"""
Stage 1: Tracked Raster Alpha Mask Generation & Master Reference Frame Selection
Implements:
  1. Raster alpha mask sequence loading & generation (supports SAM 2 / Ultralytics SAM if installed,
     plus OpenCV GrabCut + Farneback Dense Optical Flow mask propagation fallback, and synthetic
     articulated mask sequence generation for testing).
  2. Multi-metric Master Reference Frame Selection (Eq. 1):
       Score(t) = w1 * Area(M_t) + w2 * N_defects(M_t) - w3 * BlurPenalty(I_t)
"""

import os
import glob
import tempfile
import importlib.util
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2


def resolve_sam2_checkpoint(custom_checkpoint: Optional[str] = None) -> Tuple[Optional[str], str]:
    """
    Locates a local SAM 2.1 Hiera-Large checkpoint in priority order:
      1. Explicitly supplied custom_checkpoint path
      2. Workspace-relative models/ or checkpoints/ folders
      3. Local HuggingFace hub cache (facebook/sam2.1-hiera-large)
    Returns (checkpoint_path, sam2_config_name) or (None, "") if not found.
    """
    cfg_name = "configs/sam2.1/sam2.1_hiera_l.yaml"
    if custom_checkpoint and os.path.isfile(custom_checkpoint):
        return custom_checkpoint, cfg_name

    base_dir = os.path.dirname(os.path.abspath(__file__))
    for rel in (
        os.path.join("models", "sam2.1_hiera_large.pt"),
        os.path.join("checkpoints", "sam2.1_hiera_large.pt"),
    ):
        cand = os.path.join(base_dir, rel)
        if os.path.isfile(cand):
            return cand, cfg_name

    hf_pattern = os.path.join(
        os.path.expanduser("~"),
        ".cache",
        "huggingface",
        "hub",
        "models--facebook--sam2.1-hiera-large",
        "snapshots",
        "*",
        "sam2.1_hiera_large.pt",
    )
    matches = sorted(glob.glob(hf_pattern))
    if matches and os.path.isfile(matches[0]):
        return matches[0], cfg_name

    return None, ""


def decode_binary_mask_array(img: np.ndarray, threshold: int = 127) -> np.ndarray:
    """Decodes an RGBA/BGR/grayscale array into a binary uint8 mask in {0, 255}."""
    if img.ndim == 3:
        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            if np.all(alpha > threshold) or np.all(alpha <= threshold):
                alpha = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        else:
            alpha = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        alpha = img
    return ((alpha > threshold).astype(np.uint8) * 255)


def make_mask_preview_plate(mask: np.ndarray) -> np.ndarray:
    """Generates a dark-slate RGB preview canvas for mask-only workflows."""
    h, w = mask.shape[:2]
    plate = np.full((h, w, 3), (24, 30, 42), dtype=np.uint8)
    plate[mask > 127] = (210, 218, 228)
    return plate


def create_synthetic_articulated_sequence(
    num_frames: int = 9, height: int = 320, width: int = 320
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Creates a synthetic articulated two-part object sequence (torso + swinging limb with a
    distinct concavity pinch at the joint) and corresponding textured RGB frames.
    Frame index 4 (frame 5) has the widest limb articulation and sharpest focus.
    """
    masks: List[np.ndarray] = []
    frames_rgb: List[np.ndarray] = []
    rng = np.random.default_rng(42)
    bg_texture = rng.integers(15, 45, size=(height, width, 3), dtype=np.uint8)
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")

    for idx in range(num_frames):
        mask = np.zeros((height, width), dtype=np.uint8)
        # Torso ellipse
        cv2.ellipse(mask, (130, 160), (52, 78), 0, 0, 360, 255, -1)

        # Swinging articulated arm (widest angle at idx == 4)
        angle_deg = -35.0 + 18.0 * abs(idx - 4)
        arm_center = (200, int(155 + (idx - 4) * 6))
        cv2.ellipse(mask, arm_center, (58, 30), angle_deg, 0, 360, 255, -1)

        # Carve explicit V-shaped concavity pinch at the joint (deepest at idx == 4)
        pinch_depth = 26 if idx == 4 else max(8, 22 - 4 * abs(idx - 4))
        top_pinch = np.array([[152, 80], [176, 80], [164, 80 + pinch_depth]], dtype=np.int32)
        bot_pinch = np.array([[152, 240], [176, 240], [164, 240 - pinch_depth]], dtype=np.int32)
        cv2.fillPoly(mask, [top_pinch], 0)
        cv2.fillPoly(mask, [bot_pinch], 0)

        frame = bg_texture.copy()
        fg_pattern = np.stack(
            [
                ((xx * 7 + idx * 5) % 180 + 70).astype(np.uint8),
                ((yy * 7 + idx * 3) % 180 + 70).astype(np.uint8),
                (((xx + yy) * 5) % 180 + 70).astype(np.uint8),
            ],
            axis=-1,
        )
        frame[mask > 127] = fg_pattern[mask > 127]

        if idx != 4:
            k_blur = 5 if abs(idx - 4) >= 2 else 3
            frame = cv2.GaussianBlur(frame, (k_blur, k_blur), 1.5)

        masks.append(mask)
        frames_rgb.append(frame)

    return masks, frames_rgb


def compute_convexity_defects_count(
    mask: np.ndarray,
    depth_thresh_px: float = 12.0,
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Computes significant convexity defects (articulation pinch points) of the foreground mask
    whose depth from the convex hull is >= depth_thresh_px.

    Returns:
        (num_defects, defects_list)
        Each entry in defects_list contains:
          - 'start': (x, y)
          - 'end': (x, y)
          - 'far': (x, y)  (the concave pinch point on the contour)
          - 'far_idx': int (index in contour)
          - 'depth_px': float
    """
    bin_mask = (mask > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0, []

    main_contour = max(contours, key=cv2.contourArea)
    if len(main_contour) < 5:
        return 0, []

    hull_indices = cv2.convexHull(main_contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 3:
        return 0, []

    # Sort hull indices monotonically to prevent OpenCV convexityDefects assertion errors
    hull_indices = np.sort(hull_indices, axis=0)
    try:
        defects = cv2.convexityDefects(main_contour, hull_indices)
    except cv2.error:
        return 0, []

    if defects is None:
        return 0, []

    defects_flat = defects.reshape(-1, 4)
    significant_defects: List[Dict[str, Any]] = []
    for i in range(defects_flat.shape[0]):
        s_idx, e_idx, f_idx, depth_fixed = defects_flat[i]
        depth_px = float(depth_fixed) / 256.0
        if depth_px >= depth_thresh_px:
            s_pt = tuple(int(v) for v in main_contour[s_idx, 0])
            e_pt = tuple(int(v) for v in main_contour[e_idx, 0])
            f_pt = tuple(int(v) for v in main_contour[f_idx, 0])
            significant_defects.append(
                {
                    "start": s_pt,
                    "end": e_pt,
                    "far": f_pt,
                    "far_idx": int(f_idx),
                    "depth_px": depth_px,
                }
            )

    significant_defects.sort(key=lambda d: d["depth_px"], reverse=True)
    return len(significant_defects), significant_defects


def compute_motion_blur_penalty(frame_img: Optional[np.ndarray]) -> float:
    """
    Computes motion blur penalty as the normalized inverse Laplacian variance of the frame.
    Returns 0.0 when no RGB/grayscale image is provided.
    """
    if frame_img is None:
        return 0.0
    if frame_img.ndim == 3:
        gray = cv2.cvtColor(frame_img, cv2.COLOR_RGB2GRAY)
    else:
        gray = frame_img
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    # Inverse variance scaled to [0, 1] range
    return 1.0 / (1.0 + lap_var / 100.0)


def select_master_frame(
    masks: List[np.ndarray],
    frames: Optional[List[np.ndarray]] = None,
    frame_numbers: Optional[List[int]] = None,
    w1: float = 0.45,
    w2: float = 0.40,
    w3: float = 0.15,
    defect_depth_thresh_px: float = 12.0,
) -> Dict[str, Any]:
    """
    Implements Equation (1):
      Score(t) = w1 * Area(M_t) + w2 * N_defects(M_t) - w3 * BlurPenalty(I_t)

    Normalizes Area(M_t) and N_defects(M_t) across the sequence to [0, 1] so the weights
    (w1=0.45, w2=0.40, w3=0.15) are scale-invariant across resolutions.

    Returns:
      {
        "master_index": int,
        "master_frame": int,
        "master_score": float,
        "per_frame_metrics": List[Dict[str, Any]]
      }
    """
    if not masks:
        raise ValueError("select_master_frame requires at least one mask.")

    num_frames = len(masks)
    if frame_numbers is None:
        frame_numbers = list(range(1, num_frames + 1))

    raw_areas = []
    raw_defects = []
    raw_blurs = []

    for idx, m in enumerate(masks):
        area = float(np.count_nonzero(m > 127))
        n_def, _ = compute_convexity_defects_count(m, depth_thresh_px=defect_depth_thresh_px)
        img = frames[idx] if (frames is not None and idx < len(frames)) else None
        blur_pen = compute_motion_blur_penalty(img)

        raw_areas.append(area)
        raw_defects.append(float(n_def))
        raw_blurs.append(blur_pen)

    max_area = max(raw_areas) if max(raw_areas) > 0 else 1.0
    max_def = max(raw_defects) if max(raw_defects) > 0 else 1.0
    max_blur = max(raw_blurs) if max(raw_blurs) > 0 else 1.0

    metrics = []
    best_idx = 0
    best_score = -1e9

    for idx in range(num_frames):
        norm_area = raw_areas[idx] / max_area
        norm_def = raw_defects[idx] / max_def if max_def > 0 else 0.0
        norm_blur = raw_blurs[idx] / max_blur if max_blur > 0 else 0.0

        score = w1 * norm_area + w2 * norm_def - w3 * norm_blur
        entry = {
            "index": idx,
            "frame": int(frame_numbers[idx]),
            "area_px": raw_areas[idx],
            "num_defects": int(raw_defects[idx]),
            "blur_penalty": raw_blurs[idx],
            "norm_area": norm_area,
            "norm_defects": norm_def,
            "norm_blur": norm_blur,
            "score": float(score),
        }
        metrics.append(entry)
        if score > best_score:
            best_score = score
            best_idx = idx

    return {
        "master_index": best_idx,
        "master_frame": int(frame_numbers[best_idx]),
        "master_score": float(best_score),
        "per_frame_metrics": metrics,
    }


def compute_object_vector_flow_profile(
    masks: List[np.ndarray],
    frames: Optional[List[np.ndarray]] = None,
    frame_numbers: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Computes foreground-masked object vector flow magnitude across consecutive frames.
    Uses dense Farneback optical flow when RGB frames are provided, or foreground XOR
    perimeter displacement + centroid shift on mask-only sequences.
    """
    num_frames = len(masks)
    if num_frames == 0:
        return {
            "frame_numbers": [],
            "per_frame_flow_mag": [],
            "per_frame_p90_mag": [],
            "cumulative_flow": [],
            "total_motion_px": 0.0,
            "peak_motion_px": 0.0,
        }

    if frame_numbers is None:
        frame_numbers = list(range(1, num_frames + 1))

    if num_frames == 1:
        return {
            "frame_numbers": [int(frame_numbers[0])],
            "per_frame_flow_mag": [0.0],
            "per_frame_p90_mag": [0.0],
            "cumulative_flow": [0.0],
            "total_motion_px": 0.0,
            "peak_motion_px": 0.0,
        }

    grays = None
    if frames is not None and len(frames) == num_frames:
        grays = [
            cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) if f.ndim == 3 else f
            for f in frames
        ]

    step_means: List[float] = []
    step_p90s: List[float] = []

    for t in range(num_frames - 1):
        m0 = masks[t] > 127
        m1 = masks[t + 1] > 127
        fg_union = m0 | m1

        if not np.any(fg_union):
            step_means.append(0.0)
            step_p90s.append(0.0)
            continue

        if grays is not None:
            flow = cv2.calcOpticalFlowFarneback(
                grays[t], grays[t + 1], None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            mag = np.hypot(flow[:, :, 0], flow[:, :, 1])
            fg_mags = mag[fg_union]
            mean_val = float(np.mean(fg_mags))
            p90_val = float(np.percentile(fg_mags, 90))
        else:
            xor_area = float(np.count_nonzero(m0 ^ m1))
            edge0 = cv2.morphologyEx((m0.astype(np.uint8) * 255), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            peri = float(max(1, np.count_nonzero(edge0)))
            ys0, xs0 = np.nonzero(m0)
            ys1, xs1 = np.nonzero(m1)
            c_shift = 0.0
            if len(xs0) > 0 and len(xs1) > 0:
                c_shift = float(np.hypot(np.mean(xs1) - np.mean(xs0), np.mean(ys1) - np.mean(ys0)))
            disp = (xor_area / peri) + c_shift
            mean_val = disp
            p90_val = disp * 1.4

        step_means.append(mean_val)
        step_p90s.append(p90_val)

    per_frame_flow_mag = [0.0] * num_frames
    per_frame_p90_mag = [0.0] * num_frames
    cumulative_flow = [0.0] * num_frames

    for t in range(num_frames):
        adj_means = []
        adj_p90s = []
        if t > 0:
            adj_means.append(step_means[t - 1])
            adj_p90s.append(step_p90s[t - 1])
        if t < num_frames - 1:
            adj_means.append(step_means[t])
            adj_p90s.append(step_p90s[t])
        per_frame_flow_mag[t] = float(np.mean(adj_means)) if adj_means else 0.0
        per_frame_p90_mag[t] = float(np.max(adj_p90s)) if adj_p90s else 0.0

    for t in range(num_frames - 1):
        cumulative_flow[t + 1] = cumulative_flow[t] + step_p90s[t]

    total_motion_px = float(cumulative_flow[-1])
    peak_motion_px = float(max(per_frame_p90_mag)) if per_frame_p90_mag else 0.0

    return {
        "frame_numbers": [int(f) for f in frame_numbers],
        "per_frame_flow_mag": [round(float(v), 3) for v in per_frame_flow_mag],
        "per_frame_p90_mag": [round(float(v), 3) for v in per_frame_p90_mag],
        "cumulative_flow": [round(float(v), 3) for v in cumulative_flow],
        "total_motion_px": round(total_motion_px, 3),
        "peak_motion_px": round(peak_motion_px, 3),
    }


def select_representative_master_frames(
    masks: List[np.ndarray],
    frames: Optional[List[np.ndarray]] = None,
    frame_numbers: Optional[List[int]] = None,
    w1: float = 0.45,
    w2: float = 0.40,
    w3: float = 0.15,
    defect_depth_thresh_px: float = 12.0,
    flow_motion_threshold_px: float = 22.0,
    peak_flow_trigger_px: float = 5.5,
    max_master_frames: int = 4,
    min_frame_gap: int = 3,
) -> Dict[str, Any]:
    """
    Selects the primary Master Reference Frame F_master via Eq. (1), analyzes the object's
    vector flow profile across the sequence, and automatically promotes 2+ representative
    master frames spaced across high-motion intervals when object motion exceeds threshold.
    """
    num_frames = len(masks)
    if frame_numbers is None:
        frame_numbers = list(range(1, num_frames + 1))

    base = select_master_frame(
        masks,
        frames=frames,
        frame_numbers=frame_numbers,
        w1=w1,
        w2=w2,
        w3=w3,
        defect_depth_thresh_px=defect_depth_thresh_px,
    )
    flow_profile = compute_object_vector_flow_profile(
        masks, frames=frames, frame_numbers=frame_numbers
    )

    primary_idx = int(base["master_index"])
    large_motion = (num_frames >= 4) and (
        flow_profile["total_motion_px"] >= float(flow_motion_threshold_px)
        or flow_profile["peak_motion_px"] >= float(peak_flow_trigger_px)
    )

    if not large_motion or max_master_frames <= 1:
        selected_indices = [primary_idx]
    else:
        target_k = min(
            int(max_master_frames),
            max(2, 1 + int(flow_profile["total_motion_px"] // max(1.0, float(flow_motion_threshold_px)))),
        )
        selected = [primary_idx]
        cum = flow_profile["cumulative_flow"]
        per_m = base["per_frame_metrics"]

        while len(selected) < target_k:
            s_sorted = sorted(selected)
            intervals = []
            if s_sorted[0] > 0:
                intervals.append((0, s_sorted[0]))
            for m_i in range(len(s_sorted) - 1):
                intervals.append((s_sorted[m_i], s_sorted[m_i + 1]))
            if s_sorted[-1] < num_frames - 1:
                intervals.append((s_sorted[-1], num_frames - 1))

            best_cand = None
            best_interval_motion = -1.0
            for a, b in intervals:
                interval_motion = float(cum[b] - cum[a])
                cands = [
                    idx
                    for idx in range(a, b + 1)
                    if all(abs(idx - s) >= int(min_frame_gap) for s in selected)
                ]
                if not cands:
                    continue
                if interval_motion > best_interval_motion:
                    best_interval_motion = interval_motion
                    # Inside the highest-motion interval, pick frame with highest Eq. (1) quality score
                    best_cand = max(cands, key=lambda idx: per_m[idx]["score"])

            if best_cand is None or best_interval_motion < 0.35 * float(flow_motion_threshold_px):
                break
            selected.append(best_cand)

        selected_indices = sorted(selected)

    selected_set = set(selected_indices)
    for idx, entry in enumerate(base["per_frame_metrics"]):
        entry["flow_mag_px"] = flow_profile["per_frame_flow_mag"][idx]
        entry["p90_flow_mag_px"] = flow_profile["per_frame_p90_mag"][idx]
        entry["cumulative_flow_px"] = flow_profile["cumulative_flow"][idx]
        entry["is_master"] = idx in selected_set

    return {
        "master_index": primary_idx,
        "master_frame": int(frame_numbers[primary_idx]),
        "master_score": float(base["master_score"]),
        "master_indices": selected_indices,
        "master_frames": [int(frame_numbers[i]) for i in selected_indices],
        "large_motion_detected": bool(large_motion),
        "flow_profile": flow_profile,
        "flow_motion_threshold_px": float(flow_motion_threshold_px),
        "per_frame_metrics": base["per_frame_metrics"],
    }


class TrackedMaskGenerator:
    """
    Generates or loads a tracked raster alpha mask sequence {M_t}.
    Supports:
      1. Loading existing raster alpha mask sequences from disk (PNG/JPG/EXR/TIFF or RGBA alpha).
      2. Prompt-based segmentation + temporal propagation:
         - Uses SAM 2 / Ultralytics SAM if installed in the environment.
         - Automatically falls back to OpenCV GrabCut + Dense Optical Flow (Farneback) bidirectional
           mask warping when running in lightweight environments without SAM 2 weights.
    """

    def __init__(self, sam_checkpoint: Optional[str] = None, device: str = "auto"):
        self.sam_checkpoint = sam_checkpoint
        self.device = device
        self._sam2_image_predictor = None
        self._sam2_video_predictor = None
        self._sam2_device = None
        self._sam2_ckpt = None

    def _resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def get_sam2_status(self) -> Dict[str, Any]:
        """Reports whether SAM 2.1 and CUDA are ready without eagerly loading weights into VRAM."""
        sam2_installed = importlib.util.find_spec("sam2") is not None
        ckpt, cfg = resolve_sam2_checkpoint(self.sam_checkpoint)
        cuda_available = False
        device_name = "CPU"
        try:
            import torch
            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                device_name = str(torch.cuda.get_device_name(0))
        except Exception:
            pass
        return {
            "sam2_installed": sam2_installed,
            "checkpoint_found": bool(ckpt),
            "checkpoint_path": ckpt,
            "config_name": cfg,
            "cuda": cuda_available,
            "device_name": device_name,
            "ready": bool(sam2_installed and ckpt),
        }

    def _get_sam2_image_predictor(self):
        if self._sam2_image_predictor is not None:
            return self._sam2_image_predictor, self._sam2_device
        ckpt, cfg = resolve_sam2_checkpoint(self.sam_checkpoint)
        if not ckpt:
            raise FileNotFoundError("SAM 2.1 checkpoint not found in workspace or HF cache.")
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        device = self._resolve_device()
        model = build_sam2(cfg, ckpt, device=device)
        predictor = SAM2ImagePredictor(model)
        self._sam2_image_predictor = predictor
        self._sam2_device = device
        self._sam2_ckpt = ckpt
        return predictor, device

    def _get_sam2_video_predictor(self):
        if self._sam2_video_predictor is not None:
            return self._sam2_video_predictor, self._sam2_device
        ckpt, cfg = resolve_sam2_checkpoint(self.sam_checkpoint)
        if not ckpt:
            raise FileNotFoundError("SAM 2.1 checkpoint not found in workspace or HF cache.")
        from sam2.build_sam import build_sam2_video_predictor

        device = self._resolve_device()
        predictor = build_sam2_video_predictor(cfg, ckpt, device=device)
        self._sam2_video_predictor = predictor
        self._sam2_device = device
        self._sam2_ckpt = ckpt
        return predictor, device

    def segment_single_frame_sam2(
        self,
        frame_rgb: np.ndarray,
        fg_points: Optional[List[Tuple[float, float]]] = None,
        bg_points: Optional[List[Tuple[float, float]]] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Segments a single RGB frame using local SAM 2.1 Hiera-Large on GPU/CPU from
        foreground/background point prompts and/or bounding box [x1, y1, x2, y2].
        Gracefully falls back to OpenCV GrabCut if SAM 2.1 is unavailable.
        """
        h, w = frame_rgb.shape[:2]
        fg_list = list(fg_points or [])
        bg_list = list(bg_points or [])
        if not fg_list and not bg_list and bbox is None:
            fg_list = [(float(w) * 0.5, float(h) * 0.5)]

        try:
            import torch

            predictor, device = self._get_sam2_image_predictor()
            coords = []
            labels = []
            for px, py in fg_list:
                coords.append([float(px), float(py)])
                labels.append(1)
            for px, py in bg_list:
                coords.append([float(px), float(py)])
                labels.append(0)

            pt_coords = np.array(coords, dtype=np.float32) if coords else None
            pt_labels = np.array(labels, dtype=np.int32) if labels else None
            box_arr = np.array(bbox, dtype=np.float32) if bbox is not None else None

            with torch.inference_mode():
                predictor.set_image(frame_rgb)
                masks_pred, scores, _ = predictor.predict(
                    point_coords=pt_coords,
                    point_labels=pt_labels,
                    box=box_arr,
                    multimask_output=False,
                )

            mask_u8 = ((masks_pred[0] > 0).astype(np.uint8) * 255)
            score_val = float(scores[0]) if scores is not None and len(scores) > 0 else 1.0
            return mask_u8, {
                "backend": f"SAM 2.1 Hiera-Large ({device.upper()})",
                "score": score_val,
            }
        except Exception as exc:
            gc_rect = None
            if bbox is not None:
                x1, y1, x2, y2 = [int(round(v)) for v in bbox]
                x1, x2 = max(0, min(x1, x2)), min(w - 1, max(x1, x2))
                y1, y2 = max(0, min(y1, y2)), min(h - 1, max(y1, y2))
                gc_rect = (x1, y1, max(2, x2 - x1), max(2, y2 - y1))
            clamped_fg = [(max(0, min(w - 1, int(round(x)))), max(0, min(h - 1, int(round(y))))) for x, y in fg_list]
            clamped_bg = [(max(0, min(w - 1, int(round(x)))), max(0, min(h - 1, int(round(y))))) for x, y in bg_list]
            gc_mask = self.segment_single_frame_grabcut(
                frame_rgb, bbox=gc_rect, fg_points=clamped_fg or None, bg_points=clamped_bg or None
            )
            return gc_mask, {
                "backend": "OpenCV GrabCut (Fallback)",
                "score": 0.0,
                "fallback_reason": str(exc),
            }

    def propagate_masks_sam2_video(
        self,
        frames_rgb: List[np.ndarray],
        init_idx: int = 0,
        fg_points: Optional[List[Tuple[float, float]]] = None,
        bg_points: Optional[List[Tuple[float, float]]] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        init_mask: Optional[np.ndarray] = None,
    ) -> Tuple[List[np.ndarray], str, float]:
        """
        Segments and propagates masks across frames using SAM 2.1 Video Predictor (bidirectional)
        or SAM 2.1 Image Predictor for single images, falling back to SAM 2.1 + Farneback
        Optical Flow if video propagation is unavailable.
        Returns (masks_list, backend_label, score).
        """
        if not frames_rgb:
            raise ValueError("frames_rgb must not be empty.")

        num_frames = len(frames_rgb)
        init_idx = max(0, min(num_frames - 1, int(init_idx)))

        if num_frames == 1:
            if init_mask is not None and not fg_points and not bg_points and bbox is None:
                return [(init_mask > 127).astype(np.uint8) * 255], "Provided Mask", 1.0
            m0, info = self.segment_single_frame_sam2(
                frames_rgb[0], fg_points=fg_points, bg_points=bg_points, bbox=bbox
            )
            return [m0], info["backend"], float(info.get("score", 1.0))

        # Multi-frame: try SAM 2.1 Video Predictor first
        try:
            import torch

            predictor, device = self._get_sam2_video_predictor()
            fg_list = list(fg_points or [])
            bg_list = list(bg_points or [])

            with tempfile.TemporaryDirectory(prefix="sam2_frames_") as tmpdir:
                for idx, rgb in enumerate(frames_rgb):
                    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(tmpdir, f"{idx:05d}.jpg"), bgr)

                with torch.inference_mode():
                    state = predictor.init_state(video_path=tmpdir)
                    try:
                        if init_mask is not None and not fg_list and not bg_list and bbox is None:
                            predictor.add_new_mask(
                                inference_state=state,
                                frame_idx=init_idx,
                                obj_id=1,
                                mask=(init_mask > 127),
                            )
                        else:
                            if not fg_list and not bg_list and bbox is None:
                                h, w = frames_rgb[init_idx].shape[:2]
                                fg_list = [(float(w) * 0.5, float(h) * 0.5)]
                            coords = [[float(x), float(y)] for x, y in fg_list] + [
                                [float(x), float(y)] for x, y in bg_list
                            ]
                            labels = [1] * len(fg_list) + [0] * len(bg_list)
                            pt_coords = np.array(coords, dtype=np.float32) if coords else None
                            pt_labels = np.array(labels, dtype=np.int32) if labels else None
                            box_arr = np.array(bbox, dtype=np.float32) if bbox is not None else None
                            predictor.add_new_points_or_box(
                                inference_state=state,
                                frame_idx=init_idx,
                                obj_id=1,
                                points=pt_coords,
                                labels=pt_labels,
                                box=box_arr,
                            )

                        masks_out: List[Optional[np.ndarray]] = [None] * num_frames
                        for f_idx, _, logits in predictor.propagate_in_video(
                            state, start_frame_idx=init_idx, reverse=False
                        ):
                            masks_out[int(f_idx)] = ((logits[0] > 0.0).squeeze().cpu().numpy().astype(np.uint8) * 255)

                        if init_idx > 0:
                            for f_idx, _, logits in predictor.propagate_in_video(
                                state, start_frame_idx=init_idx, reverse=True
                            ):
                                masks_out[int(f_idx)] = ((logits[0] > 0.0).squeeze().cpu().numpy().astype(np.uint8) * 255)
                    finally:
                        predictor.reset_state(state)

            if all(m is not None for m in masks_out):
                return [m for m in masks_out if m is not None], f"SAM 2.1 Video ({device.upper()})", 1.0
        except Exception:
            pass

        # Fallback: SAM 2.1 Image (or GrabCut) on init_idx + bidirectional optical flow
        score_val = 1.0
        if init_mask is not None and not fg_points and not bg_points and bbox is None:
            seed_mask = (init_mask > 127).astype(np.uint8) * 255
            backend_lbl = "Optical Flow"
        else:
            seed_mask, info = self.segment_single_frame_sam2(
                frames_rgb[init_idx], fg_points=fg_points, bg_points=bg_points, bbox=bbox
            )
            backend_lbl = f"{info['backend']} + Optical Flow"
            score_val = float(info.get("score", 0.0))

        propagated = self.propagate_mask_optical_flow(frames_rgb, seed_mask, init_idx=init_idx)
        return propagated, backend_lbl, score_val

    @staticmethod
    def load_mask_sequence(mask_paths: List[str], threshold: int = 127) -> List[np.ndarray]:
        """Loads a list of mask image files into binary uint8 masks (0 or 255)."""
        masks = []
        for p in mask_paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Mask file not found: {p}")
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Failed to decode mask image: {p}")
            masks.append(decode_binary_mask_array(img, threshold=threshold))
        return masks

    def segment_and_track(
        self,
        frames_rgb: List[np.ndarray],
        init_idx: int = 0,
        fg_points: Optional[List[Tuple[float, float]]] = None,
        bg_points: Optional[List[Tuple[float, float]]] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        use_sam: bool = True,
    ) -> Tuple[List[np.ndarray], str, float]:
        """
        Segments the prompted frame and propagates across the sequence using SAM 2.1 Video
        or OpenCV GrabCut + Dense Optical Flow fallback.
        """
        if not frames_rgb:
            raise ValueError("frames_rgb must not be empty.")
        fg_pts = [(float(p[0]), float(p[1])) for p in (fg_points or [])]
        bg_pts = [(float(p[0]), float(p[1])) for p in (bg_points or [])]
        bbox_t = tuple(float(v) for v in bbox) if (bbox and len(bbox) == 4) else None
        p_idx = max(0, min(len(frames_rgb) - 1, int(init_idx)))

        if use_sam:
            return self.propagate_masks_sam2_video(
                frames_rgb,
                init_idx=p_idx,
                fg_points=fg_pts or None,
                bg_points=bg_pts or None,
                bbox=bbox_t,
            )

        gc_rect = None
        if bbox_t is not None:
            x1, y1, x2, y2 = [int(round(v)) for v in bbox_t]
            gc_rect = (min(x1, x2), min(y1, y2), max(2, abs(x2 - x1)), max(2, abs(y2 - y1)))
        m0 = self.segment_single_frame_grabcut(
            frames_rgb[p_idx],
            bbox=gc_rect,
            fg_points=[(int(round(x)), int(round(y))) for x, y in fg_pts] or None,
            bg_points=[(int(round(x)), int(round(y))) for x, y in bg_pts] or None,
        )
        masks = self.propagate_mask_optical_flow(frames_rgb, m0, init_idx=p_idx)
        return masks, "OpenCV GrabCut + Optical Flow", 0.0

    @staticmethod
    def segment_single_frame_grabcut(
        frame_rgb: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        fg_points: Optional[List[Tuple[int, int]]] = None,
        bg_points: Optional[List[Tuple[int, int]]] = None,
    ) -> np.ndarray:
        """
        Segments a single RGB frame using OpenCV GrabCut initialized from a bounding box
        and/or foreground/background seed points.
        """
        h, w = frame_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)

        if bbox is None:
            margin_x, margin_y = max(4, w // 8), max(4, h // 8)
            bbox = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

        cv2.grabCut(frame_rgb, mask, bbox, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)

        if fg_points or bg_points:
            for px, py in (fg_points or []):
                cv2.circle(mask, (int(px), int(py)), 4, int(cv2.GC_FGD), -1)
            for px, py in (bg_points or []):
                cv2.circle(mask, (int(px), int(py)), 4, int(cv2.GC_BGD), -1)
            cv2.grabCut(frame_rgb, mask, None, bgd_model, fgd_model, 2, cv2.GC_INIT_WITH_MASK)

        out_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        return out_mask

    @staticmethod
    def propagate_mask_optical_flow(
        frames_rgb: List[np.ndarray],
        init_mask: np.ndarray,
        init_idx: int = 0,
    ) -> List[np.ndarray]:
        """
        Propagates an initial binary mask forward and backward across a video sequence
        using dense Farneback optical flow warping.
        """
        num_frames = len(frames_rgb)
        h, w = init_mask.shape[:2]
        grays = [
            cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) if f.ndim == 3 else f
            for f in frames_rgb
        ]

        masks: List[Optional[np.ndarray]] = [None] * num_frames
        masks[init_idx] = (init_mask > 127).astype(np.uint8) * 255

        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

        def _warp_step(src_mask: np.ndarray, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
            # Flow from curr_gray -> prev_gray for backward remap lookup
            flow = cv2.calcOpticalFlowFarneback(
                curr_gray, prev_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            map_x = grid_x + flow[:, :, 0]
            map_y = grid_y + flow[:, :, 1]
            warped = cv2.remap(
                src_mask, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
            )
            return ((warped > 127).astype(np.uint8) * 255)

        # Forward pass
        for t in range(init_idx + 1, num_frames):
            masks[t] = _warp_step(masks[t - 1], grays[t - 1], grays[t])

        # Backward pass
        for t in range(init_idx - 1, -1, -1):
            masks[t] = _warp_step(masks[t + 1], grays[t + 1], grays[t])

        return [m for m in masks if m is not None]

    def generate_tracked_masks(
        self,
        frames_rgb: List[np.ndarray],
        init_mask: Optional[np.ndarray] = None,
        init_idx: int = 0,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        fg_points: Optional[List[Tuple[int, int]]] = None,
        bg_points: Optional[List[Tuple[int, int]]] = None,
        use_sam: bool = False,
    ) -> List[np.ndarray]:
        """
        Generates a tracked raster alpha mask sequence from input RGB frames.
        When use_sam=True or prompts are provided without init_mask, uses SAM 2.1
        (with GrabCut + Optical Flow fallback). When init_mask is passed directly with
        use_sam=False, uses fast Farneback optical flow warping.
        """
        if not frames_rgb:
            raise ValueError("frames_rgb must not be empty.")
        if use_sam or (init_mask is None and (fg_points or bg_points or bbox)):
            masks, _, _ = self.propagate_masks_sam2_video(
                frames_rgb,
                init_idx=init_idx,
                fg_points=fg_points,
                bg_points=bg_points,
                bbox=bbox,
                init_mask=init_mask,
            )
            return masks
        if init_mask is None:
            init_mask = self.segment_single_frame_grabcut(
                frames_rgb[init_idx], bbox=bbox, fg_points=fg_points, bg_points=bg_points
            )
        return self.propagate_mask_optical_flow(frames_rgb, init_mask, init_idx=init_idx)
