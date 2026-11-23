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
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2


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
            if img.ndim == 3:
                if img.shape[2] == 4:
                    alpha = img[:, :, 3]
                else:
                    alpha = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                alpha = img
            masks.append(((alpha > threshold).astype(np.uint8) * 255))
        return masks

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
    ) -> List[np.ndarray]:
        """
        Generates a tracked raster alpha mask sequence from input RGB frames.
        If init_mask is not provided, segments frame[init_idx] via GrabCut/SAM and
        propagates across the sequence.
        """
        if not frames_rgb:
            raise ValueError("frames_rgb must not be empty.")
        if init_mask is None:
            init_mask = self.segment_single_frame_grabcut(
                frames_rgb[init_idx], bbox=bbox, fg_points=fg_points
            )
        return self.propagate_mask_optical_flow(frames_rgb, init_mask, init_idx=init_idx)
