"""
Stage 2 & 3 Unified Geometry Module (roto_geometry.py)
Integrates:
  - Shared polygon & optical-flow utilities (`compute_signed_area`, `compute_master_optical_flow`)
  - Stage 2: Kinematic-Geometric Part Decomposition & 10-15 px Seam Overlap Margin (Eq. 2)
  - Stage 3: O(N) Closed-Form Hermite-to-Bezier Analytical Contour Fitting (Eqs. 3-5),
             Cubic Bezier Rasterizer, J & F / Contour RMSE Evaluators, and Table 1 Baselines.
"""

import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2

from mask_generator import compute_convexity_defects_count


# ---------------------------------------------------------------------------
# Shared Polygon & Flow Utilities
# ---------------------------------------------------------------------------

def compute_signed_area(pts: np.ndarray) -> float:
    """Computes signed polygon area via the shoelace formula."""
    if len(pts) < 3:
        return 0.0
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def compute_master_optical_flow(
    frames_rgb: Optional[List[np.ndarray]],
    master_idx: int,
) -> Optional[np.ndarray]:
    """Computes dense Farneback optical flow at master_idx when RGB sequence has 2+ frames."""
    if frames_rgb is None or len(frames_rgb) <= 1:
        return None
    num_frames = len(frames_rgb)
    m_idx = max(0, min(num_frames - 1, int(master_idx)))
    next_idx = min(num_frames - 1, m_idx + 1) if m_idx < num_frames - 1 else max(0, m_idx - 1)
    if next_idx == m_idx:
        return None
    g0 = cv2.cvtColor(frames_rgb[m_idx], cv2.COLOR_RGB2GRAY) if frames_rgb[m_idx].ndim == 3 else frames_rgb[m_idx]
    g1 = cv2.cvtColor(frames_rgb[next_idx], cv2.COLOR_RGB2GRAY) if frames_rgb[next_idx].ndim == 3 else frames_rgb[next_idx]
    return cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)


# ---------------------------------------------------------------------------
# Stage 2: Kinematic-Geometric Part Decomposition & Seam Overlap Margin (Eq. 2)
# ---------------------------------------------------------------------------

def _contour_arc_distance(idx_a: int, idx_b: int, n_pts: int) -> int:
    diff = abs(idx_a - idx_b)
    return min(diff, n_pts - diff)


def _compute_affine_flow_residual(part_mask: np.ndarray, flow: Optional[np.ndarray]) -> float:
    """
    Fits a 2D affine motion model [u, v]^T = A [x, y, 1]^T to the optical flow vectors
    inside part_mask and returns the normalized mean squared residual error E_flow(V).
    """
    if flow is None:
        return 0.0
    ys, xs = np.nonzero(part_mask > 127)
    if len(xs) < 12:
        return 0.0

    step = max(1, len(xs) // 500)
    xs_s = xs[::step].astype(np.float64)
    ys_s = ys[::step].astype(np.float64)
    u_s = flow[ys[::step], xs[::step], 0].astype(np.float64)
    v_s = flow[ys[::step], xs[::step], 1].astype(np.float64)

    a_mat = np.stack([xs_s, ys_s, np.ones_like(xs_s)], axis=1)
    try:
        coeff_u, _, _, _ = np.linalg.lstsq(a_mat, u_s, rcond=None)
        coeff_v, _, _, _ = np.linalg.lstsq(a_mat, v_s, rcond=None)
        res_u = u_s - a_mat @ coeff_u
        res_v = v_s - a_mat @ coeff_v
        mse = float(np.mean(res_u ** 2 + res_v ** 2))
        flow_var = float(np.var(u_s) + np.var(v_s)) + 1e-5
        return min(1.0, mse / flow_var)
    except Exception:
        return 0.0


def _compute_inward_pinch_bisector(
    contour_pts: np.ndarray,
    idx: int,
    bin_mask: np.ndarray,
    stride: int,
) -> np.ndarray:
    """
    Computes the inward-pointing unit bisector vector at a contour pinch point.
    Verifies sign against bin_mask interior so the vector reliably points across the mask.
    """
    n = len(contour_pts)
    if n < 3:
        return np.zeros(2, dtype=np.float64)

    p0 = contour_pts[(idx - stride) % n]
    p1 = contour_pts[idx]
    p2 = contour_pts[(idx + stride) % n]

    v1 = p1 - p0
    v2 = p2 - p1
    u1 = v1 / (np.linalg.norm(v1) + 1e-8)
    u2 = v2 / (np.linalg.norm(v2) + 1e-8)

    b = u1 - u2
    norm_b = float(np.linalg.norm(b))
    if norm_b < 1e-4:
        return np.zeros(2, dtype=np.float64)
    b = b / norm_b

    h, w = bin_mask.shape[:2]
    probe_pos = p1 + 6.0 * b
    px, py = int(round(probe_pos[0])), int(round(probe_pos[1]))
    if 0 <= py < h and 0 <= px < w and bin_mask[py, px] == 0:
        probe_neg = p1 - 6.0 * b
        nx, ny = int(round(probe_neg[0])), int(round(probe_neg[1]))
        if 0 <= ny < h and 0 <= nx < w and bin_mask[ny, nx] > 0:
            b = -b

    return b


def _detect_local_concave_pinches(
    main_contour: np.ndarray,
    bin_mask: np.ndarray,
    min_turn_deg: float = 35.0,
) -> List[Dict[str, Any]]:
    """
    Detects local inward concave pinch corners along main_contour even when shadowed by
    wider convex protrusions (such as outstretched arms below a neck constriction).
    """
    pts = main_contour[:, 0, :].astype(np.float64)
    n = len(pts)
    if n < 24:
        return []

    h, w = bin_mask.shape[:2]
    x, y = pts[:, 0], pts[:, 1]
    orient = 1.0 if compute_signed_area(pts) > 0.0 else -1.0

    k = max(12, min(35, n // 60))
    p_prev = np.roll(pts, k, axis=0)
    p_next = np.roll(pts, -k, axis=0)
    v1 = pts - p_prev
    v2 = p_next - pts
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    dot = v1[:, 0] * v2[:, 0] + v1[:, 1] * v2[:, 1]

    concave_deg = np.where(orient * cross < 0.0, np.degrees(np.arctan2(np.abs(cross), dot)), 0.0)
    on_border = (x <= 2.0) | (x >= float(w - 3)) | (y <= 2.0) | (y >= float(h - 3))
    concave_deg[on_border] = 0.0

    win = max(20, n // 35)
    pinches: List[Dict[str, Any]] = []
    for i in range(n):
        score_i = float(concave_deg[i])
        if score_i < min_turn_deg:
            continue
        idxs = np.arange(i - win, i + win + 1) % n
        if score_i >= float(np.max(concave_deg[idxs])):
            pinches.append(
                {
                    "far": (int(round(pts[i, 0])), int(round(pts[i, 1]))),
                    "far_idx": int(i),
                    "depth_px": float(score_i * 0.8),
                }
            )

    pinches.sort(key=lambda d: d["depth_px"], reverse=True)
    return pinches


def find_optimal_cut_lines(
    mask: np.ndarray,
    flow: Optional[np.ndarray] = None,
    max_cuts: int = 2,
    defect_depth_thresh_px: float = 10.0,
    lambda_geom: float = 0.5,
    lambda_mot: float = 0.35,
    lambda_overlap: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Finds optimal articulation cut segments C = (p_a, p_b) connecting opposite concavity
    defects across the mask interior by minimizing Equation (2):
      E_total = lambda_geom * E_concavity(C) + lambda_mot * E_flow(V) + lambda_overlap * E_seam
    """
    bin_mask = (mask > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return []

    main_contour = max(contours, key=cv2.contourArea)
    n_c = len(main_contour)
    pts = main_contour[:, 0, :].astype(np.float64)
    _, hull_defects = compute_convexity_defects_count(bin_mask, depth_thresh_px=defect_depth_thresh_px)
    local_pinches = _detect_local_concave_pinches(main_contour, bin_mask)

    h, w = bin_mask.shape[:2]
    diag = float(np.hypot(h, w))
    dedup_dist = max(16.0, 0.025 * diag)

    merged_raw = sorted(list(hull_defects) + local_pinches, key=lambda d: d["depth_px"], reverse=True)
    defects: List[Dict[str, Any]] = []
    stride = max(12, min(35, n_c // 60))
    for cand in merged_raw:
        cx, cy = cand["far"]
        if any(math.hypot(cx - d["far"][0], cy - d["far"][1]) < dedup_dist for d in defects):
            continue
        entry = dict(cand)
        entry["bisector"] = _compute_inward_pinch_bisector(pts, entry["far_idx"], bin_mask, stride)
        defects.append(entry)

    if len(defects) < 2:
        return []

    candidates: List[Dict[str, Any]] = []
    for i in range(len(defects)):
        for j in range(i + 1, len(defects)):
            d1, d2 = defects[i], defects[j]
            arc_dist = _contour_arc_distance(d1["far_idx"], d2["far_idx"], n_c)
            if arc_dist < max(12, n_c // 10):
                continue

            p1 = np.array(d1["far"], dtype=np.float64)
            p2 = np.array(d2["far"], dtype=np.float64)
            chord_len = float(np.linalg.norm(p2 - p1))
            if chord_len < 4.0 or chord_len > 0.45 * diag:
                continue

            inside_count = 0
            for t_val in (0.2, 0.35, 0.5, 0.65, 0.8):
                sp = p1 + t_val * (p2 - p1)
                sx, sy = int(round(sp[0])), int(round(sp[1]))
                if 0 <= sy < h and 0 <= sx < w and bin_mask[sy, sx] > 0:
                    inside_count += 1
            if inside_count < 4:
                continue

            c_dir = (p2 - p1) / (chord_len + 1e-8)
            b1 = d1["bisector"]
            b2 = d2["bisector"]
            has_bisectors = (float(np.linalg.norm(b1)) > 0.5) and (float(np.linalg.norm(b2)) > 0.5)
            if has_bisectors:
                align1 = float(np.dot(b1, c_dir))
                align2 = float(np.dot(b2, -c_dir))
                if align1 < 0.25 or align2 < 0.25:
                    continue
                align_bonus = max(0.25, 0.5 * (align1 + align2))
            else:
                align_bonus = 1.0

            mean_depth = 0.5 * (d1["depth_px"] + d2["depth_px"])
            e_concavity = float(chord_len / (chord_len + 2.0 * mean_depth * align_bonus))
            e_seam = float(chord_len / diag)

            if flow is not None:
                cut_test = bin_mask.copy()
                cv2.line(cut_test, tuple(p1.astype(int)), tuple(p2.astype(int)), 0, thickness=3)
                num_lbl, lbl_map = cv2.connectedComponents(cut_test)
                if num_lbl >= 3:
                    residuals = []
                    for lbl_id in range(1, num_lbl):
                        comp = (lbl_map == lbl_id).astype(np.uint8) * 255
                        if np.count_nonzero(comp) > 50:
                            residuals.append(_compute_affine_flow_residual(comp, flow))
                    e_flow = float(np.mean(residuals)) if residuals else 0.0
                else:
                    e_flow = 0.5
            else:
                e_flow = 0.0

            e_total = lambda_geom * e_concavity + lambda_mot * e_flow + lambda_overlap * e_seam
            candidates.append(
                {
                    "p1": (int(p1[0]), int(p1[1])),
                    "p2": (int(p2[0]), int(p2[1])),
                    "chord_len": chord_len,
                    "e_concavity": e_concavity,
                    "e_flow": e_flow,
                    "e_seam": e_seam,
                    "e_total": e_total,
                    "idx1": d1["far_idx"],
                    "idx2": d2["far_idx"],
                }
            )

    candidates.sort(key=lambda c: c["e_total"])

    selected_cuts: List[Dict[str, Any]] = []
    used_indices: List[int] = []
    for cand in candidates:
        if len(selected_cuts) >= max_cuts:
            break
        if any(_contour_arc_distance(cand["idx1"], u, n_c) < max(10, n_c // 12) for u in used_indices):
            continue
        if any(_contour_arc_distance(cand["idx2"], u, n_c) < max(10, n_c // 12) for u in used_indices):
            continue
        selected_cuts.append(cand)
        used_indices.extend([cand["idx1"], cand["idx2"]])

    return selected_cuts


def decompose_mask_into_parts(
    mask: np.ndarray,
    flow: Optional[np.ndarray] = None,
    max_parts: int = 3,
    overlap_margin_px: int = 12,
    min_part_area_ratio: float = 0.06,
    defect_depth_thresh_px: float = 10.0,
) -> Dict[str, Any]:
    """
    Decomposes a master frame binary/alpha mask into quasi-rigid parts and expands each part
    across internal cut boundaries by overlap_margin_px (default 12 px, within 10-15 px)
    while constraining to the original outer silhouette.
    """
    bin_mask = (mask > 127).astype(np.uint8) * 255
    total_area = int(np.count_nonzero(bin_mask))
    if total_area == 0 or max_parts <= 1:
        return {
            "parts": [{"part_id": 0, "name": "Part_01", "mask": bin_mask, "core_mask": bin_mask.copy(), "area_px": total_area}],
            "cut_lines": [],
            "overlap_margin_px": int(overlap_margin_px),
            "seam_gap_ratio": 0.0,
        }

    cuts = find_optimal_cut_lines(
        bin_mask,
        flow=flow,
        max_cuts=max(1, max_parts - 1),
        defect_depth_thresh_px=defect_depth_thresh_px,
    )
    if not cuts:
        return {
            "parts": [{"part_id": 0, "name": "Part_01", "mask": bin_mask, "core_mask": bin_mask.copy(), "area_px": total_area}],
            "cut_lines": [],
            "overlap_margin_px": int(overlap_margin_px),
            "seam_gap_ratio": 0.0,
        }

    split_canvas = bin_mask.copy()
    for cut in cuts:
        cv2.line(split_canvas, cut["p1"], cut["p2"], 0, thickness=3)

    num_labels, label_map = cv2.connectedComponents(split_canvas)
    raw_part_masks: List[np.ndarray] = []
    min_area = int(total_area * min_part_area_ratio)

    for lbl in range(1, num_labels):
        comp_mask = (label_map == lbl).astype(np.uint8) * 255
        if np.count_nonzero(comp_mask) >= min_area:
            raw_part_masks.append(comp_mask)

    if len(raw_part_masks) <= 1:
        return {
            "parts": [{"part_id": 0, "name": "Part_01", "mask": bin_mask, "core_mask": bin_mask.copy(), "area_px": total_area}],
            "cut_lines": [],
            "overlap_margin_px": int(overlap_margin_px),
            "seam_gap_ratio": 0.0,
        }

    raw_part_masks.sort(key=lambda m: np.count_nonzero(m), reverse=True)

    parts_out: List[Dict[str, Any]] = []
    k_rad = max(0, int(overlap_margin_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k_rad + 1, 2 * k_rad + 1)) if k_rad > 0 else None

    union_reconstructed = np.zeros_like(bin_mask)
    for idx, p_mask in enumerate(raw_part_masks):
        if kernel is not None:
            overlapped_part = cv2.bitwise_and(cv2.dilate(p_mask, kernel, iterations=1), bin_mask)
        else:
            overlapped_part = p_mask.copy()

        union_reconstructed = cv2.bitwise_or(union_reconstructed, overlapped_part)
        parts_out.append(
            {
                "part_id": idx,
                "name": f"Part_{idx + 1:02d}",
                "mask": overlapped_part,
                "core_mask": p_mask,
                "area_px": int(np.count_nonzero(overlapped_part)),
            }
        )

    missing_seam_pixels = np.count_nonzero((bin_mask > 0) & (union_reconstructed == 0))
    seam_gap_ratio = float(missing_seam_pixels) / float(max(1, total_area))

    return {
        "parts": parts_out,
        "cut_lines": cuts,
        "overlap_margin_px": int(overlap_margin_px),
        "seam_gap_ratio": seam_gap_ratio,
    }


def rebuild_part_overlap_mask(
    core_mask: np.ndarray,
    full_mask: np.ndarray,
    overlap_margin_px: int = 12,
    boundary_offset_px: int = 0,
) -> np.ndarray:
    """
    Rebuilds a single part's mask from its core_mask with an adjustable internal overlap
    margin (constrained inside full_mask) and optional boundary offset (erosion/dilation).
    """
    core_bin = (core_mask > 127).astype(np.uint8) * 255
    full_bin = (full_mask > 127).astype(np.uint8) * 255
    if np.count_nonzero(core_bin) == 0:
        return core_bin

    k_rad = max(0, int(overlap_margin_px))
    if k_rad > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k_rad + 1, 2 * k_rad + 1))
        out = cv2.bitwise_and(cv2.dilate(core_bin, kernel, iterations=1), full_bin)
    else:
        out = core_bin.copy()

    b_off = int(boundary_offset_px)
    if b_off > 0:
        k_b = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * b_off + 1, 2 * b_off + 1))
        out = cv2.dilate(out, k_b, iterations=1)
    elif b_off < 0:
        r_e = abs(b_off)
        k_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_e + 1, 2 * r_e + 1))
        eroded = cv2.erode(out, k_e, iterations=1)
        if np.count_nonzero(eroded) > 20:
            out = eroded

    return out


def compute_parts_overlap_stats(
    parts: List[Dict[str, Any]],
    full_mask: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes overlap pixels (covered by >= 2 enabled parts) and internal seam gap ratio
    across enabled parts against full_mask.
    """
    full_bin = (full_mask > 127).astype(np.uint8) * 255
    h, w = full_bin.shape[:2]
    total_area = int(np.count_nonzero(full_bin))

    overlap_count = np.zeros((h, w), dtype=np.int32)
    union_mask = np.zeros((h, w), dtype=np.uint8)
    enabled_count = 0

    for p in parts:
        if not p.get("enabled", True):
            continue
        enabled_count += 1
        pm = p["mask"] > 127
        overlap_count[pm] += 1
        union_mask[pm] = 255

    overlap_pixels = int(np.count_nonzero(overlap_count >= 2))
    missing = int(np.count_nonzero((full_bin > 0) & (union_mask == 0)))
    seam_gap_ratio = float(missing) / float(max(1, total_area)) if enabled_count > 0 else 1.0

    return {
        "enabled_parts": enabled_count,
        "overlap_pixels": overlap_pixels,
        "seam_gap_ratio": seam_gap_ratio,
        "seam_gap_percent": round(seam_gap_ratio * 100.0, 4),
    }


# ---------------------------------------------------------------------------
# Stage 3: O(N) Closed-Form Hermite-to-Bezier Fitting & Metrics (Eqs. 3-5)
# ---------------------------------------------------------------------------

def extract_ordered_contour(mask: np.ndarray, smooth_sigma: float = 1.2) -> np.ndarray:
    """
    Extracts the primary outer contour of a binary/alpha mask as an ordered (M, 2) float64
    array of (x, y) coordinates and applies circular Gaussian smoothing to remove 1-px
    raster stair-stepping artifacts.
    """
    bin_mask = (mask > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return np.zeros((0, 2), dtype=np.float64)

    main_c = max(contours, key=cv2.contourArea)
    pts = main_c[:, 0, :].astype(np.float64)
    if len(pts) < 6 or smooth_sigma <= 0:
        return pts

    rad = max(1, int(round(3.0 * smooth_sigma)))
    if len(pts) <= 2 * rad + 1:
        return pts

    k_idx = np.arange(-rad, rad + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (k_idx / smooth_sigma) ** 2)
    kernel /= np.sum(kernel)

    pad_pts = np.vstack([pts[-rad:], pts, pts[:rad]])
    sx = np.convolve(pad_pts[:, 0], kernel, mode="valid")
    sy = np.convolve(pad_pts[:, 1], kernel, mode="valid")
    out = np.stack([sx, sy], axis=1)

    h_img, w_img = bin_mask.shape[:2]
    out[pts[:, 0] <= 0.0, 0] = 0.0
    out[pts[:, 0] >= float(w_img - 1), 0] = float(w_img - 1)
    out[pts[:, 1] <= 0.0, 1] = 0.0
    out[pts[:, 1] >= float(h_img - 1), 1] = float(h_img - 1)
    return out


def select_curvature_adaptive_anchors(
    contour_pts: np.ndarray,
    target_num_points: Optional[int] = None,
    epsilon_ratio: float = 0.0025,
    min_points: int = 8,
    max_points: int = 28,
) -> np.ndarray:
    """
    Samples a sparse closed loop of N anchor points {P_0, ..., P_{N-1}} from a dense pixel contour.
    """
    if len(contour_pts) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if len(contour_pts) <= min_points:
        return contour_pts.copy()

    if target_num_points is not None:
        n_target = max(4, int(target_num_points))
        diffs = np.diff(np.vstack([contour_pts, contour_pts[:1]]), axis=0)
        seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
        p_prev = np.roll(contour_pts, 1, axis=0)
        p_next = np.roll(contour_pts, -1, axis=0)
        v1 = contour_pts - p_prev
        v2 = p_next - contour_pts
        n1 = np.linalg.norm(v1, axis=1) + 1e-8
        n2 = np.linalg.norm(v2, axis=1) + 1e-8
        cos_ang = np.clip(np.sum(v1 * v2, axis=1) / (n1 * n2), -1.0, 1.0)
        turn_ang = np.arccos(cos_ang)

        weights = seg_lens * (1.0 + 2.5 * turn_ang)
        cum_w = np.concatenate([[0.0], np.cumsum(weights)])
        total_w = cum_w[-1]
        sample_targets = np.linspace(0.0, total_w, n_target, endpoint=False)
        indices = []
        for st in sample_targets:
            idx = int(np.searchsorted(cum_w[:-1], st, side="right") - 1)
            idx = max(0, min(len(contour_pts) - 1, idx))
            if not indices or idx != indices[-1]:
                indices.append(idx)
        return contour_pts[indices]

    peri = float(cv2.arcLength(contour_pts.astype(np.float32).reshape(-1, 1, 2), True))
    eps = max(2.5, epsilon_ratio * peri)
    approx = cv2.approxPolyDP(contour_pts.astype(np.float32).reshape(-1, 1, 2), eps, True)
    anchors = approx[:, 0, :].astype(np.float64)

    for _ in range(14):
        if len(anchors) < min_points and eps > 0.5:
            eps *= 0.82
            approx = cv2.approxPolyDP(contour_pts.astype(np.float32).reshape(-1, 1, 2), eps, True)
            anchors = approx[:, 0, :].astype(np.float64)
        elif len(anchors) > max_points:
            eps *= 1.18
            approx = cv2.approxPolyDP(contour_pts.astype(np.float32).reshape(-1, 1, 2), eps, True)
            anchors = approx[:, 0, :].astype(np.float64)
        else:
            break

    return anchors


def compute_adaptive_tensions(
    anchors: np.ndarray,
    default_tension: float = 0.5,
    cusp_angle_deg: float = 60.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes curvature-adaptive tension tau_i in [0, 1] at each anchor point P_i.
    At sharp corners where turning angle >= cusp_angle_deg, tau_i drops to 0.0.
    """
    n = len(anchors)
    if n < 3:
        return np.full(n, default_tension, dtype=np.float64), np.zeros(n, dtype=bool)

    p_prev = np.roll(anchors, 1, axis=0)
    p_next = np.roll(anchors, -1, axis=0)

    v_in = anchors - p_prev
    v_out = p_next - anchors
    len_in = np.linalg.norm(v_in, axis=1) + 1e-8
    len_out = np.linalg.norm(v_out, axis=1) + 1e-8

    cos_theta = np.clip(np.sum(v_in * v_out, axis=1) / (len_in * len_out), -1.0, 1.0)
    turn_deg = np.degrees(np.arccos(cos_theta))

    tensions = np.full(n, default_tension, dtype=np.float64)
    is_corner = turn_deg >= cusp_angle_deg

    for i in range(n):
        if turn_deg[i] >= cusp_angle_deg:
            tensions[i] = 0.0
            continue
        min_chord = min(len_in[i], len_out[i])
        full_chord = float(np.linalg.norm(p_next[i] - p_prev[i])) + 1e-8
        tensions[i] = min(tensions[i], 0.9 * (min_chord / full_chord))

    return tensions, is_corner


def contour_to_hermite_bezier(
    anchors: np.ndarray,
    tensions: Optional[np.ndarray] = None,
    default_tension: float = 0.5,
    cusp_angle_deg: float = 60.0,
) -> List[Dict[str, Any]]:
    """
    Implements Equations (3), (4), and (5) in O(N) time:
      V_i = tau_i * (P_{(i+1) mod N} - P_{(i-1) mod N})
      out_handle_i = +(1/3) * V_i,  in_handle_i = -(1/3) * V_i
    """
    pts = np.asarray(anchors, dtype=np.float64)
    num_pts = len(pts)
    if num_pts == 0:
        return []

    if tensions is None:
        tau, is_corner = compute_adaptive_tensions(
            pts, default_tension=default_tension, cusp_angle_deg=cusp_angle_deg
        )
    else:
        tau = np.asarray(tensions, dtype=np.float64).reshape(-1)
        is_corner = tau < 0.15

    tau_col = tau.reshape(num_pts, 1)
    p_prev = np.roll(pts, 1, axis=0)
    p_next = np.roll(pts, -1, axis=0)

    v_tan = tau_col * (p_next - p_prev)
    out_handles = +(1.0 / 3.0) * v_tan
    in_handles = -(1.0 / 3.0) * v_tan

    schema: List[Dict[str, Any]] = []
    for i in range(num_pts):
        schema.append(
            {
                "index": i,
                "anchor": [float(pts[i, 0]), float(pts[i, 1])],
                "in_handle": [float(in_handles[i, 0]), float(in_handles[i, 1])],
                "out_handle": [float(out_handles[i, 0]), float(out_handles[i, 1])],
                "hermite_tangent": [float(v_tan[i, 0]), float(v_tan[i, 1])],
                "tension": float(tau[i]),
                "is_corner": bool(is_corner[i]),
            }
        )
    return schema


def evaluate_hermite_segment(
    p0: np.ndarray, p1: np.ndarray, v0: np.ndarray, v1: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Evaluates the cubic Hermite polynomial p(t) from Equation (3) for t in [0, 1]."""
    t = np.asarray(t, dtype=np.float64).reshape(-1, 1)
    a = 2.0 * p0 - 2.0 * p1 + v0 + v1
    b = -3.0 * p0 + 3.0 * p1 - 2.0 * v0 - v1
    return a * (t ** 3) + b * (t ** 2) + v0 * t + p0


def evaluate_bezier_segment(
    b0: np.ndarray, b1: np.ndarray, b2: np.ndarray, b3: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Evaluates the cubic Bezier segment B(t) for t in [0, 1]."""
    t = np.asarray(t, dtype=np.float64).reshape(-1, 1)
    u = 1.0 - t
    return (u ** 3) * b0 + 3.0 * (u ** 2) * t * b1 + 3.0 * u * (t ** 2) * b2 + (t ** 3) * b3


def evaluate_cubic_bezier_loop(
    bezier_schema: List[Dict[str, Any]], samples_per_segment: int = 24
) -> np.ndarray:
    """Samples a closed cubic Bezier loop defined by bezier_schema into a dense polygon."""
    n = len(bezier_schema)
    if n == 0:
        return np.zeros((0, 2), dtype=np.float64)

    t_vals = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
    curve_pts = []
    for i in range(n):
        cp0 = bezier_schema[i]
        cp1 = bezier_schema[(i + 1) % n]
        p0 = np.array(cp0["anchor"], dtype=np.float64)
        out0 = np.array(cp0["out_handle"], dtype=np.float64)
        p1 = np.array(cp1["anchor"], dtype=np.float64)
        in1 = np.array(cp1["in_handle"], dtype=np.float64)
        curve_pts.append(evaluate_bezier_segment(p0, p0 + out0, p1 + in1, p1, t_vals))

    return np.vstack(curve_pts)


def rasterize_bezier_shapes(
    shapes_schemas: List[List[Dict[str, Any]]],
    height: int,
    width: int,
    samples_per_segment: int = 24,
) -> np.ndarray:
    """Rasterizes closed cubic Bezier shape schemas into a binary alpha mask in {0, 255}."""
    canvas = np.zeros((height, width), dtype=np.uint8)
    for schema in shapes_schemas:
        if len(schema) < 3:
            continue
        poly = evaluate_cubic_bezier_loop(schema, samples_per_segment=samples_per_segment)
        pts_int = np.round(poly).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(canvas, [pts_int], 255)
    return canvas


def compute_j_and_f_metrics(
    pred_mask: np.ndarray, gt_mask: np.ndarray, bound_tol_px: int = 2
) -> Dict[str, float]:
    """Computes DAVIS-style region similarity J (IoU %) and contour accuracy F (Boundary F1 %)."""
    pred_bin = pred_mask > 127
    gt_bin = gt_mask > 127

    inter = float(np.count_nonzero(pred_bin & gt_bin))
    union = float(np.count_nonzero(pred_bin | gt_bin))
    j_iou = (inter / union) * 100.0 if union > 0 else 100.0

    k3 = np.ones((3, 3), np.uint8)
    pred_b = cv2.morphologyEx(pred_bin.astype(np.uint8) * 255, cv2.MORPH_GRADIENT, k3) > 0
    gt_b = cv2.morphologyEx(gt_bin.astype(np.uint8) * 255, cv2.MORPH_GRADIENT, k3) > 0

    if not np.any(pred_b) and not np.any(gt_b):
        f_score = 100.0
    elif not np.any(pred_b) or not np.any(gt_b):
        f_score = 0.0
    else:
        k_tol = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * bound_tol_px + 1, 2 * bound_tol_px + 1))
        pred_dil = cv2.dilate(pred_b.astype(np.uint8), k_tol) > 0
        gt_dil = cv2.dilate(gt_b.astype(np.uint8), k_tol) > 0
        prec = float(np.count_nonzero(pred_b & gt_dil)) / float(np.count_nonzero(pred_b))
        rec = float(np.count_nonzero(gt_b & pred_dil)) / float(np.count_nonzero(gt_b))
        f_score = (2.0 * prec * rec / (prec + rec)) * 100.0 if (prec + rec) > 0 else 0.0

    return {
        "J": float(j_iou),
        "F": float(f_score),
        "J_and_F": float(0.5 * (j_iou + f_score)),
    }


def mask_to_hermite_bezier_shape(
    mask: np.ndarray,
    target_num_points: Optional[int] = None,
    epsilon_ratio: float = 0.0025,
    min_points: int = 8,
    max_points: int = 24,
    default_tension: float = 0.5,
) -> Dict[str, Any]:
    """Converts a single binary/alpha mask into a sparse cubic Bezier schema and evaluates J, F, J&F."""
    h, w = mask.shape[:2]
    contour = extract_ordered_contour(mask, smooth_sigma=1.2)
    anchors = select_curvature_adaptive_anchors(
        contour,
        target_num_points=target_num_points,
        epsilon_ratio=epsilon_ratio,
        min_points=min_points,
        max_points=max_points,
    )
    bezier_cp = contour_to_hermite_bezier(anchors, default_tension=default_tension)
    recon_mask = rasterize_bezier_shapes([bezier_cp], h, w)
    metrics = compute_j_and_f_metrics(recon_mask, mask)

    return {
        "num_control_points": len(bezier_cp),
        "control_points": bezier_cp,
        "reconstructed_mask": recon_mask,
        "metrics": metrics,
    }


def compute_contour_rmse_px(
    shapes_schemas: List[List[Dict[str, Any]]],
    gt_mask: np.ndarray,
    samples_per_segment: int = 24,
) -> float:
    """Computes symmetric contour RMSE (px) between rasterized Bezier curves and gt_mask."""
    gt_contour = extract_ordered_contour(gt_mask, smooth_sigma=0.5)
    if len(gt_contour) == 0 or not shapes_schemas:
        return 0.0

    h, w = gt_mask.shape[:2]
    pred_mask = rasterize_bezier_shapes(shapes_schemas, h, w, samples_per_segment=samples_per_segment)
    pred_contour = extract_ordered_contour(pred_mask, smooth_sigma=0.5)
    if len(pred_contour) == 0:
        return float(np.hypot(h, w))

    diff_p2g = pred_contour[:, None, :] - gt_contour[None, :, :]
    dist_p2g = np.min(np.hypot(diff_p2g[:, :, 0], diff_p2g[:, :, 1]), axis=1)

    diff_g2p = gt_contour[:, None, :] - pred_contour[None, :, :]
    dist_g2p = np.min(np.hypot(diff_g2p[:, :, 0], diff_g2p[:, :, 1]), axis=1)

    return float(np.sqrt(np.mean(np.concatenate([dist_p2g ** 2, dist_g2p ** 2]))))


# ---------------------------------------------------------------------------
# Table 1 Benchmark Baselines (Retained for Paper Evaluation)
# ---------------------------------------------------------------------------

def rdp_polygon_vectorize(
    mask: np.ndarray,
    epsilon_ratio: float = 0.0022,
    target_num_points: Optional[int] = None,
) -> Dict[str, Any]:
    """Baseline 1 (Table 1): RDP Polygon Vectorization (cv2.approxPolyDP)."""
    h, w = mask.shape[:2]
    bin_mask = (mask > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {"num_control_points": 0, "control_points": [], "reconstructed_mask": np.zeros_like(bin_mask), "metrics": {"J": 0.0, "F": 0.0, "J_and_F": 0.0}}

    main_c = max(contours, key=cv2.contourArea)
    peri = float(cv2.arcLength(main_c, True))

    if target_num_points is not None:
        low_eps, high_eps = 0.2, max(2.0, 0.08 * peri)
        approx = main_c
        for _ in range(12):
            mid_eps = 0.5 * (low_eps + high_eps)
            approx = cv2.approxPolyDP(main_c, mid_eps, True)
            if len(approx) > target_num_points:
                low_eps = mid_eps
            else:
                high_eps = mid_eps
        pts = approx[:, 0, :].astype(np.float64)
    else:
        eps = max(0.5, epsilon_ratio * peri)
        approx = cv2.approxPolyDP(main_c, eps, True)
        pts = approx[:, 0, :].astype(np.float64)

    schema = [
        {
            "index": i,
            "anchor": [float(pts[i, 0]), float(pts[i, 1])],
            "in_handle": [0.0, 0.0],
            "out_handle": [0.0, 0.0],
            "hermite_tangent": [0.0, 0.0],
            "tension": 0.0,
            "is_corner": True,
        }
        for i in range(len(pts))
    ]
    recon = rasterize_bezier_shapes([schema], h, w, samples_per_segment=4)
    return {
        "num_control_points": len(schema),
        "control_points": schema,
        "reconstructed_mask": recon,
        "metrics": compute_j_and_f_metrics(recon, mask),
    }


def sam2_autotrace_vectorize(
    mask: np.ndarray,
    step_px: float = 11.0,
) -> Dict[str, Any]:
    """Baseline 2 (Table 1): SAM 2 Auto-Trace (uniform unsmoothed spline tracing)."""
    h, w = mask.shape[:2]
    raw_contour = extract_ordered_contour(mask, smooth_sigma=0.0)
    if len(raw_contour) == 0:
        return {"num_control_points": 0, "control_points": [], "reconstructed_mask": np.zeros((h, w), dtype=np.uint8), "metrics": {"J": 0.0, "F": 0.0, "J_and_F": 0.0}}

    diffs = np.diff(np.vstack([raw_contour, raw_contour[:1]]), axis=0)
    peri = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1])))
    n_pts = max(16, int(round(peri / max(2.0, step_px))))
    idx_sample = np.linspace(0, len(raw_contour), n_pts, endpoint=False).astype(int)
    pts = raw_contour[idx_sample]

    fixed_tau = np.full(len(pts), 0.5, dtype=np.float64)
    schema = contour_to_hermite_bezier(pts, tensions=fixed_tau)
    recon = rasterize_bezier_shapes([schema], h, w)
    return {
        "num_control_points": len(schema),
        "control_points": schema,
        "reconstructed_mask": recon,
        "metrics": compute_j_and_f_metrics(recon, mask),
    }


def least_squares_bspline_fit(
    mask: np.ndarray,
    num_segments: int = 20,
    newton_iters: int = 5,
) -> Dict[str, Any]:
    """Baseline 3 (Table 1): Iterative Least-Squares Cubic Bezier Fitting (Schneider 1990)."""
    h, w = mask.shape[:2]
    contour = extract_ordered_contour(mask, smooth_sigma=1.0)
    if len(contour) < num_segments * 2:
        return mask_to_hermite_bezier_shape(mask, target_num_points=num_segments)

    n_c = len(contour)
    split_indices = np.linspace(0, n_c, num_segments, endpoint=False).astype(int)
    anchors = contour[split_indices]
    in_handles = np.zeros((num_segments, 2), dtype=np.float64)
    out_handles = np.zeros((num_segments, 2), dtype=np.float64)

    for s in range(num_segments):
        i_start = split_indices[s]
        i_end = split_indices[(s + 1) % num_segments]
        seg_pts = (
            contour[i_start : i_end + 1]
            if i_end > i_start
            else np.vstack([contour[i_start:], contour[: i_end + 1]])
        )

        p0, p3 = seg_pts[0], seg_pts[-1]
        if len(seg_pts) <= 3:
            chord = p3 - p0
            out_handles[s] = chord / 3.0
            in_handles[(s + 1) % num_segments] = -chord / 3.0
            continue

        d = np.hypot(np.diff(seg_pts[:, 0]), np.diff(seg_pts[:, 1]))
        cum = np.concatenate([[0.0], np.cumsum(d)])
        u = cum / (cum[-1] + 1e-8)

        b1 = p0 + (p3 - p0) / 3.0
        b2 = p0 + 2.0 * (p3 - p0) / 3.0

        for _ in range(max(1, newton_iters)):
            u_col = u.reshape(-1, 1)
            om = 1.0 - u_col
            a1 = 3.0 * (om ** 2) * u_col
            a2 = 3.0 * om * (u_col ** 2)
            rhs = seg_pts - ((om ** 3) * p0 + (u_col ** 3) * p3)

            a_mat = np.hstack([a1, a2])
            sol = np.linalg.solve(a_mat.T @ a_mat + 1e-6 * np.eye(2), a_mat.T @ rhs)
            b1, b2 = sol[0], sol[1]

            q_u = (om ** 3) * p0 + a1 * b1 + a2 * b2 + (u_col ** 3) * p3
            qp_u = (
                3.0 * (om ** 2) * (b1 - p0)
                + 6.0 * om * u_col * (b2 - b1)
                + 3.0 * (u_col ** 2) * (p3 - b2)
            )
            num = np.sum((q_u - seg_pts) * qp_u, axis=1)
            den = np.sum(qp_u * qp_u, axis=1) + 1e-8
            u = np.clip(u - num / den, 0.0, 1.0)
            u[0], u[-1] = 0.0, 1.0

        out_handles[s] = b1 - p0
        in_handles[(s + 1) % num_segments] = b2 - p3

    schema = [
        {
            "index": i,
            "anchor": [float(anchors[i, 0]), float(anchors[i, 1])],
            "in_handle": [float(in_handles[i, 0]), float(in_handles[i, 1])],
            "out_handle": [float(out_handles[i, 0]), float(out_handles[i, 1])],
            "hermite_tangent": [float(3.0 * out_handles[i, 0]), float(3.0 * out_handles[i, 1])],
            "tension": 0.5,
            "is_corner": False,
        }
        for i in range(num_segments)
    ]
    recon = rasterize_bezier_shapes([schema], h, w)
    return {
        "num_control_points": len(schema),
        "control_points": schema,
        "reconstructed_mask": recon,
        "metrics": compute_j_and_f_metrics(recon, mask),
    }
