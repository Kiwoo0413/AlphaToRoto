"""
Stage 2: Kinematic-Geometric Part Decomposition & Seam Overlap Margin
Implements Equation (2):
  E_total = lambda_geom * E_concavity(C) + lambda_mot * E_flow(V) + lambda_overlap * E_seam
Decomposes an articulated binary/alpha mask into quasi-rigid sub-parts at matched concavity
pinch points and applies a 10 to 15 px internal seam overlap margin (d_overlap) along the
cut normal so part-level artist edits never open internal seam gaps.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2
from mask_generator import compute_convexity_defects_count


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

    # Subsample for speed
    step = max(1, len(xs) // 500)
    xs_s = xs[::step].astype(np.float64)
    ys_s = ys[::step].astype(np.float64)
    u_s = flow[ys[::step], xs[::step], 0].astype(np.float64)
    v_s = flow[ys[::step], xs[::step], 1].astype(np.float64)

    A = np.stack([xs_s, ys_s, np.ones_like(xs_s)], axis=1)
    try:
        coeff_u, _, _, _ = np.linalg.lstsq(A, u_s, rcond=None)
        coeff_v, _, _, _ = np.linalg.lstsq(A, v_s, rcond=None)
        res_u = u_s - A @ coeff_u
        res_v = v_s - A @ coeff_v
        mse = float(np.mean(res_u ** 2 + res_v ** 2))
        flow_var = float(np.var(u_s) + np.var(v_s)) + 1e-5
        return min(1.0, mse / flow_var)
    except Exception:
        return 0.0


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
    _, defects = compute_convexity_defects_count(bin_mask, depth_thresh_px=defect_depth_thresh_px)
    if len(defects) < 2:
        return []

    h, w = bin_mask.shape[:2]
    diag = float(np.hypot(h, w))
    candidates: List[Dict[str, Any]] = []

    # Evaluate pairs of concavity defects
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

            # Check that the midpoint of the cut lies inside the foreground mask
            mid = ((p1 + p2) * 0.5).astype(int)
            if not (0 <= mid[1] < h and 0 <= mid[0] < w) or bin_mask[mid[1], mid[0]] == 0:
                continue

            # 1. Geometric concavity cost: favors deep defects and short internal cut chords
            mean_depth = 0.5 * (d1["depth_px"] + d2["depth_px"])
            e_concavity = float(chord_len / (chord_len + 2.0 * mean_depth))

            # 2. Seam length cost normalized by bounding diagonal
            e_seam = float(chord_len / diag)

            # 3. Kinematic optical flow distortion cost (if flow is available)
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

    # Greedily pick non-overlapping cuts
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
    across internal cut boundaries by overlap_margin_px (default 12 px, within the paper's
    10-15 px specification) while constraining to the original outer silhouette.

    Returns:
      {
        "parts": List[Dict[str, Any]],  # [{'part_id': int, 'name': str, 'mask': np.ndarray, 'area_px': int}, ...]
        "cut_lines": List[Dict[str, Any]],
        "overlap_margin_px": int,
        "seam_gap_ratio": float
      }
    """
    bin_mask = (mask > 127).astype(np.uint8) * 255
    total_area = int(np.count_nonzero(bin_mask))
    if total_area == 0 or max_parts <= 1:
        return {
            "parts": [{"part_id": 0, "name": "Part_01", "mask": bin_mask, "area_px": total_area}],
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
            "parts": [{"part_id": 0, "name": "Part_01", "mask": bin_mask, "area_px": total_area}],
            "cut_lines": [],
            "overlap_margin_px": int(overlap_margin_px),
            "seam_gap_ratio": 0.0,
        }

    # Draw thin separation lines across the chosen concavity cuts
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
            "parts": [{"part_id": 0, "name": "Part_01", "mask": bin_mask, "area_px": total_area}],
            "cut_lines": [],
            "overlap_margin_px": int(overlap_margin_px),
            "seam_gap_ratio": 0.0,
        }

    # Sort parts from top-left / largest area for deterministic ordering
    raw_part_masks.sort(key=lambda m: np.count_nonzero(m), reverse=True)

    # Apply internal seam overlap margin (d_overlap = 10..15 px)
    # Expanding each part with an elliptical structuring element of radius overlap_margin_px
    # and intersecting with bin_mask ensures the overlap extends ONLY into adjacent parts
    # across the internal cut seam without bloating the outer object boundary.
    parts_out: List[Dict[str, Any]] = []
    k_rad = max(0, int(overlap_margin_px))
    if k_rad > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k_rad + 1, 2 * k_rad + 1))
    else:
        kernel = None

    union_reconstructed = np.zeros_like(bin_mask)
    for idx, p_mask in enumerate(raw_part_masks):
        if kernel is not None:
            dilated = cv2.dilate(p_mask, kernel, iterations=1)
            overlapped_part = cv2.bitwise_and(dilated, bin_mask)
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
