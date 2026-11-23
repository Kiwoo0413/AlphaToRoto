"""
Stage 3: O(N) Closed-Form Hermite-to-Bezier Analytical Contour Fitting
Implements Equations (3), (4), and (5):
  - Curvature-adaptive sparse anchor extraction {P_0, P_1, ..., P_{N-1}} from raster masks
  - Hermite tangent vector estimation:
      V_i = tau_i * (P_{(i+1) mod N} - P_{(i-1) mod N})
    with automatic corner/cusp tension adaptation (tau_i -> 0 at sharp corners, tau_i = 0.5 on smooth curves)
  - Closed-form Hermite-to-Bezier handle conversion:
      B_0 = P_0,  B_1 = P_0 + (1/3) V_0,  B_2 = P_1 - (1/3) V_1,  B_3 = P_1
      out_handle = +(1/3) V_i,  in_handle = -(1/3) V_i
  - Analytical cubic Bezier evaluator & rasterizer + J (IoU) and F (Boundary F-score) metrics.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2


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
    return np.stack([sx, sy], axis=1)


def select_curvature_adaptive_anchors(
    contour_pts: np.ndarray,
    target_num_points: Optional[int] = None,
    epsilon_ratio: float = 0.008,
    min_points: int = 8,
    max_points: int = 28,
) -> np.ndarray:
    """
    Samples a sparse closed loop of N anchor points {P_0, ..., P_{N-1}} from a dense pixel contour.
    Combines Ramer-Douglas-Peucker curvature extrema detection with arc-length subdivision on
    long curved spans so smooth arcs are represented by sparse cubic Bezier segments rather than
    dense polygonal vertices.
    """
    if len(contour_pts) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if len(contour_pts) <= min_points:
        return contour_pts.copy()

    if target_num_points is not None:
        n_target = max(4, int(target_num_points))
        # Compute cumulative arc length and sample at curvature-weighted intervals
        diffs = np.diff(np.vstack([contour_pts, contour_pts[:1]]), axis=0)
        seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
        # Estimate local discrete curvature weight
        p_prev = np.roll(contour_pts, 1, axis=0)
        p_next = np.roll(contour_pts, -1, axis=0)
        v1 = contour_pts - p_prev
        v2 = p_next - contour_pts
        n1 = np.linalg.norm(v1, axis=1) + 1e-8
        n2 = np.linalg.norm(v2, axis=1) + 1e-8
        cos_ang = np.clip(np.sum(v1 * v2, axis=1) / (n1 * n2), -1.0, 1.0)
        turn_ang = np.arccos(cos_ang)  # [0, pi]

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

    # Adaptive RDP + maximum chord subdivision
    peri = float(cv2.arcLength(contour_pts.astype(np.float32).reshape(-1, 1, 2), True))
    eps = max(1.5, epsilon_ratio * peri)
    approx = cv2.approxPolyDP(contour_pts.astype(np.float32).reshape(-1, 1, 2), eps, True)
    anchors = approx[:, 0, :].astype(np.float64)

    # Adjust epsilon if point count falls outside [min_points, max_points]
    for _ in range(6):
        if len(anchors) < min_points and eps > 0.5:
            eps *= 0.65
            approx = cv2.approxPolyDP(contour_pts.astype(np.float32).reshape(-1, 1, 2), eps, True)
            anchors = approx[:, 0, :].astype(np.float64)
        elif len(anchors) > max_points:
            eps *= 1.35
            approx = cv2.approxPolyDP(contour_pts.astype(np.float32).reshape(-1, 1, 2), eps, True)
            anchors = approx[:, 0, :].astype(np.float64)
        else:
            break

    return anchors


def compute_adaptive_tensions(
    anchors: np.ndarray,
    default_tension: float = 0.5,
    cusp_angle_deg: float = 75.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes curvature-adaptive tension tau_i in [0, 1] at each anchor point P_i.
    At sharp corners where turning angle > cusp_angle_deg, tau_i smoothly drops toward 0.0
    to prevent Bezier loop overshooting; along smooth contours tau_i = default_tension (0.5).

    Returns:
        (tensions, is_corner_flags)
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
    turn_deg = np.degrees(np.arccos(cos_theta))  # 0 deg = straight line, 90+ deg = sharp corner

    tensions = np.full(n, default_tension, dtype=np.float64)
    is_corner = turn_deg >= cusp_angle_deg

    for i in range(n):
        if turn_deg[i] >= cusp_angle_deg:
            # Retract handle at sharp cusp corners
            sharpness = min(1.0, (turn_deg[i] - cusp_angle_deg) / max(1.0, 135.0 - cusp_angle_deg))
            tensions[i] = default_tension * (1.0 - 0.85 * sharpness)
        # Also clamp handle length relative to shortest adjacent chord to avoid self-loops
        min_chord = min(len_in[i], len_out[i])
        full_chord = float(np.linalg.norm(p_next[i] - p_prev[i])) + 1e-8
        max_safe_tau = 0.9 * (min_chord / full_chord)
        tensions[i] = min(tensions[i], max_safe_tau)

    return tensions, is_corner


def contour_to_hermite_bezier(
    anchors: np.ndarray,
    tensions: Optional[np.ndarray] = None,
    default_tension: float = 0.5,
    cusp_angle_deg: float = 75.0,
) -> List[Dict[str, Any]]:
    """
    Implements Equations (3), (4), and (5) in O(N) time:
      V_i = tau_i * (P_{(i+1) mod N} - P_{(i-1) mod N})
      out_handle_i = +(1/3) * V_i
      in_handle_i  = -(1/3) * V_i

    Returns a list of control point dictionaries compatible with Nuke's _rotopaint / _curveknob:
      [
        {
          "index": i,
          "anchor": [x, y],
          "in_handle": [dx_in, dy_in],    # leftTangent relative offset (lt_rel)
          "out_handle": [dx_out, dy_out], # rightTangent relative offset (rt_rel)
          "hermite_tangent": [vx, vy],
          "tension": float,
          "is_corner": bool
        },
        ...
      ]
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

    # Eq. (5): Hermite tangent vector V_i
    V = tau_col * (p_next - p_prev)

    # Eq. (4): Closed-form Hermite-to-Bezier relative handle offsets
    out_handles = +(1.0 / 3.0) * V
    in_handles = -(1.0 / 3.0) * V

    schema: List[Dict[str, Any]] = []
    for i in range(num_pts):
        schema.append(
            {
                "index": i,
                "anchor": [float(pts[i, 0]), float(pts[i, 1])],
                "in_handle": [float(in_handles[i, 0]), float(in_handles[i, 1])],
                "out_handle": [float(out_handles[i, 0]), float(out_handles[i, 1])],
                "hermite_tangent": [float(V[i, 0]), float(V[i, 1])],
                "tension": float(tau[i]),
                "is_corner": bool(is_corner[i]),
            }
        )
    return schema


def evaluate_hermite_segment(
    p0: np.ndarray, p1: np.ndarray, v0: np.ndarray, v1: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """
    Evaluates the cubic Hermite polynomial p(t) from Equation (3) for t in [0, 1]:
      p(t) = (2 P0 - 2 P1 + V0 + V1) t^3 + (-3 P0 + 3 P1 - 2 V0 - V1) t^2 + V0 t + P0
    """
    t = np.asarray(t, dtype=np.float64).reshape(-1, 1)
    a = 2.0 * p0 - 2.0 * p1 + v0 + v1
    b = -3.0 * p0 + 3.0 * p1 - 2.0 * v0 - v1
    c = v0
    d = p0
    return a * (t ** 3) + b * (t ** 2) + c * t + d


def evaluate_bezier_segment(
    b0: np.ndarray, b1: np.ndarray, b2: np.ndarray, b3: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """
    Evaluates the cubic Bezier segment B(t) for t in [0, 1]:
      B(t) = (1-t)^3 B0 + 3(1-t)^2 t B1 + 3(1-t) t^2 B2 + t^3 B3
    """
    t = np.asarray(t, dtype=np.float64).reshape(-1, 1)
    u = 1.0 - t
    return (u ** 3) * b0 + 3.0 * (u ** 2) * t * b1 + 3.0 * u * (t ** 2) * b2 + (t ** 3) * b3


def evaluate_cubic_bezier_loop(
    bezier_schema: List[Dict[str, Any]], samples_per_segment: int = 24
) -> np.ndarray:
    """
    Samples a closed cubic Bezier loop defined by bezier_schema into a dense (N * samples, 2)
    polygon array for rasterization or SVG/viewer rendering.
    """
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

        b0 = p0
        b1 = p0 + out0
        b2 = p1 + in1
        b3 = p1

        seg_pts = evaluate_bezier_segment(b0, b1, b2, b3, t_vals)
        curve_pts.append(seg_pts)

    return np.vstack(curve_pts)


def rasterize_bezier_shapes(
    shapes_schemas: List[List[Dict[str, Any]]],
    height: int,
    width: int,
    samples_per_segment: int = 24,
) -> np.ndarray:
    """
    Rasterizes one or more closed cubic Bezier shape schemas into a combined binary alpha mask
    of shape (height, width) with values in {0, 255}.
    """
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
    """
    Computes DAVIS-style region similarity J (IoU %) and contour accuracy F (Boundary F1-score %).
    """
    pred_bin = pred_mask > 127
    gt_bin = gt_mask > 127

    inter = float(np.count_nonzero(pred_bin & gt_bin))
    union = float(np.count_nonzero(pred_bin | gt_bin))
    j_iou = (inter / union) * 100.0 if union > 0 else 100.0

    # Extract 1-px morphological boundaries
    k3 = np.ones((3, 3), np.uint8)
    pred_u8 = pred_bin.astype(np.uint8) * 255
    gt_u8 = gt_bin.astype(np.uint8) * 255
    pred_b = cv2.morphologyEx(pred_u8, cv2.MORPH_GRADIENT, k3) > 0
    gt_b = cv2.morphologyEx(gt_u8, cv2.MORPH_GRADIENT, k3) > 0

    if not np.any(pred_b) and not np.any(gt_b):
        f_score = 100.0
    elif not np.any(pred_b) or not np.any(gt_b):
        f_score = 0.0
    else:
        k_tol = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * bound_tol_px + 1, 2 * bound_tol_px + 1)
        )
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
    epsilon_ratio: float = 0.008,
    min_points: int = 8,
    max_points: int = 24,
    default_tension: float = 0.5,
) -> Dict[str, Any]:
    """
    High-level convenience function converting a single binary/alpha mask into a sparse
    cubic Bezier control point schema and evaluating its reconstruction quality (J, F, J&F).
    """
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
    """
    Computes the symmetric Root Mean Squared Error (RMSE in pixels) between the evaluated
    parametric cubic Bezier curves of all parts (outer boundary) and the ground-truth mask contour.
    """
    gt_contour = extract_ordered_contour(gt_mask, smooth_sigma=0.5)
    if len(gt_contour) == 0 or not shapes_schemas:
        return 0.0

    h, w = gt_mask.shape[:2]
    pred_mask = rasterize_bezier_shapes(
        shapes_schemas, h, w, samples_per_segment=samples_per_segment
    )
    pred_contour = extract_ordered_contour(pred_mask, smooth_sigma=0.5)
    if len(pred_contour) == 0:
        return float(np.hypot(h, w))

    # Distance from each predicted boundary point to nearest GT contour point and vice versa
    diff_p2g = pred_contour[:, None, :] - gt_contour[None, :, :]
    dist_p2g = np.min(np.hypot(diff_p2g[:, :, 0], diff_p2g[:, :, 1]), axis=1)

    diff_g2p = gt_contour[:, None, :] - pred_contour[None, :, :]
    dist_g2p = np.min(np.hypot(diff_g2p[:, :, 0], diff_g2p[:, :, 1]), axis=1)

    all_sq = np.concatenate([dist_p2g ** 2, dist_g2p ** 2])
    return float(np.sqrt(np.mean(all_sq)))


def rdp_polygon_vectorize(
    mask: np.ndarray,
    epsilon_ratio: float = 0.0022,
    target_num_points: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Baseline 1 (Table 1): RDP Polygon Vectorization (cv2.approxPolyDP).
    Produces piecewise linear polygon vertices (zero tangent handles: in_handle=[0,0], out_handle=[0,0]).
    Requires a much denser vertex count N to approximate curved boundaries without tangent handles.
    """
    h, w = mask.shape[:2]
    bin_mask = (mask > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {"num_control_points": 0, "control_points": [], "reconstructed_mask": np.zeros_like(bin_mask), "metrics": {"J": 0.0, "F": 0.0, "J_and_F": 0.0}}

    main_c = max(contours, key=cv2.contourArea)
    peri = float(cv2.arcLength(main_c, True))

    if target_num_points is not None:
        # Binary search on epsilon to hit target_num_points
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
    metrics = compute_j_and_f_metrics(recon, mask)
    return {
        "num_control_points": len(schema),
        "control_points": schema,
        "reconstructed_mask": recon,
        "metrics": metrics,
    }


def sam2_autotrace_vectorize(
    mask: np.ndarray,
    step_px: float = 11.0,
) -> Dict[str, Any]:
    """
    Baseline 2 (Table 1): SAM 2 Auto-Trace (Nuke Auto-Trace / uniform spline tracing baseline).
    Traces the raw unsmoothed raster contour at uniform arc-length intervals without
    curvature-adaptive vertex pruning or cusp tension retraction, resulting in dense
    control points and stair-stepping oscillation at corners.
    """
    h, w = mask.shape[:2]
    # Unsmoothed raw raster contour (smooth_sigma=0.0) as traced directly from binary mask
    raw_contour = extract_ordered_contour(mask, smooth_sigma=0.0)
    if len(raw_contour) == 0:
        return {"num_control_points": 0, "control_points": [], "reconstructed_mask": np.zeros((h, w), dtype=np.uint8), "metrics": {"J": 0.0, "F": 0.0, "J_and_F": 0.0}}

    diffs = np.diff(np.vstack([raw_contour, raw_contour[:1]]), axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    peri = float(np.sum(seg_lens))
    n_pts = max(16, int(round(peri / max(2.0, step_px))))

    idx_sample = np.linspace(0, len(raw_contour), n_pts, endpoint=False).astype(int)
    pts = raw_contour[idx_sample]

    # Uniform fixed Catmull-Rom tension (no corner cusp detection)
    fixed_tau = np.full(len(pts), 0.5, dtype=np.float64)
    schema = contour_to_hermite_bezier(pts, tensions=fixed_tau)
    recon = rasterize_bezier_shapes([schema], h, w)
    metrics = compute_j_and_f_metrics(recon, mask)
    return {
        "num_control_points": len(schema),
        "control_points": schema,
        "reconstructed_mask": recon,
        "metrics": metrics,
    }


def least_squares_bspline_fit(
    mask: np.ndarray,
    num_segments: int = 20,
    newton_iters: int = 5,
) -> Dict[str, Any]:
    """
    Baseline 3 (Table 1): Iterative Least-Squares Cubic Spline / Bezier Fitting (Schneider 1990).
    Splits the contour into num_segments spans and iteratively fits cubic Bezier handles
    via chord-length parameterization and Newton-Raphson parameter refinement.
    """
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
        if i_end > i_start:
            seg_pts = contour[i_start : i_end + 1]
        else:
            seg_pts = np.vstack([contour[i_start:], contour[: i_end + 1]])

        p0 = seg_pts[0]
        p3 = seg_pts[-1]
        if len(seg_pts) <= 3:
            chord = p3 - p0
            out_handles[s] = chord / 3.0
            in_handles[(s + 1) % num_segments] = -chord / 3.0
            continue

        # Initial chord-length parameterization u in [0, 1]
        d = np.hypot(np.diff(seg_pts[:, 0]), np.diff(seg_pts[:, 1]))
        cum = np.concatenate([[0.0], np.cumsum(d)])
        total_len = cum[-1] + 1e-8
        u = cum / total_len

        b1 = p0 + (p3 - p0) / 3.0
        b2 = p0 + 2.0 * (p3 - p0) / 3.0

        # Iterative least-squares + Newton-Raphson reparameterization
        for _ in range(max(1, newton_iters)):
            u_col = u.reshape(-1, 1)
            om = 1.0 - u_col
            a1 = 3.0 * (om ** 2) * u_col
            a2 = 3.0 * om * (u_col ** 2)
            rhs = seg_pts - ((om ** 3) * p0 + (u_col ** 3) * p3)

            A_mat = np.hstack([a1, a2])  # (M, 2)
            # Solve 2x2 normal equations for [b1, b2]
            ata = A_mat.T @ A_mat + 1e-6 * np.eye(2)
            atb = A_mat.T @ rhs
            sol = np.linalg.solve(ata, atb)
            b1, b2 = sol[0], sol[1]

            # Newton-Raphson parameter update: u <- u - (Q(u) - P) . Q'(u) / |Q'(u)|^2
            q_u = ((om ** 3) * p0 + a1 * b1 + a2 * b2 + (u_col ** 3) * p3)
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

    schema = []
    for i in range(num_segments):
        schema.append(
            {
                "index": i,
                "anchor": [float(anchors[i, 0]), float(anchors[i, 1])],
                "in_handle": [float(in_handles[i, 0]), float(in_handles[i, 1])],
                "out_handle": [float(out_handles[i, 0]), float(out_handles[i, 1])],
                "hermite_tangent": [float(3.0 * out_handles[i, 0]), float(3.0 * out_handles[i, 1])],
                "tension": 0.5,
                "is_corner": False,
            }
        )

    recon = rasterize_bezier_shapes([schema], h, w)
    metrics = compute_j_and_f_metrics(recon, mask)
    return {
        "num_control_points": len(schema),
        "control_points": schema,
        "reconstructed_mask": recon,
        "metrics": metrics,
    }
