"""
Stage 4: End-to-End AlphaToRoto Human-in-the-Loop Conversion & Temporal Propagation Pipeline
Integrates:
  - Stage 1: Tracked Raster Alpha Mask Generation & Master Reference Frame Selection (Eq. 1)
  - Stage 2: Kinematic-Geometric Part Decomposition & 10-15 px Seam Overlap Margin (Eq. 2)
  - Stage 3: O(N) Closed-Form Hermite-to-Bezier Analytical Contour Fitting (Eqs. 3-5)
  - Stage 4: Fixed-Topology Temporal Propagation, Multi-Reference C1 Smoothstep Visibility-Weighted
             Blending (Eqs. 6-7), Chord-Aligned Relative Tangent Rotation (Eq. 8), Non-Destructive
             Artist Keyframe Editing, Oracle Editability Evaluator (NoM), and Foundry Nuke / SVG / JSON Export.
"""

import os
import re
import json
import math
import itertools
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2

from mask_generator import (
    TrackedMaskGenerator,
    select_master_frame,
    compute_object_vector_flow_profile,
    select_representative_master_frames,
)
from roto_geometry import (
    decompose_mask_into_parts,
    extract_ordered_contour,
    select_curvature_adaptive_anchors,
    compute_adaptive_tensions,
    contour_to_hermite_bezier,
    rasterize_bezier_shapes,
    compute_j_and_f_metrics,
    compute_contour_rmse_px,
    compute_signed_area,
    compute_master_optical_flow,
)


def blend_bidirectional_trajectories(
    fwd_tracks: Dict[int, Dict[int, Tuple[float, float, float]]],
    bwd_tracks: Dict[int, Dict[int, Tuple[float, float, float]]],
    start_k: int,
    end_k: int,
) -> Dict[int, Dict[int, Tuple[float, float, float]]]:
    """
    Blends forward and backward trajectories between keyframes start_k and end_k
    using C1 smoothstep s(t)=3t^2-2t^3 and visibility weighting (Eqs. 6-7),
    with exact ground-truth preservation at f == start_k and f == end_k.
    """
    blended: Dict[int, Dict[int, Tuple[float, float, float]]] = {}
    span = end_k - start_k
    all_pts = set(fwd_tracks.keys()).intersection(set(bwd_tracks.keys()))

    for pt_idx in all_pts:
        blended[pt_idx] = {}
        pt_fwd = fwd_tracks[pt_idx]
        pt_bwd = bwd_tracks[pt_idx]
        for f in range(start_k, end_k + 1):
            if f not in pt_fwd or f not in pt_bwd:
                continue
            xf, yf, vf = pt_fwd[f]
            xb, yb, vb = pt_bwd[f]
            if span <= 0:
                blended[pt_idx][f] = (xf, yf, max(vf, vb))
                continue

            t = max(0.0, min(1.0, float(f - start_k) / float(span)))
            s = 3.0 * (t ** 2) - 2.0 * (t ** 3)
            w_a = (1.0 - s) * (max(0.0, float(vf)) + 0.05)
            w_b = s * (max(0.0, float(vb)) + 0.05)
            w_sum = w_a + w_b
            if w_sum < 1e-6:
                bx, by = 0.5 * (xf + xb), 0.5 * (yf + yb)
            else:
                bx = (w_a * xf + w_b * xb) / w_sum
                by = (w_a * yf + w_b * yb) / w_sum

            if f == start_k:
                bx, by = xf, yf
            elif f == end_k:
                bx, by = xb, yb

            blended[pt_idx][f] = (bx, by, max(float(vf), float(vb)))

    return blended


def compute_rotated_tangents(
    ref_pts: List[Dict[str, Any]],
    curr_local_coords: List[Tuple[float, float]],
    scale_min: float = 0.7,
    scale_max: float = 1.4,
) -> List[Dict[str, Tuple[float, float]]]:
    """
    Implements Equation (8):
      Delta h_i(f) = S_i(f) * R(theta_i(f)) * Delta h_i(f_ref)
    Rotates and scales relative left/right tangent handles (lt_rel, rt_rel) based on
    the chord vector rotation and clamped scale ratio between neighboring control points.
    """
    num_pts = len(ref_pts)
    rotated: List[Dict[str, Tuple[float, float]]] = []
    if num_pts == 0:
        return rotated

    for i in range(num_pts):
        lt_dx, lt_dy = ref_pts[i]["lt_rel"]
        rt_dx, rt_dy = ref_pts[i]["rt_rel"]
        fc_dx, fc_dy = ref_pts[i].get("fc_rel", (0.0, 0.0))

        if num_pts >= 3:
            prev_i = (i - 1) % num_pts
            next_i = (i + 1) % num_pts

            v0_x = float(ref_pts[next_i]["local_x"]) - float(ref_pts[prev_i]["local_x"])
            v0_y = float(ref_pts[next_i]["local_y"]) - float(ref_pts[prev_i]["local_y"])
            l0 = math.hypot(v0_x, v0_y)

            vf_x = float(curr_local_coords[next_i][0]) - float(curr_local_coords[prev_i][0])
            vf_y = float(curr_local_coords[next_i][1]) - float(curr_local_coords[prev_i][1])
            lf = math.hypot(vf_x, vf_y)

            if l0 > 1e-4 and lf > 1e-4:
                rot_cos = (v0_x * vf_x + v0_y * vf_y) / (l0 * lf)
                rot_sin = (v0_x * vf_y - v0_y * vf_x) / (l0 * lf)
                scale = max(scale_min, min(scale_max, lf / l0))
            else:
                rot_cos, rot_sin, scale = 1.0, 0.0, 1.0
        else:
            rot_cos, rot_sin, scale = 1.0, 0.0, 1.0

        new_lt = (
            scale * (rot_cos * lt_dx - rot_sin * lt_dy),
            scale * (rot_sin * lt_dx + rot_cos * lt_dy),
        )
        new_rt = (
            scale * (rot_cos * rt_dx - rot_sin * rt_dy),
            scale * (rot_sin * rt_dx + rot_cos * rt_dy),
        )
        new_fc = (
            scale * (rot_cos * fc_dx - rot_sin * fc_dy),
            scale * (rot_sin * fc_dx + rot_cos * fc_dy),
        )
        rotated.append({"lt_rel": new_lt, "rt_rel": new_rt, "fc_rel": new_fc})

    return rotated


def blend_multi_ref_tracks(
    tracks_per_ref: Dict[int, Dict[int, Dict[int, Tuple[float, float, float]]]],
    ref_frames: List[int],
    start_f: int,
    end_f: int,
    blending_mode: str = "c1_vis",
) -> Dict[int, Dict[int, Tuple[float, float, float]]]:
    """
    Combines trajectories propagated from one or more reference keyframes K = {k_1, ..., k_M}
    across [start_f, end_f] using piecewise C1 smoothstep visibility-weighted bidirectional
    blending (Equations 6 and 7) on each interval [k_m, k_{m+1}], and unidirectional
    propagation outside [k_1, k_M].

    blending_mode:
      - "c1_vis": Proposed C1 smoothstep s(t)=3t^2-2t^3 + visibility weighting (Eqs. 6-7)
      - "linear": Ablation baseline w/o visibility weighting or C1 smoothstep (w_fwd=1-t, w_bwd=t)
      - "unidirectional": Forward-only propagation from each reference keyframe
    """
    sorted_refs = sorted(ref_frames)
    first_ref = sorted_refs[0]
    last_ref = sorted_refs[-1]
    first_tracks = tracks_per_ref[first_ref]
    all_pts = sorted(first_tracks.keys())

    combined: Dict[int, Dict[int, Tuple[float, float, float]]] = {
        pt_idx: {} for pt_idx in all_pts
    }

    # 1. Frames before first_ref: use backward track from first_ref
    for f in range(start_f, first_ref + 1):
        for pt_idx in all_pts:
            if f in tracks_per_ref[first_ref][pt_idx]:
                combined[pt_idx][f] = tracks_per_ref[first_ref][pt_idx][f]

    # 2. Intermediate intervals [k_m, k_{m+1}]
    for m in range(len(sorted_refs) - 1):
        k_start = sorted_refs[m]
        k_end = sorted_refs[m + 1]
        if blending_mode == "c1_vis":
            interval_blend = blend_bidirectional_trajectories(
                tracks_per_ref[k_start],
                tracks_per_ref[k_end],
                start_k=k_start,
                end_k=k_end,
            )
            for pt_idx in all_pts:
                for f, val in interval_blend.get(pt_idx, {}).items():
                    combined[pt_idx][f] = val
        elif blending_mode == "linear":
            span = max(1, k_end - k_start)
            for pt_idx in all_pts:
                for f in range(k_start, k_end + 1):
                    xf, yf, vf = tracks_per_ref[k_start][pt_idx][f]
                    xb, yb, vb = tracks_per_ref[k_end][pt_idx][f]
                    t = float(f - k_start) / float(span)
                    if f == k_start:
                        bx, by = xf, yf
                    elif f == k_end:
                        bx, by = xb, yb
                    else:
                        bx = (1.0 - t) * xf + t * xb
                        by = (1.0 - t) * yf + t * yb
                    combined[pt_idx][f] = (bx, by, max(float(vf), float(vb)))
        elif blending_mode == "unidirectional":
            for pt_idx in all_pts:
                for f in range(k_start, k_end + 1):
                    if f == k_end:
                        combined[pt_idx][f] = tracks_per_ref[k_end][pt_idx][f]
                    else:
                        combined[pt_idx][f] = tracks_per_ref[k_start][pt_idx][f]
        else:
            raise ValueError(f"Unsupported blending_mode: {blending_mode}")

    # 3. Frames after last_ref: use forward track from last_ref
    for f in range(last_ref, end_f + 1):
        for pt_idx in all_pts:
            if f in tracks_per_ref[last_ref][pt_idx]:
                combined[pt_idx][f] = tracks_per_ref[last_ref][pt_idx][f]

    return combined


def snap_point_along_normal_to_contour(
    pt_xy: Tuple[float, float],
    normal_xy: Tuple[float, float],
    target_contour: np.ndarray,
    search_radius_px: float = 18.0,
) -> Tuple[float, float]:
    """
    Refines a tracked control point position by snapping it toward the nearest target mask
    contour point within search_radius_px weighted by normal alignment.
    """
    if len(target_contour) == 0:
        return pt_xy

    p = np.array(pt_xy, dtype=np.float64)
    diffs = target_contour - p
    dists = np.hypot(diffs[:, 0], diffs[:, 1])
    min_idx = int(np.argmin(dists))
    if dists[min_idx] <= search_radius_px:
        # Soft snap (0.75 contour + 0.25 tracked) to preserve temporal smoothness while hugging boundary
        snapped = 0.75 * target_contour[min_idx] + 0.25 * p
        return float(snapped[0]), float(snapped[1])
    return float(p[0]), float(p[1])


def propagate_points_from_reference(
    ref_points: List[Dict[str, Any]],
    ref_frame: int,
    frame_numbers: List[int],
    masks: List[np.ndarray],
    frames_rgb: Optional[List[np.ndarray]] = None,
    use_contour_snapping: bool = True,
) -> Dict[int, Dict[int, Tuple[float, float, float]]]:
    """
    Propagates fixed-topology control points (count N fixed at ref_frame) forward and backward
    across frame_numbers.
    Uses pyramidal Lucas-Kanade optical flow on frames_rgb (or distance-transform contour correspondence
    on masks) combined with contour-normal snapping, returning a trajectory dictionary matching
    tracker_core.run_cotracker_point_tracking format:
      { pt_idx: { frame_num: (x, y, visibility) } }
    """
    num_frames = len(frame_numbers)
    frame_to_idx = {f: idx for idx, f in enumerate(frame_numbers)}
    ref_idx = frame_to_idx[ref_frame]
    num_pts = len(ref_points)

    tracks: Dict[int, Dict[int, Tuple[float, float, float]]] = {
        i: {} for i in range(num_pts)
    }
    for i, cp in enumerate(ref_points):
        ax, ay = float(cp["anchor"][0]), float(cp["anchor"][1])
        tracks[i][ref_frame] = (ax, ay, 1.0)

    # Pre-extract ordered contours for each frame mask
    contours_per_frame = [extract_ordered_contour(m, smooth_sigma=1.0) for m in masks]
    grays = None
    if frames_rgb is not None and len(frames_rgb) == num_frames:
        grays = [
            cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) if f.ndim == 3 else f
            for f in frames_rgb
        ]

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    def _step_propagate(from_idx: int, to_idx: int):
        f_from = frame_numbers[from_idx]
        f_to = frame_numbers[to_idx]
        prev_pts = np.array(
            [[tracks[i][f_from][0], tracks[i][f_from][1]] for i in range(num_pts)],
            dtype=np.float32,
        ).reshape(-1, 1, 2)

        vis_arr = np.ones(num_pts, dtype=np.float64)
        if grays is not None:
            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                grays[from_idx], grays[to_idx], prev_pts, None, **lk_params
            )
            # Backward-flow consistency check for visibility confidence v_f in [0, 1]
            back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
                grays[to_idx], grays[from_idx], next_pts, None, **lk_params
            )
            fb_err = np.linalg.norm(prev_pts[:, 0, :] - back_pts[:, 0, :], axis=1)
            pred_xy = next_pts[:, 0, :].astype(np.float64)
            for i in range(num_pts):
                if status[i, 0] == 1 and back_status[i, 0] == 1:
                    vis_arr[i] = float(np.exp(-fb_err[i] / 10.0))
                else:
                    vis_arr[i] = 0.25
                    pred_xy[i] = prev_pts[i, 0, :]
        else:
            # Pure mask sequence propagation via centroid shift + rotational/scale registration
            c_from = contours_per_frame[from_idx]
            c_to = contours_per_frame[to_idx]
            pred_xy = prev_pts[:, 0, :].astype(np.float64).copy()
            if len(c_from) >= 6 and len(c_to) >= 6:
                mean_from = np.mean(c_from, axis=0)
                mean_to = np.mean(c_to, axis=0)
                std_from = np.std(c_from, axis=0) + 1e-6
                std_to = np.std(c_to, axis=0) + 1e-6
                scale = np.clip(std_to / std_from, 0.75, 1.35)
                pred_xy = (pred_xy - mean_from) * scale + mean_to

        target_c = contours_per_frame[to_idx]
        for i in range(num_pts):
            px, py = float(pred_xy[i, 0]), float(pred_xy[i, 1])
            if use_contour_snapping and len(target_c) > 0:
                p_prev = pred_xy[(i - 1) % num_pts]
                p_next = pred_xy[(i + 1) % num_pts]
                chord = p_next - p_prev
                norm_len = float(np.hypot(chord[0], chord[1])) + 1e-8
                normal = (-chord[1] / norm_len, chord[0] / norm_len)
                px, py = snap_point_along_normal_to_contour(
                    (px, py), normal, target_c, search_radius_px=20.0
                )
            tracks[i][f_to] = (px, py, float(np.clip(vis_arr[i], 0.05, 1.0)))

    # Forward propagation from ref_idx
    for idx in range(ref_idx, num_frames - 1):
        _step_propagate(idx, idx + 1)

    # Backward propagation from ref_idx
    for idx in range(ref_idx, 0, -1):
        _step_propagate(idx, idx - 1)

    return tracks


def bake_part_frames_with_rotated_tangents(
    ref_keyframes_map: Dict[int, List[Dict[str, Any]]],
    blended_tracks: Dict[int, Dict[int, Tuple[float, float, float]]],
    frame_numbers: List[int],
    use_tangent_rotation: bool = True,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Applies Chord-Aligned Relative Tangent Rotation (Equation 8) across all frames while
    strictly preserving 100% of artist-edited reference keyframes in ref_keyframes_map.
    When a frame lies between two reference keyframes, rotated tangent handles from both
    neighbors are blended with C1 smoothstep so manual handle edits transition smoothly.
    When use_tangent_rotation=False (ablation study), tangent handles remain fixed to
    the reference keyframe's initial handle offsets without chord rotation.

    Returns:
      { frame_num: List[control_point_schema_dict] }
    """
    sorted_refs = sorted(ref_keyframes_map.keys())
    first_ref = sorted_refs[0]
    num_pts = len(ref_keyframes_map[first_ref])

    def _rotated_handles_from_ref(
        ref_cps: List[Dict[str, Any]],
        curr_local_coords: List[Tuple[float, float]],
    ) -> List[Dict[str, Tuple[float, float]]]:
        ref_pts_for_bridge = [
            {
                "index": i,
                "local_x": float(ref_cps[i]["anchor"][0]),
                "local_y": float(ref_cps[i]["anchor"][1]),
                "lt_rel": (float(ref_cps[i]["in_handle"][0]), float(ref_cps[i]["in_handle"][1])),
                "rt_rel": (float(ref_cps[i]["out_handle"][0]), float(ref_cps[i]["out_handle"][1])),
                "fc_rel": (0.0, 0.0),
            }
            for i in range(num_pts)
        ]
        if use_tangent_rotation:
            return compute_rotated_tangents(ref_pts_for_bridge, curr_local_coords)
        return [
            {
                "lt_rel": ref_pts_for_bridge[i]["lt_rel"],
                "rt_rel": ref_pts_for_bridge[i]["rt_rel"],
                "fc_rel": ref_pts_for_bridge[i]["fc_rel"],
            }
            for i in range(num_pts)
        ]

    baked_frames: Dict[int, List[Dict[str, Any]]] = {}

    for f in frame_numbers:
        # Non-overwriting protection rule: 100% exact preservation on reference keyframes
        if f in ref_keyframes_map:
            baked_frames[f] = [
                {
                    "index": cp["index"],
                    "anchor": list(cp["anchor"]),
                    "in_handle": list(cp["in_handle"]),
                    "out_handle": list(cp["out_handle"]),
                    "is_keyframe": True,
                    "visibility": 1.0,
                }
                for cp in ref_keyframes_map[f]
            ]
            continue

        # Find nearest preceding and succeeding reference keyframes
        base_ref = sorted_refs[0]
        next_ref: Optional[int] = None
        for rk in sorted_refs:
            if rk <= f:
                base_ref = rk
            else:
                next_ref = rk
                break

        curr_local_coords = [
            (float(blended_tracks[i][f][0]), float(blended_tracks[i][f][1]))
            for i in range(num_pts)
        ]

        rot_a = _rotated_handles_from_ref(ref_keyframes_map[base_ref], curr_local_coords)
        if next_ref is not None and base_ref < f < next_ref:
            rot_b = _rotated_handles_from_ref(ref_keyframes_map[next_ref], curr_local_coords)
            t = (f - base_ref) / float(next_ref - base_ref)
            w = 3.0 * t * t - 2.0 * t * t * t
            rotated_tangents = [
                {
                    "lt_rel": (
                        (1.0 - w) * rot_a[i]["lt_rel"][0] + w * rot_b[i]["lt_rel"][0],
                        (1.0 - w) * rot_a[i]["lt_rel"][1] + w * rot_b[i]["lt_rel"][1],
                    ),
                    "rt_rel": (
                        (1.0 - w) * rot_a[i]["rt_rel"][0] + w * rot_b[i]["rt_rel"][0],
                        (1.0 - w) * rot_a[i]["rt_rel"][1] + w * rot_b[i]["rt_rel"][1],
                    ),
                    "fc_rel": (0.0, 0.0),
                }
                for i in range(num_pts)
            ]
        else:
            rotated_tangents = rot_a

        frame_schema: List[Dict[str, Any]] = []
        for i in range(num_pts):
            cx, cy = curr_local_coords[i]
            vis = float(blended_tracks[i][f][2])
            lt_dx, lt_dy = rotated_tangents[i]["lt_rel"]
            rt_dx, rt_dy = rotated_tangents[i]["rt_rel"]
            frame_schema.append(
                {
                    "index": i,
                    "anchor": [cx, cy],
                    "in_handle": [float(lt_dx), float(lt_dy)],
                    "out_handle": [float(rt_dx), float(rt_dy)],
                    "is_keyframe": False,
                    "visibility": vis,
                }
            )
        baked_frames[f] = frame_schema

    return baked_frames


def _mask_centroid(mask: np.ndarray) -> np.ndarray:
    """Computes (x, y) foreground centroid for a binary mask."""
    ys, xs = np.nonzero(mask > 127)
    if len(xs) == 0:
        return np.array([mask.shape[1] / 2.0, mask.shape[0] / 2.0], dtype=np.float64)
    return np.array([float(np.mean(xs)), float(np.mean(ys))], dtype=np.float64)


def build_dynamic_part_masks_sequence(
    masks: List[np.ndarray],
    decomp_parts: List[Dict[str, Any]],
    master_idx: int,
    master_frames: List[int],
    frame_numbers: List[int],
    overlap_margin_px: int = 12,
) -> Dict[int, List[np.ndarray]]:
    """
    Constructs dynamic per-frame part masks across the sequence:
      1. Shifts each primary master part mask by the foreground centroid displacement on frame t
         (when displacement exceeds 5 px), dilates by 25x25, and intersects with the frame t
         foreground mask.
      2. On secondary representative master frames, re-decomposes the frame mask and matches parts
         1-to-1 by relative centroid so large articulation/translation keeps exact part boundaries.
    """
    num_parts = len(decomp_parts)
    if num_parts == 0:
        return {}
    if num_parts <= 1:
        pid0 = int(decomp_parts[0]["part_id"])
        return {pid0: [m.copy() for m in masks]}

    h, w = masks[master_idx].shape[:2]
    c_master = _mask_centroid(masks[master_idx])
    dil_kernel = np.ones((25, 25), np.uint8)
    frame_to_idx = {f: i for i, f in enumerate(frame_numbers)}

    part_masks_all: Dict[int, List[np.ndarray]] = {}
    prim_masks: List[np.ndarray] = []

    for p in decomp_parts:
        pid = int(p["part_id"])
        p_mask_master = p.get("mask") if p.get("mask") is not None else p.get("_part_master_mask")
        if p_mask_master is None:
            p_mask_master = masks[master_idx]
        prim_masks.append(p_mask_master)

        seq_list: List[np.ndarray] = []
        for t_idx, m in enumerate(masks):
            if t_idx == master_idx:
                seq_list.append(p_mask_master.copy())
            else:
                dc = _mask_centroid(m) - c_master
                if float(np.hypot(dc[0], dc[1])) > 5.0:
                    shift_m = np.float32([[1.0, 0.0, float(dc[0])], [0.0, 1.0, float(dc[1])]])
                    part_prior = cv2.warpAffine(
                        p_mask_master, shift_m, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
                    )
                else:
                    part_prior = p_mask_master
                dilated = cv2.dilate(part_prior, dil_kernel)
                cand = cv2.bitwise_and(m, dilated)
                seq_list.append(cand if np.count_nonzero(cand) > 50 else m.copy())
        part_masks_all[pid] = seq_list

    rel_prim = [_mask_centroid(pm) - c_master for pm in prim_masks]
    master_frame_num = frame_numbers[master_idx]

    for sec_f in master_frames:
        if sec_f == master_frame_num or sec_f not in frame_to_idx:
            continue
        sec_idx = frame_to_idx[sec_f]
        sec_decomp = decompose_mask_into_parts(
            masks[sec_idx],
            max_parts=num_parts,
            overlap_margin_px=overlap_margin_px,
        )
        if len(sec_decomp["parts"]) != num_parts:
            continue

        c_sec = _mask_centroid(masks[sec_idx])
        rel_sec = [_mask_centroid(sp["mask"]) - c_sec for sp in sec_decomp["parts"]]

        if num_parts <= 6:
            best_perm = min(
                itertools.permutations(range(num_parts)),
                key=lambda perm: sum(
                    float(np.linalg.norm(rel_prim[i] - rel_sec[perm[i]]))
                    for i in range(num_parts)
                ),
            )
        else:
            used = set()
            perm_list = []
            for i in range(num_parts):
                best_j = min(
                    (j for j in range(num_parts) if j not in used),
                    key=lambda j: float(np.linalg.norm(rel_prim[i] - rel_sec[j])),
                )
                used.add(best_j)
                perm_list.append(best_j)
            best_perm = tuple(perm_list)

        for i, p in enumerate(decomp_parts):
            pid = int(p["part_id"])
            part_masks_all[pid][sec_idx] = sec_decomp["parts"][best_perm[i]]["mask"].copy()

    return part_masks_all


def synthesize_linked_master_keyframe_cps(
    primary_cps: List[Dict[str, Any]],
    primary_part_mask: np.ndarray,
    sec_part_mask: np.ndarray,
    default_tension: float = 0.5,
    cusp_angle_deg: float = 70.0,
) -> List[Dict[str, Any]]:
    """
    Synthesizes topology-locked Bezier control points on a secondary representative master
    frame via monotonic arc-length-fraction + centroid/scale contour mapping, preserving exact
    vertex ordering without crossing and landing directly on the secondary part boundary.
    """
    num_pts = len(primary_cps)
    prim_pts = np.array(
        [[float(cp["anchor"][0]), float(cp["anchor"][1])] for cp in primary_cps],
        dtype=np.float64,
    )
    c_prim = extract_ordered_contour(primary_part_mask, smooth_sigma=1.2)
    c_sec = extract_ordered_contour(sec_part_mask, smooth_sigma=1.2)

    if len(c_sec) < max(6, num_pts) or num_pts < 3:
        shift = _mask_centroid(sec_part_mask) - _mask_centroid(primary_part_mask)
        fallback: List[Dict[str, Any]] = []
        for i, cp in enumerate(primary_cps):
            fallback.append(
                {
                    "index": i,
                    "anchor": [float(cp["anchor"][0] + shift[0]), float(cp["anchor"][1] + shift[1])],
                    "in_handle": list(cp["in_handle"]),
                    "out_handle": list(cp["out_handle"]),
                    "is_keyframe": True,
                    "visibility": 1.0,
                }
            )
        return fallback

    if compute_signed_area(prim_pts) * compute_signed_area(c_sec) < 0.0:
        c_sec = c_sec[::-1].copy()

    ref_cloud = c_prim if len(c_prim) >= 6 else prim_pts
    mean_p = np.mean(ref_cloud, axis=0)
    std_p = np.std(ref_cloud, axis=0) + 1e-6
    mean_s = np.mean(c_sec, axis=0)
    std_s = np.std(c_sec, axis=0) + 1e-6
    scale = np.clip(std_s / std_p, 0.75, 1.35)
    pred = (prim_pts - mean_p) * scale + mean_s

    closed_prim = np.vstack([prim_pts, prim_pts[0:1]])
    diffs = np.diff(closed_prim, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    total_perim = float(np.sum(seg_lens)) + 1e-8
    cum_lens = np.concatenate([[0.0], np.cumsum(seg_lens[:-1])])
    p_fracs = cum_lens / total_perim

    d0 = np.hypot(c_sec[:, 0] - pred[0, 0], c_sec[:, 1] - pred[0, 1])
    start_idx = int(np.argmin(d0))
    c_rolled = np.roll(c_sec, -start_idx, axis=0)
    n_c = len(c_rolled)

    chosen_indices = [0]
    search_half = max(3, int(round(0.08 * n_c)))
    for i in range(1, num_pts):
        target_idx = int(round(float(p_fracs[i]) * n_c))
        lo = max(chosen_indices[-1] + 1, target_idx - search_half)
        hi = min(n_c - (num_pts - i), target_idx + search_half)
        if lo > hi:
            best_idx = max(chosen_indices[-1] + 1, min(n_c - (num_pts - i), lo))
        else:
            window = c_rolled[lo : hi + 1]
            dists = np.hypot(window[:, 0] - pred[i, 0], window[:, 1] - pred[i, 1])
            best_idx = lo + int(np.argmin(dists))
        chosen_indices.append(best_idx)

    sec_anchors = c_rolled[chosen_indices]
    tensions, _ = compute_adaptive_tensions(
        sec_anchors, default_tension=default_tension, cusp_angle_deg=cusp_angle_deg
    )
    sec_cps = contour_to_hermite_bezier(
        sec_anchors,
        tensions=tensions,
        default_tension=default_tension,
        cusp_angle_deg=cusp_angle_deg,
    )
    for cp in sec_cps:
        cp["is_keyframe"] = True
        cp["visibility"] = 1.0
    return sec_cps


class AlphaToRotoPipeline:
    """
    End-to-End AlphaToRoto Human-in-the-Loop Conversion System transforming tracked raster alpha masks
    into artist-editable, spatiotemporally consistent parametric cubic Bezier rotoscoping shapes.
    """

    def __init__(
        self,
        overlap_margin_px: int = 12,
        max_parts: int = 3,
        min_points_per_part: int = 8,
        max_points_per_part: int = 20,
        default_tension: float = 0.5,
        blending_mode: str = "c1_vis",
        use_tangent_rotation: bool = True,
        use_contour_snapping: bool = True,
        auto_multi_master: bool = False,
        flow_motion_threshold_px: float = 22.0,
        max_master_frames: int = 4,
    ):
        self.overlap_margin_px = overlap_margin_px
        self.max_parts = max_parts
        self.min_points_per_part = min_points_per_part
        self.max_points_per_part = max_points_per_part
        self.default_tension = default_tension
        self.blending_mode = blending_mode
        self.use_tangent_rotation = use_tangent_rotation
        self.use_contour_snapping = use_contour_snapping
        self.auto_multi_master = auto_multi_master
        self.flow_motion_threshold_px = flow_motion_threshold_px
        self.max_master_frames = max_master_frames

    @staticmethod
    def select_stage1_masters(
        masks: List[np.ndarray],
        frames_rgb: Optional[List[np.ndarray]] = None,
        frame_numbers: Optional[List[int]] = None,
        auto_multi_master: bool = False,
        flow_motion_threshold_px: float = 22.0,
        max_master_frames: int = 4,
        master_frames_override: Optional[List[int]] = None,
        include_flow_profile: bool = False,
    ) -> Dict[str, Any]:
        """
        Stage 1 master selection (Eq. 1) with optional vector-flow multi-master promotion
        and explicit master_frames_override handling.
        """
        if not masks:
            raise ValueError("masks list cannot be empty.")
        num_frames = len(masks)
        if frame_numbers is None:
            frame_numbers = list(range(1, num_frames + 1))

        use_multi = bool(auto_multi_master) or bool(master_frames_override and len(master_frames_override) >= 2)
        if use_multi:
            master_info = select_representative_master_frames(
                masks,
                frames=frames_rgb,
                frame_numbers=frame_numbers,
                flow_motion_threshold_px=flow_motion_threshold_px,
                max_master_frames=max_master_frames,
            )
            if master_frames_override:
                f_to_idx = {f: i for i, f in enumerate(frame_numbers)}
                chosen = sorted({int(f) for f in master_frames_override if int(f) in f_to_idx})
                if chosen:
                    primary_f = max(
                        chosen,
                        key=lambda f: master_info["per_frame_metrics"][f_to_idx[f]]["score"],
                    )
                    master_info["master_frame"] = int(primary_f)
                    master_info["master_index"] = int(f_to_idx[primary_f])
                    master_info["master_frames"] = chosen
                    master_info["master_indices"] = [f_to_idx[f] for f in chosen]
                    chosen_set = set(chosen)
                    for idx_m, entry in enumerate(master_info["per_frame_metrics"]):
                        entry["is_master"] = frame_numbers[idx_m] in chosen_set
            return master_info

        master_info = select_master_frame(masks, frames=frames_rgb, frame_numbers=frame_numbers)
        if master_frames_override:
            f_to_idx = {f: i for i, f in enumerate(frame_numbers)}
            valid = [int(f) for f in master_frames_override if int(f) in f_to_idx]
            if valid:
                master_info["master_frame"] = valid[0]
                master_info["master_index"] = f_to_idx[valid[0]]

        master_info["master_indices"] = [int(master_info["master_index"])]
        master_info["master_frames"] = [int(master_info["master_frame"])]
        if include_flow_profile:
            flow_prof = compute_object_vector_flow_profile(
                masks, frames=frames_rgb, frame_numbers=frame_numbers
            )
            master_info["large_motion_detected"] = bool(
                flow_prof["total_motion_px"] >= float(flow_motion_threshold_px)
            )
            master_info["flow_profile"] = flow_prof
            for idx_m, entry in enumerate(master_info["per_frame_metrics"]):
                entry["flow_mag_px"] = flow_prof["per_frame_flow_mag"][idx_m]
                entry["p90_flow_mag_px"] = flow_prof["per_frame_p90_mag"][idx_m]
                entry["cumulative_flow_px"] = flow_prof["cumulative_flow"][idx_m]
                entry["is_master"] = idx_m == int(master_info["master_index"])
        else:
            master_info["large_motion_detected"] = False
            master_info["flow_profile"] = None
            for idx_m, entry in enumerate(master_info["per_frame_metrics"]):
                entry["is_master"] = idx_m == int(master_info["master_index"])
        return master_info

    @staticmethod
    def run_stage2_decompose(
        masks: List[np.ndarray],
        master_idx: int,
        frames_rgb: Optional[List[np.ndarray]] = None,
        max_parts: int = 3,
        overlap_margin_px: int = 12,
        defect_depth_thresh_px: float = 10.0,
    ) -> Dict[str, Any]:
        """Stage 2: Kinematic-geometric part decomposition & seam overlap margin (Eq. 2)."""
        master_flow = compute_master_optical_flow(frames_rgb, master_idx)
        decomp = decompose_mask_into_parts(
            masks[master_idx],
            flow=master_flow,
            max_parts=max(1, int(max_parts)),
            overlap_margin_px=max(0, int(overlap_margin_px)),
            defect_depth_thresh_px=float(defect_depth_thresh_px),
        )
        for p in decomp["parts"]:
            if "core_mask" not in p:
                p["core_mask"] = p["mask"].copy()
            p.setdefault("enabled", True)
            p.setdefault("overlap_margin_px", int(decomp["overlap_margin_px"]))
            p.setdefault("boundary_offset_px", 0)
            p.setdefault("max_points", None)
        return decomp

    @staticmethod
    def run_stage3_fit_parts(
        decomp: Dict[str, Any],
        masks: List[np.ndarray],
        master_info: Dict[str, Any],
        frame_numbers: List[int],
        frames_rgb: Optional[List[np.ndarray]] = None,
        min_points: int = 8,
        max_points: int = 16,
        target_points_per_part: Optional[int] = None,
        default_tension: float = 0.5,
        cusp_angle_deg: float = 60.0,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Stage 3: O(N) closed-form Hermite-to-Bezier fitting for all enabled parts (Eqs. 3-5)."""
        enabled_parts = [p for p in decomp["parts"] if p.get("enabled", True)]
        if not enabled_parts:
            raise ValueError("Enable at least one decomposed part in Stage 2.")

        master_idx = int(master_info["master_index"])
        master_frame = int(master_info["master_frame"])
        master_mask = masks[master_idx]
        h, w = master_mask.shape[:2]

        parts_roto: List[Dict[str, Any]] = []
        schemas_master: List[List[Dict[str, Any]]] = []

        for part in enabled_parts:
            p_mask = part["mask"]
            part_max_pts = (
                target_points_per_part
                if target_points_per_part is not None
                else int(part.get("max_points") or max_points)
            )
            contour = extract_ordered_contour(p_mask, smooth_sigma=1.2)
            anchors = select_curvature_adaptive_anchors(
                contour,
                target_num_points=target_points_per_part,
                min_points=int(min_points),
                max_points=part_max_pts,
            )
            tensions, _ = compute_adaptive_tensions(
                anchors,
                default_tension=float(default_tension),
                cusp_angle_deg=float(cusp_angle_deg),
            )
            cps = contour_to_hermite_bezier(
                anchors,
                tensions=tensions,
                default_tension=float(default_tension),
                cusp_angle_deg=float(cusp_angle_deg),
            )
            schemas_master.append(cps)

            frames_map = {
                fn: [
                    {
                        "index": cp["index"],
                        "anchor": list(cp["anchor"]),
                        "in_handle": list(cp["in_handle"]),
                        "out_handle": list(cp["out_handle"]),
                        "hermite_tangent": list(cp.get("hermite_tangent", [0.0, 0.0])),
                        "tension": float(cp.get("tension", default_tension)),
                        "is_corner": bool(cp.get("is_corner", False)),
                    }
                    for cp in cps
                ]
                for fn in frame_numbers
            }
            parts_roto.append(
                {
                    "part_id": part["part_id"],
                    "name": part["name"],
                    "num_control_points": len(cps),
                    "reference_keyframes": {master_frame: cps},
                    "tracks_per_ref": {},
                    "frames": frames_map,
                    "_part_master_mask": p_mask,
                }
            )

        recon = rasterize_bezier_shapes(schemas_master, h, w)
        jf = compute_j_and_f_metrics(recon, master_mask)
        rmse = compute_contour_rmse_px(schemas_master, master_mask)

        roto_project = {
            "width": int(w),
            "height": int(h),
            "frame_numbers": list(frame_numbers),
            "master_frame": master_frame,
            "master_index": master_idx,
            "master_frames": list(master_info.get("master_frames", [master_frame])),
            "flow_profile": master_info.get("flow_profile"),
            "master_selection": master_info,
            "overlap_margin_px": int(decomp["overlap_margin_px"]),
            "seam_gap_ratio_master": float(decomp["seam_gap_ratio"]),
            "cut_lines": decomp["cut_lines"],
            "parts": parts_roto,
            "_masks_ref": masks,
            "_frames_rgb_ref": frames_rgb,
        }
        metrics = {
            "total_points": sum(p["num_control_points"] for p in parts_roto),
            "points_per_part": [p["num_control_points"] for p in parts_roto],
            "J": round(jf["J"], 2),
            "F": round(jf["F"], 2),
            "J_and_F": round(jf["J_and_F"], 2),
            "RMSE_px": round(rmse, 2),
        }
        return roto_project, metrics

    def run_stage4_propagate(
        self,
        roto_project: Dict[str, Any],
        masks: List[np.ndarray],
        frames_rgb: Optional[List[np.ndarray]] = None,
        master_frames: Optional[List[int]] = None,
        blending_mode: Optional[str] = None,
        use_tangent_rotation: Optional[bool] = None,
        use_contour_snapping: Optional[bool] = None,
        default_tension: Optional[float] = None,
        cusp_angle_deg: float = 70.0,
    ) -> Dict[str, Any]:
        """Stage 4: Multi-master linked fixed-topology propagation & C1 baking (Eqs. 6-8)."""
        frame_numbers = roto_project["frame_numbers"]
        master_frame = int(roto_project["master_frame"])
        master_idx = int(roto_project["master_index"])
        frame_set = set(frame_numbers)
        linked_masters = sorted(
            {
                int(f)
                for f in (master_frames if master_frames is not None else roto_project.get("master_frames", [master_frame]))
                if int(f) in frame_set
            }
            | {master_frame}
        )

        b_mode = self.blending_mode if blending_mode is None else blending_mode
        rot_tan = self.use_tangent_rotation if use_tangent_rotation is None else bool(use_tangent_rotation)
        snap = self.use_contour_snapping if use_contour_snapping is None else bool(use_contour_snapping)
        tension = self.default_tension if default_tension is None else float(default_tension)

        part_masks_all = build_dynamic_part_masks_sequence(
            masks,
            roto_project["parts"],
            master_idx,
            linked_masters,
            frame_numbers,
            overlap_margin_px=int(roto_project.get("overlap_margin_px", self.overlap_margin_px)),
        )
        frame_to_idx = {f: i for i, f in enumerate(frame_numbers)}

        for part in roto_project["parts"]:
            pid = int(part["part_id"])
            num_pts = int(part["num_control_points"])
            manual_kfs = {
                int(k): v
                for k, v in part.get("manual_keyframes", {}).items()
                if int(k) in frame_set and isinstance(v, list) and len(v) == num_pts
            }
            if master_frame in manual_kfs:
                master_cps = manual_kfs[master_frame]
            else:
                master_cps = part["reference_keyframes"][master_frame]
            p_mask_master = part.get("_part_master_mask")
            if p_mask_master is None:
                p_mask_master = masks[master_idx]
            part_masks_seq = part_masks_all.get(pid, masks)
            part["_part_masks_seq"] = part_masks_seq

            raw_tracks = propagate_points_from_reference(
                master_cps,
                ref_frame=master_frame,
                frame_numbers=frame_numbers,
                masks=part_masks_seq,
                frames_rgb=frames_rgb,
                use_contour_snapping=snap,
            )
            ref_keyframes_map = {master_frame: master_cps}
            tracks_per_ref = {master_frame: raw_tracks}

            for sec_f in linked_masters:
                if sec_f == master_frame:
                    continue
                if sec_f in manual_kfs:
                    sec_cps = manual_kfs[sec_f]
                else:
                    sec_cps = synthesize_linked_master_keyframe_cps(
                        primary_cps=master_cps,
                        primary_part_mask=p_mask_master,
                        sec_part_mask=part_masks_seq[frame_to_idx[sec_f]],
                        default_tension=tension,
                        cusp_angle_deg=cusp_angle_deg,
                    )
                ref_keyframes_map[sec_f] = sec_cps
                tracks_per_ref[sec_f] = propagate_points_from_reference(
                    sec_cps,
                    ref_frame=sec_f,
                    frame_numbers=frame_numbers,
                    masks=part_masks_seq,
                    frames_rgb=frames_rgb,
                    use_contour_snapping=snap,
                )

            extra_manual = [mf for mf in sorted(manual_kfs.keys()) if mf not in ref_keyframes_map]
            if extra_manual:
                base_blended = blend_multi_ref_tracks(
                    tracks_per_ref,
                    ref_frames=sorted(ref_keyframes_map.keys()),
                    start_f=frame_numbers[0],
                    end_f=frame_numbers[-1],
                    blending_mode=b_mode,
                )
                for mf in extra_manual:
                    edited_cps = manual_kfs[mf]
                    ref_keyframes_map[mf] = edited_cps
                    m_tracks: Dict[int, Dict[int, Tuple[float, float, float]]] = {
                        i: {} for i in range(num_pts)
                    }
                    for i in range(num_pts):
                        ex = float(edited_cps[i]["anchor"][0])
                        ey = float(edited_cps[i]["anchor"][1])
                        bx_mf, by_mf, _ = base_blended[i][mf]
                        dx = ex - float(bx_mf)
                        dy = ey - float(by_mf)
                        for f in frame_numbers:
                            if f == mf:
                                m_tracks[i][f] = (ex, ey, 1.0)
                            else:
                                bx, by, bvis = base_blended[i][f]
                                dist_f = abs(f - mf)
                                vis_edit = float(np.clip(float(bvis) * np.exp(-dist_f / 25.0), 0.1, 1.0))
                                m_tracks[i][f] = (float(bx) + dx, float(by) + dy, vis_edit)
                    tracks_per_ref[mf] = m_tracks

            sorted_refs = sorted(ref_keyframes_map.keys())
            blended_tracks = blend_multi_ref_tracks(
                tracks_per_ref,
                ref_frames=sorted_refs,
                start_f=frame_numbers[0],
                end_f=frame_numbers[-1],
                blending_mode=b_mode,
            )
            part["reference_keyframes"] = ref_keyframes_map
            part["manual_keyframes"] = manual_kfs
            part["tracks_per_ref"] = tracks_per_ref
            part["frames"] = bake_part_frames_with_rotated_tangents(
                ref_keyframes_map,
                blended_tracks,
                frame_numbers,
                use_tangent_rotation=rot_tan,
            )

        roto_project["master_frames"] = linked_masters
        roto_project["_masks_ref"] = masks
        roto_project["_frames_rgb_ref"] = frames_rgb
        return roto_project

    @staticmethod
    def serialize_project(
        roto_project: Optional[Dict[str, Any]],
        include_tracks: bool = False,
        stringify_frame_keys: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Converts an internal roto_project dict into a JSON-serializable payload."""
        if roto_project is None:
            return None
        out: Dict[str, Any] = {}
        for k, v in roto_project.items():
            if k.startswith("_"):
                continue
            if k == "parts":
                out_parts = []
                for p in v:
                    pd: Dict[str, Any] = {}
                    for pk, pv in p.items():
                        if pk.startswith("_") or (not include_tracks and pk == "tracks_per_ref"):
                            continue
                        if stringify_frame_keys and pk in ("reference_keyframes", "manual_keyframes", "frames") and isinstance(pv, dict):
                            pd[pk] = {str(fk): fv for fk, fv in pv.items()}
                        else:
                            pd[pk] = pv
                    pd["manual_keyframe_frames"] = sorted(int(k) for k in p.get("manual_keyframes", {}).keys())
                    out_parts.append(pd)
                out["parts"] = out_parts
            else:
                out[k] = v
        return out

    def build_editable_roto_from_masks(
        self,
        masks: List[np.ndarray],
        frames_rgb: Optional[List[np.ndarray]] = None,
        frame_numbers: Optional[List[int]] = None,
        target_points_per_part: Optional[int] = None,
        master_frames_override: Optional[List[int]] = None,
        auto_multi_master: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Executes Stages 1 -> 2 -> 3 -> 4 on an input sequence of tracked raster alpha masks.
        Returns a complete editable Roto project dictionary supporting non-destructive artist
        keyframe edits, multi-master linking, Nuke Roto node injection, JSON, and SVG export.
        """
        if not masks:
            raise ValueError("masks list cannot be empty.")

        num_frames = len(masks)
        if frame_numbers is None:
            frame_numbers = list(range(1, num_frames + 1))

        use_multi = (
            self.auto_multi_master if auto_multi_master is None else bool(auto_multi_master)
        )
        master_info = self.select_stage1_masters(
            masks,
            frames_rgb=frames_rgb,
            frame_numbers=frame_numbers,
            auto_multi_master=use_multi,
            flow_motion_threshold_px=self.flow_motion_threshold_px,
            max_master_frames=self.max_master_frames,
            master_frames_override=master_frames_override,
            include_flow_profile=False,
        )
        decomp = self.run_stage2_decompose(
            masks,
            master_idx=int(master_info["master_index"]),
            frames_rgb=frames_rgb,
            max_parts=self.max_parts,
            overlap_margin_px=self.overlap_margin_px,
        )
        roto_project, _ = self.run_stage3_fit_parts(
            decomp,
            masks=masks,
            master_info=master_info,
            frame_numbers=frame_numbers,
            frames_rgb=frames_rgb,
            min_points=self.min_points_per_part,
            max_points=self.max_points_per_part,
            target_points_per_part=target_points_per_part,
            default_tension=self.default_tension,
        )
        return self.run_stage4_propagate(
            roto_project,
            masks=masks,
            frames_rgb=frames_rgb,
            master_frames=master_info["master_frames"],
        )

    def apply_artist_keyframe_edit(
        self,
        roto_project: Dict[str, Any],
        part_id: int,
        edited_frame: int,
        edited_control_points: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Registers a manual artist edit at edited_frame as a new reference keyframe in K,
        propagates a bidirectional trajectory from the newly edited keyframe, and re-bakes
        all intermediate frames using C1 Smoothstep Visibility-Weighted Blending (Eqs. 6-7)
        and Chord-Aligned Tangent Rotation (Eq. 8) while preserving 100% of all keyframes in K.
        """
        part = None
        for p in roto_project["parts"]:
            if p["part_id"] == part_id:
                part = p
                break
        if part is None:
            raise ValueError(f"Part ID {part_id} not found in roto_project.")

        if len(edited_control_points) != part["num_control_points"]:
            raise ValueError(
                f"Topology mismatch: expected {part['num_control_points']} control points, "
                f"got {len(edited_control_points)}."
            )

        frame_numbers = roto_project["frame_numbers"]
        masks = roto_project.get("_masks_ref")

        edited_f_int = int(edited_frame)
        edited_copy = [
            {
                "index": i,
                "anchor": [float(cp["anchor"][0]), float(cp["anchor"][1])],
                "in_handle": [float(cp["in_handle"][0]), float(cp["in_handle"][1])],
                "out_handle": [float(cp["out_handle"][0]), float(cp["out_handle"][1])],
                "is_keyframe": True,
            }
            for i, cp in enumerate(edited_control_points)
        ]

        # Store artist-edited keyframe non-destructively & mark as manual keyframe for re-tracking
        part["reference_keyframes"][edited_f_int] = edited_copy
        part.setdefault("manual_keyframes", {})[edited_f_int] = [
            {
                "index": cp["index"],
                "anchor": list(cp["anchor"]),
                "in_handle": list(cp["in_handle"]),
                "out_handle": list(cp["out_handle"]),
                "is_keyframe": True,
            }
            for cp in edited_copy
        ]

        if masks is not None and len(part["reference_keyframes"]) == 1:
            p_mask_master = part.get("_part_master_mask")
            part_masks_seq = part.get("_part_masks_seq")
            if part_masks_seq is None:
                part_masks_seq = (
                    masks
                    if (len(roto_project["parts"]) == 1 or p_mask_master is None)
                    else [
                        cv2.bitwise_and(m, cv2.dilate(p_mask_master, np.ones((25, 25), np.uint8)))
                        if np.count_nonzero(cv2.bitwise_and(m, p_mask_master)) > 30
                        else m
                        for m in masks
                    ]
                )
            new_ref_tracks = propagate_points_from_reference(
                part["reference_keyframes"][edited_f_int],
                ref_frame=edited_f_int,
                frame_numbers=frame_numbers,
                masks=part_masks_seq,
                frames_rgb=roto_project.get("_frames_rgb_ref"),
                use_contour_snapping=self.use_contour_snapping,
            )
        else:
            # Propagate artist delta offset from the currently blended trajectory so multi-master
            # precision on all reference intervals is preserved while edited points shift smoothly.
            new_ref_tracks = {i: {} for i in range(part["num_control_points"])}
            old_frame_cps = part["frames"][edited_f_int]
            for i in range(part["num_control_points"]):
                ex = float(edited_copy[i]["anchor"][0])
                ey = float(edited_copy[i]["anchor"][1])
                dx = ex - float(old_frame_cps[i]["anchor"][0])
                dy = ey - float(old_frame_cps[i]["anchor"][1])
                for f in frame_numbers:
                    if f == edited_f_int:
                        new_ref_tracks[i][f] = (ex, ey, 1.0)
                    else:
                        ox, oy = part["frames"][f][i]["anchor"]
                        vis = float(part["frames"][f][i].get("visibility", 1.0))
                        # Confidence decays gently with temporal distance from edited_frame
                        dist_f = abs(f - edited_f_int)
                        vis_edit = float(np.clip(vis * np.exp(-dist_f / 25.0), 0.1, 1.0))
                        new_ref_tracks[i][f] = (float(ox + dx), float(oy + dy), vis_edit)

        part["tracks_per_ref"][edited_f_int] = new_ref_tracks
        sorted_refs = sorted(part["reference_keyframes"].keys())

        # Equations (6) & (7): Multi-Reference Blending (C1 Smoothstep + Visibility or ablation mode)
        blended_tracks = blend_multi_ref_tracks(
            part["tracks_per_ref"],
            ref_frames=sorted_refs,
            start_f=frame_numbers[0],
            end_f=frame_numbers[-1],
            blending_mode=self.blending_mode,
        )

        # Equation (8) + Non-destructive keyframe locking
        part["frames"] = bake_part_frames_with_rotated_tangents(
            part["reference_keyframes"],
            blended_tracks,
            frame_numbers,
            use_tangent_rotation=self.use_tangent_rotation,
        )
        return roto_project

    @staticmethod
    def compute_temporal_flicker_T(roto_project: Dict[str, Any]) -> float:
        """
        Computes temporal instability / jitter T (Flicker, in px) as the mean second-order
        temporal finite difference norm ||P_{t+1} - 2 P_t + P_{t-1}||_2 across all control
        points and frames.
        """
        frame_numbers = roto_project["frame_numbers"]
        if len(frame_numbers) < 3:
            return 0.0

        accel_norms = []
        for p in roto_project["parts"]:
            n_pts = p["num_control_points"]
            for t_idx in range(1, len(frame_numbers) - 1):
                f_prev = frame_numbers[t_idx - 1]
                f_curr = frame_numbers[t_idx]
                f_next = frame_numbers[t_idx + 1]
                cps_prev = p["frames"][f_prev]
                cps_curr = p["frames"][f_curr]
                cps_next = p["frames"][f_next]
                if len(cps_prev) != n_pts or len(cps_curr) != n_pts or len(cps_next) != n_pts:
                    continue
                for i in range(n_pts):
                    p0 = np.array(cps_prev[i]["anchor"], dtype=np.float64)
                    p1 = np.array(cps_curr[i]["anchor"], dtype=np.float64)
                    p2 = np.array(cps_next[i]["anchor"], dtype=np.float64)
                    accel = p2 - 2.0 * p1 + p0
                    accel_norms.append(float(np.hypot(accel[0], accel[1])))

        return float(np.mean(accel_norms)) if accel_norms else 0.0

    @staticmethod
    def compute_sequence_seam_gap_percent(
        roto_project: Dict[str, Any],
        gt_masks: List[np.ndarray],
    ) -> float:
        """
        Computes the internal seam gap ratio (%) across the sequence when multi-part Bezier
        shapes are rasterized and compared against the interior cut zone of gt_masks.
        """
        if len(roto_project["parts"]) <= 1 or not roto_project.get("cut_lines"):
            return float(roto_project.get("seam_gap_ratio_master", 0.0) * 100.0)

        h, w = roto_project["height"], roto_project["width"]
        frame_numbers = roto_project["frame_numbers"]
        gap_ratios = [float(roto_project.get("seam_gap_ratio_master", 0.0))]

        # Build a 14-px band mask around cut lines to measure internal seam gaps across frames
        cut_band = np.zeros((h, w), dtype=np.uint8)
        for cut in roto_project["cut_lines"]:
            cv2.line(cut_band, tuple(cut["p1"]), tuple(cut["p2"]), 255, thickness=12)

        master_idx = int(roto_project.get("master_index", 0))
        c_master = _mask_centroid(gt_masks[master_idx])

        for idx, f_num in enumerate(frame_numbers):
            gt_m = (gt_masks[idx] > 127).astype(np.uint8) * 255
            total_fg = int(np.count_nonzero(gt_m))
            if total_fg == 0:
                continue
            dc = _mask_centroid(gt_m) - c_master
            if abs(dc[0]) > 0.5 or abs(dc[1]) > 0.5:
                shift_cut = np.float32([[1.0, 0.0, float(dc[0])], [0.0, 1.0, float(dc[1])]])
                frame_cut_band = cv2.warpAffine(
                    cut_band, shift_cut, (w, h), flags=cv2.INTER_NEAREST, borderValue=0
                )
            else:
                frame_cut_band = cut_band
            # Erode outer boundary by 5 px so only internal seam gaps (not outer contour curvature) are measured
            interior_fg = cv2.erode(gt_m, np.ones((9, 9), np.uint8))
            schemas = [p["frames"][f_num] for p in roto_project["parts"]]
            pred_union = rasterize_bezier_shapes(schemas, h, w)
            seam_zone_fg = (interior_fg > 0) & (frame_cut_band > 0)
            missing_in_seam = np.count_nonzero(seam_zone_fg & (pred_union == 0))
            gap_ratios.append(float(missing_in_seam) / float(total_fg))

        return float(np.mean(gap_ratios) * 100.0)

    @classmethod
    def evaluate_sequence_metrics(
        cls,
        roto_project: Dict[str, Any],
        gt_masks: List[np.ndarray],
    ) -> Dict[str, float]:
        """
        Evaluates full sequence metrics for Tables 1, 2, and 3:
          - J (%): mean region similarity (IoU)
          - F (%): mean boundary F-score
          - J_and_F (%): mean (J + F) / 2
          - RMSE_px: mean contour Root Mean Squared Error in pixels
          - T_flicker: temporal second-order acceleration jitter
          - dJ_decay: long-term J decay from reference keyframe to sequence end
          - Gap_percent: internal seam gap percentage
        """
        h, w = roto_project["height"], roto_project["width"]
        frame_numbers = roto_project["frame_numbers"]
        j_list, f_list, jf_list, rmse_list = [], [], [], []

        for idx, f_num in enumerate(frame_numbers):
            schemas = [p["frames"][f_num] for p in roto_project["parts"]]
            pred_m = rasterize_bezier_shapes(schemas, h, w)
            m = compute_j_and_f_metrics(pred_m, gt_masks[idx])
            rmse = compute_contour_rmse_px(schemas, gt_masks[idx])
            j_list.append(m["J"])
            f_list.append(m["F"])
            jf_list.append(m["J_and_F"])
            rmse_list.append(rmse)

        master_idx = int(roto_project.get("master_index", 0))
        j_ref = j_list[master_idx] if 0 <= master_idx < len(j_list) else j_list[0]
        # Measure drift against worst/furthest boundary frame
        j_end = min(j_list[0], j_list[-1])
        dj_decay = max(0.0, float(j_ref - j_end))

        return {
            "J": float(np.mean(j_list)),
            "F": float(np.mean(f_list)),
            "J_and_F": float(np.mean(jf_list)),
            "RMSE_px": float(np.mean(rmse_list)),
            "T_flicker": cls.compute_temporal_flicker_T(roto_project),
            "dJ_decay": dj_decay,
            "Gap_percent": cls.compute_sequence_seam_gap_percent(roto_project, gt_masks),
        }

    def simulate_oracle_editability_nom(
        self,
        roto_project: Dict[str, Any],
        gt_masks: List[np.ndarray],
        target_f_score: float = 90.0,
        max_modifications: int = 30,
        propagate_keyframe_edits: bool = False,
    ) -> Dict[str, Any]:
        """
        Simulates an Oracle Artist Agent that iteratively identifies the control point with the
        largest distance to the ground-truth contour and drags it to the true contour boundary
        until the sequence boundary accuracy F >= target_f_score.
        When propagate_keyframe_edits=True, editing a control point on a keyframe triggers
        multi-reference bidirectional C1 smoothstep blending (apply_artist_keyframe_edit)
        so intermediate frames benefit from the artist's correction.

        Returns:
          {
            "initial_J": float,
            "initial_F": float,
            "final_J": float,
            "final_F": float,
            "NoM": int,
            "total_control_points": int
          }
        """
        h, w = roto_project["height"], roto_project["width"]
        frame_numbers = roto_project["frame_numbers"]
        total_cps = sum(p["num_control_points"] for p in roto_project["parts"])

        def _eval_frame(f_idx: int) -> Dict[str, float]:
            f_num = frame_numbers[f_idx]
            schemas = [p["frames"][f_num] for p in roto_project["parts"]]
            pred_m = rasterize_bezier_shapes(schemas, h, w)
            return compute_j_and_f_metrics(pred_m, gt_masks[f_idx])

        init_metrics = [_eval_frame(i) for i in range(len(frame_numbers))]
        init_j = float(np.mean([m["J"] for m in init_metrics]))
        init_f = float(np.mean([m["F"] for m in init_metrics]))

        nom_count = 0
        if propagate_keyframe_edits:
            # Iterative multi-ref keyframe oracle: always pick the worst frame in the sequence
            while nom_count < max_modifications:
                curr_metrics = [_eval_frame(i) for i in range(len(frame_numbers))]
                worst_f_idx = int(np.argmin([m["F"] for m in curr_metrics]))
                if curr_metrics[worst_f_idx]["F"] >= target_f_score:
                    break

                f_num = frame_numbers[worst_f_idx]
                gt_c = extract_ordered_contour(gt_masks[worst_f_idx], smooth_sigma=0.5)
                if len(gt_c) == 0:
                    break

                worst_dist = -1.0
                worst_part = None
                worst_pt_idx = -1
                best_target_xy = None

                for p in roto_project["parts"]:
                    for cp in p["frames"][f_num]:
                        pt = np.array(cp["anchor"], dtype=np.float64)
                        dists = np.hypot(gt_c[:, 0] - pt[0], gt_c[:, 1] - pt[1])
                        min_d_idx = int(np.argmin(dists))
                        if dists[min_d_idx] > worst_dist:
                            worst_dist = float(dists[min_d_idx])
                            worst_part = p
                            worst_pt_idx = cp["index"]
                            best_target_xy = gt_c[min_d_idx]

                if worst_part is None or worst_dist < 1.0:
                    break

                edited_cps = [
                    {
                        "index": cp["index"],
                        "anchor": list(cp["anchor"]),
                        "in_handle": list(cp["in_handle"]),
                        "out_handle": list(cp["out_handle"]),
                    }
                    for cp in worst_part["frames"][f_num]
                ]
                edited_cps[worst_pt_idx]["anchor"] = [
                    float(best_target_xy[0]),
                    float(best_target_xy[1]),
                ]
                self.apply_artist_keyframe_edit(
                    roto_project,
                    part_id=worst_part["part_id"],
                    edited_frame=f_num,
                    edited_control_points=edited_cps,
                )
                nom_count += 1
        else:
            for f_idx, f_num in enumerate(frame_numbers):
                gt_c = extract_ordered_contour(gt_masks[f_idx], smooth_sigma=0.5)
                if len(gt_c) == 0:
                    continue
                m_curr = _eval_frame(f_idx)
                while m_curr["F"] < target_f_score and nom_count < max_modifications:
                    worst_dist = -1.0
                    worst_part = None
                    worst_pt_idx = -1
                    best_target_xy = None

                    for p in roto_project["parts"]:
                        for cp in p["frames"][f_num]:
                            pt = np.array(cp["anchor"], dtype=np.float64)
                            dists = np.hypot(gt_c[:, 0] - pt[0], gt_c[:, 1] - pt[1])
                            min_d_idx = int(np.argmin(dists))
                            if dists[min_d_idx] > worst_dist:
                                worst_dist = float(dists[min_d_idx])
                                worst_part = p
                                worst_pt_idx = cp["index"]
                                best_target_xy = gt_c[min_d_idx]

                    if worst_part is None or worst_dist < 1.0:
                        break

                    worst_part["frames"][f_num][worst_pt_idx]["anchor"] = [
                        float(best_target_xy[0]),
                        float(best_target_xy[1]),
                    ]
                    nom_count += 1
                    m_curr = _eval_frame(f_idx)

        final_metrics = [_eval_frame(i) for i in range(len(frame_numbers))]
        final_j = float(np.mean([m["J"] for m in final_metrics]))
        final_f = float(np.mean([m["F"] for m in final_metrics]))

        return {
            "initial_J": init_j,
            "initial_F": init_f,
            "final_J": final_j,
            "final_F": final_f,
            "NoM": nom_count,
            "total_control_points": total_cps,
        }

    @staticmethod
    def export_to_json(roto_project: Dict[str, Any], output_json_path: str):
        """Exports the parametric cubic Bezier Roto project to a clean JSON file."""
        serializable = AlphaToRotoPipeline.serialize_project(
            roto_project, include_tracks=True, stringify_frame_keys=False
        )
        os.makedirs(os.path.dirname(os.path.abspath(output_json_path)), exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)

    @staticmethod
    def export_frame_to_svg(roto_project: Dict[str, Any], frame_num: int, output_svg_path: str):
        """
        Exports all decomposed cubic Bezier Roto shapes and their tangent handles at frame_num
        to a standard SVG vector file for visual inspection or vector editor import.
        """
        w, h = roto_project["width"], roto_project["height"]
        palette = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

        lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
            f'  <rect width="{w}" height="{h}" fill="#12161F"/>',
        ]

        for p_i, part in enumerate(roto_project["parts"]):
            color = palette[p_i % len(palette)]
            cps = part["frames"][frame_num]
            n = len(cps)
            if n < 2:
                continue

            path_cmds = [f"M {cps[0]['anchor'][0]:.2f},{cps[0]['anchor'][1]:.2f}"]
            for i in range(n):
                cp0 = cps[i]
                cp1 = cps[(i + 1) % n]
                b1x = cp0["anchor"][0] + cp0["out_handle"][0]
                b1y = cp0["anchor"][1] + cp0["out_handle"][1]
                b2x = cp1["anchor"][0] + cp1["in_handle"][0]
                b2y = cp1["anchor"][1] + cp1["in_handle"][1]
                b3x, b3y = cp1["anchor"][0], cp1["anchor"][1]
                path_cmds.append(
                    f"C {b1x:.2f},{b1y:.2f} {b2x:.2f},{b2y:.2f} {b3x:.2f},{b3y:.2f}"
                )
            path_cmds.append("Z")
            d_str = " ".join(path_cmds)
            lines.append(
                f'  <path d="{d_str}" fill="{color}" fill-opacity="0.28" stroke="{color}" stroke-width="2.2"/>'
            )

            # Draw tangent handles and control point anchors
            for cp in cps:
                ax, ay = cp["anchor"]
                ix, iy = ax + cp["in_handle"][0], ay + cp["in_handle"][1]
                ox, oy = ax + cp["out_handle"][0], ay + cp["out_handle"][1]
                lines.append(
                    f'  <line x1="{ix:.2f}" y1="{iy:.2f}" x2="{ox:.2f}" y2="{oy:.2f}" stroke="#F0E442" stroke-width="1.0"/>'
                )
                lines.append(f'  <circle cx="{ix:.2f}" cy="{iy:.2f}" r="2.5" fill="#56B4E9"/>')
                lines.append(f'  <circle cx="{ox:.2f}" cy="{oy:.2f}" r="2.5" fill="#E69F00"/>')
                lines.append(
                    f'  <rect x="{ax - 3.0:.2f}" y="{ay - 3.0:.2f}" width="6" height="6" fill="#FFFFFF" stroke="{color}" stroke-width="1.5"/>'
                )

        lines.append("</svg>")
        os.makedirs(os.path.dirname(os.path.abspath(output_svg_path)), exist_ok=True)
        with open(output_svg_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    @staticmethod
    def inject_into_nuke_roto_node(
        roto_project: Dict[str, Any],
        roto_node: Optional[Any] = None,
        key_step: int = 1,
    ) -> Any:
        """
        Directly creates native cubic Bezier shapes inside a live Foundry Nuke Roto/RotoPaint
        node using Nuke's `nuke.rotopaint` (`_curveknob`) API, applying the subpixel vertical
        coordinate transform (y_nuke = H_orig - y_img) and setting center + relative left/right
        tangent handles across the frame range.
        """
        import nuke
        import nuke.rotopaint as rp

        if roto_node is None:
            roto_node = nuke.createNode("Roto", inpanel=False)

        curves_knob = roto_node["curves"]
        root_layer = getattr(curves_knob, "rootLayer", getattr(curves_knob, "root", None))
        h = float(roto_project["height"])
        frame_numbers = roto_project["frame_numbers"]

        for part in roto_project["parts"]:
            shape = rp.Shape(curves_knob)
            shape.name = part["name"]
            num_pts = part["num_control_points"]
            ref_set = set(part["reference_keyframes"].keys())

            for _ in range(num_pts):
                cp = rp.ShapeControlPoint()
                shape.append(cp)

            for idx_f, f_num in enumerate(frame_numbers):
                if (idx_f % max(1, key_step) != 0) and (f_num not in ref_set) and (f_num != frame_numbers[-1]):
                    continue
                f_cps = part["frames"][f_num]
                for i in range(num_pts):
                    cp_data = f_cps[i]
                    nuke_cp = shape[i]

                    # Subpixel Y-axis flip from image top-left to Nuke bottom-left
                    x_nuke = float(cp_data["anchor"][0])
                    y_nuke = h - float(cp_data["anchor"][1])
                    lt_dx = float(cp_data["in_handle"][0])
                    lt_dy = -float(cp_data["in_handle"][1])
                    rt_dx = float(cp_data["out_handle"][0])
                    rt_dy = -float(cp_data["out_handle"][1])

                    nuke_cp.center.addPositionKey(float(f_num), (x_nuke, y_nuke, 0.0))
                    nuke_cp.leftTangent.addPositionKey(float(f_num), (lt_dx, lt_dy, 0.0))
                    nuke_cp.rightTangent.addPositionKey(float(f_num), (rt_dx, rt_dy, 0.0))
                    nuke_cp.featherCenter.addPositionKey(float(f_num), (0.0, 0.0, 0.0))

            root_layer.append(shape)

        curves_knob.changed()
        return roto_node

    @staticmethod
    def export_to_nuke_script_py(
        roto_project: Dict[str, Any],
        output_py_path: str,
        key_step: int = 1,
        attach_autoroto_tab: bool = True,
    ):
        """
        Generates a standalone Python script that can be executed inside Foundry Nuke's
        Script Editor (or `nuke -t`) to construct the animated multi-part cubic Bezier Roto
        node with all keyframes, tangent handles, and (when available) the AlphaToRoto Tab 1 UI.
        """
        repo_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
        serializable = {
            "width": roto_project["width"],
            "height": roto_project["height"],
            "frame_numbers": roto_project["frame_numbers"],
            "master_frame": roto_project["master_frame"],
            "overlap_margin_px": int(roto_project.get("overlap_margin_px", 12)),
            "parts": AlphaToRotoPipeline.serialize_project(
                roto_project, include_tracks=False, stringify_frame_keys=True
            )["parts"],
        }
        payload_json = json.dumps(serializable)

        script = f'''"""
Auto-generated Foundry Nuke Roto Node Builder Script (EditableAlpha)
Run inside Nuke Script Editor to instantiate the artist-editable multi-part Bezier Roto node.
"""
import os
import sys
import json
import nuke
import nuke.rotopaint as rp

REPO_DIR = {repr(repo_dir)}
ATTACH_AUTOROTO_TAB = {bool(attach_autoroto_tab)}
ROTO_DATA = json.loads({repr(payload_json)})

def build_editablealpha_node(key_step={int(key_step)}, attach_tab=ATTACH_AUTOROTO_TAB):
    roto_node = nuke.createNode("Roto", inpanel=False)
    roto_node.setName("EditableAlpha_EditableParts")
    curves_knob = roto_node["curves"]
    root_layer = getattr(curves_knob, "rootLayer", getattr(curves_knob, "root", None))
    w = int(ROTO_DATA["width"])
    h = float(ROTO_DATA["height"])
    frame_numbers = ROTO_DATA["frame_numbers"]

    try:
        fmt_name = f"EditableAlpha_{{w}}x{{int(h)}}"
        nuke.addFormat(f"{{w}} {{int(h)}} 1 {{fmt_name}}")
        roto_node["format"].setValue(fmt_name)
    except Exception:
        pass

    all_ref_frames = set()
    for part in ROTO_DATA["parts"]:
        shape = rp.Shape(curves_knob)
        shape.name = part["name"]
        num_pts = int(part["num_control_points"])
        ref_set = set(int(k) for k in part["reference_keyframes"].keys())
        all_ref_frames.update(ref_set)

        for _ in range(num_pts):
            shape.append(rp.ShapeControlPoint())

        for idx_f, f_num in enumerate(frame_numbers):
            if (idx_f % max(1, key_step) != 0) and (f_num not in ref_set) and (f_num != frame_numbers[-1]):
                continue
            f_cps = part["frames"][str(f_num)]
            for i in range(num_pts):
                cp_data = f_cps[i]
                nuke_cp = shape[i]
                x_nuke = float(cp_data["anchor"][0])
                y_nuke = h - float(cp_data["anchor"][1])
                lt_dx = float(cp_data["in_handle"][0])
                lt_dy = -float(cp_data["in_handle"][1])
                rt_dx = float(cp_data["out_handle"][0])
                rt_dy = -float(cp_data["out_handle"][1])

                nuke_cp.center.addPositionKey(float(f_num), (x_nuke, y_nuke, 0.0))
                nuke_cp.leftTangent.addPositionKey(float(f_num), (lt_dx, lt_dy, 0.0))
                nuke_cp.rightTangent.addPositionKey(float(f_num), (rt_dx, rt_dy, 0.0))
                nuke_cp.featherCenter.addPositionKey(float(f_num), (0.0, 0.0, 0.0))

        root_layer.append(shape)

    curves_knob.changed()

    if attach_tab:
        try:
            if os.path.isdir(REPO_DIR) and REPO_DIR not in sys.path:
                sys.path.insert(0, REPO_DIR)
            import nuke_bridge
            nuke_bridge.setup_autoroto_knobs(roto_node)
            sorted_refs = sorted(all_ref_frames) if all_ref_frames else [int(ROTO_DATA.get("master_frame", frame_numbers[0]))]
            if roto_node.knob("ar_ref_frame"):
                roto_node["ar_ref_frame"].setValue(", ".join(str(k) for k in sorted_refs))
            if roto_node.knob("ar_start_frame"):
                roto_node["ar_start_frame"].setValue(int(frame_numbers[0]))
            if roto_node.knob("ar_end_frame"):
                roto_node["ar_end_frame"].setValue(int(frame_numbers[-1]))
            if roto_node.knob("ar_overlap_px"):
                roto_node["ar_overlap_px"].setValue(int(ROTO_DATA.get("overlap_margin_px", 12)))
            if roto_node.knob("ar_max_parts"):
                roto_node["ar_max_parts"].setValue(max(1, len(ROTO_DATA["parts"])))
            if roto_node.knob("ar_status"):
                roto_node["ar_status"].setValue(
                    f"Loaded {{len(ROTO_DATA['parts'])}} EditableAlpha part(s) (Master F{{ROTO_DATA.get('master_frame', 1)}})."
                )
            nuke_bridge.hide_intermediate_native_tabs(roto_node)
            nuke_bridge.make_autoroto_first_tab(roto_node.name(), focus=True)
        except Exception:
            pass

    return roto_node

build_alphatoroto_node = build_editablealpha_node
build_autoroto_node = build_editablealpha_node

if __name__ == "__main__":
    build_editablealpha_node()
'''
        os.makedirs(os.path.dirname(os.path.abspath(output_py_path)), exist_ok=True)
        with open(output_py_path, "w", encoding="utf-8") as f:
            f.write(script)

    @staticmethod
    def format_nuke_nk_node(
        roto_project: Dict[str, Any],
        key_step: int = 1,
        attach_autoroto_tab: bool = True,
    ) -> str:
        """
        Formats the multi-part animated cubic Bezier Roto project into a native Foundry Nuke
        (.nk / clipboard) node script so artists can paste directly via Ctrl+V or drag-and-drop.
        """
        w = int(roto_project["width"])
        h = float(roto_project["height"])
        h_int = int(h)
        cx, cy = w * 0.5, h * 0.5
        frame_numbers = [int(f) for f in roto_project["frame_numbers"]]
        all_ref_frames = set()
        curvegroup_blocks: List[str] = []

        def _comp(vals: List[float]) -> str:
            return "{" + " ".join("{%.4f}" % v for v in vals) + "}"

        def _pt_line(xs: List[float], ys: List[float]) -> str:
            return "        {" + _comp(xs) + " " + _comp(ys) + " {{0 -}}}"

        for part in roto_project["parts"]:
            raw_name = str(part.get("name", "Part_01"))
            shape_name = re.sub(r"[^A-Za-z0-9_]", "_", raw_name) or "Part_01"
            num_pts = int(part["num_control_points"])
            ref_set = {int(k) for k in part["reference_keyframes"].keys()}
            all_ref_frames.update(ref_set)

            sampled_frames = [
                f_num
                for idx_f, f_num in enumerate(frame_numbers)
                if (idx_f % max(1, int(key_step)) == 0)
                or (f_num in ref_set)
                or (f_num == frame_numbers[-1])
            ]
            if not sampled_frames and frame_numbers:
                sampled_frames = [frame_numbers[0]]

            frames_hdr = " ".join(str(f) for f in sampled_frames)
            px_point_lines: List[str] = []
            p_empty_blocks: List[str] = []

            for i in range(num_pts):
                lt_x, lt_y = [], []
                c_x, c_y = [], []
                rt_x, rt_y = [], []
                for f_num in sampled_frames:
                    f_cps = part["frames"].get(f_num)
                    if f_cps is None:
                        f_cps = part["frames"][str(f_num)]
                    cp = f_cps[i]
                    c_x.append(float(cp["anchor"][0]))
                    c_y.append(h - float(cp["anchor"][1]))
                    lt_x.append(float(cp["in_handle"][0]))
                    lt_y.append(-float(cp["in_handle"][1]))
                    rt_x.append(float(cp["out_handle"][0]))
                    rt_y.append(-float(cp["out_handle"][1]))

                px_point_lines.append(_pt_line(lt_x, lt_y))
                px_point_lines.append(_pt_line(c_x, c_y))
                px_point_lines.append(_pt_line(rt_x, rt_y))
                p_empty_blocks.extend(["        {}", "        {}", "        {}"])

            cg_block = "\n".join(
                [
                    f"    {{curvegroup {shape_name} 512 bezier",
                    "     {{cc",
                    "       {f 8192}",
                    "       {px",
                    f"        {{{frames_hdr}}}",
                    *px_point_lines,
                    "       }}",
                    "      {cc",
                    "       {f 8192}",
                    "       {p",
                    *p_empty_blocks,
                    "       }}}",
                    "     {tx 1 0}",
                    f"     {{a osw 10 osf 0 str 1 spx {cx:.4f} spy {cy:.4f} sf 1 sb 1}}}}",
                ]
            )
            curvegroup_blocks.append(cg_block)

        sorted_refs = (
            sorted(all_ref_frames)
            if all_ref_frames
            else [int(roto_project.get("master_frame", frame_numbers[0] if frame_numbers else 1))]
        )
        refs_str = ", ".join(str(k) for k in sorted_refs)
        start_f = int(frame_numbers[0]) if frame_numbers else 1
        end_f = int(frame_numbers[-1]) if frame_numbers else 1
        overlap_px = int(roto_project.get("overlap_margin_px", 12))
        num_parts = max(1, len(roto_project["parts"]))
        master_f = int(roto_project.get("master_frame", start_f))

        user_knobs_section = ""
        if attach_autoroto_tab:
            status_txt = f"Loaded {num_parts} EditableAlpha part(s) (Master F{master_f})."
            user_knobs_section = "\n".join(
                [
                    " addUserKnob {20 EditableAlpha l EditableAlpha}",
                    ' addUserKnob {26 ar_title l "" +STARTLINE T "<font size=4 color=\\"#e58934\\"><b>EditableAlpha</b></font>"}',
                    ' addUserKnob {1 ar_ref_frame l "Ref Frame"}',
                    f' ar_ref_frame "{refs_str}"',
                    ' addUserKnob {3 ar_start_frame l "Start Frame"}',
                    f" ar_start_frame {start_f}",
                    ' addUserKnob {3 ar_end_frame l "End Frame"}',
                    f" ar_end_frame {end_f}",
                    ' addUserKnob {3 ar_key_step l "Key Step"}',
                    f" ar_key_step {max(1, int(key_step))}",
                    ' addUserKnob {3 ar_max_parts l "Max Parts"}',
                    f" ar_max_parts {num_parts}",
                    ' addUserKnob {3 ar_overlap_px l "Seam Overlap (px)"}',
                    f" ar_overlap_px {overlap_px}",
                    " addUserKnob {1 ar_status l Status}",
                    f' ar_status "{status_txt}"',
                ]
            )

        lines = [
            "set cut_paste_input [stack 0]",
            "version 17.1 v2",
            "push $cut_paste_input",
            "Roto {",
            " output alpha",
            " curves {{{v x3f99999a}",
            "  {f 0}",
            "  {n",
            "   {layer Root",
            "    {f 0}",
            f"    {{t {cx:.4f} {cy:.4f}}}",
            "    {a}",
            *curvegroup_blocks,
            "   }}}}",
            f' format "{w} {h_int} 0 0 {w} {h_int} 1 EditableAlpha_{w}x{h_int}"',
            " name EditableAlpha_EditableParts",
            " selected true",
        ]
        if user_knobs_section:
            lines.append(user_knobs_section)
        lines.append("}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def export_to_nuke_nk(
        roto_project: Dict[str, Any],
        output_nk_path: str,
        key_step: int = 1,
        attach_autoroto_tab: bool = True,
    ) -> str:
        """Exports the parametric cubic Bezier Roto project as a native Foundry Nuke .nk node."""
        nk_text = AlphaToRotoPipeline.format_nuke_nk_node(
            roto_project,
            key_step=key_step,
            attach_autoroto_tab=attach_autoroto_tab,
        )
        os.makedirs(os.path.dirname(os.path.abspath(output_nk_path)), exist_ok=True)
        with open(output_nk_path, "w", encoding="utf-8") as f:
            f.write(nk_text)
        return nk_text


# Backwards-compatibility & brand aliases
EditableAlphaPipeline = AlphaToRotoPipeline
AutoRotoPipeline = AlphaToRotoPipeline
