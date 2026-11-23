"""
Stage 4: End-to-End AutoRoto Human-in-the-Loop Conversion & Temporal Propagation Pipeline
Integrates:
  - Stage 1: Tracked Raster Alpha Mask Generation & Master Reference Frame Selection (Eq. 1)
  - Stage 2: Kinematic-Geometric Part Decomposition & 10-15 px Seam Overlap Margin (Eq. 2)
  - Stage 3: O(N) Closed-Form Hermite-to-Bezier Analytical Contour Fitting (Eqs. 3-5)
  - Stage 4: Fixed-Topology Temporal Propagation, Multi-Reference C1 Smoothstep Visibility-Weighted
             Blending (Eqs. 6-7), Chord-Aligned Relative Tangent Rotation (Eq. 8), Non-Destructive
             Artist Keyframe Editing, Oracle Editability Evaluator (NoM), and Foundry Nuke / SVG / JSON Export.
"""

import os
import json
import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import cv2

from mask_generator import TrackedMaskGenerator, select_master_frame
from part_decomposition import decompose_mask_into_parts
from hermite_bezier_converter import (
    extract_ordered_contour,
    select_curvature_adaptive_anchors,
    contour_to_hermite_bezier,
    rasterize_bezier_shapes,
    compute_j_and_f_metrics,
)
from nuke_bridge import blend_bidirectional_trajectories


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
) -> Dict[int, Dict[int, Tuple[float, float, float]]]:
    """
    Combines trajectories propagated from one or more reference keyframes K = {k_1, ..., k_M}
    across [start_f, end_f] using piecewise C1 smoothstep visibility-weighted bidirectional
    blending (Equations 6 and 7) on each interval [k_m, k_{m+1}], and unidirectional
    propagation outside [k_1, k_M].
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

    # 2. Intermediate intervals [k_m, k_{m+1}]: C1 smoothstep visibility-weighted blend
    for m in range(len(sorted_refs) - 1):
        k_start = sorted_refs[m]
        k_end = sorted_refs[m + 1]
        interval_blend = blend_bidirectional_trajectories(
            tracks_per_ref[k_start],
            tracks_per_ref[k_end],
            start_k=k_start,
            end_k=k_end,
        )
        for pt_idx in all_pts:
            for f, val in interval_blend.get(pt_idx, {}).items():
                combined[pt_idx][f] = val

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
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Applies Chord-Aligned Relative Tangent Rotation (Equation 8) across all frames while
    strictly preserving 100% of artist-edited reference keyframes in ref_keyframes_map.

    Returns:
      { frame_num: List[control_point_schema_dict] }
    """
    sorted_refs = sorted(ref_keyframes_map.keys())
    first_ref = sorted_refs[0]
    num_pts = len(ref_keyframes_map[first_ref])

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

        # Find nearest preceding reference keyframe for relative tangent rotation base
        base_ref = sorted_refs[0]
        for rk in sorted_refs:
            if rk <= f:
                base_ref = rk
            else:
                break

        base_cps = ref_keyframes_map[base_ref]
        ref_pts_for_bridge = [
            {
                "index": i,
                "local_x": float(base_cps[i]["anchor"][0]),
                "local_y": float(base_cps[i]["anchor"][1]),
                "lt_rel": (float(base_cps[i]["in_handle"][0]), float(base_cps[i]["in_handle"][1])),
                "rt_rel": (float(base_cps[i]["out_handle"][0]), float(base_cps[i]["out_handle"][1])),
                "fc_rel": (0.0, 0.0),
            }
            for i in range(num_pts)
        ]

        curr_local_coords = [
            (float(blended_tracks[i][f][0]), float(blended_tracks[i][f][1]))
            for i in range(num_pts)
        ]

        # Eq. (8): Chord-aligned tangent rotation & scale clamping in [0.7, 1.4]
        rotated_tangents = compute_rotated_tangents(ref_pts_for_bridge, curr_local_coords)

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


class AutoRotoPipeline:
    """
    End-to-End Human-in-the-Loop Conversion System transforming tracked raster alpha masks
    into artist-editable, spatiotemporally consistent parametric cubic Bezier rotoscoping shapes.
    """

    def __init__(
        self,
        overlap_margin_px: int = 12,
        max_parts: int = 3,
        min_points_per_part: int = 8,
        max_points_per_part: int = 20,
        default_tension: float = 0.5,
    ):
        self.overlap_margin_px = overlap_margin_px
        self.max_parts = max_parts
        self.min_points_per_part = min_points_per_part
        self.max_points_per_part = max_points_per_part
        self.default_tension = default_tension

    def build_editable_roto_from_masks(
        self,
        masks: List[np.ndarray],
        frames_rgb: Optional[List[np.ndarray]] = None,
        frame_numbers: Optional[List[int]] = None,
        target_points_per_part: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes Stages 1 -> 2 -> 3 -> 4 on an input sequence of tracked raster alpha masks.
        Returns a complete editable Roto project dictionary that supports non-destructive
        artist keyframe edits, Nuke Roto node injection, JSON serialization, and SVG export.
        """
        if not masks:
            raise ValueError("masks list cannot be empty.")

        num_frames = len(masks)
        h, w = masks[0].shape[:2]
        if frame_numbers is None:
            frame_numbers = list(range(1, num_frames + 1))

        # Stage 1: Master Frame Selection (Eq. 1)
        master_info = select_master_frame(masks, frames=frames_rgb, frame_numbers=frame_numbers)
        master_idx = master_info["master_index"]
        master_frame = master_info["master_frame"]
        master_mask = masks[master_idx]

        # Compute optical flow at master frame if RGB frames are available
        master_flow = None
        if frames_rgb is not None and len(frames_rgb) > 1:
            next_idx = min(num_frames - 1, master_idx + 1) if master_idx < num_frames - 1 else max(0, master_idx - 1)
            if next_idx != master_idx:
                g0 = cv2.cvtColor(frames_rgb[master_idx], cv2.COLOR_RGB2GRAY) if frames_rgb[master_idx].ndim == 3 else frames_rgb[master_idx]
                g1 = cv2.cvtColor(frames_rgb[next_idx], cv2.COLOR_RGB2GRAY) if frames_rgb[next_idx].ndim == 3 else frames_rgb[next_idx]
                master_flow = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Stage 2: Kinematic-Geometric Part Decomposition & 10-15 px Seam Overlap Margin (Eq. 2)
        decomp = decompose_mask_into_parts(
            master_mask,
            flow=master_flow,
            max_parts=self.max_parts,
            overlap_margin_px=self.overlap_margin_px,
        )

        # Stage 3 & 4: Per-Part Hermite-Bezier Fitting (Eqs. 3-5) & Fixed-Topology Propagation (Eqs. 6-8)
        parts_roto: List[Dict[str, Any]] = []

        for part in decomp["parts"]:
            p_id = part["part_id"]
            p_name = part["name"]
            p_mask_master = part["mask"]

            contour = extract_ordered_contour(p_mask_master, smooth_sigma=1.2)
            anchors = select_curvature_adaptive_anchors(
                contour,
                target_num_points=target_points_per_part,
                min_points=self.min_points_per_part,
                max_points=self.max_points_per_part,
            )
            master_bezier_cps = contour_to_hermite_bezier(
                anchors, default_tension=self.default_tension
            )

            # Build per-frame target masks for this part by intersecting or tracking
            ref_keyframes_map = {master_frame: master_bezier_cps}
            raw_tracks_from_master = propagate_points_from_reference(
                master_bezier_cps,
                ref_frame=master_frame,
                frame_numbers=frame_numbers,
                masks=masks if len(decomp["parts"]) == 1 else [
                    cv2.bitwise_and(m, cv2.dilate(p_mask_master, np.ones((25, 25), np.uint8)))
                    if np.count_nonzero(cv2.bitwise_and(m, p_mask_master)) > 30 else m
                    for m in masks
                ],
                frames_rgb=frames_rgb,
                use_contour_snapping=True,
            )

            tracks_per_ref = {master_frame: raw_tracks_from_master}
            blended_tracks = blend_multi_ref_tracks(
                tracks_per_ref,
                ref_frames=[master_frame],
                start_f=frame_numbers[0],
                end_f=frame_numbers[-1],
            )
            baked_frames = bake_part_frames_with_rotated_tangents(
                ref_keyframes_map, blended_tracks, frame_numbers
            )

            parts_roto.append(
                {
                    "part_id": p_id,
                    "name": p_name,
                    "num_control_points": len(master_bezier_cps),
                    "reference_keyframes": ref_keyframes_map,
                    "tracks_per_ref": tracks_per_ref,
                    "frames": baked_frames,
                }
            )

        return {
            "width": w,
            "height": h,
            "frame_numbers": frame_numbers,
            "master_frame": master_frame,
            "master_index": master_idx,
            "master_selection": master_info,
            "overlap_margin_px": self.overlap_margin_px,
            "seam_gap_ratio_master": decomp["seam_gap_ratio"],
            "cut_lines": decomp["cut_lines"],
            "parts": parts_roto,
            "_masks_ref": masks,
            "_frames_rgb_ref": frames_rgb,
        }

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

        # Store artist-edited keyframe non-destructively
        part["reference_keyframes"][edited_frame] = [
            {
                "index": i,
                "anchor": [float(cp["anchor"][0]), float(cp["anchor"][1])],
                "in_handle": [float(cp["in_handle"][0]), float(cp["in_handle"][1])],
                "out_handle": [float(cp["out_handle"][0]), float(cp["out_handle"][1])],
                "is_keyframe": True,
            }
            for i, cp in enumerate(edited_control_points)
        ]

        if masks is not None:
            new_ref_tracks = propagate_points_from_reference(
                part["reference_keyframes"][edited_frame],
                ref_frame=edited_frame,
                frame_numbers=frame_numbers,
                masks=masks,
                frames_rgb=roto_project.get("_frames_rgb_ref"),
                use_contour_snapping=False,
            )
        else:
            # Propagate relative delta offset from existing trajectory when raw masks are not attached
            new_ref_tracks = {i: {} for i in range(part["num_control_points"])}
            old_frame_cps = part["frames"][edited_frame]
            for i in range(part["num_control_points"]):
                dx = float(edited_control_points[i]["anchor"][0]) - float(old_frame_cps[i]["anchor"][0])
                dy = float(edited_control_points[i]["anchor"][1]) - float(old_frame_cps[i]["anchor"][1])
                for f in frame_numbers:
                    ox, oy = part["frames"][f][i]["anchor"]
                    new_ref_tracks[i][f] = (float(ox + dx), float(oy + dy), 1.0)

        part["tracks_per_ref"][edited_frame] = new_ref_tracks
        sorted_refs = sorted(part["reference_keyframes"].keys())

        # Equations (6) & (7): Multi-Reference C1 Smoothstep Visibility-Weighted Blending
        blended_tracks = blend_multi_ref_tracks(
            part["tracks_per_ref"],
            ref_frames=sorted_refs,
            start_f=frame_numbers[0],
            end_f=frame_numbers[-1],
        )

        # Equation (8) + Non-destructive keyframe locking
        part["frames"] = bake_part_frames_with_rotated_tangents(
            part["reference_keyframes"], blended_tracks, frame_numbers
        )
        return roto_project

    @staticmethod
    def simulate_oracle_editability_nom(
        roto_project: Dict[str, Any],
        gt_masks: List[np.ndarray],
        target_f_score: float = 90.0,
        max_modifications: int = 30,
    ) -> Dict[str, Any]:
        """
        Simulates an Oracle Artist Agent that iteratively identifies the control point with the
        largest distance to the ground-truth contour and drags it to the true contour boundary
        until the sequence boundary accuracy F >= target_f_score.

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
        for f_idx, f_num in enumerate(frame_numbers):
            gt_c = extract_ordered_contour(gt_masks[f_idx], smooth_sigma=0.5)
            if len(gt_c) == 0:
                continue
            m_curr = _eval_frame(f_idx)
            while m_curr["F"] < target_f_score and nom_count < max_modifications:
                # Find control point across all parts with largest deviation from gt_c
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

                # Oracle drags worst control point to true contour boundary
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
        serializable = {
            k: v
            for k, v in roto_project.items()
            if not k.startswith("_")
        }
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
    ):
        """
        Generates a standalone Python script that can be executed inside Foundry Nuke's
        Script Editor (or `nuke -t`) to construct the animated multi-part cubic Bezier Roto
        node with all keyframes and tangent handles.
        """
        serializable = {
            "width": roto_project["width"],
            "height": roto_project["height"],
            "frame_numbers": roto_project["frame_numbers"],
            "master_frame": roto_project["master_frame"],
            "parts": [
                {
                    "part_id": p["part_id"],
                    "name": p["name"],
                    "num_control_points": p["num_control_points"],
                    "reference_keyframes": {str(k): v for k, v in p["reference_keyframes"].items()},
                    "frames": {str(k): v for k, v in p["frames"].items()},
                }
                for p in roto_project["parts"]
            ],
        }
        payload_json = json.dumps(serializable)

        script = f'''"""
Auto-generated Foundry Nuke Roto Node Builder Script (AutoRoto)
Run inside Nuke Script Editor to instantiate the artist-editable multi-part Bezier Roto node.
"""
import json
import nuke
import nuke.rotopaint as rp

ROTO_DATA = json.loads({repr(payload_json)})

def build_autoroto_node(key_step={int(key_step)}):
    roto_node = nuke.createNode("Roto", inpanel=False)
    roto_node.setName("AutoRoto_EditableParts")
    curves_knob = roto_node["curves"]
    root_layer = getattr(curves_knob, "rootLayer", getattr(curves_knob, "root", None))
    h = float(ROTO_DATA["height"])
    frame_numbers = ROTO_DATA["frame_numbers"]

    for part in ROTO_DATA["parts"]:
        shape = rp.Shape(curves_knob)
        shape.name = part["name"]
        num_pts = int(part["num_control_points"])
        ref_set = set(int(k) for k in part["reference_keyframes"].keys())

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
    return roto_node

if __name__ == "__main__":
    build_autoroto_node()
'''
        os.makedirs(os.path.dirname(os.path.abspath(output_py_path)), exist_ok=True)
        with open(output_py_path, "w", encoding="utf-8") as f:
            f.write(script)
