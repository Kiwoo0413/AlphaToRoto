"""
EditableAlpha External AI/Geometry Worker (editablealpha_worker.py)
Runs outside Foundry Nuke's bundled Python so NumPy, OpenCV, and optional PyTorch
execute cleanly without ABI conflicts or GUI thread blocking.

Usage:
  python editablealpha_worker.py --job <job.json> --out <result.json>
"""

import os
import sys
import json
import argparse
from typing import Any, Dict, List, Optional, Tuple

_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURR_DIR not in sys.path:
    sys.path.insert(0, _CURR_DIR)

import cv2
import numpy as np

from mask_generator import decode_binary_mask_array
from autoroto_pipeline import EditableAlphaPipeline


def _emit_progress(pct: int, msg: str) -> None:
    print(f"PROGRESS:{max(0, min(100, int(pct)))}:{msg}", flush=True)


def _load_frames_from_paths(
    frame_paths: List[str],
    use_alpha: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    masks: List[np.ndarray] = []
    frames_rgb: List[np.ndarray] = []
    total = max(1, len(frame_paths))

    for idx, p in enumerate(frame_paths):
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Failed to read frame image: {p}")

        src_for_mask = img if (use_alpha or img.ndim < 3) else img[:, :, :3]
        masks.append(decode_binary_mask_array(src_for_mask))

        if img.ndim == 2:
            frames_rgb.append(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
        elif img.shape[2] >= 3:
            frames_rgb.append(cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB))
        else:
            frames_rgb.append(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))

        if (idx + 1) % max(1, total // 4) == 0 or idx == total - 1:
            pct = 10 + int(20 * (idx + 1) / total)
            _emit_progress(pct, f"Loaded frame {idx + 1}/{total}")

    return masks, frames_rgb


def _deserialize_project(raw_project: Dict[str, Any]) -> Dict[str, Any]:
    proj = dict(raw_project)
    proj["frame_numbers"] = [int(f) for f in raw_project["frame_numbers"]]
    norm_parts = []
    for p in raw_project.get("parts", []):
        np_part = dict(p)
        np_part["reference_keyframes"] = {
            int(k): v for k, v in p.get("reference_keyframes", {}).items()
        }
        np_part["manual_keyframes"] = {
            int(k): v for k, v in p.get("manual_keyframes", {}).items()
        }
        np_part["frames"] = {
            int(k): v for k, v in p.get("frames", {}).items()
        }
        norm_parts.append(np_part)
    proj["parts"] = norm_parts
    return proj


def run_convert_job(job: Dict[str, Any]) -> Dict[str, Any]:
    frame_paths = list(job["frame_paths"])
    frame_numbers = [int(f) for f in job["frame_numbers"]]
    use_alpha = bool(job.get("use_alpha", True))

    _emit_progress(5, "Loading rendered mask sequence...")
    masks, frames_rgb = _load_frames_from_paths(frame_paths, use_alpha=use_alpha)

    master_override: Optional[List[int]] = None
    if job.get("master_frames_override"):
        master_override = [int(f) for f in job["master_frames_override"]]

    pipeline = EditableAlphaPipeline(
        overlap_margin_px=int(job.get("overlap_margin_px", 12)),
        max_parts=max(1, int(job.get("max_parts", 3))),
        min_points_per_part=max(4, int(job.get("min_points_per_part", 8))),
        max_points_per_part=max(6, int(job.get("max_points_per_part", 16))),
        default_tension=float(job.get("default_tension", 0.5)),
        auto_multi_master=bool(job.get("auto_multi_master", True)),
    )

    _emit_progress(40, "Selecting master frame(s) & decomposing parts...")
    project = pipeline.build_editable_roto_from_masks(
        masks,
        frames_rgb=frames_rgb,
        frame_numbers=frame_numbers,
        master_frames_override=master_override,
        auto_multi_master=bool(job.get("auto_multi_master", True)),
    )

    _emit_progress(85, "Evaluating sequence quality & serializing Bezier curves...")
    metrics = pipeline.evaluate_sequence_metrics(project, masks)

    nk_path = job.get("output_nk_path")
    if nk_path:
        pipeline.export_to_nuke_nk(
            project,
            nk_path,
            key_step=max(1, int(job.get("key_step", 1))),
            attach_autoroto_tab=False,
        )

    serialized = EditableAlphaPipeline.serialize_project(
        project, include_tracks=False, stringify_frame_keys=True
    )
    _emit_progress(100, "Mask-to-Roto conversion complete.")
    return {
        "ok": True,
        "mode": "convert",
        "project": serialized,
        "metrics": metrics,
        "output_nk_path": nk_path,
    }


def run_retrack_job(job: Dict[str, Any]) -> Dict[str, Any]:
    frame_paths = list(job["frame_paths"])
    frame_numbers = [int(f) for f in job["frame_numbers"]]
    use_alpha = bool(job.get("use_alpha", True))

    _emit_progress(5, "Loading mask sequence for re-tracking...")
    masks, frames_rgb = _load_frames_from_paths(frame_paths, use_alpha=use_alpha)

    project = _deserialize_project(job["project"])
    manual_edits_by_part: Dict[str, Dict[str, Any]] = job.get("manual_keyframes_by_part", {})

    for part in project["parts"]:
        pid_key = str(part["part_id"])
        part_name_key = str(part["name"])
        incoming = manual_edits_by_part.get(pid_key) or manual_edits_by_part.get(part_name_key) or {}
        if incoming:
            mk = dict(part.get("manual_keyframes", {}))
            for f_str, cps in incoming.items():
                f_int = int(f_str)
                if len(cps) == int(part["num_control_points"]):
                    mk[f_int] = [
                        {
                            "index": i,
                            "anchor": [float(cp["anchor"][0]), float(cp["anchor"][1])],
                            "in_handle": [float(cp["in_handle"][0]), float(cp["in_handle"][1])],
                            "out_handle": [float(cp["out_handle"][0]), float(cp["out_handle"][1])],
                            "is_keyframe": True,
                        }
                        for i, cp in enumerate(cps)
                    ]
                    part["reference_keyframes"][f_int] = mk[f_int]
            part["manual_keyframes"] = mk

    pipeline = EditableAlphaPipeline(
        overlap_margin_px=int(project.get("overlap_margin_px", job.get("overlap_margin_px", 12))),
        max_parts=max(1, len(project["parts"])),
        min_points_per_part=max(4, int(job.get("min_points_per_part", 8))),
        max_points_per_part=max(6, int(job.get("max_points_per_part", 16))),
        default_tension=float(job.get("default_tension", 0.5)),
        auto_multi_master=bool(job.get("auto_multi_master", True)),
    )

    _emit_progress(50, "Re-tracking while preserving manual control points...")
    updated = pipeline.run_stage4_propagate(
        project,
        masks=masks,
        frames_rgb=frames_rgb,
        master_frames=project.get("master_frames"),
        default_tension=float(job.get("default_tension", 0.5)),
        cusp_angle_deg=float(job.get("cusp_angle_deg", 60.0)),
    )

    _emit_progress(85, "Evaluating updated sequence metrics...")
    metrics = pipeline.evaluate_sequence_metrics(updated, masks)
    serialized = EditableAlphaPipeline.serialize_project(
        updated, include_tracks=False, stringify_frame_keys=True
    )
    _emit_progress(100, "Re-tracking complete.")
    return {
        "ok": True,
        "mode": "retrack",
        "project": serialized,
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="EditableAlpha External Geometry & Tracking Worker")
    parser.add_argument("--job", required=True, help="Path to input job JSON file")
    parser.add_argument("--out", required=True, help="Path to output result JSON file")
    args = parser.parse_args()

    with open(args.job, "r", encoding="utf-8") as f:
        job = json.load(f)

    mode = str(job.get("mode", "convert")).lower()
    if mode == "convert":
        result = run_convert_job(job)
    elif mode == "retrack":
        result = run_retrack_job(job)
    else:
        raise ValueError(f"Unsupported worker mode: {mode}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
