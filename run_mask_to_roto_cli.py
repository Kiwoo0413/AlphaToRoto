"""
CLI & Demo Runner for HumanInTheLoop_Mask (AutoRoto)
Converts a folder of tracked raster alpha masks (or RGB frames via mask generator) into:
  1. An artist-editable multi-part parametric cubic Bezier project (`roto_project.json`)
  2. A standalone Foundry Nuke Python script (`build_nuke_roto.py`) that builds the animated
     multi-part Roto node inside Nuke
  3. Per-frame SVG vector previews with control points and tangent handles (`svg_previews/`)
  4. Quantitative evaluation summary (Control points N, J, F, J&F, Seam Gap %, and Oracle NoM)
"""

import os
import argparse
import glob
import json
import cv2

from mask_generator import TrackedMaskGenerator
from autoroto_pipeline import AutoRotoPipeline
from test_human_in_the_loop_mask import create_synthetic_articulated_sequence


def main():
    parser = argparse.ArgumentParser(
        description="Convert tracked raster alpha masks into artist-editable parametric cubic Bezier rotoscoping shapes."
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default=None,
        help="Directory containing input alpha mask sequence (PNG/JPG). If omitted, runs on a synthetic articulated demo sequence.",
    )
    parser.add_argument(
        "--rgb_dir",
        type=str,
        default=None,
        help="Optional directory containing matching RGB frames for optical flow & blur scoring.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_output"),
        help="Output directory for JSON, Foundry Nuke Python builder script, and SVG previews.",
    )
    parser.add_argument("--overlap_px", type=int, default=12, help="Internal seam overlap margin in pixels (default: 12).")
    parser.add_argument("--max_parts", type=int, default=2, help="Maximum quasi-rigid parts to decompose (default: 2).")
    parser.add_argument("--max_points", type=int, default=16, help="Maximum Bezier control points per part (default: 16).")
    parser.add_argument("--key_step", type=int, default=1, help="Keyframe baking step for Nuke export (default: 1).")
    args = parser.parse_args()

    if args.mask_dir:
        mask_files = sorted(
            glob.glob(os.path.join(args.mask_dir, "*.png"))
            + glob.glob(os.path.join(args.mask_dir, "*.jpg"))
        )
        if not mask_files:
            raise FileNotFoundError(f"No PNG/JPG mask files found in {args.mask_dir}")
        masks = TrackedMaskGenerator.load_mask_sequence(mask_files)
        frames_rgb = None
        if args.rgb_dir and os.path.isdir(args.rgb_dir):
            rgb_files = sorted(
                glob.glob(os.path.join(args.rgb_dir, "*.png"))
                + glob.glob(os.path.join(args.rgb_dir, "*.jpg"))
            )
            if len(rgb_files) == len(masks):
                frames_rgb = [
                    cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                    for p in rgb_files
                ]
    else:
        print("[Demo Mode] Generating 9-frame articulated object mask & RGB sequence...")
        masks, frames_rgb = create_synthetic_articulated_sequence(num_frames=9)

    pipeline = AutoRotoPipeline(
        overlap_margin_px=args.overlap_px,
        max_parts=args.max_parts,
        min_points_per_part=8,
        max_points_per_part=args.max_points,
    )

    project = pipeline.build_editable_roto_from_masks(masks, frames_rgb=frames_rgb)
    os.makedirs(args.output_dir, exist_ok=True)

    json_path = os.path.join(args.output_dir, "autoroto_project.json")
    nuke_script_path = os.path.join(args.output_dir, "build_nuke_roto.py")
    svg_dir = os.path.join(args.output_dir, "svg_previews")
    os.makedirs(svg_dir, exist_ok=True)

    pipeline.export_to_json(project, json_path)
    pipeline.export_to_nuke_script_py(project, nuke_script_path, key_step=args.key_step)
    for f_num in project["frame_numbers"]:
        pipeline.export_frame_to_svg(
            project, f_num, os.path.join(svg_dir, f"frame_{f_num:04d}.svg")
        )

    oracle_eval = pipeline.simulate_oracle_editability_nom(
        project, masks, target_f_score=90.0, max_modifications=20
    )

    summary = {
        "master_frame": project["master_frame"],
        "num_parts": len(project["parts"]),
        "control_points_per_part": [p["num_control_points"] for p in project["parts"]],
        "total_control_points": oracle_eval["total_control_points"],
        "overlap_margin_px": project["overlap_margin_px"],
        "seam_gap_ratio_percent": round(project["seam_gap_ratio_master"] * 100.0, 4),
        "initial_J_percent": round(oracle_eval["initial_J"], 2),
        "initial_F_percent": round(oracle_eval["initial_F"], 2),
        "final_J_percent": round(oracle_eval["final_J"], 2),
        "final_F_percent": round(oracle_eval["final_F"], 2),
        "oracle_NoM": oracle_eval["NoM"],
        "outputs": {
            "project_json": json_path,
            "nuke_builder_script": nuke_script_path,
            "svg_previews_dir": svg_dir,
        },
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
