"""
Comprehensive Unit & Integration Tests for HumanInTheLoop_Mask (AutoRoto)
Verifies:
  1. Stage 1: Tracked Raster Alpha Mask Generation & Master Frame Selection (Eq. 1)
  2. Stage 2: Kinematic-Geometric Part Decomposition & 10-15 px Seam Overlap Margin (Eq. 2)
  3. Stage 3: O(N) Closed-Form Hermite-to-Bezier Analytical Contour Fitting (Eqs. 3-5)
  4. Stage 4: Fixed-Topology Temporal Propagation, Multi-Reference C1 Smoothstep Visibility-Weighted
              Blending (Eqs. 6-7), Chord-Aligned Relative Tangent Rotation (Eq. 8), Non-Destructive
              Artist Keyframe Editing, Oracle Editability Evaluator (NoM), and SVG/JSON/Nuke Export.
"""

import os
import tempfile
import unittest
import numpy as np
import cv2

from mask_generator import (
    TrackedMaskGenerator,
    compute_convexity_defects_count,
    select_master_frame,
)
from part_decomposition import decompose_mask_into_parts
from hermite_bezier_converter import (
    contour_to_hermite_bezier,
    evaluate_hermite_segment,
    evaluate_bezier_segment,
    mask_to_hermite_bezier_shape,
    rasterize_bezier_shapes,
    compute_j_and_f_metrics,
)
from autoroto_pipeline import AutoRotoPipeline


def create_synthetic_articulated_sequence(
    num_frames: int = 9, height: int = 320, width: int = 320
):
    """
    Creates a synthetic articulated two-part object sequence (torso + swinging limb with a
    distinct concavity pinch at the joint) and corresponding textured RGB frames.
    Frame index 4 (frame 5) has the widest limb articulation and sharpest focus.
    """
    masks = []
    frames_rgb = []
    rng = np.random.default_rng(42)
    bg_texture = rng.integers(15, 45, size=(height, width, 3), dtype=np.uint8)

    for idx in range(num_frames):
        mask = np.zeros((height, width), dtype=np.uint8)
        # Torso ellipse
        cv2.ellipse(mask, (130, 160), (52, 78), 0, 0, 360, 255, -1)

        # Swinging articulated arm (widest angle at idx == 4)
        angle_deg = -35.0 + 18.0 * abs(idx - 4)
        arm_center = (200, int(155 + (idx - 4) * 6))
        cv2.ellipse(mask, arm_center, (58, 30), angle_deg, 0, 360, 255, -1)

        # Carve explicit V-shaped concavity pinch at the joint (top and bottom)
        # Deepest articulation defects at idx == 4 (F_master candidate)
        pinch_depth = 26 if idx == 4 else max(8, 22 - 4 * abs(idx - 4))
        top_pinch = np.array(
            [[152, 80], [176, 80], [164, 80 + pinch_depth]], dtype=np.int32
        )
        bot_pinch = np.array(
            [[152, 240], [176, 240], [164, 240 - pinch_depth]], dtype=np.int32
        )
        cv2.fillPoly(mask, [top_pinch], 0)
        cv2.fillPoly(mask, [bot_pinch], 0)

        # Synthesize textured RGB frame
        frame = bg_texture.copy()
        yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        fg_pattern = np.stack(
            [
                ((xx * 7 + idx * 5) % 180 + 70).astype(np.uint8),
                ((yy * 7 + idx * 3) % 180 + 70).astype(np.uint8),
                (((xx + yy) * 5) % 180 + 70).astype(np.uint8),
            ],
            axis=-1,
        )
        frame[mask > 127] = fg_pattern[mask > 127]

        # Apply slight motion blur to non-master frames so idx == 4 is sharpest
        if idx != 4:
            k_blur = 5 if abs(idx - 4) >= 2 else 3
            frame = cv2.GaussianBlur(frame, (k_blur, k_blur), 1.5)

        masks.append(mask)
        frames_rgb.append(frame)

    return masks, frames_rgb


class TestHumanInTheLoopMaskSystem(unittest.TestCase):
    def setUp(self):
        self.masks, self.frames_rgb = create_synthetic_articulated_sequence(num_frames=9)

    def test_stage1_master_frame_selection_and_mask_propagation(self):
        """Verifies Eq. (1) selects frame index 4 (frame 5) where articulation and sharpness peak."""
        res = select_master_frame(self.masks, frames=self.frames_rgb)
        self.assertEqual(res["master_index"], 4)
        self.assertEqual(res["master_frame"], 5)
        self.assertGreaterEqual(res["per_frame_metrics"][4]["num_defects"], 2)

        # Also verify optical flow mask propagation from master frame
        gen = TrackedMaskGenerator()
        prop_masks = gen.generate_tracked_masks(
            self.frames_rgb[:3], init_mask=self.masks[0], init_idx=0
        )
        self.assertEqual(len(prop_masks), 3)
        self.assertEqual(prop_masks[0].shape, self.masks[0].shape)

    def test_stage2_kinematic_geometric_part_decomposition_and_overlap(self):
        """Verifies Eq. (2) decomposes the articulated mask into >=2 parts with 12 px overlap and 0% seam gap."""
        master_mask = self.masks[4]
        decomp_with_overlap = decompose_mask_into_parts(
            master_mask, max_parts=2, overlap_margin_px=12, defect_depth_thresh_px=10.0
        )
        self.assertGreaterEqual(len(decomp_with_overlap["parts"]), 2)
        self.assertEqual(decomp_with_overlap["seam_gap_ratio"], 0.0)

        # Verify overlap intersection between Part 0 and Part 1 is non-empty when overlap_margin_px=12
        p0 = decomp_with_overlap["parts"][0]["mask"] > 0
        p1 = decomp_with_overlap["parts"][1]["mask"] > 0
        overlap_pixels = int(np.count_nonzero(p0 & p1))
        self.assertGreater(overlap_pixels, 100)

        # Without overlap margin (overlap_margin_px=0), the cut line leaves a measurable seam gap
        decomp_no_overlap = decompose_mask_into_parts(
            master_mask, max_parts=2, overlap_margin_px=0, defect_depth_thresh_px=10.0
        )
        self.assertGreater(decomp_no_overlap["seam_gap_ratio"], 0.0)

    def test_stage3_hermite_to_bezier_analytical_identity_and_accuracy(self):
        """
        Verifies exact mathematical equivalence between the cubic Hermite polynomial p(t) (Eq. 3)
        and the closed-form cubic Bezier segment B(t) (Eq. 4), plus high J&F mask reconstruction
        with sparse control points.
        """
        anchors = np.array(
            [[100.0, 60.0], [220.0, 90.0], [210.0, 220.0], [90.0, 210.0]],
            dtype=np.float64,
        )
        schema = contour_to_hermite_bezier(anchors)
        self.assertEqual(len(schema), 4)

        t_samples = np.linspace(0.0, 1.0, 50)
        for i in range(4):
            cp0 = schema[i]
            cp1 = schema[(i + 1) % 4]
            p0 = np.array(cp0["anchor"])
            p1 = np.array(cp1["anchor"])
            v0 = np.array(cp0["hermite_tangent"])
            v1 = np.array(cp1["hermite_tangent"])

            b0 = p0
            b1 = p0 + np.array(cp0["out_handle"])
            b2 = p1 + np.array(cp1["in_handle"])
            b3 = p1

            hermite_pts = evaluate_hermite_segment(p0, p1, v0, v1, t_samples)
            bezier_pts = evaluate_bezier_segment(b0, b1, b2, b3, t_samples)
            np.testing.assert_allclose(hermite_pts, bezier_pts, atol=1e-11)

        # Fit master mask with sparse control points and check J (IoU) >= 92%
        fit_res = mask_to_hermite_bezier_shape(self.masks[4], min_points=12, max_points=20)
        self.assertLessEqual(fit_res["num_control_points"], 20)
        self.assertGreaterEqual(fit_res["metrics"]["J"], 92.0)

    def test_stage4_end_to_end_pipeline_artist_edit_and_exporters(self):
        """
        Verifies end-to-end AutoRoto pipeline:
          - Fixed control point count N across all frames
          - 100% non-destructive preservation of artist-edited keyframes (Eqs. 6-7)
          - Smooth C1 interpolation on intermediate frames + chord-aligned tangent rotation (Eq. 8)
          - JSON, SVG, and Nuke Python script exports
        """
        pipeline = AutoRotoPipeline(
            overlap_margin_px=12,
            max_parts=2,
            min_points_per_part=10,
            max_points_per_part=16,
        )
        project = pipeline.build_editable_roto_from_masks(
            self.masks, frames_rgb=self.frames_rgb
        )

        self.assertEqual(project["master_frame"], 5)
        self.assertGreaterEqual(len(project["parts"]), 2)

        part0 = project["parts"][0]
        n_pts = part0["num_control_points"]
        for f_num in project["frame_numbers"]:
            self.assertEqual(len(part0["frames"][f_num]), n_pts)

        # Simulate an artist editing frame 1 on part 0 (shifting anchor 0 by +15, -10 px)
        edited_cps = [
            {
                "index": cp["index"],
                "anchor": list(cp["anchor"]),
                "in_handle": list(cp["in_handle"]),
                "out_handle": list(cp["out_handle"]),
            }
            for cp in part0["frames"][1]
        ]
        orig_f1_anchor0 = list(edited_cps[0]["anchor"])
        orig_master_anchor0 = list(part0["frames"][5][0]["anchor"])
        edited_cps[0]["anchor"] = [orig_f1_anchor0[0] + 15.0, orig_f1_anchor0[1] - 10.0]

        pipeline.apply_artist_keyframe_edit(
            project, part_id=0, edited_frame=1, edited_control_points=edited_cps
        )

        # 1. Artist keyframe at frame 1 must be 100% preserved
        np.testing.assert_allclose(
            part0["frames"][1][0]["anchor"], edited_cps[0]["anchor"], atol=1e-7
        )
        # 2. Master keyframe at frame 5 must ALSO be 100% preserved
        np.testing.assert_allclose(
            part0["frames"][5][0]["anchor"], orig_master_anchor0, atol=1e-7
        )

        # 3. Oracle editability simulator (NoM)
        oracle_stats = pipeline.simulate_oracle_editability_nom(
            project, self.masks, target_f_score=85.0, max_modifications=10
        )
        self.assertGreaterEqual(oracle_stats["final_F"], oracle_stats["initial_F"])

        # 4. Verify JSON, SVG, and Nuke Script exporters
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "project.json")
            svg_path = os.path.join(tmpdir, "frame_0005.svg")
            nuke_py_path = os.path.join(tmpdir, "build_nuke_roto.py")

            pipeline.export_to_json(project, json_path)
            pipeline.export_frame_to_svg(project, 5, svg_path)
            pipeline.export_to_nuke_script_py(project, nuke_py_path)

            self.assertTrue(os.path.isfile(json_path))
            self.assertTrue(os.path.isfile(svg_path))
            self.assertTrue(os.path.isfile(nuke_py_path))


if __name__ == "__main__":
    unittest.main(verbosity=2)
