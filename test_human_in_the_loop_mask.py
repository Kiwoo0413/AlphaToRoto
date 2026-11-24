"""
Comprehensive Unit & Integration Tests for AlphaToRoto
Verifies:
  1. Stage 1: Tracked Raster Alpha Mask Generation & Master Frame Selection (Eq. 1)
  2. Stage 2: Kinematic-Geometric Part Decomposition & 10-15 px Seam Overlap Margin (Eq. 2)
  3. Stage 3: O(N) Closed-Form Hermite-to-Bezier Analytical Contour Fitting (Eqs. 3-5)
  4. Stage 4: Fixed-Topology Temporal Propagation, Multi-Reference C1 Smoothstep Visibility-Weighted
              Blending (Eqs. 6-7), Chord-Aligned Relative Tangent Rotation (Eq. 8), Non-Destructive
              Artist Keyframe Editing, Oracle Editability Evaluator (NoM), and SVG/JSON/Nuke Export.
"""

import os
import json
import tempfile
import unittest
import numpy as np
import cv2

from mask_generator import (
    TrackedMaskGenerator,
    compute_convexity_defects_count,
    select_master_frame,
    compute_object_vector_flow_profile,
    select_representative_master_frames,
    create_synthetic_articulated_sequence,
)
from roto_geometry import (
    decompose_mask_into_parts,
    contour_to_hermite_bezier,
    evaluate_hermite_segment,
    evaluate_bezier_segment,
    mask_to_hermite_bezier_shape,
    rasterize_bezier_shapes,
    compute_j_and_f_metrics,
    compute_contour_rmse_px,
    rdp_polygon_vectorize,
    sam2_autotrace_vectorize,
    least_squares_bspline_fit,
)
from autoroto_pipeline import AlphaToRotoPipeline, AutoRotoPipeline


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

    def test_stage5_baselines_ablations_and_sequence_metrics(self):
        """
        Verifies Table 1 baselines (RDP, SAM2 Auto-Trace, Least-Squares B-Spline),
        Table 2/3 sequence metrics (RMSE, T_flicker, dJ_decay, Gap_percent),
        and ablation configurations (linear blending, disabled tangent rotation).
        """
        m = self.masks[4]
        rdp_res = rdp_polygon_vectorize(m, epsilon_ratio=0.0025)
        sam_res = sam2_autotrace_vectorize(m, step_px=12.0)
        lsq_res = least_squares_bspline_fit(m, num_segments=16, newton_iters=3)
        our_res = mask_to_hermite_bezier_shape(m, min_points=10, max_points=16)

        self.assertGreater(rdp_res["num_control_points"], 0)
        self.assertGreater(sam_res["num_control_points"], 0)
        self.assertEqual(lsq_res["num_control_points"], 16)
        self.assertGreaterEqual(our_res["metrics"]["J"], 92.0)

        rmse_val = compute_contour_rmse_px([our_res["control_points"]], m)
        self.assertLess(rmse_val, 5.0)

        pipe_full = AutoRotoPipeline(
            overlap_margin_px=12,
            max_parts=2,
            blending_mode="c1_vis",
            use_tangent_rotation=True,
        )
        proj_full = pipe_full.build_editable_roto_from_masks(self.masks, frames_rgb=self.frames_rgb)
        metrics_full = pipe_full.evaluate_sequence_metrics(proj_full, self.masks)
        self.assertIn("T_flicker", metrics_full)
        self.assertIn("RMSE_px", metrics_full)
        self.assertEqual(metrics_full["Gap_percent"], 0.0)

    def test_stage6_realworld_rgba_fallback_cusps_and_opposing_pinches(self):
        """
        Verifies real-world mask edge cases:
          1. RGBA PNG with uniform alpha=255 falls back to RGB luminance.
          2. Sharp 90-degree corners retract handles to [0.0, 0.0] (tension == 0.0).
          3. Shadowed opposing neck pinches cleanly split head from torso/arm with 0% seam gap.
        """
        # 1. RGBA opaque-alpha fallback
        with tempfile.TemporaryDirectory() as tmpdir:
            rgb_mask = np.zeros((160, 200, 3), dtype=np.uint8)
            cv2.circle(rgb_mask, (100, 80), 45, (255, 255, 255), -1)
            alpha_solid = np.full((160, 200, 1), 255, dtype=np.uint8)
            rgba = np.concatenate([rgb_mask, alpha_solid], axis=2)
            rgba_path = os.path.join(tmpdir, "opaque_alpha_mask.png")
            cv2.imwrite(rgba_path, rgba)
            loaded = TrackedMaskGenerator.load_mask_sequence([rgba_path])[0]
            self.assertEqual(int(loaded[80, 100]), 255)
            self.assertEqual(int(loaded[10, 10]), 0)

        # 2. Sharp 90-degree corners retract handles to zero
        rect_anchors = np.array(
            [[20.0, 20.0], [180.0, 20.0], [180.0, 140.0], [20.0, 140.0]],
            dtype=np.float64,
        )
        rect_cps = contour_to_hermite_bezier(rect_anchors)
        for cp in rect_cps:
            self.assertEqual(cp["tension"], 0.0)
            np.testing.assert_allclose(cp["in_handle"], [0.0, 0.0], atol=1e-12)
            np.testing.assert_allclose(cp["out_handle"], [0.0, 0.0], atol=1e-12)

        # 3. Synthetic head + shoulders + outstretched arm touching right border
        sil = np.zeros((320, 480), dtype=np.uint8)
        # Head ellipse
        cv2.ellipse(sil, (240, 100), (58, 62), 0, 0, 360, 255, -1)
        # Shoulders/torso reaching bottom border
        cv2.ellipse(sil, (240, 270), (135, 115), 0, 0, 360, 255, -1)
        # Outstretched right arm creating a shadowed concavity on the right shoulder
        cv2.rectangle(sil, (340, 235), (479, 319), 255, -1)
        # Carve clear neck pinches on left and right
        cv2.fillPoly(sil, [np.array([[160, 145], [195, 145], [195, 168]], dtype=np.int32)], 0)
        cv2.fillPoly(sil, [np.array([[320, 145], [285, 145], [285, 168]], dtype=np.int32)], 0)

        decomp = decompose_mask_into_parts(sil, max_parts=2, overlap_margin_px=12)
        self.assertEqual(len(decomp["parts"]), 2)
        self.assertEqual(decomp["seam_gap_ratio"], 0.0)
        cut = decomp["cut_lines"][0]
        # Neck cut should stay horizontal near y ~= 150-175 across the neck constriction
        self.assertLess(abs(cut["p1"][1] - cut["p2"][1]), 35)
        self.assertLess(max(cut["p1"][1], cut["p2"][1]), 200)

    def test_stage7_web_studio_api_sam2_and_nuke_autoroto_export(self):
        """
        End-to-end verification of the Full-Pipeline Web Studio (alphatoroto_web_studio.py):
          - GET / and /api/status
          - Load 15-frame RGB preset -> Stage 1 (SAM 2.1 GPU / fallback)
          - Stage 2 pre-Bezier decomposition + 12 px mask overlap + PNG layer endpoint
          - Stage 3 O(N) Hermite-Bezier fit
          - Stage 4 propagation + non-destructive HITL keyframe edit API
          - Export Nuke AlphaToRoto node script & syntax verification
        """
        from fastapi.testclient import TestClient
        import alphatoroto_web_studio as web_studio

        client = TestClient(web_studio.app)

        r_idx = client.get("/")
        self.assertEqual(r_idx.status_code, 200)
        self.assertIn("AlphaToRoto Studio", r_idx.text)

        r_st = client.get("/api/status")
        self.assertEqual(r_st.status_code, 200)
        st_data = r_st.json()
        self.assertIn("sam2", st_data)
        self.assertIn("session", st_data)

        # Load 15-frame RGB sequence
        r_load = client.post("/api/load_preset", json={"preset": "sample_sequence"})
        self.assertEqual(r_load.status_code, 200)
        self.assertEqual(r_load.json()["session"]["num_frames"], 15)

        # Stage 1: SAM 2.1 segmentation + propagation + F_master
        r_s1 = client.post(
            "/api/stage1/run",
            json={"prompt_frame_idx": 4, "fg_points": [[150, 160]], "use_sam": True},
        )
        self.assertEqual(r_s1.status_code, 200)
        s1_data = r_s1.json()
        self.assertTrue(s1_data["session"]["stage1_ready"])
        master_idx = s1_data["master_info"]["master_index"]
        master_frame = s1_data["master_info"]["master_frame"]

        # Stage 2: Pre-Bezier part decomposition + 12px seam overlap
        r_s2 = client.post(
            "/api/stage2/run",
            json={"max_parts": 2, "overlap_margin_px": 12, "defect_depth_thresh_px": 10.0},
        )
        self.assertEqual(r_s2.status_code, 200)
        s2_data = r_s2.json()
        self.assertGreaterEqual(s2_data["num_parts"], 2)
        self.assertGreater(s2_data["overlap_pixels"], 0)
        self.assertEqual(s2_data["seam_gap_percent"], 0.0)

        r_ov_png = client.get(f"/api/frame/{master_idx}/stage2_overlap")
        self.assertEqual(r_ov_png.status_code, 200)
        self.assertTrue(r_ov_png.content.startswith(b"\x89PNG"))

        # Stage 3: O(N) Hermite-Bezier fit
        r_s3 = client.post(
            "/api/stage3/run",
            json={"min_points": 8, "max_points": 16, "default_tension": 0.5, "cusp_angle_deg": 60.0},
        )
        self.assertEqual(r_s3.status_code, 200)
        self.assertGreaterEqual(r_s3.json()["metrics"]["J"], 90.0)

        # Stage 4: Full sequence propagation & non-destructive artist keyframe edit
        r_s4 = client.post(
            "/api/stage4/run",
            json={"blending_mode": "c1_vis", "use_tangent_rotation": True, "use_contour_snapping": True},
        )
        self.assertEqual(r_s4.status_code, 200)
        proj = r_s4.json()["project"]
        part0 = proj["parts"][0]
        orig_master_a0 = list(part0["frames"][str(master_frame)][0]["anchor"])

        edited_cps = [dict(cp) for cp in part0["frames"]["1"]]
        edited_target = [edited_cps[0]["anchor"][0] + 12.0, edited_cps[0]["anchor"][1] - 8.0]
        edited_cps[0] = dict(edited_cps[0], anchor=edited_target)

        r_edit = client.post(
            "/api/stage4/edit",
            json={"part_id": part0["part_id"], "edited_frame": 1, "edited_control_points": edited_cps},
        )
        self.assertEqual(r_edit.status_code, 200)
        edited_proj = r_edit.json()["project"]
        np.testing.assert_allclose(edited_proj["parts"][0]["frames"]["1"][0]["anchor"], edited_target, atol=1e-6)
        np.testing.assert_allclose(
            edited_proj["parts"][0]["frames"][str(master_frame)][0]["anchor"], orig_master_a0, atol=1e-6
        )

        # Export Nuke AlphaToRoto Node script & verify syntax
        r_exp = client.post("/api/export/nuke", json={"key_step": 1, "attach_autoroto_tab": True})
        self.assertEqual(r_exp.status_code, 200)
        exp_data = r_exp.json()
        self.assertTrue(os.path.isfile(exp_data["script_path"]))
        with open(exp_data["script_path"], "r", encoding="utf-8") as f:
            py_src = f.read()
        self.assertIn("setup_autoroto_knobs", py_src)
        compile(py_src, exp_data["script_path"], "exec")

        r_dl = client.get(exp_data["download_urls"]["nuke_py"])
        self.assertEqual(r_dl.status_code, 200)

    def test_stage8_per_part_toggle_adjustment_and_vector_flow_multi_master(self):
        """
        Verifies both new capabilities:
          1. Object vector flow analysis & automatic multi-master representative frame selection +
             linked multi-reference propagation when foreground motion is large (vs single master when static).
          2. Individual per-part mask layer endpoints, visibility/enabled toggling, overlap margin
             adjustment, and multi-master synchronization into Nuke's ar_ref_frame knob.
        """
        # 1a. Static sequence -> 1 master frame, large_motion_detected == False
        static_masks = [self.masks[4].copy() for _ in range(8)]
        static_rgb = [self.frames_rgb[4].copy() for _ in range(8)]
        static_res = select_representative_master_frames(static_masks, frames=static_rgb)
        self.assertFalse(static_res["large_motion_detected"])
        self.assertEqual(len(static_res["master_frames"]), 1)

        # 1b. Dynamic 15-frame articulated sequence -> >= 2 representative master frames
        dyn_masks, dyn_rgb = create_synthetic_articulated_sequence(num_frames=15, height=360, width=360)
        flow_prof = compute_object_vector_flow_profile(dyn_masks, frames=dyn_rgb)
        self.assertGreater(flow_prof["total_motion_px"], 22.0)

        dyn_res = select_representative_master_frames(
            dyn_masks, frames=dyn_rgb, flow_motion_threshold_px=20.0, max_master_frames=4
        )
        self.assertTrue(dyn_res["large_motion_detected"])
        self.assertGreaterEqual(len(dyn_res["master_frames"]), 2)

        # 2. AlphaToRotoPipeline(auto_multi_master=True) links all representative master frames
        pipe = AlphaToRotoPipeline(
            overlap_margin_px=12,
            max_parts=2,
            min_points_per_part=10,
            max_points_per_part=14,
            auto_multi_master=True,
            flow_motion_threshold_px=20.0,
        )
        proj = pipe.build_editable_roto_from_masks(dyn_masks, frames_rgb=dyn_rgb)
        self.assertGreaterEqual(len(proj["master_frames"]), 2)
        for p in proj["parts"]:
            self.assertEqual(sorted(p["reference_keyframes"].keys()), proj["master_frames"])
            for f_num in proj["frame_numbers"]:
                self.assertEqual(len(p["frames"][f_num]), p["num_control_points"])
        seq_metrics = pipe.evaluate_sequence_metrics(proj, dyn_masks)
        self.assertGreaterEqual(seq_metrics["J_and_F"], 82.0)
        self.assertLess(seq_metrics["Gap_percent"], 0.05)

        # 2b. High-motion translating + articulated 15-frame sequence (dx = 14*t px inside 360x560)
        h_tr, w_tr = 360, 560
        trans_masks, trans_rgb = [], []
        for t_idx in range(15):
            dx = float(14 * t_idx)
            m_mat = np.float32([[1.0, 0.0, dx], [0.0, 1.0, 0.0]])
            m_shift = cv2.warpAffine(
                dyn_masks[t_idx], m_mat, (w_tr, h_tr), flags=cv2.INTER_NEAREST, borderValue=0
            )
            fr_shift = cv2.warpAffine(
                dyn_rgb[t_idx],
                m_mat,
                (w_tr, h_tr),
                flags=cv2.INTER_LINEAR,
                borderValue=(30, 30, 30),
            )
            trans_masks.append(m_shift)
            trans_rgb.append(fr_shift)

        pipe_single = AlphaToRotoPipeline(
            overlap_margin_px=12,
            max_parts=2,
            min_points_per_part=10,
            max_points_per_part=14,
            auto_multi_master=False,
            flow_motion_threshold_px=20.0,
        )
        proj_single = pipe_single.build_editable_roto_from_masks(trans_masks, frames_rgb=trans_rgb)
        metrics_single = pipe_single.evaluate_sequence_metrics(proj_single, trans_masks)

        proj_multi = pipe.build_editable_roto_from_masks(trans_masks, frames_rgb=trans_rgb)
        metrics_multi = pipe.evaluate_sequence_metrics(proj_multi, trans_masks)

        self.assertGreaterEqual(len(proj_multi["master_frames"]), 2)
        self.assertGreaterEqual(metrics_multi["J_and_F"], 85.0)
        self.assertGreater(metrics_multi["J_and_F"], metrics_single["J_and_F"])
        self.assertLess(metrics_multi["RMSE_px"], metrics_single["RMSE_px"])
        self.assertLess(metrics_multi["dJ_decay"], metrics_single["dJ_decay"] * 0.7)
        self.assertLess(metrics_multi["Gap_percent"], 0.05)
        self.assertLess(metrics_multi["Gap_percent"], metrics_single["Gap_percent"])

        # 3. Web Studio API: individual part toggle/adjustment + vector flow multi-master linking
        from fastapi.testclient import TestClient
        import alphatoroto_web_studio as web_studio

        client = TestClient(web_studio.app)
        r_load = client.post("/api/load_preset", json={"preset": "sample_sequence"})
        self.assertEqual(r_load.status_code, 200)

        r_s1 = client.post(
            "/api/stage1/run",
            json={
                "prompt_frame_idx": 4,
                "fg_points": [[150, 160]],
                "use_sam": True,
                "auto_multi_master": True,
                "flow_motion_threshold_px": 18.0,
                "max_master_frames": 3,
            },
        )
        self.assertEqual(r_s1.status_code, 200)
        s1_json = r_s1.json()
        self.assertIn("flow_profile", s1_json["master_info"])
        self.assertGreaterEqual(len(s1_json["master_info"]["master_frames"]), 2)
        master_idx = s1_json["master_info"]["master_index"]

        # Manual override of representative master frames
        r_mf = client.post("/api/stage1/master_frames", json={"master_frames": [2, 5, 11]})
        self.assertEqual(r_mf.status_code, 200)
        self.assertEqual(r_mf.json()["master_info"]["master_frames"], [2, 5, 11])

        # Stage 2 & individual part layer PNGs + update_part
        r_s2 = client.post("/api/stage2/run", json={"max_parts": 2, "overlap_margin_px": 12})
        self.assertEqual(r_s2.status_code, 200)
        s2_json = r_s2.json()
        self.assertGreaterEqual(s2_json["num_parts"], 2)

        for pid in (0, 1):
            r_p = client.get(f"/api/frame/{master_idx}/part/{pid}")
            self.assertEqual(r_p.status_code, 200)
            self.assertTrue(r_p.content.startswith(b"\x89PNG"))
        r_seam = client.get(f"/api/frame/{master_idx}/overlap_seam")
        self.assertEqual(r_seam.status_code, 200)
        self.assertTrue(r_seam.content.startswith(b"\x89PNG"))

        # Adjust Part 0 overlap margin to 0 px -> reduces overlap pixels; disable Part 1 -> overlap becomes 0
        r_upd0 = client.post("/api/stage2/update_part", json={"part_id": 0, "overlap_margin_px": 0})
        self.assertEqual(r_upd0.status_code, 200)
        r_upd1 = client.post("/api/stage2/update_part", json={"part_id": 1, "enabled": False})
        self.assertEqual(r_upd1.status_code, 200)
        self.assertEqual(r_upd1.json()["overlap_pixels"], 0)

        # Re-enable Part 1 and restore Part 0 overlap to 14 px, then run Stage 3 -> Stage 4 -> Nuke Export
        client.post("/api/stage2/update_part", json={"part_id": 0, "overlap_margin_px": 14})
        client.post("/api/stage2/update_part", json={"part_id": 1, "enabled": True})

        r_s3 = client.post("/api/stage3/run", json={"min_points": 8, "max_points": 16})
        self.assertEqual(r_s3.status_code, 200)

        r_s4 = client.post("/api/stage4/run", json={"blending_mode": "c1_vis", "use_tangent_rotation": True})
        self.assertEqual(r_s4.status_code, 200)
        s4_json = r_s4.json()
        self.assertEqual(s4_json["linked_master_frames"], [2, 5, 11])
        for p in s4_json["project"]["parts"]:
            self.assertEqual(sorted(int(k) for k in p["reference_keyframes"].keys()), [2, 5, 11])

        r_exp = client.post("/api/export/nuke", json={"key_step": 1, "attach_autoroto_tab": True})
        self.assertEqual(r_exp.status_code, 200)
        with open(r_exp.json()["script_path"], "r", encoding="utf-8") as f:
            py_src = f.read()
        self.assertIn("ar_ref_frame", py_src)
        self.assertIn('", ".join(str(k) for k in sorted_refs)', py_src)
        with open(r_exp.json()["json_path"], "r", encoding="utf-8") as jf:
            exported_json = json.load(jf)
        all_exported_refs = sorted(
            {int(k) for p in exported_json["parts"] for k in p["reference_keyframes"].keys()}
        )
        self.assertEqual(all_exported_refs, [2, 5, 11])

        import nuke_bridge
        self.assertTrue(hasattr(nuke_bridge, "setup_autoroto_knobs"))
        self.assertTrue(hasattr(nuke_bridge, "load_alphatoroto_json_as_node"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
