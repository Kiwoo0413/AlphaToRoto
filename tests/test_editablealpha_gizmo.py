import json
import os
import subprocess
import sys
import tempfile
import types
import unittest

import cv2
import numpy as np

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from mask_generator import create_synthetic_articulated_sequence


class _MockAnimCurve:
    def __init__(self):
        self.keys = {}

    def addPositionKey(self, frame, xyz):
        self.keys[float(frame)] = (float(xyz[0]), float(xyz[1]), float(xyz[2]))

    def getPosition(self, frame):
        return self.keys[float(frame)]


class _MockControlPoint:
    def __init__(self):
        self.center = _MockAnimCurve()
        self.leftTangent = _MockAnimCurve()
        self.rightTangent = _MockAnimCurve()
        self.featherCenter = _MockAnimCurve()


class _MockShape:
    def __init__(self, _curves_knob):
        self.name = ""
        self._pts = []

    def append(self, cp):
        self._pts.append(cp)

    def __len__(self):
        return len(self._pts)

    def __getitem__(self, idx):
        return self._pts[idx]


class _MockRootLayer(list):
    def remove(self, idx):
        del self[idx]


class _MockCurvesKnob:
    def __init__(self):
        self.rootLayer = _MockRootLayer()
        self.changed_called = False

    def changed(self):
        self.changed_called = True


class _MockRotoNode:
    def __init__(self):
        self._knobs = {"curves": _MockCurvesKnob()}

    def knob(self, name):
        return self._knobs.get(name)

    def __getitem__(self, name):
        return self._knobs[name]


class TestEditableAlphaGizmo(unittest.TestCase):
    def test_01_bridge_imports_without_numpy_or_cv2(self):
        probe = (
            "import sys; "
            "sys.modules['numpy'] = None; "
            "sys.modules['cv2'] = None; "
            f"sys.path.insert(0, {repr(REPO_DIR)}); "
            "import editablealpha_bridge as b; "
            "assert b.parse_keyframe_list_string('2, 5 11') == [2, 5, 11]; "
            "print('STDLIB_BRIDGE_OK')"
        )
        proc = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=10)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("STDLIB_BRIDGE_OK", proc.stdout)

        import editablealpha_bridge as bridge
        py = bridge.find_worker_python(sys.executable)
        self.assertIsNotNone(py)

    def test_02_gizmo_syntax_and_knobs(self):
        gizmo_path = os.path.join(REPO_DIR, "EditableAlpha.gizmo")
        self.assertTrue(os.path.isfile(gizmo_path))
        with open(gizmo_path, "r", encoding="utf-8") as f:
            txt = f.read()

        self.assertIn("Gizmo {", txt)
        self.assertIn("end_group", txt)
        self.assertIn("name EditableAlpha", txt)
        self.assertIn("name mask", txt)
        self.assertIn("name EditableAlpha_Roto", txt)
        self.assertIn("name ViewSwitch", txt)
        self.assertIn("name Output1", txt)

        for required_knob in (
            "ea_start_frame",
            "ea_end_frame",
            "ea_sync_range",
            "ea_mask_channel",
            "ea_key_step",
            "ea_max_parts",
            "ea_overlap_px",
            "ea_min_pts",
            "ea_max_pts",
            "ea_tension",
            "ea_cusp_angle",
            "ea_auto_multi_master",
            "ea_ref_frames",
            "ea_output_mode",
            "ea_view_mode",
            "ea_convert_btn",
            "ea_extract_btn",
            "ea_retrack_btn",
            "ea_status",
            "ea_python_path",
            "ea_check_env",
            "ea_last_project_json",
        ):
            self.assertIn(required_knob, txt, f"Missing knob {required_knob} in EditableAlpha.gizmo")

        # Brace balance check
        self.assertEqual(txt.count("{"), txt.count("}"), "Unbalanced braces in EditableAlpha.gizmo")

    def test_03_worker_subprocess_convert_and_retrack_preserves_manual_edits(self):
        masks, frames_rgb = create_synthetic_articulated_sequence(num_frames=8, height=320, width=320)
        worker_script = os.path.join(REPO_DIR, "editablealpha_worker.py")

        with tempfile.TemporaryDirectory() as tmpdir:
            frame_paths = []
            frame_numbers = list(range(1, len(masks) + 1))
            for idx, (m, rgb) in enumerate(zip(masks, frames_rgb), start=1):
                rgba = np.dstack([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), m])
                p = os.path.join(tmpdir, f"rgba_{idx:04d}.png")
                cv2.imwrite(p, rgba)
                frame_paths.append(p)

            job_path = os.path.join(tmpdir, "job_convert.json")
            out_path = os.path.join(tmpdir, "out_convert.json")
            with open(job_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "mode": "convert",
                        "frame_paths": frame_paths,
                        "frame_numbers": frame_numbers,
                        "use_alpha": True,
                        "max_parts": 2,
                        "overlap_margin_px": 12,
                        "min_points_per_part": 8,
                        "max_points_per_part": 14,
                        "default_tension": 0.5,
                        "cusp_angle_deg": 60.0,
                        "auto_multi_master": True,
                    },
                    f,
                )

            proc = subprocess.run(
                [sys.executable, worker_script, "--job", job_path, "--out", out_path],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(proc.returncode, 0, proc.stdout + "\n" + proc.stderr)
            self.assertIn("PROGRESS:100:", proc.stdout)

            with open(out_path, "r", encoding="utf-8") as f:
                conv_res = json.load(f)
            self.assertTrue(conv_res["ok"])
            proj = conv_res["project"]
            self.assertGreaterEqual(len(proj["parts"]), 2)
            self.assertGreaterEqual(conv_res["metrics"]["J_and_F"], 80.0)
            self.assertEqual(conv_res["metrics"]["Gap_percent"], 0.0)

            # Edit anchor + tangent handles on frame 1 and re-track
            part0 = proj["parts"][0]
            f1_cps = [dict(cp) for cp in part0["frames"]["1"]]
            edited_anchor = [f1_cps[0]["anchor"][0] + 9.5, f1_cps[0]["anchor"][1] - 6.25]
            edited_in = [f1_cps[0]["in_handle"][0] - 3.0, f1_cps[0]["in_handle"][1] + 2.5]
            edited_out = [f1_cps[0]["out_handle"][0] + 4.0, f1_cps[0]["out_handle"][1] - 1.5]
            f1_cps[0] = {
                "index": 0,
                "anchor": edited_anchor,
                "in_handle": edited_in,
                "out_handle": edited_out,
                "is_keyframe": True,
            }

            retrack_job = os.path.join(tmpdir, "job_retrack.json")
            retrack_out = os.path.join(tmpdir, "out_retrack.json")
            with open(retrack_job, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "mode": "retrack",
                        "frame_paths": frame_paths,
                        "frame_numbers": frame_numbers,
                        "use_alpha": True,
                        "project": proj,
                        "manual_keyframes_by_part": {str(part0["part_id"]): {"1": f1_cps}},
                    },
                    f,
                )

            proc2 = subprocess.run(
                [sys.executable, worker_script, "--job", retrack_job, "--out", retrack_out],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(proc2.returncode, 0, proc2.stdout + "\n" + proc2.stderr)

            with open(retrack_out, "r", encoding="utf-8") as f:
                ret_res = json.load(f)
            ret_part0 = ret_res["project"]["parts"][0]
            np.testing.assert_allclose(ret_part0["frames"]["1"][0]["anchor"], edited_anchor, atol=1e-5)
            np.testing.assert_allclose(ret_part0["frames"]["1"][0]["in_handle"], edited_in, atol=1e-5)
            np.testing.assert_allclose(ret_part0["frames"]["1"][0]["out_handle"], edited_out, atol=1e-5)
            self.assertIn(1, ret_part0["manual_keyframe_frames"])

    def test_04_roto_inject_and_extract_roundtrip(self):
        import editablealpha_bridge as bridge

        mock_rp = types.SimpleNamespace(Shape=_MockShape, ShapeControlPoint=_MockControlPoint)
        old_rp = bridge.rp
        bridge.rp = mock_rp
        try:
            proj = {
                "width": 200,
                "height": 100,
                "frame_numbers": [1, 2],
                "parts": [
                    {
                        "part_id": 0,
                        "name": "Part_01",
                        "num_control_points": 2,
                        "reference_keyframes": {"1": []},
                        "manual_keyframe_frames": [],
                        "frames": {
                            "1": [
                                {"index": 0, "anchor": [40.0, 25.0], "in_handle": [-5.0, 3.0], "out_handle": [5.0, -3.0]},
                                {"index": 1, "anchor": [80.0, 60.0], "in_handle": [-4.0, -2.0], "out_handle": [4.0, 2.0]},
                            ],
                            "2": [
                                {"index": 0, "anchor": [42.0, 27.0], "in_handle": [-5.0, 3.0], "out_handle": [5.0, -3.0]},
                                {"index": 1, "anchor": [82.0, 62.0], "in_handle": [-4.0, -2.0], "out_handle": [4.0, 2.0]},
                            ],
                        },
                    }
                ],
            }
            node = _MockRotoNode()
            bridge.inject_project_json_into_roto(proj, node, key_step=1)
            self.assertTrue(node["curves"].changed_called)
            shape = node["curves"].rootLayer[0]
            # Nuke Y = H - y_img = 100 - 25 = 75; tangent Y negated
            self.assertEqual(shape[0].center.getPosition(1.0), (40.0, 75.0, 0.0))
            self.assertEqual(shape[0].leftTangent.getPosition(1.0), (-5.0, -3.0, 0.0))
            self.assertEqual(shape[0].rightTangent.getPosition(1.0), (5.0, 3.0, 0.0))

            extracted = bridge.extract_roto_keyframes_from_node(node, [1, 2], height=100.0)
            self.assertIn("Part_01", extracted)
            np.testing.assert_allclose(extracted["Part_01"]["1"][0]["anchor"], [40.0, 25.0], atol=1e-6)
            np.testing.assert_allclose(extracted["Part_01"]["1"][0]["in_handle"], [-5.0, 3.0], atol=1e-6)
            np.testing.assert_allclose(extracted["Part_01"]["2"][1]["out_handle"], [4.0, 2.0], atol=1e-6)
        finally:
            bridge.rp = old_rp


if __name__ == "__main__":
    unittest.main()
