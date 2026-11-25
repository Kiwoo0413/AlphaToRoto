"""
EditableAlpha Pure-Stdlib Foundry Nuke Bridge (editablealpha_bridge.py)
Runs inside Foundry Nuke's bundled Python without any NumPy, OpenCV, or PyTorch
dependencies. Heavy image/geometry/tracking work is delegated to an external
Python subprocess (`editablealpha_worker.py`) via JSON IPC.
"""

import os
import re
import sys
import json
import shutil
import tempfile
import subprocess
from typing import Any, Dict, List, Optional, Tuple

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

try:
    import nuke
    import nuke.rotopaint as rp
except ImportError:
    nuke = None
    rp = None


_KNOWN_PYTHON_CANDIDATES = [
    os.environ.get("EDITABLEALPHA_PYTHON", ""),
    r"C:\Users\yido2\AppData\Local\Programs\Python\Python312\python.exe",
    "python",
    "python3",
]


def get_nuke_cache_temp_dir(subfolder: str = "EditableAlpha_Temp") -> str:
    """Returns a writable temp directory under Nuke's DiskCachePath or OS temp."""
    candidates: List[str] = []
    if nuke is not None:
        try:
            pref = nuke.toNode("preferences")
            if pref is not None:
                for knob_name in ("DiskCachePath", "localCachePath", "DiskCacheDirectory"):
                    if pref.knob(knob_name):
                        val = pref[knob_name].evaluate()
                        if val and str(val).strip():
                            candidates.append(
                                os.path.expandvars(os.path.expanduser(str(val).strip()))
                            )
        except Exception:
            pass

    for env_var in ("NUKE_DISK_CACHE", "NUKE_TEMP_DIR", "EDITABLEALPHA_CACHE_DIR"):
        v = os.environ.get(env_var)
        if v and os.path.isdir(v):
            candidates.append(v)

    for path in candidates:
        try:
            clean = path.replace("\\", "/").rstrip("/")
            os.makedirs(clean, exist_ok=True)
            if os.path.isdir(clean) and os.access(clean, os.W_OK):
                cache_dir = os.path.join(clean, subfolder).replace("\\", "/")
                os.makedirs(cache_dir, exist_ok=True)
                return cache_dir
        except Exception:
            continue

    fallback = os.path.join(tempfile.gettempdir(), subfolder).replace("\\", "/")
    os.makedirs(fallback, exist_ok=True)
    return fallback


def parse_keyframe_list_string(text: str) -> List[int]:
    """Parses comma- or whitespace-separated frame integers into a sorted unique list."""
    if not text or not str(text).strip():
        return []
    frames = set()
    for tok in re.split(r"[,;\s]+", str(text).strip()):
        if not tok:
            continue
        try:
            frames.add(int(tok))
        except ValueError:
            pass
    return sorted(frames)


def find_worker_python(override_path: str = "") -> Optional[str]:
    """Discovers an external Python interpreter with cv2 and numpy installed."""
    candidates: List[str] = []
    if override_path and str(override_path).strip():
        candidates.append(str(override_path).strip())
    for c in _KNOWN_PYTHON_CANDIDATES:
        if c and c not in candidates:
            candidates.append(c)

    for py in candidates:
        try:
            proc = subprocess.run(
                [py, "-c", "import cv2, numpy; print('OK')"],
                capture_output=True,
                text=True,
                timeout=6,
            )
            if proc.returncode == 0 and "OK" in (proc.stdout or ""):
                return py
        except Exception:
            continue
    return None


def get_root_layer(roto_node: Any) -> Any:
    """Safely retrieves the rootLayer / root from a Nuke Roto/RotoPaint node."""
    curves = roto_node["curves"]
    if hasattr(curves, "rootLayer"):
        return curves.rootLayer
    if hasattr(curves, "root"):
        return curves.root
    return None


def _ensure_roto_format(roto_node: Any, width: int, height: int) -> None:
    if nuke is None:
        return
    try:
        fmt_name = f"EditableAlpha_{int(width)}x{int(height)}"
        nuke.addFormat(f"{int(width)} {int(height)} 1 {fmt_name}")
        if roto_node.knob("format"):
            roto_node["format"].setValue(fmt_name)
    except Exception:
        pass


def inject_project_json_into_roto(
    project: Dict[str, Any],
    roto_node: Any,
    key_step: int = 1,
    clear_existing: bool = True,
) -> Any:
    """
    Injects serialized EditableAlpha cubic Bezier parts into a Nuke Roto/RotoPaint node
    with Y-axis flip (y_nuke = H - y_img) and relative left/right tangent handles.
    """
    if rp is None:
        raise RuntimeError("nuke.rotopaint is only available inside Foundry Nuke.")

    curves_knob = roto_node["curves"]
    root_layer = get_root_layer(roto_node)
    if root_layer is None:
        raise RuntimeError("Could not access rootLayer on target Roto node.")

    if clear_existing:
        while len(root_layer) > 0:
            root_layer.remove(0)

    w = int(project["width"])
    h = float(project["height"])
    _ensure_roto_format(roto_node, w, int(h))

    frame_numbers = [int(f) for f in project["frame_numbers"]]
    step = max(1, int(key_step))

    for part in project.get("parts", []):
        shape = rp.Shape(curves_knob)
        shape.name = str(part.get("name", "Part_01"))
        num_pts = int(part["num_control_points"])

        ref_set = {int(k) for k in part.get("reference_keyframes", {}).keys()}
        manual_set = {int(k) for k in part.get("manual_keyframes", {}).keys()}
        manual_set.update(int(k) for k in part.get("manual_keyframe_frames", []))
        keep_set = ref_set | manual_set

        for _ in range(num_pts):
            shape.append(rp.ShapeControlPoint())

        frames_map = part.get("frames", {})
        for idx_f, f_num in enumerate(frame_numbers):
            if (
                (idx_f % step != 0)
                and (f_num not in keep_set)
                and (f_num != frame_numbers[0])
                and (f_num != frame_numbers[-1])
            ):
                continue

            f_cps = frames_map.get(str(f_num))
            if f_cps is None:
                f_cps = frames_map.get(f_num)
            if f_cps is None:
                continue

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


def extract_roto_keyframes_from_node(
    roto_node: Any,
    frame_numbers: List[int],
    height: float,
) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """
    Reads cubic Bezier control points & tangent handles from a Nuke Roto node at
    `frame_numbers` and converts them back to image-space (top-left origin) JSON.
    """
    root_layer = get_root_layer(roto_node)
    if root_layer is None:
        return {}

    h = float(height)
    extracted: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for elem in root_layer:
        shape_name = getattr(elem, "name", "")
        if not shape_name or not hasattr(elem, "__len__"):
            continue
        num_pts = len(elem)
        if num_pts == 0:
            continue
        per_frame: Dict[str, List[Dict[str, Any]]] = {}
        for f_num in frame_numbers:
            tf = float(f_num)
            cps: List[Dict[str, Any]] = []
            for i in range(num_pts):
                cp = elem[i]
                c_pos = cp.center.getPosition(tf)
                lt_pos = cp.leftTangent.getPosition(tf)
                rt_pos = cp.rightTangent.getPosition(tf)
                cps.append(
                    {
                        "index": i,
                        "anchor": [float(c_pos[0]), h - float(c_pos[1])],
                        "in_handle": [float(lt_pos[0]), -float(lt_pos[1])],
                        "out_handle": [float(rt_pos[0]), -float(rt_pos[1])],
                        "is_keyframe": True,
                    }
                )
            per_frame[str(int(f_num))] = cps
        extracted[shape_name] = per_frame
    return extracted


def export_input_frames_via_write(
    source_node: Any,
    start_f: int,
    end_f: int,
) -> Tuple[List[str], str]:
    """Renders 8-bit RGBA PNG frames across [start_f, end_f] into Nuke's DiskCachePath."""
    if nuke is None:
        raise RuntimeError("export_input_frames_via_write requires Foundry Nuke.")

    base_cache = get_nuke_cache_temp_dir("EditableAlpha_Temp")
    temp_dir = tempfile.mkdtemp(prefix="ea_rgba_", dir=base_cache)
    pattern = os.path.join(temp_dir, "rgba_%04d.png").replace("\\", "/")

    write_node = nuke.nodes.Write(
        inputs=[source_node],
        file=pattern,
        file_type="png",
        channels="rgba",
    )
    try:
        nuke.execute(write_node, int(start_f), int(end_f), 1)
    finally:
        nuke.delete(write_node)

    paths: List[str] = []
    for f in range(int(start_f), int(end_f) + 1):
        p = os.path.join(temp_dir, f"rgba_{f:04d}.png")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Rendered frame not found: {p}")
        paths.append(p)
    return paths, temp_dir


def run_worker_subprocess(
    job_dict: Dict[str, Any],
    progress_title: str,
    worker_python: str,
) -> Dict[str, Any]:
    """Runs `editablealpha_worker.py` as an external subprocess with progress feedback."""
    worker_script = os.path.join(_PKG_DIR, "editablealpha_worker.py")
    if not os.path.isfile(worker_script):
        raise FileNotFoundError(f"Worker script missing: {worker_script}")

    cache_dir = get_nuke_cache_temp_dir("EditableAlpha_Temp")
    work_dir = tempfile.mkdtemp(prefix="ea_job_", dir=cache_dir)
    job_path = os.path.join(work_dir, "job.json")
    out_path = os.path.join(work_dir, "result.json")

    with open(job_path, "w", encoding="utf-8") as f:
        json.dump(job_dict, f)

    progress = nuke.ProgressTask(progress_title) if nuke is not None else None
    stdout_lines: List[str] = []
    try:
        proc = subprocess.Popen(
            [worker_python, worker_script, "--job", job_path, "--out", out_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\r\n")
            stdout_lines.append(line)
            if line.startswith("PROGRESS:") and progress is not None:
                parts = line.split(":", 2)
                if len(parts) == 3:
                    try:
                        progress.setProgress(int(parts[1]))
                        progress.setMessage(parts[2])
                    except Exception:
                        pass
            if progress is not None and progress.isCancelled():
                proc.terminate()
                raise RuntimeError("Cancelled by user.")

        rc = proc.wait()
        if rc != 0:
            tail = "\n".join(stdout_lines[-15:])
            raise RuntimeError(f"Worker exited with code {rc}:\n{tail}")

        with open(out_path, "r", encoding="utf-8") as f:
            return json.load(f)
    finally:
        if progress is not None:
            del progress
        shutil.rmtree(work_dir, ignore_errors=True)


def _get_knob_val(node: Any, name: str, default: Any = None) -> Any:
    k = node.knob(name)
    return k.value() if k is not None else default


def _set_status(node: Any, msg: str) -> None:
    if node is not None and node.knob("ea_status"):
        node["ea_status"].setValue(str(msg))


def check_environment_status(node: Any = None) -> Dict[str, Any]:
    """Verifies worker Python (cv2+numpy), Gizmo file, and worker script."""
    override = str(_get_knob_val(node, "ea_python_path", "")) if node is not None else ""
    py = find_worker_python(override)
    gizmo_ok = os.path.isfile(os.path.join(_PKG_DIR, "EditableAlpha.gizmo"))
    worker_ok = os.path.isfile(os.path.join(_PKG_DIR, "editablealpha_worker.py"))
    status = {
        "worker_python": py,
        "gizmo_found": gizmo_ok,
        "worker_found": worker_ok,
        "ready": bool(py and gizmo_ok and worker_ok),
    }
    msg = (
        f"Ready (Worker: {py})"
        if status["ready"]
        else f"Missing dependency — Worker Python: {py or 'NOT FOUND'}, Gizmo: {gizmo_ok}, Worker: {worker_ok}"
    )
    _set_status(node, msg)
    if nuke is not None and node is None:
        nuke.message(f"EditableAlpha Environment Check\n\n{msg}")
    return status


def on_editablealpha_sync_range(gizmo_node: Any) -> None:
    if nuke is None:
        return
    src = gizmo_node.input(0)
    if src is not None:
        try:
            start = int(src.firstFrame())
            end = int(src.lastFrame())
        except Exception:
            start = int(nuke.root()["first_frame"].value())
            end = int(nuke.root()["last_frame"].value())
    else:
        start = int(nuke.root()["first_frame"].value())
        end = int(nuke.root()["last_frame"].value())

    if gizmo_node.knob("ea_start_frame"):
        gizmo_node["ea_start_frame"].setValue(start)
    if gizmo_node.knob("ea_end_frame"):
        gizmo_node["ea_end_frame"].setValue(end)
    _set_status(gizmo_node, f"Frame range synced: {start} - {end}")


def _find_or_create_standalone_roto(gizmo_node: Any, source_node: Any) -> Any:
    target_name = f"{gizmo_node.name()}_Roto"
    existing = nuke.toNode(target_name)
    if existing is not None and existing.Class() in ("Roto", "RotoPaint"):
        return existing

    roto_node = nuke.createNode("Roto", inpanel=False)
    roto_node.setName(target_name)
    if source_node is not None:
        try:
            roto_node.setInput(0, source_node)
        except Exception:
            pass
    try:
        roto_node.setXYpos(gizmo_node.xpos() + 140, gizmo_node.ypos())
    except Exception:
        pass
    return roto_node


def _apply_project_to_gizmo_targets(
    gizmo_node: Any,
    project: Dict[str, Any],
    source_node: Any,
    key_step: int,
    output_mode_str: str,
) -> Optional[Any]:
    mode_low = str(output_mode_str).lower()
    update_internal = "internal" in mode_low
    extract_standalone = "extract" in mode_low or "standalone" in mode_low

    if update_internal:
        internal_roto = gizmo_node.node("EditableAlpha_Roto")
        if internal_roto is not None:
            inject_project_json_into_roto(project, internal_roto, key_step=key_step)

    standalone = None
    if extract_standalone:
        standalone = _find_or_create_standalone_roto(gizmo_node, source_node)
        inject_project_json_into_roto(project, standalone, key_step=key_step)
    return standalone


def _save_cached_project(gizmo_node: Any, project: Dict[str, Any]) -> str:
    cache_dir = get_nuke_cache_temp_dir("EditableAlpha_Temp")
    safe_name = re.sub(r"[^A-Za-z0-9_]", "_", gizmo_node.name()) or "EditableAlpha"
    cache_path = os.path.join(cache_dir, f"project_{safe_name}.json").replace("\\", "/")
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(project, f)
    if gizmo_node.knob("ea_last_project_json"):
        gizmo_node["ea_last_project_json"].setValue(cache_path)
    return cache_path


def on_editablealpha_convert(gizmo_node: Any) -> Optional[Dict[str, Any]]:
    """Converts the connected input mask sequence into editable multi-part Bezier Roto curves."""
    if nuke is None:
        return None

    source_node = gizmo_node.input(0)
    if source_node is None:
        nuke.message(f"Please connect an Alpha Mask or RGBA sequence to '{gizmo_node.name()}'.")
        return None

    start_f = int(_get_knob_val(gizmo_node, "ea_start_frame", nuke.root()["first_frame"].value()))
    end_f = int(_get_knob_val(gizmo_node, "ea_end_frame", nuke.root()["last_frame"].value()))
    if start_f > end_f:
        nuke.message("Start Frame must be <= End Frame.")
        return None

    worker_py = find_worker_python(str(_get_knob_val(gizmo_node, "ea_python_path", "")))
    if not worker_py:
        nuke.message(
            "Could not find an external Python with OpenCV & NumPy.\n"
            "Set 'Worker Python' on the EditableAlpha node or set EDITABLEALPHA_PYTHON."
        )
        return None

    mask_chan = str(_get_knob_val(gizmo_node, "ea_mask_channel", "alpha"))
    use_alpha = "alpha" in mask_chan.lower()
    key_step = max(1, int(_get_knob_val(gizmo_node, "ea_key_step", 1)))
    ref_override = parse_keyframe_list_string(str(_get_knob_val(gizmo_node, "ea_ref_frames", "")))
    output_mode = str(_get_knob_val(gizmo_node, "ea_output_mode", "Update Internal + Extract Roto"))

    temp_dir = None
    try:
        _set_status(gizmo_node, f"Rendering input mask frames {start_f}-{end_f}...")
        frame_paths, temp_dir = export_input_frames_via_write(source_node, start_f, end_f)
        frame_numbers = list(range(start_f, end_f + 1))

        job = {
            "mode": "convert",
            "frame_paths": frame_paths,
            "frame_numbers": frame_numbers,
            "use_alpha": use_alpha,
            "key_step": key_step,
            "max_parts": max(1, int(_get_knob_val(gizmo_node, "ea_max_parts", 3))),
            "overlap_margin_px": max(0, int(_get_knob_val(gizmo_node, "ea_overlap_px", 12))),
            "min_points_per_part": max(4, int(_get_knob_val(gizmo_node, "ea_min_pts", 8))),
            "max_points_per_part": max(6, int(_get_knob_val(gizmo_node, "ea_max_pts", 16))),
            "default_tension": float(_get_knob_val(gizmo_node, "ea_tension", 0.5)),
            "cusp_angle_deg": float(_get_knob_val(gizmo_node, "ea_cusp_angle", 60.0)),
            "auto_multi_master": bool(_get_knob_val(gizmo_node, "ea_auto_multi_master", True)),
            "master_frames_override": ref_override or None,
        }

        res = run_worker_subprocess(
            job,
            progress_title="EditableAlpha: Converting Mask to Roto...",
            worker_python=worker_py,
        )
        project = res["project"]
        metrics = res.get("metrics", {})
        _save_cached_project(gizmo_node, project)

        _apply_project_to_gizmo_targets(
            gizmo_node,
            project,
            source_node=source_node,
            key_step=key_step,
            output_mode_str=output_mode,
        )

        masters = project.get("master_frames", [project.get("master_frame", start_f)])
        if gizmo_node.knob("ea_ref_frames") and not ref_override:
            gizmo_node["ea_ref_frames"].setValue(", ".join(str(m) for m in masters))

        msg = (
            f"Converted {len(project['parts'])} part(s) | "
            f"masters=[{', '.join(str(m) for m in masters)}] | "
            f"J&F={metrics.get('J_and_F', 0)}%"
        )
        _set_status(gizmo_node, msg)
        return project
    except Exception as exc:
        _set_status(gizmo_node, f"Error: {exc}")
        nuke.message(f"EditableAlpha Conversion Error:\n{exc}")
        return None
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def on_editablealpha_extract_roto(gizmo_node: Any) -> Optional[Any]:
    """Extracts the converted Bezier curves into a standalone editable Nuke Roto node."""
    if nuke is None:
        return None

    cache_path = str(_get_knob_val(gizmo_node, "ea_last_project_json", "")).strip()
    if not cache_path or not os.path.isfile(cache_path):
        nuke.message("No converted Roto data found yet. Click 'Convert Mask to Roto' first.")
        return None

    with open(cache_path, "r", encoding="utf-8") as f:
        project = json.load(f)

    key_step = max(1, int(_get_knob_val(gizmo_node, "ea_key_step", 1)))
    roto_node = _find_or_create_standalone_roto(gizmo_node, gizmo_node.input(0))
    inject_project_json_into_roto(project, roto_node, key_step=key_step)
    _set_status(gizmo_node, f"Extracted standalone Roto '{roto_node.name()}'.")
    return roto_node


def on_editablealpha_retrack_manual(gizmo_node: Any) -> Optional[Dict[str, Any]]:
    """
    Reads manually edited Bezier control points from the extracted (or internal) Roto node
    at the current frame, re-tracks adjacent frames while preserving manual edits 100%,
    and updates the Roto node(s).
    """
    if nuke is None:
        return None

    source_node = gizmo_node.input(0)
    if source_node is None:
        nuke.message(f"Please connect the input mask to '{gizmo_node.name()}' before re-tracking.")
        return None

    cache_path = str(_get_knob_val(gizmo_node, "ea_last_project_json", "")).strip()
    if not cache_path or not os.path.isfile(cache_path):
        nuke.message("No converted project found. Run 'Convert Mask to Roto' first.")
        return None

    worker_py = find_worker_python(str(_get_knob_val(gizmo_node, "ea_python_path", "")))
    if not worker_py:
        nuke.message("Worker Python (with cv2 & numpy) not found.")
        return None

    with open(cache_path, "r", encoding="utf-8") as f:
        project = json.load(f)

    standalone = nuke.toNode(f"{gizmo_node.name()}_Roto")
    edit_source = standalone if standalone is not None else gizmo_node.node("EditableAlpha_Roto")
    if edit_source is None:
        nuke.message("Could not find a Roto node containing edited curves.")
        return None

    curr_f = int(nuke.frame())
    frame_numbers = [int(fn) for fn in project["frame_numbers"]]
    if curr_f not in frame_numbers:
        curr_f = frame_numbers[0]

    extracted = extract_roto_keyframes_from_node(edit_source, [curr_f], float(project["height"]))
    if not extracted:
        nuke.message("No Bezier shapes found to preserve.")
        return None

    start_f, end_f = frame_numbers[0], frame_numbers[-1]
    use_alpha = "alpha" in str(_get_knob_val(gizmo_node, "ea_mask_channel", "alpha")).lower()
    key_step = max(1, int(_get_knob_val(gizmo_node, "ea_key_step", 1)))
    output_mode = str(_get_knob_val(gizmo_node, "ea_output_mode", "Update Internal + Extract Roto"))

    temp_dir = None
    try:
        frame_paths, temp_dir = export_input_frames_via_write(source_node, start_f, end_f)
        job = {
            "mode": "retrack",
            "frame_paths": frame_paths,
            "frame_numbers": frame_numbers,
            "use_alpha": use_alpha,
            "project": project,
            "manual_keyframes_by_part": extracted,
            "default_tension": float(_get_knob_val(gizmo_node, "ea_tension", 0.5)),
            "cusp_angle_deg": float(_get_knob_val(gizmo_node, "ea_cusp_angle", 60.0)),
            "auto_multi_master": bool(_get_knob_val(gizmo_node, "ea_auto_multi_master", True)),
        }
        res = run_worker_subprocess(
            job,
            progress_title=f"EditableAlpha: Re-Tracking (Preserving F{curr_f})...",
            worker_python=worker_py,
        )
        updated = res["project"]
        _save_cached_project(gizmo_node, updated)
        _apply_project_to_gizmo_targets(
            gizmo_node,
            updated,
            source_node=source_node,
            key_step=key_step,
            output_mode_str=output_mode,
        )
        _set_status(gizmo_node, f"Re-tracked preserving manual edits at F{curr_f}.")
        return updated
    except Exception as exc:
        _set_status(gizmo_node, f"Re-Track Error: {exc}")
        nuke.message(f"EditableAlpha Re-Track Error:\n{exc}")
        return None
    finally:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def create_editablealpha_gizmo() -> Any:
    """Creates an EditableAlpha Gizmo node and auto-connects to the selected node."""
    if nuke is None:
        raise RuntimeError("create_editablealpha_gizmo requires Foundry Nuke.")

    sel = None
    try:
        sel = nuke.selectedNode()
    except Exception:
        pass

    node = nuke.createNode("EditableAlpha")
    if sel is not None:
        try:
            node.setInput(0, sel)
        except Exception:
            pass
    on_editablealpha_sync_range(node)
    return node
