"""
AlphaToRoto Lightweight Foundry Nuke Bridge (nuke_bridge.py)
Provides Foundry Nuke Roto/RotoPaint node integration for Web Studio outputs and
direct in-Nuke Stage 1-4 Alpha-to-Roto node conversion without external CoTracker3 bloat.
"""

import os
import re
import sys
import json
import shutil
import tempfile
from typing import List, Optional, Tuple

_curr_dir = os.path.dirname(os.path.abspath(__file__))
if _curr_dir not in sys.path:
    sys.path.insert(0, _curr_dir)

try:
    from PySide6 import QtWidgets, QtCore
except ImportError:
    try:
        from PySide2 import QtWidgets, QtCore
    except ImportError:
        QtWidgets = None
        QtCore = None

try:
    import nuke
    import nuke.rotopaint as rp
except ImportError:
    nuke = None
    rp = None

from autoroto_pipeline import AlphaToRotoPipeline, blend_bidirectional_trajectories


_NATIVE_ROTO_TABS = {"transform", "motion blur", "shape", "clone", "lifetime", "tracking"}


def get_nuke_cache_temp_dir() -> str:
    """Returns a writable AutoRoto_Temp cache directory under Nuke's disk cache or OS temp."""
    candidates = []
    try:
        pref = nuke.toNode("preferences")
        if pref:
            for knob_name in ("DiskCachePath", "localCachePath", "DiskCacheDirectory"):
                if pref.knob(knob_name):
                    val = pref[knob_name].value()
                    if val and str(val).strip():
                        candidates.append(os.path.expandvars(os.path.expanduser(str(val).strip())))
    except Exception:
        pass

    for env_var in ("NUKE_DISK_CACHE", "NUKE_TEMP_DIR", "AUTOROTO_CACHE_DIR"):
        v = os.environ.get(env_var)
        if v and os.path.isdir(v):
            candidates.append(v)

    for path in candidates:
        try:
            clean = path.replace("\\", "/").rstrip("/")
            os.makedirs(clean, exist_ok=True)
            if os.path.isdir(clean) and os.access(clean, os.W_OK):
                cache_dir = os.path.join(clean, "AutoRoto_Temp").replace("\\", "/")
                os.makedirs(cache_dir, exist_ok=True)
                return cache_dir
        except Exception:
            continue

    fallback = os.path.join(tempfile.gettempdir(), "AutoRoto_Temp").replace("\\", "/")
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


def export_source_rgba_frames(source_node, start_f: int, end_f: int) -> Tuple[List[str], Optional[str]]:
    """Renders 8-bit RGBA PNG frames across [start_f, end_f] using Nuke's native Write node."""
    base_cache = get_nuke_cache_temp_dir()
    temp_dir = tempfile.mkdtemp(prefix="autoroto_rgba_", dir=base_cache)
    pattern = os.path.join(temp_dir, "rgba_%04d.png").replace("\\", "/")

    write_node = nuke.nodes.Write(
        inputs=[source_node],
        file=pattern,
        file_type="png",
        channels="rgba",
    )
    try:
        nuke.execute(write_node, start_f, end_f, 1)
    finally:
        nuke.delete(write_node)

    paths: List[str] = []
    for f in range(start_f, end_f + 1):
        p = os.path.join(temp_dir, f"rgba_{f:04d}.png")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Rendered RGBA frame not found: {p}")
        paths.append(p)
    return paths, temp_dir


def hide_intermediate_native_tabs(node):
    """Hides secondary native Nuke Roto tabs so AlphaToRoto sits cleanly beside Roto."""
    for k in node.allKnobs():
        if isinstance(k, nuke.Tab_Knob):
            n_low = (k.name() or "").lower()
            l_low = (k.label() or "").lower()
            if any(t in n_low or t in l_low for t in _NATIVE_ROTO_TABS):
                try:
                    k.setFlag(nuke.INVISIBLE)
                    k.setVisible(False)
                except Exception:
                    pass


def toggle_native_roto_tabs(node):
    """Toggles secondary native Nuke Roto tabs visibility on demand."""
    is_hidden = False
    for k in node.allKnobs():
        if isinstance(k, nuke.Tab_Knob):
            n_low = (k.name() or "").lower()
            l_low = (k.label() or "").lower()
            if any(t in n_low or t in l_low for t in _NATIVE_ROTO_TABS):
                if k.getFlag(nuke.INVISIBLE) or not k.visible():
                    is_hidden = True
                    break

    should_hide = not is_hidden
    for k in node.allKnobs():
        if isinstance(k, nuke.Tab_Knob):
            n_low = (k.name() or "").lower()
            l_low = (k.label() or "").lower()
            if any(t in n_low or t in l_low for t in _NATIVE_ROTO_TABS):
                try:
                    if should_hide:
                        k.setFlag(nuke.INVISIBLE)
                        k.setVisible(False)
                    else:
                        k.clearFlag(nuke.INVISIBLE)
                        k.setVisible(True)
                except Exception:
                    pass

    if node.knob("ar_status"):
        state_text = "hidden (AlphaToRoto is Tab 1)" if should_hide else "restored"
        node.knob("ar_status").setValue(f"Extra tabs {state_text}.")


def make_autoroto_first_tab(node_name: Optional[str] = None, focus: bool = False):
    """Moves the AlphaToRoto tab to Tab 1 in Nuke's Properties panel."""
    if not QtWidgets:
        return

    def _do_adjust():
        app = QtWidgets.QApplication.instance()
        if not app:
            return
        for w in app.allWidgets():
            if isinstance(w, QtWidgets.QTabBar):
                try:
                    auto_idx = next(
                        (
                            i
                            for i in range(w.count())
                            if "AutoRoto" in w.tabText(i).replace("&", "").strip()
                            or "AlphaToRoto" in w.tabText(i).replace("&", "").strip()
                        ),
                        -1,
                    )
                    if auto_idx != -1:
                        if auto_idx != 0:
                            w.moveTab(auto_idx, 0)
                        if focus:
                            w.setCurrentIndex(0)
                except Exception:
                    pass

    _do_adjust()
    if QtCore:
        QtCore.QTimer.singleShot(30, _do_adjust)
        QtCore.QTimer.singleShot(120, _do_adjust)


def on_node_knob_changed(node, knob):
    """Aligns the AlphaToRoto tab when opening the node's Properties panel."""
    if not node or not knob:
        return
    try:
        if knob.name() == "showPanel":
            make_autoroto_first_tab(node.name(), focus=True)
    except Exception:
        pass


def setup_autoroto_knobs(node):
    """Equips a native Nuke Roto node with the lightweight AlphaToRoto Tab 1 UI."""
    saved_ref = ""
    saved_start = None
    saved_end = None
    saved_step = None

    if node.knob("AutoRoto") or node.knob("AutoRotoTrackerTab"):
        if node.knob("ar_ref_frame"):
            try:
                old_ref = node["ar_ref_frame"]
                saved_ref = (
                    str(int(old_ref.value()))
                    if old_ref.Class() == "Int_Knob"
                    else str(old_ref.value()).strip()
                )
            except Exception:
                pass
        if node.knob("ar_start_frame"):
            saved_start = int(node["ar_start_frame"].value())
        if node.knob("ar_end_frame"):
            saved_end = int(node["ar_end_frame"].value())
        if node.knob("ar_key_step"):
            saved_step = int(node["ar_key_step"].value())

        for k in [
            name
            for name in node.knobs().keys()
            if name.startswith("ar_") or name in ("AutoRoto", "AutoRotoTrackerTab")
        ]:
            try:
                node.removeKnob(node.knob(k))
            except Exception:
                pass

    node.addKnob(nuke.Tab_Knob("AutoRoto", "AlphaToRoto"))
    node.addKnob(
        nuke.Text_Knob(
            "ar_title",
            "",
            '<font size=4 color="#e58934"><b>AlphaToRoto</b></font> '
            '<font color="#888">Web Studio Bridge &amp; Parametric Bezier Roto</font>',
        )
    )

    curr_f = int(nuke.frame())
    start_f = int(nuke.root()["first_frame"].value())
    end_f = int(nuke.root()["last_frame"].value())

    k_ref = nuke.String_Knob("ar_ref_frame", "Ref Frame")
    k_ref.setValue(saved_ref if saved_ref else str(curr_f))
    k_ref.setTooltip('Linked representative reference frames (e.g. "2, 5, 11").')
    node.addKnob(k_ref)

    k_start = nuke.Int_Knob("ar_start_frame", "Start Frame")
    k_start.setValue(saved_start if saved_start is not None else start_f)
    node.addKnob(k_start)

    k_end = nuke.Int_Knob("ar_end_frame", "End Frame")
    k_end.setValue(saved_end if saved_end is not None else end_f)
    node.addKnob(k_end)

    k_sync = nuke.PyScript_Knob(
        "ar_sync_range",
        "Project Range",
        "import nuke_bridge; nuke_bridge.on_node_sync_range(nuke.thisNode())",
    )
    k_sync.clearFlag(nuke.STARTLINE)
    node.addKnob(k_sync)

    k_step = nuke.Int_Knob("ar_key_step", "Key Step")
    k_step.setValue(saved_step if saved_step is not None else 1)
    k_step.setRange(1, 10)
    node.addKnob(k_step)

    node.addKnob(
        nuke.Text_Knob(
            "ar_div_alpha",
            "",
            '<b><font color="#56B4E9">Alpha Mask ➔ Editable Bezier Roto (Stage 1–4)</font></b>',
        )
    )

    k_max_parts = nuke.Int_Knob("ar_max_parts", "Max Parts")
    k_max_parts.setValue(3)
    k_max_parts.setRange(1, 6)
    node.addKnob(k_max_parts)

    k_overlap = nuke.Int_Knob("ar_overlap_px", "Seam Overlap (px)")
    k_overlap.setValue(12)
    k_overlap.setRange(0, 25)
    k_overlap.clearFlag(nuke.STARTLINE)
    node.addKnob(k_overlap)

    k_max_pts = nuke.Int_Knob("ar_max_pts", "Max Points / Part")
    k_max_pts.setValue(16)
    k_max_pts.setRange(6, 32)
    node.addKnob(k_max_pts)

    k_chan = nuke.Enumeration_Knob(
        "ar_mask_channel",
        "Mask Source",
        ["alpha (Input RGBA/Alpha)", "luminance (Input RGB/BW)"],
    )
    k_chan.setValue("alpha (Input RGBA/Alpha)")
    k_chan.clearFlag(nuke.STARTLINE)
    node.addKnob(k_chan)

    btn_import = nuke.PyScript_Knob(
        "ar_import_project",
        "📂 Import Web Studio Project JSON",
        "import nuke_bridge; nuke_bridge.on_node_import_web_project(nuke.thisNode())",
    )
    btn_import.setFlag(nuke.STARTLINE)
    node.addKnob(btn_import)

    btn_convert = nuke.PyScript_Knob(
        "ar_convert_mask",
        '<b><font color="#56B4E9">✨ Convert Input Alpha Mask to Editable Multi-Part Roto</font></b>',
        "import nuke_bridge; nuke_bridge.on_node_convert_alpha_to_roto(nuke.thisNode())",
    )
    btn_convert.setFlag(nuke.STARTLINE)
    node.addKnob(btn_convert)

    btn_tabs = nuke.PyScript_Knob(
        "ar_toggle_tabs",
        "Toggle Extra Tabs",
        "import nuke_bridge; nuke_bridge.toggle_native_roto_tabs(nuke.thisNode())",
    )
    btn_tabs.setFlag(nuke.STARTLINE)
    node.addKnob(btn_tabs)

    k_status = nuke.String_Knob("ar_status", "Status")
    k_status.setValue("Ready. Import Web Studio JSON or convert connected Alpha input.")
    node.addKnob(k_status)

    hide_intermediate_native_tabs(node)
    node.knob("knobChanged").setValue(
        "import nuke_bridge\nnuke_bridge.on_node_knob_changed(nuke.thisNode(), nuke.thisKnob())\n"
    )
    make_autoroto_first_tab(node.name())


def create_autoroto_node():
    """Creates a Nuke Roto node equipped with AlphaToRoto Tab 1 UI."""
    sel = None
    try:
        sel = nuke.selectedNode()
    except Exception:
        pass

    node = nuke.createNode("Roto")
    node.setName("AlphaToRoto1")
    if sel:
        try:
            node.setInput(0, sel)
            node["format"].setValue(sel.format().name())
        except Exception:
            pass
    setup_autoroto_knobs(node)
    hide_intermediate_native_tabs(node)
    make_autoroto_first_tab(node.name(), focus=True)
    return node


create_alphatoroto_node = create_autoroto_node


def add_autoroto_to_selected():
    """Attaches the AlphaToRoto Tab 1 UI to the currently selected Roto/RotoPaint node."""
    node = nuke.selectedNode()
    if node and node.Class() in ("Roto", "RotoPaint"):
        if node.input(0):
            try:
                node["format"].setValue(node.input(0).format().name())
            except Exception:
                pass
        setup_autoroto_knobs(node)
        hide_intermediate_native_tabs(node)
        make_autoroto_first_tab(node.name(), focus=True)
        nuke.message(f"Added AlphaToRoto as Tab 1 on '{node.name()}'!")
    else:
        nuke.message("Please select a Roto or RotoPaint node first.")


def on_node_sync_range(roto_node):
    start = int(nuke.root()["first_frame"].value())
    end = int(nuke.root()["last_frame"].value())
    if roto_node.knob("ar_start_frame"):
        roto_node["ar_start_frame"].setValue(start)
    if roto_node.knob("ar_end_frame"):
        roto_node["ar_end_frame"].setValue(end)
    if roto_node.knob("ar_status"):
        roto_node["ar_status"].setValue(f"Range synced to project: {start} - {end}")


def on_node_convert_alpha_to_roto(roto_node):
    """Converts input Alpha/RGBA node connected to `roto_node` via Stage 1-4 AlphaToRotoPipeline."""
    source_node = roto_node.input(0)
    if not source_node:
        nuke.message(f"Please connect an Alpha Mask or RGBA sequence to '{roto_node.name()}'.")
        return None

    start_f = int(roto_node["ar_start_frame"].value()) if roto_node.knob("ar_start_frame") else int(nuke.root()["first_frame"].value())
    end_f = int(roto_node["ar_end_frame"].value()) if roto_node.knob("ar_end_frame") else int(nuke.root()["last_frame"].value())
    if start_f > end_f:
        nuke.message("Start Frame must be <= End Frame.")
        return None

    max_parts = int(roto_node["ar_max_parts"].value()) if roto_node.knob("ar_max_parts") else 3
    overlap_px = int(roto_node["ar_overlap_px"].value()) if roto_node.knob("ar_overlap_px") else 12
    max_pts = int(roto_node["ar_max_pts"].value()) if roto_node.knob("ar_max_pts") else 16
    key_step = int(roto_node["ar_key_step"].value()) if roto_node.knob("ar_key_step") else 1
    mask_chan = roto_node["ar_mask_channel"].value() if roto_node.knob("ar_mask_channel") else "alpha"
    use_alpha = "alpha" in str(mask_chan).lower()

    progress = nuke.ProgressTask("AlphaToRoto: Converting Alpha Mask to Multi-Part Bezier Roto...")
    temp_dir = None
    try:
        import cv2
        from mask_generator import decode_binary_mask_array

        rgba_paths, temp_dir = export_source_rgba_frames(source_node, start_f, end_f)
        frame_numbers = list(range(start_f, end_f + 1))
        masks = []
        frames_rgb = []

        for p in rgba_paths:
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Failed to read rendered frame: {p}")
            src_for_mask = img if (use_alpha or img.ndim < 3) else img[:, :, :3]
            masks.append(decode_binary_mask_array(src_for_mask))
            if img.ndim == 3:
                frames_rgb.append(cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB))
            else:
                frames_rgb.append(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))

        pipeline = AlphaToRotoPipeline(
            overlap_margin_px=overlap_px,
            max_parts=max_parts,
            min_points_per_part=8,
            max_points_per_part=max_pts,
            auto_multi_master=True,
        )
        project = pipeline.build_editable_roto_from_masks(
            masks, frames_rgb=frames_rgb, frame_numbers=frame_numbers
        )
        pipeline.inject_into_nuke_roto_node(project, roto_node=roto_node, key_step=max(1, key_step))

        master_frames = project.get("master_frames", [project["master_frame"]])
        if roto_node.knob("ar_ref_frame"):
            roto_node["ar_ref_frame"].setValue(", ".join(str(k) for k in master_frames))

        msg = (
            f"Converted {len(project['parts'])} part(s), "
            f"masters=[{', '.join(str(k) for k in master_frames)}], "
            f"seam gap={round(project['seam_gap_ratio_master'] * 100.0, 3)}%"
        )
        if roto_node.knob("ar_status"):
            roto_node["ar_status"].setValue(msg)
        nuke.message(f"AlphaToRoto Conversion Complete!\n{msg}")
        return msg
    except Exception as exc:
        if roto_node.knob("ar_status"):
            roto_node["ar_status"].setValue(f"Error: {exc}")
        nuke.message(f"AlphaToRoto Conversion Error:\n{exc}")
        return None
    finally:
        del progress
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


def load_alphatoroto_json_as_node(json_path: str, roto_node=None, key_step: int = 1):
    """Loads an exported alphatoroto_project.json from Web Studio into a Foundry Nuke Roto node."""
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"AlphaToRoto project JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        raw_project = json.load(f)

    project = dict(raw_project)
    project["frame_numbers"] = [int(fn) for fn in raw_project["frame_numbers"]]
    norm_parts = []
    all_refs = set()
    for p in raw_project.get("parts", []):
        np_part = dict(p)
        np_part["reference_keyframes"] = {int(k): v for k, v in p.get("reference_keyframes", {}).items()}
        np_part["frames"] = {int(k): v for k, v in p.get("frames", {}).items()}
        all_refs.update(np_part["reference_keyframes"].keys())
        norm_parts.append(np_part)
    project["parts"] = norm_parts

    if roto_node is None:
        roto_node = create_autoroto_node()

    AlphaToRotoPipeline.inject_into_nuke_roto_node(project, roto_node=roto_node, key_step=max(1, int(key_step)))

    sorted_refs = sorted(all_refs) if all_refs else [int(project.get("master_frame", project["frame_numbers"][0]))]
    if roto_node.knob("ar_ref_frame"):
        roto_node["ar_ref_frame"].setValue(", ".join(str(k) for k in sorted_refs))
    if roto_node.knob("ar_start_frame") and project["frame_numbers"]:
        roto_node["ar_start_frame"].setValue(int(project["frame_numbers"][0]))
    if roto_node.knob("ar_end_frame") and project["frame_numbers"]:
        roto_node["ar_end_frame"].setValue(int(project["frame_numbers"][-1]))
    if roto_node.knob("ar_overlap_px"):
        roto_node["ar_overlap_px"].setValue(int(project.get("overlap_margin_px", 12)))
    if roto_node.knob("ar_max_parts"):
        roto_node["ar_max_parts"].setValue(max(1, len(norm_parts)))
    if roto_node.knob("ar_status"):
        roto_node["ar_status"].setValue(
            f"Imported {len(norm_parts)} part(s) from {os.path.basename(json_path)} (Master F{project.get('master_frame', 1)})."
        )
    hide_intermediate_native_tabs(roto_node)
    make_autoroto_first_tab(roto_node.name(), focus=True)
    return roto_node


def on_node_import_web_project(roto_node):
    """Opens a file picker inside Nuke to import an alphatoroto_project.json from Web Studio."""
    default_dir = os.path.join(_curr_dir, "web_output")
    if not os.path.isdir(default_dir):
        default_dir = _curr_dir

    chosen_path = None
    try:
        chosen_path = nuke.getFilename(
            "Select AlphaToRoto Project JSON",
            "*.json",
            default=default_dir.replace("\\", "/") + "/",
        )
    except Exception:
        chosen_path = None

    if not chosen_path:
        return None

    key_step = int(roto_node["ar_key_step"].value()) if roto_node.knob("ar_key_step") else 1
    try:
        load_alphatoroto_json_as_node(chosen_path, roto_node=roto_node, key_step=key_step)
        nuke.message(f"AlphaToRoto: Imported project '{os.path.basename(chosen_path)}' into '{roto_node.name()}'!")
        return chosen_path
    except Exception as exc:
        if roto_node.knob("ar_status"):
            roto_node["ar_status"].setValue(f"Import Error: {exc}")
        nuke.message(f"AlphaToRoto Import Error:\n{exc}")
        return None


if QtWidgets and QtWidgets.QApplication.instance():
    try:
        make_autoroto_first_tab()
    except Exception:
        pass
