"""
AlphaToRoto Full-Pipeline Interactive Web Studio (FastAPI + Uvicorn)
Thin web adapter over `mask_generator`, `roto_geometry`, and `autoroto_pipeline`:
  Image / Sequence Input
    -> Stage 1: SAM 2.1 (GPU) Segmentation, Vector Flow & Multi-Master Selection
    -> Stage 2: Kinematic-Geometric Part Decomposition, Per-Part Toggles & Pre-Bezier Overlap
    -> Stage 3: O(N) Closed-Form Hermite-to-Bezier Contour Fitting
    -> Stage 4: Multi-Master Linked Propagation & Interactive Keyframe Editing
    -> Export / Import into Foundry Nuke AlphaToRoto Node
"""

import os
import re
import glob
import time
import argparse
from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from mask_generator import (
    TrackedMaskGenerator,
    decode_binary_mask_array,
    make_mask_preview_plate,
    create_synthetic_articulated_sequence,
    select_representative_master_frames,
)
from roto_geometry import (
    rebuild_part_overlap_mask,
    compute_parts_overlap_stats,
)
from autoroto_pipeline import AlphaToRotoPipeline


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_UI_PATH = os.path.join(BASE_DIR, "web_ui", "index.html")
WEB_OUTPUT_DIR = os.path.join(BASE_DIR, "web_output")


def _natural_sort_key(name: str):
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", name)]


def _encode_png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode PNG buffer.")
    return buf.tobytes()


def _load_sorted_images(directory: str, as_mask: bool = False) -> List[np.ndarray]:
    files = sorted(
        [
            os.path.join(directory, fn)
            for fn in os.listdir(directory)
            if fn.lower().endswith((".png", ".jpg", ".jpeg"))
        ],
        key=lambda p: _natural_sort_key(os.path.basename(p)),
    )
    if not files:
        return []
    if as_mask:
        return TrackedMaskGenerator.load_mask_sequence(files)
    imgs: List[np.ndarray] = []
    for p in files:
        bgr = cv2.imread(p, cv2.IMREAD_COLOR)
        if bgr is not None:
            imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return imgs


class PresetRequest(BaseModel):
    preset: str = "sample_sequence"
    rgb_dir: Optional[str] = None
    mask_dir: Optional[str] = None


class Stage1Request(BaseModel):
    prompt_frame_idx: int = 0
    fg_points: List[List[float]] = []
    bg_points: List[List[float]] = []
    bbox: Optional[List[float]] = None
    use_sam: bool = True
    auto_multi_master: bool = True
    flow_motion_threshold_px: float = 22.0
    max_master_frames: int = 4


class MasterFramesRequest(BaseModel):
    master_frames: List[int]
    primary_master_frame: Optional[int] = None


class Stage2Request(BaseModel):
    max_parts: int = 3
    overlap_margin_px: int = 12
    defect_depth_thresh_px: float = 10.0
    master_frame_idx: Optional[int] = None


class PartUpdateRequest(BaseModel):
    part_id: int
    enabled: Optional[bool] = None
    overlap_margin_px: Optional[int] = None
    boundary_offset_px: Optional[int] = None
    max_points: Optional[int] = None


class Stage3Request(BaseModel):
    min_points: int = 8
    max_points: int = 16
    default_tension: float = 0.5
    cusp_angle_deg: float = 60.0


class Stage4Request(BaseModel):
    blending_mode: str = "c1_vis"
    use_tangent_rotation: bool = True
    use_contour_snapping: bool = True


class Stage4EditRequest(BaseModel):
    part_id: int
    edited_frame: int
    edited_control_points: List[Dict[str, Any]]


class NukeExportRequest(BaseModel):
    key_step: int = 1
    attach_autoroto_tab: bool = True


class AlphaToRotoStudioSession:
    """In-memory interactive pipeline state for the active Web Studio workspace."""

    def __init__(self):
        self.mask_gen = TrackedMaskGenerator()
        self.frames_rgb: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
        self.frame_numbers: List[int] = []
        self.width: int = 0
        self.height: int = 0
        self.input_mode: str = "rgb"

        self.master_info: Optional[Dict[str, Any]] = None
        self.stage1_backend: str = ""
        self.stage1_score: float = 0.0

        self.stage2_decomp: Optional[Dict[str, Any]] = None
        self.stage2_overlap_rgba: Optional[np.ndarray] = None
        self.stage2_seam_only_rgba: Optional[np.ndarray] = None
        self.stage2_part_rgba_map: Dict[int, np.ndarray] = {}
        self.stage2_overlap_pixels: int = 0

        self.stage3_params: Dict[str, Any] = {}
        self.stage3_metrics: Optional[Dict[str, Any]] = None
        self.stage4_metrics: Optional[Dict[str, Any]] = None

        self.pipeline: Optional[AlphaToRotoPipeline] = None
        self.project: Optional[Dict[str, Any]] = None
        self.last_export: Optional[Dict[str, Any]] = None

    def reset_downstream(self, keep_masks: bool = False):
        if not keep_masks:
            self.masks = []
            self.master_info = None
            self.stage1_backend = ""
            self.stage1_score = 0.0
        self.stage2_decomp = None
        self.stage2_overlap_rgba = None
        self.stage2_seam_only_rgba = None
        self.stage2_part_rgba_map = {}
        self.stage2_overlap_pixels = 0
        self.stage3_metrics = None
        self.stage4_metrics = None
        self.pipeline = None
        self.project = None
        self.last_export = None

    def summary(self) -> Dict[str, Any]:
        return {
            "has_frames": len(self.frames_rgb) > 0,
            "width": self.width,
            "height": self.height,
            "num_frames": len(self.frame_numbers),
            "frame_numbers": self.frame_numbers,
            "input_mode": self.input_mode,
            "stage1_ready": len(self.masks) > 0 and self.master_info is not None,
            "stage1_backend": self.stage1_backend,
            "master_frame": self.master_info["master_frame"] if self.master_info else None,
            "master_index": self.master_info["master_index"] if self.master_info else None,
            "master_frames": self.master_info.get("master_frames", []) if self.master_info else [],
            "master_indices": self.master_info.get("master_indices", []) if self.master_info else [],
            "large_motion_detected": bool(self.master_info.get("large_motion_detected", False)) if self.master_info else False,
            "flow_profile": self.master_info.get("flow_profile") if self.master_info else None,
            "stage2_ready": self.stage2_decomp is not None,
            "stage3_ready": self.stage3_metrics is not None,
            "stage4_ready": self.stage4_metrics is not None,
            "export_ready": self.last_export is not None,
        }


def _rebuild_stage2_visual_layers(sess: AlphaToRotoStudioSession):
    """Rebuilds individual part RGBA overlays, yellow seam-overlap overlay, and overlap stats."""
    if sess.stage2_decomp is None or sess.master_info is None:
        return

    h, w = sess.height, sess.width
    master_idx = int(sess.master_info["master_index"])
    master_mask = sess.masks[master_idx]
    decomp = sess.stage2_decomp

    part_colors_bgra = [
        (248, 189, 56, 135),
        (22, 115, 249, 135),
        (129, 185, 16, 135),
        (249, 121, 232, 135),
        (180, 140, 240, 135),
        (100, 210, 210, 135),
    ]

    sess.stage2_part_rgba_map = {}
    combined_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    overlap_count = np.zeros((h, w), dtype=np.int32)
    k3 = np.ones((3, 3), np.uint8)

    for idx, p in enumerate(decomp["parts"]):
        pid = int(p["part_id"])
        pm_u8 = (p["mask"] > 127).astype(np.uint8) * 255
        pm_bool = pm_u8 > 0
        col = part_colors_bgra[idx % len(part_colors_bgra)]

        part_rgba = np.zeros((h, w, 4), dtype=np.uint8)
        part_rgba[pm_bool] = col
        edge = cv2.morphologyEx(pm_u8, cv2.MORPH_GRADIENT, k3) > 0
        part_rgba[edge] = (col[0], col[1], col[2], 240)
        sess.stage2_part_rgba_map[pid] = part_rgba

        if p.get("enabled", True):
            overlap_count[pm_bool] += 1
            combined_rgba[pm_bool] = col
            combined_rgba[edge] = (col[0], col[1], col[2], 235)

    overlap_zone = overlap_count >= 2
    seam_only_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    seam_only_rgba[overlap_zone] = (21, 204, 250, 235)
    combined_rgba[overlap_zone] = (21, 204, 250, 235)

    stats = compute_parts_overlap_stats(decomp["parts"], master_mask)
    sess.stage2_seam_only_rgba = seam_only_rgba
    sess.stage2_overlap_rgba = combined_rgba
    sess.stage2_overlap_pixels = stats["overlap_pixels"]
    decomp["seam_gap_ratio"] = stats["seam_gap_ratio"]


def _format_stage2_parts_meta(decomp: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "part_id": int(p["part_id"]),
            "name": p["name"],
            "area_px": int(p["area_px"]),
            "enabled": bool(p.get("enabled", True)),
            "overlap_margin_px": int(p.get("overlap_margin_px", decomp.get("overlap_margin_px", 12))),
            "boundary_offset_px": int(p.get("boundary_offset_px", 0)),
            "max_points": p.get("max_points"),
        }
        for p in decomp["parts"]
    ]


session = AlphaToRotoStudioSession()
app = FastAPI(title="AlphaToRoto Full-Pipeline Web Studio")


@app.get("/", response_class=HTMLResponse)
def get_index():
    if not os.path.isfile(WEB_UI_PATH):
        raise HTTPException(status_code=404, detail=f"UI template missing: {WEB_UI_PATH}")
    with open(WEB_UI_PATH, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), headers={"Cache-Control": "no-store"})


@app.get("/api/status")
def api_status():
    return {
        "sam2": session.mask_gen.get_sam2_status(),
        "session": session.summary(),
    }


@app.post("/api/upload")
async def api_upload(
    files: List[UploadFile] = File(...),
    input_mode: str = Form("rgb"),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    ordered = sorted(files, key=lambda uf: _natural_sort_key(uf.filename or ""))
    frames_rgb: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    target_size: Optional[Tuple[int, int]] = None

    for uf in ordered:
        arr = np.frombuffer(await uf.read(), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise HTTPException(status_code=400, detail=f"Failed to decode image: {uf.filename}")

        if target_size is None:
            target_size = (img.shape[1], img.shape[0])
        elif (img.shape[1], img.shape[0]) != target_size:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)

        if input_mode == "mask":
            bin_m = decode_binary_mask_array(img)
            masks.append(bin_m)
            frames_rgb.append(make_mask_preview_plate(bin_m))
        else:
            if img.ndim == 2:
                rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                rgb = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2RGB)
            else:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames_rgb.append(rgb)

    h, w = frames_rgb[0].shape[:2]
    session.frames_rgb = frames_rgb
    session.width = int(w)
    session.height = int(h)
    session.frame_numbers = list(range(1, len(frames_rgb) + 1))
    session.input_mode = input_mode
    session.reset_downstream(keep_masks=False)

    if input_mode == "mask":
        session.masks = masks
        session.master_info = select_representative_master_frames(
            masks, frames=None, frame_numbers=session.frame_numbers
        )
        session.stage1_backend = "Uploaded Alpha Mask Sequence"
        session.stage1_score = float(session.master_info["master_score"])

    return {
        "session": session.summary(),
        "master_info": session.master_info,
    }


@app.post("/api/load_preset")
def api_load_preset(req: PresetRequest):
    preset = req.preset
    if preset == "sample_sequence":
        rgb_dir = os.path.join(BASE_DIR, "sample_sequences", "rgb")
        frames_rgb = _load_sorted_images(rgb_dir) if os.path.isdir(rgb_dir) else []
        if not frames_rgb:
            _, frames_rgb = create_synthetic_articulated_sequence(num_frames=15, height=360, width=360)

        h, w = frames_rgb[0].shape[:2]
        session.frames_rgb = frames_rgb
        session.width = int(w)
        session.height = int(h)
        session.frame_numbers = list(range(1, len(frames_rgb) + 1))
        session.input_mode = "rgb"
        session.reset_downstream(keep_masks=False)
        return {"session": session.summary()}

    if preset == "user_mask":
        test_mask_glob = sorted(glob.glob(os.path.join(BASE_DIR, "Test_mask", "*.png")))
        candidates = [
            os.path.join(BASE_DIR, "Test_mask", "vlm_initial_mask.png"),
            *test_mask_glob,
            os.path.join(BASE_DIR, "sample_sequences", "masks", "mask_0005.png"),
        ]
        chosen = next((c for c in candidates if os.path.isfile(c)), None)
        if chosen:
            masks = TrackedMaskGenerator.load_mask_sequence([chosen])
        else:
            masks, _ = create_synthetic_articulated_sequence(num_frames=1, height=360, width=360)

        m0 = masks[0]
        h, w = m0.shape[:2]
        session.frames_rgb = [make_mask_preview_plate(m0)]
        session.width = int(w)
        session.height = int(h)
        session.frame_numbers = [1]
        session.input_mode = "mask"
        session.reset_downstream(keep_masks=False)
        session.masks = masks
        session.master_info = select_representative_master_frames(masks, frame_numbers=[1])
        session.stage1_backend = "Preset Binary Mask"
        session.stage1_score = float(session.master_info["master_score"])
        return {"session": session.summary(), "master_info": session.master_info}

    if preset == "local_dir":
        masks = _load_sorted_images(req.mask_dir, as_mask=True) if (req.mask_dir and os.path.isdir(req.mask_dir)) else []
        frames_rgb = _load_sorted_images(req.rgb_dir, as_mask=False) if (req.rgb_dir and os.path.isdir(req.rgb_dir)) else []
        if not frames_rgb and not masks:
            raise HTTPException(status_code=400, detail="No valid RGB or Mask images found in the specified folders.")
        if not frames_rgb:
            frames_rgb = [make_mask_preview_plate(m) for m in masks]

        h, w = frames_rgb[0].shape[:2]
        session.frames_rgb = frames_rgb
        session.width = int(w)
        session.height = int(h)
        session.frame_numbers = list(range(1, len(frames_rgb) + 1))
        session.input_mode = "mask" if masks else "rgb"
        session.reset_downstream(keep_masks=False)
        if masks:
            session.masks = masks
            session.master_info = select_representative_master_frames(
                masks, frames=frames_rgb if req.rgb_dir else None, frame_numbers=session.frame_numbers
            )
            session.stage1_backend = "Loaded Mask Sequence"
            session.stage1_score = float(session.master_info["master_score"])
        return {"session": session.summary(), "master_info": session.master_info}

    raise HTTPException(status_code=400, detail=f"Unknown preset: {preset}")


@app.get("/api/frame/{frame_idx}/{layer}")
def api_get_frame_layer(frame_idx: int, layer: str):
    if not session.frames_rgb:
        raise HTTPException(status_code=404, detail="No sequence loaded.")
    if frame_idx < 0 or frame_idx >= len(session.frames_rgb):
        raise HTTPException(status_code=404, detail=f"Frame index out of range: {frame_idx}")

    h, w = session.height, session.width
    headers = {"Cache-Control": "no-store"}

    if layer == "rgb":
        bgr = cv2.cvtColor(session.frames_rgb[frame_idx], cv2.COLOR_RGB2BGR)
        return Response(content=_encode_png_bytes(bgr), media_type="image/png", headers=headers)

    if layer == "mask":
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        if frame_idx < len(session.masks):
            m = session.masks[frame_idx]
            fg = m > 127
            rgba[fg] = (248, 189, 56, 125)
            edge = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
            rgba[edge] = (255, 225, 110, 235)
        return Response(content=_encode_png_bytes(rgba), media_type="image/png", headers=headers)

    master_idx = session.master_info["master_index"] if session.master_info else 0
    is_master_view = frame_idx == master_idx or len(session.frames_rgb) == 1

    if layer == "stage2_overlap":
        rgba = session.stage2_overlap_rgba if (session.stage2_overlap_rgba is not None and is_master_view) else np.zeros((h, w, 4), dtype=np.uint8)
        return Response(content=_encode_png_bytes(rgba), media_type="image/png", headers=headers)

    if layer == "overlap_seam":
        rgba = session.stage2_seam_only_rgba if (session.stage2_seam_only_rgba is not None and is_master_view) else np.zeros((h, w, 4), dtype=np.uint8)
        return Response(content=_encode_png_bytes(rgba), media_type="image/png", headers=headers)

    raise HTTPException(status_code=400, detail=f"Unknown layer: {layer}")


@app.get("/api/frame/{frame_idx}/part/{part_id}")
def api_get_frame_part_layer(frame_idx: int, part_id: int):
    if not session.frames_rgb:
        raise HTTPException(status_code=404, detail="No sequence loaded.")
    if frame_idx < 0 or frame_idx >= len(session.frames_rgb):
        raise HTTPException(status_code=404, detail=f"Frame index out of range: {frame_idx}")

    h, w = session.height, session.width
    master_idx = session.master_info["master_index"] if session.master_info else 0
    if part_id in session.stage2_part_rgba_map and (frame_idx == master_idx or len(session.frames_rgb) == 1):
        rgba = session.stage2_part_rgba_map[part_id]
    else:
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
    return Response(content=_encode_png_bytes(rgba), media_type="image/png", headers={"Cache-Control": "no-store"})


@app.post("/api/stage1/run")
def api_run_stage1(req: Stage1Request):
    if not session.frames_rgb:
        raise HTTPException(status_code=400, detail="Load an image or sequence first.")

    has_prompt = bool(req.fg_points or req.bg_points or (req.bbox and len(req.bbox) == 4))
    t0 = time.perf_counter()

    if session.input_mode == "mask" and session.masks and not has_prompt:
        masks = session.masks
        backend = session.stage1_backend or "Uploaded Mask Sequence"
        score = 1.0
    else:
        bbox_t = tuple(float(v) for v in req.bbox) if (req.bbox and len(req.bbox) == 4) else None
        masks, backend, score = session.mask_gen.segment_and_track(
            session.frames_rgb,
            init_idx=req.prompt_frame_idx,
            fg_points=[(float(p[0]), float(p[1])) for p in req.fg_points] or None,
            bg_points=[(float(p[0]), float(p[1])) for p in req.bg_points] or None,
            bbox=bbox_t,
            use_sam=req.use_sam,
        )

    session.reset_downstream(keep_masks=False)
    session.masks = masks
    session.stage1_backend = backend
    session.stage1_score = float(score)

    flow_frames = session.frames_rgb if session.input_mode == "rgb" else None
    session.master_info = AlphaToRotoPipeline.select_stage1_masters(
        masks,
        frames_rgb=flow_frames,
        frame_numbers=session.frame_numbers,
        auto_multi_master=bool(req.auto_multi_master),
        flow_motion_threshold_px=float(req.flow_motion_threshold_px),
        max_master_frames=max(1, int(req.max_master_frames)),
        include_flow_profile=True,
    )

    dt_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "session": session.summary(),
        "backend": backend,
        "sam_score": round(float(score), 4),
        "time_ms": round(dt_ms, 2),
        "master_info": session.master_info,
    }


@app.post("/api/stage1/master_frames")
def api_set_master_frames(req: MasterFramesRequest):
    if session.master_info is None or not session.frame_numbers:
        raise HTTPException(status_code=400, detail="Run Stage 1 (or load a mask) first.")

    f_to_idx = {f: i for i, f in enumerate(session.frame_numbers)}
    valid = sorted({int(f) for f in req.master_frames if int(f) in f_to_idx})
    if not valid:
        raise HTTPException(status_code=400, detail="At least one valid representative frame is required.")

    old_primary = int(session.master_info["master_frame"])
    per_m = session.master_info["per_frame_metrics"]
    if req.primary_master_frame is not None and int(req.primary_master_frame) in valid:
        primary_f = int(req.primary_master_frame)
    elif old_primary in valid:
        primary_f = old_primary
    else:
        primary_f = max(valid, key=lambda f: per_m[f_to_idx[f]]["score"])

    primary_idx = f_to_idx[primary_f]
    session.master_info["master_frame"] = primary_f
    session.master_info["master_index"] = primary_idx
    session.master_info["master_frames"] = valid
    session.master_info["master_indices"] = [f_to_idx[f] for f in valid]
    valid_set = set(valid)
    for idx_m, entry in enumerate(per_m):
        entry["is_master"] = session.frame_numbers[idx_m] in valid_set

    if primary_f != old_primary:
        session.reset_downstream(keep_masks=True)
    else:
        session.stage4_metrics = None
        if session.project is not None:
            session.project["master_frames"] = list(valid)

    return {
        "session": session.summary(),
        "master_info": session.master_info,
    }


@app.post("/api/stage2/run")
def api_run_stage2(req: Stage2Request):
    if not session.masks or session.master_info is None:
        raise HTTPException(status_code=400, detail="Run Stage 1 (or load a mask) first.")

    if req.master_frame_idx is not None:
        m_idx = max(0, min(len(session.masks) - 1, int(req.master_frame_idx)))
        session.master_info["master_index"] = m_idx
        session.master_info["master_frame"] = session.frame_numbers[m_idx]

    master_idx = int(session.master_info["master_index"])
    master_frame = int(session.master_info["master_frame"])
    rgb_for_flow = session.frames_rgb if session.input_mode == "rgb" else None

    t0 = time.perf_counter()
    decomp = AlphaToRotoPipeline.run_stage2_decompose(
        session.masks,
        master_idx=master_idx,
        frames_rgb=rgb_for_flow,
        max_parts=req.max_parts,
        overlap_margin_px=req.overlap_margin_px,
        defect_depth_thresh_px=req.defect_depth_thresh_px,
    )
    dt_ms = (time.perf_counter() - t0) * 1000.0

    session.stage2_decomp = decomp
    _rebuild_stage2_visual_layers(session)
    session.stage3_metrics = None
    session.stage4_metrics = None
    session.project = None

    return {
        "session": session.summary(),
        "master_frame": master_frame,
        "master_index": master_idx,
        "num_parts": len(decomp["parts"]),
        "parts": _format_stage2_parts_meta(decomp),
        "cut_lines": decomp["cut_lines"],
        "overlap_margin_px": decomp["overlap_margin_px"],
        "overlap_pixels": session.stage2_overlap_pixels,
        "seam_gap_percent": round(decomp["seam_gap_ratio"] * 100.0, 4),
        "time_ms": round(dt_ms, 2),
    }


@app.post("/api/stage2/update_part")
def api_update_stage2_part(req: PartUpdateRequest):
    if session.stage2_decomp is None or session.master_info is None:
        raise HTTPException(status_code=400, detail="Run Stage 2 (Decompose & Overlap) first.")

    decomp = session.stage2_decomp
    target = next((p for p in decomp["parts"] if int(p["part_id"]) == int(req.part_id)), None)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Part ID {req.part_id} not found.")

    if req.enabled is not None:
        target["enabled"] = bool(req.enabled)
    if req.overlap_margin_px is not None:
        target["overlap_margin_px"] = max(0, int(req.overlap_margin_px))
    if req.boundary_offset_px is not None:
        target["boundary_offset_px"] = max(-10, min(15, int(req.boundary_offset_px)))
    if req.max_points is not None:
        target["max_points"] = max(4, min(36, int(req.max_points)))

    master_mask = session.masks[int(session.master_info["master_index"])]
    rebuilt = rebuild_part_overlap_mask(
        target["core_mask"],
        master_mask,
        overlap_margin_px=target["overlap_margin_px"],
        boundary_offset_px=target["boundary_offset_px"],
    )
    target["mask"] = rebuilt
    target["area_px"] = int(np.count_nonzero(rebuilt > 127))

    _rebuild_stage2_visual_layers(session)
    session.stage3_metrics = None
    session.stage4_metrics = None
    session.project = None

    return {
        "session": session.summary(),
        "parts": _format_stage2_parts_meta(decomp),
        "overlap_pixels": session.stage2_overlap_pixels,
        "seam_gap_percent": round(decomp["seam_gap_ratio"] * 100.0, 4),
    }


@app.post("/api/stage3/run")
def api_run_stage3(req: Stage3Request):
    if session.stage2_decomp is None or session.master_info is None:
        raise HTTPException(status_code=400, detail="Run Stage 2 (Decompose & Overlap) first.")

    t0 = time.perf_counter()
    try:
        roto_project, stage3_metrics = AlphaToRotoPipeline.run_stage3_fit_parts(
            session.stage2_decomp,
            masks=session.masks,
            master_info=session.master_info,
            frame_numbers=session.frame_numbers,
            frames_rgb=(session.frames_rgb if session.input_mode == "rgb" else None),
            min_points=int(req.min_points),
            max_points=int(req.max_points),
            default_tension=float(req.default_tension),
            cusp_angle_deg=float(req.cusp_angle_deg),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    dt_ms = (time.perf_counter() - t0) * 1000.0
    stage3_metrics["Time_ms"] = round(dt_ms, 2)

    session.stage3_params = {
        "min_points": int(req.min_points),
        "max_points": int(req.max_points),
        "default_tension": float(req.default_tension),
        "cusp_angle_deg": float(req.cusp_angle_deg),
    }
    session.pipeline = AlphaToRotoPipeline(
        overlap_margin_px=int(session.stage2_decomp["overlap_margin_px"]),
        max_parts=len(roto_project["parts"]),
        min_points_per_part=int(req.min_points),
        max_points_per_part=int(req.max_points),
        default_tension=float(req.default_tension),
    )
    session.project = roto_project
    session.stage3_metrics = stage3_metrics
    session.stage4_metrics = None

    return {
        "session": session.summary(),
        "metrics": session.stage3_metrics,
        "project": AlphaToRotoPipeline.serialize_project(session.project),
    }


@app.post("/api/stage4/run")
def api_run_stage4(req: Stage4Request):
    if session.project is None or session.stage2_decomp is None or session.master_info is None:
        raise HTTPException(status_code=400, detail="Run Stage 3 (Generate Bezier) first.")

    t0 = time.perf_counter()
    master_frame = int(session.project["master_frame"])
    master_frames = sorted(
        {
            int(f)
            for f in session.master_info.get("master_frames", [master_frame])
            if int(f) in session.frame_numbers
        }
        | {master_frame}
    )

    default_tension = float(session.stage3_params.get("default_tension", 0.5))
    cusp_angle_deg = float(session.stage3_params.get("cusp_angle_deg", 60.0))

    pipe = AlphaToRotoPipeline(
        overlap_margin_px=int(session.project["overlap_margin_px"]),
        max_parts=max(1, len(session.project["parts"])),
        min_points_per_part=int(session.stage3_params.get("min_points", 8)),
        max_points_per_part=int(session.stage3_params.get("max_points", 16)),
        default_tension=default_tension,
        blending_mode=req.blending_mode,
        use_tangent_rotation=bool(req.use_tangent_rotation),
        use_contour_snapping=bool(req.use_contour_snapping),
        auto_multi_master=len(master_frames) >= 2,
    )

    frames_rgb_for_track = session.frames_rgb if (session.input_mode == "rgb" and len(session.frames_rgb) > 1) else None
    pipe.run_stage4_propagate(
        session.project,
        masks=session.masks,
        frames_rgb=frames_rgb_for_track,
        master_frames=master_frames,
        blending_mode=req.blending_mode,
        use_tangent_rotation=bool(req.use_tangent_rotation),
        use_contour_snapping=bool(req.use_contour_snapping),
        default_tension=default_tension,
        cusp_angle_deg=max(60.0, cusp_angle_deg),
    )

    session.pipeline = pipe
    seq_metrics = pipe.evaluate_sequence_metrics(session.project, session.masks)
    dt_ms = (time.perf_counter() - t0) * 1000.0
    session.stage4_metrics = {
        "linked_master_frames": master_frames,
        "J": round(seq_metrics["J"], 2),
        "F": round(seq_metrics["F"], 2),
        "J_and_F": round(seq_metrics["J_and_F"], 2),
        "RMSE_px": round(seq_metrics["RMSE_px"], 2),
        "T_flicker": round(seq_metrics["T_flicker"], 2),
        "Gap_percent": round(seq_metrics["Gap_percent"], 3),
        "Time_ms": round(dt_ms, 2),
    }

    return {
        "session": session.summary(),
        "linked_master_frames": master_frames,
        "sequence_metrics": session.stage4_metrics,
        "project": AlphaToRotoPipeline.serialize_project(session.project),
    }


@app.post("/api/stage4/edit")
def api_edit_stage4(req: Stage4EditRequest):
    if session.project is None or session.pipeline is None:
        raise HTTPException(status_code=400, detail="Run Stage 3 or Stage 4 first.")

    target_part = next((p for p in session.project["parts"] if p["part_id"] == req.part_id), None)
    if target_part is None:
        raise HTTPException(status_code=404, detail=f"Part ID {req.part_id} not found.")
    if not target_part.get("tracks_per_ref"):
        master_f = int(session.project["master_frame"])
        target_part["tracks_per_ref"] = {
            master_f: {
                cp["index"]: {
                    fn: (float(cp["anchor"][0]), float(cp["anchor"][1]), 1.0)
                    for fn in session.project["frame_numbers"]
                }
                for cp in target_part["reference_keyframes"][master_f]
            }
        }

    session.pipeline.apply_artist_keyframe_edit(
        session.project,
        part_id=req.part_id,
        edited_frame=req.edited_frame,
        edited_control_points=req.edited_control_points,
    )
    return {
        "session": session.summary(),
        "project": AlphaToRotoPipeline.serialize_project(session.project),
    }


@app.post("/api/export/nuke")
def api_export_nuke(req: NukeExportRequest):
    if session.project is None:
        raise HTTPException(status_code=400, detail="Run Stage 3 or Stage 4 first before exporting to Nuke.")

    os.makedirs(WEB_OUTPUT_DIR, exist_ok=True)
    script_name = "build_alphatoroto_node.py"
    json_name = "alphatoroto_project.json"
    master_f = int(session.project["master_frame"])
    svg_name = f"frame_{master_f:04d}.svg"

    script_path = os.path.join(WEB_OUTPUT_DIR, script_name)
    json_path = os.path.join(WEB_OUTPUT_DIR, json_name)
    svg_path = os.path.join(WEB_OUTPUT_DIR, svg_name)

    AlphaToRotoPipeline.export_to_json(session.project, json_path)
    AlphaToRotoPipeline.export_frame_to_svg(session.project, master_f, svg_path)
    AlphaToRotoPipeline.export_to_nuke_script_py(
        session.project,
        script_path,
        key_step=max(1, int(req.key_step)),
        attach_autoroto_tab=bool(req.attach_autoroto_tab),
    )

    fwd_script = script_path.replace("\\", "/")
    fwd_json = json_path.replace("\\", "/")
    fwd_repo = BASE_DIR.replace("\\", "/")

    result = {
        "script_path": script_path,
        "json_path": json_path,
        "svg_path": svg_path,
        "nuke_exec_snippet": f'exec(open(r"{fwd_script}", encoding="utf-8").read())',
        "nuke_json_snippet": (
            f'import sys; sys.path.insert(0, r"{fwd_repo}"); '
            f'import nuke_bridge; nuke_bridge.load_alphatoroto_json_as_node(r"{fwd_json}")'
        ),
        "download_urls": {
            "nuke_py": f"/api/download/{script_name}",
            "json": f"/api/download/{json_name}",
            "svg": f"/api/download/{svg_name}",
        },
    }
    session.last_export = result
    return result


@app.get("/api/download/{filename}")
def api_download(filename: str):
    safe_name = os.path.basename(filename)
    if safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    target = os.path.join(WEB_OUTPUT_DIR, safe_name)
    if not os.path.isfile(target):
        raise HTTPException(status_code=404, detail=f"File not found: {safe_name}")
    return FileResponse(target, filename=safe_name)


def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Launch the AlphaToRoto Interactive Full-Pipeline Web Studio.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    print(f"[AlphaToRoto Web Studio] Starting at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
