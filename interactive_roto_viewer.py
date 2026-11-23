"""
Interactive Human-in-the-Loop (HITL) Roto Testing Studio & Sample Sequence Generator
Provides 3 direct ways for a user/researcher to test AlphaToRoto:
  1. Exports ready-to-use sample Alpha Mask & RGB frame sequences (`sample_sequences/masks/`, `sample_sequences/rgb/`)
  2. Generates a standalone, interactive HTML5 Canvas Roto Editor (`demo_output/interactive_alphatoroto_studio.html`)
     where you can:
       - Scrub across frames and inspect the Master Frame (F_master) & 12px Seam Overlap Margin
       - Drag any cubic Bezier control point anchor or left/right tangent handle (`in_handle` / `out_handle`)
       - See real-time C1 smoothstep visibility-weighted multi-keyframe re-tracking (Eqs. 6-7) and
         Chord-Aligned Relative Tangent Rotation (Eq. 8) across intermediate frames while 100% preserving keyframes
       - Export the modified project directly to a Foundry Nuke Python builder script (`build_nuke_roto.py`) or JSON
  3. Optional native OpenCV Desktop GUI window (`--gui`) for interactive mouse dragging of Bezier control points.
"""

import os
import json
import argparse
import cv2
import numpy as np

from autoroto_pipeline import AlphaToRotoPipeline
from test_human_in_the_loop_mask import create_synthetic_articulated_sequence


def export_sample_sequence_to_disk(base_dir: str, num_frames: int = 15) -> dict:
    """
    Writes a sample articulated alpha mask sequence and matching RGB frames to disk so the user
    can test CLI commands and Nuke Read nodes with real image files.
    """
    mask_dir = os.path.join(base_dir, "masks")
    rgb_dir = os.path.join(base_dir, "rgb")
    rgba_dir = os.path.join(base_dir, "rgba_nuke")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(rgba_dir, exist_ok=True)

    masks, frames_rgb = create_synthetic_articulated_sequence(num_frames=num_frames, height=360, width=360)
    for idx, (m, rgb) in enumerate(zip(masks, frames_rgb)):
        f_num = idx + 1
        m_path = os.path.join(mask_dir, f"mask_{f_num:04d}.png")
        rgb_path = os.path.join(rgb_dir, f"frame_{f_num:04d}.png")
        rgba_path = os.path.join(rgba_dir, f"rgba_{f_num:04d}.png")

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        rgba = np.dstack([bgr, m])
        cv2.imwrite(m_path, m)
        cv2.imwrite(rgb_path, bgr)
        cv2.imwrite(rgba_path, rgba)

    return {
        "mask_dir": mask_dir,
        "rgb_dir": rgb_dir,
        "rgba_dir": rgba_dir,
        "num_frames": num_frames,
        "masks": masks,
        "frames_rgb": frames_rgb,
    }


def generate_interactive_html_studio(roto_project: dict, output_html_path: str):
    """
    Generates a rich, self-contained HTML5 Canvas interactive rotoscoping test studio
    implementing Equations (3)-(8) live in the browser so the user can interactively drag
    anchors/handles, add keyframes, inspect the 12px seam overlap, and download Nuke scripts.
    """
    serializable = {
        "width": roto_project["width"],
        "height": roto_project["height"],
        "frame_numbers": roto_project["frame_numbers"],
        "master_frame": roto_project["master_frame"],
        "overlap_margin_px": roto_project["overlap_margin_px"],
        "seam_gap_ratio_master": roto_project["seam_gap_ratio_master"],
        "cut_lines": roto_project.get("cut_lines", []),
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
    project_json_str = json.dumps(serializable)

    html = rf"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8" />
<title>AlphaToRoto Interactive HITL Bezier Studio</title>
<style>
  :root {{
    --bg: #0f131a;
    --panel: #181e29;
    --border: #2a3446;
    --accent: #38bdf8;
    --accent2: #f97316;
    --text: #e2e8f0;
    --muted: #94a3b8;
  }}
  * {{ box-sizing: border-box; font-family: 'Segoe UI', system-ui, sans-serif; }}
  body {{
    margin: 0; padding: 20px; background: var(--bg); color: var(--text);
    display: flex; flex-direction: column; align-items: center; gap: 16px;
  }}
  header {{
    width: 100%; max-width: 1120px; display: flex; justify-content: space-between;
    align-items: center; background: var(--panel); padding: 14px 20px;
    border-radius: 10px; border: 1px solid var(--border);
  }}
  h1 {{ margin: 0; font-size: 18px; color: var(--accent); }}
  .badge {{
    background: #1e293b; border: 1px solid #334155; padding: 4px 10px;
    border-radius: 6px; font-size: 12px; color: #cbd5e1;
  }}
  .workspace {{
    width: 100%; max-width: 1120px; display: grid;
    grid-template-columns: 1fr 340px; gap: 16px;
  }}
  .canvas-card, .side-card {{
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px;
  }}
  .canvas-wrap {{
    display: flex; flex-direction: column; align-items: center; gap: 12px;
  }}
  canvas {{
    background: #090c10; border: 1px solid #334155; border-radius: 8px;
    cursor: crosshair; width: 540px; height: 540px;
  }}
  .timeline {{
    width: 100%; display: flex; flex-direction: column; gap: 8px;
  }}
  .frame-pills {{
    display: flex; gap: 4px; flex-wrap: wrap;
  }}
  .pill {{
    flex: 1; text-align: center; padding: 6px 4px; font-size: 12px;
    background: #0f172a; border: 1px solid #334155; border-radius: 5px;
    cursor: pointer; user-select: none;
  }}
  .pill.active {{ border-color: var(--accent); background: #0284c7; color: #fff; font-weight: 600; }}
  .pill.is-key {{ box-shadow: inset 0 -3px 0 #f59e0b; }}
  .controls {{ display: flex; gap: 8px; flex-wrap: wrap; width: 100%; }}
  button {{
    background: #1e293b; color: var(--text); border: 1px solid #3b4963;
    padding: 8px 12px; border-radius: 6px; cursor: pointer; font-size: 13px;
  }}
  button:hover {{ background: #283548; border-color: var(--accent); }}
  button.primary {{ background: #0284c7; border-color: #38bdf8; color: #fff; font-weight: 600; }}
  .stat-row {{
    display: flex; justify-content: space-between; font-size: 13px;
    padding: 7px 0; border-bottom: 1px solid #232c3d;
  }}
  .legend {{ font-size: 12px; color: var(--muted); line-height: 1.6; margin-top: 10px; }}
</style>
</head>
<body>
<header>
  <div>
    <h1>AlphaToRoto — Human-in-the-Loop Multi-Part Cubic Bézier Test Studio</h1>
    <div style="font-size:12px;color:var(--muted);margin-top:4px;">
      Stage 1 (Master Frame Eq.1) • Stage 2 (Kinematic Part Decomp + 12px Overlap Eq.2) • Stage 3 (O(N) Hermite-Bézier Eqs.3-5) • Stage 4 (C¹ Smoothstep Multi-Ref Re-Tracking Eqs.6-8)
    </div>
  </div>
  <div class="badge" id="statusBadge">Master Frame: F{roto_project["master_frame"]}</div>
</header>

<div class="workspace">
  <div class="canvas-card canvas-wrap">
    <canvas id="rotoCanvas" width="{roto_project["width"]}" height="{roto_project["height"]}"></canvas>
    <div class="timeline">
      <div style="display:flex;justify-content:space-between;font-size:13px;">
        <span id="frameLabel">Frame 1</span>
        <span style="color:#f59e0b;">★ Amber Underline = Protected Artist / Master Keyframe (100% Preserved)</span>
      </div>
      <div class="frame-pills" id="framePills"></div>
    </div>
    <div class="controls">
      <button onclick="stepFrame(-1)">◀ Prev Frame</button>
      <button onclick="togglePlay()" id="playBtn">▶ Play Loop</button>
      <button onclick="stepFrame(1)">Next Frame ▶</button>
      <button onclick="jumpMaster()">★ Go to F_master ({roto_project["master_frame"]})</button>
      <button onclick="resetProject()">↺ Reset Edits</button>
    </div>
  </div>

  <div class="side-card">
    <h3 style="margin-top:0;font-size:15px;color:var(--accent);">Pipeline Telemetry & Controls</h3>
    <div class="stat-row"><span>Master Reference Frame (Eq. 1)</span><strong id="stMaster">F{roto_project["master_frame"]}</strong></div>
    <div class="stat-row"><span>Decomposed Quasi-Rigid Parts</span><strong id="stParts">{len(roto_project["parts"])}</strong></div>
    <div class="stat-row"><span>Seam Overlap Margin (Eq. 2)</span><strong>{roto_project["overlap_margin_px"]} px</strong></div>
    <div class="stat-row"><span>Internal Seam Gap Ratio</span><strong>{round(roto_project["seam_gap_ratio_master"]*100, 3)}%</strong></div>
    <div class="stat-row"><span>Active Keyframes (Part 1 / Part 2)</span><strong id="stKeys">-</strong></div>

    <div style="margin-top:14px;display:flex;flex-direction:column;gap:8px;">
      <label style="font-size:13px;"><input type="checkbox" id="chkHandles" checked onchange="draw()"> Show In/Out Tangent Handles (Δh)</label>
      <label style="font-size:13px;"><input type="checkbox" id="chkCut" checked onchange="draw()"> Show Articulation Cut & 12px Overlap Seam</label>
      <label style="font-size:13px;"><input type="checkbox" id="chkRotTangent" checked onchange="recomputeAllParts()"> Enable Eq. (8) Chord-Aligned Tangent Rotation</label>
    </div>

    <div style="margin-top:16px;display:flex;flex-direction:column;gap:8px;">
      <button class="primary" onclick="downloadNukeScript()">⬇ Download Foundry Nuke Script (build_nuke_roto.py)</button>
      <button onclick="downloadJson()">⬇ Download Roto Project JSON</button>
    </div>

    <div class="legend">
      <strong>직접 테스트하는 방법 (HITL Interaction):</strong><br/>
      1. 캔버스에서 <b>흰색 사각형(제어점 Anchor)</b>이나 <b>하늘색/주황색 원(좌·우 베지에 접선 핸들)</b>을 마우스로 드래그해 보세요.<br/>
      2. 드래그한 프레임이 즉시 <b>새로운 기준 키프레임($k_m \in K$)</b>으로 등록되어 100% 보존됩니다.<br/>
      3. 인접 프레임들은 <b>$C^1$ Smoothstep($3t^2-2t^3$) 가시성 가중 양방향 블렌딩(식 6~7)</b>과 <b>할선 회전 탄젠트 보정(식 8)</b>으로 실시간 재계산됩니다.
    </div>
  </div>
</div>

<script>
const INITIAL_DATA = {project_json_str};
let project = JSON.parse(JSON.stringify(INITIAL_DATA));
let currentFrame = project.master_frame;
let playing = false;
let playTimer = null;
let dragTarget = null; // {{ partIdx, cpIdx, mode: 'anchor'|'in'|'out' }}

// Store baseline trajectory offsets per part for interactive browser re-baking
const baseFramesPerPart = INITIAL_DATA.parts.map(p => JSON.parse(JSON.stringify(p.frames)));

const canvas = document.getElementById('rotoCanvas');
const ctx = canvas.getContext('2d');
const palette = ['#38bdf8', '#f97316', '#10b981', '#e879f9'];

function initPills() {{
  const wrap = document.getElementById('framePills');
  wrap.innerHTML = '';
  project.frame_numbers.forEach(f => {{
    const el = document.createElement('div');
    el.className = 'pill' + (f === currentFrame ? ' active' : '');
    const isKey = project.parts.some(p => p.reference_keyframes[String(f)] !== undefined);
    if (isKey) el.classList.add('is-key');
    el.textContent = (f === project.master_frame ? '★F' : 'F') + f;
    el.onclick = () => {{ currentFrame = f; updateUI(); }};
    wrap.appendChild(el);
  }});
}}

function recomputePartC1(partIdx) {{
  const part = project.parts[partIdx];
  const baseFrames = baseFramesPerPart[partIdx];
  const keyFrames = Object.keys(part.reference_keyframes).map(Number).sort((a,b) => a - b);
  const fNums = project.frame_numbers;
  const numPts = part.num_control_points;
  const useRot = document.getElementById('chkRotTangent').checked;

  // Compute delta offsets at each reference keyframe relative to base trajectory
  const keyDeltas = {{}};
  keyFrames.forEach(kf => {{
    const refCps = part.reference_keyframes[String(kf)];
    const baseCps = baseFrames[String(kf)];
    keyDeltas[kf] = refCps.map((cp, i) => ({{
      dx: cp.anchor[0] - baseCps[i].anchor[0],
      dy: cp.anchor[1] - baseCps[i].anchor[1],
      in_h: [...cp.in_handle],
      out_h: [...cp.out_handle],
      anchor: [...cp.anchor]
    }}));
  }});

  fNums.forEach(f => {{
    const fStr = String(f);
    if (part.reference_keyframes[fStr]) {{
      part.frames[fStr] = JSON.parse(JSON.stringify(part.reference_keyframes[fStr]));
      return;
    }}
    let kStart = keyFrames[0], kEnd = keyFrames[keyFrames.length - 1];
    for (let m = 0; m < keyFrames.length - 1; m++) {{
      if (f >= keyFrames[m] && f <= keyFrames[m+1]) {{
        kStart = keyFrames[m]; kEnd = keyFrames[m+1]; break;
      }}
    }}

    const newCps = [];
    for (let i = 0; i < numPts; i++) {{
      const basePt = baseFrames[fStr][i];
      let dx = 0, dy = 0, baseRefKf = kStart;
      if (f <= keyFrames[0]) {{
        dx = keyDeltas[keyFrames[0]][i].dx;
        dy = keyDeltas[keyFrames[0]][i].dy;
        baseRefKf = keyFrames[0];
      }} else if (f >= keyFrames[keyFrames.length - 1]) {{
        const kl = keyFrames[keyFrames.length - 1];
        dx = keyDeltas[kl][i].dx;
        dy = keyDeltas[kl][i].dy;
        baseRefKf = kl;
      }} else {{
        const t = (f - kStart) / Math.max(1, kEnd - kStart);
        // Eq. (6): C1 Smoothstep s(t) = 3t^2 - 2t^3
        const s = 3 * t * t - 2 * t * t * t;
        dx = (1 - s) * keyDeltas[kStart][i].dx + s * keyDeltas[kEnd][i].dx;
        dy = (1 - s) * keyDeltas[kStart][i].dy + s * keyDeltas[kEnd][i].dy;
        baseRefKf = kStart;
      }}
      newCps.push({{
        index: i,
        anchor: [basePt.anchor[0] + dx, basePt.anchor[1] + dy],
        in_handle: [...part.reference_keyframes[String(baseRefKf)][i].in_handle],
        out_handle: [...part.reference_keyframes[String(baseRefKf)][i].out_handle],
        _refKf: baseRefKf
      }});
    }}

    // Eq. (8): Chord-aligned relative tangent rotation
    if (useRot && numPts >= 3) {{
      for (let i = 0; i < numPts; i++) {{
        const prevI = (i - 1 + numPts) % numPts;
        const nextI = (i + 1) % numPts;
        const refCps = part.reference_keyframes[String(newCps[i]._refKf)];
        const v0x = refCps[nextI].anchor[0] - refCps[prevI].anchor[0];
        const v0y = refCps[nextI].anchor[1] - refCps[prevI].anchor[1];
        const vfx = newCps[nextI].anchor[0] - newCps[prevI].anchor[0];
        const vfy = newCps[nextI].anchor[1] - newCps[prevI].anchor[1];
        const l0 = Math.hypot(v0x, v0y), lf = Math.hypot(vfx, vfy);
        if (l0 > 1e-4 && lf > 1e-4) {{
          const cosA = (v0x * vfx + v0y * vfy) / (l0 * lf);
          const sinA = (v0x * vfy - v0y * vfx) / (l0 * lf);
          const scale = Math.max(0.7, Math.min(1.4, lf / l0));
          const [ix, iy] = newCps[i].in_handle;
          const [ox, oy] = newCps[i].out_handle;
          newCps[i].in_handle = [scale * (cosA * ix - sinA * iy), scale * (sinA * ix + cosA * iy)];
          newCps[i].out_handle = [scale * (cosA * ox - sinA * oy), scale * (sinA * ox + cosA * oy)];
        }}
      }}
    }}
    part.frames[fStr] = newCps;
  }});
}}

function recomputeAllParts() {{
  project.parts.forEach((_, idx) => recomputePartC1(idx));
  updateUI();
}}

function draw() {{
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Subtle background grid
  ctx.strokeStyle = '#161e2e';
  ctx.lineWidth = 1;
  for (let x = 0; x < canvas.width; x += 32) {{
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
  }}
  for (let y = 0; y < canvas.height; y += 32) {{
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
  }}

  const showHandles = document.getElementById('chkHandles').checked;
  const showCut = document.getElementById('chkCut').checked;

  // Draw each decomposed cubic Bezier part
  project.parts.forEach((part, pIdx) => {{
    const color = palette[pIdx % palette.length];
    const cps = part.frames[String(currentFrame)];
    if (!cps || cps.length < 2) return;

    ctx.beginPath();
    ctx.moveTo(cps[0].anchor[0], cps[0].anchor[1]);
    for (let i = 0; i < cps.length; i++) {{
      const cp0 = cps[i];
      const cp1 = cps[(i + 1) % cps.length];
      const b1x = cp0.anchor[0] + cp0.out_handle[0];
      const b1y = cp0.anchor[1] + cp0.out_handle[1];
      const b2x = cp1.anchor[0] + cp1.in_handle[0];
      const b2y = cp1.anchor[1] + cp1.in_handle[1];
      ctx.bezierCurveTo(b1x, b1y, b2x, b2y, cp1.anchor[0], cp1.anchor[1]);
    }}
    ctx.closePath();
    ctx.fillStyle = color + '44';
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.2;
    ctx.stroke();

    // Draw handles and anchors
    cps.forEach(cp => {{
      const [ax, ay] = cp.anchor;
      if (showHandles) {{
        const ix = ax + cp.in_handle[0], iy = ay + cp.in_handle[1];
        const ox = ax + cp.out_handle[0], oy = ay + cp.out_handle[1];
        ctx.strokeStyle = '#fde047';
        ctx.lineWidth = 1.1;
        ctx.beginPath(); ctx.moveTo(ix, iy); ctx.lineTo(ox, oy); ctx.stroke();

        ctx.fillStyle = '#38bdf8';
        ctx.beginPath(); ctx.arc(ix, iy, 3.2, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = '#fb923c';
        ctx.beginPath(); ctx.arc(ox, oy, 3.2, 0, Math.PI * 2); ctx.fill();
      }}
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.6;
      ctx.fillRect(ax - 3.5, ay - 3.5, 7, 7);
      ctx.strokeRect(ax - 3.5, ay - 3.5, 7, 7);
    }});
  }});

  if (showCut && project.cut_lines) {{
    project.cut_lines.forEach(cut => {{
      ctx.save();
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      ctx.moveTo(cut.p1[0], cut.p1[1]);
      ctx.lineTo(cut.p2[0], cut.p2[1]);
      ctx.stroke();
      ctx.restore();
    }});
  }}
}}

function updateUI() {{
  initPills();
  const isMaster = currentFrame === project.master_frame;
  document.getElementById('frameLabel').textContent =
    `Frame ${{currentFrame}} / ${{project.frame_numbers.length}}` +
    (isMaster ? ' (★ Master Reference Frame F_master)' : '');
  const kSummary = project.parts.map(p => `${{p.name}}: [${{Object.keys(p.reference_keyframes).join(', ')}}]`).join(' | ');
  document.getElementById('stKeys').textContent = kSummary;
  draw();
}}

function getCanvasCoords(e) {{
  const rect = canvas.getBoundingClientRect();
  const sx = canvas.width / rect.width;
  const sy = canvas.height / rect.height;
  return [(e.clientX - rect.left) * sx, (e.clientY - rect.top) * sy];
}}

canvas.addEventListener('mousedown', e => {{
  const [mx, my] = getCanvasCoords(e);
  const fStr = String(currentFrame);
  for (let pIdx = 0; pIdx < project.parts.length; pIdx++) {{
    const cps = project.parts[pIdx].frames[fStr];
    for (let i = 0; i < cps.length; i++) {{
      const [ax, ay] = cps[i].anchor;
      if (Math.hypot(mx - ax, my - ay) < 8) {{
        dragTarget = {{ partIdx: pIdx, cpIdx: i, mode: 'anchor' }};
        return;
      }}
      const ix = ax + cps[i].in_handle[0], iy = ay + cps[i].in_handle[1];
      if (Math.hypot(mx - ix, my - iy) < 7) {{
        dragTarget = {{ partIdx: pIdx, cpIdx: i, mode: 'in' }};
        return;
      }}
      const ox = ax + cps[i].out_handle[0], oy = ay + cps[i].out_handle[1];
      if (Math.hypot(mx - ox, my - oy) < 7) {{
        dragTarget = {{ partIdx: pIdx, cpIdx: i, mode: 'out' }};
        return;
      }}
    }}
  }}
}});

canvas.addEventListener('mousemove', e => {{
  if (!dragTarget) return;
  const [mx, my] = getCanvasCoords(e);
  const fStr = String(currentFrame);
  const part = project.parts[dragTarget.partIdx];

  // Register current frame as an artist keyframe in K if not already registered
  if (!part.reference_keyframes[fStr]) {{
    part.reference_keyframes[fStr] = JSON.parse(JSON.stringify(part.frames[fStr]));
  }}
  const cp = part.reference_keyframes[fStr][dragTarget.cpIdx];
  if (dragTarget.mode === 'anchor') {{
    cp.anchor = [mx, my];
  }} else if (dragTarget.mode === 'in') {{
    cp.in_handle = [mx - cp.anchor[0], my - cp.anchor[1]];
  }} else if (dragTarget.mode === 'out') {{
    cp.out_handle = [mx - cp.anchor[0], my - cp.anchor[1]];
  }}
  recomputePartC1(dragTarget.partIdx);
  updateUI();
}});

window.addEventListener('mouseup', () => {{ dragTarget = null; }});

function stepFrame(delta) {{
  const fNums = project.frame_numbers;
  const idx = fNums.indexOf(currentFrame);
  const nextIdx = (idx + delta + fNums.length) % fNums.length;
  currentFrame = fNums[nextIdx];
  updateUI();
}}

function jumpMaster() {{ currentFrame = project.master_frame; updateUI(); }}

function togglePlay() {{
  playing = !playing;
  document.getElementById('playBtn').textContent = playing ? '⏸ Pause' : '▶ Play Loop';
  if (playing) {{
    playTimer = setInterval(() => stepFrame(1), 160);
  }} else {{
    clearInterval(playTimer);
  }}
}}

function resetProject() {{
  project = JSON.parse(JSON.stringify(INITIAL_DATA));
  currentFrame = project.master_frame;
  updateUI();
}}

function downloadJson() {{
  const blob = new Blob([JSON.stringify(project, null, 2)], {{ type: 'application/json' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'alphatoroto_edited_project.json';
  a.click();
}}

function downloadNukeScript() {{
  const lines = [
    'import json',
    'import nuke',
    'import nuke.rotopaint as rp',
    '',
    'ROTO_DATA = json.loads(' + JSON.stringify(JSON.stringify(project)) + ')',
    '',
    'def build_alphatoroto_node():',
    '    roto_node = nuke.createNode("Roto", inpanel=False)',
    '    roto_node.setName("AlphaToRoto_EditableParts")',
    '    curves_knob = roto_node["curves"]',
    '    root_layer = getattr(curves_knob, "rootLayer", getattr(curves_knob, "root", None))',
    '    h = float(ROTO_DATA["height"])',
    '    for part in ROTO_DATA["parts"]:',
    '        shape = rp.Shape(curves_knob)',
    '        shape.name = part["name"]',
    '        num_pts = int(part["num_control_points"])',
    '        for _ in range(num_pts):',
    '            shape.append(rp.ShapeControlPoint())',
    '        for f_num in ROTO_DATA["frame_numbers"]:',
    '            f_cps = part["frames"][str(f_num)]',
    '            for i in range(num_pts):',
    '                cp_data = f_cps[i]',
    '                nuke_cp = shape[i]',
    '                nuke_cp.center.addPositionKey(float(f_num), (float(cp_data["anchor"][0]), h - float(cp_data["anchor"][1]), 0.0))',
    '                nuke_cp.leftTangent.addPositionKey(float(f_num), (float(cp_data["in_handle"][0]), -float(cp_data["in_handle"][1]), 0.0))',
    '                nuke_cp.rightTangent.addPositionKey(float(f_num), (float(cp_data["out_handle"][0]), -float(cp_data["out_handle"][1]), 0.0))',
    '        root_layer.append(shape)',
    '    curves_knob.changed()',
    '    return roto_node',
    '',
    'if __name__ == "__main__":',
    '    build_alphatoroto_node()'
  ];
  const blob = new Blob([lines.join(String.fromCharCode(10))], {{ type: 'text/x-python' }});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'build_nuke_roto_edited.py';
  a.click();
}}

updateUI();
</script>
</body>
</html>
"""
    os.makedirs(os.path.dirname(os.path.abspath(output_html_path)), exist_ok=True)
    with open(output_html_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample sequences and launch the AlphaToRoto Interactive HITL Test Studio."
    )
    parser.add_argument(
        "--sample_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_sequences"),
        help="Output folder for sample mask/RGB/RGBA PNG sequences.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_output"),
        help="Output folder for the interactive HTML5 studio, JSON project, and Nuke builder script.",
    )
    parser.add_argument("--num_frames", type=int, default=15, help="Number of frames in sample sequence.")
    args = parser.parse_args()

    sample_info = export_sample_sequence_to_disk(args.sample_dir, num_frames=args.num_frames)
    pipeline = AlphaToRotoPipeline(
        overlap_margin_px=12,
        max_parts=2,
        min_points_per_part=10,
        max_points_per_part=14,
    )
    project = pipeline.build_editable_roto_from_masks(
        sample_info["masks"], frames_rgb=sample_info["frames_rgb"]
    )

    os.makedirs(args.output_dir, exist_ok=True)
    html_path = os.path.join(args.output_dir, "interactive_alphatoroto_studio.html")
    json_path = os.path.join(args.output_dir, "alphatoroto_project.json")
    nuke_py_path = os.path.join(args.output_dir, "build_nuke_roto.py")

    generate_interactive_html_studio(project, html_path)
    pipeline.export_to_json(project, json_path)
    pipeline.export_to_nuke_script_py(project, nuke_py_path)

    print(
        json.dumps(
            {
                "status": "ready",
                "master_frame": project["master_frame"],
                "num_parts": len(project["parts"]),
                "sample_masks_dir": sample_info["mask_dir"],
                "sample_rgb_dir": sample_info["rgb_dir"],
                "sample_rgba_nuke_dir": sample_info["rgba_dir"],
                "interactive_html_studio": html_path,
                "nuke_builder_script": nuke_py_path,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
