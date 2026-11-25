# EditableAlpha — Script & Package Architecture

## Directory Layout

```text
e:\AlphaToRoto\
├── EditableAlpha.gizmo              # Native Foundry Nuke Gizmo (mask Input -> Roto -> ViewSwitch -> Output)
├── init.py                          # Nuke pluginAddPath & sys.path registration
├── menu.py                          # Nodes toolbar + top menu + Ctrl+Alt+E shortcut
├── editablealpha_bridge.py          # Pure-stdlib Nuke bridge (zero numpy/cv2 imports)
├── editablealpha_worker.py          # External Python subprocess CLI worker (cv2 + numpy)
├── autoroto_pipeline.py             # Multi-stage Mask-to-Bezier & C1 multi-reference tracking engine
├── roto_geometry.py                 # Kinematic-geometric part decomposition & O(N) Hermite-Bezier fitting
├── mask_generator.py                # Binary mask decoding & representative master frame selection
├── tests/
│   └── test_editablealpha_gizmo.py  # Headless unit & subprocess E2E tests
├── README.md                        # User installation & workflow guide
└── SCRIPT_STRUCTURE.md              # Architecture reference
```

## Runtime Split (Decoupled Dual-Runtime)

1. **Inside Foundry Nuke (`editablealpha_bridge.py`)**:
   - Uses only Python standard library + `nuke` / `nuke.rotopaint`.
   - Renders connected `mask` input frames into `preferences["DiskCachePath"]` (`EditableAlpha_Temp`).
   - Invokes `editablealpha_worker.py` via `subprocess.Popen` with `PROGRESS:<pct>:<msg>` streaming into `nuke.ProgressTask`.
   - Injects resulting cubic Bezier shapes into internal `EditableAlpha_Roto` and/or extracted standalone `Roto` node (`y_nuke = H - y_img`, negated tangent Y).
   - Reads artist-edited Bezier control points back from Nuke (`extract_roto_keyframes_from_node`) for `retrack`.

2. **External Worker (`editablealpha_worker.py`)**:
   - Imports `cv2`, `numpy`, `mask_generator`, `roto_geometry`, and `autoroto_pipeline`.
   - Executes `convert` (Stage 1–4) and `retrack` (Stage 4 preserving `manual_keyframes`) modes and returns JSON.
