# EditableAlpha — Foundry Nuke Mask-to-Roto Gizmo

**EditableAlpha**는 생성된 래스터 알파 마스크(매트/RGBA 시퀀스)를 입력으로 연결하면 **다중 파트 3차 베지에(Cubic Bézier) Roto 셰이프**로 자동 변환해 주는 **Foundry Nuke 17.1+ 전용 Gizmo 툴**입니다.

Nuke 내장 Python(NumPy/OpenCV 미설치 환경)과의 충돌을 피하기 위해, Nuke 내부에서는 순수 표준 라이브러리 브리지([editablealpha_bridge.py](file:///e:/AlphaToRoto/editablealpha_bridge.py))만 동작하고 무거운 기하·추적 연산은 외부 Python 워커([editablealpha_worker.py](file:///e:/AlphaToRoto/editablealpha_worker.py)) 서브프로세스로 분리 실행합니다.

![EditableAlpha Mask-to-Roto Preview](assets/editablealpha_preview.png)

![Real Mask Part Split & Cubic Bezier Roto Preview](assets/real_mask_test_preview.png)

---

## 1. Nuke 설치 방법 (1줄 등록)

`~/.nuke/init.py` (Windows 예: `C:\Users\<사용자명>\.nuke\init.py`)에 아래 2줄을 추가하면 Nuke 실행 시 `EditableAlpha.gizmo`와 메뉴/단축키가 자동 등록됩니다.

```python
import nuke
nuke.pluginAddPath(r"E:\AlphaToRoto")
```

- **툴바 메뉴**: `Nodes > EditableAlpha > EditableAlpha (Mask to Roto)`
- **단축키**: `Ctrl+Alt+E`
- **상단 메뉴**: `EditableAlpha > Check Worker Python Environment` (외부 Python `cv2`/`numpy` 연결 상태 확인)

---

## 2. 사용 방법 (Mask Input ➔ Editable Roto)

![EditableAlpha Nuke Properties UI](assets/editablealpha_properties_ui.png)

1. Nuke 그래프에서 알파 마스크 노드(예: `Read`, `Keylight`, `SAM2`/`AlphaGen` 출력 등)를 선택하고 **`Ctrl+Alt+E`** (또는 `Tab` → `EditableAlpha`)로 노드를 생성해 입력(`mask`)에 연결합니다.
2. **`Sync Range`** 버튼을 눌러 입력 시퀀스 프레임 범위(`Start Frame` / `End Frame`)를 맞춥니다.
3. 필요 시 파라미터를 조정합니다:
   - **Mask Source**: `Alpha (RGBA Input)` 또는 `Luminance (RGB/BW)`
   - **Max Parts / Seam Overlap (px)**: 관절 단위 파트 분할 개수 및 이음새 겹침 마진(기본 `12 px`)
   - **Min / Max Points / Part**: 파트당 제어점 개수
   - **Auto Multi-Master**: 움직임이 큰 시퀀스에서 대표 기준 프레임을 자동 다중 선출
   - **Output Mode**:
     - `Update Internal + Extract Roto` *(기본)*: Gizmo 내부 Roto 업데이트 + 그래프에 독립 `Roto` 노드 자동 추출
     - `Update Internal Gizmo Only`: Gizmo 내부 Roto만 갱신
     - `Extract Standalone Roto Only`: 독립 `Roto` 노드만 생성/갱신
   - **Preview**: `Converted Roto` / `Input Mask` A/B 전환
4. **`▶ Convert Mask to Roto`** 클릭 → 입력 마스크가 편집 가능한 다중 파트 베지에 `Roto` 노드로 변환됩니다.
5. **수동 수정 후 재추적**: 추출된 `Roto` 노드에서 현재 프레임의 제어점/핸들을 직접 수정한 뒤, Gizmo의 **`Re-Track (Keep Edits)`** 를 누르면 수정한 제어점이 100% 보존된 상태로 인접 프레임이 부드럽게 재추적됩니다.

---

## 3. 테스트 샘플 이미지 (Sample Test Sequences)

저장소에 포함된 15프레임 테스트 시퀀스를 Nuke `Read` 노드로 불러와 `EditableAlpha` 입력(`mask`)에 바로 연결해 테스트할 수 있습니다.

| 구분 | 경로 | 설명 | 미리보기 (F5) |
| :--- | :--- | :--- | :---: |
| **Nuke RGBA 테스트** | `sample_sequences/rgba_nuke/rgba_####.png` | `Read` 노드에 연결 즉시 Alpha 채널이 포함된 테스트 시퀀스 (`1–15f`) | ![Sample RGB](sample_sequences/rgb/frame_0005.png) |
| **바이너리 마스크** | `sample_sequences/masks/mask_####.png` | 흑백 알파 마스크 시퀀스 (`1–15f`, `Mask Source: luminance` 또는 `alpha`) | ![Sample Mask](sample_sequences/masks/mask_0005.png) |
| **원본 RGB 플레이트** | `sample_sequences/rgb/frame_####.png` | 배경이 포함된 원본 RGB 이미지 시퀀스 (`1–15f`) | — |

---

## 4. 헤드리스 테스트 실행

Nuke GUI를 켜지 않고도 워커 서브프로세스, Gizmo 구조, 수동 수정 보존, Roto 좌표 변환을 검증할 수 있습니다.

```bash
python -m unittest discover -s tests -v
```
