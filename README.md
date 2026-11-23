# AlphaToRoto

추적 생성된 픽셀 단위의 래스터 알파 마스크 시퀀스를 작업자(아티스트)가 직관적으로 수정할 수 있는 **시공간 위상 일관형 다중 파트 3차 베지에(Cubic Bézier) 로토스코핑 셰이프**로 변환하고, **Python 기반 Foundry Nuke 테스트 환경** 및 **대화형 웹/CLI 실험 환경**과 연동하는 전체 프로젝트 시스템(`AlphaToRoto`) 구현체입니다.

---

## 1. 파일 구성 (Module Architecture)

1. **`mask_generator.py` — Stage 1: 마스크 생성·전파 및 마스터 프레임 선출 (식 1)**
   - 기존 알파 마스크 시퀀스 로드(`load_mask_sequence`) 또는 초기 프레임 분할(GrabCut / SAM) 후 양방향 광학 흐름 기반 마스크 전파(`generate_tracked_masks`).
   - 전경 면적(`Area`), 볼록 껍질 대비 $12\text{ px}$ 이상 오목 결함 개수(`N_defects`), 라플라시안 모션 블러 페널티(`BlurPenalty`)를 결합해 최적의 초기 변환 기준 프레임($F_{\text{master}}$) 자동 선출(`select_master_frame`).
2. **`part_decomposition.py` — Stage 2: 기하-운동학 파트 분할 및 $10\sim 15\text{ px}$ 내부 오버랩 마진 (식 2)**
   - 마스터 프레임의 오목 결함 쌍과 광학 흐름 아핀 잔차 에너지를 최소화하는 최적 절단선 탐색(`find_optimal_cut_lines`).
   - 절단선 법선 방향으로 $d_{\text{overlap}} = 10\sim 15\text{ px}$ 내부 오버랩 마진을 부여하여 파트 개별 수정 시 내부 솔기 빈틈(Seam Gap)을 $0\%$로 억제(`decompose_mask_into_parts`).
3. **`hermite_bezier_converter.py` — Stage 3: $O(N)$ 닫힌 형태 에르미트-베지에 제어점·핸들 변환 (식 3~5)**
   - 픽셀 외곽선 계단 노이즈 스무딩 및 곡률 적응형 희소 앵커 포인트 $\{P_0, \dots, P_{N-1}\}$ 추출.
   - 할선 기반 에르미트 접선 벡터 $V_i = \tau_i (P_{i+1} - P_{i-1})$ 및 첨점($\tau_i \to 0$) 자동 감지 후, 닫힌 형태 행렬 변환($\Delta h_i^{\text{out}} = +\frac{1}{3}V_i$, $\Delta h_i^{\text{in}} = -\frac{1}{3}V_i$)으로 3차 베지에 핸들 즉시 합성(`contour_to_hermite_bezier`).
   - 베지에 폐곡선 래스터라이저(`rasterize_bezier_shapes`), $\mathcal{J}$ (IoU), $\mathcal{F}$ (Boundary F-score), `RMSE (px)` 평가 함수 및 선행연구 베이스라인(`rdp_polygon_vectorize`, `sam2_autotrace_vectorize`, `least_squares_bspline_fit`) 포함.
4. **`autoroto_pipeline.py` — Stage 4: 시공간 위상 고정 전파, 작업자 수정 키프레임 보존 재추적 및 Nuke/SVG/JSON 출력 (식 6~8)**
   - 마스터 프레임의 제어점 개수($N$)와 인덱스를 전 프레임에 고정한 채 양방향 추적 및 윤곽선 법선 스냅핑 수행.
   - 작업자가 임의의 프레임에서 제어점을 수정할 경우(`apply_artist_keyframe_edit`), 수정 키프레임을 100% 비파괴 보존하며 사이 구간을 $C^1$ 스무스스텝($s(t)=3t^2-2t^3$) 가시성 가중 블렌딩(식 6~7)과 할선 회전 탄젠트 보정(식 8)으로 재베이킹.
   - 오라클 작업자 수정 횟수(`NoM`) 시뮬레이터(`simulate_oracle_editability_nom`), 시간적 떨림(`T_flicker`), 장기 드리프트(`dJ_decay`), Foundry Nuke Roto 노드 직접 주입(`inject_into_nuke_roto_node`), 독립 실행형 Nuke Python 스크립트 생성(`export_to_nuke_script_py`), SVG/JSON 내보내기 지원.
5. **`tracker_core.py` & `nuke_bridge.py` — CoTracker 3 GPU 백엔드 및 Foundry Nuke Roto 노드 통합**
   - Meta CoTracker 3 GPU 오프라인 백엔드 및 Foundry Nuke Roto 노드 UI 브리지.
   - Nuke의 `AlphaToRoto` 탭 내에 **`✨ Convert Input Alpha Mask to Editable Multi-Part Roto`** 버튼(`on_node_convert_alpha_to_roto`)이 통합되어, Nuke 안에서 입력 알파 마스크를 즉시 다중 파트 베지에 로토 셰이프로 변환하고 다중 기준점 CoTracker 3 재추적을 수행할 수 있습니다.
6. **`interactive_roto_viewer.py` — 대화형 HTML5 베지에 편집 스튜디오 & 샘플 시퀀스 생성기**
   - 브라우저에서 직접 제어점(`Anchor`)과 좌·우 접선 핸들(`in_handle`, `out_handle`)을 마우스로 드래그하여 키프레임 수정, $C^1$ 스무스스텝 재추적, 할선 회전 탄젠트 보정, Nuke 스크립트 다운로드를 즉시 테스트할 수 있는 독립 스튜디오(`demo_output/interactive_alphatoroto_studio.html`) 및 샘플 이미지 시퀀스(`sample_sequences/`)를 생성합니다.
7. **`run_paper_experiments.py` — 논문 표 1~3 정량 실험 자동 실행기**
   - `PROJECT_HANDOVER_AND_EXPERIMENT_GUIDE.md`의 **실험 1 (표 1)**, **실험 2 (표 2)**, **실험 3 (표 3, Ablation Study)**을 일괄 실행하여 마크다운 표(`experiment_results/PAPER_TABLES_1_2_3.md`)와 JSON(`experiment_results/experiment_metrics.json`)으로 저장합니다.

---

## 2. 직접 테스트 실행 가이드 (Quick Start)

### ① 단위·통합 테스트 실행 (Stage 1~4 수식 및 파이프라인 검증)
```bash
python test_human_in_the_loop_mask.py
```

### ② 대화형 HITL 베지에 테스트 스튜디오 & 샘플 PNG 시퀀스 생성
```bash
python interactive_roto_viewer.py
```
- 실행 후 생성되는 `demo_output/interactive_alphatoroto_studio.html`을 웹 브라우저로 열면 마우스로 제어점과 접선 핸들을 드래그하며 다중 키프레임 $C^1$ 보간과 Nuke 스크립트 내보내기를 즉시 체험할 수 있습니다.
- `sample_sequences/masks/`, `sample_sequences/rgb/`, `sample_sequences/rgba_nuke/` 폴더에 테스트용 PNG 시퀀스가 함께 생성됩니다.

### ③ CLI 마스크 시퀀스 ➔ Nuke Roto 변환 실행
```bash
python run_mask_to_roto_cli.py \
  --mask_dir "./sample_sequences/masks" \
  --rgb_dir "./sample_sequences/rgb" \
  --output_dir "./demo_output" \
  --overlap_px 12 \
  --max_parts 2 \
  --max_points 16
```

### ④ 논문 실험 1~3 (표 1, 표 2, 표 3) 정량 벤치마크 실행
```bash
python run_paper_experiments.py
```
- 실행 완료 시 `experiment_results/PAPER_TABLES_1_2_3.md` 및 `experiment_results/experiment_metrics.json`에 표 1~3의 모든 측정 수치가 출력됩니다.
