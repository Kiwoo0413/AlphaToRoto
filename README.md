# HumanInTheLoop_Mask (`AutoRoto`)

추적 생성된 픽셀 단위의 래스터 알파 마스크 시퀀스를 작업자(아티스트)가 직관적으로 수정할 수 있는 **시공간 위상 일관형 다중 파트 3차 베지에(Cubic Bézier) 로토스코핑 셰이프**로 변환하고, **Python 기반 Foundry Nuke 테스트 환경**과 연동하는 전체 프로젝트 시스템 구현체입니다.

---

## 파일 구성 (Module Architecture)

1. **`mask_generator.py` — Stage 1: 마스크 생성·전파 및 마스터 프레임 선출 (식 1)**
   - 기존 알파 마스크 시퀀스 로드(`load_mask_sequence`) 또는 초기 프레임 분할(GrabCut / SAM) 후 양방향 광학 흐름 기반 마스크 전파(`generate_tracked_masks`).
   - 전경 면적(`Area`), 볼록 껍질 대비 $12\text{ px}$ 이상 오목 결함 개수(`N_defects`), 라플라시안 모션 블러 페널티(`BlurPenalty`)를 결합해 최적의 초기 변환 기준 프레임($F_{\text{master}}$) 자동 선출(`select_master_frame`).
2. **`part_decomposition.py` — Stage 2: 기하-운동학 파트 분할 및 $10\sim 15\text{ px}$ 내부 오버랩 마진 (식 2)**
   - 마스터 프레임의 오목 결함 쌍과 광학 흐름 아핀 잔차 에너지를 최소화하는 최적 절단선 탐색(`find_optimal_cut_lines`).
   - 절단선 법선 방향으로 $d_{\text{overlap}} = 10\sim 15\text{ px}$ 내부 오버랩 마진을 부여하여 파트 개별 수정 시 내부 솔기 빈틈(Seam Gap)을 $0\%$로 억제(`decompose_mask_into_parts`).
3. **`hermite_bezier_converter.py` — Stage 3: $O(N)$ 닫힌 형태 에르미트-베지에 제어점·핸들 변환 (식 3~5)**
   - 픽셀 외곽선 계단 노이즈 스무딩 및 곡률 적응형 희소 앵커 포인트 $\{P_0, \dots, P_{N-1}\}$ 추출.
   - 할선 기반 에르미트 접선 벡터 $V_i = \tau_i (P_{i+1} - P_{i-1})$ 및 첨점($\tau_i \to 0$) 자동 감지 후, 닫힌 형태 행렬 변환($\Delta h_i^{\text{out}} = +\frac{1}{3}V_i$, $\Delta h_i^{\text{in}} = -\frac{1}{3}V_i$)으로 3차 베지에 핸들 즉시 합성(`contour_to_hermite_bezier`).
   - 베지에 폐곡선 래스터라이저(`rasterize_bezier_shapes`) 및 $\mathcal{J}$ (IoU), $\mathcal{F}$ (Boundary F-score) 평가 함수 포함.
4. **`autoroto_pipeline.py` — Stage 4: 시공간 위상 고정 전파, 작업자 수정 키프레임 보존 재추적 및 Nuke/SVG/JSON 출력 (식 6~8)**
   - 마스터 프레임의 제어점 개수($N$)와 인덱스를 전 프레임에 고정한 채 양방향 추적 및 윤곽선 법선 스냅핑 수행.
   - 작업자가 임의의 프레임에서 제어점을 수정할 경우(`apply_artist_keyframe_edit`), 수정 키프레임을 100% 비파괴 보존하며 사이 구간을 $C^1$ 스무스스텝($s(t)=3t^2-2t^3$) 가시성 가중 블렌딩(식 6~7)과 할선 회전 탄젠트 보정(식 8)으로 재베이킹.
   - 오라클 작업자 수정 횟수(`NoM`) 시뮬레이터(`simulate_oracle_editability_nom`), Foundry Nuke Roto 노드 직접 주입(`inject_into_nuke_roto_node`), 독립 실행형 Nuke Python 스크립트 생성(`export_to_nuke_script_py`), SVG/JSON 내보내기 지원.
5. **`tracker_core.py` & `nuke_bridge.py`**
   - Meta CoTracker 3 GPU 오프라인 백엔드 및 Foundry Nuke Roto 노드 UI 브리지.
6. **`run_mask_to_roto_cli.py` & `test_human_in_the_loop_mask.py`**
   - 커맨드라인 데모/변환 실행기 및 Stage 1~4 전체 수식·파이프라인 단위/통합 테스트 스위트.
