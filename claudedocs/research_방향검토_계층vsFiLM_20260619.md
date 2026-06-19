# 방향 검토 — README 로드맵(딥: FiLM/nnPU) vs 대안(통계적 계층 + 일별 ISI)

**작성** 2026-06-19 · **질문**: 다음 단계를 README 로드맵(Stage4 FiLM 계층 + FiLM-Ensemble, Stage5 conformal, Stage6 nnPU 딥 PU)으로 갈지, 아니면 대안(일별 fire-weather 동역학 + 경험적베이즈/계층 partial-pooling + ISI-조건부 확산)으로 갈지.
**우리 조건**: 정답 라벨 없음 · 양성(발화점) ~700개로 희소 · 그룹 있음(시군 16, 그중 4개 zero-fire) · 테이블 피처.

## 핵심 결론 (한 줄)
> **우리 데이터 규모에선 "딥(FiLM/nnPU)"이 아니라 "통계적 계층(partial pooling) + 일별 ISI 동역학"이 더 방어 가능하고 효과 기대도 높다.** README 로드맵의 FiLM/nnPU 딥 스테이지는 데이터가 작아 과적합 위험 → 뒤로 미루고, conformal은 불확실성 보조로 유지.

## 쟁점별 근거

### ① 양성 ~700개에서 딥 PU(nnPU)·FiLM은 과적합하는가 → 그렇다 (위험)
- **nnPU가 존재하는 이유 자체가 "딥 PU의 과적합"**이다: 유연한 신경망은 PU 위험추정이 0 아래로 새며 심한 과적합 → nnPU가 음수항을 클리핑해 막지만, 근본적으로 **양성이 적을수록 위험**. PU는 양성·미분류 둘 다 표본수 하한이 있다. [nnPU Kiryo 2017](https://arxiv.org/abs/1703.00593), [PU 유한표본 한계 arXiv:2507.07354](https://arxiv.org/abs/2507.07354)
- 일반론: **딥러닝은 소규모 데이터에서 암기·과적합**. → 우리(양성 ~700, 게다가 발화점은 검증전용)에선 딥 PU/FiLM이 부적합.

### ② 지역 계층: deep conditioning(FiLM) vs 통계적 partial pooling → 후자가 정석
- **"데이터에 그룹이 있으면 계층적 베이즈가 가장 방어 가능한 베이스라인"**. partial pooling = 표본수 가중으로 pooled↔unpooled 절충 → **희소·소그룹 과적합 억제**. [brms partial pooling](https://www.r-bloggers.com/2026/03/how-to-fit-hierarchical-bayesian-models-in-r-with-brms-partial-pooling-explained/), [PyMC multilevel](https://www.pymc.io/projects/examples/en/2022.12.0/case_studies/multilevel_modeling.html)
- 소규모 테이블+그룹에서 딥은 과적합, 계층 베이즈는 정보공유로 정규화. [Deep Tabular 서베이 arXiv:2410.12034](https://arxiv.org/pdf/2410.12034)
- **zero-event 지역 처리**: 경험적 베이즈(EB) 축소가 질병지도·소지역추정의 표준 — 사건수 적은 지역의 극단 추정을 분산 기반으로 전역/이웃 평균에 끌어당겨(borrow strength) 안정화. 우리의 4개 zero-fire 시군·초소형 시군에 정확히 맞음. [공간역학 EB](https://mkram01.github.io/EPI563-SpatialEPI/disease-mapping-i-aspatial-empirical-bayes.html), [disease mapping PMC4180601](https://pmc.ncbi.nlm.nih.gov/articles/PMC4180601/)
- → "지역 계층을 무조건 세분"이 맞되, **세분의 안전장치 = partial pooling/EB shrinkage**. FiLM(딥)은 같은 목적을 데이터 더 많이 요구하며 달성.

### ③ 일별 fire weather(ISI) 동역학 vs 시즌 집계 → 일별 우위
- 시간 집계(주/월 평균)는 상관분석엔 쓰이나 **운영·예측에 필요한 일별 fine-scale을 못 담는다**. 일별 FWI가 표준. [Wildfire prediction stats arXiv:1312.6481](https://arxiv.org/pdf/1312.6481), [CFFDRS FWI](https://www.nwcg.gov/publications/pms437/cffdrs/fire-weather-index-fwi-system)
- 시즌 예보 skill은 1~2개월 넘으면 기후값 수준으로 하락 → 시즌 평균 하나로 뭉개면 신호 손실. [Global seasonal fire danger PMC10810953](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10810953/)
- → 현재 `fwi_q90` 한 값 대신 **일별 ISI(확산)·FFMC(발화준비) 동역학을 시즌 집계**가 더 타당(+조기경보 프레이밍과도 합치). EDA의 "크기는 ISI·바람이 좌우"와 직결.

### ④ conformal을 약·noisy proxy 라벨로 보정 → 대체로 유효하나 보조
- **노이즈 라벨이어도 marginal coverage는 (조건부) 유지**되나, 구간이 **넓어지고 over-cover** 경향. "anti-dispersive noise"라는 실패형이 있어 주의. Noise-Aware CP(NACP) 등 대응법 존재. [Label Noise Robustness of CP, JMLR 2024](https://www.jmlr.org/papers/volume25/23-1549/23-1549.pdf), [Noisy-label CP arXiv:2501.12749](https://arxiv.org/abs/2501.12749)
- → conformal은 "커버리지 보장" 스토리·임계값 보조로 **쓸 수 있으나**, 우리 앵커가 약하므로 보수적으로(over-cover 감안). 백본 우선순위 아님.

## 권고 — 로드맵 수정
| README 로드맵 | 검토 결과 | 권고 |
|---|---|---|
| Stage4 **FiLM 딥 계층 + FiLM-Ensemble** | 소표본 과적합 위험 | **→ 통계적 계층(EB/random-effects partial pooling)으로 대체.** FiLM은 데이터 늘면 옵션 |
| Stage5 **conformal** | noisy 라벨로도 대체로 유효(over-cover) | **유지(보조)** — 불확실성·임계 스토리 |
| Stage6 **nnPU 딥 PU** | 양성 ~700 과적합 위험 | **후순위/소실험만**. 발화점은 검증전용 유지 |
| (신규) **일별 ISI/FFMC 동역학 W** | 일별 우위 명확 | **채택(우선)** |
| (신규) **ISI-조건부 풍하확산** | 물리 타당 | 채택(조건부, recall 해치지 않는 선) |

**다음(Phase 4) 우선순위**: ① 일별 ISI/FFMC 동역학 W 재설계 → ② 계층 partial-pooling/EB 지역보정(체제→시군→격자, zero-fire는 부모로 축소) → ③ ISI-조건부 확산 → ④ (선택) conformal 임계 → ⑤ (선택, 후순위) nnPU 소실험.
