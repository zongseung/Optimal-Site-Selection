# MOSAIC — 전력설비(전주) 산불위험 분석

> **2026 날씨 빅데이터 콘테스트 · 주제1(재난안전)** — 기상·공간정보 기반 전력설비 인근 산불위험 분석
> 주최 **기상청** · 참여기관 **한국전력공사(KEPCO)**

강원도 가상 전주 **1,387,831본** 각각에 대해 산불위험 여부를 **0/1**로 분류한다. 정답 라벨은
비공개이며(presence-only/PU), 과거 발화점은 **학습 타깃이 아니라 검증·임계값 결정의 앵커**로만
쓴다. 위험 자체는 **산불 물리식 + 지역 혼합전문가(MoE)** 로 비지도에 가깝게 추정한다.

**MOSAIC** = **M**ixture-of-experts · **S**patial(공간 블록 CV·BYM2) · **A**nchored(발화점 앵커) ·
**I**gnition-physics(R = I·S·W) · **C**overage(per-regime conformal). 강원을
영동/회랑/영서/산악의 **모자이크**로 보고 각 조각을 체제 전문가가 맡는 구조가 모델의 정체성이다.

---

## 1. 모델 구조 — R = I·S·W

전주 *p*의 시즌(산불조심기간 2~5월) 위험을 세 항의 곱으로 분해한다.

```
R(p) = I(p) · S(p) · W(p)
       │       │       └─ W: 기상·계절(FWI / ISI·FFMC 동역학)
       │       └───────── S: 정적 취약 + 양간지풍 풍하노출
       └───────────────── I: 발화 성향(4-레짐 MoE)
```

### I — 발화 성향: 4-레짐 혼합전문가(MoE)
강원은 지역마다 발화 메커니즘이 다르므로(`config.REGIMES`) 4개 전문가를 둔다.

| 레짐 | 지역 | 발화 구동인자 |
|---|---|---|
| `yeongdong` 영동 | 강릉·동해·삼척 | 가뭄·FWI |
| `corridor` 회랑 | 양양·고성·속초 | 양간지풍·송전선 |
| `yeongseo` 영서 | 내륙 | 인적발화(소각·입산자) |
| `mountain` 산악 | 고지 산간 | 연료·도로 접근 |

- **게이트(gate)**: 전주 위치피처 `(경도, 고도, 양간일수)`를 표준화해 체제 앵커와의 거리로
  softmax(온도 0.6) → 4개 전문가에 대한 soft 가중(행합=1). 경계 전주는 자연 블렌딩되어
  하드 분할의 MAUP을 피한다. (`pfire/regimes.py`)
- **전문가(expert)**: 각 레짐이 동일 피처(산림거리·도로·송전선거리·FWI·양간일수·연료가연성·토지피복)를
  서로 다른 가중(`config.EXPERT_WEIGHTS`, 7성분)으로 결합. (`pfire/experts.py`)
- `I(p) = Σ_r gate_r(p) · Σ_j w_{rj} · f_j(p)`

### S — 정적 취약 + 양간지풍 풍하노출
정적 확산취약 `S_p`(사전계산)에 풍하노출을 결합한다.

- **풍하 확산 MC**(`pfire/exposure_engine.py` → Rust 커널 `rust/pfire_kernels`): 서풍 prior를 따라
  동쪽으로 번지는 이방성 풍하 커널을 몬테카를로로 모사 → 노출확률 `p_exposure`.
- **노출 v2.2**(`pfire/exposure_v2.py`): 발화가중 기대도즈 + LWR 타원 + 국지정규화로 영동 포화를
  해제한 변별 dose. 결정에 `R ← 1−(1−R)(1−w·dose)`로 결합(`EXPOSURE_V2_BLEND_W=0.25`).
  근거·검증: [`exposure_v2/README.md`](exposure_v2/README.md).

### W — 기상·계절
전주별 시즌 극값(FWI 등)과 일별 동역학(ISI·FFMC)을 [0,1]에서 선형 블렌드한다(`pfire/weather.py`).
`W = w·W_season + (1−w)·W_daily`, `W_BLEND_SEASON_WEIGHT=0.82`(고성 극값 보존 + 일별 recall 이득).

### 불확실성 — 베이지안 사후
격자 발화수·노출에 발생률 사후 `λ ~ Γ(α₀+Σy, β₀+ΣE)`를 두고, 사후를 물리 base risk에 MC 전파해
전주별 **90% credible `risk_lo`/`risk_hi`**를 산출한다(`pfire/posterior.py`). 백엔드는 독립
**Poisson-Gamma**(기본)와 공간 CAR로 이웃 정보를 빌리는 **BYM2**(`--posterior-spatial bym`,
INLA식 Laplace 근사). per-regime **conformal**로 명목 90% ≈ 실측 커버리지를 검증한다.
불확실성은 0/1 결정에 쓰지 않고 운영 분류에만 쓴다.

### 결정 — regime-anchor 배분
전역 단일 컷이 위험점수 높은 영동에 양성을 싹쓸이하는 것을 막는다. 체제별 발화앵커밀도 비례로
양성 예산(기본 prevalence 2%)을 배분한 뒤 레짐 내 위험순위로 컷을 둔다(`pfire/calibrate.py`).

### 학습 — within-regime 목적함수
가중 튜닝(`scripts/tune_weights.py --within-regime`)의 목적을 전역 top-k recall이 아니라
레짐별 within recall의 동일가중 평균으로 둔다. between-regime 기저율 지름길을 제거해 레짐 내
변별을 직접 최적화한다. 산출: `outputs/tuned_weights_within.json`.

---

## 2. 디렉토리 구조

```
OR-project/
├── README.md                  # (이 파일)
├── pyproject.toml             # Python ≥3.12, uv 관리 (.python-version=3.12)
├── pfire/                     # 핵심 파이프라인 패키지
│   ├── config.py              # 단일 진실: 경로·상수·레짐·가중·하이퍼파라미터
│   ├── io.py                  # polars 마스터 프레임 로더(pole_id 조인)
│   ├── regimes.py             # 4-레짐 soft 게이트(MoE)
│   ├── experts.py             # 체제별 물리식 전문가 → I(p)
│   ├── exposure.py            # 풍하 확산 노출 커널 래퍼(Rust + numpy 폴백)
│   ├── exposure_engine.py     # 노출 실연동(발화후보·바람표집·평면km)
│   ├── exposure_v2.py         # 변별 dose v2.2(타원+국지정규화)
│   ├── weather.py             # 기상 W(시즌극값 × 일별 ISI/FFMC 블렌드)
│   ├── hazard.py              # R = I·S·W 결합·시즌 집계
│   ├── hierarchy.py           # 계층 EB(partial pooling) 지역배율
│   ├── posterior.py           # 베이지안 사후(Poisson-Gamma / BYM2) + MC 전파
│   ├── calibrate.py           # 임계값·regime-anchor 배분·conformal
│   ├── validate.py            # 공간 블록 CV·recall·sanity
│   ├── fire_cause.py          # 화재 원인 분류(설비원인 앵커 검증)
│   ├── ablation.py            # asset-aware vs asset-blind LOGO ablation
│   ├── risk_index.py          # 위험 백분위 지수
│   ├── geo.py                 # lon/lat ↔ 평면 km 변환
│   └── submit.py              # 제출 CSV 작성·무결성 검증
├── scripts/                   # 실행 엔트리포인트(아래 §4)
│   ├── run_phase1_mvp.py      # end-to-end(Phase-4/5 통합): R → 사후 → 제출
│   ├── run_phase3.py          # 토지피복 + 조건부 풍하노출 단계 검증·제출
│   ├── tune_weights.py        # EXPERT_WEIGHTS 튜닝(--within-regime)
│   ├── make_figures.py        # 정성평가용 그림 fig1~11 생성
│   ├── figures/               # 보고서용 보조 그림 생성(루트에서 실행): eda_4regime·fig_goseong_corridor·fig_submission_uncertainty·fig_submission_map
│   ├── experiments/           # PoC(파이프라인 외): poc_lgcp·poc_lgcp_full (베이지안 포아송/LGCP)
│   └── eda_derived.py · risk_percentile.py · run_ablation*.py · validate_equipment_cause.py · …
├── used_dataset/              # 입력 데이터(스냅샷; 결과물·예보·FIRMS 제외)
│   ├── poles/                 # 가상 전주 1,387,831본 + 피처 parquet
│   ├── weather/               # AWS 일별 관측·일별 FWI/ISI/FFMC·관측소 좌표
│   ├── fire/                  # safemap 발화점(검증·임계 앵커)
│   ├── admin/                 # 행정경계 wkt
│   ├── burn/                  # 2019 고성 burn-scar(dNBR) sanity
│   └── README.md              # 데이터 명세
├── outputs/                   # 산출물(아래 §5)
│   ├── submissions/           # submission.csv·submission_p{π}.csv·test_hanjeon.csv
│   ├── figures/               # fig1~11 + eda/
│   ├── regime_threshold_analysis.json
│   ├── tuned_weights*.json · ablation_*.json · equipment_cause_validation.json
│   └── pole_dose_v2*.parquet · pole_risk_v2blend.parquet
├── exposure_v2/               # 노출 v2.2 재설계 문서·flagship 그림
├── rust/pfire_kernels/        # 풍하 확산 Rust 커널(rayon 병렬)
├── report/                    # 보고서·수식 정리(.tex / 최종공모안.md)
├── notebooks/                 # EDA.ipynb
└── tests/                     # pytest
```

---

## 3. 데이터

스냅샷 명세는 [`used_dataset/README.md`](used_dataset/README.md). 핵심:

| 구분 | 위치 | 내용 |
|---|---|---|
| 전주 | `used_dataset/poles/` | 가상 전주 1,387,831본 좌표 + 지형·연료·FWI·송전선/변전소거리·토지피복·시군 피처(parquet) |
| 기상 | `used_dataset/weather/` | AWS 일별 관측(풍향·풍속), 일별 FWI/ISI/FFMC, 관측소 109곳 좌표 |
| 발화이력 | `used_dataset/fire/` | safemap 발화점 928(검증·임계 앵커, presence-only 양성) |
| 행정·흉터 | `used_dataset/admin/`, `used_dataset/burn/` | 강원 행정경계, 2019 고성 burn-scar(dNBR) |

좌표계는 EPSG:4326(위경도), 거리·확산은 강원 대표 위도 기준 평면 근사 km. 모든 경로·상수는
`pfire/config.py`에 단일 진실로 두며 다른 모듈은 하드코딩하지 않는다.

---

## 4. 실행 방법

Python ≥3.12 가상환경(`.venv`)이 준비되어 있다. 파이프라인은 다음 순서로 실행한다.

```bash
# (가상환경 — .venv/bin/python 직접 호출도 가능)
source .venv/bin/activate

# 1) end-to-end 위험 추정 → 사후 불확실성 → 제출 (Phase-4/5 통합, 메인)
.venv/bin/python scripts/run_phase1_mvp.py \
    --prevalence 0.02 \
    --posterior-spatial bym \      # BYM2 공간 CAR 사후(기본은 poisson_gamma)
    --submission-variants          # π별 submission_p{π}.csv 변형 생성

# 2) 토지피복 + 조건부 풍하노출 단계 검증·제출 재생성
.venv/bin/python scripts/run_phase3.py --prevalence 0.02

# 3) 정성평가용 그림 fig1~11 생성
.venv/bin/python scripts/make_figures.py
```

산출 → `outputs/submissions/submission.csv`, `outputs/regime_threshold_analysis.json`,
`outputs/figures/`.

**주요 플래그(`run_phase1_mvp.py`)**

| 플래그 | 기본 | 의미 |
|---|---|---|
| `--prevalence` | 0.02 | 운영 예측양성 비율(임계 예산) |
| `--alloc-mode` | auto | 양성 배분: `auto`=공간CV로 체제별/전역 자동 채택, `regime_anchor` 등 강제 |
| `--posterior` | on | 베이지안 사후 + MC 전파 + per-regime 커버리지 |
| `--posterior-spatial` | poisson_gamma | 사후 백엔드: `poisson_gamma`(독립) / `bym`(BYM2 공간 CAR) |
| `--submission-variants` | off | `PREVALENCE_GRID` π별 제출 변형 CSV 생성 |
| `--multiplier` | auto | 계층 EB 지역배율 채택(누수안전 공간CV 기준) |

**가중 튜닝(선택)** — within-regime 목적함수로 `EXPERT_WEIGHTS` 재튜닝:

```bash
.venv/bin/python scripts/tune_weights.py --within-regime --n-random 4000 --workers 60
# → outputs/tuned_weights_within.json
```

> 재현성: `config.SEED` 고정. 평가는 항상 공간 블록(10km) CV이며 무작위-공간 격차(낙관편향)를
> 함께 보고한다.

---

## 5. 산출물

`outputs/submissions/`
- **`submission.csv`** — 제출 + 해석용 컬럼:
  `pole_id, lon, lat, decision(0/1), risk_score, regime, p_exposure, risk_lo, risk_hi, ops_priority, risk_pctile, risk_pctile_regime`
- **`submission_p{π}.csv`** — `--submission-variants`로 생성한 π별(0.005~0.10) 제출 변형.
- **`test_hanjeon.csv`** — 콘테스트 제출용 슬림형 `pole_id, lon, lat, decision`
  (가상전주 1,387,831본 위험여부 0/1).

`outputs/`
- `regime_threshold_analysis.json` — 배분·F1 민감도·공간CV recall·고성 sanity·사후 커버리지.
- `tuned_weights.json` / `tuned_weights_within.json` — 가중 튜닝 provenance.
- `ablation_asset.json` · `ablation_by_cause.json` · `equipment_cause_validation.json` — 자산 피처
  기여·설비원인 검증.
- `figures/` — `fig1_risk_map` ~ `fig11_uncertainty_map`(위험지도·체제배분·recall·F1·고성·
  분해·시군 서브플롯·커버리지·BYM vs PG·불확실성) + `eda/`.

제출 무결성은 `pfire/submit.py`가 강제 검증한다(행수 1,387,831 · `decision` 0/1만 · `pole_id` 정렬).

---

## 6. 보고서

- [`report/최종공모안.md`](report/최종공모안.md) — **최종 공모안**(공식 양식 1~5절).
- [`report/`](report/README.md) — 6쪽 논문형 보고서(`report.tex`)·수식 정리(`formulas.tex`).
- [`exposure_v2/README.md`](exposure_v2/README.md) — 풍하 노출 v2.2 재설계 근거·검증.
</content>
