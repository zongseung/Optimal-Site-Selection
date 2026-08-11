# MOSAIC — 전력설비(전주) 산불위험 분석

[![Contest](https://img.shields.io/badge/2026_날씨_빅데이터_콘테스트-주제1_재난안전-1f6feb)](https://bd.kma.go.kr)
[![Award](https://img.shields.io/badge/🏆_3등상-한국전력사장상-FFD700)](https://www.kepco.co.kr)

**[English](README.md)** | 한국어

> **🏆 2026 날씨 빅데이터 콘테스트 · 주제1(재난안전) — 3등상(한국전력사장상) 수상작**
> 주최 **기상청** · 참여기관 **한국전력공사(KEPCO)**

강원도 가상 전주 **1,387,831본** 각각에 대해 산불위험 여부를 **0/1**로 분류한다. 정답 라벨은
비공개이며(presence-only/PU), 과거 발화점은 **학습 타깃이 아니라 검증·임계값 결정의 앵커**로만
쓴다. 위험 자체는 **산불 물리식 + 지역 혼합전문가(MoE)** 로 비지도에 가깝게 추정한다.

**MOSAIC** = **M**ixture-of-experts · **S**patial(공간 블록 CV·BYM2) · **A**nchored(발화점 앵커) ·
**I**gnition-physics($R = I \cdot S \cdot W$) · **C**overage(per-regime conformal). 강원을
영동/회랑/영서/산악의 **모자이크**로 보고 각 조각을 체제 전문가가 맡는 구조가 모델의 정체성이다.

---

## 1. 모델 — $R = I \cdot S \cdot W$

산불조심기간(2~5월) 중 전주 $p$의 일 위험을 세 항의 곱으로 분해한다.

$$
h(p,t) = \underbrace{I(p)}_{\text{발화 성향}} \times \underbrace{S(p)}_{\text{확산·노출 취약}} \times \underbrace{W(g(p),t)}_{\text{그날 기상}}
$$

**왜 곱인가:** 셋 중 하나라도 0이면 위험 0이라는 AND 성격이 물리적으로 타당하고(불씨가
없거나/탈 게 없거나/날이 궂지 않으면 안 난다), 각 항을 따로 해석해 "왜 위험한가"를 항목별로
설명할 수 있다. 모든 성분은 $[0,1]$로 정규화하고 범위를 강제 검증한다(`pfire/hazard.py`).

**시즌 집계.** 일별 경로는 기하평균(로그합 평균)이라 단발 고위험일 하나가 시즌 위험을
폭주시키는 것을 완화한다.

$$
R(p) = \exp\left(\frac{1}{T}\sum_{t=1}^{T} \log\big[\mathrm{clip}(I \cdot S \cdot W_t,\ \varepsilon,\ 1)\big]\right)
$$

### 1.1 $I$ — 발화 성향: 4-레짐 혼합전문가(MoE)

강원은 지역마다 발화 메커니즘이 다르므로(`config.REGIMES`) 4개 전문가를 둔다.

| 레짐 | 지역 | 발화 구동인자 |
|---|---|---|
| `yeongdong` 영동 | 강릉·동해·삼척 | 가뭄·FWI |
| `corridor` 회랑 | 양양·고성·속초 | 양간지풍·송전선 |
| `yeongseo` 영서 | 내륙 | 인적발화(소각·입산자) |
| `mountain` 산악 | 고지 산간 | 연료·도로 접근 |

**피처 정규화**(`pfire/experts.py`) — 거리 피처는 가까울수록 위험이므로 지수 감쇠:

$$
\mathrm{feat}_{\text{dist}}(p) = \exp\!\big(-d(p)/\text{scale}\big),\qquad
\text{scale} = 100\,\text{m(산림)},\ 300\,\text{m(도로)},\ 1500\,\text{m(송전선)}
$$

FWI·양간일수는 min–max 정규화, 연료는 $\mathrm{clip}(\mu_{\text{flam}},0,1)$, 토지피복은 도메인
사전치 $\mathrm{lc\_ignition}\in[0,1]$(산림 1.00 … 수역 0.00).

**전문가** — 체제 $r$은 동일한 7개 피처를 서로 다른 가중(`config.EXPERT_WEIGHTS`, 공간블록 CV
recall 목적의 LHS+좌표상승 튜닝)으로 결합한다. 가중평균이라 $[0,1]$이 보장된다.

$$
E_r(p) = \frac{\sum_k w_{r,k}\,\mathrm{feat}_k(p)}{\sum_k w_{r,k}},\qquad
k \in \{\text{forest, road, powerline, fwi, yanggan, fuel, landcover}\}
$$

**soft 게이트**(`pfire/regimes.py`) — 게이트 피처 $z_p=(\text{경도, 고도, 양간일수})$를
z-표준화하고 체제 앵커 $\mu_r$과의 거리로 softmax:

$$
g_r(p) = \frac{\exp\!\big(-\lVert z_p-\mu_r\rVert^2/\tau\big)}{\sum_{r'}\exp\!\big(-\lVert z_p-\mu_{r'}\rVert^2/\tau\big)},
\qquad \sum_r g_r(p)=1,\quad \tau=0.6
$$

경계 전주는 자연 블렌딩되어 하드 분할의 MAUP 문제를 피한다.

$$
\boxed{\ I(p) = \sum_r g_r(p)\,E_r(p)\ } \in [0,1]
$$

### 1.2 $W$ — 기상

두 성분을 선형 블렌드한다(`pfire/weather.py`).

$$
W_{\text{season}}(p) = 0.5\,\mathrm{fwi\_q90} + 0.3\,\mathrm{hdd} + 0.2\,\mathrm{yanggan}
$$

$$
W_{\text{daily}}(p) = 0.05\,(\text{ISI 성분}) + 0.75\,(\text{FFMC 성분}) + 0.20\,(\text{양간 성분})
$$

ISI 성분은 $0.5\,(\text{고-ISI일수비율}) + 0.25\,\overline{\mathrm{ISI}} + 0.25\,\mathrm{ISI}_{q90}$
(임계 ISI ≥ 10, FFMC ≥ 90)이며, 관측소별 시즌 집계를 최근접 관측소로 전주에 부여한다. 발화
앵커가 인간발화 위주라 발화준비(FFMC)가 본체다.

$$
\boxed{\ W(p) = w_s\,W_{\text{season}}(p) + (1-w_s)\,W_{\text{daily}}(p)\ },\qquad w_s = 0.82
$$

시즌 성분이 전주 고유 극값(예: 2019 고성 인근)을 보존한다 — 일별 관측소 매핑만 쓰면 희석된다.

### 1.3 $S$ — 확산·노출 취약

정적 취약 $S_{\text{static}}$을 풍하 노출과 **전주별 조건부**로 블렌딩한다. 영동·고양간
전주에서만 블렌드 강도를 키워 전역 균일 희석을 막는다(`pfire/hazard.py`).

$$
\alpha_p = \alpha_{\text{base}} \cdot g_{\text{영동}}(p) \cdot \widehat{\mathrm{yanggan}}(p) \in [0, \alpha_{\text{base}}]
$$

$$
S(p) = (1-\alpha_p)\,S_{\text{static}}(p) + \alpha_p\,\widetilde{P}_{\text{expo}}(p)
$$

$\widetilde{P}_{\text{expo}}$는 노출확률의 순위 백분위(0 다수 분포 강건화).

### 1.4 풍하 노출 — OR 포화에서 변별 도즈로

**v1 몬테카를로 OR 커널**(`pfire/exposure.py`, `pfire/exposure_engine.py`, Rust 백엔드
`rust/pfire_kernels`): 발화후보($I$ 상위 2%)에서 바람을 표집해 비등방 확산을 모사한다.

$$
L = L_0\big(1+\alpha\,\max(0,\mathrm{align})\,\tfrac{ws}{5}\big),\qquad
\Pr(\text{도달}) = e^{-d/L}\cdot \mathrm{fuel}\cdot(1+\beta\,\mathrm{southness})
$$

$$
P_{\text{expo}}(p) = \frac{1}{S}\sum_s \mathbf{1}\big[\text{어느 발화원이라도 } p \text{ 도달}\big]
$$

**문제:** 발화원이 조밀한 영동에서 OR 확률이 $P\approx 1$로 포화되어 변별력을 잃는다.

**v2.2 발화가중 기대도즈**(`pfire/exposure_v2.py`)는 OR(포화)을 누적합(비포화)으로 바꾸고
Anderson 길이/폭비(LWR) 타원과 격자 국지정규화를 더한다.

$$
U = 2.237\,ws\ \text{(mph)},\qquad
\mathrm{LWR}(ws) = \min\!\big(0.936\,e^{0.2566U} + 0.461\,e^{-0.1548U} - 0.397,\ 8\big)
$$

$$
r = \sqrt{(d_\parallel/L_\parallel)^2 + (d_\perp/W_\perp)^2},\qquad
\mathrm{dose}(p) = \sum_g I_g\,\big\langle e^{-r}\big\rangle_{\text{바람}}\,\mathrm{fuel}_p\,(1+\beta\,s_p)
$$

풍하는 $L_\parallel = \mathrm{LWR}\cdot W_\perp$로 길게, 풍상은 $L_{\text{back}}$으로 짧게 둔다.
국지정규화가 "영동은 다 높음" 배경을 제거한다:

$$
\widetilde{e}(p) = \frac{\mathrm{dose}(p)}{\overline{\mathrm{dose}}_{\mathrm{cell}(p)}+\varepsilon},\qquad
\mathrm{dose01}(p) = \mathrm{percentile}\big(\widetilde{e}(p)\big)/100
$$

결정에는 발화 랭킹을 보존하는 확률 OR로 결합한다:

$$
\boxed{\ R'(p) = 1-\big(1-R(p)\big)\big(1-w\cdot\mathrm{dose01}(p)\big)\ },\qquad w = 0.25
$$

검증: 영동 변별 IQR $0.166 \to 0.290$, 공간CV recall $+0.0071$(5 fold-seed 일관).
근거·검증 전체: [`exposure_v2/README.md`](exposure_v2/README.md).

### 1.5 지역율 계층 보정 — 경험적 베이즈(EB)

발화 앵커가 희소(~900개)하고 일부 시군은 발화 0이다. 관측율을 그대로 쓰면 과적합하므로
Poisson–Gamma EB 축소로 부모 단계로 끌어당긴다(전역→체제→시군→격자 top-down,
`pfire/hierarchy.py`). 부모율 $m$·형제 분산 $v$로 적률추정($\alpha=m^2/v,\ \beta=m/v$)하면

$$
\tilde\lambda_g = \frac{y_g+\alpha}{n_g+\beta} = w_g\,\hat\lambda_g + (1-w_g)\,m,
\qquad w_g = \frac{n_g}{n_g+\beta}
$$

대표본은 관측율에 가깝고($w_g \to 1$), 0발화 소표본은 부모율로 축소된다($w_g \to 0$). 전주
배율은 격자 EB율을 전역 EB율로 나눈 상대배율(전역평균 ≈ 1).

### 1.6 불확실성 — 베이지안 사후 + MC 전파

`pfire/posterior.py`에 두 백엔드가 있다. 불확실성은 0/1 결정에 쓰지 않고 운영 분류에만 쓴다.

**Poisson–Gamma 켤레 사후(기본, 닫힌형):**

$$
\lambda_g \mid y_g \sim \mathrm{Gamma}(\alpha_0+y_g,\ \beta_0+n_g)
$$

zero-event 지역은 자동으로 사후 분산이 커진다. 각 draw가 물리 base risk를 곱해
$R^{(r)}(p) = \mathrm{clip}(R_{\text{base}}(p)\cdot m^{(r)}(p), 0, 1)$, 전주별
**90% credible `risk_lo`/`risk_hi`**($q_{0.05}, q_{0.95}$)를 낸다.

**BYM2 공간 CAR**(`--posterior-spatial bym`, INLA식 Laplace 근사) — 이웃 격자에서 정보를
빌린다(borrow strength):

$$
y_i \sim \mathrm{Poisson}(E_i e^{x_i}),\qquad
Q_x(\tau,\phi) = \tau\big[(1-\phi)I + \phi\,Q_{\text{ICAR}}\big]
$$

각 하이퍼 $(\tau,\phi)$에서 음의 로그사후 $f(x)=\sum_i(E_i e^{x_i}-y_i x_i)+\tfrac12 x^\top Q_x x$를
sparse Newton으로 최소화하고, Laplace 주변우도

$$
\log p(y\mid\tau,\phi) \approx \sum_i\big(y_i x_i^\star - E_i e^{x_i^\star}\big)
+\tfrac12\log|Q_x| - \tfrac12 x^{\star\top}Q_x x^\star - \tfrac12\log|H|,
\qquad H = \mathrm{diag}(E e^{x^\star}) + Q_x
$$

가 최대인 점을 채택한다. 격자 사후는 가우시안 $x_i \sim \mathcal N(x_i^\star, (H^{-1})_{ii})$.

**per-regime conformal** — 교환가능 가정만으로 유한표본 커버리지를 보장한다. 체제별 보정
점수를 정렬해

$$
\tau_r = \mathrm{score}_{(\lfloor \alpha (n_r+1)\rfloor)}
\quad\Rightarrow\quad
\Pr\big(\text{새 발화점 위험} \ge \tau_r\big) \gtrsim 1-\alpha,\qquad \alpha = 0.10
$$

명목 90% 대비 실측 커버리지: 전체 0.902(영동 0.907 · 영서 0.902 · 산간 0.897).

### 1.7 결정 — regime-anchor 배분

전역 단일 컷은 위험점수 높은 영동이 양성을 싹쓸이하게 만든다. 대신 체제별 발화앵커밀도
비례로 양성 예산(기본 prevalence 2%)을 배분한 뒤 레짐 내 위험순위로 컷을 둔다
(`pfire/calibrate.py`). 정답 양성비율 $\rho$가 미지이므로 가정별 F1 민감도 곡선을 보고한다:

$$
\mathrm{prec}_{\text{proxy}} = \frac{\rho\cdot\mathrm{recall}\cdot N}{n_{\text{pred}}},\qquad
F_1 = \frac{2\,\mathrm{prec}_{\text{proxy}}\,\mathrm{recall}}{\mathrm{prec}_{\text{proxy}}+\mathrm{recall}},
\qquad \rho \in \{0.5,1,2,3,5,10\}\%
$$

### 1.8 학습 — within-regime 목적함수

가중 튜닝(`scripts/tune_weights.py --within-regime`)의 목적을 전역 top-k recall이 아니라
레짐별 within recall의 동일가중 평균으로 둔다. between-regime 기저율 지름길을 제거해 레짐 내
변별을 직접 최적화한다. 평가는 항상 **공간 블록(10km) CV**이며 무작위-공간 격차(낙관편향)를
함께 보고한다. 전 과정은 `config.SEED`로 결정적이다.

---

## 2. 디렉토리 구조

```
OR/
├── README.md                  # 영문판 · README.ko.md — 이 파일
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
├── scripts/                   # 실행 엔트리포인트(§4)
│   ├── run_phase1_mvp.py      # 메인 end-to-end: R → 사후 → 제출
│   ├── run_phase3.py          # 토지피복 + 조건부 풍하노출 단계 검증
│   ├── tune_weights.py        # EXPERT_WEIGHTS 튜닝(--within-regime)
│   ├── predict.py             # 배포 CLI: 새 기상 입력 → 위험지도 출력
│   ├── make_figures.py        # 보고서 그림 fig1~fig11
│   ├── figures/               # 보고서 보조 그림(루트에서 실행)
│   └── eda_derived.py · risk_percentile.py · run_ablation*.py · validate_equipment_cause.py · …
├── used_dataset/              # 입력 데이터 스냅샷(내부 README 참조)
├── outputs/                   # 산출물(§5)
├── exposure_v2/               # 노출 v2.2 재설계 문서·flagship 그림
├── rust/pfire_kernels/        # 풍하 확산 Rust 커널(rayon 병렬)
├── webapp/                    # 로컬 위험지도 웹앱(표준 라이브러리만)
├── report/                    # 최종 공모안·LaTeX 보고서
├── notebooks/                 # EDA 노트북
└── tests/                     # pytest(122개)
```

---

## 3. 데이터

스냅샷 명세는 [`used_dataset/README.md`](used_dataset/README.md). 핵심:

| 구분 | 위치 | 내용 |
|---|---|---|
| 전주 | `used_dataset/poles/` | 가상 전주 1,387,831본 좌표 + 지형·연료·FWI·송전선/변전소거리·토지피복(parquet) |
| 기상 | `used_dataset/weather/` | AWS 일별 관측(풍향·풍속), 일별 FWI/ISI/FFMC, 관측소 109곳 좌표 |
| 발화이력 | `used_dataset/fire/` | safemap 발화점 928(검증·임계 앵커, presence-only 양성) |
| 행정·흉터 | `used_dataset/admin/`, `used_dataset/burn/` | 강원 행정경계, 2019 고성 burn-scar(dNBR) |

좌표계는 EPSG:4326, 거리·확산은 강원 대표 위도 기준 평면 근사 km. 모든 경로·상수는
`pfire/config.py`에 단일 진실로 두며 다른 모듈은 하드코딩하지 않는다.

---

## 4. 재현 방법

Python ≥ 3.12 가상환경(`.venv`) 기준:

```bash
# 1) end-to-end 위험 추정 → 사후 불확실성 → 제출 (메인)
.venv/bin/python scripts/run_phase1_mvp.py \
    --prevalence 0.02 \
    --posterior-spatial bym \      # BYM2 공간 CAR 사후(기본은 poisson_gamma)
    --submission-variants          # π별 submission_p{π}.csv 변형 생성

# 2) 토지피복 + 조건부 풍하노출 단계 검증
.venv/bin/python scripts/run_phase3.py --prevalence 0.02

# 3) 보고서 그림 fig1~fig11
.venv/bin/python scripts/make_figures.py
```

`run_phase1_mvp.py` 주요 플래그:

| 플래그 | 기본 | 의미 |
|---|---|---|
| `--prevalence` | 0.02 | 운영 예측양성 비율(임계 예산) |
| `--alloc-mode` | auto | 양성 배분: `auto`(공간CV 자동 채택) / `regime_anchor` / … |
| `--posterior` | on | 베이지안 사후 + MC 전파 + per-regime 커버리지 |
| `--posterior-spatial` | poisson_gamma | 사후 백엔드: `poisson_gamma` / `bym` |
| `--submission-variants` | off | `PREVALENCE_GRID` π별 제출 변형 생성 |
| `--multiplier` | auto | 계층 EB 지역배율(누수안전 공간CV 기준) |

가중 재튜닝(선택):

```bash
.venv/bin/python scripts/tune_weights.py --within-regime --n-random 4000 --workers 60
```

---

## 5. 산출물

`outputs/submissions/`
- **`submission.csv`** — 제출 + 해석용 컬럼:
  `pole_id, lon, lat, decision(0/1), risk_score, regime, p_exposure, risk_lo, risk_hi, ops_priority, risk_pctile, risk_pctile_regime`
- **`submission_p{π}.csv`** — π별(0.005~0.10) 제출 변형.
- **`test_hanjeon.csv`** — 콘테스트 제출용 슬림형 `pole_id, lon, lat, decision`.

`outputs/`
- `regime_threshold_analysis.json` — 배분·F1 민감도·공간CV recall·고성 sanity·사후 커버리지.
- `tuned_weights*.json` · `ablation_*.json` · `equipment_cause_validation.json` — 튜닝 provenance·자산 피처 검증.
- `figures/` — `fig1_risk_map` ~ `fig11_uncertainty_map` + `eda/`.

제출 무결성은 `pfire/submit.py`가 강제 검증한다(행수 1,387,831 · `decision` 0/1만 · `pole_id` 정렬).

---

## 6. 예측 도구 — "데이터 넣으면 위험 예상도"

두 사용형 모두 물리 위험 $R = I \cdot S \cdot W$를 재사용하며 정적 성분은 고정하고
입력(기상·발화점)만 반영한다.

**① CLI**(`scripts/predict.py`) — 새 기상 폴더를 넣어 `risk.csv` + Leaflet/PNG 지도 산출:

```bash
.venv/bin/python scripts/predict.py --weather <기상폴더> --out outputs/predict
#   폴더에 fwi_station_daily.parquet[, aws_obs_daily.parquet]
#   --no-exposure-v2 로 빠른(≈12초) 미리보기
```

**② 웹앱**(`webapp/`) — 브라우저에서 발화점 CSV·고위험 비율을 입력하면 지도가 갱신된다.
표준 라이브러리만 사용. 사용법: [`webapp/README.md`](webapp/README.md):

```bash
.venv/bin/python webapp/server.py --port 8642   # → http://127.0.0.1:8642
```

> 입력 기상은 $W$의 일별 성분·풍하 노출에 반영되고, 시즌 극값 성분($w_s=0.82$)은 전주
> 피처의 기후값을 유지한다. 발화점은 위험 자체가 아니라 지역배율·배분 앵커로만 반영된다(§1.7).

---

## 7. 보고서

- [`report/최종공모안.md`](report/최종공모안.md) — 최종 공모안(공식 양식 1~5절).
- [`report/`](report/README.md) — 6쪽 논문형 보고서(`report.tex`)·수식 정리(`formulas.tex`).
- [`exposure_v2/README.md`](exposure_v2/README.md) — 풍하 노출 v2.2 재설계 근거·검증.
