# MOSAIC — Wildfire Risk Analysis for Power Distribution Poles

[![Contest](https://img.shields.io/badge/2026_Weather_Big_Data_Contest-Topic_1_Disaster_Safety-1f6feb)](https://bd.kma.go.kr)
[![Award](https://img.shields.io/badge/🏆_3rd_Prize-KEPCO_President's_Award-FFD700)](https://www.kepco.co.kr)

English | **[한국어](README.ko.md)**

> **🏆 3rd Prize — KEPCO President's Award (한국전력사장상)**
> **2026 Weather Big Data Contest · Topic 1 (Disaster Safety)**
> Hosted by the **Korea Meteorological Administration (KMA)** · Partner institution **Korea Electric Power Corporation (KEPCO)**

Classify each of **1,387,831 virtual power poles** in Gangwon Province as wildfire-risky or safe (**0/1**).
Ground-truth labels are withheld (a presence-only / PU problem), so historical ignition points are used
**only as anchors for validation and threshold selection — never as training targets**. Risk itself is
estimated in an unsupervised-first manner from **wildfire physics combined with a regional
mixture-of-experts (MoE)**.

**MOSAIC** = **M**ixture-of-experts · **S**patial (block CV · BYM2) · **A**nchored (ignition anchors) ·
**I**gnition-physics ($R = I \cdot S \cdot W$) · **C**overage (per-regime conformal). The model's identity
is treating Gangwon as a *mosaic* of four fire regimes — East Coast, Foehn Corridor, Inland West, and
Mountain — each handled by its own expert.

---

## 1. Model — $R = I \cdot S \cdot W$

The daily risk of pole $p$ on day $t$ during the fire-watch season (Feb–May) is a product of three terms:

$$
h(p,t) = \underbrace{I(p)}_{\text{ignition propensity}} \times \underbrace{S(p)}_{\text{spread/exposure susceptibility}} \times \underbrace{W(g(p),t)}_{\text{weather that day}}
$$

**Why a product:** if any factor is zero there is no risk (no spark, nothing to burn, or benign
weather ⇒ no fire) — a physically sound AND semantics that also makes each factor separately
interpretable. All components are normalized to $[0,1]$ and range-validated (`pfire/hazard.py`).

**Season aggregation.** Over $T$ season days the daily path uses a geometric mean (log-sum average),
which prevents a single extreme day from dominating the season:

$$
R(p) = \exp\left(\frac{1}{T}\sum_{t=1}^{T} \log\big[\mathrm{clip}(I \cdot S \cdot W_t,\ \varepsilon,\ 1)\big]\right)
$$

### 1.1 $I$ — Ignition propensity: 4-regime mixture of experts

Ignition mechanisms differ across Gangwon (`config.REGIMES`), so four experts are used:

| Regime | Area | Ignition drivers |
|---|---|---|
| `yeongdong` East Coast | Gangneung · Donghae · Samcheok | drought · FWI |
| `corridor` Foehn Corridor | Yangyang · Goseong · Sokcho | Yangan foehn wind · transmission lines |
| `yeongseo` Inland West | inland cities | human ignition (burning · hikers) |
| `mountain` Highlands | high-elevation terrain | fuel · road access |

**Feature normalization** (`pfire/experts.py`) — distance features decay exponentially
(closer ⇒ riskier):

$$
\mathrm{feat}_{\text{dist}}(p) = \exp\!\big(-d(p)/\text{scale}\big),\qquad
\text{scale} = 100\,\text{m (forest)},\ 300\,\text{m (road)},\ 1500\,\text{m (powerline)}
$$

FWI and foehn-day counts are min–max normalized; fuel is $\mathrm{clip}(\mu_{\text{flam}},0,1)$;
land cover uses a domain prior $\mathrm{lc\_ignition}\in[0,1]$ (forest 1.00 … water 0.00).

**Experts** — each regime $r$ combines the same seven features with different weights
(`config.EXPERT_WEIGHTS`, tuned by LHS + coordinate ascent on spatial-block-CV recall);
the weighted mean keeps the output in $[0,1]$:

$$
E_r(p) = \frac{\sum_k w_{r,k}\,\mathrm{feat}_k(p)}{\sum_k w_{r,k}},\qquad
k \in \{\text{forest, road, powerline, fwi, yanggan, fuel, landcover}\}
$$

**Soft gate** (`pfire/regimes.py`) — gate features $z_p = (\text{longitude, elevation, foehn days})$
are z-standardized and softmaxed by distance to regime anchors $\mu_r$:

$$
g_r(p) = \frac{\exp\!\big(-\lVert z_p-\mu_r\rVert^2/\tau\big)}{\sum_{r'}\exp\!\big(-\lVert z_p-\mu_{r'}\rVert^2/\tau\big)},
\qquad \sum_r g_r(p)=1,\quad \tau=0.6
$$

Boundary poles blend naturally between regimes, avoiding the MAUP artifacts of hard partitions.

$$
\boxed{\ I(p) = \sum_r g_r(p)\,E_r(p)\ } \in [0,1]
$$

### 1.2 $W$ — Weather

Two components blended linearly (`pfire/weather.py`):

$$
W_{\text{season}}(p) = 0.5\,\mathrm{fwi\_q90} + 0.3\,\mathrm{hdd} + 0.2\,\mathrm{yanggan}
$$

$$
W_{\text{daily}}(p) = 0.05\,(\text{ISI comp.}) + 0.75\,(\text{FFMC comp.}) + 0.20\,(\text{foehn comp.})
$$

where the ISI component is $0.5\,(\text{high-ISI day ratio}) + 0.25\,\overline{\mathrm{ISI}} + 0.25\,\mathrm{ISI}_{q90}$
(thresholds: ISI $\ge 10$, FFMC $\ge 90$), aggregated per station over the fire season and assigned to
poles by nearest station. FFMC (ignition readiness) dominates because the anchors are mostly
human-caused ignitions.

$$
\boxed{\ W(p) = w_s\,W_{\text{season}}(p) + (1-w_s)\,W_{\text{daily}}(p)\ },\qquad w_s = 0.82
$$

The season term preserves pole-specific extremes (e.g. the 2019 Goseong fire area) that pure
station-mapped daily statistics would dilute.

### 1.3 $S$ — Spread/exposure susceptibility

Static susceptibility $S_{\text{static}}$ is blended with downwind exposure **conditionally per pole**
— only East Coast / high-foehn poles get a strong blend, preventing global dilution
(`pfire/hazard.py`):

$$
\alpha_p = \alpha_{\text{base}} \cdot g_{\text{yeongdong}}(p) \cdot \widehat{\mathrm{yanggan}}(p) \in [0, \alpha_{\text{base}}]
$$

$$
S(p) = (1-\alpha_p)\,S_{\text{static}}(p) + \alpha_p\,\widetilde{P}_{\text{expo}}(p)
$$

where $\widetilde{P}_{\text{expo}}$ is the rank-percentile of exposure probability.

### 1.4 Downwind exposure — from OR saturation to a discriminative dose

**v1 Monte Carlo OR kernel** (`pfire/exposure.py`, `pfire/exposure_engine.py`, Rust backend
`rust/pfire_kernels`): sample winds at ignition candidates (top 2% of $I$), spread anisotropically:

$$
L = L_0\big(1+\alpha\,\max(0,\mathrm{align})\,\tfrac{ws}{5}\big),\qquad
\Pr(\text{reach}) = e^{-d/L}\cdot \mathrm{fuel}\cdot(1+\beta\,\mathrm{southness})
$$

$$
P_{\text{expo}}(p) = \frac{1}{S}\sum_s \mathbf{1}\big[\text{any source reaches } p\big]
$$

**Problem:** with dense sources on the East Coast the OR probability saturates at $P\approx 1$,
destroying within-region discrimination.

**v2.2 ignition-weighted expected dose** (`pfire/exposure_v2.py`) replaces OR with a non-saturating
sum, an Anderson length-to-width-ratio (LWR) ellipse, and grid-local normalization:

$$
U = 2.237\,ws\ \text{(mph)},\qquad
\mathrm{LWR}(ws) = \min\!\big(0.936\,e^{0.2566U} + 0.461\,e^{-0.1548U} - 0.397,\ 8\big)
$$

$$
r = \sqrt{(d_\parallel/L_\parallel)^2 + (d_\perp/W_\perp)^2},\qquad
\mathrm{dose}(p) = \sum_g I_g\,\big\langle e^{-r}\big\rangle_{\text{wind}}\,\mathrm{fuel}_p\,(1+\beta\,s_p)
$$

with downwind $L_\parallel = \mathrm{LWR}\cdot W_\perp$ and upwind $L_\parallel = L_{\text{back}}$.
Local normalization removes the "everything on the East Coast is high" background:

$$
\widetilde{e}(p) = \frac{\mathrm{dose}(p)}{\overline{\mathrm{dose}}_{\mathrm{cell}(p)}+\varepsilon},\qquad
\mathrm{dose01}(p) = \mathrm{percentile}\big(\widetilde{e}(p)\big)/100
$$

The dose enters the decision through a probabilistic OR that preserves ignition ranking:

$$
\boxed{\ R'(p) = 1-\big(1-R(p)\big)\big(1-w\cdot\mathrm{dose01}(p)\big)\ },\qquad w = 0.25
$$

Validated: East-Coast discrimination IQR $0.166 \to 0.290$; spatial-CV recall $+0.0071$ (consistent
across 5 fold seeds). Full evidence: [`exposure_v2/README.md`](exposure_v2/README.md).

### 1.5 Regional-rate correction — hierarchical empirical Bayes

Ignition anchors are sparse (~900 points; some districts have zero). Raw regional rates would overfit,
so Poisson–Gamma EB shrinkage pulls each region toward its parent (global → regime → district → grid,
top-down; `pfire/hierarchy.py`). With parent mean $m$ and sibling variance $v$, moment-matching gives
$\alpha = m^2/v,\ \beta = m/v$ and the posterior-mean rate

$$
\tilde\lambda_g = \frac{y_g+\alpha}{n_g+\beta} = w_g\,\hat\lambda_g + (1-w_g)\,m,
\qquad w_g = \frac{n_g}{n_g+\beta}
$$

Large samples keep their observed rate ($w_g \to 1$); zero-fire small regions shrink to the parent
($w_g \to 0$). Pole multipliers are grid EB rates relative to the global EB rate (mean ≈ 1).

### 1.6 Uncertainty — Bayesian posterior + Monte Carlo propagation

`pfire/posterior.py` provides two backends. Neither affects the 0/1 decision; uncertainty feeds
operational prioritization only.

**Poisson–Gamma conjugate (default, closed-form):**

$$
\lambda_g \mid y_g \sim \mathrm{Gamma}(\alpha_0+y_g,\ \beta_0+n_g)
$$

Zero-event regions automatically get wide posteriors. Each posterior draw multiplies the physical
base risk, $R^{(r)}(p) = \mathrm{clip}(R_{\text{base}}(p)\cdot m^{(r)}(p), 0, 1)$, yielding per-pole
**90% credible intervals** `risk_lo`/`risk_hi` ($q_{0.05}, q_{0.95}$).

**BYM2 spatial CAR (`--posterior-spatial bym`, INLA-style Laplace approximation)** — borrows strength
from neighboring grid cells:

$$
y_i \sim \mathrm{Poisson}(E_i e^{x_i}),\qquad
Q_x(\tau,\phi) = \tau\big[(1-\phi)I + \phi\,Q_{\text{ICAR}}\big]
$$

For each hyperparameter pair the negative log-posterior
$f(x)=\sum_i(E_i e^{x_i}-y_i x_i)+\tfrac12 x^\top Q_x x$ is minimized by sparse Newton; the pair
maximizing the Laplace marginal likelihood

$$
\log p(y\mid\tau,\phi) \approx \sum_i\big(y_i x_i^\star - E_i e^{x_i^\star}\big)
+\tfrac12\log|Q_x| - \tfrac12 x^{\star\top}Q_x x^\star - \tfrac12\log|H|,
\qquad H = \mathrm{diag}(E e^{x^\star}) + Q_x
$$

is selected, giving Gaussian grid posteriors $x_i \sim \mathcal N(x_i^\star, (H^{-1})_{ii})$.

**Per-regime conformal coverage** — distribution-free finite-sample guarantee. Sorting calibration
scores per regime and setting

$$
\tau_r = \mathrm{score}_{(\lfloor \alpha (n_r+1)\rfloor)}
\quad\Rightarrow\quad
\Pr\big(\text{new ignition risk} \ge \tau_r\big) \gtrsim 1-\alpha,\qquad \alpha = 0.10
$$

Measured coverage at nominal 90%: overall 0.902 (East Coast 0.907 · Inland 0.902 · Mountain 0.897).

### 1.7 Decision — regime-anchor budget allocation

A single global cut would let the high-scoring East Coast sweep all positives. Instead the positive
budget (default prevalence 2%) is allocated across regimes proportionally to ignition-anchor density,
then cut by within-regime risk rank (`pfire/calibrate.py`). Because the true positive ratio $\rho$ is
unknown, an F1 sensitivity curve is reported over assumptions:

$$
\mathrm{prec}_{\text{proxy}} = \frac{\rho\cdot\mathrm{recall}\cdot N}{n_{\text{pred}}},\qquad
F_1 = \frac{2\,\mathrm{prec}_{\text{proxy}}\,\mathrm{recall}}{\mathrm{prec}_{\text{proxy}}+\mathrm{recall}},
\qquad \rho \in \{0.5,1,2,3,5,10\}\%
$$

### 1.8 Training — within-regime objective

Weight tuning (`scripts/tune_weights.py --within-regime`) optimizes the equal-weighted mean of
per-regime within recall rather than global top-k recall, removing the between-regime base-rate
shortcut. All evaluation uses **spatial block CV (10 km)** with the random-vs-spatial gap reported as
an optimism check. Everything is deterministic under `config.SEED`.

---

## 2. Repository layout

```
OR/
├── README.md                  # this file
├── pfire/                     # core pipeline package
│   ├── config.py              # single source of truth: paths, constants, regimes, weights
│   ├── io.py                  # polars master-frame loader (pole_id join)
│   ├── regimes.py             # 4-regime soft gate (MoE)
│   ├── experts.py             # per-regime physics experts → I(p)
│   ├── exposure.py            # downwind exposure kernel wrapper (Rust + numpy fallback)
│   ├── exposure_engine.py     # exposure orchestration (candidates, wind sampling, planar km)
│   ├── exposure_v2.py         # discriminative dose v2.2 (ellipse + local normalization)
│   ├── weather.py             # W (season extremes × daily ISI/FFMC blend)
│   ├── hazard.py              # R = I·S·W combination, season aggregation
│   ├── hierarchy.py           # hierarchical EB (partial pooling) regional multipliers
│   ├── posterior.py           # Bayesian posterior (Poisson–Gamma / BYM2) + MC propagation
│   ├── calibrate.py           # thresholds, regime-anchor allocation, conformal
│   ├── validate.py            # spatial block CV, recall, sanity checks
│   ├── fire_cause.py          # fire-cause classification (equipment-cause anchor validation)
│   ├── ablation.py            # asset-aware vs asset-blind LOGO ablation
│   ├── risk_index.py          # risk percentile index
│   ├── geo.py                 # lon/lat ↔ planar km
│   └── submit.py              # submission CSV writer + integrity checks
├── scripts/                   # entry points (§4)
│   ├── run_phase1_mvp.py      # main end-to-end: R → posterior → submission
│   ├── run_phase3.py          # land-cover + conditional exposure staged validation
│   ├── tune_weights.py        # EXPERT_WEIGHTS tuning (--within-regime)
│   ├── predict.py             # deployment CLI: new weather in → risk map out
│   ├── make_figures.py        # report figures fig1–fig11
│   ├── figures/               # report auxiliary figures (run from repo root)
│   └── eda_derived.py · risk_percentile.py · run_ablation*.py · validate_equipment_cause.py · …
├── used_dataset/              # input data snapshot (see its README)
├── outputs/                   # artifacts (§5)
├── exposure_v2/               # exposure v2.2 design docs + flagship figure
├── rust/pfire_kernels/        # downwind spread Rust kernel (rayon parallel)
├── webapp/                    # local risk-map web app (stdlib only)
├── report/                    # final contest report + LaTeX papers
├── notebooks/                 # EDA notebooks
└── tests/                     # pytest (122 tests)
```

---

## 3. Data

Snapshot spec: [`used_dataset/README.md`](used_dataset/README.md). Highlights:

| Kind | Location | Contents |
|---|---|---|
| Poles | `used_dataset/poles/` | 1,387,831 virtual poles: coordinates + terrain, fuel, FWI, powerline/substation distance, land cover (parquet) |
| Weather | `used_dataset/weather/` | daily AWS observations (wind), daily FWI/ISI/FFMC, 109 station coordinates |
| Ignitions | `used_dataset/fire/` | 928 safemap ignition points (validation/threshold anchors, presence-only) |
| Admin / burn scar | `used_dataset/admin/`, `used_dataset/burn/` | Gangwon boundaries, 2019 Goseong burn scar (dNBR) |

CRS is EPSG:4326; distances use a planar approximation at Gangwon's representative latitude. All
paths and constants live in `pfire/config.py` — no hardcoding elsewhere.

---

## 4. Reproduction

Python ≥ 3.12 with the project virtualenv (`.venv`):

```bash
# 1) end-to-end risk → posterior uncertainty → submission (main)
.venv/bin/python scripts/run_phase1_mvp.py \
    --prevalence 0.02 \
    --posterior-spatial bym \      # BYM2 spatial CAR posterior (default: poisson_gamma)
    --submission-variants          # per-π submission_p{π}.csv variants

# 2) land-cover + conditional exposure staged validation
.venv/bin/python scripts/run_phase3.py --prevalence 0.02

# 3) report figures fig1–fig11
.venv/bin/python scripts/make_figures.py
```

Key flags of `run_phase1_mvp.py`:

| Flag | Default | Meaning |
|---|---|---|
| `--prevalence` | 0.02 | operating positive ratio (threshold budget) |
| `--alloc-mode` | auto | positive allocation: `auto` (spatial-CV-selected) / `regime_anchor` / … |
| `--posterior` | on | Bayesian posterior + MC propagation + per-regime coverage |
| `--posterior-spatial` | poisson_gamma | posterior backend: `poisson_gamma` / `bym` |
| `--submission-variants` | off | per-π submission variants over `PREVALENCE_GRID` |
| `--multiplier` | auto | hierarchical EB regional multiplier (leak-safe spatial CV) |

Optional weight re-tuning:

```bash
.venv/bin/python scripts/tune_weights.py --within-regime --n-random 4000 --workers 60
```

---

## 5. Artifacts

`outputs/submissions/`
- **`submission.csv`** — submission + interpretation columns:
  `pole_id, lon, lat, decision(0/1), risk_score, regime, p_exposure, risk_lo, risk_hi, ops_priority, risk_pctile, risk_pctile_regime`
- **`submission_p{π}.csv`** — per-prevalence variants (0.005–0.10).
- **`test_hanjeon.csv`** — slim contest submission `pole_id, lon, lat, decision`.

`outputs/`
- `regime_threshold_analysis.json` — allocation, F1 sensitivity, spatial-CV recall, Goseong sanity, posterior coverage.
- `tuned_weights*.json`, `ablation_*.json`, `equipment_cause_validation.json` — tuning provenance and asset-feature validation.
- `figures/` — `fig1_risk_map` … `fig11_uncertainty_map` + `eda/`.

Submission integrity is enforced by `pfire/submit.py` (row count 1,387,831 · decision ∈ {0,1} ·
sorted `pole_id`).

---

## 6. Prediction tools — "weather in, risk map out"

Both wrappers reuse the physical risk $R = I \cdot S \cdot W$ with static factors fixed; only the
inputs (weather, ignition points) vary.

**① CLI** (`scripts/predict.py`) — point it at a weather folder to get `risk.csv` + Leaflet/PNG maps:

```bash
.venv/bin/python scripts/predict.py --weather <weather-dir> --out outputs/predict
#   weather-dir needs fwi_station_daily.parquet [, aws_obs_daily.parquet]
#   --no-exposure-v2 for a fast (~12 s) preview
```

**② Web app** (`webapp/`) — upload an ignition CSV and set a high-risk ratio in the browser; the map
updates live. Standard library only. See [`webapp/README.md`](webapp/README.md):

```bash
.venv/bin/python webapp/server.py --port 8642   # → http://127.0.0.1:8642
```

> Input weather feeds the daily component of $W$ and downwind exposure; the season-extreme component
> ($w_s = 0.82$) keeps climatological pole features. Ignition points act as regional-multiplier and
> allocation anchors — never as direct risk labels (§1.7).

---

## 7. Reports

- [`report/최종공모안.md`](report/최종공모안.md) — final contest submission (official 5-section form, Korean).
- [`report/`](report/README.md) — 6-page paper (`report.tex`) and formula reference (`formulas.tex`).
- [`exposure_v2/README.md`](exposure_v2/README.md) — exposure v2.2 redesign rationale and validation.
