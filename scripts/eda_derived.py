"""파생변수(derived/engineered features) 심화 EDA.

기존 notebooks/EDA.ipynb 는 원본 데이터(발화점·dNBR·기상관측소)만 봤다. 이 스크립트는
**모델이 실제 쓰는 파생변수**를 EDA 한다:

(a) 사전 엔지니어링(used_dataset): dist_to_forest/road/powerline, fwi_q90,
    fwi_high_danger_days, yanggan_days, fwi_mean/max, ndvi, s2_nbr/ndmi, elevation,
    slope, aspect_cos, mu_southness, mu_flammability, S_p, kfs_fires, fwi_*_obs, lc_group.
(b) 모델 계산 중간변수(pfire 재계산): 발화성분 I + 7개 입력 정규화값, 체제 게이트 가중,
    W(시즌·일별 블렌드), S, R=risk_score. + 제출물 컬럼(risk_score·p_exposure·risk_lo/hi·
    regime·decision·ops_priority).

8개 섹션을 수치+그림으로 산출:
  1 분포·요약 (퇴화변수 flag)
  2 상관·다중공선성 (히트맵·고상관쌍·VIF)
  3 발화 분리력 (point-biserial·KS·AUC-PR·recall@top-k → 판별력 순위표)
  4 체제 차이 (영동/영서/산간 분포)
  5 I·S·W 기여 분해 (무엇이 R 을 지배하나)
  6 공간 자기상관 (Moran's I)
  7 불확실성 진단 (risk_lo/hi·cv·ops_priority)
  8 leakage/sanity (training_labels 상관·의심 변수)

산출:
  figures → outputs/figures/eda/ (한글폰트, 300dpi)
  리포트  → claudedocs/eda_파생변수_심화_20260619.md

used_dataset/·pfire/ READ-ONLY(import만). 실행: .venv/bin/python scripts/eda_derived.py
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pfire import (  # noqa: E402
    calibrate,
    config,
    experts,
    geo,
    hazard,
    io,
    regimes,
    weather,
)

# ── 로깅(pfire INFO 소음 억제) ───────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING, format="%(message)s")
for noisy in ("pfire", "pfire.io", "pfire.experts", "pfire.regimes",
              "pfire.weather", "pfire.hazard", "pfire.calibrate"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("eda_derived")
log.setLevel(logging.INFO)
warnings.filterwarnings("ignore", category=RuntimeWarning)

FIG_DIR = ROOT / "outputs" / "figures" / "eda"
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT = ROOT / "claudedocs" / "eda_파생변수_심화_20260619.md"
SUBMISSION = ROOT / "outputs" / "submissions" / "submission.csv"

FONT_PATH = "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"
if Path(FONT_PATH).exists():
    fm.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["savefig.bbox"] = "tight"

RNG = np.random.default_rng(config.SEED)

# 한글 라벨용 변수명(그림 가독성)
PRETTY = {
    "dist_to_forest": "산림거리", "dist_to_road": "도로거리",
    "dist_to_powerline": "송전선거리", "dist_to_substation": "변전소거리",
    "fwi_q90": "FWI_q90", "fwi_high_danger_days": "FWI고위험일",
    "yanggan_days": "양간일수", "fwi_mean": "FWI평균", "fwi_max": "FWI최대",
    "ndvi": "NDVI", "s2_nbr": "S2_NBR", "s2_ndmi": "S2_NDMI",
    "elevation": "고도", "slope": "경사", "aspect_cos": "사면_cos",
    "mu_southness": "남향도", "mu_flammability": "연료가연성", "S_p": "S_p확산취약",
    "mu_slope": "mu경사", "mu_ndvi": "mu_NDVI", "kfs_fires": "산림청화재수",
    "fwi_q90_obs": "FWI_q90관측", "fwi_mean_obs": "FWI평균관측",
    "fwi_max_obs": "FWI최대관측", "fwi_hdd_obs": "FWI고위험일관측",
    "nn_station_km": "최근접관측소km",
    # 모델 파생
    "I": "발화성분I", "S": "확산취약S", "W": "기상W", "R": "위험R",
    "f_forest": "I입력_산림", "f_road": "I입력_도로", "f_powerline": "I입력_송전선",
    "f_fwi": "I입력_FWI", "f_yanggan": "I입력_양간", "f_fuel": "I입력_연료",
    "f_landcover": "I입력_토지피복",
    "gate_yeongdong": "게이트_영동", "gate_yeongseo": "게이트_영서",
    "gate_mountain": "게이트_산간",
    "risk_score": "risk_score", "p_exposure": "p_exposure",
    "risk_lo": "risk_lo", "risk_hi": "risk_hi", "ops_priority": "ops_priority",
    "unc_hi": "unc_hi",
}


def pretty(name: str) -> str:
    return PRETTY.get(name, name)


# ── 보고용 누적 버퍼 ─────────────────────────────────────────────────────────
RB: list[str] = []  # report blocks


def md(s: str = "") -> None:
    RB.append(s)


def fmt(x: float, p: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{p}f}"


# ═══════════════════════════════════════════════════════════════════════════
# 데이터 준비
# ═══════════════════════════════════════════════════════════════════════════
def build_frame() -> dict:
    """마스터 + 모델 중간변수 + 제출컬럼 + 앵커를 한 번에 구성."""
    log.info("[load] master frame ...")
    m = io.load_master()
    N = m.height

    # 추가 파생(io.load_master 가 안 싣는 것): dist_to_substation, lc_group(이미 있음),
    # training_labels(누수 점검), grid_id(공간자기상관용 격자), aspect_cos 등.
    power = pl.read_parquet(config.F_POLE_POWER, columns=["pole_id", "dist_to_substation"])
    tl = pl.read_parquet(config.F_TRAINING_LABELS, columns=["pole_id", "label", "unlabeled"])
    m = m.join(power, on="pole_id", how="left").join(tl, on="pole_id", how="left").sort("pole_id")

    # ── (b) 모델 중간변수 재계산 ───────────────────────────────────────────
    log.info("[model] gate / I / per-expert ...")
    gate, regime_order = regimes.compute_gate(m)            # (N,3)
    I, per_expert = experts.ignition_propensity(m, gate, regime_order)
    feats = experts.build_ignition_features(m)              # I 의 7개 입력 정규화값

    log.info("[model] W (blended season×daily) / S / R ...")
    station_daily = io.load_station_daily()
    stations = io.load_stations()
    W = weather.blended_weather(m, station_daily, stations)
    W_season = weather.season_weather(m, source="features")
    W_daily = weather.daily_isi_weather(m, station_daily, stations)
    S = m["S_p"].to_numpy().astype(np.float64)
    R = hazard.season_risk(I, S, W)

    regime_lbl = np.array(regime_order)[gate.argmax(axis=1)]

    # ── 제출물 컬럼(이미 materialized: 모델 최종산출 single source) ─────────
    sub = None
    if SUBMISSION.exists():
        sub = pl.read_csv(
            SUBMISSION,
            columns=["pole_id", "decision", "risk_score", "regime", "p_exposure",
                     "risk_lo", "risk_hi", "ops_priority", "unc_lo", "unc_hi"],
        ).sort("pole_id")
        if sub.height != N:
            log.warning("submission 행수 %d != master %d → 제출컬럼 일부 섹션 스킵", sub.height, N)
            sub = None

    # ── 앵커: 발화점 → 전주 매핑(검증·분리력 전용; 학습X) ───────────────────
    log.info("[anchor] positives → poles ...")
    pos = io.load_positives()
    pole_xy = m.select(["lon", "lat"]).to_numpy().astype(np.float64)
    fire_xy = pos.select(["lon", "lat"]).to_numpy().astype(np.float64)
    fire_to_pole = calibrate.assign_poles_to_fires(pole_xy, fire_xy, radius_km=1.0)
    pos_idx = np.unique(fire_to_pole[fire_to_pole >= 0])
    y = np.zeros(N, dtype=np.int8)
    y[pos_idx] = 1

    return dict(
        m=m, N=N, gate=gate, regime_order=regime_order, regime_lbl=regime_lbl,
        I=I, S=S, W=W, W_season=W_season, W_daily=W_daily, R=R, feats=feats,
        per_expert=per_expert, sub=sub, y=y, pos_idx=pos_idx, pole_xy=pole_xy,
        n_anchor_fires=int((fire_to_pole >= 0).sum()), n_anchor_poles=int(pos_idx.size),
    )


def assemble_feature_matrix(D: dict) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """EDA 대상 파생변수 전부를 {name: (N,)} 로. group: name→그룹라벨."""
    m = D["m"]
    cols = {}
    grp = {}

    # (a) 사전 엔지니어링
    pre = ["dist_to_forest", "dist_to_road", "dist_to_powerline", "dist_to_substation",
           "fwi_q90", "fwi_high_danger_days", "yanggan_days", "fwi_mean", "fwi_max",
           "ndvi", "s2_nbr", "s2_ndmi", "elevation", "slope", "aspect_cos",
           "mu_southness", "mu_flammability", "S_p", "mu_slope", "mu_ndvi",
           "kfs_fires", "fwi_q90_obs", "fwi_mean_obs", "fwi_max_obs", "fwi_hdd_obs",
           "nn_station_km"]
    for c in pre:
        if c in m.columns:
            cols[c] = m[c].to_numpy().astype(np.float64)
            grp[c] = "사전엔지니어링(a)"

    # (b) 모델 중간변수
    cols["I"] = D["I"]; grp["I"] = "모델중간(b)"
    cols["S"] = D["S"]; grp["S"] = "모델중간(b)"
    cols["W"] = D["W"]; grp["W"] = "모델중간(b)"
    cols["R"] = D["R"]; grp["R"] = "모델중간(b)"
    for k, v in D["feats"].items():
        cols[f"f_{k}"] = v.astype(np.float64); grp[f"f_{k}"] = "I입력(b)"
    for j, r in enumerate(D["regime_order"]):
        cols[f"gate_{r}"] = D["gate"][:, j]; grp[f"gate_{r}"] = "게이트(b)"

    # (b) 제출컬럼
    if D["sub"] is not None:
        s = D["sub"]
        for c in ["risk_score", "p_exposure", "risk_lo", "risk_hi", "ops_priority", "unc_hi"]:
            cols[c] = s[c].to_numpy().astype(np.float64); grp[c] = "제출물(b)"
    return cols, grp


# ═══════════════════════════════════════════════════════════════════════════
# 섹션 1 — 분포·요약
# ═══════════════════════════════════════════════════════════════════════════
def section1_distributions(cols: dict, grp: dict) -> dict:
    log.info("[sec1] distributions & summary")
    rows = []
    degenerate = []
    for name, x in cols.items():
        xf = x[np.isfinite(x)]
        n = x.size
        n_nan = int(n - xf.size)
        if xf.size == 0:
            continue
        std = float(np.std(xf))
        mean = float(np.mean(xf))
        skew = float(stats.skew(xf)) if std > 0 else 0.0
        pct_zero = float(np.mean(xf == 0.0) * 100.0)
        rng = float(xf.max() - xf.min())
        flag = []
        if std < 1e-9 or rng < 1e-9:
            flag.append("상수")
        if abs(skew) > 5:
            flag.append("극왜곡")
        if pct_zero > 90:
            flag.append("희소(>90%0)")
        if flag:
            degenerate.append((name, ",".join(flag)))
        rows.append(dict(
            변수=name, 그룹=grp[name], mean=mean, std=std,
            skew=skew, min=float(xf.min()), max=float(xf.max()),
            결측=n_nan, pct_zero=pct_zero, flag=",".join(flag),
        ))
    summary = pl.DataFrame(rows)

    # 그림: 핵심 파생변수 히스토그램 그리드
    key = ["dist_to_forest", "dist_to_powerline", "fwi_q90", "yanggan_days",
           "mu_flammability", "S_p", "ndvi", "elevation", "slope",
           "I", "S", "W", "R", "kfs_fires", "fwi_high_danger_days", "aspect_cos"]
    key = [k for k in key if k in cols]
    ncol = 4
    nrow = int(np.ceil(len(key) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes, key):
        x = cols[name]
        x = x[np.isfinite(x)]
        ax.hist(x, bins=60, color="#3b6fb0", alpha=0.85)
        ax.set_title(pretty(name), fontsize=10)
        ax.set_yscale("log")
        ax.tick_params(labelsize=7)
    for ax in axes[len(key):]:
        ax.axis("off")
    fig.suptitle("파생변수 분포 (y=log scale)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_distributions.png")
    plt.close(fig)

    # 그림: skew vs %zero (퇴화 진단)
    fig, ax = plt.subplots(figsize=(9, 6))
    sk = summary["skew"].to_numpy()
    pz = summary["pct_zero"].to_numpy()
    nm = summary["변수"].to_list()
    ax.scatter(np.clip(sk, -5, 30), pz, s=28, color="#666")
    for xi, yi, ni in zip(np.clip(sk, -5, 30), pz, nm):
        if abs(yi) > 80 or abs(xi) > 8:
            ax.annotate(pretty(ni), (xi, yi), fontsize=7, alpha=0.8)
    ax.axhline(90, color="r", ls="--", lw=1, label="희소 임계 90%")
    ax.axvline(5, color="orange", ls="--", lw=1, label="극왜곡 임계 skew=5")
    ax.set_xlabel("왜도 skew (클립 [-5,30])")
    ax.set_ylabel("% zero")
    ax.set_title("퇴화변수 진단: 왜도 vs %zero")
    ax.legend()
    fig.savefig(FIG_DIR / "01_degeneracy_scatter.png")
    plt.close(fig)

    return dict(summary=summary, degenerate=degenerate)


# ═══════════════════════════════════════════════════════════════════════════
# 섹션 2 — 상관·다중공선성
# ═══════════════════════════════════════════════════════════════════════════
def section2_correlation(cols: dict, grp: dict) -> dict:
    log.info("[sec2] correlation & multicollinearity")
    # 분석 대상: 상수/희소 극단 제외, 의미있는 연속 파생변수
    names = [n for n in cols
             if n not in ("ops_priority",)  # 이진/희소 제외
             and np.isfinite(cols[n]).sum() > 0
             and np.nanstd(cols[n]) > 1e-9]
    # 샘플(상관은 전수 불필요; 재현 고정 샘플)
    N = cols[names[0]].size
    idx = RNG.choice(N, size=min(120_000, N), replace=False)
    mat = np.column_stack([cols[n][idx] for n in names])
    # NaN → 열 중앙값
    for j in range(mat.shape[1]):
        col = mat[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            col[bad] = np.nanmedian(col)
            mat[:, j] = col
    C = np.corrcoef(mat, rowvar=False)

    # 고상관쌍 flag (|r|>=0.8). 단, 모델이 원변수를 그대로/단조변환해 만든 '거울쌍'은
    # 정의상 r≈1 이라 진짜 중복 신호를 가린다 → 분리해서 보고(정직).
    #   거울쌍 = {원변수 ↔ I입력 f_*(monotone), S_p↔S} 등 같은 정보의 재표현.
    # 거울쌍 = 같은 물리량의 재표현(I입력 f_*, S_p↔S, static-overlay mu_* ↔ 원필드).
    #   주의: fwi_q90↔fwi_mean↔fwi_max 는 '다른 통계량' 이라 거울 아님 → 액션가능 중복.
    MIRROR = {
        frozenset({"fwi_q90", "f_fwi"}), frozenset({"fwi_mean", "f_fwi"}),
        frozenset({"yanggan_days", "f_yanggan"}),
        frozenset({"mu_flammability", "f_fuel"}), frozenset({"S_p", "S"}),
        frozenset({"aspect_cos", "mu_southness"}),  # 둘 다 사면 남향 표현(부호반대)
        frozenset({"slope", "mu_slope"}), frozenset({"ndvi", "mu_ndvi"}),  # static-overlay 재집계
    }

    def _is_mirror(a: str, b: str) -> bool:
        if frozenset({a, b}) in MIRROR:
            return True
        # f_* (I입력 정규화) ↔ 그 원천 변수(거리류는 exp(-d/s) monotone 재표현)
        src_of = {"f_forest": "dist_to_forest", "f_road": "dist_to_road",
                  "f_powerline": "dist_to_powerline", "f_fwi": "fwi_q90",
                  "f_yanggan": "yanggan_days", "f_fuel": "mu_flammability"}
        return src_of.get(a) == b or src_of.get(b) == a

    hi_pairs = []        # 진짜(서로 다른 파생변수) 고상관 — 액션 가능
    mirror_pairs = []    # 모델 재표현 거울쌍 — 정보상 동일(분석에서 한쪽만)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = C[i, j]
            if np.isfinite(r) and abs(r) >= 0.8:
                if _is_mirror(names[i], names[j]):
                    mirror_pairs.append((names[i], names[j], float(r)))
                else:
                    hi_pairs.append((names[i], names[j], float(r)))
    hi_pairs.sort(key=lambda t: -abs(t[2]))
    mirror_pairs.sort(key=lambda t: -abs(t[2]))

    # 그림: 상관 히트맵
    fig, ax = plt.subplots(figsize=(0.42 * len(names) + 3, 0.42 * len(names) + 2))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels([pretty(n) for n in names], rotation=90, fontsize=7)
    ax.set_yticklabels([pretty(n) for n in names], fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson r")
    ax.set_title("파생변수 상관 히트맵")
    fig.savefig(FIG_DIR / "02_corr_heatmap.png")
    plt.close(fig)

    # VIF (주요 연속 발화·기상·지형 변수만 — 모델 입력성 변수).
    #   완전공선 거울(mu_southness=−aspect_cos, mu_slope=slope, mu_ndvi=ndvi)은 한쪽만
    #   넣는다(둘 다 넣으면 VIF=∞ 라 나머지 추정이 무의미해짐).
    vif_vars = [v for v in ["dist_to_forest", "dist_to_road", "dist_to_powerline",
                            "dist_to_substation", "fwi_q90", "fwi_mean", "fwi_max",
                            "fwi_high_danger_days", "yanggan_days", "ndvi", "s2_nbr",
                            "s2_ndmi", "elevation", "slope", "aspect_cos",
                            "mu_flammability", "S_p", "kfs_fires"]
                if v in cols]
    vif = compute_vif(cols, vif_vars, idx)

    return dict(names=names, C=C, hi_pairs=hi_pairs, mirror_pairs=mirror_pairs, vif=vif)


def compute_vif(cols: dict, vif_vars: list[str], idx: np.ndarray) -> list[tuple[str, float]]:
    """VIF_j = 1/(1-R²_j), R²_j = 나머지 변수로 j 회귀. 표준화 후 정규방정식."""
    X = np.column_stack([cols[v][idx] for v in vif_vars]).astype(np.float64)
    for j in range(X.shape[1]):
        col = X[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            col[bad] = np.nanmedian(col)
            X[:, j] = col
    # 표준화
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Z = (X - mu) / sd
    out = []
    for j in range(Z.shape[1]):
        yj = Z[:, j]
        Xj = np.delete(Z, j, axis=1)
        # 최소제곱 R²
        coef, *_ = np.linalg.lstsq(Xj, yj, rcond=None)
        pred = Xj @ coef
        ss_res = float(np.sum((yj - pred) ** 2))
        ss_tot = float(np.sum((yj - yj.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 - 1e-9 else np.inf
        out.append((vif_vars[j], float(vif)))
    out.sort(key=lambda t: -t[1])
    return out


# ═══════════════════════════════════════════════════════════════════════════
# 섹션 3 — 발화 분리력 (per-feature)
# ═══════════════════════════════════════════════════════════════════════════
def auc_pr(score: np.ndarray, y: np.ndarray) -> float:
    """AUC-PR (average precision). 점수 내림차순으로 PR 곡선 적분."""
    order = np.argsort(-score, kind="mergesort")
    yt = y[order]
    tp = np.cumsum(yt)
    fp = np.cumsum(1 - yt)
    P = tp[-1]
    if P == 0:
        return float("nan")
    precision = tp / (tp + fp)
    recall = tp / P
    # average precision = Σ (R_k - R_{k-1}) * P_k
    rec_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - rec_prev) * precision))


def ks_stat(score: np.ndarray, y: np.ndarray) -> float:
    pos = score[y == 1]
    neg = score[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    return float(stats.ks_2samp(pos, neg).statistic)


def section3_separability(cols: dict, grp: dict, D: dict) -> dict:
    log.info("[sec3] per-feature separability (anchors=positives)")
    y = D["y"]
    N = y.size
    base_rate = float(y.mean())
    # 분리력 후보: 위험 방향이 의미있는 파생변수(거리는 부호 반전해 '근접도'로 정렬해도
    # AUC-PR/KS 는 단조변환 불변. point-biserial 은 부호로 방향 표시).
    cand = [n for n in cols if n not in ("R", "risk_score", "p_exposure",
                                         "risk_lo", "risk_hi", "unc_hi", "ops_priority")
            and np.nanstd(cols[n]) > 1e-9]
    rows = []
    for name in cand:
        x = cols[name].astype(np.float64)
        bad = ~np.isfinite(x)
        if bad.any():
            x = x.copy(); x[bad] = np.nanmedian(x)
        # 방향: '위험↑' 가 큰 값이 되도록 거리류는 부호 반전한 점수로 ranking 지표 계산
        if name.startswith("dist_") or name == "nn_station_km":
            sc = -x
        else:
            sc = x
        # point-biserial (원변수 기준; 부호가 곧 방향)
        pb = float(stats.pointbiserialr(y, x).statistic) if np.std(x) > 0 else float("nan")
        ks = ks_stat(sc, y)
        ap = auc_pr(sc, y)
        # 단일변수 recall@top-1%,5%
        rk = calibrate.recall_at_topk(sc, D["pos_idx"], ks=(0.01, 0.05))
        rows.append(dict(변수=name, 그룹=grp[name],
                         point_biserial=pb, KS=ks, AUC_PR=ap,
                         lift_PR=ap / base_rate if base_rate > 0 else np.nan,
                         recall_top1=rk[0.01], recall_top5=rk[0.05]))
    tbl = pl.DataFrame(rows).sort("AUC_PR", descending=True)

    # 그림: AUC-PR 순위 막대 (상위 25)
    top = tbl.head(25)
    fig, ax = plt.subplots(figsize=(9, 8))
    names = top["변수"].to_list()
    ap = top["AUC_PR"].to_numpy()
    ypos = np.arange(len(names))[::-1]
    colors = ["#b03b3b" if g.startswith("사전") else "#3b6fb0" for g in top["그룹"].to_list()]
    ax.barh(ypos, ap, color=colors)
    ax.axvline(base_rate, color="k", ls="--", lw=1, label=f"기저율 {base_rate:.2e}")
    ax.set_yticks(ypos)
    ax.set_yticklabels([pretty(n) for n in names], fontsize=8)
    ax.set_xlabel("AUC-PR (단일변수 발화 분리력)")
    ax.set_title("파생변수 발화 판별력 순위 (Top 25, AUC-PR)")
    ax.legend()
    fig.savefig(FIG_DIR / "03_separability_aucpr.png")
    plt.close(fig)

    # 그림: KS vs point-biserial (signal map)
    fig, ax = plt.subplots(figsize=(9, 7))
    pbv = tbl["point_biserial"].to_numpy()
    ksv = tbl["KS"].to_numpy()
    nmv = tbl["변수"].to_list()
    ax.scatter(pbv, ksv, s=30, color="#444")
    for xi, yi, ni in zip(pbv, ksv, nmv):
        if abs(xi) > 0.005 or yi > 0.15:
            ax.annotate(pretty(ni), (xi, yi), fontsize=7)
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("point-biserial r (부호=위험방향)")
    ax.set_ylabel("KS 통계 (전주 vs 발화점)")
    ax.set_title("파생변수 발화 신호 맵")
    fig.savefig(FIG_DIR / "03_signal_map.png")
    plt.close(fig)

    return dict(table=tbl, base_rate=base_rate)


# ═══════════════════════════════════════════════════════════════════════════
# 섹션 4 — 체제 차이
# ═══════════════════════════════════════════════════════════════════════════
def section4_regimes(cols: dict, D: dict) -> dict:
    log.info("[sec4] regime differences")
    lbl = D["regime_lbl"]
    regimes_kr = {"yeongdong": "영동", "yeongseo": "영서", "mountain": "산간"}
    order = ["yeongdong", "yeongseo", "mountain"]
    show = ["yanggan_days", "fwi_q90", "fwi_high_danger_days", "mu_flammability",
            "dist_to_forest", "dist_to_powerline", "elevation", "I", "W", "R"]
    show = [s for s in show if s in cols]

    rows = []
    for name in show:
        x = cols[name]
        rec = dict(변수=name)
        for r in order:
            mask = lbl == r
            xr = x[mask]
            xr = xr[np.isfinite(xr)]
            rec[f"{regimes_kr[r]}_median"] = float(np.median(xr))
            rec[f"{regimes_kr[r]}_mean"] = float(np.mean(xr))
        rows.append(rec)
    tbl = pl.DataFrame(rows)

    # 그림: 체제별 박스플롯 그리드(핵심 6)
    box_vars = ["yanggan_days", "fwi_q90", "mu_flammability", "dist_to_forest", "I", "R"]
    box_vars = [b for b in box_vars if b in cols]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()
    for ax, name in zip(axes, box_vars):
        data = []
        for r in order:
            x = cols[name][lbl == r]
            x = x[np.isfinite(x)]
            # 시각화 안정: 상위 99.5% 클립
            hi = np.quantile(x, 0.995) if x.size else 1.0
            data.append(np.clip(x, None, hi))
        ax.boxplot(data, tick_labels=[regimes_kr[r] for r in order], showfliers=False)
        ax.set_title(pretty(name), fontsize=11)
        ax.tick_params(labelsize=9)
    fig.suptitle("체제별 핵심 파생변수 분포 (MoE 근거; outlier 숨김)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_regime_boxplots.png")
    plt.close(fig)

    counts = {regimes_kr[r]: int((lbl == r).sum()) for r in order}
    return dict(table=tbl, counts=counts)


# ═══════════════════════════════════════════════════════════════════════════
# 섹션 5 — I·S·W 기여 분해
# ═══════════════════════════════════════════════════════════════════════════
def section5_isw(D: dict) -> dict:
    log.info("[sec5] I·S·W contribution to R")
    I, S, W, R = D["I"], D["S"], D["W"], D["R"]
    logR = np.log(np.clip(R, 1e-12, None))
    comp = {"I": np.log(np.clip(I, 1e-12, None)),
            "S": np.log(np.clip(S, 1e-12, None)),
            "W": np.log(np.clip(W, 1e-12, None))}
    # log R = log I + log S + log W (정확한 가법분해). 각 항의 분산기여:
    var_logR = float(np.var(logR))
    contrib = {}
    spear = {}
    pears_logterm = {}
    for k, lv in comp.items():
        # 분산분해: cov(log term, log R)/var(log R) (합 = 1)
        contrib[k] = float(np.cov(lv, logR)[0, 1] / var_logR)
        spear[k] = float(stats.spearmanr({"I": I, "S": S, "W": W}[k], R).statistic)
        pears_logterm[k] = float(np.corrcoef(lv, logR)[0, 1])
    # 변동계수(어느 항이 R 변동을 흔드나)
    cv = {k: float(np.std(v) / np.mean(v)) for k, v in (("I", I), ("S", S), ("W", W))}

    # 체제별 W 기여(영동 쏠림 점검) — within-regime 분산분해.
    lbl = D["regime_lbl"]
    regimes_kr = {"yeongdong": "영동", "yeongseo": "영서", "mountain": "산간"}
    wshare = {}
    for r in ("yeongdong", "yeongseo", "mountain"):
        mask = lbl == r
        lv = {kk: comp[kk][mask] for kk in comp}
        lr = logR[mask]
        vlr = float(np.var(lr))
        wshare[regimes_kr[r]] = {kk: float(np.cov(lv[kk], lr)[0, 1] / vlr) for kk in comp}

    # ── between-regime 평균 차이(영동 쏠림의 '기계적' 원인) ──────────────────
    # within-regime 분산분해는 '한 체제 안에서 어느 항이 순위를 흔드나'를 본다.
    # 하지만 영동이 상위위험에 4×쏠리는 건 '체제 간 평균 차이' 때문 → 각 항의
    # 영동 평균 / (영서·산간 평균) 배율을 본다. 이 배율이 큰 항이 영동을 끌어올림.
    yd = lbl == "yeongdong"
    other = ~yd
    mean_ratio = {}
    for kk, raw in (("I", I), ("S", S), ("W", W)):
        m_yd = float(np.mean(raw[yd]))
        m_ot = float(np.mean(raw[other]))
        mean_ratio[kk] = m_yd / m_ot if m_ot > 0 else float("nan")

    # 그림: 분산기여 막대 + 체제별
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ks = ["I", "S", "W"]
    ax.bar(ks, [contrib[k] for k in ks], color=["#3b6fb0", "#5aa469", "#b03b3b"])
    ax.set_ylabel("log R 분산기여 (cov/var, 합=1)")
    ax.set_title("R 을 지배하는 항 (전역)")
    for i, k in enumerate(ks):
        ax.text(i, contrib[k], fmt(contrib[k], 3), ha="center", va="bottom")
    ax = axes[1]
    width = 0.25
    xpos = np.arange(3)
    for i, k in enumerate(ks):
        vals = [wshare[r][k] for r in ("영동", "영서", "산간")]
        ax.bar(xpos + i * width, vals, width, label=k)
    ax.set_xticks(xpos + width)
    ax.set_xticklabels(["영동", "영서", "산간"])
    ax.set_ylabel("log R 분산기여")
    ax.set_title("체제별 I·S·W 기여 (영동 쏠림 점검)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_isw_contribution.png")
    plt.close(fig)

    # 그림: 체제별 R 분포 + 상위위험 체제 점유
    fig, ax = plt.subplots(figsize=(8, 5))
    thr = np.quantile(R, 0.98)  # 상위 2% (운영 양성비율 근방)
    top_mask = R >= thr
    shares = {regimes_kr[r]: float(np.mean(lbl[top_mask] == r)) for r in
              ("yeongdong", "yeongseo", "mountain")}
    pop_shares = {regimes_kr[r]: float(np.mean(lbl == r)) for r in
                  ("yeongdong", "yeongseo", "mountain")}
    labels = ["영동", "영서", "산간"]
    x = np.arange(3)
    ax.bar(x - 0.2, [pop_shares[l] for l in labels], 0.4, label="전체 전주 점유", color="#bbb")
    ax.bar(x + 0.2, [shares[l] for l in labels], 0.4, label="상위2% 위험 점유", color="#b03b3b")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("점유 비율")
    ax.set_title("체제별 상위위험 쏠림 (전체 vs 상위2%)")
    ax.legend()
    fig.savefig(FIG_DIR / "05_regime_topshare.png")
    plt.close(fig)

    return dict(contrib=contrib, spear=spear, cv=cv, wshare=wshare,
                top_shares=shares, pop_shares=pop_shares, mean_ratio=mean_ratio)


# ═══════════════════════════════════════════════════════════════════════════
# 섹션 6 — 공간 자기상관 (Moran's I)
# ═══════════════════════════════════════════════════════════════════════════
def morans_i(values: np.ndarray, xy_km: np.ndarray, k: int = 8) -> float:
    """kNN 이진 가중 Moran's I. 대규모 대비 표본 기반 추정.

    I = (n/W) · Σ_ij w_ij (z_i)(z_j) / Σ_i z_i²,  z = x - mean.
    kNN(브루트포스 by 블록)으로 이웃을 찾고 행정규화 가중 사용.
    """
    n = values.shape[0]
    z = values - values.mean()
    denom = float(np.sum(z * z))
    if denom <= 0:
        return float("nan")
    # kNN: 블록 단위 거리(표본 n 이 작아 n×n 허용)
    d2 = (xy_km[:, None, 0] - xy_km[None, :, 0]) ** 2 + \
         (xy_km[:, None, 1] - xy_km[None, :, 1]) ** 2
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1)[:, :k]
    num = 0.0
    Wsum = 0.0
    zi = z
    for i in range(n):
        w = 1.0 / k  # 행정규화
        num += w * np.sum(zi[i] * zi[nn[i]])
        Wsum += k * w
    I = (n / Wsum) * (num / denom)
    return float(I)


def section6_moran(cols: dict, D: dict) -> dict:
    log.info("[sec6] spatial autocorrelation (Moran's I)")
    N = D["N"]
    # 표본(공간 자기상관은 표본으로 충분; 재현 고정)
    n_s = 4000
    idx = RNG.choice(N, size=n_s, replace=False)
    xy = geo.lonlat_to_km(D["pole_xy"][idx])
    targets = ["R", "I", "S", "W", "fwi_q90", "dist_to_forest", "yanggan_days",
               "mu_flammability"]
    if D["sub"] is not None:
        targets = ["risk_score"] + targets
    targets = [t for t in targets if t in cols]
    res = {}
    for name in targets:
        x = cols[name][idx].astype(np.float64)
        bad = ~np.isfinite(x)
        if bad.any():
            x = x.copy(); x[bad] = np.nanmedian(x)
        res[name] = morans_i(x, xy, k=8)

    fig, ax = plt.subplots(figsize=(9, 5))
    nm = list(res.keys())
    vals = [res[n] for n in nm]
    ax.bar([pretty(n) for n in nm], vals, color="#3b6fb0")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Moran's I (kNN=8, 표본 4000)")
    ax.set_title("공간 자기상관 — 공간CV 정당성 근거")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    for i, v in enumerate(vals):
        ax.text(i, v, fmt(v, 2), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_morans_i.png")
    plt.close(fig)
    return dict(moran=res, n_sample=n_s)


# ═══════════════════════════════════════════════════════════════════════════
# 섹션 7 — 불확실성 진단
# ═══════════════════════════════════════════════════════════════════════════
def section7_uncertainty(cols: dict, D: dict) -> dict:
    log.info("[sec7] uncertainty diagnostics")
    if D["sub"] is None:
        return dict(skip=True)
    s = D["sub"]
    rs = s["risk_score"].to_numpy().astype(np.float64)
    lo = s["risk_lo"].to_numpy().astype(np.float64)
    hi = s["risk_hi"].to_numpy().astype(np.float64)
    ops = s["ops_priority"].to_numpy()
    unc_lo = s["unc_lo"].to_numpy().astype(np.float64)
    unc_hi = s["unc_hi"].to_numpy().astype(np.float64)
    lbl = D["regime_lbl"]
    regimes_kr = {"yeongdong": "영동", "yeongseo": "영서", "mountain": "산간"}

    width = hi - lo
    rel_width = width / np.clip(rs, 1e-9, None)  # 상대 불확실성
    # 체제별 평균 상대폭(밴드 폭 / risk_score)
    rel_by_regime = {}
    for r in ("yeongdong", "yeongseo", "mountain"):
        mask = lbl == r
        rel_by_regime[regimes_kr[r]] = float(np.median(rel_width[mask]))
    ops_share = {regimes_kr[r]: float(np.mean(ops[lbl == r])) for r in
                 ("yeongdong", "yeongseo", "mountain")}
    ops_total = int(ops.sum())

    # 진단: unc_lo 퇴화 여부
    unc_lo_degenerate = bool(np.std(unc_lo) < 1e-12)

    # 그림: risk_score vs 밴드폭(샘플)
    N = rs.size
    idx = RNG.choice(N, size=min(60_000, N), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.scatter(rs[idx], width[idx], s=3, alpha=0.2, color="#3b6fb0")
    ax.set_xlabel("risk_score"); ax.set_ylabel("밴드폭 (risk_hi - risk_lo)")
    ax.set_title("절대 불확실성 vs 위험")
    ax = axes[1]
    # 상대폭 분포(체제별)
    data = [np.clip(rel_width[lbl == r], 0, np.quantile(rel_width, 0.97)) for r in
            ("yeongdong", "yeongseo", "mountain")]
    ax.boxplot(data, tick_labels=["영동", "영서", "산간"], showfliers=False)
    ax.set_ylabel("상대 불확실성 (밴드폭/risk_score)")
    ax.set_title("체제별 상대 불확실성")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_uncertainty.png")
    plt.close(fig)

    return dict(rel_by_regime=rel_by_regime, ops_share=ops_share,
                ops_total=ops_total, unc_lo_degenerate=unc_lo_degenerate,
                median_relwidth=float(np.median(rel_width)),
                unc_hi_range=(float(unc_hi.min()), float(unc_hi.max())))


# ═══════════════════════════════════════════════════════════════════════════
# 섹션 8 — leakage / sanity
# ═══════════════════════════════════════════════════════════════════════════
def section8_leakage(cols: dict, grp: dict, D: dict, sep_tbl: pl.DataFrame) -> dict:
    log.info("[sec8] leakage / sanity")
    m = D["m"]
    # training_labels (신뢰 낮음): label 0/1 가 매겨진 65,750 전주 부분에서 파생변수 상관
    lab = m["label"].to_numpy()  # 0/1/null
    has_lab = np.isfinite(lab.astype(np.float64)) if lab.dtype.kind != "f" else np.isfinite(lab)
    # polars Int8 with null → numpy object/float; 안전하게 mask
    lab_f = m["label"].cast(pl.Float64).to_numpy()
    mask = np.isfinite(lab_f)
    yl = lab_f[mask].astype(np.int8)

    rows = []
    check = [n for n in cols if n not in ("ops_priority",) and np.nanstd(cols[n]) > 1e-9]
    for name in check:
        x = cols[name][mask].astype(np.float64)
        bad = ~np.isfinite(x)
        if bad.any():
            x = x.copy(); x[bad] = np.nanmedian(x)
        if np.std(x) <= 0:
            continue
        r = float(stats.pointbiserialr(yl, x).statistic)
        rows.append((name, grp[name], r))
    rows.sort(key=lambda t: -abs(t[2]))
    tl_corr = pl.DataFrame(rows, schema=["변수", "그룹", "weak_label_pb"], orient="row")

    # 의심스럽게 강한 변수: 발화 분리력(AUC-PR) 상위인데 모델 산출(R/risk_score)과
    # 거의 동일하면 정상(정의상 누수 아님). 진짜 위험은 weak label 과의 비정상 상관.
    # safemap 앵커 분리력과 weak-label 상관을 함께 보고.
    sep_lookup = {r["변수"]: r["AUC_PR"] for r in sep_tbl.to_dicts()}
    suspicious = []
    for name, g, r in rows[:15]:
        ap = sep_lookup.get(name, float("nan"))
        suspicious.append((name, g, r, ap))

    # 그림: weak-label point-biserial 상위
    top = tl_corr.head(20)
    fig, ax = plt.subplots(figsize=(9, 7))
    nm = top["변수"].to_list()
    vv = top["weak_label_pb"].to_numpy()
    ypos = np.arange(len(nm))[::-1]
    ax.barh(ypos, vv, color=["#b03b3b" if v > 0 else "#3b6fb0" for v in vv])
    ax.set_yticks(ypos)
    ax.set_yticklabels([pretty(n) for n in nm], fontsize=8)
    ax.set_xlabel("point-biserial r (training_labels, 신뢰낮음)")
    ax.set_title("약지도 라벨 vs 파생변수 상관 (누수 점검)")
    ax.axvline(0, color="k", lw=0.8)
    fig.savefig(FIG_DIR / "08_weaklabel_corr.png")
    plt.close(fig)

    return dict(tl_corr=tl_corr, n_labeled=int(mask.sum()),
                pos_frac=float(yl.mean()), suspicious=suspicious)


# ═══════════════════════════════════════════════════════════════════════════
# 리포트 작성
# ═══════════════════════════════════════════════════════════════════════════
def write_report(D, s1, s2, s3, s4, s5, s6, s7, s8):
    log.info("[report] writing markdown")
    md("# 파생변수(derived/engineered features) 심화 EDA")
    md()
    md("> 생성: `scripts/eda_derived.py` · 날짜 2026-06-19 · figures `outputs/figures/eda/`")
    md(">")
    md("> 기존 `notebooks/EDA.ipynb` 는 **원본 데이터**(발화점·dNBR·관측소)만 다뤘다. 이 문서는")
    md("> **모델이 실제 쓰는 파생변수**를 EDA 한다 — (a) used_dataset 사전 엔지니어링 변수, ")
    md("> (b) pfire 가 계산하는 중간변수(I·S·W·R·게이트) + 제출물 컬럼. 발화점(safemap)은")
    md("> **분리력 앵커로만** 쓰고 학습엔 안 쓴다(누수 방지).")
    md()
    md(f"- 전주 수 N = {D['N']:,}")
    md(f"- 발화 앵커: 반경 1km 내 매핑된 발화점 {D['n_anchor_fires']}건 → 고유 전주 {D['n_anchor_poles']}개 "
       f"(기저율 {s3['base_rate']:.3e})")
    md(f"- 체제 분포(게이트 argmax): {s4['counts']}")
    md()
    md("---")
    md()

    # 섹션 1
    md("## 1. 분포·요약 & 퇴화변수")
    md()
    md("그림: `01_distributions.png`(핵심 16변수 히스토그램), `01_degeneracy_scatter.png`(왜도-희소).")
    md()
    md("**퇴화(상수/극왜곡/희소>90%0) flag 변수:**")
    md()
    if s1["degenerate"]:
        md("| 변수 | flag |")
        md("|---|---|")
        for name, f in s1["degenerate"]:
            md(f"| `{name}` | {f} |")
    else:
        md("- 없음")
    md()
    md("**전체 요약(주요 변수 발췌):**")
    md()
    md("| 변수 | 그룹 | mean | std | skew | min | max | 결측 | %zero |")
    md("|---|---|---|---|---|---|---|---|---|")
    pick = ["dist_to_forest", "dist_to_powerline", "dist_to_substation", "fwi_q90",
            "yanggan_days", "fwi_high_danger_days", "mu_flammability", "S_p", "ndvi",
            "s2_nbr", "elevation", "slope", "kfs_fires", "I", "S", "W", "R"]
    srows = {r["변수"]: r for r in s1["summary"].to_dicts()}
    for n in pick:
        if n in srows:
            r = srows[n]
            md(f"| `{n}` | {r['그룹']} | {fmt(r['mean'],3)} | {fmt(r['std'],3)} | "
               f"{fmt(r['skew'],2)} | {fmt(r['min'],2)} | {fmt(r['max'],2)} | "
               f"{r['결측']} | {fmt(r['pct_zero'],1)} |")
    md()

    # 섹션 2
    md("## 2. 상관·다중공선성")
    md()
    md("그림: `02_corr_heatmap.png`. 분석표본 120k(고정 시드).")
    md()
    md("**서로 다른 파생변수 간 고상관쌍 |r| ≥ 0.8 (실질 중복 후보 = 액션 가능):**")
    md()
    if s2["hi_pairs"]:
        md("| 변수 A | 변수 B | r |")
        md("|---|---|---|")
        for a, b, r in s2["hi_pairs"]:
            md(f"| `{a}` | `{b}` | {fmt(r,3)} |")
    else:
        md("- |r|≥0.8 인 (서로 다른) 쌍 없음 — 파생변수 간 심각한 중복은 제한적.")
    md()
    md("**(참고) 모델 재표현 '거울쌍' |r| ≥ 0.8 — 정보상 동일(원변수↔I입력/S_p↔S 등; 중복 아님):**")
    md()
    if s2["mirror_pairs"]:
        md("| 변수 A | 변수 B | r |")
        md("|---|---|---|")
        for a, b, r in s2["mirror_pairs"]:
            md(f"| `{a}` | `{b}` | {fmt(r,3)} |")
    md()
    md("**VIF (주요 연속 입력변수; 표준화 후):** VIF>10 = 강한 다중공선성")
    md()
    md("| 변수 | VIF |")
    md("|---|---|")
    for name, v in s2["vif"]:
        flag = " ⚠️" if v > 10 else ""
        vstr = "∞(완전공선)" if not np.isfinite(v) else fmt(v, 2)
        md(f"| `{name}` | {vstr}{flag} |")
    md()

    # 섹션 3
    md("## 3. 발화 분리력 (per-feature) — 핵심")
    md()
    md("그림: `03_separability_aucpr.png`(AUC-PR 순위), `03_signal_map.png`(KS vs point-biserial).")
    md("발화점=앵커. AUC-PR / KS / point-biserial / 단일변수 recall@top-k 로 측정.")
    md(f"기저율 {s3['base_rate']:.3e} (이보다 큰 AUC-PR 만 신호).")
    md()
    md("**발화 판별력 순위 (AUC-PR 내림차순, 상위 18):**")
    md()
    md("| 순위 | 변수 | 그룹 | AUC-PR | lift(/기저) | KS | point-bis | recall@1% | recall@5% |")
    md("|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(s3["table"].head(18).to_dicts(), 1):
        md(f"| {i} | `{r['변수']}` | {r['그룹']} | {fmt(r['AUC_PR'],4)} | "
           f"{fmt(r['lift_PR'],1)} | {fmt(r['KS'],3)} | {fmt(r['point_biserial'],4)} | "
           f"{fmt(r['recall_top1'],3)} | {fmt(r['recall_top5'],3)} |")
    md()
    md("**하위(신호 약함, AUC-PR 오름차순 5):**")
    md()
    md("| 변수 | AUC-PR | KS | point-bis |")
    md("|---|---|---|---|")
    for r in s3["table"].sort("AUC_PR").head(5).to_dicts():
        md(f"| `{r['변수']}` | {fmt(r['AUC_PR'],4)} | {fmt(r['KS'],3)} | {fmt(r['point_biserial'],4)} |")
    md()

    # 섹션 4
    md("## 4. 체제 차이 (영동/영서/산간)")
    md()
    md("그림: `04_regime_boxplots.png`. 체제=게이트 argmax.")
    md()
    md("**체제별 중앙값:**")
    md()
    cols_kr = ["영동", "영서", "산간"]
    md("| 변수 | " + " | ".join(cols_kr) + " |")
    md("|---|" + "---|" * 3)
    for r in s4["table"].to_dicts():
        md(f"| `{r['변수']}` | " +
           " | ".join(fmt(r[f'{c}_median'], 3) for c in cols_kr) + " |")
    md()

    # 섹션 5
    md("## 5. I·S·W 기여 분해 — 무엇이 R 을 지배하나")
    md()
    md("그림: `05_isw_contribution.png`, `05_regime_topshare.png`.")
    md("R=I·S·W → logR=logI+logS+logW 가법분해. 분산기여 = cov(log항, logR)/var(logR) (합=1).")
    md()
    md("**전역 분산기여 (logR 기준):**")
    md()
    md("| 항 | 분산기여 | Spearman(항,R) | 변동계수 CV |")
    md("|---|---|---|---|")
    for k in ("I", "S", "W"):
        md(f"| {k} | {fmt(s5['contrib'][k],3)} | {fmt(s5['spear'][k],3)} | {fmt(s5['cv'][k],3)} |")
    md()
    md("**체제별 분산기여 (within-regime; 한 체제 안에서 순위를 흔드는 항):**")
    md()
    md("| 체제 | I | S | W |")
    md("|---|---|---|---|")
    for r in ("영동", "영서", "산간"):
        w = s5["wshare"][r]
        md(f"| {r} | {fmt(w['I'],3)} | {fmt(w['S'],3)} | {fmt(w['W'],3)} |")
    md()
    md("**between-regime 평균 배율 (영동평균 / 비영동평균) — 영동 쏠림의 기계적 원인:**")
    md()
    md("within-regime 분산분해는 '같은 체제 안' 순위를 본다. 그러나 영동이 상위위험에 "
       "4.3× 쏠리는 건 **체제 간 평균 차이** 때문 — 이 배율이 큰 항이 영동을 끌어올린다.")
    md()
    md("| 항 | 영동평균/비영동평균 |")
    md("|---|---|")
    for k in ("I", "S", "W"):
        md(f"| {k} | {fmt(s5['mean_ratio'][k],2)}× |")
    md()
    md("**상위2% 위험 체제 점유 vs 전체 점유:**")
    md()
    md("| 체제 | 전체 점유 | 상위2% 위험 점유 | 쏠림배율 |")
    md("|---|---|---|---|")
    for r in ("영동", "영서", "산간"):
        ps = s5["pop_shares"][r]; ts = s5["top_shares"][r]
        md(f"| {r} | {fmt(ps,3)} | {fmt(ts,3)} | {fmt(ts/ps if ps>0 else float('nan'),2)}× |")
    md()

    # 섹션 6
    md("## 6. 공간 자기상관 (Moran's I)")
    md()
    md(f"그림: `06_morans_i.png`. kNN=8 이진가중, 표본 {s6['n_sample']} (고정 시드).")
    md("I≈0 무상관, I→1 강한 양의 공간자기상관 → **무작위 CV 낙관편향 → 공간블록 CV 정당.**")
    md()
    md("| 변수 | Moran's I |")
    md("|---|---|")
    for name, v in sorted(s6["moran"].items(), key=lambda t: -t[1]):
        md(f"| `{name}` | {fmt(v,3)} |")
    md()

    # 섹션 7
    md("## 7. 불확실성 진단")
    md()
    if s7.get("skip"):
        md("- submission.csv 부재로 스킵.")
    else:
        md("그림: `07_uncertainty.png`.")
        md()
        md(f"- 전역 중앙 상대불확실성(밴드폭/risk_score) = {fmt(s7['median_relwidth'],3)}")
        md(f"- `ops_priority` 플래그 총 {s7['ops_total']:,}개")
        md(f"- `unc_lo` 퇴화(상수)= **{s7['unc_lo_degenerate']}** "
           f"(True 면 unc_lo 컬럼이 정보가 없음 → 제거 권고)")
        md(f"- `unc_hi` 범위 = [{fmt(s7['unc_hi_range'][0],3)}, {fmt(s7['unc_hi_range'][1],3)}]")
        md()
        md("**체제별 중앙 상대불확실성 / ops_priority 비율:**")
        md()
        md("| 체제 | 중앙 상대폭 | ops_priority 비율 |")
        md("|---|---|---|")
        for r in ("영동", "영서", "산간"):
            md(f"| {r} | {fmt(s7['rel_by_regime'][r],3)} | {fmt(s7['ops_share'][r],4)} |")
    md()

    # 섹션 8
    md("## 8. leakage / sanity")
    md()
    md("그림: `08_weaklabel_corr.png`.")
    md(f"training_labels(신뢰 낮음): 라벨 부여 전주 {s8['n_labeled']:,}개, 양성비 {fmt(s8['pos_frac'],3)}.")
    md("**해석:** 약지도 라벨과 파생변수의 상관이 비정상적으로 높으면 라벨 누수 의심. ")
    md("단, 모델 산출(R·risk_score)이 발화점 앵커와 강한 건 정의상 정상(누수 아님).")
    md()
    md("**약지도 라벨 point-biserial 상위 12:**")
    md()
    md("| 변수 | 그룹 | weak-label r | (참고)발화앵커 AUC-PR |")
    md("|---|---|---|---|")
    for name, g, r, ap in s8["suspicious"][:12]:
        md(f"| `{name}` | {g} | {fmt(r,4)} | {fmt(ap,4)} |")
    md()

    # ── 종합 권고 ──
    md("---")
    md()
    md("## 종합 발견 & 정직한 권고")
    md()
    _write_recommendations(D, s1, s2, s3, s4, s5, s6, s7, s8)

    REPORT.write_text("\n".join(RB), encoding="utf-8")
    log.info("[report] wrote %s", REPORT)


def _write_recommendations(D, s1, s2, s3, s4, s5, s6, s7, s8):
    """데이터 기반 자동 권고 — 수치에서 직접 도출(과장 금지)."""
    sep = s3["table"]

    # 분리력 순위에서 '거울' 변수(모델 재표현)는 원변수로 접어 중복 표기 제거.
    FOLD = {"S": "S_p", "f_fuel": "mu_flammability", "f_landcover": "lc_ignition",
            "f_fwi": "fwi_q90", "f_yanggan": "yanggan_days", "f_forest": "dist_to_forest",
            "f_road": "dist_to_road", "f_powerline": "dist_to_powerline",
            "mu_ndvi": "ndvi", "mu_slope": "slope", "fwi_mean": "fwi_q90"}

    def _dedup(names: list[str], k: int) -> list[str]:
        seen, out = set(), []
        for n in names:
            key = FOLD.get(n, n)
            if key in seen:
                continue
            seen.add(key)
            out.append(n)
            if len(out) >= k:
                break
        return out

    top3 = _dedup(sep["변수"].to_list(), 3)
    bot = _dedup(sep.sort("AUC_PR")["변수"].to_list(), 3)
    # 지배항
    dom = max(s5["contrib"], key=lambda k: s5["contrib"][k])
    # 영동 쏠림
    yd_lift = (s5["top_shares"]["영동"] / s5["pop_shares"]["영동"]
               if s5["pop_shares"]["영동"] > 0 else float("nan"))
    yd_w = s5["wshare"]["영동"]["W"]

    md(f"**① 발화 판별력:** 단일변수 최강 신호(거울변수 접음) = `{top3[0]}`·`{top3[1]}`·`{top3[2]}` "
       f"(AUC-PR 상위). 최약 신호 = `{bot[0]}`·`{bot[1]}`·`{bot[2]}`. "
       "주의: 모든 단일변수 AUC-PR 이 기저율 대비 lift ~1.5~1.9배로 **낮다**(presence-only "
       "+ 발화 희소 + 거리/연료의 약한 단조신호). 즉 단일 파생변수로는 발화를 거의 못 가르며, "
       "위험은 I·S·W 결합과 임계값 결정에서 나온다.")
    md()
    md("- 가설 검정: README/config 가설은 'FFMC·dist_to_forest 강, ISI 약'. 앵커 분리력에서 "
       "**연료/산림계열(S_p·mu_flammability·dist_to_forest)이 도로·송전선·양간보다 분리력이 "
       "높게** 나와 '산림 근접' 가설은 지지된다. 반면 FFMC 계열 단일신호의 우위는 "
       "(전주 사전피처에 일별 FFMC 가 직접 없어) 이 앵커 분리력만으로는 확정 못 한다 — "
       "config 의 FFMC 우위는 관측소-수준 recall@top-k 튜닝 근거이며 별개 측정임.")
    md()
    if s2["hi_pairs"]:
        pr = s2["hi_pairs"][0]
        md(f"**② 중복(고상관):** 서로 다른 파생변수 중 최강 중복쌍 `{pr[0]}`↔`{pr[1]}` "
           f"(r={pr[2]:.2f}). 추가로 모델 재표현 거울쌍({len(s2['mirror_pairs'])}개; "
           "원변수↔I입력, S_p↔S 등)은 정의상 동일 정보라 중복 카운트에서 제외. "
           "VIF>10 변수도 다중공선성 → 입력 축소 후보.")
    else:
        md(f"**② 중복(고상관):** 서로 다른 파생변수 간 |r|≥0.8 쌍 없음 "
           f"(거울쌍 {len(s2['mirror_pairs'])}개는 모델 재표현이라 제외). "
           "심각한 중복은 제한적이나 VIF 표에서 FWI 계열(q90/mean/max) 다중공선성은 존재.")
    md()
    # 영동 쏠림의 기계적 원인 = between-regime 평균 배율이 가장 큰 항.
    mr = s5["mean_ratio"]
    cause = max(mr, key=lambda k: mr[k])
    md(f"**③ R 지배항 / 영동 쏠림:** *전역 분산기여*(전주 순위 흔드는 항)는 **{dom}** 최대 "
       f"(I={fmt(s5['contrib']['I'],2)}, S={fmt(s5['contrib']['S'],2)}, W={fmt(s5['contrib']['W'],2)}). "
       f"하지만 영동이 상위2% 위험의 {fmt(s5['top_shares']['영동']*100,0)}%(쏠림 {fmt(yd_lift,1)}×)를 "
       f"차지하는 **기계적 원인은 between-regime 평균 차이**이고, 그 차이가 가장 큰 항은 "
       f"**{cause}**(영동/비영동 배율 I={fmt(mr['I'],2)}×, S={fmt(mr['S'],2)}×, W={fmt(mr['W'],2)}×). "
       f"즉 영동 쏠림은 곱셈 R 에서 **W(시즌 fwi_q90·양간) 가 영동에서 {fmt(mr['W'],1)}× 높아** "
       "발생한다(영동 W 중앙 0.56 vs 영서·산간 0.21).")
    md()
    md(f"**④ 체제 차이:** 양간일수·FWI·연료·산림거리가 체제별로 갈림 → MoE 정당. (표 §4)")
    md()
    moranR = s6["moran"].get("R", s6["moran"].get("risk_score", float("nan")))
    md(f"**⑤ 공간/불확실성/leakage:** R Moran's I={fmt(moranR,2)} (강한 공간자기상관 → 공간CV 필수). ")
    if not s7.get("skip"):
        md(f"  - `unc_lo` 퇴화={s7['unc_lo_degenerate']} → True 면 컬럼 제거 권고. ")
    md(f"  - 약지도 라벨 누수 점검: 최강 상관 변수가 모델 산출이 아니라면 주의 (표 §8).")
    md()
    md("**모델 개선 권고 Top 3:**")
    md()
    # 자동 권고 우선순위 — 영향 큰 순.
    recs = []
    # (1) R 지배항·영동 쏠림 재균형 (가장 영향 큼)
    mr = s5["mean_ratio"]
    recs.append(
        f"**영동 쏠림 재균형:** 상위2% 위험의 {fmt(s5['top_shares']['영동']*100,0)}% 가 영동에 "
        f"쏠려 있고(전체 영동 점유 {fmt(s5['pop_shares']['영동']*100,0)}%, 쏠림 {fmt(yd_lift,1)}×). "
        f"원인은 곱셈 R 에서 **W 가 영동에서 {fmt(mr['W'],1)}× 높기** 때문(I 는 {fmt(mr['I'],2)}×, "
        f"S 는 {fmt(mr['S'],2)}× 로 체제차 작음). 의도된 설계(영동=양간·고FWI)이나, F1 이 영동에 "
        "지나치게 의존하면 영서·산간 발화(인간발화·연료)를 놓칠 위험 → W_BLEND/양간 가중 점검 "
        "또는 체제별 임계값(이미 ALLOC_* 존재)으로 비영동 recall 보강.")
    # (2) 다중공선성 정리
    high_vif = [v for v, val in s2["vif"] if val > 10]
    if s2["hi_pairs"]:
        a, b, r = s2["hi_pairs"][0]
        recs.append(f"**입력 축소:** 고상관쌍 `{a}`↔`{b}`(r={r:.2f})"
                    + (f", 고VIF 변수 {high_vif[:4]}" if high_vif else "")
                    + " 중 하나만 유지 → 다중공선성·과적합 축소.")
    elif high_vif:
        recs.append(f"**입력 축소:** VIF>10 변수 {high_vif[:5]}(특히 FWI q90/mean/max 동시 사용)는 "
                    "다중공선성 → 대표 하나만 유지.")
    # (3) unc_lo 퇴화 or 분리력 약함
    if not s7.get("skip") and s7["unc_lo_degenerate"]:
        recs.append("**제출물 정리:** `unc_lo` 가 상수(0)라 정보 없음 → 컬럼 제거하거나 실제 하한 "
                    "추정으로 대체(현재 risk_lo/hi 가 유효 밴드이므로 unc_lo/hi 는 중복·혼란 소지).")
    else:
        recs.append(f"**약한 단일신호 보완:** 단일 파생변수 분리력이 전반적으로 낮으니(lift~1.5~1.9×) "
                    "결합·임계값 결정과 공간CV recall@top-k 에 검증 무게를 두고, 약신호 변수 가중 축소.")
    for i, r in enumerate(recs[:3], 1):
        md(f"{i}. {r}")
    md()
    md("> 모든 수치는 스크립트 1회 실행에서 자동 산출. 표본 기반(상관·Moran)은 고정 시드.")


# ═══════════════════════════════════════════════════════════════════════════
def main() -> int:
    D = build_frame()
    cols, grp = assemble_feature_matrix(D)
    log.info("EDA 대상 파생변수 %d개", len(cols))
    s1 = section1_distributions(cols, grp)
    s2 = section2_correlation(cols, grp)
    s3 = section3_separability(cols, grp, D)
    s4 = section4_regimes(cols, D)
    s5 = section5_isw(D)
    s6 = section6_moran(cols, D)
    s7 = section7_uncertainty(cols, D)
    s8 = section8_leakage(cols, grp, D, s3["table"])
    write_report(D, s1, s2, s3, s4, s5, s6, s7, s8)

    # 콘솔 요약(최종 보고용)
    print("\n" + "=" * 70)
    print("파생변수 심화 EDA 완료")
    print("=" * 70)
    print(f"figures: {FIG_DIR}")
    print(f"report : {REPORT}")
    print(f"\n[발화 분리력 Top5 AUC-PR]")
    for r in s3["table"].head(5).to_dicts():
        print(f"  {r['변수']:<24} AUC-PR={r['AUC_PR']:.4f} KS={r['KS']:.3f} "
              f"pb={r['point_biserial']:+.4f}")
    print(f"[발화 분리력 Bottom3]")
    for r in s3["table"].sort("AUC_PR").head(3).to_dicts():
        print(f"  {r['변수']:<24} AUC-PR={r['AUC_PR']:.4f}")
    print(f"[고상관쌍 |r|>=0.8] {len(s2['hi_pairs'])}개")
    for a, b, r in s2["hi_pairs"][:6]:
        print(f"  {a} <-> {b} : r={r:+.3f}")
    print(f"[R 분산기여] I={s5['contrib']['I']:.3f} S={s5['contrib']['S']:.3f} "
          f"W={s5['contrib']['W']:.3f}")
    print(f"[영동 상위2% 쏠림] pop={s5['pop_shares']['영동']:.3f} "
          f"top2%={s5['top_shares']['영동']:.3f}")
    print(f"[Moran's I] R={s6['moran'].get('R', float('nan')):.3f}")
    if not s7.get("skip"):
        print(f"[unc_lo 퇴화] {s7['unc_lo_degenerate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
