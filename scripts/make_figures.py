#!/usr/bin/env python
"""정성평가용 시각자료 생성 — outputs/figures/*.png.

실행:
    .venv/bin/python scripts/make_figures.py

산출물(9장):
    fig1_risk_map.png       강원 위험지도 (전주 risk_score + 행정경계 + 2019 발화점)
    fig2_regime_alloc.png   체제 지도 + 전역컷 vs 체제별컷 양성분포 편중완화
    fig3_recall_topk.png    recall@top-k 곡선 (전역 vs 체제별 + 랜덤기준선)
    fig4_f1_sensitivity.png F1 민감도 곡선 (π별, 전역 vs 체제별, 권장 운영점)
    fig5_goseong_2019.png   2019 고성 케이스 (burn-scar + 전주 + 발화점 + 풍하화살표)
    fig6_ignition_decomp.png 한 고위험 전주의 I·S·W·regime 기여 분해
    fig7_sgg_risk_subplots.png       16개 시군 서브플롯 위험지도(risk_mean)  [Phase-5]
    fig8_sgg_uncertainty_subplots.png 16개 시군 사후 불확실성(risk_hi−risk_lo)  [Phase-5]
    fig9_coverage.png       체제별 명목 vs 실측 커버리지(conformal·credible)  [Phase-5]
    fig10_bym_vs_pg.png     독립 Poisson-Gamma vs BYM2 공간평활 시군 위험(이웃 borrow)  [Phase-5b]

파이프라인 코드는 읽기만(import 가능, 수정 금지). decision/json/데이터에서 계산.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

# 프로젝트 루트를 import 경로에 추가 (pfire 파이프라인 import — 읽기만, 수정 금지)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Polygon as MplPolygon

ROOT = Path(__file__).resolve().parents[1]
OUT_FIG = ROOT / "outputs" / "figures"
SUBMISSION = ROOT / "outputs" / "submissions" / "submission.csv"
REGIME_JSON = ROOT / "outputs" / "regime_threshold_analysis.json"
TUNED_JSON = ROOT / "outputs" / "tuned_weights.json"
ADMIN_CSV = ROOT / "used_dataset" / "admin" / "admin_gangwon.csv"
POSITIVES = ROOT / "used_dataset" / "fire" / "safemap_positives.parquet"
BURN_GEOJSON = ROOT / "used_dataset" / "burn" / "burn_goseong_2019_dnbr.geojson"
POLE_SGG = ROOT / "used_dataset" / "poles" / "pole_sgg.parquet"

# 시군 → 체제(색·정렬용). config.SGG_TO_REGIME 와 동일(읽기만; 하드코딩 대신 import).
import pfire.config as _cfg  # noqa: E402
SGG_TO_REGIME = _cfg.SGG_TO_REGIME
# zero-event(발화 0~소수) 시군 — 사후 폭이 큰 현장확인 1순위(기획서 검증 대상).
ZERO_EVENT_SGG = ("고성", "인제", "화천", "태백")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("make_figures")

# 강원 본도(state) 시군 코드는 32xxx. admin csv 엔 인접도(31/33/34/37) 도 섞여 있다.
GANGWON_SIG_PREFIX = "32"
# 체제별 색 (영동/영서/산간)
REGIME_COLORS = {"yeongdong": "#2c7fb8", "corridor": "#253494", "yeongseo": "#7fbc41", "mountain": "#d95f0e"}
REGIME_KO = {"yeongdong": "영동(해안)", "corridor": "양간지풍 회랑", "yeongseo": "영서(내륙)", "mountain": "산간(고지)"}
DPI = 300


# ───────────────────────────────────────── 한글 폰트 ─────────────────────────
def setup_korean_font() -> str:
    """NanumBarunGothic 등록 후 rcParams 설정. (EDA.ipynb B-2 셀 참고)"""
    candidates = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
    ]
    bold_paths = [
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf",
    ]
    name = None
    for path in candidates:
        if Path(path).exists():
            fm.fontManager.addfont(path)
            name = fm.FontProperties(fname=path).get_name()
            break
    # bold 변형도 등록 (fontweight='bold' 경고 방지)
    for bp in bold_paths:
        if Path(bp).exists():
            fm.fontManager.addfont(bp)
    if name is None:
        log.warning("나눔 폰트를 못 찾음 — 한글이 깨질 수 있음")
        name = "DejaVu Sans"
    plt.rcParams["font.family"] = name
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    log.info("한글 폰트 = %s", name)
    return name


# ─────────────────────────────────── WKT MultiPolygon 파서 ───────────────────
_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _parse_wkt_multipolygon(wkt: str) -> list[np.ndarray]:
    """MULTIPOLYGON WKT → 외곽 링 좌표 배열 리스트(구멍 무시, 시각화용)."""
    rings: list[np.ndarray] = []
    # 각 polygon 의 첫 (((...))) 외곽 링만 사용 (시각화 목적)
    # "((" 로 시작하는 polygon 블록 분리
    body = wkt[wkt.find("(") :]
    # 가장 바깥 괄호를 벗기고 polygon 단위로 분리
    for poly in re.split(r"\)\s*\)\s*,\s*\(\s*\(", body):
        # poly 안의 첫 링(외곽) = 첫 "(" 이후 ~ 첫 ")" 까지의 좌표
        inner = poly[poly.find("(") + 1 if poly.find("(") >= 0 else 0:]
        first_ring = inner.split(")")[0]
        nums = _NUM.findall(first_ring)
        if len(nums) < 6:
            continue
        arr = np.array(nums, dtype=float).reshape(-1, 2)
        rings.append(arr)
    return rings


def load_admin_polygons(gangwon_only: bool = True) -> list[np.ndarray]:
    """행정경계 WKT → 외곽 링 좌표 리스트."""
    adm = pl.read_csv(ADMIN_CSV)
    rings: list[np.ndarray] = []
    for sig_cd, wkt in zip(adm["sig_cd"].cast(str).to_list(), adm["wkt"].to_list()):
        if gangwon_only and not sig_cd.startswith(GANGWON_SIG_PREFIX):
            continue
        rings.extend(_parse_wkt_multipolygon(wkt))
    log.info("행정경계 링 %d개 (gangwon_only=%s)", len(rings), gangwon_only)
    return rings


def _draw_admin(ax, rings, color="#999999", lw=0.6, alpha=0.8):
    for r in rings:
        ax.plot(r[:, 0], r[:, 1], color=color, lw=lw, alpha=alpha, zorder=2)


def load_admin_polygons_by_sgg() -> dict[str, list[np.ndarray]]:
    """강원 시군 short-name(예 '강릉') → 외곽 링 리스트. 시군 서브플롯 경계용.

    admin csv 의 sig_kor_nm('강릉시','고성군'…)에서 끝의 시/군/구를 떼어
    pole_sgg 의 short-name 과 매칭한다.
    """
    adm = pl.read_csv(ADMIN_CSV)
    has_name = "sig_kor_nm" in adm.columns
    out: dict[str, list[np.ndarray]] = {}
    for row in adm.iter_rows(named=True):
        sig_cd = str(row["sig_cd"])
        if not sig_cd.startswith(GANGWON_SIG_PREFIX):
            continue
        nm = str(row["sig_kor_nm"]) if has_name else sig_cd
        short = nm[:-1] if nm and nm[-1] in ("시", "군", "구") else nm
        out.setdefault(short, []).extend(_parse_wkt_multipolygon(row["wkt"]))
    return out


def _load_burn_rings() -> list[np.ndarray]:
    """burn geojson(top-level MultiPolygon) → 외곽 링 좌표 리스트."""
    gj = json.loads(BURN_GEOJSON.read_text())
    rings: list[np.ndarray] = []
    coords = gj["coordinates"]  # MultiPolygon: [ [ [ring], [hole]... ], ... ]
    for poly in coords:
        if not poly:
            continue
        ring = np.array(poly[0], dtype=float)  # 외곽 링
        if ring.ndim == 2 and ring.shape[1] >= 2:
            rings.append(ring[:, :2])
    return rings


# ─────────────────────────────────── 공통 데이터 로드 ───────────────────────
def load_submission() -> pl.DataFrame:
    return pl.read_csv(SUBMISSION)


def load_positives() -> pl.DataFrame:
    fp = pl.read_parquet(POSITIVES)
    # 강원 박스만: safemap 에 강원 밖 좌표오염 3건(35°N대; 전남·경남·울산권)이 섞여
    # 있어 그림 하단에 점으로 찍힘 — 모델 미사용(≤1km 전주 귀속서 자동 제외)·그림 정리용.
    return (fp.select("lon", "lat", "occu_year", "occu_mt", "sgg_cd")
            .drop_nulls(["lon", "lat"])
            .filter((pl.col("lat") >= 36.8) & (pl.col("lat") <= 38.7)
                    & (pl.col("lon") >= 126.9) & (pl.col("lon") <= 129.6)))


# ════════════════════════════════════════════════════════════════════════════
# FIG 1 — 강원 위험지도
# ════════════════════════════════════════════════════════════════════════════
def fig1_risk_map(sub: pl.DataFrame, admin_rings, pos: pl.DataFrame):
    """전주를 risk_score 로 색칠 + 행정경계 + 2019 발화점★. 위험 전주 군집 시각화."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    lon = sub["lon"].to_numpy()
    lat = sub["lat"].to_numpy()
    # risk_score(단조보정 확률)는 희소발화로 계단붕괴(고유값 66개)라 시각화 부적합 →
    # R 기반 위험 백분위(risk_pctile, 균일 0~100)로 색칠해 위험 그라데이션을 보존.
    risk = sub["risk_pctile"].to_numpy()
    decision = sub["decision"].to_numpy()

    # ── (a) 위험 백분위 연속 히트 ──
    ax = axes[0]
    order = np.argsort(risk)  # 낮은값 먼저 그려 위험이 위로
    sc = ax.scatter(
        lon[order], lat[order], c=risk[order], s=0.6,
        cmap="YlOrRd", vmin=0.0, vmax=100.0,
        rasterized=True, zorder=1,
    )
    _draw_admin(ax, admin_rings)
    cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("위험 백분위 (0~100)", fontsize=10)
    ax.set_title("(a) 위험 백분위", fontsize=13, fontweight="bold")
    ax.set_xlabel("경도 (°E)")
    ax.set_ylabel("위도 (°N)")

    # ── (b) decision=1(위험) 전주 + 발화점 ──
    ax = axes[1]
    safe = decision == 0
    risky = decision == 1
    ax.scatter(lon[safe], lat[safe], s=0.3, c="#cfd8dc", rasterized=True,
               zorder=1, label="안전 전주 (decision=0)")
    ax.scatter(lon[risky], lat[risky], s=1.2, c="#c1121f", rasterized=True,
               zorder=3, label=f"위험 전주 (decision=1, {risky.sum():,}개)")
    _draw_admin(ax, admin_rings)
    if pos.height:
        ax.scatter(pos["lon"].to_numpy(), pos["lat"].to_numpy(), marker="*",
                   s=55, c="#1a1a1a", edgecolors="white", linewidths=0.4,
                   zorder=5, label=f"과거 발화점 ({pos.height}건, 검증 앵커)")
    ax.set_title("(b) 위험 판정 & 발화점",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("경도 (°E)")
    ax.set_ylabel("위도 (°N)")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9, markerscale=3)

    fig.suptitle(
        "그림 1. 강원 전주 산불 위험지도",
        fontsize=15, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "fig1_risk_map.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG 2 — 체제 지도 + 편중 완화
# ════════════════════════════════════════════════════════════════════════════
def fig2_regime_alloc(sub: pl.DataFrame, admin_rings, rj: dict):
    """영동/영서/산간 색 지도 + 전역컷 vs 체제별컷 양성분포 막대(편중 완화)."""
    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.22)

    # ── (a) 체제 지도 ──
    ax = fig.add_subplot(gs[0, 0])
    lon = sub["lon"].to_numpy()
    lat = sub["lat"].to_numpy()
    regime = sub["regime"].to_numpy()
    for r, color in REGIME_COLORS.items():
        m = regime == r
        ax.scatter(lon[m], lat[m], s=0.5, c=color, rasterized=True, zorder=1,
                   label=f"{REGIME_KO[r]} ({m.sum():,})")
    _draw_admin(ax, admin_rings)
    ax.set_title("(a) 체제 게이트(MoE)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("경도 (°E)")
    ax.set_ylabel("위도 (°N)")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9, markerscale=6)

    # ── (b) 전역컷 vs 체제별컷 양성분포 ──
    ax = fig.add_subplot(gs[0, 1])
    # 전역컷: 전역 상위 π% — 양성이 영동에 쏠림. (재현: 전역 risk_score 상위 2%)
    risk = sub["risk_score"].to_numpy()
    n = len(risk)
    k = int(round(0.02 * n))
    thr_global = np.partition(risk, n - k)[n - k]
    global_pos = risk >= thr_global
    regimes = list(REGIME_COLORS.keys())
    global_share = np.array([
        global_pos[regime == r].sum() for r in regimes
    ], dtype=float)
    global_share = 100 * global_share / global_share.sum()

    # 체제별컷: regime_threshold_analysis.json 의 share_of_positives_pct (균등 운영)
    rd = rj["regime_decision_distribution"]
    regime_share = np.array([rd[r]["share_of_positives_pct"] for r in regimes])
    regime_rate = np.array([rd[r]["positive_rate_pct"] for r in regimes])

    x = np.arange(len(regimes))
    w = 0.38
    b1 = ax.bar(x - w / 2, global_share, w, color="#9e9e9e",
                label="전역 단일컷 (상위 2%)")
    b2 = ax.bar(x + w / 2, regime_share, w,
                color=[REGIME_COLORS[r] for r in regimes],
                label="체제별 컷 (각 체제 내 2%)")
    for rect, v in zip(b1, global_share):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 1, f"{v:.0f}%",
                ha="center", fontsize=9, color="#555")
    for rect, v in zip(b2, regime_share):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 1, f"{v:.0f}%",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([REGIME_KO[r] for r in regimes], fontsize=10)
    ax.set_ylabel("위험 판정(양성) 중 각 체제 비중 (%)", fontsize=10)
    ax.set_ylim(0, max(global_share.max(), regime_share.max()) + 14)
    ax.set_title("(b) 편중 완화",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    # 양성률 주석
    ax.text(0.02, 0.96,
            "체제별 컷 → 각 체제 양성률 ≈ 2.0%로 균일\n"
            "(운영 타당성↑: 한 지역만 순시 자원 독식 방지)",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round", fc="#fff8e1", ec="#e0c060"))

    fig.suptitle(
        "그림 2. 지역 체제(MoE)와 편중 완화",
        fontsize=14, fontweight="bold", y=0.99,
    )
    fig.subplots_adjust(left=0.06, right=0.97, top=0.9, bottom=0.1)
    _save(fig, "fig2_regime_alloc.png")
    return global_share, regime_share


# ════════════════════════════════════════════════════════════════════════════
# FIG 3 — recall@top-k 곡선
# ════════════════════════════════════════════════════════════════════════════
def fig3_recall_topk(rj: dict):
    """공간블록 CV recall@top-k: 전역 vs 체제별 + 랜덤기준선."""
    cv = rj["alloc_cv_recall"]
    ks = [0.01, 0.02, 0.05, 0.1]
    kcols = ["top0.01", "top0.02", "top0.05", "top0.1"]
    kpct = [k * 100 for k in ks]

    glob = [cv["global"][c]["mean"] for c in kcols]
    glob_std = [cv["global"][c]["std"] for c in kcols]
    reg = [cv["regime_count"][c]["mean"] for c in kcols]
    reg_std = [cv["regime_count"][c]["std"] for c in kcols]
    random_baseline = ks  # 무작위로 상위 k% 뽑으면 recall ≈ k

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.errorbar(kpct, glob, yerr=glob_std, marker="o", ms=8, lw=2.2,
                color="#1f77b4", capsize=4, label="전역 단일컷 (global)")
    ax.errorbar(kpct, reg, yerr=reg_std, marker="s", ms=8, lw=2.2,
                color="#d95f0e", capsize=4, label="체제별 컷 (regime)")
    ax.plot(kpct, random_baseline, "--", color="#777", lw=1.8,
            label="랜덤 기준선 (recall = k)")
    ax.fill_between(kpct, random_baseline, glob, color="#1f77b4", alpha=0.06)

    # 배수 주석
    for kp, g, rnd in zip(kpct, glob, random_baseline):
        ax.annotate(f"×{g / rnd:.1f}", (kp, g), textcoords="offset points",
                    xytext=(6, 8), fontsize=9, color="#1f77b4", fontweight="bold")

    ax.set_xlabel("상위 위험 전주 비율 top-k (%)", fontsize=11)
    ax.set_ylabel("발화점 recall (공간 블록 CV, 홀드아웃)", fontsize=11)
    ax.set_xticks(kpct)
    ax.set_title("그림 3. 발화점 recall@top-k (랜덤 대비 ~2배)\n"
                 "",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(alpha=0.25)

    gap = rj["random_vs_spatial_gap"]
    note = (
        f"공간 블록 CV (10km GroupKFold) — 낙관편향 감시:\n"
        f"  무작위분할 recall@5% = {gap['random_recall']:.3f}\n"
        f"  공간분할 recall@5% = {gap['spatial_recall']:.3f}  (격차 {gap['gap']:+.3f})\n"
        f"→ 격차≈0: 공간 자기상관 누수로 성능을 부풀리지 않음\n\n"
        f"천장 한계: 발화점이 희소하고(706개 매핑) 인간발화 위주라\n"
        f"recall 절대값은 낮음 — 상대 순위(랭킹) 도구로 해석."
    )
    ax.text(0.97, 0.03, note, transform=ax.transAxes, fontsize=8.3,
            va="bottom", ha="right",
            bbox=dict(boxstyle="round", fc="#f3f6fb", ec="#9bb"))
    fig.tight_layout()
    _save(fig, "fig3_recall_topk.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG 4 — F1 민감도 곡선
# ════════════════════════════════════════════════════════════════════════════
def fig4_f1_sensitivity(rj: dict):
    """π(0.5~10%)별 proxy-F1, 전역 vs 체제별, 권장 운영점(2%) 표시."""
    sens = rj["f1_sensitivity_global_vs_regime"]
    pis = sorted(float(k) for k in sens.keys())
    glob_f1 = [sens[f"{p}"]["global"]["f1_proxy"] if f"{p}" in sens
               else sens[_kfmt(p)]["global"]["f1_proxy"] for p in pis]
    glob_rec = [sens[_kfmt(p)]["global"]["recall"] for p in pis]
    reg_f1 = [sens[_kfmt(p)]["regime"]["f1_proxy"] for p in pis]
    reg_rec = [sens[_kfmt(p)]["regime"]["recall"] for p in pis]
    pis_pct = [p * 100 for p in pis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ── (a) proxy-F1 ──
    ax = axes[0]
    ax.plot(pis_pct, glob_f1, marker="o", ms=7, lw=2.2, color="#1f77b4",
            label="전역 단일컷")
    ax.plot(pis_pct, reg_f1, marker="s", ms=7, lw=2.2, color="#d95f0e",
            label="체제별 컷")
    ax.axvline(2.0, color="#2a9d8f", ls="--", lw=1.8)
    ax.annotate("권장 운영점 π=2%\n(한전 점검 capacity 기반)",
                (2.0, max(glob_f1) * 0.55), fontsize=9, color="#2a9d8f",
                ha="center",
                bbox=dict(boxstyle="round", fc="#e8f6f3", ec="#2a9d8f"))
    ax.set_xlabel("가정 양성비율 π (%) — KEPCO 비공개 정답 비율 모름", fontsize=10.5)
    ax.set_ylabel("proxy F1 (발화점 앵커 기준)", fontsize=10.5)
    ax.set_title("(a) F1 민감도",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9.5)
    ax.grid(alpha=0.25)

    # ── (b) recall(=택해진 예산에서) ──
    ax = axes[1]
    ax.plot(pis_pct, glob_rec, marker="o", ms=7, lw=2.2, color="#1f77b4",
            label="전역 단일컷 recall")
    ax.plot(pis_pct, reg_rec, marker="s", ms=7, lw=2.2, color="#d95f0e",
            label="체제별 컷 recall")
    ax.plot(pis_pct, [p for p in pis], "--", color="#777", lw=1.6,
            label="랜덤 기준선")
    ax.axvline(2.0, color="#2a9d8f", ls="--", lw=1.8)
    ax.set_xlabel("가정 양성비율 π = 예산 (%)", fontsize=10.5)
    ax.set_ylabel("발화점 recall", fontsize=10.5)
    ax.set_title("(b) 예산↑ → recall↑",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9.5)
    ax.grid(alpha=0.25)

    fig.suptitle(
        "그림 4. 임계값 민감도\n"
        "",
        fontsize=13.5, fontweight="bold", y=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save(fig, "fig4_f1_sensitivity.png")


def _kfmt(p: float) -> str:
    """0.5%→'0.005' 같은 json 키 포맷 복원 (불필요 0 제거)."""
    s = f"{p:.3f}".rstrip("0").rstrip(".")
    return s


# ════════════════════════════════════════════════════════════════════════════
# FIG 5 — 2019 고성 케이스
# ════════════════════════════════════════════════════════════════════════════
def fig5_goseong(sub: pl.DataFrame, pos: pl.DataFrame):
    """burn-scar + 인근 전주(위험/안전) + 발화점★ + 풍하(서→동) 화살표."""
    burn_rings = _load_burn_rings()
    all_v = np.vstack(burn_rings)
    lon0, lon1 = all_v[:, 0].min(), all_v[:, 0].max()
    lat0, lat1 = all_v[:, 1].min(), all_v[:, 1].max()
    # 흉터 위·아래·동서로 약간의 여백 (풍하=동쪽을 조금 더). 종횡비 squash 방지.
    span_lon = lon1 - lon0
    span_lat = lat1 - lat0
    bbox = (lon0 - 0.45 * span_lon, lon1 + 0.55 * span_lon,
            lat0 - 0.55 * span_lat, lat1 + 0.55 * span_lat)

    lon = sub["lon"].to_numpy()
    lat = sub["lat"].to_numpy()
    m = (lon >= bbox[0]) & (lon <= bbox[1]) & (lat >= bbox[2]) & (lat <= bbox[3])
    s_lon = lon[m]
    s_lat = lat[m]
    s_dec = sub["decision"].to_numpy()[m]
    # risk_score(단조보정)는 계단붕괴라 risk_pctile(R 기반, 균일)로 색칠.
    s_risk = sub["risk_pctile"].to_numpy()[m]

    fig, ax = plt.subplots(figsize=(12, 7.5))

    # burn scar — 채움은 옅게(zorder 낮음), 외곽선은 진하게(점 위로) 그려 형상 유지
    for i, r in enumerate(burn_rings):
        ax.add_patch(MplPolygon(r, closed=True, facecolor="#7a3b2e",
                                edgecolor="none", alpha=0.30, zorder=2,
                                label="2019 고성 burn-scar (dNBR)" if i == 0 else None))
    for r in burn_rings:
        ax.plot(r[:, 0], r[:, 1], color="#4a1c12", lw=1.0, alpha=0.9, zorder=4.5)

    # 인근 전주를 risk_score(연속)로 색칠 → 풍하 방향 위험 gradient 가시화.
    # (decision=1 은 흉터=대부분 산림이라 가장자리에 적음 → 연속 점수가 더 정직)
    # 국지 risk 분포가 좁아 색 대비가 약하므로 국지 [min, max] 로 스트레치.
    vmin = float(np.quantile(s_risk, 0.05)) if s_risk.size else 0.0
    vmax = float(s_risk.max()) if s_risk.size else 1.0
    order = np.argsort(s_risk)
    sc = ax.scatter(s_lon[order], s_lat[order], c=s_risk[order], s=9,
                    cmap="YlOrRd", vmin=vmin, vmax=vmax, zorder=3,
                    alpha=0.75, edgecolors="none", rasterized=True)
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("전주 위험 백분위 risk_pctile (국지 스케일)", fontsize=10)
    # decision=1 위험 전주는 테두리 강조
    risky = s_dec == 1
    if risky.sum():
        ax.scatter(s_lon[risky], s_lat[risky], s=40, facecolors="none",
                   edgecolors="#7a0010", linewidths=1.1, zorder=5,
                   label=f"위험 판정(decision=1) 전주 ({int(risky.sum())}개)")

    # 발화 추정점(★, burn-scar 서쪽 끝) + safemap 발화점
    gpos = pos.filter(
        (pl.col("lon") >= bbox[0]) & (pl.col("lon") <= bbox[1])
        & (pl.col("lat") >= bbox[2]) & (pl.col("lat") <= bbox[3])
    )
    ig_lon, ig_lat = lon0, (lat0 + lat1) / 2
    ax.scatter([ig_lon], [ig_lat], marker="*", s=480, c="#ffd000",
               edgecolors="#7a5c00", linewidths=1.3, zorder=7,
               label="발화 추정점 (흉터 서쪽 끝)")
    if gpos.height:
        ax.scatter(gpos["lon"].to_numpy(), gpos["lat"].to_numpy(), marker="*",
                   s=130, c="#1a1a1a", edgecolors="white", linewidths=0.5,
                   zorder=6, label=f"safemap 발화점 ({gpos.height}건)")

    # 풍하(양간지풍 서→동) 화살표
    mid_lat = (lat0 + lat1) / 2
    arr = FancyArrowPatch(
        (bbox[0] + 0.02 * span_lon, mid_lat), (bbox[1] - 0.04 * span_lon, mid_lat),
        arrowstyle="-|>", mutation_scale=30, color="#1565c0", lw=3.0,
        alpha=0.85, zorder=4,
    )
    ax.add_patch(arr)
    ax.text((lon0 + lon1) / 2, bbox[3] - 0.12 * (bbox[3] - bbox[2]),
            "양간지풍 풍하 방향 (서 → 동, ~6.7km)",
            color="#1565c0", fontsize=13, fontweight="bold", ha="center")

    # 풍하 위험 동향: 서/동 절반 전주 평균 risk 비교 (흉터 신장과 정합 메시지)
    mid_lon = (lon0 + lon1) / 2
    west = s_lon < mid_lon
    east = ~west
    west_mean = float(s_risk[west].mean()) if west.any() else 0.0
    east_mean = float(s_risk[east].mean()) if east.any() else 0.0

    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_xlabel("경도 (°E)")
    ax.set_ylabel("위도 (°N)")
    ax.set_title(
        "그림 5. 2019 고성-속초 산불 케이스\n"
        "",
        fontsize=12.5, fontweight="bold",
    )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.92)

    n_near = int(m.sum())
    ax.text(0.015, 0.985,
            f"케이스 범위 전주 {n_near:,}개\n"
            f"발화원: KEPCO 특고압선 아크 (2023 법원 배상 인정)\n"
            f"흉터 신장: 서→동 ~6.7km (양간지풍 풍하)\n"
            f"전주 평균 risk  서 {west_mean:.4f} / 동 {east_mean:.4f}",
            transform=ax.transAxes, fontsize=8.8, va="top",
            bbox=dict(boxstyle="round", fc="#fff3e0", ec="#d99"))
    fig.tight_layout()
    _save(fig, "fig5_goseong_2019.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG 6 — 발화 분해 예시 (한 고위험 전주의 I·S·W·regime 기여)
# ════════════════════════════════════════════════════════════════════════════
def fig6_ignition_decomp():
    """한 고위험 전주의 발화 성향 I 를 피처별 기여로 분해 (왜 위험한가)."""
    import pfire.config as cfg
    import pfire.experts as experts
    import pfire.regimes as regimes
    import pfire.io as io

    master = io.load_master()
    gate, regime_order = regimes.compute_gate(master)  # (N,3)
    feats = experts.build_ignition_features(master)    # dict key→(N,)
    per_expert = {r: experts.expert_score(feats, r) for r in regime_order}
    expert_mat = np.stack([per_expert[r] for r in regime_order], axis=1)
    I = np.clip(np.sum(gate * expert_mat, axis=1), 0.0, 1.0)

    # S, W 근사 (decision 산출과 동일 의미의 해석용 성분):
    #  S = 정적 확산취약 S_p, W = 시즌 기상(FWI 강도 + 고위험일 + 양간) 가중 — README 3.5
    S = master["S_p"].to_numpy().astype(np.float64)
    fwi_n = experts._minmax01(master["fwi_q90"].to_numpy().astype(np.float64))
    hdd_n = experts._minmax01(master["fwi_high_danger_days"].to_numpy().astype(np.float64))
    ygn_n = experts._minmax01(master["yanggan_days"].to_numpy().astype(np.float64))
    W = 0.5 * fwi_n + 0.3 * hdd_n + 0.2 * ygn_n

    # 대표 고위험 전주: 영동 체제에서 I 상위 (양간지풍 시나리오 설명력↑)
    regime_idx = gate.argmax(axis=1)
    yd = regime_order.index("yeongdong")
    cand = np.where(regime_idx == yd)[0]
    pick = int(cand[np.argmax(I[cand])])
    dom_regime = regime_order[gate[pick].argmax()]

    # 피처별 기여 = Σ_r gate_r * w_r[key] * feat_key  (정규화 전 raw 기여)
    feat_keys = ["forest", "road", "powerline", "fwi", "yanggan", "fuel", "landcover"]
    feat_ko = {
        "forest": "산림 근접", "road": "도로 근접", "powerline": "송전선 근접",
        "fwi": "FWI 강도", "yanggan": "양간풍 일수", "fuel": "연료량", "landcover": "토지피복",
    }
    contrib = {}
    for key in feat_keys:
        c = 0.0
        for ci, r in enumerate(regime_order):
            w = cfg.EXPERT_WEIGHTS[r]
            wsum = sum(w.values())
            c += gate[pick, ci] * (w.get(key, 0.0) / wsum) * feats[key][pick]
        contrib[key] = c
    total_I = sum(contrib.values())

    fig = plt.figure(figsize=(15, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 0.85], wspace=0.32)

    # ── (a) 피처별 발화 기여 (수평 막대) ──
    ax = fig.add_subplot(gs[0, 0])
    items = sorted(contrib.items(), key=lambda kv: kv[1])
    labels = [feat_ko[k] for k, _ in items]
    vals = [v for _, v in items]
    colors = ["#c1121f" if v >= max(vals) * 0.5 else "#fb8c00" if v > 0 else "#cccccc"
              for v in vals]
    ax.barh(labels, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.02, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlabel("발화 성향 I 기여도 (가중·정규화)", fontsize=10.5)
    ax.set_title(f"(a) 발화 성향 I={total_I:.3f} 분해\n"
                 f"우세 체제 = {REGIME_KO[dom_regime]}",
                 fontsize=12, fontweight="bold")

    # ── (b) 체제 게이트 가중 (pie) ──
    ax = fig.add_subplot(gs[0, 1])
    g = gate[pick]
    ax.pie(g, labels=[REGIME_KO[r] for r in regime_order],
           colors=[REGIME_COLORS[r] for r in regime_order],
           autopct="%1.0f%%", startangle=90,
           wedgeprops=dict(edgecolor="white", linewidth=1.5),
           textprops=dict(fontsize=10))
    ax.set_title("(b) 체제 soft 게이트 g(p)\n(MoE 전문가 혼합 비중)",
                 fontsize=12, fontweight="bold")

    # ── (c) I × S × W = 하루 위험 ──
    ax = fig.add_subplot(gs[0, 2])
    comps = ["I\n발화성향", "S\n확산취약", "W\n기상노출"]
    cvals = [I[pick], S[pick], W[pick]]
    bars = ax.bar(comps, cvals, color=["#c1121f", "#fb8c00", "#1565c0"])
    for b, v in zip(bars, cvals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(cvals) * 1.25)
    ax.set_ylabel("성분 값 [0,1]", fontsize=10.5)
    h = cvals[0] * cvals[1] * cvals[2]
    ax.set_title(f"(c) 위험 = I × S × W\n= {cvals[0]:.2f}×{cvals[1]:.2f}×{cvals[2]:.2f} = {h:.4f}",
                 fontsize=12, fontweight="bold")

    info = (
        f"전주 pole_id={int(master['pole_id'][pick])} | "
        f"위치 ({master['lon'][pick]:.3f}°E, {master['lat'][pick]:.3f}°N) | "
        f"시군={master['sgg'][pick]} | "
        f"산림거리={master['dist_to_forest'][pick]:.0f}m, "
        f"송전선거리={master['dist_to_powerline'][pick]:.0f}m, "
        f"양간일={master['yanggan_days'][pick]:.1f}"
    )
    fig.suptitle(
        "그림 6. 고위험 전주 위험 분해\n"
        + info,
        fontsize=12.5, fontweight="bold", y=1.0,
    )
    fig.subplots_adjust(left=0.1, right=0.97, top=0.84, bottom=0.1)
    _save(fig, "fig6_ignition_decomp.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG 7 — 16개 시군 서브플롯 위험지도 (risk_mean)  [Phase-5]
# ════════════════════════════════════════════════════════════════════════════
def _sgg_order() -> list[str]:
    """시군을 체제(영동→영서→산간)·이름 순으로 정렬한 리스트."""
    reg_rank = {"yeongdong": 0, "corridor": 1, "yeongseo": 2, "mountain": 3}
    return sorted(SGG_TO_REGIME.keys(),
                  key=lambda s: (reg_rank.get(SGG_TO_REGIME[s], 9), s))


def load_pole_sgg() -> pl.DataFrame:
    """pole_id → sgg 매핑(시군 서브플롯용; used_dataset 읽기 전용)."""
    return pl.read_parquet(POLE_SGG)


def fig7_sgg_risk_subplots(sub: pl.DataFrame, sgg_of_pole: np.ndarray):
    """16개 시군 격자 서브플롯 — 각 시군 위험지도(risk_mean/risk_score 색칠).

    어디가 위험한지 시군 단위로 본다. 색은 risk_mean(없으면 risk_score). 전역
    분위 색척도로 시군 간 절대 위험을 비교할 수 있게 한다.
    """
    lon = sub["lon"].to_numpy()
    lat = sub["lat"].to_numpy()
    # risk_score(단조보정)는 계단붕괴 → risk_pctile(R 기반, 균일) 우선.
    risk = (sub["risk_pctile"].to_numpy() if "risk_pctile" in sub.columns
            else sub["risk_score"].to_numpy())
    vmax = float(np.quantile(risk, 0.995))
    sgg_rings = load_admin_polygons_by_sgg()

    order = _sgg_order()
    fig, axes = plt.subplots(4, 4, figsize=(18, 17))
    axes = axes.ravel()
    for ax, sgg in zip(axes, order):
        m = sgg_of_pole == sgg
        if not m.any():
            ax.set_visible(False)
            continue
        rk = risk[m]
        o = np.argsort(rk)
        sc = ax.scatter(lon[m][o], lat[m][o], c=rk[o], s=2.0, cmap="YlOrRd",
                        vmin=0.0, vmax=vmax, rasterized=True)
        for r in sgg_rings.get(sgg, []):
            ax.plot(r[:, 0], r[:, 1], color="#444", lw=0.8, alpha=0.85, zorder=3)
        reg = SGG_TO_REGIME.get(sgg, "?")
        star = " ★" if sgg in ZERO_EVENT_SGG else ""
        ax.set_title(f"{sgg} ({REGIME_KO.get(reg, reg)}){star}\n전주 {int(m.sum()):,}개 · "
                     f"평균위험 {rk.mean():.3f}",
                     fontsize=10.5, fontweight="bold",
                     color=REGIME_COLORS.get(reg, "#333"))
        ax.tick_params(labelsize=7)
    # 남는 축 숨김
    for ax in axes[len(order):]:
        ax.set_visible(False)
    cbar = fig.colorbar(sc, ax=axes.tolist(), fraction=0.012, pad=0.01)
    cbar.set_label("위험 점수 risk_mean (사후평균; 전역 분위 색척도)", fontsize=11)
    fig.suptitle(
        "그림 7. 시군별 위험지도\n"
        "★ = zero-event 시군",
        fontsize=15, fontweight="bold", y=0.995,
    )
    fig.subplots_adjust(left=0.04, right=0.92, top=0.93, bottom=0.04,
                        hspace=0.30, wspace=0.18)
    _save(fig, "fig7_sgg_risk_subplots.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG 8 — 16개 시군 불확실성 서브플롯 (risk_hi − risk_lo)  [Phase-5]
# ════════════════════════════════════════════════════════════════════════════
def fig8_sgg_uncertainty_subplots(sub: pl.DataFrame, sgg_of_pole: np.ndarray):
    """시군별 위험 사후 폭(risk_hi−risk_lo) 서브플롯 — '위험하지만 불확실' 가시화.

    폭이 넓은 전주 = 사후가 넓다(현장확인 1순위). zero-event 시군은 발화 앵커가
    없어 부모로 수렴하되 폭이 커지는 경향을 시군 단위로 본다.
    """
    if "risk_hi" not in sub.columns or "risk_lo" not in sub.columns:
        log.warning("risk_hi/risk_lo 없음 — fig8(불확실성) 생략. posterior 경로로 재생성 필요.")
        return
    lon = sub["lon"].to_numpy()
    lat = sub["lat"].to_numpy()
    risk_mean = (sub["risk_mean"].to_numpy() if "risk_mean" in sub.columns
                 else sub["risk_score"].to_numpy())
    # 상대 불확실성(cv 유사) = 위험폭 / 평균. 절대폭은 위험 큰 영동이 지배하므로,
    # zero-event(저위험) 지역의 불확실성을 보려면 상대값을 써야 한다(절대폭은 오해 유발).
    width = (sub["risk_hi"].to_numpy() - sub["risk_lo"].to_numpy()) / np.maximum(risk_mean, 1e-9)
    width = np.clip(width, 0.0, float(np.quantile(width, 0.99)))  # 저위험 전주 분모폭주 방지
    vmax = float(width.max())
    sgg_rings = load_admin_polygons_by_sgg()

    order = _sgg_order()
    fig, axes = plt.subplots(4, 4, figsize=(18, 17))
    axes = axes.ravel()
    sgg_mean_w = {}
    for ax, sgg in zip(axes, order):
        m = sgg_of_pole == sgg
        if not m.any():
            ax.set_visible(False)
            continue
        w = width[m]
        sgg_mean_w[sgg] = float(w.mean())
        o = np.argsort(w)
        sc = ax.scatter(lon[m][o], lat[m][o], c=w[o], s=2.0, cmap="viridis",
                        vmin=0.0, vmax=vmax, rasterized=True)
        for r in sgg_rings.get(sgg, []):
            ax.plot(r[:, 0], r[:, 1], color="#555", lw=0.8, alpha=0.85, zorder=3)
        reg = SGG_TO_REGIME.get(sgg, "?")
        star = " ★" if sgg in ZERO_EVENT_SGG else ""
        ax.set_title(f"{sgg} ({REGIME_KO.get(reg, reg)}){star}\n평균 불확실 {w.mean():.2f}",
                     fontsize=10.5, fontweight="bold",
                     color=REGIME_COLORS.get(reg, "#333"))
        ax.tick_params(labelsize=7)
    for ax in axes[len(order):]:
        ax.set_visible(False)
    cbar = fig.colorbar(sc, ax=axes.tolist(), fraction=0.012, pad=0.01)
    cbar.set_label("상대 불확실성 (위험폭 ÷ 평균; 클수록 불확실)", fontsize=11)
    # zero-event 시군 상대 불확실성이 큰지 한 줄 요약
    ze = [s for s in ZERO_EVENT_SGG if s in sgg_mean_w]
    ze_mean = np.mean([sgg_mean_w[s] for s in ze]) if ze else float("nan")
    all_mean = np.mean(list(sgg_mean_w.values())) if sgg_mean_w else float("nan")
    fig.suptitle(
        "그림 8. 시군별 사후 불확실성\n"
        f"★ zero-event 시군 평균 {ze_mean:.2f} vs 전체 평균 {all_mean:.2f} "
        "(발화 앵커 없는 지역일수록 상대 불확실성 큼)",
        fontsize=14.5, fontweight="bold", y=0.995,
    )
    fig.subplots_adjust(left=0.04, right=0.92, top=0.93, bottom=0.04,
                        hspace=0.30, wspace=0.18)
    _save(fig, "fig8_sgg_uncertainty_subplots.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG 9 — 체제별 명목 vs 실측 커버리지  [Phase-5]
# ════════════════════════════════════════════════════════════════════════════
def fig9_coverage(rj: dict):
    """체제별 명목(1−α) vs 홀드아웃 발화점 실측 커버리지 — conformal 보장 확인."""
    pc = rj.get("posterior_coverage", {})
    cov_conf = pc.get("coverage_conformal")
    cov_cred = pc.get("coverage_credible")
    if not cov_conf:
        log.warning("coverage_conformal 없음 — fig9(커버리지) 생략. posterior 경로로 재생성 필요.")
        return
    nominal = cov_conf.get("all", {}).get("nominal", 0.90)
    groups = ["all", "yeongdong", "yeongseo", "mountain"]
    glabels = ["전체", REGIME_KO["yeongdong"], REGIME_KO["yeongseo"], REGIME_KO["mountain"]]

    conf_emp = [cov_conf.get(g, {}).get("empirical", np.nan) for g in groups]
    cred_emp = ([cov_cred.get(g, {}).get("empirical", np.nan) for g in groups]
                if cov_cred else [np.nan] * len(groups))
    n_test = [cov_conf.get(g, {}).get("n_test", 0) for g in groups]

    fig, ax = plt.subplots(figsize=(11, 6.8))
    x = np.arange(len(groups))
    w = 0.36
    b1 = ax.bar(x - w / 2, conf_emp, w, color="#d95f0e",
                label="conformal 실측 커버리지 (per-regime/Mondrian)")
    b2 = ax.bar(x + w / 2, cred_emp, w, color="#2c7fb8",
                label="베이지안 credible 실측 커버리지")
    ax.axhline(nominal, color="#2a9d8f", ls="--", lw=2.0,
               label=f"명목 목표 {nominal*100:.0f}%")
    # 합격대(±3%p) 음영
    ax.axhspan(nominal - 0.03, nominal + 0.03, color="#2a9d8f", alpha=0.10,
               label="합격대 (명목 ±3%p)")
    for rect, v, nt in zip(b1, conf_emp, n_test):
        if np.isfinite(v):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.01,
                    f"{v:.2f}", ha="center", fontsize=9, fontweight="bold",
                    color="#d95f0e")
    for rect, v in zip(b2, cred_emp):
        if np.isfinite(v):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 0.01,
                    f"{v:.2f}", ha="center", fontsize=9, color="#2c7fb8")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{g}\n(발화 {int(nt)}건)" for g, nt in zip(glabels, n_test)],
                       fontsize=10)
    ax.set_ylabel("홀드아웃 발화점 실측 포함율 (커버리지)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "그림 9. 체제별 커버리지 (명목 90% vs 실측)\n"
        "",
        fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9.5, framealpha=0.95)
    ax.grid(axis="y", alpha=0.25)
    note = (
        "conformal = 분포가정 없는 유한표본 커버리지 보장(per-regime).\n"
        "credible = 베이지안 사후예측 구간(모델 오설정 시 under-cover 가능 → conformal 로 보정).\n"
        "각 fold 의 train 발화점만으로 임계/사후 추정 → test 발화점 미사용(누수안전)."
    )
    ax.text(0.02, 0.03, note, transform=ax.transAxes, fontsize=8.4,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round", fc="#f3f6fb", ec="#9bb"))
    fig.tight_layout()
    _save(fig, "fig9_coverage.png")


# ════════════════════════════════════════════════════════════════════════════
# FIG 10 — 독립 Poisson-Gamma vs BYM2 공간평활 시군 위험  [Phase-5b]
# ════════════════════════════════════════════════════════════════════════════
def fig10_bym_vs_pg(rj: dict):
    """시군별 독립(PG) vs BYM2 공간평활 상대배율 비교 — 이웃 borrow 효과.

    posterior_spatial_backend='bym' 로 실행한 산출(post_artifact.bym_vs_poisson_gamma)에서
    시군별 pg_mult/bym_mult 를 나란히 그려, zero-event 시군이 이웃에서 borrow 해 PG(부모
    붕괴) 대비 어떻게 움직였는지 본다. BYM 데이터 없으면 생략(poisson_gamma 백엔드).
    """
    pc = rj.get("posterior_coverage", {})
    cmp = pc.get("bym_vs_poisson_gamma")
    if not cmp:
        log.warning("bym_vs_poisson_gamma 없음 — fig10 생략 "
                    "(--posterior-spatial bym 으로 재생성 필요).")
        return
    rows = cmp["sgg"]
    # 체제·이름 순 정렬로 보기 좋게.
    reg_rank = {"yeongdong": 0, "corridor": 1, "yeongseo": 2, "mountain": 3}
    rows = sorted(rows, key=lambda r: (reg_rank.get(SGG_TO_REGIME.get(r["sgg"], "?"), 9),
                                       r["sgg"]))
    sggs = [r["sgg"] for r in rows]
    pg = np.array([r["pg_mult"] for r in rows])
    bym = np.array([r["bym_mult"] for r in rows])

    fig, ax = plt.subplots(figsize=(13, 6.8))
    x = np.arange(len(sggs))
    w = 0.4
    ax.bar(x - w / 2, pg, w, color="#9e9e9e", label="독립 Poisson-Gamma (공간상관 없음)")
    ax.bar(x + w / 2, bym, w,
           color=[REGIME_COLORS.get(SGG_TO_REGIME.get(s, "?"), "#555") for s in sggs],
           label="BYM2 공간 CAR (이웃 borrow)")
    ax.axhline(1.0, color="#444", ls="--", lw=1.2, label="전역 평균(상대배율 1)")
    for xi, s in zip(x, sggs):
        if s in ZERO_EVENT_SGG:
            ax.annotate("★", (xi, max(pg[list(x).index(xi)], bym[list(x).index(xi)]) + 0.05),
                        ha="center", fontsize=12, color="#c1121f")
    ax.set_xticks(x)
    ax.set_xticklabels(sggs, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("시군 상대 위험 배율 (전역율 기준)", fontsize=11)
    mi = cmp.get("moran", {})
    bm = cmp.get("borrow_mean_watch", float("nan"))
    ax.set_title(
        "그림 10. Poisson-Gamma vs BYM2 공간평활\n"
        f"Moran's I(매끄러움) PG={mi.get('pg', float('nan')):.3f} → BYM={mi.get('bym', float('nan')):.3f} | "
        f"zero-event 평균 borrow(추정 이동)={bm:.3f}",
        fontsize=12.5, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(axis="y", alpha=0.25)
    note = (
        "독립 PG: 발화 0 시군은 부모(체제→전역)로만 수렴 — 이웃 정보 미사용.\n"
        "BYM2: 격자 rook 인접 그래프로 이웃 강도를 빌려(borrow) 사후를 공간적으로 평활.\n"
        "★ = zero-event 시군(고성·인제·화천·태백). recall·커버리지 레버 아님 — 사후 공간정밀도 확장."
    )
    ax.text(0.02, 0.97, note, transform=ax.transAxes, fontsize=8.4, va="top",
            bbox=dict(boxstyle="round", fc="#f3f6fb", ec="#9bb"))
    fig.tight_layout()
    _save(fig, "fig10_bym_vs_pg.png")


# ─────────────────────────────────────── 저장 헬퍼 ──────────────────────────
def _save(fig, name: str):
    path = OUT_FIG / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("저장: %s", path)


# ════════════════════════════════════════════════════════════════════════════
# FIG 11 — 강원 전역 사후 불확실성 지도 (MC credible)  [Phase-5]
# ════════════════════════════════════════════════════════════════════════════
def fig11_uncertainty_map(sub: pl.DataFrame, admin_rings, pos: pl.DataFrame):
    """체제별(영동/영서/산간) 사후 불확실성 서브플롯 — 지역별로 줌해 가시성 확보.

    전 강원 한 장(점 138만 겹침)은 가시성이 떨어져, 체제(MoE 지역) 단위로 쪼개 각
    패널을 그 지역 bbox 로 확대한다. fig8(16시군·상대폭)과 달리 여기선 3개 큰 패널로
    절대 신뢰폭과 '현장확인 우선'을 본다.
      상단: 절대 사후폭 (risk_hi − risk_lo) — 위험 추정의 ± 신뢰폭(체제 간 vmax 공유).
      하단: 위험 상위(상위 10%) ∩ 상대 불확실 상위(상위 20%) = 현장확인·능동라벨 우선
            전주 강조 + 그 지역 발화점★.
    risk_lo/risk_hi 는 posterior.propagate_risk_posterior 의 MC 사후예측 분위(기본 90%).
    """
    if "risk_lo" not in sub.columns or "risk_hi" not in sub.columns:
        log.warning("risk_lo/risk_hi 없음 — fig11(불확실성 지도) 생략. posterior 경로로 재생성 필요.")
        return
    if "regime" not in sub.columns:
        log.warning("regime 열 없음 — fig11 체제별 서브플롯 생략.")
        return

    lon = sub["lon"].to_numpy()
    lat = sub["lat"].to_numpy()
    reg = sub["regime"].to_numpy().astype(str)
    lo = sub["risk_lo"].to_numpy()
    hi = sub["risk_hi"].to_numpy()
    abs_w = np.clip(hi - lo, 0.0, None)
    mid = 0.5 * (lo + hi)                       # 사후평균 프록시(risk_mean 미동봉 시)
    rel_w = abs_w / np.maximum(mid, 1e-9)       # 상대 불확실성(cv 유사)
    if "risk_pctile" in sub.columns:
        risk_pct = sub["risk_pctile"].to_numpy()
    else:
        rs = sub["risk_score"].to_numpy()
        risk_pct = 100.0 * (np.argsort(np.argsort(rs)) / max(len(rs) - 1, 1))

    # 체제 간 비교 가능하도록 공통 척도.
    vmax = float(np.quantile(abs_w, 0.999))
    rel_thr = float(np.quantile(rel_w, 0.80))   # '불확실 상위 20%' 전역 임계
    fx = pos["lon"].to_numpy() if pos.height else np.empty(0)
    fy = pos["lat"].to_numpy() if pos.height else np.empty(0)

    order_reg = ["yeongdong", "yeongseo", "mountain"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    sc = None
    for j, rk in enumerate(order_reg):
        m = reg == rk
        if not m.any():
            continue
        x, y = lon[m], lat[m]
        padx = 0.03 * (x.max() - x.min() + 1e-6)
        pady = 0.03 * (y.max() - y.min() + 1e-6)
        xlim = (x.min() - padx, x.max() + padx)
        ylim = (y.min() - pady, y.max() + pady)
        fin = (fx >= xlim[0]) & (fx <= xlim[1]) & (fy >= ylim[0]) & (fy <= ylim[1])

        # ── 상단: 절대 사후폭 ──
        ax = axes[0, j]
        o = np.argsort(abs_w[m])                # 좁은 폭 먼저 → 넓은(불확실) 폭 위로
        sc = ax.scatter(x[o], y[o], c=abs_w[m][o], s=1.4, cmap="viridis",
                        vmin=0.0, vmax=vmax, rasterized=True, zorder=1)
        _draw_admin(ax, admin_rings)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_title(f"{REGIME_KO.get(rk, rk)} — 사후 신뢰폭", fontsize=12, fontweight="bold")
        ax.set_xlabel("경도 (°E)"); ax.set_ylabel("위도 (°N)")

        # ── 하단: 현장확인 우선 ──
        ax = axes[1, j]
        pri = m & (risk_pct >= 90.0) & (rel_w >= rel_thr)
        oth = m & ~pri
        ax.scatter(lon[oth], lat[oth], s=0.8, c="#cfd8dc", rasterized=True,
                   zorder=1, label="기타 전주")
        ax.scatter(lon[pri], lat[pri], s=2.2, c="#6a1b9a", rasterized=True,
                   zorder=3, label=f"현장확인 우선 {int(pri.sum()):,}개")
        _draw_admin(ax, admin_rings)
        if fin.any():
            ax.scatter(fx[fin], fy[fin], marker="*", s=60, c="#1a1a1a",
                       edgecolors="white", linewidths=0.4, zorder=5,
                       label=f"발화점 {int(fin.sum())}건")
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_title(f"{REGIME_KO.get(rk, rk)} — 위험∩불확실(현장확인 우선)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("경도 (°E)"); ax.set_ylabel("위도 (°N)")
        ax.legend(loc="lower right", fontsize=7.5, framealpha=0.9, markerscale=3)

    if sc is not None:
        cb = fig.colorbar(sc, ax=axes[0, :].tolist(), fraction=0.025, pad=0.02)
        cb.set_label("사후 신뢰폭  risk_hi - risk_lo  (90% credible)", fontsize=10)

    fig.suptitle("그림 11. 체제별 사후 불확실성 (MC 베이지안 credible) — 상단 신뢰폭 · 하단 현장확인 우선",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, "fig11_uncertainty_map.png")


# ──────────────────────────────────────────── main ─────────────────────────
def main():
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    setup_korean_font()

    rj = json.loads(REGIME_JSON.read_text())
    sub = load_submission()
    pos = load_positives()
    admin_rings = load_admin_polygons(gangwon_only=True)

    log.info("=== fig1 위험지도 ===")
    fig1_risk_map(sub, admin_rings, pos)
    log.info("=== fig2 체제·편중완화 ===")
    fig2_regime_alloc(sub, admin_rings, rj)
    log.info("=== fig3 recall@top-k ===")
    fig3_recall_topk(rj)
    log.info("=== fig4 F1 민감도 ===")
    fig4_f1_sensitivity(rj)
    log.info("=== fig5 고성 2019 ===")
    fig5_goseong(sub, pos)
    log.info("=== fig6 발화 분해 ===")
    fig6_ignition_decomp()

    # Phase-5: 시군 서브플롯 · 불확실성 · 커버리지
    sgg_df = load_pole_sgg().sort("pole_id")
    # 제출은 pole_id 정렬이므로 행 순서가 일치(검증: 길이 동일).
    if sgg_df.height == sub.height:
        sgg_of_pole = sgg_df["sgg"].to_numpy()
    else:
        log.warning("pole_sgg 행수 != 제출 — pole_id 조인으로 정합")
        joined = sub.select("pole_id").join(sgg_df, on="pole_id", how="left")
        sgg_of_pole = joined["sgg"].to_numpy()
    log.info("=== fig7 시군 위험 서브플롯 ===")
    fig7_sgg_risk_subplots(sub, sgg_of_pole)
    log.info("=== fig8 시군 불확실성 서브플롯 ===")
    fig8_sgg_uncertainty_subplots(sub, sgg_of_pole)
    log.info("=== fig11 강원 전역 불확실성 지도 ===")
    fig11_uncertainty_map(sub, admin_rings, pos)
    log.info("=== fig9 체제별 커버리지 ===")
    fig9_coverage(rj)
    log.info("=== fig10 BYM2 vs Poisson-Gamma(공간 borrow) ===")
    fig10_bym_vs_pg(rj)

    log.info("완료 — outputs/figures/ 생성(BYM 백엔드면 fig10 포함)")


if __name__ == "__main__":
    main()
