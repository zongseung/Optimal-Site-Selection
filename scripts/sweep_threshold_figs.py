"""임계값(양성비율) 5~15% 스윕 — 컷마다 별도 폴더에 그림 저장.

목적: "위험(1)으로 상위 몇 %를 잡을까"를 5/7.5/10/12.5/15% 로 바꿔가며
  ① decision 공간 지도, ② 위험분포+컷선+지표(recall@cut·proxy-F1·체제구성)를
  컷마다 다른 폴더(outputs/threshold_sweep/prev_XX/)에 그리고,
  전체 비교용 민감도 곡선을 _summary/ 에 그린다.

위험점수 R 은 현재 운영 파이프라인과 동일하게 재계산한다:
  R = I(MoE) × S(S_p) × W(blended) × 지역배율(hierarchy)  → 전역 top-k 컷.
재현 검증: top-10% 컷이 기존 submission_p0.1.csv 의 decision 과 일치하는지 자가확인.

실행:
    .venv/bin/python scripts/sweep_threshold_figs.py
"""
from __future__ import annotations

import glob
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pfire import (  # noqa: E402
    calibrate,
    config,
    experts,
    hierarchy,
    io,
    regimes,
    hazard,
    weather,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("sweep")

PREVALENCES = (0.05, 0.075, 0.10, 0.125, 0.15)
OUT = config.OUT / "threshold_sweep"
ADMIN_CSV = ROOT / "used_dataset" / "admin" / "admin_gangwon.csv"

REGIME_COLORS = {"yeongdong": "#1f77b4", "yeongseo": "#2ca02c", "mountain": "#8c564b"}
REGIME_KO = {"yeongdong": "영동", "yeongseo": "영서", "mountain": "산간"}


# ── 한글 폰트 ────────────────────────────────────────────────────────────
def setup_korean_font() -> None:
    cands = sorted(glob.glob("/usr/share/fonts/truetype/nanum/*.ttf")) + \
            sorted(glob.glob("/usr/share/fonts/opentype/noto/NotoSansCJK*.*"))
    name = None
    for p in cands:
        try:
            fm.fontManager.addfont(p)
            fp = fm.FontProperties(fname=p)
            n = fp.get_name()
            if name is None and ("Nanum" in n or "Noto Sans" in n):
                name = n
        except Exception:
            continue
    if name:
        plt.rcParams["font.family"] = name
    plt.rcParams["axes.unicode_minus"] = False
    for k in ("figure.facecolor", "axes.facecolor", "savefig.facecolor"):
        plt.rcParams[k] = "white"
    logger.info("한글 폰트 = %s", name)


# ── admin WKT 경계 ───────────────────────────────────────────────────────
def _parse_wkt_multipolygon(wkt: str) -> list[np.ndarray]:
    rings: list[np.ndarray] = []
    body = wkt[wkt.find("("):]
    import re
    for grp in re.findall(r"\(([-0-9.,\s]+)\)", body):
        pts = []
        for pair in grp.split(","):
            xy = pair.split()
            if len(xy) == 2:
                pts.append((float(xy[0]), float(xy[1])))
        if len(pts) >= 3:
            rings.append(np.array(pts))
    return rings


def load_admin_polygons() -> list[np.ndarray]:
    if not ADMIN_CSV.exists():
        return []
    adm = pl.read_csv(ADMIN_CSV)
    col_cd = "sig_cd" if "sig_cd" in adm.columns else adm.columns[0]
    col_wkt = "wkt" if "wkt" in adm.columns else adm.columns[-1]
    rings: list[np.ndarray] = []
    for sig_cd, wkt in zip(adm[col_cd].cast(str).to_list(), adm[col_wkt].to_list()):
        if wkt and str(sig_cd).startswith("32"):
            rings.extend(_parse_wkt_multipolygon(wkt))
    return rings


def draw_admin(ax, rings, color="#888888", lw=0.6):
    for r in rings:
        ax.plot(r[:, 0], r[:, 1], color=color, lw=lw, alpha=0.8, zorder=3)


# ── R 재계산 (운영 파이프라인 동일) ──────────────────────────────────────
def compute_production_risk():
    logger.info("데이터 로드 + R 재계산 …")
    master = io.load_master()
    positives = io.load_positives()
    pole_xy = master.select(["lon", "lat"]).to_numpy().astype(np.float64)

    gate, regime_order = regimes.compute_gate(master)
    regime_lbl = np.array(regime_order)[gate.argmax(axis=1)]
    I, _ = experts.ignition_propensity(master, gate, regime_order)
    S = np.clip(master["S_p"].to_numpy().astype(np.float64), 0.0, 1.0)

    station_daily = io.load_station_daily()
    stations = io.load_stations()
    W = weather.blended_weather(master, station_daily, stations)
    R_phys = hazard.season_risk(I, S, W)

    mult = hierarchy.regional_multiplier(
        master, positives.select(["lon", "lat"]), normalize="mean")
    R_mult = np.clip(R_phys * mult, 0.0, 1.0)

    fire_xy = positives.select(["lon", "lat"]).to_numpy().astype(np.float64)
    fire_to_pole = calibrate.assign_poles_to_fires(pole_xy, fire_xy, radius_km=1.0)
    return master, pole_xy, regime_lbl, R_phys, R_mult, fire_to_pole, fire_xy


def _decision_topk(R: np.ndarray, prevalence: float) -> np.ndarray:
    _, dec = calibrate.decide_threshold(R, prevalence)
    return np.asarray(dec).astype(np.int8)


def pick_production_R(R_phys, R_mult) -> tuple[np.ndarray, str, float]:
    """기존 submission_p0.1 의 decision 과 top-10% 일치도가 높은 R 채택(재현 검증)."""
    ref = config.SUBMISSIONS / "submission_p0.1.csv"
    if not ref.exists():
        logger.warning("기준 submission_p0.1.csv 없음 → R=I·S·W×배율 사용")
        return R_mult, "R_mult(배율 채택·검증생략)", float("nan")
    dec_ref = pl.read_csv(ref)["decision"].to_numpy().astype(np.int8)
    best = None
    for name, R in (("R_mult(I·S·W×배율)", R_mult), ("R_phys(I·S·W)", R_phys)):
        dec = _decision_topk(R, 0.10)
        agree = float((dec == dec_ref).mean())
        logger.info("재현 검증 %s: top-10%% vs submission_p0.1 일치=%.4f", name, agree)
        if best is None or agree > best[2]:
            best = (R, name, agree)
    return best


# ── 그림 ─────────────────────────────────────────────────────────────────
def fig_decision_map(folder, pole_xy, decision, fire_xy, admin, prevalence, n_pos):
    fig, ax = plt.subplots(figsize=(7.6, 8.4))
    safe = decision == 0
    risky = decision == 1
    ax.scatter(pole_xy[safe, 0], pole_xy[safe, 1], s=0.25, c="#d9dde1",
               rasterized=True, linewidths=0, zorder=1)
    ax.scatter(pole_xy[risky, 0], pole_xy[risky, 1], s=0.9, c="#c1121f",
               rasterized=True, linewidths=0, zorder=2,
               label=f"위험(1) {n_pos:,}개")
    draw_admin(ax, admin)
    if fire_xy is not None and len(fire_xy):
        ax.scatter(fire_xy[:, 0], fire_xy[:, 1], marker="*", s=22,
                   c="#111111", edgecolors="white", linewidths=0.3, zorder=4,
                   label=f"과거 발화점(강원) {len(fire_xy)}")
    ax.set_title(f"강원 전주 위험판정 — 상위 {prevalence*100:.1f}% 위험\n"
                 f"(decision=1: {n_pos:,} / 1,387,831)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("경도"); ax.set_ylabel("위도")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9, markerscale=4)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()
    fig.savefig(folder / "decision_map.png", dpi=130)
    plt.close(fig)


def fig_metrics(folder, R, decision, regime_lbl, prevalence, m):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    # (좌) 위험분포 + 컷선
    ax = axes[0]
    thr = m["threshold"]
    Rpos = R[R > 0]
    ax.hist(np.log10(Rpos + 1e-12), bins=80, color="#9ecae1", edgecolor="white")
    ax.axvline(np.log10(thr + 1e-12), color="#c1121f", lw=2,
               label=f"컷 thr={thr:.2e}")
    ax.set_title(f"위험점수 R 분포 (log10) — 상위 {prevalence*100:.1f}% 컷",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("log10 R"); ax.set_ylabel("전주 수")
    ax.legend(fontsize=9)
    # (우) 체제 구성 + 지표 텍스트
    ax = axes[1]
    flagged = decision == 1
    regs = ["yeongdong", "yeongseo", "mountain"]
    counts = [int(((regime_lbl == r) & flagged).sum()) for r in regs]
    ax.bar([REGIME_KO[r] for r in regs], counts,
           color=[REGIME_COLORS[r] for r in regs])
    for i, c in enumerate(counts):
        ax.text(i, c, f"{c:,}", ha="center", va="bottom", fontsize=10)
    ax.set_title("위험(1) 전주의 체제 구성", fontsize=12, fontweight="bold")
    ax.set_ylabel("위험 전주 수")
    txt = (f"양성비율 π = {prevalence*100:.1f}%\n"
           f"위험(1) 전주 = {int(flagged.sum()):,}\n"
           f"발화점 recall = {m['recall']:.3f}\n"
           f"proxy-F1 = {m['f1_proxy']:.3f}\n"
           f"(무작위 대비 recall lift ≈ {m['recall']/prevalence:.2f}×)")
    ax.text(0.98, 0.97, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=10.5, bbox=dict(boxstyle="round", fc="#fff6e0", ec="#da3"))
    fig.suptitle(f"임계값 진단 — 상위 {prevalence*100:.1f}%",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(folder / "metrics.png", dpi=130)
    plt.close(fig)


def fig_summary(curve: dict):
    ps = [p * 100 for p in curve]
    recall = [curve[p]["recall"] for p in curve]
    f1 = [curve[p]["f1_proxy"] for p in curve]
    npos = [curve[p]["predicted_positive"] for p in curve]
    lift = [curve[p]["recall"] / p for p in curve]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(ps, recall, "o-", color="#1f77b4", label="발화점 recall@컷")
    ax.plot(ps, f1, "s-", color="#c1121f", label="proxy-F1")
    ax.set_xlabel("양성비율 π (위험으로 잡는 상위 %)")
    ax.set_ylabel("값")
    ax.set_title("(a) recall · proxy-F1 vs 임계값 (5~15%)",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=10)
    for x, y in zip(ps, f1):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8, color="#c1121f")

    ax = axes[1]
    ax.plot(ps, lift, "d-", color="#2ca02c")
    ax.axhline(1.0, color="#999", ls="--", lw=1, label="무작위(=1×)")
    ax.set_xlabel("양성비율 π (위험으로 잡는 상위 %)")
    ax.set_ylabel("recall lift (무작위 대비 배수)")
    ax.set_title("(b) 무작위 대비 recall lift", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend(fontsize=10)
    for x, y, n in zip(ps, lift, npos):
        ax.annotate(f"{y:.2f}×\n{int(n/1000)}k", (x, y),
                    textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8)
    fig.suptitle("임계값 민감도 — 상위 5~15% (전역 단일컷)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    sd = OUT / "_summary"
    sd.mkdir(parents=True, exist_ok=True)
    fig.savefig(sd / "sensitivity_5_15.png", dpi=140)
    plt.close(fig)


def main() -> int:
    setup_korean_font()
    admin = load_admin_polygons()
    logger.info("admin 경계 ring=%d", len(admin))

    master, pole_xy, regime_lbl, R_phys, R_mult, fire_to_pole, fire_xy = \
        compute_production_risk()
    R, rname, agree = pick_production_R(R_phys, R_mult)
    logger.info("채택 R = %s (재현 일치 %.4f)", rname, agree)

    curve = calibrate.f1_sensitivity(R, fire_to_pole, prevalence_grid=PREVALENCES)

    # 지도용 발화점: 강원 박스 안만(타지역 좌표오염 3개 제외 — 모델엔 이미 미사용).
    _m = ((fire_xy[:, 1] >= 36.9) & (fire_xy[:, 1] <= 38.7) &
          (fire_xy[:, 0] >= 126.9) & (fire_xy[:, 0] <= 129.6))
    fire_xy_plot = fire_xy[_m]
    logger.info("지도 발화점: 강원 박스 %d/%d (타지역 %d 제외)",
                len(fire_xy_plot), len(fire_xy), int((~_m).sum()))

    summary_rows = []
    for p in PREVALENCES:
        m = curve[p]
        decision = _decision_topk(R, p)
        tag = f"prev_{str(round(p*100,1)).replace('.0','').replace('.','')}"
        folder = OUT / tag
        folder.mkdir(parents=True, exist_ok=True)
        logger.info("[%s] π=%.1f%% 위험=%d recall=%.3f F1p=%.3f → %s",
                    tag, p * 100, int(decision.sum()), m["recall"],
                    m["f1_proxy"], folder)
        fig_decision_map(folder, pole_xy, decision, fire_xy_plot, admin, p,
                         int(decision.sum()))
        fig_metrics(folder, R, decision, regime_lbl, p, m)
        rec = dict(prevalence=p, n_positive=int(decision.sum()),
                   threshold=m["threshold"], recall=m["recall"],
                   f1_proxy=m["f1_proxy"], recall_lift=m["recall"] / p,
                   regime={r: int(((regime_lbl == r) & (decision == 1)).sum())
                           for r in ("yeongdong", "yeongseo", "mountain")})
        (folder / "metrics.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2))
        summary_rows.append(rec)

    fig_summary(curve)
    sd = OUT / "_summary"
    (sd / "summary.json").write_text(json.dumps(
        dict(adopted_R=rname, reproduction_agreement=agree, rows=summary_rows),
        ensure_ascii=False, indent=2))
    logger.info("=== 완료 === 폴더: %s", OUT)
    for r in summary_rows:
        logger.info("  π=%.1f%% | 위험 %7d | recall=%.3f | F1p=%.3f | lift=%.2f×",
                    r["prevalence"] * 100, r["n_positive"], r["recall"],
                    r["f1_proxy"], r["recall_lift"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
