"""exposure_v2 v2.1(LWR 타원) + v2.2(국지 baseline 정규화) — 회랑 날카롭게.

v2.1 타원 커널 (Anderson 1983 LWR · Richards 1990 타원):
  발화원→전주 변위를 풍하방향 u 로 분해: d∥(풍하성분)·d⊥(횡성분).
  reach = exp(−√[(d∥/L)² + (d⊥/W)²]),  L/W = LWR(풍속)(상한 8),
  풍하(d∥≥0)는 길게(L), 풍상(d∥<0)은 짧게(L_back), 횡(W)은 좁게 → 좁고 긴 회랑.
v2.2 국지정규화: dose / (격자셀 평균 dose)  → "영동 다 높음" 배경 제거(상대 노출).

비교: v2.0 iso(저장 parquet) vs v2.1 타원 vs v2.2 타원+정규화.
측정: 영동 IQR(변별)·회랑 집중도(상위10% dose 점유율)·2019 방향·누수안전 CV.

실행:  .venv/bin/python exposure_v2/proto_v1_ellipse.py   (백그라운드 권장, ~3분)
산출:  outputs/pole_dose_v2_ellipse.parquet + exposure_v2/fig_v1_corridor.png
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pfire import (config, calibrate, experts, exposure_engine, geo, io,  # noqa: E402
                   regimes, validate)
from pfire.risk_index import risk_percentile  # noqa: E402
from sweep_threshold_figs import setup_korean_font, load_admin_polygons, draw_admin  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("v1ell")

S = 16
MAXD = config.SPREAD_MAX_DIST_KM
BETA = config.SPREAD_SOUTHNESS_BETA
W_CROSS = 0.35      # 횡풍 스케일(km) — 작을수록 좁은 회랑
L_BACK = 0.35       # 풍상(backing) 스케일(km)
NORM_CELL_KM = 4.0  # v2.2 국지정규화 격자 셀
KS = config.TOPK_FOR_RECALL


def anderson_lwr(ws_ms):
    """Anderson(1983) 길이/폭비. ws m/s → mph. 상한 8."""
    U = np.asarray(ws_ms) * 2.237
    lwr = 0.936 * np.exp(0.2566 * U) + 0.461 * np.exp(-0.1548 * U) - 0.397
    return np.minimum(lwr, 8.0)


def main() -> int:
    master = io.load_master()
    pole_xy = master.select(["lon", "lat"]).to_numpy().astype(np.float64)
    positives = io.load_positives()
    gate, order = regimes.compute_gate(master)
    regime_lbl = np.array(order)[gate.argmax(1)]
    I, _ = experts.ignition_propensity(master, gate, order)
    from pfire import weather
    R_ign = np.clip(I * np.clip(master["S_p"].to_numpy().astype(np.float64), 0, 1)
                    * weather.season_weather(master, source="features"), 0, 1)

    sources = np.asarray(exposure_engine.ignition_candidates(I, gate, order), dtype=np.int64)
    Ig = np.clip(I[sources], 1e-6, None)
    km = geo.lonlat_to_km(pole_xy)
    src_km = km[sources]
    fuel = np.clip(master["mu_flammability"].to_numpy().astype(np.float64), 0, 1)
    south_fac = 1.0 + BETA * (master["mu_southness"].to_numpy().astype(np.float64) * 2 - 1)

    station_daily = io.load_station_daily()
    aws_daily = io.load_aws_daily()
    winds = {}
    for r in order:
        wd, ws = exposure_engine.sample_wind(r, S, station_daily, aws_daily, seed=config.SEED)
        winds[r] = (np.deg2rad(wd + 180.0), np.asarray(ws))

    logger.info("cKDTree + 반경질의 …")
    tree = cKDTree(km)
    nbr_lists = tree.query_ball_point(src_km, MAXD, workers=-1)

    dose_r = {r: np.zeros(km.shape[0]) for r in order}
    logger.info("타원 도즈 누적(발화원 %d × 체제 %d × 바람 %d) …", src_km.shape[0], len(order), S)
    for gi in range(src_km.shape[0]):
        nbr = np.asarray(nbr_lists[gi], dtype=np.int64)
        if nbr.size == 0:
            continue
        dx = km[nbr, 0] - src_km[gi, 0]
        dy = km[nbr, 1] - src_km[gi, 1]
        d2 = dx * dx + dy * dy
        ff = fuel[nbr] * south_fac[nbr]
        for r in order:
            theta, ws = winds[r]
            ct, st = np.cos(theta), np.sin(theta)                 # (S,)
            d_par = dx[:, None] * ct[None, :] + dy[:, None] * st[None, :]   # (K,S)
            d_perp = np.sqrt(np.clip(d2[:, None] - d_par * d_par, 0.0, None))
            L_along = (anderson_lwr(ws) * W_CROSS)[None, :]        # (1,S)
            along = np.where(d_par >= 0, L_along, L_BACK)
            rr = np.sqrt((d_par / np.maximum(along, 1e-6)) ** 2
                         + (d_perp / W_CROSS) ** 2)
            reach = np.exp(-rr).mean(1) * ff
            dose_r[r][nbr] += Ig[gi] * np.clip(reach, 0.0, 1.0)

    dose_ell = np.zeros(km.shape[0])
    for ri, r in enumerate(order):
        dose_ell += gate[:, ri] * dose_r[r]

    # v2.2 국지정규화(격자셀 평균 대비)
    ix = np.floor((km[:, 0] - km[:, 0].min()) / NORM_CELL_KM).astype(np.int64)
    iy = np.floor((km[:, 1] - km[:, 1].min()) / NORM_CELL_KM).astype(np.int64)
    cell = iy * (ix.max() + 1) + ix
    csum = np.bincount(cell, weights=dose_ell)
    ccnt = np.bincount(cell)
    cmean = csum / np.maximum(ccnt, 1)
    dose_ell_rel = dose_ell / (cmean[cell] + 1e-9)

    # v2.0 iso(저장본) 로드
    iso = pl.read_parquet(config.OUT / "pole_dose_v2.parquet").sort("pole_id")
    dose_iso01 = iso["dose01"].to_numpy()

    variants = {
        "v2.0 iso": dose_iso01,
        "v2.1 타원": risk_percentile(dose_ell) / 100.0,
        "v2.2 타원+정규화": risk_percentile(dose_ell_rel) / 100.0,
    }

    yd = regime_lbl == "yeongdong"

    def iqr(x):
        return float(np.subtract(*np.quantile(x, [0.75, 0.25])))

    def conc(x):  # 상위10% 점유율(집중도) — 높을수록 회랑 집중
        thr = np.quantile(x, 0.90)
        return float(x[x >= thr].sum() / (x.sum() + 1e-9))

    logger.info("=== 회랑 날카로움 비교 ===")
    logger.info("변형".ljust(18) + "영동IQR".rjust(9) + "집중도(top10%)".rjust(14) + "  2019동향")
    fire_xy = positives.select(["lon", "lat"]).to_numpy().astype(np.float64)
    for name, v in variants.items():
        dw = validate.exposure_downwind_goseong(pole_xy, v)
        logger.info(name.ljust(18) + f"{iqr(v[yd]):.3f}".rjust(9)
                    + f"{conc(v):.3f}".rjust(14) + f"   east={dw['east_frac']:.2f}")

    # 누수안전 CV (참고: recall 은 확산이라 큰 변화 기대 안 함)
    f2p = calibrate.assign_poles_to_fires(pole_xy, fire_xy, radius_km=1.0)
    blocks = validate.spatial_blocks(pole_xy)
    logger.info("=== 누수안전 CV (OR w=0.5, 참고) ===")
    base = validate.spatial_cv_recall(R_ign, f2p, blocks)
    logger.info("  baseline 평균=%.4f", np.nanmean([base[k]["mean"] for k in KS]))
    for name, v in variants.items():
        Ror = np.clip(1 - (1 - R_ign) * (1 - 0.5 * v), 0, 1)
        cv = validate.spatial_cv_recall(Ror, f2p, blocks)
        logger.info("  +%s 평균=%.4f", name, np.nanmean([cv[k]["mean"] for k in KS]))

    # 저장
    pl.DataFrame({"pole_id": master["pole_id"].to_numpy(),
                  "dose_ellipse": np.round(dose_ell, 5),
                  "dose_ellipse01": np.round(variants["v2.1 타원"], 4),
                  "dose_ellipse_rel01": np.round(variants["v2.2 타원+정규화"], 4)}
                 ).write_parquet(config.OUT / "pole_dose_v2_ellipse.parquet")

    # 그림: 3패널 회랑 비교
    setup_korean_font()
    admin = load_admin_polygons()
    lon, lat = pole_xy[:, 0], pole_xy[:, 1]
    gx, gy = config.GOSEONG_2019_LONLAT
    fig, axes = plt.subplots(1, 3, figsize=(19, 7.5))
    for a, (name, v) in zip(axes, variants.items()):
        sc = a.scatter(lon, lat, c=v, s=0.3, cmap="inferno", vmin=0, vmax=1,
                       rasterized=True, linewidths=0)
        draw_admin(a, admin)
        a.scatter([gx], [gy], marker="*", s=200, c="#39ff14", edgecolors="black",
                  linewidths=0.7, zorder=5)
        a.set_title(f"{name}\n영동IQR={iqr(v[yd]):.3f}", fontsize=12, fontweight="bold")
        a.set_xlabel("경도"); a.set_aspect("equal", adjustable="datalim")
    axes[0].set_ylabel("위도")
    fig.suptitle("풍하 노출 회랑 날카롭게: v2.0 iso → v2.1 LWR타원 → v2.2 +국지정규화",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "fig_v1_corridor.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    logger.info("그림 저장: %s | === 완료 ===", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
