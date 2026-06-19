"""exposure_v2 v2.0 full-N — 발화가중 기대도즈(전 전주) + 누수안전 CV 기여 검증.

dose(p) = Σ_r gate_r(p) · Σ_g I_g · mean_s[exp(−d/L_s)] · fuel_p·(1+β·south_p)
  (체제별 바람 r 로 도즈 산출 후 gate 결합 — 배포 compute_exposure 구조와 동형, OR→합)
이웃질의는 scipy cKDTree(반경 5km). dose01 = 전역 순위 백분위/100.

검증: baseline R_발화 vs R = 1−(1−R)(1−w·dose01) 의 **누수안전 공간CV recall@top-k**
  (R 에 발화데이터 없음 → 고정 R 평가) + 영동 top2% share + 2019 고성 방향 sanity.

실행:  .venv/bin/python exposure_v2/proto_v0_fulln_cv.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pfire import (config, experts, exposure_engine, geo, io, regimes,  # noqa: E402
                   validate)
from pfire.risk_index import risk_percentile  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("v0fulln")

S = 16
MAXD = config.SPREAD_MAX_DIST_KM
L0 = config.SPREAD_LENGTH_SCALE_KM
ALPHA = config.SPREAD_WIND_ANISO
BETA = config.SPREAD_SOUTHNESS_BETA
KS = config.TOPK_FOR_RECALL


def main() -> int:
    master = io.load_master()
    pole_xy = master.select(["lon", "lat"]).to_numpy().astype(np.float64)
    positives = io.load_positives()
    gate, order = regimes.compute_gate(master)
    regime_lbl = np.array(order)[gate.argmax(1)]
    I, _ = experts.ignition_propensity(master, gate, order)
    R_ign = np.clip((I * np.clip(master["S_p"].to_numpy().astype(np.float64), 0, 1)
                     * _Wseason(master)), 0.0, 1.0)

    sources = np.asarray(exposure_engine.ignition_candidates(I, gate, order), dtype=np.int64)
    Ig = np.clip(I[sources], 1e-6, None)
    km = geo.lonlat_to_km(pole_xy)
    src_km = km[sources]
    fuel = np.clip(master["mu_flammability"].to_numpy().astype(np.float64), 0, 1)
    south_fac = 1.0 + BETA * (master["mu_southness"].to_numpy().astype(np.float64) * 2 - 1)

    station_daily = io.load_station_daily()
    aws_daily = io.load_aws_daily()
    # 체제별 바람(배포와 동일): 영동 양간 prior, 그 외 관측.
    winds = {}
    for r in order:
        wd, ws = exposure_engine.sample_wind(r, S, station_daily, aws_daily, seed=config.SEED)
        winds[r] = (np.deg2rad(wd + 180.0), ws)

    logger.info("cKDTree 구축(N=%d) + 반경질의(원=%d, r=%.1fkm) …", km.shape[0], src_km.shape[0], MAXD)
    tree = cKDTree(km)
    nbr_lists = tree.query_ball_point(src_km, MAXD, workers=-1)

    dose_r = {r: np.zeros(km.shape[0]) for r in order}
    logger.info("도즈 누적(발화원 %d × 체제 %d × 바람 %d) …", src_km.shape[0], len(order), S)
    for gi in range(src_km.shape[0]):
        nbr = np.asarray(nbr_lists[gi], dtype=np.int64)
        if nbr.size == 0:
            continue
        dxy = km[nbr] - src_km[gi]
        d = np.sqrt((dxy * dxy).sum(1))
        phi = np.arctan2(dxy[:, 1], dxy[:, 0])
        ff = fuel[nbr] * south_fac[nbr]
        for r in order:
            theta, ws = winds[r]
            align = np.cos(phi[:, None] - theta[None, :])
            L = L0 * (1.0 + ALPHA * np.maximum(0.0, align) * (ws[None, :] / 5.0))
            reach = np.exp(-d[:, None] / np.maximum(L, 1e-6)).mean(1) * ff
            dose_r[r][nbr] += Ig[gi] * np.clip(reach, 0.0, 1.0)

    dose = np.zeros(km.shape[0])
    for ri, r in enumerate(order):
        dose += gate[:, ri] * dose_r[r]
    dose01 = risk_percentile(dose) / 100.0
    logger.info("dose: mean=%.4g p99=%.4g frac>0=%.3f | 영동내 dose01 std=%.3f IQR=%.3f",
                dose.mean(), np.quantile(dose, .99), float(np.mean(dose > 0)),
                float(dose01[regime_lbl == "yeongdong"].std()),
                float(np.subtract(*np.quantile(dose01[regime_lbl == "yeongdong"], [.75, .25]))))

    # ── 누수안전 CV: baseline vs OR 결합 ──
    fire_xy = positives.select(["lon", "lat"]).to_numpy().astype(np.float64)
    from pfire import calibrate
    f2p = calibrate.assign_poles_to_fires(pole_xy, fire_xy, radius_km=1.0)
    blocks = validate.spatial_blocks(pole_xy)

    def OR(w):
        return np.clip(1.0 - (1.0 - R_ign) * (1.0 - w * dose01), 0.0, 1.0)

    def yd_share(R, prev=0.02):
        cut = max(1, int(np.ceil(prev * R.shape[0])))
        top = np.argsort(-R)[:cut]
        return float(np.mean(regime_lbl[top] == "yeongdong"))

    variants = [("baseline R_발화", R_ign)]
    for w in (0.25, 0.5, 1.0):
        variants.append((f"OR dose (w={w:g})", OR(w)))

    logger.info("=== 누수안전 공간CV recall@top-k + 영동 top2%% share ===")
    logger.info("변형".ljust(20) + "".join(f"top{int(100*k)}%".rjust(9) for k in KS) + "   평균   영동share")
    for name, Rv in variants:
        cv = validate.spatial_cv_recall(Rv, f2p, blocks)
        rec = {k: cv[k]["mean"] for k in KS}
        avg = float(np.nanmean([rec[k] for k in KS]))
        logger.info(name.ljust(20) + "".join(f"{rec[k]:.4f}".rjust(9) for k in KS)
                    + f"   {avg:.4f}   {yd_share(Rv):.3f}")

    dw = validate.exposure_downwind_goseong(pole_xy, dose)
    logger.info("2019 고성 방향: 노출전주 %d/%d east_frac=%.2f mean_dlon=%+.4f (dose가 동향?)",
                int(dw["n_exposed"]), int(dw["n_local"]), dw["east_frac"], dw["mean_dlon_deg"])

    # dose 저장(재사용)
    out = config.OUT / "pole_dose_v2.parquet"
    pl.DataFrame({"pole_id": master["pole_id"].to_numpy(),
                  "dose": dose, "dose01": np.round(dose01, 4)}).write_parquet(out)
    logger.info("저장: %s | === 완료 ===", out)
    return 0


def _Wseason(master):
    from pfire import weather
    return weather.season_weather(master, source="features")


if __name__ == "__main__":
    raise SystemExit(main())
