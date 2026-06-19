"""exposure_v2 v2.2 확정 검증 — w 스윕 × 5 fold-seed (신호 vs 잡음) + 낙관편향 격차.

v2.2 dose(국지정규화)를 R 발화와 OR 결합:  R(w)=1−(1−R발화)(1−w·dose).
"+0.004 recall" 이 진짜인지: **5개 fold-seed 의 mean±std** 로 baseline 대비 Δ 가
잡음대(std)를 넘는지 판정. + 무작위-공간 격차(편향) 확인.

실행:  .venv/bin/python exposure_v2/proto_v2_confirm.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pfire import config, calibrate, experts, io, regimes, validate, weather  # noqa: E402
from prototype_sgg_pooling import cv_recall, make_folds, KS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("v2confirm")

WS = (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS = (20260618, 1, 2, 3, 4)


def main() -> int:
    master = io.load_master()
    pole_xy = master.select(["lon", "lat"]).to_numpy().astype(np.float64)
    positives = io.load_positives()
    gate, order = regimes.compute_gate(master)
    I, _ = experts.ignition_propensity(master, gate, order)
    S = np.clip(master["S_p"].to_numpy().astype(np.float64), 0, 1)
    W = weather.season_weather(master, source="features")
    R_ign = np.clip(I * S * W, 0.0, 1.0)

    ell = pl.read_parquet(config.OUT / "pole_dose_v2_ellipse.parquet").sort("pole_id")
    dose = ell["dose_ellipse_rel01"].to_numpy()   # v2.2 국지정규화 [0,1]

    fire_xy = positives.select(["lon", "lat"]).to_numpy().astype(np.float64)
    f2p = calibrate.assign_poles_to_fires(pole_xy, fire_xy, radius_km=1.0)
    blocks = validate.spatial_blocks(pole_xy)

    def OR(w):
        return np.clip(1.0 - (1.0 - R_ign) * (1.0 - w * dose), 0.0, 1.0)

    # 5 seed × w 스윕: 각 평균 recall(top-k 평균) 수집
    logger.info("=== v2.2 OR — w 스윕 × %d fold-seed (평균 recall±std) ===", len(SEEDS))
    Rw = {w: OR(w) for w in WS}
    seed_scores = {w: [] for w in WS}
    for sd in SEEDS:
        fold = make_folds(blocks, seed=sd)
        for w in WS:
            cv = cv_recall(Rw[w], fold, f2p, KS, None)
            seed_scores[w].append(float(np.nanmean([cv[k] for k in KS])))
    base_mean = np.mean(seed_scores[0.0])
    base_std = np.std(seed_scores[0.0])
    logger.info("  w      평균recall   std      Δ vs baseline   신호?(|Δ|>합성std)")
    for w in WS:
        m, sdv = np.mean(seed_scores[w]), np.std(seed_scores[w])
        delta = m - base_mean
        combined = np.hypot(sdv, base_std)
        sig = "신호" if (w > 0 and delta > combined) else ("잡음내" if w > 0 else "기준")
        logger.info("  %.2f   %.4f    ±%.4f   %+.4f         %s (합성std=%.4f)",
                    w, m, sdv, delta, sig, combined)

    # 최선 w 의 top-k 상세 + 무작위-공간 격차
    best_w = max([w for w in WS if w > 0], key=lambda w: np.mean(seed_scores[w]))
    logger.info("=== 최선 w=%.2f 상세(seed=%d) + 낙관편향 격차 ===", best_w, config.SEED)
    fold0 = make_folds(blocks, seed=config.SEED)
    cv_b = cv_recall(R_ign, fold0, f2p, KS, None)
    cv_w = cv_recall(Rw[best_w], fold0, f2p, KS, None)
    for k in KS:
        logger.info("  top%-3d%%  baseline=%.4f  +v2.2(w=%.2f)=%.4f  Δ=%+.4f",
                    int(100 * k), cv_b[k], best_w, cv_w[k], cv_w[k] - cv_b[k])
    gap_b = validate.random_vs_spatial_gap(R_ign, f2p, blocks, k=0.05)
    gap_w = validate.random_vs_spatial_gap(Rw[best_w], f2p, blocks, k=0.05)
    logger.info("  무작위-공간 격차@5%%: baseline=%.4f  +v2.2=%.4f  (≈0이면 낙관편향 없음)",
                gap_b["gap"], gap_w["gap"])
    logger.info("=== 완료 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
