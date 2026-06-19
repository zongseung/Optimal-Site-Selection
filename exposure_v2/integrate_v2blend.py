"""exposure_v2 통합 — v2.2 dose 를 결정 위험에 OR 결합(w=0.5) standalone.

R_blend = 1 − (1 − R발화)(1 − 0.5·dose_v2.2)   → 체제별 배분 컷 → decision.
파이프라인(run_phase1_mvp/pfire)을 *수정하지 않고* 결과를 산출(다른 세션 충돌 회피).
하단에 run_phase1_mvp 병합용 패치 스니펫 출력.

실행:  .venv/bin/python exposure_v2/integrate_v2blend.py
산출:  outputs/pole_risk_v2blend.parquet
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
from pfire.risk_index import risk_percentile, risk_percentile_by_group  # noqa: E402
from prototype_sgg_pooling import cv_recall, make_folds, KS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("v2blend")

W_BLEND = 0.5
PREV = 0.02


def main() -> int:
    master = io.load_master()
    pole_xy = master.select(["lon", "lat"]).to_numpy().astype(np.float64)
    positives = io.load_positives()
    gate, order = regimes.compute_gate(master)
    regime_lbl = np.array(order)[gate.argmax(1)]
    I, _ = experts.ignition_propensity(master, gate, order)
    S = np.clip(master["S_p"].to_numpy().astype(np.float64), 0, 1)
    W = weather.season_weather(master, source="features")
    R_ign = np.clip(I * S * W, 0.0, 1.0)

    dose = pl.read_parquet(config.OUT / "pole_dose_v2_ellipse.parquet") \
        .sort("pole_id")["dose_ellipse_rel01"].to_numpy()
    R_blend = np.clip(1.0 - (1.0 - R_ign) * (1.0 - W_BLEND * dose), 0.0, 1.0)

    # 결정: 체제별 배분(regime_count, 배포와 동일) — baseline vs blend
    _, dec_base = calibrate.decide_threshold_per_regime(
        R_ign, regime_lbl, PREV, calibrate.ALLOC_COUNT, anchor_count=None)
    _, dec_blend = calibrate.decide_threshold_per_regime(
        R_blend, regime_lbl, PREV, calibrate.ALLOC_COUNT, anchor_count=None)

    logger.info("=== v2.2 통합 결정(체제배분 prev=%.0f%%) ===", PREV * 100)
    logger.info("  양성: baseline=%d  blend=%d  (변동 전주=%d, %.2f%%)",
                int(dec_base.sum()), int(dec_blend.sum()),
                int((dec_base != dec_blend).sum()),
                100.0 * (dec_base != dec_blend).mean())
    for r in order:
        m = regime_lbl == r
        logger.info("    %s: baseline 양성=%d → blend 양성=%d",
                    r, int(dec_base[m].sum()), int(dec_blend[m].sum()))

    # recall 확정(기본 fold)
    fire_xy = positives.select(["lon", "lat"]).to_numpy().astype(np.float64)
    f2p = calibrate.assign_poles_to_fires(pole_xy, fire_xy, radius_km=1.0)
    blocks = validate.spatial_blocks(pole_xy)
    fold = make_folds(blocks, seed=config.SEED)
    cb = cv_recall(R_ign, fold, f2p, KS, None)
    cw = cv_recall(R_blend, fold, f2p, KS, None)
    logger.info("=== 공간CV recall: baseline vs v2.2 blend ===")
    for k in KS:
        logger.info("  top%-3d%%  %.4f → %.4f  (Δ%+.4f)", int(100 * k), cb[k], cw[k], cw[k] - cb[k])

    # 산출 저장(표시용 백분위 포함)
    pctile = np.round(risk_percentile_by_group(R_blend, regime_lbl), 2)
    out = config.OUT / "pole_risk_v2blend.parquet"
    pl.DataFrame({"pole_id": master["pole_id"].to_numpy(),
                  "R_ign": np.round(R_ign, 6), "dose_v22": np.round(dose, 4),
                  "R_blend": np.round(R_blend, 6),
                  "decision_blend": dec_blend.astype(np.int8),
                  "risk_pctile_regime": pctile}).write_parquet(out)
    logger.info("저장: %s", out)

    logger.info("=== run_phase1_mvp 병합 패치(다른 세션 멈춘 뒤) ===")
    logger.info("  # 6단계(R 확정) 직후, decision 전:")
    logger.info("  #   dose = pl.read_parquet(config.OUT/'pole_dose_v2_ellipse.parquet')...['dose_ellipse_rel01']")
    logger.info("  #   R = np.clip(1-(1-R)*(1-0.5*dose), 0, 1)   # exposure_v2 v2.2 결합")
    logger.info("=== 완료 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
