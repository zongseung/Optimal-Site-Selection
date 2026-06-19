"""토지피복/연료 신호 효과 — 누수안전 공간CV recall (있을 때 vs 없을 때).

물음: 발화항 I 에 **landcover(토지피복 발화계수)** 를 넣으면 발화점 recall@top-k 가
오르나? (= 공간을 더 쪼개는 대신 *새로운 종류*의 연속신호로 천장을 올리는가)

비교(동일 게이트·S·W, R=I×S×W 만 다름):
  - equal 6성분(landcover 제외)  vs  equal 7성분(+landcover)   → 토지피복 *원신호*
  - tuned 7성분(config, 배포)    vs  tuned 에서 landcover→0    → 현 튜닝에서의 한계기여
  - landcover 단독                                              → 순수 분리력
검증: prototype_sgg_pooling 의 누수안전 공간CV 하네스 재사용(가중 고정 → 누수 없음).

실행:  .venv/bin/python scripts/prototype_landcover_signal.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pfire import config, experts, io, regimes, validate, weather  # noqa: E402
from sweep_threshold_figs import compute_production_risk  # noqa: E402
from prototype_sgg_pooling import cv_recall, make_folds, KS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("lc_signal")

# config 의 전문가 성분 키(landcover 포함 여부 자동 감지).
FEAT_KEYS = tuple(config.EXPERT_WEIGHTS[config.REGIME_YEONGDONG].keys())


def combine_I(weights, feats, gate, regime_order, drop=()):
    """MoE 발화성향 I = Σ_r gate_r · (Σ_k w[r,k]·feat_k / Σw). drop=제외 성분."""
    keys = [k for k in FEAT_KEYS if k not in drop and k in feats]
    N = next(iter(feats.values())).shape[0]
    expert_mat = np.zeros((N, len(regime_order)))
    for ri, r in enumerate(regime_order):
        w = {k: weights[r][k] for k in keys}
        wsum = sum(w.values())
        if wsum <= 0:
            expert_mat[:, ri] = 0.5
            continue
        sc = np.zeros(N)
        for k in keys:
            sc += w[k] * feats[k]
        expert_mat[:, ri] = sc / wsum
    I = np.einsum("nr,nr->n", gate, expert_mat)
    return np.clip(I, 0.0, 1.0)


def main() -> int:
    master, pole_xy, regime_lbl, R_phys, R_mult, fire_to_pole, fire_xy = \
        compute_production_risk()
    gate, regime_order = regimes.compute_gate(master)
    feats = experts.build_ignition_features(master)
    has_lc = "landcover" in feats and "landcover" in FEAT_KEYS
    logger.info("성분=%s | landcover 존재=%s", FEAT_KEYS, has_lc)
    if not has_lc:
        logger.warning("landcover 성분 없음 — Phase3 미반영 상태. 실험 의미 제한.")

    S = np.clip(master["S_p"].to_numpy().astype(np.float64), 0.0, 1.0)
    W = weather.blended_weather(master, io.load_station_daily(), io.load_stations())

    blocks = validate.spatial_blocks(pole_xy)
    fold = make_folds(blocks)

    eq = {r: {k: 1.0 for k in FEAT_KEYS} for r in regime_order}  # 동일가중
    cfg = config.EXPERT_WEIGHTS

    def R_of(weights, drop=()):
        I = combine_I(weights, feats, gate, regime_order, drop=drop)
        return np.clip(I * S * W, 0.0, 1.0)

    variants = [
        ("equal 6성분 (landcover 제외)", R_of(eq, drop=("landcover",))),
        ("equal 7성분 (+landcover)",     R_of(eq)),
        ("tuned 7성분 (config·배포)",     R_of(cfg)),
        ("tuned, landcover→0",           R_of(cfg, drop=("landcover",))),
    ]
    # landcover 단독 분리력
    if has_lc:
        I_lc = np.clip(feats["landcover"].astype(np.float64), 0.0, 1.0)
        variants.append(("landcover 단독 (I=lc)", np.clip(I_lc * S * W, 0.0, 1.0)))

    logger.info("=== 누수안전 공간블록 CV recall@top-k (토지피복 효과) ===")
    hdr = "변형".ljust(30) + "".join(f"top{int(100*k)}%".rjust(9) for k in KS) + "   평균"
    logger.info(hdr)
    res = {}
    for name, R in variants:
        rec = cv_recall(R, fold, fire_to_pole, KS, None)
        res[name] = rec
        avg = float(np.nanmean([rec[k] for k in KS]))
        logger.info(name.ljust(30) + "".join(f"{rec[k]:.4f}".rjust(9) for k in KS) + f"   {avg:.4f}")

    # 요약 델타
    def avg(n):
        return float(np.nanmean([res[n][k] for k in KS]))
    logger.info("=== 토지피복 한계기여(평균 recall Δ) ===")
    logger.info("  equal: +landcover Δ = %+.4f", avg("equal 7성분 (+landcover)") - avg("equal 6성분 (landcover 제외)"))
    logger.info("  tuned: landcover 유무 Δ = %+.4f", avg("tuned 7성분 (config·배포)") - avg("tuned, landcover→0"))
    logger.info("=== 완료 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
