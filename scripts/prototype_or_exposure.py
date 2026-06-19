"""확산(노출)을 전주 화재위험에 확률 OR 로 결합 — 프로토타입·검증.

R_전주 = 1 − (1 − R_발화)(1 − w·노출)   (확률 OR: 발화 랭킹 유지 + 풍하 전주 상승)

문제: raw p_exposure 는 영동에서 포화(평균 0.90)·R_발화는 작음(0.06) → raw OR 면
영동이 top 을 도배(게이트가 이미 아는 정보) → 발화 recall 붕괴. 그래서 두 형태 비교:
  - rawOR : 1−(1−R)(1−w·pe)              ← 영동 쏠림 예상
  - regOR : 1−(1−R)(1−w·pe_regime%)      ← 체제 *내* 노출 백분위(체제간 차는 게이트에 맡김)

측정(고정 R, 누수안전): ① 발화 recall@top-k(유지되나) ② top-2% 영동 share(쏠리나)
③ 2019 고성 풍하방향 sanity(노출이 동향인가). p_exposure 는 직전 submission 재사용.

실행:  .venv/bin/python scripts/prototype_or_exposure.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pfire import config, validate, calibrate  # noqa: E402
from pfire.risk_index import risk_percentile_by_group  # noqa: E402
from sweep_threshold_figs import compute_production_risk  # noqa: E402
from prototype_sgg_pooling import make_folds, KS  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("or_exp")


def top_share(R, regime_lbl, prevalence=0.02, target="yeongdong"):
    """전역 top-prevalence 에서 target 체제 비율 + 그들의 평균노출."""
    n = R.shape[0]
    cut = max(1, int(np.ceil(prevalence * n)))
    order = np.argsort(-R)
    top = order[:cut]
    return float(np.mean(regime_lbl[top] == target))


def cv_recall_fixed(R, fire_to_pole, blocks):
    cv = validate.spatial_cv_recall(R, fire_to_pole, blocks)
    return {k: cv[k]["mean"] for k in KS}


def main() -> int:
    master, pole_xy, regime_lbl, R_phys, R_mult, fire_to_pole, fire_xy = \
        compute_production_risk()
    R = np.clip(R_phys, 0.0, 1.0)

    # p_exposure 재사용(직전 submission). pole_id 정렬 동일 → 직접 정렬.
    sub = pl.read_csv(config.SUBMISSIONS / "submission.csv").sort("pole_id")
    pe = sub["p_exposure"].to_numpy().astype(np.float64)
    assert pe.shape[0] == R.shape[0], "행수 불일치"
    pe_reg = risk_percentile_by_group(pe, regime_lbl) / 100.0   # 체제 내 노출 백분위 [0,1]

    blocks = validate.spatial_blocks(pole_xy)

    def OR(Rig, expo, w):
        return np.clip(1.0 - (1.0 - Rig) * (1.0 - w * np.clip(expo, 0, 1)), 0.0, 1.0)

    variants = [("baseline R_발화", R)]
    for w in (0.5, 1.0):
        variants.append((f"rawOR (w={w:g})", OR(R, pe, w)))
    for w in (0.5, 1.0):
        variants.append((f"regOR 체제내% (w={w:g})", OR(R, pe_reg, w)))

    logger.info("=== 확산 OR 결합: 발화 recall vs 영동 쏠림 ===")
    hdr = "변형".ljust(24) + "".join(f"top{int(100*k)}%".rjust(8) for k in KS) \
        + "   평균    top2%영동share"
    logger.info(hdr)
    base_share = None
    for name, Rv in variants:
        rec = cv_recall_fixed(Rv, fire_to_pole, blocks)
        avg = float(np.nanmean([rec[k] for k in KS]))
        share = top_share(Rv, regime_lbl, 0.02, "yeongdong")
        if base_share is None:
            base_share = share
        logger.info(name.ljust(24) + "".join(f"{rec[k]:.4f}".rjust(8) for k in KS)
                    + f"   {avg:.4f}   {share:.3f}")

    # 방향 sanity (p_exposure 자체, w 무관)
    dw = validate.exposure_downwind_goseong(pole_xy, pe)
    logger.info("=== 2019 고성 풍하방향 sanity (노출) ===")
    logger.info("  노출전주 %d/%d  east_frac=%.2f (>0.5=서→동 일치)  mean_dlon=%+.4f도",
                int(dw["n_exposed"]), int(dw["n_local"]), dw["east_frac"], dw["mean_dlon_deg"])
    logger.info(f"  (참고) 전역 top2pct 영동share: baseline={base_share:.3f}")
    logger.info("=== 완료 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
