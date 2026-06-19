"""Phase-4 ② 데모 — 계층 partial pooling(경험적 베이즈) 지역 보정.

pfire.hierarchy.regional_multiplier 를 마스터+발화점으로 산출해 다음을 보고한다.
  ⒜ 체제·시군별 EB 배율 분포(관측율·부모율·EB율·축소가중·배율).
  ⒝ 발화 0/소수 지역(화천·인제·고성·태백)이 부모 단계로 축소됐는지 확인.
  ⒞ 누수방지 공간블록 CV(config.SPATIAL_CV_BLOCK_KM, 기본 10km):
       각 폴드의 **train fold 발화점으로만** regional_multiplier 를 재계산하고
       (test fold 발화점은 multiplier 산출에 미사용), 그 배율 **단독**으로
       hold-out 폴드의 test 발화점 recall@top-{1,2,5,10}% 를 측정.

"배율 단독" recall 은 배율이 그 자체로 발화점을 상위에 올리는 분리력을 본다(통합 전
독립 진단). 배율은 곱셈 파이프라인(I·S·W)에 곱해 쓰는 보정자이므로, 단독 성능이
물리식 본체보다 낮은 것은 정상이다(지역 prior 의 기여만 측정).

실행:
    .venv/bin/python scripts/phase4_hierarchy_demo.py [--block-km 10] [--n-folds 5]
        [--anchor-km 1.0] [--normalize mean|unit]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pfire import calibrate, config, hierarchy, io, validate  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase4_hierarchy_demo")

KS = config.TOPK_FOR_RECALL
# 발화 0/소수로 부모 축소를 확인할 표적 시군(태사 기획서 명시).
TARGET_SGG = ("화천", "인제", "고성", "태백")


def _fmt_cv(cv: dict[float, dict[str, float]]) -> str:
    return " ".join(
        f"top{int(k * 100)}={cv[k]['mean']:.4f}±{cv[k]['std']:.4f}" for k in KS
    )


def leakage_safe_spatial_cv(
    master: pl.DataFrame,
    positives: pl.DataFrame,
    *,
    block_km: float,
    n_folds: int,
    anchor_km: float,
    normalize: str,
) -> dict[float, dict[str, float]]:
    """누수방지 공간블록 CV — fold 별 train 발화점으로만 배율 재계산.

    공간 블록(block_km)을 폴드로 나눈 뒤, 각 폴드 f 에 대해:
      1) train 발화점 = f 가 아닌 블록에 귀속된 발화점만 추려 hierarchy.regional_multiplier
         재계산(test fold 발화점 정보 미사용 → 누수 차단).
      2) hold-out(f 블록) 전주만으로 그 배율의 상위 k% 가 f 의 test 발화점을 회수하는지
         recall@top-k 측정.
    폴드별 recall 의 평균±표준편차 반환.

    Parameters
    ----------
    master : polars.DataFrame
        전주 마스터.
    positives : polars.DataFrame
        발화점 전체.
    block_km : float
        공간 블록 한 변(km).
    n_folds : int
        폴드 수.
    anchor_km : float
        발화점→전주 귀속 반경(배율 산출·평가 앵커 공통).
    normalize : str
        'mean'|'unit'.

    Returns
    -------
    dict[float, dict]
        k → {mean, std, n_folds_with_pos}.
    """
    pole_xy = master.select(["lon", "lat"]).to_numpy().astype(np.float64)
    fire_xy = positives.select(["lon", "lat"]).to_numpy().astype(np.float64)

    # 전주 공간 블록 → 폴드.
    blocks = validate.spatial_blocks(pole_xy, block_km=block_km)
    uniq = np.unique(blocks)
    rng = np.random.default_rng(config.SEED)
    rng.shuffle(uniq)
    fold_of_block = {b: i % n_folds for i, b in enumerate(uniq)}
    pole_fold = np.array([fold_of_block[b] for b in blocks])

    # 각 발화점을 최근접 전주에 귀속 → 그 전주의 폴드/인덱스(평가 앵커).
    f2p = calibrate.assign_poles_to_fires(pole_xy, fire_xy, radius_km=anchor_km)
    fire_pole = f2p                      # (M,) 전주 인덱스 또는 -1
    fire_fold = np.where(fire_pole >= 0, pole_fold[np.clip(fire_pole, 0, None)], -1)

    per_k: dict[float, list[float]] = {k: [] for k in KS}
    for f in range(n_folds):
        # train 발화점 = 다른 폴드 블록에 귀속된 발화점(누수 차단).
        train_mask = (fire_pole >= 0) & (fire_fold != f)
        fires_train = positives.filter(pl.Series(train_mask))
        # train 발화점으로만 배율 산출(test fold 발화점 미사용).
        mult = hierarchy.regional_multiplier(
            master, fires_train, anchor_radius_km=anchor_km, normalize=normalize
        )
        # hold-out 폴드 전주 내에서 배율 단독 recall.
        sel = pole_fold == f
        if sel.sum() == 0:
            continue
        risk_f = mult[sel]
        # test 발화점-전주(이 폴드 귀속) → 폴드 내부 인덱스로 변환.
        glob2local = -np.ones(master.height, dtype=np.int64)
        glob2local[np.nonzero(sel)[0]] = np.arange(int(sel.sum()))
        test_fire_pole = fire_pole[(fire_fold == f) & (fire_pole >= 0)]
        pos_local = glob2local[test_fire_pole]
        pos_local = pos_local[pos_local >= 0]
        if pos_local.size == 0:
            continue
        rec = calibrate.recall_at_topk(risk_f, pos_local, KS)
        for k in KS:
            per_k[k].append(rec[k])

    out: dict[float, dict[str, float]] = {}
    for k in KS:
        arr = np.array(per_k[k], dtype=np.float64)
        out[k] = dict(
            mean=float(arr.mean()) if arr.size else float("nan"),
            std=float(arr.std()) if arr.size else float("nan"),
            n_folds_with_pos=float(arr.size),
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Phase-4 ② 계층 EB 지역보정 데모")
    ap.add_argument("--block-km", type=float, default=config.SPATIAL_CV_BLOCK_KM,
                    help="공간 블록 한 변(km).")
    ap.add_argument("--n-folds", type=int, default=5, help="공간 CV 폴드 수.")
    ap.add_argument("--anchor-km", type=float, default=1.0,
                    help="발화점→전주 귀속 반경(km).")
    ap.add_argument("--normalize", choices=("mean", "unit"), default="mean",
                    help="배율 정규화(mean=전역평균≈1, unit=[0,1]).")
    args = ap.parse_args()
    np.random.seed(config.SEED)

    logger.info("=== 1. 데이터 로드 ===")
    master = io.load_master()
    positives = io.load_positives()
    logger.info("master rows=%d, positives=%d", master.height, positives.height)

    logger.info("=== 2. 전데이터 EB 배율 산출 ===")
    mult = hierarchy.regional_multiplier(
        master, positives, anchor_radius_km=args.anchor_km, normalize=args.normalize
    )
    logger.info("배율 형상=%d 유한=%s 최소=%.4f 평균=%.4f 최대=%.4f",
                mult.shape[0], bool(np.all(np.isfinite(mult))),
                float(mult.min()), float(mult.mean()), float(mult.max()))

    logger.info("=== ⒜ 체제·시군별 EB 배율 분포 ===")
    tbl = hierarchy.region_rate_table(master, positives, anchor_radius_km=args.anchor_km)
    with pl.Config(tbl_rows=30, tbl_width_chars=200, fmt_float="full"):
        print("\n--- 체제(regime) 단계 ---")
        print(tbl.filter(pl.col("level") == "regime")
              .sort("eb_rate")
              .select(["name", "n_poles", "n_fires", "obs_rate", "parent_rate",
                       "eb_rate", "shrink_w", "mult"]))
        print("\n--- 시군(sgg) 단계 (n_fires 오름차순) ---")
        print(tbl.filter(pl.col("level") == "sgg")
              .sort("n_fires")
              .select(["name", "regime", "n_poles", "n_fires", "obs_rate",
                       "parent_rate", "eb_rate", "shrink_w", "mult"]))

    logger.info("=== ⒝ 발화 0/소수 시군 → 부모 축소 확인 ===")
    sgg_tbl = tbl.filter(pl.col("level") == "sgg")
    for name in TARGET_SGG:
        row = sgg_tbl.filter(pl.col("name") == name)
        if row.height == 0:
            logger.warning("  %s: 시군 없음(스킵)", name)
            continue
        r = row.row(0, named=True)
        # 축소 판정: EB율이 부모율에 가까운가(|eb-parent| << |obs-parent| or obs nan).
        gap_parent = abs(r["eb_rate"] - r["parent_rate"])
        denom = max(r["eb_rate"], r["parent_rate"], 1e-12)
        rel = gap_parent / denom
        shrunk = (np.isnan(r["obs_rate"]) and rel < 0.05) or (r["shrink_w"] < 0.2)
        logger.info(
            "  %s(%s): n전주=%d n발화=%d obs=%.3e parent=%.3e eb=%.3e "
            "shrink_w=%.3f mult=%.3f → 부모축소=%s",
            name, r["regime"], int(r["n_poles"]), int(r["n_fires"]),
            r["obs_rate"], r["parent_rate"], r["eb_rate"], r["shrink_w"],
            r["mult"], "예" if shrunk else "부분",
        )

    logger.info("=== ⒞ 누수방지 공간블록 CV — '배율 단독' recall@top-k ===")
    logger.info("(block=%.0fkm, folds=%d, train fold 발화점으로만 배율 산출)",
                args.block_km, args.n_folds)
    cv = leakage_safe_spatial_cv(
        master, positives,
        block_km=args.block_km, n_folds=args.n_folds,
        anchor_km=args.anchor_km, normalize=args.normalize,
    )
    logger.info("[배율 단독] %s", _fmt_cv(cv))
    print("\n--- 누수방지 공간CV recall@top-k (배율 단독) ---")
    for k in KS:
        print(f"  top{int(k*100):>2d}%: mean={cv[k]['mean']:.4f} "
              f"std={cv[k]['std']:.4f} (n_folds_with_pos={int(cv[k]['n_folds_with_pos'])})")

    # 참고: random baseline = k(top-k 면 우연 recall ≈ k). 배율이 그 이상이면 신호 有.
    print("\n  참고 random baseline(top-k 우연 recall ≈ k): "
          + " ".join(f"top{int(k*100)}={k:.4f}" for k in KS))

    logger.info("=== 완료 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
