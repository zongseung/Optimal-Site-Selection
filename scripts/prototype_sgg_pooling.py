"""시군구 부분풀링(전역→체제→시군구 수축) 프로토타입 — 누수안전 공간CV 검증.

물음: 체제(영동/영서/산간) 위에 **시군구**를 한 겹 더 부분풀링으로 얹으면
발화점 recall@top-k 가 오르나? (= "한 번 더 조건부 분할"이 데이터로 정당한가)

식 (닫힌형, 학습 없음):
  RR_g = (y_g/n_g)/λ̄ ,  w_g = y_g/(y_g+κ) ,  m_g = w_g·RR_g + (1−w_g)·m_부모
  체제도 동일하게 전역(=1)에서 수축. 최종 R' = R×m (랭킹은 전역스케일 불변).
  y=발화점수, n=전주수, λ̄=전역 발화율. 불 적은 시군구는 자동으로 체제로 수축.

검증(정직):
  - **누수안전 CV**: fold f 평가 시 배율은 train fold 발화점으로만 계산(test fold 미사용).
  - **in-sample CV**: 배율을 전체 발화점으로 계산(누수) → 누수안전과의 격차 = 암기 정도.
  - baseline(R=I·S·W) vs +체제풀링 vs +시군구풀링(κ별) 비교.

실행:  .venv/bin/python scripts/prototype_sgg_pooling.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pfire import config, validate  # noqa: E402
from sweep_threshold_figs import compute_production_risk  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("sgg_pool")

KS = config.TOPK_FOR_RECALL
KAPPAS = (3.0, 10.0, 30.0)
N_FOLDS = 5


def make_folds(blocks: np.ndarray, n_folds: int = N_FOLDS, seed: int = config.SEED) -> np.ndarray:
    """validate.spatial_cv_recall 과 동일한 공간블록→fold 배정(비교 가능성)."""
    uniq = np.unique(blocks)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    fob = {b: i % n_folds for i, b in enumerate(uniq)}
    return np.array([fob[b] for b in blocks])


def _recall_in_fold(Rp, test_mask, pos_mask, ks):
    """test fold 내부 recall@top-k (calibrate.recall_at_topk 와 동일 의미)."""
    R_f = Rp[test_mask]
    pos_idx = np.nonzero(pos_mask[test_mask])[0]
    if pos_idx.size == 0:
        return None
    order = np.argsort(-R_f)
    rank = np.empty_like(order)
    rank[order] = np.arange(R_f.shape[0])
    n = R_f.shape[0]
    out = {}
    for k in ks:
        cut = max(1, int(np.ceil(k * n)))
        out[k] = float((rank[pos_idx] < cut).mean())
    return out


def build_multiplier(train_fire_pole, sgg_code, n_sgg_codes, sgg_parent_regime,
                     regime_code, n_regime, n_g, kappa, level):
    """전역→체제→시군구 부분풀링 배율 (N,). level='regime'|'sgg'.

    train_fire_pole : 학습용 발화점이 귀속된 전주 인덱스(누수안전: fold별 train만).
    """
    N = sgg_code.shape[0]
    y_total = train_fire_pole.shape[0]
    if y_total == 0:
        return np.ones(N)
    lam_bar = y_total / N

    # 체제 수축: RR_r, m_r
    y_r = np.bincount(regime_code[train_fire_pole], minlength=n_regime).astype(float)
    n_r = np.bincount(regime_code, minlength=n_regime).astype(float)
    RR_r = np.where(n_r > 0, (y_r / np.maximum(n_r, 1)) / lam_bar, 1.0)
    w_r = y_r / (y_r + kappa)
    m_r = w_r * RR_r + (1.0 - w_r) * 1.0          # (n_regime,)
    if level == "regime":
        return m_r[regime_code]

    # 시군구 수축(부모=체제): RR_g, m_g
    y_g = np.bincount(sgg_code[train_fire_pole], minlength=n_sgg_codes).astype(float)
    RR_g = np.where(n_g > 0, (y_g / np.maximum(n_g, 1)) / lam_bar, 1.0)
    w_g = y_g / (y_g + kappa)
    m_g = w_g * RR_g + (1.0 - w_g) * m_r[sgg_parent_regime]   # (n_sgg,)
    return m_g[sgg_code]


def cv_recall(R_base, fold, fire_to_pole, ks, build_fn, n_folds=N_FOLDS):
    """누수안전 공간CV recall@top-k. build_fn(train_fire_pole)->m (N,); None=baseline."""
    mapped = fire_to_pole >= 0
    fire_pole = fire_to_pole[mapped].astype(int)
    fire_fold = fold[fire_pole]
    pos_mask = np.zeros(R_base.shape[0], dtype=bool)
    pos_mask[fire_pole] = True
    per_k = {k: [] for k in ks}
    for f in range(n_folds):
        test_mask = fold == f
        if not test_mask.any():
            continue
        if build_fn is None:
            Rp = R_base
        else:
            train_fp = fire_pole[fire_fold != f]
            Rp = R_base * build_fn(train_fp)
        rec = _recall_in_fold(Rp, test_mask, pos_mask, ks)
        if rec:
            for k in ks:
                per_k[k].append(rec[k])
    return {k: (float(np.mean(per_k[k])) if per_k[k] else float("nan")) for k in ks}


def main() -> int:
    master, pole_xy, regime_lbl, R_phys, R_mult, fire_to_pole, fire_xy = \
        compute_production_risk()
    R = np.clip(R_phys, 0.0, 1.0)

    # 시군구·체제 정수 코드화
    sgg = master["sgg"].to_list()
    sgg_uniq, sgg_code = np.unique(np.array(sgg, dtype=object), return_inverse=True)
    reg_uniq, regime_code = np.unique(regime_lbl, return_inverse=True)
    n_sgg, n_reg = sgg_uniq.shape[0], reg_uniq.shape[0]
    n_g = np.bincount(sgg_code, minlength=n_sgg).astype(float)
    # 시군구 부모 체제 = 그 시군구 전주들의 다수 체제
    parent_counts = np.zeros((n_sgg, n_reg), dtype=np.int64)
    np.add.at(parent_counts, (sgg_code, regime_code), 1)
    sgg_parent_regime = parent_counts.argmax(axis=1)
    logger.info("시군구=%d개, 체제=%d개, 매핑발화점=%d",
                n_sgg, n_reg, int((fire_to_pole >= 0).sum()))

    blocks = validate.spatial_blocks(pole_xy)
    fold = make_folds(blocks)

    def mk(level, kappa):
        return lambda tfp: build_multiplier(
            tfp, sgg_code, n_sgg, sgg_parent_regime, regime_code, n_reg,
            n_g, kappa, level)

    # 전체발화점 배율(in-sample, 누수) — 암기정도 진단용
    all_fp = fire_to_pole[fire_to_pole >= 0].astype(int)

    rows = []
    base = cv_recall(R, fold, fire_to_pole, KS, None)
    rows.append(("baseline (I·S·W)", base))
    for kappa in KAPPAS:
        rows.append((f"+체제풀링 (κ={kappa:g})",
                     cv_recall(R, fold, fire_to_pole, KS, mk("regime", kappa))))
    for kappa in KAPPAS:
        rows.append((f"+시군구풀링 누수안전 (κ={kappa:g})",
                     cv_recall(R, fold, fire_to_pole, KS, mk("sgg", kappa))))
    # in-sample(누수) 시군구 — 같은 배율을 전체발화점으로 고정
    for kappa in KAPPAS:
        m_all = build_multiplier(all_fp, sgg_code, n_sgg, sgg_parent_regime,
                                 regime_code, n_reg, n_g, kappa, "sgg")
        rows.append((f"+시군구풀링 in-sample (κ={kappa:g})",
                     cv_recall(R, fold, fire_to_pole, KS, lambda _t, m=m_all: m)))

    # 출력 표
    hdr = "변형".ljust(34) + "".join(f"top{int(100*k)}%".rjust(9) for k in KS) + "   평균"
    logger.info("=== 누수안전 공간블록 CV recall@top-k (체제 vs 시군구) ===")
    logger.info(hdr)
    for name, rec in rows:
        avg = float(np.nanmean([rec[k] for k in KS]))
        line = name.ljust(34) + "".join(f"{rec[k]:.4f}".rjust(9) for k in KS) + f"   {avg:.4f}"
        logger.info(line)

    # 무작위-공간 격차(최선 시군구 κ, in-sample 배율로 진단)
    best_kappa = KAPPAS[1]
    m_all = build_multiplier(all_fp, sgg_code, n_sgg, sgg_parent_regime,
                             regime_code, n_reg, n_g, best_kappa, "sgg")
    gap = validate.random_vs_spatial_gap(np.clip(R * m_all, 0, 1), fire_to_pole, blocks, k=0.05)
    logger.info("=== 낙관편향 진단(시군구 in-sample κ=%g, @5%%) ===", best_kappa)
    logger.info("  무작위=%.4f 공간=%.4f 격차=%.4f (≈0이면 편향없음)",
                gap["random_recall"], gap["spatial_recall"], gap["gap"])
    logger.info("=== 완료 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
