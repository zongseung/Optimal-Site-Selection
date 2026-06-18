"""보정·임계값 — 발화점 앵커 기반.

설계 근거: 정답 라벨이 없으므로 발화점(검증 앵커)으로만 보정한다.
(1) recall@top-k: 위험 상위 k% 가 발화점을 얼마나 포함하는가(분리력).
(2) 단조 보정: 위험점수→확률을 직접 구현한 isotonic 유사(PAVA)로 단조화.
(3) F1 민감도: 정답 양성비율 미지 → PREVALENCE_GRID 가정별 가상 F1 곡선으로
임계값 민감도를 보고하고, 운영 양성비율을 골라 임계값을 결정한다.
sklearn 미사용(직접 구현).
"""
from __future__ import annotations

import logging

import numpy as np

from pfire import config, geo

logger = logging.getLogger(__name__)


def assign_poles_to_fires(
    pole_xy: np.ndarray,
    fire_xy: np.ndarray,
    radius_km: float = 1.0,
) -> np.ndarray:
    """각 발화점에 반경 내 최근접 전주를 매핑(검증 앵커용).

    Parameters
    ----------
    pole_xy : numpy.ndarray, shape (N, 2)
        전주 (lon, lat).
    fire_xy : numpy.ndarray, shape (M, 2)
        발화점 (lon, lat).
    radius_km : float
        매핑 허용 반경. 이보다 멀면 매핑 안 함(-1).

    Returns
    -------
    numpy.ndarray, shape (M,)
        각 발화점의 최근접 전주 인덱스. 반경 밖이면 -1.
    """
    pole_km = geo.lonlat_to_km(pole_xy)
    px, py = pole_km[:, 0], pole_km[:, 1]
    out = np.full(fire_xy.shape[0], -1, dtype=np.int64)
    for i in range(fire_xy.shape[0]):
        fx, fy = geo.point_to_km(fire_xy[i, 0], fire_xy[i, 1])
        d2 = (px - fx) ** 2 + (py - fy) ** 2
        j = int(d2.argmin())
        if np.sqrt(d2[j]) <= radius_km:
            out[i] = j
    n_ok = int((out >= 0).sum())
    logger.info("발화점→전주 매핑: %d/%d (반경 %.1fkm)",
                n_ok, fire_xy.shape[0], radius_km)
    return out


def recall_at_topk(
    risk: np.ndarray,
    positive_pole_idx: np.ndarray,
    ks: tuple[float, ...] = config.TOPK_FOR_RECALL,
) -> dict[float, float]:
    """위험 상위 k% 가 발화점-전주를 포함하는 비율(recall@top-k).

    Parameters
    ----------
    risk : numpy.ndarray, shape (N,)
        전주 위험 점수.
    positive_pole_idx : numpy.ndarray
        발화점에 매핑된 전주 인덱스(>=0).
    ks : tuple[float]
        상위 비율 지점들(0..1).

    Returns
    -------
    dict[float, float]
        k → recall.
    """
    pos = np.unique(positive_pole_idx[positive_pole_idx >= 0])
    if pos.size == 0:
        logger.warning("양성 전주 0 → recall NaN")
        return {k: float("nan") for k in ks}
    order = np.argsort(-risk)  # 내림차순
    rank = np.empty_like(order)
    rank[order] = np.arange(risk.shape[0])
    n = risk.shape[0]
    res: dict[float, float] = {}
    for k in ks:
        cut = max(1, int(np.ceil(k * n)))
        in_top = rank[pos] < cut
        res[k] = float(in_top.mean())
    return res


def isotonic_fit(
    x: np.ndarray, y: np.ndarray, weight: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """PAVA 단조 회귀(비감소). sklearn 없이 직접 구현.

    Parameters
    ----------
    x : numpy.ndarray
        예측자(위험). 정렬해 사용.
    y : numpy.ndarray
        타깃(0/1 등).
    weight : numpy.ndarray or None
        표본 가중.

    Returns
    -------
    x_sorted : numpy.ndarray
        정렬된 x(중복 제거 안 함).
    y_iso : numpy.ndarray
        x_sorted 에 대응하는 단조(비감소) 적합값.
    """
    order = np.argsort(x, kind="mergesort")
    xs = x[order].astype(np.float64)
    ys = y[order].astype(np.float64)
    w = np.ones_like(ys) if weight is None else weight[order].astype(np.float64)

    # Pool Adjacent Violators
    vals = ys.copy()
    wts = w.copy()
    # 블록 표현
    block_val: list[float] = []
    block_w: list[float] = []
    block_len: list[int] = []
    for i in range(len(vals)):
        block_val.append(vals[i])
        block_w.append(wts[i])
        block_len.append(1)
        while len(block_val) > 1 and block_val[-2] > block_val[-1]:
            v2, w2, l2 = block_val.pop(), block_w.pop(), block_len.pop()
            v1, w1, l1 = block_val.pop(), block_w.pop(), block_len.pop()
            nw = w1 + w2
            block_val.append((v1 * w1 + v2 * w2) / nw)
            block_w.append(nw)
            block_len.append(l1 + l2)
    y_iso = np.empty_like(vals)
    pos = 0
    for v, ln in zip(block_val, block_len):
        y_iso[pos:pos + ln] = v
        pos += ln
    return xs, y_iso


def calibrate_probability(
    risk: np.ndarray,
    positive_pole_idx: np.ndarray,
    n_neg_sample: int = 200_000,
) -> np.ndarray:
    """위험점수를 단조보정해 사이비 확률(pseudo-probability)로 변환.

    발화점-전주를 양성(1), 나머지에서 표본추출한 전주를 음성(0)으로 두고
    isotonic 적합 → 위험 분위에 대한 단조 확률. 절대확률이 아닌 단조 보정값.

    Parameters
    ----------
    risk : numpy.ndarray, shape (N,)
        위험 점수.
    positive_pole_idx : numpy.ndarray
        양성 전주 인덱스.
    n_neg_sample : int
        음성 표본 수(결정성: config.SEED).

    Returns
    -------
    numpy.ndarray, shape (N,)
        단조 보정 확률 ∈ [0,1].
    """
    n = risk.shape[0]
    pos = np.unique(positive_pole_idx[positive_pole_idx >= 0])
    if pos.size == 0:
        logger.warning("양성 0 → 보정 생략, min-max 반환")
        lo, hi = float(risk.min()), float(risk.max())
        return (risk - lo) / (hi - lo + 1e-12)

    rng = np.random.default_rng(config.SEED)
    neg_pool = np.setdiff1d(np.arange(n), pos, assume_unique=False)
    neg = rng.choice(neg_pool, size=min(n_neg_sample, neg_pool.size), replace=False)

    x = np.concatenate([risk[pos], risk[neg]])
    y = np.concatenate([np.ones(pos.size), np.zeros(neg.size)])
    xs, y_iso = isotonic_fit(x, y)
    # 보정맵을 전체 risk 에 단조 보간.
    p = np.interp(risk, xs, y_iso, left=y_iso[0], right=y_iso[-1])
    return np.clip(p, 0.0, 1.0)


def f1_sensitivity(
    risk: np.ndarray,
    positive_pole_idx: np.ndarray,
    prevalence_grid: tuple[float, ...] = config.PREVALENCE_GRID,
) -> dict[float, dict[str, float]]:
    """양성비율 가정별 가상 F1·임계값 곡선.

    각 가정 양성비율 ρ 에서 위험 상위 ρ 분위를 임계값으로 잡고(예측양성=ρ·N),
    발화점-전주 회수율(recall) 과 prior 기반 가상 precision 으로 F1 을 추정한다.
    정답 음성 라벨이 없으므로 precision 은 prior·recall 로 근사한 상한 지표다.

    Parameters
    ----------
    risk : numpy.ndarray, shape (N,)
        위험 점수.
    positive_pole_idx : numpy.ndarray
        양성 전주 인덱스.
    prevalence_grid : tuple[float]
        가정 양성비율들.

    Returns
    -------
    dict[float, dict]
        ρ → {threshold, predicted_positive, recall, f1_proxy}.
    """
    pos = np.unique(positive_pole_idx[positive_pole_idx >= 0])
    n = risk.shape[0]
    out: dict[float, dict[str, float]] = {}
    # 순위 기반 컷: 동점(보정확률 ties)으로 예측양성이 목표비율을 초과하는
    # 퇴행을 막는다. tie_break 로 결정적 순위를 만든다(고위험 우선).
    rank = _stable_rank_desc(risk)
    for rho in prevalence_grid:
        cut = max(1, int(np.ceil(rho * n)))
        pred_pos = rank < cut
        thr = float(risk[rank == (cut - 1)][0]) if (rank == (cut - 1)).any() else float(risk.min())
        n_pred = int(pred_pos.sum())
        if pos.size:
            recall = float(pred_pos[pos].mean())
        else:
            recall = float("nan")
        # 가상 precision: 예측양성 중 진짜양성이 prior ρ 만큼 있다고 가정.
        prec_proxy = (rho * recall * n) / max(n_pred, 1)
        f1 = (2 * prec_proxy * recall / (prec_proxy + recall)
              if (prec_proxy + recall) > 0 else 0.0)
        out[rho] = dict(threshold=thr, predicted_positive=float(n_pred),
                        recall=recall, f1_proxy=float(f1))
    return out


def decide_threshold(
    risk: np.ndarray, prevalence: float
) -> tuple[float, np.ndarray]:
    """운영 양성비율로 임계값 결정 → 0/1 decision.

    Parameters
    ----------
    risk : numpy.ndarray, shape (N,)
        위험 점수(보정확률 권장).
    prevalence : float
        목표 예측양성 비율(0..1).

    Returns
    -------
    threshold : float
        결정 임계값.
    decision : numpy.ndarray, shape (N,)
        0/1 (int8).
    """
    if not 0.0 < prevalence < 1.0:
        raise ValueError("prevalence 는 (0,1)")
    n = risk.shape[0]
    cut = max(1, int(np.ceil(prevalence * n)))
    rank = _stable_rank_desc(risk)
    decision = (rank < cut).astype(np.int8)
    thr = float(risk[rank == (cut - 1)][0]) if (rank == (cut - 1)).any() else float(risk.min())
    logger.info("임계값 결정: prevalence=%.3f thr=%.5g 양성=%d (%.3f%%)",
                prevalence, thr, int(decision.sum()),
                100.0 * decision.mean())
    return thr, decision


def _stable_rank_desc(
    risk: np.ndarray, tie_break: np.ndarray | None = None
) -> np.ndarray:
    """내림차순 결정적 순위(0=최고위험). 동점은 tie_break(or index)로 분해.

    보정확률 등 ties 가 많은 점수에서 정확히 상위 k 개를 고르기 위한 순위.
    tie_break 가 없으면 인덱스 역순으로 결정적 분해한다.

    Parameters
    ----------
    risk : numpy.ndarray, shape (N,)
        위험 점수.
    tie_break : numpy.ndarray or None
        동점 분해용 보조 점수(클수록 우선). None=인덱스.

    Returns
    -------
    numpy.ndarray, shape (N,)
        각 원소의 0-기반 순위(0=최상위).
    """
    n = risk.shape[0]
    if tie_break is None:
        tie_break = np.arange(n, dtype=np.float64)[::-1]
    # lexsort: 마지막 키가 1차. (-risk, -tie_break) 오름차순 = risk 내림차순.
    order = np.lexsort((-tie_break, -risk))
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    return rank
