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

# 배분 방식 식별자(체제별 예산 배분). run/검증/테스트가 공유하는 단일 진실.
ALLOC_GLOBAL = "global"            # ⒜ 전역 단일 컷(현행) — 체제 무시, 전체 상위 비율.
ALLOC_EQUAL = "regime_equal"       # ⒝ 체제별 동일비율 — 각 체제 내 top-x%(x=전역 prevalence).
ALLOC_COUNT = "regime_count"       # ⒞ 체제별 전주수 비례 — 예산을 체제 전주수 비율로(=전역 동일).
ALLOC_ANCHOR = "regime_anchor"     # ⒟ 체제별 발화점 앵커밀도 비례 — 발화 잦은 체제에 예산 더.
ALLOC_MODES = (ALLOC_GLOBAL, ALLOC_EQUAL, ALLOC_COUNT, ALLOC_ANCHOR)


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


def _f1_from_decision(
    decision: np.ndarray,
    pos: np.ndarray,
    rho: float,
) -> dict[str, float]:
    """주어진 0/1 decision 의 발화점 앵커 proxy-F1/recall/precision.

    정답 음성 라벨이 없으므로 precision 은 prior ρ 기반 가상치다(f1_sensitivity 와
    동일 정의). 예측양성 중 진짜양성이 prior ρ 만큼 있다고 가정한다.
    """
    n = decision.shape[0]
    n_pred = int(decision.sum())
    recall = float(decision[pos].mean()) if pos.size else float("nan")
    prec_proxy = (rho * recall * n) / max(n_pred, 1)
    f1 = (2 * prec_proxy * recall / (prec_proxy + recall)
          if (prec_proxy + recall) > 0 else 0.0)
    return dict(predicted_positive=float(n_pred), recall=recall,
                precision_proxy=float(prec_proxy), f1_proxy=float(f1))


def f1_sensitivity_compare(
    risk: np.ndarray,
    positive_pole_idx: np.ndarray,
    regime_labels: np.ndarray,
    alloc_mode: str = ALLOC_COUNT,
    prevalence_grid: tuple[float, ...] = config.PREVALENCE_GRID,
    anchor_count: dict[str, float] | None = None,
    regimes: tuple[str, ...] = config.REGIMES,
) -> dict[float, dict[str, dict[str, float]]]:
    """양성비율 π 가정별 전역 vs 체제별 컷의 proxy-F1/recall/precision 비교.

    각 π 에서 (전역 단일컷) 과 (체제별 컷, alloc_mode) 을 같은 예산(ceil(π·N))으로
    만들고, 발화점 앵커 recall·prior 기반 가상 precision·F1 을 각각 보고한다.
    "정답비율 몰라도 이 구간에서 안정적"을 보이기 위한 민감도 표(곡선)다.

    Parameters
    ----------
    risk : numpy.ndarray, shape (N,)
        위험 점수.
    positive_pole_idx : numpy.ndarray
        발화점→전주 인덱스(>=0).
    regime_labels : numpy.ndarray of str, shape (N,)
        전주별 우세 체제.
    alloc_mode : str
        체제별 컷 배분 방식(기본 ALLOC_COUNT=전주수비례).
    prevalence_grid : tuple[float]
        가정 양성비율 π 들.
    anchor_count : dict[str, float] or None
        ALLOC_ANCHOR 일 때만 사용(체제→발화수). 누수주의: 전역 산출은 전체 앵커 허용.
    regimes : tuple[str]
        체제 순서.

    Returns
    -------
    dict[float, dict]
        π → {"global": {...}, "regime": {...}}. 각 항목은 predicted_positive,
        recall, precision_proxy, f1_proxy.
    """
    pos = np.unique(positive_pole_idx[positive_pole_idx >= 0])
    n = risk.shape[0]
    out: dict[float, dict[str, dict[str, float]]] = {}
    for rho in prevalence_grid:
        _, dec_g = decide_threshold(risk, rho)
        _, dec_r = decide_threshold_per_regime(
            risk, regime_labels, rho, alloc_mode, anchor_count, regimes)
        out[rho] = dict(
            **{"global": _f1_from_decision(dec_g, pos, rho)},
            regime=_f1_from_decision(dec_r, pos, rho),
        )
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


def allocate_regime_budget(
    regime_labels: np.ndarray,
    total_budget: int,
    mode: str,
    anchor_count: dict[str, float] | None = None,
    regimes: tuple[str, ...] = config.REGIMES,
) -> dict[str, int]:
    """체제별 양성 예산(개수)을 배분한다. 총합 = total_budget(전역 동일 예산).

    배분 방식(mode):
      - ALLOC_EQUAL  : 각 체제 내 동일비율(top-x%). 체제 전주수 n_r 에 비례하므로
                       n_r 비율로 total_budget 을 쪼갠다(= ALLOC_COUNT 와 수학적 동일).
      - ALLOC_COUNT  : 체제 전주수 비례(= ALLOC_EQUAL). 두 라벨을 모두 두는 것은
                       프레이밍(동일비율 vs 전주수비례)을 명시하기 위함이다.
      - ALLOC_ANCHOR : 발화점 앵커밀도(체제별 발화수 / 전주수) 비례. 발화가 잦은
                       체제에 예산을 더 준다. anchor_count(체제→발화수) 필수.
                       발화 0 체제도 굶지 않도록 라플라스 평활(+1 사건)로 가중.

    Parameters
    ----------
    regime_labels : numpy.ndarray of str, shape (N,)
        전주별 우세 체제 라벨(gate argmax).
    total_budget : int
        총 양성 개수(전역 동일). >=1.
    mode : str
        ALLOC_EQUAL|ALLOC_COUNT|ALLOC_ANCHOR. ALLOC_GLOBAL 은 체제배분이 아니므로
        여기서 처리하지 않는다(decide_threshold 전역 사용).
    anchor_count : dict[str, float] or None
        ALLOC_ANCHOR 일 때 체제→발화점수. None 이면 균등 폴백(경고).
    regimes : tuple[str]
        체제 순서(config.REGIMES).

    Returns
    -------
    dict[str, int]
        체제 → 양성 개수. 총합 == min(total_budget, N). 각 체제 예산 <= 그 체제 전주수.

    Raises
    ------
    ValueError
        mode 가 체제배분 모드가 아닐 때.
    """
    if mode not in (ALLOC_EQUAL, ALLOC_COUNT, ALLOC_ANCHOR):
        raise ValueError(f"체제배분 mode 아님: {mode!r} (전역은 decide_threshold 사용)")
    n = regime_labels.shape[0]
    total_budget = int(min(max(total_budget, 0), n))
    n_r = {r: int((regime_labels == r).sum()) for r in regimes}

    if mode in (ALLOC_EQUAL, ALLOC_COUNT):
        weight = {r: float(n_r[r]) for r in regimes}
    else:  # ALLOC_ANCHOR
        if anchor_count is None:
            logger.warning("ALLOC_ANCHOR 인데 anchor_count 없음 → 전주수비례 폴백")
            weight = {r: float(n_r[r]) for r in regimes}
        else:
            # 앵커밀도 = (발화수+1) / 전주수. 발화 0 체제도 작은 양수 가중(굶주림 방지).
            # 예산 = 밀도 × 전주수 ∝ (발화수+1). 즉 사건수에 라플라스 평활.
            weight = {r: float(anchor_count.get(r, 0.0)) + 1.0 for r in regimes}

    return _largest_remainder(weight, total_budget, cap=n_r, regimes=regimes)


def _largest_remainder(
    weight: dict[str, float],
    total: int,
    cap: dict[str, int],
    regimes: tuple[str, ...],
) -> dict[str, int]:
    """가중을 정수 예산으로 배분(최대잉여법). 총합=total, 각 항목 <= cap.

    floor 배분 후 잔여를 소수부 큰 순서로 +1. cap(체제 전주수) 초과는 다른
    체제로 재배분해 총합을 보존한다(결정적: 체제 순서 tie-break).
    """
    wsum = sum(weight[r] for r in regimes)
    if wsum <= 0:
        # 가중 전무 → 전주수 cap 비례 균등 폴백.
        wsum = float(sum(cap[r] for r in regimes)) or 1.0
        weight = {r: float(cap[r]) for r in regimes}

    alloc = {r: 0 for r in regimes}
    remaining = total
    # 반복 배분: cap 초과분을 남은 체제로 재분배(최대 len(regimes) 패스면 수렴).
    active = [r for r in regimes if cap[r] > 0]
    for _ in range(len(regimes) + 1):
        if remaining <= 0 or not active:
            break
        aw = sum(weight[r] for r in active) or 1.0
        raw = {r: weight[r] / aw * remaining for r in active}
        floors = {r: int(np.floor(raw[r])) for r in active}
        # cap 적용.
        for r in active:
            floors[r] = min(floors[r] + alloc[r], cap[r]) - alloc[r]
            floors[r] = max(floors[r], 0)
        for r in active:
            alloc[r] += floors[r]
        remaining = total - sum(alloc.values())
        if remaining <= 0:
            break
        # 잔여 1개씩: 소수부(or 여력) 큰 순. cap 미달 체제만.
        frac = sorted(
            [r for r in active if alloc[r] < cap[r]],
            key=lambda r: (-(raw[r] - np.floor(raw[r])), regimes.index(r)),
        )
        for r in frac:
            if remaining <= 0:
                break
            alloc[r] += 1
            remaining -= 1
        active = [r for r in regimes if alloc[r] < cap[r]]
    return alloc


def decide_threshold_per_regime(
    risk: np.ndarray,
    regime_labels: np.ndarray,
    prevalence: float,
    mode: str,
    anchor_count: dict[str, float] | None = None,
    regimes: tuple[str, ...] = config.REGIMES,
) -> tuple[dict[str, float], np.ndarray]:
    """체제별 분리 랭킹/컷 → 0/1 decision. 총 양성 = prevalence·N(전역 동일 예산).

    각 전주의 우세 체제(regime_labels) 안에서만 위험순위를 매겨, 그 체제에 배분된
    예산(allocate_regime_budget)만큼 상위를 위험(1)으로 둔다. 전역 단일컷이 위험점수
    높은 한 체제(영동)를 싹쓸이하는 것을 막아, 영서·산간의 체제-상대 고위험 전주도
    잡는다. 총 양성 예산은 전역과 동일하게 맞춘다(공정 비교).

    Parameters
    ----------
    risk : numpy.ndarray, shape (N,)
        위험 점수.
    regime_labels : numpy.ndarray of str, shape (N,)
        전주별 우세 체제(gate argmax 등).
    prevalence : float
        목표 전역 예측양성 비율(0..1). 총 예산 = ceil(prevalence·N).
    mode : str
        ALLOC_EQUAL|ALLOC_COUNT|ALLOC_ANCHOR.
    anchor_count : dict[str, float] or None
        ALLOC_ANCHOR 용 체제→발화점수.
    regimes : tuple[str]
        체제 순서.

    Returns
    -------
    thresholds : dict[str, float]
        체제 → 그 체제 내 결정 임계값(이상이면 1). 빈 체제는 +inf(아무도 양성 아님).
    decision : numpy.ndarray, shape (N,)
        0/1 (int8). 합 == allocate_regime_budget 총합.
    """
    if not 0.0 < prevalence < 1.0:
        raise ValueError("prevalence 는 (0,1)")
    n = risk.shape[0]
    total_budget = max(1, int(np.ceil(prevalence * n)))
    budget = allocate_regime_budget(
        regime_labels, total_budget, mode, anchor_count, regimes)

    decision = np.zeros(n, dtype=np.int8)
    thresholds: dict[str, float] = {}
    for r in regimes:
        sel = np.nonzero(regime_labels == r)[0]
        b = int(budget.get(r, 0))
        if sel.size == 0 or b <= 0:
            thresholds[r] = float("inf")
            continue
        risk_r = risk[sel]
        rank_r = _stable_rank_desc(risk_r)
        top = rank_r < b
        decision[sel[top]] = 1
        thresholds[r] = float(risk_r[rank_r == (b - 1)][0]) if (rank_r == (b - 1)).any() \
            else float(risk_r.min())
    logger.info("체제별 임계값(mode=%s): 예산=%s 총양성=%d (%.3f%%)",
                mode, {r: int(budget[r]) for r in regimes},
                int(decision.sum()), 100.0 * decision.mean())
    return thresholds, decision


def regime_anchor_count(
    regime_labels: np.ndarray,
    positive_pole_idx: np.ndarray,
    regimes: tuple[str, ...] = config.REGIMES,
) -> dict[str, float]:
    """체제별 발화점(전주에 매핑된) 개수 집계 — ALLOC_ANCHOR 가중용.

    Parameters
    ----------
    regime_labels : numpy.ndarray of str, shape (N,)
        전주별 우세 체제.
    positive_pole_idx : numpy.ndarray
        발화점→전주 인덱스(>=0). 누수주의: train fold 만 넘길 것.
    regimes : tuple[str]
        체제 순서.

    Returns
    -------
    dict[str, float]
        체제 → 발화점수(그 체제 전주에 매핑된 발화점 수).
    """
    pos = positive_pole_idx[positive_pole_idx >= 0]
    pos = np.unique(pos)
    cnt = {r: 0.0 for r in regimes}
    for r in regimes:
        if pos.size:
            cnt[r] = float((regime_labels[pos] == r).sum())
    return cnt


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


# ──────────────────────────────────────────────────────────────────────────
# per-regime (Mondrian) conformal — 0/1 결정의 분포가정 없는 커버리지 보장 (Phase-5)
# ──────────────────────────────────────────────────────────────────────────
# 설계 근거(기획서_phase5): 베이지안 credible interval 은 모델 오설정 시 under-cover
# 한다. conformal 은 교환가능(exchangeability) 가정만으로 유한표본 커버리지를 건다.
# 우리 목표: "위험집합 {risk ≥ τ_r} 이 (그 체제) 발화점의 (1−α) 를 포함" 보장.
#
# 비순응점수(nonconformity): 발화점일수록 risk 가 높아야 하므로 발화점의 위험
# 점수 자체를 score(낮을수록 비순응 큼)로 본다. 보정집합(=fold train 발화점에
# 매핑된 전주의 risk)에서 α 분위(하위 α)를 임계 τ_r 로 잡으면, 같은 분포의 새
# 발화점 risk 가 τ_r 이상일 확률이 ~(1−α) 다(Mondrian: 체제별 분리).
#
# 누수안전: 보정집합은 **fold train 발화점만** 쓴다(test 발화점 미사용). conformal
# 분위는 (1−α)(n+1) 보정으로 유한표본 보수성을 둔다.


def conformal_threshold_per_regime(
    risk: np.ndarray,
    cal_pole_idx: np.ndarray,
    regime_labels: np.ndarray,
    alpha: float = 0.10,
    regimes: tuple[str, ...] = config.REGIMES,
) -> dict[str, float]:
    """체제별 conformal 임계 τ_r — "위험집합이 발화점 (1−α) 포함" 보장.

    각 체제 r 에서 보정집합(cal_pole_idx 중 체제 r 전주)의 위험 점수를 비순응점수로
    보고, 유한표본 보정 분위 q = floor(α·(n_r+1))/n_r 에 해당하는 risk 하위 분위를
    임계 τ_r 로 잡는다. risk ≥ τ_r 인 전주를 위험(1)으로 두면, 같은 분포의 새 발화점이
    그 집합에 들 확률(=커버리지)이 ≈(1−α) 다(Mondrian/per-regime conformal).

    **누수안전(필수)**: cal_pole_idx 는 **fold train 발화점에 매핑된 전주**만 넘겨야
    한다(test fold 발화점 미사용). 보정집합 발화점은 per-pole 라벨이 아니라 교환가능
    표본으로만 쓴다.

    Parameters
    ----------
    risk : numpy.ndarray, shape (N,)
        전주 위험 점수.
    cal_pole_idx : numpy.ndarray
        보정 발화점에 매핑된 전주 인덱스(>=0). 누수안전: train fold 만.
    regime_labels : numpy.ndarray of str, shape (N,)
        전주별 우세 체제.
    alpha : float
        목표 miscoverage(기본 0.10 → 90% 커버리지). (0,1).
    regimes : tuple[str]
        체제 순서.

    Returns
    -------
    dict[str, float]
        체제 → conformal 임계 τ_r. 보정 발화점이 없는 체제는 -inf(전부 포함 →
        보수적으로 100% 커버, 정보 없음).

    Raises
    ------
    ValueError
        alpha 가 (0,1) 밖일 때.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha 는 (0,1)")
    cal = np.unique(cal_pole_idx[cal_pole_idx >= 0])
    out: dict[str, float] = {}
    for r in regimes:
        in_r = cal[regime_labels[cal] == r] if cal.size else cal
        n_r = int(in_r.size)
        if n_r == 0:
            # 보정 발화점 없음 → 임계 -inf(전부 위험집합 포함 = 100% 커버, 보수).
            out[r] = float("-inf")
            continue
        scores = np.sort(risk[in_r])  # 오름차순(낮을수록 비순응 큼)
        # 유한표본 보정: 하위 ceil(α(n+1)) 번째 점수를 임계로(미만은 잘라냄).
        # rank(1-based) = floor(α(n+1)). 0 이면(α 작거나 n 작음) 최소값 미만 → 전부 포함.
        rank = int(np.floor(alpha * (n_r + 1)))
        if rank <= 0:
            out[r] = float("-inf")
        elif rank > n_r:
            out[r] = float(scores[-1])
        else:
            out[r] = float(scores[rank - 1])
    logger.info("conformal 임계(α=%.2f, 목표커버=%.0f%%): %s",
                alpha, 100 * (1 - alpha),
                {r: (round(out[r], 5) if np.isfinite(out[r]) else "-inf") for r in regimes})
    return out


def conformal_decision(
    risk: np.ndarray,
    regime_labels: np.ndarray,
    thresholds: dict[str, float],
    regimes: tuple[str, ...] = config.REGIMES,
) -> np.ndarray:
    """체제별 conformal 임계로 0/1 decision(risk ≥ τ_r → 1).

    Parameters
    ----------
    risk : numpy.ndarray, shape (N,)
        위험 점수.
    regime_labels : numpy.ndarray of str, shape (N,)
        전주별 우세 체제.
    thresholds : dict[str, float]
        conformal_threshold_per_regime 출력(체제 → τ_r).
    regimes : tuple[str]
        체제 순서.

    Returns
    -------
    numpy.ndarray, shape (N,)
        0/1 (int8).
    """
    decision = np.zeros(risk.shape[0], dtype=np.int8)
    for r in regimes:
        sel = regime_labels == r
        tau = thresholds.get(r, float("-inf"))
        decision[sel & (risk >= tau)] = 1
    logger.info("conformal decision: 양성=%d (%.3f%%)",
                int(decision.sum()), 100.0 * decision.mean())
    return decision


def empirical_coverage(
    lo: np.ndarray,
    hi: np.ndarray,
    test_pole_idx: np.ndarray,
    point: np.ndarray,
    regime_labels: np.ndarray | None = None,
    regimes: tuple[str, ...] = config.REGIMES,
) -> dict[str, dict[str, float]]:
    """credible/예측 구간 [lo,hi] 의 홀드아웃 발화점 실측 커버리지.

    홀드아웃(test) 발화점에 매핑된 전주에서, 그 전주의 점추정 point(예: risk_mean)가
    아니라 — 커버리지 정의대로 — 해당 전주의 구간 [lo, hi] 가 **그 전주의 위험을
    포함하는지**를 본다. 여기서 "관측"은 발화점이 실제로 일어난 전주의 위험으로,
    point(risk_mean)를 관측 대리(observed proxy)로 쓴다: 구간이 point 를 담는 비율이
    명목 커버리지(예: 90%)에 수렴하는지 확인한다(자기일관 sanity). 실질 검증은 validate
    .coverage_holdout 에서 fold 누수안전으로 수행하며, 이 함수는 그 핵심 집계 단위다.

    Parameters
    ----------
    lo, hi : numpy.ndarray, shape (N,)
        구간 하/상한(risk_lo, risk_hi).
    test_pole_idx : numpy.ndarray
        홀드아웃 발화점에 매핑된 전주 인덱스(>=0).
    point : numpy.ndarray, shape (N,)
        점추정(risk_mean) — 구간 포함 여부의 대상값.
    regime_labels : numpy.ndarray of str or None
        주어지면 체제별 커버리지도 집계.
    regimes : tuple[str]
        체제 순서.

    Returns
    -------
    dict[str, dict]
        {'all': {n, covered, coverage}, <regime>: {...}, ...}. coverage = 포함비율.
    """
    test = np.unique(test_pole_idx[test_pole_idx >= 0])
    out: dict[str, dict[str, float]] = {}

    def _cov(idx: np.ndarray) -> dict[str, float]:
        if idx.size == 0:
            return dict(n=0.0, covered=0.0, coverage=float("nan"))
        inside = (point[idx] >= lo[idx] - 1e-12) & (point[idx] <= hi[idx] + 1e-12)
        return dict(n=float(idx.size), covered=float(inside.sum()),
                    coverage=float(inside.mean()))

    out["all"] = _cov(test)
    if regime_labels is not None and test.size:
        for r in regimes:
            out[r] = _cov(test[regime_labels[test] == r])
    return out


def conformal_coverage(
    risk: np.ndarray,
    test_pole_idx: np.ndarray,
    regime_labels: np.ndarray,
    thresholds: dict[str, float],
    regimes: tuple[str, ...] = config.REGIMES,
) -> dict[str, dict[str, float]]:
    """conformal 임계의 홀드아웃 발화점 실측 커버리지(체제별).

    홀드아웃 발화점에 매핑된 전주가 위험집합(risk ≥ τ_r)에 드는 비율(=발화점 커버리지)을
    전역·체제별로 집계한다. 명목(1−α)에 수렴하면 conformal 보장이 실측으로 성립.

    Parameters
    ----------
    risk : numpy.ndarray, shape (N,)
        위험 점수.
    test_pole_idx : numpy.ndarray
        홀드아웃 발화점→전주 인덱스(>=0).
    regime_labels : numpy.ndarray of str, shape (N,)
        전주별 우세 체제.
    thresholds : dict[str, float]
        conformal 임계(체제 → τ_r). 누수안전: train fold 로 추정한 것.
    regimes : tuple[str]
        체제 순서.

    Returns
    -------
    dict[str, dict]
        {'all': {n, covered, coverage}, <regime>: {...}}. coverage = 위험집합 포함비율.
    """
    test = np.unique(test_pole_idx[test_pole_idx >= 0])
    out: dict[str, dict[str, float]] = {}

    def _cov(idx: np.ndarray) -> dict[str, float]:
        if idx.size == 0:
            return dict(n=0.0, covered=0.0, coverage=float("nan"))
        covered = np.zeros(idx.size, dtype=bool)
        for r in regimes:
            sel = regime_labels[idx] == r
            covered[sel] = risk[idx][sel] >= thresholds.get(r, float("-inf"))
        return dict(n=float(idx.size), covered=float(covered.sum()),
                    coverage=float(covered.mean()))

    out["all"] = _cov(test)
    if test.size:
        for r in regimes:
            out[r] = _cov(test[regime_labels[test] == r])
    return out
