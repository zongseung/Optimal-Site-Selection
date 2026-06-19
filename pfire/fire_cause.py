"""화재 원인 분류 + 원인별 검증 — 설비원인 화재 앵커 분리(실험 B).

설계 근거(claudedocs/기획서_ablation_설비원인검증_20260619.md §4): 우리 모델의
유일한 novelty("전주=전력설비 자산을 발화원으로 모델")의 효과는 인간발화가 아니라
**설비·전기 원인 화재**에서 나타나야 한다(메커니즘 정합). safemap_positives.resn
(238종 자유텍스트)을 config.CAUSE_* 검토가능 사전으로 5범주에 배정하고, 원인별로
recall@top-k·위험 백분위·부트스트랩 CI 를 모델-비의존(R 만 받음)으로 계산한다.

정직성: 설비(grid_electric) n≈18 로 매우 희소하다. pooled recall(train/test 미분리)은
**상대 비교(설비 vs 인간; 후속 aware vs blind)** 에만 쓰고 절대값은 보수적으로 해석한다.
모든 함수는 순수·재사용 가능(오케스트레이터의 A×B 병합에서 그대로 호출).
"""
from __future__ import annotations

import logging

import numpy as np

from pfire import config
from pfire.calibrate import assign_poles_to_fires, recall_at_topk

logger = logging.getLogger(__name__)


def classify_cause(resn: list[str | None]) -> np.ndarray:
    """resn 자유텍스트 → 원인 5범주 라벨 배열.

    각 resn 문자열에 대해 config.CAUSE_PRIORITY 순서로 범주를 검사하고, 그 범주
    키워드(config.CAUSE_*) 중 하나라도 부분문자열로 포함되면 그 범주를 배정한다
    (우선순위 grid_electric > work_spark > natural > human → 혼합 문구 시 설비 신호를
    우선해 과소계수 방지). 어떤 범주에도 매칭 안 되거나 None/빈 문자열이면 "unknown".

    Parameters
    ----------
    resn : list[str | None]
        발화점별 원인 자유텍스트(결측 None 가능).

    Returns
    -------
    numpy.ndarray, shape (M,)
        dtype object/str. 값 ∈ {"grid_electric","work_spark","natural","human",
        "unknown"}. config.CAUSE_PRIORITY 와 일치.
    """
    # 범주명 → 키워드 튜플 매핑(config 단일 진실). "unknown" 은 폴백이라 키워드 없음.
    keywords: dict[str, tuple[str, ...]] = {
        "grid_electric": config.CAUSE_GRID_ELECTRIC,
        "work_spark": config.CAUSE_WORK_SPARK,
        "natural": config.CAUSE_NATURAL,
        "human": config.CAUSE_HUMAN,
    }
    out = np.empty(len(resn), dtype=object)
    for i, text in enumerate(resn):
        label = "unknown"
        if text is not None and str(text).strip():
            s = str(text)
            for cat in config.CAUSE_PRIORITY:
                kws = keywords.get(cat)
                if kws and any(kw in s for kw in kws):
                    label = cat
                    break
        out[i] = label

    # 분류 분포 로깅(silent 금지). 매핑 불가(unknown) 개수도 명시.
    cats, counts = np.unique(out.astype(str), return_counts=True)
    dist = {str(c): int(n) for c, n in zip(cats, counts)}
    n_unknown = dist.get("unknown", 0)
    logger.info("원인 분류(n=%d): %s | unknown(무매칭)=%d",
                len(resn), dist, n_unknown)
    return out


def _global_ranks(R: np.ndarray) -> np.ndarray:
    """전역 위험 순위(0=최고위험). recall_at_topk 와 동일 규칙(내림차순 argsort).

    한 번 계산해 원인 그룹·부트스트랩 리샘플 전체에서 재사용한다(그룹별 재정렬 금지).

    Parameters
    ----------
    R : numpy.ndarray, shape (N,)
        전역 전주 위험 점수.

    Returns
    -------
    numpy.ndarray, shape (N,)
        각 전주의 0-기반 내림차순 순위(0=최상위).
    """
    order = np.argsort(-R)
    rank = np.empty_like(order)
    rank[order] = np.arange(R.shape[0])
    return rank


def _recall_from_ranks(
    rank: np.ndarray, pos_idx: np.ndarray, n: int, ks: tuple[float, ...]
) -> dict[float, float]:
    """미리 계산한 전역 순위로 양성 전주의 recall@top-k(전역 랭킹 기준).

    recall_at_topk 와 동일한 컷(ceil(k·n))을 쓰되, argsort 를 재실행하지 않는다.
    """
    pos = np.unique(pos_idx[pos_idx >= 0])
    if pos.size == 0:
        return {k: float("nan") for k in ks}
    res: dict[float, float] = {}
    for k in ks:
        cut = max(1, int(np.ceil(k * n)))
        res[k] = float((rank[pos] < cut).mean())
    return res


def recall_by_cause(
    R: np.ndarray,
    pole_xy: np.ndarray,
    fire_xy: np.ndarray,
    cause: np.ndarray,
    ks: tuple[float, ...] = config.TOPK_FOR_RECALL,
    radius_km: float = 1.0,
) -> dict[str, dict]:
    """원인 그룹별 pooled recall@top-k(전역 R 랭킹 기준) — 모델 비의존.

    각 원인 그룹의 발화점을 calibrate.assign_poles_to_fires 로 전주에 매핑한 뒤,
    **전역** R 순위에서 그 매핑 전주들이 상위 k% 에 드는 비율(recall)을 잰다.
    전역 순위는 한 번만 argsort 해 모든 그룹에서 재사용한다(그룹별 재정렬 금지).

    누수주의: pooled(train/test 미분리)이므로 절대 recall 이 아니라 **그룹 간 상대
    비교**(설비 vs 인간)에만 쓴다. 임의의 R 을 받으므로 오케스트레이터의 aware vs blind
    비교에 그대로 재사용 가능하다.

    Parameters
    ----------
    R : numpy.ndarray, shape (N,)
        전역 전주 위험 점수(어떤 모델 산출이든 가능).
    pole_xy : numpy.ndarray, shape (N, 2)
        전주 (lon, lat).
    fire_xy : numpy.ndarray, shape (M, 2)
        발화점 (lon, lat). cause 와 같은 순서.
    cause : numpy.ndarray, shape (M,)
        classify_cause 출력(발화점별 원인 라벨).
    ks : tuple[float]
        recall@top-k 평가 지점(0..1).
    radius_km : float
        발화점→전주 매핑 허용 반경.

    Returns
    -------
    dict[str, dict]
        cause → {"n_fires", "n_mapped", recall@k...}. recall 키는 float k.
    """
    n = R.shape[0]
    rank = _global_ranks(R)
    out: dict[str, dict] = {}
    for grp in sorted(set(str(c) for c in cause)):
        sel = np.asarray([str(c) == grp for c in cause], dtype=bool)
        sub_xy = fire_xy[sel]
        if sub_xy.shape[0] == 0:
            row: dict = dict(n_fires=0, n_mapped=0)
            row.update({k: float("nan") for k in ks})
            out[grp] = row
            continue
        f2p = assign_poles_to_fires(pole_xy, sub_xy, radius_km=radius_km)
        n_mapped = int((f2p >= 0).sum())
        rec = _recall_from_ranks(rank, f2p, n, ks)
        # recall 키는 float k(예약 키워드 충돌 방지로 ** 미사용). 명시 병합.
        row = dict(n_fires=int(sub_xy.shape[0]), n_mapped=n_mapped)
        row.update(rec)
        out[grp] = row
    return out


def risk_percentile_by_cause(
    R: np.ndarray,
    pole_xy: np.ndarray,
    fire_xy: np.ndarray,
    cause: np.ndarray,
    radius_km: float = 1.0,
) -> dict[str, list[float]]:
    """원인 그룹별, 매핑 전주의 전역 위험 백분위 리스트(박스플롯용).

    백분위는 전역 R 의 경험분포에서 각 전주 위험의 순위 백분위(0..1, 클수록 고위험)다.
    그룹별 중앙값·Q1·Q3 를 스크립트에서 산출하도록 raw 리스트를 반환한다.

    Parameters
    ----------
    R : numpy.ndarray, shape (N,)
        전역 위험 점수.
    pole_xy : numpy.ndarray, shape (N, 2)
        전주 (lon, lat).
    fire_xy : numpy.ndarray, shape (M, 2)
        발화점 (lon, lat).
    cause : numpy.ndarray, shape (M,)
        classify_cause 출력.
    radius_km : float
        발화점→전주 매핑 반경.

    Returns
    -------
    dict[str, list[float]]
        cause → 매핑 전주의 위험 백분위 리스트(∈[0,1]). 매핑 0이면 빈 리스트.
    """
    n = R.shape[0]
    order = np.argsort(R)  # 오름차순 → 낮을수록 작은 백분위
    pctile = np.empty(n, dtype=np.float64)
    pctile[order] = np.linspace(0.0, 1.0, n)
    out: dict[str, list[float]] = {}
    for grp in sorted(set(str(c) for c in cause)):
        sel = np.asarray([str(c) == grp for c in cause], dtype=bool)
        sub_xy = fire_xy[sel]
        if sub_xy.shape[0] == 0:
            out[grp] = []
            continue
        f2p = assign_poles_to_fires(pole_xy, sub_xy, radius_km=radius_km)
        mapped = np.unique(f2p[f2p >= 0])
        out[grp] = [float(pctile[j]) for j in mapped]
    return out


def bootstrap_recall_ci(
    R: np.ndarray,
    pole_xy: np.ndarray,
    fire_xy_subset: np.ndarray,
    B: int = 2000,
    ks: tuple[float, ...] = config.TOPK_FOR_RECALL,
    radius_km: float = 1.0,
    seed: int = config.SEED,
) -> dict[float, dict[str, float]]:
    """부분집합 발화점 복원추출 부트스트랩 recall@top-k 95% CI(백분위법).

    fire_xy_subset 의 발화점을 전주에 매핑한 뒤, 매핑 전주 인덱스를 복원추출로 B 회
    리샘플해 각 리샘플의 recall(전역 R 순위 기준)을 재계산한다. 전역 순위는 한 번만
    argsort 해 모든 리샘플에서 재사용한다(리샘플별 재정렬 금지). 95% CI 는 백분위
    [2.5, 97.5].

    희소 n(설비 ≈18) 대응: 표본이 작을수록 CI 가 넓다 — 이는 정직한 불확실성 표현이다.

    Parameters
    ----------
    R : numpy.ndarray, shape (N,)
        전역 위험 점수.
    pole_xy : numpy.ndarray, shape (N, 2)
        전주 (lon, lat).
    fire_xy_subset : numpy.ndarray, shape (m, 2)
        한 원인 그룹의 발화점 (lon, lat).
    B : int
        부트스트랩 리샘플 수.
    ks : tuple[float]
        recall@top-k 지점.
    radius_km : float
        발화점→전주 매핑 반경.
    seed : int
        난수 시드(결정성).

    Returns
    -------
    dict[float, dict]
        k → {"mean", "ci_lo", "ci_hi"}. 매핑 발화점 0 이면 모두 nan.
    """
    n = R.shape[0]
    rank = _global_ranks(R)
    f2p = assign_poles_to_fires(pole_xy, fire_xy_subset, radius_km=radius_km)
    mapped = f2p[f2p >= 0]
    out: dict[float, dict[str, float]] = {}
    if mapped.size == 0:
        logger.warning("부트스트랩: 매핑 발화점 0 → recall NaN")
        return {k: dict(mean=float("nan"), ci_lo=float("nan"),
                        ci_hi=float("nan")) for k in ks}

    rng = np.random.default_rng(seed)
    m = mapped.shape[0]
    cuts = {k: max(1, int(np.ceil(k * n))) for k in ks}
    # 각 매핑 전주가 그 컷 안에 드는지 사전 계산 → 리샘플은 부울 평균만.
    in_top = {k: (rank[mapped] < cuts[k]) for k in ks}
    samples: dict[float, np.ndarray] = {k: np.empty(B, dtype=np.float64) for k in ks}
    for b in range(B):
        pick = rng.integers(0, m, size=m)
        for k in ks:
            samples[k][b] = float(in_top[k][pick].mean())

    for k in ks:
        arr = samples[k]
        out[k] = dict(
            mean=float(arr.mean()),
            ci_lo=float(np.percentile(arr, 2.5)),
            ci_hi=float(np.percentile(arr, 97.5)),
        )
    return out
