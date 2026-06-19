"""위험 표시용 백분위 지수 — 순위 보존 재척도(canonical 단일 구현).

근거: R=I·S·W(×배율)는 1보다 작은 성분들의 곱이라 절대값이 작다(mean≈0.06).
0/1 decision 은 순위로만 결정되므로 절대 스케일은 무의미하지만, 제출/그림에
raw 값(예: 0.011)을 쓰면 "위험=0.011"로 보여 정성평가에서 약해 보인다.
순위 백분위(0=최저, 100=최고)로 재표시하면 의미 보존 + 가독성↑, **순위·decision 불변**.

표시값 변환의 단일 진실. submit/figures/scripts 모두 여기서 import 한다.
"""
from __future__ import annotations

import numpy as np


def risk_percentile(risk: np.ndarray, ties: str = "ordinal") -> np.ndarray:
    """위험 점수 → 백분위 지수 [0,100] (0=최저위험, 100=최고위험).

    Parameters
    ----------
    risk : numpy.ndarray, shape (N,)
        위험 점수(연속). 클수록 위험.
    ties : {"ordinal", "average"}
        동점 처리. "ordinal"(기본)=동점도 안정정렬로 distinct 백분위 부여
        (R 이 거의 전부 고유값일 때 정확). "average"=동점에 평균 백분위.

    Returns
    -------
    numpy.ndarray, shape (N,)
        백분위 지수 ∈ [0,100]. risk 와 단조(순위 보존).
    """
    risk = np.asarray(risk, dtype=np.float64)
    n = risk.shape[0]
    if n == 0:
        return risk.copy()
    if n == 1:
        return np.array([100.0])
    order = np.argsort(risk, kind="mergesort")  # 오름차순(안정)
    pct = np.empty(n, dtype=np.float64)
    pct[order] = np.linspace(0.0, 100.0, n)
    if ties == "average":
        # 동점 그룹에 평균 백분위 부여(완전 단조 보존, 동점 동일값).
        uniq, inv, counts = np.unique(risk, return_inverse=True, return_counts=True)
        if uniq.size != n:
            sums = np.zeros(uniq.size, dtype=np.float64)
            np.add.at(sums, inv, pct)
            pct = (sums / counts)[inv]
    return pct


def risk_percentile_by_group(
    risk: np.ndarray, group: np.ndarray, ties: str = "ordinal"
) -> np.ndarray:
    """체제(그룹) **내** 백분위 [0,100]. 각 그룹 안에서의 순위 백분위.

    글로벌 백분위는 decision 이 체제별 배분일 때 정렬이 깨진다(전역 중위 전주가
    자기 체제 상위라 위험(1)이 될 수 있음). 체제 내 백분위는 "자기 지역 기준
    상위 몇 %"라 배분 decision 과 일관된다.

    Parameters
    ----------
    risk : numpy.ndarray, shape (N,)
        위험 점수.
    group : numpy.ndarray, shape (N,)
        그룹 라벨(체제 등). 각 고유값별로 독립 백분위.
    ties : {"ordinal", "average"}
        risk_percentile 와 동일.

    Returns
    -------
    numpy.ndarray, shape (N,)
        체제 내 백분위 ∈ [0,100].
    """
    risk = np.asarray(risk, dtype=np.float64)
    group = np.asarray(group)
    out = np.empty(risk.shape[0], dtype=np.float64)
    for g in np.unique(group):
        m = group == g
        out[m] = risk_percentile(risk[m], ties=ties)
    return out


def cut_to_percentile(prevalence: float) -> float:
    """상위 prevalence 비율 컷 → 백분위 임계(예: 0.05 → 95.0)."""
    return 100.0 * (1.0 - float(prevalence))
