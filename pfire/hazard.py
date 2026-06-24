"""위험 결합 — h(p,t) = I(p) × S(p) × W(grid,t), 시즌 집계 R(p).

설계 근거: 발화 성향 I(고정), 확산/노출 취약 S(고정), 그날 기상 W(가변)의
곱셈 결합이 본체다. 셋 중 하나라도 낮으면 위험이 낮아지는 AND 성격(물리적
타당). MVP 는 W 가 시즌통계라 일별 곱이 곧 시즌 위험 R(p) 와 같다. 일별
가변 W 가 들어오면 로그합(기하평균)으로 시즌집계해 한 번의 고위험일 폭주를
완화한다.
"""
from __future__ import annotations

import logging

import numpy as np

from pfire import config

logger = logging.getLogger(__name__)


def _validate_unit(name: str, x: np.ndarray) -> None:
    """성분이 [0,1] · NaN 없음인지 검증(silent 금지)."""
    if np.isnan(x).any():
        raise ValueError(f"{name} 에 NaN 존재")
    if x.min() < -1e-9 or x.max() > 1 + 1e-9:
        raise ValueError(f"{name} 가 [0,1] 범위 밖: [{x.min():.4g},{x.max():.4g}]")


def daily_hazard(I: np.ndarray, S: np.ndarray, W: np.ndarray) -> np.ndarray:
    """하루 위험 h = I × S × W (성분곱).

    Parameters
    ----------
    I, S, W : numpy.ndarray, shape (N,)
        각 [0,1] 성분.

    Returns
    -------
    numpy.ndarray, shape (N,)
        일 위험 h ∈ [0,1].
    """
    for n, x in (("I", I), ("S", S), ("W", W)):
        _validate_unit(n, x)
    if not (I.shape == S.shape == W.shape):
        raise ValueError("I/S/W 형상 불일치")
    return I * S * W


def season_risk(
    I: np.ndarray,
    S: np.ndarray,
    W_daily: np.ndarray | list[np.ndarray],
) -> np.ndarray:
    """산불조심기간 위험 R(p).

    W_daily 가 단일 (N,) 이면 MVP 시즌통계로 보고 R = I×S×W.
    여러 날 (T 개 (N,) 의 리스트/스택)이면 일별 h 의 기하평균(로그합 평균)으로
    집계해 단발 고위험일의 과대평가를 완화한다.

    Parameters
    ----------
    I, S : numpy.ndarray, shape (N,)
        고정 성분.
    W_daily : numpy.ndarray or list[numpy.ndarray]
        (N,) 시즌통계 또는 (T,N)/리스트 일별 기상.

    Returns
    -------
    numpy.ndarray, shape (N,)
        시즌 위험 R ∈ [0,1].
    """
    W_arr = np.asarray(W_daily, dtype=np.float64)
    if W_arr.ndim == 1:
        R = daily_hazard(I, S, W_arr)
        logger.info("season R(p) [MVP single-W]: mean=%.5f p99=%.5f",
                    float(R.mean()), float(np.quantile(R, 0.99)))
        return R
    if W_arr.ndim != 2:
        raise ValueError(f"W_daily 차원 이상: {W_arr.ndim}")
    # (T, N) 일별 h 의 기하평균: exp(mean_t log(h+eps)).
    eps = 1e-9
    base = (I * S)[None, :]
    h = np.clip(base * W_arr, eps, 1.0)
    R = np.exp(np.mean(np.log(h), axis=0))
    R = np.clip(R, 0.0, 1.0)
    logger.info("season R(p) [%d-day geomean]: mean=%.5f p99=%.5f",
                W_arr.shape[0], float(R.mean()), float(np.quantile(R, 0.99)))
    return R


def blend_exposure(
    S_static: np.ndarray, p_exposure: np.ndarray, alpha: float = 0.5
) -> np.ndarray:
    """정적 S 와 풍하 노출확률 p_exposure 를 블렌딩해 S 성분을 증강(전역 균일 α).

    Parameters
    ----------
    S_static : numpy.ndarray, shape (N,)
        pole_static_overlay.S_p 정규화 값.
    p_exposure : numpy.ndarray, shape (N,)
        exposure 커널 P(노출) ∈ [0,1].
    alpha : float
        블렌딩 가중(0=정적만, 1=노출만).

    Returns
    -------
    numpy.ndarray, shape (N,)
        증강 S ∈ [0,1].
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha 는 [0,1]")
    _validate_unit("S_static", S_static)
    _validate_unit("p_exposure", p_exposure)
    return np.clip((1 - alpha) * S_static + alpha * p_exposure, 0.0, 1.0)


def _percentile_norm(x: np.ndarray) -> np.ndarray:
    """순위 백분위 정규화 → [0,1]. 동점은 평균 백분위, 단조 보존(이상치 강건).

    p_exposure 처럼 0 이 다수인 분포에서 절대값 min-max 보다 안정적이다.
    """
    n = x.shape[0]
    if n == 0:
        return x.astype(np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    if n > 1:
        ranks /= (n - 1)
    return ranks


def conditional_alpha(
    gate: np.ndarray,
    regime_order: list[str],
    yanggan_days: np.ndarray,
    alpha_base: float,
    yeongdong_regime: str,
) -> np.ndarray:
    """전주별 조건부 풍하노출 블렌드 강도 α_p ∈ [0, α_base].

    α_p = α_base × (영동 게이트 가중)_p × (양간강도 정규화)_p.
    영동·고양간 전주에서만 α_p 가 커지고, 영서·산악·저양간은 ≈0 → 전역 희석 방지.

    Parameters
    ----------
    gate : numpy.ndarray, shape (N, R)
        regimes.compute_gate 출력(행합=1).
    regime_order : list[str]
        gate 열 순서(config.REGIMES).
    yanggan_days : numpy.ndarray, shape (N,)
        전주별 양간지풍 일수(원시값; 내부에서 백분위 정규화).
    alpha_base : float
        최대 블렌드 강도(0..1). 0 이면 전역 비활성(정적 S 유지).
    yeongdong_regime : str
        영동 체제명(config.REGIME_YEONGDONG).

    Returns
    -------
    numpy.ndarray, shape (N,)
        조건부 α_p ∈ [0, α_base].

    Raises
    ------
    ValueError
        alpha_base 가 [0,1] 밖이거나 형상 불일치.
    KeyError
        regime_order 에 영동 체제가 없을 때.
    """
    if not 0.0 <= alpha_base <= 1.0:
        raise ValueError("alpha_base 는 [0,1]")
    if gate.ndim != 2 or gate.shape[1] != len(regime_order):
        raise ValueError(f"gate 형상 {gate.shape} 가 regime_order 와 불일치")
    if yanggan_days.shape[0] != gate.shape[0]:
        raise ValueError("yanggan_days 와 gate 행수 불일치")
    if yeongdong_regime not in regime_order:
        raise KeyError(f"regime_order 에 영동 체제 없음: {yeongdong_regime}")
    # 풍하노출은 wind-driven 체제(영동+회랑) 모두에 적용 → 해당 게이트 가중 합산.
    foehn_cols = [regime_order.index(r) for r in config.FOEHN_REGIMES
                  if r in regime_order]
    yd_gate = np.clip(gate[:, foehn_cols].sum(axis=1), 0.0, 1.0)   # 영동+회랑 게이트 [0,1]
    yang_norm = _percentile_norm(yanggan_days)            # 양간강도 [0,1]
    alpha_p = alpha_base * yd_gate * yang_norm
    return np.clip(alpha_p, 0.0, alpha_base)


def blend_exposure_conditional(
    S_static: np.ndarray,
    p_exposure: np.ndarray,
    alpha_p: np.ndarray,
) -> np.ndarray:
    """전주별 조건부 α_p 로 정적 S 와 풍하노출을 블렌딩.

    S_final = (1 − α_p)·S_static + α_p·Pexposure_norm. p_exposure 는 분포 강건성을
    위해 백분위 정규화(_percentile_norm) 후 섞는다(0 다수 분포 안정화).

    Parameters
    ----------
    S_static : numpy.ndarray, shape (N,)
        정적 S(S_p).
    p_exposure : numpy.ndarray, shape (N,)
        풍하 노출확률 P(노출) ∈ [0,1].
    alpha_p : numpy.ndarray, shape (N,)
        전주별 조건부 블렌드 강도(conditional_alpha 출력) ∈ [0,1].

    Returns
    -------
    numpy.ndarray, shape (N,)
        증강 S ∈ [0,1].

    Raises
    ------
    ValueError
        형상 불일치 또는 α_p 범위 위반.
    """
    _validate_unit("S_static", S_static)
    _validate_unit("p_exposure", p_exposure)
    if not (S_static.shape == p_exposure.shape == alpha_p.shape):
        raise ValueError("S_static/p_exposure/alpha_p 형상 불일치")
    if alpha_p.min() < -1e-9 or alpha_p.max() > 1 + 1e-9:
        raise ValueError(f"alpha_p [0,1] 밖: [{alpha_p.min():.4g},{alpha_p.max():.4g}]")
    p_norm = _percentile_norm(p_exposure)
    a = np.clip(alpha_p, 0.0, 1.0)
    return np.clip((1.0 - a) * S_static + a * p_norm, 0.0, 1.0)
