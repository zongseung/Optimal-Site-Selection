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
    """정적 S 와 풍하 노출확률 p_exposure 를 블렌딩해 S 성분을 증강.

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
