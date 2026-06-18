"""체제별 물리식 전문가 — 발화 성향 I(p) 성분.

설계 근거: 같은 피처(발화기하 forest/road/powerline + 기상 fwi/yanggan + 연료 fuel)를
체제마다 다른 가중(config.EXPERT_WEIGHTS)으로 결합한다 → "어디서 왜 위험한가"가
체제별로 다르게 표현된다(영서=도로/입산자, 영동=송전선/양간풍, 산간=연료/산림).
거리 피처는 가까울수록 위험↑ 이므로 exp(-d/scale) 로 0..1 정규화한 순수 함수.
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

from pfire import config

logger = logging.getLogger(__name__)

# 거리 감쇠 스케일(m) — 가까울수록 위험. EDA 분포(forest median 37m, road 90m,
# powerline 2.1km) 기준으로 설정. exp(-d/scale).
DIST_SCALE_FOREST_M: float = 100.0
DIST_SCALE_ROAD_M: float = 300.0
DIST_SCALE_POWERLINE_M: float = 1500.0


def _decay(dist_m: np.ndarray, scale_m: float) -> np.ndarray:
    """근접도 점수: exp(-d/scale) ∈ (0,1]. 가까울수록 1."""
    return np.exp(-np.clip(dist_m, 0.0, None) / scale_m)


def _minmax01(x: np.ndarray) -> np.ndarray:
    """[0,1] min-max 정규화. 상수열은 0.5 로."""
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if hi - lo < 1e-12:
        return np.full_like(x, 0.5, dtype=np.float64)
    return (x - lo) / (hi - lo)


def build_ignition_features(master: pl.DataFrame) -> dict[str, np.ndarray]:
    """전주별 발화 성향 정규화 피처(0..1, 클수록 위험)를 만든다.

    Parameters
    ----------
    master : polars.DataFrame
        dist_to_forest, dist_to_road, dist_to_powerline, fwi_q90,
        yanggan_days, mu_flammability, ndvi 포함.

    Returns
    -------
    dict[str, numpy.ndarray]
        키 forest/road/powerline/fwi/yanggan/fuel → (N,) 정규화 점수.
        config.EXPERT_WEIGHTS 의 키와 일치한다.

    Raises
    ------
    KeyError
        필요한 컬럼이 없을 때.
    """
    need = ["dist_to_forest", "dist_to_road", "dist_to_powerline",
            "fwi_q90", "yanggan_days", "mu_flammability"]
    for c in need:
        if c not in master.columns:
            raise KeyError(f"발화피처에 필요한 컬럼 없음: {c}")

    d_forest = master["dist_to_forest"].to_numpy().astype(np.float64)
    d_road = master["dist_to_road"].to_numpy().astype(np.float64)
    d_power = master["dist_to_powerline"].to_numpy().astype(np.float64)
    fwi = master["fwi_q90"].to_numpy().astype(np.float64)
    yanggan = master["yanggan_days"].to_numpy().astype(np.float64)
    fuel = master["mu_flammability"].to_numpy().astype(np.float64)

    feats = {
        "forest": _decay(d_forest, DIST_SCALE_FOREST_M),
        "road": _decay(d_road, DIST_SCALE_ROAD_M),
        "powerline": _decay(d_power, DIST_SCALE_POWERLINE_M),
        "fwi": _minmax01(fwi),
        "yanggan": _minmax01(yanggan),
        "fuel": np.clip(fuel, 0.0, 1.0),
    }
    for k, v in feats.items():
        if v.shape[0] != master.height:
            raise ValueError(f"발화피처 '{k}' 길이 불일치")
    return feats


def expert_score(feats: dict[str, np.ndarray], regime: str) -> np.ndarray:
    """단일 체제 전문가의 발화 성향 점수(가중합, 0..1).

    Parameters
    ----------
    feats : dict[str, numpy.ndarray]
        build_ignition_features 출력.
    regime : str
        config.REGIMES 중 하나.

    Returns
    -------
    numpy.ndarray, shape (N,)
        해당 체제 전문가의 I 점수 ∈ [0,1].

    Raises
    ------
    KeyError
        알 수 없는 체제명일 때.
    """
    if regime not in config.EXPERT_WEIGHTS:
        raise KeyError(f"알 수 없는 체제: {regime}")
    w = config.EXPERT_WEIGHTS[regime]
    wsum = sum(w.values())
    if wsum <= 0:
        raise ValueError(f"체제 {regime} 가중 합 <= 0")
    score = np.zeros(next(iter(feats.values())).shape[0], dtype=np.float64)
    for key, weight in w.items():
        if key not in feats:
            raise KeyError(f"전문가 가중 키 '{key}' 에 해당하는 피처 없음")
        score += weight * feats[key]
    return score / wsum  # 가중평균 → [0,1] 보장


def ignition_propensity(
    master: pl.DataFrame, gate_weights: np.ndarray, regime_order: list[str]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """MoE 발화 성향 I(p) = Σ_r gate_r * expert_r.

    Parameters
    ----------
    master : polars.DataFrame
        발화 피처 컬럼 포함.
    gate_weights : numpy.ndarray, shape (N, 3)
        regimes.compute_gate 출력(행합=1).
    regime_order : list[str]
        gate_weights 열 순서.

    Returns
    -------
    I : numpy.ndarray, shape (N,)
        MoE 결합 발화 성향 ∈ [0,1].
    per_expert : dict[str, numpy.ndarray]
        체제별 전문가 점수(검증/해석용).
    """
    if gate_weights.shape[0] != master.height:
        raise ValueError("게이트 행수와 master 행수 불일치")
    feats = build_ignition_features(master)
    per_expert = {r: expert_score(feats, r) for r in regime_order}
    expert_mat = np.stack([per_expert[r] for r in regime_order], axis=1)  # (N,3)
    I = np.sum(gate_weights * expert_mat, axis=1)
    I = np.clip(I, 0.0, 1.0)
    logger.info("ignition I(p): mean=%.4f p50=%.4f p99=%.4f",
                float(I.mean()), float(np.median(I)), float(np.quantile(I, 0.99)))
    return I, per_expert
