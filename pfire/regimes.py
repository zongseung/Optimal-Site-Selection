"""체제(regime) soft 게이트 — 전주 피처 → (N, R) 체제 가중(행합=1).
R=레짐 수(len(config.REGIMES), 현재 4: 영동·회랑·영서·산악).

설계 근거: 영동/회랑/영서/산악은 발화·확산 메커니즘이 다르다(영동 양간일 2.33·FWI 9.2
vs 영서 0.3·5.24). 양간지풍 회랑(양양·고성·속초)을 별도 체제로 분리한다. 하드 시군
분할 대신 전주 피처(경도·고도·양간일)를 표준화해 체제 앵커와의 거리 기반 softmax 로
soft 가중을 준다 → 경계 전주는 자연 블렌딩, 소표본 시군(고성179·화천364·인제377)도
피처로 안정적으로 배치된다. SGG_TO_REGIME 은 앵커 도출·검증용으로만 쓴다(하드 라벨 아님).
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

from pfire import config

logger = logging.getLogger(__name__)

# 게이트 피처: 경도(영동/영서 분리), 고도(산간), 양간일(영동 강풍 신호).
GATE_FEATURES: tuple[str, ...] = ("lon", "elevation", "yanggan_days")
# softmax 온도 — 작을수록 하드. EDA 기반 적당히 부드럽게.
GATE_TEMPERATURE: float = 0.6


def _standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """열별 z-표준화. (표준화행렬, 평균, 표준편차) 반환."""
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (x - mu) / sd, mu, sd


def _regime_anchors(
    feats_z: np.ndarray, regime_of_pole: np.ndarray
) -> dict[str, np.ndarray]:
    """SGG_TO_REGIME 로 매핑된 전주들의 표준화피처 중심 = 체제 앵커.

    Parameters
    ----------
    feats_z : numpy.ndarray, shape (N, F)
        표준화된 게이트 피처.
    regime_of_pole : numpy.ndarray of str, shape (N,)
        시군→체제 매핑 결과(미매핑은 빈 문자열).

    Returns
    -------
    dict[str, numpy.ndarray]
        체제명 → 앵커 벡터 (F,).
    """
    anchors: dict[str, np.ndarray] = {}
    for r in config.REGIMES:
        mask = regime_of_pole == r
        if mask.sum() == 0:
            logger.warning("체제 %s 앵커 표본 0 → 전역평균 사용", r)
            anchors[r] = feats_z.mean(axis=0)
        else:
            anchors[r] = feats_z[mask].mean(axis=0)
    return anchors


def compute_gate(master: pl.DataFrame) -> tuple[np.ndarray, list[str]]:
    """전주별 체제 soft 가중을 계산한다.

    표준화된 게이트 피처와 체제 앵커(시군 매핑으로 도출) 간 유클리드 거리를
    음의 거리 제곱 / 온도 로 softmax 하여 행합=1 가중을 만든다.

    Parameters
    ----------
    master : polars.DataFrame
        최소 GATE_FEATURES + 'sgg' 컬럼을 포함.

    Returns
    -------
    weights : numpy.ndarray, shape (N, R)
        체제 가중(행합=1). 열 순서는 반환되는 regime_order 와 동일.
    regime_order : list[str]
        config.REGIMES 순서(yeongdong, corridor, yeongseo, mountain).

    Raises
    ------
    KeyError
        필요한 게이트 피처가 없을 때.
    """
    for c in (*GATE_FEATURES, "sgg"):
        if c not in master.columns:
            raise KeyError(f"게이트에 필요한 컬럼 없음: {c}")

    feats = master.select(GATE_FEATURES).to_numpy().astype(np.float64)
    feats_z, _, _ = _standardize(feats)

    sgg = master["sgg"].to_list()
    regime_of_pole = np.array(
        [config.SGG_TO_REGIME.get(s, "") for s in sgg], dtype=object
    )
    anchors = _regime_anchors(feats_z, regime_of_pole)

    regime_order = list(config.REGIMES)
    anchor_mat = np.stack([anchors[r] for r in regime_order], axis=0)  # (3, F)

    # 각 전주-앵커 거리 제곱 (N, R)
    diff = feats_z[:, None, :] - anchor_mat[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)

    logits = -dist2 / max(GATE_TEMPERATURE, 1e-6)
    logits -= logits.max(axis=1, keepdims=True)  # 수치안정
    ex = np.exp(logits)
    weights = ex / ex.sum(axis=1, keepdims=True)

    _validate_gate(weights)
    _log_gate_agreement(weights, regime_of_pole, regime_order)
    return weights, regime_order


def _validate_gate(weights: np.ndarray) -> None:
    """게이트 출력 무결성: 행합=1, [0,1], NaN 없음."""
    if weights.ndim != 2 or weights.shape[1] != len(config.REGIMES):
        raise ValueError(f"게이트 형상 이상: {weights.shape}")
    rs = weights.sum(axis=1)
    if not np.allclose(rs, 1.0, atol=1e-6):
        raise ValueError(f"게이트 행합 != 1 (max dev {np.abs(rs - 1).max():.2e})")
    if np.any(weights < -1e-9) or np.any(weights > 1 + 1e-9):
        raise ValueError("게이트 가중 [0,1] 범위 위반")
    if np.isnan(weights).any():
        raise ValueError("게이트에 NaN 존재")


def _log_gate_agreement(
    weights: np.ndarray, regime_of_pole: np.ndarray, regime_order: list[str]
) -> None:
    """soft 게이트 argmax 와 시군 하드 매핑의 일치율 로깅(검증)."""
    argmax = np.array([regime_order[i] for i in weights.argmax(axis=1)])
    mapped = regime_of_pole != ""
    if mapped.sum():
        agree = float((argmax[mapped] == regime_of_pole[mapped]).mean())
        logger.info("게이트 argmax vs 시군매핑 일치율 = %.3f (매핑전주 %d)",
                    agree, int(mapped.sum()))
    shares = weights.mean(axis=0)
    logger.info("체제 평균가중 %s",
                {r: round(float(s), 3) for r, s in zip(regime_order, shares)})
