"""풍하 노출 v2.2 — 변별 가능한 발화가중 기대도즈(타원 + 국지정규화).

설계 근거(exposure_v2/README.md): 기존 OR 노출은 영동에서 P≈1 로 포화(변별 0).
이를 (1)발화가중 기대도즈(OR→합), (2)Anderson LWR 타원 커널, (3)격자 국지정규화로
바꿔 변별 그라데이션을 만든다. 검증: 영동 IQR 0.166→0.290, OR-blend(w=0.5)에서
공간CV recall +0.0071(5 fold-seed 신호, 무작위-공간 격차≈0). 자세히는 exposure_v2/.

산출 dose01 ∈ [0,1] 은 위험 결합용:  R ← 1−(1−R)(1−w·dose01),  w=config.EXPOSURE_V2_BLEND_W.
"""
from __future__ import annotations

import logging

import numpy as np

from pfire import config, exposure_engine, geo

logger = logging.getLogger(__name__)


def _anderson_lwr(ws_ms: np.ndarray) -> np.ndarray:
    """Anderson(1983) 길이/폭비. ws m/s → mph. 상한 config.EXPOSURE_V2_LWR_CAP."""
    U = np.asarray(ws_ms) * 2.237
    lwr = 0.936 * np.exp(0.2566 * U) + 0.461 * np.exp(-0.1548 * U) - 0.397
    return np.minimum(lwr, config.EXPOSURE_V2_LWR_CAP)


def compute_dose01(
    master,
    I: np.ndarray,
    gate: np.ndarray,
    regime_order: list[str],
    station_daily=None,
    aws_daily=None,
    *,
    seed: int = config.SEED,
) -> np.ndarray:
    """v2.2 풍하 노출 dose01 ∈ [0,1] (전주별, 변별 그라데이션).

    타원 커널(LWR) · 발화원 I 가중 기대도즈 · 격자 국지정규화 · 순위 백분위.
    scipy.cKDTree 로 반경질의(없으면 ImportError — pyproject 에 scipy 필요).

    Parameters
    ----------
    master : polars.DataFrame
        lon, lat, mu_flammability, mu_southness 포함.
    I : numpy.ndarray (N,)
        발화 성향(발화후보 선정·가중).
    gate : numpy.ndarray (N, R)
        체제 soft 가중.
    regime_order : list[str]
        gate 열 순서.
    station_daily, aws_daily : polars.DataFrame or None
        바람 표집 소스(없으면 기본 분포).
    seed : int
        결정성 시드.

    Returns
    -------
    numpy.ndarray (N,)
        dose01 ∈ [0,1] 순위 백분위.
    """
    from scipy.spatial import cKDTree  # 지역 import(설치 안 됐을 때 명확한 실패)
    from pfire.risk_index import risk_percentile

    S = config.EXPOSURE_V2_N_SIMS
    W_CROSS = config.EXPOSURE_V2_W_CROSS
    L_BACK = config.EXPOSURE_V2_L_BACK
    MAXD = config.SPREAD_MAX_DIST_KM
    BETA = config.SPREAD_SOUTHNESS_BETA
    CELL = config.EXPOSURE_V2_NORM_CELL_KM

    pole_xy = master.select(["lon", "lat"]).to_numpy().astype(np.float64)
    km = geo.lonlat_to_km(pole_xy)
    sources = np.asarray(exposure_engine.ignition_candidates(
        I, gate, regime_order, seed=seed), dtype=np.int64)
    Ig = np.clip(I[sources], 1e-6, None)
    src_km = km[sources]
    fuel = np.clip(master["mu_flammability"].to_numpy().astype(np.float64), 0, 1)
    south_fac = 1.0 + BETA * (master["mu_southness"].to_numpy().astype(np.float64) * 2 - 1)

    winds = {}
    for r in regime_order:
        wd, ws = exposure_engine.sample_wind(r, S, station_daily, aws_daily, seed=seed)
        winds[r] = (np.deg2rad(wd + 180.0), np.asarray(ws))

    tree = cKDTree(km)
    nbr_lists = tree.query_ball_point(src_km, MAXD, workers=-1)

    dose_r = {r: np.zeros(km.shape[0]) for r in regime_order}
    for gi in range(src_km.shape[0]):
        nbr = np.asarray(nbr_lists[gi], dtype=np.int64)
        if nbr.size == 0:
            continue
        dx = km[nbr, 0] - src_km[gi, 0]
        dy = km[nbr, 1] - src_km[gi, 1]
        d2 = dx * dx + dy * dy
        ff = fuel[nbr] * south_fac[nbr]
        for r in regime_order:
            theta, ws = winds[r]
            ct, st = np.cos(theta), np.sin(theta)
            d_par = dx[:, None] * ct[None, :] + dy[:, None] * st[None, :]
            d_perp = np.sqrt(np.clip(d2[:, None] - d_par * d_par, 0.0, None))
            L_along = (_anderson_lwr(ws) * W_CROSS)[None, :]
            along = np.where(d_par >= 0, L_along, L_BACK)
            rr = np.sqrt((d_par / np.maximum(along, 1e-6)) ** 2 + (d_perp / W_CROSS) ** 2)
            reach = np.exp(-rr).mean(1) * ff
            dose_r[r][nbr] += Ig[gi] * np.clip(reach, 0.0, 1.0)

    dose = np.zeros(km.shape[0])
    for ri, r in enumerate(regime_order):
        dose += gate[:, ri] * dose_r[r]

    # 격자 국지정규화(영동 배경 제거)
    ix = np.floor((km[:, 0] - km[:, 0].min()) / CELL).astype(np.int64)
    iy = np.floor((km[:, 1] - km[:, 1].min()) / CELL).astype(np.int64)
    cell = iy * (ix.max() + 1) + ix
    cmean = np.bincount(cell, weights=dose) / np.maximum(np.bincount(cell), 1)
    dose_rel = dose / (cmean[cell] + 1e-9)

    dose01 = risk_percentile(dose_rel) / 100.0
    logger.info("exposure_v2 dose01: mean=%.3f p99=%.3f (발화원 %d, 바람 %d)",
                float(dose01.mean()), float(np.quantile(dose01, 0.99)),
                int(sources.size), S)
    return dose01


def blend_into_risk(R: np.ndarray, dose01: np.ndarray,
                    w: float = config.EXPOSURE_V2_BLEND_W) -> np.ndarray:
    """확률 OR 결합:  R' = 1 − (1 − R)(1 − w·dose01). 발화 랭킹 유지 + 풍하 상승."""
    if not 0.0 <= w <= 1.0:
        raise ValueError("blend weight 는 [0,1]")
    return np.clip(1.0 - (1.0 - R) * (1.0 - w * np.clip(dose01, 0, 1)), 0.0, 1.0)
