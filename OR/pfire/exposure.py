"""풍하 확산 노출 P(노출) — Rust 커널 래퍼 + numpy 폴백.

설계 근거: 확산/노출 취약 S 를 비등방(풍하) MC 확산 프록시로 추정한다.
성능 본체는 Rust(`pfire_kernels`). 없으면 동일 시그니처 numpy 벡터 폴백을
쓴다(작은 M·S 스모크용). 의미론은 claudedocs/CONTRACT_rust_python.md 준수:
시뮬 s 풍하방향 θ=wind_dir+180, reach_prob=exp(-d/L)*fuel*(1+β·southness),
L=L0*(1+α·max(0,align)·ws/5), 전주 P(노출)=reached 비율(발화원 OR).

좌표 규약(CONTRACT): pole_xy 는 **평면 근사 km**(lon/lat 아님). Rust·폴백 모두
받은 좌표를 그대로 km 로 본다. lon/lat → km 변환은 호출자
(pfire.exposure_engine.lonlat_to_km)가 책임진다.
"""
from __future__ import annotations

import logging

import numpy as np

from pfire import config

logger = logging.getLogger(__name__)

try:  # 우선 Rust 커널.
    import pfire_kernels  # type: ignore

    _HAS_RUST = True
    logger.info("pfire_kernels(Rust) 로드 — Rust 커널 사용")
except ImportError:
    pfire_kernels = None  # type: ignore
    _HAS_RUST = False
    logger.info("pfire_kernels 없음 — numpy 폴백 사용")


def has_rust() -> bool:
    """Rust 커널 사용 가능 여부."""
    return _HAS_RUST


def _validate_inputs(
    pole_xy: np.ndarray,
    ignition_idx: np.ndarray,
    wind_dir_deg: np.ndarray,
    wind_speed: np.ndarray,
    fuel: np.ndarray,
    southness: np.ndarray,
) -> None:
    """계약 시그니처 형상/범위 검증."""
    n = pole_xy.shape[0]
    if pole_xy.ndim != 2 or pole_xy.shape[1] != 2:
        raise ValueError(f"pole_xy 는 (N,2): {pole_xy.shape}")
    if wind_dir_deg.shape != wind_speed.shape:
        raise ValueError("wind_dir_deg/wind_speed 형상 불일치")
    if fuel.shape[0] != n or southness.shape[0] != n:
        raise ValueError("fuel/southness 길이 != N")
    if ignition_idx.size and (ignition_idx.max() >= n or ignition_idx.min() < 0):
        raise ValueError("ignition_idx 범위 밖")


def _fallback_simulate(
    pole_xy: np.ndarray,
    ignition_idx: np.ndarray,
    wind_dir_deg: np.ndarray,
    wind_speed: np.ndarray,
    fuel: np.ndarray,
    southness: np.ndarray,
    max_dist_km: float,
    length_scale_km: float,
    wind_aniso: float,
    southness_beta: float,
    seed: int,
) -> np.ndarray:
    """numpy 벡터화 폴백 — 계약 의미론 동일(작은 M·S 스모크).

    발화원 g 마다 반경 max_dist_km 내 전주만 평가. CONTRACT 대로 pole_xy 는
    이미 **평면 근사 km** 좌표(lon/lat 아님)로 받는다(Rust 커널과 동일). 좌표
    변환은 호출자(exposure_engine.lonlat_to_km)가 책임진다. 시뮬 s 별 독립
    시드(seed+s)로 reached 를 OR 결합, 전주 P(노출)=reached 비율.
    """
    n = pole_xy.shape[0]
    p_exposure = np.zeros(n, dtype=np.float64)
    if ignition_idx.size == 0 or wind_dir_deg.size == 0:
        return p_exposure

    # CONTRACT: pole_xy 는 이미 평면 근사 km. 추가 변환 없음(Rust 커널과 동일).
    x_km = pole_xy[:, 0]
    y_km = pole_xy[:, 1]
    fuel_c = np.clip(fuel, 0.0, 1.0)
    south_fac = 1.0 + southness_beta * southness

    n_sims = wind_dir_deg.shape[0]
    reached_any = np.zeros(n, dtype=np.float64)  # 시뮬 OR 누적

    for s in range(n_sims):
        rng = np.random.default_rng(seed + s)
        theta = np.deg2rad(wind_dir_deg[s] + 180.0)  # from→to
        ws = wind_speed[s]
        reached_s = np.zeros(n, dtype=bool)
        for g in ignition_idx:
            dx = x_km - x_km[g]
            dy = y_km - y_km[g]
            d = np.sqrt(dx * dx + dy * dy)
            near = d <= max_dist_km
            if not near.any():
                continue
            phi = np.arctan2(dy[near], dx[near])
            align = np.cos(phi - theta)
            L = length_scale_km * (1.0 + wind_aniso * np.maximum(0.0, align) * (ws / 5.0))
            L = np.maximum(L, 1e-6)
            reach_prob = np.exp(-d[near] / L) * fuel_c[near] * south_fac[near]
            reach_prob = np.clip(reach_prob, 0.0, 1.0)
            draw = rng.random(reach_prob.shape[0]) < reach_prob
            idx_near = np.nonzero(near)[0]
            reached_s[idx_near[draw]] = True
        reached_any += reached_s
    p_exposure = reached_any / float(n_sims)
    return np.clip(p_exposure, 0.0, 1.0)


def simulate_exposure(
    pole_xy: np.ndarray,
    ignition_idx: np.ndarray,
    wind_dir_deg: np.ndarray,
    wind_speed: np.ndarray,
    fuel: np.ndarray,
    southness: np.ndarray,
    max_dist_km: float = config.SPREAD_MAX_DIST_KM,
    length_scale_km: float = config.SPREAD_LENGTH_SCALE_KM,
    wind_aniso: float = config.SPREAD_WIND_ANISO,
    southness_beta: float = config.SPREAD_SOUTHNESS_BETA,
    seed: int = config.SEED,
) -> np.ndarray:
    """전주별 풍하 노출확률 P(노출) ∈ [0,1] (계약 단일 진입 함수).

    Rust 커널이 있으면 위임하고, 없으면 numpy 폴백을 쓴다. 시그니처·반환
    형상/의미는 claudedocs/CONTRACT_rust_python.md 와 동일.

    Parameters
    ----------
    pole_xy : numpy.ndarray, shape (N, 2)
        전주 (lon, lat). 내부에서 평면근사 km 변환.
    ignition_idx : numpy.ndarray, shape (M,)
        발화 후보 전주 인덱스(uint32 권장).
    wind_dir_deg : numpy.ndarray, shape (S,)
        MC 표집 풍향(기상학적 from-방향, 도).
    wind_speed : numpy.ndarray, shape (S,)
        MC 표집 풍속(m/s).
    fuel : numpy.ndarray, shape (N,)
        전주별 연료/가연성 [0,1].
    southness : numpy.ndarray, shape (N,)
        남서사면 정렬도 [-1,1].
    max_dist_km, length_scale_km, wind_aniso, southness_beta : float
        확산 파라미터(기본값 config).
    seed : int
        재현성 시드.

    Returns
    -------
    numpy.ndarray, shape (N,)
        전주별 P(노출) ∈ [0,1].
    """
    pole_xy = np.asarray(pole_xy, dtype=np.float64)
    ignition_idx = np.asarray(ignition_idx, dtype=np.uint32)
    wind_dir_deg = np.asarray(wind_dir_deg, dtype=np.float64)
    wind_speed = np.asarray(wind_speed, dtype=np.float64)
    fuel = np.asarray(fuel, dtype=np.float64)
    southness = np.asarray(southness, dtype=np.float64)
    _validate_inputs(pole_xy, ignition_idx, wind_dir_deg, wind_speed, fuel, southness)

    if _HAS_RUST:
        return np.asarray(
            pfire_kernels.simulate_exposure(
                pole_xy, ignition_idx, wind_dir_deg, wind_speed, fuel, southness,
                max_dist_km, length_scale_km, wind_aniso, southness_beta, seed,
            ),
            dtype=np.float64,
        )
    return _fallback_simulate(
        pole_xy, ignition_idx, wind_dir_deg, wind_speed, fuel, southness,
        max_dist_km, length_scale_km, wind_aniso, southness_beta, seed,
    )
