"""좌표 기하 유틸 — lon/lat ↔ 평면 근사 km 변환(단일 구현).

설계 근거: 강원 영역의 거리/확산 계산은 위경도를 강원 대표 위도(config.LAT0_DEG)
기준 cos 보정 평면(km)으로 근사한다. 같은 변환식이 exposure/calibrate/validate
여러 곳에서 반복되던 것을 여기 한 곳으로 모은다(동작 동일, 식 불변).

변환식(CONTRACT 와 동일):
    x_km = lon * cos(LAT0) * (π/180) * EARTH_R_KM
    y_km = lat *            (π/180) * EARTH_R_KM
"""
from __future__ import annotations

import numpy as np

from pfire import config

# cos(LAT0) 경도 보정 계수와 도→라디안 환산 계수.
#   곱셈 결합 순서는 기존 인라인 구현(lon * cos_lat * (π/180) * R)과 정확히
#   동일하게 유지한다 → 부동소수점 비트까지 같은 결과(회귀 방지).
_COS_LAT0: float = float(np.cos(np.deg2rad(config.LAT0_DEG)))
_RAD: float = np.pi / 180.0


def lonlat_to_km(lonlat: np.ndarray) -> np.ndarray:
    """lon/lat (N,2) → 평면 근사 km (N,2). config.LAT0_DEG cos 보정.

    Parameters
    ----------
    lonlat : numpy.ndarray, shape (N, 2)
        (lon, lat) 도(degree).

    Returns
    -------
    numpy.ndarray, shape (N, 2)
        (x_km, y_km) 평면 근사 좌표(float64).
    """
    lonlat = np.asarray(lonlat, dtype=np.float64)
    x_km = lonlat[:, 0] * _COS_LAT0 * _RAD * config.EARTH_R_KM
    y_km = lonlat[:, 1] * _RAD * config.EARTH_R_KM
    return np.column_stack([x_km, y_km]).astype(np.float64)


def point_to_km(lon: float, lat: float) -> tuple[float, float]:
    """단일 (lon, lat) 점 → 평면 근사 (x_km, y_km).

    Parameters
    ----------
    lon, lat : float
        경도/위도(도).

    Returns
    -------
    tuple[float, float]
        (x_km, y_km).
    """
    x_km = lon * _COS_LAT0 * _RAD * config.EARTH_R_KM
    y_km = lat * _RAD * config.EARTH_R_KM
    return x_km, y_km
