"""기상 위험 W(grid, t) 성분.

설계 근거: W 는 "그날 격자 기상" 의 위험. MVP 는 전주별 사전계산 시즌통계
(pole_features.fwi_q90 / fwi_high_danger_days / yanggan_days, 또는 pole_fwi_obs)를
정규화해 W 로 쓴다. 추가로 daily_high_danger_days() 가 fwi_station_daily 에서
전주→최근접 관측소 매핑 후 산불조심기간(2~5월) 일별 FWI 고위험일을 집계하는
조기경보 프레이밍 경로를 제공한다(일별 엔진).
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

from pfire import config

logger = logging.getLogger(__name__)

# 일별 엔진에서 "고위험일" FWI 임계값(관측소 일별 FWI 분포 상위; EDA p75≈10.9).
DAILY_FWI_HIGH_THRESHOLD: float = 10.0


def _minmax01(x: np.ndarray) -> np.ndarray:
    """[0,1] min-max 정규화. 상수열은 0.5."""
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    if hi - lo < 1e-12:
        return np.full_like(x, 0.5, dtype=np.float64)
    return (x - lo) / (hi - lo)


def season_weather(master: pl.DataFrame, source: str = "features") -> np.ndarray:
    """MVP 기상 위험 W(p): 전주 사전계산 시즌통계 정규화 결합.

    fwi_q90(강도)·fwi_high_danger_days(빈도)·yanggan_days(강풍) 를 각각 0..1
    정규화하여 가중 결합. 관측기반(pole_fwi_obs)도 선택 가능.

    Parameters
    ----------
    master : polars.DataFrame
        fwi_q90, fwi_high_danger_days, yanggan_days (또는 *_obs) 포함.
    source : {"features", "obs"}
        features = pole_features 통계, obs = pole_fwi_obs 관측기반.

    Returns
    -------
    numpy.ndarray, shape (N,)
        기상 위험 W ∈ [0,1].

    Raises
    ------
    ValueError
        source 가 부적절하거나 컬럼이 없을 때.
    """
    if source == "features":
        cols = ("fwi_q90", "fwi_high_danger_days", "yanggan_days")
    elif source == "obs":
        cols = ("fwi_q90_obs", "fwi_hdd_obs", "yanggan_days")
    else:
        raise ValueError(f"source 는 'features'|'obs' 여야 함: {source!r}")
    for c in cols:
        if c not in master.columns:
            raise ValueError(f"기상 컬럼 없음: {c} (source={source})")

    intensity = _minmax01(master[cols[0]].to_numpy().astype(np.float64))
    frequency = _minmax01(master[cols[1]].to_numpy().astype(np.float64))
    wind = _minmax01(master[cols[2]].to_numpy().astype(np.float64))

    # 강도 0.5 + 빈도 0.3 + 강풍(양간) 0.2 — 영동 양간풍 신호 반영.
    W = 0.5 * intensity + 0.3 * frequency + 0.2 * wind
    W = np.clip(W, 0.0, 1.0)
    logger.info("season W(p) [%s]: mean=%.4f p50=%.4f p99=%.4f",
                source, float(W.mean()), float(np.median(W)),
                float(np.quantile(W, 0.99)))
    return W


def _nearest_station(
    pole_xy: np.ndarray, stn_xy: np.ndarray, stn_ids: np.ndarray
) -> np.ndarray:
    """각 전주를 최근접 관측소에 매핑(평면근사 거리).

    Parameters
    ----------
    pole_xy : numpy.ndarray, shape (N, 2)
        전주 (lon, lat).
    stn_xy : numpy.ndarray, shape (K, 2)
        관측소 (lon, lat).
    stn_ids : numpy.ndarray, shape (K,)
        관측소 id.

    Returns
    -------
    numpy.ndarray, shape (N,)
        각 전주의 최근접 관측소 id.
    """
    cos_lat = np.cos(np.deg2rad(config.LAT0_DEG))
    out = np.empty(pole_xy.shape[0], dtype=stn_ids.dtype)
    # 청크로 (N×K) 거리 메모리 폭발 방지.
    chunk = 200_000
    sx = stn_xy[:, 0] * cos_lat
    sy = stn_xy[:, 1]
    for start in range(0, pole_xy.shape[0], chunk):
        end = min(start + chunk, pole_xy.shape[0])
        px = pole_xy[start:end, 0] * cos_lat
        py = pole_xy[start:end, 1]
        d2 = (px[:, None] - sx[None, :]) ** 2 + (py[:, None] - sy[None, :]) ** 2
        out[start:end] = stn_ids[d2.argmin(axis=1)]
    return out


def daily_high_danger_days(
    master: pl.DataFrame,
    station_daily: pl.DataFrame,
    stations: pl.DataFrame,
    years: tuple[int, ...] | None = None,
) -> np.ndarray:
    """일별 엔진: 전주별 산불조심기간 평균 "고위험일 수" 를 집계해 W 로 정규화.

    각 전주를 최근접 관측소에 매핑 후, 해당 관측소의 산불조심기간(2~5월)
    일별 FWI 가 임계값을 넘은 날 수를 연 평균하여 전주에 부여한다(조기경보
    프레이밍). 정규화된 [0,1] 점수를 반환.

    Parameters
    ----------
    master : polars.DataFrame
        lon, lat 포함.
    station_daily : polars.DataFrame
        stn, month, year, fwi 포함.
    stations : polars.DataFrame
        stnId, lon, lat 포함.
    years : tuple[int] or None
        집계 연도 제한(None=전체).

    Returns
    -------
    numpy.ndarray, shape (N,)
        일별엔진 기반 W ∈ [0,1].
    """
    sd = station_daily.filter(pl.col("month").is_in(list(config.SEASON_MONTHS)))
    if years is not None:
        sd = sd.filter(pl.col("year").is_in(list(years)))
    # 관측소·연도별 고위험일 수 → 관측소별 연평균.
    per_stn_year = (
        sd.with_columns((pl.col("fwi") >= DAILY_FWI_HIGH_THRESHOLD).alias("hd"))
        .group_by(["stn", "year"])
        .agg(pl.col("hd").sum().alias("hd_days"))
        .group_by("stn")
        .agg(pl.col("hd_days").mean().alias("hdd_mean"))
    )
    stn_map = dict(zip(per_stn_year["stn"].to_list(),
                       per_stn_year["hdd_mean"].to_list()))

    stn_xy = stations.select(["lon", "lat"]).to_numpy().astype(np.float64)
    stn_ids = stations["stnId"].to_numpy()
    pole_xy = master.select(["lon", "lat"]).to_numpy().astype(np.float64)

    nearest = _nearest_station(pole_xy, stn_xy, stn_ids)
    global_mean = float(np.mean(list(stn_map.values()))) if stn_map else 0.0
    hdd = np.array([stn_map.get(int(s), global_mean) for s in nearest],
                   dtype=np.float64)
    W = _minmax01(hdd)
    logger.info("daily-engine W(p): stations=%d global_hdd_mean=%.2f",
                len(stn_map), global_mean)
    return W
