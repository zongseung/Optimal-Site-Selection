"""풍하 노출 P(노출) 실연동 — 발화후보·바람표집·평면km 좌표 준비.

설계 근거: 정적 S_p 를 비등방(풍하) MC 확산 프록시로 증강한다(기획서 5장).
발화점 라벨은 학습 타깃이 아니므로(검증·임계 앵커 전용), 발화후보는
**발화성향 I 상위 분위 전주**로 둔다("위험 환경에서 불이 시작된다" 가정).
바람은 관측 분포에서 표집하되, 영동(양간지풍)은 풍향 prior(서풍 ≈270°±좁게)를
쓴다 — 일평균 AWS 풍향은 해풍에 희석돼 강풍 산불 이벤트의 서풍 신호를 못 담는다.
풍속은 관측 FWI 일자료(fwi_station_daily.ws, 강풍 신호 보존)에서 표집.

좌표: exposure 커널(Rust + 폴백)은 CONTRACT 상 **평면 근사 km** 좌표를 받는다
(config.LAT0_DEG cos 보정). 여기서 lon/lat → km 변환을 책임진다.
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

from pfire import config, exposure, geo

logger = logging.getLogger(__name__)

# 발화후보 분위: I 상위 IGNITION_TOP_Q 전주를 후보로(설비/입산 무관, 위험환경 가정).
IGNITION_TOP_Q: float = 0.02
# 발화후보 상한(성능·과밀 방지). 상위 분위가 이보다 많으면 균등 thinning.
IGNITION_MAX: int = 8000
# 양간지풍 prior(영동): 풍향 평균·표준편차(기상학적 from-방향, 도). 서풍≈270°±좁게.
YANGGAN_DIR_MEAN_DEG: float = 270.0
YANGGAN_DIR_STD_DEG: float = 20.0
# 비영동 풍향: 관측 분포에서 표집(없으면 넓은 표집).


def _regime_seed_offset(regime: str) -> int:
    """체제별 결정적 시드 오프셋(체제마다 다른 바람 표집을 재현 가능하게).

    config.REGIMES 내 인덱스를 오프셋으로 쓴다. (이전 구현은 hash(regime) 를
    썼으나 파이썬 문자열 hash 는 PYTHONHASHSEED 로 프로세스마다 salt 되어 같은
    seed 라도 실행마다 결과가 달라졌다 — CONTRACT 의 결정성 위반. 인덱스 기반은
    프로세스 무관하게 항상 같다.)

    Parameters
    ----------
    regime : str
        config.REGIMES 중 하나. 미등록 체제는 오프셋 0.

    Returns
    -------
    int
        결정적 시드 오프셋.
    """
    try:
        return config.REGIMES.index(regime)
    except ValueError:
        return 0


def lonlat_to_km(pole_xy: np.ndarray) -> np.ndarray:
    """lon/lat (N,2) → 평면 근사 km (N,2). config.LAT0_DEG cos 보정.

    :func:`pfire.geo.lonlat_to_km` 의 얇은 래퍼(호환 API 유지). 좌표 변환의
    단일 구현은 :mod:`pfire.geo` 에 있다.

    Parameters
    ----------
    pole_xy : numpy.ndarray, shape (N, 2)
        전주 (lon, lat) 도.

    Returns
    -------
    numpy.ndarray, shape (N, 2)
        (x_km, y_km) 평면 근사.
    """
    return geo.lonlat_to_km(pole_xy)


def ignition_candidates(I: np.ndarray, gate: np.ndarray, regime_order: list[str],
                        top_q: float = IGNITION_TOP_Q,
                        max_n: int = IGNITION_MAX,
                        seed: int = config.SEED) -> np.ndarray:
    """발화후보 = I 상위 분위 전주(발화점 라벨 미사용).

    상위 분위가 max_n 을 넘으면 I 가중 표집으로 thinning(고위험 우선·결정성).

    Parameters
    ----------
    I : numpy.ndarray, shape (N,)
        발화 성향(MoE). 높을수록 위험.
    gate, regime_order : 미사용(시그니처 호환·향후 체제별 후보 확장 여지).
    top_q : float
        상위 분위(0..1).
    max_n : int
        후보 상한.
    seed : int
        결정성 시드.

    Returns
    -------
    numpy.ndarray, shape (M,)
        발화후보 전주 인덱스(uint32).
    """
    n = I.shape[0]
    thr = float(np.quantile(I, 1.0 - top_q))
    cand = np.nonzero(I >= thr)[0]
    if cand.size == 0:
        cand = np.argsort(-I)[:max(1, int(top_q * n))]
    if cand.size > max_n:
        rng = np.random.default_rng(seed)
        w = I[cand]
        w = w / w.sum()
        cand = rng.choice(cand, size=max_n, replace=False, p=w)
        cand.sort()
    logger.info("발화후보: I>=%.4g 상위 %d개(상한 %d)", thr, cand.size, max_n)
    return cand.astype(np.uint32)


def _season_wind_speed(station_daily: pl.DataFrame, n_sims: int,
                       rng: np.random.Generator) -> np.ndarray:
    """산불조심기간 관측 풍속 분포에서 n_sims 표집(fwi_station_daily.ws).

    ws 가 없거나 비면 기본 분포(3~12 m/s 균등)로 폴백.
    """
    if station_daily is None or "ws" not in station_daily.columns:
        return rng.uniform(3.0, 12.0, n_sims)
    sd = station_daily.filter(
        pl.col("month").is_in(list(config.SEASON_MONTHS)) & (pl.col("ws") >= 0)
    )
    ws_pool = sd["ws"].drop_nulls().to_numpy().astype(np.float64)
    if ws_pool.size == 0:
        return rng.uniform(3.0, 12.0, n_sims)
    return rng.choice(ws_pool, size=n_sims, replace=True)


def _season_wind_dir(aws_daily: pl.DataFrame | None, n_sims: int,
                     rng: np.random.Generator) -> np.ndarray:
    """비영동 풍향 표집: 관측(aws_obs_daily.wd, 유효치) 분포 또는 넓은 표집."""
    if aws_daily is not None and "wd" in aws_daily.columns:
        df = aws_daily
        if "month" not in df.columns and "obs_date" in df.columns:
            df = df.with_columns(pl.col("obs_date").dt.month().alias("month"))
        if "month" in df.columns:
            df = df.filter(pl.col("month").is_in(list(config.SEASON_MONTHS)))
        wd_pool = df.filter(pl.col("wd") >= 0)["wd"].drop_nulls().to_numpy().astype(np.float64)
        if wd_pool.size > 0:
            return rng.choice(wd_pool, size=n_sims, replace=True)
    return rng.uniform(0.0, 360.0, n_sims)


def sample_wind(regime: str, n_sims: int,
                station_daily: pl.DataFrame | None = None,
                aws_daily: pl.DataFrame | None = None,
                seed: int = config.SEED) -> tuple[np.ndarray, np.ndarray]:
    """체제별 MC 풍향·풍속 표집.

    영동·회랑(config.FOEHN_REGIMES)은 양간지풍 prior(서풍 ≈270°±좁게)로 풍향 표집,
    그 외는 관측 풍향 분포(또는 넓은 표집). 풍속은 산불조심기간 관측 분포에서 표집.

    Parameters
    ----------
    regime : str
        config.REGIMES 중 하나(영동만 prior 분기).
    n_sims : int
        MC 시뮬 수(S).
    station_daily : polars.DataFrame or None
        fwi_station_daily (ws 표집용).
    aws_daily : polars.DataFrame or None
        aws_obs_daily (비영동 풍향 표집용).
    seed : int
        결정성 시드.

    Returns
    -------
    wind_dir_deg : numpy.ndarray, shape (S,)
        기상학적 from-방향(도).
    wind_speed : numpy.ndarray, shape (S,)
        풍속(m/s).
    """
    rng = np.random.default_rng(seed + _regime_seed_offset(regime))
    if regime in config.FOEHN_REGIMES:
        wd = (rng.normal(YANGGAN_DIR_MEAN_DEG, YANGGAN_DIR_STD_DEG, n_sims)) % 360.0
    else:
        wd = _season_wind_dir(aws_daily, n_sims, rng)
    ws = _season_wind_speed(station_daily, n_sims, rng)
    return wd.astype(np.float64), ws.astype(np.float64)


def compute_exposure(
    master: pl.DataFrame,
    I: np.ndarray,
    gate: np.ndarray,
    regime_order: list[str],
    station_daily: pl.DataFrame | None = None,
    aws_daily: pl.DataFrame | None = None,
    n_sims: int = config.SPREAD_N_SIMS,
    top_q: float = IGNITION_TOP_Q,
    seed: int = config.SEED,
) -> np.ndarray:
    """풍하 노출확률 P(노출) 실연동(체제별 바람 prior로 시뮬 결합).

    발화후보(I 상위 분위)에서, 체제별로 표집한 바람으로 exposure 커널을
    돌려 P(노출)을 얻고 체제별 결과를 전주 우세체제 가중으로 결합한다.
    영동은 양간 서풍 prior 라 풍하(동쪽)로 신장된 노출, 영서/산간은 관측 풍향.

    Parameters
    ----------
    master : polars.DataFrame
        lon, lat, mu_flammability, mu_southness 포함.
    I : numpy.ndarray, shape (N,)
        발화 성향(발화후보 선정용).
    gate : numpy.ndarray, shape (N, 3)
        체제 soft 가중(행합=1) — 체제별 노출 결합 가중.
    regime_order : list[str]
        gate 열 순서.
    station_daily, aws_daily : polars.DataFrame or None
        바람 표집 소스.
    n_sims : int
        체제별 MC 시뮬 수.
    top_q : float
        발화후보 상위 분위.
    seed : int
        결정성 시드.

    Returns
    -------
    numpy.ndarray, shape (N,)
        풍하 노출확률 P(노출) ∈ [0,1].
    """
    pole_lonlat = master.select(["lon", "lat"]).to_numpy().astype(np.float64)
    pole_km = lonlat_to_km(pole_lonlat)
    fuel = np.clip(master["mu_flammability"].to_numpy().astype(np.float64), 0.0, 1.0)
    southness = master["mu_southness"].to_numpy().astype(np.float64) * 2.0 - 1.0

    ign = ignition_candidates(I, gate, regime_order, top_q=top_q, seed=seed)

    # 체제별 노출을 우세체제 가중으로 결합(영동만 양간 prior로 풍하 동향).
    p_combined = np.zeros(pole_km.shape[0], dtype=np.float64)
    for ri, regime in enumerate(regime_order):
        wd, ws = sample_wind(regime, n_sims, station_daily, aws_daily, seed=seed)
        p_r = exposure.simulate_exposure(
            pole_km, ign, wd, ws, fuel, southness,
            max_dist_km=config.SPREAD_MAX_DIST_KM,
            length_scale_km=config.SPREAD_LENGTH_SCALE_KM,
            wind_aniso=config.SPREAD_WIND_ANISO,
            southness_beta=config.SPREAD_SOUTHNESS_BETA,
            seed=seed + ri,
        )
        p_combined += gate[:, ri] * p_r
        logger.info("체제 %s 노출: mean=%.4f p99=%.4f (rust=%s)",
                    regime, float(p_r.mean()), float(np.quantile(p_r, 0.99)),
                    exposure.has_rust())
    p_combined = np.clip(p_combined, 0.0, 1.0)
    logger.info("결합 P(노출): mean=%.4f p99=%.4f frac>0=%.4f",
                float(p_combined.mean()), float(np.quantile(p_combined, 0.99)),
                float((p_combined > 0).mean()))
    return p_combined
