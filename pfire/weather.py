"""기상 위험 W(grid, t) 성분.

설계 근거: W 는 "그날 격자 기상" 의 위험.

경로(우선순위 = Phase-4 결합 W):
  1. blended_weather()    — **Phase-4 본체(기본 W)**. 전주고유 시즌 극값(season_weather:
     fwi_q90·yanggan — 2019 고성 극값 포착)과 일별 ISI/FFMC 동역학(daily_isi_weather:
     top-10% recall 이득)을 [0,1] 선형 블렌드(가중 config.W_BLEND_SEASON_WEIGHT).
     일별만으로는 전주고유 극값을 희석해 고성 sanity 가 퇴행(0.698→0.230)하므로,
     시즌 극값을 함께 보존해 고성 순위를 회복하면서 recall 을 유지·개선한다.
  2. daily_isi_weather()  — 일별 ISI(초기확산 준비)·FFMC(발화준비) 동역학 성분(블렌드
     구성요소). 시즌 평균 대신 일별 변동 반영(문헌·EDA: claudedocs/
     research_방향검토_계층vsFiLM_20260619.md). 가중·임계는 config 의 W_* 섹션.
  3. season_weather()     — 전주별 사전계산 시즌통계(fwi_q90 등) 정규화(블렌드 구성요소).
  4. daily_high_danger_days() — (폴백) 전주→최근접 관측소 일별 FWI 고위험일 빈도.

모든 경로 산출은 [0,1] W(p) 1벡터 → hazard.season_risk 단일-W 경로(R=I×S×W).
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


def _per_station_isi_ffmc(
    station_daily: pl.DataFrame,
    years: tuple[int, ...] | None = None,
) -> pl.DataFrame:
    """관측소별 일별 ISI/FFMC 동역학 → 산불조심기간 시즌 집계.

    각 관측소(stn)에 대해 산불조심기간(config.SEASON_MONTHS) 일별 ISI·FFMC 분포에서
    확산준비(ISI)·발화준비(FFMC) 시즌 통계를 만든다(연 평균이 아니라 시즌 전체 일
    풀에서 집계 — 관측소별 fire-weather 기후값).

    집계 항목(관측소별):
      - isi_hi_frac  : 고-ISI(>= config.DAILY_ISI_HIGH_THRESHOLD) 일 비율 (확산준비 빈도)
      - isi_mean     : 평균 ISI (기저 확산성)
      - isi_q90      : ISI 90 분위 (극단 확산일 강도)
      - ffmc_hi_frac : 고-FFMC(>= config.DAILY_FFMC_HIGH_THRESHOLD) 일 비율 (발화준비 빈도)

    Parameters
    ----------
    station_daily : polars.DataFrame
        stn, month, year, isi, ffmc 포함(fwi_station_daily).
    years : tuple[int] or None
        집계 연도 제한(None=전체).

    Returns
    -------
    polars.DataFrame
        컬럼: stn, isi_hi_frac, isi_mean, isi_q90, ffmc_hi_frac.

    Raises
    ------
    ValueError
        필수 컬럼(isi/ffmc)이 없을 때.
    """
    for c in ("stn", "month", "isi", "ffmc"):
        if c not in station_daily.columns:
            raise ValueError(f"station_daily 에 컬럼 없음: {c}")

    sd = station_daily.filter(pl.col("month").is_in(list(config.SEASON_MONTHS)))
    if years is not None:
        sd = sd.filter(pl.col("year").is_in(list(years)))

    agg = (
        sd.group_by("stn")
        .agg(
            (pl.col("isi") >= config.DAILY_ISI_HIGH_THRESHOLD)
            .mean().alias("isi_hi_frac"),
            pl.col("isi").mean().alias("isi_mean"),
            pl.col("isi").quantile(0.90, interpolation="linear").alias("isi_q90"),
            (pl.col("ffmc") >= config.DAILY_FFMC_HIGH_THRESHOLD)
            .mean().alias("ffmc_hi_frac"),
        )
        .sort("stn")
    )
    return agg


def daily_isi_weather(
    master: pl.DataFrame,
    station_daily: pl.DataFrame,
    stations: pl.DataFrame,
    years: tuple[int, ...] | None = None,
    nn_station_col: str = "nn_station_km",
) -> np.ndarray:
    """Phase-4 ① 일별 fire-weather(ISI/FFMC) 동역학 W(p) ∈ [0,1].

    관측소별 산불조심기간 일별 ISI(초기확산 준비)·FFMC(발화 준비) 동역학을 시즌
    집계(_per_station_isi_ffmc)하고, 각 전주를 최근접 관측소(평면근사 km)에 매핑해
    부여한다. 여기에 전주고유 양간(강풍·방향 prior) 성분을 결합한다.

        ISI성분  = w_freq·고ISI일수비율 + w_mean·평균ISI + w_q90·ISI q90   (관측소→전주)
        FFMC성분 = 고FFMC일수비율                                          (관측소→전주)
        양간성분 = master.yanggan_days                                     (전주고유)
        W = w_isi·ISI성분 + w_ffmc·FFMC성분 + w_yanggan·양간성분

    각 성분은 전주 모집단에서 [0,1] min-max 정규화 후 결합(가중·임계는 config.W_*).
    시즌 평균 한 값(fwi_q90)보다 일별 변동을 담아 예측 skill 우위(문헌·EDA 근거).
    가중은 공간블록CV 발화점 recall 로 튜닝됨: 발화점 앵커가 인간발화 위주라
    FFMC(발화준비) 가 본체, ISI(확산준비) 는 소수 유지(상세 근거는 config.W_* 주석).

    Parameters
    ----------
    master : polars.DataFrame
        lon, lat, yanggan_days 포함. nn_station_col 이 있으면 사전계산 최근접거리는
        진단 로깅에만 쓰고 매핑은 좌표 최근접으로 직접 계산(좌표 단일 기준 유지).
    station_daily : polars.DataFrame
        stn, month, year, isi, ffmc 포함(fwi_station_daily).
    stations : polars.DataFrame
        stnId, lon, lat 포함(관측소 좌표).
    years : tuple[int] or None
        집계 연도 제한(None=전체).
    nn_station_col : str
        master 의 사전계산 최근접거리 컬럼명(진단 로깅용; 없으면 무시).

    Returns
    -------
    numpy.ndarray, shape (N,)
        일별-ISI/FFMC 동역학 W ∈ [0,1].

    Raises
    ------
    ValueError
        필수 컬럼 결측 또는 정규화 후 NaN 잔존.
    """
    if "yanggan_days" not in master.columns:
        raise ValueError("master 에 yanggan_days 없음(양간 성분 필요)")

    agg = _per_station_isi_ffmc(station_daily, years=years)
    stn_ids_agg = agg["stn"].to_numpy()
    isi_hi = agg["isi_hi_frac"].to_numpy().astype(np.float64)
    isi_mean = agg["isi_mean"].to_numpy().astype(np.float64)
    isi_q90 = agg["isi_q90"].to_numpy().astype(np.float64)
    ffmc_hi = agg["ffmc_hi_frac"].to_numpy().astype(np.float64)

    # 관측소 id → 행 인덱스 (전주 매핑 후 gather).
    id_to_row = {int(s): i for i, s in enumerate(stn_ids_agg)}

    stn_xy = stations.select(["lon", "lat"]).to_numpy().astype(np.float64)
    stn_ids = stations["stnId"].to_numpy()
    pole_xy = master.select(["lon", "lat"]).to_numpy().astype(np.float64)
    nearest = _nearest_station(pole_xy, stn_xy, stn_ids)

    # 집계에 없는 관측소(예: AWS 신설)는 전역 평균으로 폴백(silent 금지: 카운트 로깅).
    g_isi_hi = float(isi_hi.mean()) if isi_hi.size else 0.0
    g_isi_mean = float(isi_mean.mean()) if isi_mean.size else 0.0
    g_isi_q90 = float(isi_q90.mean()) if isi_q90.size else 0.0
    g_ffmc_hi = float(ffmc_hi.mean()) if ffmc_hi.size else 0.0

    n = pole_xy.shape[0]
    p_isi_hi = np.empty(n, dtype=np.float64)
    p_isi_mean = np.empty(n, dtype=np.float64)
    p_isi_q90 = np.empty(n, dtype=np.float64)
    p_ffmc_hi = np.empty(n, dtype=np.float64)
    n_fallback = 0
    for i, s in enumerate(nearest):
        row = id_to_row.get(int(s))
        if row is None:
            n_fallback += 1
            p_isi_hi[i], p_isi_mean[i] = g_isi_hi, g_isi_mean
            p_isi_q90[i], p_ffmc_hi[i] = g_isi_q90, g_ffmc_hi
        else:
            p_isi_hi[i], p_isi_mean[i] = isi_hi[row], isi_mean[row]
            p_isi_q90[i], p_ffmc_hi[i] = isi_q90[row], ffmc_hi[row]
    if n_fallback:
        logger.info("daily-ISI W: 집계 없는 관측소 매핑 전주 %d → 전역평균 폴백",
                    n_fallback)

    # 성분 정규화(전주 모집단 기준 [0,1]) 후 가중 결합.
    isi_component = (
        config.W_ISI_FREQ_WEIGHT * _minmax01(p_isi_hi)
        + config.W_ISI_MEAN_WEIGHT * _minmax01(p_isi_mean)
        + config.W_ISI_Q90_WEIGHT * _minmax01(p_isi_q90)
    )
    ffmc_component = _minmax01(p_ffmc_hi)
    yanggan = master["yanggan_days"].to_numpy().astype(np.float64)
    yanggan_component = _minmax01(yanggan)

    W = (
        config.W_ISI_COMPONENT_WEIGHT * isi_component
        + config.W_FFMC_COMPONENT_WEIGHT * ffmc_component
        + config.W_YANGGAN_COMPONENT_WEIGHT * yanggan_component
    )
    W = np.clip(W, 0.0, 1.0)
    if np.isnan(W).any():
        raise ValueError("daily-ISI W 에 NaN 잔존(정규화/매핑 점검)")

    if nn_station_col in master.columns:
        nn = master[nn_station_col].to_numpy().astype(np.float64)
        logger.info("daily-ISI W: nn_station_km p50=%.2f p90=%.2f",
                    float(np.nanmedian(nn)), float(np.nanquantile(nn, 0.90)))
    logger.info(
        "daily-ISI W(p): stations_agg=%d isi_comp(mean=%.3f) ffmc_comp(mean=%.3f) "
        "yang_comp(mean=%.3f) | W mean=%.4f p50=%.4f p99=%.4f",
        len(id_to_row), float(isi_component.mean()), float(ffmc_component.mean()),
        float(yanggan_component.mean()), float(W.mean()),
        float(np.median(W)), float(np.quantile(W, 0.99)),
    )
    return W


def blended_weather(
    master: pl.DataFrame,
    station_daily: pl.DataFrame,
    stations: pl.DataFrame,
    years: tuple[int, ...] | None = None,
    season_weight: float = config.W_BLEND_SEASON_WEIGHT,
    source: str = "features",
    nn_station_col: str = "nn_station_km",
) -> np.ndarray:
    """Phase-4 **기본 W**: 전주고유 시즌 극값 × 일별 ISI/FFMC 동역학 블렌드.

    두 성분을 [0,1] 공간에서 선형 결합한다(단순 대체 아님):

        W = season_weight · W_season + (1 − season_weight) · W_daily

      - W_season = season_weather(master, source)
          전주고유 시즌 극값(fwi_q90·고위험일·yanggan_days)을 [0,1] 결합. 2019 고성
          인근 같은 **전주 고유 극값**(fwi_q90 pctile≈0.97·yanggan pctile≈0.99)을
          포착해 sanity 위험순위를 지킨다.
      - W_daily = daily_isi_weather(master, station_daily, stations, years)
          관측소별 일별 ISI(확산준비)/FFMC(발화준비) 동역학 시즌집계 + 양간. 공간블록
          CV 발화점 recall(특히 top-10%) 이득의 출처.

    설계 근거(config.W_BLEND_SEASON_WEIGHT 주석): 일별 W 만 쓰면 전주를 최근접 관측소
    시즌집계로 매핑해 전주고유 극값을 희석 → 고성 sanity 가 퇴행(0.698→0.230)한다.
    곱셈 결합 R=I·S·W 에서 고성은 S 가 매우 낮아 W 크기로만 상위 위험을 유지하므로,
    블렌드 가중을 시즌 쪽(기본 0.82)으로 둬 W 크기 극값을 지키면 고성 순위가 회복되며,
    남은 일별 성분이 recall 이득을 유지·개선한다(공간CV recall@top-k 가 시즌W 이상).

    Parameters
    ----------
    master : polars.DataFrame
        lon, lat, yanggan_days 및 season_weather 가 요구하는 시즌통계 컬럼 포함.
    station_daily : polars.DataFrame
        stn, month, year, isi, ffmc 포함(fwi_station_daily).
    stations : polars.DataFrame
        stnId, lon, lat 포함(관측소 좌표).
    years : tuple[int] or None
        일별 성분 집계 연도 제한(None=전체).
    season_weight : float
        시즌 극값 성분 가중 ∈ [0,1]. 나머지(1−season_weight)는 일별 성분.
        기본 config.W_BLEND_SEASON_WEIGHT.
    source : {"features", "obs"}
        시즌 성분 출처(season_weather).
    nn_station_col : str
        일별 성분의 진단 로깅용 사전계산 최근접거리 컬럼명.

    Returns
    -------
    numpy.ndarray, shape (N,)
        결합 기상 위험 W ∈ [0,1].

    Raises
    ------
    ValueError
        season_weight 가 [0,1] 밖이거나 결합 후 NaN 잔존.
    """
    if not 0.0 <= season_weight <= 1.0:
        raise ValueError(f"season_weight 는 [0,1]: {season_weight!r}")

    W_season = season_weather(master, source=source)
    W_daily = daily_isi_weather(
        master, station_daily, stations, years=years, nn_station_col=nn_station_col
    )
    W = season_weight * W_season + (1.0 - season_weight) * W_daily
    W = np.clip(W, 0.0, 1.0)
    if np.isnan(W).any():
        raise ValueError("blended W 에 NaN 잔존(성분/가중 점검)")
    logger.info(
        "blended W(p): season_weight=%.2f | season(mean=%.4f) daily(mean=%.4f) "
        "→ W mean=%.4f p50=%.4f p99=%.4f",
        season_weight, float(W_season.mean()), float(W_daily.mean()),
        float(W.mean()), float(np.median(W)), float(np.quantile(W, 0.99)),
    )
    return W
