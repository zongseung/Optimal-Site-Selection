"""데이터 로더 — polars 기반, pole_id 기준 마스터 프레임 조인.

설계 근거: 모든 전주 parquet 은 pole_id 0..1387830 으로 정렬·정합되어 있다(검증함).
필요 컬럼만 골라 단일 마스터 프레임으로 합치고, 소수 결측(ndvi/s2 3개)은
체제·전역 통계로 메우는 순수 로더 계층. used_dataset/ 은 READ-ONLY.
"""
from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from pfire import config

logger = logging.getLogger(__name__)

# config 단일 진실의 별칭(하위호환 — 모듈 직접 참조용).
N_POLES_EXPECTED = config.N_POLES_EXPECTED


def _read(path: Path, columns: list[str] | None = None) -> pl.DataFrame:
    """parquet 한 개를 읽고 존재/스키마를 검증한다.

    Parameters
    ----------
    path : Path
        읽을 parquet 경로.
    columns : list[str] or None
        선택할 컬럼. None 이면 전체.

    Returns
    -------
    polars.DataFrame
        읽은 프레임.

    Raises
    ------
    FileNotFoundError
        파일이 없을 때.
    """
    if not path.exists():
        raise FileNotFoundError(f"필수 데이터 파일 없음: {path}")
    df = pl.read_parquet(path, columns=columns)
    logger.debug("loaded %s rows=%d cols=%s", path.name, df.height, df.columns)
    return df


def load_master() -> pl.DataFrame:
    """전주 마스터 프레임을 구성한다(pole_id 기준 left-join).

    pole_features 를 기준으로 power / static_overlay / kfs / sgg / fwi_obs 를
    pole_id 로 조인하고, 소수 결측을 채운 뒤 반환한다.

    Returns
    -------
    polars.DataFrame
        pole_id 정렬·중복없는 마스터 프레임. 행수 == 1,387,831.

    Raises
    ------
    ValueError
        행수가 기대치와 다르거나 pole_id 정렬/유일성이 깨질 때.
    """
    feats = _read(
        config.F_POLE_FEATURES,
        [
            "pole_id", "lon", "lat", "grid_id", "elevation", "slope",
            "aspect", "aspect_cos", "ndvi", "dist_to_forest", "dist_to_road",
            "fwi_q90", "fwi_high_danger_days", "yanggan_days",
            "fwi_mean", "fwi_max", "n_seasons", "s2_ndmi", "s2_nbr",
        ],
    )
    power = _read(config.F_POLE_POWER, ["pole_id", "dist_to_powerline", "dist_to_substation"])
    static = _read(
        config.F_POLE_STATIC,
        ["pole_id", "mu_slope", "mu_southness", "mu_flammability", "mu_ndvi", "S_p"],
    )
    kfs = _read(config.F_POLE_KFS, ["pole_id", "kfs_fires"])
    sgg = _read(config.F_POLE_SGG, ["pole_id", "sgg"])
    landcover = _read(config.F_POLE_LANDCOVER, ["pole_id", "lc_group"])
    fwi_obs = _read(
        config.F_POLE_FWI_OBS,
        ["pole_id", "fwi_q90_obs", "fwi_mean_obs", "fwi_max_obs", "fwi_hdd_obs",
         "nn_station_km", "extrap_flag"],
    )

    master = (
        feats
        .join(power, on="pole_id", how="left")
        .join(static, on="pole_id", how="left")
        .join(kfs, on="pole_id", how="left")
        .join(sgg, on="pole_id", how="left")
        .join(landcover, on="pole_id", how="left")
        .join(fwi_obs, on="pole_id", how="left")
        .sort("pole_id")
    )

    master = _fill_missing(master)
    _validate_master(master)
    logger.info("master frame ready: rows=%d cols=%d", master.height, master.width)
    return master


def _fill_missing(df: pl.DataFrame) -> pl.DataFrame:
    """소수 결측 컬럼을 전역 중앙값으로 채운다.

    ndvi/s2_ndmi/s2_nbr 에 각 3개의 null 이 있다(EDA 확인). 비지도 위험
    추정에서 소수 결측은 전역 중앙값으로 대치해도 영향이 무시할 수준이다.
    silent 금지: 채운 개수를 로깅한다.
    """
    fill_cols = ["ndvi", "s2_ndmi", "s2_nbr", "fwi_q90_obs", "fwi_mean_obs",
                 "fwi_max_obs", "fwi_hdd_obs", "nn_station_km", "mu_southness",
                 "mu_flammability", "mu_ndvi", "S_p", "kfs_fires"]
    out = df
    for c in fill_cols:
        if c not in out.columns:
            continue
        n_null = out[c].null_count()
        if n_null > 0:
            med = out[c].median()
            out = out.with_columns(pl.col(c).fill_null(med))
            logger.info("filled %d null in '%s' with median=%.4g", n_null, c, med)
    # kfs_fires 의미상 0 결측은 0 으로(화재 없음). median 대신 0 우선.
    if "kfs_fires" in out.columns:
        out = out.with_columns(pl.col("kfs_fires").fill_null(0.0))
    return out


def _validate_master(df: pl.DataFrame) -> None:
    """마스터 프레임 무결성 검증(행수·정렬·유일성)."""
    if df.height != N_POLES_EXPECTED:
        raise ValueError(f"마스터 행수 {df.height} != 기대 {N_POLES_EXPECTED}")
    if not df["pole_id"].is_sorted():
        raise ValueError("pole_id 정렬 깨짐")
    if df["pole_id"].n_unique() != df.height:
        raise ValueError("pole_id 중복 존재")


def load_positives() -> pl.DataFrame:
    """발화점(safemap) 을 정제해 반환한다(검증·임계값 앵커 전용).

    occu_mt 가 더러움("03" 과 "3" 혼재)이라 정수로 정제한다. 강원 전주 좌표
    bounding box 밖 발화점은 검증 대상이 아니므로 표시(in_bbox)한다.

    Returns
    -------
    polars.DataFrame
        컬럼: lon, lat, occu_date, occu_year(int), occu_mt(int), resn, ar, sgg_cd, in_season.
    """
    df = _read(config.F_SAFEMAP_POSITIVES)
    df = df.with_columns(
        pl.col("occu_mt").str.strip_chars().cast(pl.Int32, strict=False).alias("occu_mt"),
        pl.col("occu_year").str.strip_chars().cast(pl.Int32, strict=False).alias("occu_year"),
    )
    n_bad = df["occu_mt"].null_count()
    if n_bad:
        logger.warning("occu_mt 정수변환 실패 %d 건(드롭하지 않고 보존)", n_bad)
    df = df.with_columns(
        pl.col("occu_mt").is_in(list(config.SEASON_MONTHS)).alias("in_season")
    )
    keep = ["lon", "lat", "occu_date", "occu_year", "occu_mt", "resn", "ar",
            "sgg_cd", "in_season"]
    df = df.select([c for c in keep if c in df.columns])
    logger.info("positives loaded: %d (in-season %d)",
                df.height, int(df["in_season"].sum()))
    return df


def load_stations() -> pl.DataFrame:
    """관측소 좌표/표고 CSV 로드.

    Returns
    -------
    polars.DataFrame
        컬럼: stnId, name, type, lon, lat, elev_dem.
    """
    if not config.F_STATIONS.exists():
        raise FileNotFoundError(f"관측소 좌표 파일 없음: {config.F_STATIONS}")
    return pl.read_csv(config.F_STATIONS)


def load_station_daily() -> pl.DataFrame:
    """관측소 일별 FWI 프레임 로드(일별 엔진용).

    Returns
    -------
    polars.DataFrame
        컬럼: stn, date, year, month, ws, isi, fwi, ...
    """
    return _read(config.F_FWI_STATION_DAILY)


def load_aws_daily() -> pl.DataFrame:
    """관측소 일별 AWS 관측(풍향 wd·풍속 ws) 로드 — exposure 풍향 표집용.

    wd 결측 sentinel(-9.9) 은 로더에서 거르지 않고 보존한다(소비 측에서
    wd>=0 필터). obs_date 에서 month 를 파생해 산불조심기간 필터를 돕는다.

    Returns
    -------
    polars.DataFrame
        컬럼: obs_date, stn, name, wd, ws, ... (+ month 파생).
    """
    df = _read(config.F_AWS_OBS_DAILY)
    if "month" not in df.columns and "obs_date" in df.columns:
        df = df.with_columns(pl.col("obs_date").dt.month().alias("month"))
    return df
