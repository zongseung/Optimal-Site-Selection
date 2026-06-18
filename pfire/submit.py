"""제출 CSV 작성 — 필수 스키마 + 해석용 추가 컬럼.

설계 근거: 채점은 pole_id·decision(0/1) 의 F1. 정성평가(활용성)를 위해
risk_score·regime·p_exposure·불확실성(unc_lo/unc_hi) 을 추가 컬럼으로 동봉한다.
제출 무결성(행수 1,387,831·decision 0/1만·pole_id 정렬)을 강제 검증한다.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import polars as pl

from pfire import config

logger = logging.getLogger(__name__)

# config 단일 진실의 별칭(하위호환 — 모듈/테스트 직접 참조용).
N_POLES_EXPECTED = config.N_POLES_EXPECTED
REQUIRED_COLS = ("pole_id", "lon", "lat", "decision")


def build_submission(
    pole_id: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    decision: np.ndarray,
    risk_score: np.ndarray,
    regime: np.ndarray | None = None,
    p_exposure: np.ndarray | None = None,
    unc_lo: np.ndarray | None = None,
    unc_hi: np.ndarray | None = None,
) -> pl.DataFrame:
    """제출 프레임 구성 + 스키마 검증.

    Parameters
    ----------
    pole_id, lon, lat : numpy.ndarray
        전주 식별/좌표.
    decision : numpy.ndarray
        0/1 결정.
    risk_score : numpy.ndarray
        위험 점수(보정확률 권장).
    regime : numpy.ndarray or None
        체제 라벨(문자열).
    p_exposure : numpy.ndarray or None
        풍하 노출확률(있으면).
    unc_lo, unc_hi : numpy.ndarray or None
        불확실성 하/상한.

    Returns
    -------
    polars.DataFrame
        제출 프레임.

    Raises
    ------
    ValueError
        스키마/행수/결정값 위반 시.
    """
    n = pole_id.shape[0]
    data: dict[str, object] = {
        "pole_id": np.asarray(pole_id, dtype=np.int64),
        "lon": np.asarray(lon, dtype=np.float64),
        "lat": np.asarray(lat, dtype=np.float64),
        "decision": np.asarray(decision, dtype=np.int8),
        "risk_score": np.asarray(risk_score, dtype=np.float64),
    }
    if regime is not None:
        data["regime"] = np.asarray(regime)
    if p_exposure is not None:
        data["p_exposure"] = np.asarray(p_exposure, dtype=np.float64)
    if unc_lo is not None:
        data["unc_lo"] = np.asarray(unc_lo, dtype=np.float64)
    if unc_hi is not None:
        data["unc_hi"] = np.asarray(unc_hi, dtype=np.float64)

    for k, v in data.items():
        if np.asarray(v).shape[0] != n:
            raise ValueError(f"컬럼 '{k}' 길이 {np.asarray(v).shape[0]} != {n}")

    df = pl.DataFrame(data)
    validate_submission(df)
    return df


def validate_submission(df: pl.DataFrame) -> None:
    """제출 무결성 강제 검증(silent 금지)."""
    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"제출 필수 컬럼 누락: {c}")
    if df.height != N_POLES_EXPECTED:
        raise ValueError(f"제출 행수 {df.height} != {N_POLES_EXPECTED}")
    dec = set(df["decision"].unique().to_list())
    if not dec.issubset({0, 1}):
        raise ValueError(f"decision 에 0/1 외 값: {dec}")
    if df["pole_id"].n_unique() != df.height:
        raise ValueError("pole_id 중복")
    if not df["pole_id"].is_sorted():
        raise ValueError("pole_id 정렬 깨짐")
    logger.info("제출 검증 통과: 행=%d 양성=%d (%.3f%%)",
                df.height, int(df["decision"].sum()),
                100.0 * df["decision"].mean())


def write_submission(df: pl.DataFrame, name: str = "submission.csv") -> Path:
    """제출 CSV 를 outputs/submissions/ 에 기록.

    Parameters
    ----------
    df : polars.DataFrame
        build_submission 출력.
    name : str
        파일명.

    Returns
    -------
    Path
        기록된 경로.
    """
    validate_submission(df)
    config.SUBMISSIONS.mkdir(parents=True, exist_ok=True)
    path = config.SUBMISSIONS / name
    df.write_csv(path)
    logger.info("제출 CSV 기록: %s (%d행)", path, df.height)
    return path
