"""제출 CSV 작성 — 필수 스키마 + 해석용 추가 컬럼.

설계 근거: 채점은 pole_id·decision(0/1) 의 F1. 정성평가(활용성)를 위해
risk_score·regime·p_exposure·사후 credible(risk_lo/risk_hi)·운영플래그(ops_priority)
를 추가 컬럼으로 동봉한다. 제출 무결성(행수 1,387,831·decision 0/1만·pole_id 정렬)을
강제 검증한다.

불확실성은 베이지안 사후 MC credible(risk_lo/risk_hi)을 단일 진실로 쓴다. 구
unc_lo/unc_hi(extrap·관측소거리 휴리스틱 밴드)는 risk_score 스케일과 안 맞아 unc_lo 가
0 으로 퇴화해 제거했다(파라미터는 하위호환으로 남기되 러너는 더 이상 넘기지 않는다).

Phase-5 추가 컬럼:
  - p_exposure(항목1): 풍하 노출확률 복구(README 스펙 일치). 체제별 재생성 때 누락분.
  - risk_lo / risk_hi(credible): 사후예측 90% 신뢰구간(posterior.propagate_risk_posterior).
  - ops_priority(항목2): 영동 풍하 고노출 전주의 운영 우선 플래그(조기경보 활용; F1
    결정과 별개의 운영 산출).
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
    risk_lo: np.ndarray | None = None,
    risk_hi: np.ndarray | None = None,
    ops_priority: np.ndarray | None = None,
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
        풍하 노출확률(항목1 복구). 있으면 [0,1] 검증.
    risk_lo, risk_hi : numpy.ndarray or None
        사후예측 credible interval 하/상한(propagate_risk_posterior). 있으면 lo<=hi 검증.
    ops_priority : numpy.ndarray or None
        영동 풍하 고노출 운영 우선 플래그(항목2). 0/1.
    unc_lo, unc_hi : numpy.ndarray or None
        (deprecated) 구 해석용 불확실성 하/상한. 퇴화로 제거됨 — 넘기지 말 것.
        파라미터는 하위호환을 위해서만 남긴다(유효 밴드는 risk_lo/risk_hi).

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
    if risk_lo is not None:
        data["risk_lo"] = np.asarray(risk_lo, dtype=np.float64)
    if risk_hi is not None:
        data["risk_hi"] = np.asarray(risk_hi, dtype=np.float64)
    if ops_priority is not None:
        data["ops_priority"] = np.asarray(ops_priority, dtype=np.int8)
    if unc_lo is not None:
        data["unc_lo"] = np.asarray(unc_lo, dtype=np.float64)
    if unc_hi is not None:
        data["unc_hi"] = np.asarray(unc_hi, dtype=np.float64)

    for k, v in data.items():
        if np.asarray(v).shape[0] != n:
            raise ValueError(f"컬럼 '{k}' 길이 {np.asarray(v).shape[0]} != {n}")

    # 새 컬럼 도메인 검증(silent 금지).
    if p_exposure is not None:
        pe = data["p_exposure"]
        if np.isnan(pe).any() or pe.min() < -1e-9 or pe.max() > 1 + 1e-9:
            raise ValueError(f"p_exposure [0,1] 밖: [{pe.min():.4g},{pe.max():.4g}]")
    if risk_lo is not None and risk_hi is not None:
        if (data["risk_hi"] < data["risk_lo"] - 1e-9).any():
            raise ValueError("risk_hi < risk_lo 인 행 존재(credible interval 깨짐)")
    if ops_priority is not None:
        op = set(np.unique(data["ops_priority"]).tolist())
        if not op.issubset({0, 1}):
            raise ValueError(f"ops_priority 0/1 외 값: {op}")

    df = pl.DataFrame(data)
    validate_submission(df)
    return df


def ops_priority_flag(
    regime_labels: np.ndarray,
    p_exposure: np.ndarray | None,
    p_exposure_quantile: float = 0.90,
    yeongdong_regime: str = config.REGIME_YEONGDONG,
) -> np.ndarray:
    """영동 풍하 고노출 운영 우선 플래그(항목2 — 조기경보 활용).

    영동(양간지풍 wind-driven) 체제이면서 풍하 노출확률 p_exposure 가 영동 내부
    상위분위(기본 q90) 이상인 전주를 1 로 둔다. F1 채점 decision 과 **별개의 운영
    산출**로, "발화 시 풍하로 빠르게 번지는 고노출 전주"를 조기경보·우선순시 대상으로
    표시한다(2019 고성형 시나리오 대비).

    Parameters
    ----------
    regime_labels : numpy.ndarray of str, shape (N,)
        전주별 우세 체제.
    p_exposure : numpy.ndarray or None, shape (N,)
        풍하 노출확률. None 이면 전부 0(노출 미산출 시 운영플래그 없음).
    p_exposure_quantile : float
        영동 내부 노출 상위분위 임계(기본 0.90).
    yeongdong_regime : str
        영동 체제명.

    Returns
    -------
    numpy.ndarray, shape (N,)
        0/1 (int8). 영동·고노출 전주만 1.
    """
    n = regime_labels.shape[0]
    flag = np.zeros(n, dtype=np.int8)
    if p_exposure is None:
        logger.info("ops_priority: p_exposure 없음 → 전부 0(운영플래그 미산출)")
        return flag
    yd = regime_labels == yeongdong_regime
    if not yd.any():
        return flag
    thr = float(np.quantile(p_exposure[yd], p_exposure_quantile))
    flag[yd & (p_exposure >= thr)] = 1
    logger.info("ops_priority(영동 풍하 고노출 q%.0f): %d개 (영동 %d 중)",
                100 * p_exposure_quantile, int(flag.sum()), int(yd.sum()))
    return flag


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
