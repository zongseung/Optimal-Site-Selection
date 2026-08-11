"""계층 partial pooling(경험적 베이즈) 지역 보정 — Phase4 ②.

설계 근거(claudedocs/research_방향검토_계층vsFiLM_20260619.md ②):
    "데이터에 그룹이 있으면 계층적 베이즈가 가장 방어 가능한 베이스라인". 발화점이
    ~700개로 희소하고 시군 16개 중 일부(화천·인제·고성·태백)는 발화 0~소수다. 이런
    소그룹에서 관측율(전주당 발화점 수)을 그대로 쓰면 극단(0 또는 과대) 추정으로
    과적합한다. 경험적 베이즈(EB) 축소(shrinkage)는 질병지도·소지역추정(small-area
    estimation)의 표준으로, 사건수 적은 지역의 추정을 분산 기반으로 부모 단계 평균에
    끌어당겨(borrow strength) 안정화한다(zero-event 지역은 부모율로 수렴).

계층(부모→자식, 전역 포함 4단):
    전역(1) → 체제(3, config.REGIMES) → 시군(16, config.SGG_TO_REGIME 키) → 격자(106, grid_id).
    각 단계의 EB 사후평균을 자식 단계의 부모 사전평균으로 연쇄 사용한다(top-down).
    즉 격자율은 자기 시군 EB율로, 시군율은 자기 체제 EB율로, 체제율은 전역율로 축소된다.

방법(Poisson–Gamma 경험적 베이즈 — 소지역추정 표준):
    지역 g 의 노출 n_g(전주 수)와 사건 y_g(발화점 수)에서 관측율 λ̂_g = y_g / n_g.
    부모 단계의 율 m(사전평균)과 자식들의 율 분산 v 로 Gamma(α, β) 사전을 적률추정
    (α = m²/v, β = m/v)한 뒤, 사후평균
        λ̃_g = (y_g + α) / (n_g + β)
             = w_g · λ̂_g + (1 − w_g) · m,   w_g = n_g / (n_g + β)
    를 쓴다. 표본(전주수) 큰 지역은 관측율에 가깝고(w_g→1), 작은/0발화 지역은 부모율
    m 으로 축소된다(w_g→0). 이것이 partial pooling 의 닫힌형이다.

출력(배율):
    전주별 격자 EB율을 전역 EB율로 나눠 **상대 배율**(전역평균≈1)로 정규화한다. 곱셈
    파이프라인(hazard.season_risk 의 I·S·W)에 자연스러운 단위가 되도록 1 중심이며,
    normalize="unit" 옵션으로 [0,1] 백분위 정규화도 제공한다.

누수 방지(중요):
    `fires` 를 인자로 받아 **부분집합(train fold)만으로** 율을 추정할 수 있다. 공간
    블록 CV 에서 각 fold 마다 train fold 발화점으로만 regional_multiplier 를 재계산하면
    test fold 발화점 정보가 배율에 새지 않는다. 발화점은 per-pole 학습 라벨이 아니라
    **지역 집계 앵커**(전주당 발화 강도 추정용)로만 쓴다 — 어떤 전주도 자신이 양성이라는
    라벨을 받지 않는다.
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

from pfire import config, geo

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# EB 하이퍼파라미터 (모듈 상수 — 근거 주석 동반)
# ──────────────────────────────────────────────────────────────────────────
# 부모율이 0(또는 극소)이면 Gamma 적률추정이 퇴화(α,β→0/∞)한다. 사전평균·분산의
# 하한 floor 로 수치안정을 둔다. 값은 전형적 전주당 발화 강도(~700발화/1.39M전주 ≈
# 5e-4) 대비 충분히 작은 정칙화 수준.
_RATE_FLOOR: float = 1e-9          # 율(λ) 하한 — log/나눗셈 0 방지, 부모율 0 정칙화
_VAR_FLOOR_FRAC: float = 1e-3      # 사전분산 v 하한 = max(v, (m·_VAR_FLOOR_FRAC)²) 근사용

# 적률추정 분산이 음수/0 으로 퇴화할 때의 폴백: 사전을 약하게(β 작게) 둬 거의 unpooled.
# 반대로 자식 표본이 1개뿐이라 분산을 못 구하면 부모로 강하게 축소(β 크게)한다.
# β(=유효 사전 표본수, "prior pseudo-count of poles")의 안전 범위.
_BETA_MIN: float = 1.0             # 최소 사전 의사표본(전주수) — 최소한의 정칙화
_BETA_MAX: float = 1e9             # 최대(분산 추정 불가 시 부모로 강축소)

# 발화점→지역 앵커 매핑 반경(km). 이보다 먼 발화점은 그 지역의 사건으로 세지 않는다.
# config.SPATIAL_CV_BLOCK_KM 와 무관한 "발화-전주 근접" 기준(calibrate.assign_poles_to_fires
# 의 1km 검증 앵커와 일관). None 이면 거리 무관 최근접 전주의 지역에 귀속.
_ANCHOR_RADIUS_KM: float = 1.0


def _eb_poisson_gamma(
    y: np.ndarray, n: np.ndarray, parent_rate: np.ndarray
) -> np.ndarray:
    """Poisson–Gamma 경험적 베이즈 사후평균 율(λ̃) — 부모율로 축소.

    각 자식 g 에 대해 부모율 m_g(=parent_rate[g]) 를 사전평균으로 두고, 같은 부모를
    공유하는 자식들의 (가중)율 분산을 적률추정해 Gamma(α, β) 사전을 만든 뒤 사후평균
    λ̃_g = (y_g + α_g)/(n_g + β_g) 를 반환한다.

    Parameters
    ----------
    y : numpy.ndarray, shape (G,)
        자식별 사건수(발화점 수).
    n : numpy.ndarray, shape (G,)
        자식별 노출(전주 수). 0 이면 율 정의 안 됨 → 부모율 반환.
    parent_rate : numpy.ndarray, shape (G,)
        자식별 부모 단계 사전평균 율 m_g(이미 EB 추정된 상위단계 율).

    Returns
    -------
    numpy.ndarray, shape (G,)
        자식별 EB 사후평균 율 λ̃_g (>= 0).

    Notes
    -----
    적률추정: 부모율 m 과 자식 관측율 λ̂=y/n 의 노출가중 분산 v 에서
    α = m²/v, β = m/v. v 가 퇴화(<=0)하면 β 를 _BETA_MAX 로(강축소), m 이 0 에
    가까우면 _RATE_FLOOR 로 정칙화한다. β 는 [_BETA_MIN, _BETA_MAX] 로 클립.
    """
    y = np.asarray(y, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64)
    m = np.maximum(np.asarray(parent_rate, dtype=np.float64), _RATE_FLOOR)

    lam_hat = np.where(n > 0, y / np.maximum(n, _RATE_FLOOR), m)

    out = np.empty_like(lam_hat)
    # 같은 부모(=동일 m)끼리 묶어 분산 적률추정. 부모율은 float 라 round 키로 그룹.
    # (상위단계 EB 율은 부모별로 상수이므로 동일 부모 자식들은 같은 m 값을 공유한다.)
    keys = np.round(m / _RATE_FLOOR).astype(np.int64)
    for key in np.unique(keys):
        sel = keys == key
        m_g = float(m[sel][0])
        n_g = n[sel]
        lam_g = lam_hat[sel]
        ntot = n_g.sum()
        if ntot <= 0:
            out[sel] = m_g
            continue
        # 노출가중 분산(부모율 m_g 중심). 자식 1개뿐이면 0 → 부모로 강축소.
        v = float(np.sum(n_g * (lam_g - m_g) ** 2) / ntot)
        v_floor = (m_g * _VAR_FLOOR_FRAC) ** 2
        v = max(v, v_floor)
        if v <= 0.0:
            beta = _BETA_MAX
        else:
            beta = m_g / v
        beta = float(np.clip(beta, _BETA_MIN, _BETA_MAX))
        alpha = beta * m_g  # = m²/v (α = m·β)
        out[sel] = (y[sel] + alpha) / (n_g + beta)
    return out


def _assign_fires_to_poles(
    pole_xy: np.ndarray, fire_xy: np.ndarray, radius_km: float
) -> np.ndarray:
    """각 발화점을 최근접 전주에 귀속(지역 앵커). 반경 밖이면 -1.

    calibrate.assign_poles_to_fires 와 동일 기하(geo 평면근사). 여기서는 hierarchy
    독립성을 위해 자체 구현하되 식은 동일하다.

    Parameters
    ----------
    pole_xy : numpy.ndarray, shape (N, 2)
        전주 (lon, lat).
    fire_xy : numpy.ndarray, shape (M, 2)
        발화점 (lon, lat).
    radius_km : float
        귀속 허용 반경. None/<=0 이면 거리무관 최근접.

    Returns
    -------
    numpy.ndarray, shape (M,)
        각 발화점의 최근접 전주 인덱스(반경 밖이면 -1).
    """
    pole_km = geo.lonlat_to_km(pole_xy)
    px, py = pole_km[:, 0], pole_km[:, 1]
    out = np.full(fire_xy.shape[0], -1, dtype=np.int64)
    use_radius = radius_km is not None and radius_km > 0
    for i in range(fire_xy.shape[0]):
        fx, fy = geo.point_to_km(fire_xy[i, 0], fire_xy[i, 1])
        d2 = (px - fx) ** 2 + (py - fy) ** 2
        j = int(d2.argmin())
        if (not use_radius) or np.sqrt(d2[j]) <= radius_km:
            out[i] = j
    return out


def _level_counts(
    label_of_pole: np.ndarray, fire_label: np.ndarray
) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    """한 계층 단계의 지역별 노출(전주수)·사건수 집계.

    Parameters
    ----------
    label_of_pole : numpy.ndarray of object/str, shape (N,)
        전주별 지역 라벨(체제명/시군명/격자id).
    fire_label : numpy.ndarray of object/str, shape (M_mapped,)
        귀속된 발화점들의 지역 라벨.

    Returns
    -------
    uniq : numpy.ndarray
        지역 라벨 유일값.
    n_by : dict[str, float]
        지역 → 전주 수.
    y_by : dict[str, float]
        지역 → 발화점 수(없으면 0).
    """
    uniq, cnt = np.unique(label_of_pole, return_counts=True)
    n_by = {str(u): float(c) for u, c in zip(uniq, cnt)}
    y_by = {str(u): 0.0 for u in uniq}
    if fire_label.size:
        fu, fc = np.unique(fire_label, return_counts=True)
        for u, c in zip(fu, fc):
            if str(u) in y_by:
                y_by[str(u)] += float(c)
    return uniq, n_by, y_by


def regional_multiplier(
    poles: pl.DataFrame,
    fires: pl.DataFrame,
    *,
    sgg_col: str = "sgg",
    grid_col: str = "grid_id",
    lon_col: str = "lon",
    lat_col: str = "lat",
    anchor_radius_km: float = _ANCHOR_RADIUS_KM,
    normalize: str = "mean",
    seed: int = config.SEED,
) -> np.ndarray:
    """전주별 지역 위험 배율을 계층 경험적 베이즈(EB) 축소로 추정한다.

    체제(3)→시군(16)→격자(106) (전역 포함 4단) 계층에서 전주당 발화 강도(율)를
    Poisson–Gamma EB 로 부모 단계 값에 축소(partial pooling)해 추정하고, 격자 EB율을
    전역 EB율로 정규화한 **상대 배율**(전역평균≈1)을 전주별로 broadcast 한다. 발화
    0~소수 지역(화천·인제·고성·태백 등)은 부모(시군→체제→전역)율로 수렴한다.

    **누수 방지(필수)**: `fires` 를 인자로 받으므로, 공간 블록 CV 에서 각 fold 의
    train fold 발화점만 담은 `fires` 를 넘기면 그 fold 기준의 배율이 산출된다(test
    fold 정보 미사용). 발화점은 per-pole 학습 라벨이 아니라 지역 집계 앵커로만 쓴다 —
    어떤 전주도 자신이 양성이라는 라벨을 받지 않으며, 배율은 그 전주가 속한 지역의
    축소 추정 율에서만 나온다.

    Parameters
    ----------
    poles : polars.DataFrame
        마스터(또는 부분집합). 최소 [lon_col, lat_col, sgg_col, grid_col] 포함.
        행 순서가 곧 출력 순서다(pole_id 정렬 마스터 권장).
    fires : polars.DataFrame
        발화점(검증/지역 앵커). 최소 [lon_col, lat_col] 포함. 부분집합(train fold)
        가능. 빈 프레임이면 모든 배율이 1(부모=전역, 정보 없음)로 수렴한다.
    sgg_col : str, optional
        시군 컬럼명(기본 'sgg').
    grid_col : str, optional
        격자 컬럼명(기본 'grid_id').
    lon_col, lat_col : str, optional
        좌표 컬럼명.
    anchor_radius_km : float, optional
        발화점→전주 귀속 반경(km). <=0 이면 거리무관 최근접. 이보다 먼 발화점은
        어떤 지역 사건으로도 세지 않는다(전역 사건수에서도 제외).
    normalize : {'mean', 'unit'}, optional
        'mean'(기본): 전역 EB율로 나눈 상대 배율(전역평균≈1, 곱셈 결합용).
        'unit': 배율을 [0,1] 백분위 정규화(순위 보존, 이상치 강건).
    seed : int, optional
        결정성용 시드(현 구현은 무작위성 없음 — API 일관성·재현성 명시용).

    Returns
    -------
    numpy.ndarray, shape (N,)
        전주별 지역 배율. normalize='mean' 이면 전역 가중평균≈1(>0, 유한),
        'unit' 이면 [0,1].

    Raises
    ------
    KeyError
        필요한 컬럼이 없을 때.
    ValueError
        normalize 가 알 수 없는 값일 때.

    Notes
    -----
    체제 매핑은 config.SGG_TO_REGIME(시군→체제). 매핑에 없는 시군은 'global' 부모로
    직결(체제 단계를 건너뜀) — 미지 시군의 silent 누락 대신 전역으로 축소.
    """
    if normalize not in ("mean", "unit"):
        raise ValueError(f"normalize 는 'mean'|'unit': {normalize!r}")
    for c in (lon_col, lat_col, sgg_col, grid_col):
        if c not in poles.columns:
            raise KeyError(f"poles 에 필요한 컬럼 없음: {c}")
    for c in (lon_col, lat_col):
        if c not in fires.columns:
            raise KeyError(f"fires 에 필요한 컬럼 없음: {c}")

    _ = np.random.default_rng(seed)  # 결정성 계약(현재 무작위성 없음).

    n_poles = poles.height
    pole_xy = poles.select([lon_col, lat_col]).to_numpy().astype(np.float64)
    sgg_arr = np.asarray(poles[sgg_col].to_list(), dtype=object)
    grid_arr = np.asarray(poles[grid_col].to_list(), dtype=object)
    # 시군→체제 라벨(미지 시군은 'global' — 체제 단계 우회).
    regime_arr = np.array(
        [config.SGG_TO_REGIME.get(str(s), "global") for s in sgg_arr], dtype=object
    )

    # ── 발화점 → 전주 귀속 → 지역 라벨 ─────────────────────────────────────
    fire_xy = fires.select([lon_col, lat_col]).to_numpy().astype(np.float64)
    if fire_xy.shape[0]:
        f2p = _assign_fires_to_poles(pole_xy, fire_xy, anchor_radius_km)
        mapped = f2p[f2p >= 0]
    else:
        mapped = np.empty(0, dtype=np.int64)
    fire_regime = regime_arr[mapped] if mapped.size else np.empty(0, dtype=object)
    fire_sgg = sgg_arr[mapped] if mapped.size else np.empty(0, dtype=object)
    fire_grid = grid_arr[mapped] if mapped.size else np.empty(0, dtype=object)
    n_fires_anchored = int(mapped.size)

    # 귀속 발화점 0 → 지역 신호 없음 → 균일 배율(곱셈 항등). 모든 단계가 동일
    # 부모율로 수렴하므로 EB 를 거치지 않고 바로 단위 배율을 반환한다.
    if n_fires_anchored == 0:
        logger.info("regional_multiplier: 귀속 발화점 0 → 균일 배율(=1) 반환 "
                    "(전주=%d, normalize=%s)", n_poles, normalize)
        if normalize == "unit":
            return np.full(n_poles, 0.5, dtype=np.float64)
        return np.ones(n_poles, dtype=np.float64)

    # ── 전역(1단): 율 = 전체 귀속 발화수 / 전체 전주수 ───────────────────────
    global_rate = max(n_fires_anchored / max(n_poles, 1), _RATE_FLOOR)

    # ── 체제(2단): 부모 = 전역 ──────────────────────────────────────────────
    reg_uniq, reg_n, reg_y = _level_counts(regime_arr, fire_regime)
    reg_y_arr = np.array([reg_y[str(r)] for r in reg_uniq])
    reg_n_arr = np.array([reg_n[str(r)] for r in reg_uniq])
    reg_parent = np.full(reg_uniq.size, global_rate)
    reg_rate = _eb_poisson_gamma(reg_y_arr, reg_n_arr, reg_parent)
    reg_rate_by = {str(r): float(v) for r, v in zip(reg_uniq, reg_rate)}

    # ── 시군(3단): 부모 = 그 시군의 체제 EB율 ───────────────────────────────
    sgg_uniq, sgg_n, sgg_y = _level_counts(sgg_arr, fire_sgg)
    # 각 시군의 부모 체제율(대표 전주의 regime 으로 조회).
    sgg_to_reg = {str(s): config.SGG_TO_REGIME.get(str(s), "global") for s in sgg_uniq}
    sgg_y_arr = np.array([sgg_y[str(s)] for s in sgg_uniq])
    sgg_n_arr = np.array([sgg_n[str(s)] for s in sgg_uniq])
    sgg_parent = np.array([reg_rate_by[sgg_to_reg[str(s)]] for s in sgg_uniq])
    sgg_rate = _eb_poisson_gamma(sgg_y_arr, sgg_n_arr, sgg_parent)
    sgg_rate_by = {str(s): float(v) for s, v in zip(sgg_uniq, sgg_rate)}

    # ── 격자(4단): 부모 = 그 격자의 시군 EB율 ───────────────────────────────
    # 격자의 대표 시군 = 그 격자 전주의 최빈 시군(격자가 시군 경계를 걸칠 수 있음).
    grid_to_sgg = _dominant_parent(grid_arr, sgg_arr)
    grid_uniq, grid_n, grid_y = _level_counts(grid_arr, fire_grid)
    grid_y_arr = np.array([grid_y[str(g)] for g in grid_uniq])
    grid_n_arr = np.array([grid_n[str(g)] for g in grid_uniq])
    grid_parent = np.array([sgg_rate_by[grid_to_sgg[str(g)]] for g in grid_uniq])
    grid_rate = _eb_poisson_gamma(grid_y_arr, grid_n_arr, grid_parent)
    grid_rate_by = {str(g): float(v) for g, v in zip(grid_uniq, grid_rate)}

    # ── 전주별 격자율 broadcast → 배율 ──────────────────────────────────────
    pole_rate = np.array([grid_rate_by[str(g)] for g in grid_arr], dtype=np.float64)
    mult = pole_rate / max(global_rate, _RATE_FLOOR)

    if normalize == "unit":
        mult = _percentile_norm(mult)
    else:
        # 전역 가중평균(전주 균등가중)을 정확히 1 로 맞춘다(드리프트 보정).
        mean_mult = float(mult.mean())
        if mean_mult > 0:
            mult = mult / mean_mult

    if not np.all(np.isfinite(mult)):
        raise ValueError("배율에 비유한값 존재(내부 오류)")
    if mult.min() < 0:
        raise ValueError("배율 음수(내부 오류)")

    logger.info(
        "regional_multiplier: 전주=%d 귀속발화=%d 전역율=%.3e | 배율 mean=%.3f "
        "min=%.3f p50=%.3f max=%.3f (normalize=%s)",
        n_poles, n_fires_anchored, global_rate, float(mult.mean()),
        float(mult.min()), float(np.median(mult)), float(mult.max()), normalize,
    )
    return mult


def _dominant_parent(
    child_of_pole: np.ndarray, parent_of_pole: np.ndarray
) -> dict[str, str]:
    """각 자식 라벨의 대표(최빈) 부모 라벨 매핑.

    격자가 시군 경계를 걸칠 수 있어, 격자→시군은 그 격자 전주의 최빈 시군으로 정한다.

    Parameters
    ----------
    child_of_pole : numpy.ndarray, shape (N,)
        전주별 자식 라벨(예: grid_id).
    parent_of_pole : numpy.ndarray, shape (N,)
        전주별 부모 라벨(예: sgg).

    Returns
    -------
    dict[str, str]
        자식 라벨 → 최빈 부모 라벨.
    """
    out: dict[str, str] = {}
    for c in np.unique(child_of_pole):
        sel = child_of_pole == c
        parents, cnt = np.unique(parent_of_pole[sel], return_counts=True)
        out[str(c)] = str(parents[int(cnt.argmax())])
    return out


def _percentile_norm(x: np.ndarray) -> np.ndarray:
    """순위 백분위 정규화 → [0,1]. 동점은 평균순위 근사, 단조 보존(이상치 강건).

    hazard._percentile_norm 와 동일 의도(독립 모듈이라 자체 보유).
    """
    n = x.shape[0]
    if n == 0:
        return x.astype(np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    if n > 1:
        ranks /= (n - 1)
    return ranks


def region_rate_table(
    poles: pl.DataFrame,
    fires: pl.DataFrame,
    *,
    sgg_col: str = "sgg",
    lon_col: str = "lon",
    lat_col: str = "lat",
    anchor_radius_km: float = _ANCHOR_RADIUS_KM,
) -> pl.DataFrame:
    """체제·시군 단계의 EB 율과 축소 진단을 표로 반환(데모/검증용).

    regional_multiplier 와 동일한 집계·EB 로직을 시군 단계까지 노출해, 발화 0/소수
    시군이 부모(체제→전역)율로 얼마나 축소됐는지 확인할 수 있게 한다.

    Parameters
    ----------
    poles : polars.DataFrame
        마스터(또는 부분집합).
    fires : polars.DataFrame
        발화점(train fold 부분집합 가능).
    sgg_col, lon_col, lat_col : str
        컬럼명.
    anchor_radius_km : float
        발화점 귀속 반경.

    Returns
    -------
    polars.DataFrame
        컬럼: level('regime'|'sgg'), name, regime(체제), n_poles, n_fires,
        obs_rate(관측율 y/n), parent_rate(부모 사전율), eb_rate(EB 사후율),
        shrink_w(축소 가중 w=n/(n+β) 근사=eb가 obs에 가까운 정도),
        mult(eb_rate/global_rate). global_rate 는 metadata 로 첫 행 parent 에 반영.
    """
    pole_xy = poles.select([lon_col, lat_col]).to_numpy().astype(np.float64)
    sgg_arr = np.asarray(poles[sgg_col].to_list(), dtype=object)
    regime_arr = np.array(
        [config.SGG_TO_REGIME.get(str(s), "global") for s in sgg_arr], dtype=object
    )
    fire_xy = fires.select([lon_col, lat_col]).to_numpy().astype(np.float64)
    if fire_xy.shape[0]:
        f2p = _assign_fires_to_poles(pole_xy, fire_xy, anchor_radius_km)
        mapped = f2p[f2p >= 0]
    else:
        mapped = np.empty(0, dtype=np.int64)
    fire_regime = regime_arr[mapped] if mapped.size else np.empty(0, dtype=object)
    fire_sgg = sgg_arr[mapped] if mapped.size else np.empty(0, dtype=object)

    global_rate = max(int(mapped.size) / max(poles.height, 1), _RATE_FLOOR)

    reg_uniq, reg_n, reg_y = _level_counts(regime_arr, fire_regime)
    reg_y_arr = np.array([reg_y[str(r)] for r in reg_uniq])
    reg_n_arr = np.array([reg_n[str(r)] for r in reg_uniq])
    reg_parent = np.full(reg_uniq.size, global_rate)
    reg_rate = _eb_poisson_gamma(reg_y_arr, reg_n_arr, reg_parent)
    reg_rate_by = {str(r): float(v) for r, v in zip(reg_uniq, reg_rate)}

    sgg_uniq, sgg_n, sgg_y = _level_counts(sgg_arr, fire_sgg)
    sgg_to_reg = {str(s): config.SGG_TO_REGIME.get(str(s), "global") for s in sgg_uniq}
    sgg_y_arr = np.array([sgg_y[str(s)] for s in sgg_uniq])
    sgg_n_arr = np.array([sgg_n[str(s)] for s in sgg_uniq])
    sgg_parent = np.array([reg_rate_by[sgg_to_reg[str(s)]] for s in sgg_uniq])
    sgg_rate = _eb_poisson_gamma(sgg_y_arr, sgg_n_arr, sgg_parent)

    rows = []
    for r, n, y, p, e in zip(reg_uniq, reg_n_arr, reg_y_arr, reg_parent, reg_rate):
        obs = y / n if n > 0 else float("nan")
        rows.append(dict(level="regime", name=str(r), regime=str(r),
                         n_poles=float(n), n_fires=float(y), obs_rate=float(obs),
                         parent_rate=float(p), eb_rate=float(e),
                         mult=float(e / global_rate)))
    for s, n, y, p, e in zip(sgg_uniq, sgg_n_arr, sgg_y_arr, sgg_parent, sgg_rate):
        obs = y / n if n > 0 else float("nan")
        rows.append(dict(level="sgg", name=str(s), regime=sgg_to_reg[str(s)],
                         n_poles=float(n), n_fires=float(y), obs_rate=float(obs),
                         parent_rate=float(p), eb_rate=float(e),
                         mult=float(e / global_rate)))
    df = pl.DataFrame(rows)
    # shrink_w: EB 가 관측율에 얼마나 붙었나 = (eb-parent)/(obs-parent), 0..1 근사.
    df = df.with_columns(
        pl.when((pl.col("obs_rate") - pl.col("parent_rate")).abs() < 1e-15)
        .then(pl.lit(0.0))
        .otherwise((pl.col("eb_rate") - pl.col("parent_rate"))
                   / (pl.col("obs_rate") - pl.col("parent_rate")))
        .alias("shrink_w")
    )
    return df
