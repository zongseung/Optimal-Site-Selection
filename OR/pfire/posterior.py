"""사후분포(posterior) 불확실성 — Phase-5.

설계 근거(claudedocs/기획서_phase5_사후분포_커버리지_20260619.md):
    사후분포의 *본체* 는 **베이지안 계층 Poisson–Gamma 켤레(closed-form, MCMC 아님)**
    이다. 격자/시군 발화수 → posterior Gamma(α0+Σy, β0+Σexposure). 부모(체제→전역)
    단계의 (α0, β0) 가 사전(prior)이 되어 **zero-event 지역은 부모로 수렴하면서 사후가
    자연히 넓어진다**(불확실성 자동 표현). hierarchy.py 의 EB 점추정(사후평균)을
    여기서 **full posterior(분포)** 로 승급한다.

    이 모듈은 두 층을 제공한다.
      [층1] regional_rate_posterior — 지역(격자) 발화율의 posterior Gamma(α, β).
      [층2] propagate_risk_posterior — MC 로 지역율 사후를 draw 해 물리 base_risk
            (I·S·W·exposure) 에 곱하고 전주별 위험의 사후예측(mean·credible)을 낸다.
            (선택) Bayesian bootstrap 으로 물리 가중 변동을 합성한다.

누수 방지(중요):
    hierarchy.regional_multiplier 와 동일하게 `fires` 를 인자로 받으므로, 공간 블록 CV
    에서 각 fold 의 **train fold 발화점만** 담은 `fires` 를 넘기면 그 fold 기준 사후가
    산출된다(test fold 발화점 미사용). 발화점은 per-pole 라벨이 아니라 **지역 집계
    앵커**(전주당 발화 강도 추정용)로만 쓴다 — 어떤 전주도 자신이 양성이라는 라벨을
    받지 않으며, 사후는 그 전주가 속한 지역의 켤레 추정에서만 나온다.

결정성:
    모든 무작위(사후 draw·Bayesian bootstrap)는 config.SEED 로 시드된
    numpy.random.default_rng 를 쓴다 → 동일 입력·동일 seed 면 동일 출력.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu

from pfire import config, hierarchy

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────
# 켤레 사전(prior) 안정화 상수 — hierarchy 의 EB 정칙화와 정합.
# ──────────────────────────────────────────────────────────────────────────
# 부모율 m·사전분산 v 에서 (α0, β0)=(m²/v, m/v) 를 적률추정한다. v 가 퇴화하거나
# m≈0 이면 사전이 발산하므로 hierarchy 와 같은 floor/clip 으로 안정화한다.
_RATE_FLOOR: float = hierarchy._RATE_FLOOR
_VAR_FLOOR_FRAC: float = hierarchy._VAR_FLOOR_FRAC
_BETA_MIN: float = hierarchy._BETA_MIN
_BETA_MAX: float = hierarchy._BETA_MAX
_ANCHOR_RADIUS_KM: float = hierarchy._ANCHOR_RADIUS_KM


@dataclass(frozen=True)
class RatePosterior:
    """전주별(격자 broadcast) 발화율 posterior Gamma(α, β).

    Gamma(shape=α, rate=β) 의 평균은 α/β, 분산은 α/β². zero-event 격자는 α 가
    부모 사전(α0)으로 수렴하고 노출(전주수)이 작아 β 도 작아져 **분산이 커진다**
    (사후가 넓다 = 불확실성↑). draw_rate 로 사후 표집을 한다.

    Attributes
    ----------
    alpha : numpy.ndarray, shape (N,)
        전주별 posterior Gamma shape α = α0 + Σy(그 전주 격자의 발화수).
    beta : numpy.ndarray, shape (N,)
        전주별 posterior Gamma rate β = β0 + Σexposure(그 전주 격자의 전주수).
    global_rate : float
        전역 발화율(전체 귀속발화/전체전주) — 상대배율 정규화 기준.
    grid_alpha : dict[str, float]
        격자 id → posterior α(진단/검증용).
    grid_beta : dict[str, float]
        격자 id → posterior β(진단/검증용).
    sgg_of_grid : dict[str, str]
        격자 id → 대표 시군(진단용).
    sgg_alpha : dict[str, float]
        시군 → 시군단계 posterior α(모든 16 시군 보장; zero-event 진단용).
    sgg_beta : dict[str, float]
        시군 → 시군단계 posterior β(모든 16 시군 보장; zero-event 진단용).
    """

    alpha: np.ndarray
    beta: np.ndarray
    global_rate: float
    grid_alpha: dict[str, float]
    grid_beta: dict[str, float]
    sgg_of_grid: dict[str, str]
    sgg_alpha: dict[str, float]
    sgg_beta: dict[str, float]

    @property
    def mean(self) -> np.ndarray:
        """전주별 사후평균 율 α/β (shape (N,))."""
        return self.alpha / np.maximum(self.beta, _RATE_FLOOR)

    def draw_rate(
        self, n_draws: int, rng: np.random.Generator,
        pole_subset: np.ndarray | None = None,
    ) -> np.ndarray:
        """지역율 사후에서 (n_draws, M) 표집(상대배율, 전역평균≈1 스케일).

        각 전주 i 에서 Gamma(α_i, scale=1/β_i) 를 n_draws 회 뽑아 전역율로 나눠
        **상대배율**(전역평균≈1, 곱셈 결합용)로 반환한다. 같은 격자 전주는 동일
        (α, β) 라 같은 분포를 공유한다(격자별 1회 표집 후 broadcast 로 효율화).

        Parameters
        ----------
        n_draws : int
            사후 draw 수.
        rng : numpy.random.Generator
            결정성 RNG(config.SEED 로 시드된 것 권장).
        pole_subset : numpy.ndarray or None
            주어지면 그 전주 인덱스만 broadcast(메모리 절약 — 커버리지 CV 용).
            None 이면 전 전주(M=N). **격자 표집은 항상 전체 격자에 대해 같은 seed
            로 하므로** subset 여부와 무관하게 같은 전주는 같은 draw 를 받는다(결정성).

        Returns
        -------
        numpy.ndarray, shape (n_draws, M)
            전주별 지역율 사후 draw(전역율 정규화 상대배율, >=0). M=len(pole_subset)
            또는 N.
        """
        # 격자 단위로 한 번만 표집해 broadcast(메모리·시간 절약).
        grids = list(self.grid_alpha.keys())
        a = np.array([self.grid_alpha[g] for g in grids], dtype=np.float64)
        b = np.array([self.grid_beta[g] for g in grids], dtype=np.float64)
        # Gamma(shape=a, scale=1/b) → 율. (n_draws, n_grid).
        draws_g = rng.gamma(shape=a[None, :], scale=1.0 / np.maximum(b[None, :], _RATE_FLOOR),
                            size=(n_draws, a.size))
        draws_g = draws_g / max(self.global_rate, _RATE_FLOOR)
        # 전주 → 격자 인덱스 broadcast(필요시 subset 만).
        col = self._pole_grid_idx
        if pole_subset is not None:
            col = col[pole_subset]
        return draws_g[:, col]

    # _pole_grid_idx 는 regional_rate_posterior 가 채운다(전주별 격자 열 인덱스).
    _pole_grid_idx: np.ndarray = None  # type: ignore[assignment]


def _gamma_prior_from_parent(
    m: float, v: float
) -> tuple[float, float]:
    """부모율 m·사전분산 v → 켤레 Gamma 사전 (α0, β0). 적률추정·안정화.

    α0 = m²/v, β0 = m/v. v 퇴화/m≈0 은 hierarchy 와 동일하게 floor/clip 한다.
    zero-event 격자에서 이 사전이 사후를 지배(부모로 수렴)하고, v 가 클수록 β0 가
    작아 사후가 넓어진다.

    Parameters
    ----------
    m : float
        부모 단계 사전평균 율(이미 EB 추정된 상위단계 율).
    v : float
        같은 부모를 공유하는 자식들의 노출가중 율 분산.

    Returns
    -------
    tuple[float, float]
        (α0, β0). β0 ∈ [_BETA_MIN, _BETA_MAX], α0 = β0·m.
    """
    m = max(float(m), _RATE_FLOOR)
    v_floor = (m * _VAR_FLOOR_FRAC) ** 2
    v = max(float(v), v_floor)
    beta0 = _BETA_MAX if v <= 0.0 else m / v
    beta0 = float(np.clip(beta0, _BETA_MIN, _BETA_MAX))
    alpha0 = beta0 * m
    return alpha0, beta0


def regional_rate_posterior(
    poles: pl.DataFrame,
    fires: pl.DataFrame,
    *,
    sgg_col: str = "sgg",
    grid_col: str = "grid_id",
    lon_col: str = "lon",
    lat_col: str = "lat",
    anchor_radius_km: float = _ANCHOR_RADIUS_KM,
    seed: int = config.SEED,
) -> RatePosterior:
    """지역(격자) 발화율의 베이지안 계층 Poisson–Gamma 켤레 사후를 추정한다.

    계층 전역(1)→체제(3)→시군(16)→격자(106) 에서 부모 단계의 EB 사후평균 율을
    자식 단계 켤레 **사전(prior)** (α0, β0)=(m²/v, m/v) 으로 쓰고, 자식의 노출 n·
    사건 y 로 **posterior Gamma(α0+y, β0+n)** 를 닫힌형으로 얻는다(MCMC 불필요).
    격자 단계 사후를 전주별로 broadcast 한 :class:`RatePosterior` 를 반환한다.

    **zero-event 지역(고성·인제·화천·태백 등)**: y=0 이라 α=α0(부모 사전 지배),
    노출이 작은 소표본 격자는 β=β0+n 도 작아 **사후 분산 α/β² 가 커진다** → 점추정은
    부모율로 수렴하되 credible interval 이 넓다(불확실성 정직 표현).

    **누수 방지(필수)**: `fires` 부분집합(train fold)만 넘기면 그 fold 기준 사후가
    산출된다. 발화점은 per-pole 라벨이 아니라 지역 집계 앵커로만 쓴다.

    Parameters
    ----------
    poles : polars.DataFrame
        마스터(또는 부분집합). 최소 [lon_col, lat_col, sgg_col, grid_col]. 행
        순서가 곧 출력(alpha/beta) 순서다.
    fires : polars.DataFrame
        발화점(검증/지역 앵커). 최소 [lon_col, lat_col]. train fold 부분집합 가능.
        빈 프레임이면 모든 지역이 전역 사전으로 수렴(정보 없음 → 균일 배율 mean=1).
    sgg_col, grid_col, lon_col, lat_col : str
        컬럼명.
    anchor_radius_km : float
        발화점→전주 귀속 반경(km). <=0 이면 거리무관 최근접.
    seed : int
        결정성 시드(현 사후 추정 자체는 무작위성 없음 — API 일관성·재현성 명시).

    Returns
    -------
    RatePosterior
        전주별 posterior Gamma(α, β) + 진단 사전(격자별 α/β, 격자→시군).

    Raises
    ------
    KeyError
        필요한 컬럼이 없을 때.
    """
    for c in (lon_col, lat_col, sgg_col, grid_col):
        if c not in poles.columns:
            raise KeyError(f"poles 에 필요한 컬럼 없음: {c}")
    for c in (lon_col, lat_col):
        if c not in fires.columns:
            raise KeyError(f"fires 에 필요한 컬럼 없음: {c}")

    _ = np.random.default_rng(seed)  # 결정성 계약(사후 추정 자체는 무작위 없음).

    n_poles = poles.height
    pole_xy = poles.select([lon_col, lat_col]).to_numpy().astype(np.float64)
    sgg_arr = np.asarray(poles[sgg_col].to_list(), dtype=object)
    grid_arr = np.asarray(poles[grid_col].to_list(), dtype=object)
    regime_arr = np.array(
        [config.SGG_TO_REGIME.get(str(s), "global") for s in sgg_arr], dtype=object
    )

    # ── 발화점 → 전주 귀속 → 지역 라벨 ─────────────────────────────────────
    fire_xy = fires.select([lon_col, lat_col]).to_numpy().astype(np.float64)
    if fire_xy.shape[0]:
        f2p = hierarchy._assign_fires_to_poles(pole_xy, fire_xy, anchor_radius_km)
        mapped = f2p[f2p >= 0]
    else:
        mapped = np.empty(0, dtype=np.int64)
    fire_regime = regime_arr[mapped] if mapped.size else np.empty(0, dtype=object)
    fire_sgg = sgg_arr[mapped] if mapped.size else np.empty(0, dtype=object)
    fire_grid = grid_arr[mapped] if mapped.size else np.empty(0, dtype=object)
    n_anchored = int(mapped.size)

    global_rate = max(n_anchored / max(n_poles, 1), _RATE_FLOOR)

    # ── 체제(부모=전역) EB 사후평균 율 (사전 m 으로 쓰기 위해) ──────────────
    reg_uniq, reg_n, reg_y = hierarchy._level_counts(regime_arr, fire_regime)
    reg_y_arr = np.array([reg_y[str(r)] for r in reg_uniq])
    reg_n_arr = np.array([reg_n[str(r)] for r in reg_uniq])
    reg_parent = np.full(reg_uniq.size, global_rate)
    reg_rate = hierarchy._eb_poisson_gamma(reg_y_arr, reg_n_arr, reg_parent)
    reg_rate_by = {str(r): float(v) for r, v in zip(reg_uniq, reg_rate)}

    # ── 시군(부모=체제) EB 사후평균 율 ──────────────────────────────────────
    sgg_uniq, sgg_n, sgg_y = hierarchy._level_counts(sgg_arr, fire_sgg)
    sgg_to_reg = {str(s): config.SGG_TO_REGIME.get(str(s), "global") for s in sgg_uniq}
    sgg_y_arr = np.array([sgg_y[str(s)] for s in sgg_uniq])
    sgg_n_arr = np.array([sgg_n[str(s)] for s in sgg_uniq])
    sgg_parent = np.array([reg_rate_by[sgg_to_reg[str(s)]] for s in sgg_uniq])
    sgg_rate = hierarchy._eb_poisson_gamma(sgg_y_arr, sgg_n_arr, sgg_parent)
    sgg_rate_by = {str(s): float(v) for s, v in zip(sgg_uniq, sgg_rate)}

    # ── 시군 단계 posterior Gamma(α, β) — 모든 16 시군 보장(zero-event 진단) ─
    # 부모(체제)별로 묶어 시군 관측율 분산 v 를 적률추정 → 체제별 켤레 사전 (α0,β0).
    # 시군 posterior = (α0 + y_sgg, β0 + n_sgg). 격자 dominant 누락과 무관하게
    # 발화 0~소수 시군(고성·인제·화천·태백)도 항상 포함된다.
    sgg_alpha: dict[str, float] = {}
    sgg_beta: dict[str, float] = {}
    sgg_lam = np.where(sgg_n_arr > 0, sgg_y_arr / np.maximum(sgg_n_arr, _RATE_FLOOR),
                       np.nan)
    sgg_parent_key = np.array([sgg_to_reg[str(s)] for s in sgg_uniq], dtype=object)
    for pk in np.unique(sgg_parent_key):
        sel = sgg_parent_key == pk
        m_s = float(sgg_parent[sel][0])
        n_sel = sgg_n_arr[sel]
        lam_sel = np.where(np.isfinite(sgg_lam[sel]), sgg_lam[sel], m_s)
        ntot = n_sel.sum()
        v = float(np.sum(n_sel * (lam_sel - m_s) ** 2) / ntot) if ntot > 0 else 0.0
        a0, b0 = _gamma_prior_from_parent(m_s, v)
        for s in sgg_uniq[sel]:
            ss = str(s)
            sgg_alpha[ss] = a0 + float(sgg_y[ss])
            sgg_beta[ss] = b0 + float(sgg_n[ss])

    # ── 격자 단계: 부모=시군율 → 켤레 사전 (α0, β0) → posterior Gamma ──────
    grid_to_sgg = hierarchy._dominant_parent(grid_arr, sgg_arr)
    grid_uniq, grid_n, grid_y = hierarchy._level_counts(grid_arr, fire_grid)

    # 같은 시군(부모)을 공유하는 격자들의 노출가중 율 분산 v 를 적률추정해
    # 시군별 켤레 사전 (α0, β0) 을 만든다. 이는 hierarchy._eb_poisson_gamma 의
    # 적률추정과 동일 식이되, 여기서는 (α0, β0) 자체를 보존해 사후 분포를 만든다.
    grid_lam = np.array(
        [grid_y[str(g)] / grid_n[str(g)] if grid_n[str(g)] > 0 else float("nan")
         for g in grid_uniq]
    )
    grid_parent_m = np.array([sgg_rate_by[grid_to_sgg[str(g)]] for g in grid_uniq])

    grid_alpha: dict[str, float] = {}
    grid_beta: dict[str, float] = {}
    # 부모(시군)별로 묶어 분산 적률추정 → 그 부모의 사전 (α0, β0) 을 자식 격자들에 적용.
    parent_key = np.array([grid_to_sgg[str(g)] for g in grid_uniq], dtype=object)
    for pk in np.unique(parent_key):
        sel = parent_key == pk
        m_g = float(grid_parent_m[sel][0])
        n_sel = np.array([grid_n[str(g)] for g in grid_uniq[sel]])
        lam_sel = np.where(np.isfinite(grid_lam[sel]), grid_lam[sel], m_g)
        ntot = n_sel.sum()
        if ntot > 0:
            v = float(np.sum(n_sel * (lam_sel - m_g) ** 2) / ntot)
        else:
            v = 0.0
        alpha0, beta0 = _gamma_prior_from_parent(m_g, v)
        for g in grid_uniq[sel]:
            gs = str(g)
            y_g = float(grid_y[gs])
            n_g = float(grid_n[gs])
            grid_alpha[gs] = alpha0 + y_g          # posterior shape
            grid_beta[gs] = beta0 + n_g            # posterior rate

    # ── 전주별 broadcast ────────────────────────────────────────────────────
    grids = list(grid_alpha.keys())
    g_to_col = {g: i for i, g in enumerate(grids)}
    pole_grid_idx = np.array([g_to_col[str(g)] for g in grid_arr], dtype=np.int64)
    alpha_pole = np.array([grid_alpha[str(g)] for g in grid_arr], dtype=np.float64)
    beta_pole = np.array([grid_beta[str(g)] for g in grid_arr], dtype=np.float64)

    post = RatePosterior(
        alpha=alpha_pole, beta=beta_pole, global_rate=global_rate,
        grid_alpha=grid_alpha, grid_beta=grid_beta,
        sgg_of_grid={str(g): grid_to_sgg[str(g)] for g in grid_uniq},
        sgg_alpha=sgg_alpha, sgg_beta=sgg_beta,
    )
    object.__setattr__(post, "_pole_grid_idx", pole_grid_idx)

    mean_mult = float((post.mean / max(global_rate, _RATE_FLOOR)).mean())
    logger.info(
        "regional_rate_posterior: 전주=%d 귀속발화=%d 전역율=%.3e | 격자=%d "
        "평균배율(사후평균/전역)=%.3f",
        n_poles, n_anchored, global_rate, len(grid_alpha), mean_mult,
    )
    return post


def sgg_rate_posterior_table(post: RatePosterior) -> pl.DataFrame:
    """시군별 사후 율 요약표(평균·credible 폭) — zero-event 진단/검증용.

    **시군 단계 posterior Gamma(α, β)** 를 직접 써 16개 시군을 모두 집계한다(격자
    dominant 매핑에서 누락되는 소표본 시군 — 고성·인제·화천 — 도 항상 포함). 각
    시군의 사후 율 평균(α/β)·표준편차(√α/β)로 mult_mean(평균/전역율)·cv(변동계수)를
    낸다. **발화 0~소수 시군(고성·인제·화천·태백)은 cv 가 커야 한다**(사후가 넓음 =
    불확실성 자동 표현). 표는 cv 내림차순(불확실 상위) 정렬.

    Parameters
    ----------
    post : RatePosterior
        regional_rate_posterior 출력(sgg_alpha/sgg_beta 보유).

    Returns
    -------
    polars.DataFrame
        컬럼: sgg, n_fires(시군 귀속 발화수), mult_mean(사후평균/전역율),
        rel_sd(상대 사후표준편차), cv(변동계수=sd/mean=1/√α). cv 클수록 사후 넓음.
    """
    g = max(post.global_rate, _RATE_FLOOR)
    rows = []
    for sgg in post.sgg_alpha:
        a = float(post.sgg_alpha[sgg])
        b = float(post.sgg_beta[sgg])
        mean_rate = a / max(b, _RATE_FLOOR)
        sd_rate = float(np.sqrt(a)) / max(b, _RATE_FLOOR)   # √(α/β²)
        cv = float(1.0 / np.sqrt(max(a, _RATE_FLOOR)))      # sd/mean = 1/√α
        # 시군 귀속 발화수(=α − 사전α0 근사). 보고용: posterior α 에서 사전 차이.
        rows.append(dict(sgg=sgg, alpha=a, beta=b,
                         mult_mean=float(mean_rate / g),
                         rel_sd=float(sd_rate / g), cv=cv))
    return pl.DataFrame(rows).sort("cv", descending=True)


def propagate_risk_posterior(
    base_risk: np.ndarray,
    rate_posterior: "RatePosterior | SpatialRatePosterior",
    *,
    n_draws: int = 200,
    seed: int = config.SEED,
    q_lo: float = 0.05,
    q_hi: float = 0.95,
    weight_bootstrap: bool = False,
    weight_bootstrap_cv: float = 0.10,
    pole_subset: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """MC 로 지역율 사후를 전파해 전주별 위험 사후예측을 만든다.

    각 draw r 에서 지역율 사후배율 m^(r)(draw_rate, 전역평균≈1)을 물리 base_risk
    (=I·S·W·exposure 결합 위험)에 곱해 위험 사후예측 R^(r)=clip(base_risk·m^(r), 0,1)
    을 만든다. n_draws 회 표집해 전주별 평균(risk_mean)·분위(risk_lo=q_lo, risk_hi=q_hi)
    를 credible interval 로 반환한다.

    (선택) Bayesian bootstrap: 우도 없는 물리 가중·임계 불확실성을 합성한다. 각 draw
    마다 전역 곱셈 인자 w^(r) ~ 로그정규(평균 1, 변동계수 weight_bootstrap_cv) 를 뽑아
    base_risk 스케일을 흔든다(앵커·EXPERT_WEIGHTS 재표집 효과의 1차 근사). 순위는
    보존(전역 스칼라)하되 위험 크기 불확실성을 credible 폭에 반영한다.

    누수 주의(docstring): rate_posterior 는 fold train 발화점만으로 추정한 것을 넘겨야
    test 발화점이 사후에 새지 않는다(regional_rate_posterior 의 fires 부분집합).

    Parameters
    ----------
    base_risk : numpy.ndarray, shape (N,)
        물리 결합 위험 R=I·S·W(·exposure), [0,1]. 배율·표집이 곱해질 본체.
    rate_posterior : RatePosterior or SpatialRatePosterior
        regional_rate_posterior(독립 Poisson-Gamma) 또는 bym2_spatial_posterior
        (BYM2 공간 CAR) 출력. 둘 다 draw_rate(상대배율 표집) 인터페이스를 제공하므로
        백엔드 교체만으로 동일 전파가 된다.
    n_draws : int
        사후예측 MC draw 수(>=2). 클수록 credible 추정 안정.
    seed : int
        결정성 시드.
    q_lo, q_hi : float
        credible interval 분위(기본 0.05/0.95 → 90% 구간).
    weight_bootstrap : bool
        True 면 Bayesian bootstrap 물리가중 변동(로그정규 전역인자)을 합성.
    weight_bootstrap_cv : float
        가중 부트스트랩 로그정규 변동계수(0..). 0 이면 변동 없음.
    pole_subset : numpy.ndarray or None
        주어지면 그 전주 인덱스만 사후예측을 계산한다(메모리·시간 절약 — 커버리지
        CV 처럼 소수 전주만 필요할 때). 출력 배열은 len(pole_subset) 길이. 격자
        표집은 전체 격자 기준 같은 seed → subset 여부와 무관하게 결정성 보존.

    Returns
    -------
    dict[str, numpy.ndarray]
        {'risk_mean','risk_lo','risk_hi'} 각 shape (M,), [0,1]. M=len(pole_subset)
        또는 N. risk_lo<=risk_mean 근사<=risk_hi(분위라 엄밀 단조 보장은 아님).

    Raises
    ------
    ValueError
        형상/분위/draw 수가 부적합할 때.
    """
    base_risk = np.asarray(base_risk, dtype=np.float64)
    n = base_risk.shape[0]
    if rate_posterior.alpha.shape[0] != n:
        raise ValueError(
            f"base_risk(N={n}) 와 rate_posterior(N={rate_posterior.alpha.shape[0]}) 불일치")
    if n_draws < 2:
        raise ValueError("n_draws 는 >=2")
    if not (0.0 <= q_lo < q_hi <= 1.0):
        raise ValueError("0<=q_lo<q_hi<=1 이어야 함")

    rng = np.random.default_rng(seed)
    mult_draws = rate_posterior.draw_rate(n_draws, rng, pole_subset=pole_subset)
    base_sub = base_risk if pole_subset is None else base_risk[pole_subset]

    if weight_bootstrap and weight_bootstrap_cv > 0:
        # 로그정규(평균 1): mu = -σ²/2, σ = log(1+cv²)^0.5.
        sigma = float(np.sqrt(np.log(1.0 + weight_bootstrap_cv ** 2)))
        mu = -0.5 * sigma ** 2
        w = rng.lognormal(mean=mu, sigma=sigma, size=(n_draws, 1))
        mult_draws = mult_draws * w

    # R^(r) = clip(base_risk · m^(r), 0, 1). (n_draws, M).
    risk_draws = np.clip(base_sub[None, :] * mult_draws, 0.0, 1.0)

    risk_mean = risk_draws.mean(axis=0)
    risk_lo = np.quantile(risk_draws, q_lo, axis=0)
    risk_hi = np.quantile(risk_draws, q_hi, axis=0)

    logger.info(
        "propagate_risk_posterior: draws=%d 가중부트=%s | risk_mean p50=%.4g "
        "평균폭(hi-lo)=%.4g",
        n_draws, weight_bootstrap, float(np.median(risk_mean)),
        float(np.mean(risk_hi - risk_lo)),
    )
    return {"risk_mean": risk_mean, "risk_lo": risk_lo, "risk_hi": risk_hi}


# ══════════════════════════════════════════════════════════════════════════
# Phase-5b: BYM2 공간 CAR 사후(INLA식 Laplace 근사)
# ══════════════════════════════════════════════════════════════════════════
# 설계 근거(기획서_phase5 §6 한계 + 본 확장): Poisson-Gamma 켤레는 격자가 부모로만
# 수렴하고 **이웃끼리 정보를 안 빌린다**. BYM2 공간 CAR 로 확장해 이웃 격자에서 강도를
# 빌려(borrow strength) zero-event 격자의 사후를 더 정교하게 만든다.
#
# 모델(격자 i):
#   y_i ~ Poisson(E_i · exp(x_i)),   E_i = n_i · global_rate  (노출=전주수×전역율)
#   x = BYM2 잠재장 = (1/√τ)·(√(1−φ)·v + √φ·u_scaled)
#       v ~ N(0, I)               (비구조 iid)
#       u ~ ICAR(이웃)            (구조적; 분산 1 로 스케일된 u_scaled)
#   하이퍼 (τ,φ): τ 전체정밀도, φ∈[0,1] 구조비. x_i=0 이 상대위험 1(전역율).
#
# 추정(INLA 코어, MCMC 아님):
#   하이퍼 (τ,φ) 를 격자(config.BYM2_*_GRID)로 두고 각 점에서
#     ① 잠재장 x 의 사후모드 x* = argmax [Poisson loglik + GMRF prior] (sparse Newton)
#     ② 모드의 sparse Hessian H = -∂²/∂x² → x 의 가우시안 사후 N(x*, H⁻¹)
#     ③ Laplace 주변우도 log p(y|τ,φ) ≈ loglik(x*) + logprior(x*) − ½ log|H| + const
#   를 계산하고 ③ 가 최대인 (τ,φ) 의 (x*, diag(H⁻¹)) 를 채택한다(결정적).
#   → 격자별 로그상대위험 사후평균 μ_i=x*_i·, 분산 s²_i=(H⁻¹)_ii.
#
# 반환: :class:`SpatialRatePosterior` — RatePosterior 와 동일 인터페이스(alpha/beta/
#   global_rate/grid_*/draw_rate/_pole_grid_idx)라 propagate_risk_posterior 가 그대로
#   받는다. draw_rate 는 격자별 로그정규(μ_i, s²_i) 상대배율(exp(x), 전역≈1)을 표집한다.


def _parse_grid_rowcol(grid_ids: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """격자 id "행_열"(예 "70_130") → (행, 열, 파싱성공마스크) 정수 배열.

    형식이 "정수_정수" 가 아닌 격자는 마스크 False 로 표시(인접 계산에서 고립 취급).

    Parameters
    ----------
    grid_ids : list[str]
        격자 id 리스트(순서 보존).

    Returns
    -------
    rows : numpy.ndarray, shape (G,)
        행 인덱스(파싱 실패는 -1).
    cols : numpy.ndarray, shape (G,)
        열 인덱스(파싱 실패는 -1).
    ok : numpy.ndarray of bool, shape (G,)
        파싱 성공 마스크.
    """
    pat = re.compile(r"^(\d+)_(\d+)$")
    rows = np.full(len(grid_ids), -1, dtype=np.int64)
    cols = np.full(len(grid_ids), -1, dtype=np.int64)
    ok = np.zeros(len(grid_ids), dtype=bool)
    for i, g in enumerate(grid_ids):
        m = pat.match(str(g))
        if m:
            rows[i] = int(m.group(1))
            cols[i] = int(m.group(2))
            ok[i] = True
    return rows, cols, ok


@dataclass(frozen=True)
class GridAdjacency:
    """격자 rook 인접(행±1 또는 열±1) 희소 그래프 + 진단.

    Attributes
    ----------
    grids : list[str]
        격자 id(노드 순서 = 행렬 인덱스).
    W : scipy.sparse.csr_matrix, shape (G, G)
        대칭 0/1 인접행렬(자기루프 없음). rook 이웃.
    n_neighbors : numpy.ndarray, shape (G,)
        격자별 이웃 수(=W 행합).
    isolated : numpy.ndarray of bool, shape (G,)
        고립(이웃 0) 격자 마스크 — 구조항 못 쓰므로 부모(독립) 폴백 대상.
    n_components : int
        연결성분 수(1 이면 전체 연결).
    component_label : numpy.ndarray, shape (G,)
        격자별 연결성분 라벨.
    """

    grids: list[str]
    W: sp.csr_matrix
    n_neighbors: np.ndarray
    isolated: np.ndarray
    n_components: int
    component_label: np.ndarray

    @property
    def n_grids(self) -> int:
        return len(self.grids)

    @property
    def mean_neighbors(self) -> float:
        return float(self.n_neighbors.mean()) if self.n_neighbors.size else 0.0


def build_grid_adjacency(grid_ids: list[str]) -> GridAdjacency:
    """격자 id "행_열" 리스트 → rook 인접(행±1 또는 열±1) 희소 그래프.

    같은 행에서 열이 1 차이거나, 같은 열에서 행이 1 차이인 격자쌍을 이웃으로 잇는다
    (rook 4-이웃). 대각(queen) 은 잇지 않는다. 그래프 연결성·고립 격자를 점검해
    반환한다(고립 격자는 BYM2 에서 구조항 없이 iid 만 적용 → 부모 폴백).

    Parameters
    ----------
    grid_ids : list[str]
        고유 격자 id 리스트("행_열" 형식). 순서가 곧 그래프 노드 인덱스다.

    Returns
    -------
    GridAdjacency
        인접행렬 W·이웃수·고립마스크·연결성분 진단.
    """
    grids = list(grid_ids)
    G = len(grids)
    rows, cols, ok = _parse_grid_rowcol(grids)
    # (행,열) → 격자 인덱스. 파싱 성공한 것만.
    pos_to_idx: dict[tuple[int, int], int] = {}
    for i in range(G):
        if ok[i]:
            pos_to_idx[(int(rows[i]), int(cols[i]))] = i

    ii: list[int] = []
    jj: list[int] = []
    for i in range(G):
        if not ok[i]:
            continue
        r, c = int(rows[i]), int(cols[i])
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            j = pos_to_idx.get((r + dr, c + dc))
            if j is not None and j != i:
                ii.append(i)
                jj.append(j)
    if ii:
        data = np.ones(len(ii), dtype=np.float64)
        W = sp.coo_matrix((data, (np.array(ii), np.array(jj))), shape=(G, G)).tocsr()
        W = W.maximum(W.T)  # 대칭화(상호 이웃 보장)
        W.setdiag(0.0)
        W.eliminate_zeros()
    else:
        W = sp.csr_matrix((G, G), dtype=np.float64)

    n_neighbors = np.asarray(W.sum(axis=1)).ravel()
    isolated = n_neighbors <= 0
    n_comp, comp = connected_components(W, directed=False)
    logger.info(
        "build_grid_adjacency: 격자=%d 간선=%d 평균이웃=%.2f 고립=%d 연결성분=%d",
        G, int(W.nnz // 2), float(n_neighbors.mean()) if G else 0.0,
        int(isolated.sum()), int(n_comp),
    )
    return GridAdjacency(
        grids=grids, W=W, n_neighbors=n_neighbors, isolated=isolated,
        n_components=int(n_comp), component_label=comp,
    )


def _icar_precision(adj: GridAdjacency, jitter: float) -> sp.csr_matrix:
    """ICAR 구조 정밀도 Q_u = D − W (+ jitter·I). rook 그래프 라플라시안.

    D=대각 이웃수, W=인접행렬. Q_u 는 (연결성분당) rank-deficient(상수 영공간)라
    sum-to-zero 가 필요하지만, BYM2 에서는 작은 jitter·I 를 더해 양정치로 만들어
    Cholesky/LU 가 돌게 한다(질병지도 표준 트릭). 고립 격자는 D=0 이라 jitter 만 남아
    iid 처럼 동작한다(구조 평활 없음 → 부모 폴백).

    Parameters
    ----------
    adj : GridAdjacency
        격자 인접 그래프.
    jitter : float
        대각 정칙화(>0).

    Returns
    -------
    scipy.sparse.csr_matrix, shape (G, G)
        양정치 근사 ICAR 정밀도.
    """
    G = adj.n_grids
    deg = sp.diags(adj.n_neighbors, format="csr")
    Q = (deg - adj.W).tocsr()
    Q = Q + jitter * sp.identity(G, format="csr")
    return Q.tocsr()


def _laplace_latent_mode(
    y: np.ndarray, E: np.ndarray, Q_prior: sp.csr_matrix,
    tol: float, max_iter: int,
) -> tuple[np.ndarray, sp.csr_matrix]:
    """sparse Newton 으로 GMRF+Poisson 잠재장 x 의 사후모드와 정밀(Hessian)을 구한다.

    목적(음의 로그사후, 상수 제외):
        f(x) = Σ_i [E_i·exp(x_i) − y_i·x_i] + ½ xᵀ Q_prior x
    경사 g(x) = E·exp(x) − y + Q_prior x,  헤시안 H(x) = diag(E·exp(x)) + Q_prior.
    H 는 항상 양정치(Q_prior 양정치 + 대각 양수)라 Newton 이 안정적으로 수렴한다.
    스텝은 damped(half-stepping)로 발산을 막는다. 결정적(무작위 없음).

    Parameters
    ----------
    y : numpy.ndarray, shape (G,)
        격자별 사건수(발화수).
    E : numpy.ndarray, shape (G,)
        격자별 노출(전주수×전역율). >0.
    Q_prior : scipy.sparse.csr_matrix, shape (G, G)
        BYM2 잠재 prior 정밀도(양정치).
    tol : float
        경사 무한노름 수렴 기준.
    max_iter : int
        최대 Newton 반복.

    Returns
    -------
    x : numpy.ndarray, shape (G,)
        잠재장 사후모드.
    H : scipy.sparse.csr_matrix, shape (G, G)
        모드에서의 정밀(Hessian) = diag(E·exp(x)) + Q_prior.
    """
    G = y.shape[0]
    x = np.zeros(G, dtype=np.float64)
    Q = Q_prior.tocsr()
    for _ in range(max_iter):
        lam = E * np.exp(x)
        grad = lam - y + Q.dot(x)
        if np.max(np.abs(grad)) < tol:
            break
        H = (sp.diags(lam, format="csr") + Q).tocsc()
        lu = splu(H)
        step = lu.solve(grad)
        # damped Newton: f 가 줄 때까지 스텝 절반(과한 exp 발산 방지).
        f0 = float(np.sum(lam - y * x) + 0.5 * x.dot(Q.dot(x)))
        t = 1.0
        for _ls in range(30):
            x_new = x - t * step
            lam_new = E * np.exp(x_new)
            f_new = float(np.sum(lam_new - y * x_new) + 0.5 * x_new.dot(Q.dot(x_new)))
            if np.isfinite(f_new) and f_new <= f0 + 1e-12:
                break
            t *= 0.5
        x = x - t * step
    lam = E * np.exp(x)
    H = (sp.diags(lam, format="csr") + Q).tocsr()
    return x, H


def _sparse_logdet(H: sp.csr_matrix) -> float:
    """양정치 sparse 행렬의 log|H| (LU 대각의 로그합). 결정적.

    Parameters
    ----------
    H : scipy.sparse.csr_matrix
        양정치 sparse 행렬.

    Returns
    -------
    float
        log|H|.
    """
    lu = splu(H.tocsc())
    du = lu.U.diagonal()
    dl = lu.L.diagonal()
    # |H| = |L||U|. permutation 부호는 양정치라 +. 안전하게 abs 로그.
    return float(np.sum(np.log(np.abs(du))) + np.sum(np.log(np.abs(dl))))


def _diag_inv(H: sp.csr_matrix) -> np.ndarray:
    """sparse 양정치 H 의 역행렬 대각 diag(H⁻¹) — 열별 solve(결정적).

    G(=106) 가 작아 단위벡터 G 회 solve 로 정확한 대각을 얻는다(근사 아님).

    Parameters
    ----------
    H : scipy.sparse.csr_matrix, shape (G, G)
        양정치 정밀(Hessian).

    Returns
    -------
    numpy.ndarray, shape (G,)
        diag(H⁻¹) (>0).
    """
    G = H.shape[0]
    lu = splu(H.tocsc())
    inv_diag = np.empty(G, dtype=np.float64)
    e = np.zeros(G, dtype=np.float64)
    for i in range(G):
        e[i] = 1.0
        col = lu.solve(e)
        inv_diag[i] = col[i]
        e[i] = 0.0
    return np.maximum(inv_diag, _RATE_FLOOR)


def _bym2_prior_precision(
    Q_icar: sp.csr_matrix, tau: float, phi: float, n_grids: int,
) -> sp.csr_matrix:
    """BYM2 잠재 prior 정밀도 Q_x(τ,φ) — 구조 ICAR + 비구조 iid 혼합의 marginal 근사.

    엄밀 BYM2 는 (u, v) 결합이지만, INLA 구현에서 흔히 쓰는 잠재 marginal 정밀도
    근사를 쓴다: x ~ N(0, Σ), Σ = τ⁻¹[(1−φ)I + φ·Q_icar⁻], Q_icar⁻ 는 (scaled)
    ICAR 일반화역. 여기서는 결합 대신 **정밀도 보간** Q_x = τ·[(1−φ)·I + φ·Q_icar]
    를 쓴다(φ→0 이면 iid τ·I = 독립 가우시안, φ→1 이면 구조 ICAR 지배). 이 형태는
    (a) 항상 양정치(jitter 포함), (b) φ 로 "이웃 빌림 강도" 를 직접 조절, (c) sparse
    유지 — 라는 우리 목적(이웃 borrow + 결정적 Laplace)에 충분하다.

    Parameters
    ----------
    Q_icar : scipy.sparse.csr_matrix, shape (G, G)
        jitter 포함 ICAR 정밀도(양정치).
    tau : float
        전체 정밀도(>0). 클수록 강축소(부모 수렴).
    phi : float
        구조비 ∈[0,1]. 클수록 공간평활(이웃 borrow) 강함.
    n_grids : int
        격자 수 G.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (G, G)
        BYM2 prior 정밀도 Q_x.
    """
    I = sp.identity(n_grids, format="csr")
    Qx = tau * ((1.0 - phi) * I + phi * Q_icar)
    return Qx.tocsr()


@dataclass(frozen=True)
class SpatialRatePosterior:
    """격자별 BYM2 공간 CAR 잠재 사후(로그상대위험 가우시안) + RatePosterior 호환 인터페이스.

    잠재장 x_i ~ N(μ_i, s²_i)(로그상대위험). 상대배율 = exp(x_i)(전역율 기준, ≈1 중심).
    draw_rate 는 격자별 로그정규(μ_i, s²_i)를 표집해 propagate_risk_posterior 에 그대로
    공급한다(RatePosterior 와 동일 메모리·결정성 계약).

    Attributes
    ----------
    alpha, beta : numpy.ndarray, shape (N,)
        propagate 호환용. RatePosterior 와 형식 일치(여기선 lognormal 모멘트 매칭으로
        Gamma(α,β) 근사: 평균 exp(μ+s²/2), 분산에서 (α,β) 적률추정). draw_rate 는
        이를 쓰지 않고 lognormal 을 직접 표집하므로 진단·호환 표시용이다.
    global_rate : float
        전역 발화율(상대배율 정규화 기준 — exp(x) 가 이미 상대배율이라 1.0).
    grid_mu, grid_s2 : dict[str, float]
        격자 id → 잠재 사후평균 μ·분산 s²(로그상대위험).
    grid_alpha, grid_beta : dict[str, float]
        격자 id → lognormal 모멘트매칭 Gamma(α,β)(진단/호환).
    sgg_of_grid : dict[str, str]
        격자 id → 대표 시군.
    sgg_alpha, sgg_beta : dict[str, float]
        시군 사후 Gamma(독립 Poisson-Gamma 경로와 동일 — 시군 진단 일관성 유지).
    tau, phi : float
        채택된 BYM2 하이퍼(주변우도 최대).
    adjacency : GridAdjacency
        쓰인 인접 그래프(진단).
    """

    alpha: np.ndarray
    beta: np.ndarray
    global_rate: float
    grid_mu: dict[str, float]
    grid_s2: dict[str, float]
    grid_alpha: dict[str, float]
    grid_beta: dict[str, float]
    sgg_of_grid: dict[str, str]
    sgg_alpha: dict[str, float]
    sgg_beta: dict[str, float]
    tau: float
    phi: float
    adjacency: GridAdjacency

    @property
    def mean(self) -> np.ndarray:
        """전주별 사후평균 상대배율 exp(μ_i + s²_i/2)(lognormal 평균)."""
        mu = self._pole_mu
        s2 = self._pole_s2
        return np.exp(mu + 0.5 * s2)

    def draw_rate(
        self, n_draws: int, rng: np.random.Generator,
        pole_subset: np.ndarray | None = None,
    ) -> np.ndarray:
        """격자별 로그정규(μ_i, s²_i) 상대배율(전역평균≈1)을 (n_draws, M) 표집.

        x_i ~ N(μ_i, s²_i) → 상대배율 exp(x_i)(전역율 기준 ≈1 중심). 같은 격자 전주는
        동일 (μ, s²) 라 격자별 1회 표집 후 broadcast(RatePosterior.draw_rate 와 동일 계약).

        Parameters
        ----------
        n_draws : int
            사후 draw 수.
        rng : numpy.random.Generator
            결정성 RNG.
        pole_subset : numpy.ndarray or None
            주어지면 그 전주 인덱스만 broadcast.

        Returns
        -------
        numpy.ndarray, shape (n_draws, M)
            상대배율 사후 draw(>=0).
        """
        grids = list(self.grid_mu.keys())
        mu = np.array([self.grid_mu[g] for g in grids], dtype=np.float64)
        s = np.sqrt(np.array([self.grid_s2[g] for g in grids], dtype=np.float64))
        z = rng.standard_normal(size=(n_draws, mu.size))
        draws_g = np.exp(mu[None, :] + s[None, :] * z)
        col = self._pole_grid_idx
        if pole_subset is not None:
            col = col[pole_subset]
        return draws_g[:, col]

    _pole_grid_idx: np.ndarray = None  # type: ignore[assignment]
    _pole_mu: np.ndarray = None        # type: ignore[assignment]
    _pole_s2: np.ndarray = None        # type: ignore[assignment]


def bym2_spatial_posterior(
    poles: pl.DataFrame,
    fires: pl.DataFrame,
    *,
    sgg_col: str = "sgg",
    grid_col: str = "grid_id",
    lon_col: str = "lon",
    lat_col: str = "lat",
    anchor_radius_km: float = _ANCHOR_RADIUS_KM,
    tau_grid: tuple[float, ...] = config.BYM2_TAU_GRID,
    phi_grid: tuple[float, ...] = config.BYM2_PHI_GRID,
    icar_jitter: float = config.BYM2_ICAR_JITTER,
    newton_tol: float = config.BYM2_NEWTON_TOL,
    newton_max_iter: int = config.BYM2_NEWTON_MAX_ITER,
    seed: int = config.SEED,
) -> SpatialRatePosterior:
    """BYM2 공간 CAR 격자 사후를 INLA식 Laplace 근사로 추정한다(MCMC 아님, 결정적).

    격자 i 발화수 y_i ~ Poisson(E_i·exp(x_i)), E_i=전주수·전역율(노출). 잠재 로그상대
    위험 x 는 BYM2(구조 ICAR 이웃평활 + 비구조 iid, 혼합비 φ·전체정밀도 τ). 하이퍼
    (τ,φ)를 격자(tau_grid×phi_grid)로 두고 각 점에서 Laplace 주변우도를 계산해 최대인
    점을 채택한다. 그 점의 잠재 사후모드 x* 와 sparse Hessian 으로 격자별 가우시안
    사후 N(μ_i=x*_i, s²_i=(H⁻¹)_ii)를 얻는다.

    **이웃 borrow(핵심)**: zero-event 격자(y=0)도 이웃 격자의 강도에서 정보를 빌려
    독립 Poisson-Gamma(부모로만 수렴)보다 사후평균이 이웃 쪽으로 당겨지고 분산이
    더 정교해진다. 고립 격자(이웃 0)는 구조항이 없어 iid 만 작동 → 부모(독립) 수준 폴백.

    **누수 방지**: `fires` 부분집합(train fold)만 넘기면 그 fold 기준 사후가 산출된다.
    발화점은 per-pole 라벨이 아니라 지역 집계 앵커로만 쓴다.

    Parameters
    ----------
    poles : polars.DataFrame
        마스터(또는 부분집합). 최소 [lon_col, lat_col, sgg_col, grid_col].
    fires : polars.DataFrame
        발화점(검증/지역 앵커). 최소 [lon_col, lat_col]. train fold 부분집합 가능.
    sgg_col, grid_col, lon_col, lat_col : str
        컬럼명.
    anchor_radius_km : float
        발화점→전주 귀속 반경(km).
    tau_grid, phi_grid : tuple[float]
        BYM2 하이퍼 (τ,φ) 탐색 격자.
    icar_jitter : float
        ICAR 정밀도 대각 jitter(양정치화).
    newton_tol, newton_max_iter : float, int
        잠재 모드 Newton 수렴 기준·최대반복.
    seed : int
        결정성 시드(추정 자체는 무작위 없음 — draw_rate 에서만 사용).

    Returns
    -------
    SpatialRatePosterior
        격자별 BYM2 잠재 사후 + propagate 호환 인터페이스.

    Raises
    ------
    KeyError
        필요한 컬럼이 없을 때.
    """
    for c in (lon_col, lat_col, sgg_col, grid_col):
        if c not in poles.columns:
            raise KeyError(f"poles 에 필요한 컬럼 없음: {c}")
    for c in (lon_col, lat_col):
        if c not in fires.columns:
            raise KeyError(f"fires 에 필요한 컬럼 없음: {c}")

    _ = np.random.default_rng(seed)  # 결정성 계약(추정 무작위 없음).

    n_poles = poles.height
    pole_xy = poles.select([lon_col, lat_col]).to_numpy().astype(np.float64)
    sgg_arr = np.asarray(poles[sgg_col].to_list(), dtype=object)
    grid_arr = np.asarray(poles[grid_col].to_list(), dtype=object)

    # ── 발화점 → 전주 귀속 → 격자 사건수 y_i, 노출(전주수) n_i ─────────────────
    fire_xy = fires.select([lon_col, lat_col]).to_numpy().astype(np.float64)
    if fire_xy.shape[0]:
        f2p = hierarchy._assign_fires_to_poles(pole_xy, fire_xy, anchor_radius_km)
        mapped = f2p[f2p >= 0]
    else:
        mapped = np.empty(0, dtype=np.int64)
    fire_grid = grid_arr[mapped] if mapped.size else np.empty(0, dtype=object)
    n_anchored = int(mapped.size)
    global_rate = max(n_anchored / max(n_poles, 1), _RATE_FLOOR)

    grid_uniq, grid_n, grid_y = hierarchy._level_counts(grid_arr, fire_grid)
    grids = [str(g) for g in grid_uniq]
    G = len(grids)
    y = np.array([grid_y[g] for g in grids], dtype=np.float64)
    n_grid = np.array([grid_n[g] for g in grids], dtype=np.float64)
    # 노출 E_i = 전주수 · 전역율 → λ_i=E_i·exp(x_i), x_i=0 ⇔ 상대위험 1(전역율).
    E = np.maximum(n_grid * global_rate, _RATE_FLOOR)

    grid_to_sgg = hierarchy._dominant_parent(grid_arr, sgg_arr)

    # ── 격자 인접 그래프 + ICAR 정밀도 ──────────────────────────────────────
    adj = build_grid_adjacency(grids)
    Q_icar = _icar_precision(adj, icar_jitter)

    # ── 하이퍼 (τ,φ) 격자 탐색: Laplace 주변우도 최대 ─────────────────────────
    # log p(y|τ,φ) ≈ [Σ(y·x − E·exp(x))] (Poisson loglik, 상수 제외)
    #               + ½·log|Q_x| − ½·xᵀQ_x x  (GMRF prior, 상수 제외)
    #               − ½·log|H|                 (Laplace 정규화; H=모드 Hessian)
    best = None  # (logml, tau, phi, x, H)
    if n_anchored == 0:
        # 발화 0 → 신호 없음. 약한 prior(τ 큰·φ 작은) 로 x≈0(상대위험 1) 수렴.
        tau_grid = (max(tau_grid),)
        phi_grid = (min(phi_grid),)
    for tau in tau_grid:
        for phi in phi_grid:
            Q_x = _bym2_prior_precision(Q_icar, tau, phi, G)
            x_mode, H = _laplace_latent_mode(y, E, Q_x, newton_tol, newton_max_iter)
            lam = E * np.exp(x_mode)
            loglik = float(np.sum(y * x_mode - lam))
            logdet_Qx = _sparse_logdet(Q_x.tocsr())
            quad = float(x_mode.dot(Q_x.dot(x_mode)))
            logdet_H = _sparse_logdet(H)
            logml = loglik + 0.5 * logdet_Qx - 0.5 * quad - 0.5 * logdet_H
            if best is None or logml > best[0]:
                best = (logml, float(tau), float(phi), x_mode, H)

    assert best is not None
    _, tau_star, phi_star, x_star, H_star = best
    s2 = _diag_inv(H_star)  # diag(H⁻¹) = 잠재 사후분산

    grid_mu = {grids[i]: float(x_star[i]) for i in range(G)}
    grid_s2 = {grids[i]: float(s2[i]) for i in range(G)}

    # lognormal(μ,s²) 모멘트매칭 Gamma(α,β)(호환/진단): 평균 m=exp(μ+s²/2),
    # 분산 var=(exp(s²)−1)·m² → α=m²/var, β=m/var.
    grid_alpha: dict[str, float] = {}
    grid_beta: dict[str, float] = {}
    for g in grids:
        mu_g, s2_g = grid_mu[g], grid_s2[g]
        m = float(np.exp(mu_g + 0.5 * s2_g))
        var = float((np.exp(s2_g) - 1.0) * m * m)
        var = max(var, (m * _VAR_FLOOR_FRAC) ** 2, _RATE_FLOOR)
        grid_alpha[g] = max(m * m / var, _RATE_FLOOR)
        grid_beta[g] = max(m / var, _RATE_FLOOR)

    # ── 시군 사후 Gamma(독립 경로와 동일 — 시군 진단 일관성) ──────────────────
    pg = regional_rate_posterior(
        poles, fires, sgg_col=sgg_col, grid_col=grid_col,
        lon_col=lon_col, lat_col=lat_col, anchor_radius_km=anchor_radius_km,
        seed=seed,
    )

    # ── 전주별 broadcast ────────────────────────────────────────────────────
    g_to_col = {g: i for i, g in enumerate(grids)}
    pole_grid_idx = np.array([g_to_col[str(g)] for g in grid_arr], dtype=np.int64)
    pole_mu = np.array([grid_mu[str(g)] for g in grid_arr], dtype=np.float64)
    pole_s2 = np.array([grid_s2[str(g)] for g in grid_arr], dtype=np.float64)
    alpha_pole = np.array([grid_alpha[str(g)] for g in grid_arr], dtype=np.float64)
    beta_pole = np.array([grid_beta[str(g)] for g in grid_arr], dtype=np.float64)

    post = SpatialRatePosterior(
        alpha=alpha_pole, beta=beta_pole, global_rate=1.0,  # exp(x) 이미 상대배율
        grid_mu=grid_mu, grid_s2=grid_s2,
        grid_alpha=grid_alpha, grid_beta=grid_beta,
        sgg_of_grid={g: grid_to_sgg[g] for g in grids},
        sgg_alpha=pg.sgg_alpha, sgg_beta=pg.sgg_beta,
        tau=tau_star, phi=phi_star, adjacency=adj,
    )
    object.__setattr__(post, "_pole_grid_idx", pole_grid_idx)
    object.__setattr__(post, "_pole_mu", pole_mu)
    object.__setattr__(post, "_pole_s2", pole_s2)

    mult_mean = float(np.exp(pole_mu + 0.5 * pole_s2).mean())
    logger.info(
        "bym2_spatial_posterior: 전주=%d 귀속발화=%d 전역율=%.3e | 격자=%d "
        "τ*=%.1f φ*=%.2f 평균배율(exp(μ+s²/2))=%.3f 고립격자=%d",
        n_poles, n_anchored, global_rate, G, tau_star, phi_star, mult_mean,
        int(adj.isolated.sum()),
    )
    return post


def spatial_sgg_rate_table(post: SpatialRatePosterior) -> pl.DataFrame:
    """BYM2 시군별 사후 율 요약표(격자 사후를 노출가중 집계) — borrow 효과 진단.

    각 시군의 격자 사후 상대배율 exp(μ_i+s²_i/2)을 전주수 가중평균해 시군 mult_mean 을,
    격자 잠재 표준편차의 가중평균으로 rel_sd·cv 를 낸다. 독립 Poisson-Gamma(부모 수렴)
    대비 BYM 은 zero-event 시군이 이웃 쪽으로 당겨지는 정도를 본다. cv 내림차순 정렬.

    Parameters
    ----------
    post : SpatialRatePosterior
        bym2_spatial_posterior 출력.

    Returns
    -------
    polars.DataFrame
        컬럼: sgg, mult_mean(시군 사후 상대배율), rel_sd(잠재 사후 표준편차 평균),
        cv(rel_sd/mult_mean). cv 클수록 사후 넓음.
    """
    # 시군 → 격자들.
    by_sgg: dict[str, list[str]] = {}
    for g, s in post.sgg_of_grid.items():
        by_sgg.setdefault(s, []).append(g)
    rows = []
    for sgg, gs in by_sgg.items():
        mults = np.array([np.exp(post.grid_mu[g] + 0.5 * post.grid_s2[g]) for g in gs])
        sds = np.array([np.sqrt(post.grid_s2[g]) for g in gs])
        mult_mean = float(mults.mean())
        rel_sd = float(sds.mean())
        cv = float(rel_sd / max(mult_mean, _RATE_FLOOR))
        rows.append(dict(sgg=sgg, mult_mean=mult_mean, rel_sd=rel_sd, cv=cv))
    return pl.DataFrame(rows).sort("cv", descending=True)
