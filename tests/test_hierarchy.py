"""pfire.hierarchy 단위 테스트 — 계층 경험적 베이즈(EB) 지역 보정.

검증 항목:
  - 출력 형상=N, 유한, 양수, 전역평균≈1(normalize='mean'), [0,1](normalize='unit').
  - 축소 동작: 고발화(대표본) 지역 → 관측율 근접, 0발화(소표본) 지역 → 부모율 수렴.
  - 부분집합(train fold) 입력에서도 정상 동작(누수방지 재계산 경로).
  - 빈 발화점 입력 → 정보 없음 → 모든 배율 1.

데이터 의존 없이 합성 프레임으로 결정적으로 검증한다(used_dataset 불필요).
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from pfire import config, hierarchy


# ──────────────────────────────────────────────────────────────────────────
# 합성 프레임 구성기
# ──────────────────────────────────────────────────────────────────────────
def _synthetic_poles() -> pl.DataFrame:
    """체제 2개·시군 5개·격자 여러 개를 가진 합성 전주 프레임(실데이터 스케일 모사).

    전주당 발화 강도(율)가 작도록(~1e-3) 지역을 크게 잡아, 소표본·0발화 지역이
    부모로 강하게 축소되게 한다(실데이터: 화천 364전주·0발화 → shrink_w<0.01).

    - 춘천(g_big_hot, yeongseo, 대표본): 발화 다수 → 관측율 근접.
    - 원주(g_big_cold, yeongseo, 대표본): 발화 0 → 부모(yeongseo)율로 축소.
    - 강릉(g_yd_big, yeongdong, 대표본): 발화 다수 → yeongdong 체제 앵커 고정.
    - 속초(g_tiny_zero, yeongdong, 소표본): 발화 0 → 부모(yeongdong)율로 강축소.
    - 양양(g_tiny_one, yeongdong, 소표본): 발화 소수(1) → 부모로 강축소.

    config.SGG_TO_REGIME 의 실제 시군명을 빌려 체제 매핑이 동작하도록 한다.
    """
    rows = []
    pid = 0

    def add(sgg: str, grid: str, lon0: float, lat0: float, n: int):
        nonlocal pid
        for i in range(n):
            rows.append(dict(
                pole_id=pid,
                lon=lon0 + (i % 100) * 1e-4,
                lat=lat0 + (i // 100) * 1e-4,
                sgg=sgg, grid_id=grid,
            ))
            pid += 1

    # 대표본 yeongseo
    add("춘천", "g_big_hot", 127.70, 37.90, 150_000)   # 발화 다수 예정
    add("원주", "g_big_cold", 126.20, 37.35, 130_000)  # 발화 0(좌표 분리)
    # 대표본 yeongdong(체제 앵커 안정화)
    add("강릉", "g_yd_big", 128.90, 37.75, 100_000)    # 발화 다수 예정
    # 소표본 yeongdong
    add("속초", "g_tiny_zero", 130.59, 38.20, 180)     # 발화 0
    add("양양", "g_tiny_one", 131.20, 38.05, 360)      # 발화 소수(1)
    return pl.DataFrame(rows)


def _fires_for(poles: pl.DataFrame, n_hot: int, n_tiny_one: int) -> pl.DataFrame:
    """big_hot·yd_big 격자에 발화점, tiny_one 격자에 n_tiny_one개 발화점 생성.

    big_hot(춘천)·yd_big(강릉)에 n_hot개씩 둬 두 체제 모두 발화 신호가 있게 하고
    (yeongdong 체제 앵커 안정화), tiny_one(양양)에 소수만 둔다. 발화점 좌표는 해당
    격자 첫 전주 좌표에 미세 오프셋(<<1km)을 줘 최근접 귀속이 그 전주가 되게 한다.
    """
    rows = []
    for grid, cnt in (("g_big_hot", n_hot), ("g_yd_big", n_hot), ("g_tiny_one", n_tiny_one)):
        sub = poles.filter(pl.col("grid_id") == grid)
        base_lon = sub["lon"][0]
        base_lat = sub["lat"][0]
        for i in range(cnt):
            rows.append(dict(lon=base_lon + (i % 50) * 1e-5, lat=base_lat + (i // 50) * 1e-5))
    if not rows:
        return pl.DataFrame({"lon": [], "lat": []}, schema={"lon": pl.Float64, "lat": pl.Float64})
    return pl.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────
# 출력 계약: 형상·유한·양수·정규화
# ──────────────────────────────────────────────────────────────────────────
def test_output_shape_finite_positive_mean1():
    poles = _synthetic_poles()
    fires = _fires_for(poles, n_hot=200, n_tiny_one=1)
    mult = hierarchy.regional_multiplier(poles, fires)
    assert mult.shape == (poles.height,)
    assert np.all(np.isfinite(mult))
    assert np.all(mult > 0.0)
    # 전역(전주 균등) 평균 ≈ 1.
    assert mult.mean() == pytest.approx(1.0, abs=1e-9)


def test_unit_normalize_range():
    poles = _synthetic_poles()
    fires = _fires_for(poles, n_hot=200, n_tiny_one=1)
    mult = hierarchy.regional_multiplier(poles, fires, normalize="unit")
    assert mult.shape == (poles.height,)
    assert mult.min() >= 0.0 and mult.max() <= 1.0
    assert np.all(np.isfinite(mult))


def test_bad_normalize_raises():
    poles = _synthetic_poles()
    fires = _fires_for(poles, n_hot=10, n_tiny_one=1)
    with pytest.raises(ValueError):
        hierarchy.regional_multiplier(poles, fires, normalize="bogus")


def test_missing_column_raises():
    poles = _synthetic_poles().drop("grid_id")
    fires = _fires_for(_synthetic_poles(), n_hot=10, n_tiny_one=1)
    with pytest.raises(KeyError):
        hierarchy.regional_multiplier(poles, fires)


# ──────────────────────────────────────────────────────────────────────────
# 축소 동작: 고발화→관측율 근접, 0발화→부모율 수렴
# ──────────────────────────────────────────────────────────────────────────
def test_high_fire_region_near_observed():
    """대표본·고발화 격자의 EB율은 관측율에 가깝다(shrink_w 높음)."""
    poles = _synthetic_poles()
    fires = _fires_for(poles, n_hot=400, n_tiny_one=1)
    tbl = hierarchy.region_rate_table(poles, fires)
    # 춘천(big_hot 시군) — 대표본+고발화 → shrink_w 크고 obs≈eb.
    chun = tbl.filter((pl.col("level") == "sgg") & (pl.col("name") == "춘천")).row(0, named=True)
    assert chun["n_fires"] >= 1
    assert chun["shrink_w"] > 0.5
    # EB율이 관측율 쪽으로 끌림(부모보다 관측에 가까움).
    assert abs(chun["eb_rate"] - chun["obs_rate"]) < abs(chun["eb_rate"] - chun["parent_rate"])


def test_zero_fire_small_region_shrinks_to_parent():
    """0발화 소표본 시군의 EB율 ≈ 부모(체제)율 (강축소)."""
    poles = _synthetic_poles()
    fires = _fires_for(poles, n_hot=200, n_tiny_one=1)
    tbl = hierarchy.region_rate_table(poles, fires)
    # 속초(tiny_zero, yeongdong, 발화0) → 부모 yeongdong율로 수렴.
    sok = tbl.filter((pl.col("level") == "sgg") & (pl.col("name") == "속초")).row(0, named=True)
    assert sok["n_fires"] == 0
    assert sok["obs_rate"] == 0.0
    # EB율이 부모율에 매우 가깝다(관측 0 에 머물지 않음).
    assert sok["eb_rate"] == pytest.approx(sok["parent_rate"], rel=0.1)
    # 축소 가중 작음(관측에 거의 안 붙음).
    assert sok["shrink_w"] < 0.2


def test_zero_fire_region_not_zero_multiplier():
    """0발화 지역이라도 배율이 0/극단이 아니라 부모 수준(>0)으로 안정."""
    poles = _synthetic_poles()
    fires = _fires_for(poles, n_hot=200, n_tiny_one=1)
    mult = hierarchy.regional_multiplier(poles, fires)
    # 속초(tiny_zero) 전주들의 배율은 모두 양수이고 극단(≈0)이 아니다.
    sok_mask = (poles["sgg"] == "속초").to_numpy()
    sok_mult = mult[sok_mask]
    assert np.all(sok_mult > 0.0)
    assert sok_mult.min() > 0.1  # 부모(yeongdong)율 수준 — 0 으로 붕괴하지 않음.


def test_hot_region_multiplier_exceeds_cold():
    """고발화 지역 배율 > 0발화 지역 배율 (분리력)."""
    poles = _synthetic_poles()
    fires = _fires_for(poles, n_hot=400, n_tiny_one=1)
    mult = hierarchy.regional_multiplier(poles, fires)
    hot = mult[(poles["grid_id"] == "g_big_hot").to_numpy()].mean()
    cold = mult[(poles["grid_id"] == "g_big_cold").to_numpy()].mean()
    assert hot > cold


# ──────────────────────────────────────────────────────────────────────────
# 부분집합(train fold) 입력 — 누수방지 재계산 경로
# ──────────────────────────────────────────────────────────────────────────
def test_subset_fires_runs_and_differs():
    """발화점 부분집합 입력에서도 정상 동작하고, 전체와 결과가 달라진다."""
    poles = _synthetic_poles()
    fires_full = _fires_for(poles, n_hot=400, n_tiny_one=1)
    fires_half = fires_full.head(fires_full.height // 2)
    mult_full = hierarchy.regional_multiplier(poles, fires_full)
    mult_half = hierarchy.regional_multiplier(poles, fires_half)
    assert mult_half.shape == (poles.height,)
    assert np.all(np.isfinite(mult_half)) and np.all(mult_half > 0)
    assert mult_half.mean() == pytest.approx(1.0, abs=1e-9)
    # 부분집합이면 hot 격자 발화수가 줄어 배율 분포가 바뀐다.
    assert not np.allclose(mult_full, mult_half)


def test_empty_fires_all_ones():
    """빈 발화점 → 지역 정보 없음 → 모든 배율이 정확히 1."""
    poles = _synthetic_poles()
    empty = pl.DataFrame({"lon": [], "lat": []},
                         schema={"lon": pl.Float64, "lat": pl.Float64})
    mult = hierarchy.regional_multiplier(poles, empty)
    assert mult.shape == (poles.height,)
    assert np.allclose(mult, 1.0)


def test_determinism():
    """같은 입력 → 같은 출력(결정성)."""
    poles = _synthetic_poles()
    fires = _fires_for(poles, n_hot=200, n_tiny_one=1)
    a = hierarchy.regional_multiplier(poles, fires, seed=config.SEED)
    b = hierarchy.regional_multiplier(poles, fires, seed=config.SEED)
    np.testing.assert_array_equal(a, b)


# ──────────────────────────────────────────────────────────────────────────
# EB 핵심 함수 직접 검증
# ──────────────────────────────────────────────────────────────────────────
def test_eb_poisson_gamma_shrinks_zero_to_parent():
    """_eb_poisson_gamma: 0사건 소표본은 부모율로, 대표본 고사건은 관측율로."""
    parent = 1e-3
    # 부모율 근처(저분산) 형제 다수 + 소표본0사건 1개 + 대표본고사건 1개.
    #   형제 분산이 작아야(realistic) 소표본이 부모로 강축소된다(EB 정석).
    #   여기에 부모율 근처 형제(대표본)들을 둬 between-group 분산을 작게 만든다.
    y = np.array([0.0, 1.0, 99.0, 101.0, 500.0])
    n = np.array([10.0, 100_000.0, 100_000.0, 100_000.0, 50_000.0])
    # obs: 0, ~1e-5, ~1e-3, ~1e-3, 1e-2. 부모율 1e-3 근처 형제 다수 → 저분산.
    parent_rate = np.full(5, parent)
    eb = hierarchy._eb_poisson_gamma(y, n, parent_rate)
    obs_last = y[-1] / n[-1]
    # 0사건 소표본(n=10) → 부모율 근접(관측 0 에 머물지 않음).
    assert eb[0] == pytest.approx(parent, rel=0.3)
    assert eb[0] > 0.0
    # 대표본 고사건 → 관측율 쪽으로 끌림(부모보다 관측에 가까움).
    assert abs(eb[-1] - obs_last) < abs(eb[-1] - parent)
    assert np.all(eb >= 0) and np.all(np.isfinite(eb))
