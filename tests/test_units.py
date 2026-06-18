"""핵심 함수 단위테스트 — 데이터 없이도 도는 합성 입력 기반.

검증 항목: 게이트 행합=1, 위험 단조성, exposure 폴백 형상/범위, 보정 단조,
제출 스키마(행수 1,387,831·decision 0/1만), 좌표변환 단일구현 일치,
바람표집 결정성, exposure 블렌드 폴백(alpha=0), 제출 무결성 게이트.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from pfire import (
    calibrate,
    config,
    exposure,
    exposure_engine,
    experts,
    geo,
    hazard,
    regimes,
    submit,
    validate,
)


def _synthetic_master(n: int = 2000, seed: int = 0) -> pl.DataFrame:
    """게이트·전문가·기상에 필요한 컬럼을 가진 합성 마스터."""
    rng = np.random.default_rng(seed)
    sgg_pool = list(config.SGG_TO_REGIME.keys())
    return pl.DataFrame({
        "pole_id": np.arange(n, dtype=np.int64),
        "lon": rng.uniform(127.5, 129.4, n),
        "lat": rng.uniform(37.0, 38.3, n),
        "elevation": rng.uniform(0, 1400, n),
        "yanggan_days": rng.uniform(0, 9, n),
        "sgg": rng.choice(sgg_pool, n),
        "dist_to_forest": rng.uniform(0, 2000, n),
        "dist_to_road": rng.uniform(0, 5000, n),
        "dist_to_powerline": rng.uniform(0, 20000, n),
        "fwi_q90": rng.uniform(1, 14, n),
        "fwi_high_danger_days": rng.uniform(0, 1, n),
        "mu_flammability": rng.uniform(0, 1, n),
        "mu_southness": rng.uniform(0, 1, n),
        "ndvi": rng.uniform(-0.5, 0.7, n),
        "S_p": rng.uniform(0.05, 0.98, n),
        "nn_station_km": rng.uniform(0, 30, n),
        "extrap_flag": rng.integers(0, 2, n).astype(bool),
    })


# ── 게이트 ──────────────────────────────────────────────────────────────
def test_gate_rows_sum_to_one():
    m = _synthetic_master()
    w, order = regimes.compute_gate(m)
    assert w.shape == (m.height, 3)
    assert order == list(config.REGIMES)
    assert np.allclose(w.sum(axis=1), 1.0, atol=1e-6)
    assert (w >= 0).all() and (w <= 1).all()


def test_gate_rejects_missing_column():
    m = _synthetic_master().drop("yanggan_days")
    with pytest.raises(KeyError):
        regimes.compute_gate(m)


# ── 전문가 / I ──────────────────────────────────────────────────────────
def test_ignition_in_unit_range():
    m = _synthetic_master()
    w, order = regimes.compute_gate(m)
    I, per = experts.ignition_propensity(m, w, order)
    assert I.shape[0] == m.height
    assert I.min() >= 0.0 and I.max() <= 1.0
    assert set(per.keys()) == set(config.REGIMES)


def test_ignition_monotonic_in_proximity():
    """발화원에 더 가까울수록(거리↓) I 가 단조 증가해야 한다."""
    rng = np.random.default_rng(1)
    n = 500
    base = dict(
        pole_id=np.arange(n), lon=np.full(n, 128.0), lat=np.full(n, 37.5),
        elevation=np.full(n, 300.0), yanggan_days=np.full(n, 1.0),
        sgg=np.array(["춘천"] * n),
        dist_to_road=np.full(n, 100.0), dist_to_powerline=np.full(n, 1000.0),
        fwi_q90=np.full(n, 6.0), fwi_high_danger_days=np.full(n, 0.2),
        mu_flammability=np.full(n, 0.4), mu_southness=np.full(n, 0.5),
        ndvi=np.full(n, 0.1), S_p=np.full(n, 0.5),
        nn_station_km=np.full(n, 5.0), extrap_flag=np.zeros(n, bool),
    )
    base["dist_to_forest"] = np.linspace(2000, 0, n)  # 거리 감소
    m = pl.DataFrame(base)
    feats = experts.build_ignition_features(m)
    sc = experts.expert_score(feats, config.REGIME_YEONGSEO)
    assert np.all(np.diff(sc) >= -1e-9), "거리 감소 시 위험 비감소여야"


# ── hazard ──────────────────────────────────────────────────────────────
def test_hazard_multiplicative_and_bounds():
    rng = np.random.default_rng(2)
    I = rng.uniform(0, 1, 100)
    S = rng.uniform(0, 1, 100)
    W = rng.uniform(0, 1, 100)
    h = hazard.daily_hazard(I, S, W)
    assert np.allclose(h, I * S * W)
    assert h.min() >= 0 and h.max() <= 1


def test_season_risk_geomean_shape():
    rng = np.random.default_rng(3)
    I = rng.uniform(0.1, 1, 50)
    S = rng.uniform(0.1, 1, 50)
    W_daily = rng.uniform(0.1, 1, (7, 50))
    R = hazard.season_risk(I, S, W_daily)
    assert R.shape == (50,)
    assert R.min() >= 0 and R.max() <= 1


# ── exposure 폴백 ────────────────────────────────────────────────────────
def test_exposure_fallback_shape_and_range():
    # CONTRACT: pole_xy 는 평면 근사 km. 작은 km 영역(0~3km)에 조밀 배치 →
    # max_dist_km=5 안에서 발화원 주변이 닿는다.
    rng = np.random.default_rng(4)
    n = 300
    pole_xy = np.column_stack([rng.uniform(0, 3, n), rng.uniform(0, 3, n)])
    ign = np.array([0, 1, 2, 3, 4], dtype=np.uint32)
    p = exposure.simulate_exposure(
        pole_xy, ign,
        wind_dir_deg=np.array([270.0, 315.0]),
        wind_speed=np.array([5.0, 8.0]),
        fuel=np.full(n, 1.0),       # 연료 최대 → 발화원 인근 닿음 보장
        southness=rng.uniform(-1, 1, n),
    )
    assert p.shape == (n,)
    assert p.min() >= 0 and p.max() <= 1
    # 발화원 인근(연료 1·근거리)은 노출 발생.
    assert p.max() > 0


def test_exposure_deterministic():
    rng = np.random.default_rng(5)
    n = 200
    pole_xy = np.column_stack([rng.uniform(0, 2, n), rng.uniform(0, 2, n)])  # km
    ign = np.array([0, 1], dtype=np.uint32)
    kw = dict(wind_dir_deg=np.array([270.0]), wind_speed=np.array([6.0]),
              fuel=rng.uniform(0, 1, n), southness=rng.uniform(-1, 1, n))
    p1 = exposure.simulate_exposure(pole_xy, ign, **kw)
    p2 = exposure.simulate_exposure(pole_xy, ign, **kw)
    assert np.array_equal(p1, p2)


# ── exposure 엔진(실연동) ─────────────────────────────────────────────────
def test_lonlat_to_km_scale_and_shape():
    """lon/lat → km 변환: 형상 보존 + 1도≈수십~111km 스케일."""
    ll = np.array([[128.0, 38.0], [128.0, 38.1], [128.1, 38.0]])
    km = exposure_engine.lonlat_to_km(ll)
    assert km.shape == ll.shape
    # 위도 0.1도 ≈ 11.1km
    dy = km[1, 1] - km[0, 1]
    assert 10.0 < dy < 12.0
    # 경도 0.1도(위도37.7 보정) ≈ 8.8km < 위도분
    dx = km[2, 0] - km[0, 0]
    assert 7.0 < dx < 10.0


def test_ignition_candidates_top_quantile():
    """발화후보 = I 상위 분위(라벨 미사용). 상한·결정성 확인."""
    rng = np.random.default_rng(0)
    n = 50_000
    I = rng.uniform(0, 1, n)
    gate = np.full((n, 3), 1 / 3)
    cand = exposure_engine.ignition_candidates(
        I, gate, list(config.REGIMES), top_q=0.02, max_n=8000)
    assert cand.dtype == np.uint32
    assert cand.size <= 8000
    # 후보는 모두 상위 분위(분위 임계 이상).
    thr = np.quantile(I, 0.98)
    assert (I[cand] >= thr - 1e-9).all()
    # 결정성.
    cand2 = exposure_engine.ignition_candidates(
        I, gate, list(config.REGIMES), top_q=0.02, max_n=8000)
    assert np.array_equal(cand, cand2)


def test_sample_wind_yeongdong_westerly_prior():
    """영동 양간지풍 prior: 풍향이 서풍(≈270°) 근방에 모인다."""
    wd, ws = exposure_engine.sample_wind(
        config.REGIME_YEONGDONG, n_sims=512, station_daily=None, aws_daily=None)
    assert wd.shape == (512,) and ws.shape == (512,)
    assert (wd >= 0).all() and (wd <= 360).all()
    # 270±std 안에 대부분 들어와야(서풍 prior).
    near_west = np.abs(((wd - 270.0 + 180) % 360) - 180) <= 3 * exposure_engine.YANGGAN_DIR_STD_DEG
    assert near_west.mean() > 0.95
    assert (ws > 0).all()


def test_exposure_downwind_eastward_under_westerly():
    """서풍(양간) 하 비등방 신장: 같은 거리의 동(풍하) 전주가 서(풍상)보다 노출↑.

    발화원 좌우 동일 거리(중·원거리)에 전주를 두고 서풍(270° from→동쪽 풍하)으로
    exposure → 동쪽 전주의 노출확률이 서쪽보다 높아야(L 신장으로 풍하 도달↑).
    실제 2019 고성 흉터(W→E ~6.7km)와 같은 방향성. 단거리(L0 내)는 거의 등방이라
    풍하 신장이 분명히 나타나는 중거리(≈2~3km)에서 비교.
    """
    cos = np.cos(np.deg2rad(config.LAT0_DEG))
    lon0, lat0 = 128.50, 38.21
    # 발화원(idx0) + 동/서 대칭쌍 여러 거리(km).
    dists_km = [1.0, 2.0, 3.0, 4.0]
    rows = [[lon0, lat0]]
    for dk in dists_km:
        dlon = dk / (cos * (np.pi / 180.0) * config.EARTH_R_KM)
        rows.append([lon0 + dlon, lat0])  # 동(풍하)
        rows.append([lon0 - dlon, lat0])  # 서(풍상)
    pole_ll = np.array(rows)
    pole_km = exposure_engine.lonlat_to_km(pole_ll)
    n = pole_ll.shape[0]
    p = exposure.simulate_exposure(
        pole_km, np.array([0], np.uint32),
        wind_dir_deg=np.full(256, 270.0),  # 서풍 from→동 풍하
        wind_speed=np.full(256, 12.0),
        fuel=np.full(n, 1.0), southness=np.zeros(n),
    )
    east = p[1::2]   # 동쪽 전주들
    west = p[2::2]   # 서쪽 전주들
    # 중·원거리 전체 합에서 풍하(동) 노출이 풍상(서)보다 커야.
    assert east.sum() > west.sum(), f"풍하(동) {east.sum():.2f} > 풍상(서) {west.sum():.2f}"


# ── 보정 ────────────────────────────────────────────────────────────────
def test_isotonic_monotone():
    rng = np.random.default_rng(6)
    x = rng.uniform(0, 1, 500)
    y = (x + rng.normal(0, 0.3, 500) > 0.5).astype(float)
    xs, yiso = calibrate.isotonic_fit(x, y)
    assert np.all(np.diff(yiso) >= -1e-9)
    assert xs.shape == yiso.shape


def test_decide_threshold_prevalence():
    rng = np.random.default_rng(7)
    risk = rng.uniform(0, 1, 10000)
    thr, dec = calibrate.decide_threshold(risk, 0.05)
    assert set(np.unique(dec)).issubset({0, 1})
    assert abs(dec.mean() - 0.05) < 0.01


def test_decide_threshold_robust_to_ties():
    """동점이 많은 점수에서도 예측양성이 목표비율을 초과하지 않아야(퇴행 방지)."""
    risk = np.zeros(10000)
    risk[:300] = np.linspace(1.0, 0.5, 300)  # 상위 300만 구분, 나머지는 동점 0
    _, dec = calibrate.decide_threshold(risk, 0.02)  # 목표 200개
    # ties 가 있어도 정확히 200개(value-threshold 였다면 9700+ 가 양성).
    assert dec.sum() == 200


def test_recall_at_topk_perfect():
    risk = np.linspace(0, 1, 1000)
    pos = np.array([999, 998, 997])  # 최상위
    rec = calibrate.recall_at_topk(risk, pos, (0.01,))
    assert rec[0.01] == 1.0


# ── 제출 스키마 ──────────────────────────────────────────────────────────
def test_submission_schema_validation():
    n = submit.N_POLES_EXPECTED
    pid = np.arange(n, dtype=np.int64)
    df = submit.build_submission(
        pole_id=pid, lon=np.zeros(n), lat=np.zeros(n),
        decision=np.zeros(n, dtype=np.int8), risk_score=np.zeros(n),
    )
    assert df.height == n
    assert set(df["decision"].unique().to_list()).issubset({0, 1})
    for c in submit.REQUIRED_COLS:
        assert c in df.columns


def test_submission_rejects_bad_decision():
    n = submit.N_POLES_EXPECTED
    dec = np.zeros(n, dtype=np.int8)
    dec[0] = 2  # 불법
    with pytest.raises(ValueError):
        submit.build_submission(
            pole_id=np.arange(n), lon=np.zeros(n), lat=np.zeros(n),
            decision=dec, risk_score=np.zeros(n),
        )


def test_submission_rejects_wrong_rowcount():
    n = 100
    with pytest.raises(ValueError):
        submit.build_submission(
            pole_id=np.arange(n), lon=np.zeros(n), lat=np.zeros(n),
            decision=np.zeros(n, dtype=np.int8), risk_score=np.zeros(n),
        )


def test_submission_rejects_unsorted_pole_id():
    """pole_id 정렬이 깨지면 거부(제출 무결성 게이트)."""
    n = submit.N_POLES_EXPECTED
    pid = np.arange(n, dtype=np.int64)
    pid[0], pid[1] = pid[1], pid[0]  # 정렬 위반
    with pytest.raises(ValueError):
        submit.build_submission(
            pole_id=pid, lon=np.zeros(n), lat=np.zeros(n),
            decision=np.zeros(n, dtype=np.int8), risk_score=np.zeros(n),
        )


def test_submission_rejects_duplicate_pole_id():
    """pole_id 중복이면 거부(제출 무결성 게이트)."""
    n = submit.N_POLES_EXPECTED
    pid = np.arange(n, dtype=np.int64)
    pid[-1] = pid[-2]  # 중복(정렬은 유지)
    with pytest.raises(ValueError):
        submit.build_submission(
            pole_id=pid, lon=np.zeros(n), lat=np.zeros(n),
            decision=np.zeros(n, dtype=np.int8), risk_score=np.zeros(n),
        )


def test_submission_n_poles_matches_config():
    """제출 행수 불변이 config 단일 진실과 일치(중복 정의 방지)."""
    assert submit.N_POLES_EXPECTED == config.N_POLES_EXPECTED


# ── 좌표 변환(geo 단일 구현) ───────────────────────────────────────────────
def test_geo_lonlat_to_km_matches_engine_wrapper():
    """geo.lonlat_to_km == exposure_engine.lonlat_to_km(래퍼). 비트 동일."""
    rng = np.random.default_rng(11)
    ll = np.column_stack([rng.uniform(127.5, 129.4, 1000),
                          rng.uniform(37.0, 38.3, 1000)])
    a = geo.lonlat_to_km(ll)
    b = exposure_engine.lonlat_to_km(ll)
    assert np.array_equal(a, b)


def test_geo_point_matches_vector():
    """point_to_km(스칼라) 가 lonlat_to_km(벡터) 와 비트 동일."""
    lon, lat = 128.5, 38.21
    vx, vy = geo.lonlat_to_km(np.array([[lon, lat]]))[0]
    px, py = geo.point_to_km(lon, lat)
    assert px == vx and py == vy


def test_geo_latitude_scale_km():
    """위도 0.1도 ≈ 11.1km, 경도 0.1도(37.7 보정) < 위도분."""
    km = geo.lonlat_to_km(np.array([[128.0, 38.0], [128.0, 38.1], [128.1, 38.0]]))
    dy = km[1, 1] - km[0, 1]
    dx = km[2, 0] - km[0, 0]
    assert 10.0 < dy < 12.0
    assert 7.0 < dx < 10.0
    assert dx < dy  # cos 보정으로 경도분이 더 짧다


# ── 바람 표집 결정성(PYTHONHASHSEED 회귀 방지) ─────────────────────────────
def test_sample_wind_deterministic_across_calls():
    """같은 seed → 같은 풍향/풍속. (과거 hash(regime) 가 프로세스마다 salt 돼

    실행마다 결과가 달라지던 결정성 위반을 막는다 — config.REGIMES 인덱스 기반.)
    """
    for regime in config.REGIMES:
        wd1, ws1 = exposure_engine.sample_wind(regime, n_sims=128, seed=config.SEED)
        wd2, ws2 = exposure_engine.sample_wind(regime, n_sims=128, seed=config.SEED)
        assert np.array_equal(wd1, wd2)
        assert np.array_equal(ws1, ws2)


def test_sample_wind_distinct_per_regime_seed():
    """체제마다 다른 시드 오프셋(서로 다른 표집)."""
    offsets = {r: exposure_engine._regime_seed_offset(r) for r in config.REGIMES}
    assert len(set(offsets.values())) == len(config.REGIMES)
    # 미등록 체제는 0 으로 폴백.
    assert exposure_engine._regime_seed_offset("__unknown__") == 0


# ── hazard: 블렌드/검증 ───────────────────────────────────────────────────
def test_blend_exposure_alpha0_returns_static():
    """alpha=0(의도된 결정): S_static 그대로(노출은 결정에 미반영)."""
    rng = np.random.default_rng(12)
    s = rng.uniform(0, 1, 200)
    pe = rng.uniform(0, 1, 200)
    out = hazard.blend_exposure(s, pe, alpha=0.0)
    assert np.allclose(out, s)


def test_blend_exposure_alpha1_returns_exposure():
    rng = np.random.default_rng(13)
    s = rng.uniform(0, 1, 200)
    pe = rng.uniform(0, 1, 200)
    out = hazard.blend_exposure(s, pe, alpha=1.0)
    assert np.allclose(out, pe)


def test_blend_exposure_rejects_bad_alpha():
    s = np.full(10, 0.5)
    pe = np.full(10, 0.5)
    with pytest.raises(ValueError):
        hazard.blend_exposure(s, pe, alpha=1.5)


def test_daily_hazard_rejects_nan_and_out_of_range():
    good = np.full(5, 0.5)
    with pytest.raises(ValueError):
        hazard.daily_hazard(good, good, np.array([0.5, np.nan, 0.5, 0.5, 0.5]))
    with pytest.raises(ValueError):
        hazard.daily_hazard(good, good, np.array([0.5, 1.5, 0.5, 0.5, 0.5]))


def test_season_risk_single_w_equals_product():
    """단일 W(MVP) 경로는 R == I*S*W."""
    rng = np.random.default_rng(14)
    I = rng.uniform(0, 1, 100)
    S = rng.uniform(0, 1, 100)
    W = rng.uniform(0, 1, 100)
    R = hazard.season_risk(I, S, W)
    assert np.allclose(R, I * S * W)


# ── 보정: 폴백·단조 ───────────────────────────────────────────────────────
def test_calibrate_probability_empty_positives_minmax():
    """양성 0 → min-max 폴백(범위 [0,1], 단조 보존)."""
    rng = np.random.default_rng(15)
    risk = rng.uniform(0, 1, 1000)
    p = calibrate.calibrate_probability(risk, np.array([], dtype=np.int64))
    assert p.min() >= 0.0 and p.max() <= 1.0
    # min-max 는 순위를 보존: risk argsort 와 p argsort 동일.
    assert np.array_equal(np.argsort(risk), np.argsort(p))


def test_calibrate_probability_monotone_in_risk():
    """보정확률은 위험에 대해 단조 비감소(isotonic)."""
    rng = np.random.default_rng(16)
    n = 5000
    risk = rng.uniform(0, 1, n)
    # 고위험일수록 양성일 확률↑ 인 양성 집합.
    pos = np.nonzero(rng.uniform(0, 1, n) < risk * 0.05)[0]
    p = calibrate.calibrate_probability(risk, pos)
    order = np.argsort(risk)
    assert np.all(np.diff(p[order]) >= -1e-9)


# ── validate: 공간 블록 ───────────────────────────────────────────────────
def test_spatial_blocks_assigns_nearby_together():
    """같은 좌표 전주는 같은 블록, 먼 전주는 다른 블록."""
    pole_xy = np.array([
        [128.00, 38.00], [128.001, 38.001],  # 거의 같은 위치
        [128.50, 38.40],                       # ~40km 이상 떨어짐
    ])
    blocks = validate.spatial_blocks(pole_xy, block_km=10.0)
    assert blocks.shape == (3,)
    assert blocks[0] == blocks[1]
    assert blocks[2] != blocks[0]


def test_assign_poles_to_fires_radius():
    """반경 내 발화점만 매핑, 밖이면 -1."""
    pole_xy = np.array([[128.0, 38.0], [128.1, 38.1]])
    fire_xy = np.array([[128.0, 38.0], [129.3, 37.0]])  # 1개는 근접, 1개는 먼 곳
    mapped = calibrate.assign_poles_to_fires(pole_xy, fire_xy, radius_km=1.0)
    assert mapped[0] == 0      # 정확히 첫 전주
    assert mapped[1] == -1     # 반경 밖
