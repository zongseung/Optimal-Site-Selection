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
    weather,
)


def _synthetic_master(n: int = 2000, seed: int = 0) -> pl.DataFrame:
    """게이트·전문가·기상에 필요한 컬럼을 가진 합성 마스터."""
    rng = np.random.default_rng(seed)
    sgg_pool = list(config.SGG_TO_REGIME.keys())
    lc_pool = list(config.LC_IGNITION.keys())
    return pl.DataFrame({
        "pole_id": np.arange(n, dtype=np.int64),
        "lon": rng.uniform(127.5, 129.4, n),
        "lat": rng.uniform(37.0, 38.3, n),
        "elevation": rng.uniform(0, 1400, n),
        "yanggan_days": rng.uniform(0, 9, n),
        "sgg": rng.choice(sgg_pool, n),
        "lc_group": rng.choice(lc_pool, n),
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
        lc_group=np.array(["산림"] * n),  # 토지피복 상수 → forest 거리 단조성만 검증
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


def test_spatial_cv_recall_subset_mask_restricts_eval():
    """subset_mask 가 주어지면 그 부분집합 전주만으로 recall 측정(영동 부분집합용)."""
    rng = np.random.default_rng(20)
    n = 4000
    pole_xy = np.column_stack([rng.uniform(127.5, 129.4, n),
                               rng.uniform(37.0, 38.3, n)])
    risk = rng.uniform(0, 1, n)
    blocks = validate.spatial_blocks(pole_xy)
    f2p = rng.choice(n, size=30, replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[: n // 2] = True
    full = validate.spatial_cv_recall(risk, f2p, blocks, ks=(0.05,))
    sub = validate.spatial_cv_recall(risk, f2p, blocks, ks=(0.05,), subset_mask=mask)
    # 둘 다 정상 딕셔너리(키·필드)이고 [0,1] 범위.
    for d in (full, sub):
        assert set(d.keys()) == {0.05}
        m = d[0.05]["mean"]
        assert np.isnan(m) or (0.0 <= m <= 1.0)


# ── Phase3 ①: 토지피복 발화신호(lc_ignition) ────────────────────────────────
def test_lc_ignition_maps_to_config_priors():
    """lc_group → lc_ignition 매핑이 config.LC_IGNITION 사전치와 정확히 일치."""
    groups = ["산림", "초지", "나지", "농업", "시가화", "습지", "수역"]
    lc = experts.landcover_ignition(groups)
    expected = np.array([config.LC_IGNITION[g] for g in groups])
    assert np.array_equal(lc, expected)
    # 산림 최고(1.0), 수역 최저(0.0).
    assert lc.max() == 1.0 and lc.min() == 0.0
    assert lc[0] == 1.0  # 산림


def test_lc_ignition_unknown_uses_default():
    """미지/결측 lc_group 은 config.LC_IGNITION_DEFAULT 로 폴백."""
    lc = experts.landcover_ignition(["산림", "기타미지", None])
    assert lc[0] == config.LC_IGNITION["산림"]
    assert lc[1] == config.LC_IGNITION_DEFAULT
    assert lc[2] == config.LC_IGNITION_DEFAULT


def test_build_features_includes_landcover_in_unit_range():
    """build_ignition_features 가 landcover 성분을 [0,1] 로 포함(7성분)."""
    m = _synthetic_master()
    feats = experts.build_ignition_features(m)
    assert "landcover" in feats
    assert set(feats.keys()) == {"forest", "road", "powerline", "fwi",
                                 "yanggan", "fuel", "landcover"}
    lc = feats["landcover"]
    assert lc.shape[0] == m.height
    assert lc.min() >= 0.0 and lc.max() <= 1.0


def test_build_features_requires_lc_group():
    """lc_group 누락 시 명시적 KeyError(silent 금지)."""
    m = _synthetic_master().drop("lc_group")
    with pytest.raises(KeyError):
        experts.build_ignition_features(m)


def test_expert_weights_have_landcover_and_sum_one():
    """config.EXPERT_WEIGHTS 각 체제가 landcover 키를 갖고 simplex 합≈1.

    값은 튜너가 4자리 반올림한 simplex(step=0.05) 라 합이 정확히 1 이 아닐 수
    있다(expert_score 가 wsum 으로 정규화하므로 무해). 반올림 오차 허용.
    """
    for r in config.REGIMES:
        w = config.EXPERT_WEIGHTS[r]
        assert "landcover" in w
        assert abs(sum(w.values()) - 1.0) < 1e-3  # 4자리 반올림 오차 허용


# ── Phase3 ②: 조건부 풍하노출(conditional alpha / blend) ────────────────────
def _gate_3regime(n: int, yd_frac: float, rng) -> tuple[np.ndarray, list[str]]:
    """영동 우세 yd_frac 비율, 나머지는 영서/산간 우세인 합성 게이트(행합=1)."""
    order = list(config.REGIMES)
    yd_col = order.index(config.REGIME_YEONGDONG)
    gate = rng.uniform(0, 0.2, (n, 3))
    is_yd = rng.uniform(0, 1, n) < yd_frac
    # 영동 전주는 영동열을 크게, 비영동은 다른 열을 크게.
    for i in range(n):
        if is_yd[i]:
            gate[i, yd_col] = 0.8
        else:
            other = [c for c in range(3) if c != yd_col]
            gate[i, rng.choice(other)] = 0.8
    gate = gate / gate.sum(axis=1, keepdims=True)
    return gate, order


def test_conditional_alpha_range_and_zero_outside_yeongdong():
    """α_p ∈ [0, α_base]; 영동 게이트 0 또는 양간 0 인 전주는 α_p≈0(전역 희석 방지)."""
    rng = np.random.default_rng(21)
    n = 1000
    gate, order = _gate_3regime(n, yd_frac=0.3, rng=rng)
    yang = rng.uniform(0, 9, n)
    a_base = 0.5
    alpha_p = hazard.conditional_alpha(gate, order, yang, alpha_base=a_base,
                                       yeongdong_regime=config.REGIME_YEONGDONG)
    assert alpha_p.shape == (n,)
    assert alpha_p.min() >= 0.0 and alpha_p.max() <= a_base + 1e-9
    # 비영동(영동게이트 낮음)은 영동보다 평균 α_p 가 작아야(조건부 집중).
    yd_col = order.index(config.REGIME_YEONGDONG)
    yd = gate.argmax(axis=1) == yd_col
    assert alpha_p[yd].mean() > alpha_p[~yd].mean()


def test_conditional_alpha_zero_base_disables():
    """α_base=0 이면 모든 α_p=0(조건부노출 전역 비활성=정적 S 유지)."""
    rng = np.random.default_rng(22)
    n = 500
    gate, order = _gate_3regime(n, yd_frac=0.5, rng=rng)
    yang = rng.uniform(0, 9, n)
    alpha_p = hazard.conditional_alpha(gate, order, yang, alpha_base=0.0,
                                       yeongdong_regime=config.REGIME_YEONGDONG)
    assert np.allclose(alpha_p, 0.0)


def test_conditional_alpha_rejects_bad_base():
    rng = np.random.default_rng(23)
    gate, order = _gate_3regime(100, 0.3, rng)
    yang = rng.uniform(0, 9, 100)
    with pytest.raises(ValueError):
        hazard.conditional_alpha(gate, order, yang, alpha_base=1.5,
                                 yeongdong_regime=config.REGIME_YEONGDONG)


def test_blend_conditional_alpha0_returns_static():
    """α_p=0 전부면 blend_exposure_conditional 은 S_static 그대로."""
    rng = np.random.default_rng(24)
    n = 300
    s = rng.uniform(0, 1, n)
    pe = rng.uniform(0, 1, n)
    out = hazard.blend_exposure_conditional(s, pe, np.zeros(n))
    assert np.allclose(out, s)


def test_blend_conditional_moves_toward_exposure_rank():
    """α_p>0 인 전주는 S 가 노출 백분위 쪽으로 이동(조건부 증강 동작)."""
    n = 200
    s = np.full(n, 0.5)
    pe = np.linspace(0.0, 1.0, n)            # 노출 백분위 = 0..1 단조
    alpha_p = np.full(n, 0.5)
    out = hazard.blend_exposure_conditional(s, pe, alpha_p)
    # 노출 백분위 높은 전주(끝)는 0.5보다 커지고, 낮은 전주(앞)는 작아져야.
    assert out[-1] > 0.5 and out[0] < 0.5
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_blend_conditional_shape_mismatch_raises():
    s = np.full(10, 0.5)
    pe = np.full(10, 0.5)
    with pytest.raises(ValueError):
        hazard.blend_exposure_conditional(s, pe, np.zeros(9))


# ── Phase4 ①: 일별 ISI/FFMC 동역학 W ──────────────────────────────────────
def _synthetic_stations(k: int = 6) -> pl.DataFrame:
    """합성 관측소 좌표(강원 bbox 내 격자)."""
    lons = np.linspace(127.6, 129.3, k)
    lats = np.linspace(37.1, 38.2, k)
    return pl.DataFrame({
        "stnId": np.arange(100, 100 + k, dtype=np.int64),
        "lon": lons, "lat": lats,
    })


def _synthetic_station_daily(stations: pl.DataFrame, seed: int = 0) -> pl.DataFrame:
    """합성 관측소 일별 ISI/FFMC(산불조심기간+비시즌 혼재).

    관측소마다 ISI/FFMC 수준을 다르게 둬(앞쪽 관측소 = 고확산) 분별력 검증 가능.
    """
    rng = np.random.default_rng(seed)
    ids = stations["stnId"].to_list()
    rows_stn, rows_month, rows_year, rows_isi, rows_ffmc = [], [], [], [], []
    for rank, sid in enumerate(ids):
        level = 1.0 - rank / max(1, len(ids) - 1)   # 0..1, 앞 관측소가 높음
        for year in (2020, 2021):
            for month in range(1, 13):
                for _ in range(20):
                    rows_stn.append(int(sid))
                    rows_month.append(month)
                    rows_year.append(year)
                    rows_isi.append(float(np.clip(rng.normal(2 + 12 * level, 3), 0, 40)))
                    rows_ffmc.append(float(np.clip(rng.normal(80 + 15 * level, 5), 0, 99)))
    return pl.DataFrame({
        "stn": rows_stn, "month": rows_month, "year": rows_year,
        "isi": rows_isi, "ffmc": rows_ffmc,
    })


def test_daily_isi_weather_shape_and_range():
    """일별 ISI/FFMC W: 형상 보존 + [0,1] + NaN 없음."""
    m = _synthetic_master(n=1500, seed=3)
    st = _synthetic_stations()
    sd = _synthetic_station_daily(st)
    W = weather.daily_isi_weather(m, sd, st)
    assert W.shape == (m.height,)
    assert W.min() >= 0.0 and W.max() <= 1.0
    assert not np.isnan(W).any()


def test_daily_isi_weather_deterministic():
    """같은 입력 → 같은 W(결정성)."""
    m = _synthetic_master(n=800, seed=4)
    st = _synthetic_stations()
    sd = _synthetic_station_daily(st)
    W1 = weather.daily_isi_weather(m, sd, st)
    W2 = weather.daily_isi_weather(m, sd, st)
    assert np.array_equal(W1, W2)


def test_per_station_isi_ffmc_seasonal_only():
    """집계는 산불조심기간(2~5월)만 사용 — 비시즌 행 추가가 결과를 바꾸지 않아야."""
    st = _synthetic_stations(k=4)
    sd = _synthetic_station_daily(st, seed=7)
    season_only = sd.filter(pl.col("month").is_in(list(config.SEASON_MONTHS)))
    a = weather._per_station_isi_ffmc(sd).sort("stn")
    b = weather._per_station_isi_ffmc(season_only).sort("stn")
    for c in ("isi_hi_frac", "isi_mean", "isi_q90", "ffmc_hi_frac"):
        assert np.allclose(a[c].to_numpy(), b[c].to_numpy())


def test_per_station_isi_ffmc_higher_for_high_isi_station():
    """ISI 수준이 높은 관측소가 isi_hi_frac·isi_mean 도 높아야(분별력)."""
    st = _synthetic_stations(k=5)
    sd = _synthetic_station_daily(st, seed=9)
    agg = weather._per_station_isi_ffmc(sd).sort("stn")
    isi_mean = agg["isi_mean"].to_numpy()
    # 첫 관측소(level=1, 고확산)가 마지막(level=0, 저확산)보다 평균 ISI 높음.
    assert isi_mean[0] > isi_mean[-1]


def test_daily_isi_weather_requires_yanggan():
    """yanggan_days 누락 시 명시적 ValueError(silent 금지)."""
    m = _synthetic_master(n=300).drop("yanggan_days")
    st = _synthetic_stations()
    sd = _synthetic_station_daily(st)
    with pytest.raises(ValueError):
        weather.daily_isi_weather(m, sd, st)


def test_per_station_isi_ffmc_rejects_missing_isi():
    """station_daily 에 isi 컬럼 없으면 ValueError."""
    st = _synthetic_stations(k=3)
    sd = _synthetic_station_daily(st).drop("isi")
    with pytest.raises(ValueError):
        weather._per_station_isi_ffmc(sd)


def test_w_config_weights_sum_to_one():
    """W 결합/ISI 내부 가중이 simplex(합≈1)."""
    assert abs(config.W_ISI_COMPONENT_WEIGHT + config.W_FFMC_COMPONENT_WEIGHT
               + config.W_YANGGAN_COMPONENT_WEIGHT - 1.0) < 1e-9
    assert abs(config.W_ISI_FREQ_WEIGHT + config.W_ISI_MEAN_WEIGHT
               + config.W_ISI_Q90_WEIGHT - 1.0) < 1e-9


def test_w_blend_season_weight_in_unit_range():
    """결합 W 의 시즌 가중이 [0,1] (나머지는 일별 성분)."""
    assert 0.0 <= config.W_BLEND_SEASON_WEIGHT <= 1.0


# ── Phase4 ①: 결합(블렌드) W ────────────────────────────────────────────────
def test_blended_weather_shape_and_range():
    """결합 W: 형상 보존 + [0,1] + NaN 없음."""
    m = _synthetic_master(n=1500, seed=5)
    st = _synthetic_stations()
    sd = _synthetic_station_daily(st)
    W = weather.blended_weather(m, sd, st)
    assert W.shape == (m.height,)
    assert W.min() >= 0.0 and W.max() <= 1.0
    assert not np.isnan(W).any()


def test_blended_weather_is_convex_mix_of_components():
    """결합 W = w·season + (1−w)·daily (성분의 볼록결합)."""
    m = _synthetic_master(n=1200, seed=6)
    st = _synthetic_stations()
    sd = _synthetic_station_daily(st)
    w = 0.7
    Ws = weather.season_weather(m, source="features")
    Wd = weather.daily_isi_weather(m, sd, st)
    Wb = weather.blended_weather(m, sd, st, season_weight=w)
    assert np.allclose(Wb, np.clip(w * Ws + (1 - w) * Wd, 0.0, 1.0))


def test_blended_weather_endpoints_match_components():
    """season_weight=1 → season W, =0 → daily W (끝점 일치)."""
    m = _synthetic_master(n=900, seed=7)
    st = _synthetic_stations()
    sd = _synthetic_station_daily(st)
    Ws = weather.season_weather(m, source="features")
    Wd = weather.daily_isi_weather(m, sd, st)
    assert np.allclose(weather.blended_weather(m, sd, st, season_weight=1.0), Ws)
    assert np.allclose(weather.blended_weather(m, sd, st, season_weight=0.0), Wd)


def test_blended_weather_deterministic():
    """같은 입력 → 같은 결합 W."""
    m = _synthetic_master(n=800, seed=8)
    st = _synthetic_stations()
    sd = _synthetic_station_daily(st)
    assert np.array_equal(weather.blended_weather(m, sd, st),
                          weather.blended_weather(m, sd, st))


def test_blended_weather_rejects_bad_weight():
    """season_weight 가 [0,1] 밖이면 ValueError."""
    m = _synthetic_master(n=300, seed=9)
    st = _synthetic_stations()
    sd = _synthetic_station_daily(st)
    with pytest.raises(ValueError):
        weather.blended_weather(m, sd, st, season_weight=1.5)


def test_compare_weather_recall_returns_gap():
    """validate.compare_weather_recall: 두 위험맵의 recall·gap 딕셔너리 반환."""
    rng = np.random.default_rng(31)
    n = 4000
    pole_xy = np.column_stack([rng.uniform(127.5, 129.4, n),
                               rng.uniform(37.0, 38.3, n)])
    blocks = validate.spatial_blocks(pole_xy)
    f2p = rng.choice(n, size=40, replace=False)
    R_a = rng.uniform(0, 1, n)
    R_b = rng.uniform(0, 1, n)
    out = validate.compare_weather_recall(R_a, R_b, f2p, blocks, ks=(0.05,))
    assert set(out.keys()) == {0.05}
    d = out[0.05]
    assert set(d.keys()) == {"recall_a", "recall_b", "gap"}
    assert np.isclose(d["gap"], d["recall_b"] - d["recall_a"])


# ── Phase4 ②: 누수안전 계층 EB 배율 공간CV ─────────────────────────────────
def _synthetic_master_with_grid(n: int = 4000, seed: int = 0) -> pl.DataFrame:
    """sgg·grid_id 를 가진 합성 마스터(배율 CV 용)."""
    rng = np.random.default_rng(seed)
    return pl.DataFrame({
        "lon": rng.uniform(127.5, 129.4, n),
        "lat": rng.uniform(37.0, 38.3, n),
        "sgg": rng.choice(list(config.SGG_TO_REGIME.keys()), n),
        "grid_id": rng.choice([f"g{i}" for i in range(40)], n),
    })


def test_spatial_cv_recall_multiplier_keys_and_range():
    """누수안전 배율 CV: k 키·[0,1]/NaN 범위·딕셔너리 필드."""
    rng = np.random.default_rng(40)
    m = _synthetic_master_with_grid(n=4000, seed=40)
    pole_xy = m.select(["lon", "lat"]).to_numpy().astype(np.float64)
    blocks = validate.spatial_blocks(pole_xy)
    base_risk = rng.uniform(0, 1, m.height)
    f2p = rng.choice(m.height, size=40, replace=False)
    out = validate.spatial_cv_recall_multiplier(
        base_risk, m, f2p, blocks, ks=(0.05, 0.10))
    assert set(out.keys()) == {0.05, 0.10}
    for k in out:
        mean = out[k]["mean"]
        assert set(out[k].keys()) == {"mean", "std", "n_folds_with_pos"}
        assert np.isnan(mean) or (0.0 <= mean <= 1.0)


def test_spatial_cv_recall_multiplier_deterministic():
    """같은 입력 → 같은 누수안전 배율 CV(결정성)."""
    rng = np.random.default_rng(41)
    m = _synthetic_master_with_grid(n=3000, seed=41)
    pole_xy = m.select(["lon", "lat"]).to_numpy().astype(np.float64)
    blocks = validate.spatial_blocks(pole_xy)
    base_risk = rng.uniform(0, 1, m.height)
    f2p = rng.choice(m.height, size=30, replace=False)
    a = validate.spatial_cv_recall_multiplier(base_risk, m, f2p, blocks, ks=(0.05,))
    b = validate.spatial_cv_recall_multiplier(base_risk, m, f2p, blocks, ks=(0.05,))
    assert a[0.05]["mean"] == b[0.05]["mean"] or (
        np.isnan(a[0.05]["mean"]) and np.isnan(b[0.05]["mean"]))


def test_fold_assignment_matches_spatial_cv_partition():
    """_fold_assignment 가 spatial_cv_recall 내부 폴드 분할과 동일(누수안전 CV 일관성)."""
    rng = np.random.default_rng(42)
    n = 2000
    pole_xy = np.column_stack([rng.uniform(127.5, 129.4, n),
                               rng.uniform(37.0, 38.3, n)])
    blocks = validate.spatial_blocks(pole_xy)
    fold = validate._fold_assignment(blocks, n_folds=5)
    assert fold.shape == (n,)
    assert set(np.unique(fold)).issubset(set(range(5)))
    # 같은 블록은 같은 폴드(블록 단위 그룹 분할).
    for b in np.unique(blocks):
        assert np.unique(fold[blocks == b]).size == 1


# ── 체제별(per-regime) 임계값 / 배분 ───────────────────────────────────────
def _regime_labels(n: int, seed: int = 0) -> np.ndarray:
    """전주별 우세 체제 라벨(영동 편중을 흉내내는 합성)."""
    rng = np.random.default_rng(seed)
    return rng.choice(list(config.REGIMES), n, p=[0.22, 0.48, 0.30]).astype("<U16")


def test_allocate_regime_budget_sums_to_total():
    """체제별 예산 합 == 전역 예산(전역 동일 예산 보장)."""
    labels = _regime_labels(100_000, seed=1)
    total = int(np.ceil(0.02 * labels.shape[0]))
    for mode in (calibrate.ALLOC_EQUAL, calibrate.ALLOC_COUNT):
        b = calibrate.allocate_regime_budget(labels, total, mode)
        assert sum(b.values()) == total
        assert set(b.keys()) == set(config.REGIMES)
    # anchor 모드: 발화수+1 평활 → 합도 정확히 total.
    pos = np.random.default_rng(2).choice(labels.shape[0], 200, replace=False)
    ac = calibrate.regime_anchor_count(labels, pos)
    b = calibrate.allocate_regime_budget(labels, total, calibrate.ALLOC_ANCHOR, ac)
    assert sum(b.values()) == total


def test_allocate_regime_equal_equals_count():
    """⒝ 체제별 동일비율 == ⒞ 체제별 전주수 비례(수학적 동치)."""
    labels = _regime_labels(50_000, seed=3)
    total = int(np.ceil(0.03 * labels.shape[0]))
    be = calibrate.allocate_regime_budget(labels, total, calibrate.ALLOC_EQUAL)
    bc = calibrate.allocate_regime_budget(labels, total, calibrate.ALLOC_COUNT)
    assert be == bc


def test_allocate_budget_respects_cap():
    """체제 예산은 그 체제 전주수를 넘지 않는다(초과분은 다른 체제로 재배분)."""
    # 영동 전주가 극소수인데 예산이 크면, 영동 cap 초과분이 다른 체제로 간다.
    labels = np.array(["yeongdong"] * 5 + ["yeongseo"] * 9995, dtype="<U16")
    total = 100
    b = calibrate.allocate_regime_budget(labels, total, calibrate.ALLOC_COUNT)
    assert b["yeongdong"] <= 5
    assert sum(b.values()) == total


def test_decide_threshold_per_regime_sum_equals_budget():
    """체제별 컷 decision 합 == ceil(prevalence·N)(전역 예산 동일)."""
    rng = np.random.default_rng(4)
    n = 80_000
    labels = _regime_labels(n, seed=4)
    risk = rng.uniform(0, 1, n)
    risk[labels == "yeongdong"] += 0.5  # 영동 위험 쏠림 흉내
    for mode in (calibrate.ALLOC_EQUAL, calibrate.ALLOC_COUNT):
        _, dec = calibrate.decide_threshold_per_regime(risk, labels, 0.02, mode)
        assert dec.sum() == int(np.ceil(0.02 * n))
        assert set(np.unique(dec)).issubset({0, 1})


def test_decide_threshold_per_regime_monotone_within_regime():
    """체제별 컷: 같은 체제 안에서 decision=1 전주의 위험이 decision=0 전주 이상(단조)."""
    rng = np.random.default_rng(5)
    n = 60_000
    labels = _regime_labels(n, seed=5)
    risk = rng.uniform(0, 1, n)
    _, dec = calibrate.decide_threshold_per_regime(
        risk, labels, 0.05, calibrate.ALLOC_COUNT)
    for r in config.REGIMES:
        sel = labels == r
        if dec[sel].sum() > 0 and (dec[sel] == 0).sum() > 0:
            assert risk[sel][dec[sel] == 1].min() >= risk[sel][dec[sel] == 0].max() - 1e-9


def test_decide_threshold_per_regime_anchor_shifts_budget():
    """⒟ 앵커밀도: 발화 잦은 체제로 예산이 쏠린다(전주수비례 대비)."""
    n = 60_000
    labels = _regime_labels(n, seed=6)
    risk = np.random.default_rng(6).uniform(0, 1, n)
    total = int(np.ceil(0.03 * n))
    # 산간에만 발화 몰아주기 → anchor 가 산간 예산을 count 대비 늘려야.
    mtn_idx = np.nonzero(labels == "mountain")[0][:300]
    ac = calibrate.regime_anchor_count(labels, mtn_idx)
    b_cnt = calibrate.allocate_regime_budget(labels, total, calibrate.ALLOC_COUNT)
    b_anc = calibrate.allocate_regime_budget(labels, total, calibrate.ALLOC_ANCHOR, ac)
    assert b_anc["mountain"] > b_cnt["mountain"]


def test_decide_threshold_per_regime_rejects_bad_prevalence():
    labels = _regime_labels(1000)
    risk = np.random.default_rng(0).uniform(0, 1, 1000)
    with pytest.raises(ValueError):
        calibrate.decide_threshold_per_regime(risk, labels, 0.0, calibrate.ALLOC_COUNT)


def test_allocate_regime_budget_rejects_global_mode():
    """ALLOC_GLOBAL 은 체제배분 함수가 아니다(전역은 decide_threshold)."""
    labels = _regime_labels(1000)
    with pytest.raises(ValueError):
        calibrate.allocate_regime_budget(labels, 20, calibrate.ALLOC_GLOBAL)


# ── F1 민감도: 전역 vs 체제별 ──────────────────────────────────────────────
def test_f1_sensitivity_compare_shape():
    """f1_sensitivity_compare: π별 global/regime 항목과 필드 형상."""
    rng = np.random.default_rng(7)
    n = 40_000
    labels = _regime_labels(n, seed=7)
    risk = rng.uniform(0, 1, n)
    pos = rng.choice(n, 150, replace=False)
    out = calibrate.f1_sensitivity_compare(risk, pos, labels)
    assert set(out.keys()) == set(config.PREVALENCE_GRID)
    fields = {"predicted_positive", "recall", "precision_proxy", "f1_proxy"}
    for rho in config.PREVALENCE_GRID:
        assert set(out[rho].keys()) == {"global", "regime"}
        for side in ("global", "regime"):
            m = out[rho][side]
            assert fields.issubset(m.keys())
            assert 0.0 <= m["recall"] <= 1.0
            assert 0.0 <= m["f1_proxy"] <= 1.0
            # 같은 예산: global·regime 예측양성 동일(전역 동일 예산).
            assert m["predicted_positive"] == out[rho]["global"]["predicted_positive"]


def test_f1_sensitivity_compare_monotone_recall_in_prevalence():
    """π 가 커지면 (같은 배분 내) recall 은 비감소(예측양성 증가)."""
    rng = np.random.default_rng(8)
    n = 40_000
    labels = _regime_labels(n, seed=8)
    risk = rng.uniform(0, 1, n)
    pos = rng.choice(n, 200, replace=False)
    out = calibrate.f1_sensitivity_compare(risk, pos, labels)
    grid = sorted(config.PREVALENCE_GRID)
    for side in ("global", "regime"):
        recalls = [out[rho][side]["recall"] for rho in grid]
        assert all(recalls[i] <= recalls[i + 1] + 1e-9 for i in range(len(recalls) - 1))


# ── 배분방식별 공간CV recall (validate) ────────────────────────────────────
def test_spatial_cv_recall_allocation_shape_and_range():
    """배분 비교 CV: mode 키·k 키·[0,1]/NaN 범위."""
    rng = np.random.default_rng(9)
    n = 8000
    pole_xy = np.column_stack([rng.uniform(127.5, 129.4, n),
                               rng.uniform(37.0, 38.3, n)])
    labels = _regime_labels(n, seed=9)
    risk = rng.uniform(0, 1, n)
    blocks = validate.spatial_blocks(pole_xy)
    f2p = rng.choice(n, size=80, replace=False)
    modes = (calibrate.ALLOC_GLOBAL, calibrate.ALLOC_COUNT, calibrate.ALLOC_ANCHOR)
    out = validate.spatial_cv_recall_allocation(
        risk, f2p, labels, blocks, prevalence=0.05, modes=modes, ks=(0.05, 0.10))
    assert set(out.keys()) == set(modes)
    for mode in modes:
        assert set(out[mode].keys()) == {0.05, 0.10}
        for k in out[mode]:
            mean = out[mode][k]["mean"]
            assert set(out[mode][k].keys()) == {"mean", "std", "n_folds_with_pos"}
            assert np.isnan(mean) or (0.0 <= mean <= 1.0)


def test_spatial_cv_recall_allocation_deterministic():
    """같은 입력 → 같은 배분 비교 CV(결정성; anchor 누수안전 포함)."""
    rng = np.random.default_rng(10)
    n = 6000
    pole_xy = np.column_stack([rng.uniform(127.5, 129.4, n),
                               rng.uniform(37.0, 38.3, n)])
    labels = _regime_labels(n, seed=10)
    risk = rng.uniform(0, 1, n)
    blocks = validate.spatial_blocks(pole_xy)
    f2p = rng.choice(n, size=60, replace=False)
    a = validate.spatial_cv_recall_allocation(
        risk, f2p, labels, blocks, prevalence=0.05,
        modes=(calibrate.ALLOC_ANCHOR,), ks=(0.05,))
    b = validate.spatial_cv_recall_allocation(
        risk, f2p, labels, blocks, prevalence=0.05,
        modes=(calibrate.ALLOC_ANCHOR,), ks=(0.05,))
    ma, mb = a[calibrate.ALLOC_ANCHOR][0.05]["mean"], b[calibrate.ALLOC_ANCHOR][0.05]["mean"]
    assert ma == mb or (np.isnan(ma) and np.isnan(mb))


# ── Phase-5: 베이지안 사후분포 (posterior.py) ──────────────────────────────
from pfire import posterior  # noqa: E402


def _poles_with_zero_event() -> pl.DataFrame:
    """대표본 고발화 격자 + 소표본 zero-event 격자(사후 폭 비교용)."""
    rows = []
    pid = 0

    def add(sgg, grid, lon0, lat0, n):
        nonlocal pid
        for i in range(n):
            rows.append(dict(pole_id=pid, lon=lon0 + (i % 100) * 1e-4,
                             lat=lat0 + (i // 100) * 1e-4, sgg=sgg, grid_id=grid))
            pid += 1

    add("춘천", "g_hot", 127.70, 37.90, 120_000)    # 발화 다수 예정
    add("강릉", "g_yd", 128.90, 37.75, 80_000)      # 발화 다수 예정
    add("고성", "g_zero", 128.45, 38.30, 200)        # zero-event 소표본
    return pl.DataFrame(rows)


def _fires_hot(poles: pl.DataFrame, n: int = 150) -> pl.DataFrame:
    rows = []
    for grid in ("g_hot", "g_yd"):
        sub = poles.filter(pl.col("grid_id") == grid)
        bl, ba = sub["lon"][0], sub["lat"][0]
        for i in range(n):
            rows.append(dict(lon=bl + (i % 50) * 1e-5, lat=ba + (i // 50) * 1e-5))
    return pl.DataFrame(rows)


def test_regional_rate_posterior_shape_and_params():
    """사후 Gamma(α,β) 형상=N, 양수, mean 배율 전역평균≈1."""
    poles = _poles_with_zero_event()
    fires = _fires_hot(poles)
    post = posterior.regional_rate_posterior(poles, fires)
    assert post.alpha.shape == (poles.height,)
    assert post.beta.shape == (poles.height,)
    assert np.all(post.alpha > 0) and np.all(post.beta > 0)
    mult = post.mean / post.global_rate
    assert np.isfinite(mult).all()
    # 전역 가중평균(전주 균등)은 정확히 1은 아니어도 1 근처(축소·소표본 영향).
    assert 0.5 < float(mult.mean()) < 1.5


def test_zero_event_region_has_wider_posterior():
    """zero-event 소표본 격자(고성)의 사후 변동계수 cv 가 고발화 격자보다 크다."""
    poles = _poles_with_zero_event()
    fires = _fires_hot(poles)
    post = posterior.regional_rate_posterior(poles, fires)
    a_zero, b_zero = post.grid_alpha["g_zero"], post.grid_beta["g_zero"]
    a_hot, b_hot = post.grid_alpha["g_hot"], post.grid_beta["g_hot"]
    cv_zero = (a_zero ** 0.5) / a_zero  # √α/β ÷ (α/β) = 1/√α
    cv_hot = (a_hot ** 0.5) / a_hot
    assert cv_zero > cv_hot   # zero-event → 사후 넓음(불확실성↑)


def test_propagate_risk_posterior_credible_ordering_and_range():
    """risk_lo<=risk_hi, [0,1], 형상=N."""
    poles = _poles_with_zero_event()
    fires = _fires_hot(poles)
    post = posterior.regional_rate_posterior(poles, fires)
    base = np.full(poles.height, 0.3)
    out = posterior.propagate_risk_posterior(base, post, n_draws=100)
    for key in ("risk_mean", "risk_lo", "risk_hi"):
        assert out[key].shape == (poles.height,)
        assert out[key].min() >= -1e-9 and out[key].max() <= 1 + 1e-9
    assert np.all(out["risk_hi"] >= out["risk_lo"] - 1e-9)


def test_propagate_risk_posterior_deterministic():
    """같은 seed → 같은 사후예측(결정성)."""
    poles = _poles_with_zero_event()
    fires = _fires_hot(poles)
    post = posterior.regional_rate_posterior(poles, fires)
    base = np.full(poles.height, 0.3)
    a = posterior.propagate_risk_posterior(base, post, n_draws=80)["risk_mean"]
    b = posterior.propagate_risk_posterior(base, post, n_draws=80)["risk_mean"]
    assert np.array_equal(a, b)


def test_propagate_weight_bootstrap_widens_interval():
    """Bayesian bootstrap 물리가중 변동 → credible 폭 확대."""
    poles = _poles_with_zero_event()
    fires = _fires_hot(poles)
    post = posterior.regional_rate_posterior(poles, fires)
    base = np.full(poles.height, 0.3)
    w0 = posterior.propagate_risk_posterior(base, post, n_draws=120)
    w1 = posterior.propagate_risk_posterior(base, post, n_draws=120,
                                            weight_bootstrap=True,
                                            weight_bootstrap_cv=0.25)
    width0 = float(np.mean(w0["risk_hi"] - w0["risk_lo"]))
    width1 = float(np.mean(w1["risk_hi"] - w1["risk_lo"]))
    assert width1 >= width0


def test_propagate_rejects_shape_and_draws():
    """형상 불일치·draw<2 거부."""
    poles = _poles_with_zero_event()
    post = posterior.regional_rate_posterior(poles, _fires_hot(poles))
    with pytest.raises(ValueError):
        posterior.propagate_risk_posterior(np.zeros(10), post, n_draws=50)
    with pytest.raises(ValueError):
        posterior.propagate_risk_posterior(np.full(poles.height, 0.3), post, n_draws=1)


def test_regional_rate_posterior_missing_column_raises():
    poles = _poles_with_zero_event().drop("grid_id")
    with pytest.raises(KeyError):
        posterior.regional_rate_posterior(poles, _fires_hot(_poles_with_zero_event()))


# ── Phase-5: per-regime conformal + 커버리지 (calibrate.py) ─────────────────
def _risk_with_biased_anchors(n=40000, seed=0):
    """위험점수 + 고위험 편중 발화점(conformal 이 의미있도록)."""
    rng = np.random.default_rng(seed)
    risk = rng.random(n)
    labels = rng.choice(list(config.REGIMES), n).astype("<U16")
    p = risk / risk.sum()
    cal = rng.choice(n, size=400, replace=False, p=p)
    test = rng.choice(n, size=400, replace=False, p=p)
    return risk, labels, cal, test


def test_conformal_threshold_per_regime_keys_and_finite():
    risk, labels, cal, _ = _risk_with_biased_anchors()
    thr = calibrate.conformal_threshold_per_regime(risk, cal, labels, alpha=0.10)
    assert set(thr.keys()) == set(config.REGIMES)
    for r in config.REGIMES:
        assert np.isfinite(thr[r]) or thr[r] == float("-inf")


def test_conformal_threshold_rejects_bad_alpha():
    risk, labels, cal, _ = _risk_with_biased_anchors(n=2000)
    with pytest.raises(ValueError):
        calibrate.conformal_threshold_per_regime(risk, cal, labels, alpha=1.5)


def test_conformal_coverage_near_nominal():
    """편중 발화점에서 conformal 실측 커버리지가 명목 90% 근처(±0.08)."""
    risk, labels, cal, test = _risk_with_biased_anchors(n=60000, seed=3)
    thr = calibrate.conformal_threshold_per_regime(risk, cal, labels, alpha=0.10)
    cov = calibrate.conformal_coverage(risk, test, labels, thr)
    assert abs(cov["all"]["coverage"] - 0.90) < 0.08


def test_conformal_decision_is_binary():
    risk, labels, cal, _ = _risk_with_biased_anchors(n=5000)
    thr = calibrate.conformal_threshold_per_regime(risk, cal, labels, alpha=0.10)
    dec = calibrate.conformal_decision(risk, labels, thr)
    assert set(np.unique(dec).tolist()).issubset({0, 1})
    assert dec.shape == risk.shape


def test_empirical_coverage_full_when_interval_wide():
    """구간이 [0,1] 이면 커버리지=1; 좁으면 <1 (자기일관)."""
    n = 5000
    rng = np.random.default_rng(7)
    point = rng.random(n)
    test = rng.choice(n, size=200, replace=False)
    labels = rng.choice(list(config.REGIMES), n).astype("<U16")
    wide = calibrate.empirical_coverage(np.zeros(n), np.ones(n), test, point,
                                        regime_labels=labels)
    assert wide["all"]["coverage"] == 1.0
    narrow = calibrate.empirical_coverage(point + 0.4, point + 0.5, test, point)
    assert narrow["all"]["coverage"] == 0.0


# ── Phase-5: 제출 새 컬럼 (submit.py) ───────────────────────────────────────
def test_submission_includes_phase5_columns():
    n = config.N_POLES_EXPECTED
    pid = np.arange(n, dtype=np.int64)
    lon = np.full(n, 128.0); lat = np.full(n, 38.0)
    dec = np.zeros(n, dtype=np.int8); dec[:50] = 1
    risk = np.linspace(0, 1, n)
    regime = np.array(["yeongdong"] * n)
    pexp = np.linspace(0, 1, n)
    op = submit.ops_priority_flag(regime, pexp)
    lo = np.clip(risk - 0.05, 0, 1); hi = np.clip(risk + 0.05, 0, 1)
    df = submit.build_submission(pid, lon, lat, dec, risk, regime=regime,
                                 p_exposure=pexp, risk_lo=lo, risk_hi=hi,
                                 ops_priority=op)
    for c in ("p_exposure", "risk_lo", "risk_hi", "ops_priority"):
        assert c in df.columns
    assert set(df["ops_priority"].unique().to_list()).issubset({0, 1})


def test_submission_rejects_inverted_credible():
    n = config.N_POLES_EXPECTED
    pid = np.arange(n, dtype=np.int64)
    risk = np.linspace(0, 1, n)
    lo = np.clip(risk - 0.05, 0, 1); hi = np.clip(risk + 0.05, 0, 1)
    with pytest.raises(ValueError):
        submit.build_submission(pid, np.full(n, 128.0), np.full(n, 38.0),
                                np.zeros(n, dtype=np.int8), risk,
                                risk_lo=hi, risk_hi=lo)  # 뒤집힘


def test_ops_priority_only_yeongdong():
    n = 3000
    rng = np.random.default_rng(11)
    labels = rng.choice(list(config.REGIMES), n).astype("<U16")
    pexp = rng.random(n)
    op = submit.ops_priority_flag(labels, pexp)
    # 1 로 표시된 전주는 모두 영동.
    assert np.all(labels[op == 1] == config.REGIME_YEONGDONG)


def test_ops_priority_none_exposure_all_zero():
    labels = np.array(["yeongdong", "yeongseo", "mountain"] * 10).astype("<U16")
    op = submit.ops_priority_flag(labels, None)
    assert op.sum() == 0


# ── Phase-5: 커버리지 검증 (validate.py) ────────────────────────────────────
def test_coverage_conformal_holdout_shape():
    """누수안전 conformal 커버리지: 키·명목·실측 범위."""
    rng = np.random.default_rng(5)
    n = 8000
    pole_xy = np.column_stack([rng.uniform(127.5, 129.4, n),
                               rng.uniform(37.0, 38.3, n)])
    risk = rng.random(n)
    labels = rng.choice(list(config.REGIMES), n).astype("<U16")
    blocks = validate.spatial_blocks(pole_xy)
    # 발화점→전주: 고위험 편중.
    p = risk / risk.sum()
    f2p = rng.choice(n, size=120, replace=False, p=p)
    out = validate.coverage_conformal_holdout(risk, f2p, labels, blocks, alpha=0.10)
    assert "all" in out and set(out["all"].keys()) == {"nominal", "empirical", "n_test", "n_folds"}
    assert out["all"]["nominal"] == 0.90
    e = out["all"]["empirical"]
    assert np.isnan(e) or (0.0 <= e <= 1.0)


def test_coverage_credible_holdout_shape():
    """누수안전 credible 커버리지: posterior 전파 경로 동작·범위."""
    m = _synthetic_master_with_grid(n=5000, seed=6)
    pole_xy = m.select(["lon", "lat"]).to_numpy().astype(np.float64)
    rng = np.random.default_rng(6)
    base_risk = rng.uniform(0, 0.5, m.height)
    labels = rng.choice(list(config.REGIMES), m.height).astype("<U16")
    blocks = validate.spatial_blocks(pole_xy)
    f2p = rng.choice(m.height, size=80, replace=False)
    out = validate.coverage_credible_holdout(
        base_risk, m, f2p, labels, blocks, n_draws=40, n_folds=3)
    assert "all" in out
    e = out["all"]["empirical"]
    assert np.isnan(e) or (0.0 <= e <= 1.0)


# ── Phase-5b: BYM2 공간 CAR 사후 (posterior.py) ─────────────────────────────
def _lattice_poles(rows: int = 5, cols: int = 5, n_per: int = 1500,
                   sggs: tuple[str, ...] = ("춘천", "강릉", "고성", "인제", "원주"),
                   r0: int = 70, c0: int = 130) -> pl.DataFrame:
    """rows×cols 격자 lattice 합성 전주(격자 id "행_열"). 각 행이 다른 시군."""
    out = []
    pid = 0
    for r in range(rows):
        sgg = sggs[r % len(sggs)]
        for c in range(cols):
            gid = f"{r0 + r}_{c0 + c}"
            for i in range(n_per):
                out.append(dict(pole_id=pid, lon=128.0 + c * 0.05 + (i % 50) * 1e-4,
                                lat=37.7 + r * 0.05 + (i // 50) * 1e-4,
                                sgg=sgg, grid_id=gid))
                pid += 1
    return pl.DataFrame(out)


def _fires_in_grids(poles: pl.DataFrame, grids: tuple[str, ...], n_each: int) -> pl.DataFrame:
    rows = []
    for gid in grids:
        sub = poles.filter(pl.col("grid_id") == gid)
        bl, ba = sub["lon"][0], sub["lat"][0]
        for i in range(n_each):
            rows.append(dict(lon=bl + (i % 20) * 1e-5, lat=ba + (i // 20) * 1e-5))
    return pl.DataFrame(rows)


def test_build_grid_adjacency_rook_structure():
    """rook 인접: 평균이웃·고립 0·전체연결, "행_열" 파싱."""
    grids = [f"{70 + r}_{130 + c}" for r in range(5) for c in range(5)]
    adj = posterior.build_grid_adjacency(grids)
    assert adj.n_grids == 25
    # 5x5 lattice 의 rook 평균이웃 = (4·2 + 12·3 + 9·4)/25 = 3.2.
    assert abs(adj.mean_neighbors - 3.2) < 1e-9
    assert int(adj.isolated.sum()) == 0
    assert adj.n_components == 1
    # 대칭·자기루프 없음.
    W = adj.W
    assert (W != W.T).nnz == 0
    assert W.diagonal().sum() == 0


def test_build_grid_adjacency_isolated_and_unparsable():
    """떨어진 격자는 고립, 비형식 id 는 파싱 실패→고립."""
    grids = ["70_130", "70_131", "90_200", "badid"]
    adj = posterior.build_grid_adjacency(grids)
    # 70_130 ↔ 70_131 만 이웃; 90_200·badid 는 고립.
    assert adj.n_neighbors[0] == 1 and adj.n_neighbors[1] == 1
    assert adj.isolated[2] and adj.isolated[3]
    assert adj.n_components == 3  # {70_130,70_131}, {90_200}, {badid}


def test_bym2_posterior_shape_finite_and_interface():
    """BYM2 사후: 격자 μ/s² 유한·s²>0, RatePosterior 호환 인터페이스(alpha/draw_rate)."""
    poles = _lattice_poles()
    fires = _fires_in_grids(poles, ("72_132", "72_133"), 40)
    post = posterior.bym2_spatial_posterior(poles, fires)
    assert post.alpha.shape == (poles.height,)
    assert np.all(post.beta > 0)
    assert all(np.isfinite(v) for v in post.grid_mu.values())
    assert all(s2 > 0 for s2 in post.grid_s2.values())
    assert 0.0 <= post.phi <= 1.0 and post.tau > 0
    # propagate 호환: draw_rate → (n_draws, N), 비음수.
    rng = np.random.default_rng(config.SEED)
    d = post.draw_rate(50, rng)
    assert d.shape == (50, poles.height)
    assert d.min() >= 0.0


def test_bym2_posterior_deterministic():
    """같은 입력 → 같은 BYM2 사후(결정적 Laplace; 추정 무작위 없음)."""
    poles = _lattice_poles(n_per=1000)
    fires = _fires_in_grids(poles, ("72_132",), 40)
    a = posterior.bym2_spatial_posterior(poles, fires)
    b = posterior.bym2_spatial_posterior(poles, fires)
    for g in a.grid_mu:
        assert a.grid_mu[g] == b.grid_mu[g]
        assert a.grid_s2[g] == b.grid_s2[g]
    assert a.tau == b.tau and a.phi == b.phi


def test_bym2_borrows_strength_in_zero_event_neighbor():
    """합성 sanity: zero-event 격자가 고발화 이웃에서 borrow → PG(부모붕괴)보다 위로.

    중앙 격자에 발화를 몰아주면, 그 **이웃** 격자(발화 0)는 독립 PG 에서 부모로
    붕괴(≈전역 미만)하지만 BYM2 는 이웃 강도를 빌려 사후평균이 부모보다 위로 당겨진다.
    이것이 BYM 이웃 평활·zero-region borrow 복원의 핵심 증거.
    """
    poles = _lattice_poles(n_per=1500)
    # 중앙 72_132 에만 발화 집중(이웃 71_132/73_132/72_131/72_133 은 발화 0).
    fires = _fires_in_grids(poles, ("72_132",), 80)
    pg = posterior.regional_rate_posterior(poles, fires)
    bym = posterior.bym2_spatial_posterior(poles, fires)
    # 핫 격자: 둘 다 높음.
    pg_hot = (pg.grid_alpha["72_132"] / pg.grid_beta["72_132"]) / pg.global_rate
    bym_hot = float(np.exp(bym.grid_mu["72_132"] + 0.5 * bym.grid_s2["72_132"]))
    assert pg_hot > 1.5 and bym_hot > 1.5
    # 핫의 직접 이웃(발화 0): PG 는 부모로 붕괴(<<핫), BYM 은 이웃 borrow 로 PG 보다 위.
    nb = "71_132"
    pg_nb = (pg.grid_alpha[nb] / pg.grid_beta[nb]) / pg.global_rate
    bym_nb = float(np.exp(bym.grid_mu[nb] + 0.5 * bym.grid_s2[nb]))
    assert bym_nb > pg_nb            # borrow: 이웃에서 강도를 빌려 위로 당겨짐
    assert bym_nb > bym_hot * 0.01   # 부모 0 붕괴가 아니라 의미있게 양수


def test_bym2_zero_event_wider_posterior_than_hot():
    """발화 0 격자의 잠재 사후분산 s² 이 고발화 격자보다 크다(불확실성↑)."""
    poles = _lattice_poles(n_per=1500)
    fires = _fires_in_grids(poles, ("72_132",), 80)
    bym = posterior.bym2_spatial_posterior(poles, fires)
    assert bym.grid_s2["70_130"] > bym.grid_s2["72_132"]


def test_bym2_smooth_field_recovers_gradient():
    """합성 매끄러운 장: 발화 강도가 공간 gradient 면 BYM 사후도 단조 gradient 복원.

    열(col)이 커질수록 발화수를 늘린 lattice 에서, BYM 격자 사후평균이 같은 행에서
    열 방향으로 (대체로) 증가해야(이웃 평활로 매끄러운 단조장 복원).
    """
    poles = _lattice_poles(rows=1, cols=6, n_per=2000, sggs=("춘천",))
    # 열 c 에 c*15 발화(왼→오 증가 gradient).
    rows = []
    for c in range(6):
        gid = f"70_{130 + c}"
        sub = poles.filter(pl.col("grid_id") == gid)
        bl, ba = sub["lon"][0], sub["lat"][0]
        for i in range(c * 15):
            rows.append(dict(lon=bl + (i % 20) * 1e-5, lat=ba + (i // 20) * 1e-5))
    fires = pl.DataFrame(rows) if rows else pl.DataFrame({"lon": [], "lat": []},
                                                         schema={"lon": pl.Float64, "lat": pl.Float64})
    bym = posterior.bym2_spatial_posterior(poles, fires)
    mu = [bym.grid_mu[f"70_{130 + c}"] for c in range(6)]
    # 끝(고발화) 이 시작(0발화)보다 확실히 높고, 전반적 증가 추세(스피어만≈단조).
    assert mu[-1] > mu[0]
    incr = sum(1 for c in range(5) if mu[c + 1] >= mu[c] - 1e-9)
    assert incr >= 4   # 6점 중 최소 4 구간 비감소(매끄러운 단조장 복원)


def test_morans_i_higher_for_smooth_field():
    """Moran's I: 매끄러운 장이 무작위 장보다 공간자기상관 높다."""
    grids = [f"{70 + r}_{130 + c}" for r in range(6) for c in range(6)]
    adj = posterior.build_grid_adjacency(grids)
    # 매끄러운 장 = 행+열(이웃 유사), 무작위 장 = 셔플.
    rc = np.array([[int(g.split("_")[0]), int(g.split("_")[1])] for g in grids])
    smooth = (rc[:, 0] + rc[:, 1]).astype(float)
    rng = np.random.default_rng(0)
    noise = rng.permutation(smooth)
    i_smooth = validate._morans_i(smooth, adj.W)
    i_noise = validate._morans_i(noise, adj.W)
    assert i_smooth > i_noise
    assert i_smooth > 0.5   # 강한 양의 공간자기상관


def test_compare_bym_vs_poisson_gamma_structure():
    """비교 헬퍼: sgg 리스트·moran·borrow 필드와 zero-event borrow>0."""
    poles = _lattice_poles(n_per=1500)
    fires = _fires_in_grids(poles, ("72_132", "72_133"), 40)
    pg = posterior.regional_rate_posterior(poles, fires)
    bym = posterior.bym2_spatial_posterior(poles, fires)
    out = validate.compare_bym_vs_poisson_gamma(pg, bym, watch_sgg=("고성", "인제"))
    assert set(out.keys()) == {"sgg", "moran", "borrow_mean_watch"}
    assert {"pg", "bym"} == set(out["moran"].keys())
    for row in out["sgg"]:
        assert set(row.keys()) == {"sgg", "pg_mult", "bym_mult", "delta_mult",
                                   "borrow", "is_watch"}
        assert row["borrow"] >= 0.0
    # zero-event 시군은 borrow(추정 이동량)가 0 이상(이웃 정보로 움직임).
    assert np.isfinite(out["borrow_mean_watch"])
