"""실험 A ablation 경량 단위테스트 — 합성 polars master 기반(데이터 없이 도는).

검증: (1) build_risk 출력 형상=(N,)·범위 [0,1], (2) feature_set 선택이 정확히
요청한 피처열만 사용(가중 형상·결정성), (3) substation 부재 합성입력에서 'plus' 가
'aware' 로 폴백. dist_to_substation 미포함 합성이라 blind/aware 위주로 본다.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from pfire import ablation, config, experts, regimes, weather


def _synthetic_master(n: int = 1500, seed: int = 0) -> pl.DataFrame:
    """게이트·전문가·기상에 필요한 컬럼만 가진 합성 마스터(substation 없음)."""
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
        "S_p": rng.uniform(0.05, 0.98, n),
    })


def _shared_from_master(master: pl.DataFrame) -> dict:
    """load_ablation_shared 의 합성판(io·앵커 없이 build_risk 검증에 필요한 키만)."""
    gate, regime_order = regimes.compute_gate(master)
    feats = experts.build_ignition_features(master)
    W = weather.season_weather(master, source="features")
    S = np.clip(master["S_p"].to_numpy().astype(np.float64), 0.0, 1.0)
    return dict(master=master, gate=gate, regime_order=list(regime_order),
                feats=feats, S=S, W=W)


def _row_simplex_weights(d: int, seed: int = 1) -> np.ndarray:
    """(3, d) 행합=1 가중(테스트용)."""
    rng = np.random.default_rng(seed)
    w = rng.random((3, d))
    return w / w.sum(axis=1, keepdims=True)


def test_build_risk_shape_and_range():
    """build_risk 출력은 (N,) 이며 [0,1] 범위."""
    m = _synthetic_master()
    shared = _shared_from_master(m)
    keys = config.FEATURE_SET_ASSET_AWARE  # forest..+powerline (7)
    w = _row_simplex_weights(len(keys))
    R = ablation.build_risk(shared, keys, w)
    assert R.shape == (m.height,)
    assert np.isfinite(R).all()
    assert (R >= 0.0).all() and (R <= 1.0).all()


def test_feature_set_selection_uses_exact_columns():
    """blind 와 aware 의 d 가 정확히 요청 피처 수이고, weights 형상 불일치는 오류.

    또한 aware(=blind+powerline)에서 powerline 가중을 0 으로 두면 blind 와 동일 R 이
    나와, build_risk 가 정확히 feature_set 의 키만 가중함을 확인한다.
    """
    m = _synthetic_master()
    shared = _shared_from_master(m)

    blind = config.FEATURE_SET_ASSET_BLIND          # 6
    aware = config.FEATURE_SET_ASSET_AWARE          # 7 (=blind + powerline 끝에)
    assert aware[:len(blind)] == blind and aware[-1] == "powerline"

    # 형상 검증: 잘못된 열 수는 ValueError.
    with pytest.raises(ValueError):
        ablation.build_risk(shared, aware, _row_simplex_weights(len(blind)))

    # blind 가중 w_b, aware 가중 = [w_b, 0](powerline=0) → 재정규화 없이 동일 R.
    w_b = _row_simplex_weights(len(blind), seed=3)
    w_a = np.concatenate([w_b, np.zeros((3, 1))], axis=1)  # powerline 가중 0
    R_blind = ablation.build_risk(shared, blind, w_b)
    R_aware0 = ablation.build_risk(shared, aware, w_a)
    assert np.allclose(R_blind, R_aware0, atol=1e-12)


def test_plus_falls_back_when_substation_absent():
    """substation 컬럼 없는 합성입력에서 'plus' feature_set 은 'aware'(7키)로 폴백."""
    m = _synthetic_master()
    shared = _shared_from_master(m)
    assert "substation" not in shared["feats"]
    keys = ablation._resolve_feature_set("plus", shared["feats"])
    assert keys == config.FEATURE_SET_ASSET_AWARE  # substation 제외 폴백
    # 폴백 키로 build_risk 가 정상 동작(7 가중).
    R = ablation.build_risk(shared, "plus", _row_simplex_weights(len(keys)))
    assert R.shape == (m.height,) and (R >= 0).all() and (R <= 1).all()
