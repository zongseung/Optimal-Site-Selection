"""웹앱 위험 예측 엔진.

물리 위험 R_base = I·S·W 를 **한 번** 계산해 parquet 으로 캐시하고, 웹에서 들어온
입력(발화점·고위험 비율·배분방식)에 따라 지역배율(계층 EB)과 임계 배분만 다시
적용해 격자 위험지도를 반환한다. 무거운 물리 계산은 캐시로 1회만 수행하므로,
발화점 입력·슬라이더 조정은 초 단위로 반응한다.

핵심: 발화점은 per-pole 라벨이 아니라 **지역 앵커**다. 입력 발화점은
  1) hierarchy.regional_multiplier 로 지역 위험 배율을 재추정(발화 잦은 지역↑),
  2) 체제앵커 배분(decide_threshold_per_regime + regime_anchor_count)으로
     고위험 예산을 발화밀도 비례로 나눠 어디를 위험(1)으로 볼지 재결정
하는 데 쓰인다(README §1 결정 철학과 동일).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

# OR 루트 임포트 경로 보장(pfire / scripts).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pfire import calibrate, config, hierarchy  # noqa: E402
from pfire.risk_index import risk_percentile  # noqa: E402
from scripts.predict import (  # noqa: E402
    cells_from_agg, compute_risk, legend_stops, _grid_aggregate,
)

logger = logging.getLogger("webapp.engine")

CACHE = config.OUT / "predict" / "_engine_cache.parquet"
GRID_DEG = 0.02
GRID_DEG_SPREAD = 0.004  # 확산 데모 격자(≈450m; 발화점 주변 세밀)


class Engine:
    """캐시된 물리 위험 + 전주 메타(좌표·체제·시군·격자)."""

    def __init__(self, df: pl.DataFrame):
        self.df = df  # pole_id, lon, lat, sgg, grid_id, regime, R_base
        self.lon = df["lon"].to_numpy().astype(np.float64)
        self.lat = df["lat"].to_numpy().astype(np.float64)
        self.regime_lbl = df["regime"].to_numpy()
        self.R_base = df["R_base"].to_numpy().astype(np.float64)
        self.pole_xy = np.column_stack([self.lon, self.lat])
        self.center = (float(np.median(self.lat)), float(np.median(self.lon)))
        self.n_total = df.height
        # 확산 데모용(지연 로드): 연료·사면·평면 km 좌표.
        self._fuel = None
        self._southness = None
        self._pole_km = None

    def _ensure_spread_arrays(self) -> None:
        """확산 시뮬 입력(연료·사면·km좌표)을 최초 1회 지연 로드."""
        if self._fuel is not None:
            return
        from pfire import exposure_engine
        st = (pl.read_parquet(
                config.F_POLE_STATIC,
                columns=["pole_id", "mu_flammability", "mu_southness"])
              .sort("pole_id"))
        self._fuel = np.clip(st["mu_flammability"].to_numpy().astype(np.float64), 0, 1)
        self._southness = st["mu_southness"].to_numpy().astype(np.float64) * 2.0 - 1.0
        self._pole_km = exposure_engine.lonlat_to_km(self.pole_xy)
        logger.info("확산 배열 로드 완료(연료·사면·km좌표)")

    def spread(
        self, lon0: float, lat0: float, *,
        wind_deg: float = 270.0, wind_speed: float = 8.0,
        n_sims: int = 96, thresh: float = 0.01,
    ) -> dict:
        """발화점(lon0,lat0)에서 풍하 확산 노출확률을 모사해 지도 셀로 반환.

        기존 확산 커널(pfire.exposure.simulate_exposure, Rust)을 그대로 사용한다.
        발화원 1개라 반경 컷오프(config.SPREAD_MAX_DIST_KM) 내만 계산 — 초 미만.
        """
        from pfire import exposure
        self._ensure_spread_arrays()

        cos_lat = np.cos(np.deg2rad(config.LAT0_DEG))
        d2 = ((self.lon - lon0) * cos_lat) ** 2 + (self.lat - lat0) ** 2
        ign = int(d2.argmin())

        rng = np.random.default_rng(config.SEED)
        wdir = (rng.normal(wind_deg, 25.0, size=n_sims)) % 360.0
        wspd = np.clip(rng.normal(wind_speed, wind_speed * 0.25, size=n_sims), 1.0, None)

        p = exposure.simulate_exposure(
            self._pole_km, np.array([ign], dtype=np.uint32),
            wdir, wspd, self._fuel, self._southness, seed=config.SEED)

        exp = p > thresh
        cells = _prob_cells(self.lon[exp], self.lat[exp], p[exp], GRID_DEG_SPREAD)
        return {
            "ignition": [float(self.lat[ign]), float(self.lon[ign])],
            "downwind_deg": (wind_deg + 180.0) % 360.0,
            "cells": cells,
            "grid_deg": GRID_DEG_SPREAD,
            "n_exposed": int(exp.sum()),
            "max_prob": round(float(p.max()) if p.size else 0.0, 3),
            "radius_km": config.SPREAD_MAX_DIST_KM,
        }


def _prob_cells(lon: np.ndarray, lat: np.ndarray, p: np.ndarray,
                grid_deg: float) -> list[dict]:
    """노출확률 전주 → 격자 셀(최대확률 색; YlOrRd). 확산 오버레이용."""
    from matplotlib import colormaps, colors as mcolors
    if len(lon) == 0:
        return []
    cmap = colormaps["YlOrRd"]
    vmax = max(0.2, float(p.max()))
    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    ci = np.floor(lon / grid_deg).astype(np.int64)
    cj = np.floor(lat / grid_deg).astype(np.int64)
    agg = (pl.DataFrame({"ci": ci, "cj": cj, "p": p})
           .group_by(["ci", "cj"]).agg(pl.col("p").max().alias("p")))
    cells = []
    for row in agg.iter_rows(named=True):
        r0 = row["cj"] * grid_deg
        c0 = row["ci"] * grid_deg
        pr = float(row["p"])
        cells.append({
            "b": [round(r0, 5), round(c0, 5),
                  round(r0 + grid_deg, 5), round(c0 + grid_deg, 5)],
            "c": mcolors.to_hex(cmap(norm(pr))),
            "p": round(pr, 3),
        })
    return cells


def load_engine(*, force: bool = False, use_exposure_v2: bool = False) -> Engine:
    """엔진 로드 — 캐시 있으면 즉시 로드, 없으면 물리 위험 1회 계산 후 캐시.

    Parameters
    ----------
    force : bool
        True 면 캐시 무시하고 재계산.
    use_exposure_v2 : bool
        R_base 에 풍하 노출 v2.2 결합 포함 여부(느림; 기본 False 로 빠른 시작).
    """
    if CACHE.exists() and not force:
        logger.info("engine cache 로드: %s", CACHE)
        return Engine(pl.read_parquet(CACHE))

    logger.info("engine 최초 계산(캐시 없음, exposure_v2=%s) …", use_exposure_v2)
    res = compute_risk(None, use_exposure_v2=use_exposure_v2)
    m = res["master"]
    df = pl.DataFrame({
        "pole_id": m["pole_id"].to_numpy(),
        "lon": m["lon"].to_numpy(),
        "lat": m["lat"].to_numpy(),
        "sgg": m["sgg"].to_numpy(),
        "grid_id": m["grid_id"].to_numpy(),
        "regime": res["regime_lbl"],
        "R_base": res["R"].astype(np.float64),
    })
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(CACHE)
    logger.info("engine cache 저장: %s (행수=%d)", CACHE, df.height)
    return Engine(df)


def predict(
    engine: Engine, *,
    prevalence: float = 0.02,
    alloc: str = "global",
    fires_xy: np.ndarray | None = None,
) -> dict:
    """입력에 따라 위험지도(격자 셀)와 요약을 계산해 dict 로 반환.

    Parameters
    ----------
    engine : Engine
        캐시된 물리 위험.
    prevalence : float
        고위험(decision=1) 예측양성 비율(전역 예산).
    alloc : {"global", "regime_anchor"}
        배분방식. regime_anchor 는 발화점이 있을 때만 유효(없으면 global 폴백).
    fires_xy : numpy.ndarray (M,2) or None
        입력 발화점(lon, lat). 있으면 지역배율 재계산 + 체제앵커 배분에 사용.
    """
    R = engine.R_base.copy()
    n_fires = 0
    mult_applied = False

    if fires_xy is not None and len(fires_xy) > 0:
        n_fires = int(len(fires_xy))
        fires_df = pl.DataFrame({"lon": fires_xy[:, 0], "lat": fires_xy[:, 1]})
        poles_df = engine.df.select(["lon", "lat", "sgg", "grid_id"])
        mult = hierarchy.regional_multiplier(poles_df, fires_df, normalize="mean")
        R = np.clip(R * mult, 0.0, 1.0)
        mult_applied = True

    if alloc == "regime_anchor" and n_fires > 0:
        fire_to_pole = calibrate.assign_poles_to_fires(
            engine.pole_xy, fires_xy, radius_km=1.0)
        anchor = calibrate.regime_anchor_count(engine.regime_lbl, fire_to_pole)
        _, decision = calibrate.decide_threshold_per_regime(
            R, engine.regime_lbl, prevalence,
            calibrate.ALLOC_ANCHOR, anchor_count=anchor)
        alloc_used = "regime_anchor"
    else:
        _, decision = calibrate.decide_threshold(R, prevalence)
        alloc_used = "global"

    pct = np.round(risk_percentile(R), 2)
    agg = _grid_aggregate(engine.lon, engine.lat, pct, decision,
                          engine.regime_lbl, GRID_DEG)
    cells = cells_from_agg(agg, GRID_DEG)

    regime_dist = {r: int(decision[engine.regime_lbl == r].sum())
                   for r in config.REGIMES}

    return {
        "cells": cells,
        "center": [engine.center[0], engine.center[1]],
        "grid_deg": GRID_DEG,
        "n_cells": len(cells),
        "n_total": engine.n_total,
        "n_high_total": int(decision.sum()),
        "prevalence": prevalence,
        "alloc": alloc_used,
        "n_fires": n_fires,
        "mult_applied": mult_applied,
        "regime_dist": regime_dist,
        "legend": legend_stops(),
    }
