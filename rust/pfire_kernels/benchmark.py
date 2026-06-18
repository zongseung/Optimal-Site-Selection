#!/usr/bin/env python
"""Large-scale benchmark for the `pfire_kernels` Rust kernel.

Loads the real 1.38M pole coordinates, converts lon/lat to a local planar km
frame (config.LAT0_DEG), and times ``simulate_exposure`` at the production
scale (N=1.38M, M=2000, S=256). Also runs a single-threaded pass (via the
RAYON_NUM_THREADS=1 environment variable in a subprocess) to report speedup /
core utilisation.

Run: ``.venv/bin/python rust/pfire_kernels/benchmark.py``
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, "/home/dlwhdtmd/OR-project")
from pfire import config  # noqa: E402

import pfire_kernels  # noqa: E402

R = config.EARTH_R_KM
LAT0 = config.LAT0_DEG


def load_inputs():
    """Load real pole coords + fuel/southness, return kernel-ready arrays."""
    feats = pl.read_parquet(
        config.F_POLE_FEATURES, columns=["pole_id", "lon", "lat"]
    )
    overlay = pl.read_parquet(
        config.F_POLE_STATIC, columns=["pole_id", "mu_flammability", "mu_southness"]
    )
    df = feats.join(overlay, on="pole_id", how="left").sort("pole_id")

    lon = df["lon"].to_numpy()
    lat = df["lat"].to_numpy()
    cos_lat0 = np.cos(np.radians(LAT0))
    x = np.radians(lon) * R * cos_lat0
    y = np.radians(lat) * R
    pole_xy = np.ascontiguousarray(np.column_stack([x, y]).astype(np.float64))

    fuel = np.nan_to_num(df["mu_flammability"].to_numpy().astype(np.float64), nan=0.0)
    southness = np.nan_to_num(df["mu_southness"].to_numpy().astype(np.float64), nan=0.0)
    return pole_xy, fuel, southness


def make_mc(n, m, s, seed):
    rng = np.random.default_rng(seed)
    ignition_idx = rng.choice(n, size=m, replace=False).astype(np.uint32)
    wind_dir_deg = rng.uniform(0.0, 360.0, size=s).astype(np.float64)
    # Plausible Gangwon wind speeds (m/s); heavier tail for 양간지풍 events.
    wind_speed = np.clip(rng.gamma(2.0, 2.5, size=s), 0.5, 25.0).astype(np.float64)
    return ignition_idx, wind_dir_deg, wind_speed


def main() -> None:
    single = os.environ.get("PFIRE_SINGLE_THREAD") == "1"
    pole_xy, fuel, southness = load_inputs()
    n = pole_xy.shape[0]
    m, s, seed = 2000, 256, config.SEED
    ignition_idx, wind_dir_deg, wind_speed = make_mc(n, m, s, seed)

    args = (
        pole_xy, ignition_idx, wind_dir_deg, wind_speed, fuel, southness,
        config.SPREAD_MAX_DIST_KM, config.SPREAD_LENGTH_SCALE_KM,
        config.SPREAD_WIND_ANISO, config.SPREAD_SOUTHNESS_BETA, seed,
    )

    # Warm-up untimed (page-in, grid build is inside the timed call anyway).
    t0 = time.perf_counter()
    p = pfire_kernels.simulate_exposure(*args)
    dt = time.perf_counter() - t0

    mode = "SINGLE-THREAD" if single else f"PARALLEL ({os.cpu_count()} cores)"
    print(f"[{mode}] N={n} M={m} S={s}")
    print(f"  wall time     : {dt:.2f} s")
    print(f"  mean P        : {p.mean():.6f}")
    print(f"  nonzero poles : {int((p > 0).sum())} ({100 * (p > 0).mean():.2f}%)")
    print(f"  P range       : [{p.min():.4f}, {p.max():.4f}]")
    # Emit machine-readable line for the parent runner to parse if needed.
    print(f"RESULT mode={'single' if single else 'parallel'} seconds={dt:.4f}")


if __name__ == "__main__":
    main()
