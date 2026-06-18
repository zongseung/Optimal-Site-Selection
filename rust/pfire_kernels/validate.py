#!/usr/bin/env python
"""Accuracy validation for the `pfire_kernels` Rust kernel.

Compares the Rust ``simulate_exposure`` against a small-scale pure-numpy
reference implementing the *same* contract semantics
(``claudedocs/CONTRACT_rust_python.md``).

Because the Rust kernel uses ChaCha8 per-simulation RNG streams while the numpy
reference uses numpy's Generator, the two cannot share an identical random
stream. The contract explicitly permits a *statistically close* match, so we
verify that the mean absolute error of per-pole P(exposure) is small at a
simulation count high enough to suppress Monte-Carlo noise.

Run: ``.venv/bin/python rust/pfire_kernels/validate.py``
"""
from __future__ import annotations

import numpy as np

import pfire_kernels

DEG2RAD = np.pi / 180.0


def numpy_reference(
    pole_xy: np.ndarray,
    ignition_idx: np.ndarray,
    wind_dir_deg: np.ndarray,
    wind_speed: np.ndarray,
    fuel: np.ndarray,
    southness: np.ndarray,
    max_dist_km: float,
    length_scale_km: float,
    wind_aniso: float,
    southness_beta: float,
    seed: int,
) -> np.ndarray:
    """Vectorised numpy reference implementing the contract semantics."""
    n = pole_xy.shape[0]
    n_sims = wind_dir_deg.shape[0]
    fuel_clip = np.clip(fuel, 0.0, 1.0)
    south_term = 1.0 + southness_beta * southness
    hit = np.zeros(n, dtype=np.float64)

    gx = pole_xy[ignition_idx, 0]
    gy = pole_xy[ignition_idx, 1]

    for s in range(n_sims):
        # Per-sim independent RNG seeded with seed + s (matches Rust scheme).
        rng = np.random.default_rng(np.uint64(seed) + np.uint64(s))
        theta = (wind_dir_deg[s] + 180.0) * DEG2RAD
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        aniso_speed = wind_aniso * (wind_speed[s] / 5.0)

        reached = np.zeros(n, dtype=bool)
        for g in range(ignition_idx.shape[0]):
            dx = pole_xy[:, 0] - gx[g]
            dy = pole_xy[:, 1] - gy[g]
            d = np.sqrt(dx * dx + dy * dy)
            within = d <= max_dist_km
            idx = np.nonzero(within)[0]
            di = d[idx]
            with np.errstate(invalid="ignore", divide="ignore"):
                align = np.where(di > 0, (dx[idx] * cos_t + dy[idx] * sin_t) / di, 0.0)
            align_pos = np.maximum(0.0, align)
            ll = length_scale_km * (1.0 + aniso_speed * align_pos)
            reach_prob = np.exp(-di / ll) * fuel_clip[idx] * south_term[idx]
            draws = rng.random(idx.shape[0])
            reached[idx] |= draws < reach_prob
        hit += reached

    return hit / n_sims


def main() -> None:
    rng = np.random.default_rng(0)
    # Contract validation scale is N~2000, M~20. We run a small S=32 noisy check
    # plus a high-S check (S=2048) where per-pole Monte-Carlo noise is small
    # enough to confirm the two implementations agree on the *probabilities*
    # (not just the random streams, which differ between ChaCha8 and numpy).
    n, m = 2000, 20
    seed = 20260618

    # Random poles spread over ~20x20 km, plausible fuel/southness ranges.
    pole_xy = rng.uniform(0.0, 20.0, size=(n, 2)).astype(np.float64)
    ignition_idx = rng.choice(n, size=m, replace=False).astype(np.uint32)
    fuel = rng.uniform(0.0, 1.0, size=n).astype(np.float64)
    southness = rng.uniform(-1.0, 1.0, size=n).astype(np.float64)

    overall_ok = True
    for s in (32, 2048):
        wind_dir_deg = rng.uniform(0.0, 360.0, size=s).astype(np.float64)
        wind_speed = rng.uniform(1.0, 12.0, size=s).astype(np.float64)
        args = (
            pole_xy, ignition_idx, wind_dir_deg, wind_speed, fuel, southness,
            5.0, 0.8, 1.5, 0.3, seed,
        )

        p_rust = pfire_kernels.simulate_exposure(*args)
        p_ref = numpy_reference(*args)

        assert p_rust.shape == (n,)
        assert np.all(p_rust >= 0.0) and np.all(p_rust <= 1.0)

        diff = np.abs(p_rust - p_ref)
        mae = float(diff.mean())
        maxe = float(diff.max())
        mask = (p_rust > 0) | (p_ref > 0)
        corr = float(np.corrcoef(p_rust[mask], p_ref[mask])[0, 1]) if mask.sum() > 1 else 1.0

        print(f"N={n} M={m} S={s} seed={seed}")
        print(f"  rust mean P  : {p_rust.mean():.6f}   ref mean P: {p_ref.mean():.6f}")
        print(f"  nonzero poles: rust={int((p_rust > 0).sum())} ref={int((p_ref > 0).sum())}")
        print(f"  MAE          : {mae:.6f}")
        print(f"  max abs err  : {maxe:.6f}")
        print(f"  correlation  : {corr:.6f}")

        # At S=2048 binomial std ~ 0.5/sqrt(2048) ~ 0.011, so MAE should be small.
        # At S=32 we only require the *aggregate* mean P to match closely.
        if s >= 2048:
            ok = mae < 0.012 and corr > 0.99
        else:
            ok = abs(p_rust.mean() - p_ref.mean()) < 0.01 and corr > 0.85
        print("  ACCURACY:", "PASS" if ok else "FAIL")
        overall_ok &= ok

    print("OVERALL:", "PASS" if overall_ok else "FAIL")
    if not overall_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
