# pfire_kernels

Performance-critical **Rust** kernels for the pfire wildfire-risk model, exposed
to Python via **PyO3 / maturin** as the `pfire_kernels` module.

The single entry point, `simulate_exposure`, runs a Monte-Carlo anisotropic
*downwind* fire-spread proxy over ~1.38M utility poles (전주) and returns each
pole's probability of fire exposure. It implements the fixed interface in
[`claudedocs/CONTRACT_rust_python.md`](../../claudedocs/CONTRACT_rust_python.md).

## Layout

```
rust/pfire_kernels/
├── Cargo.toml        # crate + release profile (LTO, codegen-units=1)
├── pyproject.toml    # maturin build backend
├── src/
│   ├── lib.rs        # PyO3 binding + simulation core + unit tests
│   └── grid.rs       # uniform-grid bucket spatial index (radius queries)
├── validate.py       # accuracy check vs. a pure-numpy reference
├── benchmark.py      # large-scale (N=1.38M, M=2000, S=256) timing
└── README.md
```

## Build / install

Requires Rust 1.96+ and `maturin` in the project venv
(`/home/dlwhdtmd/OR-project/.venv`):

```bash
# one-time: install the build tool into the venv
uv pip install maturin

# build the release extension and install it editable into the venv
cd rust/pfire_kernels
PATH="$HOME/.cargo/bin:$PATH" \
  /home/dlwhdtmd/OR-project/.venv/bin/maturin develop --release
```

## Python usage

```python
import numpy as np
import pfire_kernels

p = pfire_kernels.simulate_exposure(
    pole_xy,        # (N, 2) float64 — pole coords, planar km (config.LAT0_DEG)
    ignition_idx,   # (M,)   uint32  — indices into pole_xy of ignition sources
    wind_dir_deg,   # (S,)   float64 — MC wind FROM-direction, degrees
    wind_speed,     # (S,)   float64 — MC wind speed, m/s
    fuel,           # (N,)   float64 — flammability in [0, 1]
    southness,      # (N,)   float64 — SW-aspect alignment in [-1, 1]
    max_dist_km=5.0,        # spread cutoff radius
    length_scale_km=0.8,    # distance-decay scale L0
    wind_aniso=1.5,         # downwind elongation alpha
    southness_beta=0.3,     # SW-aspect correction beta
    seed=20260618,          # reproducibility seed
)
# p: (N,) float64, per-pole P(exposure) in [0, 1]
```

## Semantics

Per simulation `s`, the downwind direction is `theta_s = wind_dir_deg[s] + 180°`.
For each ignition source `g`, every pole `p` within `max_dist_km`:

```
d         = ||g - p||                       (km)
align     = cos(angle(g→p) − theta_s)       in [-1, 1]
L         = length_scale_km * (1 + wind_aniso * max(0, align) * wind_speed[s]/5)
reach_prob = exp(-d / L) * clip(fuel[p],0,1) * (1 + southness_beta * southness[p])
reached[p] |= (uniform() < reach_prob)      OR-combined across ignition sources
```

`P(exposure)[p]` is the fraction of the `S` simulations in which `p` was reached.

## Determinism

Simulation `s` uses an independent `ChaCha8` RNG seeded with `seed + s`. Within a
simulation, uniform draws are consumed in a fixed order — ignition sources in
`ignition_idx` order, candidate poles in spatial-bucket order — so identical
inputs always yield identical output, regardless of thread count (verified:
parallel and single-thread runs produce the same mean P to full precision).

## Performance

- Parallelised across the simulation (S) axis with **rayon**; 64-core target.
- Fixed-radius neighbour queries use a uniform square-cell grid (cell size =
  `max_dist_km`), reducing each ignition query to the surrounding 3×3 cells.
- Measured (N=1,387,831, M=2000, S=256, this machine, 64 cores):
  - parallel: **~5.9 s**
  - single-thread: **~184.5 s** → **~31× speedup**

## Development

```bash
cd rust/pfire_kernels
PATH="$HOME/.cargo/bin:$PATH" cargo fmt
PATH="$HOME/.cargo/bin:$PATH" cargo clippy --release --all-targets -- -D warnings
PATH="$HOME/.cargo/bin:$PATH" cargo test --release      # Rust unit tests

# Python-side checks (after `maturin develop --release`)
/home/dlwhdtmd/OR-project/.venv/bin/python validate.py   # accuracy vs numpy
/home/dlwhdtmd/OR-project/.venv/bin/python benchmark.py  # large-scale timing
```
