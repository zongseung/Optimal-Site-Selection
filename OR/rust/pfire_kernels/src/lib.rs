//! `pfire_kernels` — performance-critical Rust kernels for the pfire wildfire
//! risk model, exposed to Python via PyO3/maturin.
//!
//! The single public entry point [`simulate_exposure`] runs a Monte-Carlo
//! anisotropic *downwind* fire-spread proxy and returns, for every utility pole
//! (전주), the probability that it is exposed to fire across the simulations.
//!
//! See `claudedocs/CONTRACT_rust_python.md` for the fixed Python-facing
//! signature and semantics this module implements.

mod grid;

use grid::Grid;
use numpy::ndarray::{ArrayView1, ArrayView2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

/// Degrees-to-radians conversion factor.
const DEG2RAD: f64 = std::f64::consts::PI / 180.0;

/// Core simulation, decoupled from Python types so it can be unit-tested and
/// reused from Rust benchmarks.
///
/// Parameters mirror the Python contract. `pole_xy` is `[[x, y]; N]` in km,
/// `ignition_idx` indexes into `pole_xy`, and `wind_dir_deg`/`wind_speed` have
/// length `S` (the number of Monte-Carlo simulations).
///
/// Determinism: simulation `s` uses an independent RNG seeded with
/// `seed + s`. Within a simulation the random draws are consumed in a fixed
/// order — for each ignition source (in `ignition_idx` order), for each
/// candidate pole within the cutoff radius (in spatial-bucket iteration order).
/// The same inputs therefore always yield the same `P(exposure)`.
///
/// Returns the per-pole exposure probability in `[0, 1]` (length `N`).
#[allow(clippy::too_many_arguments)]
pub fn simulate_exposure_core(
    pole_xy: &[[f64; 2]],
    ignition_idx: &[u32],
    wind_dir_deg: &[f64],
    wind_speed: &[f64],
    fuel: &[f64],
    southness: &[f64],
    max_dist_km: f64,
    length_scale_km: f64,
    wind_aniso: f64,
    southness_beta: f64,
    seed: i64,
) -> Vec<f64> {
    let n = pole_xy.len();
    let n_sims = wind_dir_deg.len();

    if n == 0 || n_sims == 0 || ignition_idx.is_empty() {
        return vec![0.0; n];
    }

    // Build the spatial index once; shared read-only across worker threads.
    let grid = Grid::build(pole_xy, max_dist_km);
    let max_dist_sq = max_dist_km * max_dist_km;

    // Precompute clipped fuel once: clip(fuel, 0, 1). southness_beta * southness
    // is left per-pole because it depends only on `southness`, also precomputed.
    let fuel_clip: Vec<f64> = fuel.iter().map(|&f| f.clamp(0.0, 1.0)).collect();
    let south_term: Vec<f64> = southness
        .iter()
        .map(|&s| 1.0 + southness_beta * s)
        .collect();

    // Per-simulation reached masks are reduced into a single per-pole hit count.
    // Each simulation produces an independent boolean mask; summing the masks
    // across simulations and dividing by `n_sims` gives the exposure prob.
    //
    // We parallelise across simulations (the S axis) and reduce u32 hit counts.
    let hit_counts = (0..n_sims)
        .into_par_iter()
        .fold(
            || vec![0u32; n],
            |mut acc, s| {
                // Independent, reproducible RNG per simulation.
                let sim_seed = (seed as u64).wrapping_add(s as u64);
                let mut rng = ChaCha8Rng::seed_from_u64(sim_seed);

                // Downwind direction θ_s = (from-direction) + 180°, in radians.
                let theta = (wind_dir_deg[s] + 180.0) * DEG2RAD;
                let cos_t = theta.cos();
                let sin_t = theta.sin();
                // Wind elongation multiplier component: wind_aniso * speed / 5.
                let aniso_speed = wind_aniso * (wind_speed[s] / 5.0);

                // Per-simulation reached mask (OR across ignition sources).
                let mut reached = vec![false; n];

                for &g in ignition_idx {
                    let gx = pole_xy[g as usize][0];
                    let gy = pole_xy[g as usize][1];

                    grid.for_each_candidate(gx, gy, |pid| {
                        let p = pid as usize;
                        let dx = pole_xy[p][0] - gx;
                        let dy = pole_xy[p][1] - gy;
                        let d_sq = dx * dx + dy * dy;
                        if d_sq > max_dist_sq {
                            return;
                        }
                        let d = d_sq.sqrt();

                        // align = cos(φ - θ) where φ = atan2(dy, dx).
                        // cos(φ - θ) = cos φ cos θ + sin φ sin θ
                        //            = (dx cos θ + dy sin θ) / d.
                        // d == 0 (ignition pole itself) → treat align as 0.
                        let align = if d > 0.0 {
                            (dx * cos_t + dy * sin_t) / d
                        } else {
                            0.0
                        };
                        let align_pos = if align > 0.0 { align } else { 0.0 };

                        // L = length_scale * (1 + aniso * max(0, align) * speed/5).
                        let l = length_scale_km * (1.0 + aniso_speed * align_pos);
                        // L is strictly positive when length_scale_km > 0.
                        let reach_prob = (-d / l).exp() * fuel_clip[p] * south_term[p];

                        // One uniform draw per (ignition, candidate-pole) pair,
                        // in fixed iteration order → deterministic.
                        if rng.gen::<f64>() < reach_prob {
                            reached[p] = true;
                        }
                    });
                }

                for (a, &r) in acc.iter_mut().zip(reached.iter()) {
                    if r {
                        *a += 1;
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0u32; n],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    *x += *y;
                }
                a
            },
        );

    let inv = 1.0 / n_sims as f64;
    hit_counts.iter().map(|&c| c as f64 * inv).collect()
}

/// Monte-Carlo downwind fire-spread exposure kernel (Python entry point).
///
/// See [`simulate_exposure_core`] and the project contract for full semantics.
/// Validates input shapes/lengths and raises `ValueError` on mismatch rather
/// than panicking. Releases the GIL during the parallel compute.
#[pyfunction]
#[pyo3(signature = (
    pole_xy,
    ignition_idx,
    wind_dir_deg,
    wind_speed,
    fuel,
    southness,
    max_dist_km,
    length_scale_km,
    wind_aniso,
    southness_beta,
    seed,
))]
#[allow(clippy::too_many_arguments)]
fn simulate_exposure<'py>(
    py: Python<'py>,
    pole_xy: PyReadonlyArray2<'py, f64>,
    ignition_idx: PyReadonlyArray1<'py, u32>,
    wind_dir_deg: PyReadonlyArray1<'py, f64>,
    wind_speed: PyReadonlyArray1<'py, f64>,
    fuel: PyReadonlyArray1<'py, f64>,
    southness: PyReadonlyArray1<'py, f64>,
    max_dist_km: f64,
    length_scale_km: f64,
    wind_aniso: f64,
    southness_beta: f64,
    seed: i64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let xy: ArrayView2<f64> = pole_xy.as_array();
    if xy.ncols() != 2 {
        return Err(PyValueError::new_err(format!(
            "pole_xy must have shape (N, 2), got (_, {})",
            xy.ncols()
        )));
    }
    let n = xy.nrows();

    let fuel_v: ArrayView1<f64> = fuel.as_array();
    let south_v: ArrayView1<f64> = southness.as_array();
    if fuel_v.len() != n {
        return Err(PyValueError::new_err(format!(
            "fuel length {} != N {}",
            fuel_v.len(),
            n
        )));
    }
    if south_v.len() != n {
        return Err(PyValueError::new_err(format!(
            "southness length {} != N {}",
            south_v.len(),
            n
        )));
    }

    let wd: ArrayView1<f64> = wind_dir_deg.as_array();
    let ws: ArrayView1<f64> = wind_speed.as_array();
    if wd.len() != ws.len() {
        return Err(PyValueError::new_err(format!(
            "wind_dir_deg length {} != wind_speed length {}",
            wd.len(),
            ws.len()
        )));
    }

    if length_scale_km <= 0.0 || !length_scale_km.is_finite() {
        return Err(PyValueError::new_err(
            "length_scale_km must be a finite positive number",
        ));
    }
    if !max_dist_km.is_finite() || max_dist_km < 0.0 {
        return Err(PyValueError::new_err(
            "max_dist_km must be a finite non-negative number",
        ));
    }

    let ign: ArrayView1<u32> = ignition_idx.as_array();
    // Validate ignition indices are in range before entering the kernel.
    for &g in ign.iter() {
        if (g as usize) >= n {
            return Err(PyValueError::new_err(format!(
                "ignition_idx contains out-of-range index {g} (N = {n})"
            )));
        }
    }

    // Copy inputs into contiguous owned buffers so the GIL can be released for
    // the (long) parallel compute. The pole arrays are the large ones; copying
    // 1.38M * 2 f64 (~22 MB) is negligible next to the simulation cost.
    let pole_xy_vec: Vec<[f64; 2]> = xy.rows().into_iter().map(|r| [r[0], r[1]]).collect();
    let ign_vec: Vec<u32> = ign.to_vec();
    let wd_vec: Vec<f64> = wd.to_vec();
    let ws_vec: Vec<f64> = ws.to_vec();
    let fuel_vec: Vec<f64> = fuel_v.to_vec();
    let south_vec: Vec<f64> = south_v.to_vec();

    let result = py.allow_threads(move || {
        simulate_exposure_core(
            &pole_xy_vec,
            &ign_vec,
            &wd_vec,
            &ws_vec,
            &fuel_vec,
            &south_vec,
            max_dist_km,
            length_scale_km,
            wind_aniso,
            southness_beta,
            seed,
        )
    });

    Ok(result.into_pyarray(py))
}

/// Python module definition: exposes `simulate_exposure`.
#[pymodule]
fn pfire_kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_exposure, m)?)?;
    m.add(
        "__doc__",
        "Rust Monte-Carlo downwind wildfire spread kernels.",
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Determinism: identical inputs + seed must give identical output.
    #[test]
    fn deterministic_repeat() {
        let pole_xy = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]];
        let ign = vec![0u32];
        let wd = vec![45.0, 90.0, 200.0];
        let ws = vec![3.0, 6.0, 1.0];
        // Ignition pole (index 0) has fuel = 1.0 and non-negative southness, so
        // its reach_prob = exp(0) * 1.0 * (1 + 0.3 * 0.1) >= 1 → always reached.
        let fuel = vec![1.0, 0.9, 0.5, 0.7];
        let south = vec![0.1, -0.2, 0.3, 0.0];
        let a = simulate_exposure_core(
            &pole_xy, &ign, &wd, &ws, &fuel, &south, 5.0, 0.8, 1.5, 0.3, 42,
        );
        let b = simulate_exposure_core(
            &pole_xy, &ign, &wd, &ws, &fuel, &south, 5.0, 0.8, 1.5, 0.3, 42,
        );
        assert_eq!(a, b);
        // Ignition pole reach_prob >= 1 → reached in every simulation → P == 1.
        assert!((a[0] - 1.0).abs() < 1e-12);
    }

    /// Probabilities must lie in [0, 1] and be 0 beyond the cutoff radius.
    #[test]
    fn bounds_and_cutoff() {
        let pole_xy = vec![[0.0, 0.0], [100.0, 100.0]];
        let ign = vec![0u32];
        let wd = vec![0.0; 16];
        let ws = vec![5.0; 16];
        let fuel = vec![1.0, 1.0];
        let south = vec![0.0, 0.0];
        let p = simulate_exposure_core(
            &pole_xy, &ign, &wd, &ws, &fuel, &south, 5.0, 0.8, 1.5, 0.3, 7,
        );
        for &v in &p {
            assert!((0.0..=1.0).contains(&v));
        }
        // Far pole (distance ~141 km >> 5 km cutoff) never reached.
        assert_eq!(p[1], 0.0);
    }

    /// Degenerate inputs (no ignition sources or no simulations) return an
    /// all-zero exposure vector of length N, never panicking.
    #[test]
    fn empty_inputs_return_zeros() {
        let pole_xy = vec![[0.0, 0.0], [1.0, 1.0]];
        let fuel = vec![1.0, 1.0];
        let south = vec![0.0, 0.0];
        // No ignition sources.
        let p = simulate_exposure_core(
            &pole_xy,
            &[],
            &[0.0; 4],
            &[5.0; 4],
            &fuel,
            &south,
            5.0,
            0.8,
            1.5,
            0.3,
            1,
        );
        assert_eq!(p, vec![0.0, 0.0]);
        // No simulations.
        let p = simulate_exposure_core(
            &pole_xy,
            &[0u32],
            &[],
            &[],
            &fuel,
            &south,
            5.0,
            0.8,
            1.5,
            0.3,
            1,
        );
        assert_eq!(p, vec![0.0, 0.0]);
    }

    /// Anisotropy: under a westerly wind (from-direction 270° → blowing east),
    /// poles downwind (east) are reached more often in aggregate than the
    /// symmetric poles upwind (west), because the reach length L is elongated
    /// downwind. Summed over several mid-range distances to be robust to the
    /// per-pole Monte-Carlo noise of a single distance. Mirrors the Python
    /// `test_exposure_downwind_eastward_under_westerly`.
    #[test]
    fn downwind_exposure_exceeds_upwind() {
        // Ignition at origin; symmetric east/west pairs at several distances.
        // Layout: [ignition, east@1, west@1, east@2, west@2, ...].
        let mut pole_xy = vec![[0.0, 0.0]];
        for &dk in &[1.0_f64, 2.0, 3.0, 4.0] {
            pole_xy.push([dk, 0.0]); // east (downwind)
            pole_xy.push([-dk, 0.0]); // west (upwind)
        }
        let n = pole_xy.len();
        let ign = vec![0u32];
        let wd = vec![270.0; 256]; // from-west → downwind east
        let ws = vec![12.0; 256];
        let fuel = vec![1.0; n];
        let south = vec![0.0; n];
        let p = simulate_exposure_core(
            &pole_xy, &ign, &wd, &ws, &fuel, &south, 5.0, 0.8, 1.5, 0.3, 7,
        );
        let east: f64 = (1..n).step_by(2).map(|i| p[i]).sum();
        let west: f64 = (2..n).step_by(2).map(|i| p[i]).sum();
        assert!(
            east > west,
            "downwind (east) sum {east} should exceed upwind (west) sum {west}"
        );
    }
}
