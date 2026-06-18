//! Uniform-grid bucket spatial index for fast fixed-radius neighbour queries.
//!
//! Points are binned into square cells whose side length equals the query
//! radius (`max_dist_km`). A fixed-radius query around any point therefore only
//! needs to inspect the 3x3 block of cells centred on the query cell, which
//! turns an O(N) scan per ignition source into an O(local density) scan.
//!
//! The index stores point indices in a CSR-like layout (`cell_start` offsets
//! into a flat `point_ids` array) so that lookups are allocation-free and
//! cache friendly. It is built once and shared (read-only) across all rayon
//! worker threads.

/// Uniform square-cell bucket grid over a set of 2-D points.
pub struct Grid {
    /// Minimum x coordinate (km) of the bounding box.
    min_x: f64,
    /// Minimum y coordinate (km) of the bounding box.
    min_y: f64,
    /// Cell side length (km); equal to the fixed query radius.
    cell: f64,
    /// Number of cells along the x axis.
    nx: usize,
    /// Number of cells along the y axis.
    ny: usize,
    /// CSR offsets: `cell_start[c]..cell_start[c + 1]` indexes `point_ids`.
    cell_start: Vec<u32>,
    /// Flat array of point indices, grouped by cell.
    point_ids: Vec<u32>,
}

impl Grid {
    /// Build a grid over `xy` (row-major `[x0, y0, x1, y1, ...]`) using `cell`
    /// (km) as both the cell size and the intended query radius.
    ///
    /// `cell` is clamped to a small positive value so a degenerate radius never
    /// produces a division by zero or a zero-width grid.
    pub fn build(xy: &[[f64; 2]], cell: f64) -> Self {
        let cell = if cell.is_finite() && cell > 1e-9 {
            cell
        } else {
            1e-9
        };

        let n = xy.len();
        if n == 0 {
            return Self {
                min_x: 0.0,
                min_y: 0.0,
                cell,
                nx: 1,
                ny: 1,
                cell_start: vec![0, 0],
                point_ids: Vec::new(),
            };
        }

        let mut min_x = f64::INFINITY;
        let mut min_y = f64::INFINITY;
        let mut max_x = f64::NEG_INFINITY;
        let mut max_y = f64::NEG_INFINITY;
        for p in xy {
            min_x = min_x.min(p[0]);
            min_y = min_y.min(p[1]);
            max_x = max_x.max(p[0]);
            max_y = max_y.max(p[1]);
        }

        let nx = (((max_x - min_x) / cell).floor() as usize) + 1;
        let ny = (((max_y - min_y) / cell).floor() as usize) + 1;
        let n_cells = nx * ny;

        // Counting sort into CSR layout: first count, then scan to offsets,
        // then scatter point ids into their cell buckets.
        let mut counts = vec![0u32; n_cells + 1];
        let cell_of = |p: &[f64; 2]| -> usize {
            let cx = (((p[0] - min_x) / cell).floor() as usize).min(nx - 1);
            let cy = (((p[1] - min_y) / cell).floor() as usize).min(ny - 1);
            cy * nx + cx
        };

        for p in xy {
            counts[cell_of(p) + 1] += 1;
        }
        for i in 1..=n_cells {
            counts[i] += counts[i - 1];
        }
        let cell_start = counts.clone();

        let mut point_ids = vec![0u32; n];
        let mut cursor = counts;
        for (i, p) in xy.iter().enumerate() {
            let c = cell_of(p);
            point_ids[cursor[c] as usize] = i as u32;
            cursor[c] += 1;
        }

        Self {
            min_x,
            min_y,
            cell,
            nx,
            ny,
            cell_start,
            point_ids,
        }
    }

    /// Invoke `f(point_index)` for every point in the 3x3 cell block centred on
    /// `(x, y)`. This is a *candidate* set: callers must still apply the exact
    /// distance test, since the 3x3 block covers slightly more than the radius.
    #[inline]
    pub fn for_each_candidate<F: FnMut(u32)>(&self, x: f64, y: f64, mut f: F) {
        let cx = ((x - self.min_x) / self.cell).floor();
        let cy = ((y - self.min_y) / self.cell).floor();

        // Clamp the centre cell into range; out-of-range query points still get
        // their nearest border block, which is correct because anything within
        // `cell` of the bounding box edge falls in the border cells.
        let cx = cx.clamp(0.0, (self.nx - 1) as f64) as isize;
        let cy = cy.clamp(0.0, (self.ny - 1) as f64) as isize;

        let x0 = (cx - 1).max(0) as usize;
        let x1 = (cx + 1).min(self.nx as isize - 1) as usize;
        let y0 = (cy - 1).max(0) as usize;
        let y1 = (cy + 1).min(self.ny as isize - 1) as usize;

        for gy in y0..=y1 {
            let row = gy * self.nx;
            for gx in x0..=x1 {
                let c = row + gx;
                let s = self.cell_start[c] as usize;
                let e = self.cell_start[c + 1] as usize;
                for &pid in &self.point_ids[s..e] {
                    f(pid);
                }
            }
        }
    }
}
