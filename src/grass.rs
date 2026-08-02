//! GPU grass: a Ghost-of-Tsushima-style procedural blade field.
//!
//! The CPU scans grass-top columns around the camera into per-LOD cell
//! lists (one 16-byte record per column - position + seed). Everything
//! else is procedural on the GPU: the vertex shader grows each blade from
//! the cell seed as a wind-bent Bezier ribbon (shaders/grass.wgsl), the
//! fragment shader runs the foliage shading stack and manually depth-tests
//! against the raymarcher's primary-hit depth, exactly like the falling-
//! leaf pass. Drawn BEFORE the TAA resolve so thin blades get temporal
//! anti-aliasing (the technique the whole approach leans on).
//!
//! LOD bands follow the GoT recipe: dense multi-segment blades near, fewer
//! wider blades further out (width compensation keeps coverage), a
//! dithered fade into the grass-top turf shading beyond.

use crate::voxel::{World, MAT_FLOWER, MAT_GRASS, MAT_TALL_GRASS, MAT_TALL_GRASS_DRY};
use crate::world_dims::m_to_vox;

// LOD band radii. How far away the eye can still tell a blade from turf is a
// property of the METRE, not of the grid: written as the bare 20/44/90 they
// were, the whole field would have pulled in to 2/4.4/9 m when the voxel
// went 25 cm -> 10 cm, and the raster grass would have ended well inside
// raymarch.wgsl's GI_MAX_T (22.5 m) instead of exactly at it.
pub const GRASS_LOD0_T: f32 = m_to_vox(5.0); // 5 m
pub const GRASS_LOD1_T: f32 = m_to_vox(11.0); // 11 m
pub const GRASS_LOD2_T: f32 = m_to_vox(22.5); // 22.5 m: SYNC raymarch GI_MAX_T

/// Edge of one grass CELL, in metres. SYNC: `GRASS_CELL_M` in grass.wgsl,
/// which spreads a cell's blade roots across exactly this footprint.
///
/// A cell is ONE instance carrying `lod_shape()[lod].0` blades, so cells per
/// square metre *is* the blade density, the CPU column-scan cost and the
/// renderer's MAX_GRASS_CELLS occupancy, all at once. The scan used to walk
/// one cell per voxel COLUMN, which only meant "0.25 m" because a voxel was
/// 0.25 m. At 10 cm the identical code means 100 cells/m^2 instead of 16:
/// 6.25x the blades, 6.25x the scan, and ~159 k cells against a 32,768-cell
/// buffer - LOD0 would have eaten the entire budget and the mid and far
/// bands would have vanished as a hard ring. Pinning the footprint in metres
/// puts all three back exactly where they were measured.
pub const CELL_M: f32 = 0.25;
/// ...and in voxels (2.5 at 10 cm). Deliberately fractional: the scan walks
/// a CELL lattice and only floors when it needs a voxel to read the ground
/// column from, so the field stays on a 0.25 m grid at any voxel size.
pub const CELL_VOX: f32 = m_to_vox(CELL_M);

/// Half-width of the dithered LOD band edges: +-0.75 m of per-cell jitter so
/// a band border never resolves into a ring.
const BAND_JITTER: f32 = m_to_vox(0.75);

/// Vertical half-window the column walk searches around the terrain-noise
/// height, for hand-built lab platforms and player edits that the noise
/// knows nothing about. 4 m of real terrain deviation, not 16 voxels of it.
const COLUMN_SEARCH: i32 = m_to_vox(4.0) as i32;

/// blades x segments per LOD band; a segment is one 6-vertex quad.
/// Two shape languages share the pipeline: fine spikes (default) and the
/// chunky Hytale-proportioned paddles (VOXELG_GRASS_CHUNKY=1) - fewer,
/// larger, bolder shapes.
pub fn lod_shape() -> [(u32, u32); 3] {
    if std::env::var("VOXELG_GRASS_CHUNKY").is_ok() {
        [(32, 4), (16, 2), (8, 1)]
    } else {
        [(80, 5), (42, 3), (20, 1)]
    }
}

/// Vertices per cell instance in a band.
pub fn verts_per_cell(lod: usize) -> u32 {
    let (blades, segs) = lod_shape()[lod];
    blades * segs * 6
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GrassCell {
    /// World-space position of the blade roots: the top FACE of the grass
    /// block (y = block top). xz is the CELL's min corner; the shader
    /// scatters this cell's roots over the CELL_M x CELL_M footprint from
    /// there.
    pub pos: [f32; 3],
    pub seed: u32,
}

pub struct GrassField {
    pub cells: [Vec<GrassCell>; 3],
    last_center: Option<(i32, i32)>,
    /// Bumped on every rebuild so the renderer knows to re-upload.
    pub version: u64,
}

fn col_seed(wx: i32, wz: i32) -> u32 {
    let x = wx as u32;
    let z = wz as u32;
    let mut v = x.wrapping_mul(1664525).wrapping_add(1013904223);
    v ^= z.wrapping_mul(22695477).wrapping_add(0x9E3779B9);
    v ^= v >> 16;
    v.wrapping_mul(2654435769)
}

fn blocks_blades(above: u8) -> bool {
    // Water (and any transparent fluid) on top: no blades underwater.
    (5..=12).contains(&above)
}

fn coexists(above: u8) -> bool {
    above == 0
        || above == MAT_TALL_GRASS
        || above == MAT_TALL_GRASS_DRY
        || above == MAT_FLOWER
}

impl GrassField {
    pub fn new() -> Self {
        Self { cells: [Vec::new(), Vec::new(), Vec::new()], last_center: None, version: 0 }
    }

    /// Rebuild if the camera strayed >= 2 CELLS (0.5 m, the same real-world
    /// hysteresis the old ">= 2 voxels" bought at 25 cm) from the last scan
    /// centre. Returns true when the lists changed.
    /// `y_hint`: explicit column search range for hand-built (lab) worlds
    /// where the terrain-noise height is meaningless; None = terrain hint.
    pub fn maybe_rebuild(&mut self, world: &World, cam: glam::Vec3, y_hint: Option<(i32, i32)>) -> bool {
        let c = ((cam.x / CELL_VOX).floor() as i32, (cam.z / CELL_VOX).floor() as i32);
        if let Some(lc) = self.last_center {
            if (lc.0 - c.0).abs() < 2 && (lc.1 - c.1).abs() < 2 {
                return false;
            }
        }
        self.last_center = Some(c);
        self.rebuild(world, c, y_hint);
        true
    }

    fn rebuild(&mut self, world: &World, center: (i32, i32), y_hint: Option<(i32, i32)>) {
        for v in &mut self.cells {
            v.clear();
        }
        // The scan walks the CELL lattice, one spare ring past the far band
        // (the old `+1`), so its cost and its output count are fixed in
        // metres instead of exploding with the voxel.
        let r = (GRASS_LOD2_T / CELL_VOX).ceil() as i32 + 1;
        let r2 = (r * r) as f32;
        for dz in -r..=r {
            for dx in -r..=r {
                let d2 = (dx * dx + dz * dz) as f32;
                if d2 > r2 {
                    continue;
                }
                let cx = center.0 + dx;
                let cz = center.1 + dz;
                // The voxel column the cell reads its ground from: the cell's
                // min corner. One sample per 0.25 m cell, exactly what the
                // 25 cm build took when a cell WAS a voxel.
                let wx = (cx as f32 * CELL_VOX).floor() as i32;
                let wz = (cz as f32 * CELL_VOX).floor() as i32;
                // Terrain height is the search hint; hand-built lab platforms
                // and player edits are caught by the wide +-4 m walk.
                let (y0, y1) = match y_hint {
                    Some(r) => r,
                    None => {
                        let h = crate::voxel::sample_terrain(wx as f32 + 0.5, wz as f32 + 0.5, world.seed).h;
                        (h - COLUMN_SEARCH, h + COLUMN_SEARCH)
                    }
                };
                let mut found = None;
                for y in (y0..=y1).rev() {
                    if world.material_at_world(wx, y, wz) == MAT_GRASS {
                        let above = world.material_at_world(wx, y + 1, wz);
                        if blocks_blades(above) {
                            break;
                        }
                        if coexists(above) {
                            found = Some(y);
                        }
                        break;
                    }
                }
                let Some(gy) = found else { continue };
                let dist = d2.sqrt() * CELL_VOX;
                // Dithered band edges so LOD borders never form a ring.
                let jit = ((col_seed(cx, cz) & 0xFF) as f32 / 255.0 * 2.0 - 1.0) * BAND_JITTER;
                let lod = if dist < GRASS_LOD0_T + jit * 0.5 {
                    0
                } else if dist < GRASS_LOD1_T + jit {
                    1
                } else if dist < GRASS_LOD2_T + jit {
                    2
                } else {
                    continue;
                };
                self.cells[lod].push(GrassCell {
                    pos: [cx as f32 * CELL_VOX, (gy + 1) as f32, cz as f32 * CELL_VOX],
                    seed: col_seed(cx, cz),
                });
            }
        }
        self.version += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pull `const <name>: f32 = <literal>;` out of a WGSL source.
    fn wgsl_f32_const(src: &str, name: &str) -> f32 {
        let pat = format!("const {name}: f32 = ");
        let i = src.find(&pat).unwrap_or_else(|| panic!("`{name}` not found in grass.wgsl"));
        let rest = &src[i + pat.len()..];
        let end = rest.find(';').expect("no `;` after the const initialiser");
        rest[..end]
            .trim()
            .parse::<f32>()
            .unwrap_or_else(|e| panic!("`{name}` is not a plain f32 literal: {e}"))
    }

    /// grass.wgsl is the ONE shader assembled without build.rs's world_consts
    /// prelude (renderer::grass_source is `common.wgsl + grass.wgsl`), so it
    /// has to carry its own copy of the voxel scale and of the cell
    /// footprint. Drift in either rescales every blade, clump, wind
    /// wavelength and fade distance in the field by exactly the factor the
    /// metres-not-voxels rewrite exists to prevent - and it would only ever
    /// surface as "the grass looks wrong". Pin both to the Rust originals.
    #[test]
    fn shader_scale_matches_world_dims() {
        let src = include_str!("../shaders/grass.wgsl");
        assert_eq!(
            wgsl_f32_const(src, "VOX_PER_M"),
            crate::world_dims::VOXELS_PER_METRE,
            "grass.wgsl VOX_PER_M drifted from world_dims::VOXELS_PER_METRE"
        );
        assert_eq!(
            wgsl_f32_const(src, "GRASS_CELL_M"),
            CELL_M,
            "grass.wgsl GRASS_CELL_M drifted from grass::CELL_M: the shader \
             would scatter each cell's roots over the wrong footprint"
        );
    }

    /// The scan emits at most one cell per lattice point inside the far
    /// band, and the renderer's cell buffer is a hard MAX_GRASS_CELLS: it
    /// fills LOD0 first and silently truncates, so an overflow does not
    /// degrade, it deletes the mid and far bands as a ring around the
    /// camera. This is the bound that a per-voxel (rather than per-0.25 m)
    /// cell blew by ~5x at 10 cm.
    #[test]
    fn worst_case_cell_count_fits_the_renderer_buffer() {
        let r = (GRASS_LOD2_T / CELL_VOX).ceil() as i32 + 1;
        let r2 = (r * r) as f32;
        let mut n = 0usize;
        for dz in -r..=r {
            for dx in -r..=r {
                if ((dx * dx + dz * dz) as f32) <= r2 {
                    n += 1;
                }
            }
        }
        assert!(
            n <= crate::renderer::MAX_GRASS_CELLS,
            "a fully-grassed scan emits {n} cells, over the {} the renderer \
             can hold",
            crate::renderer::MAX_GRASS_CELLS
        );
    }
}
