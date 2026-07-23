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

pub const GRASS_LOD0_T: f32 = 20.0;
pub const GRASS_LOD1_T: f32 = 44.0;
pub const GRASS_LOD2_T: f32 = 90.0;

/// blades x segments per LOD band; a segment is one 6-vertex quad.
pub const LOD_SHAPE: [(u32, u32); 3] = [(56, 5), (30, 3), (14, 1)];

/// Vertices per cell instance in a band.
pub fn verts_per_cell(lod: usize) -> u32 {
    let (blades, segs) = LOD_SHAPE[lod];
    blades * segs * 6
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GrassCell {
    /// World-space position of the blade roots: the top FACE of the grass
    /// block (y = block top). xz is the voxel min corner.
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

    /// Rebuild if the camera strayed >= 2 voxels from the last scan centre.
    /// Returns true when the lists changed.
    /// `y_hint`: explicit column search range for hand-built (lab) worlds
    /// where the terrain-noise height is meaningless; None = terrain hint.
    pub fn maybe_rebuild(&mut self, world: &World, cam: glam::Vec3, y_hint: Option<(i32, i32)>) -> bool {
        let c = (cam.x.floor() as i32, cam.z.floor() as i32);
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
        let r = GRASS_LOD2_T as i32 + 1;
        let r2 = (GRASS_LOD2_T + 1.0) * (GRASS_LOD2_T + 1.0);
        for dz in -r..=r {
            for dx in -r..=r {
                let d2 = (dx * dx + dz * dz) as f32;
                if d2 > r2 {
                    continue;
                }
                let wx = center.0 + dx;
                let wz = center.1 + dz;
                // Terrain height is the search hint; hand-built lab platforms
                // and player edits are caught by the wide +-16 walk.
                let (y0, y1) = match y_hint {
                    Some(r) => r,
                    None => {
                        let h = crate::voxel::sample_terrain(wx as f32 + 0.5, wz as f32 + 0.5, world.seed).h;
                        (h - 16, h + 16)
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
                let dist = d2.sqrt();
                // Dithered band edges so LOD borders never form a ring.
                let jit = (col_seed(wx, wz) & 0xFF) as f32 / 255.0 * 6.0 - 3.0;
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
                    pos: [wx as f32, (gy + 1) as f32, wz as f32],
                    seed: col_seed(wx, wz),
                });
            }
        }
        self.version += 1;
    }
}
