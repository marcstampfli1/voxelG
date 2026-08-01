// Bit-packed voxel world with a 3-level u64 hierarchy.
//
// Hierarchy (every level fits in a single u64):
//
//   level   cell-size voxels   one cell stores                   axes
//   ─────   ───────────────    ──────────────────────────────    ─────
//   L1      4³  = 64           u64 = 1 bit per voxel             64×16×64 cells
//   L2      16³ = 4³ bricks    u64 = 1 bit per child brick       16× 4×16 cells
//   L3      64³ = 4³ tiles     u64 = 1 bit per child tile         4× 1× 4 cells
//
// At every level the cell is a 4³ subgrid → exactly one u64. So a single
// bit-test "is this 64³ region of the world empty?" reads 8 bytes; if so we
// skip that whole region during ray traversal.
//
// Within a brick, voxels are ordered (x, z, y) — y is the slowest axis. That
// makes a 4×4 horizontal layer 16 contiguous bits, so falling-sand physics on
// a u64 is `intra = sand & (~occupancy << 16)` — see physics.rs. Tile-in-chunk
// and brick-in-tile linearisations follow the same convention.

use glam::UVec3;

use crossbeam_channel::{Receiver, Sender};

/// A request to a background worker: generate this slot's bricks for this world
/// chunk under this seed. Pure — the worker touches no shared state.
type GenRequest = (u32, glam::IVec3, u64);
/// A finished slot from a worker: (slot, world_chunk, data). The worker also
/// computes the derived per-brick movable + uniform masks and the chunk's 8
/// tile-uniform flags, so the main-thread install is a cheap copy + bit-set
/// instead of re-scanning 64 voxels per brick (that rescan was the chunk-load
/// lag spike — checklist: physics/streaming off the frame thread).
type GenResult = (u32, glam::IVec3, SlotData);

/// A generated storage chunk plus everything the install needs, all computed on
/// the worker thread. `STORAGE_CHUNK_BRICKS³` bricks; `tile_uniform` is the 2³
/// tiles that tile the chunk.
pub struct SlotData {
    pub bricks: Vec<Brick>,
    pub movable: Vec<u64>,
    pub brick_uniform: Vec<u8>,
    pub tile_uniform: [u8; 8],
}

impl SlotData {
    /// Compute the derived masks from freshly generated bricks (pure; runs on
    /// the worker). Brick scratch layout is x + y*8 + z*64 (see gen_slot_bricks).
    pub fn from_bricks(bricks: Vec<Brick>) -> Self {
        let n = bricks.len();
        let mut movable = vec![0u64; n];
        let mut brick_uniform = vec![0u8; n];
        for (i, b) in bricks.iter().enumerate() {
            movable[i] = brick_movable_mask(b);
            brick_uniform[i] = brick_uniform_of(b);
        }
        // tile uniform for the chunk's 2x2x2 tiles (each 4x4x4 child bricks).
        let mut tile_uniform = [0u8; 8];
        let scb = STORAGE_CHUNK_BRICKS as usize; // 8
        for dtz in 0..2usize {
            for dty in 0..2usize {
                for dtx in 0..2usize {
                    let first = (dtx * 4) + (dty * 4) * scb + (dtz * 4) * scb * scb;
                    let m0 = brick_uniform[first];
                    let mut uniform = m0 != 0;
                    if uniform {
                        'scan: for bz in 0..4usize {
                            for by in 0..4usize {
                                for bx in 0..4usize {
                                    let idx = (dtx * 4 + bx) + (dty * 4 + by) * scb + (dtz * 4 + bz) * scb * scb;
                                    if brick_uniform[idx] != m0 {
                                        uniform = false;
                                        break 'scan;
                                    }
                                }
                            }
                        }
                    }
                    tile_uniform[dtx + dty * 2 + dtz * 4] = if uniform { m0 } else { 0 };
                }
            }
        }
        SlotData { bricks, movable, brick_uniform, tile_uniform }
    }
}

// World dimensions live in `src/world_dims.rs` so build.rs can generate the
// matching WGSL constants from the exact same source. Re-export them here so
// every existing `crate::voxel::WORLD_*` reference keeps working unchanged.
pub use crate::world_dims::*;

#[inline(always)]
pub const fn storage_chunk_idx(cx: u32, cy: u32, cz: u32) -> u32 {
    cx + cy * WORLD_STORE_CX + cz * WORLD_STORE_CX * WORLD_STORE_CY
}

#[derive(Clone, Copy)]
pub struct ChunkMeta {
    pub generated: bool,
}

pub const MAT_AIR: u8 = 0;
pub const MAT_SAND: u8 = 1;
pub const MAT_GRASS: u8 = 2;
pub const MAT_DIRT: u8 = 3;
pub const MAT_STONE: u8 = 4;
// 8 water-level variants encode mass per voxel (DwarfCorp-style cellular
// fluid). L8 = a full cell of water (also what set_voxel places); the
// physics step bleeds level into neighbours each tick.
pub const MAT_WATER_L1: u8 = 5;
pub const MAT_WATER_L2: u8 = 6;
pub const MAT_WATER_L3: u8 = 7;
pub const MAT_WATER_L4: u8 = 8;
pub const MAT_WATER_L5: u8 = 9;
pub const MAT_WATER_L6: u8 = 10;
pub const MAT_WATER_L7: u8 = 11;
pub const MAT_WATER_L8: u8 = 12;
pub const MAT_WATER: u8 = MAT_WATER_L8; // alias for callers that just want "full water"
pub const MAT_WOOD: u8 = 13;
pub const MAT_LEAVES: u8 = 14;
pub const MAT_SNOW: u8 = 15;
pub const MAT_LAVA: u8 = 16;
pub const MAT_ICE: u8 = 17;
pub const MAT_GLASS: u8 = 18;
pub const MAT_COAL: u8 = 19;
pub const MAT_IRON: u8 = 20;
pub const MAT_GOLD: u8 = 21;
pub const MAT_DIAMOND: u8 = 22;
pub const MAT_WOOD_BIRCH: u8 = 23;
pub const MAT_WOOD_PINE: u8 = 24;
pub const MAT_LEAVES_BIRCH: u8 = 25;
pub const MAT_LEAVES_PINE: u8 = 26;
pub const MAT_LEAVES_AUTUMN: u8 = 27;
pub const MAT_SMOKE: u8 = 28;
pub const MAT_FIRE: u8 = 29;
pub const MAT_FLOWER: u8 = 30;
pub const MAT_TALL_GRASS: u8 = 31;
pub const MAT_CACTUS: u8 = 32;
/// Invisible canopy-fringe cells painted around tree leaves (radius + 1):
/// they give the raymarcher's DDA a cell to stand in so the big Better
/// Leaves tuft quads that PROTRUDE out of leaf blocks are visible from the
/// side, not only from above. Never rendered as a cube, never casts
/// shadows, skipped by picking.
pub const MAT_LEAF_FRINGE: u8 = 33;
/// Dry straw tuft decoration for sand (desert/savanna) and snow (tundra)
/// tops - same cross-quad renderer as tall grass, its own sprite + palette.
pub const MAT_TALL_GRASS_DRY: u8 = 34;
/// Continuous ground turf: fills the cell above every grass block. Rendered
/// as analytic 3D blade segments near the camera (raymarch turf_blade_hit),
/// invisible past ~48 voxels where the grass-top combed-sheen shading
/// carries the field. Transparent to RT/GI rays (bounce comes from the
/// grass block below); never occludes shadow rays.
pub const MAT_TURF: u8 = 35;
/// Leafy bush decoration: a micro-voxel dome with leaf-cutout faces and
/// oak-tuft crown cards (renderer bush_hit). Scattered on grass tops.
pub const MAT_BUSH: u8 = 36;
/// Tree TEST cells (lab only, no worldgen): an 8x8x8 block of these
/// marches one shared 128^3 wood+leaf volume - real 3D voxel leaves.
pub const MAT_TREE_TEST: u8 = 37;

#[inline(always)]
pub fn is_leaf_mat(m: u8) -> bool {
    m == MAT_LEAVES || m == MAT_LEAVES_BIRCH || m == MAT_LEAVES_PINE || m == MAT_LEAVES_AUTUMN
}
#[inline(always)]
pub fn is_wood_mat(m: u8) -> bool {
    m == MAT_WOOD || m == MAT_WOOD_BIRCH || m == MAT_WOOD_PINE
}
pub const MAX_WATER_LEVEL: u8 = 8;

#[inline(always)]
pub fn is_water_mat(m: u8) -> bool {
    m >= MAT_WATER_L1 && m <= MAT_WATER_L8
}

#[inline(always)]
pub fn is_movable_mat(m: u8) -> bool {
    m == MAT_SAND || is_water_mat(m) || m == MAT_SMOKE
}

#[inline(always)]
pub fn water_level_of(m: u8) -> u8 {
    if is_water_mat(m) { m - MAT_WATER_L1 + 1 } else { 0 }
}

#[inline(always)]
pub fn water_mat_for_level(level: u8) -> u8 {
    if level == 0 { MAT_AIR } else { MAT_WATER_L1 + (level.min(MAX_WATER_LEVEL) - 1) }
}

#[inline(always)]
pub const fn brick_voxel_idx(x: u32, y: u32, z: u32) -> u32 {
    x + z * BRICK_DIM + y * BRICK_DIM * BRICK_DIM
}

#[inline(always)]
pub const fn brick_idx(bx: u32, by: u32, bz: u32) -> u32 {
    bx + by * WORLD_BRICKS_X + bz * WORLD_BRICKS_X * WORLD_BRICKS_Y
}

/// Inverse of [`brick_idx`]: linear brick index → (bx, by, bz). Kept next to
/// the forward fn so the two can't drift.
#[inline(always)]
pub const fn brick_coords(bi: u32) -> (u32, u32, u32) {
    let bx = bi % WORLD_BRICKS_X;
    let by = (bi / WORLD_BRICKS_X) % WORLD_BRICKS_Y;
    let bz = bi / (WORLD_BRICKS_X * WORLD_BRICKS_Y);
    (bx, by, bz)
}

#[inline(always)]
pub const fn tile_idx(tx: u32, ty: u32, tz: u32) -> u32 {
    tx + ty * WORLD_TILES_X + tz * WORLD_TILES_X * WORLD_TILES_Y
}

#[inline(always)]
pub const fn chunk_idx(cx: u32, cy: u32, cz: u32) -> u32 {
    cx + cy * WORLD_CHUNKS_X + cz * WORLD_CHUNKS_X * WORLD_CHUNKS_Y
}

/// Movable-voxel mask for a brick: bit i set iff voxel i is occupied AND its
/// material is movable. Single source for every movable_mask rebuild.
#[inline]
pub fn brick_movable_mask(b: &Brick) -> u64 {
    let mut m = 0u64;
    for i in 0..64usize {
        m |= (is_movable_mat(b.materials[i]) as u64) << i;
    }
    m & b.occupancy
}

/// A brick is uniform iff fully solid and every voxel shares one nonzero
/// material; returns that material (0 = not uniform).
#[inline]
pub fn brick_uniform_of(b: &Brick) -> u8 {
    if b.occupancy != !0u64 {
        return 0;
    }
    let m0 = b.materials[0];
    if m0 == 0 {
        return 0;
    }
    for i in 1..(BRICK_VOXELS as usize) {
        if b.materials[i] != m0 {
            return 0;
        }
    }
    m0
}

/// A tile is uniform iff all 64 child bricks are uniform with the same
/// material. `brick_uniform` is the world's per-brick uniform array; `ti` the
/// linear tile index. Returns that material (0 = not uniform).
#[inline]
pub fn tile_uniform_of(brick_uniform: &[u8], ti: u32) -> u8 {
    let tx = ti % WORLD_TILES_X;
    let ty = (ti / WORLD_TILES_X) % WORLD_TILES_Y;
    let tz = ti / (WORLD_TILES_X * WORLD_TILES_Y);
    let (bx0, by0, bz0) = (tx * 4, ty * 4, tz * 4);
    let m0 = brick_uniform[brick_idx(bx0, by0, bz0) as usize];
    if m0 == 0 {
        return 0;
    }
    for dz in 0..4 {
        for dy in 0..4 {
            for dx in 0..4 {
                if brick_uniform[brick_idx(bx0 + dx, by0 + dy, bz0 + dz) as usize] != m0 {
                    return 0;
                }
            }
        }
    }
    m0
}

#[inline(always)]
pub const fn l4_idx(l4x: u32, l4y: u32, l4z: u32) -> u32 {
    l4x + l4y * WORLD_L4_X + l4z * WORLD_L4_X * WORLD_L4_Y
}

/// Bit position of a child chunk inside its L4 cell's u64 (same x + z*4 + y*16
/// linearisation every level uses).
#[inline(always)]
pub const fn chunk_bit_in_l4(lx: u32, ly: u32, lz: u32) -> u32 {
    lx + lz * 4 + ly * 16
}

#[inline(always)]
pub const fn brick_bit_in_tile(lx: u32, ly: u32, lz: u32) -> u32 {
    lx + lz * 4 + ly * 16
}

#[inline(always)]
pub const fn tile_bit_in_chunk(lx: u32, ly: u32, lz: u32) -> u32 {
    lx + lz * 4 + ly * 16
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Brick {
    pub occupancy: u64,
    pub materials: [u8; BRICK_VOXELS as usize],
}

impl Brick {
    pub const EMPTY: Self = Self {
        occupancy: 0,
        materials: [0; BRICK_VOXELS as usize],
    };

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.occupancy == 0
    }

    /// Every voxel occupied, i.e. the brick holds no air at all. The exact
    /// complement of `is_empty` on the same word, so the two can never disagree
    /// about what "occupied" means.
    #[inline(always)]
    pub fn is_full(&self) -> bool {
        self.occupancy == u64::MAX
    }

    #[inline(always)]
    pub fn set(&mut self, x: u32, y: u32, z: u32, mat: u8) {
        let i = brick_voxel_idx(x, y, z);
        let bit = 1u64 << i;
        if mat == MAT_AIR {
            self.occupancy &= !bit;
            self.materials[i as usize] = 0;
        } else {
            self.occupancy |= bit;
            self.materials[i as usize] = mat;
        }
    }
}

pub struct World {
    pub bricks: Vec<Brick>,
    pub tile_mask: Vec<u64>,
    pub chunk_mask: Vec<u64>,
    /// L4 occupancy: one u64 per 256³-voxel cell, one bit per child chunk. The
    /// coarsest pyramid level — lets the DDA skip a 256³ empty region in a
    /// single bit test (checklist: L4 level).
    pub l4_mask: Vec<u64>,
    pub movable_mask: Vec<u64>,
    /// Per-brick "this whole brick is one material" hint. 0 = not uniform;
    /// any non-zero value = uniform with that material id. Lets the DDA
    /// skip the whole brick in one step instead of walking 4 voxels.
    pub brick_uniform: Vec<u8>,
    /// Per-tile uniform hint (same idea at the 16-voxel scale). When set
    /// the DDA can skip 16 voxels in one step.
    pub tile_uniform: Vec<u8>,
    pub active_bricks: Vec<u32>,
    pub dirty_bricks: Vec<u32>,
    /// Sparse per-voxel light field allocation (docs/VOXEL_LIGHTING_PLAN.md).
    /// Only the brick -> block binding lives here; the light records themselves
    /// are GPU-side and never read back.
    pub light: crate::voxlight::LightField,
    /// Reusable physics scratch buffers (a sorted snapshot of active_bricks and
    /// the per-tick "touched" set), kept here so the CA tick allocates nothing —
    /// previously it cloned active_bricks twice per tick (checklist: physics).
    pub phys_scratch: Vec<u32>,
    pub phys_touched: Vec<u32>,
    pub all_dirty: bool,
    pub chunk_meta: Vec<ChunkMeta>,
    pub seed: u64,
    /// Sliding-window origin in chunk coords (xz only — y axis is fixed).
    /// Voxels stored locally at index `(x, y, z)` correspond to world voxel
    /// `(world_origin.x * 32 + x, y, world_origin.z * 32 + z)`. As the camera
    /// moves the origin shifts and edge chunks regenerate to give "infinite"
    /// terrain. y stays in [0, WORLD_VOXELS_Y).
    pub world_origin_chunk: glam::IVec2,
    /// For each slot, the world chunk coord it currently holds. None = stale.
    pub slot_world_chunk: Vec<Option<glam::IVec3>>,
    /// Tiles whose mask was cleared (slot recycled) without any brick edit, so
    /// the GPU must re-upload them to render that region as sky immediately. The
    /// incremental brick-upload path derives its dirty tiles from dirty_bricks,
    /// which a mask-only clear doesn't touch — hence this side list.
    pub mask_dirty_tiles: Vec<u32>,
    /// Async chunk generation. shift_origin sends (slot, world_chunk, seed) to a
    /// pool of background worker threads; install_finished_chunks pulls finished
    /// bricks back and stitches them in on the main thread. This keeps the
    /// expensive noise generation off the frame thread — the chunk-load hitch.
    gen_req_tx: Sender<GenRequest>,
    gen_res_rx: Receiver<GenResult>,
    /// Outstanding (requested but not yet received) generation jobs.
    in_flight: usize,
    /// Persistent voxel edits keyed by *world* voxel coord. Survives chunk
    /// unload/regen — applied on top of fresh noise when a chunk reloads,
    /// and synced over the network so all clients agree on player builds.
    pub edits: std::collections::HashMap<(i32, i32, i32), u8>,
}

impl World {
    pub fn new() -> Self {
        Self::with_seed(0xC0FFEE_F00D_BEEFu64)
    }

    pub fn with_seed(seed: u64) -> Self {
        // Spawn a small pool of generation workers. Each pulls (slot, chunk,
        // seed) jobs and pushes back finished bricks; gen_slot_bricks is a pure
        // function so there is no shared state and no locking.
        let (gen_req_tx, gen_req_rx) = crossbeam_channel::unbounded::<GenRequest>();
        let (gen_res_tx, gen_res_rx) = crossbeam_channel::unbounded::<GenResult>();
        let n_workers = std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(2))
            .unwrap_or(4)
            .clamp(2, 6);
        for _ in 0..n_workers {
            let rx = gen_req_rx.clone();
            let tx = gen_res_tx.clone();
            std::thread::Builder::new()
                .name("chunkgen".into())
                .spawn(move || {
                    while let Ok((slot, world_chunk, seed)) = rx.recv() {
                        let bricks = gen_slot_bricks(world_chunk, seed);
                        // Compute the derived masks here, off the frame thread.
                        let data = SlotData::from_bricks(bricks);
                        if tx.send((slot, world_chunk, data)).is_err() {
                            break;
                        }
                    }
                })
                .expect("spawn chunkgen worker");
        }
        Self {
            bricks: vec![Brick::EMPTY; WORLD_BRICKS_TOTAL as usize],
            tile_mask: vec![0u64; WORLD_TILES_TOTAL as usize],
            chunk_mask: vec![0u64; WORLD_CHUNKS_TOTAL as usize],
            l4_mask: vec![0u64; WORLD_L4_TOTAL as usize],
            movable_mask: vec![0u64; WORLD_BRICKS_TOTAL as usize],
            brick_uniform: vec![0u8; WORLD_BRICKS_TOTAL as usize],
            tile_uniform: vec![0u8; WORLD_TILES_TOTAL as usize],
            active_bricks: Vec::with_capacity(4096),
            dirty_bricks: Vec::with_capacity(4096),
            light: crate::voxlight::LightField::new(crate::voxlight::LIGHT_BLOCKS_MAX),
            phys_scratch: Vec::with_capacity(4096),
            phys_touched: Vec::with_capacity(8192),
            all_dirty: true,
            chunk_meta: vec![ChunkMeta { generated: false }; WORLD_STORE_CHUNKS as usize],
            seed,
            world_origin_chunk: glam::IVec2::ZERO,
            slot_world_chunk: vec![None; WORLD_STORE_CHUNKS as usize],
            mask_dirty_tiles: Vec::with_capacity(2048),
            gen_req_tx,
            gen_res_rx,
            in_flight: 0,
            edits: std::collections::HashMap::new(),
        }
    }

    /// World-voxel offset of the loaded window's lower corner.
    pub fn world_origin_voxel(&self) -> glam::IVec3 {
        glam::IVec3::new(
            self.world_origin_chunk.x * STORAGE_CHUNK_VOXELS as i32,
            0,
            self.world_origin_chunk.y * STORAGE_CHUNK_VOXELS as i32,
        )
    }

    /// Record a persistent edit at WORLD-voxel coords and (if it's currently
    /// inside the loaded window) apply it locally. The edit map drives
    /// replay-on-regen so builds survive crossing the chunk-streaming edge.
    pub fn apply_edit(&mut self, wx: i32, wy: i32, wz: i32, mat: u8) {
        self.edits.insert((wx, wy, wz), mat);
        let origin = self.world_origin_voxel();
        // Bounds: only apply locally if the world voxel is inside the loaded
        // window (relative to origin).
        let rel_x = wx - origin.x;
        let rel_y = wy - origin.y;
        let rel_z = wz - origin.z;
        if rel_x < 0 || rel_y < 0 || rel_z < 0
            || (rel_x as u32) >= WORLD_VOXELS_X
            || (rel_y as u32) >= WORLD_VOXELS_Y
            || (rel_z as u32) >= WORLD_VOXELS_Z
        {
            return;
        }
        // Storage is TOROIDAL — the GPU shader maps world voxels to slots via
        // `pos_mod(wx, WORLD_VOXELS_X)`. Naive `wx - origin.x` only matches
        // that when origin == 0; for any other origin (player walked away
        // from spawn) we'd write to the wrong brick. Use the same mapping
        // the shader uses.
        let lx = wx.rem_euclid(WORLD_VOXELS_X as i32) as u32;
        let ly = rel_y as u32;
        let lz = wz.rem_euclid(WORLD_VOXELS_Z as i32) as u32;
        self.set_voxel(lx, ly, lz, mat);
    }

    /// Target origin chunk-coord for a camera at the given world position.
    /// Centres the loaded window on the camera.
    pub fn target_origin_chunk(camera_world: glam::Vec3) -> glam::IVec2 {
        let cam_cx = (camera_world.x / STORAGE_CHUNK_VOXELS as f32).floor() as i32;
        let cam_cz = (camera_world.z / STORAGE_CHUNK_VOXELS as f32).floor() as i32;
        let half_x = (WORLD_STORE_CX as i32) / 2;
        let half_z = (WORLD_STORE_CZ as i32) / 2;
        glam::IVec2::new(cam_cx - half_x, cam_cz - half_z)
    }

    /// Local voxel coords for `world_voxel`, given the current origin.
    /// Returns None if `world_voxel` is outside the loaded window.
    pub fn world_to_local(&self, world_voxel: glam::IVec3) -> Option<glam::UVec3> {
        let origin_vox = glam::IVec3::new(
            self.world_origin_chunk.x * STORAGE_CHUNK_VOXELS as i32,
            0,
            self.world_origin_chunk.y * STORAGE_CHUNK_VOXELS as i32,
        );
        let local = world_voxel - origin_vox;
        if local.x < 0 || local.y < 0 || local.z < 0
            || local.x >= WORLD_VOXELS_X as i32
            || local.y >= WORLD_VOXELS_Y as i32
            || local.z >= WORLD_VOXELS_Z as i32 { return None; }
        Some(glam::UVec3::new(local.x as u32, local.y as u32, local.z as u32))
    }

    /// Shift the sliding window using TOROIDAL slot indexing. A slot at
    /// store-coord `(sx, _, sz)` represents the world chunk in the loaded
    /// window whose `mod WORLD_STORE_*` equals `(sx, sz)` — so a +1 origin
    /// shift only invalidates the single column of slots that just dropped
    /// out of the window.
    pub fn shift_origin(&mut self, new_origin: glam::IVec2) {
        if new_origin == self.world_origin_chunk { return; }
        self.world_origin_chunk = new_origin;
        let store_x = WORLD_STORE_CX as i32;
        let store_z = WORLD_STORE_CZ as i32;
        for cz in 0..WORLD_STORE_CZ {
            for cy in 0..WORLD_STORE_CY {
                for cx in 0..WORLD_STORE_CX {
                    // For slot (cx, cy, cz), the world chunk currently in
                    // the window with `wc mod store == slot` is computed via
                    // the offset from origin's mod.
                    let want_x = new_origin.x + (cx as i32 - new_origin.x).rem_euclid(store_x);
                    let want_z = new_origin.y + (cz as i32 - new_origin.y).rem_euclid(store_z);
                    let want = glam::IVec3::new(want_x, cy as i32, want_z);
                    let slot = storage_chunk_idx(cx, cy, cz) as usize;
                    if self.slot_world_chunk[slot] != Some(want) {
                        // Render this slot as SKY immediately by clearing only its
                        // mask bits (cheap — no brick zeroing, no 4.7 MB clear
                        // upload). The stale brick data is simply never read while
                        // the tile bits are 0, and the async install overwrites it.
                        self.clear_slot_masks(cx, cy, cz);
                        self.slot_world_chunk[slot] = Some(want);
                        if self.gen_req_tx.send((slot as u32, want, self.seed)).is_ok() {
                            self.in_flight += 1;
                        }
                    }
                }
            }
        }
    }

    /// Number of chunk-generation jobs still in flight on the worker pool.
    pub fn pending_gen(&self) -> usize {
        self.in_flight
    }

    /// Install up to `budget` finished chunks from the worker pool onto the main
    /// thread (the cheap stitch + edit replay). This is the per-frame upload
    /// budget: capping installs caps how many bricks get marked dirty (and thus
    /// uploaded) per frame, so a chunk cross streams in smoothly over several
    /// frames instead of spiking. Returns the number installed.
    pub fn install_finished_chunks(&mut self, budget: u32) -> u32 {
        let mut installed = 0u32;
        while installed < budget {
            let (slot, want, data) = match self.gen_res_rx.try_recv() {
                Ok(r) => r,
                Err(_) => break,
            };
            self.in_flight = self.in_flight.saturating_sub(1);
            // Discard stale results: the origin may have shifted again while
            // this slot was generating, reassigning it to a different chunk.
            if self.slot_world_chunk[slot as usize] != Some(want) {
                continue;
            }
            self.install_slot(slot, want, &data);
            installed += 1;
        }
        installed
    }

    /// Block until every queued + in-flight generation job has been processed.
    /// Used by tests and "must be fully loaded now" paths; the per-frame loop
    /// uses the budgeted `install_finished_chunks` instead.
    pub fn process_pending_gen_blocking(&mut self) {
        while self.in_flight > 0 {
            let (slot, want, data) = match self.gen_res_rx.recv() {
                Ok(r) => r,
                Err(_) => break,
            };
            self.in_flight = self.in_flight.saturating_sub(1);
            if self.slot_world_chunk[slot as usize] == Some(want) {
                self.install_slot(slot, want, &data);
            }
        }
    }

    fn install_slot(&mut self, slot: u32, want: glam::IVec3, data: &SlotData) {
        let cx = slot % WORLD_STORE_CX;
        let cy = (slot / WORLD_STORE_CX) % WORLD_STORE_CY;
        let cz = slot / (WORLD_STORE_CX * WORLD_STORE_CY);
        self.apply_slot_data(cx, cy, cz, data);
        self.replay_edits_for_chunk(want);
    }

    /// Replay persistent edits that fall inside `world_chunk` on top of freshly
    /// generated terrain, so player builds survive the streaming round-trip.
    fn replay_edits_for_chunk(&mut self, world_chunk: glam::IVec3) {
        if self.edits.is_empty() {
            return;
        }
        let cv = STORAGE_CHUNK_VOXELS as i32;
        let origin = self.world_origin_voxel();
        // Collect first so we don't hold a borrow on self.edits across set_voxel.
        let mut to_apply: Vec<(i32, i32, i32, u8)> = Vec::new();
        for (&(wx, wy, wz), &mat) in &self.edits {
            if wx.div_euclid(cv) == world_chunk.x
                && wy.div_euclid(cv) == world_chunk.y
                && wz.div_euclid(cv) == world_chunk.z
            {
                to_apply.push((wx, wy, wz, mat));
            }
        }
        for (wx, wy, wz, mat) in to_apply {
            let rel_x = wx - origin.x;
            let rel_y = wy - origin.y;
            let rel_z = wz - origin.z;
            if rel_x < 0 || rel_y < 0 || rel_z < 0
                || (rel_x as u32) >= WORLD_VOXELS_X
                || (rel_y as u32) >= WORLD_VOXELS_Y
                || (rel_z as u32) >= WORLD_VOXELS_Z
            {
                continue;
            }
            let lx = wx.rem_euclid(WORLD_VOXELS_X as i32) as u32;
            let ly = rel_y as u32;
            let lz = wz.rem_euclid(WORLD_VOXELS_Z as i32) as u32;
            self.set_voxel(lx, ly, lz, mat);
        }
    }

    /// Make a recycled slot render as SKY immediately, cheaply: clear only its
    /// hierarchy MASK bits (tile/chunk/L4) + tile-uniform + the slot's movable
    /// bits. The stale brick voxel data is left untouched (never read while the
    /// tile bits are 0) and is overwritten when the async install lands — so
    /// this does no brick zeroing and no big brick upload (it queues just the
    /// touched tiles for a tiny mask upload). active_bricks may keep stale
    /// entries but physics skips them (movable == 0).
    fn clear_slot_masks(&mut self, slot_cx: u32, slot_cy: u32, slot_cz: u32) {
        let base_bx = slot_cx * STORAGE_CHUNK_BRICKS;
        let base_by = slot_cy * STORAGE_CHUNK_BRICKS;
        let base_bz = slot_cz * STORAGE_CHUNK_BRICKS;
        // Stop physics touching the slot's now-hidden bricks, and hand the
        // slot's light blocks back to the pool. The block records are left as
        // they are: nothing maps to them while unbound, and rebinding queues a
        // reset, so recycling a slot never touches GPU memory.
        for dz in 0..STORAGE_CHUNK_BRICKS {
            for dy in 0..STORAGE_CHUNK_BRICKS {
                for dx in 0..STORAGE_CHUNK_BRICKS {
                    let bi = brick_idx(base_bx + dx, base_by + dy, base_bz + dz);
                    self.movable_mask[bi as usize] = 0;
                    self.light.release(bi);
                }
            }
        }
        // Clear the slot's 2x2x2 tiles and propagate empties up to chunk + L4.
        let base_tx = base_bx / 4;
        let base_ty = base_by / 4;
        let base_tz = base_bz / 4;
        for dtz in 0..2u32 {
            for dty in 0..2u32 {
                for dtx in 0..2u32 {
                    let tx = base_tx + dtx;
                    let ty = base_ty + dty;
                    let tz = base_tz + dtz;
                    if tx >= WORLD_TILES_X || ty >= WORLD_TILES_Y || tz >= WORLD_TILES_Z {
                        continue;
                    }
                    let ti = tile_idx(tx, ty, tz);
                    if self.tile_mask[ti as usize] == 0 {
                        continue; // already empty
                    }
                    self.tile_mask[ti as usize] = 0;
                    self.tile_uniform[ti as usize] = 0;
                    self.mask_dirty_tiles.push(ti);
                    // Clear this tile's bit in its chunk; if the chunk empties,
                    // clear its L4 bit too.
                    let (cx, cy, cz) = (tx / 4, ty / 4, tz / 4);
                    let ci = chunk_idx(cx, cy, cz);
                    let cprev = self.chunk_mask[ci as usize];
                    self.chunk_mask[ci as usize] &= !(1u64 << tile_bit_in_chunk(tx & 3, ty & 3, tz & 3));
                    if cprev != 0 && self.chunk_mask[ci as usize] == 0 {
                        let li = l4_idx(cx / 4, cy / 4, cz / 4);
                        self.l4_mask[li as usize] &= !(1u64 << chunk_bit_in_l4(cx & 3, cy & 3, cz & 3));
                    }
                }
            }
        }
    }

    /// Install a worker-computed slot (bricks + precomputed masks) into the flat
    /// world arrays. The movable / brick-uniform / tile-uniform values were all
    /// computed on the worker (SlotData::from_bricks), so the main thread only
    /// copies + sets mask bits — no 64-voxel rescans. This is what removes the
    /// chunk-load lag spike.
    fn apply_slot_data(&mut self, slot_cx: u32, slot_cy: u32, slot_cz: u32, data: &SlotData) {
        let base_bx = slot_cx * STORAGE_CHUNK_BRICKS;
        let base_by = slot_cy * STORAGE_CHUNK_BRICKS;
        let base_bz = slot_cz * STORAGE_CHUNK_BRICKS;
        for dz in 0..STORAGE_CHUNK_BRICKS {
            for dy in 0..STORAGE_CHUNK_BRICKS {
                for dx in 0..STORAGE_CHUNK_BRICKS {
                    let idx =
                        (dx + dy * STORAGE_CHUNK_BRICKS + dz * STORAGE_CHUNK_BRICKS * STORAGE_CHUNK_BRICKS)
                            as usize;
                    let bx = base_bx + dx;
                    let by = base_by + dy;
                    let bz = base_bz + dz;
                    let bi = brick_idx(bx, by, bz);
                    self.bricks[bi as usize] = data.bricks[idx];
                    self.brick_uniform[bi as usize] = data.brick_uniform[idx];
                    self.set_movable(bi, data.movable[idx]);
                    self.refresh_masks_for_brick(bx, by, bz);
                    self.mark_brick_dirty(bi);
                }
            }
        }
        // Tile-uniform flags were precomputed for the chunk's 2x2x2 tiles.
        let base_tx = base_bx / 4;
        let base_ty = base_by / 4;
        let base_tz = base_bz / 4;
        for dtz in 0..2u32 {
            for dty in 0..2u32 {
                for dtx in 0..2u32 {
                    let tx = base_tx + dtx;
                    let ty = base_ty + dty;
                    let tz = base_tz + dtz;
                    if tx < WORLD_TILES_X && ty < WORLD_TILES_Y && tz < WORLD_TILES_Z {
                        let local = (dtx + dty * 2 + dtz * 4) as usize;
                        self.tile_uniform[tile_idx(tx, ty, tz) as usize] = data.tile_uniform[local];
                    }
                }
            }
        }
    }

    /// Regenerate one slot. Slot must already be cleared (by `clear_slot`).
    pub fn regenerate_slot(&mut self, slot_cx: u32, slot_cy: u32, slot_cz: u32, world_chunk: glam::IVec3, seed: u64) {
        let sea_level: u32 = 64;
        let (s_x, s_z) = seed_offset_xz(seed);

        let local_x0 = slot_cx * STORAGE_CHUNK_VOXELS;
        let local_y0 = slot_cy * STORAGE_CHUNK_VOXELS;
        let local_z0 = slot_cz * STORAGE_CHUNK_VOXELS;
        let world_x0 = world_chunk.x * STORAGE_CHUNK_VOXELS as i32;
        let world_y0 = world_chunk.y * STORAGE_CHUNK_VOXELS as i32;
        let world_z0 = world_chunk.z * STORAGE_CHUNK_VOXELS as i32;

        // First, clear all bricks in this slot.
        for dz in 0..STORAGE_CHUNK_BRICKS {
            for dy in 0..STORAGE_CHUNK_BRICKS {
                for dx in 0..STORAGE_CHUNK_BRICKS {
                    let bx = local_x0 / BRICK_DIM + dx;
                    let by = local_y0 / BRICK_DIM + dy;
                    let bz = local_z0 / BRICK_DIM + dz;
                    let bi = brick_idx(bx, by, bz) as usize;
                    self.bricks[bi] = Brick::EMPTY;
                    self.movable_mask[bi] = 0;
                }
            }
        }

        // Generate using WORLD coords for noise so neighbouring chunks line
        // up seamlessly.
        for dz in 0..STORAGE_CHUNK_VOXELS {
            for dx in 0..STORAGE_CHUNK_VOXELS {
                let wx = (world_x0 + dx as i32) as f32 + s_x;
                let wz = (world_z0 + dz as i32) as f32 + s_z;
                let hilly     = fbm_2d(wx * 0.007, wz * 0.007, 4);
                let mountains = fbm_2d(wx * 0.0025, wz * 0.0025, 3).max(0.0).powf(1.6) * 1.8;
                let detail    = fbm_2d(wx * 0.04, wz * 0.04, 2) * 0.4;
                let h_signed = (sea_level as f32 + 14.0 + hilly * 22.0 + mountains * 40.0 + detail * 3.0)
                    .clamp(2.0, (WORLD_VOXELS_Y - 1) as f32) as i32;
                // Lower-frequency biome noise → larger, less-jumpy biomes.
                let temperature = fbm_2d(wx * 0.0006, wz * 0.0006, 3);
                let humidity    = fbm_2d(wx * 0.0008 + 100.0, wz * 0.0008 + 100.0, 3);
                let biome = pick_biome(temperature, humidity, h_signed as u32, sea_level);
                let local_x = local_x0 + dx;
                let local_z = local_z0 + dz;

                for dy in 0..STORAGE_CHUNK_VOXELS {
                    let world_y = world_y0 + dy as i32;
                    let local_y = local_y0 + dy;
                    if world_y < 0 || world_y >= WORLD_VOXELS_Y as i32 { continue; }
                    if world_y > h_signed { break; }
                    let cn  = value_noise_3d(wx * 0.045, world_y as f32 * 0.085, wz * 0.045);
                    let cn2 = value_noise_3d(wx * 0.110, world_y as f32 * 0.060, wz * 0.110);
                    if world_y > 4 && world_y + 3 < h_signed && (cn + cn2 * 0.6) > 0.30 { continue; }
                    let h_u32 = h_signed as u32;
                    let mat = if world_y as u32 >= h_u32 {
                        biome.top_block(h_u32, sea_level)
                    } else if (world_y as u32) + 4 >= h_u32 {
                        biome.subsoil()
                    } else {
                        stone_or_ore(wx, world_y as f32, wz, h_u32)
                    };
                    self.write_voxel_unchecked(local_x, local_y, local_z, mat);
                }
                // Sea fill
                if (h_signed as u32) < sea_level {
                    for wy in ((h_signed + 1) as i32)..=(sea_level as i32) {
                        if wy < world_y0 || wy >= world_y0 + STORAGE_CHUNK_VOXELS as i32 { continue; }
                        let local_y = local_y0 + (wy - world_y0) as u32;
                        if local_y < WORLD_VOXELS_Y {
                            self.write_voxel_unchecked(local_x, local_y, local_z, MAT_WATER);
                        }
                    }
                }
            }
        }

        // CRITICAL: write_voxel_unchecked bypasses every bookkeeping mask, so
        // we must rebuild them per-brick now. Without this the shader's
        // hierarchical DDA sees the chunk as empty (tile/chunk bits unset)
        // and renders sky, even though the bricks have content on the CPU.
        for dz in 0..STORAGE_CHUNK_BRICKS {
            for dy in 0..STORAGE_CHUNK_BRICKS {
                for dx in 0..STORAGE_CHUNK_BRICKS {
                    let bx = local_x0 / BRICK_DIM + dx;
                    let by = local_y0 / BRICK_DIM + dy;
                    let bz = local_z0 / BRICK_DIM + dz;
                    let bi = brick_idx(bx, by, bz);
                    let new_movable = brick_movable_mask(&self.bricks[bi as usize]);
                    self.set_movable(bi, new_movable);
                    self.refresh_masks_for_brick(bx, by, bz);
                    self.mark_brick_dirty(bi);
                }
            }
        }
    }

    /// Insert `bi` into the sorted `active_bricks` list if absent. Single
    /// keeper of the "active_bricks stays sorted" invariant that the rest of
    /// the engine relies on for `binary_search`.
    #[inline]
    pub fn mark_active(&mut self, bi: u32) {
        if let Err(pos) = self.active_bricks.binary_search(&bi) {
            self.active_bricks.insert(pos, bi);
        }
    }

    /// Remove `bi` from the sorted `active_bricks` list if present.
    #[inline]
    pub fn unmark_active(&mut self, bi: u32) {
        if let Ok(pos) = self.active_bricks.binary_search(&bi) {
            self.active_bricks.remove(pos);
        }
    }

    /// Set `movable_mask[bi]` and keep `active_bricks` in sync with the
    /// empty↔movable transition.
    #[inline]
    pub fn set_movable(&mut self, bi: u32, new_movable: u64) {
        let was_movable = self.movable_mask[bi as usize] != 0;
        self.movable_mask[bi as usize] = new_movable;
        let is_movable = new_movable != 0;
        if was_movable != is_movable {
            if is_movable {
                self.mark_active(bi);
            } else {
                self.unmark_active(bi);
            }
        }
    }

    pub fn recompute_movable_for_brick(&mut self, bi: u32) {
        let new_mask = brick_movable_mask(&self.bricks[bi as usize]);
        self.set_movable(bi, new_mask);
    }

    pub fn rebuild_active_bricks(&mut self) {
        self.active_bricks.clear();
        for (i, m) in self.movable_mask.iter().enumerate() {
            if *m != 0 {
                self.active_bricks.push(i as u32);
            }
        }
        // Already in ascending order because we walk indices in order.
    }

    pub fn mark_brick_dirty(&mut self, bi: u32) {
        if !self.all_dirty {
            self.dirty_bricks.push(bi);
        }
    }

    /// Bring light-field block bindings in line with the bricks that changed.
    ///
    /// Called ONCE PER FRAME from the renderer over the existing dirty list,
    /// deliberately NOT from `mark_brick_dirty`: physics calls that for every
    /// touched brick every tick, and running the shell test there would tax the
    /// CPU hot path to compute something consumed only once, at draw time.
    pub fn sync_light_shell_dirty(&mut self) {
        // Move the list out so the shell walk can take &mut self, then put it
        // back: the caller still needs it for the brick upload. Moving a Vec
        // costs nothing, so this stays allocation-free.
        let changed = std::mem::take(&mut self.dirty_bricks);
        for &bi in &changed {
            // A brick's own change can flip its NEIGHBOURS' shell membership
            // too: the empty brick above a surface joins the shell the moment
            // that surface appears. Evaluate the closed neighbourhood.
            self.eval_light_shell(bi);
            let (nb, n) = self.brick_neighbours(bi);
            for k in 0..n {
                self.eval_light_shell(nb[k]);
            }
            // The geometry under this brick's light moved, so whatever it had
            // accumulated is stale.
            self.light.invalidate(bi);
        }
        self.dirty_bricks = changed;
    }

    /// Rebind the whole world's light shell. Used on a full-dirty frame (world
    /// init, teleport), where every brick is effectively new.
    pub fn sync_light_shell_all(&mut self) {
        for bi in 0..WORLD_BRICKS_TOTAL {
            self.eval_light_shell(bi);
        }
    }

    fn eval_light_shell(&mut self, bi: u32) {
        if self.brick_needs_light(bi) {
            self.light.allocate(bi);
        } else {
            self.light.release(bi);
        }
    }

    /// A brick needs light storage when it can hold air next to solid: it holds
    /// at least one AIR voxel, and either it is non-empty (so it holds both) or
    /// it touches a non-empty brick (the open air directly above a surface).
    /// Conservative by one brick on the air side, which is exactly what keeps
    /// the sampler's eight-tap neighbourhood populated right at a surface
    /// instead of falling off the edge of the allocated region.
    ///
    /// A FULLY SOLID brick is excluded, and that exclusion is the difference
    /// between a shell and a volume. Light lives in air: the update pass writes
    /// epoch 0 for every solid voxel and `voxlight_sample` drops every tap that
    /// lands in one, so a block bound to a brick with no air holds 64 records
    /// that nothing can ever read - while still costing 512 bytes of pool and a
    /// workgroup of update work every time its slice of the work list comes
    /// round. On the demo world that is 310,545 of the 370,719 bricks the old
    /// rule asked for (measured), i.e. five sixths of the storage and of the
    /// update dispatch, and it is what made the pool overflow by 3x and strand
    /// whole cameras on the per-pixel fallback.
    ///
    /// Excluding them changes nothing the sampler can observe: every voxel of a
    /// full brick is solid, so every tap into it was already dropped. That is an
    /// argument, so it was also measured - benchmarking both rules side by side
    /// moves every FIELD A/B delta by at most 0.05 ms while saving 134 MB of
    /// pool and 0.22 ms of update per frame (docs/rt/BASELINE-per-voxel-
    /// lighting.md, round D).
    fn brick_needs_light(&self, bi: u32) -> bool {
        let brick = &self.bricks[bi as usize];
        if brick.is_full() {
            return false;
        }
        if !brick.is_empty() {
            return true;
        }
        let (nb, n) = self.brick_neighbours(bi);
        for k in 0..n {
            if !self.bricks[nb[k] as usize].is_empty() {
                return true;
            }
        }
        false
    }

    /// The face-adjacent storage bricks. x/z wrap toroidally (the storage
    /// window is periodic on those axes); y is clamped, so a brick at the world
    /// floor or ceiling simply reports fewer neighbours.
    fn brick_neighbours(&self, bi: u32) -> ([u32; 6], usize) {
        let (bx, by, bz) = brick_coords(bi);
        let wrap = |v: i64, m: u32| -> u32 { v.rem_euclid(m as i64) as u32 };
        let mut out = [0u32; 6];
        let mut n = 0usize;
        out[n] = brick_idx(wrap(bx as i64 + 1, WORLD_BRICKS_X), by, bz);
        n += 1;
        out[n] = brick_idx(wrap(bx as i64 - 1, WORLD_BRICKS_X), by, bz);
        n += 1;
        out[n] = brick_idx(bx, by, wrap(bz as i64 + 1, WORLD_BRICKS_Z));
        n += 1;
        out[n] = brick_idx(bx, by, wrap(bz as i64 - 1, WORLD_BRICKS_Z));
        n += 1;
        if by + 1 < WORLD_BRICKS_Y {
            out[n] = brick_idx(bx, by + 1, bz);
            n += 1;
        }
        if by > 0 {
            out[n] = brick_idx(bx, by - 1, bz);
            n += 1;
        }
        (out, n)
    }

    /// Refresh tile/chunk bits for a brick after the brick's occupancy may
    /// have changed. Called by physics and by set_voxel().
    pub fn refresh_masks_for_brick(&mut self, bx: u32, by: u32, bz: u32) {
        let bi = brick_idx(bx, by, bz);
        let solid = !self.bricks[bi as usize].is_empty();
        let (tx, ty, tz) = (bx / 4, by / 4, bz / 4);
        let ti = tile_idx(tx, ty, tz);
        let bit = brick_bit_in_tile(bx & 3, by & 3, bz & 3);
        let prev = self.tile_mask[ti as usize];
        if solid {
            self.tile_mask[ti as usize] |= 1u64 << bit;
        } else {
            self.tile_mask[ti as usize] &= !(1u64 << bit);
        }
        let now = self.tile_mask[ti as usize];
        if (prev == 0) != (now == 0) {
            let (cx, cy, cz) = (tx / 4, ty / 4, tz / 4);
            let ci = chunk_idx(cx, cy, cz);
            let cbit = tile_bit_in_chunk(tx & 3, ty & 3, tz & 3);
            let cprev = self.chunk_mask[ci as usize];
            if now == 0 {
                self.chunk_mask[ci as usize] &= !(1u64 << cbit);
            } else {
                self.chunk_mask[ci as usize] |= 1u64 << cbit;
            }
            let cnow = self.chunk_mask[ci as usize];
            // Propagate a chunk empty↔non-empty transition up to the L4 level.
            if (cprev == 0) != (cnow == 0) {
                let li = l4_idx(cx / 4, cy / 4, cz / 4);
                let lbit = chunk_bit_in_l4(cx & 3, cy & 3, cz & 3);
                if cnow == 0 {
                    self.l4_mask[li as usize] &= !(1u64 << lbit);
                } else {
                    self.l4_mask[li as usize] |= 1u64 << lbit;
                }
            }
        }
    }

    pub fn set_voxel(&mut self, x: u32, y: u32, z: u32, mat: u8) {
        if x >= WORLD_VOXELS_X || y >= WORLD_VOXELS_Y || z >= WORLD_VOXELS_Z {
            return;
        }
        let (bx, by, bz) = (x / BRICK_DIM, y / BRICK_DIM, z / BRICK_DIM);
        let (lx, ly, lz) = (x % BRICK_DIM, y % BRICK_DIM, z % BRICK_DIM);
        let bi = brick_idx(bx, by, bz);
        let was_empty = self.bricks[bi as usize].is_empty();
        self.bricks[bi as usize].set(lx, ly, lz, mat);
        let is_empty = self.bricks[bi as usize].is_empty();
        if was_empty != is_empty {
            self.refresh_masks_for_brick(bx, by, bz);
        }
        self.recompute_movable_for_brick(bi);
        self.recompute_uniform_for_brick(bi);
        // The tile this brick lives in may have lost its uniform status.
        let ti = tile_idx(bx / 4, by / 4, bz / 4);
        self.recompute_uniform_for_tile(ti);
        self.mark_brick_dirty(bi);
    }

    /// Recompute brick_uniform[bi] from the brick's current contents.
    pub fn recompute_uniform_for_brick(&mut self, bi: u32) {
        self.brick_uniform[bi as usize] = brick_uniform_of(&self.bricks[bi as usize]);
    }

    /// Recompute tile_uniform[ti] from its 64 child bricks. Tile is uniform
    /// iff every child brick is uniform with the same material.
    pub fn recompute_uniform_for_tile(&mut self, ti: u32) {
        self.tile_uniform[ti as usize] = tile_uniform_of(&self.brick_uniform, ti);
    }

    /// Recompute ALL uniform flags from current brick contents. Use after
    /// bulk gen / fill_demo_terrain. O(total_voxels) — runs in parallel.
    pub fn rebuild_all_uniform(&mut self) {
        use rayon::prelude::*;
        let bricks = &self.bricks;
        self.brick_uniform = bricks.par_iter().map(brick_uniform_of).collect();
        // Tiles depend on the brick_uniform array we just computed.
        let bu = &self.brick_uniform;
        self.tile_uniform = (0..WORLD_TILES_TOTAL as u32)
            .into_par_iter()
            .map(|ti| tile_uniform_of(bu, ti))
            .collect();
    }

    pub fn rebuild_all_masks(&mut self) {
        self.tile_mask.iter_mut().for_each(|m| *m = 0);
        self.chunk_mask.iter_mut().for_each(|m| *m = 0);
        self.l4_mask.iter_mut().for_each(|m| *m = 0);
        for bz in 0..WORLD_BRICKS_Z {
            for by in 0..WORLD_BRICKS_Y {
                for bx in 0..WORLD_BRICKS_X {
                    if !self.bricks[brick_idx(bx, by, bz) as usize].is_empty() {
                        let (tx, ty, tz) = (bx / 4, by / 4, bz / 4);
                        let ti = tile_idx(tx, ty, tz) as usize;
                        self.tile_mask[ti] |= 1u64 << brick_bit_in_tile(bx & 3, by & 3, bz & 3);
                    }
                }
            }
        }
        for tz in 0..WORLD_TILES_Z {
            for ty in 0..WORLD_TILES_Y {
                for tx in 0..WORLD_TILES_X {
                    let ti = tile_idx(tx, ty, tz) as usize;
                    if self.tile_mask[ti] != 0 {
                        let (cx, cy, cz) = (tx / 4, ty / 4, tz / 4);
                        let ci = chunk_idx(cx, cy, cz) as usize;
                        self.chunk_mask[ci] |= 1u64 << tile_bit_in_chunk(tx & 3, ty & 3, tz & 3);
                    }
                }
            }
        }
        for cz in 0..WORLD_CHUNKS_Z {
            for cy in 0..WORLD_CHUNKS_Y {
                for cx in 0..WORLD_CHUNKS_X {
                    let ci = chunk_idx(cx, cy, cz) as usize;
                    if self.chunk_mask[ci] != 0 {
                        let li = l4_idx(cx / 4, cy / 4, cz / 4) as usize;
                        self.l4_mask[li] |= 1u64 << chunk_bit_in_l4(cx & 3, cy & 3, cz & 3);
                    }
                }
            }
        }
    }

    /// Top-level demo generation: walks every storage chunk and generates it
    /// (terrain + ores + sea + trees). Trees place into neighbour chunks so
    /// the tree pass runs after the terrain pass for the whole world.
    pub fn fill_demo_terrain(&mut self) {
        use rayon::prelude::*;
        let seed = self.seed;
        // Parallel slot generation — 1024 chunks otherwise = 10s wall time
        // serial. Rayon parallel + serial merge brings this well under 1s.
        let total_slots = (WORLD_STORE_CX * WORLD_STORE_CY * WORLD_STORE_CZ) as usize;
        let scratches: Vec<(usize, glam::IVec3, Vec<Brick>)> = (0..total_slots)
            .into_par_iter()
            .map(|i| {
                let i = i as u32;
                let slot_cx = i % WORLD_STORE_CX;
                let slot_cy = (i / WORLD_STORE_CX) % WORLD_STORE_CY;
                let slot_cz = i / (WORLD_STORE_CX * WORLD_STORE_CY);
                let world_chunk = glam::IVec3::new(slot_cx as i32, slot_cy as i32, slot_cz as i32);
                let scratch = gen_slot_bricks(world_chunk, seed);
                (i as usize, world_chunk, scratch)
            })
            .collect();
        for (i, world_chunk, scratch) in scratches {
            let i = i as u32;
            let slot_cx = i % WORLD_STORE_CX;
            let slot_cy = (i / WORLD_STORE_CX) % WORLD_STORE_CY;
            let slot_cz = i / (WORLD_STORE_CX * WORLD_STORE_CY);
            let base_bx = slot_cx * STORAGE_CHUNK_BRICKS;
            let base_by = slot_cy * STORAGE_CHUNK_BRICKS;
            let base_bz = slot_cz * STORAGE_CHUNK_BRICKS;
            for db_z in 0..STORAGE_CHUNK_BRICKS {
                for db_y in 0..STORAGE_CHUNK_BRICKS {
                    for db_x in 0..STORAGE_CHUNK_BRICKS {
                        let scratch_idx = (db_x
                            + db_y * STORAGE_CHUNK_BRICKS
                            + db_z * STORAGE_CHUNK_BRICKS * STORAGE_CHUNK_BRICKS)
                            as usize;
                        let bi = brick_idx(base_bx + db_x, base_by + db_y, base_bz + db_z);
                        self.bricks[bi as usize] = scratch[scratch_idx];
                    }
                }
            }
            let slot_idx = storage_chunk_idx(slot_cx, slot_cy, slot_cz) as usize;
            self.slot_world_chunk[slot_idx] = Some(world_chunk);
        }

        self.rebuild_all_masks();
        self.rebuild_all_uniform();
        for bi in 0..WORLD_BRICKS_TOTAL {
            self.movable_mask[bi as usize] = brick_movable_mask(&self.bricks[bi as usize]);
        }
        self.rebuild_active_bricks();
        self.all_dirty = true;
        for cm in self.chunk_meta.iter_mut() { cm.generated = true; }
        // Initial slot ↔ world chunk mapping (origin starts at 0).
        for cz in 0..WORLD_STORE_CZ {
            for cy in 0..WORLD_STORE_CY {
                for cx in 0..WORLD_STORE_CX {
                    let slot = storage_chunk_idx(cx, cy, cz) as usize;
                    self.slot_world_chunk[slot] = Some(glam::IVec3::new(cx as i32, cy as i32, cz as i32));
                }
            }
        }
    }

    /// Generate one 32×32×32-voxel storage chunk. Idempotent — if the chunk
    /// is already generated it returns immediately. World gen calls this in
    /// a triple-loop, but the hook is here for future on-demand streaming.
    pub fn generate_chunk_terrain(&mut self, cx: u32, cy: u32, cz: u32, seed: u64) {
        let ci = storage_chunk_idx(cx, cy, cz) as usize;
        if self.chunk_meta[ci].generated { return; }
        let sea_level: u32 = 64;
        let (s_x, s_z) = seed_offset_xz(seed);

        let x0 = cx * STORAGE_CHUNK_VOXELS;
        let y0 = cy * STORAGE_CHUNK_VOXELS;
        let z0 = cz * STORAGE_CHUNK_VOXELS;
        let x1 = (x0 + STORAGE_CHUNK_VOXELS).min(WORLD_VOXELS_X);
        let y1 = (y0 + STORAGE_CHUNK_VOXELS).min(WORLD_VOXELS_Y);
        let z1 = (z0 + STORAGE_CHUNK_VOXELS).min(WORLD_VOXELS_Z);

        for z in z0..z1 {
            for x in x0..x1 {
                let fx = x as f32 + s_x;
                let fz = z as f32 + s_z;
                let hilly     = fbm_2d(fx * 0.007, fz * 0.007, 4);
                let mountains = fbm_2d(fx * 0.0025, fz * 0.0025, 3).max(0.0).powf(1.6) * 1.8;
                let detail    = fbm_2d(fx * 0.04, fz * 0.04, 2) * 0.4;
                let h = (sea_level as f32 + 14.0 + hilly * 22.0 + mountains * 40.0 + detail * 3.0)
                    .clamp(2.0, (WORLD_VOXELS_Y - 1) as f32) as u32;
                let temperature = fbm_2d(fx * 0.0006, fz * 0.0006, 3);
                let humidity    = fbm_2d(fx * 0.0008 + 100.0, fz * 0.0008 + 100.0, 3);
                let biome = pick_biome(temperature, humidity, h, sea_level);
                let y_top = y1.min(h + 1);
                for y in y0..y_top {
                    let cn  = value_noise_3d(fx * 0.045, y as f32 * 0.085, fz * 0.045);
                    let cn2 = value_noise_3d(fx * 0.110, y as f32 * 0.060, fz * 0.110);
                    if y > 4 && y + 3 < h && (cn + cn2 * 0.6) > 0.30 { continue; }
                    let mat = if y >= h {
                        biome.top_block(h, sea_level)
                    } else if y + 4 >= h {
                        biome.subsoil()
                    } else {
                        stone_or_ore(fx, y as f32, fz, h)
                    };
                    self.write_voxel_unchecked(x, y, z, mat);
                }
                // Sea fill within this chunk's Y range.
                if h < sea_level && y0 <= sea_level {
                    let yy0 = (h + 1).max(y0);
                    let yy1 = sea_level.min(y1.saturating_sub(1));
                    for y in yy0..=yy1 {
                        self.write_voxel_unchecked(x, y, z, MAT_WATER);
                    }
                }
            }
        }
        self.chunk_meta[ci].generated = true;
    }

    #[inline]
    fn write_voxel_unchecked(&mut self, x: u32, y: u32, z: u32, mat: u8) {
        let (bx, by, bz) = (x / BRICK_DIM, y / BRICK_DIM, z / BRICK_DIM);
        let (lx, ly, lz) = (x % BRICK_DIM, y % BRICK_DIM, z % BRICK_DIM);
        let bi = brick_idx(bx, by, bz);
        self.bricks[bi as usize].set(lx, ly, lz, mat);
    }

    pub fn dims_voxels(&self) -> UVec3 {
        UVec3::new(WORLD_VOXELS_X, WORLD_VOXELS_Y, WORLD_VOXELS_Z)
    }

    /// Material at a WORLD voxel coord using the same toroidal slot mapping the
    /// shader and CPU raycaster use. Returns MAT_AIR if empty or outside the
    /// loaded window. Single source of truth for "what's at this world voxel".
    pub fn material_at_world(&self, wx: i32, wy: i32, wz: i32) -> u8 {
        let origin = self.world_origin_voxel();
        let rel = glam::IVec3::new(wx - origin.x, wy - origin.y, wz - origin.z);
        if rel.x < 0 || rel.y < 0 || rel.z < 0
            || rel.x as u32 >= WORLD_VOXELS_X
            || rel.y as u32 >= WORLD_VOXELS_Y
            || rel.z as u32 >= WORLD_VOXELS_Z
        {
            return MAT_AIR;
        }
        let sx = wx.rem_euclid(WORLD_VOXELS_X as i32) as u32;
        let sy = rel.y as u32;
        let sz = wz.rem_euclid(WORLD_VOXELS_Z as i32) as u32;
        let bi = brick_idx(sx / BRICK_DIM, sy / BRICK_DIM, sz / BRICK_DIM) as usize;
        let b = &self.bricks[bi];
        let vi = brick_voxel_idx(sx % BRICK_DIM, sy % BRICK_DIM, sz % BRICK_DIM);
        if (b.occupancy & (1u64 << vi)) == 0 {
            return MAT_AIR;
        }
        b.materials[vi as usize]
    }
}

/// Decompose a 64-bit world seed into a pair of (x, z) float offsets used to
/// shift noise queries. Different seeds → different terrain by sampling a
/// different region of the same infinite noise field.
#[inline(always)]
pub fn seed_offset_xz(seed: u64) -> (f32, f32) {
    let hi = ((seed >> 32) as u32) as i32 as f32;
    let lo = ((seed & 0xFFFF_FFFF) as u32) as i32 as f32;
    (hi * 0.01734, lo * 0.02153)
}

// ---------- value noise ----------

#[inline(always)]
fn hash2(x: i32, z: i32) -> f32 {
    let h = (x as u32)
        .wrapping_mul(0x9E3779B1)
        .wrapping_add((z as u32).wrapping_mul(0x85EBCA77));
    let h = h.wrapping_mul(0xC2B2AE3D);
    let h = h ^ (h >> 16);
    let h = h.wrapping_mul(0x85EBCA6B);
    ((h & 0xFFFFFF) as f32) / (0xFFFFFF as f32) * 2.0 - 1.0
}

#[inline(always)]
fn hash3(x: i32, y: i32, z: i32) -> f32 {
    let h = (x as u32)
        .wrapping_mul(0x9E3779B1)
        .wrapping_add((y as u32).wrapping_mul(0x85EBCA77))
        .wrapping_add((z as u32).wrapping_mul(0xC2B2AE3D));
    let h = h.wrapping_mul(0xD2B74407);
    let h = h ^ (h >> 16);
    let h = h.wrapping_mul(0x85EBCA6B);
    ((h & 0xFFFFFF) as f32) / (0xFFFFFF as f32) * 2.0 - 1.0
}

#[inline(always)]
fn smoothstep(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

pub fn value_noise_2d(x: f32, z: f32) -> f32 {
    let xi = x.floor() as i32;
    let zi = z.floor() as i32;
    let xf = smoothstep(x - xi as f32);
    let zf = smoothstep(z - zi as f32);
    let v00 = hash2(xi, zi);
    let v10 = hash2(xi + 1, zi);
    let v01 = hash2(xi, zi + 1);
    let v11 = hash2(xi + 1, zi + 1);
    let a = v00 * (1.0 - xf) + v10 * xf;
    let b = v01 * (1.0 - xf) + v11 * xf;
    a * (1.0 - zf) + b * zf
}

/// Ridge noise: 1 - |fbm|. Output in [0, 1] with thin "ridge" lines along
/// the fbm = 0 contours. Used for rivers + ravines.
pub fn ridge_noise_2d(x: f32, z: f32) -> f32 {
    let n = fbm_2d(x, z, 4);
    (1.0 - n.abs()).clamp(0.0, 1.0)
}

pub fn fbm_2d(x: f32, z: f32, octaves: u32) -> f32 {
    let mut total = 0.0;
    let mut amp = 1.0;
    let mut freq = 1.0;
    let mut max_amp = 0.0;
    for _ in 0..octaves {
        total += value_noise_2d(x * freq, z * freq) * amp;
        max_amp += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    total / max_amp
}

/// Per-column terrain sample. `water_top` is the y level the topmost water
/// voxel reaches (0 = no water). A river fills its carved channel up to
/// 1 voxel above the surrounding terrain so the player sees a brimming
/// stream, not a sliver at the bottom of a ditch.
pub struct TerrainSample {
    pub h: i32,
    pub water_top: i32,
    pub is_river: bool,
}

pub fn sample_terrain(wx: f32, wz: f32, seed: u64) -> TerrainSample {
    let (s_x, s_z) = seed_offset_xz(seed);
    let px = wx + s_x;
    let pz = wz + s_z;

    let warp_x = fbm_2d(px * 0.005, pz * 0.005, 2) * 8.0;
    let warp_z = fbm_2d(px * 0.005 + 50.0, pz * 0.005 + 50.0, 2) * 8.0;
    let wpx = px + warp_x;
    let wpz = pz + warp_z;

    // Hills — strong amplitude so terrain is genuinely rolling.
    let base = fbm_2d(wpx * 0.012, wpz * 0.012, 4) * 22.0;

    // Mountains — huge (amp 110), more common (mask shifted +0.2 so most of
    // the map has at least some elevation contribution; peaks reach the
    // world's roof).
    let mountain_mask = (fbm_2d(wpx * 0.0028, wpz * 0.0028, 2) + 0.2).max(0.0);
    let mountain_amp = mountain_mask.min(1.0);
    let mountain_h = fbm_2d(wpx * 0.009, wpz * 0.009, 5).max(0.0).powf(1.15)
        * mountain_amp * 110.0;

    // Ravines DISABLED. They use the same ridge-noise mechanism as rivers and
    // carve thin winding channels that dip below sea level and fill with water,
    // reading as rivers. Cut forced to 0; uncomment the two lines to restore.
    // let ravine_n = ridge_noise_2d(wpx * 0.012, wpz * 0.012);
    // let ravine_cut = ((ravine_n - 0.97).max(0.0) * 20.0).min(1.0) * 6.0;
    let ravine_cut = 0.0_f32;

    // Sea level raised (38 → 64) AND world ceiling doubled (192 → 256) so
    // lakes/seas have real depth and mountains still loom above with
    // headroom. base_h offset stays at 8 so terrain typically sits ~72,
    // ~8 voxels above sea, with mountains pushing well into the 200s.
    let sea_level: f32 = 64.0;
    let base_h = sea_level + 8.0 + base + mountain_h - ravine_cut;

    // Rivers DISABLED. The noise-blend river carve is kept below (commented)
    // so it can be restored: uncomment the block, then derive `h` from
    // `h_blended` and `is_river` from `actual_strength` as in the bottom two
    // lines of the block. With it off, terrain comes straight from `base_h`
    // and no column is ever flagged a river (oceans still fill via water_top).
    /*
    // Rivers via SMOOTH BLEND with strict low-elevation gating. Rivers only
    // appear where terrain is naturally near sea level; they smoothly blend
    // the bed down so water (always at sea_level) shows in the channel.
    let river_n = ridge_noise_2d(px * 0.0050 + 1000.0, pz * 0.0050 + 1000.0);
    let river_strength_raw = ((river_n - 0.85) / 0.15).clamp(0.0, 1.0);
    let elevation_above_sea = (base_h - sea_level).max(0.0);
    // Hard cutoff at +8 vox above sea — full strength up to +4, linear
    // fade-out from +4 to +8, zero past that. No rivers on hills.
    let elevation_fade = if elevation_above_sea < 4.0 {
        1.0
    } else if elevation_above_sea < 8.0 {
        1.0 - (elevation_above_sea - 4.0) * 0.25
    } else {
        0.0
    };
    let actual_strength = river_strength_raw * elevation_fade;

    // Blend terrain DOWN toward bed_target. At full strength terrain reaches
    // sea_level - 3 → 3 voxels of water. At river edges it tapers back to
    // base_h naturally.
    let bed_target = sea_level - 3.0;
    let h_blended = base_h * (1.0 - actual_strength) + bed_target * actual_strength;
    */
    let h = base_h.clamp(2.0, (WORLD_VOXELS_Y - 1) as f32);
    let h_i = h as i32;
    let is_river = false;

    // Single GLOBAL water level. Anywhere terrain dips below sea_level (ocean
    // or river) fills with water to sea_level. Cannot overflow because every
    // water cell shares the same surface.
    let water_top = if h_i < sea_level as i32 { sea_level as i32 } else { 0 };
    TerrainSample { h: h_i, water_top, is_river }
}

/// Pure function: produce one storage chunk's worth of bricks from a world
/// chunk coord + seed. No shared state — safe to call from rayon workers.
/// Returns 512 bricks in (x, y, z) order with x innermost.
pub fn gen_slot_bricks(world_chunk: glam::IVec3, seed: u64) -> Vec<Brick> {
    let total = (STORAGE_CHUNK_BRICKS * STORAGE_CHUNK_BRICKS * STORAGE_CHUNK_BRICKS) as usize;
    let mut bricks: Vec<Brick> = vec![Brick::EMPTY; total];
    let sea_level: u32 = 64;
    let (s_x, s_z) = seed_offset_xz(seed);
    let world_x0 = world_chunk.x * STORAGE_CHUNK_VOXELS as i32;
    let world_y0 = world_chunk.y * STORAGE_CHUNK_VOXELS as i32;
    let world_z0 = world_chunk.z * STORAGE_CHUNK_VOXELS as i32;

    for dz in 0..STORAGE_CHUNK_VOXELS {
        for dx in 0..STORAGE_CHUNK_VOXELS {
            let wx_int = world_x0 + dx as i32;
            let wz_int = world_z0 + dz as i32;
            let wx = wx_int as f32;
            let wz = wz_int as f32;
            let ts = sample_terrain(wx, wz, seed);
            let h_signed = ts.h;
            let h_u32 = h_signed as u32;

            // Skip the whole column if it's entirely below or above this
            // chunk's Y range AND has no water that reaches into our range.
            let col_top = h_signed.max(ts.water_top);
            let col_bottom = 0;
            if col_top < world_y0 || col_bottom >= world_y0 + STORAGE_CHUNK_VOXELS as i32 { continue; }

            let temperature = fbm_2d((wx + s_x) * 0.0006, (wz + s_z) * 0.0006, 3);
            let humidity = fbm_2d((wx + s_x) * 0.0008 + 100.0, (wz + s_z) * 0.0008 + 100.0, 3);
            let biome = pick_biome(temperature, humidity, h_u32, sea_level);

            // Compute the Y range that actually overlaps this chunk to skip
            // iterating Y values above terrain (was iterating empty air).
            let y_start = world_y0.max(0);
            let y_end = (world_y0 + STORAGE_CHUNK_VOXELS as i32).min(WORLD_VOXELS_Y as i32);
            // Seal the top 5 voxels below a water column so caves don't
            // perforate the river/lake bed and let the water drain into
            // them. Caves are still allowed deeper underground.
            let has_water_above = ts.water_top > h_signed;
            let cave_seal_y = if has_water_above { h_signed - 5 } else { i32::MIN };
            for world_y in y_start..y_end {
                if world_y > h_signed { break; }
                let in_water_seal = world_y >= cave_seal_y;
                if !in_water_seal {
                    let cn = value_noise_3d(wx * 0.045, world_y as f32 * 0.085, wz * 0.045);
                    let cn2 = value_noise_3d(wx * 0.110, world_y as f32 * 0.060, wz * 0.110);
                    if world_y > 4 && world_y + 3 < h_signed && (cn + cn2 * 0.6) > 0.30 { continue; }
                }
                let mat = if ts.is_river && world_y as u32 >= h_u32 {
                    MAT_SAND
                } else if world_y as u32 >= h_u32 {
                    biome.top_block(h_u32, sea_level)
                } else if (world_y as u32) + 4 >= h_u32 {
                    biome.subsoil()
                } else {
                    stone_or_ore(wx, world_y as f32, wz, h_u32)
                };
                let dy = (world_y - world_y0) as u32;
                write_into_scratch(&mut bricks, dx, dy, dz, mat);
            }

            // Water fill — rivers brim above their carved banks, ocean fills
            // any column with terrain below sea level.
            if ts.water_top > h_signed {
                let fill_top = ts.water_top;
                let fill_bottom = (h_signed + 1).max(world_y0);
                let fill_end = fill_top.min(world_y0 + STORAGE_CHUNK_VOXELS as i32 - 1);
                if fill_bottom <= fill_end {
                    for wy in fill_bottom..=fill_end {
                        let dy = (wy - world_y0) as u32;
                        write_into_scratch(&mut bricks, dx, dy, dz, MAT_WATER);
                    }
                }
            }

            // Surface decoration: tall grass + flowers on grass tops (with
            // clustered flower meadows), dry straw tufts on desert/savanna
            // sand and tundra snow. Never over water; beaches stay clean.
            if ts.water_top == 0 && (h_signed as u32) < WORLD_VOXELS_Y - 1 {
                let surface_top = biome.top_block(h_u32, sea_level);
                let dec_y = h_signed + 1;
                if dec_y >= world_y0 && dec_y < world_y0 + STORAGE_CHUNK_VOXELS as i32 {
                    let h = hash3(wx_int, dec_y, wz_int);
                    let v = h * 0.5 + 0.5; // 0..1
                    let dec_mat = match surface_top {
                        MAT_GRASS => {
                            let (mut fp, mut gp) = biome.flora_probs();
                            // Meadow patches: low-frequency noise clusters the
                            // flowers into wildflower fields instead of a
                            // uniform sprinkle.
                            let meadow = fbm_2d(
                                wx_int as f32 * 0.02 + 7.0,
                                wz_int as f32 * 0.02 - 3.0,
                                2,
                            ) > 0.30;
                            if meadow {
                                // Tussock patches: meadow regions grow
                                // clustered tuft fields, elsewhere stays
                                // sparse - patches, not a carpet.
                                fp *= 3.0;
                                gp *= 12.0;
                            }
                            if v > 1.0 - fp {
                                MAT_FLOWER
                            } else if v > 1.0 - fp - gp {
                                MAT_TALL_GRASS
                            } else {
                                // Bushes: rare scattered singles, plus BIG
                                // 2x2x2 bushes anchored to the even lattice
                                // - every column of an anchor block derives
                                // the same decision, so the four columns
                                // assemble one super-bush (renderer marches
                                // one shared dome across the block).
                                let ax = wx_int & !1;
                                let az = wz_int & !1;
                                let big = hash3(ax, 977, az) * 0.5 + 0.5 > 0.9955;
                                if big {
                                    MAT_BUSH
                                } else if v > 1.0 - fp - gp - 0.004 {
                                    MAT_BUSH
                                } else {
                                    0u8
                                }
                            }
                        }
                        MAT_SAND if matches!(biome, Biome::Desert | Biome::Savanna) => {
                            if v > 0.99 { MAT_TALL_GRASS_DRY } else { 0u8 }
                        }
                        MAT_SNOW if matches!(biome, Biome::Tundra) => {
                            if v > 0.995 { MAT_TALL_GRASS_DRY } else { 0u8 }
                        }
                        _ => 0u8,
                    };
                    if dec_mat != 0 {
                        let dy = (dec_y - world_y0) as u32;
                        write_into_scratch(&mut bricks, dx, dy, dz, dec_mat);
                        // Big bushes are two cells tall: the aligned block
                        // spans 2x2x2 and the renderer marches one dome.
                        if dec_mat == MAT_BUSH {
                            let ax = wx_int & !1;
                            let az = wz_int & !1;
                            let big = hash3(ax, 977, az) * 0.5 + 0.5 > 0.9955;
                            if big && dec_y + 1 >= world_y0
                                && dec_y + 1 < world_y0 + STORAGE_CHUNK_VOXELS as i32
                            {
                                write_into_scratch(&mut bricks, dx, dy + 1, dz, MAT_BUSH);
                            }
                        }
                    }
                }
            }
        }
    }

    // ---------- TREE PASS ----------
    // Trees with their *base* in this chunk or any of the 8 xz neighbours.
    // Small trees (canopy radius ≤ 5 vox), so a 1-chunk scan covers them.
    // We write straight into the brick scratch — no per-voxel allocation.
    let chunk_min = (world_x0, world_y0, world_z0);
    let chunk_max = (
        world_x0 + STORAGE_CHUNK_VOXELS as i32,
        world_y0 + STORAGE_CHUNK_VOXELS as i32,
        world_z0 + STORAGE_CHUNK_VOXELS as i32,
    );
    for ncz in -1..=1i32 {
        for ncx in -1..=1i32 {
            let src_chunk = glam::IVec2::new(world_chunk.x + ncx, world_chunk.z + ncz);
            let trees = trees_for_chunk(src_chunk, seed, sea_level);
            for tree in trees {
                // Vertical overlap rejection.
                let tree_top = tree.base_y + 22;
                if tree.base_y > chunk_max.1 || tree_top < chunk_min.1 { continue; }
                paint_tree(&tree, &mut bricks, chunk_min, chunk_max);
            }
        }
    }

    bricks
}

#[inline(always)]
fn try_write_tree_voxel(
    bricks: &mut [Brick],
    wx: i32, wy: i32, wz: i32, mat: u8,
    cmin: (i32, i32, i32), cmax: (i32, i32, i32),
) {
    if wx < cmin.0 || wx >= cmax.0 { return; }
    if wy < cmin.1 || wy >= cmax.1 { return; }
    if wz < cmin.2 || wz >= cmax.2 { return; }
    let dx = (wx - cmin.0) as u32;
    let dy = (wy - cmin.1) as u32;
    let dz = (wz - cmin.2) as u32;
    let bb_x = dx / BRICK_DIM;
    let bb_y = dy / BRICK_DIM;
    let bb_z = dz / BRICK_DIM;
    let bb_idx = (bb_x + bb_y * STORAGE_CHUNK_BRICKS
        + bb_z * STORAGE_CHUNK_BRICKS * STORAGE_CHUNK_BRICKS) as usize;
    let vi = brick_voxel_idx(dx % BRICK_DIM, dy % BRICK_DIM, dz % BRICK_DIM);
    // Trees never overwrite existing solid voxels (terrain wins) — except
    // canopy fringe, which real tree parts always replace.
    let occupied = (bricks[bb_idx].occupancy & (1u64 << vi)) != 0;
    let is_fringe = occupied && bricks[bb_idx].materials[vi as usize] == MAT_LEAF_FRINGE;
    if !occupied || (is_fringe && mat != MAT_LEAF_FRINGE) {
        bricks[bb_idx].set(dx % BRICK_DIM, dy % BRICK_DIM, dz % BRICK_DIM, mat);
    }
}

#[derive(Clone, Copy)]
struct TreeSpec {
    base_x: i32,
    base_y: i32,
    base_z: i32,
    ttype: u32,
    hash: u32,
}

/// Deterministic tree positions for a given (xz) chunk.
fn trees_for_chunk(chunk_xz: glam::IVec2, seed: u64, sea_level: u32) -> Vec<TreeSpec> {
    let (s_x, s_z) = seed_offset_xz(seed);
    // Climate at chunk centre — coarse enough that whole forests stay in
    // the same biome.
    let cx_center = (chunk_xz.x as f32 + 0.5) * STORAGE_CHUNK_VOXELS as f32;
    let cz_center = (chunk_xz.y as f32 + 0.5) * STORAGE_CHUNK_VOXELS as f32;
    let temperature = fbm_2d((cx_center + s_x) * 0.0006, (cz_center + s_z) * 0.0006, 3);
    let humidity = fbm_2d((cx_center + s_x) * 0.0008 + 100.0,
                           (cz_center + s_z) * 0.0008 + 100.0, 3);
    let biome = pick_biome(temperature, humidity, sea_level + 10, sea_level);
    let density = biome.tree_density();
    let n_candidates = (density * 5.0).round() as u32;
    if n_candidates == 0 { return Vec::new(); }

    // Patch noise — clearings AND dense thickets within the same biome (gives
    // trees grove/glade clustering). Cacti want an EVEN scatter, so deserts
    // skip it and use a flat multiplier so every desert chunk gets the same
    // count instead of thicket-and-clearing clumps.
    let patch_mul = if matches!(biome, Biome::Desert) {
        1.0
    } else {
        let patch_raw = fbm_2d(cx_center * 0.0035, cz_center * 0.0035, 2);
        ((patch_raw + 0.4).max(0.0) * 1.6).min(2.0)
    };
    let n = ((n_candidates as f32) * patch_mul) as u32;
    if n == 0 { return Vec::new(); }

    let chunk_hash = hash_chunk(chunk_xz.x, chunk_xz.y, seed);
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n {
        let h = chunk_hash.wrapping_mul(2654435761).wrapping_add(i.wrapping_mul(7919));
        let dx = (h & 31) as i32;
        let dz = ((h >> 5) & 31) as i32;
        let wx = chunk_xz.x * STORAGE_CHUNK_VOXELS as i32 + dx;
        let wz = chunk_xz.y * STORAGE_CHUNK_VOXELS as i32 + dz;
        let ts = sample_terrain(wx as f32, wz as f32, seed);
        if ts.is_river || (ts.h as u32) <= sea_level + 1 { continue; }
        if ts.h + 22 >= WORLD_VOXELS_Y as i32 { continue; }
        let h_terrain = ts.h;
        let local_t = fbm_2d((wx as f32 + s_x) * 0.0006, (wz as f32 + s_z) * 0.0006, 3);
        let local_h = fbm_2d((wx as f32 + s_x) * 0.0008 + 100.0,
                             (wz as f32 + s_z) * 0.0008 + 100.0, 3);
        let local_biome = pick_biome(local_t, local_h, h_terrain as u32, sea_level);
        let ttype = match local_biome {
            // Sandy / arid biomes read as desert — no leafy trees there.
            Biome::Beach | Biome::Savanna => continue,
            Biome::Desert => 4, // cacti, scattered across the desert
            _ => local_biome.tree_type(h),
        };
        out.push(TreeSpec { base_x: wx, base_y: h_terrain + 1, base_z: wz, ttype, hash: h });
    }
    out
}

fn hash_chunk(cx: i32, cz: i32, seed: u64) -> u32 {
    let s = (seed as u32) ^ ((seed >> 32) as u32);
    let mut h = (cx as u32).wrapping_mul(0x9E3779B1);
    h = h.wrapping_add((cz as u32).wrapping_mul(0x85EBCA77));
    h = h.wrapping_add(s.wrapping_mul(0xC2B2AE3D));
    h = h ^ (h >> 16);
    h = h.wrapping_mul(0xD2B74407);
    h ^ (h >> 13)
}

// ---------- branched-tree generator ----------
// Writes the tree's voxels DIRECTLY into the brick scratch — no per-voxel
// allocation, no sort. The bounds check is the only overhead per voxel.

fn paint_tree(
    t: &TreeSpec,
    bricks: &mut [Brick],
    cmin: (i32, i32, i32),
    cmax: (i32, i32, i32),
) {
    let base = glam::IVec3::new(t.base_x, t.base_y, t.base_z);
    let h = t.hash;
    match t.ttype {
        // Pine: tall slender trunk, stacked conical leaf disks.
        2 => {
            let trunk_h = 10 + (h % 6) as i32;
            let trunk_top = base + glam::IVec3::new(0, trunk_h, 0);
            paint_line(bricks, cmin, cmax, base, trunk_top, 0, MAT_WOOD_PINE);
            let layers: i32 = 6;
            for i in 0..layers {
                let t_f = i as f32 / layers as f32;
                let y = base.y + (trunk_h as f32 * (0.35 + t_f * 0.78)) as i32;
                let r = ((1.0 - t_f).powf(0.85) * 3.5 + 1.0) as i32;
                paint_canopy(bricks, cmin, cmax, glam::IVec3::new(base.x, y, base.z), r, MAT_LEAVES_PINE);
            }
        }
        // Birch: slim trunk + small leaf cluster.
        1 => {
            let trunk_h = 8 + (h % 5) as i32;
            let trunk_top = base + glam::IVec3::new(0, trunk_h, 0);
            paint_line(bricks, cmin, cmax, base, trunk_top, 0, MAT_WOOD_BIRCH);
            let n = 2 + (h % 2) as i32;
            for b in 0..n {
                let angle = (b as f32 / n as f32) * std::f32::consts::TAU
                    + branch_jitter(h, b as u32, 0) * 0.5;
                let len = 2 + ((h.wrapping_mul(b as u32 + 1)) % 3) as i32;
                let sy = base.y + (trunk_h as f32 * 0.7) as i32;
                let end = glam::IVec3::new(
                    base.x + (angle.cos() * len as f32) as i32,
                    sy + 1,
                    base.z + (angle.sin() * len as f32) as i32,
                );
                paint_line(bricks, cmin, cmax, glam::IVec3::new(base.x, sy, base.z), end, 0, MAT_WOOD_BIRCH);
                paint_canopy(bricks, cmin, cmax, end, 2, MAT_LEAVES_BIRCH);
            }
            paint_canopy(bricks, cmin, cmax, trunk_top, 3, MAT_LEAVES_BIRCH);
        }
        // Cactus (saguaro): thick column + 0-2 arms that go out then bend up.
        4 => {
            // Small bodied saguaro: a 2-wide column (between the too-thin
            // 1-wide and the too-chunky 3-wide). Arms branch low and rise to
            // just below the top so the silhouette reads at this small scale.
            // Thin saguaro: 1-wide trunk + 1-wide arms branching PERPENDICULAR
            // (the layout still reads 3D, not coplanar) with a gap to the trunk.
            let col_h = 3 + (h % 6) as i32; // 3..8 (wide height variation)
            let top = base + glam::IVec3::new(0, col_h, 0);
            paint_line(bricks, cmin, cmax, base, top, 0, MAT_CACTUS);
            let n_arms = (h % 3) as i32; // 0, 1 or 2
            let sets: [[(i32, i32); 2]; 4] = [
                [(1, 0), (0, 1)],
                [(-1, 0), (0, -1)],
                [(0, 1), (-1, 0)],
                [(0, -1), (1, 0)],
            ];
            let set = sets[(h & 3) as usize];
            for a in 0..n_arms {
                let (dx, dz) = set[a as usize % 2];
                let sy = base.y + (col_h as f32 * 0.4) as i32;
                let arm_top = base.y + (col_h as f32 * 0.9) as i32;
                let ax = base.x + dx * 2; // 2 out → a 1-voxel gap to the thin trunk
                let az = base.z + dz * 2;
                // Elbow bridging trunk → arm at the branch height.
                paint_line(bricks, cmin, cmax,
                    glam::IVec3::new(base.x, sy, base.z),
                    glam::IVec3::new(ax, sy, az), 0, MAT_CACTUS);
                // Vertical arm, gapped from the trunk.
                paint_line(bricks, cmin, cmax,
                    glam::IVec3::new(ax, sy, az),
                    glam::IVec3::new(ax, arm_top, az), 0, MAT_CACTUS);
            }
        }
        // Oak / autumn: wider canopy, a few branches.
        _ => {
            let leaf_mat = if t.ttype == 3 { MAT_LEAVES_AUTUMN } else { MAT_LEAVES };
            let trunk_h = 8 + (h % 5) as i32;
            let trunk_top = base + glam::IVec3::new(0, trunk_h, 0);
            paint_line(bricks, cmin, cmax, base, trunk_top, 0, MAT_WOOD);
            let n = 3 + (h % 2) as i32;
            for b in 0..n {
                let angle = (b as f32 / n as f32) * std::f32::consts::TAU
                    + branch_jitter(h, b as u32, 0) * 0.6;
                let len = 3 + ((h.wrapping_mul(b as u32 + 7)) % 3) as i32;
                let sy = base.y + (trunk_h as f32 * 0.65) as i32;
                let end = glam::IVec3::new(
                    base.x + (angle.cos() * len as f32) as i32,
                    sy + (len as f32 * 0.5) as i32,
                    base.z + (angle.sin() * len as f32) as i32,
                );
                paint_line(bricks, cmin, cmax, glam::IVec3::new(base.x, sy, base.z), end, 0, MAT_WOOD);
                paint_canopy(bricks, cmin, cmax, end, 3, leaf_mat);
            }
            paint_canopy(bricks, cmin, cmax, trunk_top, 4, leaf_mat);
        }
    }
}

// A canopy blob = a fringe shell (radius + 1, widened one EXTRA cell on
// the horizontal axes, only into empty cells) plus the leaf sphere itself
// (replacing its own interior fringe). The fringe ring is what lets the
// renderer show tuft quads and leaf-cloud cards protruding sideways; the
// horizontal widening gives the outer ring where overhanging cards render
// instead of being clipped at the shell boundary.
fn paint_canopy(
    bricks: &mut [Brick], cmin: (i32, i32, i32), cmax: (i32, i32, i32),
    c: glam::IVec3, r: i32, leaf_mat: u8,
) {
    paint_fringe_shell(bricks, cmin, cmax, c, r + 1);
    paint_sphere(bricks, cmin, cmax, c, r, leaf_mat);
}

// Sphere of radius r stretched +1 cell along +-x/+-z: the horizontal axes
// shrink toward the sphere test by one cell first, so the vertical extent
// stays r while the sides gain one ring.
fn paint_fringe_shell(
    bricks: &mut [Brick], cmin: (i32, i32, i32), cmax: (i32, i32, i32),
    center: glam::IVec3, r: i32,
) {
    let r2 = r * r;
    for dy in -r..=r {
        for dx in -(r + 1)..=(r + 1) {
            for dz in -(r + 1)..=(r + 1) {
                let hx = (dx.abs() - 1).max(0);
                let hz = (dz.abs() - 1).max(0);
                if hx * hx + dy * dy + hz * hz > r2 { continue; }
                try_write_tree_voxel(bricks, center.x + dx, center.y + dy, center.z + dz, MAT_LEAF_FRINGE, cmin, cmax);
            }
        }
    }
}

// thickness=0 → 1-voxel-wide line (no spheres along the line). Otherwise a
// small radius is splatted at each step. Keep small to avoid voxel blowup.
fn paint_line(
    bricks: &mut [Brick], cmin: (i32, i32, i32), cmax: (i32, i32, i32),
    a: glam::IVec3, b: glam::IVec3, thickness: i32, mat: u8,
) {
    let d = b - a;
    let len = ((d.x * d.x + d.y * d.y + d.z * d.z) as f32).sqrt();
    let steps = (len * 1.5).ceil() as i32;
    if steps <= 0 {
        try_write_tree_voxel(bricks, a.x, a.y, a.z, mat, cmin, cmax);
        return;
    }
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let cx = (a.x as f32 + d.x as f32 * t).round() as i32;
        let cy = (a.y as f32 + d.y as f32 * t).round() as i32;
        let cz = (a.z as f32 + d.z as f32 * t).round() as i32;
        if thickness == 0 {
            try_write_tree_voxel(bricks, cx, cy, cz, mat, cmin, cmax);
        } else {
            let r2 = thickness * thickness;
            for dy in -thickness..=thickness {
                for dx in -thickness..=thickness {
                    for dz in -thickness..=thickness {
                        if dx * dx + dy * dy + dz * dz > r2 { continue; }
                        try_write_tree_voxel(bricks, cx + dx, cy + dy, cz + dz, mat, cmin, cmax);
                    }
                }
            }
        }
    }
}

fn paint_sphere(
    bricks: &mut [Brick], cmin: (i32, i32, i32), cmax: (i32, i32, i32),
    center: glam::IVec3, r: i32, mat: u8,
) {
    let r2 = r * r;
    for dy in -r..=r {
        for dx in -r..=r {
            for dz in -r..=r {
                if dx * dx + dy * dy + dz * dz > r2 { continue; }
                try_write_tree_voxel(bricks, center.x + dx, center.y + dy, center.z + dz, mat, cmin, cmax);
            }
        }
    }
}

fn branch_jitter(hash: u32, b: u32, salt: u32) -> f32 {
    let h = hash
        .wrapping_mul(0x9E3779B1)
        .wrapping_add(b.wrapping_mul(2654435761))
        .wrapping_add(salt.wrapping_mul(40503));
    ((h & 0xFFFF) as f32 / 65535.0) * 2.0 - 1.0
}

#[inline]
fn write_into_scratch(bricks: &mut [Brick], dx: u32, dy: u32, dz: u32, mat: u8) {
    let bb_x = dx / BRICK_DIM;
    let bb_y = dy / BRICK_DIM;
    let bb_z = dz / BRICK_DIM;
    let bb_idx = (bb_x + bb_y * STORAGE_CHUNK_BRICKS + bb_z * STORAGE_CHUNK_BRICKS * STORAGE_CHUNK_BRICKS)
        as usize;
    bricks[bb_idx].set(dx % BRICK_DIM, dy % BRICK_DIM, dz % BRICK_DIM, mat);
}

// ---------------- biome + world-gen helpers ----------------

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Biome {
    Tundra,
    Plains,
    Forest,
    Jungle,
    Savanna,
    Desert,
    Beach,
    Mountain,
}

pub fn pick_biome(temp: f32, humid: f32, h: u32, sea_level: u32) -> Biome {
    if h > sea_level + 36 { return Biome::Mountain; }
    if h <= sea_level + 1 { return Biome::Beach; }
    if temp < -0.20 { return Biome::Tundra; }
    if temp > 0.25 && humid < -0.05 { return Biome::Desert; }
    if temp > 0.15 && humid > 0.25 { return Biome::Jungle; }
    if temp > 0.10 && humid < 0.10 { return Biome::Savanna; }
    if humid > 0.20 { return Biome::Forest; }
    Biome::Plains
}

impl Biome {
    pub fn top_block(self, h: u32, sea_level: u32) -> u8 {
        match self {
            Biome::Tundra => MAT_SNOW,
            Biome::Desert | Biome::Beach | Biome::Savanna => MAT_SAND,
            Biome::Mountain => if h > sea_level + 55 { MAT_SNOW } else { MAT_STONE },
            _ => MAT_GRASS,
        }
    }
    pub fn subsoil(self) -> u8 {
        match self {
            Biome::Desert | Biome::Beach => MAT_SAND,
            Biome::Mountain => MAT_STONE,
            Biome::Savanna => MAT_DIRT,
            _ => MAT_DIRT,
        }
    }
    pub fn tree_type(self, hash: u32) -> u32 {
        match self {
            Biome::Tundra | Biome::Mountain => 2, // pine
            Biome::Plains => if hash % 3 == 0 { 1 } else { 0 }, // birch/oak
            Biome::Forest => match hash % 4 { 0 => 1, 1 => 3, _ => 0 }, // birch/autumn/oak
            Biome::Jungle => match hash % 4 { 0 => 3, _ => 0 },  // oak/autumn — dense
            Biome::Savanna => 0,
            _ => 0,
        }
    }
    /// (flower, tall grass) probability per grass-top voxel. Close to the
    /// previous global constants (flower 0.015, grass 0.065) - plains get a
    /// mild boost, and meadow patches multiply the flower probability 3x
    /// locally. Denser than this measurably slows every view containing
    /// grass tops (more occupied cells = more DDA descents).
    pub fn flora_probs(self) -> (f32, f32) {
        match self {
            Biome::Plains => (0.020, 0.08),
            Biome::Forest => (0.010, 0.065),
            Biome::Jungle => (0.012, 0.09),
            _ => (0.015, 0.065),
        }
    }
    /// Trees per chunk multiplier — Jungle is dense, Savanna sparse.
    /// Trees per chunk in a "dense patch" of this biome. Clearings (low
    /// patch noise) bring it down to zero, dense patches scale by ~2x.
    pub fn tree_density(self) -> f32 {
        match self {
            Biome::Jungle => 1.2,   // very dense
            Biome::Forest => 0.55,  // dense
            Biome::Tundra => 0.30,  // scattered pines (denser)
            Biome::Plains => 0.20,  // occasional oaks/birch (was ~empty)
            Biome::Mountain => 0.14,
            Biome::Savanna => 0.0,  // sandy — reads as desert, keep it bare
            Biome::Desert => 0.20,  // ~1 per chunk, evenly scattered (flat patch_mul)
            _ => 0.0,               // Beach
        }
    }
    /// Elevation contribution — mountains are noticeably taller, jungles are
    /// rolling, plains are nearly flat.
    pub fn height_mult(self) -> f32 {
        match self {
            Biome::Mountain => 2.5,
            Biome::Jungle => 1.4,
            Biome::Forest => 1.0,
            Biome::Plains => 0.4,
            Biome::Savanna => 0.6,
            Biome::Tundra => 1.2,
            Biome::Desert => 0.5,
            Biome::Beach => 0.2,
        }
    }
}

/// Replace some stone voxels with ore. Rarer / more valuable ores cluster
/// deeper. Three noise scales give chunkier veins instead of single specks.
pub fn stone_or_ore(x: f32, y: f32, z: f32, h: u32) -> u8 {
    let depth = (h as f32 - y).max(0.0);
    let n1 = value_noise_3d(x * 0.27, y * 0.27, z * 0.27);
    let n2 = value_noise_3d(x * 0.55, y * 0.55, z * 0.55);
    let combined = n1 + n2 * 0.30;
    if depth > 30.0 && combined > 0.50 { return MAT_DIAMOND; }
    if depth > 20.0 && combined > 0.36 { return MAT_GOLD; }
    if depth > 10.0 && combined > 0.24 { return MAT_IRON; }
    if combined > 0.32 { return MAT_COAL; }
    MAT_STONE
}

pub fn place_tree(world: &mut World, cx: i32, base_y: u32, cz: i32, ttype: u32, hash: u32) {
    // Scaled-up trees: trunks ~15 wide, canopies ~20 radius, heights ~50-70.
    let (trunk_mat, leaf_mat, trunk_h, canopy_r, trunk_r, conical) = match ttype {
        0 => (MAT_WOOD,       MAT_LEAVES,        45 + (hash % 20), 22i32, 7i32, false),
        1 => (MAT_WOOD_BIRCH, MAT_LEAVES_BIRCH,  55 + (hash % 20), 18,    6,    false),
        2 => (MAT_WOOD_PINE,  MAT_LEAVES_PINE,   65 + (hash % 20), 22,    7,    true),
        3 => (MAT_WOOD,       MAT_LEAVES_AUTUMN, 45 + (hash % 20), 22,    7,    false),
        _ => (MAT_WOOD,       MAT_LEAVES,        45,               22,    7,    false),
    };
    let trunk_r2 = trunk_r * trunk_r;

    // Thick trunk: circular cross-section instead of a 3x3 box.
    for dy in 0..trunk_h {
        for dx in -trunk_r..=trunk_r {
            for dz in -trunk_r..=trunk_r {
                if dx * dx + dz * dz > trunk_r2 { continue; }
                let wx = cx + dx;
                let wz = cz + dz;
                let wy = base_y + dy;
                if wx >= 0 && wz >= 0
                && (wx as u32) < WORLD_VOXELS_X
                && (wz as u32) < WORLD_VOXELS_Z
                && wy < WORLD_VOXELS_Y {
                    world.write_voxel_unchecked(wx as u32, wy, wz as u32, trunk_mat);
                }
            }
        }
    }

    // Canopy
    if conical {
        // Pine: stack of decreasing-radius disks.
        let layers: i32 = 18;
        for layer in 0..layers {
            // Radius shrinks toward the top of the pine.
            let r = ((canopy_r * (layers - layer)) / layers).max(2);
            let wy_signed = base_y as i32 + trunk_h as i32 - 4 + layer * 2;
            if wy_signed < 0 { continue; }
            let wy = wy_signed as u32;
            if wy >= WORLD_VOXELS_Y { continue; }
            for dx in -r..=r {
                for dz in -r..=r {
                    if dx * dx + dz * dz > r * r { continue; }
                    let wx = cx + dx;
                    let wz = cz + dz;
                    if wx < 0 || wz < 0 { continue; }
                    if (wx as u32) >= WORLD_VOXELS_X || (wz as u32) >= WORLD_VOXELS_Z { continue; }
                    let bi = brick_idx((wx as u32) / BRICK_DIM, wy / BRICK_DIM, (wz as u32) / BRICK_DIM) as usize;
                    let vi = brick_voxel_idx((wx as u32) % BRICK_DIM, wy % BRICK_DIM, (wz as u32) % BRICK_DIM);
                    if (world.bricks[bi].occupancy & (1u64 << vi)) == 0 {
                        world.write_voxel_unchecked(wx as u32, wy, wz as u32, leaf_mat);
                    }
                }
            }
        }
    } else {
        let r = canopy_r;
        let canopy_cy = base_y as i32 + trunk_h as i32 + 1;
        for dy in -r..=r {
            for dx in -r..=r {
                for dz in -r..=r {
                    let dd = dx * dx + dy * dy + dz * dz;
                    if dd > r * r { continue; }
                    let wx = cx + dx;
                    let wz = cz + dz;
                    let wy = canopy_cy + dy;
                    if wx < 0 || wz < 0 || wy < 0 { continue; }
                    if (wx as u32) >= WORLD_VOXELS_X
                    || (wz as u32) >= WORLD_VOXELS_Z
                    || (wy as u32) >= WORLD_VOXELS_Y { continue; }
                    let bi = brick_idx((wx as u32) / BRICK_DIM, (wy as u32) / BRICK_DIM, (wz as u32) / BRICK_DIM) as usize;
                    let vi = brick_voxel_idx((wx as u32) % BRICK_DIM, (wy as u32) % BRICK_DIM, (wz as u32) % BRICK_DIM);
                    if (world.bricks[bi].occupancy & (1u64 << vi)) == 0 {
                        world.write_voxel_unchecked(wx as u32, wy as u32, wz as u32, leaf_mat);
                    }
                }
            }
        }
    }
}

pub fn value_noise_3d(x: f32, y: f32, z: f32) -> f32 {
    let xi = x.floor() as i32;
    let yi = y.floor() as i32;
    let zi = z.floor() as i32;
    let xf = smoothstep(x - xi as f32);
    let yf = smoothstep(y - yi as f32);
    let zf = smoothstep(z - zi as f32);
    let v000 = hash3(xi, yi, zi);
    let v100 = hash3(xi + 1, yi, zi);
    let v010 = hash3(xi, yi + 1, zi);
    let v110 = hash3(xi + 1, yi + 1, zi);
    let v001 = hash3(xi, yi, zi + 1);
    let v101 = hash3(xi + 1, yi, zi + 1);
    let v011 = hash3(xi, yi + 1, zi + 1);
    let v111 = hash3(xi + 1, yi + 1, zi + 1);
    let a = v000 * (1.0 - xf) + v100 * xf;
    let b = v010 * (1.0 - xf) + v110 * xf;
    let c = v001 * (1.0 - xf) + v101 * xf;
    let d = v011 * (1.0 - xf) + v111 * xf;
    let ab = a * (1.0 - yf) + b * yf;
    let cd = c * (1.0 - yf) + d * yf;
    ab * (1.0 - zf) + cd * zf
}
