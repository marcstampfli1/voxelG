// SINGLE SOURCE OF TRUTH for world dimensions.
//
// Included by `src/voxel.rs` (which re-exports everything) AND by `build.rs`,
// which generates the matching WGSL constants into `$OUT_DIR/world_consts.wgsl`
// and prepends them to every shader at pipeline-creation time. Editing a value
// here updates the Rust storage, the CPU raycaster, the temporal projector and
// all three shaders at once — no more Rust/WGSL drift (checklist: architecture).

pub const BRICK_DIM: u32 = 4;
pub const BRICK_VOXELS: u32 = BRICK_DIM * BRICK_DIM * BRICK_DIM;

pub const WORLD_BRICKS_X: u32 = 128;
pub const WORLD_BRICKS_Y: u32 = 64;
pub const WORLD_BRICKS_Z: u32 = 128;
pub const WORLD_BRICKS_TOTAL: u32 = WORLD_BRICKS_X * WORLD_BRICKS_Y * WORLD_BRICKS_Z;

pub const WORLD_VOXELS_X: u32 = WORLD_BRICKS_X * BRICK_DIM;
pub const WORLD_VOXELS_Y: u32 = WORLD_BRICKS_Y * BRICK_DIM;
pub const WORLD_VOXELS_Z: u32 = WORLD_BRICKS_Z * BRICK_DIM;

pub const WORLD_TILES_X: u32 = WORLD_BRICKS_X / 4;
pub const WORLD_TILES_Y: u32 = WORLD_BRICKS_Y / 4;
pub const WORLD_TILES_Z: u32 = WORLD_BRICKS_Z / 4;
pub const WORLD_TILES_TOTAL: u32 = WORLD_TILES_X * WORLD_TILES_Y * WORLD_TILES_Z;

pub const WORLD_CHUNKS_X: u32 = (WORLD_TILES_X + 3) / 4;
pub const WORLD_CHUNKS_Y: u32 = (WORLD_TILES_Y + 3) / 4;
pub const WORLD_CHUNKS_Z: u32 = (WORLD_TILES_Z + 3) / 4;
pub const WORLD_CHUNKS_TOTAL: u32 = WORLD_CHUNKS_X * WORLD_CHUNKS_Y * WORLD_CHUNKS_Z;

// L4: one more pyramid level. A single u64 covers a 4³-chunk cell = a 256³-voxel
// region, so one bit test skips up to a quarter-million empty voxels. This is
// the level that keeps the DDA cheap as voxels shrink toward ~10 cm.
pub const WORLD_L4_X: u32 = (WORLD_CHUNKS_X + 3) / 4;
pub const WORLD_L4_Y: u32 = (WORLD_CHUNKS_Y + 3) / 4;
pub const WORLD_L4_Z: u32 = (WORLD_CHUNKS_Z + 3) / 4;
pub const WORLD_L4_TOTAL: u32 = WORLD_L4_X * WORLD_L4_Y * WORLD_L4_Z;

// ---- GI irradiance probe grid (world-space DDGI-style cache) ----
// One irradiance probe every PROBE_SPACING voxels, centered in its cell. The
// grid is world-space and toroidal like the voxel storage, so it follows the
// streaming window at O(1) and per-pixel GI is a cheap trilinear probe sample
// instead of a fresh bounce ray. PROBE_SPACING must divide the world dims.
pub const PROBE_SPACING: u32 = 8;
pub const PROBE_DIM_X: u32 = WORLD_VOXELS_X / PROBE_SPACING; // 64
pub const PROBE_DIM_Y: u32 = WORLD_VOXELS_Y / PROBE_SPACING; // 32
pub const PROBE_DIM_Z: u32 = WORLD_VOXELS_Z / PROBE_SPACING; // 64
pub const PROBE_TOTAL: u32 = PROBE_DIM_X * PROBE_DIM_Y * PROBE_DIM_Z; // 131072

// ---- storage chunks (the "chunked world") ----
// A storage chunk holds 8x8x8 bricks = 32x32x32 voxels. Generation, dirty
// tracking and GPU streaming all operate at this granularity.
pub const STORAGE_CHUNK_BRICKS: u32 = 8;
pub const STORAGE_CHUNK_VOXELS: u32 = STORAGE_CHUNK_BRICKS * BRICK_DIM;
pub const WORLD_STORE_CX: u32 = WORLD_BRICKS_X / STORAGE_CHUNK_BRICKS;
pub const WORLD_STORE_CY: u32 = WORLD_BRICKS_Y / STORAGE_CHUNK_BRICKS;
pub const WORLD_STORE_CZ: u32 = WORLD_BRICKS_Z / STORAGE_CHUNK_BRICKS;
pub const WORLD_STORE_CHUNKS: u32 = WORLD_STORE_CX * WORLD_STORE_CY * WORLD_STORE_CZ;
