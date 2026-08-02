// SINGLE SOURCE OF TRUTH for world dimensions.
//
// Included by `src/voxel.rs` (which re-exports everything) AND by `build.rs`,
// which generates the matching WGSL constants into `$OUT_DIR/world_consts.wgsl`
// and prepends them to every shader at pipeline-creation time. Editing a value
// here updates the Rust storage, the CPU raycaster, the temporal projector and
// all three shaders at once — no more Rust/WGSL drift (checklist: architecture).

pub const BRICK_DIM: u32 = 4;
pub const BRICK_VOXELS: u32 = BRICK_DIM * BRICK_DIM * BRICK_DIM;

// ---- physical scale ----
// The SINGLE SOURCE OF TRUTH for "how big is a voxel in the real world".
//
// Everything the renderer and worldgen do is in voxel units, which is right for
// them - they never need metres. GAMEPLAY does: a player is 1.8 m tall, gravity
// is 9.81 m/s^2 and a step you can walk up is ~0.35 m, and those numbers must
// not silently change meaning when the voxel does. So body sizes and movement
// tuning are written in SI and converted here, exactly once.
//
// One voxel is 10 cm (was 25 cm until docs/SCALE_TO_10CM.md landed). The player
// stayed 1.8 m tall across that change without a single gameplay constant being
// retuned, which is the whole point of routing sizes through this rather than
// hardcoding voxel counts.
//
// This is ALSO the conversion the renderer's distance budgets and worldgen's
// feature sizes go through. A view distance or a mountain amplitude written as a
// bare voxel count silently shrinks by 2.5x when this constant moves; written as
// `m_to_vox(175.0)` it does not. Everything scale-dependent in `raymarch.wgsl`
// and `voxel.rs` is written that way now, and `build.rs` emits VOXELS_PER_METRE
// into the shader prelude so WGSL uses the same one source of truth.
pub const VOXEL_METRES: f32 = 0.10;
pub const VOXELS_PER_METRE: f32 = 1.0 / VOXEL_METRES;

/// Metres -> voxels. Use at every gameplay constant so the SI value stays
/// visible in the source (`m_to_vox(1.8)` reads as "1.8 metres", `7.2` does not).
#[inline(always)]
pub const fn m_to_vox(metres: f32) -> f32 {
    metres * VOXELS_PER_METRE
}

/// Voxels -> metres. For reporting/measuring a simulated quantity back in SI.
#[inline(always)]
pub const fn vox_to_m(voxels: f32) -> f32 {
    voxels * VOXEL_METRES
}

// The loaded streaming window, in bricks. MUST be a multiple of 80: 16 keeps the
// tile (x4) and chunk (x4) levels of the pyramid exact, and 5 keeps
// PROBE_SPACING (20 voxels) dividing the voxel extent. Y only needs 16 and 5 as
// well, which 160 satisfies.
//
// 400 x 160 x 400 bricks = 1600 x 640 x 1600 voxels = 160 x 64 x 160 m at 10 cm.
// That is 1.25x today's 128 m per horizontal axis (1.56x the area) at 2.5x the
// linear resolution, and it was chosen by measurement, not by guess: see the
// size ladder in docs/SCALE_TO_10CM.md, which reports what 320 (128 m, the
// same world as the 25 cm build) and 400 each cost per pass.
pub const WORLD_BRICKS_X: u32 = 400;
pub const WORLD_BRICKS_Y: u32 = 160;
pub const WORLD_BRICKS_Z: u32 = 400;
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

// ---- compute dispatch tiling ----
// Workgroups carried by the X dimension of a linearised compute dispatch.
//
// `dispatch_workgroups` caps EVERY dimension at 65,535 - that is WebGPU's
// `maxComputeWorkgroupsPerDimension` floor and what wgpu reports by default.
// A one-invocation-per-brick pass is WORLD_BRICKS_TOTAL / 64 workgroups, which
// was 16,384 at 25 cm and is 400,000 at 10 cm: the GPU physics dispatch went
// from comfortably inside the limit to a hard validation abort, with nothing in
// between to warn about it. So a world-sized dispatch is TILED - X carries this
// many workgroups and Y carries however many rows are needed - and the shader
// rebuilds the linear index with `linear_wg` / the same multiply.
//
// 32,768 is half the limit and a power of two, so the reconstruction is a shift
// and there is room for the limit to be reported lower by some future adapter
// without the value needing to move. `renderer::linear_dispatch` is the only
// place that converts a workgroup COUNT into a dispatch; nothing else should
// open-code the division.
pub const DISPATCH_ROW_WGS: u32 = 32_768;

// ---- per-voxel light field (docs/VOXEL_LIGHTING_PLAN.md) ----
// Resident light blocks, one per brick that carries lit-shell air. It lives here
// rather than in `src/voxlight.rs` because the SHADER needs it too: the URGENT
// list is appended to `vl_live_bricks` at exactly this offset, so both sides
// must agree and `build.rs` emits it into the shader prelude. The rationale for
// the value is on `voxlight::LIGHT_BLOCKS_MAX`, which re-exports this.
// 2^22. The lit shell is a SURFACE, so it grows with the square of the linear
// resolution: at 25 cm the demo world bound 63,903 blocks, and 2.5x finer voxels
// over a 1.25x wider window is 6.25 x 1.56 = 9.8x that. The old 131,072 ceiling
// would have been overrun by 8x, and overflow does not degrade gracefully -
// `allocate` fills in brick-index order, which is z-major, so the shortfall
// lands as a hard geographic band (that exact failure is the round-D story on
// `voxlight::LIGHT_BLOCKS_MAX`).
//
// MEASURED, not estimated: a demo world with a coast, a forest and a 38 m peak
// in it binds 1,118,187 blocks. 2^21 held it but with only 47% spare, which is
// not headroom for a rougher seed - the shell scales with SURFACE, and caves,
// cliffs and dense canopy are all surface.
//
// What pays for it is LIGHT_RECORDS_PER_BLOCK dropping 64 -> 8 (records at 2x
// voxel spacing): a block is 64 B instead of 512 B, so 32x the blocks cost 4x
// the pool - 256 MiB against 64 MiB - and the field is still finer in metres
// than the 25 cm build shipped. Measured headroom on the demo world is in
// `the_demo_world_light_shell_fits_the_pool_and_still_covers_it`.
pub const LIGHT_BLOCKS_MAX: u32 = 4_194_304;

// Edge of a light record's cell, in voxels. A record covers a
// LIGHT_RECORD_STEP^3 group, so a brick holds (BRICK_DIM/STEP)^3 of them.
//
// 2 at 10 cm is a 20 cm light field: FINER than the 25 cm one that shipped, at
// 1/8 the storage. 1 (a record per voxel, which is what 25 cm ran) would put the
// demo world's shell at ~5 M blocks x 512 B = 2.5 GB of pool for detail below
// the size of the pixel it is interpolated across. 4 (40 cm) is the next step up
// and is the knob to turn if the pool ever needs to shrink again.
//
// It MUST divide BRICK_DIM: the update pass indexes records within one brick and
// the sampler folds a voxel coord to a record coord by a shift, and both would
// straddle a brick boundary otherwise.
pub const LIGHT_RECORD_STEP: u32 = 2;
pub const LIGHT_RECORD_DIM: u32 = BRICK_DIM / LIGHT_RECORD_STEP;
pub const LIGHT_RECORDS_PER_BLOCK: u32 = LIGHT_RECORD_DIM * LIGHT_RECORD_DIM * LIGHT_RECORD_DIM;

/// A record is two u32 words:
///   word0: sun_vis u8 | ao u8 | epoch u8 | flags u8
///   word1: point-light radiance, packed RGB9E5
pub const LIGHT_RECORD_WORDS: u32 = 2;

/// u32 words per block. The shader needs this to address the pool, so it is
/// derived here and emitted by `build.rs` rather than written out twice.
pub const LIGHT_BLOCK_WORDS: u32 = LIGHT_RECORDS_PER_BLOCK * LIGHT_RECORD_WORDS;

// Urgent-list slots reserved past the work list in the same buffer. This is a
// PER-DISPATCH budget, not a queue length: the CPU queue is unbounded and drains
// at most this many bricks per dispatch, which is what stops a chunk install
// (3,072 bricks dirtied at once, more after the neighbourhood walk) from turning
// into one enormous workgroup launch on a single frame. 512 is about half a
// normal sweep round on the demo world, so a burst costs less per frame than the
// sweep it rides alongside.
pub const LIGHT_URGENT_BUDGET: u32 = 512;

// ---- GI irradiance probe grid (world-space DDGI-style cache) ----
// One irradiance probe every PROBE_SPACING voxels, centered in its cell. The
// grid is world-space and toroidal like the voxel storage, so it follows the
// streaming window at O(1) and per-pixel GI is a cheap trilinear probe sample
// instead of a fresh bounce ray. PROBE_SPACING must divide the world dims.
//
// The spacing is a REAL-WORLD 2 m, not a voxel count: an irradiance cache is a
// property of the room, not of the grid, so 8 voxels at 25 cm and 20 voxels at
// 10 cm are the same cache. Leaving it at 8 would have bought a 2.5x finer probe
// grid nobody asked for and 15.6x the probes to update.
pub const PROBE_SPACING: u32 = (2.0 * VOXELS_PER_METRE) as u32; // 2 m
pub const PROBE_DIM_X: u32 = WORLD_VOXELS_X / PROBE_SPACING; // 80
pub const PROBE_DIM_Y: u32 = WORLD_VOXELS_Y / PROBE_SPACING; // 32
pub const PROBE_DIM_Z: u32 = WORLD_VOXELS_Z / PROBE_SPACING; // 80
pub const PROBE_TOTAL: u32 = PROBE_DIM_X * PROBE_DIM_Y * PROBE_DIM_Z; // 204800

// ---- storage chunks (the "chunked world") ----
// A storage chunk holds 8x8x8 bricks = 32x32x32 voxels. Generation, dirty
// tracking and GPU streaming all operate at this granularity.
pub const STORAGE_CHUNK_BRICKS: u32 = 8;
pub const STORAGE_CHUNK_VOXELS: u32 = STORAGE_CHUNK_BRICKS * BRICK_DIM;
pub const WORLD_STORE_CX: u32 = WORLD_BRICKS_X / STORAGE_CHUNK_BRICKS;
pub const WORLD_STORE_CY: u32 = WORLD_BRICKS_Y / STORAGE_CHUNK_BRICKS;
pub const WORLD_STORE_CZ: u32 = WORLD_BRICKS_Z / STORAGE_CHUNK_BRICKS;
pub const WORLD_STORE_CHUNKS: u32 = WORLD_STORE_CX * WORLD_STORE_CY * WORLD_STORE_CZ;
