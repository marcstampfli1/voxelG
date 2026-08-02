// Hierarchical DDA raymarch through a 3-level bit pyramid.
//
//   L3 chunk_mask  — bit per child tile  (one u64 covers a 64³ voxel region)
//   L2 tile_mask   — bit per child brick (one u64 covers a 16³ voxel region)
//   L1 brick.occ   — bit per voxel       (one u64 covers a  4³ voxel region)
//
// On a hit we compute per-corner ambient occlusion by reading 12 neighbour
// occupancy bits across the hit face (4 corners × 3 samples each). With the
// hierarchy, each "is this neighbour solid?" lookup is ~3 u32 fetches max
// (chunk → tile → brick) and most return false at L3/L2 immediately.

// Camera uniform layout is defined in shaders/common.wgsl (shared prelude).

// Toroidal storage: world voxel coords get folded into [0, WORLD_VOXELS_*)
// for the lookup. As the camera shifts the origin, only the small edge
// region's slots get reused — the rest of storage stays put.
fn world_to_slot_voxel(wv: vec3<i32>) -> vec3<i32> {
    return vec3<i32>(
        pos_mod(wv.x, WORLD_VOXELS_X),
        wv.y,
        pos_mod(wv.z, WORLD_VOXELS_Z),
    );
}

struct Brick {
    occ_lo: u32,
    occ_hi: u32,
    materials: array<u32, 16>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> bricks: array<Brick>;
@group(0) @binding(2) var<storage, read> tile_mask: array<u32>;
@group(0) @binding(3) var<storage, read> chunk_mask: array<u32>;
@group(0) @binding(4) var<uniform> palette: array<vec4<f32>, 256>;
// HDR scene target: sun glints, sky disc and backlit foliage exceed 1.0;
// the post stack (shaders/post.wgsl) tonemaps to LDR before the TAA.
@group(0) @binding(5) var output_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(6) var beam_depth: texture_2d<f32>;
@group(0) @binding(7) var<storage, read> tile_dirty: array<u32>;

struct PlayersBuf {
    count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    positions: array<vec4<f32>, 16>,
};
@group(0) @binding(8) var<storage, read> players: PlayersBuf;

// Uniform-material lookup tables. brick_uniform[bi] / tile_uniform[ti] = 0 means
// the brick/tile is non-uniform (must traverse children); any non-zero value is
// the material that fills it entirely. Packed 4 bytes per u32.
@group(0) @binding(9) var<storage, read> brick_uniform_packed: array<u32>;
@group(0) @binding(10) var<storage, read> tile_uniform_packed: array<u32>;

// L4 occupancy: one u64 (= 2 u32) per 256-voxel cell, one bit per child chunk.
// The coarsest pyramid level — one bit test skips a 256 region.
@group(0) @binding(11) var<storage, read> l4_mask: array<u32>;

// Half-res volumetrics: cs_clouds writes the cloud march into cloud_out at half
// resolution (binding 14); cs_main samples it bilinearly (binding 12 + sampler
// 13) and composites — ~4x fewer cloud marches on sky-facing views (#14).
@group(0) @binding(12) var cloud_in: texture_2d<f32>;
@group(0) @binding(13) var cloud_samp: sampler;
@group(0) @binding(14) var cloud_out: texture_storage_2d<rgba16float, write>;

// Lighting G-buffer (binding 16): per pixel, the hit position relative to
// world_origin plus a packed shadow/AO word. TWO consumers, both of which need
// it and neither of which is lighting reuse:
//   - cs_taa reprojects a terrain pixel into last frame by its stored world
//     position, which is what lets TAA accumulate across camera motion;
//   - the grass pass lights each blade from the ground pixel at its root, so a
//     blade sits in the same shadow the turf under it does.
//
// IT USED TO BE A PAIR. Binding 15 held the PREVIOUS frame's copy, cs_main
// reprojected each hit into it and reused the stored shadow/AO on a position
// match, and the frame ended with a full-resolution Rgba32Float copy from one to
// the other. That whole mechanism is gone, for two measured reasons:
//
//   - IT WAS A SECOND LIGHTING PATH, and it WON. `reuse_shadow` was tested
//     before `vlf.valid`, so wherever reprojection hit - i.e. on a still camera,
//     which is the only state it engages in - a smooth world-space gradient was
//     overridden by a value some earlier frame had traced from ONE binary ray.
//     Shadows visibly changed character when the camera stopped moving.
//   - IT COST MORE THAN IT SAVED. The per-voxel field now answers 96.6% of
//     shaded pixels on terrain and 99.9% on water (`LiveFrameRig::coverage`), so
//     the reprojection was re-deriving what the field had already given away.
//     Timed on a still camera with it compiled out, cs_main went 4.73 -> 4.30 ms
//     on terrain (the cache COST 9.9%) and moved within noise on foliage and
//     water. Removing it also drops 33 MB of VRAM and the 66 MB/frame of copy
//     traffic that ping-pong needed - fixed cost that a sky-facing frame paid in
//     full for nothing.
@group(0) @binding(16) var light_out: texture_storage_2d<rgba32float, write>;

// G-buffer word: 8-bit shadow + 8-bit AO + a spare 16 bits. The spare used to
// hold the sun altitude the shadow was traced at, which drove the staleness
// dither that decided when to re-trace; there is nothing to re-trace now.
// SYNC: grass.wgsl unpacks these bits.
fn pack_light_cache(shadow: f32, ao: f32) -> u32 {
    let s8 = u32(round(clamp(shadow, 0.0, 1.0) * 255.0));
    let a8 = u32(round(clamp(ao, 0.0, 1.0) * 255.0));
    return (s8 << 24u) | (a8 << 16u) | 1u;
}

// Deferred transparent pass (#16). cs_main records each transparent (water-top /
// glass) hit here as (t_hit, kind, facet/face code, facet payload) and writes a
// cheap placeholder colour; the separate cs_transparent pass does the expensive
// refraction/dispersion so the opaque-majority warps in cs_main stay coherent
// (less 8x8 divergence). A read_write storage buffer (not a texture) lets both
// passes share this binding without a read/write aliasing hazard.
// Deferred transparent records, one per pixel. Explicit u32 schema (bit
// patterns must survive exactly; f32 lanes are not bit-stable for packed
// payloads on all hardware):
//   x = bitcast<u32>(t_hit)
//   y = kind: TR_NONE / TR_WATER_TOP / TR_WATER_FACE / TR_GLASS
//   z = TR_WATER_TOP: pack2x16float(the cell's QUANTIZED facet slope, which is
//       the whole normal - cs_transparent adds nothing per pixel);
//       TR_WATER_FACE/TR_GLASS: axis face code (encode_face_normal)
//   w = TR_WATER_TOP: the cell's wave band + solid-neighbour mask
//       (water_pack_facet), which drive the facet tone and the foam; 0 otherwise
//
// The buffer used to carry a THIRD region, a full-res temporal reflection
// history for the water mirror. Stylized water has no reflection to accumulate,
// so that region is gone and this allocation lost a full-res RGBA32 image
// (33 MB at 1920x1080).
@group(0) @binding(17) var<storage, read_write> transp_buf: array<vec4<u32>>;

const TR_NONE:       u32 = 0u;
const TR_WATER_TOP:  u32 = 1u;
const TR_WATER_FACE: u32 = 2u;
const TR_GLASS:      u32 = 3u;

// Authored 16x16 foliage sprites, 2 bits per texel (0 transparent, 1 primary,
// 2 secondary/dark, 3 accent). Drawn as ASCII art in src/sprites.rs and
// encoded at startup — hand-made cutout art, not hash noise.
@group(0) @binding(18) var<storage, read> sprites: array<u32>;
// Primary-hit depth (t along the primary ray; 1e9 = sky). Water and glass
// pixels carry the SURFACE t (cs_transparent re-shades at the same t) and
// player boxes are included, so the falling-leaf pass can depth-test
// against the world without any rasterized geometry. Persists across
// temporal-differential clean tiles exactly like output_tex.
@group(0) @binding(19) var depth_out: texture_storage_2d<r32float, write>;
// cs_compose inputs: the geometry colour written by cs_main/cs_transparent
// and the exported depth, re-read as sampled textures. In the main bind
// group these two slots point at unrelated textures (history, beam) to
// keep per-pass usage scopes conflict-free.
@group(0) @binding(20) var geom_in: texture_2d<f32>;
@group(0) @binding(21) var depth_in: texture_2d<f32>;

// The SPR_* / TUFT_* atlas constants are GENERATED from src/sprites.rs and
// prepended to this source (see renderer::raymarch_source) - single source
// of truth for indices and word offsets.

// Texel (x, y) of a sprite; y = 0 is the sprite's bottom row. 16 u32s per
// sprite, bit (y*16 + x)*2.
fn sprite_texel(sprite: u32, x: u32, y: u32) -> u32 {
    let bit = (y * 16u + x) * 2u;
    let w = sprites[sprite * 16u + (bit >> 5u)];
    return (w >> (bit & 31u)) & 3u;
}

fn brick_uniform_mat(bi: i32) -> u32 {
    let w = brick_uniform_packed[bi >> 2];
    let shift = u32(bi & 3) * 8u;
    return (w >> shift) & 0xFFu;
}

fn tile_uniform_mat(ti: i32) -> u32 {
    let w = tile_uniform_packed[ti >> 2];
    let shift = u32(ti & 3) * 8u;
    return (w >> shift) & 0xFFu;
}

fn ray_aabb_t(origin: vec3<f32>, inv_dir: vec3<f32>, mn: vec3<f32>, mx: vec3<f32>) -> f32 {
    let t0 = (mn - origin) * inv_dir;
    let t1 = (mx - origin) * inv_dir;
    let tmin3 = min(t0, t1);
    let tmax3 = max(t0, t1);
    let t_enter = max(max(tmin3.x, tmin3.y), max(tmin3.z, 0.0));
    let t_exit = min(min(tmax3.x, tmax3.y), tmax3.z);
    if (t_enter >= t_exit || t_exit < 0.0) { return 1e30; }
    return t_enter;
}

fn player_color_for(id: u32) -> vec3<f32> {
    let h = (id * 2654435761u) & 0xFFu;
    let r = f32((h * 73u) & 0xFFu) / 255.0;
    let g = f32((h * 41u + 91u) & 0xFFu) / 255.0;
    let b = f32((h * 113u + 53u) & 0xFFu) / 255.0;
    return vec3<f32>(0.5 + r * 0.5, 0.5 + g * 0.5, 0.5 + b * 0.5);
}

// World-dimension consts (BRICK_DIM, WORLD_BRICKS_*, WORLD_VOXELS_*,
// WORLD_TILES_*, WORLD_CHUNKS_*, WORLD_L4_*) are injected at the top of this
// module from $OUT_DIR/world_consts.wgsl, generated by build.rs from
// src/world_dims.rs. Do not redeclare them here.

fn brick_voxel_idx(lx: i32, ly: i32, lz: i32) -> i32 {
    return lx + lz * 4 + ly * 16;
}

fn world_brick_idx(bx: i32, by: i32, bz: i32) -> i32 {
    return bx + by * WORLD_BRICKS_X + bz * WORLD_BRICKS_X * WORLD_BRICKS_Y;
}

fn world_tile_idx(tx: i32, ty: i32, tz: i32) -> i32 {
    return tx + ty * WORLD_TILES_X + tz * WORLD_TILES_X * WORLD_TILES_Y;
}

fn world_chunk_idx(cx: i32, cy: i32, cz: i32) -> i32 {
    return cx + cy * WORLD_CHUNKS_X + cz * WORLD_CHUNKS_X * WORLD_CHUNKS_Y;
}

fn tile_has_child(ti: i32, child_lin: i32) -> bool {
    let base = ti * 2;
    if (child_lin < 32) {
        return (tile_mask[base] & (1u << u32(child_lin))) != 0u;
    }
    return (tile_mask[base + 1] & (1u << u32(child_lin - 32))) != 0u;
}

fn chunk_has_child(ci: i32, child_lin: i32) -> bool {
    let base = ci * 2;
    if (child_lin < 32) {
        return (chunk_mask[base] & (1u << u32(child_lin))) != 0u;
    }
    return (chunk_mask[base + 1] & (1u << u32(child_lin - 32))) != 0u;
}

fn world_l4_idx(x: i32, y: i32, z: i32) -> i32 {
    return x + y * WORLD_L4_X + z * WORLD_L4_X * WORLD_L4_Y;
}

// Is this child chunk occupied within its L4 cell?
fn l4_has_child(li: i32, child_lin: i32) -> bool {
    let base = li * 2;
    if (child_lin < 32) {
        return (l4_mask[base] & (1u << u32(child_lin))) != 0u;
    }
    return (l4_mask[base + 1] & (1u << u32(child_lin - 32))) != 0u;
}

// Is the entire 256-voxel L4 cell empty? One test skips a quarter-million voxels.
fn l4_cell_empty(li: i32) -> bool {
    let base = li * 2;
    return (l4_mask[base] | l4_mask[base + 1]) == 0u;
}

fn brick_voxel_solid(bi: i32, vi: i32) -> bool {
    let b = bricks[bi];
    if (vi < 32) {
        return (b.occ_lo & (1u << u32(vi))) != 0u;
    }
    return (b.occ_hi & (1u << u32(vi - 32))) != 0u;
}

fn brick_voxel_material(bi: i32, vi: i32) -> u32 {
    let word = vi / 4;
    let byte = vi - word * 4;
    return (bricks[bi].materials[word] >> u32(byte * 8)) & 0xFFu;
}

// LOD support: representative material of an entire brick — used at far
// distance where we terminate the DDA at brick granularity instead of
// per-voxel. Picks the topmost solid voxel so distant terrain reads as its
// surface (grass/snow/sand) rather than its hidden interior (stone/dirt).
fn brick_topmost_material(bi: i32) -> u32 {
    let b = bricks[bi];
    // Brick layout: voxel idx = x + z*4 + y*16. So y=3 layer = bits 48..63.
    // Walk from top y=3 layer down. MAT_TURF (the continuous blade layer on
    // every grass top) is skipped like air: from LOD distance the surface IS
    // the grass block under it, and reporting turf here would disable the
    // brick/tile LOD fast paths across every meadow. A brick holding ONLY
    // turf still reports turf, and its callers fall through to the per-voxel
    // descent (whose beyond-TURF_T dispatcher miss steps to the ground).
    let occ_hi = b.occ_hi;
    let occ_lo = b.occ_lo;
    // y=3 layer is occ_hi >> 16 (16 bits at bit 48..63).
    let y3 = (occ_hi >> 16u) & 0xFFFFu;
    if (y3 != 0u) {
        let bit = firstTrailingBit(y3);
        let m = brick_voxel_material(bi, i32(48u + bit));
        if (m != MAT_TURF) { return m; }
    }
    // y=2 layer = occ_hi & 0xFFFF (bits 32..47).
    let y2 = occ_hi & 0xFFFFu;
    if (y2 != 0u) {
        let bit = firstTrailingBit(y2);
        let m = brick_voxel_material(bi, i32(32u + bit));
        if (m != MAT_TURF) { return m; }
    }
    // y=1 layer = occ_lo >> 16 (bits 16..31).
    let y1 = (occ_lo >> 16u) & 0xFFFFu;
    if (y1 != 0u) {
        let bit = firstTrailingBit(y1);
        let m = brick_voxel_material(bi, i32(16u + bit));
        if (m != MAT_TURF) { return m; }
    }
    // y=0 layer = occ_lo & 0xFFFF (bits 0..15).
    let y0 = occ_lo & 0xFFFFu;
    if (y0 != 0u) {
        let bit = firstTrailingBit(y0);
        let m = brick_voxel_material(bi, i32(bit));
        if (m != MAT_TURF) { return m; }
    }
    // Occupied but nothing except turf: report turf (callers' foliage guard
    // then takes the per-voxel descent). NEVER 0 here - a mat-0 LOD cube.
    return select(0u, MAT_TURF, (occ_lo | occ_hi) != 0u);
}

fn is_voxel_solid(world_v: vec3<i32>) -> bool {
    // Bounds: voxel must be inside the loaded window.
    let rel = world_v - camera.world_origin;
    if (rel.x < 0 || rel.x >= WORLD_VOXELS_X
     || rel.y < 0 || rel.y >= WORLD_VOXELS_Y
     || rel.z < 0 || rel.z >= WORLD_VOXELS_Z) {
        return false;
    }
    // Fold into slot voxel for storage lookup.
    let v = world_to_slot_voxel(world_v);
    let bp = v >> vec3<u32>(2u);
    let tp = v >> vec3<u32>(4u);
    let cp = v >> vec3<u32>(6u);
    let ci = world_chunk_idx(cp.x, cp.y, cp.z);
    let tile_lin = (tp.x & 3) + (tp.z & 3) * 4 + (tp.y & 3) * 16;
    if (!chunk_has_child(ci, tile_lin)) { return false; }
    let ti = world_tile_idx(tp.x, tp.y, tp.z);
    let brick_lin = (bp.x & 3) + (bp.z & 3) * 4 + (bp.y & 3) * 16;
    if (!tile_has_child(ti, brick_lin)) { return false; }
    let bi = world_brick_idx(bp.x, bp.y, bp.z);
    let local = v - bp * BRICK_DIM;
    let vi = brick_voxel_idx(local.x, local.y, local.z);
    return brick_voxel_solid(bi, vi);
}

// Material at a WORLD voxel coord, or 0 (MAT_AIR) if empty / outside window.
// Used by `camera_in_water` for the underwater post-effect.
fn voxel_material_at(world_v: vec3<i32>) -> u32 {
    let rel = world_v - camera.world_origin;
    if (rel.x < 0 || rel.x >= WORLD_VOXELS_X
     || rel.y < 0 || rel.y >= WORLD_VOXELS_Y
     || rel.z < 0 || rel.z >= WORLD_VOXELS_Z) { return 0u; }
    let v = world_to_slot_voxel(world_v);
    let bp = v >> vec3<u32>(2u);
    let tp = v >> vec3<u32>(4u);
    let cp = v >> vec3<u32>(6u);
    let ci = world_chunk_idx(cp.x, cp.y, cp.z);
    let tile_lin = (tp.x & 3) + (tp.z & 3) * 4 + (tp.y & 3) * 16;
    if (!chunk_has_child(ci, tile_lin)) { return 0u; }
    let ti = world_tile_idx(tp.x, tp.y, tp.z);
    let brick_lin = (bp.x & 3) + (bp.z & 3) * 4 + (bp.y & 3) * 16;
    if (!tile_has_child(ti, brick_lin)) { return 0u; }
    let bi = world_brick_idx(bp.x, bp.y, bp.z);
    let local = v - bp * BRICK_DIM;
    let vi = brick_voxel_idx(local.x, local.y, local.z);
    if (!brick_voxel_solid(bi, vi)) { return 0u; }
    return brick_voxel_material(bi, vi);
}

// ---------------------------------------------------------------------------
// Per-voxel light field (docs/VOXEL_LIGHTING_PLAN.md).
//
// Shadows, AO and local light are properties of AIR voxels touching geometry,
// not of pixels. Shading fetches and interpolates them; nothing here traces.
// Sited after the voxel accessors above because the sampler needs
// is_voxel_solid / world_brick_idx / brick_voxel_idx, and before shade().
// ---------------------------------------------------------------------------

const VL_NONE: u32 = 0xFFFFFFFFu;
const VL_RECORD_WORDS: u32 = u32(LIGHT_RECORD_WORDS);
const VL_BLOCK_WORDS: u32 = u32(LIGHT_BLOCK_WORDS);
// Voxel edge of one record's cell. A record covers a VL_STEP^3 group, so the
// light lattice is coarser than the voxel grid; see LIGHT_RECORD_STEP.
const VL_STEP: i32 = LIGHT_RECORD_STEP;
const VL_STEP_F: f32 = f32(LIGHT_RECORD_STEP);

/// Index of the record covering in-brick voxel `local`.
/// Mirrors `voxlight::light_record_idx` (x + z*D + y*D^2, one level up from
/// `brick_voxel_idx`); the update pass inverts exactly this.
fn light_record_idx(local: vec3<i32>) -> i32 {
    let r = local / VL_STEP;
    return r.x + r.z * LIGHT_RECORD_DIM + r.y * LIGHT_RECORD_DIM * LIGHT_RECORD_DIM;
}

/// Occupancy bits of the VL_STEP^3 voxel group `r` (in-brick record coords)
/// within brick `bi`, as a mask over ONE half of the brick's 64-bit occupancy.
///
/// The whole group lives in one half by construction: the brick voxel index is
/// x + z*4 + y*16, so a 2^3 group is the bit pattern {0,1,4,5,16,17,20,21}
/// (= 0x00330033) shifted by 2*rx + 8*rz, which never crosses bit 31, and the
/// y half is picked by ry. ONE storage word answers "is any of these 8 solid",
/// which is what keeps the coarser gate as cheap per tap as the per-voxel one
/// was - the gate runs eight times per shaded pixel.
///
/// PINNED to VL_STEP == 2 by `voxlight::the_group_mask_matches_the_record_step`.
fn vl_group_occ(bi: i32, r: vec3<i32>) -> u32 {
    let pat = 0x00330033u << u32(2 * r.x + 8 * r.z);
    let b = bricks[bi];
    return select(b.occ_lo, b.occ_hi, r.y == 1) & pat;
}

/// Does the record cell hold a REAL OPAQUE OCCLUDER, so light must not be
/// interpolated through it?
///
/// The per-voxel field asked this of one voxel. A record cell is 8 of them, and
/// "ANY of the 8 is opaque" is the rule that keeps a one-voxel wall opaque at
/// the coarser spacing: a wall thinner than a record cell still lands inside
/// SOME cell, and that cell then vetoes every tap that would have crossed it.
/// (`voxlight_does_not_leak_through_a_one_voxel_wall` is the test.)
fn vl_group_blocks(bi: i32, r: vec3<i32>) -> bool {
    var m = vl_group_occ(bi, r);
    // Empty group: the common case in the lit shell, and it answers in one load.
    if (m == 0u) { return false; }
    let vbase = select(0, 32, r.y == 1);
    loop {
        if (m == 0u) { break; }
        let b = i32(firstTrailingBit(m));
        m = m & (m - 1u);
        // Foliage is a semi-transparent volume, not a wall: it carries light and
        // never vetoes. Stone, water and glass do. Same rule as the per-voxel
        // field, applied to whichever voxels of the group are occupied.
        if (!is_foliage_mat(brick_voxel_material(bi, vbase + b))) { return true; }
    }
    return false;
}

struct VoxLightParams {
    live_count: u32,
    round: u32,
    update_div: u32,
    sun_rays: u32,
    sun_cone: f32,
    light_count: u32,
    fold: f32,
    ao_strength: f32,
    // Length of the NEAR prefix of `vl_live_bricks`. The SWEEP walks that prefix
    // on the `update_div` cadence and the remainder `far_div` times more rarely -
    // the whole camera-awareness policy is these two numbers.
    near_count: u32,
    // Extra division applied to the FAR remainder, on top of `update_div`.
    far_div: u32,
    // Bricks in the URGENT list, which lives at VL_URGENT_BASE in
    // `vl_live_bricks`. These are dispatched IN FULL and before anything else:
    // they are the blocks whose record is unreadable or whose geometry just
    // changed, so they cannot wait for a sweep slice to come round.
    urgent_count: u32,
    // 1 when this dispatch also advances the periodic sweep, 0 when it is urgent
    // work only. The sweep is paced by SUN MOTION, not by frames, so most frames
    // of a fast-rendering session carry no sweep at all - and a frozen sun with
    // nothing dirty means the pass is not dispatched at all.
    sweep: u32,
};

struct VlPointLight {
    // xyz = WORLD position, w = radius in voxels.
    pos_radius: vec4<f32>,
    // rgb = radiance, a unused.
    color: vec4<f32>,
};

// Group 0, continuing the render layout: the light field is world data with a
// single owner, exactly like the brick pyramid it sits beside. A separate group
// would have to be added to every pipeline layout and bound in every pass,
// including eleven test harnesses, for no isolation benefit.
@group(0) @binding(22) var<storage, read_write> vl_pool: array<u32>;
@group(0) @binding(23) var<storage, read> vl_block_of_brick: array<u32>;
@group(0) @binding(24) var<storage, read> vl_live_bricks: array<u32>;
@group(0) @binding(25) var<uniform> vl_params: VoxLightParams;
@group(0) @binding(26) var<storage, read> vl_lights: array<VlPointLight>;

// Shared-exponent HDR packing for the local-light term. One word holds a
// radiance that can exceed 1.0 without an extra buffer.
fn vl_unpack_rgb9e5(p: u32) -> vec3<f32> {
    let e = i32((p >> 27u) & 0x1Fu);
    let scale = exp2(f32(e - 24));
    return vec3<f32>(f32(p & 0x1FFu), f32((p >> 9u) & 0x1FFu), f32((p >> 18u) & 0x1FFu)) * scale;
}

fn vl_pack_rgb9e5(c: vec3<f32>) -> u32 {
    let cc = max(c, vec3<f32>(0.0));
    let m = max(max(cc.r, cc.g), max(cc.b, 1e-6));
    let e = clamp(i32(floor(log2(m))) + 16, 0, 31);
    let q = clamp(vec3<i32>(round(cc * exp2(f32(24 - e)))), vec3<i32>(0), vec3<i32>(511));
    return u32(q.x) | (u32(q.y) << 9u) | (u32(q.z) << 18u) | (u32(e) << 27u);
}

/// One trilinear tap: where its record lives, and whether it must be dropped.
///
/// The two questions share an ENTIRE hierarchy descent - bounds test, toroidal
/// fold, brick index, in-brick voxel index - so they are answered together. This
/// used to be `vl_record_word` plus a separate `is_voxel_solid`, i.e. two full
/// descents per tap, eight times per shaded pixel.
struct VlTap {
    /// First pool word of the record, or VL_NONE when the cell is outside the
    /// window or its brick has no light block.
    word: u32,
    /// True when the record cell holds a REAL OPAQUE OCCLUDER, so light must not
    /// be interpolated through it. See `vl_group_blocks`.
    blocks: bool,
};

/// `rc` is a WORLD RECORD-CELL coordinate, i.e. a world voxel divided by
/// VL_STEP. The cell's low voxel is `rc * VL_STEP`, and because the window
/// extent and the brick dim are both multiples of VL_STEP, the whole cell folds
/// into one brick and never straddles a slot seam.
fn vl_tap(rc: vec3<i32>) -> VlTap {
    var o: VlTap;
    o.word = VL_NONE;
    o.blocks = false;
    let world_v = rc * VL_STEP;
    let rel = world_v - camera.world_origin;
    if (rel.x < 0 || rel.x >= WORLD_VOXELS_X
     || rel.y < 0 || rel.y >= WORLD_VOXELS_Y
     || rel.z < 0 || rel.z >= WORLD_VOXELS_Z) {
        return o;
    }
    let v = world_to_slot_voxel(world_v);
    let bp = v >> vec3<u32>(2u);
    let bi = world_brick_idx(bp.x, bp.y, bp.z);
    let block = vl_block_of_brick[u32(bi)];
    if (block == VL_NONE) { return o; }
    let local = v - bp * BRICK_DIM;
    o.word = block * VL_BLOCK_WORDS + u32(light_record_idx(local)) * VL_RECORD_WORDS;

    // OCCUPANCY, through the same mask hierarchy `is_voxel_solid` walks. The
    // tile and chunk masks are cleared the frame a streaming slot is recycled
    // while its brick occupancy words are not, so reading `brick_voxel_solid`
    // without them would see a recycled slot's stale geometry as solid.
    let tp = v >> vec3<u32>(4u);
    let cp = v >> vec3<u32>(6u);
    let ci = world_chunk_idx(cp.x, cp.y, cp.z);
    let tile_lin = (tp.x & 3) + (tp.z & 3) * 4 + (tp.y & 3) * 16;
    if (!chunk_has_child(ci, tile_lin)) { return o; }
    let ti = world_tile_idx(tp.x, tp.y, tp.z);
    let brick_lin = (bp.x & 3) + (bp.z & 3) * 4 + (bp.y & 3) * 16;
    if (!tile_has_child(ti, brick_lin)) { return o; }
    // OCCUPIED IS NOT THE SAME AS OPAQUE, and conflating them is what left over
    // half of a canopy view without a light record.
    //
    // The gate exists to stop light interpolating through a one-voxel WALL. A
    // canopy is not a wall: it is a semi-transparent volume that light passes
    // through, and the engine already draws that distinction everywhere else -
    // `shadow_voxel_occludes` runs foliage through a sub-voxel cutout instead of
    // blocking outright, and `ao_occluder` refuses to let decoration stamp AO.
    // The light field never inherited it, so on canopy - where the cell against a
    // leaf face is usually ANOTHER leaf - all eight taps were dropped and the
    // sampler had nothing to return.
    //
    // So foliage CARRIES a record (see `cs_voxel_light_update`) and never vetoes
    // a tap. Stone, water and glass still do: they are opaque, they occlude
    // shadow rays, and they are what the wall-leak rule was written for
    // (`voxlight_does_not_leak_through_a_one_voxel_wall`).
    o.blocks = vl_group_blocks(bi, local / VL_STEP);
    return o;
}

struct VoxLight {
    sun: f32,
    ao: f32,
    point: vec3<f32>,
    valid: bool,
};

fn vl_load(word: u32) -> VoxLight {
    var o: VoxLight;
    let w0 = vl_pool[word];
    o.sun = f32(w0 & 0xFFu) * (1.0 / 255.0);
    o.ao = f32((w0 >> 8u) & 0xFFu) * (1.0 / 255.0);
    o.point = vl_unpack_rgb9e5(vl_pool[word + 1u]);
    // epoch 0 means the record has never held a converged estimate (freshly
    // bound block, or a solid voxel), so it must not be blended in.
    o.valid = ((w0 >> 16u) & 0xFFu) != 0u;
    return o;
}

/// Opacity-gated trilinear fetch of the light field for a surface point.
///
/// This is the whole point of the rework: `sun` comes back as a CONTINUOUS
/// value, so no binary visibility test survives into the pixel and the
/// penumbra is smooth by construction rather than by TAA convergence. It is
/// also the ONLY sun-visibility and AO mechanism in the frame - there is no
/// per-pixel path underneath it any more.
fn voxlight_sample(p_world: vec3<f32>, n: vec3<f32>) -> VoxLight {
    var o: VoxLight;
    o.sun = 0.0;
    o.ao = 1.0;
    o.point = vec3<f32>(0.0);
    o.valid = false;

    // Early-out on the CENTRE voxel before doing anything eight times.
    //
    // MEASURED: with an unpopulated field the full eight-tap loop cost 1.25 ms
    // per frame at 1920x1080 on the terrain scene (shade 9.24 -> 10.01 ms) just
    // to discover there was nothing to read. Any voxel outside the lit shell -
    // open sky, deep interior, or anything past the pool ceiling - answers in
    // ONE table lookup instead of eight.
    //
    // `vl_tap` returns before the occupancy walk when the brick has no block, so
    // this miss costs exactly what the old `vl_record_word` cost.
    let ps = p_world + n * 0.5;
    if (vl_tap(vec3<i32>(floor(ps / VL_STEP_F))).word == VL_NONE) { return o; }

    // Step into the air voxel against the face, then place the lattice on RECORD
    // centres so the eight taps straddle the surface. A record sits at the centre
    // of the LOW voxel of its VL_STEP^3 cell, i.e. at `rc * VL_STEP + 0.5`, so
    // the lattice is uniform with spacing VL_STEP and the interpolation below is
    // the same trilinear it always was, one level coarser.
    //
    // Anchoring on the low voxel rather than on the cell's geometric centre is
    // deliberate: it keeps the sample point a VOXEL centre (safe as a shadow-ray
    // origin, and exactly what the per-voxel field did) and it keeps the nearest
    // record ~1.5 voxels off a surface whichever parity the surface has, instead
    // of alternating between 2 and 3.
    let g = (ps - vec3<f32>(0.5)) / VL_STEP_F;
    let b = floor(g);
    let f = g - b;
    let base = vec3<i32>(b);

    var acc_sun = 0.0;
    var acc_ao = 0.0;
    var acc_pt = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var k = 0u; k < 8u; k = k + 1u) {
        let off = vec3<i32>(i32(k & 1u), i32((k >> 1u) & 1u), i32((k >> 2u) & 1u));
        let fw = select(1.0 - f, f, vec3<bool>(off.x == 1, off.y == 1, off.z == 1));
        let w = fw.x * fw.y * fw.z;
        if (w <= 0.0) { continue; }
        let c = base + off;
        // Light lives in AIR and in FOLIAGE. Interpolating through an OPAQUE
        // cell is exactly how light leaks across a one-voxel wall, so those taps
        // are dropped and the surviving weights renormalised below; a leaf or a
        // grass tuft carries its own record and is read like any other.
        let tap = vl_tap(c);
        if (tap.blocks || tap.word == VL_NONE) { continue; }
        let s = vl_load(tap.word);
        if (!s.valid) { continue; }
        acc_sun = acc_sun + s.sun * w;
        acc_ao = acc_ao + s.ao * w;
        acc_pt = acc_pt + s.point * w;
        wsum = wsum + w;
    }
    if (wsum > 0.0) {
        let inv = 1.0 / wsum;
        o.sun = acc_sun * inv;
        o.ao = acc_ao * inv;
        o.point = acc_pt * inv;
        o.valid = true;
    }
    return o;
}

fn sky(dir: vec3<f32>) -> vec3<f32> {
    return sky_color(dir);
}

/// Sky-like colour with **no stars, sun disc, or cloud emitters** — used for
/// the fog blend on distant terrain so far-away blocks don't visibly show
/// pinpoint stars through them at night.
// Atmospheric fog amount at hit distance t. Clear out to FOG-start, then a
// linear ramp saturating by ~350: the old t/280-from-zero curve hazed the
// whole midfield ("everything looks a bit foggy"); this keeps the near/mid
// field crisp while still saturating before the 400+ water/LOD switches so
// fog keeps hiding them. ONE curve for terrain, water, glass AND clouds -
// the horizon melts consistently (clouds previously stayed fully crisp while
// the ground fogged out).
fn fog_amount(t: f32) -> f32 {
    return clamp((t - 15.0 * VOXELS_PER_METRE) / (85.0 * VOXELS_PER_METRE), 0.0, 0.85);
}

fn fog_atmospheric(dir: vec3<f32>) -> vec3<f32> {
    let s = sun_dir();
    let day_t = sun_intensity(s);
    let up = clamp(dir.y, -0.2, 1.0);
    let zen_f = smoothstep(0.0, 0.55, up);
    let day_zenith  = vec3<f32>(0.22, 0.50, 0.95);
    let day_horizon = vec3<f32>(0.78, 0.86, 0.95);
    let day_sky = mix(day_horizon, day_zenith, zen_f);
    let night_zenith  = vec3<f32>(0.025, 0.035, 0.065);
    let night_horizon = vec3<f32>(0.055, 0.065, 0.110);
    let night_sky = mix(night_horizon, night_zenith, zen_f);
    return mix(night_sky, day_sky, day_t);
}

struct Hit {
    hit: bool,
    mat: u32,
    normal: vec3<f32>,
    voxel: vec3<i32>,
    last_axis: i32,
    t_hit: f32,
    // Sub-voxel colour tint (leaf shade, blade gradient, petal/stem colour).
    // (1,1,1) for plain cube hits. Carried in the Hit so secondary rays can't
    // read a stale value, which a module-global tint could leak.
    tint: vec3<f32>,
};

// IGN (interleaved gradient noise) — high-quality low-discrepancy per-pixel
// hash. Used for shadow PCF jitter so adjacent pixels get well-distributed
// offsets without forming visible patterns.
fn ign(x: f32, y: f32, frame: f32) -> f32 {
    return fract(52.9829189 * fract(0.06711056 * (x + frame * 5.588238) + 0.00583715 * (y + frame * 4.182857)));
}

// Profiling toggles (const-folded out when false): PROFILE_FLAT skips all
// shading to isolate traversal cost; PROFILE_NO_L4 skips the L4/chunk coarse
// skips. Both off for normal rendering.
override PROFILE_FLAT_F: f32 = 0.0;
const PROFILE_FLAT: bool = false;
const PROFILE_NO_L4: bool = false;

// Diagnostic override (flicker rig): freeze the per-frame shading-noise
// phase (shadow cone jitter, GI sample jitter) on a static camera. The TAA
// camera sub-pixel jitter is separate and unaffected. Default = live
// behaviour.
override JIT_PHASE_FREEZE: f32 = 0.0;

// Transparent-pass cost-split toggles (timing-only, const-folded out in
// normal builds): each disables one component of water shading so the
// harness attributes transp milliseconds by differencing runs. Never
// shipped on - the outputs they produce are placeholders.
override PROF_TRANSP_NO_REFR: f32 = 0.0;
override PROF_TRANSP_REFR_FLATSHADE: f32 = 0.0;

// COVERAGE PROBE for the per-voxel light field (timing/diagnostic only). Every
// shaded surface returns GREEN when `voxlight_sample` answered for it and RED
// when it fell through to the per-pixel path, so a readback counts the exact
// fraction of shaded pixels the field actually serves. That fraction is the
// precondition for deleting the fallback: the plan has called for removing the
// per-pixel shadow cone and AO since the beginning, and doing it without first
// knowing what the field does NOT cover would render those surfaces black.
override PROF_VLF_COVERAGE: f32 = 0.0;

// Primary ray direction for a normalized screen uv (0..1). Shared by cs_main
// (full-res, jittered) and cs_clouds (half-res). aspect uses the full-res
// resolution ratio, which is identical at half res.
fn ray_dir_uv(uv: vec2<f32>) -> vec3<f32> {
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let aspect = camera.resolution.x / camera.resolution.y;
    return normalize(
        camera.forward
        + camera.right * (ndc.x * camera.tan_half_fov * aspect)
        + camera.up    * (ndc.y * camera.tan_half_fov)
    );
}

// t at which the ray enters the cloud slab (or a large value if it never does,
// looking away from / parallel to the slab). Used to reapply terrain occlusion
// to the precomputed half-res clouds without re-marching.
fn cloud_slab_near(dir: vec3<f32>) -> f32 {
    // Camera INSIDE the slab: every direction starts in cloud immediately. The
    // horizontal-ray epsilon below must not fire here - it carved a 1-2 px "no
    // cloud" band at the exact horizon, seen as a dark line THROUGH the clouds
    // when flying inside them.
    if (camera.origin.y >= CLOUD_BASE && camera.origin.y <= CLOUD_TOP) { return 0.0; }
    // Outside the slab, a near-horizontal ray never reaches it.
    if (abs(dir.y) < 1e-3) { return 1e9; }
    let inv_dy = 1.0 / dir.y;
    var t_in  = (CLOUD_BASE - camera.origin.y) * inv_dy;
    var t_out = (CLOUD_TOP  - camera.origin.y) * inv_dy;
    if (t_in > t_out) { let tmp = t_in; t_in = t_out; t_out = tmp; }
    let t_start = max(t_in, 0.0);
    if (t_out <= t_start) { return 1e9; }
    return t_start;
}

// Half-res volumetric pass: march the clouds once per 2×2 block, store into
// cloud_out. cs_main bilinearly upsamples + composites. No terrain occlusion
// here (cs_main reapplies it cheaply via cloud_slab_near) — ~4× fewer marches.
@compute @workgroup_size(8, 8, 1)
fn cs_clouds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let half_res = (vec2<u32>(camera.resolution) + vec2<u32>(1u)) / vec2<u32>(2u);
    if (gid.x >= half_res.x || gid.y >= half_res.y) { return; }
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / vec2<f32>(half_res);
    let dir = ray_dir_uv(uv);
    let clouds = render_clouds(camera.origin, dir, 1e9, vec2<f32>(f32(gid.x), f32(gid.y)));
    textureStore(cloud_out, vec2<i32>(i32(gid.x), i32(gid.y)), clouds);
}

// The per-cell FACET of the last water surface the primary trace() resolved -
// its quantized slope, and its band + shore mask packed by `water_pack_facet` -
// transported to the deferred record write in cs_main. Private vars instead of
// Hit fields on purpose: growing Hit bloats registers in every tracer
// (trace_no_water runs per refraction ray) - measured +1.8 ms on the water
// scenario when Hit last grew. trace() resets both, so a far water cube-top can
// never read a stale facet.
var<private> water_facet_grad: vec2<f32> = vec2<f32>(0.0);
var<private> water_facet_code: u32 = 0u;

// Axis-aligned face normal <-> small code, for the deferred transparent buffer.
fn encode_face_normal(n: vec3<f32>) -> u32 {
    if (n.x > 0.5) { return 0u; } else if (n.x < -0.5) { return 1u; }
    else if (n.y > 0.5) { return 2u; } else if (n.y < -0.5) { return 3u; }
    else if (n.z > 0.5) { return 4u; } else { return 5u; }
}
fn decode_face_normal(c: u32) -> vec3<f32> {
    if (c == 0u) { return vec3<f32>(1.0, 0.0, 0.0); }
    if (c == 1u) { return vec3<f32>(-1.0, 0.0, 0.0); }
    if (c == 2u) { return vec3<f32>(0.0, 1.0, 0.0); }
    if (c == 3u) { return vec3<f32>(0.0, -1.0, 0.0); }
    if (c == 4u) { return vec3<f32>(0.0, 0.0, 1.0); }
    return vec3<f32>(0.0, 0.0, -1.0);
}

// Is the eye below the water surface? Shared by cs_main (trace-path choice)
// and cs_compose (underwater post-effect). Reads the SAME `water_facet` the
// tracer draws, so the surface the camera thinks it is under is the surface on
// screen - exactly, now, rather than to within the old corner-vs-centre
// mismatch: the facet is a per-cell constant, so there is nothing left to
// approximate.
fn camera_in_water() -> bool {
    let cam_voxel_chk = vec3<i32>(floor(camera.origin));
    let cam_mat_chk = voxel_material_at(cam_voxel_chk);
    var in_water = is_water_mat(cam_mat_chk);
    if (in_water && !is_water_mat(voxel_material_at(cam_voxel_chk + vec3<i32>(0, 1, 0)))) {
        let lf = f32(cam_mat_chk - MAT_WATER_L1 + 1u) * 0.125;
        let fc = water_facet(cam_voxel_chk, lf, 0.0);
        let lp_y = camera.origin.y - f32(cam_voxel_chk.y);
        in_water = lp_y <= fc.h;
    }
    return in_water;
}

// Deferred transparent shading pass (#16): runs after cs_main, shades only the
// pixels cs_main flagged as water-top/glass (the expensive reflection/refraction
// + dispersion), then re-applies clouds + god-rays to match cs_main's compositing.
// Opaque/sky pixels early-out, leaving cs_main's output untouched.
@compute @workgroup_size(8, 8, 1)
fn cs_transparent(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = vec2<i32>(camera.resolution);
    if (i32(gid.x) >= res.x || i32(gid.y) >= res.y) { return; }
    let rec = transp_buf[gid.y * u32(res.x) + gid.x];
    if (rec.y == TR_NONE) { return; } // not a transparent pixel

    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5) + camera.jitter) / camera.resolution;
    let dir = ray_dir_uv(uv);

    var hit: Hit;
    hit.hit = true;
    hit.t_hit = bitcast<f32>(rec.x);
    hit.last_axis = 0;
    hit.voxel = vec3<i32>(0);
    hit.tint = vec3<f32>(1.0);
    var col: vec3<f32>;
    if (rec.y != TR_GLASS) {
        hit.mat = MAT_WATER_L8;
        if (rec.y == TR_WATER_TOP) {
            // Top surface: the cell's QUANTIZED slope, straight from the
            // record. No per-pixel wave-field term is added on top any more -
            // that term is exactly what made the surface read smooth, and one
            // constant normal per cell is what "flat shaded" means. Every pixel
            // of a facet resolves the identical normal by construction now,
            // rather than to within a tolerance.
            hit.normal = water_facet_normal(unpack2x16float(rec.z));
        } else {
            hit.normal = decode_face_normal(rec.z);
        }
        // ONE shade_water_top call site: it inlines the refraction trace, and
        // duplicating it doubles cs_transparent's code size (measured ~+1.8 ms
        // on the water scenario).
        col = shade_water_top(hit, camera.origin, dir, rec.w);
    } else {
        hit.mat = MAT_GLASS;
        hit.normal = decode_face_normal(rec.z);
        col = shade_glass(hit, camera.origin, dir);
    }

    // Clouds/god rays/underwater are applied by cs_compose (per-frame terms
    // must not bake into cached pixels) - store the plain shaded colour.
    textureStore(output_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(col, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = vec2<u32>(camera.resolution);
    if (gid.x >= res.x || gid.y >= res.y) { return; }

    // Temporal-differential gate.
    let tiles_w = (i32(camera.resolution.x) + 7) / 8;
    let tile_x = i32(gid.x) / 8;
    let tile_y = i32(gid.y) / 8;
    let tile_idx = tile_x + tile_y * tiles_w;
    let word = tile_idx >> 5;
    let bit = tile_idx & 31;
    if ((tile_dirty[word] & (1u << u32(bit))) == 0u) { return; }

    // Per-pixel + per-frame jitter, threaded through shading.
    // Per-frame-varying jitter ONLY while TAA accumulates (static camera):
    // unaveraged time-varying jitter is pure shimmer on a moving camera, so
    // motion freezes the phase - every pixel samples the same cone offset each
    // frame (temporally rock-solid, spatially soft-dithered penumbra), and the
    // time-varying softness resumes the moment the camera rests.
    let jit_phase = select(select(0.0, camera.time * 60.0, camera.taa_blend > 0.0),
                           0.0, JIT_PHASE_FREEZE > 0.5);
    let pix_jitter = ign(f32(gid.x), f32(gid.y), jit_phase);

    // Sub-pixel jitter for temporal anti-aliasing (zero unless accumulating).
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5) + camera.jitter) / camera.resolution;
    let dir = ray_dir_uv(uv);

    // Beam pre-pass: read the coarse first-tile-hit t for this 8×8 block
    // and start the per-pixel ray there. A 16-voxel safety margin (one
    // tile) keeps grazing-angle pixels from over-skipping.
    let beam_xy = vec2<i32>(i32(gid.x / 8u), i32(gid.y / 8u));
    let beam_t_raw = textureLoad(beam_depth, beam_xy, 0).r;
    let beam_skip = max(0.0, beam_t_raw - 16.0);
    let ray_origin = camera.origin + dir * beam_skip;

    // If the camera itself sits in a water voxel, the primary ray must
    // skip water (we're inside it) and find the first NON-water surface.
    // Otherwise the ray would hit the water voxel it's already inside and
    // render an opaque wall in our face.
    let cam_in_water = camera_in_water();
    var hit: Hit;
    // Hardware-RT primary (RT variant, RT_PRIMARY override): the RT core does
    // the empty-space traversal. Off by default; falls through to the software
    // beam + hierarchical DDA below.
    var rt_done = false;
    hit = rt_primary_or_none(camera.origin, dir, cam_in_water, &rt_done);
    if (!rt_done) {
        if (cam_in_water) {
            // Skip beam-skip when underwater — beam pre-pass doesn't know about
            // the camera being inside water and may have advanced past real geo.
            hit = trace_no_water(camera.origin, dir, MAX_RAY_DIST);
        } else {
            hit = trace(ray_origin, dir);
            if (hit.hit) {
                hit.t_hit = hit.t_hit + beam_skip;
            }
        }
    }
    var col: vec3<f32>;
    if (PROFILE_FLAT || PROFILE_FLAT_F > 0.5) {
        col = select(sky(dir), palette[hit.mat].rgb, hit.hit);
        textureStore(output_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(col, 1.0));
        return;
    }
    // Lighting G-buffer for cs_taa's reprojection and the grass pass's blade
    // lighting; sentinel position = "not a terrain pixel, do not reproject".
    var gbuf = vec4<f32>(1e9, 1e9, 1e9, 0.0);
    // Deferred transparent record (kind TR_NONE = opaque/none).
    var transp = vec4<u32>(0u);
    if (hit.hit) {
        if (is_water_mat(hit.mat)) {
            // Defer ALL water (patches, walls, undersides) to cs_transparent.
            // Up-facing surface hits carry the rest-gradient; walls and
            // undersides carry their axis face code.
            if (hit.normal.y > 0.9) {
                transp = vec4<u32>(bitcast<u32>(hit.t_hit), TR_WATER_TOP,
                                   pack2x16float(water_facet_grad), water_facet_code);
            } else {
                transp = vec4<u32>(bitcast<u32>(hit.t_hit), TR_WATER_FACE,
                                   encode_face_normal(hit.normal), 0u);
            }
            col = sky(dir); // cheap placeholder (overwritten by cs_transparent)
        } else if (hit.mat == MAT_GLASS) {
            transp = vec4<u32>(bitcast<u32>(hit.t_hit), TR_GLASS,
                               encode_face_normal(hit.normal), 0u);
            col = sky(dir);
        } else {
            let gi_p = camera.origin + dir * hit.t_hit;
            let indirect = indirect_light(gi_p, hit.normal, pix_jitter,
                                          vec2<i32>(i32(gid.x), i32(gid.y)), hit.t_hit, dir);
            var out_light = vec2<f32>(0.0);
            col = shade(hit, camera.origin, dir, vl_unknown(), indirect,
                        &out_light);
            // Only stable cube faces go in the G-buffer. An oblique sub-voxel hit
            // (grass or flower cross-quad) has no well-defined face position to
            // reproject by.
            if (hit.last_axis >= 0) {
                let hitpos_rel =
                    (camera.origin - vec3<f32>(camera.world_origin)) + dir * hit.t_hit;
                gbuf = vec4<f32>(hitpos_rel,
                                 bitcast<f32>(pack_light_cache(out_light.x, out_light.y)));
            }
        }
    } else {
        col = sky(dir);
    }
    textureStore(light_out, vec2<i32>(i32(gid.x), i32(gid.y)), gbuf);
    transp_buf[gid.y * u32(res.x) + gid.x] = transp;

    // Remote-player markers: each player is a 1.6×2×1.6 box in world coords.
    // Pick the closest hit (player vs terrain) and override colour if a
    // player marker wins.
    let inv_dir_p = vec3<f32>(safe_inv(dir.x), safe_inv(dir.y), safe_inv(dir.z));
    var closest_t = select(1e30, hit.t_hit, hit.hit);
    for (var pi: u32 = 0u; pi < players.count; pi = pi + 1u) {
        let pp = players.positions[pi];
        let center = pp.xyz;
        let bmin = center - vec3<f32>(0.8, 0.5, 0.8);
        let bmax = center + vec3<f32>(0.8, 1.5, 0.8);
        let t = ray_aabb_t(camera.origin, inv_dir_p, bmin, bmax);
        if (t < closest_t) {
            closest_t = t;
            let pid = u32(pp.w);
            col = player_color_for(pid);
        }
    }
    textureStore(depth_out, vec2<i32>(i32(gid.x), i32(gid.y)),
                 vec4<f32>(min(closest_t, 1e9), 0.0, 0.0, 0.0));

    // Clouds, god rays and the underwater post-effect are NOT applied here:
    // they are per-frame-varying, and anything time-varying baked into this
    // TILE-GATED pass freezes at a different phase per 8x8 tile (the
    // rotating anim-refresh re-traces ~1/8 of tiles per frame), which is
    // the diffused checkerboard this engine fought repeatedly. cs_compose
    // applies them for EVERY pixel EVERY frame from the exported depth.
    textureStore(output_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(col, 1.0));
}

// Full-screen per-frame compose: geometry colour + volumetric clouds by
// depth + god rays + underwater post-effect. Runs after cs_transparent and
// before TAA, unconditionally for every pixel - per-frame-varying terms
// live HERE and nowhere upstream, so the temporal-differential tile cache
// can never freeze them out of phase.
@compute @workgroup_size(8, 8, 1)
fn cs_compose(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = vec2<i32>(camera.resolution);
    if (i32(gid.x) >= res.x || i32(gid.y) >= res.y) { return; }
    let pix = vec2<i32>(i32(gid.x), i32(gid.y));
    var col = textureLoad(geom_in, pix, 0).rgb;
    let t_hit = textureLoad(depth_in, pix, 0).r;
    let uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5) + camera.jitter) / camera.resolution;
    let dir = ray_dir_uv(uv);

    // Density-driven cloud shadows: every ground pixel samples the SAME
    // cloud_density field the sky pass renders, along the sun ray through
    // the slab. Denser cloud overhead = deeper shade, and the shadow field
    // drifts with the clouds by construction. THREE FIXED taps averaged -
    // NOT a per-frame IGN-jittered tap: the jitter only vanished if TAA
    // averaged it, which it cannot on fluttering canopy edges or the
    // post-TAA falling-leaf pass, so leaves flickered under cloud shade.
    // cloud_density is smooth, so fixed taps need no dither and are stable
    // frame-to-frame (only the slow cloud drift moves them). Full-screen
    // per-frame only (mechanism rule); skipped for near-horizontal sun and
    // scaled by sun intensity so night is untouched.
    if (t_hit < 1.0e8) {
        let s = sun_dir();
        let s_int = sun_intensity(s);
        // Low-sun fade: the tap positions scale as 1/s.y, so near the horizon
        // they RACE horizontally (d/ds ~ 1/s.y^2) and the cloud pattern sweeps
        // the ground as accelerating waves of light ("rings washing over the
        // land at sunset"). Physically a near-grazing sun through a cloud layer
        // is extinct and diffuse - no crisp travelling shadows - so the term
        // fades out smoothly before the sweep regime instead of hard-cutting.
        // Fade window raised twice: residual rings were still visible in the
        // old 0.10-0.30 band, where partial-strength taps still race (and the
        // higher, denser clouds made both the sweep faster and the shade
        // stronger). Below 0.22 the term is fully OFF.
        let horiz_fade = smoothstep(0.22, 0.42, s.y);
        if (s_int > 0.0 && horiz_fade > 0.0) {
            let p_ground = camera.origin + dir * t_hit;
            let st_in = (CLOUD_BASE - p_ground.y) / s.y;
            let st_out = (CLOUD_TOP - p_ground.y) / s.y;
            // NOTE: do NOT gate this with a per-pixel sun-occlusion trace.
            // `dir` carries the TAA sub-pixel jitter, so a hard occluded/not
            // boolean flips frame-to-frame at shadow edges and the cloud
            // shade JITTERS while standing still (TAA never settles). Roofed
            // surfaces are instead kept dim by the probe GI (less skylight
            // reaches them), so an ungated cloud multiply over them is negligible.
            if (st_in > 0.0) {
                var d = 0.0;
                d = d + cloud_density_coarse(p_ground + s * mix(st_in, st_out, 0.2), camera.sun_time);
                d = d + cloud_density_coarse(p_ground + s * mix(st_in, st_out, 0.5), camera.sun_time);
                d = d + cloud_density_coarse(p_ground + s * mix(st_in, st_out, 0.8), camera.sun_time);
                let occl = 1.0 - exp(-d * 1.1);
                col = col * (1.0 - occl * 0.42 * s_int * horiz_fade);
            }
        }
    }

    let uv_cloud = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / camera.resolution;
    let clouds = textureSampleLevel(cloud_in, cloud_samp, uv_cloud, 0.0);
    let t_cloud = cloud_slab_near(dir);
    if (t_hit >= t_cloud) {
        // Same distance-haze curve as terrain: far clouds melt into the horizon
        // instead of hanging crisp over fogged-out ground.
        let cf = 1.0 - fog_amount(t_cloud);
        col = col * (1.0 - clouds.a * cf) + clouds.rgb * cf;
    }

    // God rays: depth-weighted 4-tap upsample of the half-res occlusion
    // fraction (cs_godrays); colour and phase are exact per pixel, so the
    // shared half-res part is only the smooth occlusion field - no haloing
    // across silhouettes thanks to the depth weights.
    {
        let s = sun_dir();
        let s_int = sun_intensity(s);
        let cos_sun = dot(dir, s);
        let phase = phase_hg(cos_sun, 0.7) * 4.0;
        if (s_int > 0.0 && phase >= 0.05) {
            let half_res = (res + vec2<i32>(1)) / 2;
            let base_idx = res.x * res.y;
            let hx = clamp(i32(gid.x) / 2, 0, half_res.x - 2);
            let hy = clamp(i32(gid.y) / 2, 0, half_res.y - 2);
            let t_ref = min(t_hit, 200.0);
            var fsum = 0.0;
            var wsum = 0.0;
            for (var k: i32 = 0; k < 4; k = k + 1) {
                let rec = transp_buf[base_idx + (hy + (k >> 1)) * half_res.x + hx + (k & 1)];
                let f = bitcast<f32>(rec.x);
                let d = min(bitcast<f32>(rec.y), 200.0);
                let wd = 1.0 / (1.0 + abs(d - t_ref) * 0.15);
                fsum = fsum + f * wd;
                wsum = wsum + wd;
            }
            col += sun_color(s) * (fsum / max(wsum, 1e-4)) * phase * 0.22 * s_int;
        }
    }

    if (camera_in_water()) {
        let t_eye = min(t_hit, 80.0);
        let absorb = vec3<f32>(0.32, 0.16, 0.06);
        let trans = exp(-absorb * (t_eye * 0.18));
        let water_col = vec3<f32>(0.04, 0.18, 0.28);
        col = col * trans + water_col * (vec3<f32>(1.0) - trans);
    }

    textureStore(output_tex, pix, vec4<f32>(col, 1.0));
}

const MAT_GRASS:           u32 = 2u;
const MAT_WATER_L1:        u32 = 5u;
const MAT_WATER_L8:        u32 = 12u;
const MAT_LEAVES:          u32 = 14u;
const MAT_GLASS:           u32 = 18u;
const MAT_LEAVES_BIRCH:    u32 = 25u;
const MAT_LEAVES_PINE:     u32 = 26u;
const MAT_LEAVES_AUTUMN:   u32 = 27u;
const MAT_FLOWER:          u32 = 30u;
const MAT_TALL_GRASS:      u32 = 31u;
const MAT_LEAF_FRINGE:     u32 = 33u;
const MAT_TALL_GRASS_DRY:  u32 = 34u;
const MAT_TURF:            u32 = 35u;
const MAT_BUSH:            u32 = 36u;
const MAT_TREE_TEST:       u32 = 37u;

fn is_water_mat(m: u32) -> bool {
    return m >= MAT_WATER_L1 && m <= MAT_WATER_L8;
}
fn is_transparent_mat(m: u32) -> bool {
    return is_water_mat(m) || m == MAT_GLASS;
}
fn is_uniform_optimisable(m: u32) -> bool {
    // Skip foliage (sub-voxel cutout) + transparent (refraction / wave anim).
    return !is_foliage_mat(m) && !is_transparent_mat(m);
}
fn is_foliage_mat(m: u32) -> bool {
    return m == MAT_LEAVES || m == MAT_LEAVES_BIRCH
        || m == MAT_LEAVES_PINE || m == MAT_LEAVES_AUTUMN
        || m == MAT_FLOWER || m == MAT_TALL_GRASS
        || m == MAT_LEAF_FRINGE || m == MAT_TALL_GRASS_DRY
        || m == MAT_TURF || m == MAT_BUSH || m == MAT_TREE_TEST;
}
fn is_leaf_block_mat(m: u32) -> bool {
    return m == MAT_LEAVES || m == MAT_LEAVES_BIRCH
        || m == MAT_LEAVES_PINE || m == MAT_LEAVES_AUTUMN;
}
// Ground decoration (single sub-voxel sprites). Far away these must NOT be
// drawn as solid cubes (that's the "pink flower blocks" bug) — they vanish
// instead. Leaves, by contrast, stay solid cubes far away so tree canopies
// don't disappear.
fn is_decoration_mat(m: u32) -> bool {
    return m == MAT_FLOWER || m == MAT_TALL_GRASS || m == MAT_LEAF_FRINGE
        || m == MAT_TALL_GRASS_DRY || m == MAT_TURF || m == MAT_BUSH
        || m == MAT_TREE_TEST;
}

struct SubHit {
    hit: bool,
    t_hit: f32,
    normal: vec3<f32>,
    color_tint: vec3<f32>,  // multiplier for palette colour (1,1,1 = no change)
}

// ---------- Wind: ONE consistent direction at any moment ----------
// A single wind direction blowing across the whole map, with slow rotation
// and time-varying strength. Per-voxel phase offsets the strength so not
// every blade peaks together, but the direction is shared so the whole
// scene leans the same way at the same time.
// Slower, smoother wind — direction rotates very gradually, strength
// oscillates calmly. Was way too fast before, made foliage look glitchy.
// Formula (CPU side, camera.rs): angle = time*0.04 + 0.4*sin(time*0.12),
// dir = (cos, sin). Uploaded per frame; reading it here is free.
fn wind_dir_now() -> vec2<f32> {
    return vec2<f32>(camera.wind_x, camera.wind_z);
}

// Traveling gust field: plane waves moving ALONG the wind direction in
// world XZ (a broad front every ~314 voxels plus a ~57-voxel ripple), range
// [0.25, 1.0]. This is what stops the whole map swaying in lockstep - a
// gust visibly travels across a field. Long wavelengths keep neighbouring
// blocks coherent (one tree never tears apart).
// SHADOW rays sample the foliage cutouts at a FROZEN wind phase: one shadow
// sample per pixel per frame cannot resolve moving blades - the binary dapple
// churns, then the staleness cache pops it ("small shadows jumping around").
// Primary rays keep full animation (trees visibly wave); the dapple pattern
// still drifts with the SUN, it just stops boiling. Set around the cutout
// tests in shadow_voxel_occludes only.
var<private> shadow_wind_freeze: bool = false;

// The voxel a light-field gather ORIGINATES in, so its own contents cannot
// occlude it (`shadow_voxel_occludes`).
//
// Only foliage needs this, and only since foliage started carrying records. An
// air voxel can never occlude a ray leaving it, but a leaf voxel gathers at its
// OWN centre, and `trace_any` (and `rt_brick_occludes`) test the origin cell
// first - so without this every leaf's sun ray would be decided by whether the
// tuft cutout happens to sit in front of the block centre, and a canopy would
// gather a hash pattern instead of light. A record is a property of the cell's
// NEIGHBOURHOOD; its own contents are what the record describes, not an occluder
// of it.
//
// Guarded by a bool rather than an impossible sentinel coordinate: `&&`
// short-circuits, so the render path pays one bool test per occluding candidate
// and never the coordinate compare.
//
// The skip is a RECORD CELL, not a single voxel: a record covers a VL_STEP^3
// group, so all of the group's own foliage has to be excluded, not just the one
// voxel the sample point happens to sit in.
var<private> shadow_skip_active: bool = false;
var<private> shadow_skip_voxel: vec3<i32> = vec3<i32>(0);

fn wind_time() -> f32 {
    return select(camera.time, 41.7, shadow_wind_freeze);
}

fn wind_gust(p_xz: vec2<f32>, wdir: vec2<f32>) -> f32 {
    let s = dot(p_xz, wdir);
    let front  = 0.5 + 0.5 * sin(s * 0.020 - wind_time() * 0.9);
    let ripple = 0.5 + 0.5 * sin(s * 0.11  - wind_time() * 2.1);
    return 0.25 + 0.75 * front * (0.6 + 0.4 * ripple);
}

fn wind_offset(voxel_min: vec3<f32>, phase: f32, base_amp: f32) -> vec2<f32> {
    let wdir = wind_dir_now();
    // Gust is a function of voxel_min only, so both cross planes and both
    // tuft quads of one block share a single shear (the anti-X-split
    // contract in sprite_cross_hit).
    let strength = base_amp * wind_gust(voxel_min.xz, wdir)
        * (0.70 + 0.30 * sin(wind_time() * 0.55 + phase));
    return wdir * strength;
}

// ---------- LEAVES: Motschen's Better Leaves, ported exactly ----------------
// Geometry and textures from "Motschen's Better Leaves Lite"
// (github.com/TeamMidnightDust/BetterLeavesLite, MIT; oak_leaves.png,
// birch_leaves.png and spruce_leaves.png converted by
// examples/convert_tuft.rs): every leaf block is a cutout CUBE whose faces
// sample the CENTRE 16x16 of the species' pre-rounded 32x32 tuft texture,
// PLUS two big double-sided diagonal quads (species-scaled, rotated 22.5 and
// -45 degrees about Y, one slightly off-centre) carrying the FULL round
// ragged tuft. Four variants (the pack's blockstate y-rotations 0/90/180/270)
// are picked per block by hash. The huge overhanging tufts from every block
// interleave into dense bushy canopies.
fn tuft_texel(tuft: u32, x: u32, y: u32) -> u32 {
    let bit = (y * 32u + x) * 2u;
    let w = sprites[TUFT_BASE_WORDS + tuft * 64u + (bit >> 5u)];
    return (w >> (bit & 31u)) & 3u;
}
// Tuft tones: 1 = dark, 2 = mid, 3 = bright (greyscale in the pack, tinted
// by the leaf palette colour here just like Minecraft's biome tint).
fn tuft_tone(val: u32, scale: f32) -> vec3<f32> {
    var b = 0.92;
    if (val == 1u) { b = 0.60; }
    if (val == 3u) { b = 1.24; }
    return vec3<f32>(b * scale);
}

// Which tuft texture a leaf material carries. Oak and autumn share the oak
// art (autumn differs by palette + mottle); birch and pine have their own
// ports from the same pack.
fn leaf_tuft_index(mat: u32) -> u32 {
    if (mat == MAT_LEAVES_BIRCH) { return TUFT_BIRCH; }
    if (mat == MAT_LEAVES_PINE)  { return TUFT_PINE; }
    return TUFT_OAK;
}

// Per-species quad half-extents (u, v): pine tufts narrower and taller
// (conifer), birch a touch wider and shorter, oak/autumn the original
// 2.3 x 2.0 blocks.
fn leaf_quad_ext(tuft: u32) -> vec2<f32> {
    if (tuft == TUFT_PINE)  { return vec2<f32>(0.95, 1.05); }
    if (tuft == TUFT_BIRCH) { return vec2<f32>(1.05, 0.95); }
    return vec2<f32>(1.15, 1.0);
}

// Autumn per-voxel mottle: a warm red-orange to gold-green patchwork so an
// autumn canopy reads as turning leaves, not one flat orange.
fn leaf_species_tint(mat: u32, vh: f32) -> vec3<f32> {
    if (mat == MAT_LEAVES_AUTUMN) {
        return mix(vec3<f32>(1.20, 0.72, 0.45), vec3<f32>(0.95, 1.25, 0.75), fract(vh * 8.0));
    }
    return vec3<f32>(1.0);
}

// Material at voxel+off, with a register-resident fast path when the offset
// stays inside the DDA's already-loaded home brick (slot_v/bp/bi are the hit
// cell's slot voxel, brick coords and brick index the trace loop already
// holds). A negative or out-of-brick candidate falls back to the full
// hierarchical point query; no signed % anywhere (arithmetic >> keeps
// negatives out of the fast path by failing the bp comparison). Shared by
// canopy AO and the water surface neighbourhood.
fn neighbor_material(voxel: vec3<i32>, slot_v: vec3<i32>, bp: vec3<i32>, bi: i32, off: vec3<i32>) -> u32 {
    let s = slot_v + off;
    let nbp = s >> vec3<u32>(2u);
    if (nbp.x == bp.x && nbp.y == bp.y && nbp.z == bp.z) {
        let local = s - nbp * BRICK_DIM;
        let vi = brick_voxel_idx(local.x, local.y, local.z);
        if (!brick_voxel_solid(bi, vi)) { return 0u; }
        return brick_voxel_material(bi, vi);
    }
    return voxel_material_at(voxel + off);
}

// Occluders for canopy AO: anything solid except the invisible fringe shell
// and ground decoration (counting those would darken faces with invisible
// geometry - the exact bug generic cube AO has on canopies, see shade()).
fn leaf_occluder(voxel: vec3<i32>, slot_v: vec3<i32>, bp: vec3<i32>, bi: i32, off: vec3<i32>) -> bool {
    let m = neighbor_material(voxel, slot_v, bp, bi, off);
    return m != 0u && !is_decoration_mat(m);
}

// Sky-weighted 6-probe occupancy AO for leaf cells, range [0.36, 1.0]:
// cells under more canopy darken, exposed crown cells stay lit, so canopies
// read volumetric instead of flat. Evaluated at the DDA cell containing the
// visible point (fringe-shell hits are exposed by construction and stay
// bright). Replaces generic cube AO for leaf materials - exactly one
// occlusion mechanism per material class.
fn leaf_canopy_ao(cell: vec3<i32>, slot_v: vec3<i32>, bp: vec3<i32>, bi: i32) -> f32 {
    var occ = 0.0;
    if (leaf_occluder(cell, slot_v, bp, bi, vec3<i32>(0, 1, 0)))  { occ += 0.28; }
    if (leaf_occluder(cell, slot_v, bp, bi, vec3<i32>(0, 2, 0)))  { occ += 0.12; }
    if (leaf_occluder(cell, slot_v, bp, bi, vec3<i32>(1, 0, 0)))  { occ += 0.06; }
    if (leaf_occluder(cell, slot_v, bp, bi, vec3<i32>(-1, 0, 0))) { occ += 0.06; }
    if (leaf_occluder(cell, slot_v, bp, bi, vec3<i32>(0, 0, 1)))  { occ += 0.06; }
    if (leaf_occluder(cell, slot_v, bp, bi, vec3<i32>(0, 0, -1))) { occ += 0.06; }
    return 1.0 - occ;
}

// The two big diagonal tuft quads of one leaf block (Better Leaves model:
// 2.3 x 2.0 blocks — the pack geometry scaled up ~15% per user taste — at
// 22.5 / -45 degrees plus the block hash 90-degree rotation, quad 1 slightly
// off-centre). Tested over [t_lo, t_hi] so both the owning cell and the
// surrounding fringe cells can render their part of the quads.
fn leaf_bl_quads(voxel_min: vec3<f32>, origin: vec3<f32>, dir: vec3<f32>, t_lo: f32, t_hi: f32, mat: u32) -> SubHit {
    var out: SubHit;
    out.hit = false;
    out.color_tint = vec3<f32>(1.0);
    let tuft = leaf_tuft_index(mat);
    let ext = leaf_quad_ext(tuft);
    let vh = hash3f(voxel_min);
    let vox_shade = 0.90 + fract(vh * 32.0) * 0.20;
    let species = leaf_species_tint(mat, vh);
    let yrot = floor(vh * 4.0) * 1.5707963;
    let phase = voxel_min.x * 0.31 + voxel_min.z * 0.41 + vh * 6.28;
    let wind = wind_offset(voxel_min, phase, 0.11);

    var best_t: f32 = 1e30;
    for (var q: i32 = 0; q < 2; q = q + 1) {
        let ang = select(0.3926991, -0.7853982, q == 1) + yrot;
        let ca = cos(ang);
        let sa = sin(ang);
        var c = voxel_min + vec3<f32>(0.5, 0.5, 0.5);
        if (q == 1) {
            c = voxel_min + vec3<f32>(0.0625 + 0.531 * ca, 0.5, 0.1875 - 0.531 * sa);
        }
        let n = vec3<f32>(sa, 0.0, ca);
        let denom = dot(dir, n);
        if (abs(denom) < 1e-4) { continue; }
        let t = dot(c - origin, n) / denom;
        if (t <= t_lo || t >= min(t_hi, best_t)) { continue; }
        let lp = origin + dir * t - c;
        let lv = lp.y;
        if (abs(lv) > ext.y) { continue; }
        var lu = lp.x * ca - lp.z * sa;
        if (abs(lu) > ext.x) { continue; }
        // Gentle waving-mod shear, stronger toward the tuft top.
        lu = lu - (wind.x * ca - wind.y * sa) * (lv / (2.0 * ext.y) + 0.5);
        let tx = u32(clamp((lu / ext.x * 0.5 + 0.5) * 32.0, 0.0, 31.0));
        let ty = u32(clamp(2.0 + (lv / (2.0 * ext.y) + 0.5) * 28.0, 0.0, 31.0));
        let val = tuft_texel(tuft, tx, ty);
        if (val == 0u) { continue; }
        best_t = t;
        out.hit = true;
        out.t_hit = t;
        out.normal = select(n, -n, denom > 0.0);
        out.color_tint = tuft_tone(val, vox_shade) * species;
    }
    return out;
}

// Horizontal canopy cap for the fringe cell directly above a leaf block:
// the block's full tuft laid flat at cap_y, rotated by its 90-degree hash
// variant. Deliberately lean - kept OUT of leaf_bl_quads because inlining
// it there bloats the hottest leaf function's registers (measured +0.14 ms
// on the foliage scenario even with the branch disabled). Trig-free (the
// four rotations are exact selects) and without geometric wind shear (the
// 0.06-block amplitude is sub-texel on a flat cap, and shade()'s sway still
// animates the normal), so a canopy-top traversal pays a hash, one plane
// test and one texel fetch.
fn leaf_cap_hit(nb_min: vec3<f32>, origin: vec3<f32>, dir: vec3<f32>, cap_y: f32, t_lo: f32, t_hi: f32, mat: u32) -> SubHit {
    var out: SubHit;
    out.hit = false;
    out.color_tint = vec3<f32>(1.0);
    if (abs(dir.y) < 1e-4) { return out; }
    let t = (cap_y - origin.y) / dir.y;
    if (t <= t_lo || t >= t_hi) { return out; }
    let vh = hash3f(nb_min);
    let k = i32(floor(vh * 4.0));
    let ca = select(select(1.0, -1.0, k == 2), 0.0, (k & 1) == 1);
    let sa = select(select(0.0, 1.0, k == 1), -1.0, k == 3);
    let lp = origin + dir * t - (nb_min + vec3<f32>(0.5, 0.0, 0.5));
    let u = lp.x * ca - lp.z * sa;
    let v = lp.x * sa + lp.z * ca;
    if (abs(u) > 1.15 || abs(v) > 1.15) { return out; }
    let val = tuft_texel(leaf_tuft_index(mat),
                         u32(clamp((u / 1.15 * 0.5 + 0.5) * 32.0, 0.0, 31.0)),
                         u32(clamp((v / 1.15 * 0.5 + 0.5) * 32.0, 0.0, 31.0)));
    if (val == 0u) { return out; }
    out.hit = true;
    out.t_hit = t;
    out.normal = vec3<f32>(0.0, select(1.0, -1.0, dir.y > 0.0), 0.0);
    out.color_tint = tuft_tone(val, 0.90 + fract(vh * 32.0) * 0.20) * leaf_species_tint(mat, vh);
    return out;
}

// Corner of the leaf-cloud tier: within this distance every fringe cell
// scatters individual leaf-silhouette cards through its VOLUME; beyond it
// the cheap horizontal cap tuft covers canopy tops instead (two-tier LOD,
// same pattern as the water corner/centre tiers).
const LEAF_CLOUD_T: f32 = 8.0 * VOXELS_PER_METRE;

// Which leaf silhouette a species' cards carry.
fn leaf_card_sprite(mat: u32) -> u32 {
    if (mat == MAT_LEAVES_BIRCH) { return SPR_LEAF_BIRCH; }
    if (mat == MAT_LEAVES_PINE)  { return SPR_LEAF_NEEDLE; }
    return SPR_LEAF_OAK;
}

// The volumetric leaf cloud: K leaf-silhouette cards (0.44 x 0.60 blocks)
// hash-scattered through the fringe cell's volume, normals biased outward
// from the canopy with wide per-leaf scatter. Varied depth gives parallax,
// per-leaf tone spread gives separation, sky shows through the gaps at the
// crown edge - individual leaves instead of a flat textured shell. Zero
// trig in the loop: orientation variety comes entirely from the 3D normal
// scatter (in-plane rotation would cost 2 trig per card per ray and adds
// little once normals are scattered).
fn leaf_cloud_hit(cell_min: vec3<f32>, origin: vec3<f32>, dir: vec3<f32>, t_lo: f32, t_hi: f32, mat: u32, outward: vec3<f32>, block_vh: f32) -> SubHit {
    var out: SubHit;
    out.hit = false;
    out.color_tint = vec3<f32>(1.0);
    let sprite = leaf_card_sprite(mat);
    let base_h = hash3f(cell_min + vec3<f32>(0.11, 0.53, 0.29));
    // Colour comes from the ADJACENT LEAF BLOCK's voxel hash (block_vh) -
    // the same hash its cube faces and cap tufts shade with - so cards sit
    // on a matching block: dark-green leaves on dark-green blocks, gold on
    // gold in an autumn mottle. base_h (the fringe-cell hash) drives only
    // GEOMETRY (placement, orientation, wind phase).
    let species = leaf_species_tint(mat, block_vh);
    let block_shade = 0.90 + fract(block_vh * 32.0) * 0.20;
    let wind = wind_offset(cell_min, base_h * 6.28, 0.15);
    var best_t: f32 = 1e30;
    for (var k: i32 = 0; k < 12; k = k + 1) {
        // Per-leaf channels from one base hash via a golden-ratio lattice.
        let hk = fract(base_h * 71.7 + f32(k) * 0.6180339);
        let h1 = fract(hk * 13.91);
        let h2 = fract(hk * 41.23);
        let h3 = fract(hk * 97.51);
        // Cluster the cards around the SHARED LEAF FACE (its centre is half
        // a cell inward along the outward axis; for a notch's diagonal
        // outward that point is the notch edge): scatter around it and pull
        // slightly inward, so no card hangs laterally off the canopy corner.
        let face_c = cell_min + vec3<f32>(0.5) - outward * 0.5;
        let c = face_c + (vec3<f32>(h1, h2, h3) - vec3<f32>(0.5)) * 0.85 - outward * 0.10;
        // Wide per-leaf size spread: small sprigs through big fans read as
        // a real crown; the horizontal overhang ring renders whatever pokes
        // past the cell, so large cards no longer clip.
        let scale = 0.70 + h3 * 0.85;
        let n = normalize(outward + (vec3<f32>(h1, h2, h3) - vec3<f32>(0.5)) * 1.4);
        let denom = dot(dir, n);
        if (abs(denom) < 1e-4) { continue; }
        let t = dot(c - origin, n) / denom;
        if (t <= t_lo || t >= min(t_hi, best_t)) { continue; }
        let lp = origin + dir * t - c;
        var up_ref = vec3<f32>(0.0, 1.0, 0.0);
        if (abs(n.y) > 0.9) { up_ref = vec3<f32>(1.0, 0.0, 0.0); }
        let u_base = normalize(cross(up_ref, n));
        let v_base = cross(n, u_base);
        // Per-leaf in-plane orientation from a trig-free 8-angle table -
        // without it every leaf tip points up. (cos, sin) of k*45deg.
        let h4 = fract(hk * 57.77);
        var oc = 1.0;
        var os = 0.0;
        if (h4 >= 0.125 && h4 < 0.250) { oc = 0.7071; os = 0.7071; }
        else if (h4 >= 0.250 && h4 < 0.375) { oc = 0.0; os = 1.0; }
        else if (h4 >= 0.375 && h4 < 0.500) { oc = -0.7071; os = 0.7071; }
        else if (h4 >= 0.500 && h4 < 0.625) { oc = -1.0; os = 0.0; }
        else if (h4 >= 0.625 && h4 < 0.750) { oc = -0.7071; os = -0.7071; }
        else if (h4 >= 0.750 && h4 < 0.875) { oc = 0.0; os = -1.0; }
        else if (h4 >= 0.875) { oc = 0.7071; os = -0.7071; }
        let u_ax = u_base * oc + v_base * os;
        let v_ax = v_base * oc - u_base * os;
        let hw = 0.22 * scale;
        let hh = 0.30 * scale;
        let lv = dot(lp, v_ax);
        if (abs(lv) > hh) { continue; }
        var lu = dot(lp, u_ax);
        // Gust shear, stronger toward the leaf tip.
        lu = lu - (wind.x * u_ax.x + wind.y * u_ax.z) * (lv + 0.5);
        if (abs(lu) > hw) { continue; }
        let tx = u32(clamp((lu / hw * 0.5 + 0.5) * 16.0, 0.0, 15.0));
        let ty = u32(clamp((lv / hh * 0.5 + 0.5) * 16.0, 0.0, 15.0));
        let val = sprite_texel(sprite, tx, ty);
        if (val == 0u) { continue; }
        var tone = 0.95;
        if (val == 2u) { tone = 0.62; }
        if (val == 3u) { tone = 1.30; }
        // Narrow per-leaf spread AROUND the block's own shade: individuals
        // still separate (sprite tone tiers + this spread), but the
        // ensemble average equals the block face's brightness, so cards
        // never read brighter or yellower than the canopy they sit on
        // (the old 0.78..1.25 spread + 18% hue wobble did exactly that).
        let leaf_shade = 0.88 + hk * 0.24;
        let hue = vec3<f32>(1.0 + (h1 - 0.5) * 0.08, 1.0, 1.0 - (h1 - 0.5) * 0.08);
        best_t = t;
        out.hit = true;
        out.t_hit = t;
        out.normal = select(n, -n, denom > 0.0);
        out.color_tint = vec3<f32>(tone * block_shade * leaf_shade) * species * hue;
    }
    return out;
}

struct CloudCtx {
    mat: u32,
    vh: f32,
    outward: vec3<f32>,
}

// EXACTLY mirrors the accumulation inside leaf_fringe_hit's neighbour loop
// (first leaf neighbour in the same 0..6 order = mat + block hash; outward
// = accumulated -dirs). The outer overhang ring reconstructs the INNER
// cell's card set with this - keep the two in lockstep or overhanging
// cards pop at the ring boundary. Separate on purpose: folding this into
// the main loop would add 6 hierarchical probes to EVERY fringe cell,
// while only the sparse outer ring needs the reconstruction.
fn leaf_cloud_context(voxel: vec3<i32>) -> CloudCtx {
    var ctx: CloudCtx;
    ctx.mat = 0u;
    ctx.vh = 0.5;
    ctx.outward = vec3<f32>(0.0);
    for (var i: i32 = 0; i < 6; i = i + 1) {
        var off = vec3<i32>(0);
        if (i == 0) { off.x = 1; } else if (i == 1) { off.x = -1; }
        else if (i == 2) { off.y = 1; } else if (i == 3) { off.y = -1; }
        else if (i == 4) { off.z = 1; } else { off.z = -1; }
        let nb = voxel + off;
        let nb_mat = voxel_material_at(nb);
        if (!is_leaf_block_mat(nb_mat)) { continue; }
        if (ctx.mat == 0u) {
            ctx.mat = nb_mat;
            ctx.vh = hash3f(vec3<f32>(f32(nb.x), f32(nb.y), f32(nb.z)));
        }
        ctx.outward = ctx.outward - vec3<f32>(f32(off.x), f32(off.y), f32(off.z));
    }
    ctx.outward = normalize(ctx.outward + vec3<f32>(0.0, 1e-4, 0.0));
    return ctx;
}

// A fringe cell renders the parts of its neighbouring leaf blocks tuft
// quads that protrude into it — this is what makes tufts visible from the
// SIDE (rays grazing past a canopy never enter the leaf cells themselves,
// only the invisible fringe shell worldgen paints around them).
fn leaf_fringe_hit(voxel: vec3<i32>, origin: vec3<f32>, dir: vec3<f32>) -> SubHit {
    var out: SubHit;
    out.hit = false;
    out.color_tint = vec3<f32>(1.0);
    let voxel_min = vec3<f32>(f32(voxel.x), f32(voxel.y), f32(voxel.z));
    let inv_dir = vec3<f32>(safe_inv(dir.x), safe_inv(dir.y), safe_inv(dir.z));
    let t0 = (voxel_min - origin) * inv_dir;
    let t1 = (voxel_min + vec3<f32>(1.0) - origin) * inv_dir;
    let tmin3 = min(t0, t1);
    let tmax3 = max(t0, t1);
    let t_enter = max(max(tmin3.x, tmin3.y), max(tmin3.z, 0.0));
    let t_exit = min(min(tmax3.x, tmax3.y), tmax3.z);
    if (t_enter >= t_exit) { return out; }

    var best_t: f32 = 1e30;
    var cloud_mat: u32 = 0u;
    var cloud_vh: f32 = 0.5;
    var outward = vec3<f32>(0.0, 1.0, 0.0);
    for (var i: i32 = 0; i < 6; i = i + 1) {
        var off = vec3<i32>(0);
        if (i == 0) { off.x = 1; } else if (i == 1) { off.x = -1; }
        else if (i == 2) { off.y = 1; } else if (i == 3) { off.y = -1; }
        else if (i == 4) { off.z = 1; } else { off.z = -1; }
        let nb = voxel + off;
        let nb_mat = voxel_material_at(nb);
        if (!is_leaf_block_mat(nb_mat)) { continue; }
        if (cloud_mat == 0u) {
            cloud_mat = nb_mat;
            // The first contributing leaf block's voxel hash: the cloud
            // cards colour-match THIS block (species mottle + shade).
            cloud_vh = hash3f(vec3<f32>(f32(nb.x), f32(nb.y), f32(nb.z)));
            outward = vec3<f32>(0.0);
        }
        // Accumulate ALL adjacent leaf directions: for a flat canopy face
        // the sum is the face normal, and in a step notch (leaf neighbours
        // in two directions) it is the DIAGONAL bisector - cards anchor
        // into the notch and bridge the blocky step.
        outward = outward - vec3<f32>(f32(off.x), f32(off.y), f32(off.z));
        let nb_min = vec3<f32>(f32(nb.x), f32(nb.y), f32(nb.z));
        var qh = leaf_bl_quads(nb_min, origin, dir, max(t_enter - 0.05, 0.0), t_exit + 0.05, nb_mat);
        if (qh.hit && qh.t_hit < best_t) {
            best_t = qh.t_hit;
            // shade() tints by palette[MAT_LEAF_FRINGE]; fold in the ratio to
            // the neighbour real leaf colour (autumn/birch/pine correctness).
            qh.color_tint = qh.color_tint * palette[nb_mat].rgb
                / max(palette[MAT_LEAF_FRINGE].rgb, vec3<f32>(1e-3));
            out = qh;
        }
        if (i == 3 && t_enter >= LEAF_CLOUD_T) {
            // Far tier only: beyond the leaf cloud, canopy tops fall back to
            // the neighbour's horizontal cap tuft at 0.30 into this cell so
            // distant tops stay fluffy instead of flat tiled cube faces.
            var ch = leaf_cap_hit(nb_min, origin, dir, voxel_min.y + 0.30,
                                  max(t_enter - 0.05, 0.0), min(t_exit + 0.05, best_t), nb_mat);
            if (ch.hit) {
                best_t = ch.t_hit;
                ch.color_tint = ch.color_tint * palette[nb_mat].rgb
                    / max(palette[MAT_LEAF_FRINGE].rgb, vec3<f32>(1e-3));
                out = ch;
            }
        }
    }
    // The volumetric leaf cloud fills this fringe cell with individual
    // leaves (all six shell directions - canopy SIDES included, which is
    // where eye-level views look).
    // Opposing leaf pairs cancel the sum: bias up so normalize is safe.
    outward = normalize(outward + vec3<f32>(0.0, 1e-4, 0.0));
    if (cloud_mat != 0u && t_enter < LEAF_CLOUD_T) {
        var lh = leaf_cloud_hit(voxel_min, origin, dir,
                                max(t_enter - 0.05, 0.0), min(t_exit + 0.05, best_t), cloud_mat, outward, cloud_vh);
        if (lh.hit) {
            lh.color_tint = lh.color_tint * palette[cloud_mat].rgb
                / max(palette[MAT_LEAF_FRINGE].rgb, vec3<f32>(1e-3));
            out = lh;
        }
    } else if (cloud_mat == 0u && t_enter < LEAF_CLOUD_T) {
        // Outer overhang ring (the horizontally-widened fringe shell): no
        // leaf neighbour at distance 1, so look laterally at distance 2 and
        // render the INNER fringe cell's IDENTICAL card set restricted to
        // this cell's ray window - cards poking past the inner cell
        // continue here seamlessly instead of clipping at the shell plane.
        for (var i: i32 = 0; i < 4; i = i + 1) {
            var off = vec3<i32>(0);
            if (i == 0) { off.x = 2; } else if (i == 1) { off.x = -2; }
            else if (i == 2) { off.z = 2; } else { off.z = -2; }
            if (!is_leaf_block_mat(voxel_material_at(voxel + off))) { continue; }
            let inner = voxel + off / 2;
            let ctx = leaf_cloud_context(inner);
            if (ctx.mat == 0u) { continue; }
            let inner_min = vec3<f32>(f32(inner.x), f32(inner.y), f32(inner.z));
            var lh = leaf_cloud_hit(inner_min, origin, dir,
                                    max(t_enter - 0.05, 0.0), min(t_exit + 0.05, best_t), ctx.mat, ctx.outward, ctx.vh);
            if (lh.hit) {
                lh.color_tint = lh.color_tint * palette[ctx.mat].rgb
                    / max(palette[MAT_LEAF_FRINGE].rgb, vec3<f32>(1e-3));
                out = lh;
                best_t = lh.t_hit;
            }
        }
    }
    return out;
}

fn leaf_bl_hit(voxel: vec3<i32>, origin: vec3<f32>, dir: vec3<f32>, mat: u32) -> SubHit {
    var out: SubHit;
    out.hit = false;
    out.color_tint = vec3<f32>(1.0);
    let voxel_min = vec3<f32>(f32(voxel.x), f32(voxel.y), f32(voxel.z));
    let inv_dir = vec3<f32>(safe_inv(dir.x), safe_inv(dir.y), safe_inv(dir.z));
    let t0 = (voxel_min - origin) * inv_dir;
    let t1 = (voxel_min + vec3<f32>(1.0) - origin) * inv_dir;
    let tmin = min(t0, t1);
    let tmax = max(t0, t1);
    let t_enter = max(max(tmin.x, tmin.y), max(tmin.z, 0.0));
    let t_exit = min(min(tmax.x, tmax.y), tmax.z);
    if (t_enter >= t_exit) { return out; }

    let vh = hash3f(voxel_min);
    let vox_shade = 0.90 + fract(vh * 32.0) * 0.20;
    let tuft = leaf_tuft_index(mat);
    let species = leaf_species_tint(mat, vh);

    var best_t: f32 = 1e30;
    var best_n = vec3<f32>(0.0, 1.0, 0.0);
    var tint = vec3<f32>(1.0);

    // ---- the two big diagonal tuft quads (shared tester) ----
    // Overhang beyond this cell is rendered by the surrounding fringe cells,
    // so the own-cell test stays tight.
    let qh = leaf_bl_quads(voxel_min, origin, dir, max(t_enter - 0.05, 0.0), t_exit + 0.05, mat);
    if (qh.hit) {
        best_t = qh.t_hit;
        best_n = qh.normal;
        tint = qh.color_tint;
    }

    // ---- the cutout cube faces (centre 16x16 of the tuft) ----
    var entry_axis: i32 = 0;
    if (tmin.x >= tmin.y && tmin.x >= tmin.z) { entry_axis = 0; }
    else if (tmin.y >= tmin.z) { entry_axis = 1; }
    else { entry_axis = 2; }
    if (t_enter < best_t) {
        let lp_e = origin + dir * t_enter - voxel_min;
        var uv_e: vec2<f32>;
        if (entry_axis == 0) { uv_e = vec2<f32>(lp_e.z, lp_e.y); }
        else if (entry_axis == 1) { uv_e = vec2<f32>(lp_e.x, lp_e.z); }
        else { uv_e = vec2<f32>(lp_e.x, lp_e.y); }
        let val = tuft_texel(tuft, 8u + u32(clamp(uv_e.x * 16.0, 0.0, 15.0)),
                             8u + u32(clamp(uv_e.y * 16.0, 0.0, 15.0)));
        if (val != 0u) {
            var n = vec3<f32>(0.0);
            if (entry_axis == 0) { n.x = select(1.0, -1.0, dir.x > 0.0); }
            else if (entry_axis == 1) { n.y = select(1.0, -1.0, dir.y > 0.0); }
            else { n.z = select(1.0, -1.0, dir.z > 0.0); }
            best_t = t_enter;
            best_n = n;
            tint = tuft_tone(val, vox_shade) * species;
        }
    }
    if (best_t >= 1e29) {
        // Entry face was a hole and no quad caught the ray: try the exit
        // face (interior faces read a shade darker).
        var exit_axis: i32 = 0;
        if (tmax.x <= tmax.y && tmax.x <= tmax.z) { exit_axis = 0; }
        else if (tmax.y <= tmax.z) { exit_axis = 1; }
        else { exit_axis = 2; }
        let lp_x = origin + dir * t_exit - voxel_min;
        var uv_x: vec2<f32>;
        if (exit_axis == 0) { uv_x = vec2<f32>(lp_x.z, lp_x.y); }
        else if (exit_axis == 1) { uv_x = vec2<f32>(lp_x.x, lp_x.z); }
        else { uv_x = vec2<f32>(lp_x.x, lp_x.y); }
        let val = tuft_texel(tuft, 8u + u32(clamp(uv_x.x * 16.0, 0.0, 15.0)),
                             8u + u32(clamp(uv_x.y * 16.0, 0.0, 15.0)));
        if (val != 0u) {
            var n = vec3<f32>(0.0);
            if (exit_axis == 0) { n.x = select(-1.0, 1.0, dir.x > 0.0); }
            else if (exit_axis == 1) { n.y = select(-1.0, 1.0, dir.y > 0.0); }
            else { n.z = select(-1.0, 1.0, dir.z > 0.0); }
            best_t = t_exit;
            best_n = n;
            tint = tuft_tone(val, vox_shade * 0.78) * species;
        }
    }

    if (best_t < 1e29) {
        out.hit = true;
        out.t_hit = best_t;
        out.normal = best_n;
        out.color_tint = tint;
    }
    return out;
}

// ---------- GRASS + FLOWERS: crossed quads with authored sprites ----------
// The classic rasterizer trick, ported to the raymarcher: every decoration
// voxel is two diagonal planes ("X") carrying a hand-drawn 16x16 sprite.
// Two plane intersections + one texel fetch replaces the old 22-blade
// procedural bundle (22 AABBs x 5 samples in the hottest DDA loop).
// The whole quad shears sideways with the wind, weighted by height, exactly
// like a vertex-shader wave on a crossed billboard.
fn cross_sprite_tint(mat: u32, sprite: u32, val: u32, v: f32, vh: f32) -> vec3<f32> {
    if (mat == MAT_TALL_GRASS) {
        // Macro-calm sprite shading: bright shared base fusing with the
        // ground, ONE quantized value step to a lighter top band, darker
        // secondary texels kept mild, and only a whisper of per-cell
        // variance (per-cell brightness lotteries read as noise).
        let stp = smoothstep(0.48, 0.56, v);
        let b = (0.94 + 0.24 * stp) * select(1.0, 0.86, val == 2u);
        let ground = mix(vec3<f32>(1.0),
                         palette[MAT_GRASS].rgb / max(palette[MAT_TALL_GRASS].rgb, vec3<f32>(1e-3)),
                         0.6);
        return vec3<f32>(b) * (0.96 + vh * 0.08) * ground;
    }
    if (mat == MAT_TALL_GRASS_DRY) {
        // Pale straw: brightness ramp along the stalk, small per-clump spread.
        return vec3<f32>(0.55 + 0.70 * v) * (0.90 + vh * 0.20);
    }
    // Flowers: absolute colours (palette MAT_FLOWER is white). Poppy and
    // daisy values are the previous ratio constants multiplied through the
    // old palette entry, so their look is unchanged.
    return flower_color(sprite, val);
}

// Absolute flower colours per species and tone. '#' = petal body, '*' =
// accent (centre or bright fringe), 'o' = stem/leaf green - shared by the
// stem and any collar ring around a head.
fn flower_color(sprite: u32, val: u32) -> vec3<f32> {
    if (val == 2u) { return vec3<f32>(0.200, 0.450, 0.130); } // stem/leaf
    if (sprite == SPR_POPPY) {
        if (val == 3u) { return vec3<f32>(0.120, 0.090, 0.050); } // dark centre
        return vec3<f32>(0.900, 0.160, 0.130);                    // red petals
    }
    if (sprite == SPR_DAISY) {
        if (val == 3u) { return vec3<f32>(1.000, 0.820, 0.250); } // yellow centre
        return vec3<f32>(0.930, 0.930, 0.880);                    // white petals
    }
    if (sprite == SPR_TULIP) {
        if (val == 3u) { return vec3<f32>(0.950, 0.450, 0.280); } // lit rim
        return vec3<f32>(0.860, 0.300, 0.240);                    // rose-red cup
    }
    if (sprite == SPR_CORNFLOWER) {
        if (val == 3u) { return vec3<f32>(0.470, 0.580, 0.960); } // bright fringe
        return vec3<f32>(0.280, 0.380, 0.880);                    // cornflower blue
    }
    // Dandelion.
    if (val == 3u) { return vec3<f32>(1.000, 0.900, 0.320); }     // bright core
    return vec3<f32>(0.950, 0.790, 0.240);                        // yellow puff
}

fn sprite_cross_hit(voxel: vec3<i32>, origin: vec3<f32>, dir: vec3<f32>, mat: u32) -> SubHit {
    var out: SubHit;
    out.hit = false;
    out.color_tint = vec3<f32>(1.0);
    let voxel_min = vec3<f32>(f32(voxel.x), f32(voxel.y), f32(voxel.z));
    let vh = hash3f(voxel_min);
    if (vh > 0.92) { return out; } // sparse gaps, same density as before

    var sprite = SPR_TALL_GRASS_A;
    if (mat == MAT_TALL_GRASS) {
        // Blade-shape variants on an independent hash channel.
        let r = fract(vh * 8.0);
        if (r >= 0.4) { sprite = SPR_TALL_GRASS_B; }
        if (r >= 0.75) { sprite = SPR_TALL_GRASS_C; }
    } else if (mat == MAT_TALL_GRASS_DRY) {
        sprite = SPR_DRY_TUFT;
    } else if (mat == MAT_FLOWER) {
        // Five species on an independent hash channel; the flower sprites
        // are contiguous in the atlas so the pick is an index offset.
        sprite = SPR_POPPY + min(u32(fract(vh * 128.0) * 5.0), 4u);
    }
    // Per-clump height (grass and straw only): flowers keep hs = 1.0, their
    // stems must reach the ground plane at full sprite height. Grass rides
    // the same rolling height field as the near-tier blades.
    var hs = 1.0;
    if (mat != MAT_FLOWER) {
        hs = (0.70 + fract(vh * 4.0) * 0.30)
            * flora_field(voxel_min.xz + vec2<f32>(0.5));
    }

    let voxel_center = voxel_min + vec3<f32>(0.5);
    let phase = voxel_min.x * 0.40 + voxel_min.z * 0.55 + vh * 6.28;
    let wind = wind_offset(voxel_min, phase, 0.32);
    let mirror_u = fract(vh * 16.0) > 0.5;

    var best_t: f32 = 1e30;
    var best_n = vec3<f32>(0.0, 1.0, 0.0);
    var tint = vec3<f32>(1.0);

    for (var i: i32 = 0; i < 2; i = i + 1) {
        var pn: vec3<f32>;
        var pt: vec3<f32>;
        if (i == 0) {
            pn = vec3<f32>(0.7071, 0.0, 0.7071);
            pt = vec3<f32>(0.7071, 0.0, -0.7071);
        } else {
            pn = vec3<f32>(0.7071, 0.0, -0.7071);
            pt = vec3<f32>(0.7071, 0.0, 0.7071);
        }
        let denom = dot(dir, pn);
        if (abs(denom) < 0.0001) { continue; }
        let t = dot(voxel_center - origin, pn) / denom;
        if (t < 0.0 || t >= best_t) { continue; }
        let p_hit = origin + dir * t;
        let local = p_hit - voxel_min;
        if (local.x < 0.0 || local.x > 1.0
         || local.y < 0.0 || local.y > 1.0
         || local.z < 0.0 || local.z > 1.0) { continue; }

        let v = local.y;
        if (v > hs) { continue; } // above this clump's height: transparent
        let vn = v / hs;          // normalized height along the clump
        // Shear the sampling space by the wind in WORLD xz, weighted by
        // height — identically for both planes, so the two quads keep
        // intersecting in one vertical line. (Shearing each plane along its
        // own tangent split the X into two separate stems.)
        let sx = local.x - wind.x * v;
        let sz = local.z - wind.y * v;
        let s_w = (sx - 0.5) * pt.x + (sz - 0.5) * pt.z;

        let u = clamp((s_w + 0.70711) / 1.41421, 0.0, 0.99999);
        var tx = u32(u * 16.0);
        if (mirror_u) { tx = 15u - tx; }
        let ty = u32(clamp(vn * 16.0, 0.0, 15.0));
        let val = sprite_texel(sprite, tx, ty);
        if (val == 0u) { continue; }

        best_t = t;
        best_n = select(pn, -pn, denom > 0.0);
        tint = cross_sprite_tint(mat, sprite, val, vn, vh);
    }

    if (best_t < 1e30) {
        out.hit = true;
        out.t_hit = best_t;
        out.normal = best_n;
        out.color_tint = tint;
    }
    return out;
}

// Near-tier volumetric grass (docs/FLORA_PLAN.md stage 2): K single-blade
// CARDS rooted in the cell, gathered into 2-3 sub-clumps, yawed around the
// 8-angle fan with jitter, leaning outward from their sub-clump centre and
// shearing with the wind (tip-weighted, via wind_offset - which routes
// through wind_time(), so shadow rays sample the frozen phase exactly like
// the cross quads). Same primitives as sprite_cross_hit: plane test, sprite
// texel alpha, cross_sprite_tint ramp.
const FLORA_NEAR_T: f32 = 7.0 * VOXELS_PER_METRE;
const FLORA_BLADES: i32 = 18;

// Rolling grass height: a smooth ~18-voxel field so blade and tuft heights
// cohere regionally (tall waves and short hollows, the "higher and lower"
// carpet look) instead of dicing at random per cell. Both tiers scale by
// the same field, so the near/far handoff keeps silhouette heights.
fn flora_field(xz: vec2<f32>) -> f32 {
    return 0.60 + 0.40 * vnoise3(vec3<f32>(xz.x * 0.055, 3.7, xz.y * 0.055));
}

fn flora_clump_hit(voxel: vec3<i32>, origin: vec3<f32>, dir: vec3<f32>, mat: u32) -> SubHit {
    var out: SubHit;
    out.hit = false;
    out.color_tint = vec3<f32>(1.0);
    let voxel_min = vec3<f32>(f32(voxel.x), f32(voxel.y), f32(voxel.z));
    let vh = hash3f(voxel_min);
    if (vh > 0.92) { return out; } // same sparse gaps as the cross tier
    let phase = voxel_min.x * 0.40 + voxel_min.z * 0.55 + vh * 6.28;
    let wind = wind_offset(voxel_min, phase, 0.32);
    // One field sample per clump: the intra-cell field delta (~2% height)
    // is invisible, and 18 vnoise3 calls per cell are not.
    let field = flora_field(voxel_min.xz + vec2<f32>(0.5));

    var best_t: f32 = 1e30;
    var best_n = vec3<f32>(0.0, 1.0, 0.0);
    var tint = vec3<f32>(1.0);

    for (var i: i32 = 0; i < FLORA_BLADES; i = i + 1) {
        let bh = hash3f(voxel_min + vec3<f32>(f32(i) * 7.13 + 0.31, 3.7, f32(i) * 2.9 + 0.17));
        // Sub-clump centre (3 per cell) spread across the WHOLE footprint,
        // blade root jittered around it; the tussock owns its full voxel.
        let sc = f32(i % 3);
        let cx = 0.15 + 0.70 * fract(vh * 23.0 + sc * 0.37);
        let cz = 0.15 + 0.70 * fract(vh * 57.0 + sc * 0.71);
        let root2 = clamp(
            vec2<f32>(cx, cz) + (vec2<f32>(fract(bh * 13.0), fract(bh * 29.0)) - vec2<f32>(0.5)) * 0.26,
            vec2<f32>(0.03), vec2<f32>(0.97));
        // Room to the nearest cell wall: wall-adjacent blades get thin and
        // stand straight so nothing clips mid-blade at the (invisible) wall;
        // interior blades stay fat and spray outward over their heads.
        let edge_room = min(min(root2.x, 1.0 - root2.x), min(root2.y, 1.0 - root2.y));
        // Yaw: the 8-angle fan plus jitter; card tangent/normal from it.
        let ang = (f32(i & 7) / 8.0) * 6.2832 + fract(bh * 5.0) * 0.7;
        let ca = cos(ang);
        let sa = sin(ang);
        let pt2 = vec2<f32>(ca, sa);
        let pn = vec3<f32>(-sa, 0.0, ca);
        let h = (0.60 + fract(bh * 3.0) * 0.40) * field;
        let denom = dot(dir, pn);
        if (abs(denom) < 1e-4) { continue; }
        let rootw = voxel_min + vec3<f32>(root2.x, 0.0, root2.y);
        let t = dot(rootw - origin, pn) / denom;
        if (t < 0.0 || t >= best_t) { continue; }
        let pw = origin + dir * t;
        // Stay inside the cell like the cross quads do - hits beyond the
        // cell walls would break the DDA's front-to-back ordering. All
        // three axes: the card plane extends above the cell too.
        let cl = pw - voxel_min;
        if (cl.x < 0.0 || cl.x > 1.0 || cl.y > 1.0
         || cl.z < 0.0 || cl.z > 1.0) { continue; }
        let p = pw - rootw;
        if (p.y < 0.0 || p.y > h) { continue; }
        let vn = p.y / h;
        // Outward lean from the sub-clump centre plus tip-weighted wind sway
        // (v^2: roots stay planted, tips ride the gusts). The combined
        // displacement is clamped to the cell so gust-blown tips crowd
        // against the wall instead of being sliced off by it.
        let lean = (root2 - vec2<f32>(0.5)) * 0.5 * clamp(edge_room * 3.0, 0.0, 1.0);
        let sway = clamp(
            lean * p.y + wind * (p.y * p.y),
            vec2<f32>(0.03) - root2, vec2<f32>(0.97) - root2);
        let uoff = dot(p.xz - sway, pt2);
        let wq = clamp(edge_room, 0.06, 0.26); // card half-width
        if (abs(uoff) > wq) { continue; }
        let u = clamp(uoff / wq * 0.5 + 0.5, 0.0, 0.99999);
        var sprite = SPR_BLADE_A + (u32(fract(bh * 97.0) * 3.0) % 3u);
        if (mat == MAT_TALL_GRASS_DRY) {
            // Dry clumps: forked straw, one seed head per clump on blade 0.
            sprite = select(SPR_BLADE_DRY, SPR_SEED_HEAD, i == 0);
        }
        let val = sprite_texel(sprite, u32(u * 16.0) & 15u, u32(clamp(vn * 16.0, 0.0, 15.0)));
        if (val == 0u) { continue; }
        best_t = t;
        // Blend the card normal toward up so blades take the ground's
        // lighting instead of flipping dark at unlucky yaws (random-yaw
        // vertical cards otherwise scatter black blades through the clump).
        let face_n = select(pn, -pn, denom > 0.0);
        best_n = normalize(mix(face_n, vec3<f32>(0.0, 1.0, 0.0), 0.5));
        // Per-blade brightness spread: inner/outer blades separate visually,
        // which is what makes 18 cards read as a volume, not a flat fan.
        tint = cross_sprite_tint(mat, sprite, val, vn, vh)
            * (0.82 + 0.36 * fract(bh * 11.0));
    }

    if (best_t < 1e30) {
        out.hit = true;
        out.t_hit = best_t;
        out.normal = best_n;
        out.color_tint = tint;
    }
    return out;
}

// Continuous turf (MAT_TURF, the cell above every grass block): ANALYTIC
// tapered blade segments - real 3D grass rising out of the ground, not
// sprite cards. A 3x3 rooted lattice per cell, each blade a closest-
// approach ray/segment test with a radius tapering root->tip, bent by a
// per-blade lean and the shared wind (tip-weighted, frozen-phase safe via
// wind_offset). Misses fall through to the grass top below, whose combed
// sheen shading is the between-blades and beyond-TURF_T look.
const TURF_T: f32 = 12.0 * VOXELS_PER_METRE;

// One flat tapered ribbon piece: closest approach of the (unit) ray to the
// segment a->b, accepted when the perpendicular offset decomposes into
// less-than-halfwidth across the blade's wide axis and less-than-thickness
// through it. s01 receives the position along the segment.
struct RibbonHit {
    hit: bool,
    t: f32,
    s01: f32,
    lat: f32,       // lateral position across the width, -1..1
    perp: vec3<f32>, // perpendicular offset vector (ray point - axis point)
}

fn ribbon_seg_hit(
    origin: vec3<f32>, dir: vec3<f32>, a: vec3<f32>, b: vec3<f32>,
    wide3: vec3<f32>, hw0: f32, hw1: f32, t_cap: f32,
) -> RibbonHit {
    var r: RibbonHit;
    r.hit = false;
    let u = b - a;
    let w0 = origin - a;
    let bb = dot(dir, u);
    let c = dot(u, u);
    let d0 = dot(dir, w0);
    let e0 = dot(u, w0);
    let den = max(c - bb * bb, 1e-6);
    let s = clamp((e0 - bb * d0) / den, 0.0, 1.0);
    let t = s * bb - d0;
    if (t <= 0.0 || t >= t_cap) { return r; }
    let perp = w0 + dir * t - u * s;
    let hw = mix(hw0, hw1, s);
    let wid = dot(perp, wide3);
    if (abs(wid) > hw) { return r; }
    let thick = perp - wide3 * wid;
    if (dot(thick, thick) > 0.0009) { return r; } // ~0.03 thickness
    r.hit = true;
    r.t = t;
    r.s01 = s;
    r.lat = wid / hw;
    r.perp = perp;
    return r;
}

fn turf_blade_hit(voxel: vec3<i32>, origin: vec3<f32>, dir: vec3<f32>) -> SubHit {
    var out: SubHit;
    out.hit = false;
    out.color_tint = vec3<f32>(1.0);
    let voxel_min = vec3<f32>(f32(voxel.x), f32(voxel.y), f32(voxel.z));
    let vh = hash3f(voxel_min);
    let field = flora_field(voxel_min.xz + vec2<f32>(0.5));
    let phase = voxel_min.x * 0.40 + voxel_min.z * 0.55 + vh * 6.28;
    let wind = wind_offset(voxel_min, phase, 0.22);

    // ---- Clump (Worley cells ~2.5 voxels): blades share facing, height
    // and colour with their clump, jittered per blade - uniform randomness
    // reads as noise, clumps read as a field (the GoT distribution rule).
    // Sampled once per cell; clump borders quantise to voxels, invisible at
    // blade scale.
    let w = worley2(voxel_min.xz * 0.4);
    let clump_ang = fract(w.id * 7.13) * 6.2832;
    let clump_h = 0.70 + 0.55 * fract(w.id * 3.71);
    let clump_bright = 0.82 + 0.34 * fract(w.id * 5.23);
    let clump_hue = mix(vec3<f32>(1.0), vec3<f32>(1.22, 1.06, 0.58),
                        fract(w.id * 9.77) * 0.45);

    // Distance LOD: near = 16 curved 2-segment blades; mid = 9 single-
    // segment blades widened to keep coverage (fewer-but-wider, the GoT
    // density-compensation trick).
    let cd = length(voxel_min + vec3<f32>(0.5) - origin);
    let near = cd < 16.0;
    let n_blades = select(9, 16, near);
    let wide_mul = select(1.9, 1.0, near);

    let ground = mix(vec3<f32>(1.0),
                     palette[MAT_GRASS].rgb / max(palette[MAT_TURF].rgb, vec3<f32>(1e-3)),
                     0.6);

    var best_t: f32 = 1e30;
    var best_n = vec3<f32>(0.0, 1.0, 0.0);
    var tint = vec3<f32>(1.0);

    for (var i: i32 = 0; i < n_blades; i = i + 1) {
        let gx = f32(i % 4);
        let gz = f32(i / 4);
        let bh = hash3f(voxel_min + vec3<f32>(gx * 13.1 + 1.7, 5.3, gz * 17.9 + 0.9));
        let root2 = vec2<f32>(
            (gx + 0.10 + 0.80 * fract(bh * 7.0)) / 4.0,
            (gz + 0.10 + 0.80 * fract(bh * 11.0)) / 4.0);
        // Facing: the clump's direction plus per-blade spread.
        let fa = clump_ang + (fract(bh * 5.0) - 0.5) * 1.7;
        let fdir = vec2<f32>(cos(fa), sin(fa));
        let h = field * clump_h * (0.32 + 0.40 * fract(bh * 3.0));
        // The Bezier arc, approximated by two chained segments: the tip
        // falls over along the facing (curved blades ARE the grass look;
        // straight spikes are the plastic look). Wind adds curvature, so
        // gusts bend blades instead of tilting them rigidly.
        let curve = 0.45 + 0.55 * fract(bh * 13.0);
        let arc2 = fdir * curve * h + wind * h * 1.6;
        let arc_len = length(arc2);
        // Arc-over: the more the blade bends, the lower its tip sits.
        let droop = clamp(arc_len * 0.9, 0.0, 0.70);
        // Quadratic Bezier: P0 root, P1 above-root leaning half the arc,
        // P2 the fallen-over tip. Sampled at t = 0.45 / 0.75 / 1.0 into a
        // 3-segment polyline (near tier) - a 2-segment blade shows a paper
        // fold at its single elbow, three read as a smooth curve.
        let cp0 = vec3<f32>(root2.x, 0.0, root2.y);
        let cp1 = vec3<f32>(root2.x + arc2.x * 0.5, h * 0.85, root2.y + arc2.y * 0.5);
        let cp2 = vec3<f32>(root2.x + arc2.x, h * (1.0 - droop), root2.y + arc2.y);
        let q0 = voxel_min + cp0;
        // B(0.45), B(0.75), B(1.0) of the quadratic, xz crowded into the cell.
        var b1 = cp0 * 0.3025 + cp1 * 0.495 + cp2 * 0.2025;
        var b2 = cp0 * 0.0625 + cp1 * 0.375 + cp2 * 0.5625;
        let q1 = voxel_min + vec3<f32>(clamp(b1.x, 0.02, 0.98), b1.y, clamp(b1.z, 0.02, 0.98));
        let q2 = voxel_min + vec3<f32>(clamp(b2.x, 0.02, 0.98), b2.y, clamp(b2.z, 0.02, 0.98));
        let q3 = voxel_min + vec3<f32>(clamp(cp2.x, 0.02, 0.98), cp2.y, clamp(cp2.z, 0.02, 0.98));
        // Twist: the ribbon's wide axis rotates along the blade, so faces
        // present at varying angles instead of one uniform tape direction.
        let tw0 = (fract(bh * 31.0) - 0.5) * 0.7;
        let hw_r = 0.013 * wide_mul * (0.8 + 0.4 * fract(bh * 17.0));

        var rh: RibbonHit;
        rh.hit = false;
        var sblade = 0.0;
        var seg_a = q0;
        var seg_b = q1;
        var wide_hit = vec3<f32>(0.0);
        // Segment bounds in blade-param space and width taper per node.
        for (var k: i32 = 0; k < 3; k = k + 1) {
            if (!near && k > 1) { break; } // mid LOD: 2 segments
            var a = q0; var bq = q1; var s0 = 0.0; var s1 = 0.45;
            var w0 = hw_r; var w1 = hw_r * 0.70;
            if (k == 1) { a = q1; bq = q2; s0 = 0.45; s1 = 0.75; w0 = hw_r * 0.70; w1 = hw_r * 0.30; }
            if (k == 2) { a = q2; bq = q3; s0 = 0.75; s1 = 1.0;  w0 = hw_r * 0.30; w1 = 0.001; }
            let twa = tw0 + (s0 + s1) * 0.5 * 0.5;
            let wf = vec2<f32>(cos(fa + twa), sin(fa + twa));
            let wide3 = vec3<f32>(-wf.y, 0.0, wf.x);
            let r = ribbon_seg_hit(origin, dir, a, bq, wide3, w0, w1,
                                   select(best_t, rh.t, rh.hit));
            if (r.hit) {
                rh = r;
                sblade = mix(s0, s1, r.s01);
                seg_a = a;
                seg_b = bq;
                wide_hit = wide3;
            }
        }
        if (!rh.hit) { continue; }

        best_t = rh.t;
        // Rounded normal: flat-face normal bowed across the width (fakes a
        // curved blade cross-section - the small sharp speculars trick),
        // then blended toward up along the height so tips catch the sky.
        let tang = normalize(seg_b - seg_a);
        var nf = normalize(rh.perp - tang * dot(rh.perp, tang));
        nf = nf * select(1.0, -1.0, dot(nf, dir) > 0.0);
        var n = normalize(nf + wide_hit * rh.lat * 0.7);
        n = normalize(mix(n, vec3<f32>(0.0, 1.0, 0.0), 0.30 + 0.35 * sblade));
        best_n = n;
        // Colour: clump-coherent base (shared hue + brightness), root-dark
        // quadratic self-shadow ramp, small per-blade spread on top.
        tint = vec3<f32>(0.34 + 0.82 * sblade * sqrt(sblade))
            * clump_bright * (0.90 + 0.20 * fract(bh * 23.0))
            * ground * clump_hue;
    }

    if (best_t < 1e30) {
        out.hit = true;
        out.t_hit = best_t;
        out.normal = best_n;
        out.color_tint = tint;
    }
    return out;
}

// Micro-voxel tussock (the "voxel patches" style): each MAT_TALL_GRASS
// cell holds a baked 8x8x8 occupancy volume (three variants, atlas tail)
// ray-marched with a tiny DDA - a TRUE voxelized bush, chunky from every
// angle, no billboard planes. Wind is a linear shear of the marching
// space (the whole tuft leans with the gust and springs back). Tint
// follows the carpet rules: bright ground-coupled base, quantized height
// steps, top faces a breath lighter; sun/shadow/AO arrive through shade()
// like any other voxel surface.
fn micro_tuft_bit(variant: u32, x: i32, y: i32, z: i32) -> bool {
    let bit = u32(x + z * 16 + y * 256);
    let w = sprites[MICRO_TUFT_BASE_WORDS + variant * MICRO_TUFT_WORDS + (bit >> 5u)];
    return ((w >> (bit & 31u)) & 1u) != 0u;
}

// Shared micro-volume march for tussocks and bushes: DDA with CUTOUT
// faces - an occupied cell's face carries a hole pattern (grass: vertical
// slits, airier toward the crown; bush: round leaf holes), and a holed
// crossing lets the ray continue, exactly like the leaf-block cutouts.
struct TuftHit {
    hit: bool,
    t: f32,
    n: vec3<f32>,
    vn: f32,     // micro height 0..1 of the hit cell
    tone: f32,   // per-micro-cell quantized tone
}

fn tuft_volume_march(
    voxel_min: vec3<f32>, origin: vec3<f32>, dir: vec3<f32>,
    variant: u32, lean: vec2<f32>, is_bush: bool,
    anchor: vec3<f32>, scale: f32, h_scale: f32,
) -> TuftHit {
    var out: TuftHit;
    out.hit = false;
    // Super-volume space: the baked cube spans `scale` world cells from
    // `anchor` (1 = a single-cell tuft, 2 = a big 2x2x2 bush). The march
    // clips to the CURRENT cell so every cell renders exactly its slice.
    let o_a = (origin - anchor) / scale;
    let o_s = vec3<f32>(o_a.x - lean.x * o_a.y, o_a.y, o_a.z - lean.y * o_a.y);
    let d_s0 = vec3<f32>(dir.x - lean.x * dir.y, dir.y, dir.z - lean.y * dir.y) / scale;
    let d_s = d_s0;
    let inv = 1.0 / d_s;
    let t0v = (vec3<f32>(0.0) - o_s) * inv;
    let t1v = (vec3<f32>(1.0) - o_s) * inv;
    let tmin3 = min(t0v, t1v);
    let tmax3 = max(t0v, t1v);
    // Current-cell slab (unsheared world space).
    let o_c = origin - voxel_min;
    let tc0 = (vec3<f32>(0.0) - o_c) / dir;
    let tc1 = (vec3<f32>(1.0) - o_c) / dir;
    let tcmin = min(tc0, tc1);
    let tcmax = max(tc0, tc1);
    let t_in = max(max(max(tmin3.x, tmin3.y), max(tmin3.z, 0.0)),
                   max(max(tcmin.x, tcmin.y), tcmin.z));
    let t_out = min(min(min(tmax3.x, tmax3.y), tmax3.z),
                    min(min(tcmax.x, tcmax.y), tcmax.z));
    if (t_in >= t_out) { return out; }
    let eps = 1e-4;
    var p = (o_s + d_s * (t_in + eps)) * 16.0;
    var c = vec3<i32>(clamp(floor(p), vec3<f32>(0.0), vec3<f32>(15.0)));
    let step_i = vec3<i32>(sign(d_s));
    let inv8 = 1.0 / (d_s * 16.0);
    var t_next = vec3<f32>(1e30);
    if (d_s.x != 0.0) { t_next.x = t_in + (select(f32(c.x), f32(c.x + 1), d_s.x > 0.0) - p.x) * inv8.x; }
    if (d_s.y != 0.0) { t_next.y = t_in + (select(f32(c.y), f32(c.y + 1), d_s.y > 0.0) - p.y) * inv8.y; }
    if (d_s.z != 0.0) { t_next.z = t_in + (select(f32(c.z), f32(c.z + 1), d_s.z > 0.0) - p.z) * inv8.z; }
    let t_delta = abs(inv8);
    var t_cur = t_in;
    var axis = 1;
    for (var s = 0; s < 52; s = s + 1) {
        if (c.x < 0 || c.x > 15 || c.y < 0 || c.y > 15 || c.z < 0 || c.z > 15) { break; }
        if (t_cur > t_out) { break; }
        let my = clamp(i32(round(f32(c.y) / h_scale)), 0, 15);
        if (micro_tuft_bit(variant, c.x, my, c.z)) {
            let fy = (f32(c.y) + 0.5) / 16.0;
            // Solid faces for both families: shape and quantized tone carry
            // the read; cutouts are retired.
            {
                out.hit = true;
                out.t = t_cur;
                // Crown light: a cell with open sky above is a tip - the
                // bright accent detail lives in the voxels themselves.
                let tip = my >= 15
                    || !micro_tuft_bit(variant, c.x,
                                       clamp(i32(round(f32(c.y + 1) / h_scale)), 0, 15), c.z);
                var tip_mul = 1.0;
                if (tip) { tip_mul = select(1.10, 1.15, !is_bush); }
                var n = vec3<f32>(0.0, 1.0, 0.0);
                if (axis == 0) { n = vec3<f32>(-f32(step_i.x), 0.0, 0.0); }
                if (axis == 2) { n = vec3<f32>(0.0, 0.0, -f32(step_i.z)); }
                if (axis == 1) { n = vec3<f32>(0.0, -f32(step_i.y), 0.0); }
                out.n = n;
                out.vn = fy;
                let mh = hash3f(voxel_min + vec3<f32>(f32(c.x) * 0.37 + 1.1,
                                                      f32(c.y) * 0.53 + 2.3,
                                                      f32(c.z) * 0.71 + 3.7));
                let spread = select(0.08, 0.13, is_bush);
                out.tone = (1.0 + select(select(0.0, spread, mh > 0.66), -spread, mh < 0.33)) * tip_mul;
                return out;
            }
        }
        if (t_next.x <= t_next.y && t_next.x <= t_next.z) {
            c.x = c.x + step_i.x;
            t_cur = t_next.x;
            t_next.x = t_next.x + t_delta.x;
            axis = 0;
        } else if (t_next.y <= t_next.z) {
            c.y = c.y + step_i.y;
            t_cur = t_next.y;
            t_next.y = t_next.y + t_delta.y;
            axis = 1;
        } else {
            c.z = c.z + step_i.z;
            t_cur = t_next.z;
            t_next.z = t_next.z + t_delta.z;
            axis = 2;
        }
    }
    return out;
}

// Crown cards: two X planes through the cell centre carrying a cutout
// sprite (fluffy tuft spray for grass, the 32x32 oak leaf tuft for
// bushes), wind-sheared - the overhanging soft detail the pure volume
// lacks (the tree-canopy recipe).
fn tuft_card_hit(
    voxel_min: vec3<f32>, origin: vec3<f32>, dir: vec3<f32>,
    sprite: u32, is_bush: bool, wind: vec2<f32>, vh: f32,
) -> TuftHit {
    var out: TuftHit;
    out.hit = false;
    let voxel_center = voxel_min + vec3<f32>(0.5);
    let mirror_u = fract(vh * 16.0) > 0.5;
    var best_t = 1e30;
    for (var i = 0; i < 2; i = i + 1) {
        var pn = vec3<f32>(0.7071, 0.0, 0.7071);
        var pt = vec3<f32>(0.7071, 0.0, -0.7071);
        if (i == 1) {
            pn = vec3<f32>(0.7071, 0.0, -0.7071);
            pt = vec3<f32>(0.7071, 0.0, 0.7071);
        }
        let denom = dot(dir, pn);
        if (abs(denom) < 1e-4) { continue; }
        let t = dot(voxel_center - origin, pn) / denom;
        if (t < 0.0 || t >= best_t) { continue; }
        let p_hit = origin + dir * t;
        let local = p_hit - voxel_min;
        if (local.x < 0.0 || local.x > 1.0
         || local.y < 0.0 || local.y > 1.0
         || local.z < 0.0 || local.z > 1.0) { continue; }
        let v = local.y;
        let sx = local.x - wind.x * v;
        let sz = local.z - wind.y * v;
        let s_w = (sx - 0.5) * pt.x + (sz - 0.5) * pt.z;
        let u = clamp((s_w + 0.70711) / 1.41421, 0.0, 0.99999);
        var val = 0u;
        if (is_bush) {
            var tx = u32(u * 32.0) & 31u;
            if (mirror_u) { tx = 31u - tx; }
            val = tuft_texel(TUFT_OAK, tx, u32(clamp(v * 32.0, 0.0, 31.0)));
        } else {
            var tx = u32(u * 16.0) & 15u;
            if (mirror_u) { tx = 15u - tx; }
            val = sprite_texel(sprite, tx, u32(clamp(v * 16.0, 0.0, 15.0)));
        }
        if (val == 0u) { continue; }
        best_t = t;
        out.hit = true;
        out.t = t;
        out.n = select(pn, -pn, denom > 0.0);
        out.vn = v;
        // Card texel tones ride the same quantized scale; '*' texels are
        // the bright accent detail.
        out.tone = select(select(1.0, 1.18, val == 3u), 0.88, val == 2u);
    }
    return out;
}

fn grass_tuft_hit(voxel: vec3<i32>, origin: vec3<f32>, dir: vec3<f32>) -> SubHit {
    var out: SubHit;
    out.hit = false;
    out.color_tint = vec3<f32>(1.0);
    let voxel_min = vec3<f32>(f32(voxel.x), f32(voxel.y), f32(voxel.z));
    let vh = hash3f(voxel_min);
    let variant = u32(fract(vh * 8.0) * 3.0) % 3u;
    let phase = voxel_min.x * 0.40 + voxel_min.z * 0.55 + vh * 6.28;
    let wind = wind_offset(voxel_min, phase, 0.30);
    let lean = clamp(wind, vec2<f32>(-0.25), vec2<f32>(0.25));

    // Volume only (pure voxel read); per-tuft height scale varies whole
    // tussocks from squat to tall.
    let h_scale = 0.55 + fract(vh * 32.0) * 0.75;
    let h = tuft_volume_march(voxel_min, origin, dir, variant, lean, false,
                              voxel_min, 1.0, h_scale);
    if (!h.hit) { return out; }

    let ground = mix(vec3<f32>(1.0),
                     palette[MAT_GRASS].rgb / max(palette[MAT_TALL_GRASS].rgb, vec3<f32>(1e-3)),
                     0.6);
    // Base sits at the ground colour (no extra shading step); brightness
    // only lifts toward the tips.
    let s2 = smoothstep(0.62, 0.92, h.vn);
    let b = (1.0 + 0.16 * s2) * h.tone;
    out.hit = true;
    out.t_hit = h.t;
    // Bottom cells light like the ground they grow from (normal blended
    // toward up near the base), so tuft bases fuse with the lawn.
    out.normal = normalize(mix(vec3<f32>(0.0, 1.0, 0.0), h.n,
                               clamp(h.vn * 2.2, 0.25, 1.0)));
    out.color_tint = vec3<f32>(b) * ground;
    return out;
}

fn bush_hit(voxel: vec3<i32>, origin: vec3<f32>, dir: vec3<f32>) -> SubHit {
    var out: SubHit;
    out.hit = false;
    out.color_tint = vec3<f32>(1.0);
    let voxel_min = vec3<f32>(f32(voxel.x), f32(voxel.y), f32(voxel.z));
    let vh = hash3f(voxel_min);
    let variant = 3u + (u32(fract(vh * 8.0) * 3.0) % 3u);
    let phase = voxel_min.x * 0.40 + voxel_min.z * 0.55 + vh * 6.28;
    let wind = wind_offset(voxel_min, phase, 0.18);
    let lean = clamp(wind, vec2<f32>(-0.15), vec2<f32>(0.15));

    // Big-bush detection: a 2x2x2 lattice block of bush cells marches ONE
    // dome spanning the block (anchor at the even lattice); lone cells
    // stay single-cell domes.
    var anchor = voxel_min;
    var scale = 1.0;
    let ax = voxel.x & -2;
    let az = voxel.z & -2;
    var ay = voxel.y;
    if (voxel_material_at(vec3<i32>(voxel.x, voxel.y - 1, voxel.z)) == MAT_BUSH) {
        ay = voxel.y - 1;
    }
    // The four cells of the aligned block must all be bush.
    let corner = voxel_material_at(vec3<i32>(ax + 1, ay, az + 1));
    let corner2 = voxel_material_at(vec3<i32>(ax, ay, az));
    if (corner == MAT_BUSH && corner2 == MAT_BUSH
        && voxel_material_at(vec3<i32>(ax + 1, ay, az)) == MAT_BUSH
        && voxel_material_at(vec3<i32>(ax, ay, az + 1)) == MAT_BUSH) {
        anchor = vec3<f32>(f32(ax), f32(ay), f32(az));
        scale = 2.0;
    }
    let h = tuft_volume_march(voxel_min, origin, dir, variant, lean, true,
                              anchor, scale, 1.0);
    if (!h.hit) { return out; }

    // Leafy tint: coupled toward the leaf palette, gentle top-light ramp.
    let ground = mix(vec3<f32>(1.0),
                     palette[MAT_LEAVES].rgb / max(palette[MAT_BUSH].rgb, vec3<f32>(1e-3)),
                     0.7);
    let b = (0.92 + 0.16 * smoothstep(0.35, 0.9, h.vn)) * h.tone;
    out.hit = true;
    out.t_hit = h.t;
    out.normal = h.n;
    out.color_tint = vec3<f32>(b) * ground;
    return out;
}

// TREE TEST: march the shared 128^3 wood+leaf volume. The block anchors
// to the 8-cell lattice; each cell marches its slice at 16 micro cells
// per world cell. Leaves are REAL 3D blobs - see-through comes from true
// air between them, no cutouts anywhere.
fn tree_bit(wood: bool, x: i32, y: i32, z: i32) -> bool {
    let bit = u32(x + z * 128 + y * 16384);
    let base = select(TREE_LEAF_BASE_WORDS, TREE_WOOD_BASE_WORDS, wood);
    let w = sprites[base + (bit >> 5u)];
    return ((w >> (bit & 31u)) & 1u) != 0u;
}

fn tree_test_hit(voxel: vec3<i32>, origin: vec3<f32>, dir: vec3<f32>) -> SubHit {
    var out: SubHit;
    out.hit = false;
    out.color_tint = vec3<f32>(1.0);
    let anchor = vec3<f32>(f32(voxel.x & -8), f32(voxel.y & -8), f32(voxel.z & -8));
    let voxel_min = vec3<f32>(f32(voxel.x), f32(voxel.y), f32(voxel.z));
    // Super space: the volume spans 8 cells from the anchor.
    let o_s = (origin - anchor) / 8.0;
    let d_s = dir / 8.0;
    let inv = 1.0 / d_s;
    let t0v = (vec3<f32>(0.0) - o_s) * inv;
    let t1v = (vec3<f32>(1.0) - o_s) * inv;
    let tmin3 = min(t0v, t1v);
    let tmax3 = max(t0v, t1v);
    let o_c = origin - voxel_min;
    let tc0 = (vec3<f32>(0.0) - o_c) / dir;
    let tc1 = (vec3<f32>(1.0) - o_c) / dir;
    let tcmin = min(tc0, tc1);
    let tcmax = max(tc0, tc1);
    let t_in = max(max(max(tmin3.x, tmin3.y), max(tmin3.z, 0.0)),
                   max(max(tcmin.x, tcmin.y), tcmin.z));
    let t_out = min(min(min(tmax3.x, tmax3.y), tmax3.z),
                    min(min(tcmax.x, tcmax.y), tcmax.z));
    if (t_in >= t_out) { return out; }
    let eps = 1e-4;
    var p = (o_s + d_s * (t_in + eps)) * 128.0;
    var c = vec3<i32>(clamp(floor(p), vec3<f32>(0.0), vec3<f32>(127.0)));
    let step_i = vec3<i32>(sign(d_s));
    let invm = 1.0 / (d_s * 128.0);
    var t_next = vec3<f32>(1e30);
    if (d_s.x != 0.0) { t_next.x = t_in + (select(f32(c.x), f32(c.x + 1), d_s.x > 0.0) - p.x) * invm.x; }
    if (d_s.y != 0.0) { t_next.y = t_in + (select(f32(c.y), f32(c.y + 1), d_s.y > 0.0) - p.y) * invm.y; }
    if (d_s.z != 0.0) { t_next.z = t_in + (select(f32(c.z), f32(c.z + 1), d_s.z > 0.0) - p.z) * invm.z; }
    let t_delta = abs(invm);
    var t_cur = t_in;
    var axis = 1;
    for (var s = 0; s < 56; s = s + 1) {
        if (c.x < 0 || c.x > 127 || c.y < 0 || c.y > 127 || c.z < 0 || c.z > 127) { break; }
        if (t_cur > t_out) { break; }
        let wood = tree_bit(true, c.x, c.y, c.z);
        let leaf = !wood && tree_bit(false, c.x, c.y, c.z);
        if (wood || leaf) {
            out.hit = true;
            out.t_hit = t_cur;
            var n = vec3<f32>(0.0, 1.0, 0.0);
            if (axis == 0) { n = vec3<f32>(-f32(step_i.x), 0.0, 0.0); }
            if (axis == 2) { n = vec3<f32>(0.0, 0.0, -f32(step_i.z)); }
            if (axis == 1) { n = vec3<f32>(0.0, -f32(step_i.y), 0.0); }
            out.normal = n;
            let mh = hash3f(anchor + vec3<f32>(f32(c.x) * 0.37, f32(c.y) * 0.53, f32(c.z) * 0.71));
            let tone = 1.0 + select(select(0.0, 0.12, mh > 0.66), -0.12, mh < 0.33);
            if (wood) {
                // Bark: palette-absolute (MAT_TREE_TEST palette is white).
                out.color_tint = vec3<f32>(0.34, 0.23, 0.13) * tone;
            } else {
                let tip = c.y >= 127 || !(tree_bit(false, c.x, c.y + 1, c.z) || tree_bit(true, c.x, c.y + 1, c.z));
                out.color_tint = vec3<f32>(0.26, 0.52, 0.16) * tone
                    * select(1.0, 1.14, tip);
            }
            return out;
        }
        if (t_next.x <= t_next.y && t_next.x <= t_next.z) {
            c.x = c.x + step_i.x;
            t_cur = t_next.x;
            t_next.x = t_next.x + t_delta.x;
            axis = 0;
        } else if (t_next.y <= t_next.z) {
            c.y = c.y + step_i.y;
            t_cur = t_next.y;
            t_next.y = t_next.y + t_delta.y;
            axis = 1;
        } else {
            c.z = c.z + step_i.z;
            t_cur = t_next.z;
            t_next.z = t_next.z + t_delta.z;
            axis = 2;
        }
    }
    return out;
}

fn foliage_subvoxel(voxel: vec3<i32>, origin: vec3<f32>, dir: vec3<f32>, mat: u32) -> SubHit {
    var hit: SubHit;
    if (mat == MAT_TURF) {
        // Blades vanish (dithered) where they are sub-pixel; the grass-top
        // sheen shading carries the field beyond.
        let cd = length(vec3<f32>(voxel) + vec3<f32>(0.5) - origin);
        let edge = TURF_T + (hash3f(vec3<f32>(voxel)) - 0.5) * 6.0;
        if (cd < edge) {
            hit = turf_blade_hit(voxel, origin, dir);
        } else {
            hit.hit = false;
            hit.color_tint = vec3<f32>(1.0);
        }
        return hit;
    }
    if (mat == MAT_TALL_GRASS) {
        // Micro-voxel tussock with cutout faces and crown cards.
        hit = grass_tuft_hit(voxel, origin, dir);
    } else if (mat == MAT_BUSH) {
        hit = bush_hit(voxel, origin, dir);
    } else if (mat == MAT_TREE_TEST) {
        hit = tree_test_hit(voxel, origin, dir);
    } else if (mat == MAT_TALL_GRASS_DRY) {
        hit = sprite_cross_hit(voxel, origin, dir, mat);
    } else if (mat == MAT_FLOWER) {
        hit = sprite_cross_hit(voxel, origin, dir, mat);
    } else if (mat == MAT_LEAF_FRINGE) {
        hit = leaf_fringe_hit(voxel, origin, dir);
    } else {
        hit = leaf_bl_hit(voxel, origin, dir, mat);
    }
    return hit;
}

// Axis index of an exact axis-aligned face normal, or -1 for oblique normals
// (cross-quad sprites). Leaf cube hits keep their face axis so they qualify
// for cube AO and the reprojected lighting cache.
fn axis_from_face_normal(n: vec3<f32>) -> i32 {
    if (abs(n.x) > 0.99) { return 0; }
    if (abs(n.y) > 0.99) { return 1; }
    if (abs(n.z) > 0.99) { return 2; }
    return -1;
}

// Sun rotates east→up→west→under. Start near midday so the very first frame
// isn't dim/orange; cycle slows to ~5 minutes for a less twitchy feel.
// Bodies live in common.wgsl (shared with the falling-leaf pass). The sun
// runs on its own clock so --freeze-time can pin the day/night cycle
// without stopping water, wind or leaves.
fn sun_dir() -> vec3<f32> {
    return sun_dir_at(camera.sun_time);
}

// IQ-style fract hash. The previous sin-based hash had visible periodic
// patterns at integer-aligned coords (that's where the "chess grid" came
// from). This one is uniform across all positions we sample.
fn hash3f(pin: vec3<f32>) -> f32 {
    var q = fract(pin * vec3<f32>(0.1031, 0.1030, 0.0973));
    q = q + dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

// Bit-exact lattice hash (PCG-3D mix, Jarzynski-Olano): noise LATTICE cells
// must be hashed with integer arithmetic only. The float hash above takes
// fract() of products around 2e4, where one f32 ULP is ~2e-3: for
// knife-edge cells the result flips with whichever FMA contraction the
// driver picks PER INLINED CALL SITE, so the same lattice cell can hash
// differently on the two sides of a lattice line - a hard pattern seam in
// the middle of a flat face (the "stone rows shifted" bug; same compiler-
// freedom family as the naga OpSRem poison, but legal float behaviour).
// Integer ops have no rounding: bit-stable across every call site, every
// unrolled loop copy, every driver. Takes integer-VALUED floats (floor()
// results / lattice corners; salts encoded as whole numbers).
fn hash_lattice3(ip: vec3<f32>) -> f32 {
    var v = vec3<u32>(
        bitcast<u32>(i32(ip.x)),
        bitcast<u32>(i32(ip.y)),
        bitcast<u32>(i32(ip.z)),
    );
    v = v * vec3<u32>(1664525u) + vec3<u32>(1013904223u);
    v.x = v.x + v.y * v.z;
    v.y = v.y + v.z * v.x;
    v.z = v.z + v.x * v.y;
    v = v ^ (v >> vec3<u32>(16u));
    v.x = v.x + v.y * v.z;
    v.y = v.y + v.z * v.x;
    v.z = v.z + v.x * v.y;
    return f32(v.x) * 2.3283064e-10;
}
fn vnoise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let n000 = hash_lattice3(i + vec3<f32>(0.0, 0.0, 0.0));
    let n100 = hash_lattice3(i + vec3<f32>(1.0, 0.0, 0.0));
    let n010 = hash_lattice3(i + vec3<f32>(0.0, 1.0, 0.0));
    let n110 = hash_lattice3(i + vec3<f32>(1.0, 1.0, 0.0));
    let n001 = hash_lattice3(i + vec3<f32>(0.0, 0.0, 1.0));
    let n101 = hash_lattice3(i + vec3<f32>(1.0, 0.0, 1.0));
    let n011 = hash_lattice3(i + vec3<f32>(0.0, 1.0, 1.0));
    let n111 = hash_lattice3(i + vec3<f32>(1.0, 1.0, 1.0));
    let a = mix(n000, n100, u.x);
    let b = mix(n010, n110, u.x);
    let c = mix(n001, n101, u.x);
    let d = mix(n011, n111, u.x);
    return mix(mix(a, b, u.y), mix(c, d, u.y), u.z);
}

// Cumulus-style cloud density. Low-frequency coverage mask gates a fbm body
// so the sky has discrete clumps with empty regions between (not haze). The
// height-falloff bell concentrates density mid-slab — flat bottoms and
// rounded tops, like real cumulus.
// Coverage-and-envelope-only density for the GROUND cloud-shade taps: shadows
// need the cloud MASS silhouette, not worley florets or tower fields, and
// these taps run 3x per full-res pixel every frame - the full field there cost
// ~1.5 ms of compose. Deliberate second implementation for a hot path (the
// full field stays the one source of the RENDERED cloud).
// Base-lift sample coordinate: xz only (the underside is a heightfield, not
// volumetric), matched between the full and coarse density fields.
fn vec2p_base(pa: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(pa.x * 2.3, 17.0, pa.z * 2.3);
}

fn cloud_density_coarse(p: vec3<f32>, t: f32) -> f32 {
    let pa = p * 0.0055 + vec3<f32>(t * 0.06, 0.0, t * 0.035);
    let cov_lo = vnoise3(pa * 1.05);
    let cov_mid = vnoise3(pa * 2.6);
    let cov = smoothstep(0.55, 0.65, cov_lo * 0.60 + cov_mid * 0.40);
    if (cov <= 0.0) { return 0.0; }
    let h = clamp((p.y - CLOUD_BASE) / max(1.0, CLOUD_TOP - CLOUD_BASE), 0.0, 1.0);
    let base_lift = vnoise3(vec2p_base(pa)) * 0.14;
    let envelope = smoothstep(base_lift, base_lift + 0.05, h) * (1.0 - smoothstep(0.45, 1.0, h));
    return clamp((cov * 0.8 - 0.25) * 9.0 * envelope, 0.0, 1.0);
}

fn hash33(p: vec3<f32>) -> vec3<f32> {
    var q = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    q = q + dot(q, q.yxz + 33.33);
    return fract((q.xxy + q.yxx) * q.zyx);
}

// 8-cell 3D Worley F1 (distance to nearest jittered feature point, ~0..1).
// The 2x2x2 neighbourhood picks the cells nearest to p per axis - accurate
// enough for cloud florets at a third of the 27-cell cost. Worley is THE
// floret maker: its iso-surfaces are spheres around the feature points, so
// eroding a density field with it leaves round convex lobes (cauliflower),
// where value-noise erosion just leaves mush.
fn worley3f(p: vec3<f32>) -> f32 {
    let ip = floor(p);
    let fp = p - ip;
    let shift = vec3<i32>(
        select(-1, 0, fp.x > 0.5),
        select(-1, 0, fp.y > 0.5),
        select(-1, 0, fp.z > 0.5),
    );
    var dmin = 4.0;
    for (var k: i32 = 0; k < 8; k = k + 1) {
        let o = vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1) + shift;
        let fo = vec3<f32>(o);
        let feat = fo + hash33(ip + fo);
        let dv = feat - fp;
        dmin = min(dmin, dot(dv, dv));
    }
    return clamp(sqrt(dmin), 0.0, 1.0);
}

// `t` is the DAY clock (camera.sun_time): equal to render time in normal play,
// pinned by --freeze-time so frozen scenes freeze their clouds (and the
// rigid-ground temporal guard isolates rogue shading fields from legit drift).
fn cloud_density(p: vec3<f32>, t: f32) -> f32 {
    let pa = p * 0.0055 + vec3<f32>(t * 0.06, 0.0, t * 0.035);
    // Cumulus coverage: a low-freq clump field carved by a smoothstep
    // threshold into DISTINCT clouds with genuinely clear sky between -
    // never a linear coverage ramp, which spreads a translucent stratus
    // veil everywhere. The mid-freq term keeps clump outlines irregular.
    // Smaller clumps (higher coverage frequency), and a strong mid-frequency
    // term so each cloud reads as a CLUSTER of round masses packed together
    // (the ice-cream-scoop look) instead of one amorphous blot.
    let cov_lo = vnoise3(pa * 1.05);
    let cov_mid = vnoise3(pa * 2.6);
    let covf = cov_lo * 0.60 + cov_mid * 0.40;
    let cov = smoothstep(0.55, 0.65, covf);
    if (cov <= 0.0) { return 0.0; }
    // Body: 4 octaves of fbm. Vertical noise is scaled finer so a horizontal
    // slice doesn't look like a flat layer when viewed sideways.
    let pb = vec3<f32>(pa.x, pa.y * 3.5, pa.z);
    // Scoop shape comes from the COVERAGE peaks (round columns capped by the
    // dome), NOT from carving iso-shells out of body noise - that made curled
    // ribbons. Body is a smooth two-octave fullness modulation only.
    let n1 = vnoise3(pb);
    let n2 = vnoise3(pb * 2.7);
    let body = n1 * 0.65 + n2 * 0.35;
    // Cauliflower surface: WORLEY-F1 erosion near the density boundary only
    // (full strength at the surface, none deep inside). Worley removes the
    // material BETWEEN feature points, leaving round convex lobes bulging
    // outward - two octaves give big florets carrying small ones, the real
    // cumulus fractal.
    let det = worley3f(pb * 4.6) * 0.68 + worley3f(pb * 10.3) * 0.32;
    // Cumulus profile: sharp flat bottom, and TOWERS - a second noise field
    // picks where each clump billows upward, so cores rise as rounded
    // cauliflower heads (weak coverage stays a low base layer near the
    // slab bottom, strong tower spots climb toward CLOUD_TOP). Vertical
    // development, not just wider clumps.
    let h = clamp((p.y - CLOUD_BASE) / max(1.0, CLOUD_TOP - CLOUD_BASE), 0.0, 1.0);
    // Lumpy underside: the base onset height varies per region with the
    // wildcard field (mostly near-level like a real condensation base, but
    // never a geometric plane), and the worley erosion then sculpts what the
    // onset exposes.
    let base_lift = vnoise3(vec2p_base(pa)) * 0.14;
    let bottom_fade = smoothstep(base_lift, base_lift + 0.05, h);
    // Height varies INSIDE a cloud, not one cap per clump:
    //  - `interior` grows from the clump's edge toward its core (the raw
    //    coverage field past the carve threshold), so edges stay low and puffy
    //    while the middle billows - no more uniform-height slabs.
    //  - `tn` adds mid-frequency bumps (~55 voxels) so one big cloud carries
    //    several cauliflower heads at different heights.
    // Every clump keeps a CHUNKY opaque base (dome floor 0.30 - capping lower
    // squashed them into translucent wisps); strong interior cores climb the
    // whole taller slab.
    let interior = smoothstep(0.56, 0.84, covf);
    let tn = vnoise3(pa * 3.2 + vec3<f32>(31.0, 0.0, 17.0));
    // TENDENCY, not rule: the edge->core gradient carries most of the cap
    // height (edges flatter, middles taller), but an independent low-freq
    // field adds +-0.17 so SOME edges still billow and SOME cores stay low -
    // a strict dome-per-clump read as artificial.
    let wildcard = vnoise3(pa * 1.1 + vec3<f32>(7.0, 0.0, 43.0));
    let tower = clamp(interior * (0.30 + 0.70 * tn * tn) + (wildcard - 0.5) * 0.35, 0.0, 1.0);
    let dome = 1.0 - smoothstep(0.30 + 0.70 * tower, 1.0, h);
    let envelope = bottom_fade * dome;
    // Full round masses: coverage bounds the scoop, body only modulates its
    // fullness; the steep scale saturates cores to opaque white. The edge
    // factor (1 at the boundary, 0 deep inside) applies the floret erosion.
    // ORDER MATTERS (the Nubis lesson): erode the GRADUAL shape field first,
    // sharpen after. Eroding an already-hard field acts on a razor-thin shell
    // and cannot sculpt lobes; eroding the soft shell carves floret-sized
    // spheres, then the final remap restores the crisp opaque edge.
    let shape = clamp(cov * (0.50 + 0.50 * body) * envelope, 0.0, 1.0);
    let eroded = shape - det * det * (1.0 - shape) * 0.62;
    let d = (eroded - 0.20) * 9.0;
    return clamp(d, 0.0, 1.0);
}

fn sky_color(dir: vec3<f32>) -> vec3<f32> {
    let s = sun_dir();
    let day_t = sun_intensity(s);

    // Atmosphere: smoothstep elevation, hazier near the horizon (Rayleigh-ish
    // tint). Down-facing rays inherit the horizon color (camera looking
    // *under* shouldn't see deep-blue zenith showing through ground gaps).
    let up = clamp(dir.y, -0.2, 1.0);
    let zen_f = smoothstep(0.0, 0.55, up);
    let day_zenith  = vec3<f32>(0.22, 0.50, 0.95);
    let day_horizon = vec3<f32>(0.78, 0.86, 0.95);
    let day_sky = mix(day_horizon, day_zenith, zen_f);
    let night_zenith  = vec3<f32>(0.005, 0.010, 0.035);
    let night_horizon = vec3<f32>(0.030, 0.040, 0.090);
    let night_sky = mix(night_horizon, night_zenith, zen_f);

    var col = mix(night_sky, day_sky, day_t);

    // Sunset / sunrise warmth: warm wash near horizon weighted by alignment
    // to the sun's azimuth and the sun's altitude (peak around horizon).
    let dusk_w = smoothstep(-0.08, 0.18, s.y) * (1.0 - smoothstep(0.05, 0.40, s.y));
    let azi = clamp(dot(normalize(vec2<f32>(dir.x, dir.z)), normalize(vec2<f32>(s.x, s.z))), 0.0, 1.0);
    let warm_horizon = vec3<f32>(1.50, 0.55, 0.18) * dusk_w * pow(azi, 3.0) * (1.0 - zen_f);
    col = col + warm_horizon;

    // General halo / forward-scatter near the sun.
    let sun_align = max(0.0, dot(dir, s));
    let halo = vec3<f32>(1.0, 0.78, 0.52) * pow(sun_align, 6.0) * day_t * 0.45;
    col = col + halo;

    // Sun disc.
    if (sun_align > 0.9985) {
        col = mix(col, vec3<f32>(2.4, 2.0, 1.5), 0.9);
    }

    // Stars when sun is below the horizon. Simple high-frequency noise
    // threshold; faint and scattered.
    if (day_t < 0.30) {
        let stn = vnoise3(dir * 95.0);
        if (stn > 0.93) {
            let amp = (stn - 0.93) * 14.0 * (1.0 - day_t);
            col = col + vec3<f32>(0.9, 0.92, 1.0) * amp;
        }
    }

    // Clouds disabled for now.
    return col;
}

// ---- world-projected procedural textures ----------------------------------
// Tri-planar UV: pick the plane perpendicular to the face's dominant axis,
// then read world-space coords on that plane. Same material across multiple
// voxels reads continuous texture; voxel boundaries vanish.
fn tex_uv(p: vec3<f32>, n: vec3<f32>) -> vec2<f32> {
    let an = abs(n);
    if (an.y > an.x && an.y > an.z) { return p.xz; }    // top/bottom face
    if (an.x > an.z)                { return p.zy; }    // ±X face
    return p.xy;                                        // ±Z face
}

// Gradient (Perlin-style) noise with a quintic fade: C2-continuous, so no
// lattice creases - the value noise's smoothstep plateaus read as blobs
// with faint grid edges INSIDE blocks, which is exactly what textures must
// not do. Gradients are cheap unnormalized hash vectors; the output is
// remapped to ~0..1.
fn gvec(i: vec3<f32>) -> vec3<f32> {
    // ONE hash fanned into three channels: a third of the inlined code of
    // hashing per channel, which matters because the driver inlines every
    // instance (3x cost = minutes of driver compile, measured).
    let h = hash_lattice3(i);
    return vec3<f32>(fract(h * 5.37), fract(h * 7.79), fract(h * 9.13)) * 2.0 - vec3<f32>(1.0);
}

fn gnoise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let d000 = dot(gvec(i + vec3<f32>(0.0, 0.0, 0.0)), f - vec3<f32>(0.0, 0.0, 0.0));
    let d100 = dot(gvec(i + vec3<f32>(1.0, 0.0, 0.0)), f - vec3<f32>(1.0, 0.0, 0.0));
    let d010 = dot(gvec(i + vec3<f32>(0.0, 1.0, 0.0)), f - vec3<f32>(0.0, 1.0, 0.0));
    let d110 = dot(gvec(i + vec3<f32>(1.0, 1.0, 0.0)), f - vec3<f32>(1.0, 1.0, 0.0));
    let d001 = dot(gvec(i + vec3<f32>(0.0, 0.0, 1.0)), f - vec3<f32>(0.0, 0.0, 1.0));
    let d101 = dot(gvec(i + vec3<f32>(1.0, 0.0, 1.0)), f - vec3<f32>(1.0, 0.0, 1.0));
    let d011 = dot(gvec(i + vec3<f32>(0.0, 1.0, 1.0)), f - vec3<f32>(0.0, 1.0, 1.0));
    let d111 = dot(gvec(i + vec3<f32>(1.0, 1.0, 1.0)), f - vec3<f32>(1.0, 1.0, 1.0));
    let x0 = mix(mix(d000, d100, u.x), mix(d010, d110, u.x), u.y);
    let x1 = mix(mix(d001, d101, u.x), mix(d011, d111, u.x), u.y);
    return mix(x0, x1, u.z) * 0.6 + 0.5;
}

// Rotated-octave gradient fbm: the domain rotates between octaves so no
// lattice direction ever aligns with the world axes.
fn fbm3g(p: vec3<f32>) -> f32 {
    var q = p;
    var acc = gnoise3(q) * 0.55;
    q = vec3<f32>(q.x * 0.80 - q.z * 0.60, q.y * 0.94 + q.x * 0.34,
                  q.x * 0.60 + q.z * 0.80) * 2.13 + vec3<f32>(7.7);
    acc = acc + gnoise3(q) * 0.30;
    q = vec3<f32>(q.x * 0.80 - q.z * 0.60, q.y * 0.94 + q.x * 0.34,
                  q.x * 0.60 + q.z * 0.80) * 2.13 + vec3<f32>(3.1);
    acc = acc + gnoise3(q) * 0.15;
    return acc;
}

// Legacy 3-octave value fbm (still used outside the material system).
fn fbm3(p: vec3<f32>) -> f32 {
    return vnoise3(p) * 0.55 + vnoise3(p * 2.7) * 0.30 + vnoise3(p * 6.1) * 0.15;
}

// Thin bright line where a noise field crosses its midpoint - crack/vein
// networks for stone, ice and lava.
fn ridge_line(v: f32, w: f32) -> f32 {
    return 1.0 - smoothstep(0.0, w, abs(v - 0.5));
}

// 2D cellular (Worley) noise over PLANAR face coordinates (tex_uv): f2 - f1
// is ~0 on the border between two cells, giving angular plate borders -
// fractured rock facets and ice panes.
// PLANAR, deliberately, and never a 3D world-space field: line features
// sampled from a 3D field show a different SLICE on parallel faces at
// different depths, so crack lines stop lining up at every 1-voxel wall
// step or terrace - the exact "stone does not line up" report. A pattern
// that is a pure function of the face-plane coordinates is identical on
// all faces of the same axis BY CONSTRUCTION (guard test:
// texture_pattern_is_depth_invariant). Corner (orientation) continuity is
// explicitly NOT a goal - lighting already breaks at corners.
struct CellNoise {
    f1: f32,
    f2: f32,
    id: f32,
}

fn worley2(p: vec2<f32>) -> CellNoise {
    let ip = floor(p);
    let fp = fract(p);
    var out: CellNoise;
    out.f1 = 8.0;
    out.f2 = 8.0;
    out.id = 0.0;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let g = vec2<f32>(f32(dx), f32(dy));
            let h = hash_lattice3(vec3<f32>(ip + g, 17.0));
            let o = vec2<f32>(fract(h * 7.13), fract(h * 13.71));
            let d = length(g + o - fp);
            if (d < out.f1) {
                out.f2 = out.f1;
                out.f1 = d;
                out.id = h;
            } else if (d < out.f2) {
                out.f2 = d;
            }
        }
    }
    return out;
}

// Bark per species - side faces get vertical ridge relief; birch gets its
// signature dark horizontal lenticel scars on a pale smooth bark; pine gets
// plated scales. End grain (top/bottom) keeps growth rings.
fn bark_pattern(p: vec3<f32>, n: vec3<f32>, mat: u32) -> vec3<f32> {
    let an = abs(n);
    if (an.y > 0.5) {
        // End-grain rings, slightly irregular.
        let r = sqrt(p.x * p.x + p.z * p.z);
        let rings = 0.5 + 0.5 * sin(r * 4.5 + vnoise3(vec3<f32>(p.x * 0.4, 0.0, p.z * 0.4)) * 2.0);
        return vec3<f32>(0.72 + rings * 0.28);
    }
    let uv = tex_uv(p, n);
    if (mat == 23u) {
        // Birch: pale, smooth, with dark lenticel bands stretched wide and
        // thin (fast noise along y, slow along the face).
        let lent = smoothstep(0.68, 0.76, gnoise3(vec3<f32>(uv.x * 0.9, p.y * 5.5, 61.0)));
        let base = 1.04 + vnoise3(vec3<f32>(uv * 3.0, 62.0)) * 0.08;
        return vec3<f32>(mix(base, 0.38, lent));
    }
    if (mat == 24u) {
        // Pine: scaly plates - dark borders where two stretched noise fields
        // cross their midlines.
        let pa = ridge_line(gnoise3(vec3<f32>(uv.x * 2.4, p.y * 1.1, 67.0)), 0.06);
        let pb = ridge_line(gnoise3(vec3<f32>(uv.x * 1.2, p.y * 2.9, 68.0)), 0.06);
        let plate = fbm3g(vec3<f32>(uv * 2.0, 69.0)) * 0.20 + 0.88;
        return vec3<f32>(plate * (1.0 - max(pa, pb) * 0.38));
    }
    // Oak: deep coarse vertical ridges + fine fibre noise.
    let ridges = pow(0.5 + 0.5 * sin(uv.x * 9.0 + vnoise3(p * 0.8) * 4.0), 1.4);
    let fibre = vnoise3(vec3<f32>(p.x * 8.0, p.y * 1.2, p.z * 8.0)) * 0.10;
    return vec3<f32>(0.66 + ridges * 0.34 + fibre);
}

// All material textures return an rgb multiplier (~0.5-1.4 per channel)
// applied to the material's base palette colour: identity colours come from
// the palette, world-projected detail and SUBTLE hue variation come from
// here. Textures are continuous across faces (world-projected), and hue
// shifts stay gentle - recolouring a whole material reads as a palette
// swap, which is not the job of this function.
// High-frequency micro-grain shared by every textured material: without it
// the smooth noise reads as soft low-res blobs. Sampled in the PLANAR
// tex_uv domain like every line-producing field, so it too is identical on
// parallel faces at different depths (depth-invariance rule, see worley2).
fn micro_grain(uv: vec2<f32>) -> f32 {
    return 0.93 + vnoise3(vec3<f32>(uv * 13.0, 7.0)) * 0.10
                + vnoise3(vec3<f32>(uv * 27.0, 19.0)) * 0.05;
}

fn material_texture(p: vec3<f32>, n: vec3<f32>, mat: u32) -> vec3<f32> {
    let base = material_texture_base(p, n, mat);
    // Glass and untextured materials skip the grain (identity base).
    if (mat == 18u || (base.x == 1.0 && base.y == 1.0 && base.z == 1.0)) {
        return base;
    }
    return base * micro_grain(tex_uv(p, n));
}

fn material_texture_base(p: vec3<f32>, n: vec3<f32>, mat: u32) -> vec3<f32> {
    let uv = tex_uv(p, n);
    // Stone - fractured rock: angular facet plates from cellular noise
    // (flat-ish grey per plate, thin dark borders as cracks), a hint of
    // strata. Smooth ridge/wave lines are exactly what stone must NOT be.
    if (mat == 4u) {
        // Planar-domain warp and cells (depth-invariance rule, see worley2).
        // Strata phase comes from the smooth jig field, NEVER from w.id: a
        // per-cell constant phase makes the banding jump at every cell
        // border ("weird edges even within one block", the original planar
        // stone bug that got misread as a projection problem).
        let jig = fbm3g(vec3<f32>(uv * 0.4, 3.0)) * 0.7;
        let w = worley2(uv * 1.5 + vec2<f32>(jig));
        let facet = 0.82 + fract(w.id * 9.7) * 0.24;
        let border = smoothstep(0.10, 0.02, w.f2 - w.f1);
        // Side faces: wavy horizontal bands (p.y is in-plane there, jig
        // bends the band lines). Horizontal faces: the exposed layer is one
        // constant level - no jig, or the per-face constant would leak
        // spatial variation and break depth invariance on terraces.
        let sphase = select(jig * 3.0, 0.0, abs(n.y) > 0.5);
        let strata = 0.95 + 0.05 * sin(p.y * 1.9 + sphase);
        return vec3<f32>(facet * strata * (1.0 - border * 0.38));
    }
    // Bark - per species.
    if (mat == 13u || mat == 23u || mat == 24u) {
        return bark_pattern(p, n, mat);
    }
    // Grass block - green top with clump/blade detail and dry patches, dirt
    // sides with a ragged green fringe, dirt bottom.
    if (mat == 2u) {
        if (n.y > 0.5) {
            let clump = fbm3g(vec3<f32>(uv * 2.1, 11.0));
            let blades = vnoise3(vec3<f32>(uv * 14.0, 5.0));
            let lum = 0.78 + clump * 0.26 + blades * 0.12;
            // Dry patches: warm the reds slightly, never a full recolour.
            let dry = smoothstep(0.58, 0.76, gnoise3(vec3<f32>(uv * 0.5, 23.0)));
            return vec3<f32>(lum * (1.0 + dry * 0.20), lum, lum * (1.0 - dry * 0.10));
        }
        let dirt = vec3<f32>(1.3333, 0.4154, 0.75); // (0.40,0.27,0.15)/(0.30,0.65,0.20)
        let nn = vnoise3(vec3<f32>(uv * 1.6, 0.0));
        if (n.y < -0.5) {
            return dirt * (0.78 + nn * 0.30);
        }
        let hcol = hash3f(vec3<f32>(floor(uv.x * 16.0) * 0.37, floor(uv.y) * 0.11, 3.7));
        let fringe_depth = (2.0 + hcol * 4.0) / 16.0;
        if (fract(uv.y) > 1.0 - fringe_depth) {
            return vec3<f32>(0.85 + nn * 0.25);
        }
        return dirt * (0.78 + nn * 0.30);
    }
    // Dirt - clumpy soil with lighter pebbles and dark pores.
    if (mat == 3u) {
        let clump = fbm3g(vec3<f32>(uv * 3.2, 3.0)) * 0.30 + 0.74;
        let pebble = smoothstep(0.68, 0.76, gnoise3(vec3<f32>(uv * 6.5, 9.5))) * 0.35;
        let pore = smoothstep(0.68, 0.76, gnoise3(vec3<f32>(uv * 5.1, 17.0))) * 0.25;
        return vec3<f32>(clump + pebble - pore);
    }
    // Sand - wind ripples over fine grain, sparse glints.
    if (mat == 1u) {
        let ripple = 0.94 + 0.08 * sin(uv.x * 2.1 + vnoise3(vec3<f32>(uv * 0.4, 31.0)) * 4.0);
        let grain = vnoise3(vec3<f32>(uv * 15.0, 41.0)) * 0.12 + 0.90;
        let glint = max(0.0, (vnoise3(vec3<f32>(uv * 17.0, 43.0)) - 0.88) * 4.0);
        return vec3<f32>(ripple * grain + glint);
    }
    // Snow - soft drifts, cool shadowed dips, hard sparkles.
    if (mat == 15u) {
        let drift = 0.92 + 0.10 * gnoise3(vec3<f32>(uv * 0.8, 51.0));
        let dip = smoothstep(0.30, 0.0, gnoise3(vec3<f32>(uv * 2.3, 53.0))) * 0.10;
        let sparkle = max(0.0, (vnoise3(vec3<f32>(uv * 16.0, 0.0)) - 0.85) * 6.0);
        return vec3<f32>(drift - dip * 1.3 + sparkle, drift - dip * 0.9 + sparkle, drift + sparkle);
    }
    // Leaves carry their own art; no extra noise here.
    // Ice - large clear panes with sparse bright fracture borders and a
    // subtle per-pane clarity difference.
    if (mat == 17u) {
        let w = worley2(uv * 0.8);
        let body = 0.90 + fract(w.id * 7.3) * 0.10;
        let border = smoothstep(0.14, 0.03, w.f2 - w.f1);
        return vec3<f32>(body + border * 0.28, body + border * 0.32, body + border * 0.42);
    }
    // Coal - dark lumpy seams in the rock with glossy specks.
    if (mat == 19u) {
        let base = fbm3g(vec3<f32>(uv * 2.2, 73.0)) * 0.22 + 0.86;
        let seam = smoothstep(0.50, 0.60, fbm3g(vec3<f32>(uv * 2.6, 74.0)));
        let gloss = max(0.0, (vnoise3(vec3<f32>(uv * 13.0, 75.0)) - 0.88) * 5.0) * seam;
        return vec3<f32>(mix(base, 0.34, seam) + gloss);
    }
    // Iron - rusty warm veins through grey rock.
    if (mat == 20u) {
        let base = fbm3g(vec3<f32>(uv * 2.2, 77.0)) * 0.22 + 0.86;
        let vein = smoothstep(0.53, 0.61, fbm3g(vec3<f32>(uv * 2.9, 78.0)));
        return mix(vec3<f32>(base), vec3<f32>(1.30, 0.78, 0.52) * (base * 0.9 + 0.2), vein);
    }
    // Gold - bright glinting veins.
    if (mat == 21u) {
        let base = fbm3g(vec3<f32>(uv * 2.2, 79.0)) * 0.22 + 0.86;
        let vein = smoothstep(0.54, 0.61, fbm3g(vec3<f32>(uv * 3.1, 80.0)));
        let glint = max(0.0, (vnoise3(vec3<f32>(uv * 15.0, 81.0)) - 0.86) * 6.0) * vein;
        return mix(vec3<f32>(base), vec3<f32>(1.55, 1.22, 0.45), vein) + vec3<f32>(glint);
    }
    // Diamond - sparse cyan crystals with hard sparkle.
    if (mat == 22u) {
        let base = fbm3g(vec3<f32>(uv * 2.2, 83.0)) * 0.22 + 0.86;
        let crystal = smoothstep(0.70, 0.76, gnoise3(vec3<f32>(uv * 4.2, 84.0)));
        let sparkle = max(0.0, (vnoise3(vec3<f32>(uv * 18.0, 85.0)) - 0.84) * 7.0) * crystal;
        return mix(vec3<f32>(base), vec3<f32>(0.85, 1.45, 1.55), crystal) + vec3<f32>(sparkle);
    }
    // Lava - dark cooling crust plates over slowly pulsing glow cracks.
    if (mat == 16u) {
        let flow = gnoise3(vec3<f32>(uv * 1.1, camera.time * 0.06));
        let crack = ridge_line(flow, 0.09);
        let crust = fbm3g(vec3<f32>(uv * 2.3, 5.0)) * 0.25 + 0.45;
        let pulse = 0.85 + 0.15 * sin(camera.time * 0.8 + flow * 6.0);
        return mix(vec3<f32>(crust * 0.55), vec3<f32>(2.2, 1.35, 0.60) * pulse, crack);
    }
    // Cactus - vertical ribs with pale spine dots on the crests.
    if (mat == 32u) {
        let ribs = 0.76 + 0.24 * pow(0.5 + 0.5 * cos(uv.x * 12.6), 0.8);
        let spine = max(0.0, (vnoise3(vec3<f32>(uv.x * 6.3, uv.y * 3.2, 83.0)) - 0.86) * 6.0);
        return vec3<f32>(ribs + spine * 1.2, ribs + spine * 1.4, ribs * 0.95 + spine);
    }
    return vec3<f32>(1.0);
}

fn ambient_color() -> vec3<f32> {
    let s = sun_dir();
    let day_t = sun_intensity(s);
    let day = vec3<f32>(0.30, 0.42, 0.58);
    let night = vec3<f32>(0.04, 0.05, 0.10);
    return mix(night, day, day_t);
}

// Real Gerstner-style wave normals. Four waves with varied directions,
// wavelengths, amplitudes and steepness so the surface looks like genuine
// ocean — not a tiled sinusoid. We return the perturbed normal computed
// from the closed-form partial derivatives of the wave height field.
//
//   h_i(P,t) = A_i · cos(D_i · P · k_i - ω_i · t + φ_i)
//   ∂h_i/∂x = -A_i · k_i · D_i.x · sin(...)
//   ∂h_i/∂z = -A_i · k_i · D_i.z · sin(...)
//
// Then normal = normalize(vec3(-Σ∂h/∂x, 1, -Σ∂h/∂z)).
// The 4-wave Gerstner table, heights in VOXELS: (dir.x, dir.z, k=2π/λ, A, ω, phase).
// A proper little spectrum — one long swell, a secondary sea, and two chop
// waves — so the surface reads as traveling wavefronts, not random bobbing.
// λ = 26 / 13 / 7 / 3.5 voxels; amplitudes sum to ~0.108 voxels.
fn wave_param(i: i32) -> array<f32, 6> {
    if      (i == 0) { return array<f32, 6>(  0.97,  0.24, 0.242, 0.055, 0.90, 0.0); }
    else if (i == 1) { return array<f32, 6>(  0.83, -0.55, 0.483, 0.030, 1.35, 1.7); }
    else if (i == 2) { return array<f32, 6>( -0.40,  0.92, 0.898, 0.015, 1.95, 3.1); }
    else             { return array<f32, 6>( -0.90, -0.43, 1.795, 0.008, 2.80, 5.2); }
}

// Height + gradient of the wave field in one loop: returns (h, dh/dx, dh/dz),
// h in voxels around the resting surface.
fn water_field(xz: vec2<f32>, t: f32) -> vec3<f32> {
    var h = 0.0;
    var dx = 0.0;
    var dz = 0.0;
    for (var i: i32 = 0; i < 4; i = i + 1) {
        let w = wave_param(i);
        let phase = (xz.x * w[0] + xz.y * w[1]) * w[2] - t * w[4] + w[5];
        h = h + w[3] * cos(phase);
        let s = w[3] * w[2] * sin(phase);
        dx = dx - s * w[0];
        dz = dz - s * w[1];
    }
    return vec3<f32>(h, dx, dz);
}

// ---------- STYLIZED VOXEL WATER: one flat facet per cell -------------------
// ART DIRECTION (Marc, 2026-08-01): water is FACETED, not smooth. Every
// surface water voxel renders ONE horizontal plate at ONE quantized height with
// ONE flat normal, so the lake reads as a staircase of discrete cells that
// animate in steps - the same discretisation the rest of the world is built
// from - rather than as a continuous swell that happens to be made of voxels.
//
// This REPLACES the corner-connected bilinear/two-triangle surface. That
// surface existed to make neighbouring cells share exact corner heights so the
// lake was one watertight sheet; a faceted lake does not want to be one sheet,
// it wants to be plates. What it must still not have is HOLES, and it does not:
// where a neighbouring plate is higher, the ray enters this cell BELOW its own
// plate and the entry face itself is the hit (the step's riser). So the old
// pin rule, the step-down fold, the two-triangle split and their up-to-24
// neighbour probes per cell are all gone, and with them the separate near/far
// surface tiers - a centre-sampled facet was already what the far tier drew,
// so near and far now run the SAME code.
//
// Heights are quantized to WATER_WAVE_BANDS steps each side of the resting
// surface and the slope to WATER_SLOPE_STEP, so both the silhouette and the
// shading take a small number of discrete values per cell. Wave MOTION
// survives: the band a cell sits in changes as the wavefront passes, so the
// surface steps up and down in place.
const WATER_DETAIL_T: f32 = 24.0 * VOXELS_PER_METRE;   // beyond this, the wave field is skipped (flat rest plane)
// Beyond this, water really is a plain cube top. The 0.28-voxel drop from
// cube top (1.0) to rest surface (0.72) subtends under a pixel out here,
// and fog has saturated by 280 - while the old 96 cutoff put the drop in
// plain view: the water level visibly sank in a radius around the camera
// as it approached ("far water looks higher"). Grazing cost stays bounded:
// past 400 the first water cell stops the ray as a cube again.
const WATER_FAR_T: f32 = 100.0 * VOXELS_PER_METRE;
const WATER_BASE: f32 = 0.72;       // resting surface height inside the cell
const WATER_MIN_H: f32 = 0.02;      // floor (keeps the plate off the cell floor)
// |water_field| bound: the four amplitudes in `wave_param` sum to 0.108.
// Dividing by it turns the raw field into a normalised -1..1 wave phase, which
// is what the band quantiser wants; the stylized amplitude is then set
// independently below instead of being whatever the spectrum happened to give.
const WATER_FIELD_MAX: f32 = 0.108;
// Stylized half-range of the quantized surface, in voxels. 0.24 with 3 bands
// puts the step at 0.08 voxels - big enough that a step's riser is a visible
// sliver at close range, small enough that the surface stays inside its cell
// (0.72 +- 0.24 = 0.48..0.96, clear of both the cell floor and the cube top,
// which is what keeps the grazing fast-out and the interior-cell test valid).
const WATER_WAVE_AMP: f32 = 0.24;
const WATER_WAVE_BANDS: f32 = 3.0;  // steps each side of rest: 7 discrete heights
// Fraction of a band spent EASING across its boundary instead of sitting flat
// on it, FOR THE TONE LADDER ONLY. 0 is a pure staircase, 1 is no quantization.
//
// A TEMPORAL fix with no spatial cost, measured rather than guessed. A pure
// staircase pops a whole cell by one tone step the instant the wave carries it
// over a boundary, and `flicker_probe_rt_views` reads that plainly: the
// water_top view went from 0.135% strongly flickering pixels before this rework
// to 1.177% after, and disabling the tone ladder alone took it back to 0.262% -
// so the ladder was 78% of the strobing. Easing it reads 0.280%.
//
// It costs nothing in the look because what makes the surface read as faceted
// is the step BETWEEN NEIGHBOURING CELLS, and that is unchanged: two cells a
// band apart still differ by a full tone step with a hard edge at the cell
// boundary. Only a cell CROSSING a boundary changes gradually, over several
// frames instead of one, and 65% of cells still sit exactly on a level.
//
// THE HEIGHT IS NOT EASED, and that is the other half of the same measurement.
// Easing it too took the water_top view to a comparable 0.290% but made the
// GRAZING view worse, 0.161% -> 0.340%: from a low angle a crest's plate occludes the
// trough behind it, so a hard-quantized height holds that occlusion boundary
// still while a sliding one creeps it across pixels every frame. The height's
// quantization is load-bearing for grazing stability; the tone's was only ever
// load-bearing for the look, and the look does not need the temporal half of it.
const WATER_BAND_EASE: f32 = 0.35;
// Conservative bound on the surface's displacement from WATER_BASE, used by
// the grazing-ray fast-out. Equal to the amplitude BY CONSTRUCTION now (the
// quantiser cannot exceed its own top band), rather than a padded guess.
const WATER_WAVE_MAX: f32 = WATER_WAVE_AMP;
// Facet slope quantum. The wave field's own max slope is 0.0557 (the sum of
// A_i * k_i over `wave_param`), so the stylized slope is bounded by
// 0.0557 * WATER_WAVE_AMP / WATER_FIELD_MAX = 0.124, and a
// 0.06 step gives five tilts per axis (0, +-0.06, +-0.12): a small, countable
// set of facet orientations, which is what makes neighbouring cells read as
// separate flat faces instead of a smooth gradient.
const WATER_SLOPE_STEP: f32 = 0.06;
// Cells at or above this band foam at their crest. Band 2 of 3 rather than the
// top band alone: `band = round(t * 3)` puts band 3 at |t| >= 5/6, which the
// four-wave spectrum reaches on only a few percent of cells, so crest foam
// would be a rare speck rather than a feature of the wavefront.
const WATER_FOAM_BAND: f32 = 2.0;
// Beyond this the quarter-voxel foam stamp is sub-pixel, so it is faded out
// rather than left to alias. Inside WATER_DETAIL_T by construction: past that
// there is no wave field, hence no crest, hence nothing to foam.
const WATER_FOAM_T: f32 = 16.0 * VOXELS_PER_METRE;

struct WaterSubHit {
    hit: bool,
    t_hit: f32,
    normal: vec3<f32>,
    // The cell's QUANTIZED slope, carried to the deferred pass via the transp
    // record so cs_transparent reconstructs the same flat facet normal the
    // tracer used. Named for what it is now: there is no separate "rest"
    // gradient any more, because there is no per-pixel wave term to add on top.
    grad: vec2<f32>,
    // Wave band, -WATER_WAVE_BANDS..WATER_WAVE_BANDS. A FLOAT because the
    // staircase is eased across its boundaries in time (see WATER_BAND_EASE);
    // it lands exactly on an integer for most cells and slides between two for
    // the rest. Paired with the 4-bit mask of lateral neighbours that are not
    // water (bit 0 +x, 1 -x, 2 +z, 3 -z). Both are per-cell constants and both
    // drive foam.
    band: f32,
    shore: u32,
};

/// The one flat facet of a water cell: quantized height, quantized slope and
/// the wave band both came from.
///
/// ONE definition, called by the tracer, by `camera_in_water` and by the
/// still-image tests. A second copy of this arithmetic anywhere would let the
/// drawn surface and the surface the camera thinks it is under drift apart.
struct WaterFacet {
    h: f32,
    grad: vec2<f32>,
    band: f32,
};

/// Quantize a wave phase to the band ladder, EASING across each boundary.
///
/// Exactly `round(u)` over the flat core of a band, sliding continuously to the
/// midpoint at the boundary and picking up again on the other side - so the
/// result is continuous in `u` and still spends most of its range pinned to an
/// integer. The continuity is what stops a cell popping a whole step in one
/// frame as the wave carries it across; the flat core is what keeps
/// neighbouring cells landing on the same level and reading as one terrace.
fn water_band_quantize(u: f32) -> f32 {
    let k = round(u);
    let d = u - k;
    let core = 0.5 - WATER_BAND_EASE * 0.5;
    return k + sign(d) * smoothstep(core, 0.5, abs(d)) * 0.5;
}

fn water_facet(voxel: vec3<i32>, level_frac: f32, t_view: f32) -> WaterFacet {
    var o: WaterFacet;
    // Sampled at the CELL CENTRE, once. That single sample is the whole cell:
    // it is what makes the facet flat, and it is why every pixel of the cell
    // shades identically.
    let vc = vec2<f32>(f32(voxel.x) + 0.5, f32(voxel.z) + 0.5);
    var f = vec3<f32>(0.0);
    if (t_view <= WATER_DETAIL_T) {
        f = water_field(vc, camera.time);
    }
    // Normalised wave phase in -1..1, then the band. Centred on round() (not
    // floor) so the resting surface is a band of its own and the ladder is
    // symmetric.
    //
    // `band` is the EASED value and drives the tone; the HEIGHT takes its hard
    // integer band (`round` of the eased value returns exactly the band the
    // quantizer picked). See WATER_BAND_EASE for why the two differ.
    let t = clamp(f.x / WATER_FIELD_MAX, -1.0, 1.0);
    o.band = clamp(water_band_quantize(t * WATER_WAVE_BANDS), -WATER_WAVE_BANDS, WATER_WAVE_BANDS);
    let step_h = WATER_WAVE_AMP / WATER_WAVE_BANDS;
    o.h = clamp((WATER_BASE + round(o.band) * step_h) * level_frac, WATER_MIN_H, 1.0);
    // Slope on the same stylized scale as the height, then quantized to its own
    // ladder. Scaled by level_frac exactly as the height is, so a half-full
    // column's facet is half as steep and the two stay consistent.
    let g = f.yz * (WATER_WAVE_AMP / WATER_FIELD_MAX) * level_frac;
    o.grad = round(g / WATER_SLOPE_STEP) * WATER_SLOPE_STEP;
    return o;
}

/// The facet normal for a quantized slope. One place, so the tracer and the
/// deferred pass cannot disagree about which way a facet faces.
fn water_facet_normal(grad: vec2<f32>) -> vec3<f32> {
    return normalize(vec3<f32>(-grad.x, 1.0, -grad.y));
}

/// Pack the per-cell facet band + shore mask into the transp record's spare
/// word. The band is biased and scaled into a byte: 8 bits over a range of
/// 2*WATER_WAVE_BANDS is a resolution of 0.024 of a band, two orders finer than
/// the eased boundary it has to represent.
fn water_pack_facet(band: f32, shore: u32) -> u32 {
    let q = clamp((band + WATER_WAVE_BANDS) * (255.0 / (2.0 * WATER_WAVE_BANDS)), 0.0, 255.0);
    return u32(round(q)) | (shore << 8u);
}
fn water_unpack_band(code: u32) -> f32 {
    return f32(code & 0xFFu) * ((2.0 * WATER_WAVE_BANDS) / 255.0) - WATER_WAVE_BANDS;
}
fn water_unpack_shore(code: u32) -> u32 {
    return (code >> 8u) & 0xFu;
}

// Sub-voxel water surface for one cell the DDA landed in. `entry_n`/`t_entry`
// describe the cell's entry face, `t_exit` the exit crossing; `slot_v`/`bp`/
// `bi` are the DDA's current slot voxel and brick so neighbour probes can
// take the register-resident fast path. Misses (ray passes above the plate)
// fall through to the next DDA cell.
fn water_subvoxel(
    voxel: vec3<i32>, origin: vec3<f32>, dir: vec3<f32>, m: u32,
    entry_n: vec3<f32>, t_entry: f32, t_exit: f32,
    slot_v: vec3<i32>, bp: vec3<i32>, bi: i32,
) -> WaterSubHit {
    var out: WaterSubHit;
    out.hit = false;
    out.grad = vec2<f32>(0.0);
    out.band = 0.0;
    out.shore = 0u;
    // Interior cell (more water above): a plain cube. Its exposed faces are
    // vertical water walls / undersides.
    if (is_water_mat(neighbor_material(voxel, slot_v, bp, bi, vec3<i32>(0, 1, 0)))) {
        out.hit = true;
        out.t_hit = t_entry;
        out.normal = entry_n;
        return out;
    }

    let level_frac = f32(m - MAT_WATER_L1 + 1u) * 0.125;
    let fc = water_facet(voxel, level_frac, t_entry);
    let vmin = vec3<f32>(f32(voxel.x), f32(voxel.y), f32(voxel.z));
    let y_in = origin.y + dir.y * t_entry - vmin.y;
    let y_out = origin.y + dir.y * t_exit - vmin.y;
    // Grazing fast-out: y is monotone along the ray, so its minimum over the
    // cell is at an endpoint, and the plate can never exceed the top band.
    if (min(y_in, y_out) > WATER_BASE + WATER_WAVE_MAX) {
        return out;
    }

    let p0 = origin + dir * t_entry - vmin;
    if (p0.y <= fc.h + 1e-4) {
        // Entered BELOW this cell's plate. The entry face IS the surface here,
        // and this branch is what makes a staircase of independent plates
        // watertight: without it a ray would slip through the gap between two
        // plates at different heights.
        out.hit = true;
        out.t_hit = t_entry;
        // A LATERAL entry from a cell that ALSO holds water is the riser of an
        // internal step, and it has to be shaded as the surface. Shading it
        // with its true vertical face normal painted hard navy cracks and
        // bright grazing slivers along every band boundary - seen on the first
        // stills of this rework, and the same artifact (with the same fix) the
        // old far tier already carried. A lateral neighbour that is NOT water
        // is a genuine wall (a pool's edge, a waterfall face) and keeps its
        // face normal.
        if (abs(entry_n.y) < 0.5
            && is_water_mat(neighbor_material(voxel, slot_v, bp, bi, vec3<i32>(entry_n)))) {
            out.normal = water_facet_normal(fc.grad);
        } else {
            out.normal = entry_n;
        }
    } else if (dir.y < -1e-7) {
        let s = (fc.h - p0.y) / dir.y;
        if (s > 0.0 && t_entry + s < t_exit) {
            out.hit = true;
            out.t_hit = t_entry + s;
            out.normal = water_facet_normal(fc.grad);
        }
    }
    if (!out.hit) {
        return out;
    }
    out.grad = fc.grad;
    out.band = fc.band;
    // Shore probes LAST and only when foam can be seen: four neighbour
    // lookups, against the up-to-24 the corner surface paid on every cell.
    //
    // The test is "the neighbour is NOT WATER", i.e. this cell is on the EDGE
    // OF THE WATER BODY - deliberately wider than "adjacent to solid". A lake
    // whose rim sits BELOW its surface (the common case: water at y on ground
    // at y-1) has AIR at every lateral neighbour, so a solid-only test foams
    // nothing at all on it, which is what the first stills showed on the
    // terrace lab. Both cases read to a player as the shore.
    if (t_entry <= WATER_FOAM_T) {
        var shore = 0u;
        for (var k = 0u; k < 4u; k = k + 1u) {
            var d = vec3<i32>(1, 0, 0);
            if (k == 1u) { d = vec3<i32>(-1, 0, 0); }
            else if (k == 2u) { d = vec3<i32>(0, 0, 1); }
            else if (k == 3u) { d = vec3<i32>(0, 0, -1); }
            if (!is_water_mat(neighbor_material(voxel, slot_v, bp, bi, d))) {
                shore = shore | (1u << k);
            }
        }
        out.shore = shore;
    }
    return out;
}

// Face normal + entry distance for the cell a ray just stepped into. `last_axis`
// is the axis whose plane we crossed (0/1/2); -1 means we began inside the cell,
// so fall back to the dominant entry plane from `tmin3`. Extracted so every LOD
// branch in the tracers computes the hit face the exact same way.
struct EntryNormal { n: vec3<f32>, t_hit: f32 };

fn entry_normal_and_t(
    last_axis: i32,
    step: vec3<i32>,
    t_max: vec3<f32>,
    t_delta: vec3<f32>,
    t_enter: f32,
    tmin3: vec3<f32>,
) -> EntryNormal {
    var r: EntryNormal;
    r.n = vec3<f32>(0.0);
    r.t_hit = 0.0;
    if (last_axis == 0)      { r.n.x = -f32(step.x); r.t_hit = t_max.x - t_delta.x; }
    else if (last_axis == 1) { r.n.y = -f32(step.y); r.t_hit = t_max.y - t_delta.y; }
    else if (last_axis == 2) { r.n.z = -f32(step.z); r.t_hit = t_max.z - t_delta.z; }
    else {
        r.t_hit = t_enter;
        if      (tmin3.x >= tmin3.y && tmin3.x >= tmin3.z) { r.n.x = -f32(step.x); }
        else if (tmin3.y >= tmin3.z)                       { r.n.y = -f32(step.y); }
        else                                               { r.n.z = -f32(step.z); }
    }
    return r;
}

// Variant of `trace` that treats every water voxel as empty. Used to find
// what lies *beneath* a water surface for refraction-style transparency.
// Shared DDA setup state produced by dda_init() and unpacked by every tracer.
// `valid` is false when the ray misses the world AABB entirely.
struct DdaInit {
    valid: bool,
    ro: vec3<i32>,
    org: vec3<f32>,
    inv_dir: vec3<f32>,
    step: vec3<i32>,
    t_delta: vec3<f32>,
    tmin3: vec3<f32>,
    t_enter: f32,
    voxel: vec3<i32>,
    t_max: vec3<f32>,
    slot_v: vec3<i32>,
};

// Origin-rebased traversal setup, shared by trace / trace_no_water / trace_any.
// The integer voxel grid stays in ABSOLUTE world coords (so cell alignment and
// the toroidal slot lookup are unchanged), but every FLOAT computation is done
// relative to the window corner `ro`. At large world coords `f32(voxel)-origin`
// catastrophically cancels (the "sky through hills" bug); `f32(voxel - ro) -
// (origin - ro)` keeps both operands small and exact. The ray parameter t is a
// distance, unchanged by the rebase.
fn dda_init(origin: vec3<f32>, dir: vec3<f32>) -> DdaInit {
    var d: DdaInit;
    d.valid = false;
    let ro = camera.world_origin;
    let org = origin - vec3<f32>(ro);
    let dims = vec3<f32>(f32(WORLD_VOXELS_X), f32(WORLD_VOXELS_Y), f32(WORLD_VOXELS_Z));
    let inv_dir = vec3<f32>(safe_inv(dir.x), safe_inv(dir.y), safe_inv(dir.z));
    let t0 = (vec3<f32>(0.0) - org) * inv_dir;
    let t1 = (dims - org) * inv_dir;
    let tmin3 = min(t0, t1);
    let tmax3 = max(t0, t1);
    let t_enter = max(max(tmin3.x, tmin3.y), max(tmin3.z, 0.0));
    let t_exit = min(min(tmax3.x, tmax3.y), tmax3.z);
    if (t_enter >= t_exit || t_exit < 0.0) { return d; }

    let bias = 1e-3;
    var p = org + dir * (t_enter + bias);
    p = clamp(p, vec3<f32>(0.01), dims - vec3<f32>(0.01));
    let step = vec3<i32>(sign(dir));
    let t_delta = abs(inv_dir);

    let voxel = vec3<i32>(floor(p)) + ro;
    let vl0 = voxel - ro;
    var t_max: vec3<f32>;
    if (step.x > 0) { t_max.x = (f32(vl0.x + 1) - org.x) * inv_dir.x; } else { t_max.x = (f32(vl0.x) - org.x) * inv_dir.x; }
    if (step.y > 0) { t_max.y = (f32(vl0.y + 1) - org.y) * inv_dir.y; } else { t_max.y = (f32(vl0.y) - org.y) * inv_dir.y; }
    if (step.z > 0) { t_max.z = (f32(vl0.z + 1) - org.z) * inv_dir.z; } else { t_max.z = (f32(vl0.z) - org.z) * inv_dir.z; }

    d.valid = true;
    d.ro = ro;
    d.org = org;
    d.inv_dir = inv_dir;
    d.step = step;
    d.t_delta = t_delta;
    d.tmin3 = tmin3;
    d.t_enter = t_enter;
    d.voxel = voxel;
    d.t_max = t_max;
    // Slot voxel tracked incrementally (avoids the two per-step pos_mod folds —
    // checklist #10); skip_to_cell resyncs it after a jump, dda_step folds it.
    d.slot_v = world_to_slot_voxel(voxel);
    return d;
}

// One DDA cell advance: step the axis with the nearest t_max, fold slot_v
// toroidally on x/z, and record which axis we crossed. Identical inner loop for
// all three tracers (trace_no_water ignores the t_cur it writes).
fn dda_step(
    voxel: ptr<function, vec3<i32>>,
    slot_v: ptr<function, vec3<i32>>,
    t_max: ptr<function, vec3<f32>>,
    t_cur: ptr<function, f32>,
    last_axis: ptr<function, i32>,
    step: vec3<i32>,
    t_delta: vec3<f32>,
) {
    if ((*t_max).x < (*t_max).y && (*t_max).x < (*t_max).z) {
        *t_cur = (*t_max).x;
        (*voxel).x = (*voxel).x + step.x;
        (*slot_v).x = (*slot_v).x + step.x;
        if ((*slot_v).x >= WORLD_VOXELS_X) { (*slot_v).x = (*slot_v).x - WORLD_VOXELS_X; }
        else if ((*slot_v).x < 0) { (*slot_v).x = (*slot_v).x + WORLD_VOXELS_X; }
        (*t_max).x = (*t_max).x + t_delta.x;
        *last_axis = 0;
    } else if ((*t_max).y < (*t_max).z) {
        *t_cur = (*t_max).y;
        (*voxel).y = (*voxel).y + step.y;
        (*slot_v).y = (*slot_v).y + step.y;
        (*t_max).y = (*t_max).y + t_delta.y;
        *last_axis = 1;
    } else {
        *t_cur = (*t_max).z;
        (*voxel).z = (*voxel).z + step.z;
        (*slot_v).z = (*slot_v).z + step.z;
        if ((*slot_v).z >= WORLD_VOXELS_Z) { (*slot_v).z = (*slot_v).z - WORLD_VOXELS_Z; }
        else if ((*slot_v).z < 0) { (*slot_v).z = (*slot_v).z + WORLD_VOXELS_Z; }
        (*t_max).z = (*t_max).z + t_delta.z;
        *last_axis = 2;
    }
}

fn trace_no_water(origin: vec3<f32>, dir: vec3<f32>, t_cap: f32) -> Hit {
    var out: Hit;
    out.hit = false;
    out.mat = 0u;
    out.normal = vec3<f32>(0.0);
    out.voxel = vec3<i32>(0);
    out.last_axis = -1;
    out.t_hit = 0.0;
    out.tint = vec3<f32>(1.0);

    let init = dda_init(origin, dir);
    if (!init.valid) { return out; }
    let ro = init.ro;
    let org = init.org;
    let inv_dir = init.inv_dir;
    let step = init.step;
    let t_delta = init.t_delta;
    let tmin3 = init.tmin3;
    let t_enter = init.t_enter;
    var voxel = init.voxel;
    var t_max = init.t_max;
    var slot_v = init.slot_v;
    var last_axis: i32 = -1;
    var t_cur: f32 = t_enter;
    for (var s: i32 = 0; s < 1024; s = s + 1) {
        if (t_cur > t_cap) { return out; }
        let rel = voxel - camera.world_origin;
        if (rel.x < 0 || rel.x >= WORLD_VOXELS_X
         || rel.y < 0 || rel.y >= WORLD_VOXELS_Y
         || rel.z < 0 || rel.z >= WORLD_VOXELS_Z) { return out; }

        // slot_v is maintained incrementally (see step + skip_to_cell).
        let bp = slot_v >> vec3<u32>(2u);
        let tp = slot_v >> vec3<u32>(4u);
        let cp = slot_v >> vec3<u32>(6u);
        // Nested hierarchy: skip the coarsest empty cell (L4 → chunk → tile → brick).
        let l4p = slot_v >> vec3<u32>(8u);
        let l4i = world_l4_idx(l4p.x, l4p.y, l4p.z);
        if (l4_cell_empty(l4i)) {
            skip_to_cell(256, &voxel, &t_max, ro, org, dir, inv_dir, step, &last_axis, &slot_v);
            continue;
        }
        let chunk_lin = (cp.x & 3) + (cp.z & 3) * 4 + (cp.y & 3) * 16;
        if (!l4_has_child(l4i, chunk_lin)) {
            skip_to_cell(64, &voxel, &t_max, ro, org, dir, inv_dir, step, &last_axis, &slot_v);
            continue;
        }
        let ci = world_chunk_idx(cp.x, cp.y, cp.z);
        let tile_lin = (tp.x & 3) + (tp.z & 3) * 4 + (tp.y & 3) * 16;
        if (!chunk_has_child(ci, tile_lin)) {
            skip_to_cell(16, &voxel, &t_max, ro, org, dir, inv_dir, step, &last_axis, &slot_v);
            continue;
        }
        let ti = world_tile_idx(tp.x, tp.y, tp.z);
        let brick_lin = (bp.x & 3) + (bp.z & 3) * 4 + (bp.y & 3) * 16;
        if (!tile_has_child(ti, brick_lin)) {
            skip_to_cell(4, &voxel, &t_max, ro, org, dir, inv_dir, step, &last_axis, &slot_v);
            continue;
        }
        let bi = world_brick_idx(bp.x, bp.y, bp.z);
        let local = slot_v - bp * BRICK_DIM;
        let vi = brick_voxel_idx(local.x, local.y, local.z);
        if (brick_voxel_solid(bi, vi)) {
            let m = brick_voxel_material(bi, vi);
            if (!is_transparent_mat(m)) {
                let en = entry_normal_and_t(last_axis, step, t_max, t_delta, t_enter, tmin3);
                let n = en.n;
                let t_hit = en.t_hit;
                out.hit = true;
                out.mat = m;
                out.normal = n;
                out.voxel = voxel;
                out.last_axis = last_axis_after_entry(last_axis, tmin3);
                out.t_hit = t_hit;
                return out;
            }
            // Water cell — fall through to the regular step so we keep going.
        }

        dda_step(&voxel, &slot_v, &t_max, &t_cur, &last_axis, step, t_delta);
    }
    return out;
}

// Rec.709-ish luma weights for the foam highlight ramp. Named because two
// separate places below want "how bright is this" and a second spelling of the
// weights is a second thing to keep in step.
const WATER_SHADOW_TONE: f32 = 0.42;   // multiplier on water sitting in full shadow
const WATER_FACET_CONTRAST: f32 = 0.22; // tone spread from trough band to crest band
const WATER_SKY_MAX: f32 = 0.45;       // ceiling on the grazing Fresnel blend toward sky
const WATER_FOAM_SHORE_SUB: i32 = 2;   // shore-foam band width, in quarter-voxel texels
// Eighths of crest cells that actually break into foam. NOT every crest does,
// and the reason it has to be well under 8 is a grazing-angle effect that only
// showed up when it was measured: a crest cell's plate stands 0.16 voxels
// proud, and a shallow ray stops at the FIRST plate it dips below, so crests
// occlude the troughs behind them and take far more of the screen than their
// footprint. Band >= 2 is 13.9% of cells by area (Monte Carlo over the four
// waves in `wave_param`) but covered 24% of the terrace-ramp crop, five times
// its share. Foaming half of them puts whitecaps back at a plausible density
// from a low angle without emptying the top-down views.
const WATER_FOAM_CREST_ODDS: u32 = 4u;

/// Cheap integer hash of a water cell. Large odd multipliers so adjacent cells
/// land on unrelated values instead of walking the grid. Drives BOTH which foam
/// shape a cell stamps and how wide its shore band is, from one value, so the
/// two cannot disagree about which cell they are decorating.
fn foam_hash(cell: vec3<i32>) -> u32 {
    return (u32(cell.x) * 73856093u) ^ (u32(cell.z) * 19349663u) ^ (u32(cell.y) * 83492791u);
}

/// One texel of a foam stamp. `SPR_FOAM_*` is a 4x4 grid of sixteen 4x4 shapes
/// (see src/sprites.rs); `local` is the position inside the water cell in 0..1
/// and `h` (from `foam_hash`) picks the shape, so neighbouring cells stamp
/// DIFFERENT shapes and a shoreline gets a ragged edge instead of a repeated
/// motif.
///
/// Quarter-voxel texels on purpose: the foam has to be visibly made of the same
/// grid the world is, and a 16x16 stamp inside a voxel would be noise at any
/// distance the water is actually seen from.
fn foam_texel(sprite: u32, local: vec2<f32>, h: u32) -> u32 {
    let sub = clamp(vec2<i32>(floor(local * 4.0)), vec2<i32>(0), vec2<i32>(3));
    let tile = h & 15u;
    let tx = (tile & 3u) * 4u + u32(sub.x);
    let ty = (tile >> 2u) * 4u + u32(sub.y);
    return sprite_texel(sprite, tx, ty);
}

// STYLIZED water surface (Marc, 2026-08-01). It reads as LIT or SHADOWED, with
// discrete flat facets and hard-edged foam - not as a mirror.
//
// The reflection is gone entirely, both the per-voxel cache and the per-pixel
// mirror trace it fell back to. That is the art direction, and it is also why
// this function no longer touches the reflection history region of transp_buf,
// runs no secondary trace of its own, and needs no per-pixel jitter.
//
// What carries the look instead:
//   - LIT vs SHADOWED, from the per-voxel sun visibility the light field
//     already stores, through a HARD-ish step rather than a smooth falloff.
//   - The per-cell facet band as a flat tone step, so neighbouring cells are
//     visibly different flat faces and the wave reads as it travels.
//   - Depth absorption and the water-body tint, KEPT unchanged: they are what
//     makes this read as water rather than as blue paint.
//   - A Fresnel blend toward the sky at grazing angles, KEPT. See the note at
//     the blend itself for why.
//   - Hard-edged quantized foam at crests and shorelines.
//
// `facet_code` is the transp record's spare word: this cell's wave band and its
// solid-neighbour mask, packed by `water_pack_facet`.
fn shade_water_top(hit: Hit, origin: vec3<f32>, dir: vec3<f32>, facet_code: u32) -> vec3<f32> {
    let p_hit = origin + dir * hit.t_hit;
    // Secondary hits report their shadow/AO into this and nothing reads it: the
    // G-buffer describes the PRIMARY surface, so a refracted hit under the water
    // must not overwrite it.
    var scratch_light = vec2<f32>(0.0);
    // ONE flat normal for the whole cell (water_facet_normal of the quantized
    // slope), or an axis face on a riser. Nothing here perturbs it per pixel.
    let n = hit.normal;
    let s = sun_dir();
    let sc = sun_color(s);
    let s_int = sun_intensity(s);
    // The water cell this surface belongs to. Stepping down off the plate
    // rather than using floor(p_hit) directly: a plate can sit flush with its
    // cell's top face, where floor() names the air voxel in front.
    let cell = vec3<i32>(floor(vec3<f32>(p_hit.x, p_hit.y - 0.02, p_hit.z)));

    // ---- LIT or SHADOWED ------------------------------------------------
    // Straight from the per-voxel light field: sun visibility is a stored
    // property of the air cell above the water, so a whole plate shares one
    // value and the shadow edge lands on cell boundaries like everything else
    // in this look. Sampled about world +Y, not the facet normal, because the
    // record that matters is the one directly over the cell.
    //
    // There is NO fallback. This used to trace one shadow ray from just above the
    // cell's top face whenever the field had no record, which made water the last
    // surface in the frame carrying a second, disagreeing shadow mechanism. The
    // field answers 100.000% of water pixels (`LiveFrameRig::coverage`), and 0.0
    // for the rest is the designed term: see the same note in `shade`.
    let vlf = voxlight_sample(p_hit, vec3<f32>(0.0, 1.0, 0.0));
    if (PROF_VLF_COVERAGE > 0.5) {
        return select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vlf.valid);
    }
    let sun_vis = vlf.sun;
    // USED DIRECTLY, exactly as terrain uses it, and that is the fix.
    //
    // `sun_vis` is a CONTINUOUS fraction of the sun disc. Terrain multiplies its
    // direct term by it and comes out smooth; water pushed it through
    // `smoothstep(0.41, 0.59, sun_vis)` first - a band 45/255 wide - which
    // re-quantized the very gradient the light field exists to produce. Two
    // voxels either side of a shadow edge landed on opposite ends of that band,
    // so water read STEPPED where terrain right beside it read smooth, and any
    // wobble in the stored value became a full lit/shadowed flip (round G of
    // docs/rt/BASELINE-per-voxel-lighting.md traced the water flicker report to
    // exactly this narrowness).
    //
    // The STYLIZATION is not in this term and never was: the faceted look comes
    // from the quantized plate height, the quantized per-cell normal, the
    // per-cell tone ladder and the hard-edged foam - all untouched. Only the
    // shadow ramp changes, which is what was asked for.
    //
    // Multiplied by the day/night curve so night water is not "lit".
    let lit = sun_vis * s_int;

    // ---- refraction: primary ray bent into the water, trace through it ----
    // Snell's law via WGSL `refract`. eta = n_air / n_water ~ 1/1.33.
    let eta = 1.0 / 1.33;
    var refr_dir = refract(dir, n, eta);
    // Total internal reflection would return zero; fall back to dir.
    if (length(refr_dir) < 0.01) { refr_dir = dir; }
    let refr_origin = p_hit + dir * 0.001; // step inside the water column
    // Refraction routes through the SOFTWARE DDA even in RT builds: these rays
    // hit the bed within a few voxels, and a hardware ray query's fixed setup
    // (init + traversal state + candidate handshake) never amortizes on rays
    // that short - measured as RT transp trailing software by ~0.9 ms on
    // water-close.
    var under_col: vec3<f32>;
    var depth = 4.0;
    if (PROF_TRANSP_NO_REFR > 0.5) {
        // Cost-split probe: whole refraction component off.
        under_col = vec3<f32>(0.05, 0.15, 0.20);
    } else {
        let under = trace_no_water(refr_origin, refr_dir, SECONDARY_MAX_T);
        if (under.hit) {
            if (PROF_TRANSP_REFR_FLATSHADE > 0.5) {
                // Cost-split probe: trace kept, hit shading replaced by a
                // palette read.
                under_col = palette[under.mat].rgb;
            } else {
                // PROVEN constant: the refracted hit lies UNDER the surface and water
                // occludes shadow rays (shadow_voxel_occludes falls through to true),
                // so its sun-shadow ray always terminates in the water column above -
                // shadow_term is 0 by construction. Feed the constant through the
                // reuse path instead of tracing a per-pixel ray for a known answer.
                // Flat AO for the refracted hit (cost-split PROVEN: traced
                // AO was 2.55-2.66 ms of the transp pass - the single
                // largest water cost - while absorption + tint swamp its
                // contribution. Masked pixel diff vs traced AO from above
                // the surface: mean 1.3/255, 0.018% of pixels > 10/255;
                // identical below. Look signed off 2026-07-23). The shadow
                // term stays the proven constant 0.
                under_col = shade(under, refr_origin, refr_dir,
                                  KnownLight(0.0, 0.4, true), vec3<f32>(0.0), &scratch_light);
            }
        } else {
            under_col = sky(refr_dir) * 0.6;
        }
        // Beer-Lambert absorption — red and green are eaten faster than blue.
        depth = max(0.0, under.t_hit);
    }
    let absorb = vec3<f32>(0.55, 0.25, 0.10); // per-unit-distance attenuation
    let transmittance = exp(-absorb * depth);
    // Water tint modulated by ambient + a bit of sun colour, so the water
    // body actually goes dark at night instead of staying daytime-blue.
    let tint_base = vec3<f32>(0.10, 0.32, 0.42);
    let water_tint = tint_base * (ambient_color() * 1.6 + sc * 0.20);
    let refr_col = under_col * transmittance + water_tint * (1.0 - transmittance.x);

    // ---- the flat facet tone --------------------------------------------
    // A cell's wave band as a brightness step. THIS is what makes the surface
    // read as faceted: the quantized HEIGHT alone moves the silhouette by
    // 0.08 voxels, which is a sliver at any normal viewing distance, and the
    // quantized SLOPE only tilts the facet a few degrees. A tone ladder tied
    // to the same band turns each cell into a visibly distinct flat face, and
    // because it is driven by the band and not by the height it steps in
    // lockstep with the geometry rather than fighting it.
    let band = water_unpack_band(facet_code) / WATER_WAVE_BANDS;
    let facet_tone = 1.0 + WATER_FACET_CONTRAST * band;

    // ---- sun glint on the facet -----------------------------------------
    // Broad (pow 32, not 256) and reusing `sun_vis` rather than tracing. A
    // razor highlight over quantized facets pops a cell fully on or fully off
    // as the wave steps it into the next band, which is strobing; a broad lobe
    // moves a few percent per step.
    let hv = normalize(s - dir);
    let spec = pow(max(0.0, dot(n, hv)), 32.0);
    let glint = sc * spec * lit * 0.55;

    var col = refr_col * facet_tone * mix(WATER_SHADOW_TONE, 1.0, lit) + glint;

    // ---- Fresnel toward the sky -----------------------------------------
    // KEPT, and this was the judgement call the brief asked for. Without it a
    // lake is one flat blue field from the shore to the horizon: the facet
    // ladder gives it cell-scale texture but nothing at all changes with view
    // angle, so it reads as painted, and the far half of a big lake reads as a
    // solid rectangle. The blend is against `fog_atmospheric`, NOT `sky` -
    // the same emitter-free sky the fog uses - so the sun disc and the stars
    // cannot pop when a facet steps into the mirror direction, which is exactly
    // the strobing a quantized normal invites.
    let cos_theta = clamp(dot(-dir, n), 0.0, 1.0);
    let f0 = 0.02;
    // CAPPED, not the raw Schlick curve. Uncapped, fresnel runs to 1.0 at true
    // graze and the far half of any lake becomes a sheet of horizon sky: the
    // terrace lab's low camera came back milky grey-white with every facet
    // washed out of it. The cap keeps the view-angle gradient that stops the
    // surface reading as paint while leaving the water's own colour in charge.
    let fresnel = min(f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0), WATER_SKY_MAX);
    let sky_col = fog_atmospheric(reflect(dir, n)) * mix(0.55, 1.0, lit);
    col = mix(col, sky_col, fresnel);

    // Deep-pit floor: at a grazing angle over a deep hole the refraction is
    // fully absorbed, so the surface would read near-black. Floor it with the
    // ambient water-body blue so deep water is dark BLUE, not black.
    col = max(col, water_tint * 0.55 * mix(WATER_SHADOW_TONE, 1.0, lit));

    // ---- stylized foam ---------------------------------------------------
    // Hard-edged and quantized, at two places a stylized sea has white water:
    // the top of the wave range, and where it meets solid ground. Both are
    // PER-CELL decisions (the cell's band; the cell's solid neighbours), so
    // foam is locked to the voxel grid like everything else here; the authored
    // stamp only carves each cell's quarter-voxel silhouette so a run of foam
    // cells has a ragged edge instead of a drawn rectangle.
    //
    // No soft ramp anywhere: `foam` is 0, or it is one of two hard levels.
    var foam = 0.0;
    let foam_fade = 1.0 - smoothstep(WATER_FOAM_T * 0.75, WATER_FOAM_T, hit.t_hit);
    if (foam_fade > 0.0 && n.y > 0.5) {
        let local = p_hit.xz - floor(p_hit.xz);
        let shore = water_unpack_shore(facet_code);
        // Distance from the shore edge in quarter-voxel texels, taking the
        // NEAREST solid side. Quantized to the same 4-texel grid the stamp is
        // drawn on, so the band edge is a texel boundary and not a curve.
        let sub = clamp(vec2<i32>(floor(local * 4.0)), vec2<i32>(0), vec2<i32>(3));
        let fh = foam_hash(cell);
        // Band width VARIES PER CELL over 0, 1 or 2 quarter-voxel texels, with
        // a quarter of shore cells drawing NONE. A constant width made a pool's
        // rim a uniform stripe that read as a drawn outline rather than as
        // surf - plainly so on the terrace lab, where the camera sits a couple
        // of voxels off the water - and merely varying 1 vs 2 only softened it:
        // an unbroken band of any width is still a band. Dropping whole cells
        // is what breaks the rim into surf.
        let bw = min(i32((fh >> 6u) & 3u), WATER_FOAM_SHORE_SUB);
        var near_shore = false;
        if ((shore & 1u) != 0u && sub.x >= 4 - bw) { near_shore = true; }
        if ((shore & 2u) != 0u && sub.x < bw) { near_shore = true; }
        if ((shore & 4u) != 0u && sub.y >= 4 - bw) { near_shore = true; }
        if ((shore & 8u) != 0u && sub.y < bw) { near_shore = true; }
        var t = 0u;
        if (near_shore) {
            t = foam_texel(SPR_FOAM_SHORE, local, fh);
        } else if (round(water_unpack_band(facet_code)) >= WATER_FOAM_BAND
                   && (fh >> 10u) % 8u < WATER_FOAM_CREST_ODDS) {
            t = foam_texel(SPR_FOAM_CREST, local, fh);
        }
        // Two hard levels: body and highlight. Nothing between them.
        foam = select(select(0.0, 0.72, t == 1u), 1.0, t == 3u) * foam_fade;
    }
    // Foam colour dims at night - at dawn/dusk it picks up the warm sun tint,
    // at noon it is near-white - and, like the surface under it, drops to the
    // shadow tone when the cell is not in sun.
    //
    // The ambient term is DESATURATED before it is used. Ambient is a strongly
    // blue sky colour, and white water is white: foam that reads as "a lighter
    // patch of the same blue" is foam that does not register as foam.
    //
    // Scoped honestly, because it was measured: at NOON this changes almost
    // nothing, since foam is HDR there and the tonemapper saturates it toward
    // white with or without the desaturation (the brightest 5% of a shoreline
    // crop reads 13.9 saturation with it against 10.4 without). It earns its
    // place in the SHADOWED and night ranges, where the tone multiplier keeps
    // foam well below saturation and the blue cast is plainly visible.
    let amb = ambient_color();
    let amb_grey = vec3<f32>(dot(amb, vec3<f32>(0.2126, 0.7152, 0.0722)));
    let foam_col = (mix(amb, amb_grey, 0.75) * 2.2 + sc * 0.55)
        * mix(WATER_SHADOW_TONE, 1.0, lit);
    col = mix(col, foam_col, foam);

    let fog_t = fog_amount(hit.t_hit);
    return mix(col, fog_atmospheric(dir), fog_t);
}

// Terrain materials whose TOP faces cross-fade into each other at shared
// edges (sand beaches into grass, snow lines into stone, ...).
fn is_blend_mat(m: u32) -> bool {
    return m == 1u || m == 2u || m == 3u || m == 4u || m == 15u;
}

const BLEND_T: f32 = 16.0 * VOXELS_PER_METRE;   // beyond this an edge fade is sub-pixel
const BLEND_W: f32 = 0.42;   // fade starts this far from the shared edge

// Connected-texture cross-fade for terrain top faces: bilinear-blend the
// PALETTE colours of the two nearest in-plane neighbours (and their
// diagonal) when they hold a different blend-class material at the same
// height, then apply the OWN material's texture pattern once. Both sides
// of a shared edge compute the same 50/50 colour split exactly at the
// edge, so the fade is seamless. Deliberately colour-only: any
// per-neighbour texture evaluation gets loop-unrolled by the driver into
// N inline copies of material_texture, and N copies of the noise stack
// put libnvidia-gpucomp into minutes of pipeline compilation (measured
// via live backtrace).
fn blended_palette(p_hit: vec3<f32>, voxel: vec3<i32>, m: u32) -> vec3<f32> {
    let own = palette[m].rgb;
    let local = fract(p_hit.xz); // (x, z) in-face position
    let sx = select(-1, 1, local.x > 0.5);
    let sz = select(-1, 1, local.y > 0.5);
    let dx = select(local.x, 1.0 - local.x, local.x > 0.5);
    let dz = select(local.y, 1.0 - local.y, local.y > 0.5);
    let ax = smoothstep(BLEND_W, 0.0, dx) * 0.5;
    let az = smoothstep(BLEND_W, 0.0, dz) * 0.5;
    if (ax <= 0.0 && az <= 0.0) { return own; }
    var cx = own;
    var cz = own;
    var cd = own;
    if (ax > 0.0) {
        let nm = voxel_material_at(voxel + vec3<i32>(sx, 0, 0));
        if (is_blend_mat(nm)) { cx = palette[nm].rgb; }
    }
    if (az > 0.0) {
        let nm = voxel_material_at(voxel + vec3<i32>(0, 0, sz));
        if (is_blend_mat(nm)) { cz = palette[nm].rgb; }
    }
    if (ax > 0.0 && az > 0.0) {
        let nm = voxel_material_at(voxel + vec3<i32>(sx, 0, sz));
        if (is_blend_mat(nm)) { cd = palette[nm].rgb; }
    }
    return own * (1.0 - ax) * (1.0 - az)
         + cx * ax * (1.0 - az)
         + cz * (1.0 - ax) * az
         + cd * ax * az;
}

// Beyond this the one-bounce indirect (RT variant) is faded out: its contribution
// is small at distance and fog covers it, and it is the priciest per-pixel term.
// Pulled in from 120 so fewer pixels pay for GI (perf).
const GI_MAX_T: f32 = 22.5 * VOXELS_PER_METRE;

// Shader-pure grass turf: micro-normal/sheen terms fade out by here (the
// blade-scale detail is sub-pixel long before this).
const GRASS_SHADE_T: f32 = 24.0 * VOXELS_PER_METRE;

// A sun-visibility and AO pair a CALLER can prove, so `shade` neither traces nor
// samples for it.
//
// This is what is left of the old four-flag reuse protocol once the screen-space
// reprojection cache is gone. It has exactly one live user and it is not a
// cache: the refracted hit under a water surface is PROVEN to be in shadow
// (water occludes shadow rays, so its sun ray always terminates in the column
// above) and is given flat AO by a measured decision, so re-deriving either
// would be paying for a known answer.
struct KnownLight {
    sun: f32,
    ao: f32,
    known: bool,
};

fn vl_unknown() -> KnownLight {
    return KnownLight(0.0, 1.0, false);
}

/// Shade one opaque hit.
///
/// Sun visibility and AO come from ONE place and there is no second one: the
/// per-voxel light field. `known` overrides it for the one caller that can PROVE
/// the answer (the refracted hit below a water surface), which is a constant
/// rather than a rival derivation.
fn shade(
    hit: Hit, origin: vec3<f32>, dir: vec3<f32>,
    known: KnownLight, indirect: vec3<f32>,
    out_light: ptr<function, vec2<f32>>,
) -> vec3<f32> {
    let p_hit = origin + dir * hit.t_hit;
    // Terrain top faces near the camera cross-fade their palette colour into
    // differing blend-class neighbours (connected textures); the texture
    // pattern is evaluated exactly once either way.
    var pal = palette[hit.mat].rgb;
    if (hit.normal.y > 0.5 && is_blend_mat(hit.mat) && hit.t_hit < BLEND_T) {
        pal = blended_palette(p_hit, hit.voxel, hit.mat);
    }
    let tex = material_texture(p_hit, hit.normal, hit.mat);
    // hit.tint carries the sub-voxel colour (leaf shade, blade gradient,
    // petal/stem); (1,1,1) for plain cube hits.
    var base = pal * tex * hit.tint;
    // Soil under the blade sward: the ground between blades lives in their
    // occlusion shadow (GoT grounds read dark under the grass). Constant
    // (view- and distance-independent) so it can never read as a moving or
    // camera-following artifact.
    if (hit.mat == MAT_GRASS && hit.normal.y > 0.5) {
        base *= 0.94;
    }
    // Skip the cube-face AO for sub-voxel sphere hits (foliage). The curved
    // sphere normal already gives rim/falloff that reads as 3D.
    // AO (12 hierarchical neighbour lookups) only near the camera — its
    // contact-shadow detail is invisible far away, so skip it past AO_DIST and
    // for sub-voxel foliage hits. Leaf-block faces use leaf_canopy_ao in the
    // tracer instead: generic cube AO counts the invisible fringe shell as
    // solid and darkens exposed canopy faces (and would double-occlude on
    // top of the canopy AO).
    // (fringe included: its cap tufts return axis-aligned normals, and cube
    // AO on a fringe cell would re-introduce the invisible-shell darkening.)
    let skip_ao = hit.last_axis < 0 || hit.t_hit > AO_DIST
        || is_leaf_block_mat(hit.mat) || hit.mat == MAT_LEAF_FRINGE;
    // Per-voxel light field: ONE fetch stands in for the AO evaluation and the
    // sun-shadow ray. Sampled off the GEOMETRIC normal (hit.normal), not the
    // foliage-perturbed shading normal, because it addresses the air voxel
    // against the real face.
    //
    // A `known` answer still wins: that caller is asserting a PROVEN constant
    // (the refracted-water hit below the surface has shadow 0 and flat AO by
    // construction), and consulting the field there would pay for a lookup to
    // re-derive an answer already in hand.
    let vlf = voxlight_sample(p_hit, hit.normal);
    if (PROF_VLF_COVERAGE > 0.5) {
        return select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), vlf.valid);
    }
    // AO comes from the field and nowhere else. `voxlight_sample` returns 1.0
    // (unoccluded) when it has no record, so an unanswered surface is lit by
    // ambient rather than blackened by a guess.
    //
    // NOTE: the old sky_access() hack (a straight-up + 4 side shadow rays to fake
    // sky occlusion) is GONE, and so is the per-pixel `compute_ao` that stood
    // under this. The sky is not a hard overhead light - it is the sun scattered
    // by the atmosphere, a dim diffuse area source, and the world-space probe GI
    // models exactly that.
    var ao = select(vlf.ao, 1.0, skip_ao);
    if (known.known) { ao = known.ao; }

    // ---- swaying foliage ----
    // Leaves and grass-tops flutter their shading normal with a wind-advected
    // field. TWO hard rules, each a former "cloud shadow" artifact class
    // (guard test: no_field_scale_luma_waves):
    //  - NEVER modulate brightness (base) with a spatial field: field-scale
    //    luma waves read as shadows flying over the terrain.
    //  - The flutter field must be HIGH-frequency (wavelength ~1 voxel, i.e.
    //    leaf-sized) at CONSTANT amplitude. Long wavelengths, or scaling the
    //    amplitude by the traveling wind_gust envelope, re-create patch-scale
    //    luma waves through the nonlinear n.l response. Gust energy belongs
    //    to GEOMETRY (wind_offset card/quad shear), which moves silhouettes,
    //    not shading fields.
    var n = hit.normal;
    if (is_foliage_mat(hit.mat)) {
        let t = camera.time;
        let wd = wind_dir_now();
        let sway = gnoise3(vec3<f32>(
            (p_hit.x - wd.x * t * 2.2) * 0.9,
            (p_hit.z - wd.y * t * 2.2) * 0.9,
            t * 0.8,
        )) * 2.0 - 1.0;
        let amp = select(0.0, 0.0, hit.mat == MAT_FLOWER || hit.mat == MAT_TALL_GRASS
                                   || hit.mat == MAT_TALL_GRASS_DRY); // EXPERIMENT: flutter off
        n.x += sway * amp;
        n.z += sway * amp * 0.7;
        n = normalize(n);
    }
    // Rigid cube tops (MAT_GRASS included) get NO flutter at all: any
    // time-varying shading on geometry that visibly cannot move reads as a
    // shadow passing over it. Grass motion is carried by the tall-grass
    // cross-quad geometry standing ON the block, never by the block face.

    // ---- shader-pure grass turf (water-grade shading, zero geometry) ----
    // Realism carried entirely by shading, like the water surface: a STATIC
    // combed blade-direction field tilts the normal and feeds an anisotropic
    // sheen, and blade-scale micro-normals make the turf sparkle under sun
    // and TAA. Everything here is time-invariant (statics obey the no-luma-
    // waves rule above: only gust-scale TRAVELING fields read as shadows);
    // aliveness stays with the blade geometry standing on the turf.
    var g_comb = vec2<f32>(0.0);
    var g_fade = 0.0;
    var g_streak = 0.5;
    if (hit.mat == MAT_GRASS && hit.normal.y > 0.5 && hit.last_axis >= 0
        && hit.t_hit < GRASS_SHADE_T) {
        g_fade = 1.0 - smoothstep(GRASS_SHADE_T * 0.55, GRASS_SHADE_T, hit.t_hit);
        // Comb: which way the turf lies, swirling over ~20 voxels.
        let th = vnoise3(vec3<f32>(p_hit.x * 0.045, 9.1, p_hit.z * 0.045)) * 6.2832;
        g_comb = vec2<f32>(cos(th), sin(th));
        // Blade-scale micro-normal in the COMB FRAME: the noise domain is
        // stretched ~3x along the lay and compressed across it, so the
        // perturbation forms fine streaks (combed blades), not felt blobs.
        // Two octaves; isotropic noise here reads as moss, streaks as grass.
        let perp = vec2<f32>(-g_comb.y, g_comb.x);
        let q = vec2<f32>(dot(p_hit.xz, g_comb) * 4.5, dot(p_hit.xz, perp) * 14.0);
        let s1x = vnoise3(vec3<f32>(q.x, 4.2, q.y)) * 2.0 - 1.0;
        let s1z = vnoise3(vec3<f32>(q.x, 8.9, q.y)) * 2.0 - 1.0;
        let s2x = vnoise3(vec3<f32>(q.x * 2.7, 14.3, q.y * 2.7)) * 2.0 - 1.0;
        let s2z = vnoise3(vec3<f32>(q.x * 2.7, 21.7, q.y * 2.7)) * 2.0 - 1.0;
        let mnx = s1x * 0.24 + s2x * 0.17;
        let mnz = s1z * 0.24 + s2z * 0.17;
        // Streak mask reused to break the sheen into blade glints.
        g_streak = vnoise3(vec3<f32>(q.x * 1.7, 31.9, q.y * 1.7));
        n = normalize(n + vec3<f32>(g_comb.x, 0.0, g_comb.y) * 0.12
                        + vec3<f32>(mnx, 0.0, mnz) * g_fade);
    } else if (hit.mat == MAT_TURF && hit.t_hit < GRASS_SHADE_T) {
        // Turf BLADE hits share the sheen + backlight terms (grass is shiny
        // and translucent - the two foliage terms flat diffuse lacks). The
        // blade already carries a real rounded normal, so no perturbation;
        // the area comb stands in for the blade tangent (sheen is a field
        // response, per-blade variation comes from the normals).
        g_fade = 1.0 - smoothstep(GRASS_SHADE_T * 0.55, GRASS_SHADE_T, hit.t_hit);
        let th = vnoise3(vec3<f32>(p_hit.x * 0.045, 9.1, p_hit.z * 0.045)) * 6.2832;
        g_comb = vec2<f32>(cos(th), sin(th));
        g_streak = 0.8; // blades ARE the streaks: full glint
    }

    let s = sun_dir();
    let s_int = sun_intensity(s);
    var n_dot_l = max(0.0, dot(n, s));
    // Foliage light response: leaf cards catch the sun on wildly-varied
    // (often sun-facing) normals, so raw n.l swings from 0 to 1 and leaves
    // brighten/darken with the light FAR more than a flat block face (which
    // only ever sees n.l = sun elevation). Blend the foliage n.l partway
    // toward that flat response so leaves track light like the blocks do -
    // this touches only the LIGHT magnitude; the sway MOTION is untouched.
    if (is_foliage_mat(hit.mat)) {
        n_dot_l = mix(n_dot_l, max(0.0, s.y), 0.45);
    }
    // CONTINUOUS sun visibility straight from the field, and from nothing else.
    // No binary occlusion test survives into the pixel, so the penumbra is a real
    // gradient rather than a dither pattern that only resolves once TAA converges
    // (and never resolved at all while the camera moved, because jit_phase
    // froze). There is no second path underneath: the jittered per-pixel cone
    // that used to sit here is deleted.
    //
    // `voxlight_sample` returns 0.0 when it cannot answer, and that is a DESIGNED
    // term rather than a leftover default. What is left unanswered, once foliage
    // carries records, is a point whose whole trilinear neighbourhood is opaque -
    // i.e. a surface buried inside geometry, for which "no direct sun" is the
    // correct answer and not a guess - plus the one or two frames between a brick
    // being edited or streamed in and the urgent list gathering it. Neither goes
    // black: ambient and probe GI still light the surface exactly as they light
    // anything else in shadow.
    var shadow_term = vlf.sun;
    if (known.known) { shadow_term = known.sun; }
    // Report the terms so cs_main can put them in the G-buffer the grass pass
    // reads. PURELY an output now - nothing feeds them back in.
    *out_light = vec2<f32>(shadow_term, ao);

    let direct = sun_color(s) * (n_dot_l * shadow_term);
    let ambient = ambient_color() * ao;
    // One-bounce indirect passed in by the caller (temporally accumulated in
    // cs_main for the RT variant; 0 for software and for secondary rays).
    // Local lights arrive already gathered and shadow-tested in the field, so
    // adding lights costs the per-pixel path nothing.
    var lit = base * (direct + ambient + indirect + vlf.point);

    // Grass turf specular terms (see the comb block above): Kajiya-Kay
    // sheen along the combed blade tangent - the gloss real grass throws
    // when sun, view and lay-direction line up - plus a warm translucent
    // backlight when looking toward a low sun. Additive specular energy,
    // not an albedo modulation.
    if (g_fade > 0.0 && s_int > 0.0) {
        let bt = normalize(vec3<f32>(g_comb.x * 0.45, 1.0, g_comb.y * 0.45));
        let hv = normalize(s - dir);
        let tdh = dot(bt, hv);
        let sin_th = sqrt(max(0.0, 1.0 - tdh * tdh));
        // Gloss grows at grazing view like every rough surface's fresnel.
        let graze = pow(1.0 - max(0.0, dot(hit.normal, -dir)), 2.0);
        // The streak mask splits the pooled highlight into blade glints:
        // without it the sheen reads as caustic spots on a smooth surface.
        let glint = 0.25 + 0.75 * smoothstep(0.35, 0.75, g_streak);
        let sheen = pow(sin_th, 24.0) * (0.25 + 0.75 * graze) * glint;
        let back = pow(max(0.0, dot(dir, s)), 6.0)
            * clamp(1.0 - s.y * 1.4, 0.0, 1.0);
        lit += sun_color(s) * g_fade * (0.3 + 0.7 * shadow_term)
            * (sheen * 0.30 * vec3<f32>(1.0, 1.0, 0.85)
               + back * 0.30 * vec3<f32>(0.55, 0.85, 0.30));
    }

    let fog_t = fog_amount(hit.t_hit);
    return mix(lit, fog_atmospheric(dir), fog_t);
}

// Glass — Fresnel reflection + per-channel refraction (chromatic dispersion),
// Total Internal Reflection handling, and a specular sun glint. The cyan
// tint compounds with travel distance for chunky glass blocks.
fn shade_glass(hit: Hit, origin: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let p_hit = origin + dir * hit.t_hit;
    // See `shade_water_top`: secondary hits report into a scratch nothing reads.
    var scratch_light = vec2<f32>(0.0);
    let n = hit.normal;
    let s = sun_dir();
    let sc = sun_color(s);
    let jit = fract(p_hit.x * 17.0 + p_hit.z * 23.0 + camera.time * 13.0);

    // Glass keeps its PER-PIXEL mirror. It predates the per-voxel reflection
    // field and outlived it: a window pane is a small, near-flat, high-Fresnel
    // surface where a true mirror is the whole effect, and it is not the
    // stylized-water surface this branch reworked.
    let refl_dir = reflect(dir, n);
    let refl_origin = p_hit + n * 0.01;
    var refl_col: vec3<f32>;
    let refl_hit = trace(refl_origin, refl_dir);
    if (refl_hit.hit) {
        refl_col = shade(refl_hit, refl_origin, refl_dir, vl_unknown(), vec3<f32>(0.0), &scratch_light);
    } else {
        refl_col = sky(refl_dir);
    }

    // Chromatic dispersion: shift the refractive index slightly per channel.
    // R refracts less than B, so a flat glass face shows a faint rainbow at
    // grazing angles. Three trace calls is more expensive — only do it when
    // we'd actually see the dispersion (cos_theta < 0.9, i.e. near edges).
    let cos_theta_pre = clamp(dot(-dir, n), 0.0, 1.0);
    let eta_r = 1.0 / 1.48;
    let eta_g = 1.0 / 1.50;
    let eta_b = 1.0 / 1.52;

    var refr_dir_r = refract(dir, n, eta_r);
    var refr_dir_g = refract(dir, n, eta_g);
    var refr_dir_b = refract(dir, n, eta_b);

    // Total internal reflection on any channel → fall back to the reflection.
    let tir = length(refr_dir_g) < 0.01;
    if (tir) {
        let fog_t = fog_amount(hit.t_hit);
        return mix(refl_col, fog_atmospheric(dir), fog_t);
    }
    if (length(refr_dir_r) < 0.01) { refr_dir_r = refr_dir_g; }
    if (length(refr_dir_b) < 0.01) { refr_dir_b = refr_dir_g; }

    let refr_origin = p_hit + dir * 0.001;
    var glass_col: vec3<f32>;
    if (cos_theta_pre > 0.92) {
        // Near head-on: dispersion invisible — single trace, save 2/3 cost.
        let under = trace_secondary(refr_origin, refr_dir_g, SECONDARY_MAX_T);
        var under_col: vec3<f32>;
        if (under.hit) { under_col = shade(under, refr_origin, refr_dir_g, vl_unknown(), vec3<f32>(0.0), &scratch_light); }
        else { under_col = sky(refr_dir_g); }
        let depth = max(0.0, under.t_hit);
        let tint = vec3<f32>(0.05, 0.02, 0.02) * depth;
        glass_col = under_col * exp(-tint);
    } else {
        let ur = trace_secondary(refr_origin, refr_dir_r, SECONDARY_MAX_T);
        let ug = trace_secondary(refr_origin, refr_dir_g, SECONDARY_MAX_T);
        let ub = trace_secondary(refr_origin, refr_dir_b, SECONDARY_MAX_T);
        var cr = select(sky(refr_dir_r).r, shade(ur, refr_origin, refr_dir_r, vl_unknown(), vec3<f32>(0.0), &scratch_light).r, ur.hit);
        var cg = select(sky(refr_dir_g).g, shade(ug, refr_origin, refr_dir_g, vl_unknown(), vec3<f32>(0.0), &scratch_light).g, ug.hit);
        var cb = select(sky(refr_dir_b).b, shade(ub, refr_origin, refr_dir_b, vl_unknown(), vec3<f32>(0.0), &scratch_light).b, ub.hit);
        let depth_g = max(0.0, ug.t_hit);
        let tint = vec3<f32>(0.05, 0.02, 0.02) * depth_g;
        glass_col = vec3<f32>(cr, cg, cb) * exp(-tint);
    }

    // Specular sun glint on the glass face — bright pinpoint highlight when
    // the surface aligns the sun reflection toward the camera.
    let h_vec = normalize(s - dir);
    let spec = pow(max(0.0, dot(n, h_vec)), 200.0);
    var shadow = 0.0;
    if (spec > 0.002 && sun_intensity(s) > 0.0 && dot(n, s) > 0.0) {
        // FROM THE FIELD, like every other sun-visibility question on a surface.
        // This was the last per-pixel shadow ray left in the frame: one binary
        // test for the glint, which switched a pane's highlight on and off at a
        // shadow edge instead of fading it across one. Glass keeps its per-pixel
        // MIRROR - it predates all of this, and a pane is exactly where a true
        // mirror is the whole effect - but it does not keep a private shadow
        // mechanism. Still gated on the glint being visible at all, so a pane
        // facing away from the sun pays for no fetch.
        shadow = voxlight_sample(p_hit, n).sun;
    }

    let cos_theta = clamp(dot(-dir, n), 0.0, 1.0);
    let f0 = 0.04;
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);

    let fog_t = fog_amount(hit.t_hit);
    let combined = mix(glass_col, refl_col, fresnel) + sc * spec * shadow * 1.2;
    return mix(combined, fog_atmospheric(dir), fog_t);
}

// Volumetric clouds: raymarch a horizontal slab. Altitude lowered (was
// 200-250) so clouds sit inside the world Y = 192 — view rays past
// mountains can actually reach the cloud band instead of stopping at the
// world ceiling.
// Slab must stay inside the world's vertical extent (Y = 256, 64 bricks
// of 4) so rays under a cloud that also cross terrain keep a consistent
// depth story; 165..235 leaves headroom for the tower tops.
const CLOUD_BASE: f32 = 47.5 * VOXELS_PER_METRE;
const CLOUD_TOP:  f32 = 72.5 * VOXELS_PER_METRE;

fn render_clouds(origin: vec3<f32>, dir: vec3<f32>, t_terrain: f32, pix: vec2<f32>) -> vec4<f32> {
    // Slab intersection. A horizontal ray (|dir.y| ~ 0) is parallel to the
    // slab: OUTSIDE it never enters; INSIDE it stays in cloud for the whole
    // marchable range. (The old unconditional early-out here carved a dark
    // horizon line straight through the clouds when flying inside them.)
    var t_in: f32;
    var t_out: f32;
    if (abs(dir.y) < 1e-3) {
        if (origin.y < CLOUD_BASE || origin.y > CLOUD_TOP) { return vec4<f32>(0.0); }
        t_in = 0.0;
        t_out = 1.0e9;
    } else {
        let inv_dy = 1.0 / dir.y;
        t_in  = (CLOUD_BASE - origin.y) * inv_dy;
        t_out = (CLOUD_TOP  - origin.y) * inv_dy;
        if (t_in > t_out) { let tmp = t_in; t_in = t_out; t_out = tmp; }
    }
    let t_start = max(t_in, 0.0);
    let t_end   = min(t_out, t_terrain);
    if (t_end <= t_start + 0.5) { return vec4<f32>(0.0); }
    // Clamp the march to the STREAMED world window in XZ: clouds otherwise
    // hang over the unrendered void past the window edge, flagging exactly
    // where the world ends. Hidden there, the sky fades to horizon haze the
    // same way the missing terrain does.
    // Inflated by half the window span per side: clouds reach out to DOUBLE
    // the distance the streamed world does, then stop (no clouds over the
    // deep void, but no hard cut at the exact terrain edge either).
    let wspan = vec2<f32>(f32(WORLD_VOXELS_X), f32(WORLD_VOXELS_Z));
    let wmin = vec2<f32>(f32(camera.world_origin.x), f32(camera.world_origin.z)) - wspan * 0.5;
    let wmax = wmin + wspan * 2.0;
    var t_win = 1.0e9;
    if (abs(dir.x) > 1e-5) {
        t_win = min(t_win, max((wmin.x - origin.x) / dir.x, (wmax.x - origin.x) / dir.x));
    }
    if (abs(dir.z) > 1e-5) {
        t_win = min(t_win, max((wmin.y - origin.z) / dir.z, (wmax.y - origin.z) / dir.z));
    }
    // Distance-clamp the slab — beyond this clouds blend into atmospheric fog.
    let t_far_clamp = min(t_end, min(t_start + 600.0, t_win));

    let s = sun_dir();
    let sc = sun_color(s);

    // Half the marching steps of the original full-res march — the TAA pass
    // temporally accumulates the result on a static camera, so the lower
    // per-frame sample count is upsampled over time instead of in one frame
    // (checklist: clouds at reduced res + temporal upsample).
    // 16 steps: the worley florets (~30 voxels) need finer sampling than the
    // slab-scale march that covered smooth clouds - undersampled florets read
    // as rough lines instead of round masses. Half-res keeps this cheap.
    let N: i32 = 16;
    let step_t = (t_far_clamp - t_start) / f32(N);
    // Per-frame time-varying jitter, averaged by TAA into a smooth march.
    // History: this was once time-varying, then made spatial-only because
    // the composite lived in the TILE-GATED main pass (stale tiles froze
    // the jitter at mismatched phases = checkerboard #1). But the static
    // dither is its own artifact: every half-res pixel marches at a
    // permanently different phase, printing a fixed dither lattice into
    // the cloud alpha - the diffused checkerboard that travels with cloud
    // shade over terrain (checkerboard #2). With the composite now in the
    // full-screen per-frame cs_compose, time-varying jitter is finally
    // CORRECT: every pixel re-marches every frame and TAA averages the
    // phases. Do not de-time this again - move work out of tile-gated
    // passes instead.
    let h = ign(pix.x, pix.y, camera.time * 60.0);

    // Henyey-Greenstein forward-scatter — gives the "silver lining" effect
    // when looking toward the sun through cloud edges.
    let cos_sun = dot(dir, s);
    let phase = phase_hg(cos_sun, 0.65) * 4.0 + 0.5;

    var transmittance: f32 = 1.0;
    var scattered: vec3<f32> = vec3<f32>(0.0);
    // Day/night-aware ambient (was hardcoded blue — clouds glowed at night).
    // Scale ambient_color a bit so daytime clouds still read as bright.
    let ambient = ambient_color() * 1.05 + sc * 0.06;

    for (var i: i32 = 0; i < N; i = i + 1) {
        let t = t_start + (f32(i) + h) * step_t;
        let p = origin + dir * t;
        let d = cloud_density(p, camera.sun_time);
        if (d < 0.01) { continue; }

        // 3 NON-UNIFORM jittered cone samples toward the sun: two same-spaced
        // taps quantize the self-shadow into iso-bands that crawl with the
        // camera ("shadow lines across the clouds"); staggered distances plus
        // the per-frame jitter break the banding, and the near tap keeps the
        // floret-scale contrast that makes lobes read as volumetric balls.
        var sun_dens: f32 = 0.0;
        sun_dens = sun_dens + cloud_density(p + s * ((h - 0.5) * 2.0 + 3.5), camera.sun_time);
        sun_dens = sun_dens + cloud_density(p + s * ((h - 0.5) * 3.0 + 8.0), camera.sun_time);
        sun_dens = sun_dens + cloud_density(p + s * ((h - 0.5) * 5.0 + 14.0), camera.sun_time);
        let sun_t = exp(-sun_dens * 0.60);
        let local_col = ambient + sc * sun_t * phase;

        let sample_t = exp(-d * step_t * 0.14);
        let alpha = (1.0 - sample_t) * transmittance;
        scattered = scattered + local_col * alpha;
        transmittance = transmittance * sample_t;
        if (transmittance < 0.02) { break; }
    }

    let alpha = 1.0 - transmittance;
    return vec4<f32>(scattered, alpha);
}

// Henyey-Greenstein phase function — anisotropic single-scatter widely used
// for clouds/atmosphere. g ∈ [-1, 1]: positive = forward-scatter (Mie-like),
// matches real sunbeam behaviour. Returns the *relative* phase; we apply our
// own brightness scaling.
fn phase_hg(cos_th: f32, g: f32) -> f32 {
    let denom = 1.0 + g * g - 2.0 * g * cos_th;
    return (1.0 - g * g) / (4.0 * 3.14159265 * pow(max(denom, 1e-3), 1.5));
}

// Volumetric god rays. Henyey-Greenstein phase (g≈0.7) gives the natural
// "halo gets brighter as you look closer to the sun" falloff. IGN jitter is
// reused so adjacent pixels get well-distributed offsets — important for
// noise that the temporal-differential pass can average away.
// Occlusion fraction of the god-ray march (0 = fully shadowed shafts, 1 =
// clear path). Computed at HALF RESOLUTION by cs_godrays - the shadow_occluded
// march is the priciest per-pixel term in compose - and upsampled per full-res
// pixel with exact per-pixel colour/phase terms.
fn god_ray_frac(origin: vec3<f32>, dir: vec3<f32>, t_far: f32, pix: vec2<f32>) -> f32 {
    let s = sun_dir();
    if (sun_intensity(s) <= 0.0) { return 0.0; }
    let phase = phase_hg(dot(dir, s), 0.7) * 4.0;
    if (phase < 0.05) { return 0.0; }
    let t_max = min(t_far, 140.0);
    if (t_max <= 1.0) { return 0.0; }
    // Sample count scales with phase - looking right at the sun gets denser
    // sampling for a smooth halo; off-axis stays cheap.
    let N: i32 = select(3, 6, phase > 0.40);
    let step_t = t_max / f32(N);
    let h = ign(pix.x, pix.y, camera.time * 60.0);
    var sum = 0.0;
    for (var i: i32 = 0; i < N; i = i + 1) {
        let t = (f32(i) + h) * step_t;
        let p = origin + dir * t;
        // God-ray shafts only need NEARBY occluders - a short occlusion cap lets
        // the hierarchical trace bail out far sooner than a full shadow ray.
        if (!shadow_occluded(p + s * 0.5, s, GOD_RAY_OCCL_DIST)) {
            // Distance-weighted contribution: nearer scatter looks brighter.
            sum = sum + exp(-t * 0.008);
        }
    }
    return sum / f32(N);
}

// Half-res god-ray pass: one occlusion march per 2x2 block, written (with the
// source depth for the bilateral upsample) into the TAIL of transp_buf - the
// per-pixel transparent records there were already consumed by cs_transparent
// this frame, so the scratch reuse needs no extra binding. Runs between
// cs_transparent and cs_compose.
@compute @workgroup_size(8, 8, 1)
fn cs_godrays(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = vec2<i32>(camera.resolution);
    let half_res = (res + vec2<i32>(1)) / 2;
    if (i32(gid.x) >= half_res.x || i32(gid.y) >= half_res.y) { return; }
    let px = min(vec2<i32>(i32(gid.x) * 2, i32(gid.y) * 2), res - vec2<i32>(1));
    let t_hit = textureLoad(depth_in, px, 0).r;
    let uv = (vec2<f32>(px) + vec2<f32>(1.0)) / camera.resolution;
    let dir = ray_dir_uv(uv);
    let frac = god_ray_frac(camera.origin, dir, min(t_hit, 200.0),
                            vec2<f32>(f32(px.x), f32(px.y)));
    // Scratch tail lives BEYOND the per-pixel records (disjoint by
    // construction - records persist across frames under tile-gating).
    let hidx = res.x * res.y + i32(gid.y) * half_res.x + i32(gid.x);
    transp_buf[hidx] = vec4<u32>(bitcast<u32>(frac), bitcast<u32>(t_hit), 0u, 0u);
}

// Stripped-down DDA — same hierarchy as `trace()` but returns the moment we
// know the ray is occluded. No normal / material work.
// Does a solid voxel actually block a shadow / sky-access ray? Invisible canopy
// fringe never does; ground decoration (tufts, flowers) and leaves use the same
// near/far cutout rules the primary DDA applies, so the software DDA and the RT
// occlusion path agree pixel-for-pixel. `voxel` is the WORLD voxel; `t_cur` its
// distance along the ray. Shared by trace_any (software) and rt_brick_occludes
// (hardware RT) - the ONE source of the shadow occluder rule.
fn shadow_voxel_occludes(voxel: vec3<i32>, m: u32, t_cur: f32, origin: vec3<f32>, dir: vec3<f32>) -> bool {
    // A light-field gather never occludes itself: see `shadow_skip_active`.
    if (shadow_skip_active
        && all(voxel >= shadow_skip_voxel)
        && all(voxel < shadow_skip_voxel + vec3<i32>(VL_STEP))) { return false; }
    if (m == MAT_LEAF_FRINGE || m == MAT_TURF) {
        // Invisible canopy fringe never occludes shadow rays; turf blades
        // are below shadow scale (their root-dark ramp is the self-shadow)
        // and testing them on every ground shadow ray would be pure cost.
        return false;
    }
    if (is_decoration_mat(m)) {
        // Ground decoration: near, the cutout gives dappled micro-shadow; far, a
        // tuft is 90% air and blocking it as a solid cube stamps a square shadow
        // per tuft across every meadow, so far decorations don't block.
        if (t_cur <= FOLIAGE_NEAR_T) {
            shadow_wind_freeze = true;
            let occ = foliage_subvoxel(voxel, origin, dir, m).hit;
            shadow_wind_freeze = false;
            return occ;
        }
        return false;
    }
    if (is_foliage_mat(m)) {
        // Leaves: far canopies block as solid cubes (they really are dense);
        // near ones pay the cutout test.
        if (t_cur > FOLIAGE_NEAR_T) { return true; }
        shadow_wind_freeze = true;
        let occ = foliage_subvoxel(voxel, origin, dir, m).hit;
        shadow_wind_freeze = false;
        return occ;
    }
    return true;
}

fn trace_any(origin: vec3<f32>, dir: vec3<f32>, max_dist: f32) -> bool {
    let init = dda_init(origin, dir);
    if (!init.valid) { return false; }
    let ro = init.ro;
    let org = init.org;
    let inv_dir = init.inv_dir;
    let step = init.step;
    let t_delta = init.t_delta;
    let t_enter = init.t_enter;
    var voxel = init.voxel;
    var t_max = init.t_max;
    var slot_v = init.slot_v;
    var last_axis: i32 = -1;
    var t_cur = t_enter;
    for (var s: i32 = 0; s < 768; s = s + 1) {
        if (t_cur > max_dist) { return false; }
        let rel = voxel - camera.world_origin;
        if (rel.x < 0 || rel.x >= WORLD_VOXELS_X
         || rel.y < 0 || rel.y >= WORLD_VOXELS_Y
         || rel.z < 0 || rel.z >= WORLD_VOXELS_Z) { return false; }

        // slot_v is maintained incrementally (see step + skip_to_cell).
        let bp = slot_v >> vec3<u32>(2u);
        let tp = slot_v >> vec3<u32>(4u);
        let cp = slot_v >> vec3<u32>(6u);
        // Nested hierarchy: skip the coarsest empty cell (L4 → chunk → tile → brick).
        let l4p = slot_v >> vec3<u32>(8u);
        let l4i = world_l4_idx(l4p.x, l4p.y, l4p.z);
        if (l4_cell_empty(l4i)) {
            skip_to_cell(256, &voxel, &t_max, ro, org, dir, inv_dir, step, &last_axis, &slot_v);
            t_cur = axis_select(t_max, last_axis) - axis_select(t_delta, last_axis);
            continue;
        }
        let chunk_lin = (cp.x & 3) + (cp.z & 3) * 4 + (cp.y & 3) * 16;
        if (!l4_has_child(l4i, chunk_lin)) {
            skip_to_cell(64, &voxel, &t_max, ro, org, dir, inv_dir, step, &last_axis, &slot_v);
            t_cur = axis_select(t_max, last_axis) - axis_select(t_delta, last_axis);
            continue;
        }
        let ci = world_chunk_idx(cp.x, cp.y, cp.z);
        let tile_lin = (tp.x & 3) + (tp.z & 3) * 4 + (tp.y & 3) * 16;
        if (!chunk_has_child(ci, tile_lin)) {
            skip_to_cell(16, &voxel, &t_max, ro, org, dir, inv_dir, step, &last_axis, &slot_v);
            t_cur = axis_select(t_max, last_axis) - axis_select(t_delta, last_axis);
            continue;
        }
        let ti = world_tile_idx(tp.x, tp.y, tp.z);
        let brick_lin = (bp.x & 3) + (bp.z & 3) * 4 + (bp.y & 3) * 16;
        if (!tile_has_child(ti, brick_lin)) {
            skip_to_cell(4, &voxel, &t_max, ro, org, dir, inv_dir, step, &last_axis, &slot_v);
            t_cur = axis_select(t_max, last_axis) - axis_select(t_delta, last_axis);
            continue;
        }
        let bi = world_brick_idx(bp.x, bp.y, bp.z);
        let local = slot_v - bp * BRICK_DIM;
        let vi = brick_voxel_idx(local.x, local.y, local.z);
        if (brick_voxel_solid(bi, vi)) {
            let m = brick_voxel_material(bi, vi);
            if (shadow_voxel_occludes(voxel, m, t_cur, origin, dir)) { return true; }
        }

        dda_step(&voxel, &slot_v, &t_max, &t_cur, &last_axis, step, t_delta);
    }
    return false;
}

// A cell occludes ambient light only if it holds an actually-solid block:
// grass tufts, flowers, dry straw and the invisible canopy fringe occupy
// their cells (the DDA must find them) but must NOT stamp AO squares onto
// the ground - thousands of hash-scattered tufts otherwise read as a
// diffused checkerboard-pattern shadow carpet. The material fetch is
// gated on the occupancy bit, so probes over open ground stay one load.
fn ao_occluder(c: vec3<i32>) -> bool {
    if (!is_voxel_solid(c)) { return false; }
    return !is_decoration_mat(voxel_material_at(c));
}

fn axis_select(v: vec3<f32>, ax: i32) -> f32 {
    if (ax == 0) { return v.x; }
    if (ax == 1) { return v.y; }
    return v.z;
}

// LOD: past this many voxels of distance, terminate the DDA at brick
// granularity instead of per-voxel.
const LOD_BRICK_T: f32 = 100.0 * VOXELS_PER_METRE;

// Even further out, terminate at TILE (16-voxel) granularity: far terrain is
// blocky but each occupied tile costs one hit instead of a brick/voxel descent
// (checklist: tile-level LOD for far terrain).
const TILE_LOD_T: f32 = 130.0 * VOXELS_PER_METRE;

// Distance-based ray budget (in voxels). Traversal stops here regardless of how
// many DDA steps it took — replaces the old fixed voxel-step count so small
// voxels can't run the loop out before reaching far geometry (holes-through-
// terrain) and so empty rays don't waste steps. Comfortably covers the loaded
// window (the camera sits at its centre).
const MAX_RAY_DIST: f32 = 175.0 * VOXELS_PER_METRE;
// Secondary rays (water reflection/refraction, glass) stop here: fog is 41%
// by 200 and the fresnel-weighted reflection of far geometry is
// indistinguishable from the sky it fades into - while traversal cost scales
// with range. Applied identically to the software and RT secondary paths.
const SECONDARY_MAX_T: f32 = 50.0 * VOXELS_PER_METRE;

// Beyond this distance, sub-voxel foliage (sprite cross-quads, leaf cutout
// faces) is treated as a solid cube rather than ray-marched. Authored-sprite
// foliage is cheap (2 plane tests + 1 texel fetch vs the old 22-blade
// procedural bundle), so the detail radius is much wider than the old 72.
const FOLIAGE_NEAR_T: f32 = 32.0 * VOXELS_PER_METRE;

// Shadow / occlusion rays give up past this distance (treated as lit). Far
// shadows contribute little and are the most expensive secondary rays
// (checklist: cheaper secondary rays / coarse shadows).
const SHADOW_MAX_DIST: f32 = 120.0 * VOXELS_PER_METRE;

// Beyond this distance, skip per-corner ambient occlusion (its contact-shadow
// detail is sub-pixel far away). 12 hierarchical lookups/pixel saved on the
// bulk of the screen.
const AO_DIST: f32 = 16.0 * VOXELS_PER_METRE;

// God-ray occlusion cap: shafts only need nearby occluders, so the per-step
// occlusion test bails out much sooner than a full-length shadow ray.
const GOD_RAY_OCCL_DIST: f32 = 40.0 * VOXELS_PER_METRE;

// Resolve ONE solid voxel the primary ray reached: decide whether it is a real
// visible hit and, if so, fill `out`. Shared by the software DDA (trace) and the
// hardware-RT primary (trace_rt) so both render leaves/decoration cutouts, water
// sub-voxel surfaces and opaque cubes IDENTICALLY. `en_*` is the entry face
// (from entry_normal_and_t), `entry_last_axis` the cube last_axis, `t_exit_cell`
// the cell exit t. Returns false when the ray should keep going (cutout miss,
// far decoration, above the water patch).
// The single source of truth for "this voxel resolves to a plain opaque cube at
// this distance" - i.e. none of the sub-voxel branches below fire. Used both by
// resolve_solid_voxel (its opaque case) and by the RT primary's fast path, which
// takes the cube hit directly and skips the slot/brick-pos setup + this call when
// the answer is already a cube (the ~90% opaque-pixel case). Glass counts as an
// opaque cube here: the cube face is the primary hit, refraction is a shade pass.
fn resolves_as_opaque_cube(m: u32, t_cur: f32) -> bool {
    return !(is_foliage_mat(m) && t_cur <= FOLIAGE_NEAR_T)
        && !is_decoration_mat(m)
        && !(is_water_mat(m) && t_cur <= WATER_FAR_T);
}

fn resolve_solid_voxel(voxel: vec3<i32>, m: u32, slot_v: vec3<i32>, bp: vec3<i32>, bi: i32,
                       t_cur: f32, origin: vec3<f32>, dir: vec3<f32>,
                       en_n: vec3<f32>, en_t: f32, entry_last_axis: i32, t_exit_cell: f32,
                       out: ptr<function, Hit>) -> bool {
    // Opaque cube (the common case): entry face + t straight through.
    if (resolves_as_opaque_cube(m, t_cur)) {
        (*out).hit = true;
        (*out).mat = m;
        (*out).normal = en_n;
        (*out).voxel = voxel;
        (*out).last_axis = entry_last_axis;
        (*out).t_hit = en_t;
        return true;
    }
    // Near foliage/decoration: per-blade procedural cutout. Far foliage falls to
    // the opaque cube branch (canopies survive); far decoration is invisible.
    if (is_foliage_mat(m) && t_cur <= FOLIAGE_NEAR_T) {
        let fh = foliage_subvoxel(voxel, origin, dir, m);
        if (fh.hit) {
            (*out).hit = true;
            (*out).mat = m;
            (*out).normal = fh.normal;
            (*out).voxel = voxel;
            (*out).last_axis = axis_from_face_normal(fh.normal);
            (*out).t_hit = fh.t_hit;
            var tint = fh.color_tint;
            if ((is_leaf_block_mat(m) || m == MAT_LEAF_FRINGE) && fh.t_hit < AO_DIST) {
                tint = tint * leaf_canopy_ao(voxel, slot_v, bp, bi);
            }
            (*out).tint = tint;
            return true;
        }
        return false; // cutout missed
    }
    if (is_decoration_mat(m)) {
        return false; // far decoration invisible
    }
    // The only remaining case is near water (resolves_as_opaque_cube ruled out
    // opaque, foliage and decoration above).
    let wh = water_subvoxel(voxel, origin, dir, m, en_n, en_t, t_exit_cell, slot_v, bp, bi);
    if (wh.hit) {
        (*out).hit = true;
        (*out).mat = m;
        (*out).normal = wh.normal;
        (*out).voxel = voxel;
        (*out).last_axis = -1; // sub-voxel hit (water is always deferred)
        (*out).t_hit = wh.t_hit;
        water_facet_grad = wh.grad;
        water_facet_code = water_pack_facet(wh.band, wh.shore);
        return true;
    }
    return false; // ray passed above the plate
}

fn trace(origin: vec3<f32>, dir: vec3<f32>) -> Hit {
    water_facet_grad = vec2<f32>(0.0);
    water_facet_code = 0u;
    var out: Hit;
    out.hit = false;
    out.mat = 0u;
    out.normal = vec3<f32>(0.0);
    out.voxel = vec3<i32>(0);
    out.last_axis = -1;
    out.t_hit = 0.0;
    out.tint = vec3<f32>(1.0);

    let init = dda_init(origin, dir);
    if (!init.valid) { return out; }
    let ro = init.ro;
    let org = init.org;
    let inv_dir = init.inv_dir;
    let step = init.step;
    let t_delta = init.t_delta;
    let tmin3 = init.tmin3;
    let t_enter = init.t_enter;
    var voxel = init.voxel;
    var t_max = init.t_max;
    var slot_v = init.slot_v;
    var last_axis: i32 = -1;
    var t_cur: f32 = t_enter;
    let max_steps: i32 = 1024;
    for (var s: i32 = 0; s < max_steps; s = s + 1) {
        // Distance budget — primary termination (the step cap is just a backstop).
        if (t_cur > MAX_RAY_DIST) { return out; }
        // Bounds check on the LOADED WINDOW (world coords).
        let rel = voxel - camera.world_origin;
        if (rel.x < 0 || rel.x >= WORLD_VOXELS_X
         || rel.y < 0 || rel.y >= WORLD_VOXELS_Y
         || rel.z < 0 || rel.z >= WORLD_VOXELS_Z) {
            return out;
        }

        // slot_v is maintained incrementally (see step + skip_to_cell).
        let bp = slot_v >> vec3<u32>(2u);
        let tp = slot_v >> vec3<u32>(4u);
        let cp = slot_v >> vec3<u32>(6u);

        // ---- Nested hierarchical descent: skip the COARSEST empty cell ----
        // L4 (256) → chunk (64) → tile (16) → brick (4) → voxel. Each empty
        // level skips its whole cell in one DDA jump, so empty space (most of a
        // ray) costs O(coarse steps) instead of O(voxels) — this is what keeps
        // traversal cheap as voxels shrink.
        if (!PROFILE_NO_L4) {
            let l4p = slot_v >> vec3<u32>(8u);
            let l4i = world_l4_idx(l4p.x, l4p.y, l4p.z);
            if (l4_cell_empty(l4i)) {
                skip_to_cell(256, &voxel, &t_max, ro, org, dir, inv_dir, step, &last_axis, &slot_v);
                t_cur = axis_select(t_max, last_axis) - axis_select(t_delta, last_axis);
                continue;
            }
            let chunk_lin = (cp.x & 3) + (cp.z & 3) * 4 + (cp.y & 3) * 16;
            if (!l4_has_child(l4i, chunk_lin)) {
                skip_to_cell(64, &voxel, &t_max, ro, org, dir, inv_dir, step, &last_axis, &slot_v);
                t_cur = axis_select(t_max, last_axis) - axis_select(t_delta, last_axis);
                continue;
            }
        }

        let ci = world_chunk_idx(cp.x, cp.y, cp.z);
        let tile_in_chunk_lin = (tp.x & 3) + (tp.z & 3) * 4 + (tp.y & 3) * 16;
        if (!chunk_has_child(ci, tile_in_chunk_lin)) {
            skip_to_cell(16, &voxel, &t_max, ro, org, dir, inv_dir, step, &last_axis, &slot_v);
            t_cur = axis_select(t_max, last_axis) - axis_select(t_delta, last_axis);
            continue;
        }

        let ti = world_tile_idx(tp.x, tp.y, tp.z);

        // ---- Tile-level LOD: far terrain terminates at the (occupied) tile ----
        if (t_cur > TILE_LOD_T) {
            let lm = tile_representative_material(ti, tp);
            if (lm != 0u && is_uniform_optimisable(lm)) {
                let en = entry_normal_and_t(last_axis, step, t_max, t_delta, t_enter, tmin3);
                let n = en.n;
                let t_hit = en.t_hit;
                out.hit = true;
                out.mat = lm;
                out.normal = n;
                out.voxel = voxel;
                out.last_axis = last_axis_after_entry(last_axis, tmin3);
                out.t_hit = t_hit;
                return out;
            }
        }

        // ---- Fast-skip: uniform 16-voxel tile (one material throughout) ----
        // Whole 4096-voxel tile is one opaque material — surface at entry face.
        let tum = tile_uniform_mat(ti);
        if (tum != 0u && is_uniform_optimisable(tum)) {
            let en = entry_normal_and_t(last_axis, step, t_max, t_delta, t_enter, tmin3);
            let n = en.n;
            let t_hit = en.t_hit;
            out.hit = true;
            out.mat = tum;
            out.normal = n;
            out.voxel = voxel;
            out.last_axis = last_axis_after_entry(last_axis, tmin3);
            out.t_hit = t_hit;
            return out;
        }

        let brick_in_tile_lin = (bp.x & 3) + (bp.z & 3) * 4 + (bp.y & 3) * 16;
        if (!tile_has_child(ti, brick_in_tile_lin)) {
            skip_to_cell(4, &voxel, &t_max, ro, org, dir, inv_dir, step, &last_axis, &slot_v);
            t_cur = axis_select(t_max, last_axis) - axis_select(t_delta, last_axis);
            continue;
        }

        let bi = world_brick_idx(bp.x, bp.y, bp.z);

        // ---- Fast-skip: uniform 4-voxel brick (one material throughout) ----
        let bum = brick_uniform_mat(bi);
        if (bum != 0u && is_uniform_optimisable(bum)) {
            let en = entry_normal_and_t(last_axis, step, t_max, t_delta, t_enter, tmin3);
            let n = en.n;
            let t_hit = en.t_hit;
            out.hit = true;
            out.mat = bum;
            out.normal = n;
            out.voxel = voxel;
            out.last_axis = last_axis_after_entry(last_axis, tmin3);
            out.t_hit = t_hit;
            return out;
        }

        // ---- LOD: brick-level early termination at far distance ----
        // If we're far enough away that voxels are sub-pixel anyway, take
        // the brick's representative material and return — saves the inner
        // per-voxel DDA loop (up to ~7 steps per brick).
        if (t_cur > LOD_BRICK_T) {
            let b = bricks[bi];
            if ((b.occ_lo | b.occ_hi) == 0u) {
                // Empty brick — skip the whole 4-voxel cell.
                skip_to_cell(4, &voxel, &t_max, ro, org, dir, inv_dir, step, &last_axis, &slot_v);
                t_cur = axis_select(t_max, last_axis) - axis_select(t_delta, last_axis);
                continue;
            }
            let m = brick_topmost_material(bi);
            // Collapse the brick to its representative material — UNLESS the top
            // is foliage (a flower/grass/leaf cube would look wrong); in that
            // case fall through to the per-voxel descent below.
            if (!is_foliage_mat(m)) {
                let en = entry_normal_and_t(last_axis, step, t_max, t_delta, t_enter, tmin3);
                let n = en.n;
                let t_hit = en.t_hit;
                out.hit = true;
                out.mat = m;
                out.normal = n;
                out.voxel = voxel;
                out.last_axis = last_axis_after_entry(last_axis, tmin3);
                out.t_hit = t_hit;
                return out;
            }
        }

        let local = slot_v - bp * BRICK_DIM;
        let vi = brick_voxel_idx(local.x, local.y, local.z);
        if (brick_voxel_solid(bi, vi)) {
            let m = brick_voxel_material(bi, vi);
            let en = entry_normal_and_t(last_axis, step, t_max, t_delta, t_enter, tmin3);
            let t_exit_cell = min(t_max.x, min(t_max.y, t_max.z));
            let ela = last_axis_after_entry(last_axis, tmin3);
            if (resolve_solid_voxel(voxel, m, slot_v, bp, bi, t_cur, origin, dir,
                                    en.n, en.t_hit, ela, t_exit_cell, &out)) {
                return out;
            }
            // cutout / far-decoration / above-water miss -> keep stepping.
        }

        dda_step(&voxel, &slot_v, &t_max, &t_cur, &last_axis, step, t_delta);
    }
    return out;
}

// Representative material of a 16-voxel tile for far tile-LOD: the topmost
// solid voxel of the tile's first occupied child brick. tp = tile coord in slot
// space.
fn tile_representative_material(ti: i32, tp: vec3<i32>) -> u32 {
    let base = ti * 2;
    var lin: i32 = -1;
    let lo = tile_mask[base];
    if (lo != 0u) {
        lin = i32(firstTrailingBit(lo));
    } else {
        let hi = tile_mask[base + 1];
        if (hi != 0u) { lin = 32 + i32(firstTrailingBit(hi)); }
    }
    if (lin < 0) { return 0u; }
    let lx = lin & 3;
    let lz = (lin >> 2) & 3;
    let ly = (lin >> 4) & 3;
    let bi = world_brick_idx(tp.x * 4 + lx, tp.y * 4 + ly, tp.z * 4 + lz);
    return brick_topmost_material(bi);
}

fn last_axis_after_entry(la: i32, tmin3: vec3<f32>) -> i32 {
    if (la >= 0) { return la; }
    if (tmin3.x >= tmin3.y && tmin3.x >= tmin3.z) { return 0; }
    if (tmin3.y >= tmin3.z) { return 1; }
    return 2;
}

fn skip_to_cell(
    cell_size: i32,
    voxel: ptr<function, vec3<i32>>,
    t_max: ptr<function, vec3<f32>>,
    ro: vec3<i32>,        // rebase reference (window corner, world voxels)
    org: vec3<f32>,       // ray origin in window-local coords (= origin - ro)
    dir: vec3<f32>,
    inv_dir: vec3<f32>,
    step: vec3<i32>,
    last_axis: ptr<function, i32>,
    slot_v: ptr<function, vec3<i32>>,
) {
    // Cell alignment is in ABSOLUTE world coords (cell_size divides
    // WORLD_VOXELS so this matches the toroidal slot cells). Only the float t
    // math is done relative to `ro` for precision.
    let cell_origin = vec3<i32>(
        (*voxel).x - pos_mod((*voxel).x, cell_size),
        (*voxel).y - pos_mod((*voxel).y, cell_size),
        (*voxel).z - pos_mod((*voxel).z, cell_size),
    );
    // Local-space cell boundary (small integers → exact in f32).
    var bnd: vec3<f32>;
    bnd.x = f32(select(cell_origin.x, cell_origin.x + cell_size, step.x > 0) - ro.x);
    bnd.y = f32(select(cell_origin.y, cell_origin.y + cell_size, step.y > 0) - ro.y);
    bnd.z = f32(select(cell_origin.z, cell_origin.z + cell_size, step.z > 0) - ro.z);
    let t_face = (bnd - org) * inv_dir;
    let eps = 1e-6;
    var t_min: f32 = 1e30;
    var ax: i32 = 0;
    if (step.x != 0 && t_face.x > eps && t_face.x < t_min) { t_min = t_face.x; ax = 0; }
    if (step.y != 0 && t_face.y > eps && t_face.y < t_min) { t_min = t_face.y; ax = 1; }
    if (step.z != 0 && t_face.z > eps && t_face.z < t_min) { t_min = t_face.z; ax = 2; }
    let bias = 1e-3;
    let p_new = org + dir * (t_min + bias);
    var nv = vec3<i32>(floor(p_new)) + ro;
    // Integer-snap the crossed axis exactly (float floor can land on the wrong
    // side of a boundary; the cell math is exact).
    if (ax == 0) {
        if (step.x > 0) { nv.x = cell_origin.x + cell_size; }
        else            { nv.x = cell_origin.x - 1; }
    } else if (ax == 1) {
        if (step.y > 0) { nv.y = cell_origin.y + cell_size; }
        else            { nv.y = cell_origin.y - 1; }
    } else {
        if (step.z > 0) { nv.z = cell_origin.z + cell_size; }
        else            { nv.z = cell_origin.z - 1; }
    }
    (*voxel) = nv;
    // t_max for the new cell, computed in local coords.
    let vl = nv - ro;
    if (step.x > 0) { (*t_max).x = (f32(vl.x + 1) - org.x) * inv_dir.x; } else { (*t_max).x = (f32(vl.x) - org.x) * inv_dir.x; }
    if (step.y > 0) { (*t_max).y = (f32(vl.y + 1) - org.y) * inv_dir.y; } else { (*t_max).y = (f32(vl.y) - org.y) * inv_dir.y; }
    if (step.z > 0) { (*t_max).z = (f32(vl.z + 1) - org.z) * inv_dir.z; } else { (*t_max).z = (f32(vl.z) - org.z) * inv_dir.z; }
    *last_axis = ax;
    // A skip jumps the voxel arbitrarily, so the incrementally-tracked slot
    // coord must be recomputed here (the per-step path keeps it in sync cheaply).
    *slot_v = world_to_slot_voxel(nv);
}
