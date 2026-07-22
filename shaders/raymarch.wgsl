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
@group(0) @binding(5) var output_tex: texture_storage_2d<rgba8unorm, write>;
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

// Reprojected shadow/AO cache (#12). light_in = previous frame's G-buffer
// (xyz = hit pos relative to world_origin, w = pack2x16float(shadow, ao));
// light_out = this frame's. cs_main reprojects each hit into last frame's screen
// and reuses the cached shadow/AO when the stored position matches (else traces).
@group(0) @binding(15) var light_in: texture_2d<f32>;
@group(0) @binding(16) var light_out: texture_storage_2d<rgba32float, write>;

// Deferred transparent pass (#16). cs_main records each transparent (water-top /
// glass) hit here as (t_hit, mat_code, normal_code, flag) and writes a cheap
// placeholder colour; the separate cs_transparent pass does the expensive
// reflection/refraction so the opaque-majority warps in cs_main stay coherent
// (less 8x8 divergence). A read_write storage buffer (not a texture) lets both
// passes share this binding without a read/write aliasing hazard.
// Deferred transparent records, one per pixel. Explicit u32 schema (bit
// patterns must survive exactly; f32 lanes are not bit-stable for packed
// payloads on all hardware):
//   x = bitcast<u32>(t_hit)
//   y = kind: TR_NONE / TR_WATER_TOP / TR_WATER_FACE / TR_GLASS
//   z = TR_WATER_TOP: pack2x16float(rest-gradient of the surface, i.e. the
//       level/step part without the wave field - cs_transparent adds the
//       per-pixel field gradient on top); TR_WATER_FACE/TR_GLASS: axis face
//       code (encode_face_normal)
//   w = spare (0)
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
    // Walk from top y=3 layer down.
    let occ_hi = b.occ_hi;
    let occ_lo = b.occ_lo;
    // y=3 layer is occ_hi >> 16 (16 bits at bit 48..63).
    let y3 = (occ_hi >> 16u) & 0xFFFFu;
    if (y3 != 0u) {
        let bit = firstTrailingBit(y3);
        return brick_voxel_material(bi, i32(48u + bit));
    }
    // y=2 layer = occ_hi & 0xFFFF (bits 32..47).
    let y2 = occ_hi & 0xFFFFu;
    if (y2 != 0u) {
        let bit = firstTrailingBit(y2);
        return brick_voxel_material(bi, i32(32u + bit));
    }
    // y=1 layer = occ_lo >> 16 (bits 16..31).
    let y1 = (occ_lo >> 16u) & 0xFFFFu;
    if (y1 != 0u) {
        let bit = firstTrailingBit(y1);
        return brick_voxel_material(bi, i32(16u + bit));
    }
    // y=0 layer = occ_lo & 0xFFFF (bits 0..15).
    let y0 = occ_lo & 0xFFFFu;
    if (y0 != 0u) {
        let bit = firstTrailingBit(y0);
        return brick_voxel_material(bi, i32(bit));
    }
    return 0u;
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

fn sky(dir: vec3<f32>) -> vec3<f32> {
    return sky_color(dir);
}

/// Sky-like colour with **no stars, sun disc, or cloud emitters** — used for
/// the fog blend on distant terrain so far-away blocks don't visibly show
/// pinpoint stars through them at night.
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
const PROFILE_FLAT: bool = false;
const PROFILE_NO_L4: bool = false;

// Reprojected shadow/AO cache (#12). Set false to fall back to tracing shadow+AO
// every frame (e.g. if reprojection ghosting is ever observed). REPROJ_EPS2 is
// the squared world-space distance (voxels²) within which a reprojected sample
// is accepted as the same surface.
const REPROJECT_LIGHTING: bool = true;
const REPROJ_EPS2: f32 = 0.5;

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

// Water-top rest gradient (terrace/level slope without the wave field) of
// the LAST water surface the primary trace() resolved, transported to the
// deferred record write in cs_main. A private var instead of a Hit field on
// purpose: growing Hit bloats registers in every tracer (trace_no_water
// runs per reflection ray) - measured +1.8 ms on the water scenario.
// trace() resets it, so a far water cube-top can never read a stale value.
var<private> water_grad_rest: vec2<f32> = vec2<f32>(0.0);

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
// and cs_compose (underwater post-effect). Deliberate approximation:
// own-level height + the field at the eye's XZ, NOT the full corner patch -
// this runs per pixel, and the 16 corner probes would be full-screen cost
// while swimming. It only diverges from the drawn patch at terrace lips,
// where a few cm of eye-height mismatch in the underwater tint is
// imperceptible.
fn camera_in_water() -> bool {
    let cam_voxel_chk = vec3<i32>(floor(camera.origin));
    let cam_mat_chk = voxel_material_at(cam_voxel_chk);
    var in_water = is_water_mat(cam_mat_chk);
    if (in_water && !is_water_mat(voxel_material_at(cam_voxel_chk + vec3<i32>(0, 1, 0)))) {
        let lf = f32(cam_mat_chk - MAT_WATER_L1 + 1u) * 0.125;
        let f = water_field(camera.origin.xz, camera.time);
        let s = clamp((WATER_BASE + f.x) * lf, WATER_MIN_H, 1.0);
        let lp_y = camera.origin.y - f32(cam_voxel_chk.y);
        in_water = lp_y <= s;
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
            // Top surface: the carried rest-gradient (terrace/level slope,
            // zero on flat lakes) plus the exact per-pixel field gradient
            // (the same field the patches displace by).
            let p_hit = camera.origin + dir * hit.t_hit;
            let g = unpack2x16float(rec.z);
            let f = water_field(p_hit.xz, camera.time);
            hit.normal = normalize(vec3<f32>(-(g.x + f.y), 1.0, -(g.y + f.z)));
        } else {
            hit.normal = decode_face_normal(rec.z);
        }
        // ONE shade_water_top call site: it inlines the reflection and
        // refraction traces, and duplicating it doubles cs_transparent's
        // code size (measured ~+1.8 ms on the water scenario).
        col = shade_water_top(hit, camera.origin, dir);
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
    let pix_jitter = ign(f32(gid.x), f32(gid.y), camera.time * 60.0);

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
    if (cam_in_water) {
        // Skip beam-skip when underwater — beam pre-pass doesn't know about
        // the camera being inside water and may have advanced past real geo.
        hit = trace_no_water(camera.origin, dir);
    } else {
        hit = trace(ray_origin, dir);
        if (hit.hit) {
            hit.t_hit = hit.t_hit + beam_skip;
        }
    }
    var col: vec3<f32>;
    if (PROFILE_FLAT) {
        col = select(sky(dir), palette[hit.mat].rgb, hit.hit);
        textureStore(output_tex, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(col, 1.0));
        return;
    }
    // G-buffer for the shadow/AO reprojection cache; sentinel pos = never reused.
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
                                   pack2x16float(water_grad_rest), 0u);
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
            // Solid terrain faces have stable (view-independent) shadow + AO, so
            // reproject them from last frame's G-buffer and reuse on a position
            // match. Leaf cutout faces are stable cube faces too (axis >= 0);
            // only oblique sub-voxel hits (grass/flower cross-quads) re-trace.
            let hitpos_rel = (camera.origin - vec3<f32>(camera.world_origin)) + dir * hit.t_hit;
            let cacheable = hit.last_axis >= 0;
            var light = vec2<f32>(0.0);
            var reuse = false;
            if (REPROJECT_LIGHTING && cacheable && camera.reproject_lighting > 0.5) {
                let abs_pos = hitpos_rel + vec3<f32>(camera.world_origin);
                let d = abs_pos - camera.prev_origin;
                let pz = dot(d, camera.prev_forward);
                if (pz > 0.01) {
                    let aspect = camera.resolution.x / camera.resolution.y;
                    let px = dot(d, camera.prev_right);
                    let py = dot(d, camera.prev_up);
                    let ndc = vec2<f32>(px / (pz * camera.tan_half_fov * aspect),
                                        py / (pz * camera.tan_half_fov));
                    let uvp = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
                    if (uvp.x >= 0.0 && uvp.x < 1.0 && uvp.y >= 0.0 && uvp.y < 1.0) {
                        let pc = vec2<i32>(uvp * camera.resolution);
                        let g = textureLoad(light_in, pc, 0);
                        let dpos = g.xyz - hitpos_rel;
                        if (dot(dpos, dpos) < REPROJ_EPS2) {
                            light = unpack2x16float(bitcast<u32>(g.w));
                            reuse = true;
                        }
                    }
                }
            }
            col = shade(hit, camera.origin, dir, pix_jitter, reuse, &light);
            if (cacheable) {
                gbuf = vec4<f32>(hitpos_rel, bitcast<f32>(pack2x16float(light)));
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
    // the slab (one IGN-jittered sample; TAA averages the penumbra over
    // frames). Denser cloud overhead = deeper shade, and the shadow field
    // drifts with the clouds by construction - it can never disagree with
    // what is visibly above. Per-frame full-screen only (mechanism rule):
    // this term animates every frame and must never enter a tile-gated
    // pass. Skipped for near-horizontal sun (shadows would race across
    // the map) and scaled by sun intensity so night is untouched.
    if (t_hit < 1.0e8) {
        let s = sun_dir();
        let s_int = sun_intensity(s);
        if (s_int > 0.0 && s.y > 0.05) {
            let p_ground = camera.origin + dir * t_hit;
            let st_in = (CLOUD_BASE - p_ground.y) / s.y;
            let st_out = (CLOUD_TOP - p_ground.y) / s.y;
            if (st_in > 0.0) {
                let jj = ign(f32(gid.x), f32(gid.y), camera.time * 60.0 + 17.0);
                let ps = p_ground + s * mix(st_in, st_out, 0.15 + 0.7 * jj);
                let d = cloud_density(ps, camera.time);
                let occl = 1.0 - exp(-d * 3.0);
                col = col * (1.0 - occl * 0.42 * s_int);
            }
        }
    }

    let uv_cloud = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / camera.resolution;
    let clouds = textureSampleLevel(cloud_in, cloud_samp, uv_cloud, 0.0);
    if (t_hit >= cloud_slab_near(dir)) {
        col = col * (1.0 - clouds.a) + clouds.rgb;
    }

    col += god_rays(camera.origin, dir, min(t_hit, 200.0), vec2<f32>(f32(gid.x), f32(gid.y)));

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
        || m == MAT_LEAF_FRINGE || m == MAT_TALL_GRASS_DRY;
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
        || m == MAT_TALL_GRASS_DRY;
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
fn wind_gust(p_xz: vec2<f32>, wdir: vec2<f32>) -> f32 {
    let s = dot(p_xz, wdir);
    let front  = 0.5 + 0.5 * sin(s * 0.020 - camera.time * 0.9);
    let ripple = 0.5 + 0.5 * sin(s * 0.11  - camera.time * 2.1);
    return 0.25 + 0.75 * front * (0.6 + 0.4 * ripple);
}

fn wind_offset(voxel_min: vec3<f32>, phase: f32, base_amp: f32) -> vec2<f32> {
    let wdir = wind_dir_now();
    // Gust is a function of voxel_min only, so both cross planes and both
    // tuft quads of one block share a single shear (the anti-X-split
    // contract in sprite_cross_hit).
    let strength = base_amp * wind_gust(voxel_min.xz, wdir)
        * (0.70 + 0.30 * sin(camera.time * 0.55 + phase));
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
const LEAF_CLOUD_T: f32 = 32.0;

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
        let scale = 0.92 + h3 * 0.48;
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
        // Dark base -> bright tip, darker secondary texels, per-voxel hue.
        let b = (0.60 + 0.55 * v) * select(1.0, 0.72, val == 2u);
        // Couple the blade hue 60% toward the ground-block palette so grass
        // tracks its terrain colour from the one palette source instead of
        // floating over it.
        let ground = mix(vec3<f32>(1.0),
                         palette[MAT_GRASS].rgb / max(palette[MAT_TALL_GRASS].rgb, vec3<f32>(1e-3)),
                         0.6);
        return vec3<f32>(b) * (0.85 + vh * 0.30) * ground;
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
        return vec3<f32>(0.950, 0.150, 0.120);                    // red petals
    }
    if (sprite == SPR_DAISY) {
        if (val == 3u) { return vec3<f32>(1.150, 0.850, 0.150); } // yellow centre
        return vec3<f32>(0.950, 0.950, 0.900);                    // white petals
    }
    if (sprite == SPR_TULIP) {
        if (val == 3u) { return vec3<f32>(1.100, 0.550, 0.200); } // lit rim
        return vec3<f32>(1.000, 0.320, 0.100);                    // red-orange cup
    }
    if (sprite == SPR_CORNFLOWER) {
        if (val == 3u) { return vec3<f32>(0.480, 0.620, 1.100); } // bright fringe
        return vec3<f32>(0.240, 0.350, 0.950);                    // cornflower blue
    }
    // Dandelion.
    if (val == 3u) { return vec3<f32>(1.150, 1.000, 0.250); }     // bright core
    return vec3<f32>(1.050, 0.820, 0.120);                        // yellow puff
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
    // stems must reach the ground plane at full sprite height.
    var hs = 1.0;
    if (mat != MAT_FLOWER) { hs = 0.70 + fract(vh * 4.0) * 0.30; }

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

fn foliage_subvoxel(voxel: vec3<i32>, origin: vec3<f32>, dir: vec3<f32>, mat: u32) -> SubHit {
    var hit: SubHit;
    if (mat == MAT_TALL_GRASS || mat == MAT_FLOWER || mat == MAT_TALL_GRASS_DRY) {
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
fn cloud_density(p: vec3<f32>, t: f32) -> f32 {
    let pa = p * 0.0055 + vec3<f32>(t * 0.06, 0.0, t * 0.035);
    // Cumulus coverage: a low-freq clump field carved by a smoothstep
    // threshold into DISTINCT clouds with genuinely clear sky between -
    // never a linear coverage ramp, which spreads a translucent stratus
    // veil everywhere. The mid-freq term keeps clump outlines irregular.
    let cov_lo = vnoise3(pa * 0.70);
    let cov_mid = vnoise3(pa * 2.1);
    let cov = smoothstep(0.54, 0.66, cov_lo * 0.75 + cov_mid * 0.25);
    if (cov <= 0.0) { return 0.0; }
    // Body: 4 octaves of fbm. Vertical noise is scaled finer so a horizontal
    // slice doesn't look like a flat layer when viewed sideways.
    let pb = vec3<f32>(pa.x, pa.y * 3.5, pa.z);
    let n1 = vnoise3(pb);
    let n2 = vnoise3(pb * 2.7);
    let n3 = vnoise3(pb * 6.3);
    let n4 = vnoise3(pb * 13.1);
    let body = n1 * 0.50 + n2 * 0.28 + n3 * 0.15 + n4 * 0.07;
    // Cumulus profile: sharp flat bottom, and a dome whose height rises
    // with clump strength - weak edges stay low, cores tower to the slab
    // top, which reads as rounded cauliflower heads instead of a layer.
    let h = clamp((p.y - CLOUD_BASE) / max(1.0, CLOUD_TOP - CLOUD_BASE), 0.0, 1.0);
    let bottom_fade = smoothstep(0.0, 0.08, h);
    let dome = 1.0 - smoothstep(0.30 + 0.60 * cov, 1.0, h);
    let envelope = bottom_fade * dome;
    let d = (body - 0.32) * cov * 5.0 * envelope;
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

// ---------- VOXEL WATER: connected per-corner surface -----------------------
// The water surface is real sub-voxel geometry: every surface water voxel
// (no water above) renders a BILINEAR PATCH over its four top-corner
// heights. Each corner is derived from the up-to-4 water columns sharing it
// (own + 2 orthogonal + 1 diagonal):
//   - PIN rule: any corner-sharing column with water at y+1 pins the corner
//     to exactly 1.0 (no wave term), so the surface rises to meet the upper
//     cube's bottom with zero gap - this is what knits diagonally-touching
//     and different-height water into one connected surface.
//   - otherwise the corner is the MEAN level_frac of the corner-sharing
//     water columns at the same y (non-water columns don't contribute, so
//     shores keep their wall behaviour), scaled into WATER_BASE and
//     displaced by the wave field sampled AT the corner's world XZ.
// Shared corners are accumulated in a fixed world order, so any of the 4
// cells sharing a corner computes bitwise-identical heights: cross-cell
// continuity is exact by construction, and the below-waterline wall branch
// provably never fires on internal water-water faces (a patch restricted to
// a cell edge is the lerp of that edge's shared corners). Flat lakes reduce
// algebraically to the previous centre-sampled plane. Shading uses the
// carried rest-gradient plus the exact per-pixel field normal
// (cs_transparent); only the ray-patch intersection is per-cell geometry.
const WATER_DETAIL_T: f32 = 96.0;   // beyond this, water is a plain cube top
// Corner-connected patches only this near: a 1-voxel terrace step subtends
// >2 px inside this range and the connection is visible; beyond it the
// cheap centre-plane facet takes over (water_subvoxel_far) - horizon-
// skimming rays traverse ~50 surface cells per pixel and paying the
// 8-probe pin ring for each measured +25% on the water scenario.
const WATER_NEAR_T: f32 = 48.0;
const WATER_BASE: f32 = 0.72;       // resting surface height inside the cell
const WATER_MIN_H: f32 = 0.02;      // corner floor (keeps the patch off the cell floor)
// Conservative bound on |wave field height|: the four amplitudes sum to
// ~0.108 voxels (see wave_param), padded a little. Used by the grazing-ray
// fast-out: with no pinned corner the surface cannot exceed
// WATER_BASE + WATER_WAVE_MAX.
const WATER_WAVE_MAX: f32 = 0.12;

struct WaterSubHit {
    hit: bool,
    t_hit: f32,
    normal: vec3<f32>,
    // Rest-only surface gradient at the hit (terrace/level slope without the
    // wave field), carried to the deferred pass via Hit.aux.
    grad_rest: vec2<f32>,
};

// Far-tier surface (WATER_NEAR_T..WATER_DETAIL_T): the previous centre-
// sampled tilted facet. At this distance a cell is a few pixels, terrace
// connectivity is sub-pixel, and the facet needs no neighbour probes (the
// caller already did the interior +Y check). Margin-clamped inside the
// cell so it can never poke into neighbours; the near/far boundary
// mismatch is the corner-vs-centre field delta (~1e-2 voxels), invisible
// at 48+ units.
fn water_subvoxel_far(
    voxel: vec3<i32>, origin: vec3<f32>, dir: vec3<f32>, m: u32,
    entry_n: vec3<f32>, t_entry: f32, t_exit: f32,
) -> WaterSubHit {
    var out: WaterSubHit;
    out.hit = false;
    out.grad_rest = vec2<f32>(0.0);
    let level_frac = f32(m - MAT_WATER_L1 + 1u) * 0.125;
    let vc = vec2<f32>(f32(voxel.x) + 0.5, f32(voxel.z) + 0.5);
    let f = water_field(vc, camera.time);
    let slope = clamp(f.yz, vec2<f32>(-0.30), vec2<f32>(0.30)) * level_frac;
    let margin = 0.5 * (abs(slope.x) + abs(slope.y)) + 0.02;
    let h = clamp(WATER_BASE + f.x, margin, 1.0 - margin) * level_frac;
    let vmin = vec3<f32>(f32(voxel.x), f32(voxel.y), f32(voxel.z));
    let p0 = origin + dir * t_entry - vmin;
    let s0 = h + slope.x * (p0.x - 0.5) + slope.y * (p0.z - 0.5);
    if (p0.y <= s0 + 1e-4) {
        out.hit = true;
        out.t_hit = t_entry;
        out.normal = entry_n;
        return out;
    }
    let denom = dir.y - slope.x * dir.x - slope.y * dir.z;
    if (denom < -1e-6) {
        let s = (p0.y - s0) / (-denom);
        if (t_entry + s < t_exit) {
            out.hit = true;
            out.t_hit = t_entry + s;
            out.normal = normalize(vec3<f32>(-slope.x, 1.0, -slope.y));
            return out;
        }
    }
    return out;
}

// One corner's (h, h_rest). `lf9`/`up9` describe the 3x3 column
// neighbourhood: lf9[(ox+1)+(oz+1)*3] = level_frac of the column at lateral
// offset (ox, oz) if it holds water at this y (0 otherwise); up9 bit i set =
// that column holds water at y+1. Corner (cx, cz) in {0,1}^2.
// The k-loop enumerates the corner's 4 sharing columns in increasing z then
// x WORLD order - keep it that way, bitwise cross-cell equality depends on
// the accumulation order.
fn water_corner_h(lf9: ptr<function, array<f32, 9>>, up9: u32, voxel: vec3<i32>, cx: i32, cz: i32) -> vec2<f32> {
    var pinned = false;
    var sum = 0.0;
    var cnt = 0.0;
    for (var k: i32 = 0; k < 4; k = k + 1) {
        let ox = cx - 1 + (k & 1);
        let oz = cz - 1 + (k >> 1);
        let idx = (ox + 1) + (oz + 1) * 3;
        if (((up9 >> u32(idx)) & 1u) == 1u) { pinned = true; }
        let lf = (*lf9)[idx];
        if (lf > 0.0) {
            sum = sum + lf;
            cnt = cnt + 1.0;
        }
    }
    if (pinned) { return vec2<f32>(1.0, 1.0); }
    let avg = sum / cnt; // own column always counts: cnt >= 1
    let f = water_field(vec2<f32>(f32(voxel.x + cx), f32(voxel.z + cz)), camera.time);
    let h_rest = clamp(WATER_BASE * avg, WATER_MIN_H, 1.0);
    let h = clamp((WATER_BASE + f.x) * avg, WATER_MIN_H, 1.0);
    return vec2<f32>(h, h_rest);
}

// Sub-voxel water surface for one cell the DDA landed in. `entry_n`/`t_entry`
// describe the cell's entry face, `t_exit` the exit crossing; `slot_v`/`bp`/
// `bi` are the DDA's current slot voxel and brick so neighbour probes can
// take the register-resident fast path. Misses (ray passes above the patch)
// fall through to the next DDA cell.
fn water_subvoxel(
    voxel: vec3<i32>, origin: vec3<f32>, dir: vec3<f32>, m: u32,
    entry_n: vec3<f32>, t_entry: f32, t_exit: f32,
    slot_v: vec3<i32>, bp: vec3<i32>, bi: i32,
) -> WaterSubHit {
    var out: WaterSubHit;
    out.hit = false;
    out.grad_rest = vec2<f32>(0.0);
    // Interior cell (more water above): a plain cube. Its exposed faces are
    // vertical water walls / undersides.
    if (is_water_mat(neighbor_material(voxel, slot_v, bp, bi, vec3<i32>(0, 1, 0)))) {
        out.hit = true;
        out.t_hit = t_entry;
        out.normal = entry_n;
        return out;
    }
    // Far tier: centre-plane facet, no neighbourhood probes.
    if (t_entry > WATER_NEAR_T) {
        return water_subvoxel_far(voxel, origin, dir, m, entry_n, t_entry, t_exit);
    }

    // Stage A - pin probes: water at y+1 in the 8 lateral columns (the own
    // column has none, the interior check above just returned).
    var up9: u32 = 0u;
    for (var oz: i32 = -1; oz <= 1; oz = oz + 1) {
        for (var ox: i32 = -1; ox <= 1; ox = ox + 1) {
            if (ox == 0 && oz == 0) { continue; }
            if (is_water_mat(neighbor_material(voxel, slot_v, bp, bi, vec3<i32>(ox, 1, oz)))) {
                up9 = up9 | (1u << u32((ox + 1) + (oz + 1) * 3));
            }
        }
    }
    // Grazing fast-out: no pinned corner means the surface cannot exceed
    // WATER_BASE + WATER_WAVE_MAX (level averaging only lowers it). y is
    // monotone along the ray, so its min over the cell is at an endpoint.
    let vmin = vec3<f32>(f32(voxel.x), f32(voxel.y), f32(voxel.z));
    let y_in = origin.y + dir.y * t_entry - vmin.y;
    let y_out = origin.y + dir.y * t_exit - vmin.y;
    if (up9 == 0u && min(y_in, y_out) > WATER_BASE + WATER_WAVE_MAX) {
        return out;
    }

    // Stage B - level probes: level_frac of the 3x3 columns at this y.
    var lf9: array<f32, 9>;
    lf9[4] = f32(m - MAT_WATER_L1 + 1u) * 0.125; // own column
    for (var oz: i32 = -1; oz <= 1; oz = oz + 1) {
        for (var ox: i32 = -1; ox <= 1; ox = ox + 1) {
            if (ox == 0 && oz == 0) { continue; }
            let lm = neighbor_material(voxel, slot_v, bp, bi, vec3<i32>(ox, 0, oz));
            if (is_water_mat(lm)) {
                lf9[(ox + 1) + (oz + 1) * 3] = f32(lm - MAT_WATER_L1 + 1u) * 0.125;
            }
        }
    }

    // The four corner heights (h, h_rest) and the bilinear coefficients
    // S(x,z) = h00 + a1 x + a2 z + a3 xz over the unit cell.
    let c00 = water_corner_h(&lf9, up9, voxel, 0, 0);
    let c10 = water_corner_h(&lf9, up9, voxel, 1, 0);
    let c01 = water_corner_h(&lf9, up9, voxel, 0, 1);
    let c11 = water_corner_h(&lf9, up9, voxel, 1, 1);
    let a1 = c10.x - c00.x;
    let a2 = c01.x - c00.x;
    let a3 = c00.x - c10.x - c01.x + c11.x;
    let r1 = c10.y - c00.y;
    let r2 = c01.y - c00.y;
    let r3 = c00.y - c10.y - c01.y + c11.y;

    let p0 = origin + dir * t_entry - vmin;
    let s0 = c00.x + a1 * p0.x + a2 * p0.z + a3 * p0.x * p0.z;
    if (p0.y <= s0 + 1e-4) {
        // Entered below the waterline: the entry face IS the water surface -
        // a side wall at a shore/terrace drop, or the underside. Shared
        // corners guarantee this never fires between two same-y water cells.
        out.hit = true;
        out.t_hit = t_entry;
        out.normal = entry_n;
        out.grad_rest = vec2<f32>(r1 + r3 * p0.z, r2 + r3 * p0.x);
        return out;
    }
    // Entered above the patch: g(s) = y(s) - S(x(s), z(s)) is an exact
    // quadratic in the ray parameter; C = g(0) > 0, so the smallest root in
    // range is the downward crossing. Stable q-form roots; |A| ~ 0 falls
    // back to the plane case (today's math shape).
    let a_q = -a3 * dir.x * dir.z;
    let b_q = dir.y - a1 * dir.x - a2 * dir.z - a3 * (p0.x * dir.z + p0.z * dir.x);
    let c_q = p0.y - s0;
    var s_hit = -1.0;
    let s_max = t_exit - t_entry;
    if (abs(a_q) < 1e-7) {
        if (b_q < -1e-6) {
            let s = c_q / (-b_q);
            if (s < s_max) { s_hit = s; }
        }
    } else {
        let disc = b_q * b_q - 4.0 * a_q * c_q;
        if (disc >= 0.0) {
            let q = -0.5 * (b_q + sign(b_q) * sqrt(disc));
            let ra = q / a_q;
            let rb = c_q / q;
            let lo = min(ra, rb);
            let hi = max(ra, rb);
            var s = -1.0;
            if (lo > 0.0) { s = lo; } else if (hi > 0.0) { s = hi; }
            if (s > 0.0 && s < s_max) { s_hit = s; }
        }
    }
    if (s_hit >= 0.0) {
        let ph = p0 + dir * s_hit;
        let grad = vec2<f32>(a1 + a3 * ph.z, a2 + a3 * ph.x);
        out.hit = true;
        out.t_hit = t_entry + s_hit;
        out.normal = normalize(vec3<f32>(-grad.x, 1.0, -grad.y));
        out.grad_rest = vec2<f32>(r1 + r3 * ph.z, r2 + r3 * ph.x);
        return out;
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

fn trace_no_water(origin: vec3<f32>, dir: vec3<f32>) -> Hit {
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

fn shade_water_top(hit: Hit, origin: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let p_hit = origin + dir * hit.t_hit;
    // The plate/wall normal is real (quantised) geometry now — shading uses it
    // directly instead of a per-pixel Gerstner fake. Every pixel of a plate
    // shares one normal, so the secondary rays below stay warp-coherent.
    let n = hit.normal;
    let s = sun_dir();
    let sc = sun_color(s);
    // Per-pixel jitter substitute for shading-of-reflections — derive from
    // hit position since we're not in cs_main scope.
    let jit = fract(p_hit.x * 17.0 + p_hit.z * 23.0 + camera.time * 13.0);
    // Secondary rays don't use the reprojection cache.
    var no_cache = vec2<f32>(0.0);

    // ---- reflection off the plate ----
    // trace_no_water: the reflection origin sits INSIDE the water cell (the
    // plate is below the cube top), so a plain trace would hit the very cell
    // it started in. Skipping water also keeps the reflection showing terrain
    // and sky rather than the surface's own neighbouring plates.
    let refl_dir = reflect(dir, n);
    let refl_origin = p_hit + n * 0.01;
    let refl_hit = trace_no_water(refl_origin, refl_dir);
    var refl_col: vec3<f32>;
    if (refl_hit.hit) {
        refl_col = shade(refl_hit, refl_origin, refl_dir, jit, false, &no_cache);
    } else {
        refl_col = sky(refl_dir);
    }

    // ---- refraction: primary ray bent into the water, trace through it ----
    // Snell's law via WGSL `refract`. eta = n_air / n_water ≈ 1/1.33.
    let eta = 1.0 / 1.33;
    var refr_dir = refract(dir, n, eta);
    // Total internal reflection would return zero; fall back to dir.
    if (length(refr_dir) < 0.01) { refr_dir = dir; }
    let refr_origin = p_hit + dir * 0.001; // step inside the water column
    let under = trace_no_water(refr_origin, refr_dir);
    var under_col: vec3<f32>;
    if (under.hit) {
        under_col = shade(under, refr_origin, refr_dir, jit, false, &no_cache);
    } else {
        under_col = sky(refr_dir) * 0.6;
    }
    // Beer-Lambert absorption — red and green are eaten faster than blue.
    let depth = max(0.0, under.t_hit);
    let absorb = vec3<f32>(0.55, 0.25, 0.10); // per-unit-distance attenuation
    let transmittance = exp(-absorb * depth);
    // Water tint modulated by ambient + a bit of sun colour, so the water
    // body actually goes dark at night instead of staying daytime-blue.
    let tint_base = vec3<f32>(0.10, 0.32, 0.42);
    let water_tint = tint_base * (ambient_color() * 1.6 + sc * 0.20);
    let refr_col = under_col * transmittance + water_tint * (1.0 - transmittance.x);

    // ---- Fresnel mix of reflection and refraction ----
    let cos_theta = clamp(dot(-dir, n), 0.0, 1.0);
    let f0 = 0.02;
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);

    // ---- specular sun glint (sharper for stronger highlight) ----
    let h = normalize(s - dir);
    let spec = pow(max(0.0, dot(n, h)), 256.0);
    var shadow = 0.0;
    if (sun_intensity(s) > 0.0 && dot(n, s) > 0.0) {
        // The hit sits inside the water cell (plate below the cube top), and
        // water voxels count as solid for trace_any — a shadow ray from p_hit
        // would self-occlude. Lift the origin to just above the cell's top
        // face: the column above a surface cell is air by construction.
        let glint_origin = vec3<f32>(p_hit.x, floor(p_hit.y) + 1.001, p_hit.z);
        shadow = select(1.0, 0.0, trace_any(glint_origin, s, SHADOW_MAX_DIST));
    }

    // ---- shoreline foam: triggered by shallow water (under.t_hit small) ----
    // The closer the underwater hit, the brighter the white foam contribution.
    // Wave-crest noise modulates so foam looks like spray, not a flat ring.
    var foam = 0.0;
    if (under.hit && under.t_hit < 2.5) {
        let shore = 1.0 - clamp(under.t_hit / 2.5, 0.0, 1.0);
        // Field heights are in voxels (±~0.11), so scale the crest gate up.
        let crest = clamp(water_field(p_hit.xz, camera.time).x * 9.0 + 0.5, 0.0, 1.0);
        foam = shore * crest * 0.85;
    }

    // ---- caustics: brighten the underwater colour where the surface wave
    // gradient focuses light. Approximation: |∇h| → dispersion factor, where
    // small gradient = focused beams. Only applies to the refracted column.
    let caustic = 0.6 + 0.8 * pow(max(0.0, n.y), 18.0);

    var col = mix(refr_col * caustic, refl_col, fresnel) + sc * spec * shadow * 1.4;
    // Foam colour also dims at night — at dawn/dusk it picks up the warm
    // sun tint, at noon it's bright white, at night it fades into ambient.
    let foam_col = ambient_color() * 1.5 + sc * 0.50;
    col = mix(col, foam_col, foam);
    let fog_t = clamp(hit.t_hit / 280.0, 0.0, 0.85);
    return mix(col, fog_atmospheric(dir), fog_t);
}

// Thin wrapper: full shade with no lighting reuse (used by reflection/refraction
// secondary rays, which aren't cached).
// `reuse_light`: when true, the shadow + AO terms are taken from *light (the
// reprojected cache) instead of being traced. When false they are computed and
// Terrain materials whose TOP faces cross-fade into each other at shared
// edges (sand beaches into grass, snow lines into stone, ...).
fn is_blend_mat(m: u32) -> bool {
    return m == 1u || m == 2u || m == 3u || m == 4u || m == 15u;
}

const BLEND_T: f32 = 64.0;   // beyond this an edge fade is sub-pixel
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

// written back into *light so the caller can store them for next frame.// written back into *light so the caller can store them for next frame.
fn shade(
    hit: Hit, origin: vec3<f32>, dir: vec3<f32>, pix_jit: f32,
    reuse_light: bool, light: ptr<function, vec2<f32>>,
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
    var ao: f32;
    if (reuse_light) { ao = (*light).y; }
    else { ao = select(compute_ao(hit, origin, dir), 1.0, skip_ao); }

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
        let amp = select(0.35, 0.45, hit.mat == MAT_FLOWER || hit.mat == MAT_TALL_GRASS
                                   || hit.mat == MAT_TALL_GRASS_DRY);
        n.x += sway * amp;
        n.z += sway * amp * 0.7;
        n = normalize(n);
    }
    // Rigid cube tops (MAT_GRASS included) get NO flutter at all: any
    // time-varying shading on geometry that visibly cannot move reads as a
    // shadow passing over it. Grass motion is carried by the tall-grass
    // cross-quad geometry standing ON the block, never by the block face.

    let s = sun_dir();
    let s_int = sun_intensity(s);
    let p_off = p_hit + n * 0.001;
    let n_dot_l = max(0.0, dot(n, s));
    var shadow_term = 0.0;
    if (reuse_light) {
        shadow_term = (*light).x;
    } else if (n_dot_l > 0.0 && s_int > 0.0) {
        // ONE jittered shadow ray (was 2). The per-pixel + per-frame jitter
        // (pix_jit rotates each frame) plus the TAA history accumulation average
        // the single sample into a soft penumbra over time — at half the cost.
        // Shadows are the single most expensive per-pixel term, so this is the
        // biggest shading win.
        let golden = 2.39996323; // 137.5° in radians
        let cone = 0.07;
        let theta = pix_jit * golden;
        let radius = cone * sqrt(pix_jit * 0.5);
        // Offset in the plane perpendicular to the sun so the penumbra is
        // uniform regardless of sun azimuth.
        var tangent = normalize(cross(s, vec3<f32>(0.0, 1.0, 0.0)));
        if (length(cross(s, vec3<f32>(0.0, 1.0, 0.0))) < 0.01) {
            tangent = vec3<f32>(1.0, 0.0, 0.0);
        }
        let bitangent = cross(s, tangent);
        let off = (tangent * cos(theta) + bitangent * sin(theta)) * radius;
        let ss = normalize(s + off);
        shadow_term = select(0.0, 1.0, !trace_any(p_off, ss, SHADOW_MAX_DIST));
    }
    // Hand the freshly-computed terms back so the caller can cache them.
    if (!reuse_light) { *light = vec2<f32>(shadow_term, ao); }

    let direct = sun_color(s) * (n_dot_l * shadow_term);
    let ambient = ambient_color() * ao;
    let lit = base * (direct + ambient);

    let fog_t = clamp(hit.t_hit / 280.0, 0.0, 0.85);
    return mix(lit, fog_atmospheric(dir), fog_t);
}

// Glass — Fresnel reflection + per-channel refraction (chromatic dispersion),
// Total Internal Reflection handling, and a specular sun glint. The cyan
// tint compounds with travel distance for chunky glass blocks.
fn shade_glass(hit: Hit, origin: vec3<f32>, dir: vec3<f32>) -> vec3<f32> {
    let p_hit = origin + dir * hit.t_hit;
    let n = hit.normal;
    let s = sun_dir();
    let sc = sun_color(s);
    let jit = fract(p_hit.x * 17.0 + p_hit.z * 23.0 + camera.time * 13.0);
    // Secondary rays don't use the reprojection cache.
    var no_cache = vec2<f32>(0.0);

    let refl_dir = reflect(dir, n);
    let refl_origin = p_hit + n * 0.01;
    let refl_hit = trace(refl_origin, refl_dir);
    var refl_col: vec3<f32>;
    if (refl_hit.hit) {
        refl_col = shade(refl_hit, refl_origin, refl_dir, jit, false, &no_cache);
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
        let fog_t = clamp(hit.t_hit / 280.0, 0.0, 0.85);
        return mix(refl_col, fog_atmospheric(dir), fog_t);
    }
    if (length(refr_dir_r) < 0.01) { refr_dir_r = refr_dir_g; }
    if (length(refr_dir_b) < 0.01) { refr_dir_b = refr_dir_g; }

    let refr_origin = p_hit + dir * 0.001;
    var glass_col: vec3<f32>;
    if (cos_theta_pre > 0.92) {
        // Near head-on: dispersion invisible — single trace, save 2/3 cost.
        let under = trace_no_water(refr_origin, refr_dir_g);
        var under_col: vec3<f32>;
        if (under.hit) { under_col = shade(under, refr_origin, refr_dir_g, jit, false, &no_cache); }
        else { under_col = sky(refr_dir_g); }
        let depth = max(0.0, under.t_hit);
        let tint = vec3<f32>(0.05, 0.02, 0.02) * depth;
        glass_col = under_col * exp(-tint);
    } else {
        let ur = trace_no_water(refr_origin, refr_dir_r);
        let ug = trace_no_water(refr_origin, refr_dir_g);
        let ub = trace_no_water(refr_origin, refr_dir_b);
        var cr = select(sky(refr_dir_r).r, shade(ur, refr_origin, refr_dir_r, jit, false, &no_cache).r, ur.hit);
        var cg = select(sky(refr_dir_g).g, shade(ug, refr_origin, refr_dir_g, jit, false, &no_cache).g, ug.hit);
        var cb = select(sky(refr_dir_b).b, shade(ub, refr_origin, refr_dir_b, jit, false, &no_cache).b, ub.hit);
        let depth_g = max(0.0, ug.t_hit);
        let tint = vec3<f32>(0.05, 0.02, 0.02) * depth_g;
        glass_col = vec3<f32>(cr, cg, cb) * exp(-tint);
    }

    // Specular sun glint on the glass face — bright pinpoint highlight when
    // the surface aligns the sun reflection toward the camera.
    let h_vec = normalize(s - dir);
    let spec = pow(max(0.0, dot(n, h_vec)), 200.0);
    var shadow = 0.0;
    if (sun_intensity(s) > 0.0 && dot(n, s) > 0.0) {
        shadow = select(1.0, 0.0, trace_any(refl_origin, s, SHADOW_MAX_DIST));
    }

    let cos_theta = clamp(dot(-dir, n), 0.0, 1.0);
    let f0 = 0.04;
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - cos_theta, 5.0);

    let fog_t = clamp(hit.t_hit / 280.0, 0.0, 0.85);
    let combined = mix(glass_col, refl_col, fresnel) + sc * spec * shadow * 1.2;
    return mix(combined, fog_atmospheric(dir), fog_t);
}

// Volumetric clouds: raymarch a horizontal slab. Altitude lowered (was
// 200-250) so clouds sit inside the world Y = 192 — view rays past
// mountains can actually reach the cloud band instead of stopping at the
// world ceiling.
const CLOUD_BASE: f32 = 165.0;
const CLOUD_TOP:  f32 = 192.0;

fn render_clouds(origin: vec3<f32>, dir: vec3<f32>, t_terrain: f32, pix: vec2<f32>) -> vec4<f32> {
    // Slab intersection. A horizontal ray (|dir.y| ~ 0) gets nothing because
    // the slab is thin compared to the marchable distance.
    if (abs(dir.y) < 1e-3) { return vec4<f32>(0.0); }
    let inv_dy = 1.0 / dir.y;
    var t_in  = (CLOUD_BASE - origin.y) * inv_dy;
    var t_out = (CLOUD_TOP  - origin.y) * inv_dy;
    if (t_in > t_out) { let tmp = t_in; t_in = t_out; t_out = tmp; }
    let t_start = max(t_in, 0.0);
    let t_end   = min(t_out, t_terrain);
    if (t_end <= t_start + 0.5) { return vec4<f32>(0.0); }
    // Distance-clamp the slab — beyond this clouds blend into atmospheric fog.
    let t_far_clamp = min(t_end, t_start + 600.0);

    let s = sun_dir();
    let sc = sun_color(s);

    // Half the marching steps of the original full-res march — the TAA pass
    // temporally accumulates the result on a static camera, so the lower
    // per-frame sample count is upsampled over time instead of in one frame
    // (checklist: clouds at reduced res + temporal upsample).
    let N: i32 = 6;
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
        let d = cloud_density(p, camera.time);
        if (d < 0.01) { continue; }

        // 2 cone samples toward the sun for self-shadowing (TAA accumulates).
        var sun_dens: f32 = 0.0;
        for (var j: i32 = 1; j <= 2; j = j + 1) {
            let pj = p + s * f32(j) * 9.0;
            sun_dens = sun_dens + cloud_density(pj, camera.time);
        }
        let sun_t = exp(-sun_dens * 0.85);
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
fn god_rays(origin: vec3<f32>, dir: vec3<f32>, t_far: f32, pix: vec2<f32>) -> vec3<f32> {
    let s = sun_dir();
    let s_int = sun_intensity(s);
    if (s_int <= 0.0) { return vec3<f32>(0.0); }
    let cos_sun = dot(dir, s);
    // Normalize HG phase to a 0..~1 scale at g=0.7 — peak ≈ 0.65 forward, ≈ 0.014 back.
    let phase = phase_hg(cos_sun, 0.7) * 4.0;
    if (phase < 0.05) { return vec3<f32>(0.0); }

    let t_max = min(t_far, 140.0);
    if (t_max <= 1.0) { return vec3<f32>(0.0); }
    // Sample count scales with phase — looking right at the sun gets denser
    // sampling for a smooth halo; off-axis stays cheap.
    // Reduced step count; TAA accumulates the god-ray term across frames.
    let N: i32 = select(3, 6, phase > 0.40);
    let step_t = t_max / f32(N);
    let h = ign(pix.x, pix.y, camera.time * 60.0);
    var sum = 0.0;
    for (var i: i32 = 0; i < N; i = i + 1) {
        let t = (f32(i) + h) * step_t;
        let p = origin + dir * t;
        // God-ray shafts only need NEARBY occluders — a short occlusion cap lets
        // the hierarchical trace bail out far sooner than a full shadow ray.
        if (!trace_any(p + s * 0.5, s, GOD_RAY_OCCL_DIST)) {
            // Distance-weighted contribution: nearer scatter looks brighter.
            sum = sum + exp(-t * 0.008);
        }
    }
    let frac = sum / f32(N);
    return sun_color(s) * frac * phase * 0.22 * s_int;
}

// Stripped-down DDA — same hierarchy as `trace()` but returns the moment we
// know the ray is occluded. No normal / material work.
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
            if (m == MAT_LEAF_FRINGE) {
                // Invisible canopy fringe never occludes shadow rays.
            } else if (is_decoration_mat(m)) {
                // Ground decoration (grass tufts, flowers, straw): the near
                // cutout gives dappled micro-shadow; far away a tuft is 90%
                // air, and blocking as a solid cube stamped a square shadow
                // per tuft across every meadow. Far decorations don't block.
                if (t_cur <= FOLIAGE_NEAR_T) {
                    let fh = foliage_subvoxel(voxel, origin, dir, m);
                    if (fh.hit) { return true; }
                }
            } else if (is_foliage_mat(m)) {
                // Leaves: far canopies block as solid cubes (they really are
                // dense); near ones pay the cutout test.
                if (t_cur > FOLIAGE_NEAR_T) { return true; }
                let fh = foliage_subvoxel(voxel, origin, dir, m);
                if (fh.hit) { return true; }
            } else {
                return true;
            }
        }

        dda_step(&voxel, &slot_v, &t_max, &t_cur, &last_axis, step, t_delta);
    }
    return false;
}

// Bit-packed AO. We project the hit point onto the face we entered through,
// compute fractional (fa, fb) coords on that face, sample 4 corner AOs, and
// bilinear-interpolate. Each corner samples 3 neighbours (two side voxels
// and the diagonal) — classic "Minecraft" AO formula, but every lookup is a
// hierarchical bit test rather than a struct fetch.
fn compute_ao(hit: Hit, origin: vec3<f32>, dir: vec3<f32>) -> f32 {
    let p_hit = origin + dir * hit.t_hit;
    let v = hit.voxel;
    let na = hit.last_axis;
    if (na < 0) { return 1.0; }
    // n_dir is +1 or -1 — the outward-facing component of the normal axis.
    let n_dir = i32(hit.normal[na]);
    var n_off = vec3<i32>(0);
    if (na == 0) { n_off.x = n_dir; }
    else if (na == 1) { n_off.y = n_dir; }
    else { n_off.z = n_dir; }

    var da_pos: vec3<i32>;
    var db_pos: vec3<i32>;
    var fa: f32;
    var fb: f32;
    let local_frac = p_hit - vec3<f32>(f32(v.x), f32(v.y), f32(v.z));
    if (na == 0) {
        da_pos = vec3<i32>(0, 1, 0); db_pos = vec3<i32>(0, 0, 1);
        fa = local_frac.y; fb = local_frac.z;
    } else if (na == 1) {
        da_pos = vec3<i32>(1, 0, 0); db_pos = vec3<i32>(0, 0, 1);
        fa = local_frac.x; fb = local_frac.z;
    } else {
        da_pos = vec3<i32>(1, 0, 0); db_pos = vec3<i32>(0, 1, 0);
        fa = local_frac.x; fb = local_frac.y;
    }
    let da_neg = -da_pos;
    let db_neg = -db_pos;
    let base = v + n_off;

    let ao00 = ao_corner(base, da_neg, db_neg);
    let ao10 = ao_corner(base, da_pos, db_neg);
    let ao01 = ao_corner(base, da_neg, db_pos);
    let ao11 = ao_corner(base, da_pos, db_pos);

    let fa_c = clamp(fa, 0.0, 1.0);
    let fb_c = clamp(fb, 0.0, 1.0);
    let ao_x0 = mix(ao00, ao10, fa_c);
    let ao_x1 = mix(ao01, ao11, fa_c);
    return mix(ao_x0, ao_x1, fb_c);
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

fn ao_corner(face_base: vec3<i32>, da: vec3<i32>, db: vec3<i32>) -> f32 {
    let s1 = ao_occluder(face_base + da);
    let s2 = ao_occluder(face_base + db);
    let cd = ao_occluder(face_base + da + db);
    // Full occlusion if both side voxels are solid (corner case).
    if (s1 && s2) { return 0.35; }
    let cnt = i32(s1) + i32(s2) + i32(cd);
    return 1.0 - f32(cnt) * 0.22;
}

fn axis_select(v: vec3<f32>, ax: i32) -> f32 {
    if (ax == 0) { return v.x; }
    if (ax == 1) { return v.y; }
    return v.z;
}

// LOD: past this many voxels of distance, terminate the DDA at brick
// granularity instead of per-voxel.
const LOD_BRICK_T: f32 = 400.0;

// Even further out, terminate at TILE (16-voxel) granularity: far terrain is
// blocky but each occupied tile costs one hit instead of a brick/voxel descent
// (checklist: tile-level LOD for far terrain).
const TILE_LOD_T: f32 = 520.0;

// Distance-based ray budget (in voxels). Traversal stops here regardless of how
// many DDA steps it took — replaces the old fixed voxel-step count so small
// voxels can't run the loop out before reaching far geometry (holes-through-
// terrain) and so empty rays don't waste steps. Comfortably covers the loaded
// window (the camera sits at its centre).
const MAX_RAY_DIST: f32 = 700.0;

// Beyond this distance, sub-voxel foliage (sprite cross-quads, leaf cutout
// faces) is treated as a solid cube rather than ray-marched. Authored-sprite
// foliage is cheap (2 plane tests + 1 texel fetch vs the old 22-blade
// procedural bundle), so the detail radius is much wider than the old 72.
const FOLIAGE_NEAR_T: f32 = 128.0;

// Shadow / occlusion rays give up past this distance (treated as lit). Far
// shadows contribute little and are the most expensive secondary rays
// (checklist: cheaper secondary rays / coarse shadows).
const SHADOW_MAX_DIST: f32 = 480.0;

// Beyond this distance, skip per-corner ambient occlusion (its contact-shadow
// detail is sub-pixel far away). 12 hierarchical lookups/pixel saved on the
// bulk of the screen.
const AO_DIST: f32 = 64.0;

// God-ray occlusion cap: shafts only need nearby occluders, so the per-step
// occlusion test bails out much sooner than a full-length shadow ray.
const GOD_RAY_OCCL_DIST: f32 = 160.0;

fn trace(origin: vec3<f32>, dir: vec3<f32>) -> Hit {
    water_grad_rest = vec2<f32>(0.0);
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
            // Near foliage gets the full per-blade procedural cutout. Far
            // foliage: leaves become solid cubes (the else branch) so canopies
            // survive, but ground decoration (flowers / tall grass) is skipped
            // entirely — drawing it as a solid cube is the "pink blocks" bug.
            if (is_foliage_mat(m) && t_cur <= FOLIAGE_NEAR_T) {
                let fh = foliage_subvoxel(voxel, origin, dir, m);
                if (fh.hit) {
                    out.hit = true;
                    out.mat = m;
                    out.normal = fh.normal;
                    out.voxel = voxel;
                    // Leaf cutout hits are stable cube faces (axis >= 0): they
                    // get cube AO + the lighting cache. Cross-quad sprites
                    // return oblique normals -> -1, as before.
                    out.last_axis = axis_from_face_normal(fh.normal);
                    out.t_hit = fh.t_hit;
                    var tint = fh.color_tint;
                    // Canopy occupancy AO (primary rays only - shadow rays
                    // must not pay for it). Tuft quads previously had ZERO
                    // occlusion; cube faces switch from the generic cube AO,
                    // which counted the invisible fringe shell as solid.
                    if ((is_leaf_block_mat(m) || m == MAT_LEAF_FRINGE) && fh.t_hit < AO_DIST) {
                        tint = tint * leaf_canopy_ao(voxel, slot_v, bp, bi);
                    }
                    out.tint = tint;
                    return out;
                }
                // cutout missed → fall through to the DDA step below.
            } else if (is_decoration_mat(m)) {
                // Far decoration → invisible; fall through to the DDA step.
            } else if (is_water_mat(m) && t_cur <= WATER_DETAIL_T) {
                // Near water: sub-voxel patch surface. A miss means the ray
                // passed above the patch — keep stepping.
                let en = entry_normal_and_t(last_axis, step, t_max, t_delta, t_enter, tmin3);
                let t_exit_cell = min(t_max.x, min(t_max.y, t_max.z));
                let wh = water_subvoxel(voxel, origin, dir, m, en.n, en.t_hit, t_exit_cell,
                                        slot_v, bp, bi);
                if (wh.hit) {
                    out.hit = true;
                    out.mat = m;
                    out.normal = wh.normal;
                    out.voxel = voxel;
                    out.last_axis = -1; // sub-voxel hit (water is always deferred)
                    out.t_hit = wh.t_hit;
                    water_grad_rest = wh.grad_rest;
                    return out;
                }
            } else {
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
