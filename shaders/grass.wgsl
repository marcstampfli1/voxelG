// GPU grass blades: Ghost-of-Tsushima-style procedural ribbon geometry.
//
// One instance = one grass-top CELL (16 bytes: position + seed). The
// vertex shader grows GRASS_BLADES wind-bent quadratic-Bezier ribbons per
// cell, GRASS_SEGS quads each, fully procedurally - blade facing comes
// from a Worley clump field (blades share facing/height/colour with their
// clump), width tapers root->tip with a view-space minimum so distant
// blades never alias to dust. The fragment shader manually depth-tests
// against the raymarcher's primary-hit depth (leaves.wgsl pattern), reuses
// the raymarch light cache for sun shadow + AO (grass is lit and shadowed
// by the same terms as the ground it stands on), and shades with the
// foliage stack: root-dark gradient, rounded normals, Kajiya-Kay sheen,
// translucent backlight.
//
// Drawn onto the compose output BEFORE the TAA resolve: thin ribbons rely
// on TAA to integrate to smooth anti-aliased blades (the GoT approach).
// Prepended with common.wgsl (Camera + sun helpers).

struct GrassCell {
    pos: vec3<f32>,
    seed: u32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> cells: array<GrassCell>;
@group(0) @binding(2) var scene_depth: texture_2d<f32>;
@group(0) @binding(3) var light_cache: texture_2d<f32>;
// The raymarched terrain colour (geometry buffer, pre-compose): the source
// blades INHERIT their base colour from - the exact lit ground the player
// sees at the blade's root, texture + shadow + AO + GI included.
@group(0) @binding(4) var terrain_color: texture_2d<f32>;
@group(0) @binding(5) var lin_sampler: sampler;

override GRASS_BLADES: u32 = 24u;
override GRASS_SEGS: u32 = 4u;
override GRASS_WIDTH_MUL: f32 = 1.0;
/// Set from Rust (renderer.rs create_grass_pipelines) to grass::GRASS_LOD2_T,
/// which is m_to_vox(22.5). The default is only what the naga validation
/// test compiles against.
override GRASS_FAR_T: f32 = 225.0;
override GRASS_CHUNKY: f32 = 0.0;

// ---- physical scale (LOCAL copy - pinned by a test) --------------------
// This is the one shader renderer.rs assembles WITHOUT build.rs's
// world_consts prelude: grass_source() is `COMMON_WGSL + this file`, so
// VOXELS_PER_METRE is not in scope here the way it is in raymarch/taa/beam/
// physics. Every length below is still written in METRES and converted
// through this single local factor, because a bare voxel count silently
// changes what it MEANS when the voxel changes size - blade widths, clump
// size, wind wavelengths and the fog ramp would all have shrunk 2.5x at
// 25 cm -> 10 cm, turning a meadow into moss.
// SYNC: src/world_dims.rs VOXELS_PER_METRE. `grass::shader_scale_matches_
// world_dims` fails the build if this drifts. (Adding WORLD_CONSTS_WGSL to
// grass_source() and deleting this is strictly better; until then the test
// is what keeps the two honest.)
const VOX_PER_M: f32 = 10.0;

// Footprint of one grass CELL (one instance, GRASS_BLADES blades). SYNC:
// grass::CELL_M. The CPU emits one cell per CELL_M x CELL_M of grass top and
// the roots below scatter across exactly that square, so blades per square
// metre is fixed in metres rather than following the voxel.
const GRASS_CELL_M: f32 = 0.25;
const GRASS_CELL_VOX: f32 = GRASS_CELL_M * VOX_PER_M;

// Wind gust wavelengths, as the real sizes they always were: a ~79 m front
// rolling across the meadow with a ~14 m ripple riding it. `s` below is a
// voxel-space distance, so the per-metre rate is divided down once here.
// SYNC: raymarch.wgsl wind_gust - the same two numbers live there and must
// be converted the same way or blades, cross quads and tree cards stop
// riding one wind.
const WIND_FRONT_K: f32 = 0.080 / VOX_PER_M;  // 2*pi/0.080 = 78.5 m
const WIND_RIPPLE_K: f32 = 0.44 / VOX_PER_M;  // 2*pi/0.44  = 14.3 m
// Rolling height field: ~4.5 m of tall waves and short hollows.
// SYNC: raymarch.wgsl flora_field.
const FLORA_FIELD_K: f32 = 0.222 / VOX_PER_M;
// Clump cells: ~0.65 m Voronoi patches sharing facing/height/colour.
// SYNC: raymarch.wgsl worley2 in turf_blade_hit.
const CLUMP_CELL_VOX: f32 = 0.65 * VOX_PER_M;
// Hill-scale hue drift over the carpet fallback: ~7.6 m features.
const HUE_DRIFT_K: f32 = 0.132 / VOX_PER_M;
// Wind PHASE gradient: neighbouring blades desync over ~4 m / ~2.9 m.
// SYNC: raymarch.wgsl's per-voxel `phase` in flora/turf/leaf quads.
const PHASE_KX: f32 = 1.60 / VOX_PER_M;
const PHASE_KZ: f32 = 2.20 / VOX_PER_M;
// Blade geometry. The old code multiplied the dimensionless height product
// by an implicit "1.0" that was one voxel, i.e. 25 cm; BLADE_H_UNIT is that
// unit made explicit, so a blade stays roughly 15..38 cm tall.
const BLADE_H_UNIT: f32 = 0.25 * VOX_PER_M;
const BLADE_HW: f32 = 0.009 * VOX_PER_M;      // 9 mm half-width at the root
const DROOP_K: f32 = 1.4 / VOX_PER_M;         // droop per metre of bend
// Chunky style: value strata banded over 26 cm of WORLD height.
const BAND_H: f32 = 0.2625 * VOX_PER_M;
// Near-plane guard, and the numerator of the pass's reversed-z depth
// (depth = NEAR_T / view_z, so keeping the two equal keeps the depth
// distribution identical in metres). 1.25 cm.
const NEAR_T: f32 = 0.0125 * VOX_PER_M;
// Manual depth test against the raymarch primary hit: a 5 mm absolute bias
// plus a 0.2% slope term (a RATIO, so scale-free).
const DEPTH_BIAS: f32 = 0.005 * VOX_PER_M;
// A root-colour tap only counts when its depth agrees with the root's own
// distance to within 0.75 m.
const ROOT_DEPTH_TOL: f32 = 0.75 * VOX_PER_M;
// Distance haze: clear to 10 m, then an exponential ramp at 0.026 per metre.
const FOG_START: f32 = 10.0 * VOX_PER_M;
const FOG_K: f32 = 0.026 / VOX_PER_M;

// ---- small local copies (SYNC comments point at the originals) ----------

// SYNC: raymarch.wgsl hash3f.
fn hash3f(pin: vec3<f32>) -> f32 {
    var q = fract(pin * vec3<f32>(0.1031, 0.1030, 0.0973));
    q = q + dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

// SYNC: raymarch.wgsl wind_gust/wind_offset (same field so blades, cross
// quads and tree cards all ride one wind).
fn wind_gust(p_xz: vec2<f32>, wdir: vec2<f32>) -> f32 {
    let s = dot(p_xz, wdir);
    let front = 0.5 + 0.5 * sin(s * WIND_FRONT_K - camera.time * 0.9);
    let ripple = 0.5 + 0.5 * sin(s * WIND_RIPPLE_K - camera.time * 2.1);
    return 0.25 + 0.75 * front * (0.6 + 0.4 * ripple);
}

fn wind_off(p_xz: vec2<f32>, phase: f32, base_amp: f32) -> vec2<f32> {
    let wdir = vec2<f32>(camera.wind_x, camera.wind_z);
    let strength = base_amp * wind_gust(p_xz, wdir)
        * (0.70 + 0.30 * sin(camera.time * 0.55 + phase));
    return wdir * strength;
}

// SYNC: raymarch.wgsl flora_field (rolling height coherence, same layout so
// the raster blades and the shader turf agree on tall/short regions).
fn hash_l3(ip: vec3<f32>) -> f32 {
    var v = vec3<u32>(bitcast<u32>(i32(ip.x)), bitcast<u32>(i32(ip.y)), bitcast<u32>(i32(ip.z)));
    v = v * vec3<u32>(1664525u) + vec3<u32>(1013904223u);
    v.x = v.x + v.y * v.z;
    v.y = v.y + v.z * v.x;
    v.z = v.z + v.x * v.y;
    v = v ^ (v >> vec3<u32>(16u));
    v.x = v.x + v.y * v.z;
    return f32(v.x) * 2.3283064e-10;
}
fn vnoise3g(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = mix(hash_l3(i), hash_l3(i + vec3<f32>(1.0, 0.0, 0.0)), u.x);
    let b = mix(hash_l3(i + vec3<f32>(0.0, 1.0, 0.0)), hash_l3(i + vec3<f32>(1.0, 1.0, 0.0)), u.x);
    let c = mix(hash_l3(i + vec3<f32>(0.0, 0.0, 1.0)), hash_l3(i + vec3<f32>(1.0, 0.0, 1.0)), u.x);
    let d = mix(hash_l3(i + vec3<f32>(0.0, 1.0, 1.0)), hash_l3(i + vec3<f32>(1.0, 1.0, 1.0)), u.x);
    return mix(mix(a, b, u.y), mix(c, d, u.y), u.z);
}
fn flora_field_g(xz: vec2<f32>) -> f32 {
    return 0.60 + 0.40 * vnoise3g(vec3<f32>(xz.x * FLORA_FIELD_K, 3.7, xz.y * FLORA_FIELD_K));
}

// Clump field: a cheap jittered-lattice Voronoi (SYNC: same role as
// raymarch worley2, cells ~0.65 m). Returns the clump id hash.
fn clump_id(xz: vec2<f32>) -> f32 {
    let p = xz / CLUMP_CELL_VOX;
    let ip = floor(p);
    let fp = fract(p);
    var best = 1e9;
    var id = 0.0;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let cell = ip + vec2<f32>(f32(dx), f32(dz));
            let h = hash_l3(vec3<f32>(cell.x, 17.0, cell.y));
            let site = vec2<f32>(f32(dx), f32(dz)) + vec2<f32>(h, fract(h * 57.31)) - fp;
            let d = dot(site, site);
            if (d < best) {
                best = d;
                id = h;
            }
        }
    }
    return id;
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,          // x: side -1..1, y: height 0..1 along blade
    @location(1) view_t: f32,
    // Fallback base colour (palette carpet) for roots whose ground pixel is
    // occluded or off-screen; inherited terrain colour is the primary path.
    @location(2) @interpolate(flat) albedo0: vec3<f32>,
    // Shared tip lighten factor (whisper of per-blade spread).
    @location(3) @interpolate(flat) tip_mul: f32,
    // The blade ROOT's screen position and view distance: base colour and
    // lighting are read from the terrain buffers where the blade GROWS,
    // never where its fragment lands (a fragment's own texel belongs to
    // whatever is behind it).
    @location(4) @interpolate(flat) root_px: vec2<f32>,
    @location(5) @interpolate(flat) root_dist: f32,
    // World height of the fragment and of the ground: the chunky style
    // draws its value bands at shared WORLD heights, so the strata run as
    // one horizontal field surface instead of per-blade gradients.
    @location(6) wy: f32,
    @location(7) @interpolate(flat) gy: f32,
};

fn bez(cp0: vec3<f32>, cp1: vec3<f32>, cp2: vec3<f32>, t: f32) -> vec3<f32> {
    let it = 1.0 - t;
    return cp0 * (it * it) + cp1 * (2.0 * it * t) + cp2 * (t * t);
}

@vertex
fn vs_grass(@builtin(vertex_index) vid: u32,
            @builtin(instance_index) iid: u32) -> VsOut {
    var o: VsOut;
    let cell = cells[iid];
    let per_blade = GRASS_SEGS * 6u;
    let blade = vid / per_blade;
    let rem = vid % per_blade;
    let seg = rem / 6u;
    let corner = rem % 6u;
    // Quad corners as (t-along-segment, side).
    var ct: f32;
    var cs: f32;
    switch corner {
        case 0u: { ct = 0.0; cs = -1.0; }
        case 1u: { ct = 0.0; cs = 1.0; }
        case 2u: { ct = 1.0; cs = -1.0; }
        case 3u: { ct = 1.0; cs = -1.0; }
        case 4u: { ct = 0.0; cs = 1.0; }
        default: { ct = 1.0; cs = 1.0; }
    }
    let t0 = (f32(seg) + ct) / f32(GRASS_SEGS); // 0..1 along the blade

    // ---- per-blade parameters, all from hashes (zero per-blade CPU data)
    let sf = f32(cell.seed & 0xFFFFu) / 65535.0;
    let bh = hash3f(vec3<f32>(sf * 511.0, f32(blade) * 7.13 + 0.31, f32(blade) * 2.9 + sf * 97.0));
    let root2 = vec2<f32>(fract(bh * 13.7), fract(bh * 41.9));
    // Roots scatter across the cell's own 0.25 m footprint, whatever that is
    // in voxels - the CPU emits one cell per GRASS_CELL_M square.
    let rootw = cell.pos + vec3<f32>(root2.x, 0.0, root2.y) * GRASS_CELL_VOX;

    let cid = clump_id(rootw.xz);
    let clump_ang = fract(cid * 7.13) * 6.2832;
    let clump_h = 0.70 + 0.55 * fract(cid * 3.71);
    let field = flora_field_g(rootw.xz);

    let fa = clump_ang + (fract(bh * 5.0) - 0.5) * 1.5;
    let fdir = vec2<f32>(cos(fa), sin(fa));
    var hfrac = 0.55 + 0.45 * fract(bh * 3.0);
    // Chunky field: heights pull toward one even canopy level - an even
    // surface is what reads as a FIELD instead of individual shapes.
    hfrac = mix(hfrac, 0.88, GRASS_CHUNKY * 0.55);
    let h = field * clump_h * hfrac * (1.02 + GRASS_CHUNKY * 0.18) * BLADE_H_UNIT;
    let curve = 0.16 + 0.30 * fract(bh * 13.0);
    // Wind bends the CURVE (control points), not the whole blade rigidly.
    // curve/base_amp/1.7 are all RATIOS of h, so only the phase gradient
    // (a spatial frequency) needs converting.
    let phase = rootw.x * PHASE_KX + rootw.z * PHASE_KZ + bh * 6.28;
    let wind = wind_off(rootw.xz, phase, 0.30);
    let arc2 = fdir * curve * h + wind * h * 1.7;
    // Gentle arc: meadow blades bow, they do not hook over. Tips stay
    // above ~80% height even in gusts.
    let droop = clamp(length(arc2) * DROOP_K, 0.0, 0.20);
    let cp0 = rootw;
    let cp1 = rootw + vec3<f32>(arc2.x * 0.5, h * 0.85, arc2.y * 0.5);
    let cp2 = rootw + vec3<f32>(arc2.x, h * (1.0 - droop), arc2.y);

    let p = bez(cp0, cp1, cp2, t0);
    // Analytic Bezier derivative: exact per-vertex tangent, interpolated
    // across the quad for smooth curvature shading.
    var tang = (cp1 - cp0) * (2.0 * (1.0 - t0)) + (cp2 - cp1) * (2.0 * t0);
    if (dot(tang, tang) < 1e-8) { tang = vec3<f32>(0.0, 1.0, 0.0); }
    tang = normalize(tang);

    // Ribbon wide axis: across the facing, twisted slightly along the blade.
    let tw = (fract(bh * 31.0) - 0.5) * 0.3 + t0 * 0.15;
    let wf = vec2<f32>(cos(fa + tw), sin(fa + tw));
    let wide3 = vec3<f32>(-wf.y, 0.0, wf.x);

    // Width: taper root->tip.
    // Two silhouettes: flat tapered spike (default), or the chunky blunt
    // paddle (Hytale-proportioned bold shapes) when GRASS_CHUNKY is set.
    let spike = 1.0 - t0 * 0.98;
    let paddle = 1.0 - pow(t0, 2.2) * 0.92;
    let plump = mix(spike, paddle, GRASS_CHUNKY);
    var hw = BLADE_HW * (1.0 + GRASS_CHUNKY * 1.15) * GRASS_WIDTH_MUL
        * (0.8 + 0.4 * fract(bh * 17.0)) * plump;

    let wp0 = p + wide3 * hw * cs;
    let d = wp0 - camera.origin;
    let z = dot(d, camera.forward);
    if (z < NEAR_T) {
        o.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return o;
    }
    // View-space minimum width (GoT trick): a blade never projects thinner
    // than ~0.6 px, so distant grass shimmers less and stays visible; TAA
    // integrates the slight overcoverage away.
    let px_w = hw * 2.0 / (z * camera.tan_half_fov * 2.0 / camera.resolution.y);
    if (px_w < 0.6) {
        hw = hw * (0.6 / max(px_w, 1e-3));
    }
    let wp = p + wide3 * hw * cs;

    let d2 = wp - camera.origin;
    let z2 = dot(d2, camera.forward);
    let aspect = camera.resolution.x / camera.resolution.y;
    var ndc = vec2<f32>(dot(d2, camera.right) / (z2 * camera.tan_half_fov * aspect),
                        dot(d2, camera.up) / (z2 * camera.tan_half_fov));
    // Same TAA sub-pixel jitter as the raymarch rays: the blades sit on the
    // jittered sample grid, and the TAA resolve integrates them.
    ndc = ndc - vec2<f32>(camera.jitter.x, -camera.jitter.y) * 2.0 / camera.resolution;
    o.pos = vec4<f32>(ndc.x * z2, ndc.y * z2, NEAR_T, z2);
    o.uv = vec2<f32>(cs, t0);
    o.view_t = length(d2);
    o.wy = wp.y;
    o.gy = rootw.y;
    // Project the root to screen space (stable: no jitter - a jittered grid
    // made the sampled texel alternate per frame, flickering the lighting).
    let rd = rootw - camera.origin;
    let rz = max(dot(rd, camera.forward), NEAR_T);
    let rndc = vec2<f32>(dot(rd, camera.right) / (rz * camera.tan_half_fov * aspect),
                         dot(rd, camera.up) / (rz * camera.tan_half_fov));
    o.root_px = vec2<f32>(rndc.x * 0.5 + 0.5, 0.5 - rndc.y * 0.5) * camera.resolution;
    o.root_dist = length(rd);

    // Fallback carpet (only for occluded / off-screen roots): the palette
    // green with the hill-scale hue drift - never per-blade variance.
    let ground = vec3<f32>(0.30, 0.65, 0.20); // palette[MAT_GRASS], SYNC renderer default_palette
    let hue_t = vnoise3g(vec3<f32>(rootw.x * HUE_DRIFT_K, 12.5, rootw.z * HUE_DRIFT_K));
    o.albedo0 = ground * mix(vec3<f32>(1.02, 0.98, 0.85), vec3<f32>(1.22, 1.08, 0.62), hue_t);
    // ONE shared tip lighten - zero per-blade colour variance, exactly as
    // the macro-calm description states.
    o.tip_mul = 1.30;
    return o;
}

// SYNC: raymarch.wgsl fog_amount shape (approximate; grass ends at
// GRASS_FAR_T where fog is still mild, so a matched curve suffices).
fn fog_amount_g(t: f32) -> f32 {
    return 1.0 - exp(-max(t - FOG_START, 0.0) * FOG_K);
}

@fragment
fn fs_grass(in: VsOut) -> @location(0) vec4<f32> {
    let scene_t = textureLoad(scene_depth, vec2<i32>(in.pos.xy), 0).r;
    if (in.view_t > scene_t + DEPTH_BIAS + scene_t * 0.002) {
        discard;
    }

    // ---- TERRAIN INHERITANCE (the BotW/Genshin field rule, exact) ----
    // The blade's base colour IS the rendered ground at its root: sample
    // the raymarch geometry buffer (texture + shadow + AO + GI already
    // applied) at the root's screen position, 2x2 averaged. A texel only
    // counts when its depth agrees with the root's distance - texels
    // showing an occluder or the sky fall back to the palette carpet lit
    // by the cached light terms. No relighting of the inherited colour:
    // reconstruction is what drifted from the visible ground before.
    // Motion-stable root lookup. The projected root SLIDES across the
    // screen while the camera moves: nearest-texel reads snap from texel to
    // texel (steppy) and the geometry buffer re-renders each frame under
    // sub-pixel ray jitter (wobbly). BILINEAR samples at the exact
    // fractional position vary continuously as the lookup slides, and a
    // 5-tap cross (+-2 px) averages the per-frame jitter noise the way TAA
    // does for terrain pixels. Each tap is depth-validated at its centre.
    let resf = camera.resolution;
    let rpx = clamp(in.root_px, vec2<f32>(2.5), resf - 2.5);
    var inherited = vec3<f32>(0.0);
    var w_inherit = 0.0;
    var shadow = 0.0;
    var ao = 0.0;
    var w_light = 0.0;
    for (var k = 0; k < 5; k = k + 1) {
        var off = vec2<f32>(0.0);
        if (k == 1) { off = vec2<f32>(2.0, 0.0); }
        if (k == 2) { off = vec2<f32>(-2.0, 0.0); }
        if (k == 3) { off = vec2<f32>(0.0, 2.0); }
        if (k == 4) { off = vec2<f32>(0.0, -2.0); }
        let sp = rpx + off;
        let p2 = vec2<i32>(sp);
        let d = textureLoad(scene_depth, p2, 0).r;
        if (abs(d - in.root_dist) < ROOT_DEPTH_TOL) {
            inherited += textureSampleLevel(terrain_color, lin_sampler, sp / resf, 0.0).rgb;
            w_inherit += 1.0;
        }
        // SYNC: raymarch.wgsl pack_light_cache (packed bits: nearest unpack).
        let c = textureLoad(light_cache, p2, 0);
        let pw = bitcast<u32>(c.w);
        if (pw != 0u && c.x < 1e8) {
            shadow += f32(pw >> 24u) / 255.0;
            ao += f32((pw >> 16u) & 0xFFu) / 255.0;
            w_light += 1.0;
        }
    }
    if (w_light > 0.0) {
        shadow /= w_light;
        ao /= w_light;
    } else {
        shadow = 1.0;
        ao = 1.0;
    }

    let s = sun_dir_at(camera.sun_time);
    let s_int = sun_intensity(s);
    let sc = sun_color(s);
    let sblade = in.uv.y;

    var base: vec3<f32>;
    if (w_inherit > 0.0) {
        base = inherited / w_inherit;
    } else {
        // Root ground not visible: palette carpet lit like flat ground by
        // the cached terms (the closest reconstruction available).
        let ndl = max(0.0, (s.y + 0.35) / 1.35);
        let sky_f = 0.25 + 0.75 * s_int;
        base = in.albedo0
            * (sc * ndl * shadow * s_int
               + (vec3<f32>(0.26, 0.33, 0.45) * ao + vec3<f32>(0.10, 0.15, 0.06)) * sky_f);
    }

    // Deliberate VALUE STEPS along the height on top of the inherited
    // colour (the stylized-art rule: clean readable bands, not a smooth
    // gradient): carpet -> mid band (+13%) -> tip band (shared lighten).
    // Band coordinate: per-blade height for spikes, shared WORLD height
    // for the chunky style - strata as one horizontal field surface.
    let hband = clamp((in.wy - in.gy) / BAND_H, 0.0, 1.0);
    let band = mix(sblade, hband, GRASS_CHUNKY);
    let step_w = 0.03 + GRASS_CHUNKY * 0.02;
    let s1 = smoothstep(0.52 - step_w, 0.52 + step_w, band);
    let s2 = smoothstep(0.83 - step_w, 0.83 + step_w, band);
    // Lightening follows the sun: a tip only brightens where its ground is
    // actually lit (gate by the root shadow term). Ungated lightening made
    // shaded canopies float brighter than their terrain and pushed sunlit
    // tips over the bloom knee - grass that visibly emitted light.
    let lg = 0.30 + 0.70 * shadow;
    let tip_mul = 1.0 + (mix(min(in.tip_mul, 1.22), 1.12, GRASS_CHUNKY) - 1.0) * lg;
    var col = base * (1.0 + (0.13 - GRASS_CHUNKY * 0.03) * s1 * lg);
    col = mix(col, base * tip_mul, s2 * 0.9);

    // Field-level tip glow toward a low sun (one shared response across
    // the whole field - kept by request: the glow reads as backlit tips).
    let res = camera.resolution;
    let gndc = vec2<f32>((in.pos.x / res.x) * 2.0 - 1.0, 1.0 - (in.pos.y / res.y) * 2.0);
    let aspect_g = res.x / res.y;
    let vdir2 = normalize(camera.forward
        + camera.right * gndc.x * camera.tan_half_fov * aspect_g
        + camera.up * gndc.y * camera.tan_half_fov);
    if (s_int > 0.0) {
        let back = pow(max(0.0, dot(vdir2, s)), 5.0) * clamp(1.2 - s.y * 1.2, 0.0, 1.0);
        col = col + sc * shadow * back * 0.15 * vec3<f32>(0.70, 0.95, 0.35)
            * smoothstep(0.6, 1.0, band);
    }

    // Fog toward the horizon haze so far grass melts into the fogged
    // terrain instead of popping against it.
    let fog = fog_amount_g(in.view_t);
    let haze = mix(vec3<f32>(0.60, 0.70, 0.85), vec3<f32>(0.75, 0.80, 0.95), 0.5) * (0.35 + 0.65 * s_int);
    col = mix(col, haze, fog);
    // Alpha 0 marks blade pixels for the TAA resolve: swaying thin blades
    // must not inherit the terrain history behind them (ghost blades).
    return vec4<f32>(col, 0.0);
}
