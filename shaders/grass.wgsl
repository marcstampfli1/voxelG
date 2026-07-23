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

override GRASS_BLADES: u32 = 24u;
override GRASS_SEGS: u32 = 4u;
override GRASS_WIDTH_MUL: f32 = 1.0;
override GRASS_FAR_T: f32 = 90.0;

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
    let front = 0.5 + 0.5 * sin(s * 0.020 - camera.time * 0.9);
    let ripple = 0.5 + 0.5 * sin(s * 0.11 - camera.time * 2.1);
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
    return 0.60 + 0.40 * vnoise3g(vec3<f32>(xz.x * 0.055, 3.7, xz.y * 0.055));
}

// Clump field: a cheap jittered-lattice Voronoi (SYNC: same role as
// raymarch worley2, cells ~2.6 voxels). Returns the clump id hash.
fn clump_id(xz: vec2<f32>) -> f32 {
    let p = xz / 2.6;
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
    @location(2) @interpolate(flat) tangent: vec3<f32>,
    @location(3) @interpolate(flat) wide3: vec3<f32>,
    @location(4) @interpolate(flat) albedo0: vec3<f32>, // root colour
    @location(5) @interpolate(flat) albedo1: vec3<f32>, // tip colour
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
    let rootw = cell.pos + vec3<f32>(root2.x, 0.0, root2.y);

    let cid = clump_id(rootw.xz);
    let clump_ang = fract(cid * 7.13) * 6.2832;
    let clump_h = 0.70 + 0.55 * fract(cid * 3.71);
    let field = flora_field_g(rootw.xz);

    let fa = clump_ang + (fract(bh * 5.0) - 0.5) * 1.5;
    let fdir = vec2<f32>(cos(fa), sin(fa));
    let h = field * clump_h * (0.45 + 0.55 * fract(bh * 3.0));
    let curve = 0.35 + 0.55 * fract(bh * 13.0);
    // Wind bends the CURVE (control points), not the whole blade rigidly.
    let phase = rootw.x * 0.40 + rootw.z * 0.55 + bh * 6.28;
    let wind = wind_off(rootw.xz, phase, 0.30);
    let arc2 = fdir * curve * h + wind * h * 1.7;
    let droop = clamp(length(arc2) * 0.9, 0.0, 0.70);
    let cp0 = rootw;
    let cp1 = rootw + vec3<f32>(arc2.x * 0.5, h * 0.85, arc2.y * 0.5);
    let cp2 = rootw + vec3<f32>(arc2.x, h * (1.0 - droop), arc2.y);

    let p = bez(cp0, cp1, cp2, t0);
    let p_next = bez(cp0, cp1, cp2, min(t0 + 0.25, 1.0));
    var tang = p_next - p;
    if (dot(tang, tang) < 1e-8) { tang = vec3<f32>(0.0, 1.0, 0.0); }
    tang = normalize(tang);

    // Ribbon wide axis: across the facing, twisted slightly along the blade.
    let tw = (fract(bh * 31.0) - 0.5) * 0.7 + t0 * 0.4;
    let wf = vec2<f32>(cos(fa + tw), sin(fa + tw));
    let wide3 = vec3<f32>(-wf.y, 0.0, wf.x);

    // Width: taper root->tip.
    var hw = 0.032 * GRASS_WIDTH_MUL * (0.8 + 0.4 * fract(bh * 17.0)) * (1.0 - t0 * 0.80);

    let wp0 = p + wide3 * hw * cs;
    let d = wp0 - camera.origin;
    let z = dot(d, camera.forward);
    if (z < 0.05) {
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
    o.pos = vec4<f32>(ndc.x * z2, ndc.y * z2, 0.05, z2);
    o.uv = vec2<f32>(cs, t0);
    o.view_t = length(d2);
    o.tangent = tang;
    o.wide3 = wide3;

    // ---- colour: clump-coherent, root-dark -> tip-bright, dry skew ----
    let ground = vec3<f32>(0.30, 0.65, 0.20); // palette[MAT_GRASS], SYNC renderer default_palette
    let dry = fract(cid * 9.77);
    // Stylized lush field: only occasional clumps skew warm, and gently.
    let hue = mix(vec3<f32>(1.0), vec3<f32>(1.22, 1.04, 0.62),
                  smoothstep(0.75, 1.0, dry) * 0.45);
    let cb = (0.88 + 0.24 * fract(cid * 5.23)) * (0.92 + 0.16 * fract(bh * 23.0));
    o.albedo0 = ground * hue * cb * 0.42;
    o.albedo1 = ground * hue * cb * 1.35;
    return o;
}

// SYNC: raymarch.wgsl fog_amount shape (approximate; grass ends at
// GRASS_FAR_T where fog is still mild, so a matched curve suffices).
fn fog_amount_g(t: f32) -> f32 {
    return 1.0 - exp(-max(t - 40.0, 0.0) * 0.0065);
}

@fragment
fn fs_grass(in: VsOut) -> @location(0) vec4<f32> {
    let scene_t = textureLoad(scene_depth, vec2<i32>(in.pos.xy), 0).r;
    if (in.view_t > scene_t + 0.02 + scene_t * 0.002) {
        discard;
    }
    // Reuse the raymarch light cache: the packed shadow/AO of the surface
    // behind this pixel - the ground the blade stands on. Grass shadows and
    // ambient occlusion stay consistent with the world for free.
    // SYNC: raymarch.wgsl pack_light_cache.
    let lc = textureLoad(light_cache, vec2<i32>(in.pos.xy), 0);
    let pw = bitcast<u32>(lc.w);
    var shadow = f32(pw >> 24u) / 255.0;
    var ao = f32((pw >> 16u) & 0xFFu) / 255.0;
    // Sky / no-hit background pixels carry no cache: a blade silhouetted
    // against the sky is fully exposed - lit, not black.
    if (pw == 0u || lc.x >= 1e8) {
        shadow = 1.0;
        ao = 1.0;
    }

    let s = sun_dir_at(camera.sun_time);
    let s_int = sun_intensity(s);
    let sc = sun_color(s);
    let sblade = in.uv.y;

    // Rounded normal: flat face bowed across the width, blended toward up
    // along the height (small sharp speculars at tips).
    var nf = normalize(cross(in.tangent, in.wide3));
    nf = nf * select(1.0, -1.0, nf.y < 0.0);
    var n = normalize(nf + in.wide3 * in.uv.x * 0.65);
    n = normalize(mix(n, vec3<f32>(0.0, 1.0, 0.0), 0.25 + 0.35 * sblade));

    // Wrapped diffuse: foliage responds softer than a hard lambert.
    let ndl = max(0.0, (dot(n, s) + 0.35) / 1.35);
    let direct = sc * ndl * shadow * s_int;
    // Sky ambient with the blade-depth gradient (dark in the sward, bright
    // at the tips) modulated by the ground's cached AO.
    let sky_f = 0.25 + 0.75 * s_int;
    let ambient = vec3<f32>(0.30, 0.34, 0.40) * sky_f * ao * (0.45 + 0.55 * sblade);

    let albedo = mix(in.albedo0, in.albedo1, sblade * sblade * 0.6 + sblade * 0.4);
    var col = albedo * (direct + ambient);

    // Sheen/backlight need the view ray; reconstruct it from the pixel.
    let res = camera.resolution;
    let ndc = vec2<f32>((in.pos.x / res.x) * 2.0 - 1.0, 1.0 - (in.pos.y / res.y) * 2.0);
    let aspect = res.x / res.y;
    let vdir2 = normalize(camera.forward
        + camera.right * ndc.x * camera.tan_half_fov * aspect
        + camera.up * ndc.y * camera.tan_half_fov);
    if (s_int > 0.0) {
        let hv = normalize(s - vdir2);
        let tdh = dot(normalize(in.tangent), hv);
        let sheen = pow(sqrt(max(0.0, 1.0 - tdh * tdh)), 32.0);
        let back = pow(max(0.0, dot(vdir2, s)), 6.0) * clamp(1.0 - s.y * 1.4, 0.0, 1.0);
        col = col + sc * shadow
            * (sheen * 0.25 * vec3<f32>(1.0, 1.0, 0.85)
               + back * 0.30 * vec3<f32>(0.55, 0.85, 0.30) * (0.4 + 0.6 * sblade));
    }

    // Fog toward the horizon haze so far grass melts into the fogged
    // terrain instead of popping against it.
    let fog = fog_amount_g(in.view_t);
    let haze = mix(vec3<f32>(0.60, 0.70, 0.85), vec3<f32>(0.75, 0.80, 0.95), 0.5) * (0.35 + 0.65 * s_int);
    col = mix(col, haze, fog);
    return vec4<f32>(col, 1.0);
}
