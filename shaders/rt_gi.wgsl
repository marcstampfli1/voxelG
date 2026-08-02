// One-bounce diffuse global illumination on the RT cores (RT variant only).
// From a shaded surface point we cosine-sample the hemisphere around its normal
// and trace short RT rays; each ray that hits another surface gathers the light
// that surface reflects (its own sun term * albedo), and each ray that escapes
// gathers skylight. The average is the indirect irradiance - the "light bounces
// off one surface to lightly illuminate another" the sky-access hack could only
// fake. Assembled after the shared resolve_brick + rt_shadow (for the world
// TLAS, resolve_brick, shadow_occluded) and alongside raymarch.wgsl (palette,
// sun_color, ambient_color, brick_voxel_material, sky_color).

const GI_RAYS: i32 = 1;      // one fresh sample/frame; reprojection accumulates it
const GI_DIST: f32 = 10.0 * VOXELS_PER_METRE;  // bounce ray reach (10 m); near light dominates
const GI_STRENGTH: f32 = 1.0;
const GI_SKY: f32 = 0.32;    // soft skylight fill (NOT a hard overhead emitter)

// Nearest RT hit of `dir` from window-local origin `o`, within GI_DIST. Returns
// t (or -1). On a hit, `hv`=hit voxel (window-local), `hn`=face normal,
// `hbi`=storage brick index, `hfv`=brick-local coords (for the material read).
fn gi_trace(o: vec3<f32>, dir: vec3<f32>, hv: ptr<function, vec3<i32>>,
            hn: ptr<function, vec3<i32>>, hbi: ptr<function, i32>,
            hfv: ptr<function, vec3<i32>>) -> f32 {
    var rq: ray_query;
    rayQueryInitialize(&rq, world_tlas, RayDesc(0u, 0xFFu, 0.02, GI_DIST, o, dir));
    var best_t = 1.0e30;
    var found = false;
    while (rayQueryProceed(&rq)) {
        let c = rayQueryGetCandidateIntersection(&rq);
        if (c.kind == RAY_QUERY_INTERSECTION_AABB) {
            let bi = i32(rt_brick_map[c.primitive_index]);
            if (!rt_brick_active(bi)) { continue; }
            let a = rt_aabbs[c.primitive_index];
            let bmin = vec3<f32>(a.min_x, a.min_y, a.min_z);
            var fv = vec3<i32>(0);
            var fnrm = vec3<i32>(0);
            // GI rays see through water/glass (same contract as the probe
            // gather in gi_probes.wgsl).
            let t = resolve_brick(bi, bmin, o, dir, 0.0, best_t, true, &fv, &fnrm);
            if (t >= 0.0) {
                rayQueryGenerateIntersection(&rq, t);
                if (t < best_t) {
                    best_t = t;
                    *hv = vec3<i32>(bmin) + fv;
                    *hn = fnrm;
                    *hbi = bi;
                    *hfv = fv;
                    found = true;
                }
            }
        }
    }
    if (found) { return best_t; }
    return -1.0;
}

// Indirect irradiance at world point `p` with surface normal `n`. `seed` is the
// per-pixel per-frame jitter (TAA averages the few samples into a clean bounce).
fn rt_gather_indirect(p: vec3<f32>, n: vec3<f32>, seed: f32) -> vec3<f32> {
    let o = p - vec3<f32>(camera.world_origin);
    // Tangent frame around n.
    let up = select(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), abs(n.z) > 0.9);
    let tang = normalize(cross(n, up));
    let bitan = cross(n, tang);

    let s = sun_dir();
    let sc = sun_color(s);
    let amb = ambient_color();

    var acc = vec3<f32>(0.0);
    for (var k: i32 = 0; k < GI_RAYS; k = k + 1) {
        // Two decorrelated randoms per sample (hashed from position + seed + k).
        let r1 = hash3f(vec3<f32>(p.xy + vec2<f32>(seed * 37.0, f32(k) * 11.3), p.z));
        let r2 = hash3f(vec3<f32>(p.zx + vec2<f32>(f32(k) * 7.1, seed * 53.0), p.y));
        // Cosine-weighted hemisphere direction.
        let rr = sqrt(r1);
        let phi = 6.2831853 * r2;
        let dl = vec3<f32>(rr * cos(phi), rr * sin(phi), sqrt(max(0.0, 1.0 - r1)));
        let dir = normalize(tang * dl.x + bitan * dl.y + n * dl.z);

        var hv = vec3<i32>(0);
        var hn = vec3<i32>(0);
        var hbi = 0;
        var hfv = vec3<i32>(0);
        let t = gi_trace(o, dir, &hv, &hn, &hbi, &hfv);
        if (t >= 0.0) {
            // Light this bounce surface reflects: its own direct sun (with a
            // shadow ray) times its albedo, plus a little skylight fill.
            let vi = brick_voxel_idx(hfv.x, hfv.y, hfv.z);
            let m = brick_voxel_material(hbi, vi);
            let q_albedo = palette[m].rgb;
            let qn = vec3<f32>(hn);
            let q_ndl = max(0.0, dot(qn, s));
            var q_shadow = 0.0;
            if (q_ndl > 0.0) {
                let q_world = (o + dir * t) + vec3<f32>(camera.world_origin) + qn * 0.03;
                q_shadow = select(1.0, 0.0, shadow_occluded(q_world, s, SHADOW_MAX_DIST));
            }
            acc = acc + q_albedo * (sc * (q_ndl * q_shadow) + amb * 0.35);
        } else {
            // Escaped to sky: gather skylight from that direction.
            acc = acc + sky_color(dir) * GI_SKY;
        }
    }
    return (acc / f32(GI_RAYS)) * GI_STRENGTH;
}

// Temporal GI accumulation buffers (group 1, RT variant only). gi_in = last
// frame's accumulated indirect; gi_out = this frame's. Only the LEGACY
// per-pixel path (GI_PROBE_MODE = 0) uses them; the shipped probe cache carries
// its own temporal accumulation in world space and needs neither.
@group(1) @binding(3) var gi_in: texture_2d<f32>;
@group(1) @binding(4) var gi_out: texture_storage_2d<rgba32float, write>;

// Pipeline-override toggle (1 = on). The game leaves it on; the occlusion-parity
// A/B and any perf comparison set it to 0 to render RT with shadows/AO but no
// one-bounce indirect, so RT still matches the software frame there.
override GI_ENABLE: f32 = 1.0;

// 1 = sample the world-space irradiance probe cache (O(1)/pixel, converges in the
// probes, works while moving); 0 = the legacy per-pixel bounce ray + screen-space
// temporal reproject. The game runs probe mode; the A/B harness flips it.
override GI_PROBE_MODE: f32 = 1.0;

// Accumulated one-bounce indirect at world point `p` (surface normal `n`) for
// output pixel `px`, hit distance `t`. Traces ONE fresh bounce this frame and
// blends it with the reprojected previous-frame value.
fn indirect_light(p: vec3<f32>, n: vec3<f32>, seed: f32, px: vec2<i32>, t: f32, v: vec3<f32>) -> vec3<f32> {
    if (GI_ENABLE < 0.5 || t >= GI_MAX_T) {
        textureStore(gi_out, px, vec4<f32>(0.0));
        return vec3<f32>(0.0);
    }
    let fade = 1.0 - smoothstep(GI_MAX_T * 0.6, GI_MAX_T, t);
    // Probe cache: one cheap trilinear sample, no per-pixel ray or reproject
    // (the probes carry the temporal accumulation and are motion-invariant).
    if (GI_PROBE_MODE > 0.5) {
        let gi = sample_probes(p, n, v) * (GI_STRENGTH * fade);
        textureStore(gi_out, px, vec4<f32>(gi, 1.0));
        return gi;
    }
    var acc = rt_gather_indirect(p, n, seed) * fade;

    // ITS SCREEN-SPACE REPROJECTION IS GONE, and this path is the legacy
    // REFERENCE rather than the shipped one, so that is a deliberate narrowing
    // and not a silent regression. It reprojected a hit into last frame's screen
    // and accepted the stored GI on a POSITION match - and the positions it
    // matched against lived in `light_in`, the previous-frame half of the
    // shadow/AO reprojection cache. That cache is gone (see the binding comment
    // in raymarch.wgsl), and reinstating a 33 MB full-res position history for a
    // path `GI_PROBE_MODE` has replaced in the shipped renderer would be paying
    // the exact cost this rework removed. A static camera still accumulates
    // through TAA; what is lost is accumulation across MOTION, which is one of
    // the reasons the probe cache replaced this path in the first place.
    let prev = textureLoad(gi_in, px, 0);
    if (camera.reproject_ok > 0.5 && prev.a > 0.5) {
        acc = mix(acc, prev.rgb, 0.9); // 90% history: ~10-frame convergence
    }
    textureStore(gi_out, px, vec4<f32>(acc, 1.0));
    return acc;
}
