// World-space irradiance probe cache (DDGI-style), RT variant only.
//
// A coarse grid of irradiance probes covers the streaming world window: one
// probe every PROBE_SPACING voxels, centered in its cell. Each probe stores the
// incoming radiance around it as spherical-harmonic L1 coefficients (per colour:
// 1 constant + 3 linear = 4 coeffs). Two operations:
//
//   * UPDATE (cs_gi_probe_update): amortized over frames, a rotating subset of
//     probes each cast a batch of RT rays over the sphere, gather the radiance
//     each ray sees (bounced surface sun*albedo + skylight), project it into SH
//     and blend into the stored coefficients with temporal hysteresis. The gather
//     converges to ground truth over frames - it is not an approximation.
//
//   * SAMPLE (sample_probes): per shaded pixel, trilinearly interpolate the 8
//     surrounding probes and evaluate their SH irradiance in the surface normal
//     direction. O(1) - a handful of buffer reads, no rays. This is what replaces
//     the per-pixel bounce ray, decoupling GI cost from screen resolution AND
//     from camera motion (probes are world-space, so there is no disocclusion).
//
// The grid is toroidal like the voxel storage: a probe slot folds its world grid
// coordinate mod PROBE_DIM, and the update resets a slot's accumulation when the
// streaming window moves a different world cell into it (meta.xyz mismatch), so
// newly-exposed probes converge fresh instead of smearing stale light.

struct GiProbe {
    sh_r: vec4<f32>,   // SH-L1 radiance coefficients, red   (c0, c1y, c1z, c1x)
    sh_g: vec4<f32>,   // green
    sh_b: vec4<f32>,   // blue
    // Visibility moments for Chebyshev leak prevention: mean visible distance
    // and mean squared distance, each projected into SH-L1. At sample time the
    // one-tailed Chebyshev bound sigma^2/(sigma^2 + (d - mu)^2) down-weights a
    // probe whose line of sight toward the shaded point is statistically
    // blocked - light stops bleeding through thin walls, sealed spaces darken.
    sh_dist: vec4<f32>,
    sh_dist2: vec4<f32>,
    info: vec4<f32>,   // xyz = world grid coord this slot holds; w = sample age
    // Staging accumulator: the in-progress sum of this cycle's per-round
    // subset estimates. The DISPLAYED coefficients above only ever step by a
    // COMPLETE 64-direction estimate (fold at the last epoch), so a static
    // scene converges to an exact constant - no per-round ripple, no random
    // walk. Sampling reads only the displayed fields.
    st_r: vec4<f32>,
    st_g: vec4<f32>,
    st_b: vec4<f32>,
    st_dist: vec4<f32>,
    st_dist2: vec4<f32>,
    st_info: vec4<f32>, // x = subset estimates accumulated this cycle
};

@group(1) @binding(5) var<storage, read_write> gi_probes: array<GiProbe>;

// ---- physical scale ----
// Everything below that is a DISTANCE is written in metres and converted
// through VOXELS_PER_METRE (build.rs emits it into the world_consts prelude
// this shader is assembled behind, from src/world_dims.rs). Probe biases,
// the Chebyshev slack and the relocation budget are all properties of the
// ROOM the cache describes, not of the grid it is stored on: as bare voxel
// counts every one of them would have shrunk 2.5x when the voxel did, and a
// surface bias that no longer clears the surface is exactly how DDGI starts
// self-shadowing.
//
// Query bias off the surface, along the normal AND back toward the camera,
// so the Chebyshev activation front lands in open air.
const GI_SURFACE_BIAS: f32 = 0.25 * VOXELS_PER_METRE;   // 25 cm each way
// Chebyshev slack: full weight within this of the probe's mean visible
// distance, so the falloff factor is exactly 1.0 at the crossing.
const GI_CHEB_SLACK: f32 = 0.0875 * VOXELS_PER_METRE;   // 8.75 cm
// Variance floor as a standard deviation: 12.5 cm of transition (~1 voxel
// at 25 cm, ~1.25 voxels at 10 cm) is what reads as soft shading rather
// than a hard edge, and it is the LENGTH that has to hold, not the voxel
// count. Squared here because the test compares against a variance.
const GI_CHEB_SIGMA: f32 = 0.125 * VOXELS_PER_METRE;
const GI_CHEB_VAR_FLOOR: f32 = GI_CHEB_SIGMA * GI_CHEB_SIGMA;
// Gather-ray tmin (self-hit guard) and the shadow-ray origin lift off the
// hit face.
const GI_RAY_TMIN: f32 = 0.005 * VOXELS_PER_METRE;      // 5 mm
const GI_SHADOW_BIAS: f32 = 0.0075 * VOXELS_PER_METRE;  // 7.5 mm
// Probe relocation: nudge away when the nearest surface is within
// GI_RELOC_NEAR, at most GI_RELOC_STEP per round, never further than
// GI_RELOC_MAX from the cell centre (well inside the 2 m cell either way);
// drift home once clear by GI_RELOC_CLEAR.
const GI_RELOC_NEAR: f32 = 0.30 * VOXELS_PER_METRE;
const GI_RELOC_STEP: f32 = 0.0875 * VOXELS_PER_METRE;
const GI_RELOC_MAX: f32 = 0.875 * VOXELS_PER_METRE;
const GI_RELOC_CLEAR: f32 = 0.625 * VOXELS_PER_METRE;

// ---- probe grid geometry ----

// Linear slot index for a world probe-grid coordinate (toroidal fold).
fn probe_slot(g: vec3<i32>) -> i32 {
    let s = vec3<i32>(
        pos_mod(g.x, PROBE_DIM_X),
        pos_mod(g.y, PROBE_DIM_Y),
        pos_mod(g.z, PROBE_DIM_Z),
    );
    return s.x + s.y * PROBE_DIM_X + s.z * PROBE_DIM_X * PROBE_DIM_Y;
}

// World voxel-space position (absolute) of probe grid coordinate g: cell center.
fn probe_world_pos(g: vec3<i32>) -> vec3<f32> {
    return (vec3<f32>(g) + vec3<f32>(0.5)) * f32(PROBE_SPACING);
}

// ---- SH-L1 basis + cosine-lobe irradiance eval ----

// Real SH-L1 basis at unit direction d.
fn sh_basis(d: vec3<f32>) -> vec4<f32> {
    return vec4<f32>(0.282095, 0.488603 * d.y, 0.488603 * d.z, 0.488603 * d.x);
}

// Irradiance/pi in direction n from radiance-projected SH coefficients (cosine
// lobe convolution: a0 = pi, a1 = 2pi/3; divide by pi for the Lambertian 1/pi).
// The result is the incoming-radiance factor to multiply by the receiver albedo.
//
// Deringed: an SH-L1 with a strong lobe (bright sky above, dark below - the
// underwater/shadow-interior case) evaluates NEGATIVE for opposed normals,
// and a hard max() clamp turns that zero-crossing into crisp curved black
// edges on geometry (seen as unphysical voxel-edge/diagonal shadows). Cap
// the linear lobe per channel at 95% of the DC term so E(n) is positive for
// every n: identical result wherever the old eval did not clamp; the
// formerly crushed-black zones become smooth dim gradients instead.
fn sh_eval_irradiance(p: GiProbe, n: vec3<f32>) -> vec3<f32> {
    let dc = vec3<f32>(p.sh_r.x, p.sh_g.x, p.sh_b.x) * 0.282095;
    let a1 = 0.488603 * 0.666667;
    let lv = vec3<f32>(n.y, n.z, n.x); // stored coefficient order (c1y, c1z, c1x)
    let lin = vec3<f32>(dot(p.sh_r.yzw, lv), dot(p.sh_g.yzw, lv), dot(p.sh_b.yzw, lv)) * a1;
    let lmax = vec3<f32>(length(p.sh_r.yzw), length(p.sh_g.yzw), length(p.sh_b.yzw)) * a1;
    let s = min(vec3<f32>(1.0), 0.95 * dc / max(lmax, vec3<f32>(1e-6)));
    return max(vec3<f32>(0.0), dc + s * lin);
}

// Reconstruct a scalar SH-L1 field (distance moments) toward direction d.
fn sh_eval_scalar(c: vec4<f32>, d: vec3<f32>) -> f32 {
    // Uniform-sphere MC projection pairs with the plain basis; the 4pi/N weight
    // is already in the coefficients.
    return dot(c, sh_basis(d));
}

// ---- per-pixel sample: trilinear over the 8 surrounding probes ----

fn sample_probes(p_world: vec3<f32>, n: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    // Query point biased off the surface along the normal AND back toward
    // the camera (DDGI surface bias, v = camera->surface ray direction): the
    // Chebyshev activation front then lands in open air instead of exactly
    // on the wall it protects, where its transition band painted probe-grid-
    // scale shapes onto flat surfaces. Irradiance is still evaluated with
    // the true surface normal.
    let p_q = p_world + n * GI_SURFACE_BIAS - v * GI_SURFACE_BIAS;
    // Grid coordinate of p (probes sit at cell centers, hence the -0.5 shift).
    let pg = p_q / f32(PROBE_SPACING) - vec3<f32>(0.5);
    let base = vec3<i32>(floor(pg));
    // Smoothstepped trilinear factor: C1 interpolation, so the piecewise-
    // linear 45-degree iso-line creases of plain trilinear cannot form.
    let fl = pg - floor(pg);
    let f = fl * fl * (3.0 - 2.0 * fl);

    var acc = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var i: i32 = 0; i < 8; i = i + 1) {
        let off = vec3<i32>(i & 1, (i >> 1) & 1, (i >> 2) & 1);
        let g = base + off;
        // trilinear weight
        let tw = vec3<f32>(
            mix(1.0 - f.x, f.x, f32(off.x)),
            mix(1.0 - f.y, f.y, f32(off.y)),
            mix(1.0 - f.z, f.z, f32(off.z)),
        );
        var w = tw.x * tw.y * tw.z;
        if (w <= 0.0) { continue; }
        let pr = gi_probes[probe_slot(g)];
        // Skip a probe that holds a different world cell (not yet re-gathered
        // after streaming) or has never been gathered.
        if (pr.info.w < 0.5 || any(pr.info.xyz != vec3<f32>(g))) { continue; }
        // Wrap-toward-normal weight: probes on the far side of the surface should
        // not leak. Directional weight from the probe->point vector vs normal.
        // The probe's RELOCATED position anchors all geometric terms.
        let dir_pp = probe_world_pos(g) + pr.st_info.yzw - p_q;
        let d = normalize(dir_pp + n * 0.001);
        let backface = max(0.0, dot(d, n));
        w = w * (backface * backface + 0.05);
        // Chebyshev visibility: from the PROBE's view, is the shaded point
        // beyond its mean visible distance in that direction? If so a wall
        // sits between them - bound the contribution by the variance test
        // instead of letting the light tunnel through.
        let dist_pp = length(dir_pp);
        let to_point = -d; // probe -> point
        let mu = sh_eval_scalar(pr.sh_dist, to_point);
        // Continuous one-tailed Chebyshev: full weight within the slack, then
        // a smooth variance falloff measured FROM the slack boundary, so the
        // factor is 1.0 exactly at the crossing. The previous form gated on
        // the same threshold but measured delta from mu itself, stepping the
        // weight ~7x at the crossing - once the probe field became temporally
        // stable, that step stood still as crisp block-aligned GI edges on
        // walls (the threshold surface cutting through geometry).
        let delta = max(dist_pp - (mu + GI_CHEB_SLACK), 0.0);
        if (delta > 0.0) {
            let mu2 = sh_eval_scalar(pr.sh_dist2, to_point);
            // Variance floor (was 0.02 voxels^2): the falloff used to
            // collapse within ~5 cm, a razor-thin front that still read as a
            // hard edge; ~12.5 cm of transition reads as soft shading.
            let variance = max(mu2 - mu * mu, GI_CHEB_VAR_FLOOR);
            // No lower clamp (was 0.02): a probe buried in terrain or far
            // behind a wall must be able to vanish from the blend entirely -
            // the old floor let near-black probes leak dark diamonds into
            // the interpolation. The wsum epsilon fallback handles the
            // all-probes-dead case.
            w = w * clamp(variance / (variance + delta * delta), 0.0, 1.0);
        }
        acc = acc + sh_eval_irradiance(pr, n) * w;
        wsum = wsum + w;
    }
    if (wsum <= 1e-4) { return vec3<f32>(0.0); }
    return acc / wsum;
}

// ---- probe update: RT sphere gather + SH projection + temporal blend ----

// Radiance seen along a gather ray from world point o (window-local), dir d.
// Returns (rgb, visible distance) - the distance feeds the Chebyshev moments.
// Reports the nearest surface seen across calls via near_t/near_n (hit
// distance + face normal), feeding probe relocation.
fn probe_ray_radiance(o: vec3<f32>, d: vec3<f32>, near_t: ptr<function, f32>, near_n: ptr<function, vec3<f32>>) -> vec4<f32> {
    let s = sun_dir();
    let sc = sun_color(s);
    let amb = ambient_color();
    var rq: ray_query;
    rayQueryInitialize(&rq, world_tlas, RayDesc(0u, 0xFFu, GI_RAY_TMIN, GI_DIST, o, d));
    var best_t = 1.0e30;
    var hv = vec3<i32>(0);
    var hn = vec3<i32>(0);
    var hbi = 0;
    var hfv = vec3<i32>(0);
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
            // skip_transparent: GI rays see THROUGH water/glass. Without
            // this the gather treated the water surface as a lit solid wall
            // AND recorded it into the Chebyshev visibility moments, so
            // underwater probes believed a ceiling of geometry hung above
            // them - the root of every underwater-only GI artifact (patchy
            // floor shadows, wall shapes). Water absorption stays the
            // primary path's job.
            let t = resolve_brick(bi, bmin, o, d, 0.0, best_t, true, &fv, &fnrm);
            if (t >= 0.0) {
                rayQueryGenerateIntersection(&rq, t);
                if (t < best_t) {
                    best_t = t; hv = vec3<i32>(bmin) + fv; hn = fnrm; hbi = bi; hfv = fv; found = true;
                }
            }
        }
    }
    if (!found) {
        // Escaped to sky: gather skylight; visible distance = the ray's reach.
        return vec4<f32>(sky_color(d) * GI_SKY, GI_DIST);
    }
    let vi = brick_voxel_idx(hfv.x, hfv.y, hfv.z);
    let m = brick_voxel_material(hbi, vi);
    let q_albedo = palette[m].rgb;
    let qn = vec3<f32>(hn);
    if (best_t < *near_t) {
        *near_t = best_t;
        *near_n = qn;
    }
    let q_ndl = max(0.0, dot(qn, s));
    var q_shadow = 0.0;
    if (q_ndl > 0.0) {
        let q_world = (o + d * best_t) + vec3<f32>(camera.world_origin) + qn * GI_SHADOW_BIAS;
        q_shadow = select(1.0, 0.0, shadow_occluded(q_world, s, SHADOW_MAX_DIST));
    }
    return vec4<f32>(q_albedo * (sc * (q_ndl * q_shadow) + amb * 0.35), best_t);
}

// Few fresh rays per probe per update; the fixed direction-set cycling (below)
// + hysteresis integrate the full 64-direction sphere over GI_DIR_EPOCHS
// rounds. Every probe stays live as the camera moves.
const PROBE_RAYS: i32 = 8;
// Blend of each COMPLETE 64-direction estimate into the displayed SH (the
// old per-round EMA at 0.94/0.97 chased every 8-ray subset: random subsets
// random-walked, deterministic subsets rippled at the cycle period). With
// complete deterministic estimates the static-scene fixpoint is exact, so
// this constant only paces response to REAL lighting change: 0.6 converges
// in 2-3 cycles (~0.5-2 s), faster than the old 0.97 per-round EMA.
const PROBE_FOLD_HYSTERESIS: f32 = 0.6;
// Gather-direction epochs: PROBE_RAYS x GI_DIR_EPOCHS = 64 fixed spherical-
// Fibonacci directions per probe, one 8-ray subset per update round. A
// DETERMINISTIC periodic set makes a static scene converge to a constant;
// the old camera.time-hashed jitter re-rolled the directions every round, so
// each 8-ray estimate was a fresh random variable and the hysteresis EMA
// random-walked forever - visible as probe-cell patches slowly growing and
// shrinking wherever indirect light dominates (shadow interiors, underwater),
// and only with RT on (probes are RT-only). MUST match GI_PROBE_DIR_EPOCHS
// in renderer.rs (camera.gi_round counts frames mod DIV*EPOCHS).
const GI_DIR_EPOCHS: u32 = 8u;

// Amortization: update only 1/GI_UPDATE_DIV of the probes each frame, strided by
// camera.gi_round so the whole grid refreshes over GI_UPDATE_DIV frames. The
// dispatch launches PROBE_TOTAL/GI_UPDATE_DIV threads; thread g owns world probe
// g*DIV + round. This is the big per-frame win: the gather is the fixed cost that
// runs even on a static camera, so cutting it 1/DIV directly lifts the framerate.
override GI_UPDATE_DIV: f32 = 8.0;

@compute @workgroup_size(64)
fn cs_gi_probe_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let div = i32(GI_UPDATE_DIV);
    let idx = i32(gid.x) * div + (camera.gi_round % div);
    if (idx >= PROBE_DIM_X * PROBE_DIM_Y * PROBE_DIM_Z) { return; }

    // The slot's world grid coordinate for the CURRENT window: unfold the linear
    // slot to a base grid coord, then shift by the window origin in probe units.
    let sx = idx % PROBE_DIM_X;
    let sy = (idx / PROBE_DIM_X) % PROBE_DIM_Y;
    let sz = idx / (PROBE_DIM_X * PROBE_DIM_Y);
    // World grid coordinate = local slot + window origin offset (in probe cells),
    // folded so the probe sits inside the active window near the camera.
    let origin_g = camera.world_origin / PROBE_SPACING;
    let base = vec3<i32>(sx, sy, sz);
    // Choose the world cell in [origin_g, origin_g + PROBE_DIM) whose slot == base.
    var g = base;
    g.x = origin_g.x + pos_mod(base.x - origin_g.x, PROBE_DIM_X);
    g.z = origin_g.z + pos_mod(base.z - origin_g.z, PROBE_DIM_Z);
    // Y is not streamed (window covers full height), so the slot maps directly.
    g.y = base.y;

    let wpos = probe_world_pos(g);               // absolute world voxel position

    var pr = gi_probes[idx];
    let is_new = pr.info.w < 0.5 || any(pr.info.xyz != vec3<f32>(g));
    // Relocation offset (st_info.yzw): rays originate from the relocated
    // position, and the sampler uses it for all geometric terms. A fresh
    // slot starts unrelocated.
    let off_prev = select(pr.st_info.yzw, vec3<f32>(0.0), is_new);
    let o = wpos - vec3<f32>(camera.world_origin) + off_prev; // window-local

    // Gather over this round's 8-direction subset of the fixed 64-point
    // spherical-Fibonacci set. k is strided by GI_DIR_EPOCHS so every subset
    // spans the full z range (no per-round hemisphere bias). The set is the
    // SAME for every probe, deliberately: with a deterministic quadrature the
    // per-probe error is a frozen bias, and per-probe rotations turn that
    // bias into a stable blotch pattern between neighbouring probes (seen in
    // shadow interiors/underwater, where GI is all the light). A shared set
    // makes the bias spatially uniform, so neighbour differences are real
    // lighting only.
    let epoch = f32((u32(camera.gi_round) / u32(GI_UPDATE_DIV)) % GI_DIR_EPOCHS);
    let n_total = f32(PROBE_RAYS) * f32(GI_DIR_EPOCHS);
    var near_t = 1.0e30;
    var near_n = vec3<f32>(0.0);
    var c_r = vec4<f32>(0.0);
    var c_g = vec4<f32>(0.0);
    var c_b = vec4<f32>(0.0);
    var c_d = vec4<f32>(0.0);
    var c_d2 = vec4<f32>(0.0);
    let wpp = 12.566371 / f32(PROBE_RAYS); // 4*pi / N (uniform-sphere MC weight)
    for (var k: i32 = 0; k < PROBE_RAYS; k = k + 1) {
        let fib_i = f32(k) * f32(GI_DIR_EPOCHS) + epoch;
        let z = 1.0 - 2.0 * (fib_i + 0.5) / n_total;
        let r = sqrt(max(0.0, 1.0 - z * z));
        let phi = 2.3999632 * fib_i; // golden angle
        let d = vec3<f32>(r * cos(phi), z, r * sin(phi));
        let rad = probe_ray_radiance(o, d, &near_t, &near_n);
        let b = sh_basis(d);
        c_r = c_r + (wpp * rad.x) * b;
        c_g = c_g + (wpp * rad.y) * b;
        c_b = c_b + (wpp * rad.z) * b;
        c_d = c_d + (wpp * rad.w) * b;
        c_d2 = c_d2 + (wpp * rad.w * rad.w) * b;
    }

    if (is_new) {
        // Seed the display with this round's subset so a fresh slot lights
        // immediately; the first complete cycle replaces it with the full
        // 64-direction estimate.
        pr.sh_r = c_r;
        pr.sh_g = c_g;
        pr.sh_b = c_b;
        pr.sh_dist = c_d;
        pr.sh_dist2 = c_d2;
        pr.st_r = vec4<f32>(0.0);
        pr.st_g = vec4<f32>(0.0);
        pr.st_b = vec4<f32>(0.0);
        pr.st_dist = vec4<f32>(0.0);
        pr.st_dist2 = vec4<f32>(0.0);
        pr.st_info = vec4<f32>(0.0);
    } else {
        pr.st_r = pr.st_r + c_r;
        pr.st_g = pr.st_g + c_g;
        pr.st_b = pr.st_b + c_b;
        pr.st_dist = pr.st_dist + c_d;
        pr.st_dist2 = pr.st_dist2 + c_d2;
        pr.st_info.x = pr.st_info.x + 1.0;
        // Cycle complete: fold the averaged FULL-sphere estimate into the
        // display. Only complete (or window-shift-truncated) estimates ever
        // reach the displayed SH, so identical scenes produce identical
        // steps: static scene => constant probes, by construction.
        if (u32(epoch) == GI_DIR_EPOCHS - 1u) {
            let cnt = max(pr.st_info.x, 1.0);
            pr.sh_r = mix(pr.st_r / cnt, pr.sh_r, PROBE_FOLD_HYSTERESIS);
            pr.sh_g = mix(pr.st_g / cnt, pr.sh_g, PROBE_FOLD_HYSTERESIS);
            pr.sh_b = mix(pr.st_b / cnt, pr.sh_b, PROBE_FOLD_HYSTERESIS);
            pr.sh_dist = mix(pr.st_dist / cnt, pr.sh_dist, PROBE_FOLD_HYSTERESIS);
            pr.sh_dist2 = mix(pr.st_dist2 / cnt, pr.sh_dist2, PROBE_FOLD_HYSTERESIS);
            pr.st_r = vec4<f32>(0.0);
            pr.st_g = vec4<f32>(0.0);
            pr.st_b = vec4<f32>(0.0);
            pr.st_dist = vec4<f32>(0.0);
            pr.st_dist2 = vec4<f32>(0.0);
            pr.st_info = vec4<f32>(0.0);
        }
    }
    // ---- probe relocation (DDGI) ----
    // A probe hugging (or inside) geometry measures crevice darkness and
    // paints grid-scale dark footprints into the interpolation. Nudge it
    // away from the nearest surface it saw this round (bounded step, clamped
    // to stay well inside its own cell); drift home once clear so a changed
    // world re-relocates. near_t is measured from the relocated origin.
    var off = off_prev;
    if (near_t < GI_RELOC_NEAR) {
        off = clamp(off + near_n * min(GI_RELOC_NEAR - near_t, GI_RELOC_STEP),
                    vec3<f32>(-GI_RELOC_MAX), vec3<f32>(GI_RELOC_MAX));
    } else if (near_t > GI_RELOC_CLEAR) {
        off = off * 0.98;
    }
    pr.st_info = vec4<f32>(pr.st_info.x, off);
    pr.info = vec4<f32>(vec3<f32>(g), 1.0);
    gi_probes[idx] = pr;
}
