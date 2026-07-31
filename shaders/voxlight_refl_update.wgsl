// Per-voxel reflection cache: the amortized update pass.
//
// Concatenated LAST into the raymarch module, after voxlight_update.wgsl,
// because it calls `trace_secondary` and `shade` (whose definitions differ
// between the software and RT variants) plus `vl_brick_world_base` from the
// light-field update pass.
//
// Dispatch shape is identical to cs_voxel_light_update: one workgroup per live
// brick, one invocation per voxel (workgroup_size 64 IS BRICK_VOXELS, so
// `local_invocation_index` is the brick voxel index and needs no bounds check),
// and the same `wg.x * div + round % div` slicing of the work list.
//
// The AMORTIZATION is NOT identical, and the difference is load-bearing. This
// pass visits a block once every `refl_update_div` = 64 rounds and gathers a
// COMPLETE stratified hemisphere on each visit (`refl_rays` = VL_REFL_RAYS = 8
// elevation strata); the light pass visits every 8 rounds and gathers a partial
// estimate. Same rays per frame either way - 64/8 = 8 times rarer, 8 times as
// many rays - but not the same result, and the difference was MEASURED, not
// argued:
//
// SAME RAYS PER FRAME IS NOT THE SAME COST, and the measurement says so: the
// reflection pass went 0.23 -> 0.51 ms/frame across this reshape. The ray count
// really is unchanged (16,448 -> 16,896 ray slots on the demo world), but the
// DISPATCH SHAPE is not: 2050 live blocks at div 8 is 257 workgroups, at div 64
// it is 33, so eight serial traces per thread replace eight workgroups' worth
// of latency hiding. On this world it is worth paying - water-close still saves
// 2.84 ms of shading against a 1.64 ms total update - but it is a real cost,
// and it is worst on worlds with LITTLE reflective geometry, where there are
// too few blocks to fill the machine. A wider world has more of both.
//
//   1 ray per visit, folded at 0.125:  the shaded value of a water voxel swung
//   22.7-46.8% peak to peak over 32 consecutive rounds with a static scene, a
//   static sun and a FIXED query direction, and the stored E[L*d.z] sat at
//   -0.17 where the true moment is unambiguously POSITIVE (the only dark thing
//   in the hemisphere is a wall at -z). That is the whole "flickery and not
//   correctly reflecting" report: an exponential average of single rays drawn
//   from directions whose radiance spans an order of magnitude never converges
//   to the hemisphere mean, it random-walks behind the direction cycle, and at
//   any instant the record holds a recency-weighted sample of the last two or
//   three rays rather than an estimate of anything.
//
//   8 rays per visit: the same measurement reads 0.0% - the record is BITWISE
//   constant once converged - and m_z reads +0.035, the right sign. Pinned by
//   `voxlight_refl_record_is_stable_across_rounds`.
//
// Those two ranges were re-measured from scratch against the shipped estimator
// and its predecessor (`voxlight_refl_diagnose`, section C). An earlier note
// here quoted 41-75% and 1.5-3.6%; neither reproduced, and the numbers above
// are the ones that do. The 22.7-46.8% figure is the FULL old build, estimator
// and reconstruction together; the old estimator against today's corrected
// reconstruction reads 16.9-34.5%, because the 4/3 tangential over-weight
// documented at `voxlight_reflection` amplified the estimator's swing as well
// as its steady-state error.
//
// The fix is the estimator, not the fold. Lowering the fold would only trade
// swing for lag: the input sequence is not noise about the right answer, it is
// a cycle through eight different answers, and no first-order filter over it
// is the hemisphere mean at any instant. A complete estimate per visit is,
// which is why a static scene now converges to a constant rather than to a
// limit cycle. It is the same conclusion the GI probe grid reached
// (gi_probes.wgsl, "fold only finished estimates").
//
// Determinism is the design constraint, as it is in the light pass: each visit
// samples a fixed stratified direction set, so a static scene and static sun
// converge to a constant instead of random-walking.
//
// The record layout, the DC bias on the directional terms and the
// unwritten-vs-black rule are all documented at the sampler in raymarch.wgsl
// (search VL_REFL_RECORD_WORDS); this file is the only writer.

// Rays in one gather. Mirrors VOXLIGHT_REFL_RAYS on the CPU, and with
// VOXLIGHT_REFL_UPDATE_DIV = VOXLIGHT_UPDATE_DIV * this, gathering completely
// but rarely costs the same rays per frame as gathering partially but often.
const VL_REFL_RAYS: u32 = 8u;

/// Outward normal of the face a glass voxel reflects from, or (0,0,0) when the
/// voxel is fully enclosed and can reflect nothing.
///
/// APPROXIMATION, and the sharpest edge of the per-voxel scheme: one record
/// describes ONE hemisphere, but a glass voxel has up to six exposed faces and
/// a window pane has two opposed ones. The dominant face is picked by how much
/// open space sits in front of it, so a pane records the side that actually
/// sees the world rather than whichever face was enumerated first. A view of
/// the other face is caught by the sampler (the fit reconstructs past -DC
/// there) and falls back to the per-pixel trace, so the failure mode is cost,
/// never a wrong colour.
fn vlr_glass_face(v: vec3<i32>) -> vec3<f32> {
    var best = vec3<f32>(0.0);
    var best_score = -1;
    // +Y is enumerated first so it wins ties: a glass block's top is the face
    // that most often carries a visible reflection.
    for (var f = 0u; f < 6u; f = f + 1u) {
        var d = vec3<i32>(0, -1, 0);
        if (f == 0u) { d = vec3<i32>(0, 1, 0); }
        else if (f == 1u) { d = vec3<i32>(1, 0, 0); }
        else if (f == 2u) { d = vec3<i32>(-1, 0, 0); }
        else if (f == 3u) { d = vec3<i32>(0, 0, 1); }
        else if (f == 4u) { d = vec3<i32>(0, 0, -1); }
        // A face buried against another block reflects nothing.
        if (is_voxel_solid(v + d)) { continue; }
        var score = 1;
        if (!is_voxel_solid(v + d * 2)) { score = score + 1; }
        if (!is_voxel_solid(v + d * 3)) { score = score + 1; }
        if (score > best_score) {
            best_score = score;
            best = vec3<f32>(d);
        }
    }
    return best;
}

/// Ray `i` of the gather's spherical-Fibonacci hemisphere set.
///
/// cos(theta) uniform in (0,1] is uniform in SOLID ANGLE, which is what the
/// moment estimator in cs_voxel_refl_update assumes: it fits the SH-L1 from
/// E[L] and E[L * d], and those are only the right moments for an unbiased
/// hemisphere sample. Stratified elevation plus a golden-angle azimuth is the
/// standard low-discrepancy set for that measure, and it is the same
/// construction the GI probe grid uses.
///
/// THE SET DOES NOT DEPEND ON THE ROUND. That is deliberate and it is the
/// second half of the flicker fix. Rotating the set between visits does not
/// make the estimate better - the estimate is already complete and unbiased in
/// expectation - it only replaces a FIXED quadrature error with a VARYING one,
/// which is to say it converts bias into flicker. For a cache that is read by
/// every frame and whose entire job is to be stable, that trade is backwards:
/// a static scene must give the same eight rays and therefore bitwise the same
/// record, visit after visit, for ever.
///
/// It matters more than the general argument suggests because `sky_color` has
/// a HARD-EDGED sun disc (`sun_align > 0.9985`, radiance ~2.4 against a sky
/// around 0.5). It covers 0.15% of the hemisphere, so about one visit in eighty
/// lands a ray in it, and that visit's DC comes out ~35% high. Under a rotating
/// set that is a large intermittent spike in the stored value; under a fixed
/// one it is a constant, and a constant that is wrong by a few percent in a
/// single voxel's cached reflection is invisible where a spike is not.
///
/// What is given up is the averaging-out of quadrature error over a cycle.
/// That error is spatially coherent - every voxel uses the same eight
/// directions - so it reads as a slight, smooth, unchanging bias in the cached
/// radiance rather than as structure. Determinism was already this pass's
/// stated design constraint; this is what honouring it costs and buys.
fn vlr_ray_dir(n: vec3<f32>, t: vec3<f32>, b: vec3<f32>, i: u32, rays: u32) -> vec3<f32> {
    let ct = (f32(i) + 0.5) / f32(rays);
    let st = sqrt(max(0.0, 1.0 - ct * ct));
    let phi = 6.28318531 * fract(f32(i) * 0.6180339887);
    return normalize(t * (st * cos(phi)) + b * (st * sin(phi)) + n * ct);
}

/// Zero a record. An all-zero record reads as "never written" at the sampler,
/// so this is also what makes a RECYCLED block safe: a non-reflective voxel in
/// a reused block falls back to the per-pixel path instead of showing the
/// previous tenant's radiance.
fn vlr_clear(word: u32) {
    refl_pool[word] = 0u;
    refl_pool[word + 1u] = 0u;
    refl_pool[word + 2u] = 0u;
    refl_pool[word + 3u] = 0u;
}

@compute @workgroup_size(64, 1, 1)
fn cs_voxel_refl_update(@builtin(workgroup_id) wg: vec3<u32>,
                        @builtin(local_invocation_index) li: u32) {
    let div = max(1u, vl_params.refl_update_div);
    // This round's slice of the work list.
    let idx = wg.x * div + (vl_params.round % div);
    if (idx >= vl_params.refl_live_count) { return; }
    let brick = refl_live_bricks[idx];
    let block = refl_block_of_brick[brick];
    if (block == VL_NONE) { return; }

    // local_invocation_index IS the brick voxel index; invert
    // brick_voxel_idx (lx + lz*4 + ly*16).
    let lx = i32(li % 4u);
    let lz = i32((li / 4u) % 4u);
    let ly = i32(li / 16u);
    let wv = vl_brick_world_base(brick) + vec3<i32>(lx, ly, lz);
    let word = block * VL_REFL_BLOCK_WORDS + li * VL_REFL_RECORD_WORDS;

    // Reflective materials only, read through the shared material accessor and
    // the shared predicates - the material ids live in exactly one place.
    let m = voxel_material_at(wv);
    var nv = vec3<f32>(0.0);
    if (is_water_mat(m)) {
        // Water reflects at its TOP only. A cell with anything above it is
        // submerged and is never shaded by shade_water_top's plate path, and a
        // lake is mostly submerged cells, so skipping them keeps the whole ray
        // budget on actual surfaces.
        if (!is_voxel_solid(wv + vec3<i32>(0, 1, 0))) {
            // APPROXIMATION: the pole is world up, not the animated Gerstner
            // facet normal. Water surfaces are horizontal tops; the facets tilt
            // a few degrees over roughly 0.1 voxel of wave amplitude, far below
            // what a hemisphere-wide L1 fit resolves, so a per-facet pole would
            // buy nothing and would make the stored value move with the waves.
            nv = vec3<f32>(0.0, 1.0, 0.0);
        }
    } else if (m == MAT_GLASS) {
        nv = vlr_glass_face(wv);
    }
    if (all(nv == vec3<f32>(0.0))) {
        vlr_clear(word);
        return;
    }

    // Tangent frame for the hemisphere sweep. `nv` is always axis aligned here,
    // so the branch picks a reference that is never parallel to it.
    var t = normalize(cross(nv, vec3<f32>(0.0, 1.0, 0.0)));
    if (abs(nv.y) > 0.9) { t = normalize(cross(nv, vec3<f32>(1.0, 0.0, 0.0))); }
    let b = cross(nv, t);

    // Half a voxel plus a hair puts the origin just outside the reflective
    // cell, so a ray cannot start inside the surface it belongs to. For water
    // that is the cell's TOP FACE rather than the wave plate a fraction of a
    // voxel below it; at per-voxel resolution that offset is by definition
    // unresolvable.
    let p0 = vec3<f32>(wv) + vec3<f32>(0.5) + nv * 0.55;

    let rays = max(1u, vl_params.refl_rays);
    // Shading jitter that depends on NEITHER time NOR the round, unlike the
    // per-pixel callers. Same reason the direction set does not: anything that
    // varies between visits of a static scene is stored flicker. Varying with
    // position keeps it from banding across a surface.
    let jit = fract(f32(wv.x) * 17.0 + f32(wv.z) * 23.0);
    // Secondary rays don't use the reprojection cache.
    var no_cache = vec2<f32>(0.0);

    // Raw SH-L1 moments of this visit's rays: a0 = sum L, a{x,y,z} = sum L*d.
    var a0 = vec3<f32>(0.0);
    var ax = vec3<f32>(0.0);
    var ay = vec3<f32>(0.0);
    var az = vec3<f32>(0.0);
    for (var i = 0u; i < rays; i = i + 1u) {
        let d = vlr_ray_dir(nv, t, b, i, rays);
        // trace_secondary skips transparent voxels, so a ray never hits the
        // water sheet or pane it started from, or its neighbours.
        let h = trace_secondary(p0, d, SECONDARY_MAX_T);
        var col: vec3<f32>;
        if (h.hit) {
            col = shade(h, p0, d, jit, false, false, false, &no_cache, vec3<f32>(0.0));
        } else {
            col = sky(d);
        }
        // The |m_a| <= m0 bound the DC bias relies on needs L >= 0.
        col = max(col, vec3<f32>(0.0));
        a0 = a0 + col;
        ax = ax + col * d.x;
        ay = ay + col * d.y;
        az = az + col * d.z;
    }
    let inv = 1.0 / f32(rays);
    var m0 = a0 * inv;
    var mx = ax * inv;
    var my = ay * inv;
    var mz = az * inv;

    // Fold a FINISHED estimate into the stored one. Every visit's estimate is
    // already a complete stratified hemisphere over a FIXED direction set, so
    // on a static scene the fold is a no-op - the value converges because the
    // estimate is whole and repeatable, not because the filter smoothed it -
    // and its only job is to ease the record across a moving sun and changing
    // geometry. That is why it can be as fast as the light field's 0.35 without
    // reintroducing swing, where the old shape needed 0.125 and swung anyway.
    //
    // A record that has never been written takes the fresh estimate outright
    // rather than blending into whatever the previous tenant of the block left
    // behind.
    //
    // The MOMENTS are folded, not a per-visit fit. A least-squares fit from a
    // handful of rays is degenerate and would swing visit to visit, while the
    // moments are unbiased estimates that average correctly - the fit is done
    // once, at sample time, from the converged moments.
    let q0w = refl_pool[word];
    let q1w = refl_pool[word + 1u];
    let q2w = refl_pool[word + 2u];
    let q3w = refl_pool[word + 3u];
    if ((q0w | q1w | q2w | q3w) != 0u) {
        let q0 = vl_unpack_rgb9e5(q0w);
        let fold = vl_params.refl_fold;
        mx = mix(vl_unpack_rgb9e5(q1w) - q0, mx, fold);
        my = mix(vl_unpack_rgb9e5(q2w) - q0, my, fold);
        mz = mix(vl_unpack_rgb9e5(q3w) - q0, mz, fold);
        m0 = mix(q0, m0, fold);
    }

    // Floor the DC so a written record can never pack to all zeros; that floor
    // is the whole mechanism behind the sampler reading all-zero as "never
    // written" instead of guessing (see the sampler's block comment).
    let dc = max(m0, vec3<f32>(VL_REFL_MIN));
    // |E[L * d_a]| <= E[L] holds by construction, but the previous round's
    // moments came back through 9-bit mantissas, so re-establish the bound
    // explicitly rather than letting vl_pack_rgb9e5's clamp swallow a negative.
    refl_pool[word] = vl_pack_rgb9e5(dc);
    refl_pool[word + 1u] = vl_pack_rgb9e5(clamp(mx, -dc, dc) + dc);
    refl_pool[word + 2u] = vl_pack_rgb9e5(clamp(my, -dc, dc) + dc);
    refl_pool[word + 3u] = vl_pack_rgb9e5(clamp(mz, -dc, dc) + dc);
}
