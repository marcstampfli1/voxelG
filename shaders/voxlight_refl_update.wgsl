// Per-voxel reflection cache: the amortized update pass.
//
// Concatenated LAST into the raymarch module, after voxlight_update.wgsl,
// because it calls `trace_secondary` and `shade` (whose definitions differ
// between the software and RT variants) plus `vl_brick_world_base` from the
// light-field update pass.
//
// Shape is deliberately identical to cs_voxel_light_update: one workgroup per
// live brick, one invocation per voxel (workgroup_size 64 IS BRICK_VOXELS, so
// `local_invocation_index` is the brick voxel index and needs no bounds check),
// the same `wg.x * div + round % div` slicing of the work list, and the same
// epoch cycling of a fixed direction set. Two amortization schemes that could
// drift out of step is exactly the failure the light field was built to avoid.
//
// Determinism is the design constraint, as it is there: each round samples a
// fixed slice of a fixed hemisphere direction set, so a static scene and static
// sun converge to a constant instead of random-walking.
//
// The record layout, the DC bias on the directional terms and the
// unwritten-vs-black rule are all documented at the sampler in raymarch.wgsl
// (search VL_REFL_RECORD_WORDS); this file is the only writer.

const VL_REFL_EPOCHS: u32 = 8u;

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

/// Direction `i` of this round's slice of the hemisphere direction set.
///
/// cos(theta) uniform in (0,1] is uniform in SOLID ANGLE, which is what the
/// moment estimator in cs_voxel_refl_update assumes: it fits the SH-L1 from
/// E[L] and E[L * d], and those are only the right moments for an unbiased
/// hemisphere sample.
///
/// The stratum index INTERLEAVES the rounds (i * EPOCHS + epoch) instead of
/// blocking them (epoch * rays + i, the way the sun pass slices its disc). The
/// sun disc is a couple of degrees wide, so a clustered slice costs it nothing;
/// a whole hemisphere is not, and a round that sampled only one elevation band
/// would fold a badly biased estimate in every round.
///
/// The epochs are then visited in BIT-REVERSED order. The fold below is an
/// exponential average, so it weights recent rounds more heavily; if the epochs
/// walked the elevation strata monotonically, that recency bias would line up
/// with elevation and the stored value would breathe up and down the hemisphere
/// once per cycle. Bit reversal is a permutation of the same eight epochs - the
/// full direction set is untouched - so consecutive rounds jump ACROSS the
/// hemisphere and every point in the cycle is weighted over its whole range.
fn vlr_ray_dir(n: vec3<f32>, t: vec3<f32>, b: vec3<f32>,
               i: u32, rays: u32, epoch: u32) -> vec3<f32> {
    // 3-bit reversal (b2 b1 b0 -> b0 b1 b2); VL_REFL_EPOCHS is 8.
    let er = ((epoch & 1u) << 2u) | (epoch & 2u) | ((epoch >> 2u) & 1u);
    let total = rays * VL_REFL_EPOCHS;
    let j = i * VL_REFL_EPOCHS + er;
    let ct = (f32(j) + 0.5) / f32(total);
    let st = sqrt(max(0.0, 1.0 - ct * ct));
    // Azimuth: an evenly spaced ring WITHIN the round, rotated by the golden
    // fraction per epoch so the full eight-round set spirals instead of
    // retracing one ring eight times.
    let phi = 6.28318531 * (f32(i) / f32(rays) + f32(er) * 0.6180339887);
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
    let div = max(1u, vl_params.update_div);
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
    let epoch = (vl_params.round / div) % VL_REFL_EPOCHS;
    // Time-free shading jitter, unlike the per-pixel callers. This pass FOLDS
    // its result, so a jitter that moved every frame would make the stored
    // value shimmer instead of converge; it varies with position and epoch so a
    // full cycle still averages over jitter phases.
    let jit = fract(f32(wv.x) * 17.0 + f32(wv.z) * 23.0 + f32(epoch) * 0.6180339887);
    // Secondary rays don't use the reprojection cache.
    var no_cache = vec2<f32>(0.0);

    // Raw SH-L1 moments of this round's rays: a0 = sum L, a{x,y,z} = sum L*d.
    var a0 = vec3<f32>(0.0);
    var ax = vec3<f32>(0.0);
    var ay = vec3<f32>(0.0);
    var az = vec3<f32>(0.0);
    for (var i = 0u; i < rays; i = i + 1u) {
        let d = vlr_ray_dir(nv, t, b, i, rays, epoch);
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

    // Fold exactly the way the sun pass folds sun_vis: an exponential average
    // over the direction epochs, so the stored value converges to the mean over
    // the WHOLE direction set and then stops changing. A record that has never
    // been written takes the fresh estimate outright rather than blending into
    // whatever the previous tenant of the block left behind.
    //
    // The MOMENTS are folded, not a per-round fit. A least-squares fit from two
    // or four rays is degenerate and would swing wildly round to round, while
    // the moments are unbiased estimates that average correctly - the fit is
    // done once, at sample time, from the converged moments.
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
