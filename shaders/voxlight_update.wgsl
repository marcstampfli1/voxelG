// Per-voxel light field: the amortized update pass.
//
// Concatenated LAST into the raymarch module, because it calls both
// `shadow_occluded` (whose definition differs between the software and RT
// variants) and the AO/sun helpers from the render body.
//
// One workgroup per brick, one invocation per voxel: workgroup_size 64 matches
// BRICK_VOXELS exactly, so `local_invocation_index` IS the brick voxel index
// and no bounds check or remainder loop is needed.
//
// Determinism is the design constraint. Each round samples a fixed subset of a
// fixed direction set (spherical Fibonacci over the sun disc), cycled by epoch,
// so a static scene and static sun converge to a constant instead of
// random-walking. That is what makes the result stable rather than noisy, and
// it is the same shape the GI probe grid already uses (gi_probes.wgsl:248).

const VL_SUN_EPOCHS: u32 = 8u;
// Weighted 6-face + 12-edge occupancy. Face neighbours block twice the solid
// angle of an edge neighbour, so 2*6 + 12 = 24 is a fully enclosed voxel.
const VL_AO_DENOM: f32 = 24.0;

/// World voxel coord of a storage brick's minimum corner.
///
/// Inverts the storage linearisation, then undoes the toroidal fold using the
/// current window origin: the window holds exactly one world voxel per storage
/// cell, so `world = origin + ((storage - origin) mod extent)`. y is never
/// folded (world_to_slot_voxel passes it through), so storage y IS world y.
fn vl_brick_world_base(brick: u32) -> vec3<i32> {
    let bi = i32(brick);
    let sbx = bi % WORLD_BRICKS_X;
    let sby = (bi / WORLD_BRICKS_X) % WORLD_BRICKS_Y;
    let sbz = bi / (WORLD_BRICKS_X * WORLD_BRICKS_Y);
    let sv = vec3<i32>(sbx, sby, sbz) * BRICK_DIM;
    let o = camera.world_origin;
    return vec3<i32>(
        o.x + pos_mod(sv.x - o.x, WORLD_VOXELS_X),
        sv.y,
        o.z + pos_mod(sv.z - o.z, WORLD_VOXELS_Z),
    );
}

/// Face-independent ambient occlusion for an AIR voxel.
///
/// The per-pixel formula this replaces (`compute_ao`) evaluated four corners of
/// one FACE and bilinearly blended them. A per-voxel field cannot be
/// face-indexed, so occlusion is measured for the air cell itself and the
/// smooth gradient comes from the trilinear fetch instead of the in-face blend.
/// `ao_occluder` is reused verbatim so decoration cells (grass tufts, flowers,
/// the invisible canopy fringe) still do not stamp AO squares onto the ground.
fn vl_voxel_ao(v: vec3<i32>) -> f32 {
    var occ = 0.0;
    // 6 face neighbours, weight 2.
    if (ao_occluder(v + vec3<i32>(1, 0, 0))) { occ = occ + 2.0; }
    if (ao_occluder(v + vec3<i32>(-1, 0, 0))) { occ = occ + 2.0; }
    if (ao_occluder(v + vec3<i32>(0, 1, 0))) { occ = occ + 2.0; }
    if (ao_occluder(v + vec3<i32>(0, -1, 0))) { occ = occ + 2.0; }
    if (ao_occluder(v + vec3<i32>(0, 0, 1))) { occ = occ + 2.0; }
    if (ao_occluder(v + vec3<i32>(0, 0, -1))) { occ = occ + 2.0; }
    // 12 edge neighbours, weight 1.
    for (var a = 0u; a < 3u; a = a + 1u) {
        for (var s0 = 0u; s0 < 2u; s0 = s0 + 1u) {
            for (var s1 = 0u; s1 < 2u; s1 = s1 + 1u) {
                let d0 = select(-1, 1, s0 == 1u);
                let d1 = select(-1, 1, s1 == 1u);
                var e = vec3<i32>(0);
                if (a == 0u) { e = vec3<i32>(0, d0, d1); }
                else if (a == 1u) { e = vec3<i32>(d0, 0, d1); }
                else { e = vec3<i32>(d0, d1, 0); }
                if (ao_occluder(v + e)) { occ = occ + 1.0; }
            }
        }
    }
    return clamp(1.0 - vl_params.ao_strength * (occ / VL_AO_DENOM), 0.0, 1.0);
}

/// Fraction of the sun disc reaching `p`, estimated from this round's slice of
/// the direction set. Returns a CONTINUOUS value; the caller folds it into the
/// stored estimate.
fn vl_sun_visibility(p: vec3<f32>, epoch: u32) -> f32 {
    let s = sun_dir();
    // Tangent frame around the sun direction so the disc is sampled uniformly
    // regardless of sun azimuth (same construction the old per-pixel cone used).
    var tangent = normalize(cross(s, vec3<f32>(0.0, 1.0, 0.0)));
    if (length(cross(s, vec3<f32>(0.0, 1.0, 0.0))) < 0.01) {
        tangent = vec3<f32>(1.0, 0.0, 0.0);
    }
    let bitangent = cross(s, tangent);

    let rays = max(1u, vl_params.sun_rays);
    let total = f32(rays * VL_SUN_EPOCHS);
    var lit = 0.0;
    for (var i = 0u; i < rays; i = i + 1u) {
        // Global index into the full direction set, so successive epochs walk
        // DIFFERENT directions and a complete cycle covers the disc evenly.
        let gi = f32(epoch * rays + i);
        let ang = gi * 2.39996323;             // golden angle
        let rad = vl_params.sun_cone * sqrt((gi + 0.5) / total);
        let d = normalize(s + (tangent * cos(ang) + bitangent * sin(ang)) * rad);
        if (!shadow_occluded(p, d, SHADOW_MAX_DIST)) {
            lit = lit + 1.0;
        }
    }
    return lit / f32(rays);
}

/// Local light reaching `p`, shadow-tested per light.
fn vl_point_light(p: vec3<f32>) -> vec3<f32> {
    var sum = vec3<f32>(0.0);
    let n = vl_params.light_count;
    for (var i = 0u; i < n; i = i + 1u) {
        let l = vl_lights[i];
        let to = l.pos_radius.xyz - p;
        let d2 = dot(to, to);
        let r = l.pos_radius.w;
        if (d2 >= r * r) { continue; }
        let d = sqrt(max(d2, 1e-6));
        // Inverse-square with a smooth cutoff at the radius, so a light never
        // ends in a visible hard circle.
        let falloff = clamp(1.0 - d / r, 0.0, 1.0);
        let atten = falloff * falloff / max(d2, 1.0);
        if (shadow_occluded(p, to / d, d)) { continue; }
        sum = sum + l.color.rgb * atten;
    }
    return sum;
}

@compute @workgroup_size(64, 1, 1)
fn cs_voxel_light_update(@builtin(workgroup_id) wg: vec3<u32>,
                         @builtin(local_invocation_index) li: u32) {
    let div = max(1u, vl_params.update_div);
    // This round's slice of the work list.
    let idx = wg.x * div + (vl_params.round % div);
    if (idx >= vl_params.live_count) { return; }
    let brick = vl_live_bricks[idx];
    let block = vl_block_of_brick[brick];
    if (block == VL_NONE) { return; }

    // local_invocation_index IS the brick voxel index; invert
    // brick_voxel_idx (lx + lz*4 + ly*16).
    let lx = i32(li % 4u);
    let lz = i32((li / 4u) % 4u);
    let ly = i32(li / 16u);
    let wv = vl_brick_world_base(brick) + vec3<i32>(lx, ly, lz);
    let word = block * VL_BLOCK_WORDS + li * VL_RECORD_WORDS;

    // Light lives in AIR. A solid voxel keeps epoch 0 so the sampler's
    // validity check drops it even before the solidity gate does.
    if (is_voxel_solid(wv)) {
        vl_pool[word] = 0u;
        vl_pool[word + 1u] = 0u;
        return;
    }

    let p = vec3<f32>(wv) + vec3<f32>(0.5);
    let epoch = (vl_params.round / div) % VL_SUN_EPOCHS;

    let prev = vl_pool[word];
    let prev_epoch = (prev >> 16u) & 0xFFu;
    let fresh_sun = vl_sun_visibility(p, epoch);

    // Fold: an exponential average over the direction epochs, so the stored
    // value converges to the mean visibility over the whole disc. A freshly
    // bound or invalidated record (epoch 0) takes the fresh estimate outright
    // rather than blending into whatever the previous tenant left behind.
    var sun = fresh_sun;
    if (prev_epoch != 0u) {
        sun = mix(f32(prev & 0xFFu) * (1.0 / 255.0), fresh_sun, vl_params.fold);
    }

    let ao = vl_voxel_ao(wv);
    let pt = vl_point_light(p);

    // epoch byte: 1..255, never 0, since 0 is the "no valid history" marker.
    let stamp = 1u + (epoch % 255u);
    vl_pool[word] = (u32(round(clamp(sun, 0.0, 1.0) * 255.0)))
        | (u32(round(clamp(ao, 0.0, 1.0) * 255.0)) << 8u)
        | (stamp << 16u);
    vl_pool[word + 1u] = vl_pack_rgb9e5(pt);
}
