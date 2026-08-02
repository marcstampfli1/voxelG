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
// Determinism is the design constraint, and it is stronger than it used to be:
// a visit estimates the sun disc COMPLETELY, from a direction set that depends
// on nothing but the ray index, so a static scene under a static sun converges
// to a fixed point and then stops changing. Cycling a SUBSET of the set per
// visit - which is what this did - looks like the same thing and is not: the
// subsets are different numbers rather than repeated looks at one, so the fold
// tracked them and the record oscillated for ever. See `vl_sun_visibility`.

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

/// Face-independent ambient occlusion for a record CELL whose low voxel is `v`.
///
/// The per-pixel formula this replaces (`compute_ao`) evaluated four corners of
/// one FACE and bilinearly blended them. A per-voxel field cannot be
/// face-indexed, so occlusion is measured for the air cell itself and the
/// smooth gradient comes from the trilinear fetch instead of the in-face blend.
/// `vl_cell_occluder` keeps `ao_occluder`'s rule verbatim, so decoration cells
/// (grass tufts, flowers, the invisible canopy fringe) still do not stamp AO
/// squares onto the ground.
///
/// The neighbours are the 18 adjacent record CELLS, one whole cell out in each
/// direction, NOT the 18 adjacent voxels. That is forced by the lattice: a cell
/// containing a surface is vetoed as opaque, so the nearest LIVE record sits one
/// or two voxels off the surface depending on the parity of the surface's
/// coordinate, and a single-voxel probe finds the ground for one parity and
/// misses it for the other. Measured before this: AO on flat open ground came
/// back 255 (no contact at all) for half of all ground heights, and a wall
/// shifted one voxel along its own normal changed brightness by 4.5% - the
/// blink that `texture_pattern_is_depth_invariant` exists to catch. At
/// VL_STEP == 1 every cell is a voxel and this is the per-voxel formula again.
///
/// Reach: one cell = VL_STEP voxels = 20 cm, against the 25 cm the per-voxel
/// kernel had at 25 cm voxels. Contact AO stays a contact-scale effect.
fn vl_cell_ao(v: vec3<i32>) -> f32 {
    let s = VL_STEP;
    var occ = 0.0;
    // 6 face-neighbour cells, weight 2 (a face blocks twice an edge's solid angle).
    if (vl_cell_occluder(v + vec3<i32>( s, 0, 0))) { occ = occ + 2.0; }
    if (vl_cell_occluder(v + vec3<i32>(-s, 0, 0))) { occ = occ + 2.0; }
    if (vl_cell_occluder(v + vec3<i32>(0,  s, 0))) { occ = occ + 2.0; }
    if (vl_cell_occluder(v + vec3<i32>(0, -s, 0))) { occ = occ + 2.0; }
    if (vl_cell_occluder(v + vec3<i32>(0, 0,  s))) { occ = occ + 2.0; }
    if (vl_cell_occluder(v + vec3<i32>(0, 0, -s))) { occ = occ + 2.0; }
    // 12 edge-neighbour cells, weight 1.
    for (var a = 0u; a < 3u; a = a + 1u) {
        for (var s0 = 0u; s0 < 2u; s0 = s0 + 1u) {
            for (var s1 = 0u; s1 < 2u; s1 = s1 + 1u) {
                let d0 = select(-s, s, s0 == 1u);
                let d1 = select(-s, s, s1 == 1u);
                var e = vec3<i32>(0);
                if (a == 0u) { e = vec3<i32>(0, d0, d1); }
                else if (a == 1u) { e = vec3<i32>(d0, 0, d1); }
                else { e = vec3<i32>(d0, d1, 0); }
                if (vl_cell_occluder(v + e)) { occ = occ + 1.0; }
            }
        }
    }
    return clamp(1.0 - vl_params.ao_strength * (occ / VL_AO_DENOM), 0.0, 1.0);
}

/// Fraction of the sun disc reaching `p`, over a COMPLETE stratified sampling of
/// the disc. Returns a CONTINUOUS value; the caller folds it into the stored
/// estimate.
///
/// COMPLETE, not a rotating slice, and that is the whole point. This used to
/// trace a QUARTER of the direction set per visit, selected by an epoch counter,
/// and fold it in at 0.35 - on the argument that a full cycle averages to the
/// true soft shadow. It does not. A penumbra voxel's four quarters are FOUR
/// DIFFERENT NUMBERS, not four noisy looks at one, so the fold tracks them
/// instead of averaging them and the record oscillates for ever even on a scene
/// and a sun that never move. Measured by `voxlight_sun_is_stable_once_converged`
/// on a static scene: a converged penumbra record swung 89/255 over one epoch
/// cycle, and `shade_water_top` pushes that through a smoothstep only 45/255
/// wide, so water at a shadow edge strobed between fully lit and fully shadowed.
///
/// The same defect, with the same cause and the same fix, was found in the
/// (since deleted) reflection field in round E of
/// docs/rt/BASELINE-per-voxel-lighting.md. It was never applied to this pass.
///
/// A complete estimate is a deterministic function of geometry and sun
/// direction, so a static scene converges to a fixed point and stays there, and
/// a moving sun tracks it smoothly instead of beating against the epoch cycle.
fn vl_sun_visibility(p: vec3<f32>) -> f32 {
    let s = sun_dir();
    // Tangent frame around the sun direction so the disc is sampled uniformly
    // regardless of sun azimuth (same construction the old per-pixel cone used).
    var tangent = normalize(cross(s, vec3<f32>(0.0, 1.0, 0.0)));
    if (length(cross(s, vec3<f32>(0.0, 1.0, 0.0))) < 0.01) {
        tangent = vec3<f32>(1.0, 0.0, 0.0);
    }
    let bitangent = cross(s, tangent);

    let rays = max(1u, vl_params.sun_rays);
    var lit = 0.0;
    for (var i = 0u; i < rays; i = i + 1u) {
        // Sunflower stratification of the disc: equal-area rings from the sqrt,
        // even azimuth from the golden angle. The set depends on NOTHING but the
        // ray index, so every visit samples exactly the same directions.
        let gi = f32(i);
        let ang = gi * 2.39996323;             // golden angle
        let rad = vl_params.sun_cone * sqrt((gi + 0.5) / f32(rays));
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

/// Where the URGENT list starts inside `vl_live_bricks`. The work list can hold
/// at most `LIGHT_BLOCKS_MAX` entries, so the slots past it are free for a
/// second, short list - which is why the urgent queue needs no binding of its
/// own and no second upload path. Emitted from `src/world_dims.rs`, so the CPU
/// side cannot drift from this.
const VL_URGENT_BASE: u32 = u32(LIGHT_BLOCKS_MAX);

/// The BRICK workgroup `wg` must gather this dispatch, or VL_NONE for nothing.
///
/// TWO SOURCES, in priority order, and the split is the whole of the "do no work
/// when nothing changed" policy:
///
///  - URGENT, `wg < urgent_count`: bricks whose block has no readable record
///    (newly bound, just invalidated) or whose neighbourhood just changed. These
///    are dispatched IN FULL, every dispatch, because their pixels are shading
///    through the fallback until they are gathered. Cost scales with how much of
///    the world actually changed.
///  - the SWEEP, everything past that and only when `sweep` is set: the periodic
///    re-gather that tracks the SUN. The list is partitioned near-first
///    (`LightField::near_count`); the near prefix is walked in `update_div`
///    slices and the far remainder in `update_div * far_div` slices, so a distant
///    block is visited that many times more rarely. The two groups are told apart
///    by the workgroup index alone - one dispatch, and every invocation within a
///    workgroup takes the same path.
///
/// Nothing here depends on the visit COUNT, only on which entry is due, because
/// each visit re-estimates the sun disc completely (`vl_sun_visibility`). That is
/// also what lets the sweep be paced by sun motion instead of by frames: a round
/// skipped because the sun did not move is a round whose output would have been
/// bit-identical.
fn vl_work(wg: u32) -> u32 {
    if (wg < vl_params.urgent_count) {
        return vl_live_bricks[VL_URGENT_BASE + wg];
    }
    let s = wg - vl_params.urgent_count;
    let div = max(1u, vl_params.update_div);
    let near = min(vl_params.near_count, vl_params.live_count);
    let near_wgs = (near + div - 1u) / div;
    var idx = 0u;
    if (s < near_wgs) {
        idx = s * div + (vl_params.round % div);
        // A near workgroup must not spill into the far region: it would refresh
        // a far block at the near cadence.
        if (idx >= near) { return VL_NONE; }
    } else {
        let fdiv = div * max(1u, vl_params.far_div);
        idx = near + (s - near_wgs) * fdiv + (vl_params.round % fdiv);
    }
    if (idx >= vl_params.live_count) { return VL_NONE; }
    return vl_live_bricks[idx];
}

// Bricks one workgroup covers. A record is a VL_STEP^3 group, so a brick holds
// only LIGHT_RECORDS_PER_BLOCK of them - 8 at VL_STEP 2. A workgroup of 8 would
// leave three quarters of every warp idle on the hardware this targets, so one
// workgroup gathers several bricks and stays 64 wide. `voxlight_wgs` in
// renderer.rs converts a work-list length into a dispatch size with the same
// divisor, and there is no second copy of it.
const VL_WG_BRICKS: u32 = 64u / u32(LIGHT_RECORDS_PER_BLOCK);

@compute @workgroup_size(64, 1, 1)
fn cs_voxel_light_update(@builtin(workgroup_id) wg: vec3<u32>,
                         @builtin(local_invocation_index) li: u32) {
    let rpb = u32(LIGHT_RECORDS_PER_BLOCK);
    // Lanes are grouped by brick so the 8 lanes sharing a block also share its
    // pool cache line and its brick occupancy word.
    //
    // The dispatch is TILED over x and y (`renderer::linear_dispatch`): a whole-
    // shell sweep is already tens of thousands of workgroups at 10 cm and every
    // dispatch dimension caps at 65,535. `vl_work` already returns VL_NONE past
    // the work list, which is what makes the over-dispatched tail row free.
    let lin = wg.x + wg.y * u32(DISPATCH_ROW_WGS);
    let slot = lin * VL_WG_BRICKS + li / rpb;
    let ri = li % rpb;
    let brick = vl_work(slot);
    if (brick == VL_NONE) { return; }
    let block = vl_block_of_brick[brick];
    if (block == VL_NONE) { return; }

    // Invert light_record_idx (rx + rz*D + ry*D^2) to the record's cell, then
    // scale to its LOW VOXEL - which is the point the sampler interpolates
    // through (see `voxlight_sample`).
    let d = u32(LIGHT_RECORD_DIM);
    let rx = i32(ri % d);
    let rz = i32((ri / d) % d);
    let ry = i32(ri / (d * d));
    let r = vec3<i32>(rx, ry, rz);
    let wv = vl_brick_world_base(brick) + r * VL_STEP;
    let word = block * VL_BLOCK_WORDS + ri * VL_RECORD_WORDS;

    // WHO CARRIES A RECORD: a cell with NO opaque voxel in it. Air, and FOLIAGE.
    //
    // A cell holding any opaque voxel keeps epoch 0, so the sampler's validity
    // check drops it even before `vl_group_blocks` does - and the two agree by
    // construction, which is what stops a half-solid cell from lighting the
    // inside of a wall. Foliage is not opaque: a canopy is a semi-transparent
    // volume, and a volume wants a value AT the sample point, not at some
    // adjacent air cell that on canopy is usually another leaf. That conflation
    // is what left `voxlight_sample` with nothing to return for 54% of a canopy
    // view; see `vl_tap`.
    //
    // Read straight out of the brick rather than through `is_voxel_solid`: this
    // invocation already knows its storage brick and its in-brick record coord,
    // so the whole hierarchy descent the old gate paid - bounds, toroidal fold,
    // chunk mask, tile mask, brick index - was re-deriving what it was handed.
    // The masks are not needed here either: only a brick that currently holds a
    // light block is ever dispatched, and a recycled slot releases its block
    // before the next upload.
    let bi = i32(brick);
    var is_foliage = false;
    var occ = vl_group_occ(bi, r);
    if (occ != 0u) {
        if (vl_group_blocks(bi, r)) {
            vl_pool[word] = 0u;
            vl_pool[word + 1u] = 0u;
            return;
        }
        // Occupied but nothing opaque: the cell is all foliage.
        is_foliage = true;
    }

    // The sample point is the LOW voxel's centre, not the cell's geometric
    // centre: a voxel centre is a safe shadow-ray origin and it is exactly what
    // the per-voxel field used, so the shadow term means the same thing.
    let p = vec3<f32>(wv) + vec3<f32>(0.5);
    // A foliage cell gathers inside its own tufts, so the WHOLE cell must be
    // excluded from occluding itself. See `shadow_skip_active` in raymarch.wgsl.
    if (is_foliage) {
        shadow_skip_active = true;
        shadow_skip_voxel = wv;
    }

    let prev = vl_pool[word];
    let prev_valid = ((prev >> 16u) & 0xFFu) != 0u;
    let fresh_sun = vl_sun_visibility(p);

    // Fold: a temporal smoother over a COMPLETE estimate, so it damps the sun's
    // motion and geometry edits without ever oscillating. The input is
    // stationary for a static scene, so this converges to a fixed point and then
    // stops moving - which the rotating-slice version it replaced could not do.
    // A freshly bound or invalidated record has no history and takes the fresh
    // estimate outright rather than blending into the previous tenant's light.
    var sun = fresh_sun;
    if (prev_valid) {
        sun = mix(f32(prev & 0xFFu) * (1.0 / 255.0), fresh_sun, vl_params.fold);
    }

    let ao = vl_cell_ao(wv);
    let pt = vl_point_light(p);

    // The third byte is the "this record holds a real estimate" marker and
    // nothing else now: 0 means no history, non-zero means history. It used to
    // carry the direction epoch as well, and there are no epochs any more.
    let stamp = 1u;
    vl_pool[word] = (u32(round(clamp(sun, 0.0, 1.0) * 255.0)))
        | (u32(round(clamp(ao, 0.0, 1.0) * 255.0)) << 8u)
        | (stamp << 16u);
    vl_pool[word + 1u] = vl_pack_rgb9e5(pt);
}
