// Hardware-RT primary trace (RT variant). The RT core traverses the world BVH
// to the nearest candidate brick and the shared in-brick DDA resolves the exact
// first solid voxel, producing a Hit directly - replacing the software beam
// pre-pass + hierarchical-DDA empty-space skip with hardware traversal.
//
// This first cut resolves OPAQUE cube voxels only (no water sub-voxel surface,
// foliage cutout, or distance LOD yet), so it is gated behind the RT_PRIMARY
// override: off by default, flipped on to benchmark the traversal speedup and
// (once the sub-voxel resolve is ported) to run for real.
override RT_PRIMARY: f32 = 0.0;

fn rt_hit_none() -> Hit {
    var out: Hit;
    out.hit = false;
    out.mat = 0u;
    out.normal = vec3<f32>(0.0);
    out.voxel = vec3<i32>(0);
    out.last_axis = -1;
    out.t_hit = 0.0;
    out.tint = vec3<f32>(1.0);
    return out;
}

// March one candidate brick's 4^3 voxels (Amanatides-Woo, window-local o/dir)
// and run the SHARED resolve_solid_voxel at each solid voxel - so leaf/decoration
// cutouts, water sub-voxel surfaces and opaque cubes resolve exactly as the
// software DDA does. Returns the hit t (or -1) and fills `out`; keeps stepping
// past cutout/above-water misses. `origin_world` is the ray origin in world
// space (for the sub-voxel functions).
fn resolve_brick_full(bi: i32, bmin: vec3<f32>, o: vec3<f32>, dir: vec3<f32>,
                      origin_world: vec3<f32>, t_hi: f32, skip_transparent: bool,
                      out: ptr<function, Hit>) -> f32 {
    let inv = vec3<f32>(rt_safe_inv(dir.x), rt_safe_inv(dir.y), rt_safe_inv(dir.z));
    let tb0 = (bmin - o) * inv;
    let tb1 = (bmin + vec3<f32>(4.0) - o) * inv;
    let te = min(tb0, tb1);
    let te_xy = max(te.x, te.y);
    let tnear = max(te_xy, te.z);
    let tfar = min(min(max(tb0.x, tb1.x), max(tb0.y, tb1.y)), max(tb0.z, tb1.z));
    var t = max(tnear, 0.0);
    let t_end = min(tfar, t_hi);
    if (t > t_end) { return -1.0; }

    var last_axis: i32 = 0;
    if (te.y > te.x) { last_axis = 1; }
    if (te.z > te_xy) { last_axis = 2; }

    let p = o + dir * (t + 1e-4) - bmin;
    var v = clamp(vec3<i32>(floor(p)), vec3<i32>(0), vec3<i32>(3));
    let step = vec3<i32>(select(-1, 1, dir.x >= 0.0), select(-1, 1, dir.y >= 0.0), select(-1, 1, dir.z >= 0.0));
    let next = vec3<f32>(
        bmin.x + f32(v.x + select(0, 1, dir.x >= 0.0)),
        bmin.y + f32(v.y + select(0, 1, dir.y >= 0.0)),
        bmin.z + f32(v.z + select(0, 1, dir.z >= 0.0)),
    );
    var t_max = (next - o) * inv;
    let t_delta = abs(inv);
    let world_base = vec3<i32>(bmin) + camera.world_origin;

    var t_cur = t;
    for (var i: i32 = 0; i < 16; i = i + 1) {
        if (v.x < 0 || v.x > 3 || v.y < 0 || v.y > 3 || v.z < 0 || v.z > 3) { return -1.0; }
        if (t_cur > t_end) { return -1.0; }
        let vi = brick_voxel_idx(v.x, v.y, v.z);
        if (brick_voxel_solid(bi, vi)) {
            let m = brick_voxel_material(bi, vi);
            // Camera-underwater / secondary-ray mode: transparent voxels (water,
            // glass) are see-through, exactly like the software trace_no_water; the
            // shared step at the loop bottom keeps the march going past them.
            if (!(skip_transparent && is_transparent_mat(m))) {
                let en_n = vec3<f32>(rt_face_normal(last_axis, dir));
                if (resolves_as_opaque_cube(m, t_cur)) {
                    // Fast path (the ~90%): the cube hit IS the answer. Skip the
                    // slot/brick-pos setup + resolve_solid_voxel entirely.
                    (*out).hit = true;
                    (*out).mat = m;
                    (*out).normal = en_n;
                    (*out).voxel = world_base + v;
                    (*out).last_axis = last_axis;
                    (*out).t_hit = t_cur;
                    (*out).tint = vec3<f32>(1.0);
                    return t_cur;
                }
                // Sub-voxel material (water/foliage cutout): compute the slot/brick
                // mapping lazily and resolve; may miss and continue the march.
                let world_voxel = world_base + v;
                let slot_v = world_to_slot_voxel(world_voxel);
                let bp = slot_v >> vec3<u32>(2u);
                let t_exit_cell = min(t_max.x, min(t_max.y, t_max.z));
                if (resolve_solid_voxel(world_voxel, m, slot_v, bp, bi, t_cur, origin_world, dir,
                                        en_n, t_cur, last_axis, t_exit_cell, out)) {
                    return (*out).t_hit;
                }
            }
        }
        if (t_max.x <= t_max.y && t_max.x <= t_max.z) {
            v.x = v.x + step.x; t_cur = t_max.x; t_max.x = t_max.x + t_delta.x; last_axis = 0;
        } else if (t_max.y <= t_max.z) {
            v.y = v.y + step.y; t_cur = t_max.y; t_max.y = t_max.y + t_delta.y; last_axis = 1;
        } else {
            v.z = v.z + step.z; t_cur = t_max.z; t_max.z = t_max.z + t_delta.z; last_axis = 2;
        }
    }
    return -1.0;
}

fn trace_rt(origin: vec3<f32>, dir: vec3<f32>, skip_transparent: bool, t_cap: f32) -> Hit {
    var out = rt_hit_none();
    water_grad_rest = vec2<f32>(0.0);
    let o = origin - vec3<f32>(camera.world_origin);
    var rq: ray_query;
    rayQueryInitialize(&rq, world_tlas, RayDesc(0u, 0xFFu, 0.0, t_cap, o, dir));
    var best_t = 1.0e30;
    var best_grad = vec2<f32>(0.0);
    while (rayQueryProceed(&rq)) {
        let c = rayQueryGetCandidateIntersection(&rq);
        if (c.kind == RAY_QUERY_INTERSECTION_AABB) {
            let bi = i32(rt_brick_map[c.primitive_index]);
            if (!rt_brick_active(bi)) { continue; }
            let a = rt_aabbs[c.primitive_index];
            let bmin = vec3<f32>(a.min_x, a.min_y, a.min_z);
            var bh = rt_hit_none();
            water_grad_rest = vec2<f32>(0.0);
            let t = resolve_brick_full(bi, bmin, o, dir, origin, best_t, skip_transparent, &bh);
            let ht = bh.t_hit;
            if (t >= 0.0 && bh.hit && ht < best_t) {
                best_t = ht;
                out = bh;
                best_grad = water_grad_rest;
                rayQueryGenerateIntersection(&rq, ht);
            }
        }
    }
    water_grad_rest = best_grad;
    return out;
}

// Secondary rays (water reflection/refraction, glass) ride the RT cores in
// this variant: trace_rt with skip_transparent matches trace_no_water's
// contract (transparent voxels are see-through) at hardware traversal speed.
fn trace_secondary(o: vec3<f32>, d: vec3<f32>, cap: f32) -> Hit {
    return trace_rt(o, d, true, cap);
}

// cs_main calls this first; if RT_PRIMARY is on it returns the RT hit and sets
// done, otherwise done stays false and cs_main runs the software primary.
fn rt_primary_or_none(origin: vec3<f32>, dir: vec3<f32>, skip_transparent: bool, done: ptr<function, bool>) -> Hit {
    if (RT_PRIMARY > 0.5) {
        *done = true;
        return trace_rt(origin, dir, skip_transparent, MAX_RAY_DIST);
    }
    *done = false;
    return rt_hit_none();
}
