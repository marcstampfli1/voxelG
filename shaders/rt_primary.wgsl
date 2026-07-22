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

fn trace_rt(origin: vec3<f32>, dir: vec3<f32>) -> Hit {
    var out = rt_hit_none();
    let o = origin - vec3<f32>(camera.world_origin);
    var rq: ray_query;
    rayQueryInitialize(&rq, world_tlas, RayDesc(0u, 0xFFu, 0.0, MAX_RAY_DIST, o, dir));
    var best_t = 1.0e30;
    var best_v = vec3<i32>(0);
    var best_n = vec3<i32>(0);
    var best_bi = 0;
    var best_fv = vec3<i32>(0);
    var found = false;
    while (rayQueryProceed(&rq)) {
        let c = rayQueryGetCandidateIntersection(&rq);
        if (c.kind == RAY_QUERY_INTERSECTION_AABB) {
            let bi = i32(rt_brick_map[c.primitive_index]);
            let a = rt_aabbs[c.primitive_index];
            let bmin = vec3<f32>(a.min_x, a.min_y, a.min_z);
            var fv = vec3<i32>(0);
            var fnrm = vec3<i32>(0);
            let t = resolve_brick(bi, bmin, o, dir, 0.0, best_t, &fv, &fnrm);
            if (t >= 0.0) {
                rayQueryGenerateIntersection(&rq, t);
                if (t < best_t) {
                    best_t = t;
                    best_v = vec3<i32>(bmin) + fv;
                    best_n = fnrm;
                    best_bi = bi;
                    best_fv = fv;
                    found = true;
                }
            }
        }
    }
    if (found) {
        out.hit = true;
        out.t_hit = best_t;
        out.voxel = best_v + camera.world_origin;
        out.normal = vec3<f32>(best_n);
        out.mat = brick_voxel_material(best_bi, brick_voxel_idx(best_fv.x, best_fv.y, best_fv.z));
        out.last_axis = select(select(2, 1, best_n.y != 0), 0, best_n.x != 0);
        out.tint = vec3<f32>(1.0);
    }
    return out;
}

// cs_main calls this first; if RT_PRIMARY is on it returns the RT hit and sets
// done, otherwise done stays false and cs_main runs the software primary.
fn rt_primary_or_none(origin: vec3<f32>, dir: vec3<f32>, done: ptr<function, bool>) -> Hit {
    if (RT_PRIMARY > 0.5) {
        *done = true;
        return trace_rt(origin, dir);
    }
    *done = false;
    return rt_hit_none();
}
