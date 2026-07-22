// Hardware ray-tracing occlusion path for the render shader (RT variant only).
// `shadow_occluded` is the single dispatcher every occlusion ray in raymarch
// goes through (sun shadows, god rays, sky-access AO); the software variant
// injects a version that forwards to the DDA `trace_any`, this one forwards to
// the RT core. Assembled AFTER world consts / common (for `camera`) and the
// shared resolve_brick, alongside raymarch.wgsl which provides `bricks`,
// brick_voxel_solid and brick_voxel_idx.

// Packed AABB primitive, scalar fields (a WGSL vec3 would 16-byte align and
// break the tight 24-byte min/max packing the BLAS build uploads).
struct RtAabb {
    min_x: f32, min_y: f32, min_z: f32,
    max_x: f32, max_y: f32, max_z: f32,
    _p0: f32, _p1: f32,
};

@group(0) @binding(22) var world_tlas: acceleration_structure;
// primitive_index -> STORAGE brick index (for occupancy); the AABB min gives the
// primitive's WINDOW-LOCAL brick min for the in-brick DDA (see src/accel.rs).
@group(0) @binding(23) var<storage, read> rt_brick_map: array<u32>;
@group(0) @binding(24) var<storage, read> rt_aabbs: array<RtAabb>;

// Any-hit occlusion: true iff the ray from `origin` (WORLD space) toward `dir`
// meets any solid voxel within `max_dist`. The BLAS is in window-local space, so
// the origin is rebased by world_origin exactly as the software DDA folds world
// coords into the window.
fn rt_occluded(origin: vec3<f32>, dir: vec3<f32>, max_dist: f32) -> bool {
    let o = origin - vec3<f32>(camera.world_origin);
    var rq: ray_query;
    rayQueryInitialize(&rq, world_tlas, RayDesc(0u, 0xFFu, 0.0, max_dist, o, dir));
    while (rayQueryProceed(&rq)) {
        let c = rayQueryGetCandidateIntersection(&rq);
        if (c.kind == RAY_QUERY_INTERSECTION_AABB) {
            let bi = i32(rt_brick_map[c.primitive_index]);
            let a = rt_aabbs[c.primitive_index];
            let bmin = vec3<f32>(a.min_x, a.min_y, a.min_z);
            var fv = vec3<i32>(0);
            let t = resolve_brick(bi, bmin, o, dir, 0.0, max_dist, &fv);
            if (t >= 0.0) {
                // Any occluder within range is enough; stop the traversal.
                return true;
            }
        }
    }
    return false;
}

fn shadow_occluded(origin: vec3<f32>, dir: vec3<f32>, max_dist: f32) -> bool {
    return rt_occluded(origin, dir, max_dist);
}
