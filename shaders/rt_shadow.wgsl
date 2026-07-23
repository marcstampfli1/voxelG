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

// RT resources live in their OWN bind group (group 1) so the software pipeline's
// group-0 layout and bind group stay byte-identical - RT is strictly additive.
@group(1) @binding(0) var world_tlas: acceleration_structure;
// primitive_index -> STORAGE brick index (for occupancy); the AABB min gives the
// primitive's WINDOW-LOCAL brick min for the in-brick DDA (see src/accel.rs).
@group(1) @binding(1) var<storage, read> rt_brick_map: array<u32>;
@group(1) @binding(2) var<storage, read> rt_aabbs: array<RtAabb>;

// Live-brick gate for RT candidates: the BVH may briefly be STALE (rebuilds
// land asynchronously off the frame thread), but the tile occupancy mask is
// updated the same frame a slot is recycled or installed. One bit test makes a
// stale primitive render as sky exactly like the software DDA - correctness
// never depends on BVH freshness, only coverage of NEW bricks does (they
// appear when the async rebuild lands).
fn rt_brick_active(bi: i32) -> bool {
    let bx = bi % WORLD_BRICKS_X;
    let by = (bi / WORLD_BRICKS_X) % WORLD_BRICKS_Y;
    let bz = bi / (WORLD_BRICKS_X * WORLD_BRICKS_Y);
    let ti = world_tile_idx(bx >> 2, by >> 2, bz >> 2);
    let lin = (bx & 3) + (bz & 3) * 4 + (by & 3) * 16;
    return tile_has_child(ti, lin);
}

// March a candidate brick's 4^3 voxels (window-local o/dir) and return true if
// any solid voxel actually occludes per `shadow_voxel_occludes`. Unlike
// resolve_brick (first solid wins, for primary hits) this visits EVERY solid
// voxel so the ray passes THROUGH non-occluders - invisible fringe, far
// decorations, leaf-cutout misses - exactly as the software DDA does.
// `origin_world`/`dir` are the WORLD-space shadow ray (for the foliage cutout);
// world voxel = window-local brick min + brick-local coord + world_origin.
fn rt_brick_occludes(bi: i32, bmin: vec3<f32>, o: vec3<f32>, origin_world: vec3<f32>,
                     dir: vec3<f32>, t_hi: f32) -> bool {
    let inv = vec3<f32>(rt_safe_inv(dir.x), rt_safe_inv(dir.y), rt_safe_inv(dir.z));
    let tb0 = (bmin - o) * inv;
    let tb1 = (bmin + vec3<f32>(4.0) - o) * inv;
    let tnear = max(max(min(tb0.x, tb1.x), min(tb0.y, tb1.y)), min(tb0.z, tb1.z));
    let tfar = min(min(max(tb0.x, tb1.x), max(tb0.y, tb1.y)), max(tb0.z, tb1.z));
    var t = max(tnear, 0.0);
    let t_end = min(tfar, t_hi);
    if (t > t_end) { return false; }

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
        if (v.x < 0 || v.x > 3 || v.y < 0 || v.y > 3 || v.z < 0 || v.z > 3) { return false; }
        if (t_cur > t_end) { return false; }
        let vi = brick_voxel_idx(v.x, v.y, v.z);
        if (brick_voxel_solid(bi, vi)) {
            let m = brick_voxel_material(bi, vi);
            if (shadow_voxel_occludes(world_base + v, m, t_cur, origin_world, dir)) {
                return true;
            }
        }
        if (t_max.x <= t_max.y && t_max.x <= t_max.z) {
            v.x = v.x + step.x; t_cur = t_max.x; t_max.x = t_max.x + t_delta.x;
        } else if (t_max.y <= t_max.z) {
            v.y = v.y + step.y; t_cur = t_max.y; t_max.y = t_max.y + t_delta.y;
        } else {
            v.z = v.z + step.z; t_cur = t_max.z; t_max.z = t_max.z + t_delta.z;
        }
    }
    return false;
}

// Any-hit occlusion: true iff the ray from `origin` (WORLD space) toward `dir`
// meets any OCCLUDING voxel within `max_dist`. The BLAS is in window-local
// space, so the origin is rebased by world_origin exactly as the software DDA
// folds world coords into the window.
fn rt_occluded(origin: vec3<f32>, dir: vec3<f32>, max_dist: f32) -> bool {
    let o = origin - vec3<f32>(camera.world_origin);
    var rq: ray_query;
    rayQueryInitialize(&rq, world_tlas, RayDesc(0u, 0xFFu, 0.0, max_dist, o, dir));
    while (rayQueryProceed(&rq)) {
        let c = rayQueryGetCandidateIntersection(&rq);
        if (c.kind == RAY_QUERY_INTERSECTION_AABB) {
            let bi = i32(rt_brick_map[c.primitive_index]);
            if (rt_brick_active(bi)) {
                let a = rt_aabbs[c.primitive_index];
                let bmin = vec3<f32>(a.min_x, a.min_y, a.min_z);
                if (rt_brick_occludes(bi, bmin, o, origin, dir, max_dist)) {
                    return true;
                }
            }
        }
    }
    return false;
}

fn shadow_occluded(origin: vec3<f32>, dir: vec3<f32>, max_dist: f32) -> bool {
    return rt_occluded(origin, dir, max_dist);
}
