// Shared hardware-RT voxel primitive: the in-brick DDA that resolves the first
// solid voxel a ray crosses inside a candidate brick's 4^3 cells. This is the
// ONE `voxel_ray_query` mechanism, used by BOTH the accel_probe validation
// shader and the render shader's RT occlusion path, so there is a single place
// the coarse-to-fine resolve is defined and verified.
//
// Depends on the includer for `brick_voxel_solid(bi, vi)` and
// `brick_voxel_idx(lx, ly, lz)` (accel_probe.wgsl and raymarch.wgsl define them
// identically) and reads occupancy from whatever `bricks` binding is in scope.

fn rt_safe_inv(x: f32) -> f32 {
    // Signed large slope for near-axis-aligned rays: the slab min/max below
    // stays correct, and a near-zero axis simply never becomes the DDA's
    // stepping axis (its t_max stays huge).
    if (abs(x) < 1e-8) { return select(-1e20, 1e20, x >= 0.0); }
    return 1.0 / x;
}

// Face normal (pointing back toward the ray) for a hit whose entry face lies on
// `axis` (0=x,1=y,2=z), given the ray direction `d`.
fn rt_face_normal(axis: i32, d: vec3<f32>) -> vec3<i32> {
    if (axis == 0) { return vec3<i32>(select(-1, 1, d.x < 0.0), 0, 0); }
    if (axis == 1) { return vec3<i32>(0, select(-1, 1, d.y < 0.0), 0); }
    return vec3<i32>(0, 0, select(-1, 1, d.z < 0.0));
}

// Amanatides-Woo DDA within one brick's 4x4x4 voxels (window-local coords,
// brick min = bmin). Returns t of the entry face of the first solid voxel found
// in [t_lo, t_hi], or -1 if the ray crosses the brick with no solid voxel.
// `first_vox` receives the hit voxel's brick-local coords and `first_normal` the
// entry-face normal (for primary-ray shading; occlusion callers ignore it).
fn resolve_brick(bi: i32, bmin: vec3<f32>, o: vec3<f32>, d: vec3<f32>,
                 t_lo: f32, t_hi: f32, first_vox: ptr<function, vec3<i32>>,
                 first_normal: ptr<function, vec3<i32>>) -> f32 {
    let inv = vec3<f32>(rt_safe_inv(d.x), rt_safe_inv(d.y), rt_safe_inv(d.z));
    let tb0 = (bmin - o) * inv;
    let tb1 = (bmin + vec3<f32>(4.0) - o) * inv;
    let te = min(tb0, tb1); // per-axis entry t
    let tnear = max(max(te.x, te.y), te.z);
    let tfar = min(min(max(tb0.x, tb1.x), max(tb0.y, tb1.y)), max(tb0.z, tb1.z));
    var t = max(tnear, t_lo);
    let t_end = min(tfar, t_hi);
    if (t > t_end) { return -1.0; }

    // Axis the ray entered the brick through (the one that achieved tnear); the
    // hit face is this axis until the DDA steps to another.
    var last_axis: i32 = 0;
    if (te.y > te.x) { last_axis = 1; }
    if (te.z > max(te.x, te.y)) { last_axis = 2; }

    // Entry voxel (nudge inside so a face-grazing entry lands in the brick).
    let p = o + d * (t + 1e-4) - bmin;
    var v = clamp(vec3<i32>(floor(p)), vec3<i32>(0), vec3<i32>(3));
    let step = vec3<i32>(select(-1, 1, d.x >= 0.0), select(-1, 1, d.y >= 0.0), select(-1, 1, d.z >= 0.0));
    // t of the next voxel boundary on each axis.
    let next = vec3<f32>(
        bmin.x + f32(v.x + select(0, 1, d.x >= 0.0)),
        bmin.y + f32(v.y + select(0, 1, d.y >= 0.0)),
        bmin.z + f32(v.z + select(0, 1, d.z >= 0.0)),
    );
    var t_max = (next - o) * inv;
    let t_delta = abs(inv);

    var t_cur = t;
    for (var i: i32 = 0; i < 16; i = i + 1) {
        if (v.x < 0 || v.x > 3 || v.y < 0 || v.y > 3 || v.z < 0 || v.z > 3) { return -1.0; }
        if (t_cur > t_end) { return -1.0; }
        if (brick_voxel_solid(bi, brick_voxel_idx(v.x, v.y, v.z))) {
            *first_vox = v;
            *first_normal = rt_face_normal(last_axis, d);
            return t_cur;
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
