// Isolated validation of the hardware-RT voxel path: for each input ray, the RT
// core traverses the world BLAS to candidate brick AABBs and a per-brick DDA
// resolves the exact first solid voxel. This is the `voxel_ray_query` primitive
// that the render passes will reuse; here it just writes the hit for a CPU
// oracle to check.
enable wgpu_ray_query;

struct Brick {
    occ_lo: u32,
    occ_hi: u32,
    materials: array<u32, 16>,
};

// Packed AABB: min then max as SCALARS (a WGSL vec3 would 16-byte align and
// break the tight 24-byte min/max packing the BLAS build uses).
struct GpuAabb {
    min_x: f32, min_y: f32, min_z: f32,
    max_x: f32, max_y: f32, max_z: f32,
    _p0: f32, _p1: f32,
};

struct Ray {
    o: vec3<f32>,
    _p0: f32,
    d: vec3<f32>,
    _p1: f32,
};

struct Hit {
    t: f32,
    hit: u32,
    vx: i32,
    vy: i32,
    vz: i32,
    mat: u32,
    _p0: u32,
    _p1: u32,
};

@group(0) @binding(0) var<storage, read> bricks: array<Brick>;
@group(0) @binding(1) var acc: acceleration_structure;
@group(0) @binding(2) var<storage, read> brick_map: array<u32>;
@group(0) @binding(3) var<storage, read> aabbs: array<GpuAabb>;
@group(0) @binding(4) var<storage, read> rays: array<Ray>;
@group(0) @binding(5) var<storage, read_write> hits: array<Hit>;

fn safe_inv(x: f32) -> f32 {
    // Large finite slope for near-axis-aligned rays; avoids inf/NaN in the DDA.
    if (abs(x) < 1e-8) { return select(-1e20, 1e20, x >= 0.0); }
    return 1.0 / x;
}

fn brick_voxel_idx(lx: i32, ly: i32, lz: i32) -> i32 {
    return lx + lz * 4 + ly * 16;
}

fn brick_voxel_solid(bi: i32, vi: i32) -> bool {
    let b = bricks[bi];
    if (vi < 32) {
        return (b.occ_lo & (1u << u32(vi))) != 0u;
    }
    return (b.occ_hi & (1u << u32(vi - 32))) != 0u;
}

fn brick_voxel_material(bi: i32, vi: i32) -> u32 {
    let word = vi / 4;
    let byte = vi - word * 4;
    return (bricks[bi].materials[word] >> u32(byte * 8)) & 0xFFu;
}

// Amanatides-Woo DDA within one brick's 4x4x4 voxels (window-local coords,
// brick min = bmin). Returns t of the entry face of the first solid voxel, or
// -1 if the ray crosses the brick with no solid voxel. `first_vox` receives the
// hit voxel's brick-local coords.
fn resolve_brick(bi: i32, bmin: vec3<f32>, o: vec3<f32>, d: vec3<f32>,
                 t_lo: f32, t_hi: f32, first_vox: ptr<function, vec3<i32>>) -> f32 {
    let inv = vec3<f32>(safe_inv(d.x), safe_inv(d.y), safe_inv(d.z));
    let tb0 = (bmin - o) * inv;
    let tb1 = (bmin + vec3<f32>(4.0) - o) * inv;
    let tnear = max(max(min(tb0.x, tb1.x), min(tb0.y, tb1.y)), min(tb0.z, tb1.z));
    let tfar = min(min(max(tb0.x, tb1.x), max(tb0.y, tb1.y)), max(tb0.z, tb1.z));
    var t = max(tnear, t_lo);
    let t_end = min(tfar, t_hi);
    if (t > t_end) { return -1.0; }

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
            return t_cur;
        }
        if (t_max.x <= t_max.y && t_max.x <= t_max.z) {
            v.x = v.x + step.x; t_cur = t_max.x; t_max.x = t_max.x + t_delta.x;
        } else if (t_max.y <= t_max.z) {
            v.y = v.y + step.y; t_cur = t_max.y; t_max.y = t_max.y + t_delta.y;
        } else {
            v.z = v.z + step.z; t_cur = t_max.z; t_max.z = t_max.z + t_delta.z;
        }
    }
    return -1.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= arrayLength(&rays)) { return; }
    let ray = rays[i];
    let o = ray.o;
    let d = ray.d;

    var rq: ray_query;
    rayQueryInitialize(&rq, acc, RayDesc(0u, 0xFFu, 0.0, 1.0e9, o, d));
    // Track the nearest resolved voxel in registers: exact (no re-derivation
    // from a committed t) and independent of the order the RT core visits
    // candidates. generateIntersection still fires so the hardware culls bricks
    // beyond the current nearest hit - the coarse-to-fine speedup - but the
    // committed intersection is only used as a cross-check, not the source of
    // truth for the voxel.
    var best_t = 1.0e30;
    var best_v = vec3<i32>(0);
    var best_mat = 0u;
    var found = false;
    var n_cand = 0u;
    while (rayQueryProceed(&rq)) {
        let c = rayQueryGetCandidateIntersection(&rq);
        if (c.kind == RAY_QUERY_INTERSECTION_AABB) {
            n_cand = n_cand + 1u;
            let bi = i32(brick_map[c.primitive_index]);
            let a = aabbs[c.primitive_index];
            let bmin = vec3<f32>(a.min_x, a.min_y, a.min_z);
            var fv = vec3<i32>(0);
            let t = resolve_brick(bi, bmin, o, d, 0.0, best_t, &fv);
            if (t >= 0.0) {
                rayQueryGenerateIntersection(&rq, t);
                if (t < best_t) {
                    best_t = t;
                    best_v = vec3<i32>(bmin) + fv;
                    let vi = brick_voxel_idx(fv.x, fv.y, fv.z);
                    best_mat = brick_voxel_material(bi, vi);
                    found = true;
                }
            }
        }
    }

    var out: Hit;
    if (found) {
        out.t = best_t;
        out.hit = 1u;
        out.vx = best_v.x; out.vy = best_v.y; out.vz = best_v.z;
        out.mat = best_mat;
    } else {
        out.t = -1.0;
        out.hit = 0u;
        out.vx = 0; out.vy = 0; out.vz = 0; out.mat = 0u;
    }
    out._p0 = n_cand;
    hits[i] = out;
}
