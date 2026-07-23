// Isolated validation of the hardware-RT voxel path: for each input ray, the RT
// core traverses the world BLAS to candidate brick AABBs and the shared
// per-brick DDA (resolve_brick, from rt_voxel_query.wgsl, prepended at assembly)
// resolves the exact first solid voxel. Here it just writes the nearest hit for
// a CPU oracle to check; the render shader reuses the same resolve_brick.

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
    nx: i32,
    ny: i32,
    nz: i32,
    _p0: u32,
};

@group(0) @binding(0) var<storage, read> bricks: array<Brick>;
@group(0) @binding(1) var acc: acceleration_structure;
@group(0) @binding(2) var<storage, read> brick_map: array<u32>;
@group(0) @binding(3) var<storage, read> aabbs: array<GpuAabb>;
@group(0) @binding(4) var<storage, read> rays: array<Ray>;
@group(0) @binding(5) var<storage, read_write> hits: array<Hit>;

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
    var best_n = vec3<i32>(0);
    var best_mat = 0u;
    var found = false;
    while (rayQueryProceed(&rq)) {
        let c = rayQueryGetCandidateIntersection(&rq);
        if (c.kind == RAY_QUERY_INTERSECTION_AABB) {
            let bi = i32(brick_map[c.primitive_index]);
            let a = aabbs[c.primitive_index];
            let bmin = vec3<f32>(a.min_x, a.min_y, a.min_z);
            var fv = vec3<i32>(0);
            var fnrm = vec3<i32>(0);
            // No transparent skip: the CPU oracle checks the exact first
            // occupied voxel, water included.
            let t = resolve_brick(bi, bmin, o, d, 0.0, best_t, false, &fv, &fnrm);
            if (t >= 0.0) {
                rayQueryGenerateIntersection(&rq, t);
                if (t < best_t) {
                    best_t = t;
                    best_v = vec3<i32>(bmin) + fv;
                    best_n = fnrm;
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
        out.nx = best_n.x; out.ny = best_n.y; out.nz = best_n.z;
    } else {
        out.t = -1.0;
        out.hit = 0u;
        out.vx = 0; out.vy = 0; out.vz = 0; out.mat = 0u;
        out.nx = 0; out.ny = 0; out.nz = 0;
    }
    hits[i] = out;
}
