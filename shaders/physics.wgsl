// GPU-compute physics — stage 1 (sand), per docs/gpu-physics-design.md.
//
// Pull-only, double-buffered cellular automaton: ONE invocation per brick reads
// the brick plus its vertical neighbours from the read-only IN buffer and writes
// the brick's next state to the OUT buffer. Because every output brick is owned
// by exactly one invocation there are no write conflicts and no atomics — the
// pull model means each cell computes its own next value from its neighbours.
//
// Only SAND moves (falls straight down, conserving mass); every other material
// passes through unchanged, so this is safe to run alongside the existing world
// (stone/water/glass are untouched). Water + smoke are the next migration stages.

struct Brick {
    occ_lo: u32,
    occ_hi: u32,
    materials: array<u32, 16>,
};

@group(0) @binding(0) var<storage, read> bricks_in: array<Brick>;
@group(0) @binding(1) var<storage, read_write> bricks_out: array<Brick>;

const MAT_AIR: u32 = 0u;
const MAT_SAND: u32 = 1u;

fn vidx(lx: u32, ly: u32, lz: u32) -> u32 { return lx + lz * 4u + ly * 16u; }

fn vmat(b: Brick, vi: u32) -> u32 {
    let word = vi >> 2u;
    let byte = vi & 3u;
    return (b.materials[word] >> (byte * 8u)) & 0xFFu;
}

@compute @workgroup_size(64, 1, 1)
fn cs_physics(@builtin(global_invocation_id) gid: vec3<u32>) {
    // The dispatch is TILED over x and y: one workgroup per 64 bricks is 400,000
    // workgroups at 10 cm and every dispatch dimension caps at 65,535. Workgroup
    // size y is 1, so `gid.y` IS the workgroup row and the linear brick index is
    // that row times a full row of INVOCATIONS. `renderer::linear_dispatch` is
    // the one place that decides the tiling. The tail row over-runs `total`,
    // which the bounds check below already handled and now depends on.
    let bi = gid.x + gid.y * u32(DISPATCH_ROW_WGS) * 64u;
    let wbx = u32(WORLD_BRICKS_X);
    let wby = u32(WORLD_BRICKS_Y);
    let wbz = u32(WORLD_BRICKS_Z);
    let total = wbx * wby * wbz;
    if (bi >= total) { return; }

    let by = (bi / wbx) % wby;
    let has_above = by + 1u < wby;
    let has_below = by > 0u;

    let self_b = bricks_in[bi];
    let above_b = bricks_in[select(bi, bi + wbx, has_above)];
    let below_b = bricks_in[select(bi, bi - wbx, has_below)];

    var out_mats = array<u32, 16>();
    var occ_lo: u32 = 0u;
    var occ_hi: u32 = 0u;

    for (var ly: u32 = 0u; ly < 4u; ly = ly + 1u) {
        for (var lz: u32 = 0u; lz < 4u; lz = lz + 1u) {
            for (var lx: u32 = 0u; lx < 4u; lx = lx + 1u) {
                let vi = vidx(lx, ly, lz);
                let m = vmat(self_b, vi);

                // Neighbour materials (cross brick boundary at ly 0 / 3).
                var above_is_sand = false;
                if (ly < 3u) {
                    above_is_sand = vmat(self_b, vidx(lx, ly + 1u, lz)) == MAT_SAND;
                } else if (has_above) {
                    above_is_sand = vmat(above_b, vidx(lx, 0u, lz)) == MAT_SAND;
                }
                // below_is_air is false at the world floor (sand can't fall out).
                var below_is_air = false;
                if (ly > 0u) {
                    below_is_air = vmat(self_b, vidx(lx, ly - 1u, lz)) == MAT_AIR;
                } else if (has_below) {
                    below_is_air = vmat(below_b, vidx(lx, 3u, lz)) == MAT_AIR;
                }

                var next = m;
                if (m == MAT_SAND) {
                    if (below_is_air) { next = MAT_AIR; }
                } else if (m == MAT_AIR) {
                    if (above_is_sand) { next = MAT_SAND; }
                }

                if (next != MAT_AIR) {
                    let word = vi >> 2u;
                    let byte = vi & 3u;
                    out_mats[word] = out_mats[word] | (next << (byte * 8u));
                    if (vi < 32u) { occ_lo = occ_lo | (1u << vi); }
                    else { occ_hi = occ_hi | (1u << (vi - 32u)); }
                }
            }
        }
    }

    bricks_out[bi].occ_lo = occ_lo;
    bricks_out[bi].occ_hi = occ_hi;
    for (var w: u32 = 0u; w < 16u; w = w + 1u) {
        bricks_out[bi].materials[w] = out_mats[w];
    }
}
