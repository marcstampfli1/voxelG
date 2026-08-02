// CPU voxel-world queries, over the SAME brick/tile/chunk/L4 pyramid the shaders
// traverse. One primitive, shared by everything that needs to ask the world a
// spatial question from the CPU: character collision today, digging volumes and
// entities later (checklist: safe-primitives - the hand-rolled per-voxel loop at
// a call site is the forbidden pattern this replaces).
//
// WHY IT IS NOT A VOXEL LOOP
// A standing player is a 2.4 x 7.2 x 2.4-voxel box. Testing it a voxel at a time
// is ~42 divisions + array indexes per substep, per axis. Here the same test is
// a handful of masked u64s: a brick's `occupancy` is one bit per voxel over 4^3,
// so "does this box overlap anything in this brick" is `occupancy & range_mask`,
// and an empty tile / chunk / L4 cell skips 16^3 / 64^3 / 256^3 voxels with a
// single bit test. The sweep exploits the same thing: because a query over a
// large empty region costs about the same as a small one, the swept region is
// tested WHOLE first and only bisected when something is actually in the way.
//
// THREE THINGS THAT ARE EASY TO GET WRONG, AND ARE GOT RIGHT HERE
//
// 1. Storage is TOROIDAL. A world voxel lives at slot `pos_mod(w, WORLD_VOXELS)`,
//    not at `w - origin` (see `World::apply_edit`, `raycast.rs`, and the shaders'
//    `world_to_slot_voxel`). Anything that assumes otherwise is correct only
//    while the player stands at spawn, and silently reads the wrong bricks after
//    that. A contiguous world range therefore wraps in x and z, so it is split
//    into at most 2x2 non-wrapping slot boxes before the descent.
// 2. The MASKS are the authority, not `Brick::occupancy`. Recycling a streaming
//    slot clears its tile/chunk/L4 bits and leaves the stale brick bytes in place
//    (`World::clear_slot_masks`) - the GPU renders that region as sky, so the CPU
//    must treat it as empty too, or the player collides with terrain that is not
//    there. Every level's bit is checked on the way down.
// 3. Outside the loaded window there is NO DATA, and it reads as EMPTY. The
//    window is centred on the player, so a body can only reach the edge if
//    streaming has fallen behind; pretending an invisible wall is there would be
//    a worse lie than letting it through.

use glam::{IVec3, UVec3, Vec3};

use crate::voxel::*;

// ---------------------------------------------------------------- material set

// A u64 bitset over material ids. Materials are a dense u8 space (`MAT_COUNT`),
// so "is this material solid?" is one shift and mask instead of a chain of
// comparisons, and a caller can ask about a different class of matter (water for
// buoyancy, later) through the SAME traversal rather than a second copy of it.
const _: () = assert!(
    MAT_COUNT <= 64,
    "MatSet is a u64 bitset over material ids; material ids must stay under 64",
);

/// Fold a `const fn(u8) -> bool` material predicate into a [`MatSet`] at compile
/// time. Every id in `0..64` is offered to the predicate, so the set and the
/// predicate can never disagree.
macro_rules! mat_set {
    ($pred:path) => {{
        let mut bits = 0u64;
        let mut m = 0u8;
        while m < 64 {
            if $pred(m) {
                bits |= 1u64 << m;
            }
            m += 1;
        }
        MatSet(bits)
    }};
}

/// A set of material ids. Build one from a `const fn` predicate with
/// `mat_set!`; the folding happens at compile time.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MatSet(pub u64);

impl MatSet {
    #[inline(always)]
    pub const fn contains(self, mat: u8) -> bool {
        // mat >= 64 cannot happen (the const assert above), but shifting by >= 64
        // is UB-adjacent in release, so fold it to "not in the set" rather than
        // relying on a caller invariant (checklist: safe by construction).
        mat < 64 && (self.0 >> mat) & 1 == 1
    }

    /// Blocks a body: everything occupied EXCEPT the things you walk through.
    ///
    /// Water is out because you wade/swim through it (buoyancy will read it via
    /// [`MatSet::WATER`], through this same traversal). Foliage is out because at
    /// this voxel size a leaf tuft, a flower and ground turf are things you push
    /// through, and because MAT_LEAF_FRINGE is deliberately INVISIBLE - colliding
    /// with it would be colliding with nothing you can see. Smoke and fire are
    /// out for the obvious reason.
    ///
    /// Anything added to the material table later is solid by default, which is
    /// the safe direction: a new material behaves like stone until classified,
    /// rather than becoming a hole in the world.
    pub const SOLID: MatSet = mat_set!(mat_is_solid);
    /// Water at any of its 8 fill levels. Not solid; here for the buoyancy /
    /// submersion queries that will need the same traversal.
    pub const WATER: MatSet = mat_set!(is_water_mat);
    /// Every occupied voxel, whatever it is made of.
    pub const ANY: MatSet = mat_set!(mat_is_not_air);
}

/// Predicate behind [`MatSet::SOLID`]; see its documentation for the rationale.
#[inline(always)]
pub const fn mat_is_solid(mat: u8) -> bool {
    mat != MAT_AIR
        && !is_water_mat(mat)
        && !is_foliage_mat(mat)
        && mat != MAT_SMOKE
        && mat != MAT_FIRE
}

#[inline(always)]
const fn mat_is_not_air(mat: u8) -> bool {
    mat != MAT_AIR
}

// ----------------------------------------------------------------------- aabb

/// An axis-aligned box in WORLD VOXEL units (not metres, not slot coords).
///
/// HALF-OPEN: the box covers voxels `floor(min) ..= ceil(max) - 1` on each axis,
/// so a box whose faces sit exactly on voxel boundaries touches exactly the
/// voxels between them and none beyond. A box with zero extent on an axis covers
/// NO voxels and therefore hits nothing - if you want a probe, give it thickness.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    #[inline]
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// Box standing on `feet` (its centre in x/z, its base in y).
    #[inline]
    pub fn from_feet(feet: Vec3, half_width: f32, height: f32) -> Self {
        Self {
            min: Vec3::new(feet.x - half_width, feet.y, feet.z - half_width),
            max: Vec3::new(feet.x + half_width, feet.y + height, feet.z + half_width),
        }
    }

    #[inline]
    pub fn translated(self, by: Vec3) -> Self {
        Self { min: self.min + by, max: self.max + by }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Axis {
    X = 0,
    Y = 1,
    Z = 2,
}

impl Axis {
    pub const ALL: [Axis; 3] = [Axis::X, Axis::Y, Axis::Z];
}

// --------------------------------------------------------------- public query

/// Some material from `set` overlapping `box_`, or `None` if the box is clear.
///
/// WHICH overlapping material is returned is unspecified when several match -
/// the traversal returns the first it proves, and the order follows the storage
/// hierarchy, not the box. Callers that need a specific one (the surface under
/// the feet, say) should ask with a box that can only contain that one.
///
/// A non-finite box is a caller bug; float-to-int saturation clamps it to the
/// loaded window rather than reading out of bounds, and the debug build asserts.
#[inline]
pub fn overlap_mat(world: &World, box_: Aabb, set: MatSet) -> Option<u8> {
    debug_assert!(
        box_.min.is_finite() && box_.max.is_finite(),
        "non-finite AABB {box_:?} in a world query",
    );
    any_in_voxel_range(world, voxel_lo(box_.min), voxel_hi(box_.max), set)
}

/// `overlap_mat(..).is_some()`, for callers that only need the yes/no.
#[inline]
pub fn overlaps(world: &World, box_: Aabb, set: MatSet) -> bool {
    overlap_mat(world, box_, set).is_some()
}

/// Result of [`sweep`]: how far the box may travel before it would enter a voxel
/// in the set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sweep {
    /// Distance (voxels, always >= 0) the box may move along the requested
    /// direction. Includes a hair of contact skin so the stopped box does not
    /// re-test as overlapping next tick.
    pub free: f32,
    /// True iff the move was cut short - i.e. there is a contact.
    pub blocked: bool,
    /// Material hit, or `MAT_AIR` when nothing was. Lets a caller know what it is
    /// standing on / walked into without a second query.
    pub mat: u8,
}

/// Distance in voxels; skin so a stopped face rests just clear of the plane it
/// hit rather than exactly on it. 1e-3 voxel is 0.25 mm at the current scale:
/// far below anything visible, far above the float error of a single subtraction
/// at window-local magnitudes.
pub const CONTACT_SKIN: f32 = 1.0e-3;

/// Swept axis-aligned test: move `box_` by `delta` along `axis` and report how
/// far it gets. EXACT and tunnel-proof at any speed - the whole swept region is
/// tested as one query first, so a fast body cannot step over a thin wall, and
/// the contact plane is then found by bisection (O(log n) queries, n = voxels
/// crossed) rather than by stepping.
///
/// Only voxels the box NEWLY enters are tested. A box that already overlaps
/// something (a voxel was placed inside it, or it was spawned in a wall) is not
/// this function's problem: it may still slide within the cells it occupies, and
/// [`overlaps`] plus a caller-side push-out is the tool for that case.
pub fn sweep(world: &World, box_: Aabb, axis: Axis, delta: f32, set: MatSet) -> Sweep {
    debug_assert!(
        box_.min.is_finite() && box_.max.is_finite(),
        "non-finite AABB {box_:?} in a world sweep",
    );
    let clear = |d: f32| Sweep { free: d, blocked: false, mat: MAT_AIR };
    if !delta.is_finite() || delta == 0.0 {
        return clear(0.0);
    }
    let a = axis as usize;
    let dist = delta.abs();
    let forward = delta > 0.0;

    // The two axes that do not move keep a fixed voxel range for the whole sweep.
    let lo = voxel_lo(box_.min);
    let hi = voxel_hi(box_.max);
    if hi.x < lo.x || hi.y < lo.y || hi.z < lo.z {
        return clear(dist); // degenerate box: covers no voxels, hits nothing
    }

    // Voxel plane index of the leading face now, and where it ends up.
    debug_assert!(box_.min.cmple(box_.max).all(), "inverted AABB {box_:?}");
    let face = if forward { box_.max[a] } else { box_.min[a] };
    let start = if forward { v_hi(face) } else { v_lo(face) };
    let end = if forward { v_hi(face + dist) } else { v_lo(face - dist) };
    let n = if forward { end - start } else { start - end };
    if n <= 0 {
        return clear(dist); // no new voxel plane is entered
    }

    // Find the first blocked plane by EXPONENTIAL SEARCH from the near end,
    // then bisect the interval it lands in.
    //
    // Not a plain bisection over the whole swept range: the overwhelmingly
    // common sweep in a character controller is short and blocked immediately
    // (gravity into the floor), and testing the entire swept region first pays
    // for a large scan before learning what a single plane would have told it.
    // Measured, that cost 2-5x a naive plane-by-plane walk. Nor a plane walk:
    // that is O(distance) and collapses on the fast, mostly-empty sweeps that
    // tunnelling depends on. Galloping is 1 query when the first plane blocks
    // and O(log n) when nothing does, so it is the best of both, and every query
    // covers only planes not already proven clear.
    let planes = |from: i32, to: i32| -> Option<u8> {
        if to < from {
            return None;
        }
        let (p0, p1) = if forward { (start + from, start + to) } else { (start - to, start - from) };
        let mut l = lo;
        let mut h = hi;
        l[a] = p0;
        h[a] = p1;
        any_in_voxel_range(world, l, h, set)
    };
    let mut free_j = 0i32; // planes 1..=free_j are proven clear
    let mut reach = 1i32;
    // `hit` is the range and result of the last test that came back blocked, so
    // the contact material costs no extra query whenever that test covered
    // exactly the contact plane - which is always true of an immediate contact,
    // i.e. of gravity meeting the floor on nearly every tick.
    let (mut blocked_j, mut hit) = loop {
        let j = (free_j + reach).min(n);
        if let Some(m) = planes(free_j + 1, j) {
            break (j, (free_j + 1, j, m));
        }
        free_j = j;
        if free_j >= n {
            return clear(dist); // the whole sweep is clear
        }
        reach = reach.saturating_mul(2);
    };
    while blocked_j - free_j > 1 {
        let mid = free_j + (blocked_j - free_j) / 2;
        if let Some(m) = planes(free_j + 1, mid) {
            hit = (free_j + 1, mid, m);
            blocked_j = mid;
        } else {
            free_j = mid;
        }
    }
    // The face may advance until it reaches the plane bounding the first blocked
    // voxel layer (`start +/- (free_j + 1)`), exclusive - the box is half-open,
    // so resting exactly on that coordinate touches only free voxels.
    let s_max = if forward {
        (start + free_j + 1) as f32 - face
    } else {
        face - (start - free_j) as f32
    };
    // What we hit. The search always ends with blocked_j == free_j + 1, so if the
    // last blocked test covered exactly that one plane its material is already
    // the answer; otherwise ask for the plane by itself.
    let mat = if hit.0 == free_j + 1 && hit.1 == blocked_j {
        hit.2
    } else {
        let mut l = lo;
        let mut h = hi;
        let p = if forward { start + blocked_j } else { start - blocked_j };
        l[a] = p;
        h[a] = p;
        any_in_voxel_range(world, l, h, set).unwrap_or(MAT_AIR)
    };
    Sweep { free: (s_max - CONTACT_SKIN).clamp(0.0, dist), blocked: true, mat }
}

/// The raw integer form of [`overlap_mat`]: any material from `set` in the
/// INCLUSIVE world-voxel range `lo ..= hi`. Digging and area effects want this
/// directly; collision goes through the float box helpers above.
pub fn any_in_voxel_range(world: &World, lo: IVec3, hi: IVec3, set: MatSet) -> Option<u8> {
    if hi.x < lo.x || hi.y < lo.y || hi.z < lo.z {
        return None;
    }
    // Clip to the loaded window, in WORLD space, before anything is folded into
    // slot space - the window is what "we have data for" means.
    let org = world.world_origin_voxel();
    let rlo = (lo - org).max(IVec3::ZERO);
    let rhi = (hi - org).min(IVec3::new(
        WORLD_VOXELS_X as i32 - 1,
        WORLD_VOXELS_Y as i32 - 1,
        WORLD_VOXELS_Z as i32 - 1,
    ));
    if rhi.x < rlo.x || rhi.y < rlo.y || rhi.z < rlo.z {
        return None;
    }
    let wlo = rlo + org;
    let (nx, ny, nz) = (
        (rhi.x - rlo.x + 1) as u32,
        (rhi.y - rlo.y + 1) as u32,
        (rhi.z - rlo.z + 1) as u32,
    );
    // x/z fold toroidally into the slot grid and can wrap; y is not toroidal
    // (the window spans the whole world height) so rel_y IS the slot y.
    let sx = wlo.x.rem_euclid(WORLD_VOXELS_X as i32) as u32;
    let sz = wlo.z.rem_euclid(WORLD_VOXELS_Z as i32) as u32;
    let sy = rlo.y as u32;
    let (xs, nxs) = split_wrap(sx, nx, WORLD_VOXELS_X);
    let (zs, nzs) = split_wrap(sz, nz, WORLD_VOXELS_Z);
    for &(x0, xn) in &xs[..nxs] {
        for &(z0, zn) in &zs[..nzs] {
            let b = SlotBox {
                lo: UVec3::new(x0, sy, z0),
                hi: UVec3::new(x0 + xn - 1, sy + ny - 1, z0 + zn - 1),
            };
            if let Some(m) = descend_l4(world, &b, set) {
                return Some(m);
            }
        }
    }
    None
}

// ------------------------------------------------------------ the descent

/// A slot-space inclusive voxel range that is known in-bounds and known NOT to
/// wrap. Only [`any_in_voxel_range`] constructs one, which is what makes the
/// descent below plain integer arithmetic with no modulo in the inner loops.
struct SlotBox {
    lo: UVec3,
    hi: UVec3,
}

/// Split a slot-space span that may wrap into at most two spans that do not.
/// `n <= w` is guaranteed by the window clip in [`any_in_voxel_range`].
#[inline]
fn split_wrap(start: u32, n: u32, w: u32) -> ([(u32, u32); 2], usize) {
    debug_assert!(n <= w && start < w);
    let first = (w - start).min(n);
    if first == n {
        ([(start, n), (0, 0)], 1)
    } else {
        ([(start, first), (0, n - first)], 2)
    }
}

/// Inclusive range intersection.
#[inline(always)]
fn clip(a0: u32, a1: u32, b0: u32, b1: u32) -> (u32, u32) {
    (if a0 > b0 { a0 } else { b0 }, if a1 < b1 { a1 } else { b1 })
}

fn descend_l4(world: &World, b: &SlotBox, set: MatSet) -> Option<u8> {
    // SMALL RANGES SKIP THE UPPER LEVELS. The L4 and chunk bits are pure
    // acceleration - a tile mask is non-zero only while its chunk and L4 bits
    // are set, so testing the tile mask alone is already correct. For a range
    // covering a handful of tiles those two levels are two extra DEPENDENT loads
    // from two more arrays, i.e. two more chances to miss cache, to save nothing:
    // measured, the full descent made a body-sized query slower than a plain
    // per-voxel loop. The pyramid earns its keep on big and mostly-empty ranges,
    // and that is exactly where it is still used.
    let (tl, th) = (b.lo / (BRICK_DIM * 4), b.hi / (BRICK_DIM * 4));
    let tiles = (th.x - tl.x + 1) * (th.y - tl.y + 1) * (th.z - tl.z + 1);
    if tiles <= 8 {
        for z in tl.z..=th.z {
            for y in tl.y..=th.y {
                for x in tl.x..=th.x {
                    if let Some(m) = descend_one_tile(world, b, UVec3::new(x, y, z), set) {
                        return Some(m);
                    }
                }
            }
        }
        return None;
    }
    let (lo, hi) = (b.lo / (BRICK_DIM * 4 * 4 * 4), b.hi / (BRICK_DIM * 4 * 4 * 4));
    for z in lo.z..=hi.z {
        for y in lo.y..=hi.y {
            for x in lo.x..=hi.x {
                let mask = world.l4_mask[l4_idx(x, y, z) as usize];
                if mask == 0 {
                    continue; // 256^3 voxels skipped by one bit test
                }
                if let Some(m) = descend_chunks(world, b, UVec3::new(x, y, z), mask, set) {
                    return Some(m);
                }
            }
        }
    }
    None
}

fn descend_chunks(world: &World, b: &SlotBox, l4: UVec3, l4_mask: u64, set: MatSet) -> Option<u8> {
    let (lo, hi) = (b.lo / (BRICK_DIM * 4 * 4), b.hi / (BRICK_DIM * 4 * 4));
    let (z0, z1) = clip(lo.z, hi.z, l4.z * 4, l4.z * 4 + 3);
    let (y0, y1) = clip(lo.y, hi.y, l4.y * 4, l4.y * 4 + 3);
    let (x0, x1) = clip(lo.x, hi.x, l4.x * 4, l4.x * 4 + 3);
    for z in z0..=z1 {
        for y in y0..=y1 {
            for x in x0..=x1 {
                if l4_mask & (1u64 << chunk_bit_in_l4(x & 3, y & 3, z & 3)) == 0 {
                    continue; // 64^3 voxels
                }
                let mask = world.chunk_mask[chunk_idx(x, y, z) as usize];
                if mask == 0 {
                    continue;
                }
                if let Some(m) = descend_tiles(world, b, UVec3::new(x, y, z), mask, set) {
                    return Some(m);
                }
            }
        }
    }
    None
}

fn descend_tiles(world: &World, b: &SlotBox, chunk: UVec3, chunk_mask: u64, set: MatSet) -> Option<u8> {
    let (lo, hi) = (b.lo / (BRICK_DIM * 4), b.hi / (BRICK_DIM * 4));
    let (z0, z1) = clip(lo.z, hi.z, chunk.z * 4, chunk.z * 4 + 3);
    let (y0, y1) = clip(lo.y, hi.y, chunk.y * 4, chunk.y * 4 + 3);
    let (x0, x1) = clip(lo.x, hi.x, chunk.x * 4, chunk.x * 4 + 3);
    for z in z0..=z1 {
        for y in y0..=y1 {
            for x in x0..=x1 {
                if chunk_mask & (1u64 << tile_bit_in_chunk(x & 3, y & 3, z & 3)) == 0 {
                    continue; // 16^3 voxels
                }
                if let Some(m) = descend_one_tile(world, b, UVec3::new(x, y, z), set) {
                    return Some(m);
                }
            }
        }
    }
    None
}

/// One tile, from its mask down to individual voxel bits.
#[inline]
fn descend_one_tile(world: &World, b: &SlotBox, tile: UVec3, set: MatSet) -> Option<u8> {
    let ti = tile_idx(tile.x, tile.y, tile.z) as usize;
    let mask = world.tile_mask[ti];
    // The mask, not the brick bytes, decides whether this region exists at all -
    // a recycled streaming slot leaves stale bricks behind a zeroed mask. Checked
    // BEFORE tile_uniform, which is cleared on the same path and would otherwise
    // answer for them.
    if mask == 0 {
        return None;
    }
    let uniform = world.tile_uniform[ti];
    if uniform != 0 {
        // Whole tile is one solid material: 4096 voxels answered by one compare.
        return if set.contains(uniform) { Some(uniform) } else { None };
    }
    descend_bricks(world, b, tile, mask, set)
}

fn descend_bricks(world: &World, b: &SlotBox, tile: UVec3, tile_mask: u64, set: MatSet) -> Option<u8> {
    let (lo, hi) = (b.lo / BRICK_DIM, b.hi / BRICK_DIM);
    let (z0, z1) = clip(lo.z, hi.z, tile.z * 4, tile.z * 4 + 3);
    let (y0, y1) = clip(lo.y, hi.y, tile.y * 4, tile.y * 4 + 3);
    let (x0, x1) = clip(lo.x, hi.x, tile.x * 4, tile.x * 4 + 3);
    for z in z0..=z1 {
        for y in y0..=y1 {
            for x in x0..=x1 {
                if tile_mask & (1u64 << brick_bit_in_tile(x & 3, y & 3, z & 3)) == 0 {
                    continue; // 4^3 voxels
                }
                let bi = brick_idx(x, y, z) as usize;
                let brick = &world.bricks[bi];
                // The box's footprint inside this brick, as one u64 of voxel bits.
                let (bx0, bx1) = clip(b.lo.x, b.hi.x, x * BRICK_DIM, x * BRICK_DIM + 3);
                let (by0, by1) = clip(b.lo.y, b.hi.y, y * BRICK_DIM, y * BRICK_DIM + 3);
                let (bz0, bz1) = clip(b.lo.z, b.hi.z, z * BRICK_DIM, z * BRICK_DIM + 3);
                let want = brick_range_mask(
                    bx0 - x * BRICK_DIM,
                    bx1 - x * BRICK_DIM,
                    by0 - y * BRICK_DIM,
                    by1 - y * BRICK_DIM,
                    bz0 - z * BRICK_DIM,
                    bz1 - z * BRICK_DIM,
                );
                let hit = brick.occupancy & want;
                if hit == 0 {
                    continue;
                }
                let uniform = world.brick_uniform[bi];
                if uniform != 0 {
                    // Uniform bricks are fully solid with one material, so any
                    // overlapping bit answers for all 64.
                    if set.contains(uniform) {
                        return Some(uniform);
                    }
                    continue;
                }
                let mut bits = hit;
                while bits != 0 {
                    let i = bits.trailing_zeros() as usize;
                    bits &= bits - 1;
                    let mat = brick.materials[i];
                    if set.contains(mat) {
                        return Some(mat);
                    }
                }
            }
        }
    }
    None
}

/// Voxel-bit mask of an inclusive brick-local range, in the brick's
/// `x + z*4 + y*16` order (see [`brick_voxel_idx`]).
///
/// The two multiplies are shift-sums, not arithmetic: `xbits` is 4 bits wide and
/// `SPREAD_Z`'s bits sit at 4-bit strides, so the product is exactly
/// `sum_z(xbits << 4z)` with no carries; likewise the 16-bit `xz` against
/// `SPREAD_Y`'s 16-bit strides. Branch-free, and pinned against a naive loop by
/// `brick_range_mask_matches_a_naive_loop`.
#[inline(always)]
fn brick_range_mask(x0: u32, x1: u32, y0: u32, y1: u32, z0: u32, z1: u32) -> u64 {
    /// bit i of the index -> bit i*stride of the value.
    const fn spread(stride: u32) -> [u64; 16] {
        let mut table = [0u64; 16];
        let mut sel = 0usize;
        while sel < 16 {
            let mut bits = 0u64;
            let mut i = 0u32;
            while i < 4 {
                if (sel >> i) & 1 == 1 {
                    bits |= 1u64 << (i * stride);
                }
                i += 1;
            }
            table[sel] = bits;
            sel += 1;
        }
        table
    }
    const SPREAD_Z: [u64; 16] = spread(4);
    const SPREAD_Y: [u64; 16] = spread(16);
    debug_assert!(x1 < 4 && y1 < 4 && z1 < 4 && x0 <= x1 && y0 <= y1 && z0 <= z1);
    let xbits = ((1u64 << (x1 - x0 + 1)) - 1) << x0;
    let zsel = (((1u32 << (z1 - z0 + 1)) - 1) << z0) as usize;
    let ysel = (((1u32 << (y1 - y0 + 1)) - 1) << y0) as usize;
    (xbits * SPREAD_Z[zsel]) * SPREAD_Y[ysel]
}

// -------------------------------------------------------------- float -> voxel

/// Lowest voxel index a face at `x` touches (the box is half-open at max).
#[inline(always)]
fn v_lo(x: f32) -> i32 {
    x.floor() as i32
}

/// Highest voxel index a face at `x` touches.
#[inline(always)]
fn v_hi(x: f32) -> i32 {
    x.ceil() as i32 - 1
}

#[inline(always)]
fn voxel_lo(v: Vec3) -> IVec3 {
    IVec3::new(v_lo(v.x), v_lo(v.y), v_lo(v.z))
}

#[inline(always)]
fn voxel_hi(v: Vec3) -> IVec3 {
    IVec3::new(v_hi(v.x), v_hi(v.y), v_hi(v.z))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic PRNG (splitmix64) so a failure is reproducible; the crate
    /// has no rand dependency and this needs no statistical quality.
    struct Rng(u64);
    impl Rng {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u32) -> u32 {
            (self.next_u64() % n as u64) as u32
        }
        fn range_i(&mut self, lo: i32, hi: i32) -> i32 {
            lo + self.below((hi - lo + 1) as u32) as i32
        }
        fn unit(&mut self) -> f32 {
            (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
        }
    }

    /// Reference implementation: one voxel at a time through the world's own
    /// "what is at this world voxel" accessor. Deliberately shares NOTHING with
    /// the hierarchy path - that is what makes the equivalence test meaningful.
    fn naive_any(world: &World, lo: IVec3, hi: IVec3, set: MatSet) -> Option<u8> {
        for z in lo.z..=hi.z {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let m = world.material_at_world(x, y, z);
                    if set.contains(m) {
                        return Some(m);
                    }
                }
            }
        }
        None
    }

    /// A palette spanning every classification branch that matters: solid, water
    /// (uniform-capable), and foliage (occupied but walk-through).
    const PALETTE: [u8; 6] =
        [MAT_STONE, MAT_SAND, MAT_GLASS, MAT_WATER_L8, MAT_LEAVES, MAT_TURF];

    /// Scatter random materials over a box of world voxels, through the world's
    /// own edit path so masks, uniform hints and bricks all stay consistent.
    fn scatter(world: &mut World, rng: &mut Rng, origin: IVec3, extent: i32, fill_pct: u32) {
        for dz in 0..extent {
            for dy in 0..extent {
                for dx in 0..extent {
                    if rng.below(100) >= fill_pct {
                        continue;
                    }
                    let mat = PALETTE[rng.below(PALETTE.len() as u32) as usize];
                    world.apply_edit(origin.x + dx, origin.y + dy, origin.z + dz, mat);
                }
            }
        }
    }

    /// First slot-wrap seam strictly above `v`: the world x/z where the toroidal
    /// fold restarts at slot 0. Test geometry is placed across one of these on
    /// purpose - a range that straddles it is the case a naive `world - origin`
    /// implementation gets silently wrong.
    fn seam_after(v: i32, w: i32) -> i32 {
        (v.div_euclid(w) + 1) * w
    }

    /// A world whose window has been moved far from spawn AND is deliberately
    /// NOT slot-aligned, so the loaded window contains a wrap seam. (An origin
    /// that is a multiple of WORLD_VOXELS_X folds to slot == rel and would prove
    /// nothing.) Returns the world plus the seam coordinates inside it.
    fn far_world() -> (World, i32, i32) {
        let mut world = World::new();
        // 100_008 chunks = 3_200_256 voxels: 3.2M from spawn (the same distance
        // renders_far_from_origin uses) and 256 voxels past a slot seam.
        world.world_origin_chunk = glam::IVec2::new(100_008, 100_004);
        let org = world.world_origin_voxel();
        let seam_x = seam_after(org.x, WORLD_VOXELS_X as i32);
        let seam_z = seam_after(org.z, WORLD_VOXELS_Z as i32);
        assert!(
            seam_x - org.x > 32 && seam_x - org.x < WORLD_VOXELS_X as i32 - 32,
            "the seam must sit well inside the window for this to test anything",
        );
        (world, seam_x, seam_z)
    }

    fn fill_box(world: &mut World, origin: IVec3, extent: i32, mat: u8) {
        for dz in 0..extent {
            for dy in 0..extent {
                for dx in 0..extent {
                    world.apply_edit(origin.x + dx, origin.y + dy, origin.z + dz, mat);
                }
            }
        }
    }

    #[test]
    fn mat_count_matches_the_last_material() {
        // If a material is added without bumping MAT_COUNT, MatSet would still
        // work but nothing would force a look at mat_is_solid - and an
        // unclassified material silently behaving like stone is exactly the kind
        // of drift the const assert cannot catch.
        assert_eq!(MAT_COUNT, MAT_TREE_TEST + 1);
        assert!(MAT_COUNT <= 64);
    }

    #[test]
    fn the_solid_set_matches_its_predicate_for_every_material() {
        for m in 0..64u8 {
            assert_eq!(MatSet::SOLID.contains(m), mat_is_solid(m), "material {m}");
            assert_eq!(MatSet::WATER.contains(m), is_water_mat(m), "material {m}");
            assert_eq!(MatSet::ANY.contains(m), m != MAT_AIR, "material {m}");
        }
        // The classification a body depends on, spelled out.
        assert!(MatSet::SOLID.contains(MAT_STONE));
        assert!(MatSet::SOLID.contains(MAT_SAND));
        assert!(MatSet::SOLID.contains(MAT_SNOW));
        assert!(!MatSet::SOLID.contains(MAT_AIR));
        assert!(!MatSet::SOLID.contains(MAT_WATER_L8));
        assert!(!MatSet::SOLID.contains(MAT_LEAVES));
        assert!(!MatSet::SOLID.contains(MAT_LEAF_FRINGE));
        assert!(!MatSet::SOLID.contains(MAT_TURF));
        assert!(!MatSet::SOLID.contains(MAT_TALL_GRASS));
        assert!(!MatSet::SOLID.contains(MAT_SMOKE));
        assert!(!MatSet::SOLID.contains(MAT_FIRE));
    }

    #[test]
    fn brick_range_mask_matches_a_naive_loop() {
        // Exhaustive over every inclusive sub-range of a brick (1000 of them).
        for x0 in 0..4 {
            for x1 in x0..4 {
                for y0 in 0..4 {
                    for y1 in y0..4 {
                        for z0 in 0..4 {
                            for z1 in z0..4 {
                                let mut want = 0u64;
                                for z in z0..=z1 {
                                    for y in y0..=y1 {
                                        for x in x0..=x1 {
                                            want |= 1u64 << brick_voxel_idx(x, y, z);
                                        }
                                    }
                                }
                                assert_eq!(
                                    brick_range_mask(x0, x1, y0, y1, z0, z1),
                                    want,
                                    "range x{x0}..{x1} y{y0}..{y1} z{z0}..{z1}",
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn hierarchy_agrees_with_naive_on_random_worlds() {
        let mut world = World::new();
        let mut rng = Rng(0xA11CE);
        // Dense, sparse, and dense-next-to-a-fully-uniform-tile: the three
        // regimes the descent takes different paths through (per-voxel bits,
        // mostly-empty skipping, and the brick/tile uniform shortcuts). The
        // third origin is 16-aligned so the fills below really do make whole
        // tiles uniform.
        let rounds: [(IVec3, u32); 3] = [
            (IVec3::new(48, 32, 48), 60),
            (IVec3::new(144, 48, 144), 8),
            (IVec3::new(240, 64, 240), 45),
        ];
        for (round, (origin, fill_pct)) in rounds.into_iter().enumerate() {
            scatter(&mut world, &mut rng, origin, 20, fill_pct);
            if round == 2 {
                // Tile-aligned uniform blocks (16^3): one solid, one water. The
                // uniform shortcut answers for 4096 voxels at once and has to
                // get the non-solid one right too.
                fill_box(&mut world, origin, 16, MAT_STONE);
                fill_box(&mut world, origin + IVec3::new(16, 0, 0), 16, MAT_WATER_L8);
                assert_ne!(
                    world.tile_uniform[tile_idx(
                        origin.x as u32 / 16,
                        origin.y as u32 / 16,
                        origin.z as u32 / 16
                    ) as usize],
                    0,
                    "round 2 must actually produce a uniform tile",
                );
            }
            for _ in 0..600 {
                let lo = IVec3::new(
                    origin.x + rng.range_i(-6, 20),
                    origin.y + rng.range_i(-6, 20),
                    origin.z + rng.range_i(-6, 20),
                );
                let hi = lo + IVec3::new(rng.range_i(0, 9), rng.range_i(0, 9), rng.range_i(0, 9));
                for set in [MatSet::SOLID, MatSet::WATER, MatSet::ANY] {
                    let fast = any_in_voxel_range(&world, lo, hi, set);
                    let slow = naive_any(&world, lo, hi, set);
                    assert_eq!(
                        fast.is_some(),
                        slow.is_some(),
                        "range {lo:?}..={hi:?} set {set:?}: hierarchy {fast:?} vs naive {slow:?}",
                    );
                    if let Some(m) = fast {
                        // Unspecified WHICH material, but it must be in the set
                        // and actually present in the range.
                        assert!(set.contains(m));
                        assert!(naive_any(&world, lo, hi, MatSet(1u64 << m)).is_some());
                    }
                }
            }
        }
    }

    #[test]
    fn hierarchy_agrees_with_naive_far_from_origin() {
        // The toroidal fold is invisible at spawn (origin 0 makes slot == world)
        // and position-dependent everywhere else, so the equivalence has to be
        // proven where the window has actually moved - the same reason
        // renderer::renders_far_from_origin exists.
        let (mut world, seam_x, seam_z) = far_world();
        let mut rng = Rng(0xFA4);
        // Two regions: one ordinary, one sitting ON the x seam and one on the z
        // seam, so ranges genuinely split into two and four slot boxes.
        let regions = [
            IVec3::new(seam_x - 40, 60, seam_z - 40),
            IVec3::new(seam_x - 8, 92, seam_z - 40),
            IVec3::new(seam_x - 8, 124, seam_z - 8),
        ];
        for origin in regions {
            scatter(&mut world, &mut rng, origin, 16, 40);
            for _ in 0..500 {
                let lo = origin
                    + IVec3::new(rng.range_i(-5, 16), rng.range_i(-5, 16), rng.range_i(-5, 16));
                let hi = lo + IVec3::new(rng.range_i(0, 8), rng.range_i(0, 8), rng.range_i(0, 8));
                for set in [MatSet::SOLID, MatSet::ANY] {
                    let fast = any_in_voxel_range(&world, lo, hi, set);
                    let slow = naive_any(&world, lo, hi, set);
                    assert_eq!(
                        fast.is_some(),
                        slow.is_some(),
                        "far range {lo:?}..={hi:?} (seam x {seam_x}, z {seam_z})",
                    );
                }
            }
        }
    }

    #[test]
    fn a_uniform_tile_answers_without_descending() {
        // 16^3 of one material makes tile_uniform non-zero; the descent must
        // answer from it, and must answer CORRECTLY for a non-solid material -
        // a uniform water tile is a hole you fall into, not a floor.
        let mut world = World::new();
        fill_box(&mut world, IVec3::new(64, 32, 64), 16, MAT_STONE);
        fill_box(&mut world, IVec3::new(96, 32, 64), 16, MAT_WATER_L8);
        let ti = tile_idx(64 / 16, 32 / 16, 64 / 16) as usize;
        assert_eq!(world.tile_uniform[ti], MAT_STONE, "the stone tile should be uniform");
        let stone = Aabb::new(Vec3::new(66.0, 34.0, 66.0), Vec3::new(68.0, 36.0, 68.0));
        let water = Aabb::new(Vec3::new(98.0, 34.0, 66.0), Vec3::new(100.0, 36.0, 68.0));
        assert_eq!(overlap_mat(&world, stone, MatSet::SOLID), Some(MAT_STONE));
        assert_eq!(overlap_mat(&world, water, MatSet::SOLID), None);
        assert_eq!(overlap_mat(&world, water, MatSet::WATER), Some(MAT_WATER_L8));
    }

    #[test]
    fn the_pyramid_is_consistent_at_every_level() {
        // The small-range path in `descend_l4` consults ONLY the tile mask. That
        // is correct exactly while a non-zero tile mask implies its chunk bit and
        // L4 bit are set - an invariant of World's mask maintenance, not of this
        // module, so it is CHECKED here rather than assumed. (It is also what
        // lets the large-range path trust an unset chunk bit and skip 64^3.)
        let check = |w: &World, when: &str| {
            for tz in 0..WORLD_TILES_Z {
                for ty in 0..WORLD_TILES_Y {
                    for tx in 0..WORLD_TILES_X {
                        if w.tile_mask[tile_idx(tx, ty, tz) as usize] == 0 {
                            continue;
                        }
                        let (cx, cy, cz) = (tx / 4, ty / 4, tz / 4);
                        let cm = w.chunk_mask[chunk_idx(cx, cy, cz) as usize];
                        assert_ne!(
                            cm & (1u64 << tile_bit_in_chunk(tx & 3, ty & 3, tz & 3)),
                            0,
                            "{when}: tile ({tx},{ty},{tz}) is occupied but its chunk bit is clear",
                        );
                        let lm = w.l4_mask[l4_idx(cx / 4, cy / 4, cz / 4) as usize];
                        assert_ne!(
                            lm & (1u64 << chunk_bit_in_l4(cx & 3, cy & 3, cz & 3)),
                            0,
                            "{when}: chunk ({cx},{cy},{cz}) is occupied but its L4 bit is clear",
                        );
                    }
                }
            }
        };
        let mut w = World::new();
        w.fill_demo_terrain();
        check(&w, "demo world");
        // A streaming round trip clears masks (clear_slot_masks) and rebuilds
        // them (apply_slot_data) through two different code paths.
        w.shift_origin(glam::IVec2::new(3, -5));
        w.process_pending_gen_blocking();
        check(&w, "after streaming");
        // ... and so does an edit.
        w.apply_edit(300, 200, 300, MAT_STONE);
        check(&w, "after an edit");
    }

    #[test]
    fn a_cleared_mask_hides_its_bricks_at_every_level() {
        // Recycling a streaming slot zeroes mask bits and leaves the brick bytes
        // in place; the GPU renders sky there, so the CPU must agree.
        let mut world = World::new();
        fill_box(&mut world, IVec3::new(64, 32, 64), 4, MAT_STONE);
        let probe = Aabb::new(Vec3::new(65.0, 33.0, 65.0), Vec3::new(66.0, 34.0, 66.0));
        assert!(overlaps(&world, probe, MatSet::SOLID));
        // The TILE mask is the authoritative one - it is what a slot recycle
        // actually clears, and it is consulted on every path.
        let saved = world.tile_mask[tile_idx(4, 2, 4) as usize];
        world.tile_mask[tile_idx(4, 2, 4) as usize] = 0;
        assert!(!overlaps(&world, probe, MatSet::SOLID), "tile mask ignored");
        world.tile_mask[tile_idx(4, 2, 4) as usize] = saved;
        assert!(overlaps(&world, probe, MatSet::SOLID));
        assert_eq!(world.material_at_world(65, 33, 65), MAT_STONE, "bricks untouched");

        // The chunk and L4 bits are ACCELERATION, valid because the pyramid is
        // consistent (pinned by the test above), and only the large-range path
        // reads them - a body-sized query skips straight to the tile. So they are
        // exercised with a query big enough to take that path.
        let wide = Aabb::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(255.0, 200.0, 255.0));
        assert!(overlaps(&world, wide, MatSet::SOLID));
        let ci = chunk_idx(1, 0, 1) as usize;
        let saved_chunk = world.chunk_mask[ci];
        world.chunk_mask[ci] = 0;
        assert!(!overlaps(&world, wide, MatSet::SOLID), "chunk mask ignored");
        world.chunk_mask[ci] = saved_chunk;
        world.l4_mask[l4_idx(0, 0, 0) as usize] = 0;
        assert!(!overlaps(&world, wide, MatSet::SOLID), "L4 mask ignored");
    }

    #[test]
    fn a_uniform_tile_behind_a_cleared_mask_is_still_hidden() {
        // The tile-uniform shortcut can answer for 4096 voxels without looking
        // at a single brick, so it MUST come after the mask test, not before -
        // otherwise a recycled slot would collide as a solid block of whatever
        // it used to be. Pins the ordering inside descend_tiles.
        let mut world = World::new();
        fill_box(&mut world, IVec3::new(64, 32, 64), 16, MAT_STONE);
        let ti = tile_idx(4, 2, 4) as usize;
        assert_eq!(world.tile_uniform[ti], MAT_STONE);
        let probe = Aabb::new(Vec3::new(70.0, 38.0, 70.0), Vec3::new(72.0, 40.0, 72.0));
        assert!(overlaps(&world, probe, MatSet::SOLID));
        world.tile_mask[ti] = 0; // as a slot recycle does, but leaving the hint
        assert!(!overlaps(&world, probe, MatSet::SOLID), "uniform hint outran the mask");
    }

    #[test]
    fn outside_the_loaded_window_reads_as_empty() {
        let mut world = World::new();
        fill_box(&mut world, IVec3::new(64, 32, 64), 4, MAT_STONE);
        // Below the world floor, above its roof, and beyond the window in x:
        // no data means empty, not a wall.
        let below = Aabb::new(Vec3::new(64.0, -4.0, 64.0), Vec3::new(68.0, -1.0, 68.0));
        let above = Aabb::new(
            Vec3::new(64.0, WORLD_VOXELS_Y as f32 + 1.0, 64.0),
            Vec3::new(68.0, WORLD_VOXELS_Y as f32 + 4.0, 68.0),
        );
        let beyond = Aabb::new(Vec3::new(-40.0, 32.0, 64.0), Vec3::new(-36.0, 36.0, 68.0));
        assert!(!overlaps(&world, below, MatSet::ANY));
        assert!(!overlaps(&world, above, MatSet::ANY));
        assert!(!overlaps(&world, beyond, MatSet::ANY));
        // A box straddling the floor still sees what IS loaded.
        let straddle = Aabb::new(Vec3::new(64.0, -2.0, 64.0), Vec3::new(68.0, 34.0, 68.0));
        assert!(overlaps(&world, straddle, MatSet::SOLID));
    }

    #[test]
    fn a_zero_extent_box_touches_nothing() {
        let mut world = World::new();
        fill_box(&mut world, IVec3::new(64, 32, 64), 4, MAT_STONE);
        let flat = Aabb::new(Vec3::new(65.0, 33.0, 65.0), Vec3::new(66.0, 33.0, 66.0));
        assert!(!overlaps(&world, flat, MatSet::SOLID));
        assert!(!sweep(&world, flat, Axis::X, 4.0, MatSet::SOLID).blocked);
    }

    /// Reference sweep: creep forward in tiny steps and stop at the first overlap.
    /// O(distance) and obviously correct, which is the point.
    fn naive_sweep(world: &World, box_: Aabb, axis: Axis, delta: f32, set: MatSet) -> f32 {
        let step = 1.0 / 512.0;
        let n = (delta.abs() / step).ceil() as i32;
        let dir = delta.signum();
        let mut moved = 0.0f32;
        let start_overlap = {
            let lo = voxel_lo(box_.min);
            let hi = voxel_hi(box_.max);
            let mut cells = std::collections::HashSet::new();
            for z in lo.z..=hi.z {
                for y in lo.y..=hi.y {
                    for x in lo.x..=hi.x {
                        cells.insert((x, y, z));
                    }
                }
            }
            cells
        };
        for i in 1..=n {
            let s = (i as f32 * step).min(delta.abs());
            let moved_box = box_.translated(match axis {
                Axis::X => Vec3::new(dir * s, 0.0, 0.0),
                Axis::Y => Vec3::new(0.0, dir * s, 0.0),
                Axis::Z => Vec3::new(0.0, 0.0, dir * s),
            });
            // Only NEWLY entered voxels count, matching the sweep's contract.
            let lo = voxel_lo(moved_box.min);
            let hi = voxel_hi(moved_box.max);
            let mut hit = false;
            for z in lo.z..=hi.z {
                for y in lo.y..=hi.y {
                    for x in lo.x..=hi.x {
                        if start_overlap.contains(&(x, y, z)) {
                            continue;
                        }
                        if set.contains(world.material_at_world(x, y, z)) {
                            hit = true;
                        }
                    }
                }
            }
            if hit {
                break;
            }
            moved = s;
        }
        moved
    }

    #[test]
    fn sweep_agrees_with_a_stepped_reference() {
        let mut world = World::new();
        let mut rng = Rng(0x5EE0);
        let origin = IVec3::new(70, 40, 70);
        // Sparse, so a box placed at random usually starts in clear air (a dense
        // field would make nearly every sample a skip, and the test would prove
        // nothing while still passing).
        scatter(&mut world, &mut rng, origin, 18, 10);
        let mut compared = 0;
        for _ in 0..900 {
            let start = Vec3::new(
                origin.x as f32 - 4.0 + rng.unit() * 24.0,
                origin.y as f32 - 4.0 + rng.unit() * 24.0,
                origin.z as f32 - 4.0 + rng.unit() * 24.0,
            );
            // A body-shaped box in a random cloud is almost always already
            // intersecting something; this one is small enough to find air.
            let b = Aabb::from_feet(start, 0.45, 1.3);
            // Skip cases that start already overlapping: "how far can it move"
            // is defined only against the voxels it NEWLY enters, and the
            // stepped reference cannot express that cleanly when the box is
            // already buried.
            if overlaps(&world, b, MatSet::SOLID) {
                continue;
            }
            compared += 1;
            let axis = Axis::ALL[rng.below(3) as usize];
            let delta = (rng.unit() * 2.0 - 1.0) * 9.0;
            let got = sweep(&world, b, axis, delta, MatSet::SOLID);
            let want = naive_sweep(&world, b, axis, delta, MatSet::SOLID);
            // The stepped reference can only resolve to its step size, and the
            // primitive stops CONTACT_SKIN short of the exact plane.
            assert!(
                (got.free - want).abs() <= 1.0 / 512.0 + 2.0 * CONTACT_SKIN,
                "sweep {axis:?} by {delta} from {start:?}: got {} want {want}",
                got.free,
            );
            if !got.blocked {
                assert!(
                    (want - delta.abs()).abs() < 1.0e-4,
                    "reported clear but the reference stopped at {want} of {delta}",
                );
            }
        }
        assert!(compared > 200, "only {compared} usable samples - the field is too dense");
    }

    #[test]
    fn sweep_does_not_tunnel_through_a_thin_wall_at_any_speed() {
        // One voxel thick, and a step that crosses hundreds of voxels: a
        // step-and-test controller walks straight through this, which is the
        // whole reason the swept region is tested as one query.
        let mut world = World::new();
        for y in 30..50 {
            for z in 60..80 {
                world.apply_edit(200, y, z, MAT_STONE);
            }
        }
        // Leading face 2.8 voxels short of the wall, so every speed below
        // reaches it and the only question is whether it is noticed.
        let feet = Vec3::new(196.0, 32.0, 70.0);
        let body = Aabb::from_feet(feet, 1.2, 7.2);
        for speed in [3.0f32, 30.0, 180.0, 500.0, 5000.0] {
            let s = sweep(&world, body, Axis::X, speed, MatSet::SOLID);
            assert!(s.blocked, "speed {speed} tunnelled");
            assert_eq!(s.mat, MAT_STONE);
            // Stops with its leading face just short of the wall at x = 200.
            let face = body.max.x + s.free;
            assert!(
                (200.0 - face) >= 0.0 && (200.0 - face) < 0.01,
                "speed {speed}: leading face {face} should rest just under 200",
            );
        }
        // ... and the same wall from the other side.
        let far = Aabb::from_feet(Vec3::new(400.0, 32.0, 70.0), 1.2, 7.2);
        let s = sweep(&world, far, Axis::X, -500.0, MatSet::SOLID);
        assert!(s.blocked);
        let face = far.min.x - s.free;
        assert!((face - 201.0) >= 0.0 && (face - 201.0) < 0.01, "face {face}");
    }

    #[test]
    fn sweep_far_from_origin_matches_the_same_shape_at_spawn() {
        // Same local geometry, once at spawn and once 3.2M voxels out with the
        // swept range crossing a slot seam: the toroidal fold must make no
        // difference at all.
        let build = |world: &mut World, org: IVec3| {
            for y in 0..8 {
                for z in 0..8 {
                    world.apply_edit(org.x + 12, org.y + y, org.z + z, MAT_STONE);
                }
            }
        };
        let mut near = World::new();
        let near_org = IVec3::new(80, 40, 80);
        build(&mut near, near_org);

        let (mut far, seam_x, _) = far_world();
        // Body starts before the seam, wall sits after it, so the sweep's range
        // splits in two.
        let far_org = IVec3::new(seam_x - 6, 40, 80 + far.world_origin_voxel().z);
        build(&mut far, far_org);

        for (world, org) in [(&near, near_org), (&far, far_org)] {
            let b = Aabb::from_feet(
                Vec3::new(org.x as f32 + 2.5, org.y as f32 + 0.0, org.z as f32 + 3.5),
                1.2,
                7.2,
            );
            let s = sweep(world, b, Axis::X, 20.0, MatSet::SOLID);
            assert!(s.blocked, "origin {org:?}");
            assert_eq!(s.mat, MAT_STONE, "origin {org:?}");
            let face_local = b.max.x + s.free - org.x as f32;
            assert!(
                (12.0 - face_local) >= 0.0 && (12.0 - face_local) < 0.01,
                "origin {org:?}: local face {face_local}",
            );
        }
    }

    #[test]
    fn sweeps_resolve_contact_exactly_on_the_voxel_plane() {
        // A floor at y in [40, 44); a box dropped from above must come to rest
        // with its feet on y = 44, within the contact skin - this is what stops
        // a resting body from creeping into the ground tick after tick.
        let mut world = World::new();
        fill_box(&mut world, IVec3::new(60, 40, 60), 4, MAT_STONE);
        let b = Aabb::from_feet(Vec3::new(62.0, 50.0, 62.0), 1.0, 7.2);
        let s = sweep(&world, b, Axis::Y, -20.0, MatSet::SOLID);
        assert!(s.blocked);
        let feet = b.min.y - s.free;
        assert!((feet - 44.0).abs() <= CONTACT_SKIN * 1.5, "feet at {feet}, want 44");
        // Resting there, a further downward sweep yields nothing at all.
        let rest = Aabb::from_feet(Vec3::new(62.0, feet, 62.0), 1.0, 7.2);
        let s2 = sweep(&world, rest, Axis::Y, -1.0, MatSet::SOLID);
        assert_eq!(s2.free, 0.0, "a resting box must not sink");
        assert!(s2.blocked);
    }

    /// A NAIVE-BUT-SANE sweep: walk the crossed voxel planes one at a time and
    /// test each cross-section voxel by voxel. This is what collision looks like
    /// without the hierarchy, and it is the honest baseline to beat - unlike the
    /// 1/512-voxel `naive_sweep` above, which exists to prove correctness and
    /// would never be written for speed.
    fn plane_by_plane_sweep(world: &World, box_: Aabb, axis: Axis, delta: f32, set: MatSet) -> f32 {
        let a = axis as usize;
        let forward = delta > 0.0;
        let dist = delta.abs();
        let lo = voxel_lo(box_.min);
        let hi = voxel_hi(box_.max);
        let face = if forward { box_.max[a] } else { box_.min[a] };
        let start = if forward { v_hi(face) } else { v_lo(face) };
        let end = if forward { v_hi(face + dist) } else { v_lo(face - dist) };
        let n = if forward { end - start } else { start - end };
        for j in 1..=n {
            let p = if forward { start + j } else { start - j };
            let mut l = lo;
            let mut h = hi;
            l[a] = p;
            h[a] = p;
            if naive_any(world, l, h, set).is_some() {
                let s_max = if forward {
                    (start + j) as f32 - face
                } else {
                    face - (start - j + 1) as f32
                };
                return s_max.max(0.0);
            }
        }
        dist
    }

    /// Speed of the hierarchy query against the naive per-voxel alternative, on
    /// the real demo terrain. Ignored by default (it is a measurement, not an
    /// assertion): `cargo test --release --lib -- --ignored --nocapture speed`.
    #[test]
    #[ignore]
    fn hierarchy_vs_naive_speed() {
        use std::time::Instant;
        let mut world = World::new();
        world.fill_demo_terrain();
        let mut rng = Rng(0x9001);
        // Player-sized boxes. Two populations, because they measure different
        // things: SCATTERED over the whole terrain is memory-bound (every brick
        // is a cache miss for both implementations, which compresses the ratio),
        // while LOCAL boxes around one spot are what a body actually queries -
        // a dozen times a tick, in the same few bricks, warm in cache.
        let make = |rng: &mut Rng, spread: f32, cx: f32, cz: f32| -> Vec<Aabb> {
            (0..20_000)
                .map(|_| {
                    let x = cx + (rng.unit() - 0.5) * spread;
                    let z = cz + (rng.unit() - 0.5) * spread;
                    let s = sample_terrain(x, z, world.seed);
                    let y = s.h as f32 + rng.unit() * 8.0 - 2.0;
                    Aabb::from_feet(Vec3::new(x, y, z), 1.2, 7.2)
                })
                .collect()
        };
        let scattered = make(&mut rng, 400.0, 250.0, 250.0);
        let local = make(&mut rng, 8.0, 250.0, 250.0);

        // A dig scoop and a blast radius: the volumes slice item 2 will ask
        // about. A body-sized box is only ~150 voxels, so the per-voxel loop
        // stays in the running; these are where the hierarchy's asymptotics show.
        for (name, r) in [("dig scoop 8^3", 4.0f32), ("blast 32^3", 16.0)] {
            let boxes: Vec<Aabb> = scattered
                .iter()
                .take(2000)
                .map(|b| {
                    let c = (b.min + b.max) * 0.5;
                    Aabb::new(c - Vec3::splat(r), c + Vec3::splat(r))
                })
                .collect();
            let n = boxes.len() as f64;
            let t = Instant::now();
            let mut hits = 0usize;
            for b in &boxes {
                hits += overlaps(&world, *b, MatSet::SOLID) as usize;
            }
            let fast = t.elapsed();
            let t = Instant::now();
            let mut hits2 = 0usize;
            for b in &boxes {
                hits2 += naive_any(&world, voxel_lo(b.min), voxel_hi(b.max), MatSet::SOLID)
                    .is_some() as usize;
            }
            let slow = t.elapsed();
            assert_eq!(hits, hits2);
            eprintln!(
                "volume overlap [{name}] ({hits}/{} solid): hierarchy {:.0} ns, per-voxel {:.0} ns, {:.1}x",
                boxes.len(),
                fast.as_secs_f64() * 1e9 / n,
                slow.as_secs_f64() * 1e9 / n,
                slow.as_secs_f64() / fast.as_secs_f64(),
            );
        }

        for (name, boxes) in [("scattered", &scattered), ("local", &local)] {
            let n = boxes.len() as f64;
            let t = Instant::now();
            let mut hits = 0usize;
            for b in boxes.iter() {
                hits += overlaps(&world, *b, MatSet::SOLID) as usize;
            }
            let fast = t.elapsed();
            let t = Instant::now();
            let mut hits2 = 0usize;
            for b in boxes.iter() {
                hits2 += naive_any(&world, voxel_lo(b.min), voxel_hi(b.max), MatSet::SOLID).is_some()
                    as usize;
            }
            let slow = t.elapsed();
            assert_eq!(hits, hits2, "{name}: the two implementations disagree");
            eprintln!(
                "AABB overlap [{name}] ({hits}/{} solid): hierarchy {:.0} ns, per-voxel {:.0} ns, {:.1}x",
                boxes.len(),
                fast.as_secs_f64() * 1e9 / n,
                slow.as_secs_f64() * 1e9 / n,
                slow.as_secs_f64() / fast.as_secs_f64(),
            );

            // Sweeps, both regimes. DOWN is blocked almost immediately (gravity
            // into the floor - the common tick, and the case a plane walk is
            // best at); UP runs 40 voxels through open sky (the fast-moving,
            // mostly-empty case that tunnelling depends on, where a plane walk
            // is O(distance) and the gallop is O(log)).
            for (dir, dist) in [(-1.0f32, 0.5f32), (-1.0, 40.0), (1.0, 40.0)] {
                let t = Instant::now();
                let mut acc = 0.0f64;
                for b in boxes.iter() {
                    acc += sweep(&world, *b, Axis::Y, dir * dist, MatSet::SOLID).free as f64;
                }
                let e = t.elapsed();
                let t = Instant::now();
                let mut acc2 = 0.0f64;
                for b in boxes.iter() {
                    acc2 += plane_by_plane_sweep(&world, *b, Axis::Y, dir * dist, MatSet::SOLID) as f64;
                }
                let e2 = t.elapsed();
                eprintln!(
                    "  sweep {:>5} voxels: hierarchy {:.0} ns, plane-by-plane {:.0} ns, {:.1}x  (free totals {:.1} vs {:.1})",
                    dir * dist,
                    e.as_secs_f64() * 1e9 / n,
                    e2.as_secs_f64() * 1e9 / n,
                    e2.as_secs_f64() / e.as_secs_f64(),
                    acc,
                    acc2,
                );
            }
        }
    }
}
