// Cellular-automaton water physics following the DwarfCorp approach
// (https://www.gamedeveloper.com/programming/how-water-works-in-dwarfcorp):
// each water voxel carries an integer "level" 1..8 stored in its material
// (MAT_WATER_L1..MAT_WATER_L8). Per tick:
//   1. Gravity:  push level from a cell into the cell below, capped at L8.
//   2. Lateral:  give 1 level to a horizontal neighbour that has strictly
//      less water, so columns equalise over time. A per-cell hash chooses
//      which side gets the donation so adjacent cells don't all donate the
//      same way.
//   3. Cleanup:  any cell whose level falls to 0 becomes MAT_AIR.
//
// Optimizations layered on:
//   * Active bricks list — `World::active_bricks` is the AWAKE set (see
//     "Sleeping" below). Stone-only bricks are completely skipped.
//   * Bitmask gravity for sand (still binary).
//   * Multi-iteration intra-brick fall so a stack of floating water cells
//     collapses in one tick instead of one cell per tick.
//
// SLEEPING (the thing that makes a lake free).
//
// `active_bricks` used to mean "holds a movable voxel", which for water is
// permanent: a lake is water forever, so 30 times a second the CA re-derived
// that the lake is still a lake. A settled cell could only leave the set by
// draining to nothing, and for water "settled" and "empty" are OPPOSITES
// (sand only looked right because for sand they coincide). At 10 cm over a
// 23.5%-water window that was 168 ms a tick against a 33 ms budget.
//
// Now `active_bricks` means AWAKE. A tick records every brick whose voxels
// CHANGED (`touched`), and ends in `retire_and_wake`, which rebuilds the awake
// set as exactly `changed + the movable face-neighbours of changed` plus the
// bricks that hold a time-dependent rule (smoke). A brick that moved nothing
// and borders nothing that moved is settled, and settled costs zero.
//
// The obligation this buys is that every disturbance must WAKE what it
// disturbs, and a missed wake path is material that stops responding - much
// worse than a slow tick. There is exactly ONE way to wake, `World::wake_region`
// (brick + its six movable face neighbours), and these are all of its callers:
//
//   * `World::set_voxel`  - player edits, explosion spheres, edit replay on
//                           chunk reload. Covers "the plug under the lake was
//                           pulled" and "someone dropped sand in".
//   * `World::apply_slot_data` - a chunk streaming in, plus the settled bricks
//                           it now borders.
//   * `retire_and_wake`   - everything the CA itself disturbed: a neighbour
//                           draining, sand landing in water, smoke arriving.
//
// `physics::tests` has one test per path, and `flowing_water_is_unchanged_by_sleeping`
// pins the evolution against the pre-sleep behaviour (reproduced exactly by
// `rebuild_active_bricks()` before every tick, which is what the old awake set
// was) so a wake path that is merely LATE shows up as a diff, not as a look.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::voxel::*;

static FRAME_COUNTER: AtomicU64 = AtomicU64::new(0);

const BOTTOM_LAYER: u64 = 0xFFFF;
const TOP_LAYER: u64 = 0xFFFF << 48;

pub fn tick(world: &mut World) {
    let frame = FRAME_COUNTER.fetch_add(1, Ordering::Relaxed);
    // ONE snapshot of the awake set for all three rules. Taken out of the world
    // so it can be iterated while the world is mutated; reusable scratch, so a
    // tick allocates nothing. `touched` is the per-tick CHANGED-brick set, and
    // `world.phys_wake` collects bricks that must stay awake for a reason other
    // than having changed (see `step_brick_smoke`).
    let mut active = std::mem::take(&mut world.phys_scratch);
    active.clear();
    active.extend_from_slice(&world.active_bricks);
    active.sort_by_key(|&bi| (bi / WORLD_BRICKS_X) % WORLD_BRICKS_Y);
    let mut touched = std::mem::take(&mut world.phys_touched);
    touched.clear();
    world.phys_wake.clear();

    // 1. Sand gravity (bitmask, multi-pass), bottom-up.
    for &bi in &active {
        if world.movable_mask[bi as usize] == 0 { continue; }
        let (bx, by, bz) = brick_coords(bi);
        step_brick_sand_fall(world, bx, by, bz, &mut touched);
    }
    // 2. Water — BOTTOM-UP brick order. Top-down used to interact badly with
    // the per-brick pass-2 refill: bottom bricks would pull from the top
    // brick's bottom row AGAIN after the top brick's pass 2 already filled
    // it, draining a 1-cell gap. Bottom-up means each brick sees the FINAL
    // state of the brick above (which has already finished its passes).
    for &bi in &active {
        if world.movable_mask[bi as usize] == 0 { continue; }
        step_brick_water(world, bi, &mut touched);
    }
    // 3. Smoke — rises top-down. Re-sort the same buffer (reverse Y) rather than
    // cloning a second list.
    active.sort_by_key(|&bi| std::cmp::Reverse((bi / WORLD_BRICKS_X) % WORLD_BRICKS_Y));
    for &bi in &active {
        if world.movable_mask[bi as usize] == 0 { continue; }
        step_brick_smoke(world, bi, frame, &mut touched);
    }
    // Settle sand on bricks the fluids touched.
    settle_sand(world, &mut touched);
    // Retire everything that settled; wake the ring around everything that moved.
    retire_and_wake(world, &mut touched);
    // Return the scratch buffers for reuse next tick.
    world.phys_scratch = active;
    world.phys_touched = touched;
}

/// Rebuild the awake set from the bricks this tick CHANGED.
///
/// awake' = { changed } + { movable face neighbours of changed } + phys_wake
///
/// Everything else that was awake moved nothing and borders nothing that moved,
/// so it is settled and drops out - that is the whole win. Built as a fresh
/// sorted list rather than by removing from the old one, because retiring a
/// settled lake removes tens of thousands of entries and each `Vec::remove` is
/// an O(n) memmove.
fn retire_and_wake(world: &mut World, touched: &mut Vec<u32>) {
    touched.sort_unstable();
    touched.dedup();
    let mut next = std::mem::take(&mut world.phys_wake);
    for &bi in touched.iter() {
        if world.movable_mask[bi as usize] != 0 { next.push(bi); }
        for nb in World::face_neighbours(bi).into_iter().flatten() {
            if world.movable_mask[nb as usize] != 0 { next.push(nb); }
        }
    }
    next.sort_unstable();
    next.dedup();
    // The old awake list becomes next tick's scratch, so this swap allocates
    // nothing in the steady state.
    std::mem::swap(&mut world.active_bricks, &mut next);
    next.clear();
    world.phys_wake = next;
}

// ---------------- sand gravity ----------------

fn settle_sand(world: &mut World, touched: &mut Vec<u32>) {
    if touched.is_empty() { return; }
    // A WORKLIST of its own: `touched` is the tick's changed set and the caller
    // still needs it, so the settle iteration must not consume it.
    let mut work = std::mem::take(&mut world.phys_work);
    work.clear();
    work.extend_from_slice(touched);
    work.sort_unstable();
    work.dedup();
    work.sort_by_key(|&bi| (bi / WORLD_BRICKS_X) % WORLD_BRICKS_Y);
    let mut visited: Vec<u32> = Vec::with_capacity(work.len() * 2);
    let mut iter = 0;
    while !work.is_empty() && iter < 8 {
        iter += 1;
        let batch = std::mem::take(&mut work);
        for bi in batch {
            if visited.contains(&bi) { continue; }
            visited.push(bi);
            if world.movable_mask[bi as usize] == 0 { continue; }
            let (bx, by, bz) = brick_coords(bi);
            step_brick_sand_fall(world, bx, by, bz, touched);
            if by > 0 { work.push(brick_idx(bx, by - 1, bz)); }
        }
    }
    world.phys_work = work;
}

fn sand_mask(b: &Brick) -> u64 {
    let mut m = 0u64;
    for i in 0..64usize {
        m |= ((b.materials[i] == MAT_SAND) as u64) << i;
    }
    m & b.occupancy
}

fn step_brick_sand_fall(world: &mut World, bx: u32, by: u32, bz: u32, touched: &mut Vec<u32>) {
    let bi = brick_idx(bx, by, bz);
    if world.movable_mask[bi as usize] == 0 { return; }

    // Multi-pass intra-fall so a 4-deep stack of floaters collapses fully
    // in one tick.
    for _ in 0..3 {
        let occ = world.bricks[bi as usize].occupancy;
        let sand = sand_mask(&world.bricks[bi as usize]);
        if sand == 0 { break; }
        let empty = !occ;
        let falling = sand & (empty << 16);
        if falling == 0 { break; }
        let b = &mut world.bricks[bi as usize];
        b.occupancy ^= falling | (falling >> 16);
        let mut bits = falling;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            b.materials[i - 16] = MAT_SAND;
            b.materials[i] = MAT_AIR;
            bits &= bits - 1;
        }
        world.movable_mask[bi as usize] ^= falling | (falling >> 16);
        world.mark_brick_dirty(bi);
        touched.push(bi);
    }

    // Cross-brick fall.
    if by == 0 { return; }
    let cur_occ = world.bricks[bi as usize].occupancy;
    let sand = sand_mask(&world.bricks[bi as usize]);
    let bottom_sand = sand & BOTTOM_LAYER;
    if bottom_sand == 0 { return; }
    let below_bi = brick_idx(bx, by - 1, bz);
    let below_occ = world.bricks[below_bi as usize].occupancy;
    let below_top_empty = !below_occ & TOP_LAYER;
    let cross = bottom_sand & (below_top_empty >> 48);
    if cross == 0 { return; }

    let mut moves: [u32; 16] = [0; 16];
    let mut count = 0usize;
    let mut bits = cross;
    while bits != 0 {
        moves[count] = bits.trailing_zeros();
        count += 1;
        bits &= bits - 1;
    }

    let b = &mut world.bricks[bi as usize];
    b.occupancy ^= cross;
    for k in 0..count { b.materials[moves[k] as usize] = MAT_AIR; }
    let cur_now_empty = b.occupancy == 0;
    world.movable_mask[bi as usize] &= !cross;

    let was_empty = below_occ == 0;
    let b2 = &mut world.bricks[below_bi as usize];
    b2.occupancy |= cross << 48;
    for k in 0..count { b2.materials[(moves[k] + 48) as usize] = MAT_SAND; }
    world.movable_mask[below_bi as usize] |= cross << 48;
    if cur_now_empty && cur_occ != 0 { world.refresh_masks_for_brick(bx, by, bz); }
    if was_empty { world.refresh_masks_for_brick(bx, by - 1, bz); }
    world.mark_brick_dirty(bi);
    world.mark_brick_dirty(below_bi);
    // WAKE: both ends changed. `retire_and_wake` turns these into next tick's
    // awake set - which is how sand landing in a settled lake wakes the lake.
    touched.push(bi);
    touched.push(below_bi);
}

// ---------------- water (DwarfCorp-style level flow) ----------------

fn step_brick_water(world: &mut World, bi: u32, touched: &mut Vec<u32>) {
    // ----------------------------------------------------------------
    // Two-pass water physics:
    //
    //   Pass 1 (gravity, top-down)  — each water cell donates ALL of its
    //     level downward. The source becomes empty when its target had
    //     enough space.
    //   Pass 2 (refill, bottom-up)  — each empty/non-full cell pulls from
    //     the cell above to refill itself. The "drain" propagates UP the
    //     column instead of leaving a half-empty middle cell.
    //   Pass 3 (lateral)            — same as DwarfCorp: one level to a
    //     strictly-lower neighbour, each side gets a turn.
    //
    // Net effect: a lake draining into a cave loses ONE cell at the lake's
    // SURFACE per tick and gains one at the cave's TOP. Every cell along the
    // column stays full at L8 — no dangling L1 sliver mid-fall.
    // ----------------------------------------------------------------
    let (bx, by, bz) = brick_coords(bi);
    let snap_occ = world.bricks[bi as usize].occupancy;
    let snap_mats: [u8; 64] = world.bricks[bi as usize].materials;
    let snap_movable = world.movable_mask[bi as usize];

    let mut new_occ = snap_occ;
    let mut new_mats = snap_mats;
    let mut new_movable = snap_movable;
    let mut any_change = false;

    // ---------- PASS 1: GRAVITY (top-down) ----------
    // CRITICAL: read from a SNAPSHOT so a single drop doesn't cascade through
    // every empty cell below it in one tick. Without the snapshot, when y=21
    // donates to y=20, y=20 then processes and donates to y=19, etc — water
    // teleports through the column in one tick. With snapshot, only cells
    // that were ORIGINALLY water donate, so the column falls 1 cell/tick.
    let p1_occ = new_occ;
    let p1_mats = new_mats;
    for ly in (0u32..4).rev() {
        for lz in 0u32..4 {
            for lx in 0u32..4 {
                let i = (lx + lz * 4 + ly * 16) as i32;
                let bit = 1u64 << i;
                if (p1_occ & bit) == 0 { continue; }
                let mat = p1_mats[i as usize];
                if !is_water_mat(mat) { continue; }
                let level = water_level_of(mat) as i32;
                if level == 0 { continue; }

                if ly > 0 {
                    let below_i = i - 16;
                    let below_bit = 1u64 << below_i;
                    let below_solid_blocking = (p1_occ & below_bit) != 0
                        && !is_water_mat(p1_mats[below_i as usize]);
                    if below_solid_blocking { continue; }
                    // Use SNAPSHOT for below-level so we don't double-donate
                    // into the same cell from a snapshot+update interplay.
                    let below_level = if (p1_occ & below_bit) != 0 {
                        water_level_of(p1_mats[below_i as usize]) as i32
                    } else { 0 };
                    let space = MAX_WATER_LEVEL as i32 - below_level;
                    let transfer = level.min(space);
                    if transfer > 0 {
                        let nl = below_level + transfer;
                        new_mats[below_i as usize] = water_mat_for_level(nl as u8);
                        new_occ |= below_bit;
                        new_movable |= below_bit;
                        let src_new = level - transfer;
                        if src_new == 0 {
                            new_occ &= !bit;
                            new_movable &= !bit;
                            new_mats[i as usize] = MAT_AIR;
                        } else {
                            new_mats[i as usize] = water_mat_for_level(src_new as u8);
                        }
                        any_change = true;
                    }
                } else if by > 0 {
                    let below_bi = brick_idx(bx, by - 1, bz);
                    let below_i_in = (lx + lz * 4 + 3 * 16) as usize;
                    let below_bit = 1u64 << below_i_in;
                    let nb_occ = world.bricks[below_bi as usize].occupancy;
                    let nb_solid = (nb_occ & below_bit) != 0
                        && !is_water_mat(world.bricks[below_bi as usize].materials[below_i_in]);
                    if nb_solid { continue; }
                    let below_level = if (nb_occ & below_bit) != 0 {
                        water_level_of(world.bricks[below_bi as usize].materials[below_i_in]) as i32
                    } else { 0 };
                    let space = MAX_WATER_LEVEL as i32 - below_level;
                    let transfer = level.min(space);
                    if transfer > 0 {
                        cross_apply_water(world, below_bi, below_i_in, (below_level + transfer) as u8, touched);
                        let src_new = level - transfer;
                        if src_new == 0 {
                            new_occ &= !bit;
                            new_movable &= !bit;
                            new_mats[i as usize] = MAT_AIR;
                        } else {
                            new_mats[i as usize] = water_mat_for_level(src_new as u8);
                        }
                        any_change = true;
                    }
                }
            }
        }
    }

    // ---------- PASS 2: REFILL FROM ABOVE (bottom-up) ----------
    // CRITICAL: only refill cells that were WATER IN THE SNAPSHOT. If we
    // also refill cells that just received water in pass 1 (the falling
    // drop), we'd steal that water UP to fill the empty cell below,
    // creating a 1-cell gap right above the drop. By limiting pass 2 to
    // original sources, the drop is left alone and the source-above-it
    // refills from its own above. Net result: connected "beam".
    for ly in 0u32..4 {
        for lz in 0u32..4 {
            for lx in 0u32..4 {
                let i = (lx + lz * 4 + ly * 16) as i32;
                let bit = 1u64 << i;
                // Must have been a water source in the snapshot.
                if (p1_occ & bit) == 0 { continue; }
                let snap_mat = p1_mats[i as usize];
                if !is_water_mat(snap_mat) { continue; }

                let occupied = (new_occ & bit) != 0;
                let mat = if occupied { new_mats[i as usize] } else { MAT_AIR };
                if occupied && !is_water_mat(mat) { continue; }
                let level = if is_water_mat(mat) { water_level_of(mat) as i32 } else { 0 };
                if level >= MAX_WATER_LEVEL as i32 { continue; }
                let space = MAX_WATER_LEVEL as i32 - level;

                if ly < 3 {
                    let above_i = i + 16;
                    let above_bit = 1u64 << above_i;
                    let above_occ = (new_occ & above_bit) != 0;
                    let above_mat = if above_occ { new_mats[above_i as usize] } else { MAT_AIR };
                    if !is_water_mat(above_mat) { continue; }
                    let above_level = water_level_of(above_mat) as i32;
                    if above_level == 0 { continue; }
                    let transfer = above_level.min(space);
                    if transfer > 0 {
                        let new_level = level + transfer;
                        new_mats[i as usize] = water_mat_for_level(new_level as u8);
                        new_occ |= bit;
                        new_movable |= bit;
                        let above_new = above_level - transfer;
                        if above_new == 0 {
                            new_occ &= !above_bit;
                            new_movable &= !above_bit;
                            new_mats[above_i as usize] = MAT_AIR;
                        } else {
                            new_mats[above_i as usize] = water_mat_for_level(above_new as u8);
                        }
                        any_change = true;
                    }
                } else if by + 1 < WORLD_BRICKS_Y {
                    // Cross-brick: pull from bottom row of brick directly above.
                    let above_bi = brick_idx(bx, by + 1, bz);
                    let above_i_in = (lx + lz * 4) as usize;
                    let above_bit = 1u64 << above_i_in;
                    let nb_occ = world.bricks[above_bi as usize].occupancy;
                    let nb_mat = if (nb_occ & above_bit) != 0 {
                        world.bricks[above_bi as usize].materials[above_i_in]
                    } else { MAT_AIR };
                    if !is_water_mat(nb_mat) { continue; }
                    let above_level = water_level_of(nb_mat) as i32;
                    if above_level == 0 { continue; }
                    let transfer = above_level.min(space);
                    if transfer > 0 {
                        let new_level = level + transfer;
                        new_mats[i as usize] = water_mat_for_level(new_level as u8);
                        new_occ |= bit;
                        new_movable |= bit;
                        let above_new = above_level - transfer;
                        cross_apply_water(world, above_bi, above_i_in, above_new as u8, touched);
                        any_change = true;
                    }
                }
            }
        }
    }

    // ---------- PASS 3: LATERAL SPREAD ----------
    let mut bits = new_movable;
    while bits != 0 {
        let i = (63 - bits.leading_zeros()) as i32;
        bits &= !(1u64 << i);
        let mat = new_mats[i as usize];
        if !is_water_mat(mat) { continue; }
        let mut remaining = water_level_of(mat) as i32;
        if remaining <= 1 { continue; }

        let lx = (i & 3) as u32;
        let lz = ((i >> 2) & 3) as u32;
        let ly = ((i >> 4) & 3) as u32;
        let gx = bx * BRICK_DIM + lx;
        let gy = by * BRICK_DIM + ly;
        let gz = bz * BRICK_DIM + lz;
        let h = (gx.wrapping_mul(0x9E3779B1)
              ^ gy.wrapping_mul(0x85EBCA77)
              ^ gz.wrapping_mul(0xC2B2AE3D)) as usize;
        const DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
        for k in 0..4 {
            if remaining <= 1 { break; }
            let (dx, dz) = DIRS[(h + k) & 3];
            let tlx = lx as i32 + dx;
            let tlz = lz as i32 + dz;
            if tlx >= 0 && tlx < 4 && tlz >= 0 && tlz < 4 {
                let target = (tlx + tlz * 4 + ly as i32 * 16) as i32;
                let tbit = 1u64 << target;
                let occupied = (new_occ & tbit) != 0;
                let t_mat = new_mats[target as usize];
                if occupied && !is_water_mat(t_mat) { continue; }
                let t_level = if occupied { water_level_of(t_mat) as i32 } else { 0 };
                if remaining > t_level + 1 {
                    let nl = t_level + 1;
                    new_mats[target as usize] = water_mat_for_level(nl as u8);
                    new_occ |= tbit;
                    new_movable |= tbit;
                    remaining -= 1;
                    any_change = true;
                }
            } else {
                let nbx = bx as i32 + dx;
                let nbz = bz as i32 + dz;
                if nbx < 0 || nbx >= WORLD_BRICKS_X as i32
                || nbz < 0 || nbz >= WORLD_BRICKS_Z as i32 { continue; }
                let nbx = nbx as u32;
                let nbz = nbz as u32;
                let nb_lx = (tlx & 3) as u32;
                let nb_lz = (tlz & 3) as u32;
                let target = (nb_lx + nb_lz * 4 + ly * 16) as usize;
                let tbit = 1u64 << target;
                let nb_bi = brick_idx(nbx, by, nbz);
                let nb_occ = world.bricks[nb_bi as usize].occupancy;
                let occupied = (nb_occ & tbit) != 0;
                let t_mat = if occupied { world.bricks[nb_bi as usize].materials[target] } else { MAT_AIR };
                if occupied && !is_water_mat(t_mat) { continue; }
                let t_level = if occupied { water_level_of(t_mat) as i32 } else { 0 };
                if remaining > t_level + 1 {
                    cross_apply_water(world, nb_bi, target, (t_level + 1) as u8, touched);
                    remaining -= 1;
                }
            }
        }

        let new_level = remaining as u8;
        let new_self_mat = water_mat_for_level(new_level);
        if new_self_mat != mat {
            new_mats[i as usize] = new_self_mat;
            if new_level == 0 {
                new_occ &= !(1u64 << i);
                new_movable &= !(1u64 << i);
            }
            any_change = true;
        }
    }

    if any_change {
        let was_empty = snap_occ == 0;
        let now_empty = new_occ == 0;
        {
            let b = &mut world.bricks[bi as usize];
            b.occupancy = new_occ;
            b.materials = new_mats;
        }
        world.movable_mask[bi as usize] = new_movable;
        if was_empty != now_empty {
            world.refresh_masks_for_brick(bx, by, bz);
        }
        world.mark_brick_dirty(bi);
        touched.push(bi);
    }
    // NOTE there is deliberately no "retire when new_movable == 0" branch here
    // any more. That branch was the whole bug: it sat inside `if any_change`,
    // so a brick where nothing moved never reached it, and a FULL settled water
    // cell never empties, so it never fired even when reached. Retirement is now
    // `retire_and_wake`'s job and is driven by "did not change", not by "went
    // empty" - the two coincide for sand and are opposites for water.
}

// ---------------- smoke (rises, dissipates) ----------------

fn step_brick_smoke(world: &mut World, bi: u32, frame: u64, touched: &mut Vec<u32>) {
    let (bx, by, bz) = brick_coords(bi);

    let snap_occ = world.bricks[bi as usize].occupancy;
    let mut new_occ = snap_occ;
    let mut new_mats = world.bricks[bi as usize].materials;
    let mut new_movable = world.movable_mask[bi as usize];
    let mut any_change = false;
    let mut holds_smoke = false;

    // Walk movable bits — smoke is movable so it's in here.
    let mut bits = new_movable;
    while bits != 0 {
        let i = bits.trailing_zeros() as i32;
        bits &= bits - 1;
        let mat = new_mats[i as usize];
        if mat != MAT_SMOKE { continue; }
        holds_smoke = true;

        let lx = (i & 3) as u32;
        let lz = ((i >> 2) & 3) as u32;
        let ly = ((i >> 4) & 3) as u32;
        let gx = bx * BRICK_DIM + lx;
        let gy = by * BRICK_DIM + ly;
        let gz = bz * BRICK_DIM + lz;

        // Pseudorandom decision per (cell, frame).
        let h = (gx.wrapping_mul(0x9E3779B1)
              ^ gy.wrapping_mul(0x85EBCA77)
              ^ gz.wrapping_mul(0xC2B2AE3D)
              ^ (frame as u32).wrapping_mul(0xD2B74407)) as u32;

        // 1/40 chance to dissipate every tick — smoke fades out naturally.
        if (h % 40) == 0 {
            new_occ &= !(1u64 << i);
            new_movable &= !(1u64 << i);
            new_mats[i as usize] = MAT_AIR;
            any_change = true;
            continue;
        }

        // Try to rise intra-brick.
        if ly < 3 {
            let up_i = i + 16;
            let up_bit = 1u64 << up_i;
            if (new_occ & up_bit) == 0 {
                new_occ ^= (1u64 << i) | up_bit;
                new_movable ^= (1u64 << i) | up_bit;
                new_mats[up_i as usize] = MAT_SMOKE;
                new_mats[i as usize] = MAT_AIR;
                any_change = true;
                continue;
            }
        } else if by + 1 < WORLD_BRICKS_Y {
            // Cross-brick rise: top row of this brick → y=0 of brick above.
            let up_bi = brick_idx(bx, by + 1, bz);
            let up_i_in = (lx + lz * 4) as usize;
            let up_bit = 1u64 << up_i_in;
            let up_occ = world.bricks[up_bi as usize].occupancy;
            if (up_occ & up_bit) == 0 {
                let nb_was_empty = up_occ == 0;
                {
                    let nb = &mut world.bricks[up_bi as usize];
                    nb.occupancy |= up_bit;
                    nb.materials[up_i_in] = MAT_SMOKE;
                }
                world.movable_mask[up_bi as usize] |= up_bit;
                if nb_was_empty {
                    world.refresh_masks_for_brick(bx, by + 1, bz);
                }
                world.mark_brick_dirty(up_bi);
                touched.push(up_bi);
                new_occ &= !(1u64 << i);
                new_movable &= !(1u64 << i);
                new_mats[i as usize] = MAT_AIR;
                any_change = true;
                continue;
            }
        }

        // Can't rise — try lateral within brick.
        const DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
        for k in 0..4 {
            let (dx, dz) = DIRS[((h as usize) + k) & 3];
            let tlx = lx as i32 + dx;
            let tlz = lz as i32 + dz;
            if tlx < 0 || tlx > 3 || tlz < 0 || tlz > 3 { continue; }
            let target = tlx + tlz * 4 + ly as i32 * 16;
            let tbit = 1u64 << target;
            if (new_occ & tbit) == 0 {
                new_occ ^= (1u64 << i) | tbit;
                new_movable ^= (1u64 << i) | tbit;
                new_mats[target as usize] = MAT_SMOKE;
                new_mats[i as usize] = MAT_AIR;
                any_change = true;
                break;
            }
        }
    }

    if any_change {
        let was_empty = snap_occ == 0;
        let now_empty = new_occ == 0;
        {
            let b = &mut world.bricks[bi as usize];
            b.occupancy = new_occ;
            b.materials = new_mats;
        }
        world.movable_mask[bi as usize] = new_movable;
        if was_empty != now_empty {
            world.refresh_masks_for_brick(bx, by, bz);
        }
        world.mark_brick_dirty(bi);
        touched.push(bi);
    }
    // Smoke is the ONE rule whose outcome depends on the tick counter and not
    // only on the world: sealed smoke that cannot rise or spread still has a
    // 1/40 chance to dissipate each tick. "Nothing moved" therefore does not
    // mean "settled" here, so a brick that still holds smoke stays awake
    // WITHOUT waking its ring (nothing around it was disturbed). Retiring it
    // would make walled-in smoke immortal.
    if holds_smoke {
        world.phys_wake.push(bi);
    }
}

/// Apply a single cross-brick water set to (nb_bi, voxel index in that brick,
/// new level). Updates occupancy, materials and movable_mask, and emits the
/// dirty/refresh/touched signals.
///
/// `touched` is what wakes the neighbour: pushing `nb_bi` here is the wake path
/// for "a neighbour drained into me" and "my supply above drained", and
/// `retire_and_wake` turns it into next tick's awake set together with its own
/// ring. Nothing here writes `active_bricks` directly.
fn cross_apply_water(
    world: &mut World,
    nb_bi: u32,
    nb_vi: usize,
    new_level: u8,
    touched: &mut Vec<u32>,
) {
    let nb_bit = 1u64 << nb_vi;
    let nb_was_empty = world.bricks[nb_bi as usize].occupancy == 0;
    {
        let b = &mut world.bricks[nb_bi as usize];
        if new_level == 0 {
            b.occupancy &= !nb_bit;
            b.materials[nb_vi] = MAT_AIR;
        } else {
            b.occupancy |= nb_bit;
            b.materials[nb_vi] = water_mat_for_level(new_level);
        }
    }
    if new_level == 0 {
        world.movable_mask[nb_bi as usize] &= !nb_bit;
    } else {
        world.movable_mask[nb_bi as usize] |= nb_bit;
    }
    let nb_now_empty = world.bricks[nb_bi as usize].occupancy == 0;
    if nb_was_empty != nb_now_empty {
        let (nbx, nby, nbz) = brick_coords(nb_bi);
        world.refresh_masks_for_brick(nbx, nby, nbz);
    }
    world.mark_brick_dirty(nb_bi);
    touched.push(nb_bi);
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    // Every crafted scene is built well inside the window, so brick-boundary
    // logic runs for real and no window-edge clamp is exercised by accident.
    const OX: u32 = 216;
    const OY: u32 = 200;
    const OZ: u32 = 200;
    const INNER: u32 = 8;

    fn fill(w: &mut World, x0: u32, y0: u32, z0: u32, x1: u32, y1: u32, z1: u32, mat: u8) {
        for y in y0..y1 {
            for z in z0..z1 {
                for x in x0..x1 {
                    w.set_voxel(x, y, z, mat);
                }
            }
        }
    }

    /// A sealed stone basin holding `depth` layers of FULL (L8) water.
    ///
    /// L8 over L8 with stone at every wall has no legal transfer in any of the
    /// three water passes, so this is the shape the demo window's 23.5% water
    /// actually is: settled, and previously re-evaluated 30 times a second
    /// forever.
    fn settled_lake(depth: u32) -> World {
        let mut w = World::new();
        let top = OY + depth;
        // Floor, then the four walls up past the waterline. No lid: air above
        // still water is not a disturbance.
        fill(&mut w, OX - 1, OY - 1, OZ - 1, OX + INNER + 1, OY, OZ + INNER + 1, MAT_STONE);
        fill(&mut w, OX - 1, OY, OZ - 1, OX, top + 1, OZ + INNER + 1, MAT_STONE);
        fill(&mut w, OX + INNER, OY, OZ - 1, OX + INNER + 1, top + 1, OZ + INNER + 1, MAT_STONE);
        fill(&mut w, OX, OY, OZ - 1, OX + INNER, top + 1, OZ, MAT_STONE);
        fill(&mut w, OX, OY, OZ + INNER, OX + INNER, top + 1, OZ + INNER + 1, MAT_STONE);
        fill(&mut w, OX, OY, OZ, OX + INNER, top, OZ + INNER, MAT_WATER_L8);
        w
    }

    /// Total water LEVEL over a generous box around the crafted scenes. Water is
    /// mass and every rule here is a transfer, so this is invariant except where
    /// a scene deliberately drains off the bottom.
    fn water_volume(w: &World) -> u64 {
        let mut sum = 0u64;
        for y in OY - 40..OY + 40 {
            for z in OZ - 40..OZ + 40 {
                for x in OX - 40..OX + 60 {
                    sum += water_level_of(w.material_at_world(x as i32, y as i32, z as i32)) as u64;
                }
            }
        }
        sum
    }

    /// Order-sensitive digest of every voxel in the crafted region. Lets two
    /// evolutions be compared voxel for voxel without ever holding two 1.84 GB
    /// worlds at the same time.
    fn digest(w: &World) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        for y in OY - 40..OY + 40 {
            for z in OZ - 40..OZ + 40 {
                for x in OX - 40..OX + 60 {
                    let m = w.material_at_world(x as i32, y as i32, z as i32) as u64;
                    h = (h ^ m).wrapping_mul(0x100_0000_01b3);
                }
            }
        }
        h
    }

    fn count_water(w: &World, x0: u32, x1: u32) -> u32 {
        let mut n = 0;
        for y in OY - 20..OY + 20 {
            for z in OZ - 20..OZ + 20 {
                for x in x0..x1 {
                    if is_water_mat(w.material_at_world(x as i32, y as i32, z as i32)) {
                        n += 1;
                    }
                }
            }
        }
        n
    }

    // ------------------------------------------------------------------
    // A. Settled water sleeps.
    // ------------------------------------------------------------------

    /// THE BUG THIS PINS. A water cell used to clear its movable bit only in the
    /// `new_level == 0` branch - only when it drained to NOTHING - and the
    /// retire path sat inside `if any_change`, so a brick where nothing moved
    /// never reached it. Settled water could therefore never sleep, and a full
    /// settled cell never empties, so its bit never cleared either. Sand only
    /// looked correct because for sand "settled" and "empty" coincide; for water
    /// they are opposites.
    #[test]
    fn a_settled_lake_sleeps_and_stays_a_lake() {
        let mut w = settled_lake(4);
        let before = water_volume(&w);
        assert!(before > 0, "the fixture has to actually hold water");
        assert!(!w.active_bricks.is_empty(), "a freshly built lake must start awake");

        tick(&mut w);
        assert_eq!(
            w.active_bricks.len(),
            0,
            "a lake that cannot move anywhere must retire on the very first tick"
        );

        // And it must STAY asleep and STAY a lake.
        for _ in 0..64 {
            tick(&mut w);
            assert_eq!(w.active_bricks.len(), 0);
        }
        assert_eq!(water_volume(&w), before, "a sleeping lake must not lose mass");
    }

    // ------------------------------------------------------------------
    // Wake paths. One test per way a settled region can be disturbed. A
    // missed wake path is a lake that stops responding, which is far worse
    // than a slow one, so every one of these has to be able to FAIL.
    // ------------------------------------------------------------------

    /// WAKE PATH: a player edit (`World::set_voxel`) under settled water.
    /// Nothing in the lake's own bricks changes when the plug is pulled, so
    /// without the ring wake in `set_voxel` the lake never notices.
    #[test]
    fn a_player_edit_under_a_settled_lake_wakes_it() {
        let mut w = settled_lake(4);
        tick(&mut w);
        assert_eq!(w.active_bricks.len(), 0);

        // Drill the floor out from under the middle of the lake.
        w.set_voxel(OX + 4, OY - 1, OZ + 4, MAT_AIR);
        assert!(!w.active_bricks.is_empty(), "the edit must wake the lake");

        let above_before = count_water(&w, OX, OX + INNER);
        for _ in 0..40 {
            tick(&mut w);
        }
        assert!(
            count_water(&w, OX, OX + INNER) < above_before,
            "water must drain through the hole the edit opened"
        );
    }

    /// WAKE PATH: an explosion sphere (`app::apply_sphere` -> `apply_edit` ->
    /// `set_voxel`) - the same primitive reached through the gameplay path
    /// rather than directly.
    #[test]
    fn an_explosion_sphere_wakes_settled_water() {
        let mut w = settled_lake(4);
        tick(&mut w);
        assert_eq!(w.active_bricks.len(), 0);

        // Blow the +x wall away at the waterline.
        crate::app::apply_sphere(
            &mut w,
            (OX + INNER) as i32,
            (OY + 1) as i32,
            (OZ + 4) as i32,
            3,
            MAT_AIR,
        );
        assert!(!w.active_bricks.is_empty(), "the blast must wake the water it exposed");

        for _ in 0..40 {
            tick(&mut w);
        }
        assert!(
            count_water(&w, OX + INNER + 1, OX + INNER + 12) > 0,
            "water must pour out through the breach"
        );
    }

    /// WAKE PATH: sand landing in settled water. The lake's own cells do not
    /// change when the grain is placed three cells above the surface; the sand's
    /// cross-brick fall is what has to report the landing brick as touched.
    #[test]
    fn sand_landing_in_a_settled_lake_wakes_it() {
        let mut w = settled_lake(4);
        tick(&mut w);
        assert_eq!(w.active_bricks.len(), 0);

        let sy = OY + 4 + 3;
        w.set_voxel(OX + 4, sy, OZ + 4, MAT_SAND);
        for _ in 0..16 {
            tick(&mut w);
        }
        let mut landed = false;
        for y in OY..OY + 5 {
            if w.material_at_world((OX + 4) as i32, y as i32, (OZ + 4) as i32) == MAT_SAND {
                landed = true;
            }
        }
        assert!(landed, "sand must fall INTO the settled lake, not rest on top of it");
    }

    /// WAKE PATH: a neighbour draining. The far end of a long channel is ten
    /// bricks from the hole and is asleep when the hole opens; the drain has to
    /// travel to it one brick-ring per tick, which is faster than water moves
    /// (one CELL per tick), so the front can never outrun the wake.
    #[test]
    fn a_drain_at_one_end_propagates_across_a_settled_lake() {
        let mut w = World::new();
        let (x0, x1) = (OX, OX + 40);
        fill(&mut w, x0 - 1, OY - 1, OZ - 1, x1 + 1, OY, OZ + 5, MAT_STONE);
        fill(&mut w, x0 - 1, OY, OZ - 1, x0, OY + 3, OZ + 5, MAT_STONE);
        fill(&mut w, x1, OY, OZ - 1, x1 + 1, OY + 3, OZ + 5, MAT_STONE);
        fill(&mut w, x0, OY, OZ - 1, x1, OY + 3, OZ, MAT_STONE);
        fill(&mut w, x0, OY, OZ + 4, x1, OY + 3, OZ + 5, MAT_STONE);
        fill(&mut w, x0, OY, OZ, x1, OY + 2, OZ + 4, MAT_WATER_L8);
        for _ in 0..8 {
            tick(&mut w);
        }
        assert_eq!(w.active_bricks.len(), 0, "the channel must settle first");

        let far_before = count_water(&w, x1 - 4, x1);
        assert!(far_before > 0);
        // Open the floor at the NEAR end only.
        w.set_voxel(x0, OY - 1, OZ + 1, MAT_AIR);
        for _ in 0..400 {
            tick(&mut w);
        }
        assert!(
            count_water(&w, x1 - 4, x1) < far_before,
            "the drain must reach the far end of the channel, ten bricks away"
        );
    }

    /// WAKE PATH: a chunk streaming in. `apply_slot_data` replaces a slot's
    /// bricks wholesale, and the water it borders has nothing of its own to
    /// notice.
    #[test]
    fn a_streamed_in_slot_wakes_the_water_it_borders() {
        let mut w = settled_lake(4);
        tick(&mut w);
        assert_eq!(w.active_bricks.len(), 0);

        // The lake's +x wall sits at x = OX + INNER, which is the first voxel of
        // storage slot 7 in x. Streaming that slot in as open sky is exactly
        // "the terrain next to the lake was replaced".
        assert_eq!(OX + INNER, 7 * STORAGE_CHUNK_VOXELS);
        let n = (STORAGE_CHUNK_BRICKS * STORAGE_CHUNK_BRICKS * STORAGE_CHUNK_BRICKS) as usize;
        let data = SlotData::from_bricks(vec![Brick::EMPTY; n]);
        let (scy, scz) = (OY / STORAGE_CHUNK_VOXELS, OZ / STORAGE_CHUNK_VOXELS);
        w.apply_slot_data(7, scy, scz, &data);
        assert!(
            !w.active_bricks.is_empty(),
            "installing a slot must wake the settled water it now borders"
        );

        for _ in 0..40 {
            tick(&mut w);
        }
        assert!(
            count_water(&w, OX + INNER, OX + INNER + 8) > 0,
            "water must flow into the slot that streamed in as sky"
        );
    }

    /// WAKE PATH / TIME DEPENDENCE: smoke sealed in a stone pocket cannot rise
    /// and cannot spread, so "nothing moved" is true every tick until its 1/40
    /// dissipation roll comes up. Retiring on "nothing moved" alone would make
    /// walled-in smoke immortal, which is why `step_brick_smoke` keeps its own
    /// brick awake without waking its ring.
    #[test]
    fn sealed_smoke_still_dissipates() {
        let mut w = World::new();
        fill(&mut w, OX - 1, OY - 1, OZ - 1, OX + 2, OY + 2, OZ + 2, MAT_STONE);
        w.set_voxel(OX, OY, OZ, MAT_SMOKE);
        let mut gone = false;
        for _ in 0..2000 {
            tick(&mut w);
            if w.material_at_world(OX as i32, OY as i32, OZ as i32) != MAT_SMOKE {
                gone = true;
                break;
            }
            assert!(
                !w.active_bricks.is_empty(),
                "a brick that still holds smoke must stay awake"
            );
        }
        assert!(gone, "sealed smoke must still dissipate");
    }

    /// Sleeping must change WHICH bricks are visited and nothing else.
    ///
    /// The reference is the pre-sleep behaviour reproduced exactly, with no
    /// second implementation to drift: the old awake set was "every brick whose
    /// movable_mask is non-zero", which is precisely what
    /// `rebuild_active_bricks()` computes, so calling it before every tick runs
    /// the old visiting set through the current rules. If a wake path is missing
    /// or merely LATE, the two evolutions diverge and the digests differ - the
    /// failure mode this exists to catch.
    #[test]
    fn flowing_water_is_unchanged_by_sleeping() {
        // A scene in motion for the whole run: a full basin draining through two
        // floor holes into open air, with a sand column collapsing into it from
        // above. No smoke: smoke folds a PROCESS-wide tick counter into its
        // hash, so two runs in one process could never be compared.
        fn scene() -> World {
            let mut w = settled_lake(4);
            w.set_voxel(OX + 2, OY - 1, OZ + 2, MAT_AIR);
            w.set_voxel(OX + 5, OY - 1, OZ + 5, MAT_AIR);
            for k in 0..6 {
                w.set_voxel(OX + 3, OY + 6 + k, OZ + 3, MAT_SAND);
            }
            w
        }
        const TICKS: u32 = 60;

        let mut reference = scene();
        for _ in 0..TICKS {
            reference.rebuild_active_bricks();
            tick(&mut reference);
        }
        let (want, want_vol) = (digest(&reference), water_volume(&reference));
        drop(reference);

        let mut sleeping = scene();
        for _ in 0..TICKS {
            tick(&mut sleeping);
        }
        assert_eq!(digest(&sleeping), want, "sleeping changed the evolution of flowing water");
        assert_eq!(water_volume(&sleeping), want_vol);
    }

    /// THE MEASUREMENT, on the real demo window (23.5% water by area).
    /// `cargo test --lib physics::tests::settled_world_tick_cost -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn settled_world_tick_cost() {
        let mut w = World::new();
        w.fill_demo_terrain();
        eprintln!("awake bricks at load: {}", w.active_bricks.len());
        let mut first = 0.0;
        for i in 0..90u32 {
            let t = Instant::now();
            tick(&mut w);
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            if i == 0 {
                first = ms;
            }
            if i < 6 || i % 10 == 0 || i == 89 {
                eprintln!("tick {i:3}: {ms:9.3} ms   awake {}", w.active_bricks.len());
            }
        }
        let mut samples: Vec<f64> = Vec::new();
        for _ in 0..60 {
            let t = Instant::now();
            tick(&mut w);
            samples.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        eprintln!(
            "settled: mean {:.4} ms  p50 {:.4}  max {:.4}  awake {}  (first tick {:.1} ms)",
            mean,
            samples[samples.len() / 2],
            samples[samples.len() - 1],
            w.active_bricks.len(),
            first
        );
    }
}
