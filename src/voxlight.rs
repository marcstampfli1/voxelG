// Per-voxel light field: the sparse block pool.
//
// Design and rationale: docs/VOXEL_LIGHTING_PLAN.md.
//
// Light is a property of AIR voxels that touch solid geometry (the "lit
// shell"), not of pixels. Storing it for all 33.5M world voxels would cost
// 256 MB, but the lit shell is a thin surface: the overwhelming majority of
// bricks are either open sky or solid interior and need nothing at all. So
// storage is a POOL of fixed-size blocks, one block per brick that actually
// carries lit-shell voxels, indexed through a brick -> block table.
//
// This module owns only the ALLOCATION. It deliberately holds no light data:
// the records themselves live in a GPU buffer written by the update compute
// pass and never read back. Keeping the allocator CPU-side and the payload
// GPU-side is what lets a block be freed or invalidated in O(1) without
// touching (or moving) 512 bytes of GPU memory.
//
// Capacity is a CONSTRUCTOR ARGUMENT rather than a global constant read inside
// `allocate`: a field must never be able to hand out a block index past the end
// of the GPU pool it was handed. (This allocator briefly served a second field,
// the per-voxel reflected-radiance cache, which is why the capacity is
// per-instance; that field is gone, the reason to keep the invariant is not.)

use crate::world_dims::WORLD_BRICKS_TOTAL;

pub use crate::world_dims::LIGHT_URGENT_BUDGET;

/// Light records per block, and the voxel edge of one record's cell.
///
/// A record covers a LIGHT_RECORD_STEP^3 group of voxels, not a single voxel:
/// at 10 cm that is a 20 cm light field, FINER than the 25 cm one the per-voxel
/// version shipped, for 1/8 the storage and 1/8 the sun rays per sweep. The
/// values live in `world_dims.rs` because `build.rs` has to emit them into the
/// shader prelude - the update pass and the sampler both address the pool with
/// them, so a second copy could drift.
pub use crate::world_dims::{
    LIGHT_BLOCK_WORDS, LIGHT_RECORDS_PER_BLOCK, LIGHT_RECORD_DIM, LIGHT_RECORD_STEP,
    LIGHT_RECORD_WORDS,
};

/// Index of the record covering in-brick voxel (lx, ly, lz).
///
/// Mirrors `brick_voxel_idx`'s x + z*D + y*D^2 ordering one level up, so the
/// shader's inverse (`cs_voxel_light_update`) and this stay readable as the same
/// linearisation. FORBIDDEN: open-coding this at a call site.
#[inline(always)]
pub const fn light_record_idx(lx: u32, ly: u32, lz: u32) -> u32 {
    let d = LIGHT_RECORD_DIM;
    (lx / LIGHT_RECORD_STEP) + (lz / LIGHT_RECORD_STEP) * d + (ly / LIGHT_RECORD_STEP) * d * d
}

/// Resident light blocks.
///
/// MEASURED, no longer estimated. The demo world (a fully generated 512x256x512
/// streamed window, `World::fill_demo_terrain`) has 1,048,576 bricks: 699,991
/// empty, 38,040 partially solid, 310,545 fully solid. `brick_needs_light`
/// binds a block for every brick that can hold an air voxel next to solid,
/// which on that world is 60,174 blocks = 30.8 MB of the 67.1 MB pool, so the
/// whole streamed window is covered with 2.18x headroom and NOTHING falls back
/// to the per-pixel path for want of storage.
///
/// The 65k-130k estimate this ceiling was originally derived from was right for
/// that rule. What was wrong was the rule: it also bound a block for every
/// FULLY SOLID brick, so the demo world asked for 370,719 and 239,647 requests
/// were refused. Because `allocate` fills blocks in brick-index order and
/// `brick_idx` is z-major, that did not degrade evenly - it covered a solid
/// slab over world z 0..192 of 512 and left every camera past it on the old
/// per-pixel path. The fix was to stop binding storage that nothing can read
/// (see `World::brick_needs_light`), not to buy a 192 MiB pool to hold it.
/// Both were built and benchmarked; the numbers are in
/// docs/rt/BASELINE-per-voxel-lighting.md, round D.
///
/// Exhaustion still degrades gracefully rather than failing (see `allocate`),
/// and refusals are now reported at ERROR level with a running total
/// (`overflow_total`) so a world class that does outgrow this cannot saturate
/// unnoticed the way this one did.
///
/// The VALUE lives in `world_dims.rs` because the shader needs it: the urgent
/// list is appended to the GPU work list at this offset. This is the name the
/// rest of the engine uses and where the reasoning lives.
pub use crate::world_dims::LIGHT_BLOCKS_MAX;

/// u32 words of GPU storage backing the whole pool (128 MiB).
pub const LIGHT_POOL_WORDS: u32 = LIGHT_BLOCKS_MAX * LIGHT_BLOCK_WORDS;

/// `block_of_brick` entry meaning "this brick has no light block".
pub const LIGHT_BLOCK_NONE: u32 = u32::MAX;

/// Marks a block's accumulated estimate as invalid so the update pass restarts
/// it from scratch rather than folding new samples into stale light. Stored in
/// the record's `epoch` byte; the GPU side treats 0 as "no valid history".
pub const LIGHT_EPOCH_RESET: u32 = 0;

/// Table entries that may be swept into one `write_buffer` rather than split
/// into two, measured in brick indices (4 bytes each).
///
/// A `queue.write_buffer` costs a staging-belt allocation, a memcpy and a
/// recorded copy - order a microsecond - whichever way the bytes go, while
/// 1024 extra entries is 4 KiB of memcpy, under half a microsecond at any
/// plausible bandwidth. So merging across a gap this size is strictly cheaper
/// than emitting a second write, and the coalescer is told to do it.
pub const LIGHT_TABLE_UPLOAD_GAP: u32 = 1024;

#[derive(Copy, Clone)]
struct BlockSlot {
    /// Brick this block is bound to.
    brick: u32,
    /// Index of this block within `live`, so release is O(1).
    live_idx: u32,
}

/// Sparse allocator mapping bricks to fixed-size light blocks.
///
/// Invariants (upheld by every method, checked by `debug_assert_consistent`):
///  - `block_of_brick[b] == LIGHT_BLOCK_NONE` XOR `slots[block].brick == b`.
///  - `live` holds exactly the allocated block indices, with no duplicates.
///  - `live[slots[k].live_idx] == k` for every allocated block `k`.
///  - `free` and `live` are disjoint and together cover `0..slots.len()`.
pub struct LightField {
    /// Brick index -> block index, or `LIGHT_BLOCK_NONE`.
    block_of_brick: Vec<u32>,
    /// Per-block bookkeeping, indexed by block index.
    slots: Vec<BlockSlot>,
    /// BRICK indices of every allocated block, kept COMPACT so the update pass
    /// dispatches exactly as much work as there is lit shell, not the pool
    /// ceiling. Bricks rather than blocks: the shader needs the brick anyway to
    /// locate the voxels, and it can reach the block through `block_of_brick`,
    /// so storing bricks here removes a whole block -> brick GPU table.
    ///
    /// PARTITIONED: `live[..near_count]` is the shell close to the camera and
    /// `live[near_count..]` is everything else. See `near_count`.
    live: Vec<u32>,
    /// Length of the NEAR prefix of `live`.
    ///
    /// The SWEEP refreshes the near prefix on the normal cadence and the far
    /// remainder far more slowly. It is a REFRESH RATE and nothing else: every
    /// block keeps its storage and its converged record, so shading reads
    /// exactly the same field it read before and no pixel changes path because
    /// of this split.
    ///
    /// The sweep exists to track the SUN and nothing else. Everything that makes
    /// a record UNREADABLE - a newly bound block, an invalidated one, a
    /// neighbour whose occupancy just changed - goes on the URGENT list instead
    /// and is serviced on the next dispatch whatever tier it is in, so this
    /// partition can be as aggressive as the sun's tracking tolerance allows
    /// without ever leaving a surface without light. See `urgent`.
    near_count: usize,
    /// BRICK indices whose block must be re-gathered on the NEXT dispatch,
    /// whatever the sweep schedule says.
    ///
    /// THIS IS WHAT MAKES THE SWEEP SKIPPABLE. A record's value is a function of
    /// (geometry, sun direction, light set), so re-gathering it when none of the
    /// three moved writes bytes identical to the ones already there - measured on
    /// the demo world, 99.25% of records are bit-identical across a whole near
    /// sweep period of sun motion (`voxlight_sun_lag_error`). The sweep can
    /// therefore be paced by the sun rather than by the frame counter, and paused
    /// outright when the sun is still - but only if the cases that genuinely
    /// cannot wait have somewhere else to go. This is that somewhere.
    ///
    /// It replaces `promote_near`, which pulled the same blocks into the near
    /// TIER and left them there: promotion was one-way, so a still camera
    /// watching physics dragged the whole shell into the near group one block at
    /// a time, and even then a promoted block waited up to `update_div` rounds
    /// for its slice to come round. An urgent block is serviced on the very next
    /// dispatch and then leaves, so the tier boundary stops drifting and the
    /// worst-case latency after an edit goes from 8 frames to 1.
    ///
    /// Bricks, not blocks, because that is what the GPU work list holds. A stale
    /// entry (the brick was released, or rebound to another block) is SAFE: the
    /// shader re-reads `block_of_brick` and skips an unbound brick, and a rebound
    /// one needs the gather anyway.
    urgent: Vec<u32>,
    /// One bit per BRICK: membership of `urgent`. Brick-indexed rather than
    /// block-indexed so a release/rebind between queueing and draining cannot
    /// strand a bit on a block the brick no longer owns.
    urgent_queued: Vec<u64>,
    /// Block indices available for reuse.
    free: Vec<u32>,
    /// Blocks whose accumulation must restart (brick edited, slot recycled).
    /// Drained by the renderer each frame into a GPU upload.
    ///
    /// DEDUPED through `reset_queued`: physics dirties the same brick on
    /// consecutive frames all the time, and a block queued twice is a second
    /// 512-byte DMA that writes the same zeros.
    pending_reset: Vec<u32>,
    /// Bitset over BLOCK indices: membership of `pending_reset`.
    reset_queued: Vec<u64>,
    /// Allocation requests refused because the pool was full, since the last
    /// `take_overflow`. Surfaced rather than silently dropped.
    overflow: u32,
    /// Requests refused since the field was built, NEVER cleared.
    ///
    /// The per-frame counter above resets on every `take_overflow`, so a world
    /// that saturates once on its full-dirty init frame and then goes quiet
    /// reports zero for every frame after - which is exactly how this field ran
    /// saturated through several benchmark rounds without anyone noticing. The
    /// running total is the number that says "this world does not fit", and it
    /// is reported alongside the delta.
    overflow_total: u32,
    /// BRICK indices whose table entry changed since the last upload.
    ///
    /// The table is 4 MiB of one u32 per brick and used to go WHOLE on any
    /// binding change. In a live world that is most frames - flowing water flips
    /// bricks between empty and non-empty, streaming recycles slots - and it
    /// measured 278 MB/s of PCIe on the shipped benchmark to move a few kilobytes
    /// of real change. Recording WHICH entries moved lets the upload push only
    /// the runs that did.
    table_touched: Vec<u32>,
    /// Set when the compact work list changed in contents OR ORDER. Separate
    /// from `table_touched` because the near/far repartition reorders the list
    /// without touching a single table entry, and the two are 240 KiB and 4 MiB
    /// respectively - conflating them would re-push the big one for free.
    list_dirty: bool,
    /// Blocks in the GPU pool backing this field. Held per instance rather than
    /// read from a constant, so a field can never allocate past the buffer it
    /// was actually handed.
    blocks_max: u32,
}

impl LightField {
    /// `blocks_max` must match the block count of the GPU pool this field
    /// indexes (`LIGHT_BLOCKS_MAX`); it is the only thing stopping `allocate`
    /// from handing out an out-of-range block index.
    pub fn new(blocks_max: u32) -> Self {
        Self {
            block_of_brick: vec![LIGHT_BLOCK_NONE; WORLD_BRICKS_TOTAL as usize],
            slots: Vec::new(),
            live: Vec::new(),
            near_count: 0,
            urgent: Vec::new(),
            urgent_queued: vec![0u64; (WORLD_BRICKS_TOTAL as usize).div_ceil(64)],
            free: Vec::new(),
            pending_reset: Vec::new(),
            reset_queued: vec![0u64; (blocks_max as usize).div_ceil(64)],
            overflow: 0,
            overflow_total: 0,
            // The GPU table starts all-NONE and the pool starts zeroed
            // (epoch 0 = invalid), which is exactly the empty state, so the
            // first frame has nothing to re-upload.
            table_touched: Vec::new(),
            list_dirty: false,
            blocks_max,
        }
    }

    /// Pool ceiling this field was built for. The renderer reports it when the
    /// pool overflows, so the message names the field's own cap.
    #[inline]
    pub fn blocks_max(&self) -> u32 {
        self.blocks_max
    }

    /// The brick indices whose table entry changed since the last upload
    /// (sorted, deduped) together with the table itself, so the caller can push
    /// only the runs that moved.
    ///
    /// Both are returned from ONE call because the caller needs them together
    /// and they are two borrows of the same field; sorting in place here rather
    /// than making the caller do it keeps the "sorted and deduped" precondition
    /// of the span coalescer with the data that has to satisfy it.
    /// `clear_table_touched` acknowledges the upload.
    pub fn table_delta(&mut self) -> (&[u32], &[u32]) {
        self.table_touched.sort_unstable();
        self.table_touched.dedup();
        (&self.table_touched, &self.block_of_brick)
    }

    /// Acknowledge that `table_delta`'s runs have been pushed.
    pub fn clear_table_touched(&mut self) {
        self.table_touched.clear();
    }

    /// Whether the compact work list changed in contents or order, and clear.
    pub fn take_list_dirty(&mut self) -> bool {
        std::mem::replace(&mut self.list_dirty, false)
    }

    /// Block bound to `brick`, if any.
    #[inline]
    pub fn block_of(&self, brick: u32) -> Option<u32> {
        match self.block_of_brick[brick as usize] {
            LIGHT_BLOCK_NONE => None,
            b => Some(b),
        }
    }

    /// The compact list of bricks holding a light block. This is the update
    /// pass's work list: its length is the dispatch size.
    #[inline]
    pub fn live_bricks(&self) -> &[u32] {
        &self.live
    }

    /// Length of the NEAR prefix of the work list (see `near_count`).
    #[inline]
    pub fn near_count(&self) -> u32 {
        self.near_count as u32
    }

    /// Bricks awaiting an urgent re-gather, oldest first.
    #[inline]
    pub fn urgent(&self) -> &[u32] {
        &self.urgent
    }

    /// Forget every queued urgent visit.
    ///
    /// For the whole-world rebind only (`World::sync_light_shell_all`): every
    /// block is new there, so the queue would hold the entire shell and drain at
    /// the per-dispatch budget for a hundred frames - while the sweep, which that
    /// path also arms and which runs at the near cadence because everything is
    /// near, covers all of it in eight. The queue would be re-doing work the
    /// sweep is already doing.
    pub fn clear_urgent(&mut self) {
        for &b in &self.urgent {
            self.urgent_queued[b as usize / 64] &= !(1u64 << (b % 64));
        }
        self.urgent.clear();
    }

    /// Drop the first `n` urgent entries, which the caller has just dispatched.
    pub fn drain_urgent(&mut self, n: usize) {
        let n = n.min(self.urgent.len());
        for &b in &self.urgent[..n] {
            self.urgent_queued[b as usize / 64] &= !(1u64 << (b % 64));
        }
        self.urgent.drain(..n);
    }

    /// Queue `brick` for an urgent visit unless it is already queued.
    #[inline]
    fn queue_urgent(&mut self, brick: u32) {
        let (w, bit) = (brick as usize / 64, 1u64 << (brick % 64));
        if self.urgent_queued[w] & bit != 0 {
            return;
        }
        self.urgent_queued[w] |= bit;
        self.urgent.push(brick);
    }

    /// Move every brick for which `near` holds into the front of the work list
    /// and report whether the order changed.
    ///
    /// O(n) with one pass and at most one swap per element - no sort, no
    /// allocation - because the update pass only needs the two GROUPS separated,
    /// never the blocks ordered by distance within a group. The caller is
    /// expected to gate this on the camera having actually moved; walking 60,000
    /// entries every frame to reproduce the same partition would be its own
    /// version of the problem this exists to fix.
    pub fn repartition(&mut self, mut near: impl FnMut(u32) -> bool) -> bool {
        let mut lo = 0usize;
        let mut hi = self.live.len();
        let mut moved = false;
        while lo < hi {
            if near(self.live[lo]) {
                lo += 1;
            } else {
                hi -= 1;
                self.live.swap(lo, hi);
                self.fix_live_idx(lo);
                self.fix_live_idx(hi);
                moved = true;
            }
        }
        if moved || lo != self.near_count {
            self.list_dirty = true;
        }
        self.near_count = lo;
        moved || self.list_dirty
    }

    /// Repair the back-pointer of the block bound to `live[i]` after a swap.
    #[inline]
    fn fix_live_idx(&mut self, i: usize) {
        let brick = self.live[i];
        let block = self.block_of_brick[brick as usize];
        self.slots[block as usize].live_idx = i as u32;
    }

    /// Brick -> block table, uploaded verbatim for the shader-side lookup.
    #[inline]
    pub fn block_table(&self) -> &[u32] {
        &self.block_of_brick
    }

    #[inline]
    pub fn allocated(&self) -> usize {
        self.live.len()
    }

    /// Bind a block to `brick`, or return the existing one. Returns `None` when
    /// the pool is exhausted.
    ///
    /// Exhaustion is a soft failure ON PURPOSE: the shader falls back to the GI
    /// probe term for any voxel without a block, so a full pool costs lighting
    /// detail in the least-recently-reached region rather than crashing or
    /// stalling the frame. `take_overflow` reports it so it cannot pass silently.
    pub fn allocate(&mut self, brick: u32) -> Option<u32> {
        if let Some(b) = self.block_of(brick) {
            return Some(b);
        }
        let block = match self.free.pop() {
            Some(b) => {
                self.slots[b as usize] = BlockSlot { brick, live_idx: self.live.len() as u32 };
                // A RECYCLED block still holds the previous tenant's records,
                // and those carry a non-zero epoch, so the sampler would treat
                // them as valid light for this brick until the block's slice
                // came round. Queue it for zeroing.
                self.queue_reset(b);
                b
            }
            None => {
                if self.slots.len() as u32 >= self.blocks_max {
                    self.overflow = self.overflow.saturating_add(1);
                    self.overflow_total = self.overflow_total.saturating_add(1);
                    return None;
                }
                let b = self.slots.len() as u32;
                self.slots.push(BlockSlot { brick, live_idx: self.live.len() as u32 });
                b
            }
        };
        let was_all_near = self.near_count == self.live.len();
        self.live.push(brick);
        self.block_of_brick[brick as usize] = block;
        // "EVERYTHING IS NEAR" IS PRESERVED, and nothing else is. Until the first
        // `World::set_light_focus` there is no camera and no tier, and the field
        // is defined to treat the whole shell as near - that is what makes a
        // freshly bound world converge on the near cadence (8 frames) instead of
        // the far one (64). Once a real partition exists this test is false and a
        // newly bound block lands in the far group, which is the point: letting
        // every streamed brick into the near tier is what used to drag it back to
        // "everything is near" one block at a time.
        if was_all_near {
            self.near_count += 1;
        }
        // A NEW block has no record at all, so it also joins the URGENT list
        // whatever its tier: until it is gathered, every pixel on that surface
        // shades through the fallback.
        self.queue_urgent(brick);
        self.table_touched.push(brick);
        self.list_dirty = true;
        // A NEVER-BOUND block needs no reset: the pool buffer starts zeroed and
        // the update pass only ever writes blocks that are in the live list, so
        // an untouched slot already reads as epoch 0 (no valid history). This
        // matters at world init, where every block is fresh and queueing a
        // reset each would mean ~100k tiny GPU writes on one frame.
        Some(block)
    }

    /// Unbind `brick`'s block and return it to the pool. No-op if unbound.
    ///
    /// The block's GPU records are NOT cleared here. They are meaningless while
    /// unbound (nothing maps to them) and the reset queued by the next
    /// `allocate` makes them valid again, so freeing stays O(1) and touches no
    /// GPU memory.
    pub fn release(&mut self, brick: u32) {
        let block = match self.block_of(brick) {
            Some(b) => b,
            None => return,
        };
        self.block_of_brick[brick as usize] = LIGHT_BLOCK_NONE;
        self.table_touched.push(brick);
        self.list_dirty = true;
        // Remove from `live` while KEEPING THE NEAR PREFIX INTACT. A plain
        // swap_remove would drag the last (far) entry into the near group and
        // silently corrupt the partition, so a near entry vacates through the
        // near boundary first: at most two swaps, still O(1).
        let live_idx = self.slots[block as usize].live_idx as usize;
        let last = self.live.len() - 1;
        if live_idx < self.near_count {
            let near_last = self.near_count - 1;
            if live_idx != near_last {
                self.live.swap(live_idx, near_last);
                self.fix_live_idx(live_idx);
            }
            if near_last != last {
                self.live.swap(near_last, last);
                self.fix_live_idx(near_last);
            }
            self.near_count -= 1;
        } else if live_idx != last {
            self.live.swap(live_idx, last);
            self.fix_live_idx(live_idx);
        }
        self.live.pop();
        self.free.push(block);
    }

    /// Restart `brick`'s accumulated estimate (its geometry changed). No-op if
    /// the brick has no block, and idempotent within a frame.
    pub fn invalidate(&mut self, brick: u32) {
        if let Some(b) = self.block_of(brick) {
            self.queue_reset(b);
            // An invalidated block has just had its records zeroed, so until the
            // update pass reaches it again its voxels shade through the fallback.
            // The sweep cannot be relied on for that: it is paced by the sun and
            // stops altogether when the sun does. Urgent means the next dispatch,
            // whatever the schedule.
            self.queue_urgent(brick);
        }
    }

    /// Queue `block` for zeroing unless it is already queued.
    ///
    /// The dedup is not a micro-optimisation: physics re-dirties the same brick
    /// on consecutive ticks, and `sync_light_shell_dirty` invalidates every
    /// dirty brick, so without it one settling lake queues the same 512-byte DMA
    /// over and over.
    #[inline]
    fn queue_reset(&mut self, block: u32) {
        let (w, bit) = (block as usize / 64, 1u64 << (block % 64));
        if self.reset_queued[w] & bit != 0 {
            return;
        }
        self.reset_queued[w] |= bit;
        self.pending_reset.push(block);
    }

    /// The blocks needing an accumulation reset, SORTED so the caller's span
    /// coalescer can turn consecutive blocks into one write. Cleared by
    /// `clear_pending_reset`.
    pub fn pending_reset(&mut self) -> &[u32] {
        self.pending_reset.sort_unstable();
        &self.pending_reset
    }

    /// Acknowledge that the queued resets have been written.
    pub fn clear_pending_reset(&mut self) {
        for &b in &self.pending_reset {
            self.reset_queued[b as usize / 64] &= !(1u64 << (b % 64));
        }
        self.pending_reset.clear();
    }

    /// Number of allocation requests refused since the last call, and clear.
    pub fn take_overflow(&mut self) -> u32 {
        std::mem::replace(&mut self.overflow, 0)
    }

    /// Requests refused over this field's whole life. Never cleared, so a
    /// caller that reports only the per-frame delta can still say how much of
    /// the world has been turned away in total.
    #[inline]
    pub fn overflow_total(&self) -> u32 {
        self.overflow_total
    }

    /// Word offset of `block`'s records within the light pool buffer.
    #[inline]
    pub fn block_word_offset(block: u32) -> u32 {
        block * LIGHT_BLOCK_WORDS
    }

    #[cfg(test)]
    fn debug_assert_consistent(&self) {
        assert_eq!(
            self.live.len() + self.free.len(),
            self.slots.len(),
            "live and free must partition the allocated slots"
        );
        assert!(
            self.near_count <= self.live.len(),
            "near prefix ({}) runs past the work list ({})",
            self.near_count,
            self.live.len(),
        );
        let mut seen_bricks = std::collections::HashSet::new();
        let mut live_blocks = std::collections::HashSet::new();
        for (i, &brick) in self.live.iter().enumerate() {
            assert!(seen_bricks.insert(brick), "brick {brick} appears twice in live");
            let b = self.block_of_brick[brick as usize];
            assert_ne!(b, LIGHT_BLOCK_NONE, "live brick {brick} has no block");
            assert_eq!(self.slots[b as usize].live_idx as usize, i, "stale live_idx");
            assert_eq!(self.slots[b as usize].brick, brick, "brick/block disagree");
            assert!(live_blocks.insert(b), "block {b} is bound to two bricks");
        }
        for &b in &self.free {
            assert!(!live_blocks.contains(&b), "block {b} is both live and free");
        }
    }
}

// No `Default`: a field built without its pool capacity would either allocate
// nothing or, worse, allocate past whichever buffer it was later handed to.

#[cfg(test)]
mod tests {
    use super::*;

    /// Face-adjacent bricks of `bi`, x/z wrapping and y clamped. CPU mirror of
    /// `World::brick_neighbours`, which is private; kept local so this test
    /// states the adjacency it needs instead of the allocator handing it over.
    fn brick_face_neighbours(bi: u32) -> Vec<u32> {
        use crate::voxel::{brick_coords, brick_idx};
        use crate::world_dims::{WORLD_BRICKS_X, WORLD_BRICKS_Y, WORLD_BRICKS_Z};
        let (bx, by, bz) = brick_coords(bi);
        let wrap = |v: i64, m: u32| -> u32 { v.rem_euclid(m as i64) as u32 };
        let mut out = vec![
            brick_idx(wrap(bx as i64 + 1, WORLD_BRICKS_X), by, bz),
            brick_idx(wrap(bx as i64 - 1, WORLD_BRICKS_X), by, bz),
            brick_idx(bx, by, wrap(bz as i64 + 1, WORLD_BRICKS_Z)),
            brick_idx(bx, by, wrap(bz as i64 - 1, WORLD_BRICKS_Z)),
        ];
        if by + 1 < WORLD_BRICKS_Y {
            out.push(brick_idx(bx, by + 1, bz));
        }
        if by > 0 {
            out.push(brick_idx(bx, by - 1, bz));
        }
        out
    }

    /// The pool ceiling, checked against a REAL streamed world rather than an
    /// estimate of one.
    ///
    /// This is the test whose absence let the field run saturated: the ceiling
    /// was derived from the shape of the terrain shell, the crafted scenes bind
    /// a couple of thousand blocks, and nothing ever asked a fully generated
    /// world whether it fit. It did not - the demo world asked for 370,719
    /// blocks against a 131,072 ceiling, and because `allocate` fills brick
    /// indices in order and `brick_idx` is z-major, the refusals were not spread
    /// thin but concentrated into a hard geographic cliff.
    ///
    /// Both halves are asserted, because either one alone can pass while the
    /// feature is broken:
    ///  - it FITS, with headroom, and refuses nothing;
    ///  - and it still covers every air voxel that touches solid, so "it fits"
    ///    was not bought by leaving lit surfaces without storage.
    #[test]
    fn the_demo_world_light_shell_fits_the_pool_and_still_covers_it() {
        use crate::voxel::World;
        let mut w = World::new();
        w.fill_demo_terrain();

        let (mut empty, mut partial, mut full) = (0u32, 0u32, 0u32);
        for b in &w.bricks {
            if b.is_empty() {
                empty += 1;
            } else if b.is_full() {
                full += 1;
            } else {
                partial += 1;
            }
        }

        w.sync_light_shell_all();
        let light = w.light.allocated();
        eprintln!(
            "demo world bricks: {empty} empty, {partial} partial, {full} full, {} total",
            WORLD_BRICKS_TOTAL,
        );
        eprintln!(
            "demo world binds: light {light} / {LIGHT_BLOCKS_MAX} blocks ({:.1} MB of {:.1} MB)",
            light as f64 * (LIGHT_RECORDS_PER_BLOCK * LIGHT_RECORD_WORDS * 4) as f64 / 1e6,
            LIGHT_POOL_WORDS as f64 * 4.0 / 1e6,
        );

        assert_eq!(
            w.light.take_overflow(),
            0,
            "the demo world's lit shell ({light} blocks) does not fit the {LIGHT_BLOCKS_MAX}-block \
             pool; the refused bricks render through the per-pixel path, and because blocks are \
             handed out in z-major brick order the loss is a contiguous slab of the world rather \
             than an even thinning"
        );
        // Headroom, not just a fit: terrain seeds vary, and a ceiling that a
        // real world only just clears is one cave system away from the cliff
        // above.
        assert!(
            light * 2 <= LIGHT_BLOCKS_MAX as usize,
            "the lit shell ({light}) uses more than half the {LIGHT_BLOCKS_MAX}-block pool; \
             there is no headroom for a rougher world"
        );

        // Coverage, brick by brick, stated from what the SAMPLER needs rather
        // than copied from `brick_needs_light`: every voxel that can be a usable
        // trilinear tap must lie in a brick that has a block, or shading there
        // has no light to read at all.
        //
        // A usable tap is a CARRIER voxel - air, or foliage - next to geometry.
        // Foliage joined that set when the field stopped treating a canopy as a
        // wall (`vl_tap` in shaders/raymarch.wgsl); before that, a fully leafy
        // brick was "solid" and got nothing, which is why over half a canopy view
        // had no record.
        //
        // A carrier's occupied neighbour is either in its own brick (then that
        // brick is non-empty) or in a face-adjacent one (then that neighbour is
        // non-empty), so these two cases are the whole requirement.
        let mut missing = Vec::new();
        let mut stray = Vec::new();
        for bi in 0..WORLD_BRICKS_TOTAL {
            let b = &w.bricks[bi as usize];
            let has_block = w.light.block_of(bi).is_some();
            let has_carrier =
                !b.is_full() || b.materials.iter().any(|&m| crate::voxel::is_foliage_mat(m));
            if !has_carrier {
                // Every voxel is an opaque occluder: the update pass stamps them
                // all epoch 0 and the sampler drops every tap into them, so a
                // block here is 512 bytes and a workgroup of update work that
                // nothing can ever read.
                if has_block && stray.len() < 8 {
                    stray.push(bi);
                }
                continue;
            }
            let needs = !b.is_empty()
                || brick_face_neighbours(bi).iter().any(|&n| !w.bricks[n as usize].is_empty());
            if needs && !has_block && missing.len() < 8 {
                missing.push(bi);
            }
        }
        assert!(
            missing.is_empty(),
            "bricks holding lit carrier voxels have no light block (first few: {missing:?})"
        );
        assert!(
            stray.is_empty(),
            "bricks with no carrier voxel hold light blocks nothing can read (first few: {stray:?})"
        );

        // A FULLY LEAFY brick is the case this rule exists for, and it is worth
        // asserting the world actually contains some - otherwise the clause above
        // is untested on real data and could be deleted without anything failing.
        let leafy = (0..WORLD_BRICKS_TOTAL)
            .filter(|&bi| {
                let b = &w.bricks[bi as usize];
                b.is_full() && b.materials.iter().any(|&m| crate::voxel::is_foliage_mat(m))
            })
            .count();
        eprintln!("demo world: {leafy} fully occupied bricks carry foliage and now hold a block");
        assert!(leafy > 0, "the demo world has no dense canopy, so this rule is untested here");
    }

    /// The CPU and shader foliage sets must be the SAME set, read out of the
    /// shader rather than trusted.
    ///
    /// They are two halves of one decision: `is_foliage_mat` in raymarch.wgsl
    /// decides which voxels the light field may interpolate through and which
    /// carry a record, and `crate::voxel::is_foliage_mat` decides which bricks get
    /// storage for them. A material in the shader's set but not the CPU's is a
    /// canopy the sampler will read from a brick that was never allocated - i.e.
    /// exactly the hole this rework closed, re-opened for one material and
    /// invisible in every aggregate number.
    #[test]
    fn the_foliage_material_sets_match_the_shader() {
        let src = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/shaders/raymarch.wgsl"
        ))
        .expect("the shader source must be readable");

        // MAT_NAME -> value, from the shader's own constant declarations.
        let mut values = std::collections::HashMap::new();
        for line in src.lines() {
            let line = line.trim();
            let Some(rest) = line.strip_prefix("const MAT_") else { continue };
            let Some((name, tail)) = rest.split_once(':') else { continue };
            let Some((_, val)) = tail.split_once('=') else { continue };
            let digits: String = val.trim().chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(v) = digits.parse::<u32>() {
                values.insert(format!("MAT_{}", name.trim()), v);
            }
        }
        assert!(values.len() > 10, "failed to parse the shader's material constants");

        let start = src.find("fn is_foliage_mat(m: u32) -> bool {").expect("shader has no is_foliage_mat");
        let body = &src[start..start + src[start..].find('}').expect("unterminated is_foliage_mat")];
        let mut shader_set = std::collections::BTreeSet::new();
        for tok in body.split(|c: char| !(c.is_alphanumeric() || c == '_')) {
            if let Some(&v) = values.get(tok) {
                assert!(v <= 255, "material {tok} = {v} does not fit the CPU u8 material");
                shader_set.insert(v as u8);
            }
        }
        // Water levels are a RANGE in the shader (`is_water_mat`), and no water
        // material appears by name in is_foliage_mat, so a plain name scan is
        // complete for this function specifically.
        let cpu_set: std::collections::BTreeSet<u8> =
            (0u8..=255).filter(|&m| crate::voxel::is_foliage_mat(m)).collect();
        assert_eq!(
            shader_set, cpu_set,
            "the shader's foliage set and the CPU's disagree; the light field would allocate \
             storage for one set of materials and interpolate through another"
        );
    }

    #[test]
    fn pool_sizing_matches_the_documented_budget() {
        // WAS: 131,072 blocks x 64 records (one per voxel) x 2 words = 64 MiB.
        // NOW: 2,097,152 blocks x 8 records (one per 2x2x2 voxels) x 2 words
        // = 128 MiB. The PREMISE changed, not the invariant: at 10 cm the lit
        // shell is ~10x the blocks (it is a surface, and the resolution doubled
        // and a half twice over), and the pool pays for that with a 16x cheaper
        // block rather than with 16x the memory. See LIGHT_RECORD_STEP.
        assert_eq!(LIGHT_RECORDS_PER_BLOCK, 8);
        assert_eq!(LIGHT_BLOCK_WORDS * 4, 64, "a block is 64 bytes");
        assert_eq!(LIGHT_POOL_WORDS as u64 * 4, 128 * 1024 * 1024);
    }

    #[test]
    fn the_group_mask_matches_the_record_step() {
        // `vl_group_occ` in raymarch.wgsl answers "is any voxel of this record's
        // cell solid" with ONE shift of the constant 0x00330033 over one half of
        // the brick's 64-bit occupancy word. That identity is specific to a 2^3
        // group inside a 4^3 brick laid out x + z*4 + y*16: it is what keeps the
        // coarser gate as cheap per tap as the per-voxel one it replaced.
        //
        // If LIGHT_RECORD_STEP ever moves, the shader mask has to be rederived -
        // it will NOT simply be wrong at the edges, it will address the wrong
        // half of the brick. Rather than let that be silent, fail here.
        assert_eq!(
            LIGHT_RECORD_STEP, 2,
            "raymarch.wgsl::vl_group_occ hardcodes the 2^3 group pattern 0x00330033;              rederive it before changing the step"
        );
        assert_eq!(LIGHT_RECORD_DIM, 2);
        // The pattern, rebuilt here from the brick linearisation, must be the
        // literal the shader uses.
        let mut pat: u64 = 0;
        for dy in 0..2u32 {
            for dz in 0..2u32 {
                for dx in 0..2u32 {
                    pat |= 1u64 << crate::voxel::brick_voxel_idx(dx, dy, dz);
                }
            }
        }
        assert_eq!(pat, 0x0033_0033, "the 2^3 group bit pattern");
        // ...and every group is that pattern shifted by 2*rx + 8*rz within one
        // 32-bit half, chosen by ry. Verified exhaustively.
        for ry in 0..2u32 {
            for rz in 0..2u32 {
                for rx in 0..2u32 {
                    let mut want: u64 = 0;
                    for dy in 0..2u32 {
                        for dz in 0..2u32 {
                            for dx in 0..2u32 {
                                want |= 1u64
                                    << crate::voxel::brick_voxel_idx(
                                        rx * 2 + dx,
                                        ry * 2 + dy,
                                        rz * 2 + dz,
                                    );
                            }
                        }
                    }
                    let half = (want >> (32 * ry as u64)) as u32;
                    assert_eq!(
                        u64::from(half) << (32 * ry as u64),
                        want,
                        "group ({rx},{ry},{rz}) straddles the 32-bit halves"
                    );
                    assert_eq!(
                        half,
                        0x0033_0033u32 << (2 * rx + 8 * rz),
                        "group ({rx},{ry},{rz}) is not the shifted pattern"
                    );
                }
            }
        }
    }

    #[test]
    fn the_capacity_is_per_field_not_a_global_constant() {
        // A ceiling read from LIGHT_BLOCKS_MAX instead of from the instance
        // would let a field built for a smaller pool hand out block indices
        // past the end of that pool's buffer.
        let mut lf = LightField::new(3);
        for brick in 0..3u32 {
            assert!(lf.allocate(brick).is_some(), "block {brick} is within capacity 3");
        }
        assert_eq!(lf.allocate(3), None, "capacity 3 must refuse a fourth block");
        assert_eq!(lf.take_overflow(), 1);
        assert_eq!(lf.blocks_max(), 3);
    }

    #[test]
    fn allocate_binds_and_is_idempotent() {
        let mut lf = LightField::new(LIGHT_BLOCKS_MAX);
        let a = lf.allocate(7).unwrap();
        let b = lf.allocate(7).unwrap();
        assert_eq!(a, b, "re-allocating a bound brick must return the same block");
        assert_eq!(lf.allocated(), 1);
        assert_eq!(lf.block_of(7), Some(a));
        lf.debug_assert_consistent();
    }

    #[test]
    fn only_recycled_blocks_request_a_reset() {
        let mut lf = LightField::new(LIGHT_BLOCKS_MAX);
        lf.allocate(3);
        assert_eq!(
            lf.pending_reset().len(),
            0,
            "a never-bound block is already zeroed in the pool buffer"
        );
        // Re-binding the SAME block to a DIFFERENT brick must reset: the
        // records still hold the previous tenant's light, at a non-zero epoch
        // the sampler would otherwise trust.
        lf.release(3);
        let b = lf.allocate(99).unwrap();
        assert_eq!(b, 0, "the freed block should be reused");
        assert_eq!(lf.pending_reset(), &[0]);
    }

    #[test]
    fn release_keeps_the_live_list_compact_and_repairs_backpointers() {
        let mut lf = LightField::new(LIGHT_BLOCKS_MAX);
        for brick in 0..5u32 {
            lf.allocate(brick).unwrap();
        }
        // Release from the MIDDLE so swap-remove has to move the last entry.
        lf.release(1);
        assert_eq!(lf.allocated(), 4);
        assert_eq!(lf.block_of(1), None);
        lf.debug_assert_consistent();
        // Every surviving brick must still be in the work list, and the
        // released one must be gone from it.
        for brick in [0u32, 2, 3, 4] {
            assert!(lf.block_of(brick).is_some(), "brick {brick} still bound");
            assert!(lf.live_bricks().contains(&brick), "brick {brick} still in work list");
        }
        assert!(!lf.live_bricks().contains(&1), "released brick must leave the work list");
    }

    #[test]
    fn release_of_an_unbound_brick_is_a_noop() {
        let mut lf = LightField::new(LIGHT_BLOCKS_MAX);
        lf.release(42);
        lf.release(42);
        assert_eq!(lf.allocated(), 0);
        lf.debug_assert_consistent();
    }

    #[test]
    fn invalidate_without_a_block_is_a_noop() {
        let mut lf = LightField::new(LIGHT_BLOCKS_MAX);
        lf.invalidate(11);
        assert_eq!(lf.pending_reset().len(), 0);
    }

    #[test]
    fn freed_blocks_are_reused_before_growing_the_pool() {
        let mut lf = LightField::new(LIGHT_BLOCKS_MAX);
        let a = lf.allocate(1).unwrap();
        lf.allocate(2).unwrap();
        lf.release(1);
        let c = lf.allocate(3).unwrap();
        assert_eq!(c, a, "a freed block must be recycled, not appended");
        assert_eq!(lf.allocated(), 2, "recycling must not grow the live set");
        lf.debug_assert_consistent();
    }

    #[test]
    fn exhaustion_is_reported_and_degrades_instead_of_panicking() {
        let mut lf = LightField::new(LIGHT_BLOCKS_MAX);
        // Fill the pool exactly.
        for brick in 0..LIGHT_BLOCKS_MAX {
            assert!(lf.allocate(brick).is_some(), "block {brick} should fit");
        }
        assert_eq!(lf.allocated(), LIGHT_BLOCKS_MAX as usize);
        assert_eq!(lf.take_overflow(), 0, "no overflow while it still fits");

        // One more must refuse rather than panic or grow.
        assert_eq!(lf.allocate(LIGHT_BLOCKS_MAX), None);
        assert_eq!(lf.allocate(LIGHT_BLOCKS_MAX + 1), None);
        assert_eq!(lf.take_overflow(), 2);
        assert_eq!(lf.take_overflow(), 0, "overflow count clears when taken");

        // An already-bound brick still resolves even when the pool is full.
        assert!(lf.block_of(0).is_some());

        // Freeing one makes room again.
        lf.release(0);
        assert!(lf.allocate(LIGHT_BLOCKS_MAX).is_some());
    }

    #[test]
    fn block_offsets_are_distinct_and_in_range() {
        let last = LIGHT_BLOCKS_MAX - 1;
        assert_eq!(LightField::block_word_offset(0), 0);
        // 8 records x 2 words. Was 128 when a record was one voxel.
        assert_eq!(LightField::block_word_offset(1), LIGHT_BLOCK_WORDS);
        assert_eq!(LIGHT_BLOCK_WORDS, 16);
        let end = LightField::block_word_offset(last) + LIGHT_BLOCK_WORDS;
        assert_eq!(end, LIGHT_POOL_WORDS, "the last block must end exactly at the pool end");
    }

    /// The near/far split must survive `release`, which is the one operation
    /// that moves an element it was not asked about.
    ///
    /// A plain `swap_remove` drags the LAST entry into the hole. When the hole is
    /// in the near prefix and the last entry is far, that silently promotes a
    /// distant block to the near cadence and, worse, leaves `near_count`
    /// describing a list it no longer describes. Nothing downstream would ever
    /// notice: the field would still render, just refreshing the wrong blocks.
    #[test]
    fn release_preserves_the_near_far_partition() {
        let mut lf = LightField::new(LIGHT_BLOCKS_MAX);
        for brick in 0..8u32 {
            lf.allocate(brick).unwrap();
        }
        // Bricks 0..4 near, 4..8 far.
        lf.repartition(|b| b < 4);
        assert_eq!(lf.near_count(), 4);
        let near: std::collections::HashSet<u32> =
            lf.live_bricks()[..4].iter().copied().collect();
        assert_eq!(near, (0..4).collect(), "the near prefix must be exactly the near bricks");

        // Release from the MIDDLE of the near prefix.
        lf.release(1);
        lf.debug_assert_consistent();
        assert_eq!(lf.near_count(), 3, "the near group lost exactly one member");
        let near: std::collections::HashSet<u32> =
            lf.live_bricks()[..3].iter().copied().collect();
        assert_eq!(near, [0u32, 2, 3].into_iter().collect(), "no far brick promoted itself");
        let far: std::collections::HashSet<u32> =
            lf.live_bricks()[3..].iter().copied().collect();
        assert_eq!(far, (4..8).collect(), "the far group is intact");

        // And from the far group, where near_count must NOT move.
        lf.release(6);
        lf.debug_assert_consistent();
        assert_eq!(lf.near_count(), 3);
        let far: std::collections::HashSet<u32> =
            lf.live_bricks()[3..].iter().copied().collect();
        assert_eq!(far, [4u32, 5, 7].into_iter().collect());
    }

    /// A newly bound block goes on the URGENT list, wherever it is.
    ///
    /// It has no record at all, so until it gets one its voxels shade through the
    /// fallback. It used to be pushed into the NEAR TIER instead, which was wrong
    /// twice over: it still waited up to `update_div` rounds for its slice, and
    /// the promotion was one-way, so streaming dragged the tier boundary out
    /// until "near" meant "most of the shell". The tier is the sun's tracking
    /// radius and nothing else now; urgency is a separate list.
    #[test]
    fn a_new_block_is_urgent_not_near() {
        let mut lf = LightField::new(LIGHT_BLOCKS_MAX);
        for brick in 0..4u32 {
            lf.allocate(brick).unwrap();
        }
        // With no partition yet, everything is near - that is what makes a fresh
        // world converge on the near cadence.
        assert_eq!(lf.near_count(), 4, "with no camera focus the whole shell is near");
        lf.drain_urgent(lf.urgent().len());
        lf.repartition(|_| false);
        assert_eq!(lf.near_count(), 0, "nothing is near");
        lf.allocate(99).unwrap();
        assert_eq!(lf.near_count(), 0, "a new block must not widen the sun-tracking tier");
        assert_eq!(lf.urgent(), &[99], "it must be serviced on the next dispatch instead");
        lf.debug_assert_consistent();
        // Draining is what the dispatch acknowledges; the entry must not linger
        // and re-dispatch for ever.
        lf.drain_urgent(1);
        assert!(lf.urgent().is_empty());
        // ... and the same brick must be queueable again after a later edit.
        lf.invalidate(99);
        assert_eq!(lf.urgent(), &[99]);
    }

    /// The urgent list must drain in FIFO order and at the caller's budget, so a
    /// burst is spread over frames instead of launched in one enormous dispatch.
    ///
    /// A chunk install dirties 3,072 bricks at once. Dispatching every one of
    /// them on the frame the install lands would be a spike on exactly the frame
    /// that is already paying for the install.
    #[test]
    fn urgent_drains_oldest_first_at_the_callers_budget() {
        let mut lf = LightField::new(LIGHT_BLOCKS_MAX);
        for brick in 0..10u32 {
            lf.allocate(brick).unwrap();
        }
        assert_eq!(lf.urgent().len(), 10);
        lf.drain_urgent(4);
        assert_eq!(lf.urgent(), &[4, 5, 6, 7, 8, 9], "oldest four go first");
        // A brick still queued must not be queued twice by a second edit.
        lf.invalidate(5);
        assert_eq!(lf.urgent(), &[4, 5, 6, 7, 8, 9]);
        // One already drained can be.
        lf.invalidate(1);
        assert_eq!(lf.urgent(), &[4, 5, 6, 7, 8, 9, 1]);
        // Draining past the end is clamped, not a panic.
        lf.drain_urgent(999);
        assert!(lf.urgent().is_empty());
    }

    /// Repartitioning must not lose or duplicate a block, and must leave every
    /// back-pointer usable.
    #[test]
    fn repartition_is_a_permutation() {
        let mut lf = LightField::new(LIGHT_BLOCKS_MAX);
        for brick in 0..64u32 {
            lf.allocate(brick).unwrap();
        }
        lf.repartition(|b| b % 3 == 0);
        lf.debug_assert_consistent();
        assert_eq!(lf.near_count(), 22, "0,3,..,63 is 22 bricks");
        assert!(lf.live_bricks()[..22].iter().all(|b| b % 3 == 0));
        assert!(lf.live_bricks()[22..].iter().all(|b| b % 3 != 0));
        let all: std::collections::HashSet<u32> = lf.live_bricks().iter().copied().collect();
        assert_eq!(all, (0..64).collect(), "every brick survives the permutation");
        // Re-partitioning the other way must be just as clean, including the
        // back-pointers a later release depends on.
        lf.repartition(|b| b % 2 == 0);
        lf.debug_assert_consistent();
        assert_eq!(lf.near_count(), 32);
        for brick in 0..64u32 {
            lf.release(brick);
            lf.debug_assert_consistent();
        }
        assert_eq!(lf.allocated(), 0);
        assert_eq!(lf.near_count(), 0);
    }

    /// Queueing the same block twice must produce ONE reset.
    ///
    /// Physics re-dirties the same brick tick after tick and every dirty brick
    /// is invalidated, so without the dedup a settling lake re-queues the same
    /// 512-byte DMA indefinitely.
    #[test]
    fn resets_are_deduped_and_sorted() {
        let mut lf = LightField::new(LIGHT_BLOCKS_MAX);
        for brick in [5u32, 9, 1] {
            lf.allocate(brick).unwrap();
        }
        lf.invalidate(5);
        lf.invalidate(5);
        lf.invalidate(9);
        lf.invalidate(5);
        let blocks: Vec<u32> = lf.pending_reset().to_vec();
        assert_eq!(blocks.len(), 2, "three invalidates of two bricks are two resets");
        assert!(blocks.windows(2).all(|w| w[0] < w[1]), "sorted, so runs coalesce");
        // Clearing must also clear the membership marks, or the block could
        // never be queued again.
        lf.clear_pending_reset();
        assert!(lf.pending_reset().is_empty());
        lf.invalidate(5);
        assert_eq!(lf.pending_reset().len(), 1, "a cleared block can be queued again");
    }

    /// The table delta must name every brick whose entry moved and nothing else.
    ///
    /// This is what replaced a 4 MiB whole-table push, so an entry that changed
    /// without being recorded is a stale GPU table: the shader would read a block
    /// index for a brick that no longer owns it, i.e. one brick's light on
    /// another, and only in a live world where bindings churn.
    #[test]
    fn the_table_delta_names_exactly_the_entries_that_moved() {
        let mut lf = LightField::new(LIGHT_BLOCKS_MAX);
        lf.allocate(10).unwrap();
        lf.allocate(11).unwrap();
        {
            let (touched, _) = lf.table_delta();
            assert_eq!(touched, &[10, 11]);
        }
        lf.clear_table_touched();
        {
            let (touched, _) = lf.table_delta();
            assert!(touched.is_empty(), "a quiet frame pushes nothing");
        }
        // A no-op allocate and a no-op release must not dirty anything.
        lf.allocate(10).unwrap();
        lf.release(999);
        {
            let (touched, _) = lf.table_delta();
            assert!(touched.is_empty(), "no-ops must not push 4 MiB");
        }
        // A real release, and the value at that index must be readable through
        // the same call.
        lf.release(11);
        let (touched, table) = lf.table_delta();
        assert_eq!(touched, &[11]);
        assert_eq!(table[11], LIGHT_BLOCK_NONE);
        assert_ne!(table[10], LIGHT_BLOCK_NONE);
    }

    #[test]
    fn churn_preserves_every_invariant() {
        // Streaming frees and rebinds whole slots constantly; walk a
        // deterministic churn pattern and assert the structure never drifts.
        fn lcg(state: &mut u32) -> u32 {
            *state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            *state
        }
        let mut lf = LightField::new(LIGHT_BLOCKS_MAX);
        let mut rng = 12345u32;
        for i in 0..4000 {
            let brick = lcg(&mut rng) % 512;
            if lcg(&mut rng) % 3 == 0 {
                lf.release(brick);
            } else {
                lf.allocate(brick);
            }
            // Repartition mid-churn, as a moving camera does: allocate, release
            // and repartition all rewrite `live_idx`, and they have to agree.
            if i % 97 == 0 {
                let pivot = lcg(&mut rng) % 512;
                lf.repartition(|b| b < pivot);
                lf.debug_assert_consistent();
                assert!(lf.live_bricks()[..lf.near_count() as usize].iter().all(|&b| b < pivot));
                assert!(lf.live_bricks()[lf.near_count() as usize..].iter().all(|&b| b >= pivot));
            }
        }
        lf.debug_assert_consistent();
        // Every live block must be reachable from its brick and vice versa.
        assert_eq!(
            lf.live_bricks().len(),
            (0..512).filter(|&b| lf.block_of(b).is_some()).count(),
            "live list length must equal the number of bound bricks"
        );
    }
}
