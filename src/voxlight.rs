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
// Because it holds no payload, ONE allocator serves both sparse fields: the
// light field and the per-voxel REFLECTED RADIANCE field, which differ only in
// pool capacity, record stride and which bricks qualify. Capacity is therefore
// a constructor argument and the stride lives with the caller's buffer, so the
// two never share a number by accident.

use crate::world_dims::{BRICK_VOXELS, WORLD_BRICKS_TOTAL};

/// Light records per block. One per voxel of a brick.
pub const LIGHT_RECORDS_PER_BLOCK: u32 = BRICK_VOXELS;

/// A record is two u32 words:
///   word0: sun_vis u8 | ao u8 | epoch u8 | flags u8
///   word1: point-light radiance, packed RGB9E5
pub const LIGHT_RECORD_WORDS: u32 = 2;

/// Resident light blocks. The lit shell of the streamed window is roughly a
/// 128x128 brick sheet in xz spread over several brick layers by relief,
/// caves, overhangs and trees, which lands in the 65k-130k range; this is the
/// ceiling, and exhaustion degrades gracefully (see `allocate`).
pub const LIGHT_BLOCKS_MAX: u32 = 131_072;

/// u32 words of GPU storage backing the whole pool (64 MiB).
pub const LIGHT_POOL_WORDS: u32 = LIGHT_BLOCKS_MAX * LIGHT_RECORDS_PER_BLOCK * LIGHT_RECORD_WORDS;

/// A reflection record is four u32 words: SH-L1 directional radiance, a DC
/// term plus three directional terms, each packed RGB9E5. Directional rather
/// than one flat colour so evaluating the SH along the reflection direction
/// keeps coarse view dependence, which one colour per voxel would collapse
/// into a uniform wash (docs/VOXEL_LIGHTING_PLAN.md, "Reflections").
pub const REFL_RECORD_WORDS: u32 = 4;

/// Resident reflection blocks. An eighth of the light ceiling because the set
/// is far smaller by construction: a block is bound only for a brick that
/// actually CONTAINS water or glass, not for the whole air shell around solid
/// geometry, and reflective surfaces are a thin sheet (a lake top, a window)
/// rather than the entire terrain.
pub const REFL_BLOCKS_MAX: u32 = 16_384;

/// u32 words of GPU storage backing the reflection pool (16 MiB).
pub const REFL_POOL_WORDS: u32 = REFL_BLOCKS_MAX * LIGHT_RECORDS_PER_BLOCK * REFL_RECORD_WORDS;

/// `block_of_brick` entry meaning "this brick has no light block".
pub const LIGHT_BLOCK_NONE: u32 = u32::MAX;

/// Marks a block's accumulated estimate as invalid so the update pass restarts
/// it from scratch rather than folding new samples into stale light. Stored in
/// the record's `epoch` byte; the GPU side treats 0 as "no valid history".
pub const LIGHT_EPOCH_RESET: u32 = 0;

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
    live: Vec<u32>,
    /// Block indices available for reuse.
    free: Vec<u32>,
    /// Blocks whose accumulation must restart (brick edited, slot recycled).
    /// Drained by the renderer each frame into a GPU upload.
    pending_reset: Vec<u32>,
    /// Allocation requests refused because the pool was full, since the last
    /// `take_overflow`. Surfaced rather than silently dropped.
    overflow: u32,
    /// Set whenever a binding changed, so the renderer re-uploads the brick
    /// table and work list only when they actually differ. Without this the
    /// frame would push 4 MB of unchanged table every time.
    table_dirty: bool,
    /// Blocks in the GPU pool backing this field. Held per instance rather than
    /// read from a constant so the light and reflection fields cannot allocate
    /// past each other's buffer.
    blocks_max: u32,
}

impl LightField {
    /// `blocks_max` must match the block count of the GPU pool this field
    /// indexes (`LIGHT_BLOCKS_MAX` / `REFL_BLOCKS_MAX`); it is the only thing
    /// stopping `allocate` from handing out an out-of-range block index.
    pub fn new(blocks_max: u32) -> Self {
        Self {
            block_of_brick: vec![LIGHT_BLOCK_NONE; WORLD_BRICKS_TOTAL as usize],
            slots: Vec::new(),
            live: Vec::new(),
            free: Vec::new(),
            pending_reset: Vec::new(),
            overflow: 0,
            // The GPU table starts all-NONE and the pool starts zeroed
            // (epoch 0 = invalid), which is exactly the empty state, so the
            // first frame has nothing to re-upload.
            table_dirty: false,
            blocks_max,
        }
    }

    /// Pool ceiling this field was built for. The renderer reports it when the
    /// pool overflows, so the message names the field's own cap.
    #[inline]
    pub fn blocks_max(&self) -> u32 {
        self.blocks_max
    }

    /// Whether the brick table / work list changed since the last call, and clear.
    pub fn take_table_dirty(&mut self) -> bool {
        std::mem::replace(&mut self.table_dirty, false)
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
                self.pending_reset.push(b);
                b
            }
            None => {
                if self.slots.len() as u32 >= self.blocks_max {
                    self.overflow = self.overflow.saturating_add(1);
                    return None;
                }
                let b = self.slots.len() as u32;
                self.slots.push(BlockSlot { brick, live_idx: self.live.len() as u32 });
                b
            }
        };
        self.live.push(brick);
        self.block_of_brick[brick as usize] = block;
        self.table_dirty = true;
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
        self.table_dirty = true;
        // Swap-remove from `live`, repairing the moved entry's back-pointer.
        let live_idx = self.slots[block as usize].live_idx as usize;
        let moved_brick = *self.live.last().expect("live non-empty while a block is bound");
        self.live.swap_remove(live_idx);
        if moved_brick != brick {
            let moved_block = self.block_of_brick[moved_brick as usize];
            self.slots[moved_block as usize].live_idx = live_idx as u32;
        }
        self.free.push(block);
    }

    /// Restart `brick`'s accumulated estimate (its geometry changed). No-op if
    /// the brick has no block.
    pub fn invalidate(&mut self, brick: u32) {
        if let Some(b) = self.block_of(brick) {
            self.pending_reset.push(b);
        }
    }

    /// Drain the blocks needing an accumulation reset.
    pub fn take_pending_reset(&mut self) -> std::vec::Drain<'_, u32> {
        self.pending_reset.drain(..)
    }

    /// Number of allocation requests refused since the last call, and clear.
    pub fn take_overflow(&mut self) -> u32 {
        std::mem::replace(&mut self.overflow, 0)
    }

    /// Word offset of `block`'s records within the LIGHT pool buffer.
    #[inline]
    pub fn block_word_offset(block: u32) -> u32 {
        Self::block_word_offset_with(LIGHT_RECORD_WORDS, block)
    }

    /// Word offset of `block` in a pool whose records are `record_words` wide.
    ///
    /// Explicit stride because the reflection pool's record is twice the light
    /// record: computing its offsets with `block_word_offset` would land every
    /// block at half its true address, silently overlapping neighbours instead
    /// of failing.
    #[inline]
    pub fn block_word_offset_with(record_words: u32, block: u32) -> u32 {
        block * LIGHT_RECORDS_PER_BLOCK * record_words
    }

    #[cfg(test)]
    fn debug_assert_consistent(&self) {
        assert_eq!(
            self.live.len() + self.free.len(),
            self.slots.len(),
            "live and free must partition the allocated slots"
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

    #[test]
    fn pool_sizing_matches_the_documented_budget() {
        // 131072 blocks * 64 records * 2 words * 4 bytes = 64 MiB.
        assert_eq!(LIGHT_RECORDS_PER_BLOCK, 64);
        assert_eq!(LIGHT_POOL_WORDS as u64 * 4, 64 * 1024 * 1024);
    }

    #[test]
    fn refl_pool_sizing_matches_the_documented_budget() {
        // 16384 blocks * 64 records * 4 words * 4 bytes = 16 MiB.
        assert_eq!(REFL_POOL_WORDS as u64 * 4, 16 * 1024 * 1024);
    }

    #[test]
    fn the_capacity_is_per_field_not_a_global_constant() {
        // The reflection field is an eighth of the light field, so a ceiling
        // read from LIGHT_BLOCKS_MAX instead of the instance would let it hand
        // out block indices past the end of its own GPU pool.
        let mut lf = LightField::new(3);
        for brick in 0..3u32 {
            assert!(lf.allocate(brick).is_some(), "block {brick} is within capacity 3");
        }
        assert_eq!(lf.allocate(3), None, "capacity 3 must refuse a fourth block");
        assert_eq!(lf.take_overflow(), 1);
        assert_eq!(lf.blocks_max(), 3);
    }

    #[test]
    fn refl_block_offsets_use_the_reflection_stride() {
        // The reflection record is twice the light record, so sharing
        // `block_word_offset` would overlap every block with its neighbour.
        assert_eq!(LightField::block_word_offset_with(REFL_RECORD_WORDS, 0), 0);
        assert_eq!(LightField::block_word_offset_with(REFL_RECORD_WORDS, 1), 256);
        assert_eq!(
            LightField::block_word_offset_with(LIGHT_RECORD_WORDS, 1),
            LightField::block_word_offset(1),
            "the light stride must stay the default"
        );
        let last = REFL_BLOCKS_MAX - 1;
        let end = LightField::block_word_offset_with(REFL_RECORD_WORDS, last)
            + LIGHT_RECORDS_PER_BLOCK * REFL_RECORD_WORDS;
        assert_eq!(end, REFL_POOL_WORDS, "the last block must end exactly at the pool end");
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
            lf.take_pending_reset().count(),
            0,
            "a never-bound block is already zeroed in the pool buffer"
        );
        // Re-binding the SAME block to a DIFFERENT brick must reset: the
        // records still hold the previous tenant's light, at a non-zero epoch
        // the sampler would otherwise trust.
        lf.release(3);
        let b = lf.allocate(99).unwrap();
        assert_eq!(b, 0, "the freed block should be reused");
        assert_eq!(lf.take_pending_reset().collect::<Vec<_>>(), vec![0]);
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
        assert_eq!(lf.take_pending_reset().count(), 0);
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
        assert_eq!(LightField::block_word_offset(1), 128);
        let end = LightField::block_word_offset(last) + LIGHT_RECORDS_PER_BLOCK * LIGHT_RECORD_WORDS;
        assert_eq!(end, LIGHT_POOL_WORDS, "the last block must end exactly at the pool end");
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
        for _ in 0..4000 {
            let brick = lcg(&mut rng) % 512;
            if lcg(&mut rng) % 3 == 0 {
                lf.release(brick);
            } else {
                lf.allocate(brick);
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
