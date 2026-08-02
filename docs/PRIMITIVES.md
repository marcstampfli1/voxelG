# Primitives catalog

The named, safe-by-construction things to reuse, and the raw pattern each one
replaces. If you are about to hand-roll something below, use the primitive
instead. Use-cases and detail live in the primitive itself; this file is the
index.

Every entry here was verified present in the tree when written.

## World geometry

**`world_dims.rs` constants** - the SINGLE source of truth for every world
dimension. `build.rs` generates the matching WGSL constants from this same file,
so Rust, the CPU raycaster and every shader move together.
FORBIDDEN: a literal 1600 / 400 / 640 / 160 / 4 for a world dimension anywhere,
and any second copy of these values in WGSL. `shaders/grass.wgsl` is the one
shader assembled without that prelude and has to carry its own copy of the scale;
`grass::shader_scale_matches_world_dims` parses the WGSL and pins it.

**`brick_idx` / `brick_coords`** (`voxel.rs`) - forward and inverse brick
linearisation, deliberately adjacent in the file so they cannot drift apart.
FORBIDDEN: open-coding `bx + by*W + bz*W*H` or its inverse at a call site.

**`pos_mod` (`common.wgsl`) / `world_to_slot_voxel` (`raymarch.wgsl`)** - the
toroidal fold from world coords to storage coords.
FORBIDDEN: `%` on a possibly-negative coordinate (wrong for negatives, and on
some backends signed `%` is undefined - see the `wgsl-gpu-correctness` rule),
or any hand-rolled wrap. The CPU mirror in `raycast.rs` must match this exactly;
a mismatch is silent and position-dependent.

## CPU world queries

**`voxquery::overlap_mat` / `overlaps` / `sweep` / `any_in_voxel_range`**
(`voxquery.rs`) - the ONE way to ask the CPU world a spatial question. Descends
the brick/tile/chunk/L4 pyramid (a brick's `occupancy` is a u64 over 4x4x4, an
empty tile skips 16^3), folds toroidally exactly as the shaders do, and treats
the MASKS as the authority so a recycled streaming slot reads as the sky it is
rendered as. `sweep` tests the whole swept region as one query before bisecting,
so it is exact and cannot tunnel at any speed.
FORBIDDEN: a per-voxel loop at a call site, a hand-rolled `world - origin`
mapping, and reading `Brick::occupancy` without checking the tile bit above it.
Pinned against a naive per-voxel reference on randomised worlds, including
3.2M voxels from spawn across a slot seam.

**`voxquery::MatSet`** (`voxquery.rs`) - a u64 bitset over material ids, folded
from `const fn` predicates at compile time. `SOLID` is the single definition of
"this blocks a body".
FORBIDDEN: an ad-hoc chain of `mat != MAT_WATER && !is_foliage_mat(mat) && ...`
at a call site; extend the predicate instead, so every consumer moves together.

## GPU upload

**`upload_spans`** (`renderer.rs`) - coalesces a sorted dirty list into
contiguous runs and issues one write per run.
FORBIDDEN: one `write_buffer` per dirty element. That exact pattern cost 295
writes and 278 MB/s per frame in the light field before it was replaced.

**`pack_u8_to_u32`** (`renderer.rs`) - packs a `u8` array for a storage buffer,
since WGSL cannot index `u8` directly.
FORBIDDEN: hand-rolled shift/mask packing at a call site.

**`VoxLightBuffers`** (`renderer.rs`) - GPU resources grouped into one named
struct argument.
FORBIDDEN: adding another positional `&Buffer` / `&TextureView` parameter to
`make_compute_bg`. It already takes 22, all interchangeable types, so a
mis-ordered argument compiles cleanly and mis-wires a binding at runtime.

## Light field

**`LightField`** (`voxlight.rs`) - the sparse brick-to-block allocator. Holds
allocation only, never payload, so a block frees in O(1) without touching GPU
memory. Capacity is per instance, which is what lets one tested implementation
serve more than one field.
FORBIDDEN: a second allocator, or reading capacity from a global constant.

**`LightField::block_word_offset`** (`voxlight.rs`) - block index to word offset
for a given record stride.
FORBIDDEN: computing an offset with a hardcoded stride.

**`vl_tap`** (`raymarch.wgsl`) - answers "where is this voxel's light record"
AND "is this cell opaque" from ONE hierarchy descent.
FORBIDDEN: calling a record lookup and a solidity test separately. That is two
descents per tap, eight times per pixel.

## Material classification

**`ao_occluder`** (`raymarch.wgsl`) - "is this cell a REAL occluder", excluding
decoration (grass tufts, flowers, the invisible canopy fringe).
FORBIDDEN: using raw `is_voxel_solid` for an occlusion or shadowing decision.
Treating occupied as opaque is precisely the bug that held foliage light-field
coverage at 45.6 percent and made half a canopy fall back to the per-pixel path.

**`is_foliage_mat`** (`voxel.rs`) - the one predicate for "is this foliage",
used by the shell rule, the update pass and the sampler. Pinned across the
CPU/shader boundary by a test that parses the WGSL.
FORBIDDEN: a second material-class list, in Rust or WGSL.

## Traversal

**`resolve_solid_voxel`** (`raymarch.wgsl`) - the shared sub-voxel resolve that
decides whether a solid voxel is a real visible hit, used by BOTH the software
DDA and the hardware-RT path.
FORBIDDEN: a second resolve path. Two of these drift, and the drift shows up as
RT and software disagreeing pixel by pixel.

## Testing

**`gpu_init_serial()`** (`lib.rs`) - serialises GPU device CREATION across the
parallel test suite.
FORBIDDEN: requesting an adapter/device in a test without holding it. Some
NVIDIA/Vulkan drivers SIGSEGV when several logical devices are created
concurrently, which reads as a random flaky crash.

**`shader_cache`** (`shader_cache.rs`) - persistent pipeline cache with
graceful degradation when the feature, the key or the blob is unusable.
FORBIDDEN: creating a pipeline with `cache: None` on the raymarch module. A
cold driver compile of `cs_transparent` alone is 98 s software / 162 s RT, which
presents as a hung launch.

## Scale

**`world_dims::VOXEL_METRES`** - how big a voxel is in metres, **0.10**. Every
size that means something in the real world is written in SI and converted with
`m_to_vox` (Rust) or `VOXELS_PER_METRE` (WGSL, emitted by `build.rs`), so the
grid can move without retuning the player.
FORBIDDEN: a size expressed directly in voxels. It compiles, renders and passes
its tests while meaning something 2.5x smaller than it says - the 10 cm bump shed
a list of them (`docs/SCALE_TO_10CM.md`): clouds 18 m across instead of 45, surf
5 cm wide instead of 12.5, every biome above 3.6 m of altitude turned to bare
rock, the light field's sun-tracking radius down to 6.4 m, multiplayer relay
range down to 60 m. Two things are NOT lengths and correctly stay in voxels: a
STEP COUNT against the grid (a probe that must not skip a one-voxel wall), and a
sub-cell offset (the water plate's height inside its own cell).

**`world_dims::DISPATCH_ROW_WGS` / `renderer::linear_dispatch`** - the one way to
turn a workgroup count into `dispatch_workgroups` arguments. Tiles over x and y
because every dimension caps at 65,535, which one workgroup per brick passes six
times over at 10 cm.
FORBIDDEN: `n.div_ceil(64)` straight into a dispatch for anything sized by the
world - that is exactly how the GPU physics pass shipped as a validation abort.
Shaders reached through it MUST bounds-check their own index; the tail row
over-dispatches by design.

## Per-voxel light field

**`voxlight::light_record_idx` / `raymarch.wgsl::light_record_idx`** - the record
index for an in-brick voxel. A record covers a `LIGHT_RECORD_STEP^3` CELL, so
several voxels share one.
FORBIDDEN: indexing the pool by `brick_voxel_idx`. It reads past the end of the
block into the next tenant's records, and it shows up as a sun-visibility row
alternating 0/255 down a straight shadow edge.

**`raymarch.wgsl::vl_group_occ` / `vl_group_blocks` / `vl_cell_occluder`** - the
three questions anything can ask about a record CELL: which of its voxels are
occupied, does it hold a real opaque occluder (so light must not interpolate
through it), and does it hold an ambient occluder (so it darkens its neighbours).
All three answer the empty case in ONE storage load and only fetch materials for
the bits that are set.
FORBIDDEN: asking any of those about a single VOXEL when the lattice is coarser
than the voxel. The cell holding a surface is dead by design, so the nearest live
record is one or two voxels off the surface depending on parity, and a per-voxel
probe silently answers for one parity and not the other. That cost a quarter of
all surfaces their direct light and half of all flat ground its contact AO.

## Known gap

`raycast.rs` still reads brick occupancy WITHOUT checking the tile bit above
it, so picking can target a voxel in a recycled streaming slot that is not
rendered. `voxquery` treats the masks as the authority and does not have this
bug. Do not copy `raycast`'s pattern into new code; it is the one to fix, not
the one to follow.
