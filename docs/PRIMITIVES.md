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
FORBIDDEN: a literal 512 / 128 / 64 / 4 for a world dimension anywhere, and any
second copy of these values in WGSL.

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

**`world_dims::VOXEL_METRES`** - how big a voxel is in metres, 0.25 today. All
gameplay sizes are written in SI and converted with `m_to_vox`, so the planned
10 cm change (`docs/SCALE_TO_10CM.md`) moves the grid without retuning the
player.
FORBIDDEN: a size expressed directly in voxels, and any assumption that a voxel
is 10 cm. That assumption is already wrong, and it once put a 1.8 m player at
0.72 m tall.

## Known gap

`raycast.rs` still reads brick occupancy WITHOUT checking the tile bit above
it, so picking can target a voxel in a recycled streaming slot that is not
rendered. `voxquery` treats the masks as the authority and does not have this
bug. Do not copy `raycast`'s pattern into new code; it is the one to fix, not
the one to follow.
