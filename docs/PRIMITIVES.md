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

## Pending

The CPU AABB voxel-query primitive for collision is being built now (see
`docs/SURVIVAL_PLAN.md`). Until it lands, the only CPU solidity test is
`in_solid` inline in `raycast.rs`, which walks voxel by voxel and does NOT use
the occupancy pyramid. Do not copy that pattern into new code.
