# Per-voxel light field: shadows, AO, lights and reflections off the pixel

Goal (Marc, 2026-07-31): reflections, shadows, ambient occlusion and lights
stop being per-pixel work and become a per-voxel quantity that shading only
samples. Shadows must come out SMOOTH, not choppy.

## Diagnosis: where the choppiness actually comes from

Shadowing today is a SINGLE BINARY ray per pixel (`raymarch.wgsl:3994`):

    shadow_term = select(0.0, 1.0, !shadow_occluded(p_off, ss, SHADOW_MAX_DIST));

The only thing that makes a penumbra is that the ray direction is jittered per
pixel and per frame inside a `cone = 0.07` disc (`raymarch.wgsl:3982`) and TAA
averages the samples over time. Two consequences:

1. The softness EXISTS ONLY WHERE TAA CONVERGES. `jit_phase` is frozen while
   the camera moves (`raymarch.wgsl:595`), so during motion every pixel takes
   the same offset and the result is a hard binary edge with no penumbra at
   all.
2. A moving sun refreshes shadows as a per-pixel hash-dithered subset of
   pixels (`raymarch.wgsl:712`), so edges crawl as scattered dither rather
   than sweeping cleanly.

So "choppy" is not a tuning problem. It is the direct signature of binary
per-pixel visibility that depends on temporal accumulation for its gradient.
A per-voxel visibility FIELD, continuous in [0,1] and interpolated at sample
time, is smooth by construction: no binary test survives into the pixel.

## Where light lives: the air shell

Light is stored in AIR voxels that touch at least one solid voxel. A surface
hit on voxel `v` with face normal `n` reads the field around `v + n`, the air
voxel against that face.

This beats per-face storage on both counts:
- One record serves all faces touching that air voxel, so no 6x duplication.
- The four air voxels straddling a face edge interpolate into each other, so
  gradients are continuous ACROSS faces and around corners, not just within a
  face. This is what removes the blockiness that a naive per-voxel scheme
  would introduce.

It is the same arrangement the existing AO already implies: `compute_ao`
(`raymarch.wgsl:4417`) evaluates its four corners from `base = v + n_off`,
the adjacent air voxel.

## The record: 8 bytes per air voxel

    word0:  sun_vis u8 | ao u8 | epoch u8 | flags u8
    word1:  point-light radiance, packed RGB9E5

- `sun_vis` is CONTINUOUS soft visibility in [0,1], the fraction of the sun
  disc reaching the voxel. 8 bits is ample once trilinear interpolation
  dithers the steps.
- `ao` is the neighbour-occupancy term, unchanged in spirit from
  `compute_ao`, just evaluated once per voxel instead of once per pixel.
- `epoch` drives the deterministic direction cycling (below) and doubles as
  the "converged yet" marker after an invalidation.
- `flags` records validity and whether the voxel is in the lit shell.

## Sparse storage: a per-brick block pool

A brick is 4^3 = 64 voxels (`world_dims.rs:9`), so a brick's light block is
64 * 8 = 512 bytes. Blocks are allocated ONLY for bricks that contain at
least one lit-shell air voxel. Solid interior bricks and open sky bricks get
nothing, which is the overwhelming majority of the 1,048,576-brick world.

Pool: 131,072 blocks = 64 MiB, with an explicit free list. Sizing rationale:
the terrain shell is roughly a 128x128 brick sheet in xz, and relief, caves,
trees and overhangs spread that over several brick layers, so the resident
lit shell lands in the 65k-130k block range for the streamed window.

Resulting GPU budget alongside what already exists:

    voxel bricks      75.5 MB   (existing)
    light field       64.0 MB   (new)
    GI probe grid     25.2 MB   (existing)

Full voxel resolution is deliberate. Storing at half resolution would be 8x
cheaper but could not resolve a one-voxel step, which is exactly where
contact shadows and AO carry their detail.

## Update pass: amortized and deterministic

A new compute pass `cs_voxel_light_update` slots in directly after the GI
probe update, reusing that pass's proven amortization shape.

- The work list is the compact array of allocated blocks.
- Each frame updates 1/8 of the blocks, strided by a round counter, so the
  whole resident field refreshes in 8 frames.
- Per air voxel in the block:
  - AO from neighbour occupancy. Purely geometric, so it is written once and
    only recomputed when the brick's voxels change.
  - `sun_vis` from K rays across the sun DISC, using a deterministic
    spherical-Fibonacci direction set cycled over epochs exactly as the probe
    grid cycles its 8 rays over 8 epochs (`gi_probes.wgsl:248`). A complete
    cycle folds an exact soft-shadow estimate, so a static scene converges to
    the true penumbra and then stops changing. Determinism is what keeps the
    result stable instead of noisy.
  - Point lights gathered from the light list, radius-culled, shadow-tested.

A moving sun is handled better here than by the current dither: every epoch
re-samples against the CURRENT sun direction, so the field tracks the sun
continuously at the cost of a small lag, instead of flipping scattered
individual pixels.

## Sampling: solidity-gated trilinear

Shading replaces the shadow ray, the AO call and the reprojection lookup with
one fetch:

    light = sample_voxel_light(p_hit, n)

- Sample point `q = p_hit + n * 0.5`, so the trilinear lattice straddles the
  face rather than sitting on it.
- Trilinear over the 8 surrounding records.
- SOLIDITY GATING: any of the 8 that is solid, unallocated or invalid is
  dropped and the remaining weights are renormalised. Without this a
  one-voxel wall leaks light from its lit side to its shadowed side. This is
  the same failure the probe grid solves with its Chebyshev visibility test
  (`gi_probes.wgsl:106`); at voxel resolution the occupancy bit answers it
  exactly and more cheaply.
- If all 8 are unusable, fall back to the probe grid term alone.

## Reflections

Decision (Marc, 2026-07-31): reflections go fully per-voxel, accepting that a
view-independent cache cannot reproduce a true mirror. Recorded honestly: the
half-res reflection experiment was already rejected on looks
(`docs/PERF.md:142`), and per-voxel is a coarser reduction than that was, so
the water is expected to read flatter. Stills get captured and judged rather
than argued about.

Within that decision, the storage is chosen to give the best result a
per-voxel scheme can:
- Only REFLECTIVE voxels (water surface, glass) get a record, so this is a
  small sparse set, not another whole-world array.
- Each stores SH-L1 directional radiance rather than one flat colour. Same
  per-voxel spatial resolution, but evaluating the SH along the reflection
  direction retains coarse view dependence for free instead of collapsing the
  surface to a single wash.
- One reflection ray per reflective voxel per round, folded into the SH.

## Point lights

Scope per decision: dynamic point lights now, emissive block materials later.
A per-frame uploaded list of (position, colour, radius) is gathered into the
`point` word of each record by the update pass. Because the gather happens in
the update pass, cost scales with lit voxels rather than with lit pixels, and
adding lights does not touch the per-pixel path at all.

## What this REPLACES

This is a net simplification, not an addition. It removes:
- the per-pixel jittered shadow cone (`raymarch.wgsl:3970-3994`),
- the per-pixel `compute_ao` call (the function survives, used by the update
  pass),
- the whole screen-space lighting reprojection cache
  (`raymarch.wgsl:676-718`, `light_in`/`light_out`, `pack_light_cache`) and
  its sun-staleness dither.

One world-space mechanism replaces a screen-space cache plus a per-pixel
trace plus a staleness heuristic. No second path is left that can drift.

Secondary rays gain the most: reflection and glass hits currently re-shade
fully and deliberately bypass the cache (`raymarch.wgsl:3545`, "secondary
rays don't use the reprojection cache"), paying a shadow ray, an AO
evaluation and an 8-probe GI gather per hit. They become one field fetch.

## Invalidation

- Brick edits: the existing `dirty_bricks` path clears the block's `epoch` so
  it re-converges instead of showing stale light.
- Streaming: `clear_slot_masks` (`voxel.rs:634`) frees the blocks belonging to
  recycled slots, mirroring how it already clears occupancy masks.
- Origin shift: blocks are keyed by brick index in the toroidal window, so a
  shift invalidates exactly the slots that were recycled and nothing else.

## Verification

Nothing here is judged by eye alone or declared done off a compile.

- AO: the cached value must match the current per-pixel formula. Pixel-diff
  against the existing path before the switch-over.
- Shadows: the smoothness claim is measured, not asserted. Penumbra gradient
  width sampled across a known shadow edge, plus the existing
  `flicker_probe_rt_views` rig for temporal stability.
- Leak check: a one-voxel wall with sun on one side must stay dark on the
  other. This is the failure mode the solidity gating exists for, so it gets
  an explicit test rather than trust.
- Perf: `rt_vs_software_timing` and `VOXELG_BENCH` before and after, measured
  ON THIS MACHINE against a baseline taken on this branch. The numbers
  recorded in `docs/PERF.md` were taken on different hardware and are not
  comparable.
- Look: `dump_lookdev_views` stills read directly, including the water views
  that the reflection decision most affects.

## Staging and current status

1. DONE - block pool, GPU buffers, bind group, invalidation hooks.
2. DONE, WITH A DEVIATION - AO lives in the field, but it is NOT the old
   formula. The per-pixel version evaluated four corners of one FACE and
   bilinearly blended them; a per-voxel field cannot be face-indexed, so
   occlusion is measured for the air cell itself (weighted 6-face + 12-edge
   occupancy) and the smooth gradient comes from the trilinear fetch instead
   of the in-face blend. `ao_occluder` is reused verbatim so decoration cells
   still do not stamp AO squares on the ground. The original plan said "proven
   equal to today"; that was not achievable and the claim is withdrawn rather
   than fudged. Verified instead by pinning the value: ground probes read
   ao 201, which is `1 - 0.85 * (6/24)` to the byte.
3. DONE - soft `sun_vis` with deterministic sun-disc sampling over 8 epochs.
4. DONE - solidity-gated trilinear sampling wired into `shade`.
5. PARTIAL - point lights are gathered by the update pass, uploaded, and
   surfaced through `Renderer::set_point_lights`. Nothing in the game calls it
   yet, and there is no test covering a lit point light.
6. NOT STARTED - per-voxel reflections.
7. NOT STARTED - secondary rays read the field.
8. IN PROGRESS - baseline captured in `docs/rt/BASELINE-per-voxel-lighting.md`.

### Known follow-ups found while building

- TEMPORAL STABILITY IS UNMEASURED. Each round folds a 4-ray estimate at
  `fold = 0.35`, an effective window of roughly three rounds or twelve rays, so
  the stored value may shimmer between rounds even though the SPATIAL gradient
  measures clean. The `flicker_probe_rt_views` rig exists precisely for this
  and has not been run against the field yet. If it shimmers, the fix is the
  probe grid's shape: accumulate a complete epoch cycle in a staging slot and
  fold only finished estimates (`gi_probes.wgsl:255`), rather than lowering the
  fold and adding lag.
- AO IS RECOMPUTED EVERY ROUND for no reason. It is purely geometric, so
  eighteen occupancy lookups per voxel per round are repeated work; it only
  needs recomputing when the record is reset or its brick is edited.
- POOL SIZING IS UNVALIDATED AGAINST A REAL STREAMED WORLD. The 131072-block
  ceiling was derived from the shape of the terrain shell, and the crafted test
  worlds bind only ~2300 blocks. `take_overflow` will log if it is short, but
  nobody has yet flown a real world far enough to find out.

### What is NOT yet true

The old per-pixel machinery is still present and still runs as the fallback
for any voxel the field cannot answer for (no block, or a block that has not
converged). The plan called for DELETING the per-pixel shadow cone, the
per-pixel `compute_ao` call and the whole screen-space reprojection cache.
None of that is removed yet, so the promised simplification - one world-space
mechanism instead of a screen-space cache plus a per-pixel trace plus a
staleness heuristic - has not landed. Two paths still exist and can drift.
Removing them is only safe once the field is proven to cover the cases they
handle, which is what the measurement stage is for.
