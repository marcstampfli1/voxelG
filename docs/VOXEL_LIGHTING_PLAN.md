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
64 * 8 = 512 bytes. Blocks are allocated ONLY for bricks that can hold a
lit-shell AIR voxel. Solid interior bricks and open sky bricks get nothing,
which is the overwhelming majority of the 1,048,576-brick world.

Pool: 131,072 blocks = 64 MiB, with an explicit free list.

MEASURED against the demo world (`World::fill_demo_terrain`, a fully
generated 512x256x512 window), not estimated:

    1,048,576 bricks   699,991 empty   38,040 partial   310,545 fully solid
    lit shell bound     60,174 blocks = 30.8 MB, 46% of the pool

So the whole streamed window is covered with 2.18x headroom and nothing falls
back to the per-pixel path for want of storage. Pinned by the test
`the_demo_world_light_shell_fits_the_pool_and_still_covers_it`
(`src/voxlight.rs`), which asserts both halves: it fits, AND it still covers
every air voxel that touches solid.

That measurement was taken only after a real defect. `brick_needs_light`
originally bound a block for every NON-EMPTY brick, which includes every
fully solid one, so the demo world asked for 370,719 blocks and 239,647
requests were refused. Because `allocate` hands out blocks in brick-index
order and `brick_idx` is z-major, that did not thin out evenly: it filled a
solid slab over world z 0..192 of 512 and left every camera past it on the
old per-pixel path, a hard geographic cliff rather than graceful degradation.
A block on a fully solid brick is dead by construction - the update pass
writes epoch 0 for every solid voxel and `voxlight_sample` drops every tap
that lands in one - so those 310,545 blocks were five sixths of the storage
and five sixths of the update dispatch, buying nothing any pixel could read.
The fix was to stop binding them, not to buy a bigger buffer.

Full voxel resolution is deliberate. Storing at half resolution would be 8x
cheaper but could not resolve a one-voxel step, which is exactly where
contact shadows and AO carry their detail.

## GPU memory budget

World-resident, independent of resolution:

    voxel bricks               75.5 MB   1,048,576 * 72 B
    light pool                 67.1 MB   131,072 blocks * 512 B (60,174 live)
    light brick->block table    4.2 MB   one u32 per brick
    light work list             0.5 MB   plus a 2 KB urgent tail
    GI probe grid              25.2 MB   131,072 probes * 192 B
    occupancy pyramid + hints   1.2 MB
    ------------------------------------
                              173.7 MB

Screen-space, at 1920x1080 in the shipped RT + probe-GI configuration:

    transp records + godray scratch                  41.4 MB
    GI accumulation (gi_in + gi_out, Rgba32Float)    66.4 MB
    lighting G-buffer (Rgba32Float, ONE now)         33.2 MB
    HDR scene + geometry (Rgba16Float)               33.2 MB
    depth, LDR, resolve, bloom, cloud, beam          33.4 MB
    ------------------------------------------------------
                                                    207.6 MB

Total ~381 MB, plus the RT acceleration structures (not accounted here). The
light field is 19% of that and the SECOND largest world-resident item after
the brick pyramid it sits beside, which is proportionate for the thing that
carries all shadowing, AO and local light.

Both totals came down when the reflection field was deleted (2026-08-01): 21.1 MB
of world-resident pool and table, and 33.2 MB of screen-space reflection history
in `transp_buf`, which stylized water has nothing to accumulate into. The
screen-space column came down another 33.2 MB when the lighting reprojection
CACHE was deleted (2026-08-02): the G-buffer stays for cs_taa and the grass pass,
its previous-frame copy does not - along with the full-res Rgba32Float
copy_texture_to_texture that fed it on every single frame.

It would NOT have been proportionate the other way round, and that is
measured rather than argued. The 393,216-block pool the broken membership
rule needs (201.3 MB, i.e. 329 MB of world data and ~604 MB total) was built
and benchmarked: it binds all 370,719 blocks, and every FIELD A/B delta comes
out the same to within noise (terrain -0.92 vs -0.92 ms, foliage -0.00 vs
-0.00, water-close -2.86 vs -2.91, terrain-covered -2.11 vs -2.10). Nothing
renders differently, exactly as the "solid taps are dropped" argument says it
should. What differs is +134 MB of VRAM and a light update pass at 1.41 ms
instead of 1.19 ms per frame - +0.22 ms, +18%. The extra blocks are cheap
rather than free because `cs_voxel_light_update` early-outs on a solid voxel
before tracing anything, so 6.2x the blocks costs 1.18x the time; the price
is overwhelmingly memory, not milliseconds. Either way it is a price paid for
records nothing can read.

If a future world class genuinely outgrows 131,072 lit-shell blocks, the
answer is still not a bigger buffer: it is to bound allocation to a radius
around the camera, so cost tracks what can be seen rather than what has been
generated. The pool is already a free list with O(1) release
(`LightField::release`), and streaming already frees recycled slots, so that
bound is a policy change in `sync_light_shell_*` rather than a storage
redesign. `LightField::overflow_total` now makes the trigger for it visible.

The REFRESH half of that idea is already there (`World::set_light_focus`, round
G), and it deliberately stops short of bounding ALLOCATION: a block outside the
radius keeps its storage and its converged record and is only revisited more
rarely, so nothing falls back to the per-pixel path and no boundary is visible.
Bounding allocation would put a ring in the world where the two shading paths
meet, and that is a different, larger decision.

## Update pass: DEMAND-DRIVEN, amortized and deterministic

The compute pass `cs_voxel_light_update` runs FIRST in the frame - before the
cloud and beam pre-passes and outside the temporal-differential gate, because the
field is world state and its queue is drained on the assumption that the frame
dispatches it.

It has exactly TWO reasons to run, and when neither applies it is not encoded at
all: no compute pass, no uniform write, nothing.

1. THE URGENT LIST (`LightField::urgent`). Bricks whose block has no readable
   record - newly bound, or just invalidated by an edit. Their pixels are shading
   through the fallback until they are gathered, so they are dispatched IN FULL
   on the next frame, up to `LIGHT_URGENT_BUDGET` per dispatch so a chunk install
   cannot turn into one enormous launch. One visit converges them exactly (a
   reset record has no history, so it takes the fresh estimate outright). Cost
   scales with how much of the world actually changed.

   This replaces `promote_near`, which pulled the same blocks into the near TIER
   and left them there: promotion was one-way, so a still camera watching physics
   dragged the whole shell into the near group one block at a time, and even then
   a promoted block waited up to `update_div` rounds for its slice. Worst-case
   latency after an edit goes from 8 frames to 1, and the tier boundary stops
   drifting - which also removed the "near group is crowded" repartition trigger
   that existed only to undo the drift.

2. THE SWEEP, paced by SUN MOTION (`VoxLightSchedule`). The work list is
   PARTITIONED near-first (`LightField::near_count`, shell within
   `World::LIGHT_NEAR_RADIUS` of the camera); a round refreshes 1/8 of the near
   group and 1/64 of the far one, strided by a round counter, from the same
   dispatch. A round is issued when the sun has turned
   `VOXLIGHT_SUN_RAD_PER_ROUND`, NOT once per frame, plus a floor of one complete
   sweep per second while geometry is still settling (a placed block moves a
   shadow that can land anywhere along the sun ray, and nothing local can know
   where).

   At 60 fps and the shipped sun that is one round per frame, i.e. exactly the
   cadence this pass always had, so the LOOK is unchanged. At 400 fps it is one
   frame in seven. With the sun frozen and nothing dirty it is nothing at all.

   The justification is measured, not argued: over a whole near sweep period of
   sun motion, 99.25% of the demo world's 3.85M records come back BIT-IDENTICAL
   (`voxlight_sun_lag_error`). Re-gathering them was not slightly wasteful, it
   was almost entirely redundant. Round H of the baseline has the table, the
   per-frame-rate dispatch counts, and the two numerical defects the gate's tests
   caught (`acos` of a unit vector with itself reads 3.4e-4 rad of motion; and
   `sun_dir_at` turns at 0.02395 rad/s, not 0.025, because it normalizes a coned
   vector).

The near/far split is a refresh RATE and nothing else: every block keeps its
storage and its converged record, so shading reads the same field and no pixel
changes path. It is DISTANCE ONLY, deliberately - scoping it by the view frustum
as well was built and measured and rejected, because it doubles the peak error on
the frame you turn around for 8-12% of one pass. See `World::set_light_focus` and
round H.
- Per air voxel in the block:
  - AO from neighbour occupancy. Purely geometric, so it is written once and
    only recomputed when the brick's voxels change.
  - `sun_vis` from K rays across the sun DISC, over a COMPLETE sunflower
    stratification of it, evaluated in full on every visit.

    IT USED TO CYCLE A QUARTER OF THE SET PER VISIT over epochs, exactly as the
    probe grid cycles its 8 rays (`gi_probes.wgsl:248`), on the claim that "a
    complete cycle folds an exact soft-shadow estimate, so a static scene
    converges to the true penumbra and then stops changing". THAT CLAIM WAS
    FALSE and it is withdrawn. A penumbra voxel's four quarters are four
    different numbers rather than four noisy looks at one, so the fold tracked
    them instead of averaging them: a converged record swung 89/255 on a static
    scene under a static sun, and `shade_water_top` reads it through a smoothstep
    only 45/255 wide, so water at a shadow edge strobed between lit and
    shadowed. This is the same defect round E found in the (now deleted)
    reflection field, and it was never applied here. Measured, fixed and pinned
    by `voxlight_sun_is_stable_once_converged`; see round G of the baseline.
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

## Reflections: BUILT, MEASURED, AND DELETED

Decision (Marc, 2026-07-31): reflections go fully per-voxel. Decision (Marc,
2026-08-01): they come out entirely, and water becomes deliberately stylized
instead. The whole per-voxel reflection field is gone - the update pass, its
pipelines, its pool and tables, its three group-0 bindings, `voxlight_reflection`
and its four tests - along with the per-pixel mirror trace it fell back to and
the full-res reflection history that fed it. Glass keeps its own per-pixel
mirror, which predates all of this.

It is recorded here rather than deleted from the record, because the reason it
came out is a RESULT and not a preference.

The storage was chosen to give the best result a per-voxel scheme can: only
reflective voxels got a record; each held SH-L1 directional radiance rather than
one flat colour, so evaluating along the reflection direction kept coarse view
dependence; sampling was bilinear in the surface plane, which is the difference
between a surface and a visible grid of tiles. Three real defects were found and
fixed in it (a random-walking estimator, a moment matrix solved with the wrong
measure, and a fit evaluated about the shading normal instead of the gather
axis), and after those fixes it was measurably STABLER than the per-pixel mirror
it replaced - 26x fewer strongly flickering pixels on a grazing water view.

What could not be fixed is the basis. The sky's luma over an elevation sweep runs
0.83 (horizon) -> 0.54 (30 deg) -> 0.83 (70 deg) -> 0.75 (zenith): it has an
INTERIOR MINIMUM. An L1 reconstruction restricted to a great circle of elevations
is `a + R*cos(e - phi)`, which over [0,90] admits an interior MAXIMUM only, so a
linear fit cannot be bright at both ends with a dip between them and the
least-squares answer is to go flat. Measured, the cache read 0.60 -> 0.65 -> 0.64
across that sweep, a span of 0.06 against the sky's 0.30, and it ran between
0.67x and 1.96x the true mirror. Over the band where the reflection carried scene
content it kept 62.6% of the mirror's spatial contrast at r = 0.76: recognisable
trees became voxel-scale blobs. Raising that needs an L2 term - 9 coefficients
per channel instead of 4, roughly 37 MB of pool instead of 16.8 MB - and that is
a memory price for a photoreal effect the art direction no longer wants.

So the water was not made to reflect better. It was made to stop reflecting.

## Water instead: stylized, faceted, lit-or-shadowed

The replacement is not "the reflection term with a cheaper reflection in it". It
is a different surface.

- ONE FLAT FACET PER CELL. Every surface water voxel renders one horizontal
  plate at one quantized height (7 bands over +-0.24 voxels) with one quantized
  normal (5 slope steps per axis), both from the same Gerstner field sampled at
  the CELL CENTRE. Every pixel of a cell shades identically by construction. The
  wave still travels: a cell steps between bands as the wavefront passes.
- THE CORNER MACHINERY IS GONE with the surface it existed for. Pins, the
  step-down fold, the two-triangle split and the separate near/far tiers were
  all there to make neighbouring cells share exact corner heights so the lake was
  ONE watertight sheet. A faceted lake does not want to be one sheet. What it
  must not have is holes, and it does not: where a neighbour's plate stands
  higher the ray enters this cell BELOW its own plate and the entry face is the
  hit. Cost per surface cell went from up to 24 neighbour probes to 4, and those
  4 only inside foam range.
- LIT OR SHADOWED, not reflective, and AS SMOOTHLY AS TERRAIN.
  `shade_water_top` reads the per-voxel sun visibility the light field already
  stores - sampled about world +Y, because the record that matters is the one
  directly over the cell - and multiplies by it DIRECTLY, exactly as terrain
  does. Water in sun reads bright; water in full shadow reads at 0.42 of it.

  It used to push that value through `smoothstep(0.41, 0.59, sun_vis)` first, a
  band 45/255 wide, on the argument that a photoreal penumbra reads as a smudge
  over faceted geometry. That was wrong twice: it re-quantized the exact gradient
  the field exists to produce, so water read STEPPED beside terrain that read
  smooth (Marc, 2026-08-02); and being only 45/255 wide it turned any wobble in
  the stored value into a full lit/shadowed flip, which is the mechanism round G
  traced the water flicker report to. Measured on a wall cast across a sheet, the
  0.72-0.90 transition band goes from 1.1% to 4.0% of the shaded pixels and the
  shadowed side sits at 0.330 of the lit side.

  The stylization was never in this term: the plate quantization, the per-cell
  tone ladder and the hard-edged foam carry it, and none of them moved.
- THE FRESNEL BLEND TOWARD SKY IS KEPT, and capped at 0.45. Kept because
  without it a lake is one flat blue field from the shore to the horizon - the
  facet ladder gives cell-scale texture but nothing changes with view angle, so
  it reads as paint. Capped because the raw Schlick curve runs to 1.0 at graze
  and turned the terrace lab's low camera into a sheet of milky horizon sky with
  every facet washed out of it. It blends toward `fog_atmospheric`, the
  emitter-free sky, so a facet stepping into the mirror direction cannot pop the
  sun disc or a star.
- FOAM IS PER-CELL AND HARD-EDGED. A cell foams at a wave crest (top bands, and
  only half of those - see below) or at the edge of the water body. The stamp is
  authored ASCII art in `src/sprites.rs` following the same convention as the
  foliage sprites, but laid out as a LIBRARY of sixteen 4x4 shapes that a
  per-cell hash indexes, at quarter-voxel texels. Foam takes three values -
  none, body, highlight - and there is no ramp anywhere in it.

Two numbers here were measured rather than assumed, and both changed the design:

- CREST FOAM IS AMPLIFIED BY GRAZING ANGLE. A crest cell's plate stands 0.16
  voxels proud and a shallow ray stops at the FIRST plate it dips below, so
  crests occlude the troughs behind them. Band >= 2 is 13.9% of cells by area
  (Monte Carlo over the four waves) but covered 24% of a low-angle crop - five
  times its footprint share, and it read as a white chequerboard. Only half of
  crest cells now break.
- THE SHORE TEST IS "NOT WATER", NOT "SOLID". A lake whose rim sits below its
  surface - water at y on ground at y-1, the common case - has AIR at every
  lateral neighbour, so a solid-only test foamed nothing at all on it.

## Point lights

Scope per decision: dynamic point lights now, emissive block materials later.
A per-frame uploaded list of (position, colour, radius) is gathered into the
`point` word of each record by the update pass. Because the gather happens in
the update pass, cost scales with lit voxels rather than with lit pixels, and
adding lights does not touch the per-pixel path at all.

## What this REPLACES - and the half of it that was wrong

The plan said this removes three things. It removes ONE of them, and the reason
the other two stay is a measurement.

REMOVED: the whole screen-space lighting reprojection cache - binding 15, the
previous-frame G-buffer, `REPROJ_EPS2`, the sun-staleness dither, the four reuse
flags threaded through `shade`, 33 MB of VRAM and a full-resolution Rgba32Float
ping-pong COPY every single frame. It had to go for two reasons and only one of
them is performance:

- IT WON OVER THE FIELD. `reuse_shadow` was tested BEFORE `vlf.valid`, so on a
  still camera - the only state it engages in - a smooth world-space gradient was
  overwritten by a value some earlier frame had traced from ONE binary ray.
  Shadows visibly changed character when the camera stopped moving. That is
  precisely the "two paths that can drift" this section was written to prevent,
  and it was live for the whole of rounds A-G.
- IT COST MORE THAN IT SAVED. Timed on a still camera with it compiled out,
  cs_main went 4.73 -> 4.30 ms on terrain (the cache COST 9.9%) and moved within
  noise on foliage and water. With the field answering 96.6% of terrain pixels,
  reprojection was re-deriving what it had already been given.

`light_out` STAYS and the plan was wrong to lump it in: it is also cs_taa's
reprojection source and the grass pass's blade lighting. Only the history it was
copied into, and everything that read it, is gone.

KEPT: the per-pixel jittered shadow cone and the per-pixel `compute_ao` call, as
a FALLBACK for the voxels the field cannot answer for. Measured coverage of
shaded pixels: terrain 96.6%, water 99.9%, FOLIAGE 45.6%. The air cell against a
leaf face is usually another leaf voxel, so the sampler's solidity gate drops all
eight taps and the field has nothing to give for over half a canopy view -
deleting the fallback would render it black. It costs nothing when unused (both
branches sit behind `vlf.valid`), which is the condition this document set for
keeping it. It also explains round D's "foliage saves nothing": on that camera
the field answers for under half the pixels.

So the frame carries ONE lighting mechanism plus a fallback for what that
mechanism cannot see, rather than two mechanisms that overlap and disagree.

Secondary rays gain the most: refraction and glass hits re-shade fully and
deliberately bypass the cache ("secondary rays don't use the reprojection
cache"), paying a shadow ray, an AO evaluation and an 8-probe GI gather per hit.
They become one field fetch. Water's own reflection ray no longer exists to
gain anything (see "Reflections: BUILT, MEASURED, AND DELETED").

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
  that the water decisions most affect (`water_view`, `water_graze`,
  `water_terrace`, and `water_shadow`, which was ADDED for the stylized rework
  because no natural camera in the demo world has shadowed open water in
  frame).

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
6. BUILT, THEN DELETED - per-voxel reflections. Rounds D and E built it,
   measured it, fixed three real defects in it and then measured the residual
   as a hard limit of the L1 basis rather than a bug. Round F removes it and
   restyles the water. See "Reflections: BUILT, MEASURED, AND DELETED" above
   for what was learnt and why it did not survive; the numbers are in rounds
   D, E and F of `docs/rt/BASELINE-per-voxel-lighting.md`.
7. DONE BY CONSTRUCTION - secondary rays read the field. Refraction and glass
   hits are shaded through the same `shade`, which fetches `voxlight_sample`
   (`raymarch.wgsl`, search `let vlf =`), so a secondary hit pays one field
   fetch instead of a shadow ray plus an AO evaluation. Water's own surface now
   reads the field DIRECTLY, in `shade_water_top`, for its lit/shadowed term.
8. DONE - rounds A through H in `docs/rt/BASELINE-per-voxel-lighting.md`,
   including the populated-field A/B that rounds A-C could not measure, the live
   session round G could not see, and the sky-facing frame and pixel-coverage
   fraction round H added.
9. DONE - the pass does no work when nothing changed. Sun-paced sweep plus an
   urgent list, a frozen sun costing zero, and the frame-rate coupling removed.
   Round H.

### Known follow-ups found while building

- THE REFLECTION FIELD'S OWN FOLLOW-UPS ARE CLOSED BY DELETION, not by fixing.
  Three of them were real and were fixed before the deletion (the random-walking
  estimator, the moment matrix solved with a cosine-weighted measure while the
  update pass sampled uniformly, and a fit evaluated about the shading normal
  instead of the gather axis); two were still open when the field came out (the
  reflection pass's dispatch shape, 33 workgroups instead of 257 after the
  estimator reshape, and the L2 basis the grazing residual would need). All of
  it is recorded in "Reflections: BUILT, MEASURED, AND DELETED" above and in
  rounds D and E of the baseline, because the reasoning is worth keeping even
  though the code is not.
- THE UPDATE PASS IS CAMERA-AWARE, and this entry is CLOSED. It used to walk
  the whole resident shell every eight frames whether or not any of it was on
  screen, which on a streamed world made it the LARGEST single GPU pass in the
  frame - 0.95 to 2.09 ms, 18-32% of GPU time, ahead of the raymarch on four of
  six benchmark segments. The work list is now partitioned near-first and the far
  group refreshes eight times more rarely. Round G of the baseline has the
  numbers, the two triggers that keep the partition honest, and the test that
  proves both groups converge to the same field.
- THE PASS RUNS ONLY WHEN SOMETHING CHANGED, and this entry is CLOSED. It used to
  dispatch a full sweep round EVERY frame regardless, which at 400 fps refreshed
  the field six times more often per second than the design ever asked for and
  never stopped at all. It is now paced by sun motion with an urgent list for
  everything that cannot wait, and a frozen sun over unchanged geometry dispatches
  nothing. Round H of the baseline.
- AO IS RECOMPUTED EVERY ROUND for no reason. It is purely geometric, so
  eighteen occupancy lookups per voxel per round are repeated work; it only
  needs recomputing when the record is reset or its brick is edited. STILL OPEN,
  and cheaper than it was: with the sweep paced by the sun, a frozen-sun scene
  recomputes it zero times per frame instead of once per block per eight frames.
- POOL SIZING IS NOW VALIDATED, and it was short. See "Sparse storage" above:
  the membership rule bound a block for every fully solid brick, the demo
  world overflowed by 239,647 requests, and the loss was a contiguous z slab
  rather than an even thinning. Fixed at the rule, and pinned by
  `the_demo_world_light_shell_fits_the_pool_and_still_covers_it`. The
  overflow report was also part of the failure: it fired once, at `warn`, on
  the init frame and then read zero forever after, so several benchmark rounds
  ran saturated without it being noticed. It is now `log::error!` and quotes a
  running total (`LightField::overflow_total`).
- THE GRAZING QUESTION IS SETTLED, AND THE ANSWER IS STRUCTURAL. `cs_vl_probe`
  now returns `sky(refl_dir)` and the per-pixel mirror alongside the
  reconstruction, which is the discriminating check this entry asked for
  (`voxlight_refl_diagnose`, section A). It is cause (a), the fit, and not
  cause (b), a genuinely dim sky - but the mechanism is sharper than "under-
  reconstructing toward the rim", and it is a hard limit rather than a tuning
  error.

  Sweeping elevation over an open-sky point, the true sky luma runs
  0.83 (horizon) -> 0.62 (19 deg) -> 0.54 (30 deg) -> 0.83 (70 deg) ->
  0.75 (zenith). It has an INTERIOR MINIMUM. The L1 reconstruction restricted
  to a great circle of elevations is

      a + b.d  with  d = (0, sin e, +-cos e)   =>   a + R*cos(e - phi)

  and over e in [0,90] that sinusoid admits an interior MAXIMUM only; its
  minimum over the interval always sits at an endpoint. A linear fit therefore
  cannot be bright at both the horizon and the zenith with a dip between them,
  and the least-squares answer to a shape it cannot express is to go FLAT. That
  is exactly what is measured: the cache reads 0.60 -> 0.65 -> 0.64 across the
  same sweep, a span of 0.06 against the sky's 0.30, and it errs in the
  predicted direction at every point - too DARK at the two bright ends
  (horizon, zenith) and too BRIGHT in the dip. Over the whole sweep the cache
  sits between 0.67x and 1.96x the per-pixel mirror.

  So the residual is not a defect to be fixed at the fit; it is the L1 basis.
  Raising it needs an L2 term (the quadratic is precisely what "bright at both
  ends, dark in the middle" requires), which is 9 coefficients per channel
  instead of 4 and roughly 37 MB of reflection pool instead of 16.8 MB. That is
  a design decision with a memory price, not a bug fix, and it is NOT taken
  here. What IS fixed is the separate arithmetic defect found next to it: the
  reconstruction solved its moment matrix with `E[(d.t)^2] = 1/4`, the
  cosine-weighted value, while the update pass samples uniformly in solid angle
  and the rest of the same solve used the uniform `E[d.n] = 1/2` and
  `E[(d.n)^2] = 1/3`. That over-weighted the tangential term by exactly 4/3, so
  every horizontal swing of the reflection - the entire view-dependent part -
  came out a third too strong, and it amplified the estimator's swing as well
  as its steady-state error. See `voxlight_reflection` in `raymarch.wgsl`.

### What is NOT yet true

Point lights (stage 5) are gathered, uploaded and surfaced through
`Renderer::set_point_lights`, and nothing in the game calls it. There is no test
covering a lit point light.

Everything else this section used to list is closed by round H, and the closure
is a measurement rather than a claim: the screen-space reprojection cache is
gone, the per-pixel cone and `compute_ao` stay as a fallback because the field
answers for only 45.6% of a canopy view, and the field's coverage is now reported
by `LiveFrameRig::coverage` on every run of the live-session profiler rather than
assumed.

Water is the one surface where that fallback is a SINGLE binary ray rather than
the cone: `shade_water_top` takes `vlf.sun` when the field answers and one
`shadow_occluded` from just above the cell's top face when it does not. The
crafted test scenes bind no shell, so that path is exercised on every run rather
than left to rot.
