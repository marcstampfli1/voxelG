# Baseline and measurement rounds for the per-voxel lighting rework

The BEFORE-baseline below was taken on `feat/per-voxel-lighting` at commit
a113962 (the manifest fix only, no lighting change yet). THIS IS THE ONLY
VALID COMPARISON POINT for the rework: the numbers in docs/PERF.md were
measured on different hardware and are not comparable to anything measured
here. Rounds A through F follow it: round D is the one that measures the feature
rather than the renderer around it, and round F is the one that deletes half of
it and restyles the water.

Machine: winpc (Windows 11). Harness:
`cargo test --lib rt_vs_software_timing -- --nocapture --ignored`, test profile
(opt-level 1), 1920x1080. Total harness wall time 304 s.

Shipped configuration is RT-primary + probe GI, so that column is the one the
rework must beat.

| scenario    | software | RT-prim+PROBE-GI | main  | transp | compose |
|-------------|----------|------------------|-------|--------|---------|
| terrain     | 13.85 ms | 14.38 ms         | 13.55 | 0.07   | 0.27    |
| foliage     | 14.04 ms | 15.83 ms         | 14.89 | 0.07   | 0.28    |
| water-close |  6.75 ms |  8.16 ms         |  4.69 | 3.25   | 0.34    |

Static-camera (tile-gated) totals: terrain 14.54, foliage 15.89,
water-close 7.71 ms.

## Where the time actually goes (the number that matters)

The harness separates primary traversal from shading by running a
trace-only variant:

| scenario    | primary trace | SHADE   |
|-------------|---------------|---------|
| terrain     | 4.42 ms       | 9.24 ms |
| foliage     | 5.93 ms       | 11.79 ms |
| water-close | 2.23 ms       | 1.63 ms |

Shading is 60-70% of the frame on terrain and foliage. That is the budget the
per-voxel light field is aimed at: the shadow ray, the AO evaluation and the
probe gather all happen per pixel today and all become one field fetch.

Water-close is the opposite shape: shading is only 1.63 ms because most of the
cost sits in the transparent pass (3.25 ms), where reflection hits are
re-shaded from scratch and deliberately bypass the reprojection cache
(raymarch.wgsl:3545). That pass is what stages 6 and 7 target.

## Measurement rounds so far (RT-prim+PROBE-GI, the shipped config)

| round                          | terrain  | foliage  | water-close |
|--------------------------------|----------|----------|-------------|
| A. baseline (a113962)          | 14.38 ms | 15.83 ms | 8.16 ms     |
| B. field in, no early-out      | 15.63 ms | 17.00 ms | 8.86 ms     |
| C. field in, with early-out    | 13.62 ms | 15.17 ms | 7.93 ms     |
| D. POPULATED field, pool fixed | 12.71 ms | 15.15 ms | 4.97 ms     |

Round B is the honest cost of a MISSING early-out: `voxlight_sample` ran its
full eight-tap gated loop and paid an `is_voxel_solid` descent plus a brick
table lookup per tap before failing. Terrain shade went 9.24 -> 10.01 ms.
Round C checks the centre voxel first and answers a miss in one lookup.

READ ROUNDS A-C CAREFULLY. They are NOT evidence that the light field is
faster. The timing harness built its own bind groups with an unpopulated
`VoxLightBuffers`, so in all three the field is empty, `voxlight_sample`
returns invalid, and shading falls back to the previous per-pixel path. All
three measure the SAME renderer, plus or minus the cost of asking the field a
question it cannot answer.

Round-to-round variance is also large: the software column alone reads
13.85 / 15.11 / 13.16 / 11.84 ms for terrain across A/B/C/D, about 15% spread
on a path whose behaviour did not change. Neither round C nor the D column
above may be read as a win against A on its own; only the FIELD A/B in round
D holds everything but the field constant.

What round C does establish: the field is performance-NEUTRAL when it is not
populated, which is the precondition for it being a win when it is.

## Round D: the A/B that matters, finally run

The harness now converges a POPULATED field before it times anything, and
times the SAME frame twice - once against the populated `VoxLightBuffers` and
once against an empty one - so the difference is the feature and nothing else.
Two defects had to be fixed first, or the comparison would still have been
meaningless:

- The demo world asked for 370,719 light blocks against a 131,072 ceiling
  because `brick_needs_light` bound a block for every FULLY SOLID brick too,
  and blocks are handed out in z-major brick order, so the field covered a
  slab of world z 0..192 of 512 and the terrain and foliage cameras were
  looking at storage that did not exist. It now binds 60,174 blocks, refuses
  nothing, and covers brick z 0..=127, i.e. the whole window.
- `voxlight_reflection` read the reflection field with a nearest-voxel lookup,
  which put a visible tile grid on the water. Now bilinear in the surface
  plane.

    voxlight field: light 60174 / 131072 blocks, refl 2050 / 16384 blocks
    light blocks cover brick z 0..=127 (world z voxels 0..512 of 512)

### FIELD A/B, populated vs empty, identical frame

| scenario        | populated       | empty field | delta              |
|-----------------|-----------------|-------------|--------------------|
| terrain         | 12.71 / 12.72   | 13.64 ms    | -0.92 ms (-6.7%)   |
| foliage         | 15.15 / 15.16   | 15.16 ms    | -0.01 ms (-0.0%)   |
| water-close     |  4.97 /  4.99   |  7.89 ms    | -2.91 ms (-36.9%)  |
| terrain-covered |  7.74 /  7.72   |  9.83 ms    | -2.10 ms (-21.3%) |

Both populated runs are printed because they bracket the empty run in time,
so a drifting GPU clock shows up as a spread between them. It does not: the
two agree to 0.02 ms everywhere, which is what makes these deltas readable at
all.

### The per-frame price, which those columns do NOT include

    light pass 1.19 ms + reflection pass 0.23 ms = 1.39 ms

The shipped frame runs one round of each pass before the raymarch, so 1.39 ms
is what has to come off the shading side's saving. Against that:

| scenario        | shading saves | update costs | NET      |
|-----------------|---------------|--------------|----------|
| terrain         | 0.92 ms       | 1.39 ms      | +0.47 ms |
| foliage         | 0.01 ms       | 1.39 ms      | +1.38 ms |
| water-close     | 2.91 ms       | 1.39 ms      | -1.52 ms |
| terrain-covered | 2.10 ms       | 1.39 ms      | -0.71 ms |

So the feature is NOT a uniform win, and the headline "-36.9% on water" would
be a lie by omission. It pays for itself on water and on close terrain, and it
LOSES on the two wide-open cameras - decisively on foliage, where it saves
nothing measurable at all.

Foliage saving nothing is the sharpest result here and it has a cause in the
code, not in the noise: `shade` skips AO entirely for leaf materials
(`skip_ao` covers `is_leaf_block_mat` and `MAT_LEAF_FRINGE`), so on a screen
that is mostly canopy the field replaces only the sun-visibility ray, and the
per-pixel version of that ray is already cheap against foliage because it hits
something immediately. The field's fixed 1.39 ms is then pure cost.

Two things follow, neither of them done yet. The update pass is amortized over
`VOXLIGHT_UPDATE_DIV` frames but is NOT camera-aware, so it pays for the whole
resident shell every frame whether or not any of it is on screen; and the
per-pixel path it was meant to replace is still there, so the frame carries
both mechanisms. The saving column above is therefore the field's SUPPLEMENT
value, not its replacement value - see "What is NOT yet true" in
docs/VOXEL_LIGHTING_PLAN.md, which is exactly the work these numbers argue
for.

### Full round-D output

    rt_vs_software_timing [terrain] 1920x1080: software 11.84 ms | RT-occl 11.39 (1.04x)
      | RT-primary 12.02 (0.98x sw) | swPrim+perpixelGI 18.90 (0.63x) | RT-prim+perpixelGI 19.65 (0.60x)
      | RT-prim+PROBE-GI 12.71 (0.93x sw)
      per-pass SW: main 11.60 transp 0.06 compose 0.26 || RT-prim+GI: main 11.98 transp 0.07 compose 0.27
      primary-trace-only: SW 4.37 || RT 4.02  (=> SW shade 7.26, RT-prim+GI shade 7.97)
      static-cam 12.88 ms
    rt_vs_software_timing [foliage]: software 13.44 | RT-occl 12.36 (1.09x) | RT-primary 14.35 (0.94x)
      | swPrim+perpixelGI 23.87 (0.56x) | RT-prim+perpixelGI 26.37 (0.51x) | RT-prim+PROBE-GI 15.15 (0.89x)
      per-pass SW: main 13.11 transp 0.06 compose 0.28 || RT-prim+GI: main 14.33 transp 0.07 compose 0.28
      primary-trace-only: SW 4.49 || RT 5.04  (=> SW shade 8.62, RT-prim+GI shade 9.29)
      static-cam 15.33 ms
    rt_vs_software_timing [water-close]: software 4.11 | RT-occl 4.24 (0.97x) | RT-primary 4.32 (0.95x)
      | swPrim+perpixelGI 7.40 (0.56x) | RT-prim+perpixelGI 6.95 (0.59x) | RT-prim+PROBE-GI 4.97 (0.83x)
      per-pass SW: main 3.06 transp 0.70 compose 0.33 || RT-prim+GI: main 3.32 transp 0.81 compose 0.34
      primary-trace-only: SW 2.15 || RT 2.01  (=> SW shade 0.93, RT-prim+GI shade 1.33)
      static-cam 4.99 ms
    rt_vs_software_timing [terrain-covered]: software 7.27 | RT-occl 7.11 (1.02x) | RT-primary 7.08 (1.03x)
      | swPrim+perpixelGI 11.71 (0.62x) | RT-prim+perpixelGI 11.29 (0.64x) | RT-prim+PROBE-GI 7.74 (0.94x)
      per-pass SW: main 6.85 transp 0.11 compose 0.30 || RT-prim+GI: main 6.82 transp 0.12 compose 0.31
      primary-trace-only: SW 2.58 || RT 2.31  (=> SW shade 4.26, RT-prim+GI shade 4.50)
      static-cam 7.82 ms

`terrain-covered` is a fourth scenario ADDED in this round (a closer, steeper
terrain framing), so it has no entry in rounds A-C. The other three are the
same cameras as before.

### Comparing D against A-C

Do it carefully. The software column moved too - terrain reads 13.85 / 15.11 /
13.16 / 11.84 ms across A/B/C/D on a path whose behaviour never changed, so
this machine's run-to-run spread is around 15% and swamps most of the
differences between rounds. The FIELD A/B above is the only comparison in this
document that holds everything but the field constant, and it is the one to
quote.

### What the bilinear reflection sampler itself costs

Isolated by re-running the harness with the tap weights forced to nearest.
Water-close is the only scenario where it can matter (the others have almost
no reflective pixels), and there it moves the transparent pass from 0.75 to
0.81 ms and the frame from 4.93 to 4.97 ms: about +0.05 ms, ~1% of that
frame, to remove the tile grid. Treat those as approximate - that particular
run was disturbed (376 s wall against 186 s, and one populated repeat came
back at 13.03 ms against its own 7.73 ms partner), so only the two numbers
above, which were stable across their repeats, are quoted.

### The counterfactual: sizing the pool to the broken rule instead

Also built and measured, because "the pool is too small" has two possible
fixes and only one of them is right. With `brick_needs_light` left as it was
and `LIGHT_BLOCKS_MAX` raised to 393,216 (a 201.3 MB pool), the field binds
all 370,719 blocks and covers the same brick z 0..=127:

| metric              | shell rule fixed | pool raised to 393,216 |
|---------------------|------------------|------------------------|
| blocks bound        | 60,174           | 370,719                |
| light pool          | 67.1 MB          | 201.3 MB               |
| light update pass   | 1.19 ms          | 1.41 ms                |
| A/B terrain         | -0.92 ms         | -0.92 ms               |
| A/B foliage         | -0.01 ms         | -0.00 ms               |
| A/B water-close     | -2.91 ms         | -2.86 ms               |
| A/B terrain-covered | -2.10 ms         | -2.11 ms               |

Nothing renders differently, which is the point: every one of those extra
310,545 blocks belongs to a brick with no air in it, the update pass writes
epoch 0 for solid voxels and `voxlight_sample` drops every tap that lands in
one, so they are records no shading path can read. The price for them is
+134 MB of VRAM and +0.22 ms of update per frame. The +0.22 rather than +6x
is because `cs_voxel_light_update` early-outs on a solid voxel before tracing
anything, so 6.2x the blocks costs 1.18x the time - the waste is
overwhelmingly memory, not milliseconds.

## Look, round D

Stills re-captured with `dump_lookdev_views` and read directly.

- The reflection MOSAIC is gone. The nearest-vs-bilinear difference image over
  the grazing water view is exactly the projected voxel grid - straight lines
  converging to the horizon, max 29/255, 2.95% of pixels - and at 5x zoom the
  hard axis-aligned tile boundaries in the before frame become smooth
  gradients in the after one.
- GRAZING WATER STILL READS DARKER THAN HEAD-ON, and interpolation was never
  going to change it: over the open-sky test sheet every record is identical,
  so the blend is a no-op, and `voxlight_refl_reads_back_plausible_radiance`
  still measures luma 0.269 grazing against 0.513 head-on, to the digit.
  Whether that is the SH-L1 fit under-reconstructing toward the rim or the
  real sky being dimmer at 19 degrees than at the zenith has NOT been tested;
  the check that would settle it is to have `cs_vl_probe` return
  `sky(refl_dir)` alongside the reconstruction. OPEN.
- The views that used to be pixel-identical with and without the field are
  identical no longer, because the pool finally covers them: `meadow_top`
  mean 25.8/255 over 67% of pixels, `meadow` 8.3, `canopy_top` 4.4,
  `birch_close` 2.4. The broad dark AO smear along every terrain step and
  around every grass tuft - the "diffused checkerboard shadow carpet" that
  view exists to catch - is gone.
- Not all of that is an improvement. Step edges now read as thin hard dark
  lines rather than soft bands, and the soft canopy shadow on the ground under
  `canopy_top` is visibly weaker. That is the documented AO deviation (stage 2
  of the plan: per-air-cell occupancy instead of the per-face bilinear blend)
  showing up in a view class that had never seen it before, not a new bug -
  but it is a real look change and it is not uniformly better.
- The crafted lab worlds (`water_terrace`, `leaf_lab_*`, `material_lab_*`) are
  BIT-IDENTICAL before and after. They never overflowed the pool and their
  reflection records are uniform, so neither fix can move them - which is
  itself the check that dropping the fully solid bricks' blocks changes
  nothing observable.

## Round E: the "flickery and not correctly reflecting" report

HISTORY. Everything in this section describes the per-voxel reflection field,
which round F deletes. The code it names (`voxlight_reflection`,
`cs_voxel_refl_update`, `voxlight_refl_diagnose` and the four `voxlight_refl_*`
tests) no longer exists; the measurements are kept because they are the reason
it does not.

Round D shipped a reflection cache that a user then reported as broken:
flickery, and not reflecting correctly. Both halves were real, both were found
by measurement, and neither was the thing the report's leading theory blamed.

### What it was NOT

The leading theory was that the cache is static while the water surface is
animated, so the SH is evaluated far from where it was gathered and the wave
motion sweeps the query across a coarse fit. That is testable and it is WRONG.

- The Gerstner facet is bounded at 3.0 degrees of tilt - max slope 0.0525 from
  the four amplitudes and wavenumbers in `wave_param` - so the reflected
  direction swings at most 6.0 degrees.
- Over a full facet cone the CACHE moves 0.107 luma where the true per-pixel
  mirror moves 0.323 over the same sweep (`voxlight_refl_diagnose`, section B).
  The cache is SMOOTHER than the thing it replaces, which is the opposite of a
  flicker source, and a linear fit is smooth in direction by construction.
- The animated surface HEIGHT cannot mis-address the record either: the plate
  lives at `WATER_BASE` 0.72 plus at most `WATER_WAVE_MAX` 0.12, and a pinned
  corner caps at 1.00, so with the sampler's 0.02 step-back along the normal
  `floor(q)` always names the water cell. Ruled out by construction.
- The per-pixel fallback is not flipping either: 0 `ok` flips over 128 rounds.

### What it actually was

Three defects, all in the estimator and the fit, none in the storage:

1. THE ESTIMATOR. The update pass visited a block every `VOXLIGHT_UPDATE_DIV`
   rounds and folded ONE ray of a rotating eight-direction set at fold 0.125.
   A hemisphere over water spans an order of magnitude in radiance between the
   sky overhead and the terrain at the rim, so those are eight different
   numbers, not eight noisy looks at one. Measured on a static scene, static
   sun, fixed query direction: 22.7-46.8% peak-to-peak, with a stored
   `E[L*d.z]` of -0.17 where the true moment is unambiguously positive.
   Fixed by visiting eight times more rarely and gathering a COMPLETE
   stratified hemisphere each visit, over a direction set that does not depend
   on the round. Same rays per frame. Now 0.0% - bitwise constant - and
   `m_z` = +0.035. Pinned by `voxlight_refl_record_is_stable_across_rounds`,
   which fails at 16.6% on the old estimator.
2. THE FIT'S MEASURE. The reconstruction solved its moment matrix with
   `E[(d.t)^2] = 1/4`, the cosine-weighted value, while using the uniform
   `E[d.n] = 1/2` and `E[(d.n)^2] = 1/3` in the same solve and sampling
   uniformly in solid angle. That over-weighted the tangential term by exactly
   4/3, so the entire view-dependent part of every reflection came out a third
   too strong - and amplified the estimator's swing on top of it.
3. THE FIT'S POLE. The sampler evaluated the fit about the SHADING normal
   instead of the axis the moments were gathered about. On open water that is a
   3-degree error and worth nothing either way, but at a terrace step or pinned
   corner the facet reaches 45 degrees, and re-poling also makes
   `c = dot(refl_dir, n)` read near 1 for a mirror direction, so the
   out-of-hemisphere reject could never fire and the record answered
   confidently for directions it had never sampled. Over a 45-degree facet cone
   the cache read a washed 0.27..0.70 against the mirror's 0.05..1.93; with the
   gather axis as the pole it reads 0.00..1.93.

### Temporal stability, finally measured on real frames

`flicker_probe_rt_views` gained a grazing water view and was run as an A/B -
once on the per-pixel fallback, once with the fields live. Every number this
rig had ever produced before was taken against an empty field.

| view                        | strong flickering | faint  | breathing | mean luma |
|-----------------------------|-------------------|--------|-----------|-----------|
| water_graze, per-pixel      | 12290 (0.593%)    | 68651  | 19496     | 127       |
| water_graze, per-voxel field|   476 (0.023%)    |  6552  |   249     | 130       |

The cache flickers 26x LESS than the per-pixel reflection it replaces, at the
same brightness. That is the report's "flickery" answered, and it answers it in
the opposite direction from the theory: the per-voxel cache is the STABLE one.

### Look, round E: what the cache still costs

`dump_lookdev_views` now captures the grazing water view BOTH ways
(`water_graze_nofield` is the per-pixel mirror, `water_graze` the cache), which
is the A/B the reflection decision always needed and never had. Read directly,
and measured over the band where the reflection carries scene content (the
shoreline vegetation mirrored in the water, y 325-400):

| quantity                          | per-pixel mirror | per-voxel cache |
|-----------------------------------|------------------|-----------------|
| mean luma                         | 90.10            | 93.55           |
| spatial sd (contrast)             | 7.07             | 4.43 (62.6%)    |
| correlation with the mirror       | 1.000            | 0.757           |

So the cache keeps about 63% of the reflection's contrast and tracks it at
r = 0.76. In the frames that reads as recognisable tree shapes in the per-pixel
still becoming smooth voxel-scale blobs in the cached one, and the dark
reflected foliage lifted 3.45 luma toward the hemisphere mean. Open water
further out (y > 400) is within 0.3-0.7 luma either way, because out there the
reflection is sky only, which a linear fit handles well.

That residual is STRUCTURAL, not a remaining bug, and the shape of it is now
understood rather than guessed. The true sky luma over an elevation sweep runs
0.83 (horizon) -> 0.54 (30 deg) -> 0.83 (70 deg) -> 0.75 (zenith): it has an
INTERIOR MINIMUM. An L1 reconstruction restricted to a great circle of
elevations is `a + R*cos(e - phi)`, which over [0,90] admits an interior
MAXIMUM only, so a linear fit cannot be bright at both ends with a dip between
them and the least-squares answer is to go flat. Measured: the cache reads
0.60 -> 0.65 -> 0.64 across that sweep, a span of 0.06 against the sky's 0.30,
and it errs in the predicted direction everywhere - too dark at the two bright
ends, too bright in the dip. Over the whole sweep it sits between 0.67x and
1.96x the per-pixel mirror.

Raising that needs an L2 term, which is what "bright at both ends, dark in the
middle" requires: 9 coefficients per channel instead of 4, and roughly 37 MB of
reflection pool instead of 16.8 MB. That is a design decision with a memory
price and it is NOT taken here.

### Round E perf

Re-run on the same machine and harness. Read the FIELD A/B column, not the
absolutes: this whole run sits 10-15% hotter than round D (terrain's EMPTY
field reads 15.75 ms against round D's 13.64 on a path neither round changed),
which is the documented run-to-run spread on this machine.

| scenario        | populated     | empty field | delta round E     | delta round D |
|-----------------|---------------|-------------|-------------------|---------------|
| terrain         | 14.43 / 14.68 | 15.75 ms    | -1.19 ms (-7.6%)  | -0.92 ms      |
| foliage         | 17.42 / 17.45 | 17.47 ms    | -0.04 ms (-0.2%)  | -0.01 ms      |
| water-close     |  5.80 /  5.62 |  8.55 ms    | -2.84 ms (-33.2%) | -2.91 ms      |
| terrain-covered |  8.23 /  8.88 | 11.10 ms    | -2.54 ms (-22.9%) | -2.10 ms      |

The shading side is unchanged within noise, which is the expected result: none
of the three fixes touches how much work a pixel does, only what the record
holds and how the fit reads it.

The UPDATE side did move, and not for free:

    round D:  light pass 1.19 ms + reflection pass 0.23 ms = 1.39 ms
    round E:  light pass 1.38 ms + reflection pass 0.51 ms = 1.64 ms

The light pass's +0.19 is the run-to-run drift above. The reflection pass's
+0.28 (+122%) is the estimator reshape and it is NOT drift. Rays per frame
really are unchanged - 16,448 ray slots before, 16,896 after - but the DISPATCH
SHAPE is not: 2050 live reflection blocks at div 8 is 257 workgroups, and at
div 64 it is 33, so eight serial traces per thread replace eight workgroups'
worth of latency hiding. It is worth paying here (water-close still saves
2.84 ms against a 1.64 ms total update) and it is worst exactly where there is
least reflective geometry to fill the machine, but "same rays per frame" should
never have been read as "same cost". OPEN as a perf follow-up: the complete
gather is what fixed the flicker and must stay, so the lever is the dispatch -
e.g. splitting a visit's rays across lanes rather than looping them in one
thread, which keeps the estimate whole and puts the parallelism back.

Net per frame, as in round D:

| scenario        | shading saves | update costs | NET      |
|-----------------|---------------|--------------|----------|
| terrain         | 1.19 ms       | 1.64 ms      | +0.45 ms |
| foliage         | 0.04 ms       | 1.64 ms      | +1.60 ms |
| water-close     | 2.84 ms       | 1.64 ms      | -1.20 ms |
| terrain-covered | 2.54 ms       | 1.64 ms      | -0.90 ms |

## Round F: the reflection field deleted, water restyled

Round E ended with a reflection cache that was stable, correct in its own terms,
and still 62.6% of a mirror's contrast. Round F is the answer to that: the whole
per-voxel reflection field comes out, the per-pixel water mirror comes out with
it, and water becomes deliberately faceted and stylized (Marc, 2026-08-01). See
"Reflections: BUILT, MEASURED, AND DELETED" in docs/VOXEL_LIGHTING_PLAN.md.

Everything below was measured on the same machine and harness as rounds A-E,
with the BEFORE half taken by `git stash`ing this exact change so the two halves
differ by nothing else.

### Perf

BEFORE was re-measured for this round rather than quoted from round E; this
machine's documented run-to-run spread is around 15% and round E's absolutes are
not comparable to today's.

| scenario        | before  | after   | delta            |
|-----------------|---------|---------|------------------|
| terrain         | 15.77   | 16.64   | +0.87 (noise)    |
| foliage         | 19.26   | 19.43   | +0.17 (noise)    |
| water-close     |  6.01   |  4.12   | -1.89 (-31.4%)   |
| terrain-covered |  9.90   |  8.41   | -1.49 (-15.1%)   |

Read the terrain and foliage rows as noise, not as a regression: the SOFTWARE
column moved by more than they did on paths this change cannot touch (terrain
14.69 -> 15.65, foliage 18.37 -> 16.85), which is the spread rounds A-E already
documented. Water-close is the row that matters and it is unambiguous.

WHERE the water win comes from is not where it was expected. The transparent
pass barely moved - 1.00 -> 0.83 ms - even though it lost an entire reflection
trace plus a full secondary `shade` per pixel. The primary pass is what
collapsed:

| water-close, RT-primary | before | after |
|-------------------------|--------|-------|
| cs_main                 | 4.15   | 2.24  |
| primary trace only      | 2.50   | 1.12  |
| cs_transparent          | 1.00   | 0.83  |

The primary TRACE more than halved because the faceted surface needs almost no
neighbourhood: the corner-connected surface probed up to 24 neighbours per
surface cell (8 pin + 8 level + up to 8 step-down) to build four shared corner
heights, and a flat plate needs one probe for "is there water above" plus four
for the shore mask, and those four only inside foam range. That is a geometry
saving, and it was not the point of the change - it came free with it.

The reflection UPDATE PASS is also gone from every frame:

    round E:  light pass 1.38 ms + reflection pass 0.51 ms = 1.64 ms
    round F:  light pass 1.63 ms                           = 1.63 ms

(the light pass's own number is inside this run's spread; the reflection pass's
0.51 ms is simply not spent any more.)

### What the field is still worth, which is LESS than it was

| scenario        | A/B before        | A/B after        |
|-----------------|-------------------|------------------|
| terrain         | -0.97 ms (-5.8%)  | -1.09 ms (-6.1%) |
| foliage         | +0.70 ms (+3.6%)  | +0.02 ms (+0.1%) |
| water-close     | -3.31 ms (-34.4%) | -0.65 ms (-13.9%)|
| terrain-covered | -3.45 ms (-26.0%) | -1.48 ms (-14.9%)|

Water-close falling from -3.31 to -0.65 ms is NOT a regression in the field, and
it would be easy to report it as one. Most of what the field used to save on
water was saving the SECONDARY rays: every reflection ray that hit something ran
a full `shade`, paying a shadow ray and an AO evaluation, and the field answered
all of them from a record. There are no reflection rays now. What is left is the
water surface's own sun term, and one field fetch instead of one shadow ray is
worth 0.65 ms. The frame is 1.89 ms faster in absolute terms; the field's share
of the credit is smaller because the bill is smaller.

Net per frame against the 1.63 ms update cost, as in rounds D and E: the field
still pays for itself on terrain-covered (-1.48 vs 1.63, roughly break-even) and
no longer does on water-close. That is the same open item rounds D and E
recorded - the update pass is not camera-aware and the per-pixel path it was
meant to replace is still there - and this round does not close it.

### Temporal stability, and the one place it got worse

`flicker_probe_rt_views`, before and after, on the same machine. The non-water
views (terrain_trees, tree_shadow, meadow, underwater) are BIT-IDENTICAL across
the change, which is the check that this is a water-only rework.

| view                 | before          | after           |
|----------------------|-----------------|-----------------|
| water_top            | 2796 (0.135%)   | 5807 (0.280%)   |
| water_graze, no field| 12290 (0.593%)  | 5562 (0.268%)   |
| water_graze, field   | 474 (0.023%)    | 3833 (0.185%)   |

Stated plainly, because it cuts both ways:

- Against the PER-PIXEL MIRROR - the thing a player was actually looking at
  before the reflection cache existed - stylized water flickers 2.2x LESS
  (0.593% -> 0.268%).
- Against the REFLECTION CACHE, it flickers 8x MORE (0.023% -> 0.185%). The
  cache was a view-independent value stored per voxel and recomputed rarely; it
  was almost perfectly still by construction. Quantized animated geometry cannot
  be, and is not meant to be.
- The steep view is up 2.1x (0.135% -> 0.280%) and there is no reflection in it
  either way at that angle, so that number is the animation itself.

It was 8.7x worse than that before it was measured and fixed. The first build
read 1.177% on water_top, and the cause was found by elimination rather than
guessed:

| water_top variant            | strong flicker |
|------------------------------|----------------|
| before this rework           | 0.135%         |
| faceted, hard band ladder    | 1.177%         |
| ... with crest foam disabled | 1.162%         |
| ... with the TONE ladder off | 0.262%         |
| ... tone ladder EASED (ship) | 0.280%         |

So the foam was not it (0.015 points) and the tone ladder was 78% of it: a hard
staircase pops a cell by a whole tone step the instant the wave carries it over
a band boundary. The fix is `WATER_BAND_EASE`, which spends 35% of each band
sliding across its boundary instead of sitting on it. It costs nothing in the
look because the thing that makes the surface read as faceted is the step
BETWEEN NEIGHBOURING CELLS, which is untouched - only a cell CROSSING a boundary
changes gradually, over several frames instead of one, and 65% of cells still
sit exactly on a level.

THE HEIGHT IS DELIBERATELY NOT EASED, and that is the other half of the same
measurement. Easing it as well took water_top to the same 0.290% but pushed the
GRAZING view the wrong way, 0.161% -> 0.340%: from a low angle a crest's plate
occludes the trough behind it, so a hard-quantized height holds that occlusion
boundary still while a sliding one creeps it across pixels. The height's
quantization is load-bearing for grazing stability; the tone's was only ever
load-bearing for the look.

### Look, round F

Stills captured with `dump_lookdev_views` and read directly, including a new
`water_shadow` view - a wall standing in a flat sheet - because no natural
camera in the demo world has shadowed open water in frame, and the lit/shadowed
read is the whole of this change's part 2.

- BEFORE, the water was blue paint. A 3x crop of the middle of `water_view` is a
  single flat blue with a handful of stray dark pixels in it; the grazing crop is
  the same blue with soft dark smudges where the reflection cache was
  reconstructing shoreline vegetation. That is the reflection decision's cost
  from round D and E, seen plainly.
- AFTER, the same crops are a mosaic of flat cells at distinct blues with hard
  boundaries, and chunky near-white foam at the crests. It reads as deliberate
  stylization rather than as a surface that failed to resolve.
- `water_shadow` is the clearest of the set: lit facets on one side, navy facets
  on the other, a crisp near-vertical edge about 2-3 px wide between them, and
  the foam staying white in both. Measured by `water_reads_lit_or_shadowed`:
  the shadowed side sits at 0.316 of the lit side, and the 0.72-0.90 ratio band
  - the penumbra - is 1.1% of the shaded pixels.
- Three defects were found in the stills and fixed before this round closed, all
  of which the numbers alone would have missed:
  - step RISERS shaded with their true vertical face normal painted hard navy
    cracks and bright grazing slivers along every band boundary. A lateral entry
    from another water cell is now presented with the plate's own normal.
  - the shore-foam rule "adjacent to SOLID" foamed nothing at all on a pool
    whose rim sits below its surface, which is the common case; it is now
    "adjacent to anything that is not water".
  - the uncapped Schlick Fresnel turned the terrace lab's low camera into a
    sheet of milky horizon sky with every facet washed out of it. Capped at 0.45.
- The weakest view is still `water_terrace`, whose camera sits about two voxels
  off the water: at that magnification the shoreline foam is a large solid white
  shape rather than surf, even after breaking a quarter of shore cells out of the
  band. It is honest at that distance - foam seen from 20 cm IS a solid white
  mass - but it is the one frame in the set that does not sell itself.

## Reproducing

    cargo test --lib rt_vs_software_timing -- --nocapture --ignored
    cargo test --lib dump_lookdev_views -- --ignored --nocapture
    cargo test --lib flicker_probe_rt_views -- --ignored --nocapture

Re-run on the SAME machine after the rework and compare against this table,
not against docs/PERF.md.
