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

## Round G: measuring a LIVE session, which nothing had ever done

Round G starts from a user report - the game runs slowly - that every number in
rounds A to F contradicts. Both were true, because every number in rounds A to F
came out of `rt_vs_software_timing`, and that harness builds ONE static crafted
world: no physics tick, no streaming, no `upload_world`, no `Renderer`. Three
cost classes are invisible to it BY CONSTRUCTION, and two of them were large.

### The harness that was missing

`live_session_profile` (`src/renderer.rs`, ignored by default) runs the real
world with `physics::tick` at 30 Hz, real `shift_origin` /
`install_finished_chunks` streaming, the real `upload_voxlight`, and the real
update dispatch at the block count a streamed world actually binds. It paces
itself at 60 Hz because chunk generation is asynchronous - a loop running flat
out finishes before the worker pool produces anything and measures a world that
never streams. It reports CPU milliseconds per stage and GPU milliseconds per
pass SEPARATELY.

`GPU_PROFILE_LABELS` also gained `vlight`: the light-field update used to share
the `probe` bracket with the GI probe update, so the single largest world-space
GPU cost in the frame was unattributable in the only profiler that watches the
real renderer.

### What it found: the update pass is the largest GPU pass in the frame

With the split label, on the in-game benchmark at 1920x1080:

| segment       | vlight | probe | main | gpu total | vlight share |
|---------------|--------|-------|------|-----------|--------------|
| water_mid     | 1.02   | 0.49  | 0.72 | 5.19      | 20%          |
| water_grazing | 0.95   | 0.47  | 0.61 | 5.38      | 18%          |
| water_strafe  | 0.96   | 0.48  | 3.22 | 7.38      | 13%          |
| terrain       | 1.29   | 0.52  | 1.22 | 4.72      | 27%          |
| foliage       | 2.09   | 0.57  | 2.42 | 6.86      | 30%          |
| meadow        | 1.49   | 0.58  | 0.61 | 4.60      | 32%          |

It is LARGER THAN THE RAYMARCH on four of the six segments, and it is the one
pass in the frame whose cost has nothing to do with what is on screen: it walked
the entire resident shell every eight frames with no idea where the camera was.

### And the per-frame CPU upload, which fires in bursts

Same run, from the new `voxlight upload:` log line:

    47 / 120 frames dirty (39%), 278 MB/s table + 24 MB/s resets,
    295 write_buffer calls per frame (35,336 reset blocks),
    cpu 0.12 ms shell + 0.68 ms upload per frame

278 MB/s is the 4 MiB brick -> block table going WHOLE on any binding change, and
295 writes per frame is one 512-byte DMA per invalidated block - installing a
streamed chunk marks all 512 of a slot's bricks dirty and every one of them
invalidates its block. In the quiet segments (camera parked, water settled) it is
0 MB/s and 0 writes, which is exactly why the static harness never saw it: this
is a STUTTER, not a steady tax, and only a live session has one.

### The fixes

1. CAMERA-AWARE UPDATE. `LightField` partitions its work list near-first
   (`near_count`); `cs_voxel_light_update` walks the near prefix on the old
   cadence and the far remainder `VOXLIGHT_FAR_DIV` = 8 times more rarely, from
   ONE dispatch. It is a REFRESH RATE and nothing else - every block keeps its
   storage and its converged record, so shading reads the same field and no pixel
   changes path. Pinned by `far_shell_converges_to_the_same_light_as_near_shell`,
   which converges both groups and asserts they agree BIT FOR BIT.

   Anything with no readable record promotes itself into the near group whatever
   its distance (`promote_near`): a newly bound block, and an invalidated one.
   That is what lets the radius be 64 voxels rather than a guess at "what the
   camera can see" - measured at 128 it classified 48-77% of the shell as near,
   because lit shell is a 3D surface and a sphere that size over hills and canopy
   sweeps up an enormous amount of it.

   The partition is rebuilt when the camera drifts 16 voxels OR when promotions
   exceed a quarter of the near group. BOTH triggers are needed: promotion is
   one-way, so with the camera trigger alone a still camera watching a lake
   settle dragged the whole shell into the near group one block at a time and the
   near fraction stayed at 43-66% however far the radius was tightened.
2. DELTA TABLE UPLOAD. `LightField` records which brick entries changed;
   `upload_voxlight` pushes only those runs through the existing `upload_spans`,
   which gained a gap tolerance so a scattered handful of changes is not a
   scattered handful of DMAs. The work list has a SEPARATE flag: it is 240 KiB
   against the table's 4 MiB and it changes on a different trigger (the
   repartition reorders it without moving a single table entry).
3. BATCHED RESETS. Recycled and invalidated blocks are deduped through a bitset
   and coalesced into runs of consecutive blocks, one write each. NO gap merging
   here, unlike the table: a block between two resets is another brick's LIVE
   block and sweeping it up would erase converged light.
4. BITSET DEDUP IN THE SHELL WALK. `sync_light_shell_dirty` evaluated brick + 6
   neighbours per dirty brick, and dirty bricks arrive in contiguous blobs, so
   interior bricks were evaluated up to seven times each. A SORT was tried first
   and measured SLOWER than the redundant work it removed (0.128 ms mean against
   0.072); one bit per brick is O(1) per probe and just as local.

### Measured, `live_session_profile`, same machine, same session

600 frames, camera walking at 24 voxels/s, physics at 30 Hz, 3 origin shifts,
393,216 dirty bricks (max 3,072 on one frame).

| per frame, ms              | before            | after             |
|----------------------------|-------------------|-------------------|
| sync_light_shell_dirty     | 0.072 / p99 0.438 | 0.052 / p99 0.333 |
| set_light_focus (new)      | -                 | 0.041 / p99 1.052 |
| upload_voxlight            | 0.206 / p99 2.357 | 0.048 / p99 0.512 |
| table traffic              | 40.6 MB/s         | 2.56 MB/s         |
| write_buffer calls / frame | 75.5              | 13.2              |
| reset blocks, 600 frames   | 45,136            | 25,245            |

The light field's whole CPU cost goes 0.278 ms mean to 0.141 ms, and the TAIL -
which is what a stutter is - goes 3.0 ms to 0.5 ms.

GPU, timed in the SAME run so thermal drift cannot forge it (both lines are 200
warm-up plus 60 timed submits, back to back):

    whole shell near (49,086 blocks, 6,136 workgroups): 3.64 ms
    camera-aware     (1,246 near,      904 workgroups): 0.95 ms

3.8x, and the near prefix is 2.4-6.5% of the shell over the walk.

`rt_vs_software_timing` now prints the same pair on the demo world, and it agrees:

    light pass, whole shell near (60,174 blocks, 7,522 workgroups): 4.66 ms
    camera-aware               ( 2,692 near,     1,236 workgroups): 1.11 ms

The shipped path therefore pays 1.11 ms where the same harness measured 2.05 ms
before this round (whole shell, 4 rays per visit), on a run whose software column
reads 9% hotter - so roughly half, while ALSO doubling the estimator's angular
resolution and removing the oscillation. The FIELD A/B columns are unchanged
within their usual spread: terrain -1.51 ms, water-close -0.93 ms,
terrain-covered -4.21 ms, foliage +1.04 ms.

### The water flicker: the leading theory was wrong, and the real cause was worse

The report was that shadows on water are too flickery, and the leading theory was
that the faceted-water rework and the light field are fighting: the surface
height is quantized and animated, so the shading point steps across voxel
boundaries and lands on a different record each step.

THAT IS WRONG, and wrong by construction rather than by measurement error.
`water_plate_height_does_not_move_the_light_sample` samples the field at all
SEVEN quantized plate heights across a shadow edge and gets BYTE-IDENTICAL
answers at every one. The reason is the sampler's solidity gate: the plate lives
inside the water cell (0.48..0.96 of it), the water voxel under it is occupied so
that tap is dropped and its weight renormalised away, and the whole answer comes
from the row of air voxels ABOVE the cell - whose weight then cancels. The height
divides out exactly.

The real cause is the ESTIMATOR, and it was in the plan document as a claim: "a
complete cycle folds an exact soft-shadow estimate, so a static scene converges
to the true penumbra and then stops changing". It does not. The pass traced a
QUARTER of the sun-disc direction set per visit, selected by an epoch counter,
and folded it in at 0.35. A penumbra voxel's four quarters are four DIFFERENT
NUMBERS, not four noisy looks at one, so the fold tracked them instead of
averaging them.

Measured by `voxlight_sun_is_stable_once_converged`, static scene, static sun:

| quantity                                           | before       | after |
|----------------------------------------------------|--------------|-------|
| worst swing of a converged record over one cycle    | 93/255       | 0/255 |
| worst swing inside the penumbra band                | 89/255       | 0/255 |
| penumbra records swinging past half the water step  | 2670 of 2670 | 0     |

`shade_water_top` pushes that value through `smoothstep(0.41, 0.59, sun_vis)`, a
band 45/255 wide, so an 89/255 swing at a half-lit voxel is the ENTIRE difference
between lit water and shadowed water, every few frames, with nothing in the scene
moving. Terrain multiplies by the same value linearly and wobbles by a few
percent, which is why this reads as a water bug and not a lighting bug.

It is the same defect, with the same cause and the same fix, that round E found
and fixed in the per-voxel REFLECTION field. It was never applied to this pass.

The fix makes a visit's estimate COMPLETE: a fixed sunflower stratification of
the disc depending on nothing but the ray index, so a static scene converges to a
fixed point and stays there and a moving sun tracks it smoothly. Rays per visit
went 4 -> 8, because rays-per-visit is now also the angular RESOLUTION - the old
scheme's time-average was a 32-ray answer even though each visit traced 4, and 4
stable rays would have been a five-level penumbra ladder.

What that costs in the look, against a 32-ray reference over 123,446 records
(`eight_sun_rays_match_a_dense_estimate`): mean 0.58/255, p99 16/255, p99.9
40/255. The tail sits exactly at (1/8 + 1/32) * 255 = 39.8, which is what two
non-nested stratifications of one disc can disagree by at a hard edge, and p99 at
16/255 is what says the coarsening really is confined to the penumbra.

### Flicker rig, before and after

`flicker_probe_rt_views` gained two views it needed and never had:
`water_top_field`, and the `water_shadow` pair on the crafted wall-in-a-sheet
scene. Every water number this rig had ever produced was taken with
`voxlight: false`, i.e. against the per-pixel fallback, so "the field makes water
flicker" was untestable in it. `water_top` also turned out to frame a pond in
FULL SUN, where the lit/shadowed smoothstep is saturated and reads the same
whichever path answers - which is why its field and no-field lines agree to
eleven pixels and settle nothing.

| view                  | before          | after           |
|-----------------------|-----------------|-----------------|
| water_top (fallback)  | 5807 (0.280%)   | 5807 (0.280%)   |
| water_top_field       | 5796 (0.280%)   | 5796 (0.280%)   |
| water_graze_nofield   | 5562 (0.268%)   | 5562 (0.268%)   |
| water_graze_field     | 3833 (0.185%)   | 3802 (0.183%)   |
| water_shadow_nofield  | 2472 (0.119%)   | 2472 (0.119%)   |
| water_shadow_field    | 2472 (0.119%)   | 2472 (0.119%)   |
| terrain_trees         | 23793 (1.147%)  | 23793 (1.147%)  |
| tree_shadow           | 19691 (0.950%)  | 19691 (0.950%)  |
| meadow                | 4913 (0.237%)   | 4913 (0.237%)   |

READ THAT HONESTLY: the rig barely moves, and it CANNOT show this fix. Its worlds
are static and its sun advances 6e-5 rad per frame, so the penumbra band is a
line a couple of voxels wide and the oscillation had almost nothing to strobe.
The measurement that shows both the defect and the fix is
`voxlight_sun_is_stable_once_converged`, which reads the stored value directly
instead of hunting for its consequences in a still scene. What the rig does
establish is that nothing REGRESSED, and that the non-field views are unchanged
to the pixel.

### What this round does NOT claim

End-to-end FPS before and after is NOT quoted, because this machine's thermal
spread swamped it during the session: the same code and the same six benchmark
segments read terrain 205 fps on one run and 117 on another, and
`rt_vs_software_timing`'s own software column - a path none of this work touches -
moved 21.25 -> 24.39 ms between the before and after runs. The numbers quoted
above are the ones measured within a single run against their own control: the
GPU pass A/B, the FIELD A/B columns, and the CPU counters.

`rt_vs_software_timing` now prints the light pass BOTH ways for exactly this
reason. It binds the shell with `sync_light_shell_all` and no camera, so its
whole-shell line is the cost of refreshing a 60,000-block world every eight
frames regardless of where anyone is standing, which is what the renderer used to
do and is not what it does now. Quoting only that line overstates the live cost
several times over.

## Round H: the sky-facing frame, and paying per SECOND instead of per FRAME

Round H starts from a second user report: ~400 fps looking at the SKY before this
branch, ~180 after. A sky-facing frame has almost no geometry in it, so almost
nothing in it scales with what is visible; whatever it costs is the FIXED cost.
Nothing measured it, because `live_session_profile` timed exactly one GPU pass in
isolation and never rendered a frame.

### The harness that was missing, again

`live_session_profile` now builds the WHOLE shipped frame at 1920x1080 - clouds,
beam, GI probe update, light update, cs_main, cs_transparent, cs_compose, in
`Renderer::render`'s order and with `Renderer::new`'s override constants - on the
world the walking session leaves behind, and times each pass for four camera
segments. `sky` is new and is the one the report is about. It also reports, from
the same rig:

  - the FIELD'S PIXEL COVERAGE, by compiling cs_main and cs_transparent with a
    `PROF_VLF_COVERAGE` override that paints every shaded surface green when
    `voxlight_sample` answered and red when it fell through, then counting;
  - the frame WITH and WITHOUT the light update dispatch, back to back, which is
    the within-run control for everything below.

### What a sky-facing frame is made of

1920x1080, RT primary + probe GI, 49,086 live blocks:

| segment | vlight | clouds | beam  | probe | main   | transp | compose | frame  | frame, no vlight |
|---------|--------|--------|-------|-------|--------|--------|---------|--------|------------------|
| sky     | 0.482  | 0.947  | 0.036 | 0.493 |  0.316 | 0.110  | 0.141   |  2.558 | 2.122            |
| terrain | 0.470  | 0.414  | 0.035 | 0.485 |  4.156 | 0.227  | 0.342   |  6.143 | 5.649            |
| foliage | 0.488  | 0.456  | 0.043 | 0.503 | 15.785 | 0.116  | 0.411   | 17.987 | 16.977           |
| water   | 0.479  | 0.511  | 0.028 | 0.494 |  3.009 | 1.024  | 0.414   |  6.072 | 5.565            |

READ THE `vlight` COLUMN ACROSS, not down: 0.48 / 0.47 / 0.49 / 0.48 ms. It is
the same on a frame with nothing in it as on one that is all canopy, which is the
diagnosis stated as a measurement - this pass's cost has nothing to do with what
is on screen. `main` over the same four rows runs 0.32 to 15.8 ms.

AND IT IS NOT 3 ms. On this machine, in this harness, the whole light-field
update on a sky-facing frame is 0.44 ms of GPU (the within-run control: the same
frame back to back, 2.558 ms with the dispatch and 2.122 without) plus 0.10 ms of
CPU (`set_light_focus` + `sync_light_shell_dirty` + `upload_voxlight`, means).
That is 391 -> 471 fps if it is removed entirely, not 180 -> 400. The reported
3 ms is NOT reproduced here and this document does not claim it was found; what
WAS found is a real 0.44 ms of fixed per-frame GPU that had no business being
spent every frame, and it is now spent on about one frame in seven at the frame
rates the report was taken at.

### The dispatch gate: cost per SECOND, not per FRAME

A record's value is a function of (geometry, sun direction, light set), so
re-gathering it when none of the three moved writes bytes that are already there.
`voxlight_sun_lag_error` measures how much of the field actually moves for a given
sun lag, over 3,851,136 records of the demo world:

| lag     | sun angle | records changed | mean | p99 | max |
|---------|-----------|-----------------|------|-----|-----|
| 0.033 s | 0.0008    |  0.17%          | 0.05 |   0 |  63 |
| 0.133 s | 0.0033    |  0.75%          | 0.21 |   0 |  63 |
| 0.533 s | 0.0133    |  1.38%          | 0.40 |  31 |  95 |
| 1.067 s | 0.0267    |  1.98%          | 0.59 |  31 | 127 |
| 2.133 s | 0.0533    |  2.60%          | 1.06 |  31 | 190 |
| 4.267 s | 0.1067    |  3.15%          | 2.63 |  95 | 254 |

0.133 s is a whole NEAR sweep period at 60 fps, and 99.25% of records come back
BIT-IDENTICAL across it. The pass was not slightly wasteful; it was almost
entirely redundant.

So the sweep is now paced by SUN MOTION (`VoxLightSchedule`): a round is issued
once the sun has turned `VOXLIGHT_SUN_RAD_PER_ROUND`, plus a floor of one
complete sweep per second while geometry is still settling. At 60 fps and the
shipped sun that is one round per frame - the old cadence exactly, so the LOOK is
unchanged. At any other frame rate it is not:

    per second: 58.8 sweep rounds + 2,544 urgent bricks
    workgroups per frame:  60 fps 934   144 fps 389   240 fps 234   400 fps 140
    the frame-counter schedule:  910 at every one of those rates

That is 1.0x at 60 fps, 2.3x at 144, 3.9x at 240 and 6.5x at 400 - and zero with
the sun still.

With the sun frozen and nothing dirty it issues NOTHING and the compute pass is
not encoded at all - pinned by
`the_light_sweep_stops_when_the_sun_and_the_world_do`, which also pins the 60 fps
and 400 fps rates and the one-sweep-per-edit bound.

THE 60 fps ROW IS SLIGHTLY HIGHER THAN THE OLD SCHEDULE (934 against 910) and
that is not noise. Blocks with no readable record - newly bound, just invalidated
- used to be pushed into the near TIER and wait up to eight frames for their
slice; they go on an URGENT list now and are dispatched on the very next frame.
In a session sprinting at 24 voxels/s through streaming that is 2,544 bricks per
second, i.e. 43 workgroups per frame at 60 fps, and it buys the worst-case
latency after an edit going from 8 frames to 1. A standing player generates
almost none of it.

It was 3.3x that (8,375/s, 143 wgs/frame, and a 60 fps row of 1030 - a real
regression) until the WHOLE-WORLD path stopped queueing. `sync_light_shell_all`
binds every block in the world, so it queued the entire 60,174-block shell as
urgent, which then drained at the per-dispatch budget for a hundred frames -
re-doing work the sweep had already done, because that path also arms a full
sweep AND runs it at the near cadence (with no camera focus yet, the whole shell
is near, which `allocate` now preserves explicitly rather than by the side effect
of the promotion it lost). It clears the queue instead.

Two defects were found while building the gate, both by the tests rather than by
reading:

  - `acos(dot(a, b))` for the sun's angular step reads 3.4e-4 rad of motion
    between a unit vector and ITSELF (the dot comes back as 0.99999994 and acos
    has infinite slope at 1), which is 82% of a round. The frozen-sun case ran
    599 sweeps in 600 frames. It uses the chord, `2 asin(|a - b| / 2)`, which is
    the same angle and is exact at zero.
  - `sun_dir_at` does not turn at 0.025 rad/s. It normalizes
    `(cos a, sin a, 0.30)`, so the direction traces a cone - a circle of radius
    0.95783 on the unit sphere - and sweeps 0.02395 rad/s. Using 0.025 would have
    quietly refreshed 4.3% less often than the cadence it is meant to reproduce.

### What the field covers, and why the fallback stays

Measured per shaded pixel, at 1920x1080:

| segment | field answers | falls back to the per-pixel path |
|---------|---------------|----------------------------------|
| terrain | 96.569%       |  3.431%                          |
| water   | 99.949%       |  0.051%                          |
| foliage | 45.644%       | 54.356%                          |

FOLIAGE SETTLES IT: the per-pixel shadow cone and `compute_ao` CANNOT be deleted.
The air cell against a leaf face is usually another leaf voxel, so the sampler's
solidity gate drops every one of the eight taps and the field has nothing to give
for over half a canopy view. Removing the fallback would render it black. This is
also, at last, the cause of round D's "foliage saves nothing": on that camera the
field is answering for under half the pixels.

The fallback costs nothing when unused - both branches sit behind `vlf.valid` -
so it stays, and the plan document stops calling for its removal.

### What DID come out: the screen-space reprojection cache

The other half of "one path, not two" is gone, and it was both a cost and a
correctness bug.

It was tested BEFORE the field (`reuse_shadow` before `vlf.valid`), so wherever
reprojection hit - which is only on a STILL camera, the one state it engages in -
a smooth world-space gradient was overwritten by a value some earlier frame had
traced from ONE binary ray. Shadows changed character when the camera stopped.

And it no longer paid. Timed on a still camera with it compiled out:

| segment | cs_main, cache ON | OFF    | the cache is worth |
|---------|-------------------|--------|--------------------|
| terrain |  4.728 ms         |  4.302 | +0.426 (it COSTS)  |
| foliage | 16.598            | 16.523 | +0.075 (noise)     |
| water   |  2.967            |  2.931 | +0.036 (noise)     |
| sky     |  0.382            |  0.431 | -0.049 (noise)     |

With the field answering 96.6% of terrain, the reprojection was re-deriving what
it had already been given. Removed: binding 15, `REPROJ_EPS2`, the sun-staleness
dither, the four reuse flags threaded through `shade`, 33 MB of VRAM and the
full-res Rgba32Float ping-pong COPY every frame - 66 MB/frame of traffic that a
sky-facing frame paid in full for nothing.

`light_out` STAYS, and the plan was wrong to call for deleting it: it is also
cs_taa's reprojection source and the grass pass's blade lighting. What is gone is
the history it was copied into, and everything that read it.

The legacy per-pixel GI path (`GI_PROBE_MODE = 0`, a reference the shipped
renderer does not run) lost its screen-space reprojection with it, because the
positions it matched against lived in that history. It accumulates per pixel
instead. Narrowing a reference path is a real narrowing, and it is recorded here
rather than left to be discovered.

### The view frustum: built, measured, rejected

Scoping the near tier by a padded view cone as well as by radius was built and
measured. `voxlight_turnaround_artifact` stands the camera facing away for 4.3 s
of continuous sun motion, turns 180 degrees, and diffs every recovery frame
against a fully converged field:

| rule                                      | near | sweep    | error at the turn (mean / p99 / max, /255) |
|-------------------------------------------|------|----------|--------------------------------------------|
| radius only                               | 2692 | 1236 wgs | 0.204 /  6.5 /  54.3                       |
| frustum + radius                          | 1337 | 1089 wgs | 0.411 / 13.9 /  68.8                       |
| control: field frozen for the whole 4.3 s |      |          | 0.836 / 30.0 / 157.5                       |

It works - 1,215 of 1,351 near blocks are newly promoted by the turn, so the cone
really is culling - and it is not worth it. 12% off a sweep round (8% on the
streamed world) for DOUBLE the peak error on the frame you turn. The near tier is
only ~18% of a sweep round; the far remainder is 920 of 1,236 workgroups, so no
frustum test can make a sky-facing camera "do almost no work". The thing that
does that is the sun pacing, which takes it to zero. The tier stays
view-independent, which is what the original note on `near_count` argued and this
is the measurement it never had.

The SHIPPED rule's own turn-around number is worth having and is now pinned:
0.204/255 mean at the turn, decaying smoothly to 0.096 mean and p99 2.3 over
1.05 s. Nothing pops, because a far block keeps a VALID record - no pixel changes
shading path; the shadow is simply slightly in the wrong place and slides home.

### Water shadows are as smooth as terrain shadows

`shade_water_top` pushed the field's continuous sun visibility through
`smoothstep(0.41, 0.59, sun_vis)`, a band 45/255 wide, before using it; terrain
multiplies by the same value directly. That is why water read STEPPED beside
terrain that read smooth, and it is the same narrowness round G blamed for the
water flicker report. Water uses `sun_vis` directly now.

Measured by `water_reads_lit_or_shadowed` on a wall cast across a sheet: the
0.72-0.90 ratio band - the penumbra - goes from 1.1% to 4.0% of the shaded
pixels, and the shadowed side sits at 0.330 of the lit side (was 0.316). The band
is small either way because the physical penumbra in that scene IS small (a
0.07 rad sun cone and a wall five voxels away casts a sub-voxel one); the 3.6x
between them is the signal. That test's second assertion was INVERTED - it used
to require a hard edge - and the reversal is stated in its doc comment rather
than buried in a diff.

The stylization is untouched: the plate quantization, the per-cell tone ladder
and the hard-edged foam are all different terms.

### Look, round H

`dump_lookdev_views` captured before and after and diffed per pixel:

| view          | mean/255 | max | >8/255 |
|---------------|----------|-----|--------|
| canopy_top    | 0.029    |   1 | 0.00%  |
| forest_mid    | 0.003    |   1 | 0.00%  |
| meadow        | 0.017    |   1 | 0.00%  |
| tree_close    | 0.002    |   1 | 0.00%  |
| water_graze   | 0.037    |   1 | 0.00%  |
| water_terrace | 0.033    |   7 | 0.00%  |
| water_view    | 0.017    |   1 | 0.00%  |
| water_shadow  | 0.707    |  43 | 3.13%  |

EVERY view is within 1/255 except the one built to show a shadow across water.
Removing a whole screen-space lighting mechanism, rescheduling the update pass
and rewriting the water shadow ramp moved exactly the pixels the water shadow
ramp was meant to move and nothing else.

Read directly at 5x on `water_shadow`: BEFORE, a hard near-vertical light/dark
boundary one or two pixels wide runs down the frame, flat mid-blue on one side
and flat navy on the other. AFTER, that boundary is gone and the same region
darkens gradually across about 40 pixels. The faceted cell mosaic - distinct flat
quads at different blues with hard cell-to-cell edges - is intact in both, and
the chunky white foam is unchanged.

### What round H does NOT claim

The absolute frame times moved between the before and after runs on passes
neither touched (`clouds` reads 1.114 ms before and 0.930 after on a sky frame),
which is this machine's documented thermal spread. Every number quoted above is
either a within-run control (the frame with and without the dispatch, back to
back), a counter, or a pixel diff of two images rendered by the same build.

End-to-end fps is not quoted, for the reason round G gives.

## Reproducing

    cargo test --lib live_session_profile -- --ignored --nocapture
    cargo test --lib voxlight_sun_lag_error -- --ignored --nocapture
    cargo test --lib voxlight_turnaround_artifact -- --ignored --nocapture
    cargo test --lib rt_vs_software_timing -- --nocapture --ignored
    cargo test --lib dump_lookdev_views -- --ignored --nocapture
    cargo test --lib flicker_probe_rt_views -- --ignored --nocapture

Re-run on the SAME machine after the rework and compare against this table,
not against docs/PERF.md.
