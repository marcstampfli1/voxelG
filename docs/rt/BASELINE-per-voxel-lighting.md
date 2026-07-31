# Baseline and measurement rounds for the per-voxel lighting rework

The BEFORE-baseline below was taken on `feat/per-voxel-lighting` at commit
a113962 (the manifest fix only, no lighting change yet). THIS IS THE ONLY
VALID COMPARISON POINT for the rework: the numbers in docs/PERF.md were
measured on different hardware and are not comparable to anything measured
here. Rounds A through D follow it, and round D is the one that measures the
feature rather than the renderer around it.

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

## Reproducing

    cargo test --lib rt_vs_software_timing -- --nocapture --ignored
    cargo test --lib dump_lookdev_views -- --ignored --nocapture

Re-run on the SAME machine after the rework and compare against this table,
not against docs/PERF.md.
