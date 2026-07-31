# Before-baseline for the per-voxel lighting rework

Taken on `feat/per-voxel-lighting` at commit a113962 (the manifest fix only, no
lighting change yet). THIS IS THE ONLY VALID COMPARISON POINT for the rework:
the numbers in docs/PERF.md were measured on different hardware and are not
comparable to anything measured here.

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

Round B is the honest cost of a MISSING early-out: `voxlight_sample` ran its
full eight-tap gated loop and paid an `is_voxel_solid` descent plus a brick
table lookup per tap before failing. Terrain shade went 9.24 -> 10.01 ms.
Round C checks the centre voxel first and answers a miss in one lookup.

READ ROUND C CAREFULLY. It is NOT evidence that the light field is faster.
The timing harness builds its own bind groups with an unpopulated
`VoxLightBuffers`, so in every one of these rounds the field is empty,
`voxlight_sample` returns invalid, and shading falls back to the previous
per-pixel path. All three rounds measure the SAME renderer, plus or minus the
cost of asking the field a question it cannot answer.

Round-to-round variance is also large: the software column alone reads
13.85 / 15.11 / 13.16 ms for terrain across A/B/C, about 13% spread on a path
whose behaviour did not change. Round C landing under round A is inside that
spread and must not be reported as a win.

What round C does establish: the field is performance-NEUTRAL when it is not
populated, which is the precondition for it being a win when it is.

## Still to measure

The A/B that matters has not been run. It needs the timing harness to bind a
POPULATED field (as `voxlight_field_populates_and_is_sampled` already does)
so the shadow ray and AO evaluation are actually replaced rather than
supplemented. Until then there is no number for the feature itself.

## Reproducing

    cargo test --lib rt_vs_software_timing -- --nocapture --ignored

Re-run on the SAME machine after the rework and compare against this table,
not against docs/PERF.md.
