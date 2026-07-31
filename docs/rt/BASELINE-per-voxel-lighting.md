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

## Reproducing

    cargo test --lib rt_vs_software_timing -- --nocapture --ignored

Re-run on the SAME machine after the rework and compare against this table,
not against docs/PERF.md.
