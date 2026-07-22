# Hardware-RT rework worklog (branch feat/hardware-rt)

Durable progress log for the autonomous RT build. Newest at the bottom.
Architecture + rules: see the plan (scratchpad draft) and memory project-voxelg-rt.

## Standing rules
- Software renderer stays a working fallback behind flags; RT is a parallel path.
- Optimization + clean reusable API at EVERY step (floor not ceiling - hunt more).
- One shared `voxel_ray_query` shader primitive for all ray users.
- wgpu contributions: local fork + [patch], build against modified source; each PR
  is a standalone branch + description in wgpu-prs/, PREPARED not opened.
- Suite green at every commit; commit each verified step.

## Log
- Phase 0 DONE (commit 3a491d1): wgpu 23 -> trunk c97d22f, verified 38 tests pass.
- Phase 1 START: accel module (AABB-per-brick BLAS from world) + in-brick DDA
  resolve + RT shadows behind a flag, A/B vs software.
- Phase 1a DONE: `src/accel.rs` (`build_world_accel` -> WorldAccel{blas,tlas,
  brick_map,aabb_buf}) enumerates non-empty bricks into one AABB each; the shared
  in-brick DDA primitive lives in `shaders/accel_probe.wgsl` (candidate AABB ->
  Amanatides-Woo march of the 4^3 voxels -> generateIntersection, nearest tracked
  in registers so the resolved voxel never depends on the RT core's candidate
  order). Validated in isolation against ground truth: a crafted flat floor
  (36/36 rays hit the top face y=61 to <0.05 vox) AND the trusted CPU raycaster
  over floor+pillars (231 clean-face hits match EXACTLY, covering floor tops and
  pillar side faces = multi-brick lateral traversal; inherently-grazing edge/
  corner rays are classified and tolerated, the software DDA splits them the same
  way). Both tests skip cleanly with no RT adapter. Full suite green (40 pass).
  NEXT Phase 1b: RT-enable the render device behind a capability check and wire
  the shadow ray (main shade / water glint / reflection / god rays) through the
  same primitive, A/B vs the software `trace_any`.
- Design note: the register-tracked nearest-hit (not the committed intersection)
  is the reusable pattern for primary/AO too - it yields the exact voxel+material
  with no re-derivation from a committed t. generateIntersection still fires so
  the hardware culls bricks past the current nearest (the coarse-to-fine win).
