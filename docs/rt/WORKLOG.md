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
- Toroidal fix: accel now builds in WINDOW-LOCAL space (contiguous BVH) reading
  occupancy from the wrapped storage slot; a shifted-origin streaming test proves
  the same crafted scene resolves identically at origin 0 and (96,160). Any-hit
  RT shadow occlusion validated vs the CPU raycaster (2464 samples, 0 disagree).
- Phase 1b-ii shader path IN: shared voxel_ray_query primitive extracted to
  shaders/rt_voxel_query.wgsl (used by BOTH the probe and the render shader). The
  render shader routes ALL occlusion (sun shadows, god rays, sky-access AO)
  through one `shadow_occluded` dispatcher: the software variant forwards to the
  DDA trace_any (byte-identical to the pre-RT shader + a tiny wrapper), the RT
  variant (raymarch_source_variant(true)) enables wgpu_ray_query, binds the world
  TLAS/brick_map/aabbs at group0 22..24 and forwards to rt_occluded. Both
  variants naga-validate (proves cross-file WGSL forward refs resolve). Not yet
  wired to a pipeline - that is next.
  NEXT Phase 1b-iii: build the RT compute pipeline + bind group (TLAS + brick_map
  + aabbs) in the HEADLESS harness and A/B render software-vs-RT occlusion (must
  match within tolerance) before touching the windowed Renderer::new.
- Phase 1b-iii DONE: RT occlusion runs end-to-end through the render shader and
  MATCHES software pixel-for-pixel. RT resources bind at their own group 1
  (create_rt_bgl / make_rt_bg) so the software group-0 layout + bind group are
  byte-identical - RT is purely additive. The A/B test (rt_shadows_match_software)
  renders one cs_main frame with each occlusion path on the same demo-terrain
  scene: mean |dRGB| = 0.003 over 320x200, only 10/192000 channels differ (>24),
  i.e. grazing shadow edges. First cut was WAY off (mean 16, canopies black)
  because the software occluder is material-aware (fringe never blocks; leaves /
  decorations use foliage_subvoxel near/far cutout) while raw RT blocked every
  occupied voxel. Fixed by extracting the ONE occluder rule into
  `shadow_voxel_occludes` (raymarch.wgsl), called by BOTH trace_any and the new
  `rt_brick_occludes` (marches a candidate brick's voxels, passing THROUGH
  non-occluders like the software DDA). Test-suite parallelism: RT + render
  devices created concurrently crashed the driver, fixed with a GPU-init
  serialization lock (gpu_init_serial); RUST_TEST_THREADS=4 is stable again.
  NEXT Phase 1b-iv: wire the same group-1 recipe into the windowed Renderer::new
  behind a default-OFF capability flag (env VOXELG_RT), rebuild the accel on world
  upload/edits, select RT vs software pipelines per frame. Software stays default.
