# Performance ledger

The standing goal: the most optimized program on earth, improved indefinitely.
Process: every candidate optimization is A/B-measured on the harness BEFORE it
ships; regressions hunted proactively, not on report. Look-affecting tradeoffs
get shown with numbers before shipping, never after.

## Harness

- `cargo test --lib rt_vs_software_timing -- --nocapture --ignored`
  Scenarios: terrain / foliage / water-close (grazing water worst case), each
  with per-pass breakdowns (all-tiles-dirty = the moving-camera worst case).
- Live: `VOXELG_UNCAPPED=1 VOXELG_GPU_PROFILE=1 VOXELG_RT=1 ./target/release/voxel`
  prints `gpu pass ms` + `cpu frame ms`. Panel is 144 Hz; Mailbox paces the loop
  to refresh on static scenes - use UNCAPPED (Immediate) for headroom numbers.
- Any config where software beats RT is a finding, not a curiosity.

## Baselines (2026-07-23c, + flat refracted-hit AO; 1920x1080 all-dirty)

| scenario     | software | RT-primary       | RT-prim + probe GI |
|--------------|----------|------------------|--------------------|
| terrain      | 9.83     | 8.32  (1.18x sw) | 9.14  (1.08x sw)   |
| foliage      | 10.93    | 10.03 (1.09x sw) | 10.96 (1.00x sw)   |
| water-close  | 9.65     | 7.74  (1.25x sw) | 8.48  (1.14x sw)   |

Water-close static camera 7.34 ms. Per-pass water-close: SW transp 5.80,
RT transp 4.71. Terrain/foliage rows are the 23b measurements (unaffected
by the water change); water-close re-measured after flat AO.

Per-pass water-close: transp 9.2-9.6 ms (THE whale). Foliage main ~10.3 all-dirty.
Live static frames (tile-gated): ~1.9-2.5 ms GPU total.

## Proven (shipped)

- Moving-camera reflection reuse (one path for both camera states): the
  reflection history is reprojected by ABSOLUTE surface position into the
  previous frame and reused when the stored position matches AND the view
  ray to the point rotated < ~2 degrees since last frame (computable from
  prev_origin, zero extra storage) - the angle gate bounds reflection
  parallax by construction and is distance-adaptive for free (close water
  re-traces, far water reuses). New camera.prev_valid uniform flag (stays
  set under motion, unlike reproject_lighting; repurposed _pad8). Modeled
  strafe on water-close: transp 4.80 -> 3.88 ms (refl trace 2.61 -> 2.01);
  static bit-identical (3.55). Live: ~170 fps static / ~120 moving at
  water (Marc, motion look signed off). BENCH LESSON: rt_vs_software's
  static row regressed 7.34 -> 8.47 after the gate change because the
  bench never set a prev camera - it modeled a state the live game is
  never in; benches must model live inputs (fixed).
- Flat AO on refracted water hits: the transp cost-split attributed 2.55-2.66
  ms (the single largest water cost) to AO rays traced for bed pixels whose
  contribution absorption + tint then swamp. Replaced with a flat 0.4;
  masked pixel diff vs traced AO: mean 1.3/255, 0.018% of pixels > 10/255
  (above-water view; a submerged camera never runs this path). water-close
  transp RT 7.28 -> 4.71, frame 11.06 -> 8.48 all-dirty / 9.90 -> 7.34
  static; software gains too (transp 7.48 -> 5.80). Look signed off with
  side-by-side stills. LESSON: the first look comparison used an UNDERWATER
  view where shade_water_top's refraction branch never executes - identical
  images that proved nothing (Marc caught it). Compare look changes in a
  view that exercises the changed code path.
- Probe-GI staged deterministic gather (the "breathing shadows" root fix):
  probe update rays were hashed on camera.time, so every 8-ray round was a
  fresh random estimate and the per-round EMA random-walked forever - visible
  as probe-cell patches slowly growing/shrinking wherever indirect light
  dominates (shadow interiors, underwater; RT-only). Now each probe cycles a
  fixed 64-dir spherical-Fibonacci set (8/round, epoch = gi_round/DIV) into a
  staging SH and folds ONLY complete estimates into the displayed SH: a
  static scene converges to an exact constant. Flicker rig (long horizon,
  strong-flip gi-on vs gi-off): underwater 5853 vs 0 -> 0 vs 0, meadow 14742
  vs 2495 -> 2520 vs 2495, tree_shadow 9594 vs 5008 -> 5012 vs 5008. Perf
  NEUTRAL (same ray count, fewer hashes; timing table above re-baselined,
  RT >= sw everywhere). NOTE an intermediate attempt (deterministic dirs,
  still per-round EMA) turned the walk into a 64-frame periodic ripple and
  made tree_shadow WORSE (9594 -> 13818): never per-round-EMA subset
  estimates - fold complete spheres only.
- Glint shadow gate (spec > 0.002 bound, contribution < 1 LSB below): water
  transp SW 9.62 -> 7.51 ms, RT 9.16 -> 8.46; helps every config.
- Temporal reflection accumulation, 8x8-BLOCK staggered (static camera traces
  half the water reflections per frame, history-blended, position-validated):
  static water-close 12.99 -> 11.68 ms - now cheaper than moving, as it must
  be. LESSON: the first attempt used a PIXEL checkerboard and measured ZERO
  gain - SIMT warps containing both tracing and skipping threads pay the trace
  latency regardless. Stagger at workgroup granularity or don't bother.

- wgpu scratch-buffer cache: build_acceleration_structures 12-28 ms -> 1.2-2 ms
  CPU per call (also a prepared upstream PR, docs/rt/wgpu-prs PR 3).
- Per-brick BVH + async off-thread rebuild (frame thread: 128 KB snapshot +
  swap): restored RT edge (see baselines) with zero streaming hitch. Origin
  shifts rebuild synchronously by design (position staleness cannot be gated).
- Refracted-hit shadow = provable constant (water occludes shadow rays, so an
  underwater hit is always shadowed): transp 10.16 -> 9.61 SW / 11.20 -> 10.65
  RT. Image-identical by proof.
- Half-res god rays (depth-weighted upsample): compose spikes 6-7 -> ~0.3-0.8.
- Probe GI amortization (1/8 per frame): probe pass ~0.9 -> ~0.45 ms.
- sky_access removal (5 shadow rays/px, replaced by probe GI skylight):
  software main 16.9 -> ~10; also killed the fake overhead leaf shadows.
- Coarse cloud-density for the 3/px ground-shade taps: compose 1.3-2.5 -> ~0.6.
- Secondary-ray range cap 200 (fog owns everything beyond).
- gi ping-pong via paired bind groups (frame parity) - eliminated the per-frame
  full-res rgba32float copy outright (probe mode never even read it).

## Rejected (measured, reverted)

- Tile-level BVH (16^3 primitives): sub-ms rebuilds but gave the hardware's
  traversal advantage away (in-shader brick marching); RT fell to ~1.0x sw.
  Superseded by per-brick + async rebuild.
- All-water-tile skip for transparent-skipping rays: pure noise on water-close.
- Canopy shadow transmittance cap: no measurable gain; per-ray miss counts
  flickered dappled shadows. (Foliage cost is PRIMARY-ray blade cutouts.)
- Water secondary-shading LOD + refraction cap 40 + grazing reflection cap:
  perf ~neutral on water-close and damaged the look (user reverted).
- Motion-reprojected lighting cache, second attempt (absolute-position
  anchors kept on reuse, no re-anchoring): bench pairs measured only -0.16
  to -0.27 ms on water_strafe main (the pass is trace-dominated; the
  reusable shading pool is ~0.2 ms), and Marc saw TERRAIN WARPING in live
  motion within minutes - water immune, exactly the cacheable/non-cacheable
  split, confirming the transport still drifts visibly even with fixed
  anchors. Rejected on both counts. PARKED: the mechanism only becomes
  worth a third attempt if per-pixel shading cost ever grows to dominate
  the moving frame, and then only with a stricter validation design.
- Half-resolution reflection pass (cs_water_refl, quad-shared reflected
  content): the largest measured win of the hunt - water_mid 93 -> 169 fps,
  grazing 75 -> 155, strafe 63 -> 97 - but REJECTED BY MARC'S EYE and
  reverted (906be0c): sharp upsample showed quad edges, 4-tap surface-aware
  bilinear still edgy from distance, 9-tap tent read as blur. Per the
  standing rule, lower internal resolution only counts as an optimization
  when it is invisible; this was not. PARKED VARIANT (on Marc's ask only):
  distance-GATED half-res - full-res beyond ~30 voxels (where the flaws
  showed), half-res near where quads subtend less than a wave; would keep a
  large share of the win invisibly. The revert keeps the committed pass
  history for easy re-application.
- Fog-bounded reflection range at grazing (cap skimming rays, resolve to
  fog): NULL on the bench pairs - grazing reflections mostly hit NEARBY
  terraced terrain, not distant geometry, so range caps save nothing. The
  cost is ray count x moderate traversal. Look risk for zero gain: rejected.
- Column-anchored reflection-history validation (wave-tolerant reuse):
  NULL - reuse was already engaging; validation was not the blocker.
- 4-phase static reflection stagger (trace each 8x8 block every 4th frame
  instead of every 2nd): the live deterministic bench measured IDENTICAL fps
  on every water segment across interleaved before/after pairs (93/93,
  74/74, 62/62) with water-free control segments agreeing within 1-5% - the
  harness-projected ~0.9 ms static win does not survive the full live frame.
  Look risk for zero gain: rejected. (Also the bench tool's first catch.)
- Acquire-late render restructure (compute submitted before swapchain
  acquire): no live fps change at any distance; the ~4 ms live "CPU" is not
  an acquire stall. Reverted; hunt item 3 reopened with the discriminators.
- Compute-pass merging (hunt item, est. -0.5-1 ms CPU): MEASURED at 0.013 ms
  saved (cpu_encode_bench: split encoding 0.027 ms/frame, merged 0.014;
  encode+submit total only 0.416). The estimate was off ~50x - encoding was
  never where the live ~4 ms CPU went (it was the early swapchain acquire,
  see Proven). Candidate killed by a 2-second micro-bench before any
  refactor risk.
- Motion-reproject of the lighting cache: fps-scaled resample drift ("warping").
- Probe-lookup grid-coherence mask (world-anchored noise offset of the
  trilinear phase, to hide probe-grid periodicity): the ~1.3-voxel noise
  cells read as "random squares of shadow" on the pond floor (user-rejected).
  Superseded by the actual root fix - GI rays skipping water (the probes had
  been recording the water surface into the Chebyshev moments as solid
  geometry, the source of ALL underwater-only GI artifacts).

## Hunt list (ranked, next up)

1. Water transp (~7.3-7.5 in the 2026-07-23b baselines; Marc reports ~100 fps
   live staring at water vs the 300 floor). Refraction routing is RESOLVED
   and shipped: refractions go through the software DDA even in RT builds
   (short rays never amortize the ray-query setup; see the comment at the
   trace_no_water call in shade_water_top). NEXT: the transp_cost_split
   harness (PROF_TRANSP_* toggles) attributes the pass into refl trace/shade,
   refr trace/shade and tail - run it clean, then pick candidates where the
   milliseconds are (leading: extend temporal reflection accumulation to a
   MOVING camera - history is position-validated per pixel, unlike the
   reverted light-cache reprojection). Further shading LOD needs Marc's
   sign-off WITH numbers.
2. Foliage main (~10 ms all-dirty): primary-ray per-blade cutout tests through
   crowns. Needs an in-shader cost split first (primary vs canopy-AO vs GI
   sample); then candidates: cheaper blade parameterization, per-brick blade
   masks (skip cutout-free cells), NOT distance/res cuts.
3. CPU render() 3.7-4.6 ms while GPU 2.4 (static): NOT encoding (0.027 ms),
   NOT submit (0.416 total), and NOT an acquire stall - an acquire-late
   restructure (compute submitted before acquiring) produced NO live fps
   change at any distance and was reverted. Next discriminator: bracket the
   live render() internally (acquire / encode / submit / present) and check
   what the app-side "cpu frame ms" actually includes (world sim, uploads,
   leaffall) before attributing further.
3b. GRAZING-ANGLE water (Marc: very close + low angle drops to ~80 fps):
   reflection rays skim parallel to the surface, traveling far and visiting
   many bricks exactly when the screen is full of water. Candidates: deepen
   static reflection amortization (1/4 blocks per frame), tighter secondary
   t-cap where fog owns the result (sub-LSB proof required), grazing-aware
   reflection origin bias. Look-gated, stills + numbers.
4. GI textures rgba32float -> rgba16float (radiance fits f16; light cache must
   STAY f32 - the position match breaks at f16 precision).
5. 44 s cold pipeline compile after shader edits (driver compile bomb): shrink
   specialization count / investigate driver cache priming.
6. ReSTIR-style reuse on probe gather rays (GI phase 3, promised).
7. TAA history copy -> ping-pong (same trick as #4).
8. TAA edge ripple - PARKED after four rig-measured attempts (2026-07-23,
   flora stage 0). Baselines: meadow 2527 strong, water_top 1959. Results:
   (a) bilinear history + current-jitter unfilter on ALL paths: meadow 3314
   (thin blades smear); (b) bilinear, no compensation: meadow 2877 /
   water_top 2114 (float reprojection resurrects the jitter the old integer
   rounding cancelled); (c) jitter compensation on the reprojection path
   only: meadow 2696 / water_top 2115 (still above baseline); (d) variance
   clipping gamma 1.25 instead of min/max clamp: water_top 1746 (-11%, the
   only sub-baseline result) but meadow 3021 and tree_shadow 5820 (tight
   statistical box rejects history at thin blades). CONCLUSION: the old
   rounded fetch is accidentally near-optimal for static views; the ripple
   is the current sample's jitter surviving the 0.9 blend. Next probes if
   reopened: gamma sweep 1.5-2.0, or blend raised for converged pixels.
   RT-independent, cosmetic-tier; flora stages gate on existing baselines.

## Live bench (the end-to-end instrument)

`VOXELG_BENCH=1 ./target/release/voxel` - a deterministic in-game benchmark:
standard demo world, frozen sun t=30, RT + uncapped + 1920x1080 forced, five
fixed camera segments (water_mid / water_grazing / water_strafe / terrain /
foliage), 2 s warmup discarded per segment, avg + p1 + p99 fps printed, then
self-exit (_exit; the normal teardown path deadlocks - known shutdown bug).
Guards: battery warning (mains-online check), 140 s stall watchdog thread.
PROTOCOL: AC power only, no other GPU apps, keep both candidate binaries and
run INTERLEAVED pairs with settle gaps; the water-free terrain/foliage rows
are CONTROLS - if they disagree >~5% across runs, discard the session (a
battery session drifted them 35%). Live baseline (2026-07-23, post flat-AO +
moving-reuse): water_mid 93, water_grazing 74, water_strafe 62, terrain 218,
foliage ~200.

## Flicker rig (the temporal-stability instrument)

`cargo test --lib flicker_probe_rt_views -- --nocapture --ignored`
Full live chain (RT primary, probe GI, clouds, god rays, TAA + Halton
jitter) at native 1920x1080; world-scanned cameras (flat meadow, pond
interior, lone-tree cast shadow); 256-frame warmup then 24 readbacks at
stride 8 (in phase with the jitter cycle, so TAA ripple aliases out and
slow walks stand alone). Metrics per view: strong/faint delta-flips,
Schmitt-trigger "breathing" transitions; heatmaps to target/lookdev/
(breathing blue, strong red, faint orange). gi on/off pairs isolate
probe-GI causes; RigOpts has sun_base / animate / jit_freeze switches.

## Rules

- A/B on the harness or it does not ship. Rejected = reverted + logged here.
- Provable constants are free wins - hunt them in every hot shader path.
- This file updates in the same change as the optimization it records.
