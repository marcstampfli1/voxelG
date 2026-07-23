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

## Baselines (2026-07-23b, + glint gate, refl accum, probe staging; 1920x1080 all-dirty)

| scenario     | software | RT-primary       | RT-prim + probe GI |
|--------------|----------|------------------|--------------------|
| terrain      | 9.76     | 8.27  (1.18x sw) | 9.10  (1.07x sw)   |
| foliage      | 11.02    | 10.02 (1.10x sw) | 10.93 (1.01x sw)   |
| water-close  | 11.35    | 10.35 (1.10x sw) | 11.21 (1.01x sw)   |

Per-pass water-close: transp 9.2-9.6 ms (THE whale). Foliage main ~10.3 all-dirty.
Live static frames (tile-gated): ~1.9-2.5 ms GPU total.

## Proven (shipped)

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
3. CPU render() 3.7-4.6 ms while GPU is 2.4 (static): encoder/submit overhead
   across ~10 passes. Candidate: merge compute passes (multiple dispatches per
   pass; wgpu inserts the same storage barriers) - est. -0.5-1 ms CPU.
4. GI textures rgba32float -> rgba16float (radiance fits f16; light cache must
   STAY f32 - the position match breaks at f16 precision).
5. 44 s cold pipeline compile after shader edits (driver compile bomb): shrink
   specialization count / investigate driver cache priming.
6. ReSTIR-style reuse on probe gather rays (GI phase 3, promised).
7. TAA history copy -> ping-pong (same trick as #4).
8. TAA edge ripple: cs_taa reprojects history to a ROUNDED pixel and never
   unfilters the Halton jitter, so high-contrast edges (tuft silhouettes,
   terrace lips) oscillate on the 8-frame cycle - the flicker rig's
   GI-independent noise floor (~5k strong px on the meadow view). Candidate:
   bilinear history sample at the sub-pixel reprojected position + jitter
   compensation. RT-independent, cosmetic-tier.

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
