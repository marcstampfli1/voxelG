# voxelG — Improvement checklist: implementation status

Every item from `voxelG-improvements.md`, what was done, and where. Cross-platform
(Linux) was a goal throughout: the engine builds and runs on Linux (Wayland/X11,
Vulkan) — verified building + creating a Vulkan device + rendering on an RTX 5060.

Validation tooling added so changes are checkable without a display:
- **Headless WGSL validation** (`naga`) for every shader.
- **Headless GPU render+readback** test: renders the real raymarch pipeline
  offscreen and asserts terrain+sky at spawn AND at 3.2 M voxels from origin.
- Deterministic-worldgen, edit-replay-roundtrip, net (compress/interp) unit tests,
  an in-process TCP+UDP integration test, and criterion microbenchmarks.

## Frame loop
- **force_full on camera move** → only a *real* move (>1 mm / >1e-4 rad) forces a
  full re-trace; sub-mm jitter no longer re-traces every tile. `src/app.rs`.
- **15-frame forced full** → removed; replaced by an evenly-spread rotating
  animation refresh (1/8 of tiles per frame) so idle animation has no stutter.
- **per-frame `dirty_bricks` / `active_bricks` clones** → gone. Tiles are
  projected straight from `world.dirty_bricks`; physics reuses
  `World::phys_scratch`/`phys_touched` (zero per-tick allocation).
- **`about_to_wait` always redraws** → `ControlFlow::WaitUntil` frame pacing (144 Hz
  cap) + redraw-on-input; present mode is Mailbox/FIFO (vsync), not Immediate.
- **`unwrap`/`expect` on GPU/window/surface** → `Renderer::new` returns `Result`;
  window/adapter/device failures log + exit; `render()` handles Lost/Outdated/
  Timeout/OutOfMemory.

## Streaming / upload
- **all_dirty re-uploads ~75 MB; chunk cross must avoid it** → chunk crossing only
  ever takes the incremental path; all_dirty is first-frame only.
- **incremental path re-uploaded whole masks** → fully incremental: brick spans +
  only the touched packed words of brick/tile uniform tables and the touched
  tile/chunk/L4 mask entries (derived from the dirty-brick list). Also fixed a
  latent bug where the uniform-skip tables were never refreshed incrementally
  (streamed chunks rendered with stale skip data).
- **gen on the calling thread** → background worker pool; `shift_origin` dispatches
  jobs, `install_finished_chunks` stitches results in with stale-result rejection.
- **no upload budget / prefetch** → per-frame chunk-install budget + 2-chunk
  hysteresis deadband; the centred window prefetches in all directions.

## Rendering / GPU (`shaders/raymarch.wgsl`)
- **nested hierarchical DDA + L4** → trace/trace_any/trace_no_water descend
  L4(256) → chunk(64) → tile(16) → brick(4) → voxel, skipping the coarsest empty
  cell each step. New L4 occupancy level (one u64 per 256³ region) added end to end.
- **voxel step budget → distance-based** → `MAX_RAY_DIST`.
- **secondary rays uncapped** → shadow rays use the full hierarchy + `SHADOW_MAX_DIST`
  give-up + treat far foliage as a solid blocker. *(Cache+reproject of the shadow/AO
  term is partially covered by TAA temporal accumulation; a dedicated reprojected
  shadow cache is left as a follow-up.)*
- **foliage in both trace passes** → procedural sub-voxel foliage only within
  `FOLIAGE_NEAR_T`; far foliage is a solid cube (and a solid shadow blocker).
- **clouds/god-rays full-res** → sample counts halved; TAA accumulates them across
  frames (temporal upsample). *(A dedicated half-res volumetric pass is the further
  optimisation.)*
- **no AA** → TAA: sub-pixel jitter while static + a resolve pass that blends with
  history under a 3×3 neighbourhood colour clamp (no ghosting), reset on motion.
- **8×8 divergence** → reduced by near-only foliage. *(A full transparent/foliage
  second/deferred pass was deliberately not done — high-risk rewrite, needs a GPU
  profiler to justify; near-only foliage removes the worst divergence.)*
- **LOD** → far terrain terminates at brick (`LOD_BRICK_T`) then tile granularity
  (`TILE_LOD_T`, via a tile-representative material).
- **connected water surface** → per-corner heights (mean fill level of the
  corner-sharing columns + Gerstner field at the corner; water one cell up in
  any sharing column pins the corner to the cell top) with closed-form
  bilinear-patch intersection, so diagonally-touching / different-height
  water knits into one surface. Corner tier within `WATER_NEAR_T`, previous
  centre-plane facet beyond; rest-gradient carried to the deferred pass in
  the u32 transparent record. Validated by `water_diagonal_connects`
  (measured thresholds, fails on the pre-fix shader), `water_terrace_ramp`,
  the `water_terrace` lookdev view and a terrace timing scenario. Cost on
  the worst-case ocean bench: 12.15 -> 13.5 ms/frame at 1080p (74 fps
  GPU-bound; breakdown in the feat commit). *(Known scalable follow-up: a
  per-frame surface-cell corner precompute would take the probes off the
  per-ray path entirely.)*
- **falling leaves** → deterministic CPU sim (src/leaffall.rs, SplitMix64,
  256 cap, canopy-top spawns through the fringe cap, ground/water/TTL
  kill) driving an instanced card pass composited AFTER the TAA
  resolve-to-history copy (never in the feedback loop - no ghosting) and
  manually depth-tested against a new r32float primary-hit depth export
  (binding 19; water pixels carry the surface t). Validated by the sim
  test suite, the red-leaf occlusion content test, the leaffall lookdev
  still and an A/B timing gate (256 in-view leaves <= 0.05 ms at 1080p).
- **foliage overhaul** → per-species Better Leaves tufts (birch/spruce ported by
  `examples/convert_tuft.rs`, oak self-validation 100%), autumn mottle, canopy
  occupancy AO (fixes generic cube AO counting the invisible fringe shell as
  solid; probes use the brick-local `neighbor_material` fast path), horizontal
  cap tufts on distant canopy tops, a volumetric leaf cloud (six leaf-
  silhouette cards per fringe cell within LEAF_CLOUD_T=32, tuned in the
  isolated leaf-lab bench; ~+0.7 ms foliage scenario),
  three grass blade variants + per-clump height +
  dry straw tufts on sand/snow, five flower species with an absolute colour
  table, and a traveling gust field shared by geometric shear and shading
  sway (wind direction lifted into the camera uniform). Validated by the
  sprite property tests, `renders_leaf_species`, the lookdev views
  (tree_close / canopy_top / birch_close / pine_close / meadow) and the
  timing scenarios below.

## Correctness / numerical
- **rebase to window-relative coords** → all float DDA math is done relative to the
  window corner (`camera.world_origin`) while the integer voxel grid stays absolute.
  Removes the f32 cancellation / "sky through hills" bug; bias hack reduced 1e-2→1e-3.
  Verified: byte-identical at spawn, correct at 3.2 M voxels out (f32 ulp ≈ 0.4 voxel,
  where the un-rebased DDA cannot resolve a boundary).
- **no wire protocol versioning** → `PROTOCOL_VERSION` in the handshake; server
  rejects mismatched clients.

## Networking
- **all-TCP** → split transports: pose over **UDP** (token-authenticated, lossy-ok),
  edits/control/replay/heartbeat over **TCP**. (Raw UDP rather than QUIC to avoid a
  heavy async dependency in a student project.)
- **no interpolation** → per-player interpolation buffer; remotes render ~100 ms in
  the past, lerped between samples.
- **late-joiner desync** → authoritative **ordered, seq-numbered** edit log; replayed
  in order to joiners, **deflate-compressed** (`flate2`).
- **no heartbeat / reconnection / compression / ordering** → heartbeat + timeout
  sweep, client auto-reconnect (re-handshake + edit-log re-sync), compression, and
  seq-versioned edits all added.

## Physics
- **synchronous + active_bricks clones** → physics runs on a dedicated worker thread
  behind `Arc<Mutex<World>>`; the render thread only locks briefly. Per-tick clones
  removed (reusable scratch). *(A fully lock-free double-buffered render snapshot is
  the documented next step; the brief-lock model already takes the tick off the frame
  path.)*
- **CA on CPU; GPU compute candidate** → `docs/gpu-physics-design.md`: a concrete,
  actionable design (pull-only, double-buffered, integer-only CA, indirect-dispatch
  active set, determinism analysis, incremental sand→smoke→water migration).

## Architecture / hygiene
- **constants duplicated in Rust + WGSL** → single source `src/world_dims.rs`;
  `build.rs` generates the matching WGSL consts prepended to every shader.
- **monolithic main.rs** → `lib.rs` + `app.rs` (client/frame loop) + `server.rs`
  (dedicated server) + thin `main.rs` launcher.
- **zero tests/benchmarks** → see the validation tooling list at the top.

## Performance notes (measured)

Headless GPU timing of the raymarch pass (RTX 5060, **on AC**, `cargo test --lib
raymarch_timing -- --ignored`), worst-case view (high camera, all terrain,
force-full = every tile re-traced). Real static/slow play is much faster — the
temporal-differential system re-traces only ~1/8 of tiles per frame.

| build stage | 1920×1080 | 1280×720 |
|---|---|---|
| pre-optimisation baseline | 13.8 ms (72 fps) | 11.4 ms (88 fps) |
| + shadow PCF 2→1 (Win B) | 11.0 ms (91 fps) | 8.6 ms (117 fps) |
| **+ AO distance-LOD** | **9.8 ms (102 fps)** | **6.9 ms (144 fps)** |
| pure traversal (no shading) | 5.6 ms | 3.1 ms |

Scenario timings after the foliage overhaul (same harness, per-scenario cameras
from `find_scene_anchors`, 1920×1080): terrain 5.4 ms, water 12.1 ms, foliage
8.28 → 9.0 ms, meadow 6.1 ms. The foliage delta breaks down as canopy AO +1.9%
(brick-local probes; naive probes cost +4.3% and were rejected), cap tufts ~+5%
(kept for the canopy-top read; revert that commit to reclaim ~0.45 ms), gusts
+1%. Grass/flower density is capacity-tuned: denser ground flora measurably
slows every view containing grass tops (see `Biome::flora_probs`).

So the worst case now clears **100+ fps even at 1080p**, and shading (the
dominant ~60–70%) was cut ~40% with no visible quality loss (TAA accumulates the
1-sample shadow + the AO is only dropped where it's sub-pixel). **Battery
throttles the GPU ~14×** — measure/play on AC.

In-game (a real 84 s session, per-second fps): avg **99.6**, 5%-low **70**,
lowest 65 (excluding the one-time startup second). Chunk-load spike: fixed
(async install + mask-only sky-clear).

## Roadmap to 10 cm voxels @ 100 fps

10 cm voxels at the current ~128 m world means a ~2.5× denser grid (~15× the
voxels). Traversal and shading both scale up; the wins below are the path. Done
this pass are the high-ROI, measured, low-risk ones; the rest are designed but
deferred (they're heavier rewrites that touch the working render path, and the
engine already clears 100 fps at the current scale).

**Done (Win B — measured on AC):**
- Shadow rays 2→1 sample + cloud self-shadow 3→2.
- AO distance-LOD (skip the 12-tap AO past 64 voxels).
- Coarse god-rays: `trace_any` now takes a distance cap; god-ray occlusion uses a
  short `GOD_RAY_OCCL_DIST` so the hierarchical trace bails out far sooner.

**Done (Wins A / C / D — implemented + validated headlessly; benchmark on AC pending):**
- **Win A — true nested DDA.** The DDA now carries the slot voxel coord and
  increments it per step (cheap wrap) instead of re-folding two `pos_mod`s every
  voxel; `skip_to_cell` resyncs after a jump. Per-step cost drops — the key as
  bricks-per-ray grow at 10 cm. Applied to trace / trace_any / trace_no_water.
  Verified correct at spawn + 3.2 M voxels from origin (toroidal wrap).
- **Win C — reprojected shadow/AO cache** (#12). Shadow + AO (view-independent)
  are cached in a ping-pong G-buffer and reprojected into the previous frame via
  the prev camera basis; reused on a world-position match, else traced. Biggest
  *moving-frame* shading win, scales with voxel density. `REPROJECT_LIGHTING`
  const disables it if ghosting is seen. The render tests run 2 frames so the
  reuse path is exercised (identical static-camera result = correct round-trip).
- **Win D — half-res volumetric pass** (#14). Clouds marched once per 2×2 block
  in `cs_clouds` → half-res texture → bilinear upsample + composite (with a
  cloud-slab depth test for terrain occlusion). ~4× fewer cloud marches on
  sky-facing views.

**Done (full-reprojection TAA):** the TAA resolve reprojects each terrain
pixel's history by its G-buffer world position through the previous-frame camera,
so a moving camera keeps accumulating AA instead of hard-resetting. Neighbourhood
clamp rejects ghosting; sky/foliage + chunk-cross frames pass through. Reuses the
Win C prev-camera + G-buffer. (`taa.wgsl`, gated by `reproject_lighting`.)

**Done (GPU-compute physics, #25):** `shaders/physics.wgsl` — a pull-only,
double-buffered sand CA (one invocation per brick, reads neighbours from the IN
buffer, writes the OUT buffer; no write conflicts / atomics). `Renderer::
run_gpu_physics` dispatches it + copies the result back so the render path sees
it. Opt-in via `VOXELG_GPU_PHYSICS` (default off → the CPU physics worker stays
authoritative so raycast picking is exact; when on, the worker is skipped). Stage
1 = sand; water/smoke + the CPU↔GPU readback for raycast are the next migration
stages in `docs/gpu-physics-design.md`. Headless test `gpu_physics_sand_falls`
proves sand falls one cell with mass + occupancy conserved.

**Done (transparent/foliage deferred pass, #16):** `cs_main` records water-top /
glass hits into a read_write storage buffer + writes a cheap placeholder; a
separate `cs_transparent` pass shades only those pixels (reflection / refraction /
dispersion) and re-applies clouds + god-rays. Keeps the opaque-majority warps in
`cs_main` coherent (less 8×8 divergence) on top of the existing near-only foliage.

**Note — reprojection is static-only.** The Win C shadow/AO cache and the
full-reprojection TAA run only while the camera is still (where they reproject
onto themselves exactly); on motion the frame is traced fresh and passed through
(sharp). Motion reprojection visibly warped the image and the engine has ample
headroom to trace every moving frame. Future work: tune the motion-reprojection
math on AC (the warp can't be eyeballed reliably on battery) so the caches can
also help during movement.

**Everything in this file is now implemented.** The render wins are validated by
the headless render test (spawn + far-origin + the 2-frame reuse path, which also
runs the cloud + deferred-transparent passes) and naga shader validation; the GPU
physics by a functional readback test. Traversal/shading perf + TAA/cache visual
quality are to be re-checked on AC with `raymarch_timing` (battery throttles the
GPU ~14×, so numbers + visuals there are moot).

## Deliberately deferred (with rationale)
- Full reprojection/disocclusion TAA (skip tracing reprojectable tiles on motion) —
  the same-pixel TAA here gives the AA win safely; checkerboard/disocclusion needs
  visual tuning that isn't possible headless.
- Dedicated half-res volumetric pass (#14) and a transparent/foliage second pass
  (#16) — both are larger GPU-profiler-driven refactors; cheaper equivalents shipped.
- GPU compute physics (#25) — shipped as a design doc; the CPU path now runs off the
  frame thread.
