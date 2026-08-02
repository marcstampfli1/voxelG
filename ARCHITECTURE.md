# voxelG architecture

Theory of operation, the frame, the invariants, and the traps that have
actually cost time. Only verified claims belong here; each fact has ONE home,
and this file is updated in the same commit as the change it describes.

## What this is

A from-scratch GPU-raymarched voxel engine. Voxels are ~10 cm. There is no mesh
and no rasterised terrain: the world is a hierarchical occupancy pyramid in GPU
storage buffers, and a compute shader marches rays through it per pixel.
Optional hardware ray tracing traces the same world through an acceleration
structure of per-brick AABBs.

## Codemap

    src/world_dims.rs   SINGLE SOURCE OF TRUTH for world dimensions. Included by
                        voxel.rs AND by build.rs, which generates matching WGSL
                        constants. Change a dimension here and Rust, the CPU
                        raycaster and every shader move together.
    src/voxel.rs        World storage, the occupancy pyramid, streaming window,
                        persistent edit log, chunk generation dispatch.
    src/voxlight.rs     Sparse block allocator for the per-voxel light field.
                        Holds allocation only; the payload is GPU-side.
    src/physics.rs      Falling sand, an 8-level water CA, smoke. 30 Hz.
    src/raycast.rs      CPU DDA raycast for picking. Mirrors the shader's
                        toroidal mapping.
    src/accel.rs        Hardware-RT acceleration structure build/rebuild.
    src/renderer.rs     wgpu device, all pipelines, bind groups, the frame.
                        By far the largest module.
    src/shader_cache.rs Persistent pipeline cache (see "Cold start" below).
    src/camera.rs       Camera state and sun direction.
    src/app.rs          Window, input, frame loop, physics thread, streaming.
    src/net.rs          Wire protocol and client/server sockets.
    src/server.rs       Dedicated server: authoritative edit log, interest fan-out.
    src/leaffall.rs     Deterministic falling-leaf particle sim.
    src/grass.rs        GPU-instanced grass blade field.
    shaders/            WGSL. raymarch.wgsl is the main body; the rest are
                        preludes and passes concatenated around it.

Working in the repo: `CONTRIBUTING.md`. Reusable building blocks and the raw
patterns they replace: `docs/PRIMITIVES.md`.

## Entry points

`main.rs: main` parses the mode, then either `server::run_server(port)` for a
dedicated server or `app::run_client(..)` for the game. The client's per-frame
work is `app::render_frame`, which ends in `Renderer::render`. Physics runs on
its own thread spawned from `app`, not from the frame loop.

## Dependency direction and boundaries

    world_dims  <-  voxel  <-  physics / raycast / accel / renderer
                               renderer  <-  app  ->  net  <-  server

`world_dims` depends on nothing and everything geometric depends on it.
`voxel` owns world state and never touches wgpu. `renderer` owns all GPU
resources and never mutates world state. `app` is the only place that wires
them together.

Three boundaries matter:

- **CPU world vs GPU buffers.** The GPU copy is downstream; it is updated only
  through the dirty-brick upload path. Anything that writes GPU state without
  updating `World` creates a split brain (see the GPU physics trap below).
- **Render thread vs physics worker.** Physics runs at 30 Hz on its own thread
  and takes the `World` mutex. Hold it briefly.
- **Client vs server.** `server.rs` owns the authoritative edit log. Gameplay
  that changes the world flows through it or multiplayer diverges.

## Key flows

Waypoint chains, greppable in order.

**Frame:** `app::render_frame` -> `Renderer::render` -> clouds -> beam -> probe
-> vlight -> `cs_main` -> transparent -> compose -> grass -> post -> taa -> blit.

**Player edit:** click queued -> `raycast::raycast` -> `apply_sphere` or
`World::apply_edit` -> `World::set_voxel` -> `refresh_masks_for_brick` +
`mark_brick_dirty` -> dirty-brick upload -> `net::Message::VoxelEdit` broadcast.

**Streaming:** camera moves -> `World::shift_origin` -> `clear_slot_masks`
(which also frees that slot's light blocks) -> async chunk generation ->
`install_finished_chunks` -> `apply_slot_data` -> `replay_edits_for_chunk`.

**Light field:** `World::sync_light_shell_dirty` -> `upload_voxlight` ->
`cs_voxel_light_update` -> read by `voxlight_sample` inside `shade`.

## The world

A toroidal streaming window of 512 x 256 x 512 voxels. The window slides in x
and z as the camera moves; y is fixed. Storage wraps modulo the window, so a
world voxel maps to a storage cell by `pos_mod`, and a recycled slot is
regenerated in the background.

The occupancy pyramid is four levels: brick (4^3 voxels, one u64 of occupancy),
tile (4^3 bricks), chunk (4^3 tiles), and L4 (4^3 chunks, so one bit test skips
a 256^3 region). Every traversal, CPU or GPU, is expected to use it.

INVARIANT: all float DDA math is done RELATIVE to `camera.world_origin`, while
the integer voxel grid stays absolute. This is what keeps precision usable far
from the origin; violating it produces "sky through hills" at a few million
voxels out and is invisible near spawn. `renders_far_from_origin` guards it.

INVARIANT: the toroidal mapping in `raycast.rs` and in the shaders must agree
exactly. A mismatch is silent and position-dependent.

## The frame

`Renderer::render` dispatches, in order:

    clouds      half-res volumetric
    beam        1/8-res depth pre-pass for god rays
    probe       GI irradiance probe update (RT only, amortized)
    vlight      per-voxel light field update (amortized, sun-paced)
    main        cs_main: primary trace + shade
    transp      deferred water/glass
    compose     geometry + clouds + god rays
    grass       instanced blade render pass
    post        bright / blur H / blur V / final tonemap
    taa         temporal resolve
    blit        to swapchain

`GPU_PROFILE_LABELS` must match this list. A pass that shares another pass's
label is invisible in the only profiler that watches the real renderer, which
has happened and cost a day.

## Lighting

Shadows, AO and local light are per-VOXEL, not per-pixel: stored in air and
foliage voxels of the lit shell, sampled with a solidity-gated trilinear fetch,
and refreshed by an amortized compute pass paced by sun motion. There is no
per-pixel shadow ray any more. Full design and rationale:
`docs/VOXEL_LIGHTING_PLAN.md`. Measurements: `docs/rt/BASELINE-per-voxel-lighting.md`.

The field lives in a GPU buffer that is never read back on the render path. The
CPU owns only the brick -> block mapping.

## Traps that have actually cost time

**The game is frame-capped.** `FRAME_CAP_HZ = 144` in `app.rs` unless
`VOXELG_UNCAPPED=1` is set. Under the cap, GPU savings become idle time rather
than frames, so any performance work must be measured uncapped or it will look
like it did nothing.

**The timing harness is not the renderer.** `rt_vs_software_timing` uses static
crafted worlds: no physics tick, no streaming, and for a long time it did not
construct a real `Renderer` at all. Costs that scale with a live world are
invisible in it by construction. `live_session_profile` is the harness that
ticks physics and streams; use that for anything about real-world cost.

**Cold shader compile is minutes, not seconds.** `cs_transparent` alone took
98 s (software) and 162 s (RT) on a cold driver cache. `src/shader_cache.rs`
persists a pipeline cache to `%LOCALAPPDATA%\voxelG\shader-cache`; without it a
first launch looks like a hang. `VOXELG_NO_SHADER_CACHE=1` disables it.

**GPU physics does not write back.** With `VOXELG_GPU_PHYSICS` set, physics runs
on the GPU buffer only and the CPU `World` is never updated, so CPU-side queries
(picking, and anything survival adds) see pre-physics state. The default path is
CPU physics on a worker thread, which does update the world.

**This machine's thermal spread is wide.** The same build has measured 205 and
117 fps on different runs, and an untouched software column moved 21.25 to
24.39 ms. Quote within-run controls, never a cross-run FPS delta.

## Environment flags

    VOXELG_RT               hardware ray tracing (also needs adapter support)
    VOXELG_UNCAPPED         remove the 144 Hz frame cap
    VOXELG_GPU_PROFILE      per-pass GPU timings
    VOXELG_BENCH            deterministic in-game benchmark, self-exits
    VOXELG_GPU_PHYSICS      GPU physics path (see trap above)
    VOXELG_NO_SHADER_CACHE  disable the persistent pipeline cache
    VOXELG_SHADER_CACHE_DIR relocate the pipeline cache

## Invariants, and where each is enforced

- **Window-relative DDA math.** All float traversal is relative to
  `camera.world_origin`; the integer grid stays absolute. ENFORCED by the
  `renders_far_from_origin` test, which fails at a few million voxels out.
- **World dimensions have one home.** `world_dims.rs` feeds both Rust and, via
  `build.rs`, the generated WGSL constants. ENFORCED by construction: there is
  no second place to edit.
- **The foliage predicate is shared.** `is_foliage_mat` drives the shell rule,
  the update pass and the sampler. ENFORCED by a test that parses the WGSL and
  compares it against the Rust definition.
- **Group 0's declared storage buffers must fit the requested limit.** ENFORCED
  at bind-group-layout creation, which fails loudly but only inside GPU test
  output.
- **The toroidal mapping in `raycast.rs` must match the shaders.** NOT ENFORCED
  by any test today. A mismatch is silent and position-dependent. This is a
  known gap, not a claim of safety.
- **Every render pass needs its own `GPU_PROFILE_LABELS` entry.** NOT ENFORCED.
  A pass that shares another's label is invisible in the profiler, which has
  already hidden a pass eating 18-32 percent of frame time.

## Dev loop

See `CONTRIBUTING.md`. `docs/PERF.md` records proven AND rejected
optimisations, with numbers.
