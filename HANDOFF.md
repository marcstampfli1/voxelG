# Handoff: voxelG, branch `feat/per-voxel-lighting`

Written 2026-08-02. Everything here was verified against the tree at commit
`00825ed`, not recalled. Read `ARCHITECTURE.md` and `CONTRIBUTING.md` first;
this file only covers what is IN FLIGHT and what is known to be wrong.

## The immediate task

**The owner's target: 10 cm voxels at 100 fps minimum (a 10 ms frame) in a live
session with physics ticking and streaming.** Today the live frame is 39.5 ms
and the physics tick is 168 ms mean / 434 ms worst against a 33 ms budget.

Three deliverables, all mandatory, none conditional on another:

    A. Fix still-water simulation at the root (sleeping / settling)  - PARTIAL
    B. Move the physics CA onto the GPU and make it the real path    - NOT STARTED
    C. Reach a 10 ms live frame                                      - NOT STARTED

Do NOT propose shrinking the window or reverting to 25 cm. Both were considered
and rejected by the owner.

## A. Water sleeping - PARTIAL, 3 of 6 wake paths fail

### The bug it fixes

In `step_brick_water`, a water cell cleared its movable bit ONLY in the
`new_level == 0` branch (when it drained to nothing), and the retire path
(`world.movable_mask[bi] = new_movable` plus the `active_bricks` removal) sat
INSIDE `if any_change`. So a brick where nothing moved never reached the code
that could retire it, and a FULL settled cell never empties, so its bit never
cleared. Settled water was structurally incapable of sleeping.

Sand only appeared correct because for sand "settled" and "empty" are the same
state. For water they are opposites: a settled lake is full.

Consequence: 30 times a second, forever, the CA re-evaluated every water brick
to learn the lake was still a lake. The demo window is 23.5% water - it was
0.0% until a recent worldgen fix, which is exactly why this never surfaced at
25 cm.

### What landed and passes

- `physics::retire_and_wake(world, touched)` - retires a brick when
  `any_change == false`.
- `voxel::face_neighbours(bi) -> [Option<u32>; 6]` and `World::wake_region(bi)`
  as the wake primitives.
- `step_brick_sand_fall` now threads a `touched` list.
- Passing: `a_settled_lake_sleeps_and_stays_a_lake`,
  `flowing_water_is_unchanged_by_sleeping` (behaviour while flowing is
  unchanged), `sand_landing_in_a_settled_lake_wakes_it`,
  `sealed_smoke_still_dissipates`,
  `a_streamed_in_slot_wakes_the_water_it_borders`.

### What fails - START HERE

    a_player_edit_under_a_settled_lake_wakes_it   physics.rs:835
        a hole opened by an edit does not drain the lake above it
    an_explosion_sphere_wakes_settled_water       physics.rs:864
        a breached wall does not pour
    a_drain_at_one_end_propagates_across_a_settled_lake  physics.rs:919
        a drain does not reach ten bricks along a channel in 400 ticks

All three are WAKE failures, not sleep failures. The sleep half works.

UNVERIFIED HYPOTHESIS, stated as one: the two edit cases suggest `wake_region`
is never reached from the world-edit path (`World::apply_edit` / `set_voxel` /
`apply_sphere` in `app.rs`), and the propagation case suggests a woken brick
does not transitively wake an already-sleeping neighbour. CHECK BOTH before
fixing; do not assume this is right.

A missed wake path is a lake that stops responding, which is far worse than a
slow one. Enumerate every way a settled region can be disturbed - player edit,
explosion sphere, brick streaming in, sand landing in water, a neighbour
draining, smoke - and test each.

### The measurement that does not exist yet

`settled_world_tick_cost` is `#[ignore]`d and HAS NOT BEEN RUN. There is no
number for what sleeping actually saves. That number is the entire point of the
change. Get it before moving on: show the active-brick count and tick time
falling to near zero once a lake settles.

## B. GPU physics - not started

The path half-exists behind `VOXELG_GPU_PHYSICS` (`shaders/physics.wgsl`,
`Renderer::run_gpu_physics`, `bricks_buf_b`). It has never run at scale: it
could not even dispatch until `renderer::linear_dispatch` landed in `d9b1dad`
(400,000 workgroups against a 65,535 per-dimension cap).

Two hard requirements:
1. It must reproduce the CPU CA (sand, 8-level water, smoke). Build an
   equivalence test from an identical seeded world over N ticks and SHOW IT
   FAILING against a deliberate break before trusting it green. A parallel CA
   has ordering hazards a sequential one does not; resolving them differently is
   a design decision to document, not a discrepancy to bury.
2. FIX THE WRITEBACK. Recorded in `ARCHITECTURE.md` as a trap: the GPU path
   updates only the GPU buffer and never the CPU `World`, so `raycast` picking
   and `voxquery` collision see pre-physics state. With a player body landed
   that means falling through sand that moved. It must not stall the frame; an
   async readback with a frame or two of lag may be acceptable for collision.

The sleeping from A must carry over: cost has to scale with MOVING material,
not window volume, on the GPU too. Parallelising a wasteful simulation just
makes it a fast wasteful simulation.

## C. The frame - 39.5 ms now, 10 ms target

Measured at 10 cm / 160 m (full table in `docs/SCALE_TO_10CM.md`):

    FIXED passes    5.06 ms   (was 1.25 at 25 cm; vlight 0.60 -> 2.24,
                               GI probe 0.66 -> 2.86)
    terrain shade   6.52 ms   (was 4.71)
    primary trace   6.99 ms   (did NOT regress: -23% to +8% across scenes)
    streaming hitch 49 ms worst
    live frame     39.52 ms

Leads, in the order the cost table suggests:
- `docs/SCALE_TO_10CM.md` notes `LIGHT_RECORD_STEP` 2 -> 4 is ONE LINE that
  quarters the light sweep. Check what 40 cm light records cost VISUALLY -
  capture stills and read them; if contact shadows suffer, find the resolution
  that does not.
- GI probe 2.86 ms: check its cadence and `PROBE_SPACING` are still right at
  this scale.
- Profile the main pass before touching it. `docs/PERF.md` records what has
  already been tried AND rejected; do not re-try a rejected item without new
  evidence.
- A 49 ms streaming hitch is a stutter even at 100 fps.

IMPORTANT: an earlier ladder in `docs/SCALE_TO_10CM.md` concludes "not
shippable". That measured the CURRENT renderer at different WINDOW SIZES. It is
a cost breakdown, NOT evidence that the frame cannot be optimised. The owner
rejected that conclusion and he was right to: the frame not shrinking with
volume says nothing about whether it shrinks with optimisation.

## Hard constraints

**RESOURCES - an agent crashed the owner's machine here.** A `World` is a dense
brick array of 1.84 GB at 10 cm, and `cargo test` runs one test per core with
several tests building a full world.

    ALWAYS  $env:RUST_TEST_THREADS="1"   before any cargo test
    ALWAYS  -j 4                          on every cargo build/check/test
    NEVER   two cargo commands at once, or a build while a harness runs
    CHECK   free RAM before anything that builds worlds; not under ~8 GB

**Do not open a window or run the game binary.** The owner runs it. Everything
is verified headless.

**Measurement discipline** (each learned by getting it wrong here):
- The game is frame-capped at 144 Hz unless `VOXELG_UNCAPPED=1`. GPU savings
  become idle time under the cap - this hid several real wins.
- `rt_vs_software_timing` uses static crafted worlds with no physics tick and
  no streaming. Live-world costs are invisible in it. Use
  `live_session_profile`.
- Within-run controls only. The same build has measured 205 and 117 fps on this
  machine.
- Never iterate on visuals blind: capture stills and READ them.
- Never ask the owner whether something feels right: measure the felt quantity
  headless against a stated band.

## Where things stand

    00825ed  wip(physics): water sleeping, 3 of 6 wake paths fail   <- HEAD
    d9b1dad  10 cm voxels over a 160 m window, verdict was "no"
    6f987f4  player body, swept voxel collision, CPU world query
    609e4bd  foliage carries light, per-pixel path deleted
    67b6e15  camera-aware light sweep, screen-space cache dropped

Test suite: 130 passing at `d9b1dad`. At `00825ed` three physics tests fail as
described above; nothing else regressed.

The game is playable at `6f987f4` (25 cm, ~300 fps uncapped). At `d9b1dad` and
later it is 10 cm and roughly 25 fps, which the owner has said is unplayable -
that is what this work is fixing.

## Design context

`docs/SURVIVAL_PLAN.md` is the game design target: a survival mode built on the
two things this engine has that Minecraft does not - fine voxels and a real
per-voxel light field. The player body and collision (slice items 1 and 2) have
landed. Volumetric digging and the light-driven exposure model have not.
