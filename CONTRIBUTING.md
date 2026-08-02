# Contributing to voxelG

How to work in this repo. What the system IS and how it works lives in
`ARCHITECTURE.md`; the reusable building blocks live in `docs/PRIMITIVES.md`.
This file is only the loop and the conventions.

## Dev loop

Fast inner loop, seconds:

    cargo check --all-targets
    cargo test --lib wgsl_valid          # naga-validates every shader variant

The shader validation is worth running on every WGSL edit: it catches syntax and
type errors headlessly, without a GPU and without a driver compile.

Full gate, before a commit:

    cargo test --lib

That includes the GPU render-content gates and takes a few minutes. It must be
green. Never weaken or delete a test to make it pass: if a change genuinely
invalidates a test's premise, rewrite the test to pin the NEW intended
behaviour and say so in the commit message.

Measurement harnesses, all `#[ignore]`d so they stay out of the gate:

    cargo test --lib dump_lookdev_views     -- --ignored --nocapture   # stills -> target/lookdev
    cargo test --lib rt_vs_software_timing  -- --ignored --nocapture   # per-pass GPU timings
    cargo test --lib live_session_profile   -- --ignored --nocapture   # REAL renderer: physics + streaming
    cargo test --lib flicker_probe_rt_views -- --ignored --nocapture   # temporal stability

Running the game:

    cargo build --release
    VOXELG_RT=1 VOXELG_UNCAPPED=1 ./target/release/voxel

See `ARCHITECTURE.md` for the full flag list. Two that matter constantly:
`VOXELG_UNCAPPED`, without which a 144 Hz cap turns every GPU saving into idle
time, and `VOXELG_GPU_PROFILE` for the per-pass breakdown.

## The measurement rules

These are not style preferences; each one was learned by getting it wrong.

- **A/B on the harness or it does not ship.** Every candidate optimisation is
  measured before and after, and `docs/PERF.md` records REJECTED attempts with
  their numbers as well as accepted ones.
- **Use within-run controls.** This machine's thermal spread is wide enough that
  the same build has measured 205 and 117 fps. A cross-run FPS delta proves
  nothing; time the two variants back to back in one run.
- **`rt_vs_software_timing` is not the renderer.** It uses static crafted worlds
  with no physics tick and no streaming. Anything whose cost scales with a live
  world is invisible in it. Use `live_session_profile` for those.
- **Never iterate on visuals blind.** Capture stills with `dump_lookdev_views`
  and READ them. A number saying the mean luma moved is not a substitute for
  looking at the image.
- **Never ask a human whether it feels right.** Measure the felt quantity
  headless against a stated, sourced band, and write it as a test so a later
  change that breaks the feel fails loudly.

## Where new code goes

- A world dimension or a derived constant -> `world_dims.rs`, nowhere else.
  `build.rs` propagates it into WGSL automatically.
- A reusable, safe-by-construction helper -> implement it once, then add it to
  `docs/PRIMITIVES.md` with the raw pattern it replaces.
- A new shader entry point -> add it to the assembly in
  `raymarch_source_variant` for BOTH the software and RT variants, add its
  pipeline, and give it its OWN label in `GPU_PROFILE_LABELS`. A pass sharing
  another pass's label is invisible in the only profiler that watches the real
  renderer.
- A new GPU binding -> `create_compute_bgl` and `make_compute_bg` must agree,
  and `COMPUTE_STORAGE_BUFFERS` must cover what group 0 declares. A mismatch
  fails only at bind-group-layout creation, deep inside GPU test output.
- Gameplay that changes the world -> it must flow through the authoritative edit
  path (`server.rs` owns the log) or multiplayer diverges.

## Commit style

Conventional prefixes (`feat`, `fix`, `perf`, `docs`, `test`, `build`, `chore`)
with a scope, then a body that explains WHY and carries the numbers. Plain
ASCII, no em dashes. Small focused commits. Docs update in the SAME commit as
the change they describe - a stale map teaches the old architecture with
authority.

Never credit the AI in a commit: no `Co-Authored-By`, no "generated with".

## Before you push

    cargo test --lib          # green
    cargo build --release     # succeeds

and, if you touched anything performance-relevant, the before/after numbers in
`docs/PERF.md`.
