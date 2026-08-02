# 10 cm voxels: what it took, what it cost, and the verdict

`world_dims::VOXEL_METRES` is **0.10** in the tree. The window is 400 x 160 x 400
bricks = 1600 x 640 x 1600 voxels = **160 x 64 x 160 m**, against 128 x 64 x 128 m
at 25 cm. Everything below is measured on an RTX 5060 at 1920x1080; the captures
are in `docs/rt/`.

**The verdict is NO.** 10 cm renders correctly and looks right, and the traversal
regression everybody expected did not happen. What blocks it is CPU physics, the
amortised light/GI passes, and per-pixel shading - none of which shrink usefully
with the window. Numbers and the size ladder are at the bottom.

## What "scale" actually means here

The rule the whole change is built on: **a size that means something in the real
world is written in metres and converted once**, through `m_to_vox` on the Rust
side and `VOXELS_PER_METRE` (emitted by `build.rs`) in WGSL. A bare voxel count
is a bug whenever it stands for a length, and it is a SILENT bug: everything
still compiles, renders and passes, it just means something 2.5x smaller.

The failures found by finishing this change were, without exception, of that one
shape. They are worth listing because none of them announced itself:

| what was a bare voxel count | what it silently became at 10 cm |
|---|---|
| cloud noise frequency `p * 0.0055` | every cloud 18 m across instead of 45 m, and the ground cloud-shadow pattern 2.5x finer |
| `WATER_FOAM_SHORE_SUB = 2` quarter-texels | surf at the waterline 5 cm wide instead of 12.5 cm - a hairline |
| `pick_biome`'s `sea_level + 36` | anything over 3.6 m above the sea is a Mountain: bare rock, snow cap, tree density 0.14. The demo window rendered **0.000 green pixels** |
| `Biome::top_block`'s `sea_level + 55` | snow line 5.5 m above the sea |
| `World::LIGHT_NEAR_RADIUS = 64` | the sun-tracking radius fell to 6.4 m: **31** blocks of a 536,168-block shell classified near |
| `server::INTEREST_R = 600` | multiplayer relay range 60 m, less than half the loaded window |
| `find_species_anchor`'s 32-voxel cells, `y 60..140` | the benchmark's per-species cameras searched a 3.2 m cell in a 6..14 m band, below the forest |
| the lookdev cameras' `71.0` / `72.0` / `+5` / `-16` | the water shots ended up 10 m under the sea |

None of these is exotic. They are what is left after the obvious ones (player
size, view distance, LOD radii) have been done, and the only reliable way to find
them is to look at the pictures and at the harness output, not at the diff.

## Two REAL bugs, neither of them about size

**The light field lost a quarter of all surfaces.** A record covers a
`LIGHT_RECORD_STEP^3` = 2x2x2 voxel cell now, and a cell holding any opaque voxel
is deliberately dead (that rule is what keeps a one-voxel wall opaque -
`voxlight_does_not_leak_through_a_one_voxel_wall`). `voxlight_sample` stepped off
the surface by half a VOXEL, which lands on a cell boundary for one parity of the
surface coordinate; for a negative-facing face the whole trilinear weight then
fell on the cell CONTAINING the surface, which is vetoed, so `wsum` came out 0 and
the pixel shaded with no light at all. Three of six face directions, one of two
parities: a quarter of every surface in the world, silently. The other half of the
same defect: the AO kernel probed single VOXELS, so the nearest live record found
the ground for one parity and missed it for the other, and flat open ground
measured AO 255 - no contact shadow whatsoever - half the time.

Fixed by moving both onto the record lattice: `voxlight_sample` steps out half a
RECORD CELL, and `vl_cell_ao` asks "does the neighbouring CELL contain an
occluder" (`vl_cell_occluder`) instead of "is that one voxel solid". Both
degenerate to exactly the old code at `LIGHT_RECORD_STEP == 1`. Measured after:
AO on flat ground 201/255 = the 0.7875 the 6-face/12-edge formula predicts, in
both parities, and a wall shifted one voxel along its own normal changes by
0.0000 where it changed by 0.0453 before.

**GPU physics could not dispatch.** One workgroup per 64 bricks is 400,000
workgroups at 10 cm against a hard 65,535-per-dimension cap. That is a validation
abort, not a slow frame. World-sized dispatches now go through
`renderer::linear_dispatch`, which tiles over x and y;
`world_sized_dispatches_fit_the_workgroup_limit` checks the real brick total
against the real limit rather than a remembered number.

## Worldgen

The metre-space rewrite left the demo window on a dry shelf: **0.0% water by
area** and 12 m of relief across 160 m, so both water scenarios of
`rt_vs_software_timing` were pointed at an empty lake. Fixed by three things,
each of which now has a test:

- the continent field's wavelength came down from ~700 m to ~285 m, so which
  side of sea level a 160 m window sits on is no longer one noise sample's coin
  flip, and the land baseline sits ON sea level rather than 2 m above it;
- the sea floor is compressed toward `SEABED_MAX_DEPTH_M` instead of clamping on
  the window floor (which had flattened whole basins to one dead plane - at seed
  42 four neighbouring surface chunks came out bit-identical);
- `DEMO_SEED` is now CHOSEN by measurement, because the window is a fixed 160 m
  patch and every benchmark camera is framed inside it.

The window it frames: 23.5% under water, 36.6% of columns grassed, 23.7% under
canopy, ground from 11.3 m to 55.3 m. A coast, a forest and a 38 m peak over the
waterline, inside one window. Guarded by
`the_demo_window_holds_both_land_and_water` and `the_demo_window_grows_a_forest`.

## Memory

| buffer | 25 cm | 10 cm |
|---|---|---|
| `world.bricks` (CPU and GPU, `Brick` = 72 B) | 75 MB | **1.84 GB** |
| light pool | 64 MiB | 256 MiB |
| GI probes, G-buffers, cloud, output | ~35 MB | ~35 MB (per-pixel) |

VRAM lands around 2.2 GB, RAM around 2.1 GB per `World`. Both fit. Two
consequences that are not obvious:

- `Renderer::new` and the headless test device request limits derived from
  `world_dims` (`renderer::world_limits`), because a single 1.84 GB storage
  binding is over wgpu's 128 MB/256 MB defaults and the failure is "the game will
  not launch".
- `cargo test --lib` runs one test per core and several build a full world.
  `.cargo/config.toml` pins `RUST_TEST_THREADS`; raise it when brick storage goes
  sparse, not before.

`bricks_buf_b` (the GPU-physics ping-pong) is now allocated at one brick unless
`VOXELG_GPU_PHYSICS` is set - it was 1.84 GB of VRAM handed to a pass the default
build never dispatches.

## Measurements

Software path, 1920x1080, RTX 5060. BEFORE is `docs/rt/BEFORE_*` (25 cm, 128 m).
**The BEFORE and AFTER scenes are not the same world** - this change rewrote
worldgen and re-chose the demo seed - so read the cross-scale rows as the shape
of the change, not as a controlled delta. The 400/320/240 ladder IS controlled:
the rt harness picked identical anchors at every size.

### primary-trace-only: the regression that did not happen

This is the number the 2.5x-more-voxels worry was about - the DDA with shading
removed (`cs_main` with the shade term compiled out).

| scene | 25 cm / 128 m | 10 cm / 160 m | 10 cm / 128 m |
|---|---|---|---|
| terrain | 6.45 ms | **6.99 ms** | 8.14 ms |
| foliage | 7.95 ms | **6.12 ms** | 6.31 ms |
| terrain-covered | 4.47 ms | **2.94 ms** | 2.79 ms |
| water-close | 2.59 ms | **0.19 ms** | 0.21 ms |

Traversal is flat. 15.6x the voxels cost between -23% and +8% depending on the
shot, because the brick/tile/chunk/L4 pyramid makes a DDA step count scale with
the SURFACE the ray meets rather than with the grid it crosses, and the LOD
switches are written in metres so they land in the same places. This is the one
piece of the 10 cm story that is unambiguously good news.

### everything else

| per frame | 25 cm / 128 m | 10 cm / 160 m | 10 cm / 128 m | 10 cm / 96 m |
|---|---|---|---|---|
| shade (terrain, sw) | 4.71 ms | 6.52 ms | 6.73 ms | - |
| FIXED passes (vlight+clouds+beam+probe+transp+compose) | 1.25 ms | **5.06 ms** | 3.60 ms | 2.57 ms |
| - of which vlight | 0.60 ms | 2.24 ms | 1.78 ms | 1.34 ms |
| - of which GI probe | 0.66 ms | 2.86 ms | 1.84 ms | 1.24 ms |
| live frame, terrain | 7.48 ms | **39.52 ms** | 23.86 ms | 21.75 ms |
| live frame, foliage | 14.83 ms | 22.00 ms | 20.48 ms | 19.34 ms |
| live frame, water | 7.53 ms | 13.36 ms | 11.95 ms | 10.74 ms |
| `physics::tick` mean | 2.03 ms | **168.10 ms** | 95.83 ms | 36.99 ms |
| `physics::tick` max | 9.63 ms | 434.39 ms | 313.03 ms | 144.23 ms |
| streaming max (chunk cross) | 11.55 ms | **49.43 ms** | 26.74 ms | 7.23 ms |
| lit shell | 63,903 blocks | 1,123,509 | 749,084 | 432,518 |
| startup `sync_light_shell_all` | 23.4 ms | 579.3 ms | 338.3 ms | 198.7 ms |

Captures: `docs/rt/AFTER_400_rt_timing.txt`, `AFTER_400_live.txt`,
`AFTER_320_rt_timing.txt`, `AFTER_320_live.txt`, `AFTER_240_live.txt`.

## The verdict

**Not shippable at 160 m, and not at any size the ladder reaches.**

- The **CPU physics CA** is the wall. A tick costs 168 ms mean / 434 ms worst
  against a 33 ms budget at 30 Hz. It scales with window VOLUME, so the ladder
  buys it back slowly: 96 m still costs 37 ms mean. There is no window worth
  shipping at which the CPU CA fits. `VOXELG_GPU_PHYSICS` exists and its dispatch
  now works; finishing it (readback so CPU queries are not stale, then water and
  smoke) is a precondition for 10 cm, not an optimisation.
- The **frame** is 20-40 ms on terrain and foliage at every size tested: 25-50 fps
  against a 144 Hz cap. The ladder flattens below 128 m because the cost is
  per-pixel and ray-length bound (`MAX_RAY_DIST` is 175 m, longer than any of
  these windows), not window bound. Shrinking the world is not the lever.
- The **fixed passes** quadrupled, 1.25 -> 5.06 ms, and that is the part that was
  supposed to be density-independent. It is not: the light sweep works a shell
  that grew 17x, and each GI probe traces rays through a 2.5x finer world. Both
  are amortised and both have obvious knobs (`PROBE_SPACING`,
  `VOXLIGHT_UPDATE_DIV`, `LIGHT_RECORD_STEP` 2 -> 4); none has been tuned.
- **Streaming** is the one thing the ladder does fix. 49 ms worst-case hitch at
  160 m, 27 ms at 128 m, 7 ms at 96 m.

The largest window that keeps streaming smooth and brings physics within sight of
its budget is **96 m** - and that is smaller than the 128 m the 25 cm build
shipped, for a frame still three times slower. That is the honest shape of the
trade today.

Memory is NOT what is blocking this, and sparse brick storage would not fix it.
It would return 1.6 GB per `World` and let `cargo test` parallelise again, both
of which are worth having, but the frame time would not move.

## What is left, in the order it has to happen

1. **GPU physics, finished** - the CPU CA cannot be made to fit. Readback so
   raycast picking and collision are not stale, then water, then smoke.
2. **The amortised passes.** `PROBE_SPACING` is 2 m and the probe pass costs
   2.86 ms; the light sweep costs 2.24 ms over a 1.1 M-block shell.
   `LIGHT_RECORD_STEP` 4 (a 40 cm field) is one line and would quarter the sweep.
3. **The main pass**, which is where the rest of the frame is. Traversal is
   already cheap; shading is not.
4. **Sparse brick storage**, for the memory and for test parallelism.

Only after 1-3 is there a point in re-running this ladder.
