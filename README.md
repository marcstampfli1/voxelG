# voxelG

A real-time voxel engine in Rust where the entire world is rendered by a WGSL compute shader. There is no
triangle geometry for terrain: each frame, rays are marched through a three-level occupancy-bit pyramid
(chunk → tile → brick) directly on the GPU, and a fullscreen-triangle blit presents the result. The same
binary runs solo, as a headless server, or as a multiplayer client. Built on wgpu (GitHub trunk, for
experimental hardware ray tracing) and winit 0.30.

## Rendering

World rendering lives in `shaders/raymarch.wgsl`, with a coarse pre-pass in `shaders/beam.wgsl`.

Traversal and performance:

- **Hierarchical DDA** through a 3-level bit pyramid: a bit per 16³ tile in each 64³ chunk, a bit per 4³
  brick in each tile, and a 64-bit occupancy mask per brick. Empty space is skipped at 16- or 4-voxel
  granularity with exact cell-boundary snapping (`skip_to_cell`).
- **Uniform-brick / uniform-tile compaction**: CPU passes detect 4³ bricks and 16³ tiles filled with a
  single opaque material and publish packed byte LUTs; the DDA then terminates at the cell's entry face
  without descending the hierarchy.
- **Beam pre-pass**: at 1/8 resolution, one ray per 8×8 pixel block walks the world at tile granularity
  and writes the first-hit distance to an `r32float` texture; the main pass fast-forwards each ray to
  that depth minus a 16-voxel safety margin.
- **Temporal differential rendering**: dirty bricks (physics, edits) are projected into 8×8 screen tiles
  on the CPU (`src/temporal.rs`); the compute shader early-outs on clean tiles, and both compute passes
  are skipped entirely when nothing changed.
- **Distance LOD**: beyond 400 voxels the DDA terminates at brick granularity, shading with the brick's
  topmost solid material so distant terrain keeps its surface colour.
- **Scalable internal resolution** (`RENDER_SCALE` in `src/renderer.rs`) with a bilinear upscale in the
  blit; ships at native 1:1.

Shading and effects:

- **Water**: deliberately STYLIZED and faceted rather than photoreal. Each surface water voxel renders
  one horizontal plate at one quantized height with one flat normal, both taken from a four-wave
  Gerstner spectrum sampled at the cell centre (swell λ26, sea λ13, chop λ7 and λ3.5 voxels) and
  quantized to seven height bands and five slope steps per axis. Every pixel of a cell therefore shades
  identically and the lake reads as a staircase of discrete plates that step up and down as the
  wavefront passes; where a neighbouring plate stands higher the ray enters below this cell's plate and
  the entry face is the hit, so a staircase of independent plates is watertight without any shared-corner
  machinery. The surface reads as LIT or SHADOWED off the per-voxel light field's sun visibility through
  a hard-ish step, with a flat tone ladder per wave band, Beer-Lambert per-channel absorption over a
  Snell-refracted trace beneath the surface (η = 1/1.33), a capped Fresnel blend toward the sky at
  grazing angles, hard-edged quantized foam at wave crests and shorelines drawn from authored
  quarter-voxel stamps (`src/sprites.rs`), and a separate absorption post-effect when the camera is
  submerged. Water does not reflect: neither the per-pixel mirror nor the per-voxel reflected-radiance
  cache that briefly replaced it survives (see `docs/VOXEL_LIGHTING_PLAN.md`).
- **Glass**: Fresnel reflection plus per-channel refraction for chromatic dispersion (n = 1.48/1.50/1.52),
  total-internal-reflection fallback to the reflected ray, distance-compounding tint; the 3-trace
  dispersion path is gated to grazing angles.
- **Foliage**, resolved sub-voxel inside the DDA. Leaves port the model and textures of Motschen's
  [Better Leaves](https://github.com/TeamMidnightDust/BetterLeavesLite) resource pack (MIT): each leaf
  block is a cutout cube (faces sample the centre of its species' pre-rounded 32×32 tuft — oak, birch
  and spruce ported via `examples/convert_tuft.rs`, carried as ASCII art in `src/sprites.rs`) plus two
  big double-sided diagonal tuft quads (species-scaled, 22.5°/−45°, four hash-picked rotations per
  block) that shear gently in the wind; autumn canopies add a per-voxel red-to-gold mottle. Up close,
  a volumetric leaf cloud scatters individual leaf-silhouette cards (lobed oak, serrated birch, pine
  needle whisks) through the whole canopy shell - varied depth, size, tone and orientation with sky
  gaps at the crown edge - so crowns read as actual leaves, not texture; beyond the cloud a horizontal
  cap tuft keeps distant tops fluffy. Leaves also FALL: a deterministic CPU simulation detaches up to
  256 leaves near the camera that pendulum down through their canopies, drift with the wind, rest
  where they land and dissolve on water, drawn as depth-tested tumbling cards composited after TAA. A
  sky-weighted occupancy AO darkens canopy interiors so crowns read volumetric. Worldgen paints an
  invisible one-voxel fringe shell around every canopy whose cells render the neighbouring blocks'
  protruding tuft parts — and lay a horizontal cap tuft over canopy tops — so the bushy overhang reads
  from every angle; fringe never draws as a cube, casts no shadows and is skipped by picking. Tall
  grass (three blade shapes with per-clump height, hue-coupled to the ground palette) and five flower
  species (poppy, daisy, tulip, cornflower, dandelion, clustered into wildflower meadows) are crossed
  quads carrying authored sprites, sheared by the wind in shared world space so the X always
  intersects; dry straw tufts dot desert sand and tundra snow. Wind strength rides a traveling gust
  field, so gusts visibly move across fields instead of the whole map swaying in lockstep; grass blocks
  render dirt sides with a ragged grass fringe.
- **Lighting**: day/night sun cycle with sunset scattering, sun disc, halo and stars; single-sample
  golden-angle PCF soft shadows jittered with interleaved gradient noise (TAA accumulates the penumbra);
  bit-test ambient occlusion bilinearly interpolated across the hit face.
- **Volumetrics**: slab-raymarched cumulus clouds (fbm body under a low-frequency coverage mask, 3 cone
  samples toward the sun for self-shadowing, Henyey, Greenstein forward scattering) and god rays
  accumulated as jittered sun-visibility samples along the primary ray.
- **Tri-planar procedural materials**: world-projected rgb-multiplier textures with a shared
  micro-grain layer, continuous across voxel boundaries. Natural strata-and-crack stone, clumpy
  dirt with pebbles, wind-rippled sand, drifted sparkling snow, per-species bark (coarse oak
  ridges, birch lenticels, pine plates) with ring end-grain, streaked ice, ore veins (dark coal
  seams, rusty iron, glinting gold, cyan diamond crystals), pulsing lava crust cracks and ribbed
  spiny cactus.

## World, simulation, multiplayer

- The world is a 512×256×512-voxel sliding window stored toroidally; crossing a chunk boundary
  regenerates only edge chunks (rayon-parallel) while the rest of GPU storage stays in place.
- Terrain from layered fbm value noise: temperature/humidity biomes (plains, forest, jungle, savanna,
  desert, tundra, beach, mountain), 3D-noise caves, ore veins, trees, rivers and sea.
- Cellular-automaton physics at 30 Hz: 8-level water (DwarfCorp-style level propagation), bitmask sand
  gravity, and rising smoke, iterating only "active" bricks that contain movable voxels.
- Multiplayer (`src/net.rs`): TCP with length-prefixed bincode messages. Each connection gets a reader
  and a writer thread bridged to the single-owner game thread through crossbeam's lock-free MPMC
  channels, no shared mutexes. World state syncs by shared seed plus a persistent edit log replayed to
  joiners; pose updates are capped at 20 Hz and fan out with distance-based interest management
  (600-voxel radius). Sphere destruction travels as a single `Explode` message expanded locally on each
  client. Remote players render as ray-traced colour-hashed boxes.

## Architecture

```
src/main.rs            thin entry point (dispatches to app / server)
src/lib.rs             crate root / module wiring
src/app.rs             event loop, input, click-to-raycast pipeline
src/server.rs          dedicated server loop
src/renderer.rs        wgpu setup; beam -> raymarch -> blit passes, palette, buffers
src/shader_cache.rs    persistent wgpu::PipelineCache so the driver's shader compile
                       survives restarts (see "First-launch shader compile" below)
src/voxel.rs           brick/tile/chunk storage, noise terrain, biomes, streaming,
                       uniform-brick/tile compaction, edit log
src/world_dims.rs      world dimension constants (also emits shaders/world_consts.wgsl)
src/physics.rs         sand / 8-level water / smoke cellular automata (CPU)
src/temporal.rs        dirty-brick -> screen-tile projection for partial re-render
src/raycast.rs         CPU DDA for block picking (destroy/place)
src/accel.rs           builds the hardware-RT acceleration structure from the world (opt-in)
src/net.rs             TCP client + server over lock-free channels
src/camera.rs          fly camera
shaders/beam.wgsl      1/8-resolution first-hit depth pre-pass
shaders/raymarch.wgsl  primary tracer and all shading (~2000 lines)
shaders/rt_voxel_query.wgsl  shared in-brick DDA (resolve_brick) for the RT path
shaders/rt_shadow.wgsl RT any-hit occlusion (shadows / AO / god rays); opt-in
shaders/taa.wgsl       temporal anti-aliasing resolve
shaders/physics.wgsl   GPU compute port of the cellular-automaton physics (in progress; see docs/gpu-physics-design.md)
shaders/blit.wgsl      fullscreen-triangle present + crosshair
shaders/world_consts.wgsl  generated dimension constants (from src/world_dims.rs)
```

### Hardware ray tracing (experimental, opt-in)

Occlusion rays (sun shadows, sky-access AO, god rays) can be traced by the GPU's RT cores instead of the
software hierarchical DDA. Set `VOXELG_RT=1` on a ray-query-capable adapter to enable it; unset, the
engine is byte-identical to the software renderer, which remains the default. One AABB per non-empty
brick goes into a BLAS (the RT core skips empty space in hardware); a small in-brick DDA
(`shaders/rt_voxel_query.wgsl`) resolves the exact voxel, sharing one occluder rule (`shadow_voxel_occludes`)
with the software path so the two render identically. Progress and design notes live in `docs/rt/WORKLOG.md`.

### First-launch shader compile

`shaders/raymarch.wgsl` assembles to a very large module and the renderer builds seven compute pipelines
out of it (fourteen with `VOXELG_RT=1`). Turning that into machine code is the graphics driver's job and
it can take minutes the first time, before any window appears. `src/shader_cache.rs` keeps a
`wgpu::PipelineCache` on disk so that cost is paid once per machine instead of once per driver-cache
eviction, and every pipeline logs its own compile time at `info` so a slow start reads as progress
instead of a hang.

The blob lives in `%LOCALAPPDATA%\voxelG\shader-cache\` on Windows and `$XDG_CACHE_HOME/voxelG/shader-cache/`
(or `~/.cache/voxelG/shader-cache/`) elsewhere, one file per adapter+driver. It is safe to delete at any
time; the next launch just recompiles. wgpu validates the blob against the current device and driver, so a
stale, corrupt or foreign one is rejected and logged rather than used. The cache needs the `PIPELINE_CACHE`
feature, which wgpu implements on Vulkan only; on DX12/Metal the engine logs that and carries on without it.

- `VOXELG_SHADER_CACHE_DIR=<path>` puts the blob somewhere else.
- `VOXELG_NO_SHADER_CACHE=1` turns it off entirely.

## Building and running

Requires a GPU and driver supported by wgpu (Vulkan, Metal or DX12).

```sh
cargo run --release                          # solo
cargo run --release -- --server 7878        # headless server
cargo run --release -- --connect host:7878  # join a server
cargo run --release -- --freeze-time 40     # pin sun/water/wind at t=40s (value optional)
cargo run --release -- --speed 4            # 4x fly speed
cargo run --release -- --flycam             # start in the noclip flycam (F toggles it)
VOXELG_RT=1 cargo run --release             # hardware-RT occlusion (ray-query GPU only; see below)
```

Controls: WASD to move, mouse to look, Shift to sprint, Ctrl to crouch. Space jumps; held while pressed
into a low ledge it vaults, and against a tall wall it climbs. F toggles the noclip flycam (WASD +
Space/Ctrl, Shift for 4x), which is what the look-dev harnesses use; dropping back out of it puts the
body where the camera was. Left click destroys a sphere, right click places the selected material; keys
1, 0 select stone, sand, water, wood, leaves, glass, lava, ice, snow or smoke. Esc releases the cursor.

## Status

Experimental graphics playground, not a game. Terrain is seed-deterministic; the server keeps its edit
log in memory only (no on-disk persistence), and remote players are placeholder markers. Constants are
tuned by eye on a single machine, expect to adjust them for yours.
