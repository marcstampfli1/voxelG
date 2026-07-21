# voxelG

A real-time voxel engine in Rust where the entire world is rendered by a WGSL compute shader. There is no
triangle geometry for terrain: each frame, rays are marched through a three-level occupancy-bit pyramid
(chunk → tile → brick) directly on the GPU, and a fullscreen-triangle blit presents the result. The same
binary runs solo, as a headless server, or as a multiplayer client. Built on wgpu 23 and winit 0.30.

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

- **Water**: real sub-voxel displaced geometry — each surface water voxel renders a bilinear patch over
  four per-corner heights. A corner takes the mean fill level of the up-to-4 water columns sharing it
  (so mixed physics levels L1-L8 ramp smoothly), displaced by a continuous four-wave Gerstner spectrum
  sampled at the corner's world position (swell λ26, sea λ13, chop λ7 and λ3.5 voxels); any
  corner-sharing column with water one cell up pins the corner to the cell top, so water bodies that
  touch diagonally or at different heights knit into one connected surface instead of isolated plates.
  Shared corners are computed identically from every sharing cell, so continuity is exact and vertical
  water walls appear only at shores, waterfalls and level steps. Shading uses the carried terrace
  gradient plus the exact per-pixel field normal; the far LOD tier falls back to a centre-sampled
  facet. Schlick Fresnel mixes a traced reflection with a Snell-refracted trace beneath the surface
  (η = 1/1.33); Beer-Lambert per-channel absorption, shoreline foam from underwater hit distance gated
  by wave crests, a caustic approximation, specular sun glints, and a separate absorption post-effect
  when the camera is submerged.
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
src/voxel.rs           brick/tile/chunk storage, noise terrain, biomes, streaming,
                       uniform-brick/tile compaction, edit log
src/world_dims.rs      world dimension constants (also emits shaders/world_consts.wgsl)
src/physics.rs         sand / 8-level water / smoke cellular automata (CPU)
src/temporal.rs        dirty-brick -> screen-tile projection for partial re-render
src/raycast.rs         CPU DDA for block picking (destroy/place)
src/net.rs             TCP client + server over lock-free channels
src/camera.rs          fly camera
shaders/beam.wgsl      1/8-resolution first-hit depth pre-pass
shaders/raymarch.wgsl  primary tracer and all shading (~2000 lines)
shaders/taa.wgsl       temporal anti-aliasing resolve
shaders/physics.wgsl   GPU compute port of the cellular-automaton physics (in progress; see docs/gpu-physics-design.md)
shaders/blit.wgsl      fullscreen-triangle present + crosshair
shaders/world_consts.wgsl  generated dimension constants (from src/world_dims.rs)
```

## Building and running

Requires a GPU and driver supported by wgpu (Vulkan, Metal or DX12).

```sh
cargo run --release                          # solo
cargo run --release -- --server 7878        # headless server
cargo run --release -- --connect host:7878  # join a server
cargo run --release -- --freeze-time 40     # pin sun/water/wind at t=40s (value optional)
cargo run --release -- --speed 4            # 4x fly speed
```

Controls: WASD + Space/Shift to fly, Alt to sprint, mouse to look. Left click destroys a sphere, right
click places the selected material; keys 1, 0 select stone, sand, water, wood, leaves, glass, lava, ice,
snow or smoke. Esc releases the cursor.

## Status

Experimental graphics playground, not a game. Terrain is seed-deterministic; the server keeps its edit
log in memory only (no on-disk persistence), and remote players are placeholder markers. Constants are
tuned by eye on a single machine, expect to adjust them for yours.
