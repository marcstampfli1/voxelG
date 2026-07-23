# Flora overhaul: volumetric, beautiful, fluffy foliage everywhere

Goal (Marc, 2026-07-23): everything that grows and is NOT a tree - grass,
flowers, dry tufts, plus new classes - must look as good as or better than
the tree canopies: volumous, fluffy, alive. Fully optimized at every stage;
the deterministic bench and Marc's eye gate every step. Trees are the
quality reference and are out of scope.

## Why the trees look good (the bar, decomposed)

The canopy look comes from five things the flora must replicate:
1. VOLUME - many overlapping sprite-cutout cards at varied depths, not two
   crossed quads: silhouettes layer and occlude, edges stay organic.
2. TIERING - card cloud near, block faces mid, caps far: cost falls with
   distance while the silhouette stays continuous (no pop).
3. SHADE STRUCTURE - per-card shade ramps + species tint + hue wobble:
   depth reads through brightness, clumps do not look flat-colored.
4. MOTION - geometric wind (shear on cards) rather than luma waves: alive
   without traveling brightness patches (W1 lesson).
5. AUTHORED SPRITES - ASCII-authored atlas art (sprites.rs +
   examples/convert_tuft.rs), not procedural blobs.

## Current flora (what falls short)

Decoration voxels (MAT_TALL_GRASS 31, MAT_FLOWER 30, MAT_TALL_GRASS_DRY 34)
render as two crossed quads with one sprite: flat from above, paper-thin
edge-on, one silhouette per cell, no clumping, no volume. Also the thin
silhouettes are the largest remaining temporal-noise source (the rig's
meadow floor ~2.5k strong-flicker pixels = TAA edge ripple on 1-2 px
geometry, hunt item 8).

## Architecture: three tiers, mirroring the proven leaf system

All tiers resolve inside the existing sub-voxel resolver (the shared
resolve_solid_voxel path), so software DDA and the RT pipeline both get
them with no new passes, and cell-level sparsity (decoration cells are
2-6% of ground cells; brick/tile uniform masks already skip empties) keeps
the cost proportional to visible flora, not to the world.

### Near tier (t < ~28): volumetric card clusters
- GRASS: 8-12 tapered blade cards per decoration cell. Roots anchored on
  the existing 8-yaw table with per-cell hash jitter; outward lean 5-20
  degrees; heights 0.5-1.0 voxel with clump variance; 2-3 sub-clump
  composition per cell so cells read as tussocks, not pincushions. Blade
  sprite: tapered, slight S-curve, root-dark to tip-bright ramp (reuse the
  cross_sprite_tint ramp). Per-blade wind shear from wind_offset + gust
  (geometric motion only - never luma).
- FLOWERS: stem card + head cluster of 3-5 petal cards around the stem top
  (a miniature leaf cloud) + a center-detail sprite. The five existing
  species keep their palettes; heads nod on gusts with slight phase per
  flower; stems share the grass blade shader path.
- DRY GRASS (sand/tundra): same skeleton, straw palette, sparser blades,
  plus seed-head cards on 1-2 blades per clump.
- NEW - BUSHES: low 1.5-2.5 voxel card-cloud domes reusing the LEAF CARD
  renderer with bush sprites and ground anchoring - the single biggest
  "fluffy everywhere" win, and nearly free engineering since the leaf
  cloud machinery exists. Biome-placed (forest edges, meadow scatter).
- STRETCH - ground cover: clover/moss patch cards hugging the surface in
  high-moisture cells; only if budgets allow after bushes.

### Mid tier (28 to the existing decoration cutoff)
Current crossed quads, BUT with upgraded sprites whose silhouette matches
the near-tier clump outline (same apparent height/width/raggedness), plus
a 4-voxel cross-fade band. The tier transition must survive the bench-
style A/B: same cell rendered at t=26 vs t=30 may not pop.

### Far tier: fade to nothing (existing behavior, unchanged).

## Performance architecture (optimized FULLY, in this engine's idioms)

- Same-resolver integration: cards are plane intersections in the existing
  per-voxel resolve, warp-coherent, RT + software identical (the leaf
  cards prove the cost model).
- Sparsity first: cell masks skip all empty cells; per-brick "has flora"
  bits extend the existing uniform-brick machinery so crowded meadows pay
  and empty rock pays zero.
- Card budgets are DISTANCE-TIERED GEOMETRY, not resolution/quality cuts:
  every tier keeps a full-quality silhouette; the ONLY thing that changes
  with distance is card count, with sprite silhouettes designed to match
  across tiers (the no-res-cuts rule as applied here).
- Shadow policy: grass and flowers do NOT cast shadow rays (today's
  policy, kept - cost and the dapple-flicker lessons); any flora shadow
  sampling uses the frozen-phase discipline (shadow_wind_freeze
  precedent). BUSHES cast like leaf blocks (they are leaf-cloud
  instances) - parity with trees demands it.
- PREREQUISITE - hunt item 8 (TAA edge ripple): fluffier flora means MORE
  thin silhouettes; without the jitter-compensated history sample in
  cs_taa, every new blade adds shimmer. Stage 0 fixes it first so quality
  work lands on a stable image.
- Budgets (deterministic bench, interleaved pairs, controls, AC):
  - foliage segment: >= 195 fps after all stages (today 204) - i.e. the
    whole overhaul costs at most ~0.2 ms there.
  - meadow-class view (add a "meadow" bench segment in Stage 0): near-tier
    grass budget +0.9 ms vs today's crossed quads at the same view.
  - water/terrain segments: unchanged (controls).
  - Any stage that blows its budget iterates or dies in the ledger like
    any perf candidate.

## Art pipeline

ASCII-authored sprites in src/sprites.rs, converted via
examples/convert_tuft.rs into the existing atlas: 3 green blade variants +
1 dry, seed-head, 5 flower petal sets + centers, stem, 2 bush leaf
sprites. Palette via the existing tint tables; species/hue variance from
the established per-cell hash wobble (tight ranges - the tree color-
harmony lesson: cards and blocks must share tint sources).

## Placement: "everywhere", by biome, clumped

Replace uniform sparse scatter with clump-noise placement (hash-Poisson):
meadows = dense tussock clusters with clearings; flower FIELDS as patches
(species-coherent) rather than confetti; reeds/tall clumps ringing water
cells; dry tufts on sand/tundra; bushes at forest edges + lone-tree bases.
Density per biome is a knob with a bench gate; world regen stays
deterministic (lattice hashes only, no RNG state).

## Workflow (the rules, applied)

Lab first: a flora lab like the leaf lab - crafted rows of every variant,
lookdev stills side/top/grazing plus a near-mid transition strip. Iterate
in the lab; Marc's eye gates EVERY stage before world integration; bench
pairs before every commit; every stage its own commit; rejects reverted
and logged in PERF.md.

## Stages (each: implement -> lab stills -> Marc look gate -> bench pairs -> commit)

- Stage 0: TAA jitter-compensated history (hunt 8) + flora lab harness +
  "meadow" bench segment. Foundation; no look change to gate except LESS
  shimmer (rig meadow strong-flicker should DROP).
- Stage 1: sprite set + atlas plumbing (blades, heads, petals, bush).
- Stage 2: grass near tier (lab -> world) + mid-tier sprite parity +
  cross-fade. The core deliverable.
- Stage 3: flowers near tier (stems + heads).
- Stage 4: dry/biome variants + clumped placement/density rework.
- Stage 5: bushes (new class, leaf-cloud reuse, casts shadows).
- Stage 6: polish - wind coherence across classes, color harmony against
  tree canopies, transition audits, final bench table, PERF.md + docs.

## Risks and their gates

- Thin-geometry shimmer: Stage 0 prerequisite + rig meadow/foliage
  strong-flicker counts must not rise at any stage.
- Tier pop: transition strip stills + the t=26/t=30 A/B at every stage.
- Cost creep on non-flora views: terrain/water bench rows are controls;
  regression = the stage iterates.
- Look subjectivity: Marc's eye is the only definition of "fluffy enough";
  stills at every stage, builds on demand, reverts stay one commit.
