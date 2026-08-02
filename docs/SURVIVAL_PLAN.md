# Survival mode: a game about light and material

Goal (Marc, 2026-08-02): a survival mode that is NOT a Minecraft reskin. Every
mechanic below earns its place by exploiting something this engine has that
Minecraft does not.

The two structural advantages, and the test every mechanic must pass:

1. **10 cm voxels.** A thousand times the volume resolution. A "block" is not a
   unit of anything here, so material is a VOLUME with mass, not an item in a
   slot.
2. **A real per-voxel light field.** Sun visibility, AO and point lights are
   stored per voxel and queryable anywhere in the world as a physical quantity
   (docs/VOXEL_LIGHTING_PLAN.md). Darkness is measured, never scripted.

If a mechanic would work identically on a 1 m grid with no light field, it is
not a twist, it is filler.

This document is the full target. It is deliberately larger than the first
milestone; see "Staging" at the end.

## A. Movement and body

1. **Sub-voxel slopes, no block step-up.** Collision resolves against the real
   10 cm surface, so uneven terrain is scrambled over naturally rather than
   climbed one-metre-block at a time. Step and vault thresholds are measured in
   voxels, so terrain roughness is something the player negotiates.
2. **Momentum-based movement.** Acceleration, ground friction and air control
   rather than instant velocity changes.
3. **Stamina-driven traversal.** Climbing and vaulting cost stamina, so the
   route up a cliff is a decision rather than a held key.
4. **The player disturbs the world.** Footprints in snow and sand: the voxels
   are fine enough for a boot to leave a real depression, and the physics
   already moves loose material. Trails persist and can be tracked.
5. **Load slows you.** Carrying heavy material costs speed and stamina, which
   couples the inventory directly to the body model rather than leaving
   encumbrance as an abstract number.

## B. Material and inventory

6. **Volume and mass, not slots.** A pack holds litres. Stone is dense and
   heavy, wood is light and bulky. Hauling ore out of a mine becomes a real
   decision about what is worth the trip.
7. **Tools carve volume.** A swing removes a scoop of material, not a cube.
   Better tools take a bigger bite. A tunnel is excavated, and its shape is the
   shape you cut.
8. **Materials carry real mass and structural weight**, feeding both
   encumbrance (5) and structural integrity (23).

## C. Light as a system

9. **Body state, not three bars.** Core temperature, hydration and energy
   replace health/hunger/thirst. They interact: heat drives water loss, cold
   burns energy, exhaustion impairs temperature regulation.
10. **Sun and shade drive temperature and water loss**, read from the per-voxel
    light field at the player's position. Standing in the open at noon is
    materially different from standing under a canopy.
11. **Shelter works because it is physically shade.** Nothing is scripted: a
    roof shelters because the light field says those voxels are dark. A badly
    built shelter with a gap genuinely leaks light and heat.
12. **Night and depth drain warmth.** Caves are cold because they are dark, by
    the same measurement.
13. **Plants grow toward measured light.** Farm placement matters; a crop in a
    shaded courtyard grows differently from one in the open. Same field, same
    query.
14. **Torches and fires are real point lights** in the field, so light is a
    placeable resource rather than a decorative sprite.
15. **The light frontier.** Because darkness is measured, pushing lights into a
    cave genuinely reclaims territory from whatever lives in the dark. Progress
    underground is a lit perimeter you extend and must maintain.

## D. Enemies

16. **Light-sensitive ecology.** Creatures spawn, path and feed based on the
    measured light field rather than a hardcoded depth or time check.
17. **They avoid bright areas**, making a lit perimeter a defence and a torch a
    tactical instrument rather than a convenience.
18. **Some dig.** Enemies that erode voxels can breach a shelter, which makes
    wall thickness meaningful and connects directly to structural integrity.
19. **Damage is material loss.** Enemies are voxel bodies that visibly lose
    chunks where they are hit, instead of draining a hidden bar. At 10 cm this
    reads clearly, and it means a wounded creature is visibly wounded.

## E. Combat

20. **The world is the weapon.** Collapse a ceiling on a pursuer, cut the
    supports under a ledge, drop a structure. This is voxel destruction plus
    physics doing the work, not a scripted trap.
21. **Flood a tunnel.** The water simulation already exists; breaching into a
    water body is a usable tactic.
22. **Light as a weapon.** Against light-averse creatures, a thrown or placed
    light source is offensive, not just illumination.

## F. Building

23. **Structural integrity.** Unsupported spans collapse. Voxel physics already
    moves loose material; this extends it to a real support relationship.
24. **Real arches and beams.** At 10 cm a genuine arch, buttress or beam is
    expressible and mechanically meaningful, which no 1 m grid can offer.
25. **Mining under your own base is dangerous**, which makes excavation and
    construction the same skill rather than two unrelated ones.

## G. Crafting

26. **Material-driven, not recipe lists.** Combine by PROPERTY - hardness,
    density, mass - so a tool's behaviour follows from what it is made of.
    Discovery replaces memorisation.
27. **Shape recognition.** Build a form at 10 cm resolution and have the game
    recognise what you made. The most ambitious item here and the one most
    likely to need its own design pass; called out as a target, not a promise.

## Cross-cutting

- **Light is a resource** that affects temperature (10), enemy behaviour (16,
  17), plant growth (13) and territory (15).
- **Material is volumetric** with mass affecting carry (5, 6), structure (23)
  and tool behaviour (26).

## Risks worth naming now

- **Structural integrity is the hardest perf problem here.** A naive support
  check is a connectivity query over millions of voxels. It needs a design that
  is incremental and bounded, not a flood fill per edit.
- **Shape recognition (27)** is open-ended and could absorb unlimited time.
- **The exposure model needs a CPU-side light query.** The field currently
  lives in a GPU buffer that is never read back. Whether that is a readback, a
  small CPU mirror, or a parallel CPU query is an open design question pending
  the infrastructure survey.
- **Host authority.** There is a net/server split; gameplay state has to
  respect it or multiplayer diverges.

## Infrastructure reality (surveyed 2026-08-02)

What exists, and the design decisions it forces.

**There is no local player.** The camera IS the position (`camera.rs:5`), a
free noclip flycam at 80 voxels/sec with no gravity, grounding or collision.
Remote players are interpolated pose history only (`net.rs:128`), purely
visual, with no physics or solidity. So "player controller" means introducing a
real player BODY, which is also the thing that damage-as-material-loss (19),
encumbrance (5) and multiplayer collision will all need. Build it once, deliberately.

**Collision is greenfield, and the obvious basis is the wrong one.** The only
CPU solidity test today is `in_solid`, inline in `raycast` (`raycast.rs:72`),
and that raycast walks voxel by voxel up to 4096 steps WITHOUT touching the
L4/chunk/tile/brick pyramid the shaders use. A character AABB is roughly
6x18x6 voxels; testing it a voxel at a time, per substep, is the naive design.
The right one reuses the hierarchy: a brick's `occupancy` is a u64 covering
4x4x4 voxels, so an AABB overlap is a handful of masked u64 tests, and empty
tiles/chunks skip whole regions. That primitive does not exist on the CPU yet
and should be built once and shared by collision, digging and any future
entity (safe-primitives).

**The CPU light query: async readback of ONE record, not a full readback and
not a second implementation.** Three options were on the table:
- Full GPU readback of the light pool. Rejected: the existing readback pattern
  (`renderer.rs:3690`) uses `poll(wait_indefinitely())` and stalls the frame.
- A separate CPU sun-visibility raycast. Rejected despite being easy: it is a
  SECOND definition of "how lit is this point" that will silently drift from
  the one the renderer uses, which is exactly the two-drifting-paths failure
  the lighting rework just spent a week removing.
- CHOSEN: the CPU already owns the brick -> block mapping (`LightField::block_of`),
  so it can compute the exact word offset of the player's own light record and
  copy 8 bytes into a staging buffer, mapped asynchronously and read a frame or
  two later. One source of truth, no stall, and a 1-2 frame lag is irrelevant to
  a body temperature that moves over seconds. Needs a defined fallback for a
  player standing outside the lit shell.

**Structural integrity (23) is fully greenfield.** Physics simulates falling
sand, an 8-level water CA and smoke at 30 Hz on a worker thread
(`physics.rs:29`, `app.rs:314`), but material falls only when the cell directly
below is empty. There is no support relationship, no connectivity, nothing to
extend. This is the hardest item in the document and it is not in the slice.

**Gameplay must respect server authority.** `server.rs` owns an authoritative
edit log replayed to joiners over TCP; edits already broadcast
(`app.rs:464`). Any survival action that changes the world has to flow through
that or multiplayer diverges.

**Latent bug found while surveying:** the GPU physics path
(`VOXELG_GPU_PHYSICS`) writes only the GPU buffer and never updates the CPU
`World`, so raycast picking sees pre-physics state. Not caused by survival
work, but it will bite anything that queries the world from the CPU.

## Staging

Vertical slice first, chosen so the first playable loop already exercises BOTH
engine advantages: **dig a shelter, light a fire, survive the night.**

1. Collisions (1) and player controller (2, 3).
2. Volumetric digging (7).
3. Exposure and warmth (9, 10, 11, 12) with fire as a real point light (14).

Everything else follows once that loop is playable and its feel is proven.
