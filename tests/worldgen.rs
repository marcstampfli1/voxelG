// Determinism + edit-replay tests (checklist: tests). These protect the two
// invariants multiplayer relies on:
//   1. Worldgen is a pure function of (chunk, seed) — every client regenerates
//      identical terrain from the shared seed.
//   2. Player edits survive a chunk streaming round-trip (unload + regenerate),
//      because the edit log is replayed on top of fresh noise. A regression
//      here is exactly the "late joiner / re-entered chunk desync" bug.

use glam::IVec3;
use voxelg::voxel::{self, World, MAT_GLASS};

fn bricks_eq(a: &[voxel::Brick], b: &[voxel::Brick]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b.iter())
            .all(|(x, y)| x.occupancy == y.occupancy && x.materials == y.materials)
}

#[test]
fn worldgen_is_deterministic() {
    let seed = 0xC0FFEE_F00D_BEEFu64;
    let chunk = IVec3::new(3, 0, 5);
    let a = voxel::gen_slot_bricks(chunk, seed);
    let b = voxel::gen_slot_bricks(chunk, seed);
    assert!(bricks_eq(&a, &b), "same (chunk, seed) must produce identical bricks");
}

/// The storage-chunk row that holds the ground at (cx, cz) for `seed`.
///
/// It USED to be the literal 2, with the note "cy=2 spans the surface (terrain
/// sits ~y72)". Both halves of that were voxel facts: at a 25 cm voxel sea level
/// was row 64 and a storage chunk is 32 voxels, so the surface really was two
/// chunks up. At 10 cm sea level is row 160 and the ground sits around row 250,
/// so chunk row 2 is 25 m of solid bedrock - where worldgen is seed-independent
/// BY DESIGN (`stone_or_ore` and the cave noise take no seed), which is exactly
/// why both of these tests started reporting that the seed does nothing.
fn surface_chunk_y(cx: i32, cz: i32, seed: u64) -> i32 {
    let cv = voxel::STORAGE_CHUNK_VOXELS as f32;
    let h = voxel::sample_terrain((cx as f32 + 0.5) * cv, (cz as f32 + 0.5) * cv, seed).h;
    h / voxel::STORAGE_CHUNK_VOXELS as i32
}

#[test]
fn worldgen_depends_on_seed() {
    // The surface layer + biome + tree placement are what the seed actually
    // shifts, so the chunk has to be the one holding the ground.
    let (cx, cz) = (3, 5);
    for seed in [1u64, 2] {
        let cy = surface_chunk_y(cx, cz, seed);
        let a = voxel::gen_slot_bricks(IVec3::new(cx, cy, cz), seed);
        let b = voxel::gen_slot_bricks(IVec3::new(cx, cy, cz), seed + 100);
        assert!(
            !bricks_eq(&a, &b),
            "different seeds must produce different terrain at the surface chunk (cy {cy})"
        );
    }
}

#[test]
fn worldgen_depends_on_chunk() {
    let seed = 42;
    let cy = surface_chunk_y(0, 0, seed);
    let a = voxel::gen_slot_bricks(IVec3::new(0, cy, 0), seed);
    let b = voxel::gen_slot_bricks(IVec3::new(1, cy, 0), seed);
    assert!(
        !bricks_eq(&a, &b),
        "neighbouring surface chunks must differ (cy {cy})"
    );
}

#[test]
fn edits_survive_chunk_streaming_roundtrip() {
    let mut world = World::with_seed(0xABCD_1234_5678_9999);
    world.fill_demo_terrain();

    // A handful of edits well inside the initial window.
    let edits = [
        (10, 100, 10),
        (40, 70, 80),
        (200, 64, 130),
    ];
    for &(x, y, z) in &edits {
        world.apply_edit(x, y, z, MAT_GLASS);
        assert_eq!(world.material_at_world(x, y, z), MAT_GLASS, "edit must apply locally");
    }

    // Stream the whole window away, then back — forces every slot (including
    // the edited columns) to unload and regenerate from noise + replayed edits.
    let far = glam::IVec2::new(
        voxel::WORLD_STORE_CX as i32,
        voxel::WORLD_STORE_CZ as i32,
    );
    world.shift_origin(far);
    world.process_pending_gen_blocking();
    world.shift_origin(glam::IVec2::ZERO);
    world.process_pending_gen_blocking();

    for &(x, y, z) in &edits {
        assert_eq!(
            world.material_at_world(x, y, z),
            MAT_GLASS,
            "edit at ({x},{y},{z}) must survive the streaming round-trip"
        );
    }
}
