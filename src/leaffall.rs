//! Falling-leaf particle simulation (CPU).
//!
//! A few hundred leaves detach from canopy tops near the camera, pendulum
//! down through their own canopy, drift with the frame wind, rest briefly
//! where they land (shrinking away on water immediately), and despawn.
//! Deterministic under a fixed step: one RNG consumed in a fixed order on
//! one thread, pure f32 arithmetic - the property the sim tests pin.
//!
//! Rendering is a separate concern: `write_instances` emits flat GPU
//! instances the renderer draws in a depth-tested pass after TAA.

use crate::camera::wind_dir;
use crate::sprites::{SPR_LEAF_BIRCH, SPR_LEAF_NEEDLE, SPR_LEAF_OAK};
use crate::voxel::{
    is_leaf_mat, is_water_mat, World, MAT_AIR, MAT_FLOWER, MAT_LEAF_FRINGE, MAT_LEAVES_AUTUMN,
    MAT_LEAVES_BIRCH, MAT_LEAVES_PINE, MAT_TALL_GRASS, MAT_TALL_GRASS_DRY,
};
use crate::world_dims::{m_to_vox, WORLD_VOXELS_Y};
use glam::{Vec2, Vec3};

pub const MAX_LEAVES: usize = 256;

const SPAWN_RATE_PER_S: f32 = 10.0;
// Spawn annulus around the camera. How far away a falling leaf is still
// worth simulating is a property of the METRE - the leaf is the same leaf
// and the eye is the same eye - so these are the 15 m / 1 m they always
// were, not the 60 / 4 voxels they happened to be at 25 cm. Left bare, the
// whole effect would have collapsed into a 6 m bubble around the camera.
const SPAWN_RADIUS: f32 = m_to_vox(15.0);
const SPAWN_MIN_RADIUS: f32 = m_to_vox(1.0);
const COLUMN_PROBES: u32 = 4;
/// Sideways drift the frame wind adds while falling: 0.1125 m/s.
const WIND_DRIFT: f32 = m_to_vox(0.1125);
const LEAF_TTL: f32 = 20.0;
const REST_S: f32 = 2.0;
const SHRINK_S: f32 = 0.35;
/// Sun-visibility probe length, in 1-voxel steps. The STEP stays at one
/// voxel deliberately - it is a sampling rate against the grid, and a
/// coarser one walks straight through a one-voxel-thick canopy - so the
/// real-world 12 m reach has to be spent as a step COUNT that follows the
/// voxel. Written as 48 it would have probed 4.8 m and called every leaf
/// under a tall tree "lit".
const SHADOW_PROBE_STEPS: u32 = m_to_vox(12.0) as u32;

/// GPU instance layout; WGSL mirror in shaders/leaves.wgsl. 32 bytes: the
/// vec3 pos + size share a 16-byte row, so Rust and WGSL strides match
/// (guarded by the layout test).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LeafInstance {
    pub pos: [f32; 3],
    pub size: f32,
    pub rot: f32,
    pub tilt_phase: f32,
    pub sprite: u32,
    pub tint: u32,
}

enum LeafState {
    Falling,
    Resting(f32),
    Shrinking(f32),
}

struct Leaf {
    pos: Vec3,
    base_size: f32,
    sway_axis: Vec2,
    sway_amp: f32,
    sway_freq: f32,
    phase: f32,
    fall_speed: f32,
    spin: f32,
    spin_rate: f32,
    sprite: u32,
    tint: u32,
    age: f32,
    /// Smoothed sun-visibility factor (0.22 shadowed .. 1.0 lit), updated
    /// round-robin by cheap coarse probes toward the sun.
    shadow: f32,
    state: LeafState,
}

/// SplitMix64: tiny, deterministic, dependency-free.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

pub struct LeafSim {
    leaves: Vec<Leaf>,
    rng: Rng,
    spawn_accum: f32,
    time: f32,
    shadow_cursor: usize,
}

/// Materials a falling leaf passes through instead of landing on: air, the
/// canopy it detached from (leaves + fringe), and ground decoration.
fn passthrough(m: u8) -> bool {
    m == MAT_AIR
        || is_leaf_mat(m)
        || m == MAT_LEAF_FRINGE
        || m == MAT_FLOWER
        || m == MAT_TALL_GRASS
        || m == MAT_TALL_GRASS_DRY
}

impl LeafSim {
    pub fn new(seed: u64) -> Self {
        Self {
            leaves: Vec::with_capacity(MAX_LEAVES),
            rng: Rng(seed),
            spawn_accum: 0.0,
            time: 0.0,
            shadow_cursor: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.leaves.len()
    }

    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }

    /// Advance the simulation. `dt` should be the (clamped) frame dt; the
    /// determinism tests drive this with a fixed step. `sun` is the current
    /// sun direction (camera::sun_dir_at of the sun clock) for the shadow
    /// probes.
    pub fn step(&mut self, world: &World, cam_pos: Vec3, dt: f32, sun: Vec3) {
        self.time += dt;
        let wind = wind_dir(self.time);
        self.leaves.retain_mut(|leaf| {
            leaf.age += dt;
            if leaf.age > LEAF_TTL {
                return false;
            }
            match &mut leaf.state {
                LeafState::Falling => {
                    leaf.phase += leaf.sway_freq * dt;
                    leaf.spin += leaf.spin_rate * dt;
                    // Slower descent at the swing extremes: the classic
                    // falling-leaf rhythm.
                    let vy = leaf.fall_speed * (0.72 + 0.28 * 0.5 * (1.0 + (2.0 * leaf.phase).cos()));
                    let hvel = leaf.sway_axis * (leaf.sway_amp * leaf.sway_freq * leaf.phase.cos())
                        + wind * WIND_DRIFT;
                    leaf.pos += Vec3::new(hvel.x, -vy, hvel.y) * dt;
                    if leaf.pos.y < 1.0 {
                        return false;
                    }
                    let cell = leaf.pos.floor();
                    let m = world.material_at_world(cell.x as i32, cell.y as i32, cell.z as i32);
                    if !passthrough(m) {
                        if is_water_mat(m) {
                            leaf.pos.y = cell.y + 0.88;
                            leaf.state = LeafState::Shrinking(SHRINK_S);
                        } else {
                            leaf.pos.y = cell.y + 1.06;
                            leaf.state = LeafState::Resting(REST_S);
                        }
                    }
                    true
                }
                LeafState::Resting(t) => {
                    *t -= dt;
                    if *t <= 0.0 {
                        leaf.state = LeafState::Shrinking(SHRINK_S);
                    }
                    true
                }
                LeafState::Shrinking(t) => {
                    *t -= dt;
                    *t > 0.0
                }
            }
        });

        // Amortized lighting: coarse-march up to 16 leaves per frame toward
        // the sun (1-voxel steps out to SHADOW_PROBE_STEPS = 12 m) and smooth
        // the visibility factor.
        // Leaves/solids occlude, the invisible fringe and decoration do not.
        if !self.leaves.is_empty() {
            for _ in 0..16.min(self.leaves.len()) {
                self.shadow_cursor = (self.shadow_cursor + 1) % self.leaves.len();
                let leaf = &mut self.leaves[self.shadow_cursor];
                let mut lit = 1.0f32;
                if sun.y > 0.0 {
                    let mut p = leaf.pos + sun * 0.75;
                    for _ in 0..SHADOW_PROBE_STEPS {
                        let m = world.material_at_world(
                            p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32,
                        );
                        if m != MAT_AIR
                            && m != MAT_LEAF_FRINGE
                            && m != MAT_FLOWER
                            && m != MAT_TALL_GRASS
                            && m != MAT_TALL_GRASS_DRY
                        {
                            lit = 0.22;
                            break;
                        }
                        p += sun;
                    }
                } else {
                    lit = 0.4; // night: no direct sun to occlude
                }
                leaf.shadow += (lit - leaf.shadow) * 0.25;
            }
        }

        self.spawn_accum += SPAWN_RATE_PER_S * dt;
        while self.spawn_accum >= 1.0 {
            self.spawn_accum -= 1.0;
            if self.leaves.len() < MAX_LEAVES {
                self.try_spawn(world, cam_pos);
            }
        }
    }

    fn try_spawn(&mut self, world: &World, cam_pos: Vec3) {
        for _ in 0..COLUMN_PROBES {
            let u = self.rng.next_f32();
            let r = (SPAWN_MIN_RADIUS * SPAWN_MIN_RADIUS
                + u * (SPAWN_RADIUS * SPAWN_RADIUS - SPAWN_MIN_RADIUS * SPAWN_MIN_RADIUS))
                .sqrt();
            let a = self.rng.next_f32() * std::f32::consts::TAU;
            let x = (cam_pos.x + r * a.cos()).floor() as i32;
            let z = (cam_pos.z + r * a.sin()).floor() as i32;
            // First non-air from the top of the loaded window: canopy tops
            // are leaf cells with air above, by scan order.
            let mut hit = 0u8;
            let mut hit_y = 0i32;
            for y in (1..WORLD_VOXELS_Y as i32 - 1).rev() {
                let m = world.material_at_world(x, y, z);
                if m != MAT_AIR {
                    hit = m;
                    hit_y = y;
                    break;
                }
            }
            // Real canopies are capped by the invisible fringe shell: the
            // first non-air cell of a tree column is MAT_LEAF_FRINGE with
            // the leaf block right below it.
            if hit == MAT_LEAF_FRINGE {
                let below = world.material_at_world(x, hit_y - 1, z);
                if is_leaf_mat(below) {
                    hit = below;
                }
            }
            if !is_leaf_mat(hit) {
                continue;
            }
            let (sprite, base_mat) = match hit {
                MAT_LEAVES_BIRCH => (SPR_LEAF_BIRCH as u32, hit),
                MAT_LEAVES_PINE => (SPR_LEAF_NEEDLE as u32, hit),
                _ => (SPR_LEAF_OAK as u32, hit),
            };
            let c = crate::renderer::palette_color(base_mat);
            let shade = 0.85 + 0.30 * self.rng.next_f32();
            // Species tint sampled from the SAME mottle ramp the canopy
            // shader uses (leaf_species_tint: warm red-orange to gold-green)
            // so a fallen autumn leaf matches the tree it left; other
            // species are untinted there, and stay untinted here.
            let species = if base_mat == MAT_LEAVES_AUTUMN {
                let t = self.rng.next_f32();
                [
                    1.20 + (0.95 - 1.20) * t,
                    0.72 + (1.25 - 0.72) * t,
                    0.45 + (0.75 - 0.45) * t,
                ]
            } else {
                [1.0, 1.0, 1.0]
            };
            let tint = [
                ((c[0] * shade * species[0]).clamp(0.0, 1.0) * 255.0) as u32,
                ((c[1] * shade * species[1]).clamp(0.0, 1.0) * 255.0) as u32,
                ((c[2] * shade * species[2]).clamp(0.0, 1.0) * 255.0) as u32,
            ];
            let ang = self.rng.next_f32() * std::f32::consts::TAU;
            self.leaves.push(Leaf {
                // pos: cell centre, just clear of the canopy cell's top face -
                // a placement RELATIVE to the cell it left, so it stays in
                // cell units.
                pos: Vec3::new(x as f32 + 0.5, hit_y as f32 + 1.3, z as f32 + 0.5),
                // A leaf is 4..9 cm across, a leaf swings +-9..22 cm, and a
                // leaf falls at 0.35..0.60 m/s. All three are properties of
                // the LEAF, so they are written in SI: as bare voxel numbers
                // they would have shrunk to thumbnail sprites drifting down at
                // walking-pace-divided-by-seven the moment the voxel did.
                // (sway_freq/spin_rate are per-second, hence untouched.)
                base_size: m_to_vox(0.04) + m_to_vox(0.05) * self.rng.next_f32(),
                sway_axis: Vec2::new(ang.cos(), ang.sin()),
                sway_amp: m_to_vox(0.0875) + m_to_vox(0.1375) * self.rng.next_f32(),
                sway_freq: 1.2 + 1.0 * self.rng.next_f32(),
                phase: self.rng.next_f32() * std::f32::consts::TAU,
                fall_speed: m_to_vox(0.35) + m_to_vox(0.25) * self.rng.next_f32(),
                spin: self.rng.next_f32() * std::f32::consts::TAU,
                spin_rate: -2.5 + 5.0 * self.rng.next_f32(),
                sprite,
                tint: tint[0] | (tint[1] << 8) | (tint[2] << 16),
                age: 0.0,
                shadow: 0.65,
                state: LeafState::Falling,
            });
            return;
        }
    }

    /// Emit the GPU instances for the current state (clears `out` first).
    pub fn write_instances(&self, out: &mut Vec<LeafInstance>) {
        out.clear();
        for leaf in &self.leaves {
            let size = match leaf.state {
                LeafState::Shrinking(t) => leaf.base_size * (t / SHRINK_S).max(0.0),
                _ => leaf.base_size,
            };
            let tilt = match leaf.state {
                // Resting/landed leaves lie flat and stop tumbling.
                LeafState::Falling => leaf.phase,
                _ => 0.0,
            };
            // The alpha byte carries the smoothed sun-visibility factor.
            let shadow_bits = ((leaf.shadow.clamp(0.0, 1.0) * 255.0) as u32) << 24;
            out.push(LeafInstance {
                pos: leaf.pos.to_array(),
                size,
                rot: leaf.spin,
                tilt_phase: tilt,
                sprite: leaf.sprite,
                tint: (leaf.tint & 0x00ff_ffff) | shadow_bits,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::{MAT_LEAVES, MAT_STONE, MAT_WATER};

    const DT: f32 = 1.0 / 60.0;
    const SUN: Vec3 = Vec3::new(0.357, 0.874, 0.331);

    /// One oak canopy over stone ground; camera parked beside it.
    fn tree_world() -> (World, Vec3) {
        let mut w = World::new();
        for z in 200..280u32 {
            for x in 200..280u32 {
                w.set_voxel(x, 60, z, MAT_STONE);
            }
        }
        // A broad canopy: spawn probes are area-proportional, and the test
        // needs plentiful spawns, not marginal ones. Fringe-capped like the
        // real worldgen shells (the spawn probe must see through the cap).
        for z in 216..264u32 {
            for x in 216..264u32 {
                for y in 70..74u32 {
                    w.set_voxel(x, y, z, MAT_LEAVES);
                }
                w.set_voxel(x, 74, z, crate::voxel::MAT_LEAF_FRINGE);
            }
        }
        (w, Vec3::new(240.0, 70.0, 240.0))
    }

    /// The densest patch of canopy in the demo world, and the ground under it.
    ///
    /// The camera this replaces was a pair of voxel literals copied from the
    /// lookdev still - (48, 101, 254) - and voxel literals are precisely what a
    /// scale change invalidates. That point was 12 m x 63 m into the 128 m
    /// window a 25 cm voxel gave and is 4.8 m x 25 m into the 160 m window a
    /// 10 cm voxel gives: a corner of a different world, where this seed grows
    /// nothing. The test means "park the camera over the forest", so it now
    /// FINDS the forest instead of remembering where it used to be.
    ///
    /// Scanned on a 2 m lattice through the 8 m of air above the terrain-noise
    /// height, which is where a canopy is; both are real-world sizes, so the
    /// scan describes the same search at any voxel size.
    fn forest_anchor(w: &World) -> Vec3 {
        let step = m_to_vox(2.0) as i32;
        let reach = m_to_vox(8.0) as i32;
        let mut best = (0usize, glam::IVec2::ZERO, 0i32);
        let mut z = 0;
        while z < crate::voxel::WORLD_VOXELS_Z as i32 {
            let mut x = 0;
            while x < crate::voxel::WORLD_VOXELS_X as i32 {
                let h = crate::voxel::sample_terrain(x as f32, z as f32, w.seed).h;
                let mut n = 0usize;
                for y in h..(h + reach).min(WORLD_VOXELS_Y as i32 - 1) {
                    if is_leaf_mat(w.material_at_world(x, y, z)) {
                        n += 1;
                    }
                }
                if n > best.0 {
                    best = (n, glam::IVec2::new(x, z), h);
                }
                x += step;
            }
            z += step;
        }
        assert!(best.0 > 0, "the demo world grew no canopy at all to spawn leaves from");
        eprintln!(
            "forest anchor: {:?}, {} leaf voxels in the 8 m above ground y={}",
            best.1, best.0, best.2
        );
        // Eye height above the ground under the canopy: a place, not a voxel count.
        Vec3::new(best.1.x as f32, best.2 as f32 + m_to_vox(1.7), best.1.y as f32)
    }

    #[test]
    fn spawns_in_demo_world() {
        let mut w = World::new();
        w.fill_demo_terrain();
        let cam = forest_anchor(&w);
        let mut sim = LeafSim::new(11);
        for i in 0..240 {
            sim.step(&w, cam, 1.0 / 60.0, SUN);
            if i % 60 == 0 {
                eprintln!("step {i}: {} leaves", sim.len());
            }
        }
        eprintln!("final: {} leaves", sim.len());
        assert!(!sim.is_empty(), "no leaves spawned over the demo forest");
    }

    #[test]
    fn leaf_instance_layout() {
        assert_eq!(std::mem::size_of::<LeafInstance>(), 32);
    }

    #[test]
    fn spawn_is_deterministic() {
        let (w, cam) = tree_world();
        let mut a = LeafSim::new(7);
        let mut b = LeafSim::new(7);
        let mut c = LeafSim::new(8);
        for _ in 0..600 {
            a.step(&w, cam, DT, SUN);
            b.step(&w, cam, DT, SUN);
            c.step(&w, cam, DT, SUN);
        }
        let dump = |s: &LeafSim| {
            let mut v = Vec::new();
            s.write_instances(&mut v);
            v.iter().flat_map(|i| bytemuck::bytes_of(i).to_vec()).collect::<Vec<u8>>()
        };
        assert!(!a.is_empty(), "sim never spawned");
        assert_eq!(dump(&a), dump(&b), "same seed diverged");
        assert_ne!(dump(&a), dump(&c), "different seeds identical");
    }

    #[test]
    fn spawns_only_from_canopy() {
        let (w, cam) = tree_world();
        let mut sim = LeafSim::new(3);
        let mut seen = 0;
        for _ in 0..1200 {
            let before = sim.len();
            sim.step(&w, cam, DT, SUN);
            if sim.len() > before {
                // Newest leaf spawned this step: its column top must be leaf.
                let leaf = sim.leaves.last().unwrap();
                let (x, z) = (leaf.pos.x.floor() as i32, leaf.pos.z.floor() as i32);
                let top_y = (1..WORLD_VOXELS_Y as i32 - 1)
                    .rev()
                    .find(|&y| w.material_at_world(x, y, z) != crate::voxel::MAT_AIR)
                    .unwrap_or(0);
                let top = w.material_at_world(x, top_y, z);
                let ok = is_leaf_mat(top)
                    || (top == MAT_LEAF_FRINGE
                        && is_leaf_mat(w.material_at_world(x, top_y - 1, z)));
                assert!(ok, "spawned over material {top}");
                seen += 1;
            }
        }
        assert!(seen > 0, "no spawns observed");
    }

    #[test]
    fn ground_kill() {
        let (w, cam) = tree_world();
        let mut sim = LeafSim::new(5);
        // Run long enough for early leaves to complete fall + rest + shrink.
        let mut max_alive_age: f32 = 0.0;
        for _ in 0..3600 {
            sim.step(&w, cam, DT, SUN);
            for leaf in &sim.leaves {
                max_alive_age = max_alive_age.max(leaf.age);
                if let LeafState::Resting(_) = leaf.state {
                    // Resting leaves sit just above a solid cell.
                    let cell = leaf.pos.floor();
                    assert!(leaf.pos.y - cell.y <= 1.2);
                }
            }
        }
        // The lab world is built in VOXEL coordinates, so the drop is 13
        // voxels however big a voxel is: 13 / m_to_vox(0.35) s at the slowest
        // fall speed (9.3 s at 25 cm, 3.7 s at 10 cm), + rest + shrink, both
        // comfortably under TTL. Nothing may outlive TTL either way.
        assert!(max_alive_age <= LEAF_TTL + DT, "leaf outlived TTL: {max_alive_age}");
        assert!(!sim.is_empty());
    }

    #[test]
    fn water_kill_is_fast() {
        let (mut w, cam) = tree_world();
        // Flood the ground with water: leaves must shrink at the surface,
        // never resting the full REST_S.
        for z in 200..280u32 {
            for x in 200..280u32 {
                w.set_voxel(x, 61, z, MAT_WATER);
            }
        }
        let mut sim = LeafSim::new(5);
        for _ in 0..3600 {
            sim.step(&w, cam, DT, SUN);
            for leaf in &sim.leaves {
                if let LeafState::Resting(_) = leaf.state {
                    let cell = leaf.pos.floor();
                    let m = w.material_at_world(cell.x as i32, cell.y as i32 - 1, cell.z as i32);
                    assert!(!is_water_mat(m), "leaf resting on water");
                }
            }
        }
    }

    #[test]
    fn cap_respected() {
        let (w, cam) = tree_world();
        let mut sim = LeafSim::new(1);
        for _ in 0..10_000 {
            sim.step(&w, cam, DT, SUN);
            assert!(sim.len() <= MAX_LEAVES);
        }
    }

    #[test]
    fn ttl_kill_over_void() {
        // No ground at all: leaves fall forever and must die by TTL or the
        // world floor, never accumulating.
        let mut w = World::new();
        for z in 236..244u32 {
            for x in 236..244u32 {
                w.set_voxel(x, 120, z, MAT_LEAVES);
            }
        }
        let cam = Vec3::new(240.0, 120.0, 240.0);
        let mut sim = LeafSim::new(2);
        for _ in 0..3600 {
            sim.step(&w, cam, DT, SUN);
            for leaf in &sim.leaves {
                assert!(leaf.age <= LEAF_TTL + DT);
            }
        }
    }
}
