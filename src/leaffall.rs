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
use crate::world_dims::WORLD_VOXELS_Y;
use glam::{Vec2, Vec3};

pub const MAX_LEAVES: usize = 256;

const SPAWN_RATE_PER_S: f32 = 10.0;
const SPAWN_RADIUS: f32 = 60.0;
const SPAWN_MIN_RADIUS: f32 = 4.0;
const COLUMN_PROBES: u32 = 4;
const WIND_DRIFT: f32 = 0.45;
const LEAF_TTL: f32 = 20.0;
const REST_S: f32 = 2.0;
const SHRINK_S: f32 = 0.35;

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
        Self { leaves: Vec::with_capacity(MAX_LEAVES), rng: Rng(seed), spawn_accum: 0.0, time: 0.0 }
    }

    pub fn len(&self) -> usize {
        self.leaves.len()
    }

    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }

    /// Advance the simulation. `dt` should be the (clamped) frame dt; the
    /// determinism tests drive this with a fixed step.
    pub fn step(&mut self, world: &World, cam_pos: Vec3, dt: f32) {
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
            // Autumn leaves mottle toward warm orange like the canopy tint.
            let warm = if base_mat == MAT_LEAVES_AUTUMN { self.rng.next_f32() * 0.35 } else { 0.0 };
            let tint = [
                ((c[0] * shade * (1.0 + warm)).clamp(0.0, 1.0) * 255.0) as u32,
                ((c[1] * shade * (1.0 - 0.3 * warm)).clamp(0.0, 1.0) * 255.0) as u32,
                ((c[2] * shade * (1.0 - warm)).clamp(0.0, 1.0) * 255.0) as u32,
            ];
            let ang = self.rng.next_f32() * std::f32::consts::TAU;
            self.leaves.push(Leaf {
                pos: Vec3::new(x as f32 + 0.5, hit_y as f32 + 1.3, z as f32 + 0.5),
                base_size: 0.20 + 0.12 * self.rng.next_f32(),
                sway_axis: Vec2::new(ang.cos(), ang.sin()),
                sway_amp: 0.35 + 0.55 * self.rng.next_f32(),
                sway_freq: 1.2 + 1.0 * self.rng.next_f32(),
                phase: self.rng.next_f32() * std::f32::consts::TAU,
                fall_speed: 1.4 + 1.0 * self.rng.next_f32(),
                spin: self.rng.next_f32() * std::f32::consts::TAU,
                spin_rate: -2.5 + 5.0 * self.rng.next_f32(),
                sprite,
                tint: tint[0] | (tint[1] << 8) | (tint[2] << 16) | 0xff00_0000,
                age: 0.0,
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
            out.push(LeafInstance {
                pos: leaf.pos.to_array(),
                size,
                rot: leaf.spin,
                tilt_phase: tilt,
                sprite: leaf.sprite,
                tint: leaf.tint,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::{MAT_LEAVES, MAT_STONE, MAT_WATER};

    const DT: f32 = 1.0 / 60.0;

    /// One oak canopy over stone ground; camera parked beside it.
    fn tree_world() -> (World, Vec3) {
        let mut w = World::new();
        for z in 200..280u32 {
            for x in 200..280u32 {
                w.set_voxel(x, 60, z, MAT_STONE);
            }
        }
        // A broad canopy: spawn probes are area-proportional, and the test
        // needs plentiful spawns, not marginal ones.
        for z in 216..264u32 {
            for x in 216..264u32 {
                for y in 70..74u32 {
                    w.set_voxel(x, y, z, MAT_LEAVES);
                }
            }
        }
        (w, Vec3::new(240.0, 70.0, 240.0))
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
            a.step(&w, cam, DT);
            b.step(&w, cam, DT);
            c.step(&w, cam, DT);
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
            sim.step(&w, cam, DT);
            if sim.len() > before {
                // Newest leaf spawned this step: its column top must be leaf.
                let leaf = sim.leaves.last().unwrap();
                let (x, z) = (leaf.pos.x.floor() as i32, leaf.pos.z.floor() as i32);
                let top = (1..WORLD_VOXELS_Y as i32 - 1)
                    .rev()
                    .map(|y| w.material_at_world(x, y, z))
                    .find(|&m| m != crate::voxel::MAT_AIR)
                    .unwrap_or(0);
                assert!(is_leaf_mat(top), "spawned over material {top}");
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
            sim.step(&w, cam, DT);
            for leaf in &sim.leaves {
                max_alive_age = max_alive_age.max(leaf.age);
                if let LeafState::Resting(_) = leaf.state {
                    // Resting leaves sit just above a solid cell.
                    let cell = leaf.pos.floor();
                    assert!(leaf.pos.y - cell.y <= 1.2);
                }
            }
        }
        // Fall from y~74 to y61 takes ~13/1.4 = 9.3 s max, + rest + shrink
        // stays comfortably under TTL; nothing may outlive TTL.
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
            sim.step(&w, cam, DT);
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
            sim.step(&w, cam, DT);
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
            sim.step(&w, cam, DT);
            for leaf in &sim.leaves {
                assert!(leaf.age <= LEAF_TTL + DT);
            }
        }
    }
}
