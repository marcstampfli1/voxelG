// The local player: a real body with a size, a velocity and a footing, plus the
// controller that moves it through the voxel world.
//
// WHY A BODY AT ALL
// Until now the "player" was the camera: a noclip flycam whose position WAS the
// view (`camera.rs`). Every survival system in docs/SURVIVAL_PLAN.md needs
// something the camera cannot provide - damage as material loss (19) needs a
// volume, encumbrance (5) needs a speed the inventory can act on, multiplayer
// collision needs an extent. So the body is the authority for position and the
// camera becomes a view derived from it (`Player::eye`), with the flycam kept as
// a toggle because look-dev and several harnesses drive the camera directly.
//
// SHAPE OF THE SIMULATION
// Fixed 60 Hz steps with interpolated rendering (`PlayerSim`). This engine runs
// anywhere from 60 to 400+ fps, and a variable-dt controller silently changes
// character with frame rate: friction applied per frame, gravity integrated at
// different granularities, a jump that reaches a different height on a 240 Hz
// machine. Fixed steps make the feel numbers a property of the game rather than
// of the hardware, and the tests can then pin them.
//
// EVERY TUNING NUMBER IS SI. Voxels are a storage detail; a player is 1.8 m tall
// and gravity is m/s^2. `world_dims::VOXEL_METRES` is the single conversion, so
// when the world moves to 10 cm voxels (docs/SCALE_TO_10CM.md) the body stays
// the same size and the feel numbers do not move.

use glam::{Vec2, Vec3};

use crate::voxel::{World, MAT_AIR};
use crate::voxquery::{self, Aabb, Axis, MatSet};
use crate::world_dims::{m_to_vox, vox_to_m};

/// The body's physical size, in VOXELS, derived from SI at construction.
///
/// Data rather than constants because it has to be able to change: crouching
/// changes it today, and damage-as-material-loss (SURVIVAL_PLAN 19) and carried
/// bulk (5) will change it later. Everything downstream asks the body for its
/// box instead of assuming one.
#[derive(Clone, Copy, Debug)]
pub struct Dims {
    pub half_width: f32,
    pub stand_height: f32,
    pub crouch_height: f32,
    /// Eyes sit this far below the top of the head.
    pub eye_below_top: f32,
}

impl Default for Dims {
    fn default() -> Self {
        Self {
            // 0.6 m shoulder to shoulder, 1.8 m tall, eyes 0.15 m below the
            // crown: an ordinary adult. The AABB is square in plan because a
            // capsule would buy nothing against an axis-aligned voxel lattice.
            half_width: m_to_vox(0.30),
            stand_height: m_to_vox(1.80),
            // Two thirds of standing: low enough to pass a 1.25 m opening, high
            // enough that the eye drop reads as a crouch.
            crouch_height: m_to_vox(1.20),
            eye_below_top: m_to_vox(0.15),
        }
    }
}

impl Dims {
    #[inline]
    pub fn height(&self, crouching: bool) -> f32 {
        if crouching { self.crouch_height } else { self.stand_height }
    }

    #[inline]
    pub fn eye_height(&self, crouching: bool) -> f32 {
        self.height(crouching) - self.eye_below_top
    }
}

/// Movement feel. Fields are in VOXEL units (voxels, voxels/s, voxels/s^2) so
/// the simulation never converts; every value is written as an SI quantity in
/// `Default` with the reasoning attached, and the feel tests measure the
/// simulated result back in SI against a stated band.
#[derive(Clone, Copy, Debug)]
pub struct Tuning {
    pub walk_speed: f32,
    pub sprint_speed: f32,
    pub crouch_speed: f32,
    pub ground_accel: f32,
    pub ground_friction: f32,
    pub air_accel: f32,
    pub gravity: f32,
    pub jump_speed: f32,
    pub terminal_speed: f32,
    /// Height climbed automatically while walking. NOT a Minecraft block.
    pub step_max: f32,
    /// Tallest ledge a deliberate vault can mantle.
    pub vault_max: f32,
    /// How fast a vault gains height; sets its duration.
    pub vault_rise_rate: f32,
    pub climb_speed: f32,
    /// tan of the steepest ground that still gives footing.
    pub slope_limit_tan: f32,
    /// Velocity-proportional drag while sliding down ungrippable ground.
    pub slide_drag: f32,
    /// How far below the feet the ground probe looks.
    pub ground_probe_depth: f32,
}

impl Default for Tuning {
    fn default() -> Self {
        // ---- the reference frame these numbers were chosen in ----
        // Minecraft is the arcade ceiling (walk 4.317 m/s, jump apex 1.25 m,
        // effective gravity ~32 m/s^2, terminal 78.4 m/s, a free one-metre
        // step-up) and a real human body is the floor (walk 1.4 m/s, standing
        // vertical 0.4-0.6 m, g 9.81, terminal ~53 m/s). This game wants the
        // grounded end: SURVIVAL_PLAN asks for momentum, stamina-gated traversal
        // and load that slows you, none of which read if the body hops a metre
        // in the air. Each number therefore sits nearer the human end, and the
        // feel tests band it explicitly.
        let gravity_si = 14.0; // 1.43x real: falls resolve without the floaty
                               // hang time real g gives in first person, and
                               // well under Minecraft's ~32.
        let jump_apex_si = 0.65; // just over a real standing vertical, half of
                                 // Minecraft's. Sits between step_max (walked)
                                 // and vault_max (deliberate), so a jump buys
                                 // something a stride does not.
        Self {
            // Jog, not a stroll: this is the default pace over open terrain.
            walk_speed: m_to_vox(3.0),
            // Between Minecraft's 5.612 sprint and a real 6-8 m/s sprint.
            sprint_speed: m_to_vox(5.5),
            // Minecraft sneaks at 1.31; a real crouch-walk is about 1.0.
            crouch_speed: m_to_vox(1.3),
            // Reaches walk speed in 3.0/12 = 0.25 s: visible momentum, without
            // the controller feeling detached from the key.
            ground_accel: m_to_vox(12.0),
            // Stops from walk in 0.19 s / 0.28 m, from sprint in 0.94 m: boots
            // on dirt, and sprinting genuinely commits you.
            ground_friction: m_to_vox(16.0),
            // 0.15x ground. Over one jump's 0.6 s of airtime that buys about a
            // third of walking speed: enough to adjust where you land, nowhere
            // near enough to walk in mid-air. (0.3x ground was tried first and
            // measured 2.2 m/s of the 3.0 m/s walk - the body could effectively
            // stroll through the air, which is not a body with weight.)
            air_accel: m_to_vox(1.8),
            gravity: m_to_vox(gravity_si),
            jump_speed: m_to_vox((2.0 * gravity_si * jump_apex_si).sqrt()),
            // A touch above a real belly-down skydiver (53 m/s), far below
            // Minecraft's 78.4. Unreachable inside this world's 64 m of height,
            // but it is what a long fall converges to.
            terminal_speed: m_to_vox(55.0),
            // A tall stair riser. Building code caps a riser near 0.21 m and a
            // curb is 0.15 m, so 0.35 m is already generous for something you
            // walk up without thinking - and it is a THIRD of the one-metre
            // block step-up this design explicitly rejects.
            step_max: m_to_vox(0.35),
            // Chest height: a table, a low wall, a boulder. Above this you climb
            // instead of vaulting.
            vault_max: m_to_vox(1.30),
            vault_rise_rate: m_to_vox(2.5),
            // A scramble, not a ladder. Real climbing is slower still.
            climb_speed: m_to_vox(1.2),
            // 55 degrees. On a voxel lattice the ASCENT limit is set by step_max
            // (with a 0.35 m step you cannot climb steeper than the 1:1
            // staircase - 45 degrees, which is also where Unreal's 44.8 and
            // Source's 45.6 walkable-floor defaults sit). This threshold decides
            // something different: what you cannot keep your FOOTING on. 55
            // places it cleanly between the 45-degree 1:1 face you walk up and
            // the 63-degree 2:1 face you must slide off, with margin against the
            // quantisation of a height sample on a 10-25 cm grid.
            slope_limit_tan: 55.0_f32.to_radians().tan(),
            // Terminal slide on a 63-degree face is g*sin*cos/drag = 5.2 m/s:
            // a real loss of control, survivable at the bottom.
            slide_drag: 1.2,
            ground_probe_depth: m_to_vox(1.6),
        }
    }
}

/// What the controller is asked to do this tick. A snapshot of held state, not
/// events: the controller derives its own edges.
///
/// STAMINA SEAM (SURVIVAL_PLAN A3): stamina does not belong here as a stub. When
/// it lands it reads [`Events`] for what the body just spent and vetoes by
/// clearing `sprint` / `jump` before the input reaches this controller - the same
/// place an encumbrance limit or a paralysing effect would act. Nothing in the
/// controller has to change for it.
#[derive(Clone, Copy, Debug, Default)]
pub struct Input {
    /// x = strafe right, y = forward, each in [-1, 1].
    pub move_axis: Vec2,
    /// Facing, radians, in the same convention as `Camera::yaw`.
    pub yaw: f32,
    /// Jump, and - pressed while moving into an obstacle - vault or climb.
    pub jump: bool,
    pub sprint: bool,
    pub crouch: bool,
}

/// What the body did this tick. Consumed by presentation (footsteps, camera
/// shake) and, later, by stamina and fall damage.
#[derive(Clone, Copy, Debug, Default)]
pub struct Events {
    pub jumped: bool,
    pub vault_started: bool,
    pub climbing: bool,
    /// Downward speed at the moment of landing, voxels/s.
    pub landed: Option<f32>,
    /// Height gained this tick by step assist, vaulting or climbing (voxels).
    /// The quantity a traversal-stamina cost is proportional to.
    pub climbed: f32,
    /// Material under the feet, or `MAT_AIR` when airborne.
    pub ground_mat: u8,
}

impl Events {
    fn merge(self, other: Events) -> Events {
        Events {
            jumped: self.jumped | other.jumped,
            vault_started: self.vault_started | other.vault_started,
            climbing: self.climbing | other.climbing,
            landed: match (self.landed, other.landed) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (a, b) => a.or(b),
            },
            climbed: self.climbed + other.climbed,
            ground_mat: if other.ground_mat != MAT_AIR { other.ground_mat } else { self.ground_mat },
        }
    }
}

/// How the body is currently moving. Vault and climb are DELIBERATE traversal
/// states, entered only on an explicit press into an obstacle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Motion {
    /// Ordinary walking and ballistics.
    Free,
    /// Mantling a ledge along a path proven clear when it started, so it runs
    /// kinematically to completion.
    Vault { from: Vec3, to: Vec3, t: f32, dur: f32 },
    /// Hand over hand up a wall face.
    Climb,
}

/// The local player's body.
#[derive(Clone, Debug)]
pub struct Player {
    /// FEET centre, in world voxels. The body's authoritative position.
    pub pos: Vec3,
    /// Voxels per second.
    pub vel: Vec3,
    pub grounded: bool,
    /// Standing on ground too steep to grip (see `Tuning::slope_limit_tan`).
    pub steep: bool,
    pub crouching: bool,
    /// Smoothed eye height above the feet, so crouching does not snap the view.
    pub eye_offset: f32,
    /// Material under the feet, `MAT_AIR` when airborne.
    pub ground_mat: u8,
    pub dims: Dims,
    pub tuning: Tuning,
    pub motion: Motion,
    prev_jump: bool,
}

/// A body counts as standing on ground within this many voxels of its feet.
/// Small enough that it cannot bridge a real gap, large enough to survive the
/// contact skin a resting sweep leaves behind.
const GROUND_TOL: f32 = 0.05;

impl Player {
    pub fn new(feet: Vec3) -> Self {
        let dims = Dims::default();
        Self {
            pos: feet,
            vel: Vec3::ZERO,
            grounded: false,
            steep: false,
            crouching: false,
            eye_offset: dims.eye_height(false),
            ground_mat: MAT_AIR,
            dims,
            tuning: Tuning::default(),
            motion: Motion::Free,
            prev_jump: false,
        }
    }

    /// Drop the body at `feet`, clearing all motion state. Spawn, and leaving
    /// the flycam.
    pub fn teleport(&mut self, feet: Vec3) {
        self.pos = feet;
        self.vel = Vec3::ZERO;
        self.grounded = false;
        self.steep = false;
        self.motion = Motion::Free;
        self.eye_offset = self.dims.eye_height(self.crouching);
    }

    /// The body's box right now.
    #[inline]
    pub fn body(&self) -> Aabb {
        self.body_at(self.pos, self.crouching)
    }

    #[inline]
    pub fn body_at(&self, feet: Vec3, crouching: bool) -> Aabb {
        Aabb::from_feet(feet, self.dims.half_width, self.dims.height(crouching))
    }

    /// Eye position, i.e. where the camera goes.
    #[inline]
    pub fn eye(&self) -> Vec3 {
        self.pos + Vec3::Y * self.eye_offset
    }

    /// Ground speed the body is currently trying to reach.
    ///
    /// ENCUMBRANCE SEAM (SURVIVAL_PLAN A5): carried mass scales the result HERE,
    /// in one place, rather than being sprinkled through the accelerator.
    fn target_speed(&self, input: &Input) -> f32 {
        if self.crouching {
            self.tuning.crouch_speed
        } else if input.sprint {
            self.tuning.sprint_speed
        } else {
            self.tuning.walk_speed
        }
    }

    /// Advance the body by one FIXED timestep. Reads the world, never writes it.
    pub fn step(&mut self, world: &World, input: Input, dt: f32) -> Events {
        let mut ev = Events::default();
        let jump_edge = input.jump && !self.prev_jump;
        self.prev_jump = input.jump;

        // A vault owns the body until it finishes; its path was proven clear
        // when it started, so nothing else may perturb it.
        if matches!(self.motion, Motion::Vault { .. }) {
            self.advance_vault(dt, &mut ev);
            self.update_eye(dt);
            return ev;
        }

        self.update_crouch(world, input.crouch);

        // Footing: where the ground is, what it is, and whether it can be stood
        // on. Measured from the position the last tick left us in.
        let ground = self.probe_ground(world);
        if self.vel.y <= 0.0 {
            self.grounded = ground.contact;
        }
        self.steep = self.grounded && ground.slope_tan > self.tuning.slope_limit_tan;
        self.ground_mat = if self.grounded { ground.mat } else { MAT_AIR };
        ev.ground_mat = self.ground_mat;

        let wish = wish_dir(&input);
        let pressing = wish.length_squared() > 1.0e-4;

        // ---- deliberate traversal ----
        // Jump held while pressed into an obstacle: mantle what can be mantled,
        // cling to what cannot. Re-tested every tick while climbing, because the
        // same test is what tops a climb out onto the ledge.
        let was_climbing = self.motion == Motion::Climb;
        if input.jump && pressing {
            if (jump_edge || was_climbing) && self.try_vault(world, wish, &mut ev) {
                self.update_eye(dt);
                return ev;
            }
            if self.wall_ahead(world, wish) {
                self.motion = Motion::Climb;
            } else if was_climbing {
                self.motion = Motion::Free;
            }
        } else if was_climbing {
            self.motion = Motion::Free;
        }
        let climbing = self.motion == Motion::Climb;
        ev.climbing = climbing;

        // ---- horizontal drive ----
        let mut vh = Vec2::new(self.vel.x, self.vel.z);
        let wish2 = Vec2::new(wish.x, wish.z);
        // On a wall, drive gently INTO it so contact is kept while ascending.
        let want = wish2
            * if climbing { self.tuning.climb_speed } else { self.target_speed(&input) };
        // Rate at which the body may change horizontal velocity: full grip on
        // walkable ground, a third of it in the air, none at all on a slope it
        // cannot hold (that is what losing your footing means), and no air drag
        // when nothing is pressed so a jump keeps its momentum.
        let rate = if climbing {
            self.tuning.ground_accel
        } else if self.grounded && self.steep {
            0.0
        } else if self.grounded {
            if pressing { self.tuning.ground_accel } else { self.tuning.ground_friction }
        } else if pressing {
            self.tuning.air_accel
        } else {
            0.0
        };
        let dv = want - vh;
        let acc = rate * dt;
        if dv.length_squared() <= acc * acc {
            vh = want;
        } else {
            vh += dv.normalize() * acc;
        }
        if self.grounded && self.steep && !climbing {
            // Gravity's component along the surface, projected back onto the
            // horizontal plane: g*cos(t)*sin(t) downhill, which for a unit
            // normal is exactly g*n.y*(n.x, n.z). The vertical resolve zeroes the
            // fall every tick, so the slide has to be driven here.
            let n = ground.normal;
            vh += Vec2::new(n.x, n.z) * (self.tuning.gravity * n.y * dt);
            // Drag, so a long face reaches a terminal slide instead of
            // accelerating without bound.
            vh -= vh * (self.tuning.slide_drag * dt);
        }
        self.vel.x = vh.x;
        self.vel.z = vh.y;

        // ---- vertical drive ----
        if climbing {
            self.vel.y = self.tuning.climb_speed;
        } else if jump_edge && self.grounded && !self.steep {
            self.vel.y = self.tuning.jump_speed;
            self.grounded = false;
            ev.jumped = true;
        } else {
            self.vel.y -= self.tuning.gravity * dt;
            if self.vel.y < -self.tuning.terminal_speed {
                self.vel.y = -self.tuning.terminal_speed;
            }
        }

        // ---- integrate against the world ----
        let was_grounded = self.grounded;
        let delta = self.vel * dt;
        let assist = was_grounded && !ev.jumped && !climbing;
        self.move_and_collide(world, delta, assist, &mut ev);

        // Walking off a step should not launch the body: if it was on the ground
        // and is now falling with ground still within step range, put it down.
        if assist && !self.grounded && self.vel.y <= 0.0 {
            let snap =
                voxquery::sweep(world, self.body(), Axis::Y, -self.tuning.step_max, MatSet::SOLID);
            if snap.blocked {
                self.pos.y -= snap.free;
                self.grounded = true;
                self.vel.y = 0.0;
                self.ground_mat = snap.mat;
                ev.ground_mat = snap.mat;
            }
        }

        self.unstick(world);
        self.update_eye(dt);
        ev
    }

    // ------------------------------------------------------------- collision

    /// Move by `delta`, resolving against the world one axis at a time so a
    /// blocked axis still slides along the others.
    ///
    /// MULTIPLAYER SEAM: every axis goes through exactly one `sweep` call, so
    /// other players' boxes join by taking the min of this free distance and a
    /// box-vs-box sweep - there is no second resolution path to keep in step.
    fn move_and_collide(&mut self, world: &World, delta: Vec3, step_assist: bool, ev: &mut Events) {
        // Vertical first: this tick's footing is settled before the horizontal
        // move, which is what tells the step assist whether it may run at all.
        if delta.y != 0.0 {
            let s = voxquery::sweep(world, self.body(), Axis::Y, delta.y, MatSet::SOLID);
            self.pos.y += s.free * delta.y.signum();
            if s.blocked {
                if delta.y < 0.0 {
                    ev.landed = Some(-self.vel.y);
                    self.grounded = true;
                    self.ground_mat = s.mat;
                    ev.ground_mat = s.mat;
                }
                self.vel.y = 0.0;
            }
        }

        let before = self.pos;
        let vel_before = self.vel;
        // Larger axis first: sliding into a corner then resolves against the
        // wall the body is actually driving at.
        let (first, second) = if delta.x.abs() >= delta.z.abs() {
            (Axis::X, Axis::Z)
        } else {
            (Axis::Z, Axis::X)
        };
        let d = |a: Axis| if a == Axis::X { delta.x } else { delta.z };
        let hit_a = self.slide_axis(world, first, d(first));
        let hit_b = self.slide_axis(world, second, d(second));
        if !(hit_a || hit_b) || !step_assist || self.steep {
            return;
        }

        // Blocked while walking: retry the same move raised by up to step_max
        // and drop back down. This is how a body scrambles over 10-25 cm terrain
        // roughness instead of stopping dead at every pebble - and why there is
        // no Minecraft one-metre free climb, because step_max is where it ends.
        let plain = self.pos;
        self.pos = before;
        let lift =
            voxquery::sweep(world, self.body(), Axis::Y, self.tuning.step_max, MatSet::SOLID).free;
        if lift > 1.0e-3 {
            self.pos.y += lift;
            self.slide_axis(world, first, d(first));
            self.slide_axis(world, second, d(second));
            let drop = voxquery::sweep(world, self.body(), Axis::Y, -lift, MatSet::SOLID);
            let further = plan_dist(before, self.pos) > plan_dist(before, plain) + 1.0e-4;
            // Only accept when the body ends up STANDING on what it climbed. An
            // unblocked drop means it stepped over into open air, which is a
            // vault, not a stride.
            if drop.blocked && further {
                self.pos.y -= drop.free;
                self.grounded = true;
                self.ground_mat = drop.mat;
                ev.ground_mat = drop.mat;
                ev.climbed += (self.pos.y - before.y).max(0.0);
                // Taking a rise in your stride is NOT a collision, so give back
                // the horizontal velocity the failed low attempt zeroed. Without
                // this the body restarts from a standstill at every riser and a
                // staircase is walked at a quarter speed - measured at 11 voxels
                // of climb in 4 s against the 29 it should manage.
                self.vel.x = vel_before.x;
                self.vel.z = vel_before.z;
                return;
            }
        }
        self.pos = plain;
    }

    /// Sweep one horizontal axis, stop at contact, kill that axis' velocity.
    /// Returns true if it was blocked.
    fn slide_axis(&mut self, world: &World, axis: Axis, delta: f32) -> bool {
        if delta == 0.0 {
            return false;
        }
        let s = voxquery::sweep(world, self.body(), axis, delta, MatSet::SOLID);
        let moved = s.free * delta.signum();
        match axis {
            Axis::X => {
                self.pos.x += moved;
                if s.blocked {
                    self.vel.x = 0.0;
                }
            }
            Axis::Z => {
                self.pos.z += moved;
                if s.blocked {
                    self.vel.z = 0.0;
                }
            }
            Axis::Y => unreachable!("vertical motion is resolved by move_and_collide"),
        }
        s.blocked
    }

    /// Push the body out of anything it has ended up inside.
    ///
    /// Not defensive programming: a player can place a voxel on their own feet,
    /// physics can drop sand into them, and a streaming install can materialise
    /// terrain around them. Without this the body is stuck forever, because a
    /// sweep deliberately ignores the cells it already occupies.
    fn unstick(&mut self, world: &World) {
        if !voxquery::overlaps(world, self.body(), MatSet::SOLID) {
            return;
        }
        // Up first (where the free space usually is when something was placed
        // underfoot), then sideways, nearest distance first.
        let reach = self.dims.height(self.crouching).max(self.dims.half_width * 2.0) + 1.0;
        let mut d = 0.25;
        while d <= reach {
            for dir in [Vec3::Y, Vec3::X, Vec3::NEG_X, Vec3::Z, Vec3::NEG_Z, Vec3::NEG_Y] {
                let candidate = self.pos + dir * d;
                if !voxquery::overlaps(world, self.body_at(candidate, self.crouching), MatSet::SOLID)
                {
                    self.pos = candidate;
                    self.vel = Vec3::ZERO;
                    return;
                }
            }
            d += 0.25;
        }
        // Nothing within a body's reach is clear (buried in solid rock). Leave
        // it where it is rather than teleporting it somewhere arbitrary.
    }

    // ---------------------------------------------------------------- footing

    /// Where the ground is, what it is made of, and how steep it is.
    fn probe_ground(&self, world: &World) -> Ground {
        // Start every probe just above the feet: a resting body sits one contact
        // skin above its floor, and a probe starting exactly at the feet would
        // read that skin as a gap.
        let eps = 0.05;
        let depth = self.tuning.ground_probe_depth;

        // Contact and material come from the BODY's own box, because that is
        // what actually rests on the world - deriving them from the sample
        // columns below could disagree with where the body is really standing.
        let raised = self.body_at(self.pos + Vec3::Y * eps, self.crouching);
        let under = voxquery::sweep(world, raised, Axis::Y, -(eps + depth), MatSet::SOLID);
        // `free` is how far the RAISED box may fall, so the gap under the feet
        // is free - eps. (Comparing the other way round makes a body standing a
        // metre in the air read as grounded, because anything below counts.)
        let contact = under.blocked && (under.free - eps) <= GROUND_TOL;

        // The gradient is sampled at four column CENTRES exactly two voxels
        // apart. Both halves of that matter: a probe straddling a voxel boundary
        // silently reports the higher of two columns, and a non-integer sample
        // separation quantises the measured gradient so badly that a 45-degree
        // staircase and a 63-degree face land in the same bucket (measured: both
        // could read tan 1.04). Snapped to columns, a 1:1 face measures exactly
        // 1.0 and a 2:1 face exactly 2.0, on any terrain the worldgen makes.
        const SPAN: f32 = 2.0;
        const CORNERS: [(f32, f32); 4] = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)];
        let cx = self.pos.x.floor() + 0.5;
        let cz = self.pos.z.floor() + 0.5;
        let mut h = [f32::NAN; 4];
        for (i, (sx, sz)) in CORNERS.iter().enumerate() {
            let px = cx + sx * (SPAN * 0.5);
            let pz = cz + sz * (SPAN * 0.5);
            let probe = Aabb::new(
                Vec3::new(px - 0.25, self.pos.y + eps, pz - 0.25),
                Vec3::new(px + 0.25, self.pos.y + eps + 0.02, pz + 0.25),
            );
            let s = voxquery::sweep(world, probe, Axis::Y, -(eps + depth), MatSet::SOLID);
            if s.blocked {
                h[i] = probe.min.y - s.free;
            }
        }
        // A gradient needs all four samples. At an edge, where a sample hangs
        // over nothing, the honest answer is "unknown", and the safe reading of
        // unknown is FLAT - refusing to grip there would shove the body off
        // every ledge it walked up to.
        if h.iter().any(|v| v.is_nan()) {
            return Ground { contact, mat: under.mat, slope_tan: 0.0, normal: Vec3::Y };
        }
        let dh_dx = ((h[1] + h[3]) - (h[0] + h[2])) / (2.0 * SPAN);
        let dh_dz = ((h[2] + h[3]) - (h[0] + h[1])) / (2.0 * SPAN);
        Ground {
            contact,
            mat: under.mat,
            slope_tan: Vec2::new(dh_dx, dh_dz).length(),
            normal: Vec3::new(-dh_dx, 1.0, -dh_dz).normalize(),
        }
    }

    // ------------------------------------------------------- vault and climb

    /// Is there a wall in front worth climbing? Deliberately blind below
    /// `step_max`, so walking into ordinary terrain roughness is a stride and
    /// never a climb.
    fn wall_ahead(&self, world: &World, dir: Vec3) -> bool {
        let hw = self.dims.half_width;
        let h = self.dims.height(self.crouching);
        let c = self.pos + dir * (hw + 0.25);
        let lo = self.pos.y + self.tuning.step_max + 0.1;
        let hi = self.pos.y + h - 0.1;
        lo < hi
            && voxquery::overlaps(
                world,
                Aabb::new(Vec3::new(c.x - 0.3, lo, c.z - 0.3), Vec3::new(c.x + 0.3, hi, c.z + 0.3)),
                MatSet::SOLID,
            )
    }

    /// Start a vault if there is a ledge in `dir` that can be mantled.
    fn try_vault(&mut self, world: &World, dir: Vec3, ev: &mut Events) -> bool {
        let Some(to) = self.find_vault(world, dir) else { return false };
        let dur = ((to.y - self.pos.y) / self.tuning.vault_rise_rate).clamp(0.25, 0.6);
        self.motion = Motion::Vault { from: self.pos, to, t: 0.0, dur };
        self.crouching = false; // the destination was validated standing
        self.vel = Vec3::ZERO;
        self.grounded = false;
        ev.vault_started = true;
        true
    }

    /// The feet position a vault in `dir` would end at, if one is possible.
    fn find_vault(&self, world: &World, dir: Vec3) -> Option<Vec3> {
        let hw = self.dims.half_width;
        let h = self.dims.stand_height;
        let t = &self.tuning;
        // One body-width ahead: far enough to end up standing ON the ledge
        // rather than balanced on its lip.
        let ahead = self.pos + dir * (hw * 2.0 + 0.5);
        // A vault is a mantle, not a leap: the body must be AT the obstacle. A
        // ledge detected from a body-width away lands the body balanced on the
        // lip with its centre over the void, because the destination is measured
        // forward from wherever the vault happened to trigger.
        let face = self.pos + dir * (hw + 0.3);
        let contact = Aabb::new(
            Vec3::new(face.x - 0.25, self.pos.y + 0.05, face.z - 0.25),
            Vec3::new(face.x + 0.25, self.pos.y + t.vault_max, face.z + 0.25),
        );
        if !voxquery::overlaps(world, contact, MatSet::SOLID) {
            return None;
        }
        let top = self.pos.y + t.vault_max;
        let raised = Aabb::from_feet(Vec3::new(ahead.x, top, ahead.z), hw, h);
        if voxquery::overlaps(world, raised, MatSet::SOLID) {
            return None; // no room to stand up there
        }
        // Find the ledge by dropping the destination box through the band
        // between vault_max and step_max.
        let drop =
            voxquery::sweep(world, raised, Axis::Y, -(t.vault_max - t.step_max), MatSet::SOLID);
        if !drop.blocked {
            return None; // nothing to land on within reach
        }
        let land_y = top - drop.free;
        if land_y <= self.pos.y + t.step_max {
            return None; // step assist's job, not a vault
        }
        // The whole volume the body passes through, as one conservative box: if
        // that is clear, no path inside it can be obstructed.
        let span = Aabb::new(
            Vec3::new(self.pos.x.min(ahead.x) - hw, land_y, self.pos.z.min(ahead.z) - hw),
            Vec3::new(self.pos.x.max(ahead.x) + hw, land_y + h, self.pos.z.max(ahead.z) + hw),
        );
        if voxquery::overlaps(world, span, MatSet::SOLID) {
            return None;
        }
        // ... and the rise out of where the body stands into that volume.
        if voxquery::sweep(world, self.body(), Axis::Y, land_y - self.pos.y, MatSet::SOLID).blocked {
            return None;
        }
        Some(Vec3::new(ahead.x, land_y, ahead.z))
    }

    fn advance_vault(&mut self, dt: f32, ev: &mut Events) {
        let Motion::Vault { from, to, t, dur } = self.motion else { return };
        let t = (t + dt).min(dur);
        let u = (t / dur).clamp(0.0, 1.0);
        // Rise first, then move over: the shape of a real mantle, and it keeps
        // the body clear of the ledge lip for the whole path.
        let uy = smoothstep((u / 0.6).min(1.0));
        let uxz = smoothstep(((u - 0.35) / 0.65).clamp(0.0, 1.0));
        let prev_y = self.pos.y;
        self.pos = Vec3::new(
            from.x + (to.x - from.x) * uxz,
            from.y + (to.y - from.y) * uy,
            from.z + (to.z - from.z) * uxz,
        );
        ev.climbed += (self.pos.y - prev_y).max(0.0);
        self.vel = Vec3::ZERO;
        if t >= dur {
            self.pos = to;
            self.motion = Motion::Free;
            self.grounded = true;
        } else {
            self.motion = Motion::Vault { from, to, t, dur };
        }
    }

    // ----------------------------------------------------------------- crouch

    fn update_crouch(&mut self, world: &World, want: bool) {
        if want {
            self.crouching = true;
            return;
        }
        if !self.crouching {
            return;
        }
        // Standing up into a ceiling would push the head through it; stay down.
        if !voxquery::overlaps(world, self.body_at(self.pos, false), MatSet::SOLID) {
            self.crouching = false;
        }
    }

    fn update_eye(&mut self, dt: f32) {
        let target = self.dims.eye_height(self.crouching);
        // Crossing the whole crouch range takes 0.12 s: fast enough to read as
        // the body doing it, slow enough not to snap the view.
        let rate = (self.dims.stand_height - self.dims.crouch_height) / 0.12;
        let d = target - self.eye_offset;
        let m = rate * dt;
        self.eye_offset += d.clamp(-m, m);
    }
}

struct Ground {
    contact: bool,
    mat: u8,
    slope_tan: f32,
    normal: Vec3,
}

#[inline]
fn wish_dir(input: &Input) -> Vec3 {
    // Same basis as Camera::forward/right with pitch removed: you walk where you
    // are facing, not where your nose points.
    let (sy, cy) = input.yaw.sin_cos();
    let v = Vec3::new(sy, 0.0, cy) * input.move_axis.y + Vec3::new(cy, 0.0, -sy) * input.move_axis.x;
    let len2 = v.length_squared();
    if len2 > 1.0 { v / len2.sqrt() } else { v }
}

#[inline]
fn plan_dist(a: Vec3, b: Vec3) -> f32 {
    Vec2::new(b.x - a.x, b.z - a.z).length()
}

#[inline]
fn smoothstep(x: f32) -> f32 {
    x * x * (3.0 - 2.0 * x)
}

/// Fixed-timestep driver: steps the body at a constant rate whatever the frame
/// rate, and interpolates the eye for rendering.
///
/// The engine runs from 60 to 400+ fps. Feeding a frame delta straight into the
/// controller would make jump height, stopping distance and step assist all
/// frame-rate dependent, so the accumulator is not a nicety - it is what makes
/// the numbers in the feel tests mean anything.
pub struct PlayerSim {
    pub player: Player,
    prev_pos: Vec3,
    prev_eye: f32,
    accum: f32,
    /// Fixed ticks simulated since construction. The sim's own clock, which is
    /// what "the same amount of game happened" means independently of frames.
    pub ticks: u64,
    /// Sticky button state since the last simulated tick. At 400 fps most
    /// frames run NO tick, so a button that is pressed and released between two
    /// ticks would never be seen by the controller at all - a jump that simply
    /// does not happen, at random, more often the faster the machine. The latch
    /// holds a press until a tick consumes it, then falls back to whatever the
    /// key is actually doing.
    latch: Input,
}

impl PlayerSim {
    pub const TICK_HZ: f32 = 60.0;
    pub const TICK_DT: f32 = 1.0 / Self::TICK_HZ;
    /// Ticks one frame may replay. A hitch catches up; a five-second stall does
    /// NOT replay 300 ticks and rubber-band the body across the map.
    pub const MAX_CATCHUP: u32 = 6;

    pub fn new(feet: Vec3) -> Self {
        let player = Player::new(feet);
        Self {
            prev_pos: player.pos,
            prev_eye: player.eye_offset,
            player,
            accum: 0.0,
            ticks: 0,
            latch: Input::default(),
        }
    }

    pub fn advance(&mut self, world: &World, input: Input, frame_dt: f32) -> Events {
        self.accum += frame_dt.clamp(0.0, 1.0);
        // Buttons are sticky until a tick sees them; the analogue channels just
        // take the newest value.
        self.latch.jump |= input.jump;
        self.latch.sprint |= input.sprint;
        self.latch.crouch |= input.crouch;
        self.latch.move_axis = input.move_axis;
        self.latch.yaw = input.yaw;
        let mut ev = Events::default();
        let mut steps = 0;
        while self.accum >= Self::TICK_DT && steps < Self::MAX_CATCHUP {
            self.prev_pos = self.player.pos;
            self.prev_eye = self.player.eye_offset;
            ev = ev.merge(self.player.step(world, self.latch, Self::TICK_DT));
            // Consumed: a held key stays held, a tap falls away after the one
            // tick it was meant to affect.
            self.latch.jump = input.jump;
            self.latch.sprint = input.sprint;
            self.latch.crouch = input.crouch;
            self.accum -= Self::TICK_DT;
            self.ticks += 1;
            steps += 1;
        }
        if steps == Self::MAX_CATCHUP {
            self.accum = 0.0; // drop the backlog rather than chase it
        }
        ev
    }

    /// Fraction of a tick the renderer is ahead of the last simulated state.
    pub fn alpha(&self) -> f32 {
        (self.accum / Self::TICK_DT).clamp(0.0, 1.0)
    }

    /// Eye position for rendering: interpolated between the last two simulated
    /// states, so a 400 fps view of a 60 Hz body is smooth.
    pub fn eye(&self) -> Vec3 {
        let a = self.alpha();
        self.prev_pos.lerp(self.player.pos, a)
            + Vec3::Y * (self.prev_eye + (self.player.eye_offset - self.prev_eye) * a)
    }

    /// Drop the body somewhere new and forget the interpolation history.
    pub fn teleport(&mut self, feet: Vec3) {
        self.player.teleport(feet);
        self.prev_pos = self.player.pos;
        self.prev_eye = self.player.eye_offset;
        self.accum = 0.0;
        self.latch = Input::default();
    }
}

/// Voxels to metres, for reporting a simulated quantity in SI.
#[inline]
pub fn to_m(voxels: f32) -> f32 {
    vox_to_m(voxels)
}

// The tests below are the FEEL SPECIFICATION. Each measures a quantity the body
// actually produces, in SI, and asserts it against a band with the reasoning and
// the reference for that band written into the assertion. A later change that
// moves the feel therefore fails loudly instead of drifting quietly.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::MAT_STONE;
    use std::sync::OnceLock;

    const DT: f32 = PlayerSim::TICK_DT;

    /// Half-open box fill through the world's own edit path, so masks and
    /// uniform hints stay consistent with the bricks.
    fn fill(w: &mut World, x: (i32, i32), y: (i32, i32), z: (i32, i32), mat: u8) {
        for vz in z.0..z.1 {
            for vy in y.0..y.1 {
                for vx in x.0..x.1 {
                    w.apply_edit(vx, vy, vz, mat);
                }
            }
        }
    }

    const GROUND_Y: f32 = 64.0;

    /// Ground plane with its surface at y = 64, spanning 64 x 64 voxels from the
    /// origin. `World::new` allocates ~90 MB and spawns worker threads, so every
    /// flat-ground feel test shares this one immutable world.
    fn flat_world() -> &'static World {
        static W: OnceLock<World> = OnceLock::new();
        W.get_or_init(|| {
            let mut w = World::new();
            fill(&mut w, (0, 64), (60, 64), (0, 64), MAT_STONE);
            w
        })
    }

    /// A body standing still on the ground, settled (grounded, zero velocity).
    fn standing(world: &World, x: f32, z: f32) -> Player {
        let mut p = Player::new(Vec3::new(x, GROUND_Y + 0.5, z));
        for _ in 0..30 {
            p.step(world, Input::default(), DT);
        }
        assert!(p.grounded, "test body failed to settle on the ground");
        assert!((p.pos.y - GROUND_Y).abs() < 0.01, "settled at {} not {GROUND_Y}", p.pos.y);
        p
    }

    /// Hold the given input for `secs`.
    fn hold(p: &mut Player, world: &World, input: Input, secs: f32) {
        for _ in 0..(secs / DT).round() as i32 {
            p.step(world, input, DT);
        }
    }

    fn forward(sprint: bool) -> Input {
        Input { move_axis: Vec2::new(0.0, 1.0), sprint, ..Default::default() }
    }

    /// Forward along +x. `Camera::yaw` = 0 faces +z, so the obstacle course,
    /// which is laid out along x, needs a quarter turn.
    fn walk_x(sprint: bool, jump: bool) -> Input {
        Input {
            move_axis: Vec2::new(0.0, 1.0),
            yaw: std::f32::consts::FRAC_PI_2,
            sprint,
            jump,
            ..Default::default()
        }
    }

    fn speed_ms(p: &Player) -> f32 {
        to_m(Vec2::new(p.vel.x, p.vel.z).length())
    }

    /// Assert a measured feel quantity sits inside its stated band.
    fn band(name: &str, got: f32, lo: f32, hi: f32, unit: &str, why: &str) {
        assert!(
            got >= lo && got <= hi,
            "{name} = {got:.3} {unit}, outside the band [{lo}, {hi}] {unit}\n  band: {why}",
        );
    }

    // ------------------------------------------------------------ feel: speed

    #[test]
    fn time_to_walk_and_sprint_speed() {
        let w = flat_world();
        for (sprint, target, lo, hi) in [(false, 3.0f32, 0.10f32, 0.50f32), (true, 5.5, 0.15, 0.80)]
        {
            let mut p = standing(w, 32.0, 32.0);
            let mut t = 0.0;
            for _ in 0..600 {
                p.step(w, forward(sprint), DT);
                t += DT;
                if speed_ms(&p) >= target * 0.99 {
                    break;
                }
            }
            band(
                if sprint { "time to sprint speed" } else { "time to walk speed" },
                t,
                lo,
                hi,
                "s",
                "Minecraft's near-instant ~0.1 s is the responsive floor; past 0.5 s reads as ice. \
                 SURVIVAL_PLAN A2 asks for momentum, so instant is wrong too - the band excludes both ends",
            );
            band("reached speed", speed_ms(&p), target * 0.98, target * 1.02, "m/s", "the target itself");
        }
    }

    #[test]
    fn stopping_distance_from_walk_and_sprint() {
        let w = flat_world();
        for (sprint, lo, hi) in [(false, 0.15f32, 0.60f32), (true, 0.40, 1.60)] {
            let mut p = standing(w, 32.0, 32.0);
            hold(&mut p, w, forward(sprint), 1.5);
            let from = p.pos;
            for _ in 0..600 {
                p.step(w, Input::default(), DT);
                if speed_ms(&p) < 0.01 {
                    break;
                }
            }
            let dist = to_m(plan_dist(from, p.pos));
            band(
                if sprint { "sprint stopping distance" } else { "walk stopping distance" },
                dist,
                lo,
                hi,
                "m",
                "a body that stops dead has no momentum (SURVIVAL_PLAN A2); one that slides metres is on ice. \
                 Minecraft coasts roughly half a block after release",
            );
            assert!(speed_ms(&p) < 0.01, "still moving after a second of friction");
        }
    }

    #[test]
    fn jump_apex_and_airtime() {
        let w = flat_world();
        let mut p = standing(w, 32.0, 32.0);
        let start = p.pos.y;
        let ev = p.step(w, Input { jump: true, ..Default::default() }, DT);
        assert!(ev.jumped, "a tap of jump on flat ground must jump");
        let mut apex = p.pos.y;
        let mut airtime = DT;
        for _ in 0..600 {
            p.step(w, Input::default(), DT);
            airtime += DT;
            apex = apex.max(p.pos.y);
            if p.grounded {
                break;
            }
        }
        band(
            "jump apex",
            to_m(apex - start),
            0.40,
            1.25,
            "m",
            "a real standing vertical is 0.4-0.6 m (floor); Minecraft's 1.2522 blocks is the arcade \
             ceiling and precisely what this design rejects",
        );
        band(
            "airtime",
            airtime,
            0.50,
            0.90,
            "s",
            "Minecraft is airborne ~0.75 s per jump, a real standing jump ~0.6 s",
        );
        assert!(p.grounded && (p.pos.y - GROUND_Y).abs() < 0.01, "must land back on the floor");
    }

    #[test]
    fn terminal_velocity_is_reached_and_clamped() {
        // No ground at all: the body falls out of the loaded window and keeps
        // falling, which is the only way to see terminal velocity in a world
        // that is 64 m tall.
        let empty = World::new();
        let mut p = Player::new(Vec3::new(32.0, 100.0, 32.0));
        for _ in 0..(6.0 / DT) as i32 {
            p.step(&empty, Input::default(), DT);
        }
        let v = to_m(-p.vel.y);
        band(
            "terminal velocity",
            v,
            50.0,
            78.4,
            "m/s",
            "a real belly-down skydiver terminates near 53 m/s (floor); Minecraft's player terminal \
             is 3.92 blocks/tick = 78.4 m/s (ceiling)",
        );
        // And it is a clamp, not an asymptote: another second must not add to it.
        for _ in 0..60 {
            p.step(&empty, Input::default(), DT);
        }
        assert!((to_m(-p.vel.y) - v).abs() < 1.0e-3, "terminal speed is not clamped");
    }

    #[test]
    fn air_control_steers_but_does_not_drive() {
        // Jumping from a standstill and holding forward must NOT reach walking
        // speed in the air: air accel is a third of ground accel, and a body
        // that accelerates freely in mid-air is a flying body.
        let w = flat_world();
        let mut p = standing(w, 32.0, 32.0);
        p.step(w, Input { jump: true, ..Default::default() }, DT);
        let mut peak: f32 = 0.0;
        for _ in 0..600 {
            p.step(w, forward(false), DT);
            peak = peak.max(speed_ms(&p));
            if p.grounded {
                break;
            }
        }
        band(
            "speed gained from a standing jump with forward held",
            peak,
            0.3,
            1.8,
            "m/s",
            "must steer (> 0) but stay well under the 3.0 m/s walk; Minecraft's air acceleration is \
             about a fifth of its ground value",
        );
    }

    #[test]
    fn the_simulation_is_frame_rate_independent() {
        // The same 2 seconds of walking and one jump, at 60, 144 and 400 fps.
        //
        // What must be IDENTICAL is the character of the motion - the jump
        // reaches the same height, the body carries the same speed - because the
        // controller only ever sees fixed 60 Hz ticks. What cannot be identical
        // is the exact position after a fixed wall-clock time: 2 seconds is not
        // a whole number of ticks at every frame rate, so the runs can end one
        // tick apart in phase. The test pins the first and bounds the second,
        // rather than pretending an accumulator can do the impossible.
        let w = flat_world();
        let run = |fps: f32| {
            let mut sim = PlayerSim::new(Vec3::new(32.0, GROUND_Y + 0.5, 32.0));
            let frame = 1.0 / fps;
            let frames = (2.0 * fps).round() as i32;
            let mut input = Input::default();
            let mut apex = f32::NEG_INFINITY;
            for i in 0..frames {
                let t = i as f32 * frame;
                // Tap jump half a second in, on whatever single frame straddles
                // it - at 400 fps that pulse is 2.5 ms long and only survives
                // because PlayerSim latches it.
                input.jump = t >= 0.5 && t < 0.5 + frame;
                input.move_axis = Vec2::new(0.0, 1.0);
                sim.advance(w, input, frame);
                apex = apex.max(sim.player.pos.y);
            }
            (sim.player.pos, sim.player.vel, apex, sim.ticks)
        };
        let runs = [run(60.0), run(144.0), run(400.0)];
        let (p0, v0, apex0, ticks0) = runs[0];
        for (i, (p, v, apex, ticks)) in runs.iter().copied().enumerate() {
            let dticks = (ticks as i64 - ticks0 as i64).unsigned_abs();
            assert!(dticks <= 1, "run {i} simulated {ticks} ticks against {ticks0}");
            assert!(
                (apex - apex0).abs() < 1.0e-3,
                "run {i} jumped to {apex} against {apex0}: jump height moved with frame rate",
            );
            assert!(
                (v - v0).length() < 1.0e-3,
                "run {i} carries velocity {v:?} against {v0:?}",
            );
            // One tick of walking is walk_speed * TICK_DT = 0.2 voxels.
            let slack = (dticks as f32 + 1.0) * Tuning::default().walk_speed * DT;
            assert!(
                (p - p0).length() <= slack,
                "run {i} ended at {p:?} against {p0:?}, more than {dticks} ticks of phase apart",
            );
        }
    }

    #[test]
    fn a_frame_hitch_does_not_rubber_band_the_body() {
        // A five-second stall must not replay five seconds of walking in one
        // frame; the catch-up cap drops the backlog instead.
        let w = flat_world();
        let mut sim = PlayerSim::new(Vec3::new(32.0, GROUND_Y + 0.5, 32.0));
        sim.advance(w, Input::default(), 0.5);
        let before = sim.player.pos;
        sim.advance(w, forward(true), 5.0);
        let moved = to_m(plan_dist(before, sim.player.pos));
        assert!(moved < 1.0, "a 5 s hitch moved the body {moved:.2} m in one frame");
    }

    // ------------------------------------------------- feel: terrain contact

    /// One shared obstacle course, laid out in separate sites so every geometry
    /// test can use the same 90 MB world. Ground surface is y = 64 throughout.
    ///
    ///   x 30..36, z 10..68  six walls, 1..6 voxels tall (the step-assist ladder)
    ///   x 60..90, z 10..20  a 1:1 staircase - 45 degrees
    ///   x 100..130, z 20..30 a 2:1 staircase - 63 degrees
    ///   x 60..90, z 30..40  a 1:2 staircase - 26.6 degrees
    ///   x 20..30, z 80..90  a slab at y 70..74: 1.5 m of headroom (crouch only)
    ///   x 130..140, z 10..20 a 1.0 m ledge (vault band)
    ///   x 130..140, z 30..40 a 5 m wall (climb band)
    ///   x 150, z 50..60     a ONE voxel thick wall (tunnelling)
    ///   x 20..30, z 120..130 water and leaves floating at y 80..84 and 88..92
    fn course() -> &'static World {
        static W: OnceLock<World> = OnceLock::new();
        W.get_or_init(|| {
            let mut w = World::new();
            fill(&mut w, (0, 176), (60, 64), (0, 144), MAT_STONE);
            for h in 1..=6i32 {
                let z0 = 10 * h;
                fill(&mut w, (30, 36), (64, 64 + h), (z0, z0 + 8), MAT_STONE);
            }
            for i in 0..30i32 {
                fill(&mut w, (60 + i, 61 + i), (64, 64 + i), (10, 20), MAT_STONE);
                fill(&mut w, (100 + i, 101 + i), (64, 64 + 2 * i), (20, 30), MAT_STONE);
                fill(&mut w, (60 + i, 61 + i), (64, 64 + i / 2), (30, 40), MAT_STONE);
            }
            fill(&mut w, (20, 30), (70, 74), (80, 90), MAT_STONE);
            fill(&mut w, (130, 140), (64, 68), (10, 20), MAT_STONE);
            fill(&mut w, (130, 140), (64, 84), (30, 40), MAT_STONE);
            fill(&mut w, (150, 151), (64, 76), (50, 60), MAT_STONE);
            fill(&mut w, (20, 30), (80, 84), (120, 130), crate::voxel::MAT_WATER_L8);
            fill(&mut w, (20, 30), (88, 92), (120, 130), crate::voxel::MAT_LEAVES);
            w
        })
    }

    /// Drop a body at (x, z) from just above `y` and let it settle.
    fn settle(world: &World, x: f32, y: f32, z: f32) -> Player {
        let mut p = Player::new(Vec3::new(x, y + 0.5, z));
        for _ in 0..40 {
            p.step(world, Input::default(), DT);
        }
        p
    }

    #[test]
    fn step_assist_climbs_up_to_the_step_height_and_no_further() {
        // THE ANTI-MINECRAFT TEST. A body walks up terrain roughness without
        // thinking about it, and stops dead at anything taller - there is no
        // free one-metre block climb. Every wall in the ladder is walked into
        // for a full second with no jump.
        let w = course();
        let t = Tuning::default();
        let mut highest_climbed = 0.0f32;
        for h in 1..=6i32 {
            let z = 10.0 * h as f32 + 4.0;
            let mut p = settle(w, 26.0, 64.0, z);
            assert!(p.grounded, "wall {h}: body did not settle");
            let start_y = p.pos.y;
            hold(&mut p, w, walk_x(false, false), 1.0);
            let climbed = p.pos.y - start_y;
            let expected = (h as f32) <= t.step_max;
            if expected {
                assert!(
                    (climbed - h as f32).abs() < 0.05 && p.pos.x > 30.0,
                    "a {h}-voxel step ({:.2} m, under the {:.2} m step height) must be walked up; \
                     climbed {climbed:.2} voxels, ended at x {:.1}",
                    to_m(h as f32),
                    to_m(t.step_max),
                    p.pos.x,
                );
                highest_climbed = highest_climbed.max(h as f32);
            } else {
                assert!(
                    climbed < 0.05 && p.pos.x < 30.0,
                    "a {h}-voxel step ({:.2} m, over the {:.2} m step height) must NOT be walked \
                     up; climbed {climbed:.2} voxels, ended at x {:.1}",
                    to_m(h as f32),
                    to_m(t.step_max),
                    p.pos.x,
                );
            }
        }
        band(
            "tallest step walked up without a vault",
            to_m(highest_climbed),
            0.20,
            0.50,
            "m",
            "a curb is 0.15 m and a tall stair riser 0.21 m, so a body should take 0.2-0.5 m in \
             its stride; Minecraft's 1.0 m free block step-up is the thing this rejects",
        );
    }

    #[test]
    fn slopes_are_walkable_up_to_the_limit_and_slide_beyond_it() {
        let w = course();
        // 26.6 degrees (1 voxel up per 2 across) and 45 degrees (1:1) are both
        // ascended: 45 is where Unreal (44.8) and Source (45.6) put their
        // walkable floor, and on a voxel lattice it is also the steepest thing
        // step assist can reach.
        for (name, x0, z, expect_gain) in
            [("26.6 degree", 58.0f32, 35.0f32, 8.0f32), ("45 degree", 58.0, 15.0, 16.0)]
        {
            let mut p = settle(w, x0, 64.0, z);
            let y0 = p.pos.y;
            // Peak, not final: these ramps are 30 voxels long and a body at
            // 12 voxels/s crests them and walks off the far end within 4 s.
            let mut peak = y0;
            let mut ever_steep = false;
            for _ in 0..(4.0 / DT) as i32 {
                p.step(w, walk_x(false, false), DT);
                peak = peak.max(p.pos.y);
                ever_steep |= p.steep;
            }
            let gained = peak - y0;
            assert!(
                gained >= expect_gain && !ever_steep,
                "{name} slope: climbed {gained:.1} voxels (wanted >= {expect_gain}), ever steep: {ever_steep}",
            );
        }
        // 63.4 degrees (2:1) is past the limit: no footing. The body rests on
        // the highest column under its footprint, which on this face is one
        // column ahead of its centre - hence the 86.5 rather than 84.5.
        let mut p = settle(w, 110.0, 86.5, 25.0);
        let mut ever_steep = false;
        for _ in 0..30 {
            p.step(w, Input::default(), DT);
            ever_steep |= p.steep;
        }
        assert!(ever_steep, "the 2:1 face must read as steep while the body is on it");
        // Walking at it gains nothing: no grip, and step assist is off.
        let mut p = settle(w, 110.0, 86.5, 25.0);
        let y0 = p.pos.y;
        let mut peak = y0;
        for _ in 0..(2.0 / DT) as i32 {
            p.step(w, walk_x(false, false), DT);
            peak = peak.max(p.pos.y);
        }
        assert!(
            peak <= y0 + 0.05,
            "a 63-degree face must not be climbable, gained {:.2} voxels",
            peak - y0,
        );
        // ... and released, the body slides DOWN it (downhill is -x here).
        let mut p = settle(w, 110.0, 86.5, 25.0);
        let (x0, y0) = (p.pos.x, p.pos.y);
        hold(&mut p, w, Input::default(), 1.5);
        assert!(
            p.pos.x < x0 - 1.0 && p.pos.y < y0 - 1.0,
            "a body must slide off a 63-degree face: moved ({:.2}, {:.2}) voxels",
            p.pos.x - x0,
            p.pos.y - y0,
        );
        band(
            "slide threshold",
            55.0,
            45.0,
            65.0,
            "degrees",
            "must sit above the 45-degree 1:1 voxel staircase a body walks up and below the \
             63-degree 2:1 face it must lose its footing on",
        );
    }

    #[test]
    fn crouching_shrinks_the_body_slows_it_and_traps_it_under_a_ceiling() {
        let w = course();
        let dims = Dims::default();
        let mut p = settle(w, 25.0, 64.0, 95.0);
        let crouch = Input { crouch: true, ..Default::default() };

        // The box really shrinks.
        p.step(w, crouch, DT);
        let b = p.body();
        assert!(
            (b.max.y - b.min.y - dims.crouch_height).abs() < 1.0e-4,
            "crouched box is {} tall, want {}",
            b.max.y - b.min.y,
            dims.crouch_height,
        );

        // ... and the speed drops with it.
        let mut moving = settle(w, 25.0, 64.0, 95.0);
        hold(&mut moving, w, Input { move_axis: Vec2::new(0.0, -1.0), crouch: true, ..Default::default() }, 1.5);
        band(
            "crouched speed",
            speed_ms(&moving),
            0.8,
            1.6,
            "m/s",
            "Minecraft sneaks at 1.31 m/s and a real crouch-walk is about 1.0",
        );

        // Under the 1.5 m slab, releasing crouch must NOT stand the body up
        // through the ceiling.
        let mut under = settle(w, 25.0, 64.0, 95.0);
        hold(&mut under, w, Input { move_axis: Vec2::new(0.0, -1.0), crouch: true, ..Default::default() }, 2.0);
        assert!(under.pos.z < 90.0, "the body should have crouch-walked under the slab, z={}", under.pos.z);
        hold(&mut under, w, Input::default(), 0.5);
        assert!(under.crouching, "stood up into a ceiling");
        assert!(
            !voxquery::overlaps(w, under.body(), MatSet::SOLID),
            "the body ended up inside the ceiling",
        );
        // Walking back out, it stands up again on its own.
        hold(&mut under, w, Input { move_axis: Vec2::new(0.0, 1.0), ..Default::default() }, 2.0);
        assert!(!under.crouching, "never stood back up in the open, z={}", under.pos.z);
    }

    #[test]
    fn vaulting_is_deliberate_and_clears_what_a_stride_cannot() {
        let w = course();
        let t = Tuning::default();
        // A 1.0 m ledge: over step_max (0.35 m), under vault_max (1.30 m).
        let ledge_top = 68.0;
        assert!(4.0 > t.step_max && 4.0 <= t.vault_max, "the test ledge must sit in the vault band");

        // Walking into it without the deliberate press gets nowhere.
        let mut p = settle(w, 126.0, 64.0, 15.0);
        hold(&mut p, w, walk_x(false, false), 2.0);
        assert!(
            p.pos.y < 65.0 && p.pos.x < 130.0,
            "a 1 m ledge must not be walked up: ended at ({}, {})",
            p.pos.x,
            p.pos.y,
        );

        // Pressed into it with jump, it is mantled.
        let mut p = settle(w, 126.0, 64.0, 15.0);
        let ev = {
            let mut acc = Events::default();
            for _ in 0..(2.0 / DT) as i32 {
                acc = acc.merge(p.step(w, walk_x(false, true), DT));
            }
            acc
        };
        assert!(ev.vault_started, "a deliberate press into a 1 m ledge must start a vault");
        assert!(
            (p.pos.y - ledge_top).abs() < 0.05 && p.pos.x > 130.0,
            "the vault must end standing on the ledge: ({}, {})",
            p.pos.x,
            p.pos.y,
        );
        assert!(p.grounded && !voxquery::overlaps(w, p.body(), MatSet::SOLID));
        assert!(ev.climbed > 3.0, "the vault should report the height it gained, got {}", ev.climbed);
    }

    #[test]
    fn climbing_is_held_and_tops_out_on_the_ledge() {
        let w = course();
        // The 5 m wall is far past vault_max, so it can only be climbed.
        let into_wall = walk_x(false, true);
        let mut p = settle(w, 126.0, 64.0, 35.0);
        let y0 = p.pos.y;
        hold(&mut p, w, into_wall, 2.0);
        let gained = to_m(p.pos.y - y0);
        band(
            "height gained in 2 s of climbing",
            gained,
            1.2,
            3.5,
            "m",
            "a scramble, not a ladder: real climbing is well under 1 m/s, and this must stay slow \
             enough that stamina (SURVIVAL_PLAN A3) has something to gate",
        );
        // Letting go drops you.
        let y1 = p.pos.y;
        hold(&mut p, w, Input::default(), 0.5);
        assert!(p.pos.y < y1 - 0.5, "releasing the climb must let the body fall");

        // Held all the way, it tops out onto the wall rather than sticking to it.
        let mut p = settle(w, 126.0, 64.0, 35.0);
        let mut topped = false;
        for _ in 0..(8.0 / DT) as i32 {
            p.step(w, into_wall, DT);
            if p.grounded && p.pos.y > 80.0 {
                topped = true;
                break;
            }
        }
        assert!(topped, "the climb never reached the top of the wall, stalled at {:?}", p.pos);
        assert!(
            (p.pos.y - 84.0).abs() < 0.2 && p.pos.x > 130.0,
            "a sustained climb must top out onto the ledge, ended at ({}, {})",
            p.pos.x,
            p.pos.y,
        );
    }

    #[test]
    fn the_body_cannot_tunnel_through_a_one_voxel_wall() {
        // 750 m/s in a single tick - 50 voxels of travel against a wall one
        // voxel thick. A move-then-test controller teleports straight through.
        let w = course();
        let mut p = settle(w, 110.0, 64.0, 55.0);
        p.vel.x = 3000.0;
        p.step(w, Input::default(), DT);
        assert!(
            p.pos.x + p.dims.half_width <= 150.0,
            "tunnelled to x {} through the wall at x 150",
            p.pos.x,
        );
        assert!(!voxquery::overlaps(w, p.body(), MatSet::SOLID));
        // And the same from the far side.
        let mut p = settle(w, 170.0, 64.0, 55.0);
        p.vel.x = -3000.0;
        p.step(w, Input::default(), DT);
        assert!(p.pos.x - p.dims.half_width >= 151.0, "tunnelled backwards to x {}", p.pos.x);
    }

    #[test]
    fn water_and_foliage_are_not_floors() {
        // The material classification has to survive the whole controller, not
        // just the query: a body falls through water and through leaves, and
        // lands on the stone below.
        let w = course();
        let mut p = Player::new(Vec3::new(25.0, 95.0, 125.0));
        for _ in 0..(4.0 / DT) as i32 {
            p.step(w, Input::default(), DT);
            if p.grounded {
                break;
            }
        }
        assert!(
            p.grounded && (p.pos.y - 64.0).abs() < 0.05,
            "the body should have fallen through leaves and water to the ground, stopped at {}",
            p.pos.y,
        );
    }

    #[test]
    fn a_body_inside_solid_matter_pushes_itself_out() {
        // Placing a block on your own feet, sand falling into you, or a chunk
        // streaming in around you: the body must not be stuck forever.
        let w = course();
        let mut p = Player::new(Vec3::new(135.0, 65.0, 15.0)); // inside the ledge
        assert!(voxquery::overlaps(w, p.body(), MatSet::SOLID), "test setup must start buried");
        for _ in 0..30 {
            p.step(w, Input::default(), DT);
        }
        assert!(
            !voxquery::overlaps(w, p.body(), MatSet::SOLID),
            "still stuck inside the world at {:?}",
            p.pos,
        );
    }

    #[test]
    fn the_body_survives_real_generated_terrain() {
        // The obstacle course proves the RULES; this proves they hold on what
        // the worldgen actually makes - hills, cliffs, lakes, trees, the lot.
        // The invariant that matters is that a body never ends a tick inside
        // solid matter and never falls out of the world.
        let mut w = World::new();
        w.fill_demo_terrain();
        let seed = w.seed;
        let mut ever_grounded = false;
        for (i, yaw) in (0..8).map(|i| (i, i as f32 * std::f32::consts::FRAC_PI_4)) {
            let (x, z) = (256.0, 256.0);
            let s = crate::voxel::sample_terrain(x, z, seed);
            let mut p = Player::new(Vec3::new(x, s.h.max(s.water_top) as f32 + 2.0, z));
            let input = Input { move_axis: Vec2::new(0.0, 1.0), yaw, sprint: true, ..Default::default() };
            for tick in 0..(4.0 / DT) as i32 {
                p.step(&w, input, DT);
                ever_grounded |= p.grounded;
                assert!(
                    !voxquery::overlaps(&w, p.body(), MatSet::SOLID),
                    "heading {i}: body inside terrain at {:?} on tick {tick}",
                    p.pos,
                );
                assert!(
                    p.pos.y > 0.0 && p.pos.y < crate::voxel::WORLD_VOXELS_Y as f32,
                    "heading {i}: body left the world at {:?}",
                    p.pos,
                );
                assert!(p.pos.is_finite(), "heading {i}: position went non-finite");
            }
        }
        assert!(ever_grounded, "the body never touched the ground on real terrain");
    }

    #[test]
    fn a_resting_body_neither_sinks_nor_jitters() {
        let w = flat_world();
        let mut p = standing(w, 32.0, 32.0);
        let y = p.pos.y;
        for i in 0..600 {
            p.step(w, Input::default(), DT);
            assert!(p.grounded, "lost the ground on tick {i}");
            assert!((p.pos.y - y).abs() < 1.0e-4, "drifted to {} from {y} by tick {i}", p.pos.y);
        }
    }
}
