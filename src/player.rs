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

/// A body counts as standing on ground within this far of its feet.
///
/// 12.5 mm: small enough that it cannot bridge a real gap (the smallest gap the
/// world can make is one voxel), large enough to survive the contact skin a
/// resting sweep leaves behind (`voxquery::CONTACT_SKIN`, 1e-3 voxel).
/// In METRES because it is a property of the BODY's contact with the floor, not
/// of the lattice: written as 0.05 voxels it was 12.5 mm at 25 cm and would have
/// become 5 mm at 10 cm, tightening ground contact by 2.5x for no reason.
const GROUND_TOL: f32 = m_to_vox(0.0125);

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
        // A body's own size plus 25 cm of slack, searched in 6.25 cm steps.
        // Both were voxel literals, so the slack and the search resolution both
        // shrank with the grid.
        let reach =
            self.dims.height(self.crouching).max(self.dims.half_width * 2.0) + m_to_vox(0.25);
        let step = m_to_vox(0.0625);
        let mut d = step;
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
            d += step;
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
        let eps = GROUND_TOL;
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
        // ~50 cm apart, rounded to an EVEN number of voxels so `+- SPAN/2` from
        // a column centre lands on another column centre. Both halves still
        // matter, and the rounding is why this is not simply `m_to_vox(0.5)`:
        // an odd or fractional span would put the samples between columns again.
        const SPAN: f32 = 2.0 * SPAN_HALF;
        const SPAN_HALF: f32 = {
            let v = m_to_vox(0.25);
            // round-to-nearest, at least one voxel.
            let r = (v + 0.5) as i32;
            if r < 1 { 1.0 } else { r as f32 }
        };
        const CORNERS: [(f32, f32); 4] = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)];
        let cx = self.pos.x.floor() + 0.5;
        let cz = self.pos.z.floor() + 0.5;
        let mut h = [f32::NAN; 4];
        for (i, (sx, sz)) in CORNERS.iter().enumerate() {
            let px = cx + sx * (SPAN * 0.5);
            let pz = cz + sz * (SPAN * 0.5);
            // HALF A VOXEL wide, deliberately in LATTICE units and not metres:
            // the probe exists to read ONE column, and a box sized in metres
            // would straddle two of them at a small enough voxel and silently
            // report the taller of the pair. The thickness is a sliver for the
            // same reason - it is a degenerate box, not a distance.
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
        // 6 cm ahead of the body's face, a 15 cm probe column, with 2.5 cm of
        // clearance top and bottom. Metres: these describe where a body's shin
        // meets a wall, which does not change when the grid does.
        let c = self.pos + dir * (hw + m_to_vox(0.0625));
        let lo = self.pos.y + self.tuning.step_max + m_to_vox(0.025);
        let hi = self.pos.y + h - m_to_vox(0.025);
        let r = m_to_vox(0.075);
        lo < hi
            && voxquery::overlaps(
                world,
                Aabb::new(Vec3::new(c.x - r, lo, c.z - r), Vec3::new(c.x + r, hi, c.z + r)),
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
        let ahead = self.pos + dir * (hw * 2.0 + m_to_vox(0.125));
        // A vault is a mantle, not a leap: the body must be AT the obstacle. A
        // ledge detected from a body-width away lands the body balanced on the
        // lip with its centre over the void, because the destination is measured
        // forward from wherever the vault happened to trigger.
        let face = self.pos + dir * (hw + m_to_vox(0.075));
        let cr = m_to_vox(0.0625);
        let contact = Aabb::new(
            Vec3::new(face.x - cr, self.pos.y + GROUND_TOL, face.z - cr),
            Vec3::new(face.x + cr, self.pos.y + t.vault_max, face.z + cr),
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
    use crate::voxel::{m_to_vox_i, MAT_STONE};
    use std::sync::OnceLock;

    const DT: f32 = PlayerSim::TICK_DT;

    // ---------------------------------------------------------------------
    // THE FIXTURES ARE MEASURED IN METRES
    //
    // Every extent, obstacle height, ledge, ramp, ceiling gap and pool below is
    // a REAL-WORLD size, converted to voxels only where it is used. They used to
    // be bare voxel counts, and the day `world_dims::VOXEL_METRES` went
    // 0.25 -> 0.10 (docs/SCALE_TO_10CM.md) every one of them silently shrank
    // 2.5x: the 1 m vault ledge became a 40 cm kerb, the 1.5 m crouch gap became
    // 60 cm, the 16 m ground patch became 6.4 m and the body walked straight off
    // the end of it. Seven feel tests failed for a reason that had nothing to do
    // with the feel - the controller's own tuning went through the same change
    // without a single number moving, because it goes through `m_to_vox`.
    //
    // So the rule for this module is the rule for `Tuning`: an integer voxel
    // count written directly is FORBIDDEN. `m_to_vox(0.35)` reads as 35 cm; 3.5
    // does not. The values here are the SAME obstacle course the 25 cm build
    // described, restated in the unit it was always measured in.
    // ---------------------------------------------------------------------

    /// Surface of every flat floor in both fixtures.
    ///
    /// This is the height the worldgen calls sea level (`voxel::SEA_LEVEL_M`,
    /// i.e. `voxel::SEA_LEVEL` as a voxel row), which is exactly where the old
    /// bare `GROUND_Y = 64.0` came from: at 25 cm, voxel row 64 WAS 16 m. Tying
    /// the two together keeps a body dropped on a fixture and a body dropped on
    /// generated terrain in the same part of the world's vertical range, and it
    /// means the fixture floor moves with sea level instead of drifting off it.
    const GROUND_M: f32 = crate::voxel::SEA_LEVEL_M;
    const GROUND_Y: f32 = m_to_vox(GROUND_M);
    /// Both fixture floors are a metre of stone. Only the top face is ever
    /// touched; the thickness is there so nothing can fall through.
    const FLOOR_THICK_M: f32 = 1.0;

    /// "The body ended a manoeuvre resting on that surface", in metres.
    ///
    /// NUMERICAL slop, not a physical size: a sweep leaves a `CONTACT_SKIN` gap
    /// (1e-3 voxel) and a settle loop stops a fraction of a tick short of
    /// equilibrium. It is stated in metres anyway so it cannot silently tighten
    /// or loosen by 2.5x with the voxel - 1.25 cm is what the old bare
    /// `0.05` voxels meant at 25 cm, and 2.5 mm what the old `0.01` did.
    const SETTLE_TOL_M: f32 = 0.0125;
    const REST_TOL_M: f32 = 0.0025;

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

    /// A world position from METRES.
    fn at_m(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3::new(m_to_vox(x), m_to_vox(y), m_to_vox(z))
    }

    /// A box FACE at `metres`, as a voxel plane.
    ///
    /// ROUNDS, where `voxel::m_to_vox_i` truncates. A face is a position, and
    /// the nearest plane is the honest answer when the grid cannot land on it:
    /// a 0.25 m wall is 2.5 voxels at 10 cm, and truncating builds a 0.20 m one
    /// while claiming 0.25. Rounding also makes the conversion immune to a
    /// metre value that lands a float epsilon under an exact plane.
    fn plane(metres: f32) -> i32 {
        m_to_vox(metres).round() as i32
    }

    /// The height the grid actually builds for a requested one. A fixture states
    /// intent in metres; the lattice quantises it, and a test must measure
    /// against what is THERE, because that is all the body can climb.
    fn built_height_m(want_m: f32) -> f32 {
        to_m((plane(GROUND_M + want_m) - plane(GROUND_M)) as f32)
    }

    /// [`fill`], in METRES. Every fixture goes through this so the size of the
    /// thing being built is visible at the call site.
    fn fill_m(w: &mut World, x: (f32, f32), y: (f32, f32), z: (f32, f32), mat: u8) {
        fill(
            w,
            (plane(x.0), plane(x.1)),
            (plane(y.0), plane(y.1)),
            (plane(z.0), plane(z.1)),
            mat,
        );
    }

    /// Side of `flat_world`'s square patch.
    const FLAT_SIDE_M: f32 = 16.0;
    /// Where every flat-ground test starts: the middle of the patch. A sprint
    /// held for 1.5 s and then released covers about 8 m, and half the patch is
    /// exactly that, so the body never measures a stopping distance against thin
    /// air. (It did once the patch was read as voxels: 6.4 m of ground, and
    /// `stopping_distance_from_walk_and_sprint` reported 30 m of "coasting"
    /// that was really a fall.)
    const FLAT_CENTRE_M: f32 = FLAT_SIDE_M / 2.0;
    /// Clearance a fixture body is dropped from. 12.5 cm was the old half a
    /// voxel at 25 cm: far enough that the first tick is a real settle, close
    /// enough that nothing accelerates on the way down.
    const DROP_M: f32 = 0.125;

    /// Ground plane 16 m square with its surface at [`GROUND_M`]. `World::new`
    /// allocates 1.84 GB and spawns worker threads, so every flat-ground feel
    /// test shares this one immutable world - which is also why
    /// `.cargo/config.toml` pins RUST_TEST_THREADS. Do not add a third static.
    fn flat_world() -> &'static World {
        static W: OnceLock<World> = OnceLock::new();
        W.get_or_init(|| {
            let mut w = World::new();
            fill_m(
                &mut w,
                (0.0, FLAT_SIDE_M),
                (GROUND_M - FLOOR_THICK_M, GROUND_M),
                (0.0, FLAT_SIDE_M),
                MAT_STONE,
            );
            w
        })
    }

    /// Feet position every flat-ground test spawns at.
    fn flat_spawn() -> Vec3 {
        at_m(FLAT_CENTRE_M, GROUND_M + DROP_M, FLAT_CENTRE_M)
    }

    /// A body standing still at the centre of `flat_world`'s patch, settled
    /// (grounded, zero velocity).
    fn standing(world: &World) -> Player {
        let mut p = Player::new(flat_spawn());
        for _ in 0..30 {
            p.step(world, Input::default(), DT);
        }
        assert!(p.grounded, "test body failed to settle on the ground");
        assert!(
            (to_m(p.pos.y) - GROUND_M).abs() < REST_TOL_M,
            "settled at {:.4} m not {GROUND_M} m",
            to_m(p.pos.y),
        );
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
            let mut p = standing(w);
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
            let mut p = standing(w);
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
        let mut p = standing(w);
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
        assert!(
            p.grounded && (to_m(p.pos.y) - GROUND_M).abs() < REST_TOL_M,
            "must land back on the floor",
        );
    }

    #[test]
    fn terminal_velocity_is_reached_and_clamped() {
        // No ground at all: the body falls out of the loaded window and keeps
        // falling, which is the only way to see terminal velocity in a world
        // that is 64 m tall.
        let empty = World::new();
        // 25 m up in an empty world. Nothing here depends on the height (there
        // is no floor to hit), but it is still a place, so it is still metres.
        let mut p = Player::new(at_m(FLAT_CENTRE_M, 25.0, FLAT_CENTRE_M));
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
        let mut p = standing(w);
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
            let mut sim = PlayerSim::new(flat_spawn());
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
            // One tick of walking is walk_speed * TICK_DT = 5 cm, which is half
            // a voxel at 10 cm and was a fifth of one at 25 cm - the slack is
            // derived from the tuning, so it follows the scale on its own.
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
        let mut sim = PlayerSim::new(flat_spawn());
        sim.advance(w, Input::default(), 0.5);
        let before = sim.player.pos;
        sim.advance(w, forward(true), 5.0);
        let moved = to_m(plan_dist(before, sim.player.pos));
        assert!(moved < 1.0, "a 5 s hitch moved the body {moved:.2} m in one frame");
    }

    // ------------------------------------------------- feel: terrain contact

    // ---- the obstacle course, in metres ----
    //
    // A 44 x 36 m yard with every geometry test's site in its own patch of z, so
    // one shared world serves them all. The numbers are the 25 cm fixture's
    // voxel counts x 0.25, i.e. the sizes it always meant.

    const COURSE_X_M: f32 = 44.0;
    const COURSE_Z_M: f32 = 36.0;

    /// The step-assist ladder: six walls, each 25 cm taller than the last.
    ///
    /// The point of the ladder is that it BRACKETS `Tuning::step_max` (0.35 m)
    /// from both sides - it has to prove what a stride takes and what it
    /// refuses, and one wall on each side of the limit does that. The grid
    /// quantises the request (0.25 m is 2.5 voxels at 10 cm, so the first wall
    /// is really 0.30 m; it was exactly 0.25 m at 25 cm), which is why the test
    /// measures `built_height_m` rather than the request. The bracket survives
    /// the quantisation at both scales: 0.30 under, 0.50 over.
    const STEP_LADDER_M: [f32; 6] = [0.25, 0.50, 0.75, 1.00, 1.25, 1.50];
    /// Each wall gets its own 2 m band of z, 2.5 m apart, so a body walking at
    /// one can never see another.
    const STEP_WALL_PITCH_M: f32 = 2.5;
    const STEP_WALL_DEPTH_M: f32 = 2.0;
    const STEP_WALL_X_M: (f32, f32) = (7.5, 9.0);
    /// Bodies start 1 m short of the wall. At 12 m/s^2 the body needs 0.375 m to
    /// reach walk speed, so it arrives at the wall at full pace.
    const STEP_APPROACH_X_M: f32 = 6.5;

    fn step_wall_z0_m(n: usize) -> f32 {
        STEP_WALL_PITCH_M * (n + 1) as f32
    }

    /// Three ramps over the same 7.5 m of run, written as RISE OVER RUN.
    ///
    /// A slope is a ratio, so it is stored as one: 1.0 is 45 degrees at any
    /// voxel size, where the old `64 + i` was 45 degrees only while a voxel was
    /// as wide as it was tall in that particular loop. Each ramp is a staircase
    /// of ONE-VOXEL treads whose riser follows the ratio, which is what the 25 cm
    /// fixture built too, and it is also what keeps `Player::probe_ground`
    /// honest: that samples four column centres exactly two voxels apart, so a
    /// tread wider than two voxels would read as flat on the tread and as a
    /// cliff at the riser, and `steep` would flicker instead of measuring the
    /// slope.
    ///
    /// WHAT CHANGED AT 10 CM: the risers are now 0.1 / 0.2 / 0.1 m where they
    /// were 0.25 / 0.5 / 0.25 m. On the 2:1 face that used to put the riser over
    /// `step_max` (0.5 > 0.35) as well as the face over `slope_limit_tan`, so
    /// two independent mechanisms refused it; now only the slope limit does.
    /// That is the mechanism the test is named after, so the test got stricter,
    /// not weaker - it can no longer pass by accident.
    const RAMP_RUN_M: f32 = 7.5;
    const RAMP_45: f32 = 1.0; // 45.0 degrees
    const RAMP_63: f32 = 2.0; // 63.4 degrees
    const RAMP_26: f32 = 0.5; // 26.6 degrees
    const RAMP_45_X0_M: f32 = 15.0;
    const RAMP_63_X0_M: f32 = 25.0;
    const RAMP_26_X0_M: f32 = 15.0;
    const RAMP_45_Z_M: (f32, f32) = (2.5, 5.0);
    const RAMP_63_Z_M: (f32, f32) = (5.0, 7.5);
    const RAMP_26_Z_M: (f32, f32) = (7.5, 10.0);

    /// A staircase ramp rising `slope` metres per metre of run, one voxel column
    /// per tread.
    fn ramp(w: &mut World, x0_m: f32, slope: f32, z_m: (f32, f32)) {
        let x0 = plane(x0_m);
        let base = GROUND_Y as i32;
        let (z0, z1) = (plane(z_m.0), plane(z_m.1));
        for i in 0..m_to_vox_i(RAMP_RUN_M) {
            let rise = (i as f32 * slope) as i32;
            fill(w, (x0 + i, x0 + i + 1), (base, base + rise), (z0, z1), MAT_STONE);
        }
    }

    /// Feet height, in metres, of a body standing at world x `x_m` on the ramp
    /// starting at `x0_m`. The body rests on the highest voxel column its
    /// footprint covers, which walking uphill in +x is the one at its leading
    /// face; the continuous surface there is at most one riser above the
    /// staircase, so a body dropped at this height lands ON the ramp, never in
    /// it. (This is what the old hand-computed `86.5` was, and it is derived now
    /// so it moves with the ramp instead of having to be re-derived.)
    fn ramp_stand_m(x0_m: f32, slope: f32, x_m: f32) -> f32 {
        GROUND_M + (x_m + to_m(Dims::default().half_width) - x0_m).max(0.0) * slope
    }

    /// A slab leaving 1.5 m of headroom: under the 1.8 m standing body, over the
    /// 1.2 m crouched one.
    const CEIL_GAP_M: f32 = 1.5;
    const SLAB_THICK_M: f32 = 1.0;
    const SLAB_X_M: (f32, f32) = (5.0, 7.5);
    const SLAB_Z_M: (f32, f32) = (20.0, 22.5);
    /// 1.25 m clear of the slab: enough run-up to be crouch-walking at speed.
    const SLAB_APPROACH_Z_M: f32 = 23.75;
    const SLAB_X_MID_M: f32 = 6.25;

    /// A 1 m ledge - over `step_max` (0.35 m), under `vault_max` (1.30 m).
    const LEDGE_H_M: f32 = 1.0;
    const LEDGE_X_M: (f32, f32) = (32.5, 35.0);
    const LEDGE_Z_M: (f32, f32) = (2.5, 5.0);
    /// A 5 m wall - far past `vault_max`, so it can only be climbed.
    const WALL_H_M: f32 = 5.0;
    const WALL_X_M: (f32, f32) = (32.5, 35.0);
    const WALL_Z_M: (f32, f32) = (7.5, 10.0);
    /// Bodies start 1 m short of both.
    const FACE_APPROACH_X_M: f32 = 31.5;

    /// A wall exactly ONE VOXEL thick, whatever a voxel is - the thinnest thing
    /// the world can express, and therefore the hardest thing not to tunnel
    /// through. This is the one extent in the course that is deliberately NOT a
    /// real-world size: at 10 cm it is 10 cm of stone against a body crossing
    /// 12.5 m in one tick, which is a strictly harder test than the 25 cm it was.
    const THIN_X0_M: f32 = 37.5;
    const THIN_H_M: f32 = 3.0;
    const THIN_Z_M: (f32, f32) = (12.5, 15.0);
    const THIN_APPROACH_X_M: f32 = 27.5;
    const THIN_BEHIND_X_M: f32 = 42.5;
    const THIN_Z_MID_M: f32 = 13.75;

    /// Water and leaves floating over open floor: neither is a floor.
    const POOL_X_M: (f32, f32) = (5.0, 7.5);
    const POOL_Z_M: (f32, f32) = (30.0, 32.5);
    const WATER_Y_M: (f32, f32) = (20.0, 21.0);
    const LEAVES_Y_M: (f32, f32) = (22.0, 23.0);
    const POOL_MID_M: (f32, f32) = (6.25, 31.25);

    /// One shared obstacle course, laid out in separate sites so every geometry
    /// test can use the same 1.84 GB world. Floor surface is [`GROUND_M`]
    /// throughout. Second and last static world in this module: see
    /// `.cargo/config.toml` for why there must not be a third.
    ///
    ///   x  7.5.. 9.0, z  2.5..17.0  six walls 0.25..1.50 m (step-assist ladder)
    ///   x 15.0..22.5, z  2.5.. 5.0  a 1:1 staircase - 45 degrees
    ///   x 25.0..32.5, z  5.0.. 7.5  a 2:1 staircase - 63.4 degrees
    ///   x 15.0..22.5, z  7.5..10.0  a 1:2 staircase - 26.6 degrees
    ///   x  5.0.. 7.5, z 20.0..22.5  a slab 1.5 m up: crouch-only headroom
    ///   x 32.5..35.0, z  2.5.. 5.0  a 1.0 m ledge (vault band)
    ///   x 32.5..35.0, z  7.5..10.0  a 5 m wall (climb band)
    ///   x 37.5,       z 12.5..15.0  a ONE VOXEL thick wall (tunnelling)
    ///   x  5.0.. 7.5, z 30.0..32.5  water at 20 m, leaves at 22 m
    fn course() -> &'static World {
        static W: OnceLock<World> = OnceLock::new();
        W.get_or_init(|| {
            let mut w = World::new();
            let floor = (GROUND_M - FLOOR_THICK_M, GROUND_M);
            fill_m(&mut w, (0.0, COURSE_X_M), floor, (0.0, COURSE_Z_M), MAT_STONE);
            for (n, h) in STEP_LADDER_M.iter().copied().enumerate() {
                let z0 = step_wall_z0_m(n);
                fill_m(
                    &mut w,
                    STEP_WALL_X_M,
                    (GROUND_M, GROUND_M + h),
                    (z0, z0 + STEP_WALL_DEPTH_M),
                    MAT_STONE,
                );
            }
            ramp(&mut w, RAMP_45_X0_M, RAMP_45, RAMP_45_Z_M);
            ramp(&mut w, RAMP_63_X0_M, RAMP_63, RAMP_63_Z_M);
            ramp(&mut w, RAMP_26_X0_M, RAMP_26, RAMP_26_Z_M);
            fill_m(
                &mut w,
                SLAB_X_M,
                (GROUND_M + CEIL_GAP_M, GROUND_M + CEIL_GAP_M + SLAB_THICK_M),
                SLAB_Z_M,
                MAT_STONE,
            );
            fill_m(&mut w, LEDGE_X_M, (GROUND_M, GROUND_M + LEDGE_H_M), LEDGE_Z_M, MAT_STONE);
            fill_m(&mut w, WALL_X_M, (GROUND_M, GROUND_M + WALL_H_M), WALL_Z_M, MAT_STONE);
            let thin_x0 = plane(THIN_X0_M);
            fill(
                &mut w,
                (thin_x0, thin_x0 + 1),
                (GROUND_Y as i32, plane(GROUND_M + THIN_H_M)),
                (plane(THIN_Z_M.0), plane(THIN_Z_M.1)),
                MAT_STONE,
            );
            fill_m(&mut w, POOL_X_M, WATER_Y_M, POOL_Z_M, crate::voxel::MAT_WATER_L8);
            fill_m(&mut w, POOL_X_M, LEAVES_Y_M, POOL_Z_M, crate::voxel::MAT_LEAVES);
            w
        })
    }

    /// Drop a body at (x, z) from [`DROP_M`] above `y`, all in METRES, and let
    /// it settle onto whatever is there.
    fn settle_m(world: &World, x: f32, y: f32, z: f32) -> Player {
        let mut p = Player::new(at_m(x, y + DROP_M, z));
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
        // The comparison is in METRES on both sides: a wall's height and
        // step_max are both real-world lengths, and comparing raw voxel counts
        // was only ever right by coincidence of the scale they were written at.
        let step_max_m = to_m(t.step_max);
        let wall_face_x = m_to_vox(STEP_WALL_X_M.0);
        let mut highest_climbed_m = 0.0f32;
        for (n, want_m) in STEP_LADDER_M.iter().copied().enumerate() {
            // What the lattice built, not what was asked for: 0.25 m is 2.5
            // voxels at 10 cm, so the wall the body meets is 0.30 m. The ladder
            // still brackets step_max, which is the property that matters.
            let wall_m = built_height_m(want_m);
            let z = step_wall_z0_m(n) + STEP_WALL_DEPTH_M * 0.5;
            let mut p = settle_m(w, STEP_APPROACH_X_M, GROUND_M, z);
            assert!(p.grounded, "the {wall_m} m wall: body did not settle");
            let start_y = p.pos.y;
            hold(&mut p, w, walk_x(false, false), 1.0);
            let climbed_m = to_m(p.pos.y - start_y);
            if wall_m <= step_max_m {
                assert!(
                    (climbed_m - wall_m).abs() < SETTLE_TOL_M && p.pos.x > wall_face_x,
                    "a {wall_m:.2} m step (under the {step_max_m:.2} m step height) must be walked \
                     up; climbed {climbed_m:.3} m, ended at x {:.2} m",
                    to_m(p.pos.x),
                );
                highest_climbed_m = highest_climbed_m.max(wall_m);
            } else {
                assert!(
                    climbed_m < SETTLE_TOL_M && p.pos.x < wall_face_x,
                    "a {wall_m:.2} m step (over the {step_max_m:.2} m step height) must NOT be \
                     walked up; climbed {climbed_m:.3} m, ended at x {:.2} m",
                    to_m(p.pos.x),
                );
            }
        }
        band(
            "tallest step walked up without a vault",
            highest_climbed_m,
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
        // 26.6 degrees (1 up per 2 across) and 45 degrees (1:1) are both
        // ascended: 45 is where Unreal (44.8) and Source (45.6) put their
        // walkable floor, and on a voxel lattice it is also the steepest thing
        // step assist can reach.
        //
        // Each body starts 0.5 m short of its ramp's toe, on flat floor, in the
        // middle of the ramp's z band. The expected gains are the metres the
        // ramp actually offers inside the time budget (2 m of the 3.75 m the 1:2
        // climbs, 4 m of the 7.5 m the 1:1 does), not a voxel count that meant
        // those metres once.
        for (name, x0_m, ramp_z, expect_gain_m) in [
            ("26.6 degree", RAMP_26_X0_M, RAMP_26_Z_M, 2.0f32),
            ("45 degree", RAMP_45_X0_M, RAMP_45_Z_M, 4.0),
        ] {
            let z_m = (ramp_z.0 + ramp_z.1) * 0.5;
            let mut p = settle_m(w, x0_m - 0.5, GROUND_M, z_m);
            let y0 = p.pos.y;
            // Peak, not final: these ramps are 7.5 m long and a body at 3 m/s
            // crests them and walks off the far end within 4 s.
            let mut peak = y0;
            let mut ever_steep = false;
            for _ in 0..(4.0 / DT) as i32 {
                p.step(w, walk_x(false, false), DT);
                peak = peak.max(p.pos.y);
                ever_steep |= p.steep;
            }
            let gained_m = to_m(peak - y0);
            assert!(
                gained_m >= expect_gain_m && !ever_steep,
                "{name} slope: climbed {gained_m:.2} m (wanted >= {expect_gain_m} m), \
                 ever steep: {ever_steep}",
            );
        }
        // 63.4 degrees (2:1) is past the limit: no footing.
        //
        // These three runs are dropped onto the face and measured FROM FIRST
        // CONTACT, where they used to be pre-settled for 40 ticks. A body cannot
        // settle on a face it slides off, and `settle_m`'s 40 ticks are only the
        // right preparation for ground it can stand on. At 25 cm the body was
        // still stepping down the 0.5 m risers after 40 ticks and happened to be
        // in contact; the same physical face at 10 cm is a 0.2 m riser
        // staircase, smooth enough that a body sliding at ~1 m/s skips down it
        // ballistically instead - which IS what losing your footing on a cliff
        // looks like, but it means `steep` is only observable while the body is
        // actually touching. So the observation now starts where the contact
        // does, and asserts the same three things about the same face.
        let face_x_m = RAMP_63_X0_M + 2.5;
        let face_z_m = 6.25;
        let face_y_m = ramp_stand_m(RAMP_63_X0_M, RAMP_63, face_x_m);
        let on_face = |w: &World| {
            let mut p = Player::new(at_m(face_x_m, face_y_m + DROP_M, face_z_m));
            for _ in 0..40 {
                p.step(w, Input::default(), DT);
                if p.grounded {
                    return p;
                }
            }
            panic!("the body never reached the 2:1 face it was dropped on");
        };
        let mut p = on_face(w);
        let mut ever_steep = false;
        for _ in 0..30 {
            p.step(w, Input::default(), DT);
            ever_steep |= p.steep;
        }
        assert!(ever_steep, "the 2:1 face must read as steep while the body is on it");
        // Walking at it gains nothing: no grip, and step assist is off.
        let mut p = on_face(w);
        let y0 = p.pos.y;
        let mut peak = y0;
        for _ in 0..(2.0 / DT) as i32 {
            p.step(w, walk_x(false, false), DT);
            peak = peak.max(p.pos.y);
        }
        assert!(
            to_m(peak - y0) <= SETTLE_TOL_M,
            "a 63-degree face must not be climbable, gained {:.3} m",
            to_m(peak - y0),
        );
        // ... and released, the body slides DOWN it (downhill is -x here).
        let mut p = on_face(w);
        let (x0, y0) = (p.pos.x, p.pos.y);
        hold(&mut p, w, Input::default(), 1.5);
        assert!(
            to_m(x0 - p.pos.x) > 0.25 && to_m(y0 - p.pos.y) > 0.25,
            "a body must slide off a 63-degree face: moved ({:.2}, {:.2}) m",
            to_m(p.pos.x - x0),
            to_m(p.pos.y - y0),
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
        let start = |w: &World| settle_m(w, SLAB_X_MID_M, GROUND_M, SLAB_APPROACH_Z_M);
        let mut p = start(w);
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
        let mut moving = start(w);
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
        let mut under = start(w);
        hold(&mut under, w, Input { move_axis: Vec2::new(0.0, -1.0), crouch: true, ..Default::default() }, 2.0);
        assert!(
            to_m(under.pos.z) < SLAB_Z_M.1,
            "the body should have crouch-walked under the slab, z={:.2} m",
            to_m(under.pos.z),
        );
        hold(&mut under, w, Input::default(), 0.5);
        assert!(under.crouching, "stood up into a ceiling");
        assert!(
            !voxquery::overlaps(w, under.body(), MatSet::SOLID),
            "the body ended up inside the ceiling",
        );
        // Walking back out, it stands up again on its own.
        hold(&mut under, w, Input { move_axis: Vec2::new(0.0, 1.0), ..Default::default() }, 2.0);
        assert!(!under.crouching, "never stood back up in the open, z={:.2} m", to_m(under.pos.z));
    }

    #[test]
    fn vaulting_is_deliberate_and_clears_what_a_stride_cannot() {
        let w = course();
        let t = Tuning::default();
        // The ledge must sit in the vault band: over step_max (0.35 m), under
        // vault_max (1.30 m). Both sides of that comparison are METRES now - the
        // old `4.0` was 1 m of ledge only because a voxel happened to be 25 cm.
        let ledge_top_m = GROUND_M + LEDGE_H_M;
        let ledge_face_x = m_to_vox(LEDGE_X_M.0);
        let ledge_z_m = (LEDGE_Z_M.0 + LEDGE_Z_M.1) * 0.5;
        assert!(
            LEDGE_H_M > to_m(t.step_max) && LEDGE_H_M <= to_m(t.vault_max),
            "the test ledge must sit in the vault band",
        );

        // Walking into it without the deliberate press gets nowhere.
        let mut p = settle_m(w, FACE_APPROACH_X_M, GROUND_M, ledge_z_m);
        hold(&mut p, w, walk_x(false, false), 2.0);
        assert!(
            to_m(p.pos.y) < GROUND_M + 0.25 && p.pos.x < ledge_face_x,
            "a {LEDGE_H_M} m ledge must not be walked up: ended at ({:.2}, {:.2}) m",
            to_m(p.pos.x),
            to_m(p.pos.y),
        );

        // Pressed into it with jump, it is mantled.
        let mut p = settle_m(w, FACE_APPROACH_X_M, GROUND_M, ledge_z_m);
        let ev = {
            let mut acc = Events::default();
            for _ in 0..(2.0 / DT) as i32 {
                acc = acc.merge(p.step(w, walk_x(false, true), DT));
            }
            acc
        };
        assert!(ev.vault_started, "a deliberate press into a {LEDGE_H_M} m ledge must start a vault");
        assert!(
            (to_m(p.pos.y) - ledge_top_m).abs() < SETTLE_TOL_M && p.pos.x > ledge_face_x,
            "the vault must end standing on the ledge: ({:.2}, {:.2}) m",
            to_m(p.pos.x),
            to_m(p.pos.y),
        );
        assert!(p.grounded && !voxquery::overlaps(w, p.body(), MatSet::SOLID));
        // The mantle must report most of the ledge it gained (the same
        // three-quarters the old bare `3.0` voxels meant against a 4-voxel ledge).
        assert!(
            to_m(ev.climbed) > LEDGE_H_M * 0.75,
            "the vault should report the height it gained, got {:.2} m",
            to_m(ev.climbed),
        );
    }

    #[test]
    fn climbing_is_held_and_tops_out_on_the_ledge() {
        let w = course();
        // The 5 m wall is far past vault_max, so it can only be climbed.
        let into_wall = walk_x(false, true);
        let wall_top_m = GROUND_M + WALL_H_M;
        let wall_face_x = m_to_vox(WALL_X_M.0);
        let wall_z_m = (WALL_Z_M.0 + WALL_Z_M.1) * 0.5;
        assert!(
            WALL_H_M > to_m(Tuning::default().vault_max),
            "the test wall must be past the vault band",
        );
        let mut p = settle_m(w, FACE_APPROACH_X_M, GROUND_M, wall_z_m);
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
        assert!(
            to_m(y1 - p.pos.y) > 0.125,
            "releasing the climb must let the body fall, dropped {:.3} m",
            to_m(y1 - p.pos.y),
        );

        // Held all the way, it tops out onto the wall rather than sticking to it.
        let mut p = settle_m(w, FACE_APPROACH_X_M, GROUND_M, wall_z_m);
        let mut topped = false;
        for _ in 0..(8.0 / DT) as i32 {
            p.step(w, into_wall, DT);
            // Grounded within a metre of the top is "on the wall", not "back on
            // the floor 5 m below".
            if p.grounded && to_m(p.pos.y) > wall_top_m - 1.0 {
                topped = true;
                break;
            }
        }
        assert!(topped, "the climb never reached the top of the wall, stalled at {:?}", p.pos);
        assert!(
            (to_m(p.pos.y) - wall_top_m).abs() < 0.05 && p.pos.x > wall_face_x,
            "a sustained climb must top out onto the ledge, ended at ({:.2}, {:.2}) m",
            to_m(p.pos.x),
            to_m(p.pos.y),
        );
    }

    #[test]
    fn the_body_cannot_tunnel_through_a_one_voxel_wall() {
        // 750 m/s in a single tick - 12.5 m of travel against a wall ONE VOXEL
        // thick. A move-then-test controller teleports straight through. The
        // speed is a real-world one, so the tick still covers 12.5 m at 10 cm
        // (125 voxels) as it did at 25 cm (50), while the wall it has to notice
        // is now 2.5x thinner: strictly the harder test.
        let w = course();
        let bullet = m_to_vox(750.0);
        let near_face = m_to_vox(THIN_X0_M);
        // The far face of a ONE VOXEL wall, which is the one quantity here that
        // is honestly a voxel count rather than a length.
        let far_face = (plane(THIN_X0_M) + 1) as f32;
        let mut p = settle_m(w, THIN_APPROACH_X_M, GROUND_M, THIN_Z_MID_M);
        p.vel.x = bullet;
        p.step(w, Input::default(), DT);
        assert!(
            p.pos.x + p.dims.half_width <= near_face,
            "tunnelled to x {:.2} m through the wall at x {THIN_X0_M} m",
            to_m(p.pos.x),
        );
        assert!(!voxquery::overlaps(w, p.body(), MatSet::SOLID));
        // And the same from the far side.
        let mut p = settle_m(w, THIN_BEHIND_X_M, GROUND_M, THIN_Z_MID_M);
        p.vel.x = -bullet;
        p.step(w, Input::default(), DT);
        assert!(
            p.pos.x - p.dims.half_width >= far_face,
            "tunnelled backwards to x {:.2} m",
            to_m(p.pos.x),
        );
    }

    #[test]
    fn water_and_foliage_are_not_floors() {
        // The material classification has to survive the whole controller, not
        // just the query: a body falls through water and through leaves, and
        // lands on the stone below.
        let w = course();
        // Dropped from 0.75 m above the leaf layer, so the fall crosses leaves
        // then water then 4 m of open air before the floor.
        let mut p = Player::new(at_m(POOL_MID_M.0, LEAVES_Y_M.1 + 0.75, POOL_MID_M.1));
        for _ in 0..(4.0 / DT) as i32 {
            p.step(w, Input::default(), DT);
            if p.grounded {
                break;
            }
        }
        assert!(
            p.grounded && (to_m(p.pos.y) - GROUND_M).abs() < SETTLE_TOL_M,
            "the body should have fallen through leaves and water to the ground, stopped at \
             {:.2} m",
            to_m(p.pos.y),
        );
    }

    #[test]
    fn a_body_inside_solid_matter_pushes_itself_out() {
        // Placing a block on your own feet, sand falling into you, or a chunk
        // streaming in around you: the body must not be stuck forever.
        let w = course();
        // A quarter of the way into the 1 m ledge, horizontally in the middle of
        // it: buried on every side except up.
        let mut p = Player::new(at_m(
            (LEDGE_X_M.0 + LEDGE_X_M.1) * 0.5,
            GROUND_M + LEDGE_H_M * 0.25,
            (LEDGE_Z_M.0 + LEDGE_Z_M.1) * 0.5,
        ));
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
        // 64 m in from the corner of the window. Worldgen is a function of
        // METRES (see `voxel::climate_at`, which multiplies by VOXEL_METRES), so
        // this is literally the same landscape the 25 cm build sampled at voxel
        // 256 - the terrain the body is asked to survive did not change, only
        // how finely it is stored. Terrain height comes back as a voxel row and
        // is already sea-level-relative through `voxel::SEA_LEVEL`, so it is
        // used as it comes; only the spawn clearance above it is a length.
        for (i, yaw) in (0..8).map(|i| (i, i as f32 * std::f32::consts::FRAC_PI_4)) {
            let (x, z) = (m_to_vox(64.0), m_to_vox(64.0));
            let s = crate::voxel::sample_terrain(x, z, seed);
            let ground = s.h.max(s.water_top) as f32;
            let mut p = Player::new(Vec3::new(x, ground + m_to_vox(0.5), z));
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
        let mut p = standing(w);
        let y = p.pos.y;
        for i in 0..600 {
            p.step(w, Input::default(), DT);
            // A voxel tolerance on purpose: a body at rest must not move AT ALL,
            // so this is float noise on a subtraction, not a physical distance
            // that should scale with the world.
            assert!((p.pos.y - y).abs() < 1.0e-4, "drifted to {} from {y} by tick {i}", p.pos.y);
            assert!(p.grounded, "lost the ground on tick {i}");
        }
    }
}
