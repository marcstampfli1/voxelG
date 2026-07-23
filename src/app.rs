// Client application: window creation, input handling and the per-frame
// orchestration loop. Split out of main.rs so the frame loop, the server loop
// and process startup each live in their own module (checklist: hygiene).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use glam::Vec3;

use crate::camera::Camera;
use crate::net;
use crate::physics;
use crate::raycast;
use crate::renderer::Renderer;
use crate::temporal;
use crate::leaffall;
use crate::voxel::{
    self, World, MAT_STONE, MAT_SAND, MAT_WATER, MAT_WOOD, MAT_LEAVES, MAT_GLASS,
    MAT_LAVA, MAT_ICE, MAT_SNOW, MAT_SMOKE,
};

/// Upper bound on frame rate. The loop sleeps between frames (ControlFlow::
/// WaitUntil) instead of busy-spinning, so an idle scene no longer pins the GPU
/// at ~1800 fps; presentation is additionally vsync-paced by the swapchain.
const FRAME_CAP_HZ: f64 = 144.0;

/// Recenter the streaming window only once the camera has drifted at least this
/// many chunks off-centre. A deadband stops the window thrashing (and
/// regenerating an edge column) when the player walks back and forth across a
/// single chunk boundary (checklist: prefetch with hysteresis).
const STREAM_HYSTERESIS: i32 = 2;

/// Max finished chunks installed per frame. Caps how many bricks get marked
/// dirty (and uploaded) per frame so a chunk cross streams in over a handful of
/// frames instead of one big hitch (checklist: per-frame upload budget).
const CHUNK_INSTALL_BUDGET: u32 = 6;

/// On an idle camera, a 1/N slice of the screen's tiles is re-traced each frame
/// so animated materials (sky, water, foliage) keep moving. N frames = one full
/// refresh; spread evenly so there's no periodic full-frame stutter.
const ANIM_REFRESH_SPREAD: usize = 8;

/// 8-tap sub-pixel jitter pattern (Halton(2,3), centred to [-0.5, 0.5]) for
/// temporal anti-aliasing. Applied only while the camera is static so the
/// accumulation converges to an anti-aliased image.
/// One scripted benchmark segment: a fixed pose (plus optional +x strafe),
/// held for `dur` seconds; the first `warmup` seconds are discarded (pipeline
/// warm, tiles converging, TAA settling).
struct BenchSeg {
    name: &'static str,
    pos: Vec3,
    yaw: f32,
    pitch: f32,
    strafe: f32,
    dur: f32,
    warmup: f32,
}

pub(crate) struct BenchState {
    segments: Vec<BenchSeg>,
    idx: usize,
    seg_start: Instant,
    dts: Vec<f32>,
    results: Vec<String>,
    // Per-segment GPU-pass attribution, fed from the renderer's profiler
    // samples (1 in 64 dirty frames), deduped by its report counter. The
    // residual (segment frame ms - gpu total) exposes CPU/present cost.
    gpu_acc: [f64; 9],
    gpu_total: f64,
    gpu_n: u32,
    last_report_seen: u64,
}

impl BenchState {
    /// Segment poses use the standard demo-world anchors (512^2 island,
    /// deterministic generator): the pond at (468, 92) with floor 57 /
    /// surface 64, the leaf-heavy cell at (16, 272) with ground y 90.
    fn from_env() -> Option<Self> {
        if std::env::var("VOXELG_BENCH").is_err() {
            return None;
        }
        // Power-state guard: battery power caps CPU/GPU clocks and silently
        // poisons every number (a battery run drifted water-free control
        // segments by 35%). Warn loudly; results from a battery run must be
        // discarded.
        // On battery only if NO adapter reports online (a machine can have
        // several ports; an unplugged one must not raise a false alarm).
        if let Ok(entries) = std::fs::read_dir("/sys/class/power_supply") {
            let mut saw_adapter = false;
            let mut any_online = false;
            for e in entries.flatten() {
                if let Ok(s) = std::fs::read_to_string(e.path().join("online")) {
                    saw_adapter = true;
                    any_online |= s.trim() == "1";
                }
            }
            if saw_adapter && !any_online {
                println!("bench WARNING: running on BATTERY - results are invalid, plug in AC");
            }
        }
        // Stall watchdog on a detached thread: an unfocused Wayland window
        // can stop receiving frame callbacks, freezing the bench clock with
        // the window alive forever (seen live). The thread is immune to the
        // event loop and force-exits well past the scripted duration.
        std::thread::spawn(|| {
            // Slack covers a cold driver shader-compile (~40 s) on a fresh
            // binary plus the 30 s script.
            std::thread::sleep(std::time::Duration::from_secs(140));
            println!("bench WATCHDOG: exceeded scripted duration + slack, forcing exit");
            unsafe { libc::_exit(3) };
        });
        let segs = vec![
            BenchSeg { name: "water_mid", pos: Vec3::new(464.5, 71.0, 96.0), yaw: 0.0, pitch: -0.22, strafe: 0.0, dur: 6.0, warmup: 2.0 },
            BenchSeg { name: "water_grazing", pos: Vec3::new(468.5, 65.4, 78.0), yaw: 0.0, pitch: -0.06, strafe: 0.0, dur: 6.0, warmup: 2.0 },
            BenchSeg { name: "water_strafe", pos: Vec3::new(452.0, 70.0, 92.0), yaw: 0.6, pitch: -0.30, strafe: 12.0, dur: 6.0, warmup: 2.0 },
            BenchSeg { name: "terrain", pos: Vec3::new(256.0, 140.0, 96.0), yaw: 0.0, pitch: -0.55, strafe: 0.0, dur: 6.0, warmup: 2.0 },
            BenchSeg { name: "foliage", pos: Vec3::new(48.5, 104.0, 244.0), yaw: 0.0, pitch: -0.35, strafe: 0.0, dur: 6.0, warmup: 2.0 },
            // Low over the flat treeless grass patch (the flora overhaul's
            // standing gate: near-tier grass renders here at full density).
            BenchSeg { name: "meadow", pos: Vec3::new(280.5, 74.0, 400.0), yaw: 0.0, pitch: -0.45, strafe: 0.0, dur: 6.0, warmup: 2.0 },
        ];
        Some(Self {
            segments: segs, idx: 0, seg_start: Instant::now(),
            dts: Vec::with_capacity(4096), results: Vec::new(),
            gpu_acc: [0.0; 9], gpu_total: 0.0, gpu_n: 0, last_report_seen: 0,
        })
    }

    /// Drive the camera for this frame; returns false when the bench is done.
    fn step(&mut self, camera: &mut crate::camera::Camera, dt: f32) -> bool {
        let now = Instant::now();
        let seg = &self.segments[self.idx];
        let elapsed = (now - self.seg_start).as_secs_f32();
        camera.pos = seg.pos + Vec3::new(seg.strafe * elapsed, 0.0, 0.0);
        camera.yaw = seg.yaw;
        camera.pitch = seg.pitch;
        if elapsed > seg.warmup {
            self.dts.push(dt);
        }
        if elapsed >= seg.dur {
            let mut d = std::mem::take(&mut self.dts);
            if d.len() > 4 {
                d.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mean = d.iter().sum::<f32>() / d.len() as f32;
                let p = |q: f32| d[((d.len() - 1) as f32 * q) as usize];
                self.results.push(format!(
                    "bench [{}]: avg {:.0} fps ({:.2} ms)  p1 {:.0}  p99 {:.0}  frames {}",
                    seg.name, 1.0 / mean, mean * 1000.0, 1.0 / p(0.99), 1.0 / p(0.01), d.len()
                ));
                if self.gpu_n > 0 {
                    let n = self.gpu_n as f64;
                    let mut line = format!("  gpu [{}]:", seg.name);
                    for (i, l) in crate::renderer::GPU_PROFILE_LABELS.iter().enumerate() {
                        line.push_str(&format!(" {l} {:.2}", self.gpu_acc[i] / n));
                    }
                    let gt = self.gpu_total / n;
                    line.push_str(&format!("  | gpu {gt:.2}  residual {:.2}  (samples {})",
                        (mean as f64) * 1000.0 - gt, self.gpu_n));
                    self.results.push(line);
                }
            }
            self.gpu_acc = [0.0; 9];
            self.gpu_total = 0.0;
            self.gpu_n = 0;
            self.idx += 1;
            self.seg_start = now;
            if self.idx >= self.segments.len() {
                println!("=== VOXELG_BENCH results (demo world, frozen sun t=30, uncapped) ===");
                for r in &self.results {
                    println!("{r}");
                }
                return false;
            }
        }
        true
    }
}

pub(crate) const JITTER_PATTERN: [[f32; 2]; 8] = [
    [0.0, -0.166_666_7],
    [-0.25, 0.166_666_7],
    [0.25, -0.388_888_9],
    [-0.375, -0.055_555_6],
    [0.125, 0.277_777_8],
    [-0.125, -0.277_777_8],
    [0.375, 0.055_555_6],
    [-0.062_5, 0.388_888_9],
];

/// Expand a sphere-of-impact into the world (and its persistent edit log). Free
/// function so it can run while the world Mutex guard is held (no &mut self).
fn apply_sphere(world: &mut World, cx: i32, cy: i32, cz: i32, radius: u8, mat: u8) {
    let r = radius as i32;
    let r2 = r * r;
    for dy in -r..=r {
        for dx in -r..=r {
            for dz in -r..=r {
                if dx * dx + dy * dy + dz * dz > r2 { continue; }
                world.apply_edit(cx + dx, cy + dy, cz + dz, mat);
            }
        }
    }
}

#[derive(Default)]
struct Keys {
    forward: bool,
    back: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    sprint: bool,
}

pub struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    camera: Camera,
    /// Shared with the physics worker thread. Streaming, edits, upload and the
    /// physics tick all lock this; physics no longer runs synchronously inside
    /// the frame (checklist: physics on a worker thread).
    world: Arc<Mutex<World>>,
    /// Signals the physics worker to stop (set in App::drop).
    phys_stop: Arc<AtomicBool>,
    keys: Keys,
    last_frame: Instant,
    start_time: Instant,
    grabbed: bool,
    frames_since_log: u32,
    last_log: Instant,

    // Temporal differential bookkeeping.
    last_camera_pose: Option<(Vec3, f32, f32, f32)>,
    /// Monotonic frame counter, used to index the TAA jitter pattern.
    frame_counter: u64,
    /// CPU frame-time accumulators (logged under VOXELG_GPU_PROFILE): the GPU
    /// can sit at 500 fps while the CPU loop caps the real rate - this shows
    /// where the CPU milliseconds go. [pre(net+input), world-lock(stream+dirty+
    /// upload), render(acquire+encode+submit+present), whole frame] seconds.
    cpu_prof: [f64; 4],
    cpu_prof_n: u32,
    /// Deterministic in-game benchmark (VOXELG_BENCH=1): drives the camera
    /// through fixed scenario segments on the standard demo world, frozen
    /// sun, uncapped present, and logs real end-to-end fps per segment
    /// (everything the headless harness cannot see: present, pacing, CPU
    /// loop, streaming). Same segments every run = honest before/after.
    bench: Option<BenchState>,
    /// Rotating animation-refresh phase: each frame ~1/ANIM_REFRESH_SPREAD of
    /// the (otherwise clean) tiles are re-traced so sky/water/foliage keep
    /// animating on a still camera — spread evenly instead of one hard full
    /// re-trace every 15 frames (which stuttered).
    refresh_phase: u32,
    opts: ClientOpts,
    /// Falling-leaf simulation (None when VOXELG_NO_LEAVES is set).
    leaf_sim: Option<leaffall::LeafSim>,
    leaf_instances: Vec<leaffall::LeafInstance>,
    grass: crate::grass::GrassField,
    tile_dirty_mask: Vec<u32>,
    first_frame: bool,
    current_material: u8,

    // Multiplayer.
    net: Option<net::NetClient>,
    /// Server address (Some in connect mode), kept so we can auto-reconnect.
    server_addr: Option<String>,
    last_reconnect: Instant,
    last_sent_pose: Option<(Vec3, f32, f32)>,
    last_net_send: Instant,
    last_heartbeat: Instant,
    remote_players: std::collections::HashMap<net::PlayerId, net::RemotePlayer>,

    // Click queue. Consumed at the top of a frame after camera state for the
    // about-to-render frame is settled, so the raycast direction matches the
    // rendered crosshair.
    pending_clicks: Vec<winit::event::MouseButton>,
}

impl App {
    pub fn new(net: Option<net::NetClient>, server_addr: Option<String>, opts: ClientOpts) -> Self {
        let mut world = World::new();
        world.fill_demo_terrain();
        // Spawn above the surface at the spawn column. The default y (80) is
        // often *below* the terrain/mountains, which renders as an opaque black
        // screen ("you see nothing"); lift the camera to clear ground + water.
        let mut camera = Camera::new();
        camera.move_speed *= opts.speed.clamp(0.1, 100.0);
        let s = voxel::sample_terrain(camera.pos.x, camera.pos.z, world.seed);
        let surface = s.h.max(s.water_top) as f32;
        camera.pos.y = surface + 10.0;

        // Hand the world to a physics worker thread. It runs the fixed-step CA
        // at 30 Hz behind the shared mutex; the render thread only takes the
        // lock briefly for streaming / edits / upload, so a heavy fluid sim no
        // longer stalls the frame (checklist: physics on a worker thread).
        let world = Arc::new(Mutex::new(world));
        let phys_stop = Arc::new(AtomicBool::new(false));
        // With GPU-compute physics (#25) the CA runs on the GPU in the render
        // loop, so don't also spawn the CPU physics worker (they'd both mutate
        // the world). Default path keeps the CPU worker.
        if std::env::var("VOXELG_GPU_PHYSICS").is_err() {
            let world = world.clone();
            let stop = phys_stop.clone();
            std::thread::Builder::new()
                .name("physics".into())
                .spawn(move || {
                    let step = Duration::from_micros(1_000_000 / 30);
                    while !stop.load(Ordering::Relaxed) {
                        std::thread::sleep(step);
                        // Recover a poisoned lock rather than propagating a
                        // second panic — the render thread does the same.
                        let mut w = world.lock().unwrap_or_else(|e| e.into_inner());
                        physics::tick(&mut w);
                    }
                })
                .expect("spawn physics thread");
        }

        Self {
            window: None,
            renderer: None,
            camera,
            world,
            phys_stop,
            keys: Keys::default(),
            last_frame: Instant::now(),
            start_time: Instant::now(),
            last_camera_pose: None,
            frame_counter: 0,
            cpu_prof: [0.0; 4],
            cpu_prof_n: 0,
            bench: BenchState::from_env(),
            refresh_phase: 0,
            opts,
            leaf_sim: std::env::var("VOXELG_NO_LEAVES")
                .is_err()
                .then(|| leaffall::LeafSim::new(0x1eaf_5eed)),
            leaf_instances: Vec::new(),
            grass: crate::grass::GrassField::new(),
            tile_dirty_mask: Vec::with_capacity(2048),
            first_frame: true,
            current_material: MAT_STONE,
            grabbed: false,
            frames_since_log: 0,
            last_log: Instant::now(),
            net,
            server_addr,
            last_reconnect: Instant::now(),
            last_sent_pose: None,
            last_net_send: Instant::now(),
            last_heartbeat: Instant::now(),
            remote_players: std::collections::HashMap::new(),
            pending_clicks: Vec::new(),
        }
    }

    fn request_redraw(&self) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    /// Auto-reconnect: if the link dropped, retry every 2s. A fresh connection
    /// re-handshakes and the server replays the full ordered edit log, so we
    /// re-sync world state automatically (checklist: reconnection).
    fn maybe_reconnect(&mut self, now: Instant) {
        let Some(addr) = self.server_addr.clone() else { return; };
        let down = self.net.as_ref().map_or(true, |n| !n.is_connected());
        if down && (now - self.last_reconnect) >= Duration::from_secs(2) {
            self.last_reconnect = now;
            match net::NetClient::connect(&addr) {
                Ok(c) => {
                    log::info!("reconnected to {addr}");
                    self.net = Some(c);
                    self.last_sent_pose = None;
                    self.remote_players.clear();
                }
                Err(e) => log::warn!("reconnect to {addr} failed: {e}"),
            }
        }
    }

    fn poll_net(&mut self, now: Instant) {
        let (msgs, my_id) = match self.net.as_mut() {
            Some(net) => (net.drain(), net.my_id),
            None => return,
        };
        for msg in msgs {
            match msg {
                net::Message::PlayerUpdate { id, pos, yaw, pitch } => {
                    if Some(id) != my_id {
                        // Feed the interpolation buffer instead of snapping.
                        self.remote_players
                            .entry(id)
                            .or_insert_with(net::RemotePlayer::new)
                            .push_sample(now, pos, yaw, pitch);
                    }
                }
                net::Message::PlayerJoin { id } => {
                    log::info!("player {} joined", id);
                }
                net::Message::PlayerLeave { id } => {
                    log::info!("player {} left", id);
                    self.remote_players.remove(&id);
                }
                net::Message::VoxelEdit { wx, wy, wz, mat, .. } => {
                    self.world.lock().unwrap_or_else(|e| e.into_inner()).apply_edit(wx, wy, wz, mat);
                }
                net::Message::Explode { cx, cy, cz, radius, mat, .. } => {
                    apply_sphere(&mut self.world.lock().unwrap_or_else(|e| e.into_inner()), cx, cy, cz, radius, mat);
                }
                net::Message::EditLog { compressed } => {
                    // Ordered, compressed edit-log replay (late-join sync).
                    let edits = net::decompress_edits(&compressed);
                    log::info!("applying {} replayed edits", edits.len());
                    let mut world = self.world.lock().unwrap_or_else(|e| e.into_inner());
                    for e in edits {
                        match e {
                            net::Message::VoxelEdit { wx, wy, wz, mat, .. } => {
                                world.apply_edit(wx, wy, wz, mat);
                            }
                            net::Message::Explode { cx, cy, cz, radius, mat, .. } => {
                                apply_sphere(&mut world, cx, cy, cz, radius, mat);
                            }
                            _ => {}
                        }
                    }
                }
                net::Message::JoinAck { your_id, seed, .. } => {
                    log::info!("joined as player {} (seed {})", your_id, seed);
                }
                net::Message::VersionMismatch { server_version } => {
                    log::error!("protocol mismatch: server v{server_version}");
                }
                _ => {}
            }
        }
    }

    fn maybe_send_pose(&mut self, now: Instant) {
        let Some(net) = self.net.as_ref() else { return; };
        let pose = (self.camera.pos, self.camera.yaw, self.camera.pitch);
        let changed = self.last_sent_pose.map_or(true, |p| p != pose);
        // Cap at 20 Hz; pose goes over UDP (lossy is fine).
        if changed && (now - self.last_net_send) >= Duration::from_millis(50) {
            net.send_pose(net.my_id.unwrap_or(0), pose.0.to_array(), pose.1, pose.2);
            self.last_sent_pose = Some(pose);
            self.last_net_send = now;
        }
        // Heartbeat (TCP) so the server doesn't time us out when we're still.
        if (now - self.last_heartbeat) >= net::HEARTBEAT_INTERVAL {
            net.heartbeat();
            self.last_heartbeat = now;
        }
    }

    fn broadcast_edit_world(&self, wx: i32, wy: i32, wz: i32, mat: u8) {
        let Some(net) = self.net.as_ref() else { return; };
        net.send(net::Message::VoxelEdit { wx, wy, wz, mat, seq: 0 });
    }

    fn grab(&mut self, on: bool) {
        let Some(win) = &self.window else { return; };
        if on {
            let _ = win
                .set_cursor_grab(CursorGrabMode::Confined)
                .or_else(|_| win.set_cursor_grab(CursorGrabMode::Locked));
            win.set_cursor_visible(false);
        } else {
            let _ = win.set_cursor_grab(CursorGrabMode::None);
            win.set_cursor_visible(true);
        }
        self.grabbed = on;
    }

    fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, pressed: bool) {
        match code {
            KeyCode::KeyW => self.keys.forward = pressed,
            KeyCode::KeyS => self.keys.back = pressed,
            KeyCode::KeyA => self.keys.left = pressed,
            KeyCode::KeyD => self.keys.right = pressed,
            KeyCode::Space => self.keys.up = pressed,
            KeyCode::ShiftLeft | KeyCode::ControlLeft => self.keys.down = pressed,
            KeyCode::AltLeft => self.keys.sprint = pressed,
            KeyCode::Digit1 if pressed => self.current_material = MAT_STONE,
            KeyCode::Digit2 if pressed => self.current_material = MAT_SAND,
            KeyCode::Digit3 if pressed => self.current_material = MAT_WATER,
            KeyCode::Digit4 if pressed => self.current_material = MAT_WOOD,
            KeyCode::Digit5 if pressed => self.current_material = MAT_LEAVES,
            KeyCode::Digit6 if pressed => self.current_material = MAT_GLASS,
            KeyCode::Digit7 if pressed => self.current_material = MAT_LAVA,
            KeyCode::Digit8 if pressed => self.current_material = MAT_ICE,
            KeyCode::Digit9 if pressed => self.current_material = MAT_SNOW,
            KeyCode::Digit0 if pressed => self.current_material = MAT_SMOKE,
            // Test-fire: drop a lava block 15 voxels ahead with no raycast.
            KeyCode::KeyT if pressed => {
                let fwd = self.camera.forward();
                let p = self.camera.pos + fwd * 15.0;
                self.world.lock().unwrap_or_else(|e| e.into_inner()).apply_edit(
                    p.x.floor() as i32, p.y.floor() as i32, p.z.floor() as i32, voxel::MAT_LAVA,
                );
            }
            KeyCode::Escape if pressed => {
                if self.grabbed {
                    self.grab(false);
                } else {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    /// Run picking for queued clicks against the current (about-to-render)
    /// camera state and apply the resulting edits.
    fn consume_clicks(&mut self) {
        if self.pending_clicks.is_empty() {
            return;
        }
        use winit::event::MouseButton;
        let clicks = std::mem::take(&mut self.pending_clicks);
        let mut world = self.world.lock().unwrap_or_else(|e| e.into_inner());
        let world_origin = world.world_origin_voxel();
        for button in clicks {
            let Some(hit) = raycast::raycast(
                self.camera.pos, self.camera.forward(), &world, world_origin,
            ) else { continue; };
            let v = hit.voxel;
            match button {
                MouseButton::Left => {
                    let (cx, cy, cz) = (v[0], v[1], v[2]);
                    let r: u8 = 2;
                    apply_sphere(&mut world, cx, cy, cz, r, voxel::MAT_AIR);
                    if let Some(net) = self.net.as_ref() {
                        net.send(net::Message::Explode {
                            cx, cy, cz, radius: r, mat: voxel::MAT_AIR, seq: 0,
                        });
                    }
                }
                MouseButton::Right => {
                    let wx = v[0] + hit.normal[0];
                    let wy = v[1] + hit.normal[1];
                    let wz = v[2] + hit.normal[2];
                    world.apply_edit(wx, wy, wz, self.current_material);
                    self.broadcast_edit_world(wx, wy, wz, self.current_material);
                }
                _ => {}
            }
        }
    }

    /// The per-frame orchestration: input integration, streaming, physics,
    /// temporal-differential tile selection, GPU upload + render.
    fn render_frame(&mut self) {
        if self.renderer.is_none() {
            return;
        }

        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(1.0 / 30.0);
        self.last_frame = now;
        let cpu_t0 = now;

        // Pump network in/out before re-borrowing renderer.
        self.maybe_reconnect(now);
        self.poll_net(now);
        self.maybe_send_pose(now);

        let speed = if self.keys.sprint { 4.0 } else { 1.0 };
        let f = (self.keys.forward as i32 - self.keys.back as i32) as f32;
        let r = (self.keys.right as i32 - self.keys.left as i32) as f32;
        let u = (self.keys.up as i32 - self.keys.down as i32) as f32;
        self.camera.translate_local(dt, f * speed, r * speed, u * speed);
        // Benchmark mode: the script owns the camera (input overridden).
        if let Some(mut b) = self.bench.take() {
            if let Some(r) = self.renderer.as_ref() {
                if let Some(p) = &r.gpu_profiler {
                    if p.reports != b.last_report_seen && p.reports > 0 {
                        b.last_report_seen = p.reports;
                        for i in 0..9 {
                            b.gpu_acc[i] += p.last[i];
                        }
                        b.gpu_total += p.last_total;
                        b.gpu_n += 1;
                    }
                }
            }
            if b.step(&mut self.camera, dt) {
                self.bench = Some(b);
            } else {
                // _exit, not exit(): the process teardown path is broken (the
                // known segfault/free-on-shutdown bug) and libc atexit
                // handlers DEADLOCK from exit() - the results printed but the
                // window sat forever. _exit skips all handlers.
                unsafe { libc::_exit(0) };
            }
        }

        let cpu_t1 = Instant::now();
        // Streaming (brief world lock): keep the window centred with a
        // hysteresis deadband so boundary oscillation doesn't thrash regen.
        {
            let mut world = self.world.lock().unwrap_or_else(|e| e.into_inner());
            let cur_origin = world.world_origin_chunk;
            let target_origin = voxel::World::target_origin_chunk(self.camera.pos);
            let drift = target_origin - cur_origin;
            if drift.x.abs() >= STREAM_HYSTERESIS || drift.y.abs() >= STREAM_HYSTERESIS {
                world.shift_origin(target_origin);
            }
            // Install a budgeted number of finished chunks. shift_origin already
            // cleared + dispatched; generation + derived-mask computation run on
            // the worker pool, so the frame never does the heavy work.
            world.install_finished_chunks(CHUNK_INSTALL_BUDGET);
        }

        // Raycast queued clicks against the now-settled camera (locks internally).
        self.consume_clicks();

        // Physics now runs on its own worker thread — no inline tick here.

        // ---- temporal differential ----
        let cur_pose = (self.camera.pos, self.camera.yaw, self.camera.pitch, self.camera.fov_y);
        // #1: only a *real* camera move forces a full re-trace. A sub-mm /
        // sub-pixel jitter (float noise, micro mouse drift) no longer re-traces
        // every tile — it would just reproduce the same image.
        let camera_changed = match self.last_camera_pose {
            None => true,
            Some((pp, py, ppi, pf)) => {
                (self.camera.pos - pp).length_squared() > 1.0e-6
                    || (self.camera.yaw - py).abs() > 1.0e-4
                    || (self.camera.pitch - ppi).abs() > 1.0e-4
                    || (self.camera.fov_y - pf).abs() > 1.0e-4
            }
        };
        let (rw, rh) = self.renderer.as_ref().unwrap().size;
        let tiles_w = (rw + 7) / 8;
        let tiles_h = (rh + 7) / 8;
        let n_tiles = (tiles_w * tiles_h) as usize;
        let word_count = (n_tiles + 31) / 32;
        self.tile_dirty_mask.clear();
        self.tile_dirty_mask.resize(word_count, 0);

        // TAA: jitter + accumulate only on a static camera (needs only
        // camera_changed / first_frame, so compute before taking the lock).
        // Skip the blend for one frame after a resize — the history texture was
        // just recreated and would otherwise be read uninitialised.
        let taa_reset = self.renderer.as_mut().unwrap().take_taa_reset();
        self.frame_counter += 1;
        // TAA + reprojection caches are STATIC-ONLY: a still camera jitters +
        // accumulates (sub-pixel AA) and the shadow/AO + colour history reproject
        // onto themselves (exact, no warp). A MOVING camera passes the freshly
        // traced frame straight through (taa_blend = 0) — motion reprojection was
        // visibly warping/smearing, and the engine is fast enough to trace every
        // moving frame fresh, so we keep motion sharp instead.
        let hard_reset = self.first_frame || taa_reset || camera_changed;
        let (jitter, taa_blend) = if hard_reset {
            ([0.0_f32, 0.0_f32], 0.0_f32)
        } else {
            (JITTER_PATTERN[(self.frame_counter & 7) as usize], 0.9_f32)
        };
        let t = (now - self.start_time).as_secs_f32();
        // --freeze-time pins the DAY/NIGHT cycle only: the sun runs on its
        // own clock while water, wind and falling leaves keep animating.
        let sun_t = self.opts.freeze_time.unwrap_or(t);

        // GPU physics modifies the brick buffer directly (not via dirty_bricks),
        // so force a full re-trace each frame while it's on for the changes to show.
        let gpu_physics = self.renderer.as_ref().unwrap().gpu_physics;

        // Build the dirty-tile mask + upload world + camera under one lock so
        // the physics worker can run in the gaps between frames.
        {
            let mut world = self.world.lock().unwrap_or_else(|e| e.into_inner());
            let physics_changed = world.all_dirty || !world.dirty_bricks.is_empty();
            let force_full = self.first_frame || world.all_dirty || camera_changed || gpu_physics;
            if force_full {
                for w in self.tile_dirty_mask.iter_mut() { *w = u32::MAX; }
            } else {
                // Physics-dirty tiles (projected from the dirty bricks, no clone).
                if physics_changed {
                    let camera = &self.camera;
                    let mask = &mut self.tile_dirty_mask;
                    for &bi in &world.dirty_bricks {
                        temporal::project_brick_to_tiles(
                            bi, &world, camera, rw, rh, tiles_w, tiles_h, mask,
                        );
                    }
                }
                // #2: rotating animation refresh — re-trace ~1/spread of the
                // tiles each frame so sky/water/foliage keep animating and the
                // per-pixel shadow-staleness dither can actually fire. The
                // refreshed set is HASH-SCATTERED, never a fixed stride: a
                // stride aligns the freshly-relit tiles into repeating rows, so
                // during any fast lighting change (sunset darkening, shadow
                // edges sweeping) the stale/fresh boundary reads as visible
                // LINES / ticking blocks. Scattered updates read as fine noise
                // that TAA averages. During the dawn/dusk transition the
                // rotation runs 2x (1/4 per frame) because per-frame lighting
                // change is largest exactly there; freeze-time keeps the calm
                // cadence (the sun cannot move).
                let sun_moving = self.opts.freeze_time.is_none();
                let sy = crate::camera::sun_dir_at(sun_t).y;
                let spread = if sun_moving && sy > -0.15 && sy < 0.35 {
                    ANIM_REFRESH_SPREAD / 2
                } else {
                    ANIM_REFRESH_SPREAD
                };
                let phase = self.refresh_phase as usize % spread;
                let tw = tiles_w as usize;
                for ti in 0..n_tiles {
                    let tx = ti % tw;
                    let ty = ti / tw;
                    let h = (tx.wrapping_mul(73856093) ^ ty.wrapping_mul(19349663))
                        .wrapping_mul(2654435761);
                    if (h >> 8) % spread == phase {
                        self.tile_dirty_mask[ti >> 5] |= 1u32 << (ti & 31);
                    }
                }
                self.refresh_phase = (self.refresh_phase + 1) % ANIM_REFRESH_SPREAD as u32;
            }
            let world_origin = world.world_origin_voxel();
            let renderer = self.renderer.as_mut().unwrap();
            renderer.upload_world(&mut world);
            // Mask-only clears from recycled slots (render them as sky now).
            renderer.upload_mask_clears(&mut world);
            // GPU-compute physics step (#25): runs the sand CA on the just-uploaded
            // bricks so player edits/streaming are seen, result rendered this frame.
            if gpu_physics {
                renderer.run_gpu_physics();
            }
            renderer.update_camera(&self.camera, t, sun_t, world_origin, jitter, taa_blend);
            // Falling leaves: stepped under the already-held lock (a few
            // hundred leaves = tens of microseconds), uploaded after it.
            if let Some(sim) = &mut self.leaf_sim {
                sim.step(&world, self.camera.pos, dt, crate::camera::sun_dir_at(sun_t));
            }
            // Grass blade field: rescan grass-top columns when the camera
            // strays from the last scan centre (a few ms, rare).
            self.grass.maybe_rebuild(&world, self.camera.pos, None);
        }
        if let Some(sim) = &self.leaf_sim {
            sim.write_instances(&mut self.leaf_instances);
            self.renderer.as_mut().unwrap().upload_leaves(&self.leaf_instances);
        }
        self.renderer.as_mut().unwrap().upload_grass(&self.grass);
        // We always re-trace at least the rotating animation subset.
        let any_dirty = true;

        let renderer = self.renderer.as_mut().unwrap();
        // Render remote players at their interpolated (delayed) pose.
        let players_vec: Vec<(Vec3, u32)> = self.remote_players
            .iter()
            .filter_map(|(id, rp)| rp.sample(now).map(|(pos, _, _)| (Vec3::from_array(pos), *id)))
            .collect();
        renderer.upload_players(&players_vec);
        if any_dirty {
            renderer.upload_tile_dirty(&self.tile_dirty_mask);
        }
        let cpu_t2 = Instant::now();
        match renderer.render(any_dirty) {
            Ok(()) => {}
            // Transient surface states: reconfigure and try again next frame.
            Err(wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated) => {
                let (w, h) = renderer.surface_size;
                renderer.resize(w.max(1), h.max(1));
            }
            // Timed out / occluded / validation-erred while acquiring the swapchain
            // image; skip this frame and try again next pump.
            Err(wgpu::CurrentSurfaceTexture::Timeout
                | wgpu::CurrentSurfaceTexture::Occluded
                | wgpu::CurrentSurfaceTexture::Validation) => {}
            // Success/Suboptimal are consumed inside render() and never returned as Err.
            Err(wgpu::CurrentSurfaceTexture::Success(_)
                | wgpu::CurrentSurfaceTexture::Suboptimal(_)) => {}
        }
        if std::env::var("VOXELG_GPU_PROFILE").is_ok() {
            let cpu_t3 = Instant::now();
            self.cpu_prof[0] += (cpu_t1 - cpu_t0).as_secs_f64();
            self.cpu_prof[1] += (cpu_t2 - cpu_t1).as_secs_f64();
            self.cpu_prof[2] += (cpu_t3 - cpu_t2).as_secs_f64();
            self.cpu_prof[3] += (cpu_t3 - cpu_t0).as_secs_f64();
            self.cpu_prof_n += 1;
            if self.cpu_prof_n == 120 {
                let m = |i: usize| self.cpu_prof[i] * 1000.0 / 120.0;
                log::info!(
                    "cpu frame ms: pre {:.2}  world+upload {:.2}  render(acquire+submit+present) {:.2}  | total {:.2}",
                    m(0), m(1), m(2), m(3)
                );
                self.cpu_prof = [0.0; 4];
                self.cpu_prof_n = 0;
            }
        }
        self.last_camera_pose = Some(cur_pose);
        self.first_frame = false;

        self.frames_since_log += 1;
        if (now - self.last_log).as_secs_f32() >= 1.0 {
            log::info!("fps {:>4}  cam {:?}", self.frames_since_log, self.camera.pos);
            self.frames_since_log = 0;
            self.last_log = now;
        }
    }
}

impl Drop for App {
    fn drop(&mut self) {
        // Tell the physics worker to exit so it doesn't outlive the app.
        self.phys_stop.store(true, Ordering::Relaxed);
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("voxel")
            .with_inner_size(if std::env::var("VOXELG_BENCH").is_ok() {
                // Benchmark runs pin the render size to the panel/harness
                // resolution so every run (and the GPU bench tables) compare
                // apples to apples regardless of how the play window is sized.
                winit::dpi::LogicalSize::new(1920, 1080)
            } else {
                winit::dpi::LogicalSize::new(1280, 720)
            });
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                log::error!("failed to create window: {e}");
                event_loop.exit();
                return;
            }
        };
        let renderer = match Renderer::new(window.clone(), &self.world.lock().unwrap_or_else(|e| e.into_inner())) {
            Ok(r) => r,
            Err(e) => {
                log::error!("failed to initialise renderer: {e}");
                event_loop.exit();
                return;
            }
        };
        self.window = Some(window);
        self.renderer = Some(renderer);
        self.grab(true);
        self.request_redraw();
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(r) = self.renderer.as_mut() {
                    r.resize(size.width, size.height);
                }
                self.request_redraw();
            }
            WindowEvent::Focused(false) => self.grab(false),
            WindowEvent::MouseInput { state: ElementState::Pressed, button, .. } => {
                if !self.grabbed {
                    self.grab(true);
                } else {
                    self.pending_clicks.push(button);
                }
                self.request_redraw();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.handle_key(event_loop, code, pressed);
                }
                self.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                self.render_frame();
                if self.renderer.is_none() {
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        if !self.grabbed {
            return;
        }
        if let DeviceEvent::MouseMotion { delta } = event {
            self.camera.rotate(delta.0 as f32, delta.1 as f32);
            // Render promptly on look so mouselook feels responsive rather than
            // waiting for the next paced heartbeat.
            self.request_redraw();
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Explicit frame pacing: instead of busy-spinning (ControlFlow::Poll +
        // unconditional redraw), sleep until the next frame is due. The scene
        // animates continuously (sky / water / foliage / clouds), so we still
        // redraw every paced tick — but capped, and the CPU idles in between.
        if self.window.is_none() {
            return;
        }
        // VOXELG_UNCAPPED removes the frame-pacing cap for throughput measurement
        // (pair with the Immediate present mode the renderer selects under it).
        let cap_hz = if std::env::var("VOXELG_UNCAPPED").is_ok() { 100_000.0 } else { FRAME_CAP_HZ };
        let frame = Duration::from_secs_f64(1.0 / cap_hz);
        let since = self.last_frame.elapsed();
        if since >= frame {
            self.request_redraw();
            event_loop.set_control_flow(ControlFlow::WaitUntil(Instant::now() + frame));
        } else {
            event_loop.set_control_flow(ControlFlow::WaitUntil(self.last_frame + frame));
        }
    }
}

/// Build the winit event loop and run the client application.
/// Command-line client options.
#[derive(Clone, Copy)]
pub struct ClientOpts {
    /// Pin the shader clock (sun, water, wind) at this many seconds instead
    /// of advancing - the falling-leaf sim keeps its own clock and stays
    /// alive. None = normal time.
    pub freeze_time: Option<f32>,
    /// Fly-speed multiplier applied to the camera's base move speed.
    pub speed: f32,
}

impl Default for ClientOpts {
    fn default() -> Self {
        Self { freeze_time: None, speed: 1.0 }
    }
}

pub fn run_client(net: Option<net::NetClient>, server_addr: Option<String>, opts: ClientOpts) {
    let event_loop = EventLoop::new().expect("event loop");
    // Frame-paced: the loop wakes on input or at the next scheduled frame
    // (see App::about_to_wait), not in a tight Poll spin.
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = App::new(net, server_addr, opts);
    if let Err(e) = event_loop.run_app(&mut app) {
        log::error!("event loop exited with error: {e}");
    }
}
