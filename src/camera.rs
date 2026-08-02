use glam::Vec3;

#[derive(Clone)]
pub struct Camera {
    pub pos: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub fov_y: f32,
    pub move_speed: f32,
    pub look_sensitivity: f32,
}

impl Camera {
    pub fn new() -> Self {
        Self {
            pos: Vec3::new(256.0, 80.0, 256.0),
            yaw: 0.0,
            pitch: -0.5,
            fov_y: 70.0_f32.to_radians(),
            move_speed: 80.0,
            look_sensitivity: 0.0025,
        }
    }

    pub fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.sin() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.cos() * self.pitch.cos(),
        )
        .normalize()
    }

    pub fn right(&self) -> Vec3 {
        // Forward x world-up, normalized
        let f = self.forward();
        Vec3::new(f.z, 0.0, -f.x).normalize()
    }

    pub fn up(&self) -> Vec3 {
        self.forward().cross(self.right()).normalize()
    }

    pub fn rotate(&mut self, dx: f32, dy: f32) {
        // Mouse right (dx > 0) should turn the view right — i.e. yaw
        // *increases* so forward.x becomes positive.
        self.yaw += dx * self.look_sensitivity;
        self.pitch -= dy * self.look_sensitivity;
        let limit = std::f32::consts::FRAC_PI_2 - 0.01;
        self.pitch = self.pitch.clamp(-limit, limit);
    }

    pub fn translate_local(&mut self, dt: f32, forward: f32, right: f32, up: f32) {
        let f = self.forward();
        let r = self.right();
        let u = Vec3::Y;
        self.pos += (f * forward + r * right + u * up) * self.move_speed * dt;
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub origin: [f32; 3],
    pub _pad0: f32,
    pub forward: [f32; 3],
    pub _pad1: f32,
    pub right: [f32; 3],
    pub _pad2: f32,
    pub up: [f32; 3],
    pub tan_half_fov: f32,
    pub resolution: [f32; 2],
    pub time: f32,
    /// Frame-constant wind direction x (unit XZ vector; z lives in the
    /// former _pad6 slot). Computed here once per frame so shaders never
    /// re-derive trig of time per pixel; formula documented at the WGSL
    /// wind_dir_now().
    pub wind_x: f32,
    /// World-voxel offset of the loaded region's lower corner. The shader
    /// uses this to bounds-check rays + mod-fold world voxel coords into the
    /// toroidal slot storage.
    pub world_origin: [i32; 3],
    /// Rotating GI-probe update round (frame counter mod GI_UPDATE_DIV). The probe
    /// gather updates a strided 1/GI_UPDATE_DIV slice of probes each frame.
    pub gi_round: i32,
    /// Sub-pixel ray jitter (in pixels) for temporal anti-aliasing. Non-zero
    /// only while accumulating (static camera); 0 on motion.
    pub jitter: [f32; 2],
    /// TAA history blend weight: 0 = use current frame only (reset / motion),
    /// ~0.9 = accumulate with reprojected history.
    pub taa_blend: f32,
    /// 1.0 = last frame's screen is a valid reprojection target; 0.0 = it is not
    /// (first frame, or a chunk cross where the rebased world origin moved and
    /// every stored position means something different).
    ///
    /// It used to gate the shadow/AO reprojection cache as well, which is where
    /// the old name `reproject_lighting` came from. That cache is gone; TAA's
    /// full reprojection and the RT GI's history reuse are the consumers now.
    pub reproject_ok: f32,
    // Previous-frame camera basis, for reprojecting a hit world-point into last
    // frame's screen to look up its cached shadow/AO.
    pub prev_origin: [f32; 3],
    pub wind_z: f32,
    pub prev_forward: [f32; 3],
    /// Day/night clock (sun position). Equals `time` normally; pinned by
    /// --freeze-time while everything else animates on `time`.
    pub sun_time: f32,
    pub prev_right: [f32; 3],
    /// 1.0 = the prev_* camera fields describe a real previous frame (set by
    /// set_prev_camera). Unlike reproject_ok this stays 1.0 while the
    /// camera MOVES - consumers that reproject by absolute position (the
    /// water reflection history) stay valid under motion; the first frame's
    /// zeroed history is rejected by its stored-position check.
    pub prev_valid: f32,
    pub prev_up: [f32; 3],
    pub _pad9: f32,
}

/// Frame-constant wind direction (unit XZ). ONE definition shared by the
/// camera uniform (shaders read it as camera.wind_x/wind_z) and the CPU
/// falling-leaf simulation.
pub fn wind_dir(time: f32) -> glam::Vec2 {
    let a = time * 0.04 + 0.4 * (time * 0.12).sin();
    glam::Vec2::new(a.cos(), a.sin())
}

/// CPU mirror of the WGSL sun_dir_at in shaders/common.wgsl - keep in sync.
pub fn sun_dir_at(t: f32) -> glam::Vec3 {
    let a = t * 0.025 + 1.20;
    glam::Vec3::new(a.cos(), a.sin(), 0.30).normalize()
}

impl CameraUniform {
    pub fn from_camera(
        c: &Camera,
        width: u32,
        height: u32,
        time: f32,
        sun_time: f32,
        world_origin_voxel: glam::IVec3,
        jitter: [f32; 2],
        taa_blend: f32,
    ) -> Self {
        Self {
            origin: c.pos.to_array(),
            _pad0: 0.0,
            forward: c.forward().to_array(),
            _pad1: 0.0,
            right: c.right().to_array(),
            _pad2: 0.0,
            up: c.up().to_array(),
            tan_half_fov: (c.fov_y * 0.5).tan(),
            resolution: [width as f32, height as f32],
            time,
            wind_x: wind_dir(time).x,
            world_origin: [world_origin_voxel.x, world_origin_voxel.y, world_origin_voxel.z],
            gi_round: 0,
            jitter,
            taa_blend,
            reproject_ok: 0.0,
            prev_origin: c.pos.to_array(),
            wind_z: wind_dir(time).y,
            prev_forward: c.forward().to_array(),
            sun_time,
            prev_right: c.right().to_array(),
            prev_valid: 0.0,
            prev_up: c.up().to_array(),
            _pad9: 0.0,
        }
    }

    /// Fill in the previous-frame camera basis and say whether it is usable as a
    /// reprojection target. Called by the frame loop once it has last frame's
    /// camera.
    pub fn set_prev_camera(&mut self, prev: &Camera, enable: bool) {
        self.reproject_ok = if enable { 1.0 } else { 0.0 };
        self.prev_valid = 1.0;
        self.prev_origin = prev.pos.to_array();
        self.prev_forward = prev.forward().to_array();
        self.prev_right = prev.right().to_array();
        self.prev_up = prev.up().to_array();
    }
}
