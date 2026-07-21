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
    pub _pad4: i32,
    /// Sub-pixel ray jitter (in pixels) for temporal anti-aliasing. Non-zero
    /// only while accumulating (static camera); 0 on motion.
    pub jitter: [f32; 2],
    /// TAA history blend weight: 0 = use current frame only (reset / motion),
    /// ~0.9 = accumulate with reprojected history.
    pub taa_blend: f32,
    /// 1.0 = the lighting (shadow/AO) reprojection cache may reuse last frame's
    /// values; 0.0 = force a full trace (first frame / after a chunk cross where
    /// the rebased world origin moved, invalidating cached positions).
    pub reproject_lighting: f32,
    // Previous-frame camera basis, for reprojecting a hit world-point into last
    // frame's screen to look up its cached shadow/AO.
    pub prev_origin: [f32; 3],
    pub wind_z: f32,
    pub prev_forward: [f32; 3],
    pub _pad7: f32,
    pub prev_right: [f32; 3],
    pub _pad8: f32,
    pub prev_up: [f32; 3],
    pub _pad9: f32,
}

impl CameraUniform {
    pub fn from_camera(
        c: &Camera,
        width: u32,
        height: u32,
        time: f32,
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
            wind_x: (time * 0.04 + 0.4 * (time * 0.12).sin()).cos(),
            world_origin: [world_origin_voxel.x, world_origin_voxel.y, world_origin_voxel.z],
            _pad4: 0,
            jitter,
            taa_blend,
            reproject_lighting: 0.0,
            prev_origin: c.pos.to_array(),
            wind_z: (time * 0.04 + 0.4 * (time * 0.12).sin()).sin(),
            prev_forward: c.forward().to_array(),
            _pad7: 0.0,
            prev_right: c.right().to_array(),
            _pad8: 0.0,
            prev_up: c.up().to_array(),
            _pad9: 0.0,
        }
    }

    /// Fill in the previous-frame camera basis + enable lighting reprojection.
    /// Called by the frame loop once it has last frame's camera.
    pub fn set_prev_camera(&mut self, prev: &Camera, enable: bool) {
        self.reproject_lighting = if enable { 1.0 } else { 0.0 };
        self.prev_origin = prev.pos.to_array();
        self.prev_forward = prev.forward().to_array();
        self.prev_right = prev.right().to_array();
        self.prev_up = prev.up().to_array();
    }
}
