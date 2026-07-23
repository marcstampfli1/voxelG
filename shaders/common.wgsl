// Shared shader prelude. renderer.rs prepends this (after world_consts.wgsl)
// to raymarch.wgsl, beam.wgsl and taa.wgsl so the Camera uniform layout and
// these tiny helpers have ONE definition instead of a per-shader copy that
// silently drifts (beam previously carried a TRUNCATED Camera). Do not add
// anything here that depends on a per-shader binding (e.g. chunk_mask).

struct Camera {
    origin: vec3<f32>,
    _pad0: f32,
    forward: vec3<f32>,
    _pad1: f32,
    right: vec3<f32>,
    _pad2: f32,
    up: vec3<f32>,
    tan_half_fov: f32,
    resolution: vec2<f32>,
    time: f32,
    // Frame-constant wind direction (unit XZ), computed once on the CPU so
    // shaders never re-derive trig of time per pixel. Lives in what used to
    // be two alignment pads - zero layout growth.
    wind_x: f32,
    world_origin: vec3<i32>,
    // Rotating GI-probe update round: the probe cache updates 1/GI_UPDATE_DIV of
    // its probes each frame (strided by this counter), so the expensive gather is
    // amortized instead of paid in full every frame.
    gi_round: i32,
    jitter: vec2<f32>,
    taa_blend: f32,
    reproject_lighting: f32,
    prev_origin: vec3<f32>,
    wind_z: f32,
    prev_forward: vec3<f32>,
    // Day/night clock for the sun: equals `time` normally; pinned by
    // --freeze-time while water/wind/leaves keep animating on `time`.
    sun_time: f32,
    prev_right: vec3<f32>,
    // 1.0 = prev_* fields describe a real previous frame; stays set while
    // the camera moves (unlike reproject_lighting).
    prev_valid: f32,
    prev_up: vec3<f32>,
    _pad9: f32,
};

// Floored modulo for toroidal world-coord folding (result always in [0, b)).
// Signed `%` must never see a negative operand here: naga emits OpSRem without
// VK_KHR_maintenance8, and Vulkan makes OpSRem on negatives POISON (NVIDIA
// returns the unsigned-style remainder, so a `select(r, r+b, r<0)` fixup never
// fires). Both `%` below therefore only ever run on non-negative values.
fn pos_mod(a: i32, b: i32) -> i32 {
    if (a >= 0) { return a % b; }
    return (b - 1) - ((-1 - a) % b);
}

// Reciprocal that won't blow up on a near-zero ray component.
fn safe_inv(x: f32) -> f32 {
    if (abs(x) < 1e-8) { return 1e30; }
    return 1.0 / x;
}

// ---- Sun (shared by raymarch and the falling-leaf pass) ----
// Pure functions of time so passes that cannot see the raymarch bindings
// still light consistently.
fn sun_dir_at(t: f32) -> vec3<f32> {
    let a = t * 0.025 + 1.20;
    return normalize(vec3<f32>(cos(a), sin(a), 0.30));
}

fn sun_intensity(s: vec3<f32>) -> f32 {
    // Smoothstep into night below the horizon.
    return smoothstep(-0.05, 0.10, s.y);
}

fn sun_color(s: vec3<f32>) -> vec3<f32> {
    let h = clamp(s.y, 0.0, 1.0);
    // Sunset/sunrise = warm orange. Midday = neutral. Lerp on solar elevation.
    let warm = vec3<f32>(1.40, 0.60, 0.25);
    let mid = vec3<f32>(1.10, 1.02, 0.92);
    return mix(warm, mid, smoothstep(0.05, 0.40, h)) * sun_intensity(s);
}
