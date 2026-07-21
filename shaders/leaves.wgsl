// Falling-leaf particle pass. Instanced quads drawn onto the TAA RESOLVE
// texture AFTER the resolve->history copy, so moving leaves never enter the
// TAA feedback loop (zero ghosting by construction), and manually
// depth-tested against the raymarcher's exported primary-hit depth (water
// pixels carry the SURFACE t, so leaves sink behind water correctly).
// Prepended with common.wgsl (Camera + sun helpers) and the generated
// sprite atlas consts - see renderer::leaves_source().

struct LeafInstance {
    pos: vec3<f32>,
    size: f32,
    rot: f32,
    tilt_phase: f32,
    sprite: u32,
    tint: u32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> leaves: array<LeafInstance>;
@group(0) @binding(2) var<storage, read> sprites: array<u32>;
@group(0) @binding(3) var scene_depth: texture_2d<f32>;

// Same decoder as raymarch.wgsl sprite_texel (16 u32 words per 16x16
// sprite, 2 bits per texel, y = 0 is the bottom row).
fn sprite_texel(sprite: u32, x: u32, y: u32) -> u32 {
    let bit = (y * 16u + x) * 2u;
    let w = sprites[sprite * 16u + (bit >> 5u)];
    return (w >> (bit & 31u)) & 3u;
}

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) sprite: u32,
    @location(2) @interpolate(flat) lit: vec3<f32>,
    // Euclidean distance to the camera - matches the primary ray t the
    // depth texture stores.
    @location(3) view_t: f32,
};

@vertex
fn vs_leaf(@builtin(vertex_index) vid: u32,
           @builtin(instance_index) iid: u32) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(-1.0, 1.0));
    let leaf = leaves[iid];
    let c = corners[vid];
    var o: VsOut;
    // Camera-facing card spun in-plane; the v axis squashes with the tilt
    // phase, which is exactly how a flat tumbling plate projects - reads as
    // 3D flutter with zero per-fragment cost (0.25 floor: never edge-on
    // invisible).
    let cs = cos(leaf.rot);
    let sn = sin(leaf.rot);
    let squash = 0.25 + 0.75 * abs(cos(leaf.tilt_phase));
    let u_ax = (camera.right * cs + camera.up * sn) * leaf.size;
    let v_ax = (camera.up * cs - camera.right * sn) * (leaf.size * squash);
    let wp = leaf.pos + u_ax * c.x + v_ax * c.y;
    let d = wp - camera.origin;
    let z = dot(d, camera.forward);
    if (z < 0.05 || leaf.size <= 0.0) {
        o.pos = vec4<f32>(0.0, 0.0, 2.0, 1.0); // behind the eye / dead: clip
        o.uv = vec2<f32>(0.0);
        o.sprite = 0u;
        o.lit = vec3<f32>(0.0);
        o.view_t = 0.0;
        return o;
    }
    // Inverse of ray_dir_uv's pinhole, minus the TAA sub-pixel jitter so
    // the card sits on the same sample grid as the depth it tests against.
    let aspect = camera.resolution.x / camera.resolution.y;
    var ndc = vec2<f32>(dot(d, camera.right) / (z * camera.tan_half_fov * aspect),
                        dot(d, camera.up) / (z * camera.tan_half_fov));
    ndc = ndc - vec2<f32>(camera.jitter.x, -camera.jitter.y) * 2.0 / camera.resolution;
    o.pos = vec4<f32>(ndc.x * z, ndc.y * z, 0.5 * z, z); // w=z: perspective-correct uv
    o.uv = c * 0.5 + vec2<f32>(0.5);
    o.sprite = leaf.sprite;
    o.view_t = length(d);
    // Time-of-day light: ambient follows the sky brightness, direct sun is
    // scaled by the CPU-probed visibility factor in the alpha byte (leaves
    // under canopy or indoors go properly dark) plus a tumble glint.
    let s = sun_dir_at(camera.sun_time);
    let glint = 0.55 + 0.45 * abs(cos(leaf.tilt_phase));
    let t4 = unpack4x8unorm(leaf.tint);
    let sky_f = 0.25 + 0.75 * sun_intensity(s);
    o.lit = t4.rgb * (vec3<f32>(0.30, 0.34, 0.40) * sky_f + sun_color(s) * glint * t4.a);
    return o;
}

@fragment
fn fs_leaf(in: VsOut) -> @location(0) vec4<f32> {
    let scene_t = textureLoad(scene_depth, vec2<i32>(in.pos.xy), 0).r;
    // Manual depth test; the scaled epsilon keeps resting leaves stable on
    // the surface they landed on.
    if (in.view_t > scene_t + (0.05 + scene_t * 0.002)) {
        discard;
    }
    let tx = u32(clamp(in.uv.x * 16.0, 0.0, 15.0));
    let ty = u32(clamp(in.uv.y * 16.0, 0.0, 15.0));
    let val = sprite_texel(in.sprite, tx, ty);
    if (val == 0u) {
        discard;
    }
    var tone = 0.95; // same tones as the leaf cloud cards
    if (val == 2u) { tone = 0.62; }
    if (val == 3u) { tone = 1.30; }
    return vec4<f32>(in.lit * tone, 1.0);
}
