// Fullscreen blit pass: samples the compute-shader output into the swapchain.

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // Single oversized triangle.
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0, -1.0),
    );
    var o: VsOut;
    o.pos = vec4<f32>(positions[vid], 0.0, 1.0);
    o.uv = uvs[vid];
    return o;
}

// Native-res sample + small + crosshair at screen centre so the player can
// see exactly where their click ray is going to land. The crosshair is a
// 12-px-arm + with a 1-px black outline so it stays visible on both bright
// (sky) and dark (cave wall) backgrounds.
@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let col = textureSample(src_tex, src_sampler, in.uv);
    let dims = vec2<f32>(textureDimensions(src_tex, 0));
    let offset_px = abs((in.uv - vec2<f32>(0.5, 0.5)) * dims);
    // Arm: 12 pixels long, 1 pixel wide (with 1-pixel black outline = 3 wide).
    let arm_len = 12.0;
    let outline_thick = 2.0;
    let inner_thick = 1.0;
    // Horizontal arm
    let in_horiz_outer = offset_px.x <= arm_len && offset_px.y <= outline_thick;
    let in_horiz_inner = offset_px.x <= arm_len && offset_px.y <= inner_thick;
    // Vertical arm
    let in_vert_outer  = offset_px.y <= arm_len && offset_px.x <= outline_thick;
    let in_vert_inner  = offset_px.y <= arm_len && offset_px.x <= inner_thick;
    if (in_horiz_inner || in_vert_inner) {
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }
    if (in_horiz_outer || in_vert_outer) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    // Force opaque: the TAA resolve smuggles the grass-blade marker in its
    // alpha; presenting it would alpha-composite the window on Wayland.
    return vec4<f32>(col.rgb, 1.0);
}
