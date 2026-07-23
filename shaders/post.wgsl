// HDR post stack: bloom + filmic tonemap + colour grade.
//
// The compose/grass output is HDR (rgba16float, sun glints and backlit
// foliage exceed 1.0). This stack turns it into the final LDR frame the
// TAA resolve consumes: a half-res bright-pass feeds a separable gaussian
// bloom, then cs_post adds the blurred glow back, applies an ACES filmic
// curve (soft shoulder instead of channel clipping), grades shadows cool /
// highlights warm with a gentle saturation lift, and finishes with a
// subtle vignette. Runs BEFORE TAA so the bloom is temporally averaged.
// Prepended with common.wgsl (Camera).

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var hdr_tex: texture_2d<f32>;
@group(0) @binding(2) var bloom_in: texture_2d<f32>;
@group(0) @binding(3) var bloom_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var ldr_out: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(5) var lin_sampler: sampler;

// ---- bright pass: half res, keep energy above the knee ------------------
@compute @workgroup_size(8, 8, 1)
fn cs_bright(@builtin(global_invocation_id) gid: vec3<u32>) {
    let hw = (vec2<i32>(camera.resolution) + vec2<i32>(1)) / 2;
    let p = vec2<i32>(gid.xy);
    if (p.x >= hw.x || p.y >= hw.y) { return; }
    let f = p * 2;
    var c = textureLoad(hdr_tex, f, 0).rgb
        + textureLoad(hdr_tex, f + vec2<i32>(1, 0), 0).rgb
        + textureLoad(hdr_tex, f + vec2<i32>(0, 1), 0).rgb
        + textureLoad(hdr_tex, f + vec2<i32>(1, 1), 0).rgb;
    c *= 0.25;
    // Soft knee around 1.0: only genuinely hot pixels bloom (sun, glints,
    // backlit tips), never flat mid-tones.
    let l = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    // Higher knee: only truly hot pixels bloom, so the glow never bleeds
    // across dark blade silhouettes and makes them read transparent.
    let k = max(l - 1.35, 0.0) / max(l, 1e-4);
    textureStore(bloom_out, p, vec4<f32>(c * k, 1.0));
}

// ---- separable gaussian (9 taps), half res ------------------------------
const G_W = array<f32, 5>(0.227027, 0.194594, 0.121621, 0.054054, 0.016216);

@compute @workgroup_size(8, 8, 1)
fn cs_blur_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let hw = (vec2<i32>(camera.resolution) + vec2<i32>(1)) / 2;
    let p = vec2<i32>(gid.xy);
    if (p.x >= hw.x || p.y >= hw.y) { return; }
    var c = textureLoad(bloom_in, p, 0).rgb * G_W[0];
    for (var i = 1; i < 5; i = i + 1) {
        let o = vec2<i32>(i, 0);
        c += textureLoad(bloom_in, clamp(p + o, vec2<i32>(0), hw - 1), 0).rgb * G_W[i];
        c += textureLoad(bloom_in, clamp(p - o, vec2<i32>(0), hw - 1), 0).rgb * G_W[i];
    }
    textureStore(bloom_out, p, vec4<f32>(c, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn cs_blur_v(@builtin(global_invocation_id) gid: vec3<u32>) {
    let hw = (vec2<i32>(camera.resolution) + vec2<i32>(1)) / 2;
    let p = vec2<i32>(gid.xy);
    if (p.x >= hw.x || p.y >= hw.y) { return; }
    var c = textureLoad(bloom_in, p, 0).rgb * G_W[0];
    for (var i = 1; i < 5; i = i + 1) {
        let o = vec2<i32>(0, i);
        c += textureLoad(bloom_in, clamp(p + o, vec2<i32>(0), hw - 1), 0).rgb * G_W[i];
        c += textureLoad(bloom_in, clamp(p - o, vec2<i32>(0), hw - 1), 0).rgb * G_W[i];
    }
    textureStore(bloom_out, p, vec4<f32>(c, 1.0));
}

// ---- final: bloom add + ACES + grade + vignette -------------------------

// Narkowicz ACES fit: filmic shoulder, keeps saturated highlights alive.
fn aces(x: vec3<f32>) -> vec3<f32> {
    return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
                 vec3<f32>(0.0), vec3<f32>(1.0));
}

@compute @workgroup_size(8, 8, 1)
fn cs_post(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = vec2<i32>(camera.resolution);
    let p = vec2<i32>(gid.xy);
    if (p.x >= res.x || p.y >= res.y) { return; }
    let hdr4 = textureLoad(hdr_tex, p, 0);
    var hdr = hdr4.rgb;
    // Bilinear-upsampled bloom, additive with a restrained weight.
    let uv = (vec2<f32>(p) + vec2<f32>(0.5)) / camera.resolution;
    let bloom = textureSampleLevel(bloom_in, lin_sampler, uv, 0.0).rgb;
    hdr += bloom * 0.22;

    // Slight exposure lift into the filmic curve.
    var c = aces(hdr * 1.15);

    // Grade: cool lifted shadows against warm highlights (the two-tone
    // cinema look), then a gentle saturation boost.
    let luma = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    let shadow_w = (1.0 - luma) * (1.0 - luma);
    let high_w = luma * luma;
    c += shadow_w * vec3<f32>(-0.012, 0.004, 0.022);
    c *= vec3<f32>(1.0) + high_w * vec3<f32>(0.05, 0.02, -0.04);
    c = mix(vec3<f32>(luma), c, 1.12);

    // Alpha carries the grass-blade marker through to the TAA resolve.
    textureStore(ldr_out, p, vec4<f32>(clamp(c, vec3<f32>(0.0), vec3<f32>(1.0)), hdr4.a));
}
