// Temporal anti-aliasing resolve pass.
//
// Blends the freshly-raymarched frame with the accumulated history, gated by a
// 3x3 neighbourhood colour clamp so disocclusions / motion don't ghost. While
// the camera is static the main pass jitters its rays sub-pixel each frame and
// this accumulation converges to an anti-aliased image (small voxels stop
// shimmering). On motion the host sets taa_blend = 0, so the current frame
// passes through unchanged (sharp, no smearing).

// Camera uniform layout is defined in shaders/common.wgsl (shared prelude).

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var current_tex: texture_2d<f32>;
@group(0) @binding(2) var history_tex: texture_2d<f32>;
@group(0) @binding(3) var resolve_out: texture_storage_2d<rgba8unorm, write>;
// Hit G-buffer (xyz = hit pos rel world_origin, w != 0 → valid terrain hit) so
// the history can be reprojected by motion instead of reset on camera movement.
@group(0) @binding(4) var gbuffer: texture_2d<f32>;

@compute @workgroup_size(8, 8, 1)
fn cs_taa(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = vec2<i32>(camera.resolution);
    let p = vec2<i32>(i32(gid.x), i32(gid.y));
    if (p.x >= res.x || p.y >= res.y) { return; }

    let cur4 = textureLoad(current_tex, p, 0);
    let cur = cur4.rgb;
    // Grass-blade pixels (alpha marker 0): thin geometry sways every frame,
    // so reprojected terrain history behind a blade bleeds THROUGH it at
    // high blend weights - blades looked translucent. Same-pixel history
    // at a much lower weight keeps them solid with some temporal smoothing.
    let is_blade = cur4.a < 0.5;

    // Hard reset (first frame / after resize → taa_blend 0) or no valid
    // reprojection basis (origin shifted on a chunk cross → reproject_lighting 0)
    // → pass the current frame straight through (sharp, no smear).
    let out_a = select(1.0, 0.0, is_blade);
    if (camera.taa_blend <= 0.0 || camera.reproject_lighting < 0.5) {
        textureStore(resolve_out, p, vec4<f32>(cur, out_a));
        return;
    }

    // Reproject this pixel's surface into the previous frame to find where its
    // history lives (full-reprojection TAA — accumulates across motion instead
    // of resetting). Terrain pixels reproject by their G-buffer world position;
    // everything else falls back to the same pixel.
    var hp = p; // history pixel
    let g = textureLoad(gbuffer, p, 0);
    // Sentinel position (1e9) marks sky/foliage/water — only terrain reprojects.
    if (!is_blade && camera.reproject_lighting > 0.5 && g.x < 1e8) {
        let abs_pos = g.xyz + vec3<f32>(camera.world_origin);
        let d = abs_pos - camera.prev_origin;
        let pz = dot(d, camera.prev_forward);
        if (pz > 0.01) {
            let aspect = camera.resolution.x / camera.resolution.y;
            let ndc = vec2<f32>(
                dot(d, camera.prev_right) / (pz * camera.tan_half_fov * aspect),
                dot(d, camera.prev_up) / (pz * camera.tan_half_fov));
            let uvp = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
            if (uvp.x >= 0.0 && uvp.x < 1.0 && uvp.y >= 0.0 && uvp.y < 1.0) {
                hp = vec2<i32>(uvp * camera.resolution);
            }
        }
    }

    // 3x3 neighbourhood colour bounds of the current frame.
    var mn = cur;
    var mx = cur;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let q = clamp(p + vec2<i32>(dx, dy), vec2<i32>(0), res - vec2<i32>(1));
            let n = textureLoad(current_tex, q, 0).rgb;
            mn = min(mn, n);
            mx = max(mx, n);
        }
    }

    // Sample the REPROJECTED history, clamp into the current neighbourhood
    // (rejects ghosting/disocclusion) and blend.
    // Same-kind accumulation rule: a pixel only blends with history of its
    // own kind (the resolve stores the blade marker in alpha). Blade over
    // blade accumulates - temporal AA along the blade body; any coverage
    // flip (blade moved on or off this pixel) renders PURE current frame,
    // so terrain history can never bleed through a blade, by construction.
    let hist4 = textureLoad(history_tex, hp, 0);
    let hist_blade = hist4.a < 0.5;
    var blend = camera.taa_blend;
    if (is_blade != hist_blade) {
        blend = 0.0;
    } else if (is_blade) {
        blend = camera.taa_blend * 0.6;
    }
    let hist_clamped = clamp(hist4.rgb, mn, mx);
    let resolved = mix(cur, hist_clamped, blend);
    textureStore(resolve_out, p, vec4<f32>(resolved, out_a));
}
