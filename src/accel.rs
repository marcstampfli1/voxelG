// Hardware ray-tracing acceleration structure built from the voxel world.
//
// Coarse-to-fine: one AABB per NON-EMPTY brick goes into a bottom-level
// acceleration structure (BLAS); the GPU's RT cores traverse that BVH to skip
// empty space at hardware speed, and a tiny per-brick DDA (in the ray-query
// shader) resolves the exact voxel inside a candidate brick. This is the voxel
// mechanism proven in the rt-spike, now driven by real world data.
//
// AABBs are in WINDOW-LOCAL voxel space (brick array coords * BRICK_DIM), the
// same space the software DDA marches after rebasing by camera.world_origin, so
// ray-query callers transform the ray by world_origin exactly as trace() does.
//
// primitive_index (0..aabb_count over the packed non-empty bricks) is NOT the
// brick index (empty bricks are skipped), so `brick_map[primitive_index]` maps
// back to the brick index for the in-shader occupancy/material lookup.

use wgpu::util::DeviceExt;

use crate::voxel::{brick_idx, World, BRICK_DIM, WORLD_BRICKS_X, WORLD_BRICKS_Y, WORLD_BRICKS_Z};

/// One packed AABB primitive: min then max (two `vec3<f32>`), padded to 32 bytes
/// (stride must be a multiple of 8 and at least 24; 32 keeps it 16-byte aligned
/// for the parallel storage-buffer read in the shader).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuAabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub _pad: [f32; 2],
}

/// The world's acceleration structure plus the primitive->brick map.
pub struct WorldAccel {
    pub blas: wgpu::Blas,
    pub tlas: wgpu::Tlas,
    /// primitive_index -> brick_idx (u32), one entry per non-empty brick.
    pub brick_map: wgpu::Buffer,
    /// The packed AABBs, kept so the ray-query shader can read a candidate
    /// brick's window-local min directly (no brick-index decode needed).
    pub aabb_buf: wgpu::Buffer,
    pub aabb_count: u32,
}

/// True if the adapter can do the hardware ray-tracing we need.
pub fn adapter_supports_rt(adapter: &wgpu::Adapter) -> bool {
    adapter
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
}

/// Enumerate non-empty bricks into AABBs + the primitive->brick map, then build
/// the BLAS and a single-instance TLAS. The device MUST have been created with
/// `EXPERIMENTAL_RAY_QUERY` + `ExperimentalFeatures::enabled()` and the
/// acceleration-structure limits (see `rt_device_limits`).
pub fn build_world_accel(device: &wgpu::Device, queue: &wgpu::Queue, world: &World) -> WorldAccel {
    let mut aabbs: Vec<GpuAabb> = Vec::new();
    let mut brick_map: Vec<u32> = Vec::new();
    for bz in 0..WORLD_BRICKS_Z {
        for by in 0..WORLD_BRICKS_Y {
            for bx in 0..WORLD_BRICKS_X {
                let bi = brick_idx(bx, by, bz);
                if world.bricks[bi as usize].occupancy != 0 {
                    let mn = [
                        (bx * BRICK_DIM) as f32,
                        (by * BRICK_DIM) as f32,
                        (bz * BRICK_DIM) as f32,
                    ];
                    let e = BRICK_DIM as f32;
                    aabbs.push(GpuAabb {
                        min: mn,
                        max: [mn[0] + e, mn[1] + e, mn[2] + e],
                        _pad: [0.0, 0.0],
                    });
                    brick_map.push(bi);
                }
            }
        }
    }
    // An empty world still needs a valid (>=1 primitive) BLAS; a degenerate
    // AABB far outside the window is never hit.
    if aabbs.is_empty() {
        aabbs.push(GpuAabb {
            min: [-1.0e9, -1.0e9, -1.0e9],
            max: [-1.0e9 + 1.0, -1.0e9 + 1.0, -1.0e9 + 1.0],
            _pad: [0.0, 0.0],
        });
        brick_map.push(0);
    }
    let count = aabbs.len() as u32;

    let aabb_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("world aabb primitives"),
        contents: bytemuck::cast_slice(&aabbs),
        usage: wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::STORAGE,
    });
    let brick_map_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("aabb primitive -> brick map"),
        contents: bytemuck::cast_slice(&brick_map),
        usage: wgpu::BufferUsages::STORAGE,
    });

    let size_desc = wgpu::BlasAABBGeometrySizeDescriptor {
        primitive_count: count,
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("world blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::AABBs {
            descriptors: vec![size_desc.clone()],
        },
    );
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("world tlas"),
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        max_instances: 1,
    });
    // Identity: AABBs are already in window-local voxel space; the ray is
    // rebased by world_origin on the shader side.
    let identity: [f32; 12] = [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
    ];
    tlas[0] = Some(wgpu::TlasInstance::new(&blas, identity, 0, 0xff));

    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("build world accel"),
    });
    enc.build_acceleration_structures(
        std::iter::once(&wgpu::BlasBuildEntry {
            blas: &blas,
            geometry: wgpu::BlasGeometries::AabbGeometries(vec![wgpu::BlasAabbGeometry {
                size: &size_desc,
                stride: std::mem::size_of::<GpuAabb>() as wgpu::BufferAddress,
                aabb_buffer: &aabb_buf,
                primitive_offset: 0,
            }]),
        }),
        std::iter::once(&tlas),
    );
    queue.submit(Some(enc.finish()));

    WorldAccel {
        blas,
        tlas,
        brick_map: brick_map_buf,
        aabb_buf,
        aabb_count: count,
    }
}

#[cfg(test)]
mod tests {
    //! Isolated validation of the hardware-RT voxel resolve BEFORE it is wired
    //! into any render pass. The RT core traverses the world BLAS to candidate
    //! brick AABBs and `accel_probe.wgsl`'s per-brick DDA resolves the exact
    //! first solid voxel; we check that against (a) the analytic surface height
    //! of a crafted flat floor and (b) the trusted CPU picking raycaster over a
    //! richer scene. Skips cleanly when no RT-capable GPU is present.
    use super::*;
    use crate::raycast::raycast;
    use crate::voxel::{World, MAT_STONE};
    use glam::{IVec3, Vec3};
    use wgpu::util::DeviceExt;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct GpuRay {
        o: [f32; 3],
        _p0: f32,
        d: [f32; 3],
        _p1: f32,
    }
    impl GpuRay {
        fn new(o: Vec3, d: Vec3) -> Self {
            let d = d.normalize();
            GpuRay { o: [o.x, o.y, o.z], _p0: 0.0, d: [d.x, d.y, d.z], _p1: 0.0 }
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Default, Debug)]
    struct GpuHit {
        t: f32,
        hit: u32,
        vx: i32,
        vy: i32,
        vz: i32,
        mat: u32,
        _p0: u32,
        _p1: u32,
    }

    /// Headless device WITH the experimental ray-query feature + AS limits. The
    /// exact incantation proven in the rt-spike (`adapter.limits()` supplies the
    /// acceleration-structure limits, which default to 0). Returns None when the
    /// adapter can't ray-trace, so the suite stays green on non-RT machines.
    fn rt_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_without_display_handle_from_env()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .ok()?;
        if !adapter_supports_rt(&adapter) {
            return None;
        }
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("rt accel test device"),
            required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            trace: wgpu::Trace::Off,
        }))
        .ok()
    }

    /// Build the accel for `world`, cast `rays` through `accel_probe.wgsl`, and
    /// read the resolved hits back. Rays and hits are in window-local voxel
    /// space (== world space for an unshifted world).
    fn probe(device: &wgpu::Device, queue: &wgpu::Queue, world: &World, rays: &[GpuRay]) -> Vec<GpuHit> {
        let accel = build_world_accel(device, queue, world);

        let bricks_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("probe bricks"),
            contents: bytemuck::cast_slice(&world.bricks),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let rays_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("probe rays"),
            contents: bytemuck::cast_slice(rays),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let hits_bytes = (rays.len() * std::mem::size_of::<GpuHit>()) as u64;
        let hits_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe hits"),
            size: hits_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("probe readback"),
            size: hits_bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("accel_probe"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/accel_probe.wgsl").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("accel_probe"),
            layout: None,
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("accel_probe bg"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: bricks_buf.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::AccelerationStructure(&accel.tlas),
                },
                wgpu::BindGroupEntry { binding: 2, resource: accel.brick_map.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: accel.aabb_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: rays_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: hits_buf.as_entire_binding() },
            ],
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("probe dispatch"),
        });
        {
            let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("probe"),
                timestamp_writes: None,
            });
            cp.set_pipeline(&pipeline);
            cp.set_bind_group(0, &bg, &[]);
            let groups = (rays.len() as u32).div_ceil(64);
            cp.dispatch_workgroups(groups, 1, 1);
        }
        enc.copy_buffer_to_buffer(&hits_buf, 0, &readback, 0, hits_bytes);
        queue.submit(Some(enc.finish()));

        let slice = readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range().unwrap();
        let hits: Vec<GpuHit> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback.unmap();
        hits
    }

    /// A flat stone floor whose top solid layer is world y=60 (so its top face
    /// is the plane y=61), filling x,z in [x0,x0+n) at world origin 0.
    fn flat_floor(x0: u32, z0: u32, n: u32) -> World {
        let mut world = World::new();
        for z in z0..z0 + n {
            for x in x0..x0 + n {
                for y in 57..=60 {
                    world.set_voxel(x, y, z, MAT_STONE);
                }
            }
        }
        world
    }

    #[test]
    fn accel_flat_floor_surface_height() {
        let Some((device, queue)) = rt_device() else {
            eprintln!("accel_flat_floor_surface_height: no RT adapter, skipping");
            return;
        };
        let world = flat_floor(16, 16, 48); // covers x,z in [16,64)

        // Near-vertical rays (tilted off-axis so the DDA never divides by zero)
        // from high above a grid of floor cells. Each must hit the y=60 layer,
        // entering its top face at world y=61.
        let mut rays = Vec::new();
        let mut expect_cell = Vec::new();
        for gz in 0..6 {
            for gx in 0..6 {
                let px = 22.0 + gx as f32 * 6.0;
                let pz = 22.0 + gz as f32 * 6.0;
                let eye = Vec3::new(px, 120.0, pz);
                let dir = Vec3::new(0.03, -1.0, 0.017);
                rays.push(GpuRay::new(eye, dir));
                expect_cell.push((px, pz));
            }
        }

        let hits = probe(&device, &queue, &world, &rays);
        let mut checked = 0;
        for (i, h) in hits.iter().enumerate() {
            assert_eq!(h.hit, 1, "ray {i} missed the floor entirely: {h:?}");
            assert_eq!(h.vy, 60, "ray {i} hit y={} not the top floor layer 60", h.vy);
            // Reconstruct the hit point and check it lands on the top face plane.
            let (px, pz) = expect_cell[i];
            let eye = Vec3::new(px, 120.0, pz);
            let d = Vec3::new(0.03, -1.0, 0.017).normalize();
            let hit_y = eye.y + h.t * d.y;
            assert!(
                (hit_y - 61.0).abs() < 0.05,
                "ray {i}: RT surface y={hit_y:.4} not the floor top face 61.0 (t={})",
                h.t
            );
            checked += 1;
        }
        assert_eq!(checked, rays.len(), "every ray must have been validated");
        eprintln!("accel_flat_floor_surface_height: {checked} rays hit y=61 exactly");
    }

    fn local_safe_inv(x: f32) -> f32 {
        if x.abs() < 1e-8 { 1e30 } else { 1.0 / x }
    }

    /// Ray parameter where `o + t*d` enters voxel V's unit AABB [V, V+1].
    fn ray_box_entry(o: Vec3, d: Vec3, v: [i32; 3]) -> f32 {
        let inv = Vec3::new(local_safe_inv(d.x), local_safe_inv(d.y), local_safe_inv(d.z));
        let vmin = Vec3::new(v[0] as f32, v[1] as f32, v[2] as f32);
        let vmax = vmin + Vec3::ONE;
        let t0 = (vmin - o) * inv;
        let t1 = (vmax - o) * inv;
        t0.min(t1).max_element().max(0.0)
    }

    fn near_int(x: f32, eps: f32) -> bool {
        (x - x.round()).abs() < eps
    }

    /// How many coordinates of P sit on a voxel-lattice plane. A CLEAN face hit
    /// touches exactly one (the face the ray crossed); two or more means the ray
    /// pierced an edge or corner, where the first-solid-voxel is genuinely
    /// ambiguous between two independent float DDAs and no exact match is
    /// required (the software renderer splits these the same way).
    fn lattice_touch(p: Vec3, eps: f32) -> u32 {
        near_int(p.x, eps) as u32 + near_int(p.y, eps) as u32 + near_int(p.z, eps) as u32
    }

    #[test]
    fn accel_matches_cpu_raycast() {
        let Some((device, queue)) = rt_device() else {
            eprintln!("accel_matches_cpu_raycast: no RT adapter, skipping");
            return;
        };

        // Floor plus 3x3 stone pillars of varying height: downward, lateral and
        // grazing rays exercise multi-brick traversal and in-brick resolution.
        let mut world = flat_floor(8, 8, 56); // x,z in [8,64)
        let pillars = [(20u32, 24u32, 90u32), (40, 40, 100), (52, 18, 76), (30, 50, 84)];
        for &(px, pz, top) in &pillars {
            for y in 61..top {
                for dz in 0..3 {
                    for dx in 0..3 {
                        world.set_voxel(px + dx, y, pz + dz, MAT_STONE);
                    }
                }
            }
        }

        let mut rays = Vec::new();
        // Downward fans from high eyes onto the floor and pillar tops.
        let high_eyes = [
            Vec3::new(6.0, 96.0, 6.0),
            Vec3::new(70.0, 110.0, 12.0),
            Vec3::new(34.0, 130.0, 34.0),
            Vec3::new(12.0, 74.0, 60.0),
        ];
        for &eye in &high_eyes {
            for gz in 0..8 {
                for gx in 0..8 {
                    let target = Vec3::new(12.0 + gx as f32 * 6.0, 62.0, 12.0 + gz as f32 * 6.0);
                    rays.push(GpuRay::new(eye, target - eye));
                }
            }
        }
        // Near-horizontal fans into each pillar's -x side face (lateral,
        // multi-brick), aimed across the 3-wide face at mid heights.
        for &(px, pz, top) in &pillars {
            let eye = Vec3::new(px as f32 - 16.0, (61 + top) as f32 * 0.5, pz as f32 + 1.5);
            for gy in 0..3 {
                for gz in 0..3 {
                    let target = Vec3::new(
                        px as f32,
                        61.0 + (top - 61) as f32 * (0.25 + gy as f32 * 0.25),
                        pz as f32 + 0.5 + gz as f32,
                    );
                    rays.push(GpuRay::new(eye, target - eye));
                }
            }
        }

        let hits = probe(&device, &queue, &world, &rays);

        const EPS: f32 = 0.02;
        let mut clean_assert = 0usize; // clean face hits where RT equals CPU
        let mut clean_floor = 0usize;
        let mut clean_pillar = 0usize;
        let mut grazing = 0usize; // inherently ambiguous edge/corner rays
        for (i, h) in hits.iter().enumerate() {
            let eye = Vec3::new(rays[i].o[0], rays[i].o[1], rays[i].o[2]);
            let dir = Vec3::new(rays[i].d[0], rays[i].d[1], rays[i].d[2]).normalize();
            let oracle = raycast(eye, dir, &world, IVec3::ZERO);
            let rt_v = [h.vx, h.vy, h.vz];
            // Hit points on each side (None where that side missed).
            let cpu_p = oracle
                .as_ref()
                .map(|pick| eye + ray_box_entry(eye, dir, pick.voxel) * dir);
            let rt_p = (h.hit == 1).then(|| eye + h.t * dir);
            let cpu_graze = cpu_p.is_some_and(|p| lattice_touch(p, EPS) >= 2);
            let rt_graze = rt_p.is_some_and(|p| lattice_touch(p, EPS) >= 2);

            // Both missing is agreement (ray left the world / hit nothing).
            if h.hit == 0 && oracle.is_none() {
                continue;
            }
            let agree = h.hit == 1 && oracle.as_ref().is_some_and(|pk| pk.voxel == rt_v);
            if agree {
                // Only count as validated coverage when the shared hit is a
                // clean face (unambiguous); grazing agreements don't prove much.
                if cpu_graze {
                    grazing += 1;
                } else {
                    clean_assert += 1;
                    if rt_v[1] <= 60 { clean_floor += 1; } else { clean_pillar += 1; }
                }
                continue;
            }
            // Disagreement: tolerable ONLY if the ray grazes a voxel edge/corner
            // on either side (two independent float DDAs split such rays, as does
            // the software renderer). A disagreement where BOTH hit points are
            // clean faces is a real resolve bug.
            if cpu_graze || rt_graze {
                grazing += 1;
                continue;
            }
            panic!(
                "ray {i}: clean-face disagreement RT {}{:?} vs CPU {}{:?} (dir {:?}, cpu_p {:?}, rt_p {:?})",
                if h.hit == 1 { "hit " } else { "MISS " }, rt_v,
                if oracle.is_some() { "hit " } else { "MISS " },
                oracle.as_ref().map(|p| p.voxel).unwrap_or_default(),
                dir, cpu_p, rt_p
            );
        }

        // The resolve is only validated if plenty of CLEAN hits were checked,
        // covering both the floor and pillar side faces; and grazing splits must
        // stay a minority (a systematic half-voxel offset would make everything
        // look grazing).
        assert!(clean_assert > rays.len() / 3, "too few clean hits checked: {clean_assert}/{}", rays.len());
        assert!(clean_floor > 0 && clean_pillar > 0, "clean hits must cover floor ({clean_floor}) and pillars ({clean_pillar})");
        assert!(grazing < clean_assert, "grazing majority ({grazing} >= {clean_assert}) - suspect a systematic offset");
        eprintln!(
            "accel_matches_cpu_raycast: {clean_assert} clean hits match CPU exactly (floor {clean_floor}, pillar {clean_pillar}); {grazing} grazing rays tolerated over {} total",
            rays.len()
        );
    }

    #[test]
    fn accel_shadow_occlusion_matches_cpu() {
        // Shadow rays are any-hit occlusion tests: a surface point is in shadow
        // iff the ray toward the sun hits ANY solid voxel. This validates that
        // semantic (the same RT primitive the render passes will use for
        // shadows) against the CPU raycaster, BEFORE wiring it into the shader.
        let Some((device, queue)) = rt_device() else {
            eprintln!("accel_shadow_occlusion_matches_cpu: no RT adapter, skipping");
            return;
        };

        // Floor with a raised roof slab over a central patch: floor points under
        // the roof are shadowed from an overhead sun, open points are lit.
        let mut world = flat_floor(8, 8, 56); // x,z in [8,64), top y=60
        let (rx0, rx1, rz0, rz1) = (24u32, 44u32, 24u32, 44u32);
        for z in rz0..rz1 {
            for x in rx0..rx1 {
                world.set_voxel(x, 80, z, MAT_STONE);
            }
        }

        // Sun mostly overhead so the roof's shadow footprint ~= its own extent
        // (a steep-ish tilt keeps the rays off the pure +Y axis).
        let sun = Vec3::new(0.16, 0.95, 0.11).normalize();

        // Sample floor-cell centres; shadow ray starts just above the top face.
        let mut samples = Vec::new(); // (origin, under_roof_interior, open_clear)
        let mut rays = Vec::new();
        for z in 10..62 {
            for x in 10..62 {
                let o = Vec3::new(x as f32 + 0.5, 61.02, z as f32 + 0.5);
                // Where the sun ray crosses the roof plane (y=80), to classify
                // decisive samples with a margin from the roof footprint edge.
                let s = (80.0 - o.y) / sun.y;
                let hx = o.x + sun.x * s;
                let hz = o.z + sun.z * s;
                let m = 1.5;
                let under = hx > rx0 as f32 + m && hx < rx1 as f32 - m && hz > rz0 as f32 + m && hz < rz1 as f32 - m;
                let open = hx < rx0 as f32 - m || hx > rx1 as f32 + m || hz < rz0 as f32 - m || hz > rz1 as f32 + m;
                samples.push((o, under, open));
                rays.push(GpuRay::new(o, sun));
            }
        }

        let hits = probe(&device, &queue, &world, &rays);

        let mut decisive = 0usize;
        let mut occluded_ok = 0usize;
        let mut lit_ok = 0usize;
        let mut disagree = 0usize;
        for (i, h) in hits.iter().enumerate() {
            let (o, under, open) = samples[i];
            let rt_occ = h.hit == 1;
            let cpu_occ = raycast(o, sun, &world, IVec3::ZERO).is_some();
            // Agreement between the two occlusion oracles is required everywhere;
            // the under/open flags additionally prove the scene really casts a
            // shadow (not "all lit" or "all dark").
            if rt_occ != cpu_occ {
                if under || open {
                    disagree += 1;
                    if disagree <= 8 {
                        eprintln!("sample {i} at {o:?}: RT occ={rt_occ} CPU occ={cpu_occ} (under={under} open={open})");
                    }
                }
                continue;
            }
            if under && rt_occ {
                decisive += 1;
                occluded_ok += 1;
            } else if open && !rt_occ {
                decisive += 1;
                lit_ok += 1;
            }
        }

        assert_eq!(disagree, 0, "{disagree} decisive shadow samples disagree between RT and CPU");
        assert!(occluded_ok > 20, "too few shadowed samples under the roof: {occluded_ok}");
        assert!(lit_ok > 20, "too few lit samples in the open: {lit_ok}");
        eprintln!(
            "accel_shadow_occlusion_matches_cpu: {decisive} decisive samples agree ({occluded_ok} shadowed, {lit_ok} lit); 0 disagreements"
        );
    }
}
