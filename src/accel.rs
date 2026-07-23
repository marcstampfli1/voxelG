// Hardware ray-tracing acceleration structure built from the voxel world.
//
// Coarse-to-fine: one AABB per NON-EMPTY TILE (16^3 voxels) goes into a
// bottom-level acceleration structure (BLAS); the RT cores traverse that BVH to
// skip empty space at hardware speed, a brick-grid DDA refines the candidate
// tile to its present bricks (gated by the tile_mask bit - the streaming-
// correct presence test), and the shared per-brick DDA resolves the exact
// voxel. Tile granularity keeps the BVH at <=16384 primitives, so a streaming
// rebuild is sub-millisecond and runs every time the world changes.
//
// AABBs are in WINDOW-LOCAL voxel space: local = world_voxel - world_origin, a
// CONTIGUOUS [0, WORLD_VOXELS) box, which is the single linear space the BVH
// lives in. A ray-query caller rebases its world-space ray by world_origin
// (`o_local = origin - world_origin`) exactly as the software DDA marches world
// coords and folds them to the window.
//
// The voxel STORAGE, however, is toroidal: the AABB sits at the window-local
// tile position while `brick_map[primitive_index]` holds the STORAGE tile
// index; the in-shader `rt_tile_brick` maps (tile, brick offset) to the storage
// brick for occupancy/material reads. world_origin is always tile-aligned
// (chunk streaming shifts by 32 voxels, y never streams).

use crate::voxel::{World, WORLD_TILES_X, WORLD_TILES_Y, WORLD_TILES_Z};

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
///
/// Buffers and the BLAS are allocated once at `capacity` primitives and REUSED
/// across streaming rebuilds (`rebuild_world_accel`): the CPU re-enumerates the
/// non-empty bricks (mask-driven, sub-ms), DMAs only the changed prefix, and the
/// GPU re-builds the BVH in place. No per-rebuild allocation, and the render
/// bind group stays valid unless the capacity has to grow.
pub struct WorldAccel {
    pub blas: wgpu::Blas,
    pub tlas: wgpu::Tlas,
    /// primitive_index -> STORAGE tile index (u32), one entry per non-empty
    /// tile. The shader decodes brick/voxel storage coords from it; the
    /// window-local position comes from the AABB min in `aabb_buf`.
    pub brick_map: wgpu::Buffer,
    /// The packed AABBs, kept so the ray-query shader can read a candidate
    /// brick's window-local min directly (no brick-index decode needed).
    pub aabb_buf: wgpu::Buffer,
    pub aabb_count: u32,
    /// Primitive capacity the buffers + BLAS were created for. The BLAS is always
    /// BUILT with `capacity` primitives; entries beyond `aabb_count` are
    /// degenerate far-away boxes no ray can hit, so the build size never has to
    /// re-negotiate with the created size.
    pub capacity: u32,
    /// How many entries of `aabb_buf` currently hold REAL (non-degenerate) data;
    /// a shrink only needs to overwrite `[count, high_water)` with degenerates.
    high_water: u32,
    // CPU-side scratch reused across rebuilds (no per-rebuild allocation).
    aabb_scratch: Vec<GpuAabb>,
    map_scratch: Vec<u32>,
}

/// A degenerate AABB far outside any reachable ray range; pads the BLAS build
/// range beyond the live primitive count.
const DEGENERATE_AABB: GpuAabb = GpuAabb {
    min: [-1.0e9, -1.0e9, -1.0e9],
    max: [-1.0e9 + 1.0, -1.0e9 + 1.0, -1.0e9 + 1.0],
    _pad: [0.0, 0.0],
};

/// True if the adapter can do the hardware ray-tracing we need.
pub fn adapter_supports_rt(adapter: &wgpu::Adapter) -> bool {
    adapter
        .features()
        .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
}

/// Enumerate the non-empty BRICKS into AABBs + the primitive->brick map,
/// MASK-DRIVEN from the tile occupancy words alone (no voxel reads): one AABB
/// per set brick bit. Per-brick primitives keep the hardware BVH doing the
/// whole empty-space hierarchy - benchmarked ~1.3-1.4x over software, where a
/// tile-level BVH (in-shader brick marching) measured ~1.0x. The streaming
/// cost of the bigger build is paid OFF the frame thread (see the async
/// rebuild worker in renderer.rs); `tile_mask` arrives as a snapshot so this
/// can run on any thread. AABBs sit at WINDOW-LOCAL positions; `map` records
/// STORAGE brick indices.
pub(crate) fn enumerate_non_empty(
    tile_mask: &[u64],
    world_origin: glam::IVec3,
    aabbs: &mut Vec<GpuAabb>,
    map: &mut Vec<u32>,
) {
    aabbs.clear();
    map.clear();
    let (wob_x, wob_z) = (world_origin.x / 4, world_origin.z / 4);
    let (nbx, nbz) = ((WORLD_TILES_X * 4) as i32, (WORLD_TILES_Z * 4) as i32);
    let (tx_n, ty_n) = (WORLD_TILES_X as usize, WORLD_TILES_Y as usize);
    let e = 4.0f32;
    for (ti, &mask) in tile_mask.iter().enumerate() {
        if mask == 0 {
            continue;
        }
        let tx = (ti % tx_n) as i32;
        let ty = ((ti / tx_n) % ty_n) as i32;
        let tz = (ti / (tx_n * ty_n)) as i32;
        let mut m = mask;
        while m != 0 {
            let b = m.trailing_zeros() as i32;
            m &= m - 1;
            // brick_bit_in_tile(lx, ly, lz) = lx + lz*4 + ly*16
            let (lx, ly, lz) = (b & 3, (b >> 4) & 3, (b >> 2) & 3);
            let (sbx, sby, sbz) = (tx * 4 + lx, ty * 4 + ly, tz * 4 + lz);
            let lbx = (sbx - wob_x).rem_euclid(nbx);
            let lbz = (sbz - wob_z).rem_euclid(nbz);
            let mn = [(lbx * 4) as f32, (sby * 4) as f32, (lbz * 4) as f32];
            aabbs.push(GpuAabb {
                min: mn,
                max: [mn[0] + e, mn[1] + e, mn[2] + e],
                _pad: [0.0, 0.0],
            });
            map.push((sbx + sby * nbx + sbz * nbx * (WORLD_TILES_Y as i32 * 4)) as u32);
        }
    }
}

/// Record the BLAS + TLAS build commands. The BLAS is always built with
/// `capacity` primitives (the size it was created with) - the live prefix holds
/// the real bricks, the tail degenerate far-away boxes - so the build size never
/// has to re-negotiate with the created size.
fn encode_accel_build(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    blas: &wgpu::Blas,
    tlas: &wgpu::Tlas,
    aabb_buf: &wgpu::Buffer,
    capacity: u32,
) {
    let size_desc = wgpu::BlasAABBGeometrySizeDescriptor {
        primitive_count: capacity,
        flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
    };
    let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("build world accel"),
    });
    enc.build_acceleration_structures(
        std::iter::once(&wgpu::BlasBuildEntry {
            blas,
            geometry: wgpu::BlasGeometries::AabbGeometries(vec![wgpu::BlasAabbGeometry {
                size: &size_desc,
                stride: std::mem::size_of::<GpuAabb>() as wgpu::BufferAddress,
                aabb_buffer: aabb_buf,
                primitive_offset: 0,
            }]),
        }),
        std::iter::once(tlas),
    );
    queue.submit(Some(enc.finish()));
}

/// DMA the enumerated primitives into the persistent buffers (degenerate-padding
/// any shrink delta so the fixed-size BLAS build never sees stale reals), then
/// re-build the BLAS + TLAS on the GPU. Shared by creation and every streaming
/// rebuild - ONE upload path. Takes ownership of the vecs and parks them back in
/// the accel as scratch for the next rebuild.
fn upload_and_build(
    accel: &mut WorldAccel,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut aabbs: Vec<GpuAabb>,
    map: Vec<u32>,
) {
    let count = aabbs.len() as u32;
    debug_assert!(count <= accel.capacity);
    let t0 = std::time::Instant::now();
    let write_len = count.max(accel.high_water) as usize;
    aabbs.resize(write_len, DEGENERATE_AABB);
    if write_len > 0 {
        queue.write_buffer(&accel.aabb_buf, 0, bytemuck::cast_slice(&aabbs[..write_len]));
    }
    if count > 0 {
        queue.write_buffer(&accel.brick_map, 0, bytemuck::cast_slice(&map[..count as usize]));
    }
    let t_write = t0.elapsed();
    encode_accel_build(device, queue, &accel.blas, &accel.tlas, &accel.aabb_buf, accel.capacity);
    let t_encode = t0.elapsed() - t_write;
    if (t_write + t_encode).as_secs_f64() > 0.002 {
        log::info!(
            "  upload_and_build: write {:.2} ms ({} entries), encode+submit {:.2} ms",
            t_write.as_secs_f64() * 1000.0,
            write_len,
            t_encode.as_secs_f64() * 1000.0
        );
    }
    accel.aabb_count = count;
    accel.high_water = count;
    accel.aabb_scratch = aabbs;
    accel.map_scratch = map;
}

/// Create the capacity-sized GPU objects and build them from `aabbs`/`map`
/// (whose length is the live count; the tail up to `capacity` is degenerate).
fn create_accel_at_capacity(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    aabbs: Vec<GpuAabb>,
    map: Vec<u32>,
    capacity: u32,
) -> WorldAccel {
    let aabb_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("world aabb primitives"),
        size: capacity as u64 * std::mem::size_of::<GpuAabb>() as u64,
        usage: wgpu::BufferUsages::BLAS_INPUT
            | wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let brick_map_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("aabb primitive -> brick map"),
        size: capacity as u64 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let blas = device.create_blas(
        &wgpu::CreateBlasDescriptor {
            label: Some("world blas"),
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        },
        wgpu::BlasGeometrySizeDescriptors::AABBs {
            descriptors: vec![wgpu::BlasAABBGeometrySizeDescriptor {
                primitive_count: capacity,
                flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
            }],
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

    // high_water = capacity: the freshly-created buffer is unwritten, so the
    // first upload must cover the whole range (reals + degenerate tail).
    let mut accel = WorldAccel {
        blas,
        tlas,
        brick_map: brick_map_buf,
        aabb_buf,
        aabb_count: 0,
        capacity,
        high_water: capacity,
        aabb_scratch: Vec::new(),
        map_scratch: Vec::new(),
    };
    upload_and_build(&mut accel, device, queue, aabbs, map);
    accel
}

/// Capacity policy: half again over the live count so ordinary streaming churn
/// never grows, rounded up to a 4096 block, floored so small test worlds still
/// get room to edit into.
fn capacity_for(count: u32) -> u32 {
    let want = count + count / 2;
    want.max(16_384).div_ceil(4096) * 4096
}

/// Build the world acceleration structure. The device MUST have been created
/// with `EXPERIMENTAL_RAY_QUERY` + `ExperimentalFeatures::enabled()` and the
/// acceleration-structure limits raised (they default to 0; the renderer passes
/// `adapter.limits()` for the RT device, see `Renderer::new`).
pub fn build_world_accel(device: &wgpu::Device, queue: &wgpu::Queue, world: &World) -> WorldAccel {
    let mut aabbs = Vec::new();
    let mut map = Vec::new();
    enumerate_non_empty(&world.tile_mask, world.world_origin_voxel(), &mut aabbs, &mut map);
    let capacity = capacity_for(aabbs.len() as u32);
    create_accel_at_capacity(device, queue, aabbs, map, capacity)
}

/// Streaming rebuild, IN PLACE: re-enumerate (mask-driven, sub-ms), DMA the
/// changed prefix (plus degenerates over any shrink delta), and re-build the
/// BLAS + TLAS on the GPU. No allocation, and the buffers - and therefore the
/// render bind group - survive. Returns `true` if the capacity had to grow
/// (buffers recreated: the caller must remake its bind group).
pub fn rebuild_world_accel(
    accel: &mut WorldAccel,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tile_mask: &[u64],
    world_origin: glam::IVec3,
) -> bool {
    let t0 = std::time::Instant::now();
    let mut aabbs = std::mem::take(&mut accel.aabb_scratch);
    let mut map = std::mem::take(&mut accel.map_scratch);
    enumerate_non_empty(tile_mask, world_origin, &mut aabbs, &mut map);
    let t_enum = t0.elapsed();
    let count = aabbs.len() as u32;

    if count > accel.capacity {
        let capacity = capacity_for(count);
        *accel = create_accel_at_capacity(device, queue, aabbs, map, capacity);
        return true;
    }

    upload_and_build(accel, device, queue, aabbs, map);
    let total = t0.elapsed();
    if total.as_secs_f64() > 0.002 {
        log::info!(
            "accel rebuild breakdown: enumerate {:.2} ms ({} prims), upload+build {:.2} ms",
            t_enum.as_secs_f64() * 1000.0,
            count,
            (total - t_enum).as_secs_f64() * 1000.0
        );
    }
    false
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
        nx: i32,
        ny: i32,
        nz: i32,
        _p0: u32,
    }

    /// Headless device WITH the experimental ray-query feature + AS limits. The
    /// exact incantation proven in the rt-spike (`adapter.limits()` supplies the
    /// acceleration-structure limits, which default to 0). Returns None when the
    /// adapter can't ray-trace, so the suite stays green on non-RT machines.
    fn rt_device() -> Option<(wgpu::Device, wgpu::Queue, std::sync::MutexGuard<'static, ()>)> {
        // Hold the GPU lock for the caller's whole test (concurrent submissions
        // across devices crash the NVIDIA/Vulkan driver, not just creation).
        let gpu = crate::gpu_init_serial();
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
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("rt accel test device"),
            required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            trace: wgpu::Trace::Off,
        }))
        .ok()?;
        Some((device, queue, gpu))
    }

    /// Build the accel for `world`, cast `rays` through `accel_probe.wgsl`, and
    /// read the resolved hits back. Rays and hits are in window-local voxel
    /// space (== world space for an unshifted world).
    fn probe(device: &wgpu::Device, queue: &wgpu::Queue, world: &World, rays: &[GpuRay]) -> Vec<GpuHit> {
        let accel = build_world_accel(device, queue, world);
        probe_with_accel(device, queue, world, &accel, rays)
    }

    fn probe_with_accel(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        world: &World,
        accel: &WorldAccel,
        rays: &[GpuRay],
    ) -> Vec<GpuHit> {
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

        // enable directive first, then the SHARED voxel_ray_query primitive
        // (resolve_brick), then the probe body - the same resolve_brick the
        // render shader's RT path uses.
        let probe_src = format!(
            "enable wgpu_ray_query;\n{}\n{}\n{}",
            include_str!(concat!(env!("OUT_DIR"), "/world_consts.wgsl")),
            include_str!("../shaders/rt_voxel_query.wgsl"),
            include_str!("../shaders/accel_probe.wgsl"),
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("accel_probe"),
            source: wgpu::ShaderSource::Wgsl(probe_src.into()),
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
        let Some((device, queue, _gpu)) = rt_device() else {
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

    /// Build a clean floor + 4 stone pillars at world coords inside the window
    /// for `origin_chunk`, cast a fan of clean and grazing rays at it, and check
    /// the RT-resolved voxel against the CPU raycaster. Returns
    /// (clean_assert, clean_floor, clean_pillar, grazing, total). A shifted
    /// origin engages the toroidal storage mapping (window-local != storage).
    fn check_floor_pillars(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        origin_chunk: glam::IVec2,
    ) -> (usize, usize, usize, usize, usize) {
        let mut world = World::new();
        // Set the origin WITHOUT regenerating (keeps the world empty so the only
        // geometry is what we craft); apply_edit then places voxels at world
        // coords via the same toroidal slot mapping the shader uses.
        world.world_origin_chunk = origin_chunk;
        let wo = world.world_origin_voxel();
        let base = Vec3::new(wo.x as f32, wo.y as f32, wo.z as f32);
        let (ox, oy, oz) = (wo.x, wo.y, wo.z);

        // Floor: top solid layer at window-local y=60 (world oy+60), x,z local
        // in [8,64). Pillars: 3x3, local footprints.
        for z in 8..64 {
            for x in 8..64 {
                for y in 57..=60 {
                    world.apply_edit(ox + x, oy + y, oz + z, MAT_STONE);
                }
            }
        }
        let pillars = [(20i32, 24i32, 90i32), (40, 40, 100), (52, 18, 76), (30, 50, 84)];
        for &(px, pz, top) in &pillars {
            for y in 61..top {
                for dz in 0..3 {
                    for dx in 0..3 {
                        world.apply_edit(ox + px + dx, oy + y, oz + pz + dz, MAT_STONE);
                    }
                }
            }
        }

        // Rays are built in window-local coords then translated to world; the
        // probe gets the local ray, the CPU oracle gets the world ray.
        let mut local_dirs: Vec<(Vec3, Vec3)> = Vec::new(); // (local_eye, dir)
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
                    local_dirs.push((eye, (target - eye).normalize()));
                }
            }
        }
        for &(px, pz, top) in &pillars {
            let eye = Vec3::new(px as f32 - 16.0, (61 + top) as f32 * 0.5, pz as f32 + 1.5);
            for gy in 0..3 {
                for gz in 0..3 {
                    let target = Vec3::new(
                        px as f32,
                        61.0 + (top - 61) as f32 * (0.25 + gy as f32 * 0.25),
                        pz as f32 + 0.5 + gz as f32,
                    );
                    local_dirs.push((eye, (target - eye).normalize()));
                }
            }
        }

        let rays: Vec<GpuRay> = local_dirs.iter().map(|&(e, d)| GpuRay::new(e, d)).collect();
        let hits = probe(device, queue, &world, &rays);

        const EPS: f32 = 0.02;
        let mut clean_assert = 0usize;
        let mut clean_floor = 0usize;
        let mut clean_pillar = 0usize;
        let mut grazing = 0usize;
        for (i, h) in hits.iter().enumerate() {
            let (local_eye, dir) = local_dirs[i];
            let eye_world = local_eye + base;
            let oracle = raycast(eye_world, dir, &world, wo);
            let rt_world = [h.vx + wo.x, h.vy + wo.y, h.vz + wo.z];
            let cpu_p = oracle
                .as_ref()
                .map(|pk| eye_world + ray_box_entry(eye_world, dir, pk.voxel) * dir);
            let rt_p = (h.hit == 1).then(|| eye_world + h.t * dir);
            let cpu_graze = cpu_p.is_some_and(|p| lattice_touch(p, EPS) >= 2);
            let rt_graze = rt_p.is_some_and(|p| lattice_touch(p, EPS) >= 2);

            if h.hit == 0 && oracle.is_none() {
                continue;
            }
            let agree = h.hit == 1 && oracle.as_ref().is_some_and(|pk| pk.voxel == rt_world);
            if agree {
                if cpu_graze {
                    grazing += 1;
                } else {
                    clean_assert += 1;
                    // A clean face hit: the RT entry-face normal must match the
                    // CPU raycaster's (needed for primary-ray shading, Phase 4).
                    let cpu_n = oracle.as_ref().unwrap().normal;
                    assert_eq!(
                        [h.nx, h.ny, h.nz], cpu_n,
                        "origin ({},{}) clean hit {:?}: RT normal {:?} != CPU {:?}",
                        wo.x, wo.z, rt_world, [h.nx, h.ny, h.nz], cpu_n
                    );
                    // window-local y<=60 is the floor (rt_world.y == wo.y + local).
                    if h.vy <= 60 { clean_floor += 1; } else { clean_pillar += 1; }
                }
                continue;
            }
            if cpu_graze || rt_graze {
                grazing += 1;
                continue;
            }
            panic!(
                "origin ({},{}) ray {i}: clean-face disagreement RT {}{:?} vs CPU {}{:?} (dir {:?})",
                wo.x, wo.z,
                if h.hit == 1 { "hit " } else { "MISS " }, rt_world,
                if oracle.is_some() { "hit " } else { "MISS " },
                oracle.as_ref().map(|p| p.voxel).unwrap_or_default(), dir
            );
        }
        (clean_assert, clean_floor, clean_pillar, grazing, rays.len())
    }

    #[test]
    fn accel_matches_cpu_raycast() {
        let Some((device, queue, _gpu)) = rt_device() else {
            eprintln!("accel_matches_cpu_raycast: no RT adapter, skipping");
            return;
        };
        let (clean, floor, pillar, grazing, total) =
            check_floor_pillars(&device, &queue, glam::IVec2::ZERO);
        assert!(clean > total / 3, "too few clean hits checked: {clean}/{total}");
        assert!(floor > 0 && pillar > 0, "clean hits must cover floor ({floor}) and pillars ({pillar})");
        assert!(grazing < clean, "grazing majority ({grazing} >= {clean}) - suspect a systematic offset");
        eprintln!("accel_matches_cpu_raycast: {clean} clean hits match CPU exactly (floor {floor}, pillar {pillar}); {grazing} grazing over {total}");
    }

    #[test]
    fn accel_matches_cpu_streaming() {
        // Same clean scene at a SHIFTED origin: window-local space and toroidal
        // storage diverge, so this proves the accel's local<->storage mapping
        // (and the ray-query caller's world_origin rebase) - the origin-0 test
        // never engages the wrap.
        let Some((device, queue, _gpu)) = rt_device() else {
            eprintln!("accel_matches_cpu_streaming: no RT adapter, skipping");
            return;
        };
        let origin = glam::IVec2::new(3, 5);
        let (clean, floor, pillar, grazing, total) =
            check_floor_pillars(&device, &queue, origin);
        assert!(clean > total / 3, "too few clean streamed hits: {clean}/{total}");
        assert!(floor > 0 && pillar > 0, "clean streamed hits must cover floor ({floor}) and pillars ({pillar})");
        assert!(grazing < clean, "grazing majority under streaming ({grazing} >= {clean})");
        eprintln!("accel_matches_cpu_streaming: origin (96,160) - {clean} clean hits match CPU exactly (floor {floor}, pillar {pillar}); {grazing} grazing over {total}");
    }

    #[test]
    fn accel_rebuild_in_place_matches_cpu() {
        // The streaming rebuild path: mask-driven re-enumeration + prefix DMA +
        // GPU BVH re-build into the SAME buffers. Grow the world (new pillar),
        // shrink it (carve a hole), and stream the origin - after each in-place
        // rebuild the RT probe must agree with the CPU raycaster, and stale
        // primitives from the previous (larger) build must be gone.
        let Some((device, queue, _gpu)) = rt_device() else {
            eprintln!("accel_rebuild_in_place_matches_cpu: no RT adapter, skipping");
            return;
        };
        let mut world = flat_floor(8, 8, 56); // top solid layer y=60
        let mut accel = build_world_accel(&device, &queue, &world);
        let count0 = accel.aabb_count;
        assert!(count0 > 0, "flat floor produced no primitives");

        // -- grow in place: raise a 3x3 pillar and rebuild --
        for y in 61..75u32 {
            for dz in 0..3u32 {
                for dx in 0..3u32 {
                    world.set_voxel(30 + dx, y, 30 + dz, MAT_STONE);
                }
            }
        }
        let grew = rebuild_world_accel(&mut accel, &device, &queue, &world.tile_mask, world.world_origin_voxel());
        assert!(!grew, "a single pillar must fit the initial capacity (in-place path must run)");
        assert!(accel.aabb_count > count0, "pillar did not add primitives");
        let wo = world.world_origin_voxel();
        // A ray at pillar height must hit the pillar's side face.
        let eye = Vec3::new(20.0, 70.0, 31.5);
        let dir = Vec3::new(1.0, 0.0, 0.0);
        let h = &probe_with_accel(&device, &queue, &world, &accel, &[GpuRay::new(eye, dir)])[0];
        let oracle = raycast(eye + Vec3::new(wo.x as f32, wo.y as f32, wo.z as f32), dir, &world, wo)
            .expect("CPU must hit the new pillar");
        assert_eq!(h.hit, 1, "RT missed the pillar added by the in-place rebuild");
        assert_eq!([h.vx + wo.x, h.vy + wo.y, h.vz + wo.z], oracle.voxel, "pillar hit voxel mismatch");

        // -- shrink in place: remove the pillar again --
        let with_pillar = accel.aabb_count;
        for y in 61..75u32 {
            for dz in 0..3u32 {
                for dx in 0..3u32 {
                    world.set_voxel(30 + dx, y, 30 + dz, crate::voxel::MAT_AIR);
                }
            }
        }
        let grew = rebuild_world_accel(&mut accel, &device, &queue, &world.tile_mask, world.world_origin_voxel());
        assert!(!grew);
        assert!(accel.aabb_count < with_pillar, "shrink did not drop primitives");
        // The same ray must now sail over the floor and MISS: proves the dead
        // range was degenerate-padded, not left holding the stale pillar.
        let h = &probe_with_accel(&device, &queue, &world, &accel, &[GpuRay::new(eye, dir)])[0];
        assert_eq!(h.hit, 0, "stale pillar primitive survived the in-place shrink");

        // -- streamed rebuild: shift the window origin and rebuild in place --
        world.shift_origin(glam::IVec2::new(2, 3));
        world.process_pending_gen_blocking();
        rebuild_world_accel(&mut accel, &device, &queue, &world.tile_mask, world.world_origin_voxel());
        let wo = world.world_origin_voxel();
        // Straight down onto regenerated terrain: RT and CPU must agree.
        let eye = Vec3::new(40.0, 140.0, 40.0);
        let dir = Vec3::new(0.001, -1.0, 0.001).normalize();
        let h = &probe_with_accel(&device, &queue, &world, &accel, &[GpuRay::new(eye, dir)])[0];
        let oracle = raycast(eye + Vec3::new(wo.x as f32, wo.y as f32, wo.z as f32), dir, &world, wo);
        match (h.hit == 1, oracle) {
            (true, Some(pk)) => {
                assert_eq!([h.vx + wo.x, h.vy + wo.y, h.vz + wo.z], pk.voxel,
                    "streamed in-place rebuild: RT vs CPU voxel mismatch");
            }
            (false, None) => {}
            (rt, cpu) => panic!("streamed rebuild disagreement: RT hit={rt} CPU hit={}", cpu.is_some()),
        }
        eprintln!("accel_rebuild_in_place_matches_cpu: grow/shrink/stream all agree (counts {count0} -> {with_pillar} -> {})", accel.aabb_count);
    }

    #[test]
    fn accel_shadow_occlusion_matches_cpu() {
        // Shadow rays are any-hit occlusion tests: a surface point is in shadow
        // iff the ray toward the sun hits ANY solid voxel. This validates that
        // semantic (the same RT primitive the render passes will use for
        // shadows) against the CPU raycaster, BEFORE wiring it into the shader.
        let Some((device, queue, _gpu)) = rt_device() else {
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

    #[test]
    fn accel_empty_world_all_miss() {
        // Edge case: a world with no solid bricks. build_world_accel still needs
        // a valid (>=1 primitive) BLAS, so it emits one degenerate AABB far out;
        // every ray must MISS it. Guards against a panic on the empty-world build
        // and against the placeholder ever being hit by an in-window ray.
        let Some((device, queue, _gpu)) = rt_device() else {
            eprintln!("accel_empty_world_all_miss: no RT adapter, skipping");
            return;
        };
        let world = World::new(); // all bricks empty
        let mut rays = Vec::new();
        for gz in 0..8 {
            for gx in 0..8 {
                let eye = Vec3::new(20.0 + gx as f32 * 30.0, 200.0, 20.0 + gz as f32 * 30.0);
                rays.push(GpuRay::new(eye, Vec3::new(0.05, -1.0, 0.03)));
            }
        }
        let hits = probe(&device, &queue, &world, &rays);
        for (i, h) in hits.iter().enumerate() {
            assert_eq!(h.hit, 0, "ray {i} hit something in an empty world: {h:?}");
        }
        eprintln!("accel_empty_world_all_miss: {} rays, all miss", rays.len());
    }
}
