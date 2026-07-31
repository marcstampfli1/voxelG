// voxelG library root.
//
// The engine lives here as a library so the binary stays a thin launcher and
// so tests/benches can exercise the world, worldgen and picking code directly
// (checklist: hygiene / tests). The binary target is `src/main.rs`.

pub mod world_dims;
pub mod voxel;
pub mod voxlight;
pub mod sprites;
pub mod camera;
pub mod leaffall;
pub mod grass;
pub mod raycast;
pub mod accel;
pub mod physics;
pub mod net;
pub mod temporal;
pub mod renderer;
pub mod app;
pub mod server;

/// Serialize GPU device CREATION across the test suite. `cargo test` runs tests
/// in parallel, and some NVIDIA/Vulkan drivers crash (SIGSEGV) when several
/// logical devices are created concurrently - especially the mix of the plain
/// render device and the experimental ray-query device the RT tests build. The
/// guard is held only across adapter+device request, so GPU submissions from
/// different tests still overlap; just the racy init is serialized.
#[cfg(test)]
pub(crate) fn gpu_init_serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}
