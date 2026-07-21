// voxelG library root.
//
// The engine lives here as a library so the binary stays a thin launcher and
// so tests/benches can exercise the world, worldgen and picking code directly
// (checklist: hygiene / tests). The binary target is `src/main.rs`.

pub mod world_dims;
pub mod voxel;
pub mod sprites;
pub mod camera;
pub mod leaffall;
pub mod raycast;
pub mod physics;
pub mod net;
pub mod temporal;
pub mod renderer;
pub mod app;
pub mod server;
