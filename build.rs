#![allow(dead_code)]
// Build script: derive the WGSL world-dimension constants from the SAME Rust
// source the engine uses (`src/world_dims.rs`) and emit them as a WGSL snippet
// that renderer.rs prepends to every shader. Keeps the GPU's idea of the world
// size byte-for-byte identical to the CPU's, with no hand-maintained second copy.

use std::io::Write;
use std::path::Path;

include!("src/world_dims.rs");

fn main() {
    println!("cargo:rerun-if-changed=src/world_dims.rs");
    println!("cargo:rerun-if-changed=build.rs");

    let out_dir = std::env::var("OUT_DIR").expect("OUT_DIR");
    let path = Path::new(&out_dir).join("world_consts.wgsl");
    let mut f = std::fs::File::create(&path).expect("create world_consts.wgsl");

    writeln!(f, "// AUTO-GENERATED from src/world_dims.rs by build.rs — DO NOT EDIT.").unwrap();
    macro_rules! emit {
        ($($name:ident),* $(,)?) => {
            $( writeln!(f, "const {}: i32 = {};", stringify!($name), $name as i32).unwrap(); )*
        };
    }
    emit!(
        BRICK_DIM, BRICK_VOXELS,
        WORLD_BRICKS_X, WORLD_BRICKS_Y, WORLD_BRICKS_Z,
        WORLD_VOXELS_X, WORLD_VOXELS_Y, WORLD_VOXELS_Z,
        WORLD_TILES_X, WORLD_TILES_Y, WORLD_TILES_Z,
        WORLD_CHUNKS_X, WORLD_CHUNKS_Y, WORLD_CHUNKS_Z,
        WORLD_L4_X, WORLD_L4_Y, WORLD_L4_Z,
        PROBE_SPACING,
        PROBE_DIM_X, PROBE_DIM_Y, PROBE_DIM_Z, PROBE_TOTAL,
    );
}
