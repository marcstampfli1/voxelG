// Thin launcher: parse the run mode and dispatch into the library. All engine
// code lives in the `voxelg` library crate (src/lib.rs); see app.rs (client),
// server.rs (dedicated server) and voxel.rs (world).

use voxelg::app::ClientOpts;
use voxelg::{app, net, server};

enum Mode {
    Solo,
    Server(u16),
    Connect(String),
}

fn parse_args() -> (Mode, ClientOpts) {
    let args: Vec<String> = std::env::args().collect();
    let mut mode = Mode::Solo;
    let mut opts = ClientOpts::default();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--server" => {
                let port = args
                    .get(i + 1)
                    .and_then(|s| s.parse::<u16>().ok())
                    .unwrap_or(7878);
                mode = Mode::Server(port);
                i += 1;
            }
            "--connect" => {
                let addr = args
                    .get(i + 1)
                    .cloned()
                    .unwrap_or_else(|| "127.0.0.1:7878".to_string());
                mode = Mode::Connect(addr);
                i += 1;
            }
            // Pin the DAY/NIGHT cycle (sun position) at an optional second
            // count (default 0 = the pleasant startup sun); water, wind and
            // falling leaves keep animating.
            "--freeze-time" => {
                let val = args.get(i + 1).and_then(|s| s.parse::<f32>().ok());
                if val.is_some() {
                    i += 1;
                }
                opts.freeze_time = Some(val.unwrap_or(0.0));
            }
            // Fly-speed multiplier (e.g. --speed 3).
            "--speed" => {
                if let Some(m) = args.get(i + 1).and_then(|s| s.parse::<f32>().ok()) {
                    opts.speed = m;
                    i += 1;
                } else {
                    log::warn!("--speed needs a numeric multiplier, ignoring");
                }
            }
            other => {
                log::warn!("unknown argument {other:?} ignored");
            }
        }
        i += 1;
    }
    (mode, opts)
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let (mode, mut opts) = parse_args();
    // Benchmark mode implies its full deterministic setup: uncapped present
    // (real throughput) and a frozen sun (identical lighting every run).
    if std::env::var("VOXELG_BENCH").is_ok() {
        std::env::set_var("VOXELG_UNCAPPED", "1");
        std::env::set_var("VOXELG_RT", "1");
        // Per-segment GPU-pass attribution rides the existing profiler.
        std::env::set_var("VOXELG_GPU_PROFILE", "1");
        opts.freeze_time = Some(30.0);
    }
    match mode {
        Mode::Server(port) => server::run_server(port),
        mode => {
            let (net, server_addr) = match mode {
                Mode::Connect(addr) => {
                    let net = match net::NetClient::connect(&addr) {
                        Ok(c) => {
                            log::info!("connected to {}", addr);
                            Some(c)
                        }
                        Err(e) => {
                            log::error!("connect failed: {} (will retry)", e);
                            None
                        }
                    };
                    // Keep the address either way so the client auto-reconnects.
                    (net, Some(addr))
                }
                _ => (None, None),
            };
            app::run_client(net, server_addr, opts);
        }
    }
}
