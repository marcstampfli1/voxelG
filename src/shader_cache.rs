//! Persistent driver pipeline cache.
//!
//! WHY this exists: `shaders/raymarch.wgsl` assembles to ~4700 lines and the
//! renderer compiles six entry points out of it (`cs_main`, `cs_compose`,
//! `cs_transparent`, `cs_godrays`, `cs_clouds`, `cs_voxel_light_update`).
//! With `VOXELG_RT=1` the ray-query source is a second module and six more
//! compiles come out of that one (`cs_clouds` stays software,
//! `cs_gi_probe_update` joins), so a launch is twelve. Turning that
//! SPIR-V into machine code is the driver's job and it is minutes of work on a
//! cold driver cache, with no window on screen and nothing in the log: the
//! first launch of a release build looks like a hang.
//!
//! `wgpu::PipelineCache` hands the driver a blob it can seed itself from. We
//! own that blob's lifetime, so the compile is paid once per machine instead of
//! once per driver-cache eviction.
//!
//! Measured by `pipeline_compile_time` below on an RTX 4060 Ti / Vulkan /
//! NVIDIA 591.86, with the DRIVER's own cache redirected to a scratch directory
//! so it can be emptied independently of ours:
//!
//! The table was taken when each variant still built SEVEN entry points
//! (`cs_voxel_refl_update` has since been deleted with the reflection field),
//! so the totals are a slight over-count of what a launch pays today. It is
//! left as measured rather than rescaled by arithmetic nobody ran:
//!
//!                        no blob    our blob
//!     software (7)       110.0 s       0.9 s   <- what a DEFAULT launch pays
//!     RT (7 more)        178.3 s       1.3 s
//!     both               288.2 s       2.2 s
//!
//! `cs_transparent` is almost the whole bill: that ONE entry point is 98 s of
//! the software 110 and 162 s of the RT 178. It needs no `VOXELG_RT` to happen
//! and nothing on screen or in the log used to say it was happening, which is
//! precisely the "first launch hangs for minutes with no window" report.
//!
//! The honest caveat: this desktop driver keeps a good cache of its own, and
//! with THAT left warm the seven RT pipelines take 1.2 s with no blob at all.
//! On a healthy machine the second launch was always going to be fast; what
//! this module buys is that the FIRST one after any cache loss is fast too. The
//! driver's cache is size-capped and LRU-evicted across every application on
//! the machine, "Disk Cleanup / DirectX Shader Cache" wipes it, a driver
//! reinstall drops it, and mobile drivers frequently keep none at all. The
//! second column is the measurement that our blob fully substitutes for it.
//!
//! Everything here degrades to "no cache, just compile" rather than failing:
//! - `PIPELINE_CACHE` is a Vulkan-only feature in wgpu 30. On DX12 the adapter
//!   never reports it and `pipeline_cache_key` returns `None`, so we disable.
//! - The blob is driver AND device specific. The filename carries the adapter
//!   identity so a foreign blob is never handed to the driver, and wgpu
//!   re-validates the header anyway (magic / header version / pointer ABI /
//!   backend / vendor+device / driver validation key / length).
//! - A rejected blob (corrupt, truncated, outdated, from a newer wgpu, from
//!   another device) costs a recompile and nothing else: the stored data is
//!   offered WITHOUT `fallback` first so the rejection is logged with its
//!   reason instead of silently swallowed, then the cache is recreated empty.
//! - `Device::create_pipeline_cache` reports failure through the device error
//!   sink, whose DEFAULT HANDLER PANICS. Every creation therefore runs inside a
//!   validation error scope, which is what makes "rejected" a log line rather
//!   than a crashed launch.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Directory override, for tests and for users who want the blob elsewhere.
const DIR_ENV: &str = "VOXELG_SHADER_CACHE_DIR";
/// Escape hatch: set to disable the cache entirely (A/B timing, bug triage).
const OFF_ENV: &str = "VOXELG_NO_SHADER_CACHE";

/// A `wgpu::PipelineCache` plus the file it is loaded from / saved to.
///
/// A disabled cache (feature missing, key unavailable, env override) is a
/// perfectly usable value whose `handle()` is `None` and whose `persist()` is a
/// no-op, so callers never branch.
pub struct ShaderCache {
    cache: Option<wgpu::PipelineCache>,
    path: Option<PathBuf>,
    /// What the file currently holds, so `persist` can skip an identical
    /// rewrite (the blob is megabytes; a warm launch changes nothing). Interior
    /// mutability because checkpointing happens from `&self` in the middle of
    /// pipeline creation.
    on_disk: RefCell<Option<Vec<u8>>>,
    /// Bytes read at open time. Fixed, unlike `on_disk`, so callers and tests
    /// can ask "did we start warm?" after any number of checkpoints.
    loaded_len: usize,
}

/// A compile slower than this gets the cache checkpointed to disk immediately.
///
/// WHY: a cold first launch is minutes of compiling with no window. A user who
/// concludes it is hung and kills it would, with a single save at the end, throw
/// away every second of that work and pay it again next time. Checkpointing
/// after each expensive pipeline means the next launch resumes from wherever
/// the last one got to. The threshold keeps the cheap pipelines from turning a
/// warm start into a series of multi-megabyte rewrites.
const CHECKPOINT_AFTER: f64 = 1.0;

/// The device features this module needs, if the adapter can provide them.
///
/// `Device::create_pipeline_cache` hard-requires `PIPELINE_CACHE`, so the
/// feature has to be requested at device creation; returning an empty set when
/// the adapter lacks it keeps `request_device` succeeding unchanged.
pub fn wanted_features(adapter: &wgpu::Adapter) -> wgpu::Features {
    if std::env::var_os(OFF_ENV).is_some() {
        return wgpu::Features::empty();
    }
    adapter.features() & wgpu::Features::PIPELINE_CACHE
}

impl ShaderCache {
    /// Load the on-disk blob for this adapter and open a pipeline cache from it.
    ///
    /// Never fails: every problem downgrades to a disabled cache and a log line.
    pub fn open(device: &wgpu::Device, adapter: &wgpu::Adapter) -> Self {
        Self::open_in(device, adapter, cache_dir())
    }

    /// `open` with an explicit directory. Split out so tests can exercise the
    /// real load/reject/persist paths against a scratch directory without
    /// mutating process-global environment behind other threads' backs.
    pub fn open_in(device: &wgpu::Device, adapter: &wgpu::Adapter, dir: PathBuf) -> Self {
        let disabled = Self {
            cache: None,
            path: None,
            on_disk: RefCell::new(None),
            loaded_len: 0,
        };

        if std::env::var_os(OFF_ENV).is_some() {
            log::info!("pipeline cache: disabled by {OFF_ENV}; shader compiles will not persist");
            return disabled;
        }
        if !device.features().contains(wgpu::Features::PIPELINE_CACHE) {
            // Expected on DX12/Metal: wgpu only implements pipeline caches on
            // Vulkan. Not an error, just no persistence.
            log::info!(
                "pipeline cache: unavailable ({:?} backend does not expose PIPELINE_CACHE); \
                 first-launch shader compiles will not persist",
                adapter.get_info().backend
            );
            return disabled;
        }

        let info = adapter.get_info();
        // wgpu's own recommended key: it encodes the backend plus the
        // vendor/device pair that the blob header is validated against, so two
        // different GPUs can never collide on one file.
        let Some(key) = wgpu::util::pipeline_cache_key(&info) else {
            log::info!(
                "pipeline cache: no cache key for {:?}; shader compiles will not persist",
                info.backend
            );
            return disabled;
        };
        // The key above pins the DEVICE but not the DRIVER. A driver update
        // changes the blob's validation key, which wgpu would reject as
        // `Outdated` and silently drop; folding the driver strings into the
        // filename instead means the update starts a fresh file (and the stale
        // one is pruned below) rather than repeatedly loading a dead blob.
        let ident = format!(
            "{}|{}|{}|{:?}|{}|{}",
            info.name, info.driver, info.driver_info, info.device_type, info.vendor, info.device
        );
        let prefix = format!("voxelg-{key}-");
        let file = format!("{prefix}{:016x}.bin", fnv1a64(ident.as_bytes()));

        if let Err(e) = std::fs::create_dir_all(&dir) {
            log::warn!(
                "pipeline cache: cannot create {}: {e}; shader compiles will not persist",
                dir.display()
            );
            return disabled;
        }
        let path = dir.join(&file);

        let loaded = match std::fs::read(&path) {
            Ok(bytes) => Some(bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
            Err(e) => {
                log::warn!("pipeline cache: cannot read {}: {e}; treating as empty", path.display());
                None
            }
        };
        // `fallback: false` on the first attempt ON PURPOSE. With `true`, wgpu
        // silently discards a blob it considers outdated (driver update, wgpu
        // upgrade) and hands back an empty cache, so "we read 3 MB off disk"
        // would look identical to "the driver accepted 3 MB" and a permanently
        // useless cache file could go unnoticed for months. Asking without a
        // fallback turns that into a logged reason, and the retry below still
        // guarantees we end up with a usable cache either way.
        let (cache, accepted) = match loaded.as_deref() {
            Some(blob) => match create_guarded(device, Some(blob), false) {
                Ok(c) => {
                    log::info!(
                        "pipeline cache: HIT, {} KiB accepted from {}",
                        blob.len() / 1024,
                        path.display()
                    );
                    (c, true)
                }
                Err(e) => {
                    log::warn!(
                        "pipeline cache: stored blob at {} rejected ({e}); recompiling from scratch",
                        path.display()
                    );
                    match create_guarded(device, None, true) {
                        Ok(c) => (c, false),
                        Err(e) => {
                            log::warn!("pipeline cache: could not be created ({e}); disabled");
                            return disabled;
                        }
                    }
                }
            },
            None => {
                log::info!(
                    "pipeline cache: MISS, no blob at {} (cold driver compile, expect tens of seconds)",
                    path.display()
                );
                match create_guarded(device, None, true) {
                    Ok(c) => (c, false),
                    Err(e) => {
                        log::warn!("pipeline cache: could not be created ({e}); disabled");
                        return disabled;
                    }
                }
            }
        };

        prune_stale(&dir, &prefix, &file);
        // A rejected blob is dead weight: forget it so `persist` compares
        // against nothing and unconditionally replaces the file.
        let loaded = if accepted { loaded } else { None };
        Self {
            cache: Some(cache),
            path: Some(path),
            loaded_len: loaded.as_ref().map_or(0, |b| b.len()),
            on_disk: RefCell::new(loaded),
        }
    }

    /// The handle to hand to `ComputePipelineDescriptor::cache`.
    pub fn handle(&self) -> Option<&wgpu::PipelineCache> {
        self.cache.as_ref()
    }

    /// The blob's file, when the cache is enabled. Diagnostics and tests.
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Bytes read from disk at open time (0 on a cold start or when disabled).
    pub fn loaded_len(&self) -> usize {
        self.loaded_len
    }

    /// Write the driver's accumulated cache back to disk, if it changed.
    ///
    /// Safe to call repeatedly: an unchanged blob is not rewritten. A failure
    /// here is logged and swallowed - a cache we cannot save is a slow next
    /// launch, not a broken one, and must never take startup down with it.
    pub fn persist(&self) {
        let (Some(cache), Some(path)) = (&self.cache, &self.path) else { return };
        let t = Instant::now();
        let Some(data) = cache.get_data() else {
            log::warn!("pipeline cache: driver returned no data; nothing written");
            return;
        };
        if self.on_disk.borrow().as_deref() == Some(data.as_slice()) {
            log::info!("pipeline cache: unchanged ({} KiB), not rewritten", data.len() / 1024);
            return;
        }
        // Write-then-rename: a crash mid-write leaves the previous good blob in
        // place instead of a truncated one. (wgpu would reject the truncated
        // blob rather than crash, but a needless full recompile is the exact
        // cost this module exists to avoid.)
        let tmp = path.with_extension("tmp");
        if let Err(e) = std::fs::write(&tmp, &data) {
            log::warn!("pipeline cache: write to {} failed ({e}); continuing", tmp.display());
            let _ = std::fs::remove_file(&tmp);
            return;
        }
        if let Err(e) = std::fs::rename(&tmp, path) {
            log::warn!("pipeline cache: rename to {} failed ({e}); continuing", path.display());
            let _ = std::fs::remove_file(&tmp);
            return;
        }
        log::info!(
            "pipeline cache: wrote {} KiB to {} in {:.0} ms",
            data.len() / 1024,
            path.display(),
            t.elapsed().as_secs_f64() * 1e3
        );
        *self.on_disk.borrow_mut() = Some(data);
    }
}

/// Create a compute pipeline through `cache`, logging the driver's compile time.
///
/// WHY the log: a cold compile of one raymarch entry point is seconds, and
/// six in a row with a single line at the end is indistinguishable from a
/// hang. One line per pipeline turns "frozen" into "working, 3 of 6".
pub fn create_compute_timed<'a>(
    device: &wgpu::Device,
    cache: &'a ShaderCache,
    prefix: &str,
    mut desc: wgpu::ComputePipelineDescriptor<'a>,
) -> wgpu::ComputePipeline {
    desc.cache = cache.handle();
    let label = desc.label.unwrap_or("unnamed");
    let t = Instant::now();
    let pipeline = device.create_compute_pipeline(&desc);
    let secs = t.elapsed().as_secs_f64();
    log::info!("{prefix}: {label} in {:.0} ms", secs * 1e3);
    // Expensive result banked now, not at the end: see CHECKPOINT_AFTER.
    if secs >= CHECKPOINT_AFTER {
        cache.persist();
    }
    pipeline
}

/// `Device::create_pipeline_cache` reports failure through the device error
/// sink, whose default handler panics. Wrapping it in a validation error scope
/// converts that into a `Result` we can fall back from.
fn create_guarded(
    device: &wgpu::Device,
    data: Option<&[u8]>,
    fallback: bool,
) -> Result<wgpu::PipelineCache, String> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    // SAFETY: `data`, when present, is a blob previously produced by
    // `PipelineCache::get_data` on this machine. It cannot be proven (it came
    // off disk), which is why wgpu re-validates the header and why the caller
    // retries with `None` if this scope catches anything.
    let cache = unsafe {
        device.create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
            label: Some("voxelg pipeline cache"),
            data,
            // With `true`, wgpu absorbs the rejections it does not blame on the
            // caller (Corrupted / Truncated / Extended / Outdated / Unsupported)
            // and returns an empty cache; DeviceMismatch is an error either way.
            fallback,
        })
    };
    match pollster::block_on(scope.pop()) {
        Some(e) => Err(e.to_string()),
        None => Ok(cache),
    }
}

/// Per-user cache directory. No new crate dependency: the two platform
/// conventions we care about are one env var each.
fn cache_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os(DIR_ENV) {
        return PathBuf::from(dir);
    }
    // Windows: the roaming profile must not carry a machine-specific blob, so
    // LOCALAPPDATA (not APPDATA) is the correct home for it.
    if let Some(local) = std::env::var_os("LOCALAPPDATA") {
        return Path::new(&local).join("voxelG").join("shader-cache");
    }
    // XDG on unix.
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
        return Path::new(&xdg).join("voxelG").join("shader-cache");
    }
    if let Some(home) = std::env::var_os("HOME") {
        return Path::new(&home).join(".cache").join("voxelG").join("shader-cache");
    }
    // Last resort: still works, just does not survive a reboot on some systems.
    std::env::temp_dir().join("voxelG-shader-cache")
}

/// Drop blobs for this same adapter but an older driver.
///
/// Without this a driver update orphans a multi-megabyte file forever. Scoped
/// to our own filename prefix AND the current adapter's key, so a second GPU's
/// blob (or anything else in the directory) is never touched.
fn prune_stale(dir: &Path, prefix: &str, keep: &str) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.starts_with(prefix) || name == keep {
            continue;
        }
        match std::fs::remove_file(entry.path()) {
            Ok(()) => log::info!("pipeline cache: pruned stale blob {name}"),
            Err(e) => log::debug!("pipeline cache: could not prune {name}: {e}"),
        }
    }
}

/// FNV-1a, 64 bit. Only used to shorten an identity string into a filename, so
/// speed and collision resistance beyond "different drivers differ" are moot;
/// hand-rolled because it must not cost a new dependency.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv1a64_matches_reference_vectors() {
        // Published FNV-1a 64 test vectors: proves the constants and the
        // xor-then-multiply order, which is what makes the filename stable
        // across builds.
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn cache_dir_honours_override_then_localappdata() {
        // The override is what tests and relocation use; without it Windows
        // must land under LOCALAPPDATA, never the repo or the roaming profile.
        let dir = cache_dir();
        if std::env::var_os(DIR_ENV).is_none() {
            if let Some(local) = std::env::var_os("LOCALAPPDATA") {
                assert!(dir.starts_with(Path::new(&local)), "cache dir escaped LOCALAPPDATA: {dir:?}");
            }
            assert!(dir.ends_with("shader-cache"), "unexpected cache dir: {dir:?}");
        }
        assert!(dir.is_absolute(), "cache dir must be absolute: {dir:?}");
    }

    #[test]
    fn disabled_cache_is_usable() {
        // The degraded value must behave like a cache that simply never hits,
        // so no caller needs a branch.
        let c = ShaderCache {
            cache: None,
            path: None,
            on_disk: RefCell::new(None),
            loaded_len: 0,
        };
        assert!(c.handle().is_none());
        assert_eq!(c.loaded_len(), 0);
        assert!(c.path().is_none());
        c.persist(); // must not panic
    }

    /// Adapter + device with `PIPELINE_CACHE`, or `None` so the GPU-backed tests
    /// below skip cleanly on a machine/backend that cannot do this at all.
    fn cache_device() -> Option<(wgpu::Device, wgpu::Adapter, std::sync::MutexGuard<'static, ()>)> {
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
        if !adapter.features().contains(wgpu::Features::PIPELINE_CACHE) {
            return None;
        }
        let (device, _queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("pipeline cache test device"),
            required_features: wgpu::Features::PIPELINE_CACHE,
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::Off,
        }))
        .ok()?;
        Some((device, adapter, gpu))
    }

    /// Scratch directory unique to one test, removed on drop.
    struct Scratch(PathBuf);
    impl Scratch {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir()
                .join(format!("voxelg-cache-test-{}-{tag}", std::process::id()));
            let _ = std::fs::remove_dir_all(&dir);
            std::fs::create_dir_all(&dir).expect("scratch dir");
            Self(dir)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A 64-byte wgpu cache header that is structurally VALID but claims a
    /// different vendor/device. wgpu classes that as `DeviceMismatch`, the one
    /// rejection `fallback: true` does NOT absorb, so it reaches the device
    /// error sink whose default handler panics. This is the exact blob the
    /// error-scope guard in `create_guarded` exists for.
    fn foreign_adapter_blob() -> Vec<u8> {
        let mut b = Vec::with_capacity(64);
        b.extend_from_slice(b"WGPUPLCH"); // magic
        b.extend_from_slice(&1u32.to_be_bytes()); // header version
        b.extend_from_slice(&(size_of::<*const ()>() as u32).to_be_bytes()); // pointer ABI
        b.push(1); // wgt::Backend::Vulkan
        // adapter_key: 0xFF padding around a vendor/device pair no GPU has.
        b.extend_from_slice(&[255, 255, 255]);
        b.extend_from_slice(&0xDEAD_BEEFu32.to_be_bytes());
        b.extend_from_slice(&0xCAFE_BABEu32.to_be_bytes());
        b.extend_from_slice(&[255, 255, 255, 255]);
        b.extend_from_slice(&[0u8; 16]); // validation key
        b.extend_from_slice(&0u64.to_be_bytes()); // data size
        b.extend_from_slice(&0xFEDC_BA98_7654_3210u64.to_be_bytes()); // hash space
        assert_eq!(b.len(), 64);
        b
    }

    /// The degradation path an actual DX12/Metal machine takes. Reproduced here
    /// by simply not requesting the feature, so it is exercised on hardware that
    /// does support it rather than being assumed.
    #[test]
    fn device_without_the_feature_degrades_quietly() {
        if std::env::var_os(OFF_ENV).is_some() {
            eprintln!("device_without_the_feature_degrades_quietly: {OFF_ENV} set, skipping");
            return;
        }
        let gpu = crate::gpu_init_serial();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..wgpu::InstanceDescriptor::new_without_display_handle_from_env()
        });
        let Ok(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })) else {
            eprintln!("device_without_the_feature_degrades_quietly: no adapter, skipping");
            return;
        };
        let Ok((device, _q)) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("no-pipeline-cache device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            trace: wgpu::Trace::Off,
        })) else {
            eprintln!("device_without_the_feature_degrades_quietly: no device, skipping");
            return;
        };
        // Held for the whole test, not just device creation: the suite's other
        // GPU tests submit work, and this driver dislikes that overlapping with
        // pipeline creation on a second device.
        let _gpu = gpu;

        let scratch = Scratch::new("nofeature");
        let cache = ShaderCache::open_in(&device, &adapter, scratch.0.clone());
        assert!(cache.handle().is_none(), "must not hand out a cache it cannot create");
        assert!(cache.path().is_none());
        assert_eq!(cache.loaded_len(), 0);
        cache.persist(); // no-op, must not panic
        assert_eq!(
            std::fs::read_dir(&scratch.0).unwrap().count(),
            0,
            "a disabled cache must not leave files behind"
        );

        // And a pipeline still builds through the disabled cache.
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("degraded probe"),
            source: wgpu::ShaderSource::Wgsl(
                "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\n\
                 @compute @workgroup_size(64) fn cs(@builtin(global_invocation_id) g: vec3<u32>) {\n\
                 o[g.x] = g.x;\n}"
                    .into(),
            ),
        });
        let _p = create_compute_timed(
            &device,
            &cache,
            "test",
            wgpu::ComputePipelineDescriptor {
                label: Some("degraded probe"),
                layout: None,
                module: &module,
                entry_point: Some("cs"),
                compilation_options: Default::default(),
                cache: None,
            },
        );
    }

    #[test]
    fn corrupt_blob_is_absorbed_by_fallback() {
        let Some((device, _adapter, _gpu)) = cache_device() else {
            eprintln!("corrupt_blob_is_absorbed_by_fallback: no PIPELINE_CACHE adapter, skipping");
            return;
        };
        // Nonsense bytes fail the magic check -> `Corrupted`, which wgpu treats
        // as not-the-caller's-fault, so `fallback: true` must hand back a valid
        // empty cache rather than an error.
        let cache = create_guarded(&device, Some(&vec![0xABu8; 4096]), true)
            .expect("fallback must absorb a corrupt blob");
        drop(cache);
        // Shorter than the 64-byte header -> `Truncated`, same treatment.
        let cache = create_guarded(&device, Some(&[1u8, 2, 3]), true)
            .expect("fallback must absorb a truncated blob");
        drop(cache);
        // ...and without the fallback the SAME blob must come back as a
        // reportable error: that is what makes "HIT accepted" trustworthy.
        create_guarded(&device, Some(&vec![0xABu8; 4096]), false)
            .expect_err("without fallback a corrupt blob must be reported");
    }

    #[test]
    fn foreign_blob_is_an_error_not_a_panic() {
        let Some((device, _adapter, _gpu)) = cache_device() else {
            eprintln!("foreign_blob_is_an_error_not_a_panic: no PIPELINE_CACHE adapter, skipping");
            return;
        };
        // Without the error scope this call takes the whole process down.
        // `fallback: true` on purpose: DeviceMismatch is the one rejection wgpu
        // refuses to absorb, so even the forgiving call must surface it.
        let err = create_guarded(&device, Some(&foreign_adapter_blob()), true)
            .expect_err("a blob from another device must be reported, not accepted");
        assert!(!err.is_empty(), "error must carry a description");
    }

    #[test]
    fn open_recovers_from_a_foreign_blob_on_disk() {
        let Some((device, adapter, _gpu)) = cache_device() else {
            eprintln!("open_recovers_from_a_foreign_blob_on_disk: no PIPELINE_CACHE adapter, skipping");
            return;
        };
        let scratch = Scratch::new("foreign");
        // First open just to learn the filename this adapter maps to.
        let probe = ShaderCache::open_in(&device, &adapter, scratch.0.clone());
        let path = probe.path().expect("enabled cache must have a path").to_path_buf();
        drop(probe);

        std::fs::write(&path, foreign_adapter_blob()).expect("plant foreign blob");
        // Must survive: reject the blob, retry empty, still hand out a handle.
        let cache = ShaderCache::open_in(&device, &adapter, scratch.0.clone());
        assert!(
            cache.handle().is_some(),
            "a foreign blob must cost a recompile, not the cache"
        );
    }

    #[test]
    fn blob_round_trips_and_hits_on_reopen() {
        let Some((device, adapter, _gpu)) = cache_device() else {
            eprintln!("blob_round_trips_and_hits_on_reopen: no PIPELINE_CACHE adapter, skipping");
            return;
        };
        let scratch = Scratch::new("roundtrip");

        // Cold: no file, so nothing is loaded.
        let cold = ShaderCache::open_in(&device, &adapter, scratch.0.clone());
        assert!(cold.handle().is_some(), "PIPELINE_CACHE device must yield a cache");
        assert_eq!(cold.loaded_len(), 0, "no blob should exist yet");

        // A pipeline has to be compiled THROUGH the cache or the driver has
        // nothing to hand back.
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cache round-trip probe"),
            source: wgpu::ShaderSource::Wgsl(
                "@group(0) @binding(0) var<storage, read_write> o: array<u32>;\n\
                 @compute @workgroup_size(64) fn cs(@builtin(global_invocation_id) g: vec3<u32>) {\n\
                 o[g.x] = o[g.x] * 3u + 1u;\n}"
                    .into(),
            ),
        });
        let _p = create_compute_timed(
            &device,
            &cold,
            "test",
            wgpu::ComputePipelineDescriptor {
                label: Some("cache round-trip probe"),
                layout: None,
                module: &module,
                entry_point: Some("cs"),
                compilation_options: Default::default(),
                cache: None,
            },
        );
        cold.persist();

        let path = cold.path().expect("enabled cache must have a path").to_path_buf();
        let written = std::fs::metadata(&path).expect("persist must create the file").len();
        assert!(written >= 64, "blob must carry at least wgpu's header, got {written}");
        drop(cold);

        // Warm: the same adapter must map to the same filename AND the driver
        // must accept the blob (a rejection would log and start empty, so
        // `handle` alone is not proof - the loaded length is).
        let warm = ShaderCache::open_in(&device, &adapter, scratch.0.clone());
        assert_eq!(warm.path(), Some(path.as_path()), "filename must be stable per adapter");
        assert_eq!(
            warm.loaded_len() as u64,
            written,
            "reopen must read back exactly what persist wrote"
        );
        assert!(warm.handle().is_some());

        // Nothing new compiled, so the identical blob must not be rewritten.
        let before = std::fs::metadata(&path).unwrap().modified().unwrap();
        warm.persist();
        let after = std::fs::metadata(&path).unwrap().modified().unwrap();
        assert_eq!(before, after, "an unchanged cache must not be rewritten");
        let before_bytes = std::fs::read(&path).expect("blob must still be readable");

        // ...but a cache that DID grow must be, and the write must land on top
        // of the existing file (rename-over-existing, not a failed rename).
        let module2 = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cache round-trip probe 2"),
            source: wgpu::ShaderSource::Wgsl(
                "@group(0) @binding(0) var<storage, read_write> o: array<f32>;\n\
                 @compute @workgroup_size(32) fn cs(@builtin(global_invocation_id) g: vec3<u32>) {\n\
                 o[g.x] = sqrt(abs(o[g.x])) * 1.5 + sin(o[g.x]);\n}"
                    .into(),
            ),
        });
        let _p2 = create_compute_timed(
            &device,
            &warm,
            "test",
            wgpu::ComputePipelineDescriptor {
                label: Some("cache round-trip probe 2"),
                layout: None,
                module: &module2,
                entry_point: Some("cs"),
                compilation_options: Default::default(),
                cache: None,
            },
        );
        warm.persist();
        let grown = std::fs::read(&path).expect("blob must still be readable");
        assert_ne!(
            grown, before_bytes,
            "a newly compiled pipeline must change the persisted blob, and the write \
             must land on top of the existing file"
        );
    }

    /// Measures the REAL cost this module exists to remove: the driver compile
    /// of every raymarch entry point a launch builds, software variant then RT.
    ///
    /// Ignored because it is a wall-clock measurement, not an assertion, and a
    /// cold run takes minutes. It is a SINGLE-SHOT measurement on purpose:
    /// within one process the driver keeps its own in-memory cache, so a
    /// cold/warm A/B has to be two processes. Run it twice:
    ///
    ///   cargo test --lib -- --ignored --nocapture pipeline_compile_time
    ///
    /// The first run reports MISS and the cold time and writes the blob; the
    /// second reports HIT and the warm time. Delete the printed path to go cold
    /// again.
    ///
    /// CAVEAT that decides whether the result means anything: the driver keeps a
    /// cache of ITS OWN, and it makes the second run fast whether or not our
    /// blob is used. Isolating our contribution means pointing the driver's
    /// cache at a scratch directory and emptying that between runs. On NVIDIA
    /// (Windows and Linux) `__GL_SHADER_DISK_CACHE_PATH=<dir>` does it - checked
    /// on Windows, not assumed: the scratch directory fills up and
    /// `%LOCALAPPDATA%\NVIDIA\GLCache` stays byte-for-byte unchanged.
    ///
    /// Knobs: `VOXELG_SHADER_CACHE_DIR` moves the blob,
    /// `VOXELG_CACHE_BENCH_SKIP=cs_a,cs_b` drops entry points (use it when one
    /// is too slow to sit through - `cs_transparent` cold is ~2.5 minutes), and
    /// `VOXELG_CACHE_BENCH_VARIANT=sw|rt` measures only one of the two.
    #[test]
    #[ignore = "wall-clock driver compile measurement; run twice to see cold vs warm"]
    fn pipeline_compile_time() {
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
        .expect("no adapter");
        let info = adapter.get_info();
        eprintln!(
            "adapter: {} [{:?}] driver {} / {}",
            info.name, info.backend, info.driver, info.driver_info
        );
        eprintln!(
            "PIPELINE_CACHE supported: {}",
            adapter.features().contains(wgpu::Features::PIPELINE_CACHE)
        );
        // The RT half needs a ray-query device; the software half does not. Ask
        // for RT when the adapter has it and fall back to a plain device rather
        // than skipping the whole measurement, because the software variant is
        // what a DEFAULT launch (no VOXELG_RT) actually compiles.
        let want_rt = crate::accel::adapter_supports_rt(&adapter);
        let (device, _queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("pipeline cache bench device"),
            required_features: if want_rt {
                wgpu::Features::EXPERIMENTAL_RAY_QUERY | wanted_features(&adapter)
            } else {
                wanted_features(&adapter)
            },
            required_limits: adapter.limits(),
            memory_hints: wgpu::MemoryHints::Performance,
            experimental_features: if want_rt {
                unsafe { wgpu::ExperimentalFeatures::enabled() }
            } else {
                wgpu::ExperimentalFeatures::disabled()
            },
            trace: wgpu::Trace::Off,
        }))
        .expect("device");
        // Ignored test, so it runs alone; holding the guard costs nothing and
        // keeps a stray parallel GPU test from perturbing the measurement.
        let _gpu = gpu;

        // A directory that SURVIVES between runs, unlike Scratch.
        let dir = std::env::var_os(DIR_ENV)
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("voxelg-pipeline-cache-bench"));
        let cache = ShaderCache::open_in(&device, &adapter, dir);
        eprintln!(
            "cache: {} ({} KiB) at {:?}",
            if cache.loaded_len() > 0 { "HIT" } else { "MISS" },
            cache.loaded_len() / 1024,
            cache.path()
        );

        // Escape hatch for an entry point that is too slow to sit through, and
        // for a shader the driver's compiler cannot currently handle: without it
        // one bad entry point makes the whole measurement impossible.
        let skip = std::env::var("VOXELG_CACHE_BENCH_SKIP").unwrap_or_default();
        let skip: Vec<&str> = skip.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
        let only = std::env::var("VOXELG_CACHE_BENCH_VARIANT").unwrap_or_default();

        // Warm/cold is decided by what was on disk when the cache OPENED, not by
        // what it holds now: the software half persists before the RT half runs.
        let warm = cache.loaded_len() > 0;
        let mut grand = 0.0;
        for rt in [false, true] {
            let name = if rt { "RT" } else { "software" };
            let tag = if rt { "rt" } else { "sw" };
            if !only.is_empty() && only != tag {
                eprintln!("--- {name} variant: SKIPPED (VOXELG_CACHE_BENCH_VARIANT={only})");
                continue;
            }
            if rt && !want_rt {
                eprintln!("--- RT variant: SKIPPED (adapter has no ray query)");
                continue;
            }
            eprintln!("--- {name} variant ---");
            grand += measure_variant(&device, &cache, rt, &skip);
            // Between the halves as well as at the end: the software half alone
            // is minutes of work on a cold machine and must not be lost if the
            // RT half is interrupted.
            cache.persist();
        }
        eprintln!(
            "GRAND TOTAL: {grand:.2}s ({})",
            if warm { "warm" } else { "cold" }
        );
    }

    /// Compile one variant's entry points through `cache`, timing each, and
    /// return the total seconds. Layouts, override constants and entry points
    /// are `Renderer::new`'s, so the driver sees exactly the pipelines a launch
    /// asks it for; anything else would measure a different program.
    fn measure_variant(
        device: &wgpu::Device,
        cache: &ShaderCache,
        rt: bool,
        skip: &[&str],
    ) -> f64 {
        // The shader source is the cache KEY as far as the driver is concerned.
        // Printing its hash makes a cold/warm pair verifiable: if the two runs
        // report different hashes the shaders were edited in between and the
        // comparison is meaningless, not evidence.
        let src = crate::renderer::raymarch_source_variant(rt);
        eprintln!(
            "  source: {} bytes, fnv1a64 {:016x}",
            src.len(),
            fnv1a64(src.as_bytes())
        );
        let t_mod = Instant::now();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raymarch bench module"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        eprintln!("  shader module (naga + SPIR-V): {:.2}s", t_mod.elapsed().as_secs_f64());

        let compute_bgl = crate::renderer::create_compute_bgl(device);
        // Held out here rather than inside the `if`: the pipeline layout only
        // borrows it for the call, but keeping the binding alive documents that
        // group 1 exists solely in the RT configuration.
        let rt_bgl = rt.then(|| crate::renderer::create_rt_bgl(device));
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bench compute pl"),
            bind_group_layouts: &match &rt_bgl {
                Some(b) => vec![Some(&compute_bgl), Some(b)],
                None => vec![Some(&compute_bgl)],
            },
            immediate_size: 0,
        });
        // The shipped RT config (hardware primary trace); empty for software.
        let consts: &[(&'static str, f64)] = if rt { &[("RT_PRIMARY", 1.0)] } else { &[] };

        // Every entry point built off the shared layout, in launch order.
        // `cs_gi_probe_update` exists only in the RT source; `cs_clouds` has its
        // own layout and is measured separately below.
        let shared: &[&str] = if rt {
            &[
                "cs_godrays",
                "cs_main",
                "cs_compose",
                "cs_transparent",
                "cs_gi_probe_update",
                "cs_voxel_light_update",
            ]
        } else {
            &[
                "cs_godrays",
                "cs_main",
                "cs_compose",
                "cs_transparent",
                "cs_voxel_light_update",
            ]
        };

        let mut counted = 0;
        let mut total = 0.0;
        let mut compile = |entry: &str, layout: &wgpu::PipelineLayout| {
            if skip.contains(&entry) {
                eprintln!("    {entry}: SKIPPED");
                return;
            }
            counted += 1;
            let t = Instant::now();
            let _p = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: consts,
                    ..Default::default()
                },
                cache: cache.handle(),
            });
            let secs = t.elapsed().as_secs_f64();
            total += secs;
            eprintln!("    {entry}: {:.0} ms", secs * 1e3);
        };
        for entry in shared {
            compile(entry, &layout);
        }
        // cs_clouds is a raymarch entry point too, and a launch always pays it -
        // it just needs the narrow cloud layout instead of the render one. Only
        // the software module has it; the RT branch leaves clouds to this one.
        if !rt {
            let cloud_bgl = crate::renderer::create_cloud_bgl(device);
            let cloud_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("bench cloud pl"),
                bind_group_layouts: &[Some(&cloud_bgl)],
                immediate_size: 0,
            });
            compile("cs_clouds", &cloud_pl);
        }
        eprintln!(
            "  {counted} {} PIPELINES: {total:.2}s",
            if rt { "RT" } else { "SOFTWARE" }
        );
        total
    }

    #[test]
    fn prune_only_touches_same_adapter_blobs() {
        let scratch = Scratch::new("prune");
        let dir = &scratch.0;
        let prefix = "voxelg-wgpu_pipeline_cache_vulkan_1_2-";
        let keep = format!("{prefix}0000000000000001.bin");
        let stale = format!("{prefix}0000000000000002.bin");
        let other = "voxelg-wgpu_pipeline_cache_vulkan_9_9-000000000000000a.bin";
        let alien = "someone-elses-file.bin";
        for f in [keep.as_str(), stale.as_str(), other, alien] {
            std::fs::write(dir.join(f), b"x").unwrap();
        }

        prune_stale(dir, prefix, &keep);

        assert!(dir.join(&keep).exists(), "current blob must survive");
        assert!(!dir.join(&stale).exists(), "old driver's blob must be reclaimed");
        assert!(dir.join(other).exists(), "a second GPU's blob must survive");
        assert!(dir.join(alien).exists(), "unrelated files must never be touched");
    }
}
