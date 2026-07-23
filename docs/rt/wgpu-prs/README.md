# Prepared wgpu contributions

Upstream contributions to wgpu's experimental hardware ray-tracing API, discovered
while building voxelG's RT path. Each is PREPARED (branch + patch + a PR body ready
to paste) but NOT opened - review, run the gate, then open on github.com/gfx-rs/wgpu.

Working fork (local, cloned from the pinned commit c97d22f, offline): `~/wgpu-contrib`.

Before opening ANY of these, from `~/wgpu-contrib` run wgpu's own gate (from
`AGENTS.md`): `cargo fmt`, `cargo clippy --tests`, then `cargo xtask test` and
`cargo xtask cts --backend vulkan`. No DCO/sign-off is required. Author identity:
`mstampfli <marc.kurt.stampfli@icloud.com>` (@mstampfli).

CHANGELOG placeholder procedure (learned the hard way on PR 1): placeholders are
`#0000`, which can never be a real PR number - `#9999` WAS one, and a blind sed
also rewrote a historical entry in a released section, which wgpu's changelog
checker rejects (released sections must never change). After opening the PR and
getting the real number, replace the placeholder ONLY on the line containing
`@mstampfli`, then verify with `git diff <base> -- CHANGELOG.md` that the full
diff against base is EXACTLY the one new entry under `## Unreleased` before
force-pushing the amend.

## PR 1 - docs: correct BlasAabbGeometry stride reference + document AABB layout

- Branch: `docs/aabb-geometry-stride-and-layout` (in `~/wgpu-contrib`).
- Patch: [`0001-blas-aabb-docs.patch`](0001-blas-aabb-docs.patch).
- Status: OPENED 2026-07-23 as [#9934](https://github.com/gfx-rs/wgpu/pull/9934),
  rebased on trunk, all 60 CI checks green; awaiting maintainer review.
  (Lesson recorded below: the placeholder swap first corrupted a released
  changelog entry and failed CI; fixed with the anchored procedure.)

`BlasAabbGeometry`'s docs referred to a `size.stride` field, but
`BlasAABBGeometrySizeDescriptor` has no `stride` - the stride field lives on
`BlasAabbGeometry` itself. The patch points the docs at the correct field and
documents the packed AABB buffer layout (each primitive a minimum then a maximum
corner, two consecutive `vec3<f32>`, the 24-byte `AABB_GEOMETRY_MIN_STRIDE`), which
was only implied before.

PR body to paste:

> **Connections**
> None.
>
> **Description**
> `BlasAabbGeometry`'s documentation referred to a `size.stride` field, but
> `BlasAABBGeometrySizeDescriptor` (the `size` sub-descriptor) has no `stride`
> field - the stride lives on `BlasAabbGeometry` itself. This fixes the two stale
> references and documents the packed AABB buffer layout (minimum then maximum
> corner, two consecutive `vec3<f32>`, the 24-byte `AABB_GEOMETRY_MIN_STRIDE`),
> which was previously only implied by the `ray_aabb_compute` example.
>
> **Testing**
> Documentation only, no code change; `cargo fmt` clean and the docs build.
>
> **Squash or Rebase?**
> Squash.

## PR 2 - remove the dead `CreateBlasError::InvalidAabbStride` variant

- Branch: `cleanup/remove-dead-createblaserror-aabb-stride` (in `~/wgpu-contrib`).
- Patch: [`0002-remove-dead-createblaserror-variant.patch`](0002-remove-dead-createblaserror-variant.patch).
- Status: OPENED 2026-07-23 as [#9935](https://github.com/gfx-rs/wgpu/pull/9935),
  rebased on trunk first (variant re-verified dead there), changelog placeholder
  swapped via the anchored procedure, CI watcher armed. Connections links #9934
  as a sibling cleanup (same audit, no dependency). Note this removes a public
  (experimental) error variant, a technically-breaking change, so a maintainer may
  prefer to keep it reserved - the PR body says so and offers that alternative.

The variant is declared and matched in the `WebGpuError` impl but NEVER constructed:
`create_blas` takes no stride, so a stride error is impossible there - the real check
is `BuildAccelerationStructureError::InvalidAabbStride` at build time
(`wgpu-core/src/command/ray_tracing.rs`). Only two references exist (both in
`wgpu-core/src/ray_tracing.rs`), both removed.

PR body to paste:

> **Connections**
> None.
>
> **Description**
> `CreateBlasError::InvalidAabbStride` is declared and handled in the `WebGpuError`
> impl but is never constructed. `create_blas` has no stride input, so a stride
> error cannot originate there; AABB stride is validated only at build time as
> `BuildAccelerationStructureError::InvalidAabbStride`. This drops the dead variant
> and its match arm. It is a (technically breaking) removal from an experimental
> API - happy to instead keep it as a reserved/`#[doc(hidden)]` variant if you would
> rather not narrow the enum; let me know your preference.
>
> **Testing**
> `cargo check -p wgpu-core` passes; no other references to the variant exist.
>
> **Squash or Rebase?**
> Squash.

## PR 4 - clear error when `dxc` is missing in the passthrough tests

- Branch: `test/dxc-missing-clear-error` (in `~/wgpu-contrib`, based on
  upstream/trunk d5977df45).
- Patch: [`0004-dxc-missing-clear-error.patch`](0004-dxc-missing-clear-error.patch).
- Status: READY. Verified both ways on this box: without dxc the passthrough
  tests now panic with a message naming dxc and the remedy (was a bare
  "No such file or directory"); with dxc installed (official Linux release
  v1.9.2602.24 in ~/.local/dxc, wrapper in ~/.local/bin) all Vulkan
  passthrough tests PASS on Intel, NVIDIA and llvmpipe. Tests-only change,
  no CHANGELOG entry.

PR body to paste:

> **Connections**
> None.
>
> **Description**
> The SPIR-V and DXIL passthrough tests compile `shader.hlsl` at runtime by
> shelling out to `dxc`, but the spawn result is bare-unwrapped: on a machine
> without the DirectX Shader Compiler every passthrough test fails with only
> `No such file or directory (os error 2)` and no hint which tool is missing.
> CI installs dxc via `.github/actions/install-dxc`, so only local runs ever
> hit this. This names the tool and the remedy in the panic.
>
> **Testing**
> Without dxc on PATH the passthrough tests now fail with the new message;
> with dxc installed they pass on Vulkan (Intel ADL, NVIDIA Blackwell,
> llvmpipe) on Linux.
>
> **Squash or Rebase?**
> Squash.

## PR 3 - perf: reuse the acceleration-structure build scratch buffer

- Branch: `perf/cache-acceleration-structure-scratch` (in `~/wgpu-contrib`,
  stacked on the PR 2 branch; the diff is independent - no file overlap - so it
  applies to trunk cleanly).
- Patch: [`0003-scratch-buffer-reuse.patch`](0003-scratch-buffer-reuse.patch).
- Status: READY. `cargo clippy -p wgpu-core --tests` clean, `cargo fmt` clean;
  run `cargo xtask test` (needs cargo-nextest) before opening. CHANGELOG entry
  uses placeholder #10000 - set the real PR number.
- Evidence (voxelG, RTX 5060, Vulkan): rebuilding a ~300k-primitive AABB BLAS
  per streaming update dropped from 12-28 ms to 1.2-2.0 ms of CPU per
  `build_acceleration_structures` call (measured via a `[patch]` override).

`build_acceleration_structures` allocated a fresh scratch buffer per call
(`ScratchBuffer::new`) and freed it at submission retirement (TempResource).
Scratch size is proportional to the geometry, so per-frame rebuilds of large
acceleration structures paid a large Vulkan allocate/free every build. The fix
parks the most recent scratch on the device: `ScratchBuffer::new` takes the
parked buffer when it is big enough, `Drop` parks it back (keeping the larger).
Every `ScratchBuffer::drop` happens at a GPU-idle point for that buffer
(submission retirement, pre-submit error paths, teardown), so reuse needs no new
synchronization. The cache stores the raw hal buffer without an `Arc<Device>`
(no cycle); `Device::drop` destroys a parked buffer. A new leaf lock rank
`DEVICE_SCRATCH_BUFFER_CACHE` is registered and added to the followed-by sets of
the ranks that can hold locks around `ScratchBuffer::new`/`drop`.

PR body to paste:

> **Connections**
> None (found while building a streaming voxel world on the experimental
> ray-tracing API; happy to file a tracking issue if preferred).
>
> **Description**
> `build_acceleration_structures` allocates a fresh scratch buffer per call and
> frees it when the submission retires. Applications that rebuild large
> acceleration structures every frame (streaming worlds, dynamic scenes) pay a
> large allocate/free per build - rebuilding a ~300k-primitive AABB BLAS cost
> 12-28 ms of CPU per call on Vulkan/NVIDIA, dominated by the scratch
> allocation.
>
> This parks the most recent scratch buffer on the device and reuses it:
> `ScratchBuffer::new` takes the parked buffer when it is large enough, and
> `Drop` parks it back, keeping the larger of the two. A `ScratchBuffer` only
> drops once its submission retired (it rides `TempResource`) or before it was
> ever submitted, so a parked buffer is always GPU-idle and reuse introduces no
> new synchronization requirements. The cache holds the raw hal buffer only (no
> `Arc<Device>`, so no reference cycle); the device destroys a parked buffer in
> its own `Drop`. Memory-wise the device retains at most one scratch buffer of
> the peak scratch size; happy to add a trim policy if you would rather not
> retain it.
>
> With this change the same rebuild takes 1.2-2.0 ms per call (>10x less).
>
> **Testing**
> Existing ray-tracing tests pass (`cargo xtask test`); exercised heavily by a
> voxel engine rebuilding a ~300k-AABB BLAS per streaming update via a `[patch]`
> override, validated against a CPU raycast oracle.
>
> **Squash or Rebase?**
> Squash.
