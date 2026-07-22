# Prepared wgpu contributions

Upstream contributions to wgpu's experimental hardware ray-tracing API, discovered
while building voxelG's RT path. Each is PREPARED (branch + patch + a PR body ready
to paste) but NOT opened - review, run the gate, then open on github.com/gfx-rs/wgpu.

Working fork (local, cloned from the pinned commit c97d22f, offline): `~/wgpu-contrib`.

Before opening ANY of these, from `~/wgpu-contrib` run wgpu's own gate (from
`AGENTS.md`): `cargo fmt`, `cargo clippy --tests`, then `cargo xtask test` and
`cargo xtask cts --backend vulkan`. Add the real PR number to the CHANGELOG line
(the patches use a placeholder). No DCO/sign-off is required. Author identity:
`mstampfli <marc.kurt.stampfli@icloud.com>` (@mstampfli).

## PR 1 - docs: correct BlasAabbGeometry stride reference + document AABB layout

- Branch: `docs/aabb-geometry-stride-and-layout` (in `~/wgpu-contrib`).
- Patch: [`0001-blas-aabb-docs.patch`](0001-blas-aabb-docs.patch).
- Status: READY. Prose-doc change only (no code), fmt-clean.

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
- Status: READY, `cargo check -p wgpu-core` clean. Note this removes a public
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
