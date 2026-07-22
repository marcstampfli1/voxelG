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

## Candidate (not prepared - needs a maintainer decision)

**Remove the dead `CreateBlasError::InvalidAabbStride` variant.** It is declared
(`wgpu-core/src/ray_tracing.rs:52`) and matched in the `WebGpuError` impl (line 64)
but is NEVER constructed: `create_blas` takes no stride, so a stride error is
impossible there - the real check is `BuildAccelerationStructureError::InvalidAabbStride`
at build time (`wgpu-core/src/command/ray_tracing.rs:1086`). Removing it is a small
cleanup but a breaking change to a public (experimental) error enum, so it wants a
maintainer's call (remove vs. keep reserved) and a run of wgpu's test gate before
proposing. Left as a note rather than a blind breaking PR.
