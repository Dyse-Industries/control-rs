# MVP Plan — Numerical Models Chronology

| Document                      | Upstream Dependency                                                         | Status Badge |
|:------------------------------|:----------------------------------------------------------------------------|:-------------|
| `matrix-design.md`            | `../math/storage-design.md`, `../math/subprograms-design.md`                | Draft        |
| `polynomial-design.md`        | `../math/storage-design.md`, `../math/subprograms-design.md`, `matrix-design.md` | Draft   |
| `state-space-design.md`       | `../math/storage-design.md`, `../math/subprograms-design.md`, `matrix-design.md` | Draft   |
| `transfer-function-design.md` | `../math/storage-design.md`, `../math/subprograms-design.md`, `matrix-design.md`, `polynomial-design.md` | Draft |
| `tensor-design.md`            | `../math/storage-design.md`, `matrix-design.md`                             | Draft        |

These documents still cite the deleted `storage-subprograms-design.md` in
their body text. Status stays Draft until a dedicated retarget pass rewrites
those citations onto `storage-design.md` / `subprograms-design.md`. Do not
`/cr-implement` them against the pre-split hierarchy.
