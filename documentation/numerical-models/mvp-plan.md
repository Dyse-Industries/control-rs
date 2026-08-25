# MVP Plan — Numerical Models Chronology

| Document                      | Upstream Dependency                                                         | Status Badge |
|:------------------------------|:----------------------------------------------------------------------------|:-------------|
| `matrix-design.md`            | `../math/storage-design.md`, `../math/subprograms-design.md`                | Draft        |
| `polynomial-design.md`        | `../math/storage-design.md`, `../math/subprograms-design.md`, `matrix-design.md` | Draft   |
| `state-space-design.md`       | `../math/storage-design.md`, `../math/subprograms-design.md`, `matrix-design.md` | Draft   |
| `transfer-function-design.md` | `../math/storage-design.md`, `../math/subprograms-design.md`, `matrix-design.md`, `polynomial-design.md` | Draft |
| `tensor-design.md`            | `../math/storage-design.md`, `matrix-design.md`                             | Draft        |

The retarget pass of August 24, 2026 landed: every document now cites
`storage-design.md` Rev 1.8 and `subprograms-design.md` Rev 1.6, and no body
text references the deleted `storage-subprograms-design.md`, `MatrixStorage`,
`BlasStorage` or `StridedView`. Status stays Draft pending human review of the
three consequences the retarget surfaced, each recorded in its document's §7:

1. **Per-branch wrappers** (`matrix-design.md` §4.1.1, §5.4). The dense and
   packed storage subsystems share no supertrait, so no single struct bound
   admits both. `Matrix` binds `DenseStorage`; `PackedMatrix` binds
   `PackedStorage`. This reverses the Rev 1.31 single-struct decision and
   duplicates the constructor and inspection surface.
2. **`Complex<T>` reaches the models** (`num-traits-design.md` §4.3). Every
   ring operator and kernel now admits it. The Hermitian wrappers and the
   `Hemv`/`Hemm`/`Herk` specializations have no counterpart in
   `matrix-design.md` §4.10 yet, and `Convolution<T>` in `src/math/dsp.rs`
   is still declared over the narrower `T: Float`.
3. **Sparse subsystem unclaimed** (`storage-design.md` FR-11 to FR-15;
   `subprograms-design.md` FR-5). No wrapper in these documents consumes CSR,
   CSC, COO or sparse vectors. Whether sparse plant dynamics belong in
   `state-space-design.md` or in a document of their own is unscoped.

Stage 0 upstreams (`num-types-design.md`, `num-traits-design.md`,
`error-design.md`) currently display **Draft** badges dated August 24, 2026,
having been reopened after `../math/mvp-plan.md` last recorded them Approved.
The retarget cites their current content. Re-approve them before
`/cr-implement` runs against anything downstream.
