# MVP Plan — Math & Numerical Models Chronology

| Stage | Document / Milestone                                     | Upstream                                      | Status / Gate |
|:------|:---------------------------------------------------------|:----------------------------------------------|:--------------|
| 0     | `num-types-design.md`                                    | —                                             | Approved      |
| 0     | `num-traits-design.md`                                   | —                                             | Approved      |
| 0     | `error-design.md` (Rev 1.6)                              | —                                             | Approved      |
| 1     | `storage-design.md`                                      | `num-types`, `num-traits`, `error`            | Approved      |
| 1     | `subprograms-design.md`                                  | `storage`, `num-traits`, `error`              | Approved      |
| 1     | `../numerical-models/matrix-design.md`                   | `storage`, `subprograms`                      | Draft         |
| 1     | `../numerical-models/polynomial-design.md`               | `storage`, `subprograms`, `matrix`            | Draft         |
| 1     | `../numerical-models/state-space-design.md`              | `storage`, `subprograms`, `matrix`            | Draft         |
| 1     | `../numerical-models/transfer-function-design.md`        | `storage`, `subprograms`, `matrix`, `polynomial` | Draft      |
| 1     | `../numerical-models/tensor-design.md`                   | `storage`, `matrix`                           | Draft         |
| 3     | `src/math/storage.rs` & `subprograms.rs`                 | Stage 1 Approved                              | ...           |
| 4     | `src/matrix` & `src/polynomial`                          | Stage 3                                       | ...           |
| 5     | `src/state_space`, `src/transfer_function`, `src/tensor` | Stage 4                                       | ...           |
| 6     | HIL Verification (`control-rs-hil`)                      | Stage 5                                       | ...           |

Numerical-model documents still describe the pre-split
`storage-subprograms-design.md` hierarchy (`MatrixStorage` / `BlasStorage`).
They stay Draft until a dedicated `/cr-design-doc numerical-models/<slug>`
retarget onto `storage-design.md` and `subprograms-design.md`.
