# MVP Plan — Math & Numerical Models Chronology

| Stage | Document / Milestone                                     | Upstream                                      | Status / Gate |
|:------|:---------------------------------------------------------|:----------------------------------------------|:--------------|
| 0     | `num-types-design.md`                                    | —                                             | Approved      |
| 0     | `num-traits-design.md`                                   | —                                             | Approved      |
| 0     | `error-design.md` (Rev 1.4)                              | —                                             | Reviewed      |
| 1     | `storage-subprograms-design.md` (Rev 1.16)               | `num-types`, `num-traits`, `error`            | Draft         |
| 1     | `matrix-design.md` (Rev 1.26)                            | `storage-subprograms`                         | Reviewed      |
| 1     | `polynomial-design.md` (Rev 1.18)                        | `storage-subprograms`, `matrix`               | Reviewed      |
| 1     | `state-space-design.md` (Rev 1.13)                       | `storage-subprograms`, `matrix`               | Reviewed      |
| 1     | `transfer-function-design.md` (Rev 1.15)                 | `storage-subprograms`, `matrix`, `polynomial` | Reviewed      |
| 1     | `tensor-design.md` (Rev 1.12)                            | `Buffer`, `matrix`                            | Reviewed      |
| 3     | `src/math/storage.rs` & `subprograms.rs`                 | Stage 2                                       | ...           |
| 4     | `src/matrix` & `src/polynomial`                          | Stage 3                                       | ...           |
| 5     | `src/state_space`, `src/transfer_function`, `src/tensor` | Stage 4                                       | ...           |
| 6     | HIL Verification (`control-rs-hil`)                      | Stage 5                                       | ...           |
