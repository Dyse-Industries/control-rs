# MVP Plan — Math & Numerical Models Chronology

| Stage | Document / Milestone                                     | Upstream                                      | Status / Gate |
|:------|:---------------------------------------------------------|:----------------------------------------------|:--------------|
| 0     | `num-types-design.md`                                    | —                                             | Draft (Phase 1b) |
| 0     | `num-traits-design.md`                                   | —                                             | Draft (Phase 1b) |
| 0     | `error-design.md`                                        | —                                             | Draft (Phase 1b) |
| 1     | `storage-design.md`                                      | `num-types`, `num-traits`, `error`            | Draft (Phase 1b) |
| 1     | `subprograms-design.md`                                  | `storage`, `num-traits`, `error`              | Draft (Phase 1b) |
| 1b    | `verification-gap-plan.md`                               | Stage 0–1 math designs                        | Draft — close §2 requirements and §6 oracles before further math tests |
| 1     | `../numerical-models/matrix-design.md`                   | `storage`, `subprograms`                      | Draft         |
| 1     | `../numerical-models/polynomial-design.md`               | `storage`, `subprograms`, `matrix`            | Draft         |
| 1     | `../numerical-models/state-space-design.md`              | `storage`, `subprograms`, `matrix`            | Draft         |
| 1     | `../numerical-models/transfer-function-design.md`        | `storage`, `subprograms`, `matrix`, `polynomial` | Draft      |
| 1     | `../numerical-models/tensor-design.md`                   | `storage`, `matrix`                           | Draft         |
| 3     | `src/math/storage.rs` & `subprograms.rs`                 | Stage 1 Approved                              | ...           |
| 4     | `src/matrix` & `src/polynomial`                          | Stage 3                                       | ...           |
| 5     | `src/state_space`, `src/transfer_function`, `src/tensor` | Stage 4                                       | ...           |
| 6     | HIL Verification (`control-rs-hil`)                      | Stage 5                                       | ...           |

The numerical-model retarget landed August 24, 2026. Every Stage-1 model
document, and the four `../control-toolboxes/` documents downstream of them,
now cite `storage-design.md` Rev 1.8 and `subprograms-design.md` Rev 1.6; no
body text describes the pre-split `storage-subprograms-design.md` hierarchy
(`MatrixStorage` / `BlasStorage`). They stay Draft pending human review of
the three consequences listed in `../numerical-models/mvp-plan.md`, the
load-bearing one being that the dense and packed subsystems share no
supertrait, which forces `Matrix` and `PackedMatrix` apart
(`matrix-design.md` §4.1.1).

Because the retarget was written against `storage-design.md` Rev 1.8 and
`subprograms-design.md` Rev 1.6, any Phase-1b change to those two documents
is a re-review trigger for all five model documents.

**Badge / plan mismatch**: this table records `storage-design.md` and
`subprograms-design.md` as Draft (Phase 1b), but both still display
**Approved** status badges dated August 22, 2026. `CLAUDE.md` makes the
badge the gate that unblocks `/cr-implement`, so the two must be reconciled —
either revert the badges to Draft or drop those rows back to Approved —
before any implementation phase runs.

The five math designs are Draft pending re-review of the verification-gap
Phase 1 edits (`verification-gap-plan.md`, Stage 1b). Re-approval of those
documents is the gate for further math-layer tests and for Stage 3.
