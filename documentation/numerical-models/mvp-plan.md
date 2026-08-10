# MVP Plan

This document sequences the implementation work described in the five
`numerical-models` design docs (`matrix`, `polynomial`, `tensor`,
`state-space`, `transfer-function`), based on the interoperability
dependencies each doc declares in its own Development Plan section.

| Document                      | Status Badge |
|-------------------------------|:------------:|
| `matrix-design.md`            |    Draft     |
| `polynomial-design.md`        |    Draft     |
| `tensor-design.md`            |    Draft     |
| `state-space-design.md`       |    Draft     |
| `transfer-function-design.md` |    Draft     |

This plan is a staging reference for review/approval order and downstream
sequencing.

## Staging

| Phase                                             | Scope                                                              | Steps (from each doc)                                                                                                                                                                                         | Cross-Module Dependency                                                     | Est. Effort |
|:--------------------------------------------------|:-------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------|:------------|
| **1. Independent Cores**                          | `Matrix`, `Polynomial`, `Tensor`, `TransferFunction` cores         | Matrix 1–2 (Core Layout, Operators); Polynomial 1–3 (Storage, Arithmetic, Evaluation/Division); Tensor 1 (Core Layout & Storage); TransferFunction 1–2 (Storage Wrapper, Frequency Evaluation)                | None beyond `src/math` (shipped)                                            | ~10.5 days  |
| **2. Type-Internal Depth**                        | Advanced operations within each type                               | Matrix 3–5 (Solvers, Specializations, Factorizations); Tensor 2–3 (Contraction, Grid Interpolation/Activation); TransferFunction 3–4 (Algebra/DSP, Discretization); StateSpace 1 (Core Struct)                | StateSpace 1 needs Matrix 1 (`ContiguousStorage`)                           | ~19.5 days  |
| **3. Cross-Type Interop + State-Space Mechanics** | Conversions between types; State-Space built out                   | Matrix 7 (interop); Polynomial 4 (interop); Tensor 4–5 (Quantized type + interop); StateSpace 2–5 (Views, Interconnections, Discretization, Structural Analysis); TransferFunction 5 (State-Space conversion) | Requires Phase 1–2 cores of Matrix, Polynomial, Tensor, StateSpace to exist | ~22.5 days  |
| **4. Verification**                               | Golden-value/proptest suites, hardware profiling, cross-validation | Matrix 6; Polynomial 5; Tensor 5 (verification portion); StateSpace 6; TransferFunction 6                                                                                                                     | Runs against completed Phase 1–3 code per module                            | ~12.5 days  |
