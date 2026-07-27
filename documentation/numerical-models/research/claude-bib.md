# Design-Document Bibliographies — `control-rs` Numerical Models

Source: `numerical_models_research_topics.md`. Each section below is written as
a drop-in `## 10. References` for that design document.

**How these were built**

- Base list = the report's "Summary Mapping of References to Design Documents"
  table.
- Within each doc, references are split into tiers — *
  *practical/performance-analytical work first** (complexity bounds,
  memory/layout analysis, or reported runtime data), then
  theoretical/algorithmic foundations — per your instruction to prioritize
  practical relevance.
- Every entry has a one-line note on *what* it actually supplies, so it's clear
  why it earns a citation rather than just being a name-drop.
- Two additions beyond the table, both traceable to the report's own domain
  write-ups (flagged inline below): **Aurentz et al. (2014)** as an optional
  addition to `polynomial-design.md`, and **Moler & Van Loan (2003)** added to
  `transfer-function-design.md`.
- Domains 8 (type-level safety) and 9 (verification/testing) are marked "
  Primary: All 5 Design Docs" in the report but only partly reflected in the
  table — pulled into one shared **Cross-Cutting** block at the end instead of
  duplicated five times.
- `Horner (1819)` appeared in the table with no title/journal — completed via
  lookup.
- The report notes `matrix-design.md` already has a preliminary bibliography —
  reconcile/dedupe against what's already there rather than replacing it
  wholesale.

---

## `matrix-design.md`

*Section 10 — References*

**Practical & Performance-Analytical**

1. **Golub, G. H., & Van Loan, C. F. (2013).** *Matrix Computations* (4th ed.).
   Johns Hopkins University Press. — flop-count basis for every in-place
   factorization (O(N³/3) Cholesky, O(N³) LU/QR).
2. **Anderson, E., et al. (1999).** *LAPACK Users' Guide* (3rd ed.). SIAM. —
   reference performance/blocking conventions behind the BLAS-backed solver
   routines.
3. **Frison, G., et al. (2018).** BLASFEO: Basic Linear Algebra Subroutines for
   Embedded Optimization. *ACM Transactions on Mathematical Software*. — direct
   embedded runtime benchmarks and a panel-major vs. column-major memory-layout
   comparison.
4. **Bini, D. A., et al. (2010).** A Fast Implicit QR Eigenvalue Algorithm for
   Companion Matrices. *Linear Algebra and its Applications*, 432(8),
   2006–2031. — explicit O(N³)→O(N²) time / O(N) space reduction for the
   companion-matrix eigenvalue solver.
5. **Aurentz, J. L., Mach, T., Vandebril, R., & Watkins, D. S. (2014).** Fast
   and backward stable computation of roots of polynomials. *TW Reports*, KU
   Leuven. — speed-vs-backward-stability trade-off data for that same pipeline.
6. **Higham, N. J. (2002).** *Accuracy and Stability of Numerical Algorithms* (
   2nd ed.). SIAM. — condition-number/error-bound analysis behind the "never
   form AᵀA" and pivoting-strategy rules.
7. **Yiu, J. (2013).** *The Definitive Guide to ARM Cortex-M3 and Cortex-M4
   Processors* (3rd ed.). Newnes. — FPU register count and cache-line behavior
   the micro-architecture assumptions rely on.

**Theoretical & Algorithmic Foundations**

8. **Faddeev, D. K., & Faddeeva, V. N. (1963).** *Computational Methods of
   Linear Algebra*. W. H. Freeman and Company. — classical derivation behind the
   division-free Faddeev–LeVerrier formulation.

---

## `polynomial-design.md`

*Section 10 — References*

**Practical & Performance-Analytical**

1. **Bini, D. A., et al. (2010).** A Fast Implicit QR Eigenvalue Algorithm for
   Companion Matrices. *Linear Algebra and its Applications*, 432(8),
   2006–2031. — the O(N²)-time/O(N)-space rootfinding result the module
   implements.
2. **Horner, W. G. (1819).** A New Method of Solving Numerical Equations of All
   Orders, by Continuous Approximation. *Philosophical Transactions of the Royal
   Society of London*, 109, 308–335. — origin of the N−1-operation evaluation
   scheme; direct FLOP-count justification for `evaluate`.
3. **Claessen, K., & Hughes, J. (2000).** QuickCheck: A Lightweight Tool for
   Random Testing of Haskell Programs. *ACM SIGPLAN Notices*, 35(9), 268–279. —
   methodology behind the `proptest` identity checks (
   e.g., $P+Q=Q+P$, $P=QD+R$).

**Theoretical & Algorithmic Foundations**

4. **Henrici, P. (1974).** *Applied and Computational Complex Analysis, Volume
   1*. Wiley. — zero-location theory underpinning root-finding correctness.
5. **Faddeev, D. K., & Faddeeva, V. N. (1963).** *Computational Methods of
   Linear Algebra*. W. H. Freeman and Company. — trace-based derivation of
   Faddeev–LeVerrier.

> *Optional addition:* Aurentz, Mach, Vandebril & Watkins (2014) is filed under
`matrix-design.md` in the source table, but its subject — polynomial root
> computation — is this doc's namesake topic. Worth citing here too if the design
> doc discusses rootfinding directly rather than just the underlying matrix
> mechanics.

---

## `state-space-design.md`

*Section 10 — References*

**Practical & Performance-Analytical**

1. **Moler, C., & Van Loan, C. (2003).** Nineteen Dubious Ways to Compute the
   Exponential of a Matrix, Twenty-Five Years Later. *SIAM Review*, 45(1),
   3–49. — comparative complexity/accuracy survey across matrix-exponential
   methods; justifies scaling-and-squaring + Padé over the augmented state
   matrix.
2. **Golub, G. H., & Van Loan, C. F. (2013).** *Matrix Computations* (4th ed.).
   Johns Hopkins University Press. — complexity basis for using triangular
   solves / Hessenberg reduction instead of direct inversion when
   computing $H(s)=C(sI-A)^{-1}B+D$.

**Theoretical & Algorithmic Foundations**

3. **Kailath, T. (1980).** *Linear Systems*. Prentice-Hall. — definitional
   source for controllability/observability matrices and canonical-form
   realizations.
4. **Ogata, K. (2010).** *Modern Control Engineering* (5th ed.). Prentice
   Hall. — LTI state-space formulation and block-diagram interconnection
   algebra.
5. **Åström, K. J., & Murray, R. M. (2021).** *Feedback Systems: An Introduction
   for Scientists and Engineers* (2nd ed.). Princeton University Press. —
   similarity-transformation and closed-loop feedback derivations.

---

## `tensor-design.md`

*Section 10 — References*

**Practical & Performance-Analytical**

1. **Kolda, T. G., & Bader, B. W. (2009).** Tensor Decompositions and
   Applications. *SIAM Review*, 51(3), 455–500. — survey of
   decomposition-algorithm computational complexity, useful for scoping the
   contraction routines.
2. **Raychev, R., et al. (2021).** TinyML for Ubiquitous Edge AI. *arXiv
   preprint arXiv:2102.01255*. — memory-footprint and inference-latency figures
   for microcontroller-class deployments.
3. **Warden, P., & Situnayake, D. (2019).** *TinyML: Machine Learning with
   TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers*. O'Reilly
   Media. — worked weight/activation memory-budgeting examples on real hardware.
4. **Hennessy, J. L., & Patterson, D. A. (2017).** *Computer Architecture: A
   Quantitative Approach* (6th ed.). Morgan Kaufmann. — quantitative
   cache/memory-hierarchy modeling relevant to stride-based N-D indexing
   performance.

*(No theoretical-only tier needed — all four references carry direct
practical/empirical content.)*

---

## `transfer-function-design.md`

*Section 10 — References*

**Practical & Performance-Analytical**

1. **Franklin, G. F., Powell, J. D., & Workman, M. L. (1998).** *Digital Control
   of Dynamic Systems* (3rd ed.). Addison-Wesley. — implementation-oriented
   treatment of discretization and digital filter realization.
2. **Moler, C., & Van Loan, C. (2003).** Nineteen Dubious Ways to Compute the
   Exponential of a Matrix, Twenty-Five Years Later. *SIAM Review*, 45(1),
   3–49. — added here: §4 of the source report (whose primary doc *is*
   `transfer-function-design.md`) lists this as key literature for
   ZOH-via-matrix-exponential, even though the summary table only assigned it to
   `state-space-design.md`.

**Theoretical & Algorithmic Foundations**

3. **Oppenheim, A. V., & Schafer, R. W. (2009).** *Discrete-Time Signal
   Processing* (3rd ed.). Pearson. — FIR/IIR representation and
   frequency-response theory.
4. **Ogata, K. (2010).** *Modern Control Engineering* (5th ed.). Prentice
   Hall. — bilinear/Tustin transform derivation and frequency pre-warping.
5. **Henrici, P. (1974).** *Applied and Computational Complex Analysis, Volume
   1*. Wiley. — complex-arithmetic foundations for $H(j\omega)$ evaluation.

---

## Cross-Cutting References (append to all five)

The report marks Domain 8 (Type-Level Metaprogramming & Memory Safety) and
Domain 9 (Verification & Property-Based Testing) as **"Primary: All 5 Design
Docs,"** since every doc needs to justify its own `#![no_std]`/stack-budget
claims and its own property-based test suite. Rather than repeat this block five
times above, add it once to each doc's Section 10.

**Practical & Performance-Analytical**

1. **Claessen, K., & Hughes, J. (2000).** QuickCheck: A Lightweight Tool for
   Random Testing of Haskell Programs. *ACM SIGPLAN Notices*, 35(9), 268–279. —
   random-generation/shrinking methodology behind the `proptest` suites (also
   listed under `polynomial-design.md` above; de-dupe if merging).

**Theoretical & Algorithmic Foundations**

2. **Rust Project Developers. (2024).** *The Rustonomicon: The Dark Arts of
   Advanced and Unsafe Rust Programming*. — memory-aliasing and layout
   guarantees underpinning the `Storage<T, R, C>` trait split.

**Standards & Process Compliance**

3. **ISO. (2018).** *ISO 26262-6:2018 Road vehicles — Functional safety — Part
   6: Product development at the software level*.
4. **RTCA / EUROCAE. (2011).** *DO-178C: Software Considerations in Airborne
   Systems and Equipment Certification*.
5. **IEEE Computer Society. (2008).** *IEEE Standard for Software and System
   Test Documentation* (IEEE Std 829-2008).

---

## Note on the linked design docs

The report links to local paths (`file:///Users/mitchelldscott/control-rs/...`)
that aren't reachable from here. If you upload the actual `*.md` files, I can
merge these references directly into each one's Section 10 instead of leaving it
as copy-paste.