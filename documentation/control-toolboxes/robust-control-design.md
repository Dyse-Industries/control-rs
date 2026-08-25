# Robust Tools (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_24,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

`src/robust_tools/mod.rs` is currently an empty scaffold whose module doc
commits only to "tools for designing and analyzing robust control system."
Every mature toolbox surveyed for this design organizes that space around two
ideas: a linear-fractional (M-Delta) interconnection of a known plant with a
bounded uncertainty block and the structured singular value, µ, which
generalizes the singular value to quantify robustness against that
interconnection (MathWorks, 2026a). µ-based and margin-based robustness
analysis has documented use across automotive active-suspension design
(MathWorks, 2026b) and, per secondary literature, power-system stability
assessment (Djukanovic et al., 1998).

µ cannot be computed exactly; every surveyed toolbox instead computes upper
and lower bounds (MathWorks, 2026a) and the tightest available bound
terminates in a general-purpose linear matrix inequality (LMI) or
semidefinite-programming (SDP) solve (MathWorks, 2026a; Adams et al., 2025).
That same solver dependency, together with a plant-uncertainty representation
that surveyed toolboxes leave dynamically sized (JuliaControl, 2026a), is in
direct tension with control-rs's `no_std`, no-heap, const-generic-dimensioned
`Matrix`/`StateSpace` model. This document scopes `robust_tools`'s initial
architecture around that tension rather than around a full mu-synthesis
feature list. It treats the `modern_tools`-vs-`robust_tools` packaging
question already opened in
`documentation/control-toolboxes/research/results/modern-control.json` as
still unresolved (see §8) and out of scope for this pass, which addresses
uncertainty representation and robustness *analysis* only.

---

### 2. Requirements

#### 2.1. Functional Requirements

- **FR-1 — Single-Block Uncertain Plant**: Represent a nominal LTI plant with
  one norm-bounded, unstructured, full complex uncertainty block (‖Δ‖ ≤ 1) using
  existing `StateSpace`/`Matrix`/`TransferFunction` types.
- **FR-2 — Closed-Form Robust-Stability Check**: Provide the robust-stability
  predicate for that interconnection, equivalent to `opnorm(P.M) < 1`.
- **FR-3 — Parametric Uncertain Value**: Provide a fixed-size, `no_alloc`
  parametric uncertain real (nominal value plus asymmetric deviation), without a
  heap-backed collection.
- **FR-4 — Analysis, Not Synthesis**: Scope this phase to robustness analysis
  only; controller synthesis (µ-synthesis, DK-iteration) is excluded.

#### 2.2. Non-Functional Requirements

- **NFR-1 — Deterministic, Bounded Execution**: Any computation admitted to run
  on-target (FR-2) must execute in bounded, deterministic time with no heap
  allocation.
- **NFR-2 — Golden-Value Testability**: Every analysis routine's output must be
  checkable against an external reference per the crate's solver/estimator
  testing standard.

#### 2.3. Constraints

- **C-1 — No General Structured µ in This Phase**: No `no_std`/no-alloc SDP or
  LMI solver precedent exists; general structured µ-analysis is out of scope
  until a solver dependency is resolved separately.
- **C-2 — No Unified Dynamically Sized Uncertain-System Type**: v1 is restricted
  to the fixed, single-block interconnection in FR-1 rather than an arbitrary
  block-diagonal Δ.
- **C-3 — `no_std` / No Heap Allocator**: Inherited from the crate-wide
  constraint; no `robust_tools` type may require dynamic allocation.

---

### 3. Technical Overview

`robust_tools`'s initial scope is a robustness-*analysis* library for a
single fixed-size nominal plant plus one norm-bounded uncertainty block: the
closed-form stability check (FR2) and the parametric uncertain-value type
(FR3). It does not implement general structured µ-analysis, µ-synthesis or
DK-iteration, since those require a general LMI/SDP solver dependency and a
dynamically structured uncertainty representation for which no `no_std`
embedded precedent exists in the evidence base (Adams et al., 2025; Oxford
Control, 2026). The work draws on linear-fractional-transformation/M-Delta
theory (Doyle et al., 1991), singular-value/operator-norm computation as an
extension of `Matrix`'s existing decomposition machinery and the
numerical-conditioning caveat MATLAB documents for skewed parametric
uncertainty ranges (MathWorks, 2026c).

---

### 4. Architecture

#### 4.1. Uncertain-Plant Representation

The nominal interconnection matrix M is represented with the existing
`StateSpace<T, ...>`/`Matrix<T, R, C, S>` types; no new heap-based
uncertain-system container is introduced. Frequency-response evaluation of
$M(j\omega)$ is complex-valued, which the retargeted numerical models now
admit directly: `Complex<T>` satisfies `T: Scalar`
(`num-traits-design.md` §4.3), and the Hermitian routines `Hemv`, `Herk`
and `Heev` (`subprograms-design.md` FR-2, FR-4, FR-8) supply the
$M(j\omega)^H M(j\omega)$ singular-value machinery $\mu$-analysis and
$\mathcal{H}_\infty$ norms require. The uncertainty itself is represented purely
as a scalar norm bound (‖Δ‖ ≤ 1) on a single, unstructured, full complex
block — the one case JuliaControl documents as currently tractable without a
general LMI solve (JuliaControl, 2026a). Frequency-dependent uncertainty
weighting, the conventional `W(s)Δ` form (JuliaControl, 2026a), is not given
a dedicated uncertainty-composition operator in v1; it folds into the
existing `TransferFunction`/`StateSpace` series/feedback algebra already
provided by the crate.

#### 4.2. Robust-Stability Check

The FR2 predicate — `sup_w σ_max(M(jw)) < 1`, equivalently `opnorm(P.M) < 1`
(JuliaControl, 2026a) — reduces to a largest-singular-value (spectral norm)
computation over a fixed-size M matrix at each evaluated frequency point.
`Matrix` does not yet expose a largest-singular-value/power-iteration
primitive; adding one is a dependency this design introduces (tracked in
§8). Because the check is a bounded, deterministic linear-algebra operation
on a fixed-size matrix, it is the one robustness metric in the evidence base
plausibly capable of running on-target, not only at design time — distinct
from general µ-bound computation, which every surveyed toolbox documents as
a design-time, workstation-class step (MathWorks, 2026a; Adams et al., 2025).

#### 4.3. Parametric Uncertain Values

`UncertainReal<T>` (FR3) stores a nominal value plus asymmetric deviation
bounds, matching `ureal`'s `[-DL, +DR]` model (MathWorks, 2026c), as a plain
value type with no propagation or LFT-lifting machinery. Lifting a
collection of such values into one interconnected M-Delta system is deferred
(§8) — combining several parametric uncertainties changes FR1's "single
block" assumption and is exactly the structured-uncertainty case excluded by
§2.3.

#### 4.4. Design-Time vs. On-Target Boundary

| Capability                                    | Boundary                                                                                  |
|:----------------------------------------------|:------------------------------------------------------------------------------------------|
| Single-block `opnorm(P.M) < 1` check (FR2)    | On-target-capable: bounded, deterministic, no solver dependency.                          |
| `UncertainReal` construction/inspection (FR3) | On-target-capable: plain value type.                                                      |
| General structured µ upper/lower bound        | Design-time only; depends on an LMI/SDP solve or iterative method not scoped here (§2.3). |
| µ-synthesis / DK-iteration                    | Out of scope this phase (FR4); design-time only wherever it lands.                        |

---

### 5. Alternatives

#### 5.1. µ-Bound Strategy

- **Upper Bound via Balanced/AMI + General LMI Optimization**: The tightest
  guarantee surveyed toolboxes offer, using Osborne/Perron balancing followed
  by general-purpose LMI optimization (MathWorks, 2026a; Boyd and El Ghaoui,
  1993). Rejected for v1: pulls in a general convex-optimization solver with
  no embedded precedent (§2.3).
- **Lower Bound via Power Iteration**: An iterative eigen/power method
  (Packard et al., 1988) that avoids an LMI solve, but only ever certifies
  *non*-robustness, never proves robust stability (MathWorks, 2026a).
  Insufficient alone as the crate's only robustness primitive; not selected.
- **Closed-Form Special Case (Single Full Complex Block)**: `opnorm(P.M) < 1`
  (JuliaControl, 2026a). Selected for v1: no solver dependency and decidable
  over the existing `Matrix` type, at the cost of coverage — it applies only
  to unstructured, single-block uncertainty, not general structured Δ.

#### 5.2. LMI/SDP Solver Landscape

- **MOSEK**: Commercial solver used in dkpy's own DK-iteration example
  (Adams et al., 2025). Rejected on license and dependency-minimization
  grounds (CLAUDE.md).
- **Clarabel.rs**: The only Rust-native SDP/LMI-capable solver found in this
  research — an Apache-2.0 interior-point solver for LP/QP/SOCP/SDP problems,
  also available as a Julia package (Oxford Control, 2026; Goulart and Chen,
  2024). No `no_std`/embedded support is documented for it anywhere in the
  evidence collected. Adopting it would mean a host-only, feature-gated
  dependency; that decision is deferred to a future design review rather than
  made here.
- **Broader Open-Source SDP Landscape (SCS, SDPT3, SeDuMi)**: The research
  pass hit WebSearch rate-limiting before surveying these and recorded the
  gap explicitly in its round-2 revision notes
  (`documentation/control-toolboxes/research/queries/robust-control.json`).
  No claim is made about their suitability here.

#### 5.3. Packaging: `robust_tools` vs. `modern_tools`

Evidence shows mixed precedent across ecosystems. python-control ships
`hinfsyn` in the same package as `lqr` (python-control, 2026), while both
MATLAB and Julia keep robust-control functionality in a separate
product/package: MATLAB's Robust Control Toolbox is a distinct add-on
product from the base Control System Toolbox (MathWorks, 2026d) and
`RobustAndOptimalControl.jl` describes itself as "an extension to
ControlSystems.jl" (JuliaControl, 2026b) rather than part of it. Since
modern-control.json already opened this question without resolving it and
`modern-control-design.md` (Draft) likewise carries it forward as an open
risk rather than adjudicating it (its own §7), this document does not
re-decide it either. `robust_tools` remains scoped to uncertainty representation
and robustness *analysis* only (FR4) — the narrower slice both packaging
precedents agree belongs under a "robust" umbrella regardless of how the
synthesis question is eventually resolved.

---

### 6. Verification & Validation

1. **Golden-Value Regression Tests**: Compare the `opnorm(M) < 1` predicate
   and the underlying largest-singular-value computation against MATLAB
   `mussv`/Julia `structured_singular_value` reference outputs for known M
   matrices (testing-standards.md; MathWorks, 2026a; JuliaControl, 2026a).
2. **Property-Based Tests (`proptest`)**: Verify invariants of the spectral-
   norm computation (e.g., scale invariance, agreement with a hand-derived
   closed form for 1×1/2×2 M matrices) per the crate's invariant-heavy-code
   testing standard.
3. **HIL**: Not required in this phase. The only on-target-capable primitive
   (§4.4) is exercised through `Matrix`'s existing verification path; add a
   HIL mock harness only once/if an on-target consumer of the check is
   identified.

---

### 7. Performance & Resource Considerations

The only in-scope robustness metric (FR2) costs a largest-singular-value
computation over an M matrix already bounded by control-rs's existing 32×32
`Matrix` capacity limit (README); it introduces no new memory ceiling. All
LMI/SDP-solver-dependent computation is excluded from v1 (§2.3), so no
host-toolchain resource question (solver runtime, memory) is currently in
scope — that consideration is deferred to whichever future design formally
adopts a solver dependency.

---

### 8. Risks & Open Questions

- **LMI/SDP Solver Dependency Blocks General µ-Analysis**: No `no_std`/
  no-alloc precedent was found for computing the tightest µ upper bound;
  every surveyed toolbox's tightest bound terminates in a general LMI
  optimization (MathWorks, 2026a; Boyd and El Ghaoui, 1993) or MOSEK (Adams
  et al., 2025) and Clarabel.rs, the one Rust SDP solver found, does not
  document embedded/`no_std` status (Oxford Control, 2026). This blocks any
  extension beyond the single-full-complex-block special case until resolved.
- **nu-Gap / Disk Margin as a Nearer-Term Target**: JuliaControl documents
  the nu-gap metric's stabilization guarantee and its `ncfmargin` relative
  (JuliaControl, 2026a), but no computational algorithm or closed form for
  nu-gap was collected in the evidence base. Disk margin (`diskmargin`) was
  named as a canonical survey target in the research query but no
  algorithmic evidence for it was recorded in the results file. Both remain
  open pending a follow-up research pass before either can be scoped into
  `robust_tools`.
- **`modern_tools` vs. `robust_tools` Packaging Boundary**: Carried forward,
  unresolved, from `modern-control.json` (see §5.3). `modern-control-design.md`
  (Draft) independently leaves this same boundary open in its own §7. Revisit
  once either document is formally approved and adjudicates it.
- **Unsurveyed SDP/Robotics Territory**: The research query's own round-2
  revision notes record that WebSearch rate-limiting blocked further survey
  of the broader open-source SDP solver landscape (SCS, SDPT3, SeDuMi) and
  of robotics-specific structured-uncertainty applications
  (`documentation/control-toolboxes/research/queries/robust-control.json`).
  No claims are made about either here.
- **Skewed Parametric-Uncertainty Conditioning**: MathWorks documents that
  highly skewed `ureal` ranges cause poor numeric conditioning (MathWorks,
  2026c). Carry this caution into `UncertainReal`'s (§4.3) validation plan
  once implemented.
- **Assumption**: A largest-singular-value/power-iteration routine can be
  added to `Matrix` within its existing `no_std`/no-alloc/32×32 constraints.
  This document assumes such a routine is a tractable extension of the
  existing LU/QR decomposition infrastructure; that has not been confirmed
  by a dedicated technical spike.

---

### 9. Development Plan

| Task / Feature                                                         | Description                                                                                                                        | Estimated Effort                               |
|:-----------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------|
| **Step 1: Uncertain-Value & Uncertain-Plant Types**                    | Define `UncertainReal<T>` and the single-block M-Delta wrapper around existing `StateSpace`/`Matrix` types (FR1, FR3).             | 1.5 Days                                       |
| **Step 2: Spectral-Norm Primitive**                                    | Add a largest-singular-value/power-iteration routine to `Matrix` sufficient for the `opnorm(P.M)` check.                           | 2.5 Days                                       |
| **Step 3: Robust-Stability Check**                                     | Implement the `sup_w σ_max(M(jw)) < 1` predicate over `TransferFunction`/`StateSpace` frequency response (FR2).                    | 1.5 Days                                       |
| **Step 4: Verification**                                               | Golden-value regression tests against MATLAB/Julia references; `proptest` invariants (§6).                                         | 1.5 Days                                       |
| **Step 5 (Future Phase, Not This Approval): Solver Integration Spike** | Investigate an LMI/SDP solver dependency (e.g., Clarabel) as a host-only feature toward general structured µ; blocked on §8 risks. | Unestimated — separate design review required. |

---

### 10. References

[1] MathWorks, "mussv - Compute bounds on structured singular value (µ),"
*MATLAB Robust Control Toolbox Documentation*. [Online].
Available: https://www.mathworks.com/help/robust/ref/mussv.html. Accessed: Aug.
8, 2026.

[2] MathWorks, "Robust Control of Active Suspension," *MATLAB Robust Control
Toolbox Documentation*. [Online].
Available: https://www.mathworks.com/help/robust/gs/active-suspension-control-design.html.
Accessed: Aug. 8, 2026.

[3] M. Djukanovic et al., "Application of the structured singular value theory
for robust stability and control analysis in multimachine power systems. I.
Framework development," *IEEE Transactions on Power Systems*, 1998, doi:
10.1109/59.736270.

[4] T. E. Adams, S. Dahdah and J. R. Forbes, "dkpy: Robust Control with
Structured Uncertainty in Python," arXiv:2511.13927, 2025. [Online].
Available: https://arxiv.org/pdf/2511.13927.

[5] JuliaControl, "Uncertainty modeling," *RobustAndOptimalControl.jl
documentation*, version dev. [Online].
Available: https://juliacontrol.github.io/RobustAndOptimalControl.jl/dev/uncertainty/.
Accessed: Aug. 8, 2026.

[6] MathWorks, "ureal - Uncertain real parameter," *MATLAB Robust Control
Toolbox Documentation*. [Online].
Available: https://www.mathworks.com/help/robust/ref/ureal.html. Accessed: Aug.
8, 2026.

[7] S. Boyd and L. El Ghaoui, "Methods of centers for minimizing generalized
eigenvalues," *Linear Algebra and Its Applications*, vol. 188-189, pp. 63-111,

1993.

[8] Oxford Control, *Clarabel.rs*: Rust implementation of an interior-point
solver for conic programs (Version 0.11.1). [Online].
Available: https://github.com/oxfordcontrol/Clarabel.rs. Accessed: Aug. 8, 2026.

[9] P. J. Goulart and Y. Chen, "Clarabel: An interior-point solver for conic
programs with quadratic objectives," arXiv:2405.12762, 2024. [Online].
Available: https://arxiv.org/abs/2405.12762.

[10] kmolan, "Multicalc: Scientific computing for real time embedded systems in
no_std rust," *users.rust-lang.org*, 2026. [Online].
Available: https://users.rust-lang.org/t/multicalc-scientific-computing-for-real-time-embedded-systems-in-no-std-rust/141510.
Accessed: Aug. 8, 2026.

[11] J. Doyle, A. Packard and K. Zhou, "Review of LFTs, LMIs and mu," in
*Proc. 30th IEEE Conference on Decision and Control*, 1991, pp. 1227-1232.

[12] A. K. Packard, M. Fan and J. Doyle, "A power method for the structured
singular value," in *Proc. 1988 IEEE Conference on Decision and Control*, Dec.
1988, pp. 2132-2137.

[13] python-control, "control.hinfsyn," *Python Control Systems Library
documentation*, version 0.10.2. [Online].
Available: https://python-control.readthedocs.io/en/latest/generated/control.hinfsyn.html.
Accessed: Aug. 8, 2026.

[14] MathWorks, "Robust Control Toolbox," *MathWorks*. [Online].
Available: https://www.mathworks.com/products/robust.html. Accessed: Aug. 8,

2026.

[15] JuliaControl, *RobustAndOptimalControl.jl* (Version 0.4.51). [Online].
Available: https://github.com/JuliaControl/RobustAndOptimalControl.jl. Accessed:
Aug. 8, 2026.

---

### 11. Revision History

| Revision | Date           | Author          | Description                                                                                                                                    |
|:---------|:---------------|:----------------|:-----------------------------------------------------------------------------------------------------------------------------------------------|
| 1.0      | August 8, 2026 | @MitchellDScott | Initial draft: scoped `robust_tools` to single-block uncertainty and a closed-form stability check; deferred µ-analysis pending an SDP solver. |
| 1.1      | August 24, 2026 | @mitchelldscott | Aligned §4.1 with the retargeted models: `Matrix<T, R, C, S>` carries its storage parameter, and recorded that `Complex<T>` now satisfies `T: Scalar`, so the Hermitian routines `Hemv`/`Herk`/`Heev` (`subprograms-design.md` FR-2, FR-4, FR-8) supply the frequency-response singular-value machinery. Robust control content unchanged. Status stays Draft. |
