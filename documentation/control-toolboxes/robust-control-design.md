# Robust Control & Structured Uncertainty (Design Document)

![Status: Draft](https://img.shields.io/badge/status-draft-orange)
![Date Badge](https://img.shields.io/badge/Date-September_15,_2026-blue)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

`robust_tools` models bounded plant uncertainty and decides whether a linear
closed loop remains stable for every uncertainty in that bound. LFT algebra,
power iteration, and multi-block D-K synthesis belong in §4.

Primary usage scenarios:

- Represent a physical parameter as a nominal value plus a normalized
  deviation of known weight.
- Close a plant around an uncertainty block or a controller.
- Decide robust stability of a linear loop against a unit-norm uncertainty
  on a frequency grid, including on an embedded health monitor.

---

### 2. Requirements

#### 2.1 Functional Requirements

- **FR-1 — Bounded uncertain parameter**: A scalar physical parameter is
  represented as a nominal value plus a weight times a normalized deviation
  with $|\delta| \le 1$. An unbounded or unweighted parameter is out of
  scope.
- **FR-2 — Uncertainty or controller interconnection**: The module forms the
  closed interconnection of a plant with an uncertainty block or with a
  controller. Foreign matrix layouts are not required.
- **FR-3 — Single-block structured singular value**: For one full-complex
  uncertainty block, the module computes the structured singular value of the
  interconnection. Multi-block $\mu$ is out of scope (C-4).
- **FR-4 — Robust stability predicate**: The module reports whether the
  small-gain test holds across a caller-supplied frequency grid for
  $\|\Delta\|_\infty \le 1$. It does not certify stability off the grid.
- **FR-5 — Largest singular value**: The module computes $\bar{\sigma}$ of
  real and complex matrices so FR-3 and FR-4 have a spectral-norm primitive.

#### 2.2 Non-Functional Requirements

- **NFR-1 — Zero dynamic allocation**: $\mu$ evaluations, interconnections,
  and singular-value iterations execute in caller scratch. Heap growth is a
  defect.
- **NFR-2 — Bounded singular-value iteration**: Power iteration for
  $\bar{\sigma}$ terminates within $N_{\max} \le 50$ iterations at a
  documented residual.

#### 2.3 Constraints

- **C-1 — Inherited toolbox bounds**: Conforms to `controls-tools-design.md`
  C-1..C-4. Former C-2 and C-3 restated those bounds and are withdrawn.
- **C-4 — Offline-only multi-block synthesis**: Multi-block $\mu$-synthesis
  and general SDP/LMI solves stay host-side. They are not a target runtime
  dependency.

---

### 6. Verification & Validation

#### 6.1 Approach

`src/robust_tools` is a
stub; behavioral FRs are unverified until implementations and locators exist.

- Loops with $\|M\|_\infty < 1$ are reported robustly stable; loops with
  $\|M\|_\infty \ge 1$ are not.
- Closing an interconnection and substituting the uncertain parameter agree
  on a manufactured plant.
- Largest singular values agree with an independent dense-linear-algebra
  reference.
- Resonant plants on a frequency grid are flagged when the small-gain test
  fails.

| Method | Mechanism |
|:-------|:----------|
| Requirements-based test | Manufactured small-gain and interconnection cases |
| Back-to-back comparison | NumPy / SciPy / Julia spectral-norm reference |
| Static analysis | `cargo clippy-ci`; source inspection for heap use |
| Inspection | This document against `controls-tools-design.md` C-1..C-4 |
| Resource usage evaluation | Iteration cap on power iteration |
| On-target execution | ETS singular-value and interconnection step |
| Coverage measurement | `cargo coverage` |

Target: 90% statement coverage of `src/robust_tools` once the module contains
kernels, measured with `cargo coverage`. Excluded: `Debug` / `Display`.
The stub module has no executable surface to cover.

No validation plant exists for this module. Fitness for purpose is judged
only after an embedded health-monitor example or a host suite consumes the
robust-stability predicate.

#### 6.2 Acceptance

| Claim | Oracle | Measure | Bound |
|:------|:-------|:--------|:------|
| Small-gain classification | Manufactured $M$ with known $\|M\|_\infty$ | Predicate vs $\\|M\\|_\\infty < 1$ | Exact equality of the Boolean |
| Interconnection vs substitution | Direct parameter substitution in a state-space plant | Relative $\ell_{2}$ of closed-loop matrices | $\le 10^{-12}$ |
| Largest singular value | NumPy / SciPy `svd` on well- and ill-conditioned matrices | Relative error on $\bar{\sigma}$ | $\le 10^{-8}$ |
| Resonant-grid detection | Plant with a known critical peak on the grid | Predicate at that frequency | Exact detection of the crossing |

#### 6.3 Limits

- **FR-1..FR-5, NFR-2**: `src/robust_tools/mod.rs` contains no kernels. No
  `test:` or `oracle:` locator exists.
- **On-target singular-value latency**: no ETS suite is linked into firmware.
- **Off-grid robust stability**: FR-4 certifies only the caller-supplied
  frequency grid.
- **Multi-block $\mu$**: out of scope per C-4.
- **Withdrawn C-2, C-3**: restated inherited toolbox bounds; covered by C-1.

---

## References

[1] A. K. Packard, M. K. H. Fan, and J. C. Doyle, "A power method for the structured singular value," in *Proc. 27th IEEE Conference on Decision and Control*, Austin, TX, USA, 1988, pp. 2132--2137, doi: 10.1109/CDC.1988.194723.

[2] J. C. Doyle, A. K. Packard, and K. Zhou, "Review of LFTs, LMIs and mu," in *Proc. 30th IEEE Conference on Decision and Control*, Brighton, UK, 1991, pp. 1227--1232, doi: 10.1109/CDC.1991.261569.

[3] S. Boyd and L. {El Ghaoui}, "Method of centers for minimizing generalized eigenvalues," *Linear Algebra and its Applications*, vol. 188--189, pp. 63--111, 1993, doi: 10.1016/0024-3795(93)90466-4.

[4] P. J. Goulart and Y. Chen, "Clarabel: An interior-point solver for conic programs with quadratic objectives," *IEEE Transactions on Automatic Control*, vol. 69, no. 10, pp. 6900--6915, 2024, doi: 10.1109/TAC.2024.3392418.

[5] T. E. Adams, S. Dahdah, and J. R. Forbes, "dkpy: Robust Control with Structured Uncertainty in Python," arXiv, Rep. no. arXiv:2511.13927, 2025. [Online]. Available: https://arxiv.org/abs/2511.13927.

[6] JuliaControl, "RobustAndOptimalControl.jl: Robust and optimal control in Julia," *GitHub*, 2026. [Online]. Available: https://github.com/JuliaControl/RobustAndOptimalControl.jl.

---

### 10. Revision History

| Revision | Date              | Author          | Description                                                    |
|:---------|:------------------|:----------------|:---------------------------------------------------------------|
| 1.0      | September 3, 2026 | @MitchellDScott | Initial requirements and verification outline for robust.      |
| 1.1      | September 15, 2026 | @MitchellDScott | Need-named FRs; inherited toolbox pointer; full §6 with stub modules listed in 6.7. |
