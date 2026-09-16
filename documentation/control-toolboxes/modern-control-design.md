# Modern Control & State Estimation (Design Document)

![Status: Draft](https://img.shields.io/badge/status-draft-orange)
![Date Badge](https://img.shields.io/badge/Date-September_15,_2026-blue)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

`modern_tools` synthesizes state feedback and reconstructs plant state for
multivariable linear and nonlinear systems already represented as native
numerical models. Continuous versus discrete Riccati forms, Ackermann versus
KNV85 placement, and filter internals belong in §4.

Primary usage scenarios:

- Recover a stabilizing infinite-horizon cost matrix from a quadratic plant
  cost, then emit the corresponding state-feedback gain.
- Place closed-loop poles at a specified set of stable locations.
- Reconstruct full state from output measurements.
- Run a discrete estimator on a linear plant, and on a nonlinear plant with
  or without analytic Jacobians, on a bare-metal flight computer.

---

### 2. Requirements

#### 2.1 Functional Requirements

- **FR-1 — Stabilizing infinite-horizon cost matrix**: The module returns a
  stabilizing positive semi-definite cost matrix for a linear plant under an
  infinite-horizon quadratic cost, in both continuous time and discrete time.
  A same-horizon solve that is not stabilizing is not a solution.
- **FR-2 — Optimal state-feedback gain**: From that cost matrix and the
  caller's state and control weights, the module emits the corresponding
  linear state-feedback gain. It does not invent weights or require a foreign
  plant type.
- **FR-3 — Specified closed-loop poles**: The module computes a state-feedback
  gain whose closed-loop eigenvalues match a caller-specified set of stable
  target poles, for both single-input and multi-input plants.
- **FR-4 — State reconstruction from outputs**: The module constructs a
  full-order observer whose gain is dual to state-feedback placement, so the
  observer error decays at the specified observer poles.
- **FR-5 — Linear discrete estimator**: The module executes prediction and
  measurement updates for a linear discrete-time plant, keeping the error
  covariance symmetric and positive semi-definite.
- **FR-6 — Nonlinear estimator with linearization**: The module executes the
  same prediction and measurement cycle on a nonlinear plant given the
  caller's state-transition and measurement maps. Analytic Jacobians are
  accepted when supplied; they are not required of every caller.
- **FR-7 — Nonlinear estimator without Jacobians**: The module estimates
  nonlinear plant state without requiring analytic Jacobians. Filter internals
  (sigma-point counts, Cholesky factors) are a §4 choice.

#### 2.2 Non-Functional Requirements

- **NFR-1 — Bounded scratch memory**: Riccati iterations, decompositions, and
  estimator temporaries operate in caller-provided scratch. Heap growth during
  a step is a defect.
- **NFR-2 — Covariance stays a covariance**: Error covariance remains
  symmetric and positive semi-definite across estimator iterations.

#### 2.3 Constraints

- **C-1 — Inherited toolbox bounds**: Conforms to `controls-tools-design.md`
  C-1..C-4. Former C-2 and C-3 restated those bounds and are withdrawn.
- **C-4 — Bounded solver iterations**: Iterative Riccati solves enforce a
  compile-time maximum iteration count so termination is deterministic.

---

### 6. Verification & Validation

#### 6.1 Approach

`src/modern_tools` is a
stub; behavioral FRs are unverified until implementations and locators exist.

- Algebraic Riccati residual of a computed cost matrix stays inside the
  stated Frobenius bound on Laub-class plants.
- Closed-loop eigenvalues after optimal gain lie in the open left-half plane
  (continuous) or inside the unit circle (discrete).
- Placed poles match the requested set within the stated relative bound.
- Linear and nonlinear estimator trajectories agree with an independent
  reference to the stated bound.
- Estimator steps use only caller scratch.

| Method | Mechanism |
|:-------|:----------|
| Requirements-based test | `#[test]` residual, pole, and covariance cases on manufactured plants |
| Back-to-back comparison | Host oracle (FilterPy / MatrixEquations.jl) via `control-rs-ci` |
| Static analysis | `cargo clippy-ci`; source inspection for heap use |
| Inspection | This document against `controls-tools-design.md` C-1..C-4 |
| On-target execution | ETS estimator step on QEMU |
| Coverage measurement | `cargo coverage` |

Target: 90% statement coverage of `src/modern_tools` once the module contains
solvers, measured with `cargo coverage`. Excluded: `Debug` / `Display`
implementations. The stub module has no executable surface to cover.

No validation plant exists for this module. Fitness for purpose is judged
only after a host example or a `control-rs-validation` suite consumes a
synthesized gain or estimator the way `classical-tools-examples-design.md`
consumes classical compensators.

#### 6.2 Acceptance

| Claim | Oracle | Measure | Bound |
|:------|:-------|:--------|:------|
| Continuous / discrete ARE residual | Manufactured stabilizing $P$ on Laub benchmark plants | $\\|A^{T}P + PA - PBR^{-1}B^{T}P + Q\\|_{F} / \\|P\\|_{F}$ (discrete analogue) | $\le 10^{-8}$ |
| Closed-loop stability after LQR | Eigenvalues of $A - BK$ | Spectral radius / real-part sign | All $\mathrm{Re}(\lambda) < 0$ (c.t.) or $\|\lambda\| < 1$ (d.t.) |
| Pole placement | Requested pole set | Relative eigenvalue error | $\le 10^{-6}$ |
| Linear Kalman trajectory | FilterPy / MatrixEquations.jl on a linear Gaussian plant | Relative $\ell_{2}$ state and covariance | $\le 10^{-6}$ |
| Nonlinear estimator trajectory | Independent radar range/bearing tracking reference | Relative $\ell_{2}$ state | $\le 10^{-4}$ |

#### 6.3 Limits

- **FR-1..FR-7, NFR-2**: `src/modern_tools/mod.rs` contains no solvers.
  No `test:` or `oracle:` locator exists. This plan does not establish those
  claims until implementations land.
- **On-target step latency**: no ETS suite is linked into a firmware binary.
- **Analytic Jacobian provenance (FR-6)**: whether a caller-supplied Jacobian
  matches the nonlinear maps is not established here.
- **f32 estimator covariance**: bounds in 6.2 are f64 host claims.
- **Withdrawn C-2, C-3**: restated inherited toolbox bounds; covered by C-1.

---

## References

[1] W. F. Arnold and A. J. Laub, "Generalized eigenproblem algorithms and software for algebraic Riccati equations," *Proceedings of the IEEE*, vol. 72, no. 12, pp. 1746--1754, Dec. 1984, doi: 10.1109/PROC.1984.13083.

[2] J. Kautsky, N. K. Nichols, and P. Van Dooren, "Robust pole assignment in linear state feedback," *International Journal of Control*, vol. 41, no. 5, pp. 1129--1155, 1985, doi: 10.1080/00207178508923267.

[3] J. Ackermann, "Der Entwurf linearer Regelungssysteme im Zustandsraum," *at --- Automatisierungstechnik*, vol. 20, no. 1-12, pp. 297--300, 1972, doi: 10.1524/auto.1972.20.112.297.

[4] S. J. Julier, "The scaled unscented transformation," in *Proc. American Control Conference*, Anchorage, AK, USA, 2002, pp. 4555--4559, doi: 10.1109/ACC.2002.1025369.

[5] E. A. Wan and R. Van der Merwe, "The unscented Kalman filter for nonlinear estimation," in *Proc. IEEE 2000 Adaptive Systems for Signal Processing, Communications, and Control Symposium*, Lake Louise, AB, Canada, 2000, pp. 153--158, doi: 10.1109/ASSPCC.2000.882463.

[6] R. R. Labbe, "FilterPy: Kalman filters and optimal estimation in Python," *GitHub*, 2026. [Online]. Available: https://github.com/rlabbe/filterpy.

[7] A. Varga, "MatrixEquations.jl: Solution of linear matrix equations in Julia," *GitHub*, 2026. [Online]. Available: https://github.com/andreasvarga/MatrixEquations.jl.

---

### 10. Revision History

| Revision | Date              | Author          | Description                                                    |
|:---------|:------------------|:----------------|:---------------------------------------------------------------|
| 1.0      | September 3, 2026 | @MitchellDScott | Initial requirements and verification outline for modern.      |
| 1.1      | September 15, 2026 | @MitchellDScott | Need-named FRs; inherited toolbox pointer; full §6 with stub modules listed in 6.7. |
