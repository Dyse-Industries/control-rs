# Modern Tools (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_8,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

`src/modern_tools/mod.rs` is currently an empty scaffold. Its module doc
commits the module to "analyzing and designing linear systems using optimal
methods such as Linear Quadratic Regulator (LQR) and Kalman Filters."
`README.md`'s capability table already advertises `StateSpace` (built on the
`Matrix` primitive, capped at 32x32) as supporting "Kalman filtering, LQR" —
a claim with no implementation behind it yet. This document turns that
commitment into an architecture.

`modern_tools` sits alongside two sibling scaffolds with overlapping
concerns: `src/classical_tools/mod.rs` (root locus, Routh-Hurwitz, Bode/
Nyquist/Nichols, lead-lag/PID — frequency-domain design) and
`src/robust_tools/mod.rs` ("tools for designing and analyzing robust control
system[s]" — currently undefined). `modern_tools` is the state-space,
time-domain counterpart: pole/eigenvalue placement, LQR/LQG synthesis,
algebraic Riccati equation (ARE) solving and Kalman/observer-based state
estimation.

---

### 2. Requirements

#### 2.1. Functional Requirements

- **FR-1 — Pole/Eigenvalue Placement**: Compute a state-feedback gain $K$ (or
  observer gain $L$) placing eigenvalues of $A - BK$ (or $A - LC$) at a
  caller-specified set, via Ackermann's formula (SISO) and a distinct MIMO
  algorithm.
- **FR-2 — LQR/LQG Synthesis**: Given $(A, B, Q, R)$, compute the state-feedback
  gain $K$ minimizing the quadratic cost $J$; LQG composes this gain with a
  Kalman-filter state estimate.
- **FR-3 — Riccati Equation Solving**: Solve the continuous- and discrete-time
  ARE as a shared substrate under both LQR and steady-state Kalman gain
  computation.
- **FR-4 — Kalman/Observer Estimation Paths**: Provide steady-state Kalman
  filtering, EKF, UKF and Luenberger observer design as four distinct
  types/entry points, not one generic "filter" abstraction.

#### 2.2. Non-Functional Requirements

- **NFR-1 — `no_std`, No Heap, No Dynamic Allocation**: Consistent with the
  crate's existing constraints (`Matrix` capped at 32x32); see §4.2 and §8 for
  evidence gaps.
- **NFR-2 — Build on `Matrix`/`StateSpace`**: All solvers operate on existing
  `Matrix`/`StateSpace` types; no independent linear-algebra path is introduced.
- **NFR-3 — No LAPACK/BLAS Dependency**: `modern_tools` reuses or extends
  `Matrix`'s in-house decomposition routines rather than an external compiled
  backend.

#### 2.3. Constraints

- **C-1 — Matrix Capacity Ceiling**: Numeric methods are bounded by `Matrix`'s
  32×32/no-heap ceiling; any ARE solver design must fit inside that envelope.
- **C-2 — H-infinity Out of Scope**: H-infinity/mu-synthesis is not committed to
  `modern_tools`; it belongs to the separate `robust_tools` scaffold (§4.3, §8).

---

### 3. Technical Overview

`modern_tools` covers time-domain, state-space design and estimation:
pole/eigenvalue placement, LQR/LQG synthesis, ARE solving and the Kalman/
EKF/UKF/Luenberger estimator family. It explicitly excludes frequency-domain
classical design (`classical_tools`) and nonlinear MPC/codegen tooling
(surveyed separately in `controls-tools.json`, covering CT, acados and
FORCES Pro — none of which are state-space LQR/Kalman toolboxes in the
sense surveyed here).

The central technical problem is the ARE: $A^\top X + XA - XBR^{-1}B^\top X

+ Q = 0$ (python-control, 2026b). Every surveyed workstation toolbox solves
  it with an eigenvalue-decomposition method (Hamiltonian-matrix or
  generalized-eigenvalue/Schur-vector techniques, §4.2) backed by a compiled
  LAPACK-class library. No such backend is available to `modern_tools` and
  the research pass surfaced no toolbox describing a no_std/embedded-friendly
  equivalent (§8) — the closest evidence is `multicalc`'s no_std Kalman-filter
  demonstration (which sidesteps ARE-solving via online covariance recursion,
  not a one-shot solve) and its LQR support, which is roadmapped but not
  shipped (kmolan, 2026a). This gap, not the placement or observer-design
  requirements, is the module's primary technical risk.

---

### 4. Architecture

#### 4.1. Module Layout

```
modern_tools/
├── placement/   pole/eigenvalue placement (Ackermann SISO, KNV85-style MIMO)
├── riccati/     CARE/DARE solver(s), shared by lqr and kalman
├── lqr/         LQR gain synthesis (built on riccati)
├── kalman/      steady-state (linear) Kalman gain (built on riccati)
├── observer/    Luenberger observer design (built on placement, not riccati)
└── estimators/  EKF and UKF — distinct nonlinear estimation code paths
```

`riccati` is a shared dependency of both `lqr` and `kalman`, matching the
evidence that LQR gain and steady-state Kalman gain are the same
mathematical problem applied to $(A, B, Q, R)$ vs. its dual $(A^\top, C^\top,
Q_{\text{proc}}, R_{\text{meas}})$ (JuliaControl, 2026a). `observer` depends
on `placement`, not `riccati` — Luenberger design has no noise model and is
architecturally a placement problem (§2.1), not an estimation problem.
`estimators` (EKF/UKF) is kept separate from `kalman` because both
surveyed variants relinearize or resample every step rather than computing
one fixed gain (Labbe, 2016a; Labbe, 2016b) — a fundamentally different
runtime shape from a precomputed-gain filter.

#### 4.2. Riccati Solver: Numerical Approach

Surveyed reference implementations converge on eigenvalue-decomposition
methods:

- MATLAB's `care` forms the Hamiltonian matrix when $R$ is well-conditioned
  and $E = I$, falling back to the extended Hamiltonian pencil and the QZ
  algorithm otherwise (MathWorks, 2026a), implementing the generalized
  eigenproblem algorithms of Arnold and Laub (1984). MATLAB's newer `icare`
  (R2019a+) is a scaling/accuracy refinement of the same family, not a
  different numerical method (MathWorks, 2026a).
- Julia's `MatrixEquations.jl` — the dependency `ControlSystemsBase.lqr`
  defers to for its Riccati solve (JuliaControl, 2026a) — documents itself
  as using "orthogonal Schur vectors based methods and their extensions to
  linear matrix pencil based reduction approaches" (Varga, 2026; tagged
  `uncorroborated` in the source evidence — a secondary summary, not a
  primary-source quote).
- python-control's `lqr` exposes a `method` parameter that tries the
  `slycot` (Fortran/SLICOT) backend first, falling back to SciPy
  (python-control, 2026a) — again an external compiled-library dependency,
  not an in-house numerical method; `care`, which `lqr` calls internally
  (§2.1), inherits the same backend choice.

All three require a general eigenvalue or Schur decomposition. `Matrix`
does not currently expose one: its decomposition set is LU (partial
pivoting), $LDL^\top$, Cholesky and QR, plus a companion-matrix
eigenvalue solver scoped specifically to polynomial root-finding
(Faddeev-LeVerrier characteristic-polynomial extraction feeding a
unitary-plus-rank-one companion-matrix QR iteration). Adopting the
Hamiltonian/Schur approach as `modern_tools`'s primary ARE solver is
therefore not a self-contained decision — it first requires a general
Schur/QZ decomposition on `Matrix`, which is design work belonging to
`Matrix`, not `modern_tools` (§5.1, §8).

The alternative with actual `no_std` precedent, though only demonstrated on
the Kalman-filter side, is recursive/iterative propagation: `minikalman`
(no_std by default, static buffers, `f32`/Q16.16 fixed-point; sunsided,

2026) and `adskalman` (no_std/embedded-capable, built on `nalgebra`;
      strawlab, 2026) both ship as no_std Kalman filters without a general ARE
      solver — they iterate the discrete-time covariance predict/update step,
      which is the Riccati *difference* equation, not a one-shot algebraic
      solve.
      Iterated to convergence under standard stabilizability/detectability
      conditions, this recursion's fixed point is the ARE solution. `multicalc`
      corroborates the no_std viability of this pattern operationally: a 5-state
      EKF fused into a 1 kHz control loop over 600,000 ticks with zero
      collisions
      (kmolan, 2026a) — though this is EKF covariance recursion, not LQR gain
      computation, which the same source lists only as a roadmap item, not a
      shipped capability (kmolan, 2026a).

**Assumption** (not directly evidenced): applying the same recursive
propagation to compute an LQR gain — iterating the *control* Riccati
difference equation to a fixed point instead of Schur-decomposing the
Hamiltonian matrix — is mathematically standard but was not found described
as a production LQR method by any surveyed toolbox. This is presented here
as `modern_tools`'s own extrapolation from the Kalman-filter-side evidence,
not as an industry-precedented technique and needs its own convergence/
worst-case-execution-time analysis before being treated as load-bearing
(§8).

#### 4.3. H-infinity / Robust Control: Positioning Evidence

Every surveyed toolbox packages H-infinity/mu-synthesis separately from
LQR/Kalman: python-control's `hinfsyn` raises `ImportError` if the `slycot`
routine `sb10ad` is not loaded (python-control, 2026c) — a distinct,
optional dependency from the `lqr`/`care` path (python-control,
2026a/2026b). MATLAB ships H-infinity
synthesis (`hinfsyn`/`mixsyn`) in the separately licensed Robust Control
Toolbox (MathWorks, 2026b). Julia's synthesis docs explicitly redirect
H-infinity/H2/"more advanced LQG design" to `RobustAndOptimalControl.jl`, an
extension package outside `ControlSystems.jl` core (JuliaControl, 2026a;
JuliaControl, 2026b). Octave groups `hinfsyn`/`mixsyn`/`ncfsyn` under a
"Robust Control" category distinct from its "Optimal Control" category
(Octave-Forge Community, 2026). This evidence is directionally consistent
with keeping H-infinity out of `modern_tools`'s core and inside
`robust_tools` — but per this document's scope (§2.3), that packaging
decision is not made here (§8).

#### 4.4. Illustrative API Sketch

The following is illustrative of the module's shape, not a final API
surface — signatures, generic bounds and error types are unresolved.

```rust
// Illustrative — not a final API surface.

/// Solves AᵀX + XA - XBR⁻¹BᵀX + Q = 0 for X (python-control, 2026b).
pub trait RiccatiSolver<T, D: Dim, M: Dim> {
    fn solve_care(
        a: &Matrix<T, D, D>,
        b: &Matrix<T, D, M>,
        q: &Matrix<T, D, D>,
        r: &Matrix<T, M, M>,
        tolerance: T,
        max_iterations: usize,
    ) -> Result<Matrix<T, D, D>, ModernToolsError>;
}

/// LQR gain K such that u = -Kx minimizes ∫(xᵀQx + uᵀRu) dt.
pub fn lqr<T, D: Dim, M: Dim, S: RiccatiSolver<T, D, M>>(
    a: &Matrix<T, D, D>,
    b: &Matrix<T, D, M>,
    q: &Matrix<T, D, D>,
    r: &Matrix<T, M, M>,
) -> Result<Matrix<T, M, D>, ModernToolsError> {
    // delegates to S::solve_care, then K = R⁻¹BᵀX
    todo!()
}

/// Steady-state (linear, time-invariant) Kalman filter — fixed gain from riccati.
pub struct SteadyStateKalmanFilter<T, D: Dim, Z: Dim> {
    gain: Matrix<T, D, Z>,
    state: Matrix<T, D, U1>,
}

/// Extended Kalman filter — relinearizes every step; a distinct code path
/// from SteadyStateKalmanFilter, not a generalization of it.
pub struct ExtendedKalmanFilter<T, D: Dim, Z: Dim> {
    state: Matrix<T, D, U1>,
    covariance: Matrix<T, D, D>,
}

impl<T, D: Dim, Z: Dim> ExtendedKalmanFilter<T, D, Z> {
    /// Mirrors FilterPy's predict_update(z, HJacobian, Hx, ...) shape (Labbe, 2016a).
    pub fn predict_update<F, H>(
        &mut self,
        z: &Matrix<T, Z, U1>,
        f_jacobian: F,
        h_jacobian: H,
        q: &Matrix<T, D, D>,
        r: &Matrix<T, Z, Z>,
    ) -> Result<(), ModernToolsError>
    where
        F: Fn(&Matrix<T, D, U1>) -> Matrix<T, D, D>,
        H: Fn(&Matrix<T, D, U1>) -> Matrix<T, Z, D>,
    {
        todo!()
    }
}

/// Luenberger observer — deterministic pole placement on A - LC, no noise model.
pub struct LuenbergerObserver<T, D: Dim, Z: Dim> {
    gain: Matrix<T, D, Z>,
}
```

Riccati-solve cost is bounded by the same $D \le 32$ cap `Matrix` already
enforces (README, 2026); an iterative solve (§4.2) additionally bounds
per-step cost to existing $O(N^3)$ matrix-multiply/invert operations rather
than introducing a new $O(N^3)$-per-call decomposition kernel, at the cost
of an iteration count that is not fixed at compile time.

---

### 5. Alternatives

#### 5.1. Riccati Solver: Direct (Schur/Hamiltonian) vs. Recursive

- **Direct Hamiltonian/Schur decomposition** (Arnold and Laub, 1984;
  MathWorks, 2026a; Varga, 2026): the industry-standard method, numerically
  robust and used by every surveyed workstation toolbox. Rejected as the
  near-term default because it requires a general Schur/QZ decomposition
  `Matrix` does not currently implement (§4.2) — adopting it means taking on
  new `Matrix`-level design scope before `modern_tools` can start and no
  surveyed source documents this method running without a compiled
  LAPACK-class backend.
- **Recursive/iterative Riccati propagation** (indirect precedent: sunsided,
  2026; strawlab, 2026; kmolan, 2026a): selected as the near-term default.
  Builds only on operations `Matrix` already has (multiply, add, invert);
  matches the only `no_std` precedent the research pass found, even though
  that precedent is on the Kalman-filter side rather than LQR (§4.2,
  labeled assumption). Trade-off: iteration count to convergence is
  system-dependent, which complicates giving a fixed worst-case execution
  time — a real-time requirement direct decomposition does not share.

#### 5.2. Dependency on `classical_tools`

`modern_tools` needs eigenvalue/characteristic-polynomial machinery for
closed-loop pole extraction and (potentially) placement. That machinery —
`Matrix`'s Faddeev-LeVerrier characteristic-polynomial conversion and its
companion-matrix eigenvalue solver — already lives at the `Matrix` primitive
layer, not inside `classical_tools` (`classical_tools`'s own scope is root
locus, Routh-Hurwitz, Bode/Nyquist/Nichols and lead-lag/PID compensators;
it does not own polynomial/companion-matrix code per its module doc).
Two options follow:

- **`modern_tools` depends on `classical_tools`**: rejected. It would invert
  the crate's layering (primitives feed toolboxes; toolboxes do not depend
  on sibling toolboxes) for no benefit, since the needed functionality
  is not actually owned by `classical_tools`.
- **`modern_tools` depends on `Matrix`'s own polynomial/eigenvalue
  conversions directly** (selected): the same primitive-layer dependency
  `classical_tools` itself would use for root-locus-style computations.
  Keeps `modern_tools` and `classical_tools` as independent siblings over
  the same primitive layer, consistent with the crate architecture diagram
  in `README.md`.

---

### 6. Verification & Validation

1. **Golden-Value Regression Tests**: Per this crate's testing standards for
   solvers and estimators, ARE/LQR/Kalman-gain outputs are checked against
   trusted external references — python-control's `care`/`lqr` (python-
   control, 2026a/2026b) and MATLAB's `care`/`icare` (MathWorks, 2026a) are
   the two reference oracles surfaced by this research pass.
2. **Property-Based Tests**: Covariance-update invariants (symmetric
   positive-semidefinite $P$ after each Kalman/EKF/UKF step, closed-loop
   stability of $A - BK$/$A - LC$ eigenvalues after placement) are checked
   via `proptest` over generated system matrices, per this crate's
   invariant-heavy-code testing standard.
3. **HIL**: Not applicable. `modern_tools` is a numerical solver/estimator
   module with no hardware interface; it is exercised through the same
   host-based unit/property-test path as `Matrix` and `StateSpace`.

---

### 7. Risks & Open Questions

- **No Confirmed `no_std` ARE-Solver Precedent**: The research pass's
  central open question is unresolved. Every workstation-class Riccati
  solver surveyed depends on a compiled LAPACK-class backend
  (`slycot`/SciPy, MATLAB, `MatrixEquations.jl`). The one no_std control
  crate surveyed (`multicalc`) demonstrates a no_std EKF in production but
  lists LQR as roadmap-only, not shipped (kmolan, 2026a). §4.2's recursive-
  propagation approach is this document's proposed answer, but it is an
  extrapolation from Kalman-filter-side evidence, not a directly evidenced
  LQR technique — it needs its own validation before being load-bearing.
- **`modern_tools` vs. `robust_tools` Packaging for H-infinity/mu-synthesis**:
  §4.3's evidence — every surveyed toolbox ships H-infinity as a separate
  package/product from LQR/Kalman — leans toward excluding it from
  `modern_tools` and toward `robust_tools` as its eventual home. This
  document does not resolve that packaging question; it is carried forward
  for explicit human decision, consistent with `robust_tools` already
  existing as its own (currently undefined) scaffold.
- **Schur/QZ Decomposition Is a `Matrix`-Level Dependency, Not Yet
  Designed**: If a future revision selects the direct Hamiltonian/Schur ARE
  solver (§5.1) over the recursive default, that decision is blocked on
  `Matrix` gaining a general Schur/QZ decomposition — out of scope for this
  document and not yet proposed anywhere in the crate's design documents.
- **MIMO Pole Placement Algorithm Choice**: Ackermann's formula is SISO-only
  per every surveyed source (Octave-Forge Community, 2026; JuliaControl,
  2026a). The MIMO case needs a distinct algorithm; Kautsky, Nichols and
  Van Dooren (1985) is the evidenced precedent (it underlies MATLAB's
  `place`, per secondary/uncorroborated evidence in the source research),
  but the exact algorithm to implement is not finalized here.
- **EKF/UKF vs. `nonlinear_tools` Boundary**: `src/nonlinear_tools/mod.rs`
  is also an undefined scaffold ("tools and methods for designing and
  analyzing nonlinear control systems"). This document places EKF/UKF in
  `modern_tools`, following the research query's explicit in-scope framing
  of "Kalman filtering (linear, extended, unscented)" as one topic — but
  this boundary was not stress-tested against `nonlinear_tools`'s own
  eventual scope and may need revisiting once that scaffold is designed.

---

### 8. Development Plan

| Task / Feature                               | Description                                                                                                                                          | Estimated Effort        |
|:---------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------|
| **Phase 1: Riccati Substrate**               | Implement recursive/iterative CARE (and DARE) solving on top of existing `Matrix` operations (§4.2); establish convergence/iteration-bound behavior. | 4.0 Days                |
| **Phase 2: Placement**                       | Ackermann's formula (SISO) and a KNV85-style robust MIMO placement routine, shared by control-law and observer-gain placement.                       | 3.0 Days                |
| **Phase 3: LQR + Steady-State Kalman**       | `lqr` and `kalman` modules built on Phase 1's Riccati substrate; LQG composition of the two.                                                         | 3.0 Days                |
| **Phase 4: Observer + Nonlinear Estimators** | Luenberger observer (built on Phase 2); `ExtendedKalmanFilter` and `UnscentedKalmanFilter` as distinct estimator types.                              | 4.0 Days                |
| **Phase 5: H-infinity Scoping (Deferred)**   | Blocked on the `modern_tools`/`robust_tools` packaging decision (§7); no implementation work begins until that decision is made.                     | Not estimated (blocked) |

---

### 9. References

1. Python Control Systems Library, "control.lqr," *Python Control Systems
   Library Documentation*, Version 0.10.2. [Online].
   Available: https://python-control.readthedocs.io/en/latest/generated/control.lqr.html.
   Accessed: Aug. 8, 2026.
2. Python Control Systems Library, "control.care," *Python Control Systems
   Library Documentation*, Version 0.10.2. [Online].
   Available: https://python-control.readthedocs.io/en/latest/generated/control.care.html.
   Accessed: Aug. 8, 2026.
3. MathWorks, "care - (Not recommended) Solve continuous-time algebraic Riccati
   equation," *MATLAB Control System Toolbox Documentation*. [Online].
   Available: https://www.mathworks.com/help/control/ref/care.html. Accessed:
   Aug. 8, 2026.
4. JuliaControl, "Synthesis," *ControlSystems.jl documentation*. [Online].
   Available: https://juliacontrol.github.io/ControlSystems.jl/stable/lib/synthesis/.
   Accessed: Aug. 8, 2026.
5. J. Kautsky, N. K. Nichols and P. Van Dooren, "Robust pole assignment in
   linear state feedback," *International Journal of Control*, vol. 41, pp.
   1129–1155, 1985.
6. J. Ackermann, "Der Entwurf linearer Regelungssysteme im Zustandsraum," *at -
   Automatisierungstechnik*, vol. 20, no. 1-12, pp. 297–300, 1972, doi:
   10.1524/auto.1972.20.112.297.
7. The Octave-Forge Community, "List of Functions for the 'control' package,"
   *Octave Forge*, Version 4.2.3. [Online].
   Available: https://octave.sourceforge.io/control/overview.html. Accessed:
   Aug. 8, 2026.
8. R. R. Labbe Jr., "ExtendedKalmanFilter," *FilterPy documentation*, Version
   1.4.4. [Online].
   Available: https://filterpy.readthedocs.io/en/latest/kalman/ExtendedKalmanFilter.html.
   Accessed: Aug. 8, 2026.
9. R. R. Labbe Jr., "UnscentedKalmanFilter," *FilterPy documentation*, Version
   1.4.4. [Online].
   Available: https://filterpy.readthedocs.io/en/latest/kalman/UnscentedKalmanFilter.html.
   Accessed: Aug. 8, 2026.
10. S. J. Julier, "The scaled unscented transformation," in *Proc. American
    Control Conference*, 2002, pp. 4555–4559.
11. E. A. Wan and R. Van der Merwe, "The unscented Kalman filter for nonlinear
    estimation," in *Proc. Symp. Adaptive Syst. Signal Process., Commun.
    Contr.*, Lake Louise, AB, Canada, Oct. 2000.
12. kmolan, "Multicalc: Scientific computing for real time embedded systems in
    no_std rust," *users.rust-lang.org*. [Online].
    Available: https://users.rust-lang.org/t/multicalc-scientific-computing-for-real-time-embedded-systems-in-no-std-rust/141510.
    Accessed: Aug. 8, 2026.
13. W. F. Arnold and A. J. Laub, "Generalized eigenproblem algorithms and
    software for algebraic Riccati equations," *Proceedings of the IEEE*, vol.
    72, no. 12, pp. 1746–1754, Dec. 1984, doi: 10.1109/PROC.1984.13083.
14. A. Varga, "MatrixEquations.jl," *GitHub*. [Online].
    Available: https://github.com/andreasvarga/MatrixEquations.jl. Accessed:
    Aug. 8, 2026.
15. Python Control Systems Library, "control.hinfsyn," *Python Control Systems
    Library Documentation*, Version 0.10.2. [Online].
    Available: https://python-control.readthedocs.io/en/latest/generated/control.hinfsyn.html.
    Accessed: Aug. 8, 2026.
16. MathWorks, "Robust Control Toolbox," *MathWorks*. [Online].
    Available: https://www.mathworks.com/products/robust.html. Accessed: Aug. 8,
    2026.
17. JuliaControl, "RobustAndOptimalControl.jl," *GitHub*, Version
    0.4.51. [Online].
    Available: https://github.com/JuliaControl/RobustAndOptimalControl.jl.
    Accessed: Aug. 8, 2026.
18. sunsided, *minikalman-rs*: no_std Kalman filter with static
    buffers. [Online]. Available: https://github.com/sunsided/minikalman-rs.
    Accessed: Aug. 8, 2026.
19. strawlab, *adskalman-rs*: no_std/embedded-capable Kalman filter built on
    nalgebra. [Online]. Available: https://github.com/strawlab/adskalman-rs.
    Accessed: Aug. 8, 2026.

---

### 10. Revision History

| Revision | Date           | Author          | Description                                                                                                                          |
|:---------|:---------------|:----------------|:-------------------------------------------------------------------------------------------------------------------------------------|
| 1.0      | August 8, 2026 | @MitchellDScott | Initial draft: requirements, Riccati-solver architecture, Kalman/EKF/UKF/Luenberger split and packaging question flagged for review. |
