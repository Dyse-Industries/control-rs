# Implicit Runge–Kutta & Collocation Integrators (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_26,_2026-blue)
![Status: Draft](https://img.shields.io/badge/status-draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

Stiff chemical networks, highly damped mechanical structures, electrical power circuits, and constrained optimal control problems involve disparate dynamical time scales and algebraic invariants (Betts, 2010; Hairer and Wanner, 1996; Hindmarsh et al., 2005). When integrated with explicit methods, stiff systems suffer numerical instability unless the time step $\Delta t$ is restricted below the smallest time constant, causing extreme computational overhead (Hairer and Wanner, 1996). Furthermore, differential-algebraic equations (DAEs) and optimal control direct collocation require implicit integration schemes possessing unconditional A-stability and L-stability (Betts, 2010; Verschueren et al., 2022).

This document establishes the architecture, requirement contract, and validation suite for the implicit Runge–Kutta and collocation integrators in `control-rs::integrators::implicit`. The module satisfies three primary usage scenarios:

1. **Stiff Multi-Scale System Simulation**: Stable, efficient stepping of stiff ODEs (e.g., Van der Pol oscillators, stiff actuation circuits) without step-size collapse using L-stable Radau IIA and SDIRK methods (Hairer and Wanner, 1996; Kennedy and Carpenter, 2003).
2. **Index-1 Differential-Algebraic Equations (DAEs)**: Direct simulation of constrained multibody dynamics and circuit networks $F(t, x, \dot{x}, z) = 0$ (Hairer and Wanner, 1996; Hindmarsh et al., 2005).
3. **Optimal Control Direct Collocation**: Transcription of continuous-time dynamics into algebraic equality constraints for nonlinear programming (NLP) solvers using Gauss–Legendre and Radau quadrature polynomials (Betts, 2010; Verschueren et al., 2022).

---

### 2. Requirements

#### 2.1 Functional Requirements

- **FR-1 — Implicit Stage Nonlinear Solver**: Provide a stack-allocated Newton–Raphson solver using in-place LU decomposition to solve the coupled algebraic stage equations $k_i = f(t_n + c_i h, x_n + h \sum_{j=1}^s a_{ij} k_j, u)$ without heap allocation.
- **FR-2 — Radau IIA Fifth-Order Collocation Stepper**: Provide `RadauIIA5` (3 stages, order 5, L-stable) for highly stiff differential equations and index-1 DAEs.
- **FR-3 — Gauss–Legendre Fourth-Order Collocation Stepper**: Provide `GaussLegendre4` (2 stages, order 4, A-stable and symplectic) for symmetric collocation and direct optimal control transcription.
- **FR-4 — Singly Diagonally Implicit Runge–Kutta (SDIRK) Stepper**: Provide `Sdirk` with identical diagonal stage entries $a_{ii} = \gamma$, enabling sequential stage solves with a single LU factorization per step.
- **FR-5 — Classical Low-Order Implicit Steppers**: Provide `ImplicitEuler` (1st order, L-stable) and `CrankNicolson` (Trapezoidal, 2nd order, A-stable).
- **FR-6 — Index-1 DAE Residual Stepping**: Support constrained algebraic variables $g(t, x, z) = 0$ coupled with differential states $\dot{x} = f(t, x, z, u)$ via simultaneous stage collocation.

#### 2.2 Non-Functional Requirements

- **NFR-1 — Zero Dynamic Memory Allocation**: Stage derivative vectors, Newton iteration residuals, and $N_x \times N_x$ Jacobian matrices must allocate exclusively on the stack (`no_alloc`).
- **NFR-2 — Compile-Time Static Sizing**: State dimension $N_x$, algebraic dimension $N_z$, and stage count $s$ must be verified at compile time via `Dim`.
- **NFR-3 — Bounded Newton Convergence**: The internal nonlinear solver must abort with `LinAlgError::NewtonConvergenceFailure` after a fixed maximum number of iterations ($N_{max} \le 20$).

#### 2.3 Constraints

- **C-1 — Native Rust Implementation**: Authored natively in Rust without C code generation ([`controls-tools-design.md`](../control-toolboxes/controls-tools-design.md) C-1).
- **C-2 — Storage Decoupling**: State and matrix storage must use `control-rs::matrix::Owned` and `DenseStorage` ([`matrix-design.md`](../numerical-models/matrix-design.md)).
- **C-3 — Zero Virtual Dispatch**: Steppers must monomorphize over `Sys: SystemDynamics<T, NX, NU>`.

---

### 3. Technical Overview

For a general implicit Runge–Kutta method, the stage values $k_i$ satisfy the coupled nonlinear algebraic system (Hairer and Wanner, 1996):
$$R_i(k_1, \dots, k_s) = k_i - f\left(t_n + c_i h, \, x_n + h \sum_{j=1}^s a_{ij} k_j, \, u\right) = 0, \quad i = 1, \dots, s$$
$$x_{n+1} = x_n + h \sum_{i=1}^s b_i k_i$$

```
Implicit Stage System:
  [ k₁ - f(t_n + c₁h, x_n + h·∑ a₁ⱼ kⱼ, u) ] = [ 0 ]
  [ k₂ - f(t_n + c₂h, x_n + h·∑ a₂ⱼ kⱼ, u) ] = [ 0 ]
  ...
  [ k_s - f(t_n + c_sh, x_n + h·∑ a_{sj} kⱼ, u) ] = [ 0 ]
  
Newton-Raphson Iteration:
  (I - h·γ·J)·Δk_i = -R_i
  k_i ← k_i + Δk_i
```

#### 3.1 SDIRK Efficiency
For SDIRK methods, $A$ is lower triangular with constant diagonal $a_{ii} = \gamma$. The iteration matrix $W = I - h \gamma J$ is identical for every stage $i = 1, \dots, s$. Thus, $W$ is factored via LU decomposition **only once** per time step, and stages $k_1, \dots, k_s$ are solved sequentially via triangular back-substitution (Hairer and Wanner, 1996; Kennedy and Carpenter, 2003).

#### 3.2 Stability Classifications
- **A-Stability**: The stability region contains the entire left half-plane $\mathbb{C}^-$, ensuring decay of non-stiff modes (Hairer and Wanner, 1996).
- **L-Stability**: The method is A-stable and the stability function satisfies $|R(\infty)| = 0$, rapidly damping infinitely stiff transients and algebraic perturbations in DAEs (Hairer and Wanner, 1996). `RadauIIA5` and `ImplicitEuler` are L-stable.

---

### 4. Architecture

#### 4.1 Generic Implicit Stepper Struct

**Proposal (not in evidence)**: Define `ImplicitRungeKutta` with stack-allocated Newton iteration:

```rust
/// Generic implicit Runge-Kutta integrator with stack Newton solver.
#[derive(Debug, Clone, Copy)]
pub struct ImplicitRungeKutta<T, const STAGES: usize> {
    pub a: [[T; STAGES]; STAGES],
    pub b: [T; STAGES],
    pub c: [T; STAGES],
    pub is_sdirk: bool,
}
```

#### 4.2 Concrete Implicit Steppers

**Proposal (not in evidence)**: Expose specialized constructors:

```rust
/// Radau IIA 3-stage, 5th-order L-stable collocation stepper.
pub struct RadauIIA5;

/// Gauss-Legendre 2-stage, 4th-order A-stable symplectic collocation stepper.
pub struct GaussLegendre4;

/// Singly Diagonally Implicit Runge-Kutta (SDIRK) stepper with single LU factorization.
pub struct Sdirk;

/// Backward / Implicit Euler 1st-order L-stable stepper.
pub struct ImplicitEuler;

/// Crank-Nicolson / Implicit Trapezoidal 2nd-order A-stable stepper.
pub struct CrankNicolson;
```

---

### 5. Alternatives Considered

- **Fully Coupled Monolithic Newton Solve vs Sequential SDIRK**: Monolithic Newton solves require factoring an $(s \cdot N_x) \times (s \cdot N_x)$ matrix ($O(s^3 N_x^3)$ cost). For real-time applications, SDIRK reduces factorization to a single $N_x \times N_x$ matrix ($O(N_x^3)$ cost), dramatically reducing stack and CPU cycles (Hairer and Wanner, 1996; Kennedy and Carpenter, 2003).
- **Dynamic Sparse Jacobian Allocation**: Rejected in favor of static stack arrays (`Owned<T, NX, NX>`) to guarantee deterministic execution bounds (C-1, NFR-1).

---

### 6. Verification & Validation Plan

#### 6.1 Unit Verification
- **Stiff Van der Pol Oscillator**: Integrate the Van der Pol oscillator $\ddot{x} - \mu (1 - x^2) \dot{x} + x = 0$ with stiffness parameter $\mu = 1000$. Assert that `RadauIIA5` and `Sdirk` stably integrate the limit cycle with large step sizes ($h = 0.1$), whereas explicit `RungeKutta4` suffers catastrophic numerical instability (Hairer and Wanner, 1996).
- **Order of Convergence Verification**: Step linear stiff decay system $\dot{x} = -1000 x$ over $h \in [0.1, 0.001]$:
  - `ImplicitEuler`: slope $1.00 \pm 0.05$.
  - `CrankNicolson`: slope $2.00 \pm 0.05$.
  - `GaussLegendre4`: slope $4.00 \pm 0.05$.
  - `RadauIIA5`: slope $5.00 \pm 0.05$.
- **Index-1 DAE Pendulum with Algebraic Length Invariant**: Simulate pendulum with algebraic constraint $x^2 + y^2 - L^2 = 0$. Verify `RadauIIA5` maintains constraint satisfaction $|x^2 + y^2 - L^2| < 10^{-10}$ across all steps.

#### 6.2 Validation
- **Direct Collocation NLP Transcription**: Connect `GaussLegendre4` and `RadauIIA5` collocation points to trajectory optimization constraints and verify constraint Jacobians (Betts, 2010; Verschueren et al., 2022).

---

### 7. Performance & Resource Considerations

- **Stack Allocation Footprint**: For an $N_x = 8$ system and $s = 3$ (`RadauIIA5`), scratch storage includes 3 stage vectors ($3 \times 8 \times 8 = 192$ bytes) and one Jacobian LU matrix ($8 \times 8 \times 8 = 512$ bytes), totaling $< 1$ KB of stack space.
- **Factorization Reuse**: Reusing the single Jacobian factorization across all stages in `Sdirk` reduces linear algebra overhead by up to $70\%$ relative to fully implicit tableaus (Kennedy and Carpenter, 2003; Reynolds et al., 2023).

---

### 8. Risks & Open Questions

- **Proposal (not in evidence) 1**: Stack-allocated Newton–Raphson solver using in-crate LU decomposition (`control_rs::matrix::decomposition::LuDecomposition`).
- **Proposal (not in evidence) 2**: SDIRK single-factorization reuse engine with const-generic diagonal parameters.

---

### 9. Development Plan

| Phase | Description | Estimated Effort (1-10) |
|:------|:------------|:------------------------|
| **Phase 1: Newton Solver & SDIRK Engine** | Implement stack Newton iteration with in-crate LU factorization and SDIRK single-matrix reuse. | 4 |
| **Phase 2: Collocation Steppers** | Implement `RadauIIA5`, `GaussLegendre4`, `ImplicitEuler`, and `CrankNicolson`. | 4 |
| **Phase 3: Stiff & DAE Verification** | Implement Van der Pol stiff benchmarks and DAE constraint preservation tests. | 3 |

---

### 10. Revision History

| Version | Date | Author | Description |
|:--------|:-----|:-------|:------------|
| 1.0 | August 26, 2026 | @MitchellDScott | Initial draft: implicit Runge-Kutta, SDIRK, and collocation integration subsystem design. |

---

## References

[1] J. T. Betts, *Practical Methods for Optimal Control and Estimation Using Nonlinear Programming*, 2nd ed. Philadelphia, PA: Society for Industrial and Applied Mathematics, 2010.

[2] E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*, 2nd ed. Berlin, Germany: Springer-Verlag, 1996.

[3] A. C. Hindmarsh, P. N. Brown, K. E. Grant, S. L. Lee, R. Serban, D. E. Shumaker, and C. S. Woodward, "SUNDIALS: Suite of Nonlinear and Differential/Algebraic Equation Solvers," *ACM Transactions on Mathematical Software*, vol. 31, no. 3, pp. 363–396, 2005, doi: 10.1145/1089014.1089020.

[4] R. Verschueren, G. Frison, D. Kouzoupis, J. Frey, N. van Duijkeren, Andrea Zanelli, B. Novoselnik, T. Albin, R. Quirynen, and M. Diehl, "acados -- a modular open-source framework for fast embedded optimal control," *Mathematical Programming Computation*, vol. 14, no. 1, pp. 147–183, 2022, doi: 10.1007/s12532-021-00208-8.

[5] C. A. Kennedy and M. H. Carpenter, "Additive Runge-Kutta schemes for convection-diffusion-reaction equations," *Applied Numerical Mathematics*, vol. 44, no. 1–2, pp. 139–181, 2003, doi: 10.1016/S0168-9274(02)00138-1.

[6] D. R. Reynolds, D. J. Gardner, C. S. Woodward, and R. Chinomona, "ARKODE: A flexible IVP solver infrastructure for one-step methods," *ACM Transactions on Mathematical Software*, vol. 49, no. 2, pp. 19:1–19:30, 2023, doi: 10.1145/3588970.

[7] T.-G. Kim, *peroxide*: Comprehensive numerical computing library for Rust (Version 0.37.0). [Online]. Available: https://docs.rs/peroxide/latest/peroxide/. Accessed: Aug. 26, 2026.
