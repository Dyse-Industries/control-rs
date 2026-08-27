# Explicit Fixed-Step Runge–Kutta Integrators (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_26,_2026-blue)
![Status: Draft](https://img.shields.io/badge/status-draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

Explicit Runge–Kutta (ERK) methods form the foundational computational workhorse for real-time simulation and digital control of continuous-time dynamical systems (Butcher, 2016; Hairer et al., 1993). In embedded flight control, vehicle dynamics, and nonlinear Model Predictive Control (MPC), differential equations $\dot{x}(t) = f(t, x(t), u(t))$ must be stepped forward across fixed sample intervals $\Delta t$ with strictly deterministic execution timing and zero dynamic memory allocation (Verschueren et al., 2022).

This document specifies the architecture, requirement contract, and verification suite for the fixed-step explicit Runge–Kutta integrators in `control-rs::integrators::explicit`. The module addresses three primary usage scenarios:

1. **Embedded Control Loop Simulation**: Deterministic 1st-order (Forward Euler) and 2nd-order (Heun, Ralston) integration of continuous plant dynamics in high-rate microcontroller control loops (Verschueren et al., 2022).
2. **High-Fidelity State Propagation**: 4th-order Classical Runge–Kutta (RK4) integration for flight navigation, vehicle kinematics, and state estimation (Hairer et al., 1993; Renevey, 2024).
3. **Monomorphized Vector Field Integration**: Direct stepping of `control-rs` continuous models ([`StateSpaceCore`](../numerical-models/state-space-design.md), nonlinear dynamics) implementing `SystemDynamics<T, NX, NU>` without heap allocation or virtual dispatch overhead.

---

### 2. Requirements

#### 2.1 Functional Requirements

- **FR-1 — Compile-Time Butcher Tableau Parameterization**: Provide a generic explicit Runge–Kutta stepper `ExplicitRungeKutta<T, NX, NU, STAGES>` parameterized by compile-time Butcher tableau coefficients $A \in \mathbb{R}^{s \times s}$, $b \in \mathbb{R}^s$, and $c \in \mathbb{R}^s$ with strictly lower triangular $A$ ($a_{ij} = 0$ for $j \ge i$).
- **FR-2 — Forward Euler Stepper**: Provide a 1-stage, 1st-order explicit Euler integrator evaluating $x_{n+1} = x_n + h f(t_n, x_n, u_n)$.
- **FR-3 — Second-Order Explicit Steppers**: Provide 2-stage, 2nd-order explicit integrators including `Heun` (explicit trapezoidal, $c = [0, 1]^T, b = [1/2, 1/2]^T$) and `Ralston` ($c = [0, 2/3]^T, b = [1/4, 3/4]^T$, minimizing the local truncation error bound).
- **FR-4 — Classical Fourth-Order Runge–Kutta Stepper**: Provide the standard 4-stage, 4th-order `RungeKutta4` integrator ($c = [0, 1/2, 1/2, 1]^T, b = [1/6, 1/3, 1/3, 1/6]^T$).
- **FR-5 — In-Place System Dynamics Stepping**: Steppers must advance state vectors in place ($x \leftarrow x_{n+1}$) via caller-provided input references without intermediate vector allocation.

#### 2.2 Non-Functional Requirements

- **NFR-1 — Zero Heap Allocation**: All stage derivatives $k_1, \dots, k_s$ and temporary state accumulations must reside exclusively on the stack (`no_alloc`).
- **NFR-2 — Compile-Time Static Sizing**: State dimension $N_x$, input dimension $N_u$, and stage count $s$ must be fixed at compile time using `Dim` and const generics.
- **NFR-3 — Deterministic Constant-Time Execution**: Stepper routines must execute in a constant number of CPU cycles with no data-dependent branching or variable iteration loops.

#### 2.3 Constraints

- **C-1 — Native Rust Implementation**: Algorithms must be authored directly in Rust without external C code generation (inheriting C-1/C-2 from [`controls-tools-design.md`](../control-toolboxes/controls-tools-design.md)).
- **C-2 — Storage Subsystem Interoperability**: State and input vectors must use the crate's `Owned<T, R, C>` and `Storage` traits.
- **C-3 — Zero Virtual Dispatch Overhead**: Integration loops must fully monomorphize over the system dynamics type `Sys: SystemDynamics<T, NX, NU>`.

---

### 3. Technical Overview

An explicit $s$-stage Runge–Kutta method computes $x_{n+1} \approx x(t_n + h)$ from $x_n = x(t_n)$ via (Butcher, 2016; Hairer et al., 1993):
$$k_i = f\left(t_n + c_i h, \, x_n + h \sum_{j=1}^{i-1} a_{ij} k_j, \, u(t_n + c_i h)\right), \quad i = 1, \dots, s$$
$$x_{n+1} = x_n + h \sum_{i=1}^s b_i k_i$$

```
Stage 1: k₁ = f(t_n, x_n, u)
Stage 2: k₂ = f(t_n + c₂h, x_n + h·a₂₁k₁, u)
Stage 3: k₃ = f(t_n + c₃h, x_n + h·(a₃₁k₁ + a₃₂k₂), u)
...
Update:  x_{n+1} = x_n + h·∑ bᵢ kᵢ
```

Tableau definitions (Hairer et al., 1993):
- **Euler (1st order)**: $c = [0], A = [0], b = [1]$.
- **Heun (2nd order)**: $c = [0, 1]^T, a_{21} = 1, b = [1/2, 1/2]^T$.
- **Ralston (2nd order)**: $c = [0, 2/3]^T, a_{21} = 2/3, b = [1/4, 3/4]^T$.
- **Classical RK4 (4th order)**: $c = [0, 1/2, 1/2, 1]^T, a_{21} = 1/2, a_{32} = 1/2, a_{43} = 1, b = [1/6, 1/3, 1/3, 1/6]^T$.

---

### 4. Architecture

#### 4.1 Generic Explicit Stepper Struct

**Proposal (not in evidence)**: Define `ExplicitRungeKutta` storing Butcher tableau arrays as const fields:

```rust
/// Generic explicit Runge-Kutta integrator over static stages.
#[derive(Debug, Clone, Copy)]
pub struct ExplicitRungeKutta<T, const STAGES: usize> {
    pub a: [[T; STAGES]; STAGES],
    pub b: [T; STAGES],
    pub c: [T; STAGES],
}

impl<T: Float + Copy, const STAGES: usize> ExplicitRungeKutta<T, STAGES> {
    /// Steps `x` forward by `dt` in place using stack-allocated stage buffers.
    pub fn step<const NX: usize, const NU: usize, Sys>(
        &self,
        sys: &Sys,
        t: T,
        x: &mut Owned<T, NX, 1>,
        u: &Owned<T, NU, 1>,
        dt: T,
    ) where
        Sys: SystemDynamics<T, NX, NU>,
        Const<NX>: Dim,
        Const<NU>: Dim,
    {
        let mut k = [Owned::<T, NX, 1>::zero(); STAGES];
        let mut x_stage = Owned::<T, NX, 1>::zero();

        for i in 0..STAGES {
            // x_stage = x + dt * sum_{j=0}^{i-1} a_{ij} * k_j
            x_stage.clone_from(x);
            for j in 0..i {
                let a_ij = self.a[i][j];
                if a_ij != T::zero() {
                    for r in 0..NX {
                        if let (Some(xs), Some(&kj)) = (x_stage.get_mut(r, 0), k[j].get(r, 0)) {
                            *xs = *xs + dt * a_ij * kj;
                        }
                    }
                }
            }
            let t_stage = t + self.c[i] * dt;
            sys.evaluate_derivative(t_stage, &x_stage, u, &mut k[i]);
        }

        // x_{n+1} = x_n + dt * sum_{i=0}^{STAGES-1} b_i * k_i
        for i in 0..STAGES {
            let b_i = self.b[i];
            if b_i != T::zero() {
                for r in 0..NX {
                    if let (Some(xn), Some(&ki)) = (x.get_mut(r, 0), k[i].get(r, 0)) {
                        *xn = *xn + dt * b_i * ki;
                    }
                }
            }
        }
    }
}
```

#### 4.2 Concrete Stepper Instances

**Proposal (not in evidence)**: Expose optimized constructors for standard methods:

```rust
/// Forward Euler 1st-order stepper.
pub struct Euler;
/// Heun 2nd-order stepper.
pub struct Heun;
/// Ralston 2nd-order minimum-error stepper.
pub struct Ralston;
/// Classical 4th-order Runge-Kutta stepper.
pub struct RungeKutta4;
```

---

### 5. Alternatives Considered

- **Dynamic Stage Buffers (`Vec<T>`) vs Stack Arrays (`[Owned<T, NX, 1>; STAGES]`)**: Dynamic heap allocation (as used in standard host libraries) (Kim, 2024; Renevey, 2024) is rejected to satisfy `no_alloc` embedded constraints (C-1, NFR-1).
- **Runtime Tableau Parsing vs Compile-Time Const Tables**: Parsing tableaus from memory at runtime prevents loop unrolling and const-folding. Const generic tables allow the compiler to eliminate zero terms ($a_{ij} = 0$).

---

### 6. Verification & Validation Plan

#### 6.1 Unit Verification
- **Convergence Rate Verification**: Test on exponential decay $\dot{x} = -\lambda x$ ($x(t) = x_0 e^{-\lambda t}$) over steps $h \in [0.1, 0.001]$. Assert linear regression of $\log_{10}(\text{error})$ vs $\log_{10}(h)$ yields slopes:
  - `Euler`: $1.00 \pm 0.05$ (order 1).
  - `Heun`: $2.00 \pm 0.05$ (order 2).
  - `Ralston`: $2.00 \pm 0.05$ (order 2).
  - `RungeKutta4`: $4.00 \pm 0.05$ (order 4).
- **Harmonic Oscillator Test**: Test undamped oscillator $\ddot{x} + \omega^2 x = 0$ over $t \in [0, 10\pi]$. Verify error bounds scale as $O(h^p)$.

#### 6.2 Validation
- **State-Space Model Integration**: Step a continuous 2-state DC motor model (`StateSpaceCore`) with `RungeKutta4`; compare against analytical step response from matrix exponential series.

---

### 7. Performance & Resource Considerations

- **Stack Memory Overhead**: For an $N_x = 8$ state model, `RungeKutta4` allocates $4 \times 8 \times 8 = 256$ bytes of temporary stage buffers on the stack.
- **Loop Unrolling**: For $s \le 4$, modern LLVM backends unroll stage updates into direct FMA (Fused Multiply-Add) SIMD vector instructions.

---

### 8. Risks & Open Questions

- **Proposal (not in evidence) 1**: Const-generic `ExplicitRungeKutta` struct representation with compile-time array tables.
- **Proposal (not in evidence) 2**: Direct stage evaluation in-place using stack-allocated `[Owned<T, NX, 1>; STAGES]`.

---

### 9. Development Plan

| Phase | Description | Estimated Effort (1-10) |
|:------|:------------|:------------------------|
| **Phase 1: Tableau & Trait Definitions** | Implement `FixedStepper` trait and generic `ExplicitRungeKutta` engine with static stage arrays. | 2 |
| **Phase 2: Concrete Steppers** | Implement `Euler`, `Heun`, `Ralston`, and `RungeKutta4` constructors and optimized stage loops. | 3 |
| **Phase 3: Verification Suite** | Implement order convergence unit tests and `StateSpaceCore` integration tests. | 2 |

---

### 10. Revision History

| Version | Date | Author | Description |
|:--------|:-----|:-------|:------------|
| 1.0 | August 26, 2026 | @MitchellDScott | Initial draft: explicit fixed-step Runge-Kutta integration subsystem design. |

---

## References

[1] J. C. Butcher, *Numerical Methods for Ordinary Differential Equations*, 3rd ed. Chichester, UK: John Wiley & Sons, Ltd, 2016.

[2] E. Hairer, S. P. N{\o}rsett, and G. Wanner, *Solving Ordinary Differential Equations I: Nonstiff Problems*, 2nd ed. Berlin, Germany: Springer-Verlag, 1993.

[3] R. Verschueren, G. Frison, D. Kouzoupis, J. Frey, N. van Duijkeren, Andrea Zanelli, B. Novoselnik, T. Albin, R. Quirynen, and M. Diehl, "acados -- a modular open-source framework for fast embedded optimal control," *Mathematical Programming Computation*, vol. 14, no. 1, pp. 147–183, 2022, doi: 10.1007/s12532-021-00208-8.

[4] S. Renevey, *ode_solvers*: Numerical methods for solving ordinary differential equations in Rust (Version 0.6.2). [Online]. Available: https://docs.rs/ode_solvers/0.6.2/ode_solvers/. Accessed: Aug. 26, 2026.

[5] T.-G. Kim, *peroxide*: Comprehensive numerical computing library for Rust (Version 0.37.0). [Online]. Available: https://docs.rs/peroxide/latest/peroxide/. Accessed: Aug. 26, 2026.
