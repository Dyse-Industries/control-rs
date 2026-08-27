# Adaptive Embedded Runge–Kutta Integrators & Dense Output (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_26,_2026-blue)
![Status: Draft](https://img.shields.io/badge/status-draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

High-accuracy trajectory generation, flight guidance, neural ordinary differential equations (Neural ODEs), and scientific differential equation solving require adaptive step-size integration capable of automatically modulating step size $\Delta t$ to satisfy user-specified local error tolerances (Chen et al., 2018; Dormand and Prince, 1980; Kidger, 2022; Rackauckas and Nie, 2017; Tsitouras, 2011). Furthermore, continuous-time simulation requires dense output polynomial interpolation to evaluate states at arbitrary timestamps between solver mesh points without incurring additional ODE vector field evaluations (Kidger, 2022; Tsitouras, 2011).

This document establishes the architecture, requirement contract, and validation suite for the adaptive embedded Runge–Kutta integrators in `control-rs::integrators::adaptive`. The module satisfies three core usage scenarios:

1. **Non-Stiff Adaptive Trajectory Propagation**: Automatic step-size adjustment using the modern `Tsitouras54` (Tsit5) and classical `DormandPrince54` (Dopri5) pairs with First Same As Last (FSAL) stage reuse (Dormand and Prince, 1980; Rackauckas and Nie, 2017; Tsitouras, 2011).
2. **Continuous Dense Output Interpolation**: High-quality 4th-order polynomial dense output for inter-sample state querying in flight telemetry, sensor fusion, and event detection (Tsitouras, 2011).
3. **Neural ODEs & Differentiable Programming**: Efficient forward and backward adjoint state trajectory propagation with constant memory overhead and FSAL caching (Chen et al., 2018; Kidger, 2022).

---

### 2. Requirements

#### 2.1 Functional Requirements

- **FR-1 — Embedded Runge–Kutta Pair Framework**: Provide a generic adaptive stepper `AdaptiveRungeKutta<T, NX, NU, STAGES, ORDER>` parameterized by primary weights $b \in \mathbb{R}^s$, embedded secondary weights $\hat{b} \in \mathbb{R}^s$, stage matrix $A \in \mathbb{R}^{s \times s}$, and nodes $c \in \mathbb{R}^s$.
- **FR-2 — First Same As Last (FSAL) Optimization**: When $c_s = 1$ and $a_{sj} = b_j$, the stepper must reuse stage $k_s$ from step $n$ as stage $k_1$ of step $n+1$, reducing evaluations per accepted step from $s$ to $s-1$.
- **FR-3 — Tsitouras 5(4) Default Non-Stiff Stepper**: Provide `Tsitouras54` (order 5(4), 7 stages, FSAL) with built-in free 4th-order polynomial continuous dense output.
- **FR-4 — Dormand–Prince 5(4) Classical Stepper**: Provide `DormandPrince54` (Dopri5, order 5(4), 7 stages, FSAL) for standard classical benchmarks and validation.
- **FR-5 — Bogacki–Shampine 3(2) Low-Cost Stepper**: Provide `BogackiShampine32` (BS32, order 3(2), 4 stages, FSAL) for fast, low-precision embedded applications.
- **FR-6 — Proportional-Integral (PI) Step-Size Controller**: Implement PI step-size adaptation calculating $h_{new} = h_n \cdot \min(\text{fac}_{max}, \max(\text{fac}_{min}, \text{fac} \cdot (\text{tol}/e_{n+1})^\alpha (e_n/e_{n+1})^\beta))$ based on user-defined absolute tolerance $ATOL$ and relative tolerance $RTOL$.
- **FR-7 — Continuous Dense Output Polynomial Evaluator**: Provide `DenseOutput<T, NX>` evaluating $x(t_n + \theta h) = x_n + h \sum_{i=1}^s b_i(\theta) k_i$ for any normalized fractional offset $\theta \in [0, 1]$.

#### 2.2 Non-Functional Requirements

- **NFR-1 — Zero Dynamic Memory Allocation**: Stage buffers, error vectors, and dense output polynomials must allocate exclusively on the stack (`no_alloc`).
- **NFR-2 — Compile-Time Static Sizing**: State dimension $N_x$, input dimension $N_u$, and stage count $s$ must be validated at compile time via `Dim`.
- **NFR-3 — Bounded Step Rejection**: Step rejection loops must detect step-size collapse ($h < h_{min}$) and return structured errors (`LinAlgError::StepSizeUnderflow`) without infinite loops.

#### 2.3 Constraints

- **C-1 — Native Rust Implementation**: Authored natively in Rust without C code generation ([`controls-tools-design.md`](../control-toolboxes/controls-tools-design.md) C-1).
- **C-2 — Storage Decoupling**: State and vector storage must use `control-rs::matrix::Owned` and `DenseStorage` ([`matrix-design.md`](../numerical-models/matrix-design.md)).
- **C-3 — Zero Virtual Dispatch**: Steppers must monomorphize over `Sys: SystemDynamics<T, NX, NU>`.

---

### 3. Technical Overview

An embedded Runge–Kutta method computes two state estimates (Dormand and Prince, 1980; Hairer et al., 1993; Tsitouras, 2011):
$$x_{n+1} = x_n + h \sum_{i=1}^s b_i k_i \quad (\text{order } p), \qquad \hat{x}_{n+1} = x_n + h \sum_{i=1}^s \hat{b}_i k_i \quad (\text{order } p-1)$$
The local truncation error norm is evaluated across all state components:
$$e_{n+1} = \sqrt{\frac{1}{N_x} \sum_{r=1}^{N_x} \left(\frac{x_{n+1, r} - \hat{x}_{n+1, r}}{\text{ATOL} + \text{RTOL} \cdot \max(|x_{n, r}|, |x_{n+1, r}|)}\right)^2}$$
If $e_{n+1} \le 1.0$, the step is accepted and $x_{n+1}$ is committed; otherwise, the step is rejected and recomputed with a reduced step size $h$ (Hairer et al., 1993).

```
Step n:
  Stage 1: k₁ = f(t_n, x_n, u)   [or reused from k_s of Step n-1 via FSAL]
  Stages 2..s: kᵢ = f(t_n + cᵢh, x_n + h·∑ aᵢⱼ kⱼ, u)
  Solutions: x_{n+1} = x_n + h·∑ bᵢ kᵢ,  x̂_{n+1} = x_n + h·∑ b̂ᵢ kᵢ
  Error: e_{n+1} = ||x_{n+1} - x̂_{n+1}||_scaled
  Accept (e ≤ 1): commit x_{n+1}, cache k_s as k₁ for Step n+1, update h_new
  Reject (e > 1): retry step with smaller h
```

Dense output evaluates a continuous polynomial interpolant using precomputed tableau stage weights $b_i(\theta) = \sum_{m=1}^4 d_{im} \theta^m$ (Kidger, 2022; Tsitouras, 2011).

---

### 4. Architecture

#### 4.1 Generic Adaptive Stepper Engine

**Proposal (not in evidence)**: Implement `AdaptiveRungeKutta` with compile-time tableau constants and FSAL stage caching:

```rust
/// Adaptive embedded Runge-Kutta integrator with FSAL and dense output.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveRungeKutta<T, const STAGES: usize, const ORDER: usize> {
    pub a: [[T; STAGES]; STAGES],
    pub b: [T; STAGES],
    pub b_hat: [T; STAGES],
    pub c: [T; STAGES],
    pub d: [[T; STAGES]; 4], // Dense output polynomial weights
}
```

#### 4.2 PI Step Controller

**Proposal (not in evidence)**: Define `AdaptiveStepController`:

```rust
/// Proportional-Integral (PI) adaptive step-size controller.
#[derive(Debug, Clone, Copy)]
pub struct StepControlConfig<T> {
    pub atol: T,
    pub rtol: T,
    pub safety_factor: T, // Typically 0.9
    pub min_factor: T,    // Typically 0.2
    pub max_factor: T,    // Typically 5.0
    pub min_step: T,
    pub max_step: T,
}
```

---

### 5. Alternatives Considered

- **Non-FSAL Embedded Pairs (RKF45, Cash-Karp)**: Require evaluating $k_1$ anew at every step, wasting 14.3% of computation relative to `Tsitouras54` and `DormandPrince54` (Dormand and Prince, 1980; Tsitouras, 2011).
- **Pure Proportional Step Control ($\beta = 0$)**: Prone to step-size oscillations on stiffening dynamics; the PI controller ($\alpha = 0.7/p, \beta = 0.4/p$) dampens step-size variance (Hairer et al., 1993).

---

### 6. Verification & Validation Plan

#### 6.1 Unit Verification
- **FSAL Cache Validation**: Assert that in a multi-step continuous trajectory, the vector field derivative function is called exactly $(s-1) \times N_{\text{steps}} + 1$ times for accepted steps.
- **Error Estimation Verification**: Step linear decay ODE with known exact analytical error. Assert local error estimate matches true difference $|x_{p} - x_{p-1}|$ within $1\%$.
- **Dense Output Precision**: Compute dense output $x(t_n + \theta h)$ for $\theta \in [0.1, 0.9]$ on $y(t) = \sin(t)$. Assert interpolation error scales as $O(h^4)$ across the entire interval.
- **Arenstorf Restricted Three-Body Orbit**: Simulate chaotic three-body orbit under adaptive tolerance $\text{ATOL} = 10^{-8}, \text{RTOL} = 10^{-8}$. Assert stepper successfully negotiates close lunar flybys without step collapse.

#### 6.2 Validation
- **Neural ODE Continuous Depth Validation**: Integrate a continuous neural vector field backward and forward; assert state reconstruction error $< 10^{-6}$ using adjoint trajectory caching (Chen et al., 2018; Kidger, 2022).

---

### 7. Performance & Resource Considerations

- **Stack Budget**: For $N_x = 12$ and $s = 7$ (`Tsitouras54`), total stack scratch space is $7 \times 12 \times 8 = 672$ bytes.
- **Evaluation Throughput**: Benchmarks in scientific computing demonstrate `Tsitouras54` outperforms classical `Dopri5` by 10–20% on non-stiff problems due to optimized coefficient error constants (Rackauckas and Nie, 2017; Tsitouras, 2011).

---

### 8. Risks & Open Questions

- **Proposal (not in evidence) 1**: Const-generic `AdaptiveRungeKutta` with compile-time dense output matrices.
- **Proposal (not in evidence) 2**: In-place FSAL buffer caching without dynamic allocation.

---

### 9. Development Plan

| Phase | Description | Estimated Effort (1-10) |
|:------|:------------|:------------------------|
| **Phase 1: Embedded Tableaus & FSAL Engine** | Implement `AdaptiveRungeKutta`, `Tsitouras54`, `DormandPrince54`, and `BogackiShampine32`. | 3 |
| **Phase 2: Step Controller & Dense Output** | Implement `StepControlConfig`, PI controller, and `DenseOutput` polynomial interpolator. | 4 |
| **Phase 3: Orbit & Neural ODE Verification** | Implement Arenstorf three-body orbit tests and dense interpolation verification. | 3 |

---

### 10. Revision History

| Version | Date | Author | Description |
|:--------|:-----|:-------|:------------|
| 1.0 | August 26, 2026 | @MitchellDScott | Initial draft: adaptive embedded Runge-Kutta integrators and dense output design. |

---

## References

[1] R. T. Q. Chen, Y. Rubanova, J. Bettencourt, and D. Duvenaud, "Neural Ordinary Differential Equations," in *Advances in Neural Information Processing Systems 31 (NeurIPS 2018)*, 2018, pp. 6571–6583.

[2] J. R. Dormand and P. J. Prince, "A family of embedded Runge-Kutta formulae," *Journal of Computational and Applied Mathematics*, vol. 6, no. 1, pp. 19–26, 1980, doi: 10.1016/0771-050X(80)90013-3.

[3] P. Kidger, "On Neural Differential Equations," University of Oxford, Oxford, UK, Rep. no. arXiv:2202.02435, 2022.

[4] C. Rackauckas and Q. Nie, "DifferentialEquations.jl -- A Performant and Feature-Rich Ecosystem for Solving Differential Equations in Julia," *Journal of Open Research Software*, vol. 5, no. 1, p. 15, 2017, doi: 10.5334/jors.151.

[5] C. Tsitouras, "Runge--Kutta pairs of order 5(4) satisfying only the first column simplifying assumption," *Computers & Mathematics with Applications*, vol. 62, no. 2, pp. 770–775, 2011, doi: 10.1016/j.camwa.2011.06.002.

[6] E. Hairer, S. P. N{\o}rsett, and G. Wanner, *Solving Ordinary Differential Equations I: Nonstiff Problems*, 2nd ed. Berlin, Germany: Springer-Verlag, 1993.

[7] T.-G. Kim, *peroxide*: Comprehensive numerical computing library for Rust (Version 0.37.0). [Online]. Available: https://docs.rs/peroxide/latest/peroxide/. Accessed: Aug. 26, 2026.

[8] S. Renevey, *ode_solvers*: Numerical methods for solving ordinary differential equations in Rust (Version 0.6.2). [Online]. Available: https://docs.rs/ode_solvers/0.6.2/ode_solvers/. Accessed: Aug. 26, 2026.
