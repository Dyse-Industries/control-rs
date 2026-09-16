# Integrators (Design Document)

![Date Badge](https://img.shields.io/badge/Date-September_15,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Approved-brightgreen)
![Author Badge](https://img.shields.io/badge/Author-@mitchelldscott-blueviolet)

---

### 1. Introduction

This module advances continuous plant state between discrete controller
updates on host and bare-metal targets. Tableau coefficients, trait names,
and settings types belong in §3/§4.

Primary usage scenarios:

- Advance a plant state across a controller sample interval with a
  fixed-step explicit scheme.
- Refine that interval with a caller-chosen number of equal sub-steps.
- Advance with a local-error-controlled adaptive step, and stop when a
  caller-chosen evaluation budget is exhausted.
- Swap schemes without rewriting the plant's vector field or its state
  container.

---

### 2. Requirements

#### 2.1 Functional Requirements

- **FR-1 — State advance over intervals**: Advance a system state across an
  interval using explicit numerical Runge-Kutta schemes with either fixed step
  size or local error-controlled adaptive step size. Implicit schemes requiring
  Jacobian inversions are excluded.
- **FR-2 — Interchangeable numerical schemes**: The same plant and settings
  contract accept more than one explicit scheme. Scheme names and tableaus are
  a §4 choice.
- **FR-3 — Sub-step refinement**: Propagate state across an outer sample
  period $\Delta t$ in a caller-specified number of equal
  sub-steps $h = \Delta t / M$, decoupling the inner integration resolution from
  the outer controller sample period.
- **FR-4 — Model and vector field abstraction**: Abstract the continuous-time
  vector field definition $\dot{x}(t) = f(t, x(t), u(t))$ independently from the
  underlying integration algorithm.
- **FR-5 — Container-independent representations**: Support continuous state and
  output representations across primitive arrays, matrix types, or custom structs
  without binding solvers to a single concrete container.
- **FR-6 — Configurable integration parameters**: Support both pre-packaged
  default integration settings and caller-specified step parameters across
  simulation drivers.
- **FR-7 — Embedded error estimation and tolerance scaling**: Embedded
  Runge-Kutta tableaus must compute local truncation error vectors and scale
  them against user-specified absolute and relative
  tolerances ($\text{atol}, \text{rtol}$) to determine step acceptance and
  calculate candidate step sizes.

#### 2.2 Non-Functional Requirements

- **NFR-1 — Predictable memory usage**: Integrator stages, local error estimates,
  and intermediate evaluations operate strictly within bounded stack frames
  without dynamic memory growth or unbounded recursion.
- **NFR-2 — Bounded execution cost**: Fixed-step algorithms document exact
  derivative evaluation counts per step. Adaptive-step execution enforces a
  strict upper bound on total stage evaluations per interval advance (
  `max_attempts`), guaranteeing bounded worst-case execution latency on bare
  metal.

#### 2.3 Constraints

- **C-1 — `no_std` / `no_alloc`**: The module operates in bare-metal embedded
  environments without the Rust standard library or dynamic memory allocation.
- **C-2 — Explicit Runge-Kutta schemes only**: Solvers are restricted to
  explicit single-step tableaus; implicit linear systems and Jacobian solves are
  excluded.
- **C-3 — Zero-order hold actuation**: Exogenous input $u$ is held constant
  across each integration step $\Delta t$, matching digital-to-analog converter
  and PWM actuation.

---

### 3. Technical Overview

The module solves initial value problems (IVPs) for systems of ordinary
differential equations:
$$\dot{x}(t) = f(t, x(t), u), \quad x(t_0) = x_0$$
where $x \in \mathbb{R}^n$ is the state vector, $u \in \mathbb{R}^m$ is the
constant exogenous input across the step,
and $f: \mathbb{R} \times \mathbb{R}^n \times \mathbb{R}^m \to \mathbb{R}^n$ is
the vector field.

#### 3.1 Explicit Runge-Kutta Tableaus

Numerical propagation applies explicit Runge-Kutta methods specified by Butcher
tableaus (Butcher, 2016):
$$k_i = f\left(t_n + c_i h, \, x_n + h \sum_{j=1}^{i-1} a_{ij} k_j, \, u\right), \quad i = 1, \dots, s$$
$$x_{n+1} = x_n + h \sum_{i=1}^s b_i k_i$$
Supported methods include:

1. **Forward Euler** ($s=1$, order 1).
2. **Explicit Midpoint / Heun** ($s=2$, order 2).
3. **Classical RK4** ($s=4$, order 4).
4. **Dormand-Prince 5(4)** ($s=7$, order 5, with embedded order 4 error
   estimate) (Dormand and Prince, 1980). Under fixed-step propagation with
   First-Same-As-Last (FSAL), stage $k_7$ matches stage $k_1$ of the subsequent
   step, reducing computational cost to 6 evaluations per step (Hairer et al.,
   1993).

#### 3.2 Embedded Error Estimation & Adaptive Step Control

For embedded Runge-Kutta pairs (Dormand and Prince, 1980; Hairer et al., 1993),
two approximations of orders $p$ and $\hat{p} = p - 1$ are generated
simultaneously:
$$x_{n+1} = x_n + h \sum_{i=1}^s b_i k_i \quad (\text{order } 5), \qquad \hat{x}_{n+1} = x_n + h \sum_{i=1}^s \hat{b}_i k_i \quad (\text{order } 4)$$
The local truncation error vector is:
$$e_{n+1} = x_{n+1} - \hat{x}_{n+1} = h \sum_{i=1}^s (b_i - \hat{b}_i) k_i$$
Component-wise tolerance scaling combines absolute and relative bounds:
$$sc_i = \text{atol} + \max(|x_{n,i}|, |x_{n+1,i}|) \cdot \text{rtol}$$
The scalar error norm is computed using the root-mean-square norm:
$$\|e\|_{\text{RMS}} = \sqrt{\frac{1}{n} \sum_{i=1}^n \left(\frac{e_{n+1,i}}{sc_i}\right)^2}$$
Step size adaptation follows the standard Hairer–Nørsett–Wanner formula (Hairer
et al., 1993):
$$h_{\text{new}} = h \cdot \min\left(\text{facmax}, \, \max\left(\text{facmin}, \, \text{fac} \cdot \left(\frac{1}{\|e\|_{\text{RMS}}}\right)^{\frac{1}{p+1}}\right)\right)$$
where $\text{fac} \approx 0.9$ is a safety factor, $\text{facmin} \approx 0.2$,
and $\text{facmax} \approx 5.0$.

- **Acceptance**: If $\|e\|_{\text{RMS}} \le 1.0$, the step is accepted, state
  advances $t \leftarrow t + h, x \leftarrow x_{n+1}$,
  and $h \leftarrow h_{\text{new}}$.
- **Rejection**: If $\|e\|_{\text{RMS}} > 1.0$, the step is rejected and
  recomputed with $h \leftarrow h_{\text{new}}$ without advancing $t$ or $x$.

Following modern generic scientific computing architectures (Ahnert and
Mulansky, 2011; Rackauckas and Nie, 2017; Robinson and Allmont, 2026), the
system definition is decoupled from the state container and the solver execution
loop.

---

### 4. Architecture

#### 4.1 Module Structure

```text
integrators
├── traits
│   ├── DynamicSystem               trait: continuous vector field dx/dt = f(t, x, u)
│   ├── VectorSpace                 trait: element-wise linear combination and error norm
│   ├── SystemIntegrator            trait: single-step and interval numerical propagation
│   └── EmbeddedIntegrator          trait: embedded local error estimation for adaptive stepping
├── settings
│   ├── FixedStepSettings           struct: sub-step count (M >= 1)
│   └── AdaptiveStepSettings        struct: atol, rtol, safety factors, step limits, max_attempts
├── solvers
│   ├── Euler                       struct: 1st-order explicit Runge-Kutta
│   ├── Midpoint                    struct: 2nd-order explicit Runge-Kutta
│   ├── Rk4                         struct: classical 4th-order Runge-Kutta
│   └── DormandPrince54             struct: 5th-order Dormand-Prince tableau
└── implementations
    ├── StateSpace                  impl DynamicSystem for GenericStateSpace
    └── Closures                    blanket impl DynamicSystem for Fn/FnMut
```

#### 4.2 Core Traits

The trait signatures below adapt the container-independent concepts of Ahnert
and Mulansky (2011) and Renevey (2026) to Rust associated types and `no_std`
constraints.

```rust
/// Continuous-time dynamical system defining state derivatives.
pub trait DynamicSystem<T = f64> {
    /// State vector representation (e.g., [T; N], Owned<T, N, 1>, or caller struct).
    type State;
    /// Exogenous input representation held constant across an integration step.
    type Input;
    /// Derivative output representation (dx/dt).
    type Output;

    /// Computes the vector field dx/dt = f(t, x, u).
    fn derivative(&self, t: T, x: &Self::State, u: &Self::Input) -> Self::Output;
}

/// Linear algebra operations required on state containers for Runge-Kutta stage accumulation.
pub trait VectorSpace<T = f64>: Clone {
    /// Derivative type added to the state.
    type Derivative;

    /// Computes self + dt * derivative.
    fn scale_add(&self, dt: T, derivative: &Self::Derivative) -> Self;

    /// Computes linear combination of stages: self + dt * sum(weights[i] * stages[i]).
    fn advance_stages<const S: usize>(&self, dt: T, weights: &[T; S], stages: &[Self::Derivative; S]) -> Self;

    /// Computes linear combination of stage derivatives: dt * sum(weights[i] * stages[i]).
    fn combine_derivatives<const S: usize>(dt: T, weights: &[T; S], stages: &[Self::Derivative; S]) -> Self::Derivative;

    /// Computes RMS error norm: sqrt( (1/n) * sum( (err_i / (atol + max(|x_i|, |x_next_i|) * rtol))^2 ) ).
    fn error_norm(&self, error: &Self::Derivative, atol: T, rtol: T, x_next: &Self) -> T;
}
```

A blanket implementation of `VectorSpace<T>` is provided for standard fixed
arrays `[T; N]` where `T: crate::math::num_traits::Float + Copy`.

#### 4.3 Settings Types

```rust
/// Execution settings for fixed-step sub-stepping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedStepSettings {
    /// Number of sub-steps per outer interval advance (M >= 1).
    pub sub_steps: usize,
}

impl Default for FixedStepSettings {
    fn default() -> Self {
        Self { sub_steps: 1 }
    }
}

/// Execution settings for error-controlled adaptive step size integration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveStepSettings<T = f64> {
    /// Absolute error tolerance.
    pub atol: T,
    /// Relative error tolerance.
    pub rtol: T,
    /// Safety factor applied to the computed step adjustment (typically 0.8 to 0.9).
    pub safety_factor: T,
    /// Minimum allowed step shrinkage factor per step (typically 0.1 to 0.2).
    pub min_factor: T,
    /// Maximum allowed step growth factor per step (typically 1.5 to 5.0).
    pub max_factor: T,
    /// Minimum allowed step size to prevent numerical stall.
    pub min_step: T,
    /// Maximum allowed step size.
    pub max_step: T,
    /// Maximum allowed sub-step attempts per outer interval before returning an error.
    pub max_attempts: usize,
}

impl Default for AdaptiveStepSettings<f64> {
    fn default() -> Self {
        Self {
            atol: 1e-6,
            rtol: 1e-6,
            safety_factor: 0.9,
            min_factor: 0.2,
            max_factor: 5.0,
            min_step: 1e-12,
            max_step: 1.0,
            max_attempts: 1000,
        }
    }
}
```

#### 4.4 Integrator Traits

```rust
/// Numerical integrator capable of stepping a dynamic system with fixed settings.
pub trait SystemIntegrator<Sys: DynamicSystem<T>, T = f64> {
    /// Settings type governing step execution.
    type Settings: Copy;

    /// Advances the state across a single step of size `dt`.
    fn step(
        &self,
        system: &Sys,
        t: T,
        x: &Sys::State,
        u: &Sys::Input,
        dt: T,
    ) -> Sys::State;

    /// Advances the state across an outer interval `dt` using packaged settings.
    fn advance(
        &self,
        system: &Sys,
        t: T,
        x: &Sys::State,
        u: &Sys::Input,
        dt: T,
    ) -> Sys::State;

    /// Advances the state across an outer interval `dt` using explicit caller settings.
    fn advance_with_settings(
        &self,
        system: &Sys,
        t: T,
        x: &Sys::State,
        u: &Sys::Input,
        dt: T,
        settings: &Self::Settings,
    ) -> Sys::State;
}

/// Embedded Runge-Kutta integrator capable of local error estimation and adaptive stepping.
pub trait EmbeddedIntegrator<Sys: DynamicSystem<T>, T = f64>: SystemIntegrator<Sys, T>
where
    Sys::State: VectorSpace<T>,
{
    /// Steps state across `dt` and returns candidate state and local truncation error estimate.
    fn step_with_error(
        &self,
        system: &Sys,
        t: T,
        x: &Sys::State,
        u: &Sys::Input,
        dt: T,
    ) -> (Sys::State, <Sys::State as VectorSpace<T>>::Derivative);

    /// Advances state across outer interval `dt` using adaptive step size control.
    fn advance_adaptive(
        &self,
        system: &Sys,
        t: T,
        x: &Sys::State,
        u: &Sys::Input,
        dt: T,
        settings: &AdaptiveStepSettings<T>,
    ) -> Result<Sys::State, IntegratorError>;
}

/// Errors returned during numerical integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegratorError {
    /// Adaptive step loop exceeded maximum allowed attempts.
    MaxAttemptsExceeded,
    /// Step size shrank below minimum allowed threshold.
    StepSizeUnderflow,
}

impl core::fmt::Display for IntegratorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MaxAttemptsExceeded => {
                write!(f, "adaptive integration exceeded maximum attempt limit")
            }
            Self::StepSizeUnderflow => write!(f, "step size underflow: smaller than min_step"),
        }
    }
}

impl core::error::Error for IntegratorError {}
```

#### 4.5 Concrete Integrators & Constructors

Each integrator struct provides constructors returning `Self` configured with
settings:

```rust
/// Classical 4th-order Runge-Kutta integrator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rk4 {
    pub settings: FixedStepSettings,
}

impl Rk4 {
    pub const fn new() -> Self {
        Self { settings: FixedStepSettings { sub_steps: 1 } }
    }

    pub const fn with_substeps(sub_steps: usize) -> Self {
        Self { settings: FixedStepSettings { sub_steps } }
    }

    pub const fn with_settings(settings: FixedStepSettings) -> Self {
        Self { settings }
    }
}

/// Dormand-Prince 5(4) integrator (DOPRI5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DormandPrince54<T = f64> {
    pub fixed_settings: FixedStepSettings,
    pub adaptive_settings: Option<AdaptiveStepSettings<T>>,
}

impl<T: crate::math::num_traits::Float + Copy> DormandPrince54<T> {
    /// Creates a Dormand-Prince 5(4) integrator with default fixed-step settings (1 sub-step).
    pub const fn new() -> Self {
        Self {
            fixed_settings: FixedStepSettings { sub_steps: 1 },
            adaptive_settings: None,
        }
    }

    /// Configures with fixed sub-step count.
    pub const fn with_substeps(sub_steps: usize) -> Self {
        Self {
            fixed_settings: FixedStepSettings { sub_steps },
            adaptive_settings: None,
        }
    }

    /// Configures with error-controlled adaptive step size settings.
    pub const fn with_adaptive(settings: AdaptiveStepSettings<T>) -> Self {
        Self {
            fixed_settings: FixedStepSettings { sub_steps: 1 },
            adaptive_settings: Some(settings),
        }
    }
}
```

Equivalent fixed-step constructors (`new`, `with_substeps`, `with_settings`) are
provided for `Euler` and `Midpoint`.

#### 4.6 Downstream Generic Function Patterns

Functions consuming integrators support fixed and adaptive execution:

**Pattern A: Generic function with packaged fixed-step integrator**

```rust
pub fn simulate_interval<Sys, Int>(
    system: &Sys,
    integrator: &Int,
    t: f64,
    x: &Sys::State,
    u: &Sys::Input,
    dt: f64,
) -> Sys::State
where
    Sys: DynamicSystem<f64>,
    Int: SystemIntegrator<Sys, f64>,
{
    integrator.advance(system, t, x, u, dt)
}

fn main() {
    // Call site:
    let x_next = simulate_interval(&motor, &Rk4::with_substeps(10), t, &x, &input, ts);
}
```

**Pattern B: Generic function with adaptive step size settings**

```rust
pub fn simulate_interval_adaptive<Sys, Int>(
    system: &Sys,
    integrator: &Int,
    t: f64,
    x: &Sys::State,
    u: &Sys::Input,
    dt: f64,
    settings: &AdaptiveStepSettings<f64>,
) -> Result<Sys::State, IntegratorError>
where
    Sys: DynamicSystem<f64>,
    Sys::State: VectorSpace<f64>,
    Int: EmbeddedIntegrator<Sys, f64>,
{
    integrator.advance_adaptive(system, t, x, u, dt, settings)
}

fn main() {
    // Call site:
    let x_next = simulate_interval_adaptive(
        &motor,
        &DormandPrince54::new(),
        t,
        &x,
        &input,
        ts,
        &AdaptiveStepSettings::default(),
    )?;
}
```

#### 4.7 Integration with `GenericStateSpace`

`GenericStateSpace` implements `DynamicSystem` using its existing `derivative`
method:

```rust
impl<T: crate::math::num_traits::Scalar + Copy, const NX: usize, const NU: usize, const NY: usize>
DynamicSystem<T> for crate::state_space::StateSpace<T, NX, NU, NY>
where
    crate::math::num_types::Const<NX>: crate::math::num_types::Dim,
    crate::math::num_types::Const<NU>: crate::math::num_types::Dim,
    crate::math::num_types::Const<NY>: crate::math::num_types::Dim,
{
    type State = crate::matrix::Owned<T, NX, 1>;
    type Input = crate::matrix::Owned<T, NU, 1>;
    type Output = crate::matrix::Owned<T, NX, 1>;

    fn derivative(&self, _t: T, x: &Self::State, u: &Self::Input) -> Self::Output {
        let (x_dot, _) = self.derivative(x, u);
        x_dot
    }
}
```

---

### 5. Alternatives

- **Free functions with a runtime `Method` enum**: (Evaluated in previous
  draft).
    - *Tradeoff*: A single function
      `advance(Method::Rk4, f, x, u, dt, sub_steps)` keeps signatures simple,
      but forces branching on `Method` or relies on constant propagation to
      eliminate dead code. It does not compose with simulation drivers like
      `ClosedLoop` without passing sub-step counts and enum tags down every call
      stack.
    - *Decision*: Rejected in favor of zero-cost traits `DynamicSystem` and
      `SystemIntegrator`.
- **Dynamic heap vectors (`Vec<T>`) for state and stages**:
    - *Tradeoff*: Simplifies trait definitions by avoiding associated types or
      generics, but introduces dynamic allocation at every stage evaluation and
      step rejection.
    - *Decision*: Rejected due to C-1 (`no_std` / `no_alloc`) and NFR-1.
- **Fixed const-generic state array `[T; N]` on `DynamicSystem`**:
    - *Tradeoff*: Avoids associated types and `VectorSpace`, but prevents using
      matrix containers (`Owned<T, N, 1>`), complex structures, or SIMD
      representations.
    - *Decision*: Rejected in favor of container independence (`type State`,
      `type Output`) as established by Ahnert and Mulansky (2011) and Renevey (
      2026).
- **Separate trait hierarchy vs unified trait for adaptive stepping**:
    - *Tradeoff*: Forcing all integrators into an adaptive interface requires
      non-embedded tableaus (Euler, Midpoint, RK4) to synthesize dummy error
      estimates or return errors.
    - *Decision*: Sub-trait `EmbeddedIntegrator: SystemIntegrator` cleanly
      separates fixed-only tableaus from embedded pairs that provide error
      estimates (DOPRI5).

---

### 6. Verification & Validation

#### 6.1 Approach

The implementation must produce evidence that each tableau converges at its
stated order, that sub-step refinement is exact rather than approximate, that
adaptive control responds monotonically to tolerance and terminates, and that
the whole module runs allocation-free on target.

| Method | Mechanism |
|:-------|:----------|
| Requirements-based test | `#[test]` unit tests over the linear scalar and nonlinear test systems |
| Back-to-back comparison | Host oracle harness (`documentation/vv/oracle-harness-design.md`) against SciPy `solve_ivp` for `DormandPrince54` |
| Metamorphic relation | `#[test]` over step-halving and sub-step decomposition |
| Property-based test | `proptest` over equilibrium states and generated step sizes |
| Compile-time shape check | Const-generic state and output dimensions; `compile_fail` doctest |
| Static analysis | `cargo clippy-ci`, source inspection for heap use |
| Resource usage evaluation | Stack analysis and disassembly audit under `cargo ci` |
| On-target execution | ETS suites under QEMU (`thumbv7em`, `riscv32imac`) |
| Coverage measurement | `cargo coverage` |

Target: 90% line coverage of `src/integrators` once solvers exist, measured
with `cargo coverage`. Excluded: tableau coefficient tables (data) and
`Debug` / `Display`. The stub module has no executable surface to cover.

No validation plant consumes this module. The DC-motor and buck-converter
suites still integrate with inlined RK4. Fitness for purpose is judged only
after those suites call this module's advance.

#### 6.2 Acceptance

| Claim | Oracle | Measure | Bound |
|:------|:-------|:--------|:------|
| Order of accuracy, `Euler` | Closed-form $x(t) = e^{\lambda t}$ | Ratio of global errors at $h$ and $h/2$ | $2.0 \pm 0.2$ |
| Order of accuracy, `Midpoint` | Closed-form $x(t) = e^{\lambda t}$ | Ratio of global errors at $h$ and $h/2$ | $4.0 \pm 0.4$ |
| Order of accuracy, `Rk4` | Closed-form $x(t) = e^{\lambda t}$ | Ratio of global errors at $h$ and $h/2$ | $16.0 \pm 1.6$ |
| Order of accuracy, `DormandPrince54` | Closed-form $x(t) = e^{\lambda t}$ | Ratio of global errors at $h$ and $h/2$ | $32.0 \pm 3.2$ |
| Sub-step equivalence | `step` applied $M$ times at $dt/M$ | Absolute difference | $0$, bit-identical |
| Equilibrium invariance | $\dot{x} = 0$ | Absolute difference $\|x_{n+1} - x_n\|_\infty$ | $0$, bit-identical |
| Adaptive tolerance response | Global error at $\text{rtol} = 10^{-3}$ | Monotone decrease as rtol tightens to $10^{-6}$ | Strictly decreasing |
| Adaptive termination | Divergent stiff plant | Returned variant | `Err(IntegratorError::MaxAttemptsExceeded)` |

Order-ratio tolerances are $\pm 10\%$ of the theoretical ratio, which
accommodates the asymptotic regime not being reached exactly at finite $h$
(Hairer et al., 1993).

#### 6.3 Limits

- **FR-1..FR-7, NFR-1, NFR-2**: `src/integrators/mod.rs` contains no tableaus
  or tests. Planned 6.2 order-ratio bounds are the contract once solvers land.
- Stiff-system behaviour. C-2 restricts the module to explicit schemes.
- Dense output and interpolation between steps.
- Long-horizon energy drift for symplectic problems.
- Measured worst-case execution time on hardware.

### 7. Performance & Resource Considerations

- **Stack Storage**: Explicit Runge-Kutta stages are stored as local variables.
  Stack footprint per step is bounded:
    - `Euler`: 1 stage.
    - `Midpoint`: 2 stages.
    - `Rk4`: 4 stages + 1 working buffer.
    - `DormandPrince54`: 7 stages + 1 error vector + 1 working buffer.
- **Execution Time Determinism**: The `max_attempts` bound in
  `AdaptiveStepSettings` guarantees that adaptive step loops terminate within a
  strict instruction budget, preventing unbounded latency in embedded interrupt
  service routines.
- **Inlining & Monomorphization**: Static dispatch over `Sys: DynamicSystem`
  allows LLVM to inline derivative evaluations directly into the tableau
  multiply-accumulate loop, eliminating function call overhead.

---

### 8. Risks & Open Questions

- **Stiff Plant Dynamics**: Explicit adaptive Runge-Kutta methods will rapidly
  reduce step size toward `min_step` when encountering stiff dynamics,
  potentially triggering `MaxAttemptsExceeded`. Clear error reporting guides
  users to plant modeling or fixed-step alternatives.
- **VectorSpace Trait Bounds**: Generalizing `VectorSpace` for arbitrary user
  types requires boilerplate unless implemented for standard types (`[T; N]`,
  `Owned<T, R, C>`). Blanket implementations for core array types are mandatory.
- **Autonomous vs. Non-Autonomous Derivatives**: While $t$ is passed to
  `derivative(t, x, u)`, many physical models are autonomous (time-invariant). A
  helper adapter `AutonomousSystem` or closure wrapper will allow callers to
  omit $t$ when defining time-invariant plants.

---

### 9. Development Plan

| Phase                                    | Description                                                                                                                                      | Estimated Effort (1-10) |
|:-----------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------|:------------------------|
| Phase 1: Core Traits & Settings          | Define `DynamicSystem`, `VectorSpace`, `FixedStepSettings`, `AdaptiveStepSettings`, `IntegratorError`, and blanket implementations for `[T; N]`. | 3                       |
| Phase 2: Fixed-Step Tableaus             | Implement `SystemIntegrator` for `Euler`, `Midpoint`, `Rk4`, and fixed-step `DormandPrince54`, with order-of-accuracy tests.                     | 4                       |
| Phase 3: Adaptive Step & Embedded DOPRI5 | Implement `EmbeddedIntegrator` and `advance_adaptive` for `DormandPrince54`, with tolerance compliance and max-attempt unit tests.               | 4                       |
| Phase 4: StateSpace & Closure Bridges    | Implement `DynamicSystem` for `GenericStateSpace` and closure wrappers.                                                                          | 2                       |
| Phase 5: Example Adoptions               | Port the `dc_motor` and `buck_converter` suites in `control-rs-validation` to the new traits; verify existing simulation tests pass. Pedagogical demos live under `examples/`. | 3                       |

---

### 10. Revision History

| Date       | Author          | Change                                                                                                                                                                             |
|:-----------|:----------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2026-09-08 | @mitchelldscott | Initial draft from library-extraction review.                                                                                                                                      |
| 2026-09-08 | @mitchelldscott | Restructure around `DynamicSystem` and `SystemIntegrator` traits, container-independent state and output representations, Dormand-Prince 5(4), and packaged settings constructors. |
| 2026-09-08 | @mitchelldscott | Incorporate `AdaptiveStepSettings`, `EmbeddedIntegrator` trait, local error estimation, and adaptive step size options into `DormandPrince54`.                                     |
| 2026-09-08 | @mitchelldscott | Add `combine_derivatives` to `VectorSpace`, bound `Sys::State: VectorSpace<T>` on `EmbeddedIntegrator`, and adopt native `core::error::Error` for `IntegratorError`.               |
| 2026-09-09 | @mitchelldscott | Hardening pass: updated badge to brightgreen, reshaped FR-2/4/5/6 from trait slogans to need-named requirements, deduplicated NFR-1 vs C-1, and normalized reference numbering.  |
| 2026-09-15 | @MitchellDScott | Retarget plant oracle locators to `control-rs-validation/src/dc_motor/`. |
| 2026-09-15 | @MitchellDScott | §1 jobs; FR-2 without scheme names; stub module listed in 6.7; no module-path inspection as behavioral evidence. |
| 2026-09-16 | @MitchellDScott | Retired `vv-standards.md`: §6 authoring rules are `design-template.md` §6. |

---

## References

[1] control-rs authors, "control-rs: GenericStateSpace continuous-time
derivative
API," in *control-rs*, 2026. [Online].
Available: https://github.com/Dyse-Industries/control-rs. Accessed: Sep. 08,
2026.

[2] control-rs authors, "control-rs: Package Manifest and Workspace Lints,"
*control-rs*, 2026. [Online].
Available: https://github.com/Dyse-Industries/control-rs. Accessed: Sep. 08,
2026.

[3] J. C. Butcher, *Numerical Methods for Ordinary Differential Equations*, 3rd
ed. Chichester, UK: John Wiley & Sons, 2016, p. 93. doi: 10.1002/9781119121534.

[4] J. R. Dormand and P. J. Prince, "A family of embedded Runge-Kutta formulae,"
*J. Comput. Appl. Math.*, vol. 6, no. 1, pp. 19–26, 1980, doi:
10.1016/0771-050X(80)90013-3.

[5] E. Hairer, S. P. Nørsett, and G. Wanner, *Solving Ordinary Differential
Equations I: Nonstiff Problems*, 2nd ed. Berlin, Heidelberg: Springer-Verlag,
1993, p. 178. doi: 10.1007/978-3-662-12607-3.

[6] K. Ahnert and M. Mulansky, "Odeint -- Solving Ordinary Differential
Equations in C++," in *Proc. AIP Conf.*, vol. 1389, 2011, pp. 1586–1589. doi:
10.1063/1.3637934.

[7] C. Rackauckas and Q. Nie, "DifferentialEquations.jl -- A Performant and
Feature-Rich Ecosystem for Solving Differential Equations in Julia," *J. Open
Res. Softw.*, vol. 5, no. 1, p. 15, 2017, doi: 10.5334/jors.151.

[8] M. Robinson and A. Allmont, "diffsol: Rust crate for solving differential
equations," *J. Open Source Softw.*, vol. 11, no. 117, p. 9384, 2026, doi:
10.21105/joss.09384.

[9] S. Renevey, "ode-solvers: Numerical methods to solve ordinary differential
equations in Rust," in *ode-solvers*, 2026. [Online].
Available: https://github.com/srenevey/ode-solvers. Accessed: Sep. 08, 2026.
