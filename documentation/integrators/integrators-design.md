# Numerical Integrators & Mathematical Function Traits Redesign (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_26,_2026-blue)
![Status: Draft](https://img.shields.io/badge/status-draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

Autonomous flight control, embedded robotics, neural dynamical systems, and scientific computing require high-fidelity numerical simulation and time-stepping of continuous-time physics (Betts, 2010; Chen et al., 2018; Rackauckas and Nie, 2017). Currently, `control-rs` implements continuous-time state derivatives $\dot{x}(t) = Ax(t) + Bu(t)$ and Zero-Order Hold (ZOH) matrix exponential discretization on [`StateSpaceCore`](../numerical-models/state-space-design.md), but lacks a general, composable numerical integration subsystem across linear, nonlinear, Hamiltonian, and differential-algebraic systems.

This document establishes the architecture for the `control-rs::integrators` module and modernizes the mathematical mapping traits (`MathFunction`, `MathFunctionInto`, and `InvertibleFunction`) in `control-rs::math`. The subsystem satisfies five primary usage scenarios:

1. **Embedded Real-Time State Stepping**: Fixed-step explicit integration (Euler, Heun, Ralston, RK4) for continuous state-space models, transfer functions, and nonlinear robot kinematics executing under strict real-time deadlines (Verschueren et al., 2022).
2. **Structure-Preserving Hamiltonian Dynamics**: Symplectic and geometric integration (Symplectic Euler, Störmer–Verlet, Ruth 3rd-order, Yoshida 4th/6th-order) for mechanical manipulators, orbital mechanics, and Hamiltonian Neural Networks requiring exact energy conservation and phase-space volume preservation over long time horizons (Greydanus et al., 2019; Hairer et al., 2006; Ruth, 1983; Yoshida, 1990).
3. **Adaptive Trajectory Propagation & Neural ODEs**: High-efficiency adaptive embedded pairs (Tsitouras 5(4), Dormand–Prince 5(4), Bogacki–Shampine 3(2)) with First Same As Last (FSAL) stage reuse, Proportional-Integral (PI) step-size adaptation, continuous 4th-order polynomial dense output interpolation, and adjoint sensitivity propagation for continuous-depth neural models (Chen et al., 2018; Dormand and Prince, 1980; Kidger, 2022; Tsitouras, 2011).
4. **Stiff Systems & Implicit Optimal Control Collocation**: Implicit Runge–Kutta (IRK) methods, Gauss–Legendre collocation, Radau IIA, and Singly Diagonally Implicit Runge–Kutta (SDIRK) solvers for stiff physical networks, index-1 Differential-Algebraic Equations (DAEs), and direct trajectory optimization (Betts, 2010; Hairer and Wanner, 1996; Hindmarsh et al., 2005).
5. **Decoupled Model Transformation & Invertible Functions**: Standardized mathematical evaluation of dynamical models via modernized `MathFunction`, `MathFunctionInto`, and `InvertibleFunction` traits, decoupling physical storage and coordinate systems from integrator algorithms.

---

### 2. Requirements

#### 2.1 Functional Requirements

- **FR-1 — Modernized In-Place and Value Mathematical Functions**: Provide `MathFunction<Domain, Codomain>` and `MathFunctionInto<Domain, Codomain>` traits that allow mathematical models to evaluate outputs either by value ($y = f(x)$) or by writing into caller-provided mutable buffers ($f(x, \&\text{mut } y)$) without heap allocation.
- **FR-2 — Invertible Mathematical Functions**: Provide `InvertibleFunction<A, B>: MathFunction<A, B> + MathFunction<B, A>` and `InvertibleFunctionInto<A, B>: MathFunctionInto<A, B> + MathFunctionInto<B, A>` with forward $y = f(x)$ and inverse $x = f^{-1}(y)$ evaluation for coordinate transformations, state similarity changes, and Hamiltonian canonical momentum transforms.
- **FR-3 — Explicit Fixed-Step Runge–Kutta Steppers**: Provide explicit Runge–Kutta integrators of orders 1 through 4 (Forward Euler, Heun, Ralston, Classical RK4) parameterized by compile-time Butcher tableaus $(A, b, c)$.
- **FR-4 — Structure-Preserving Symplectic Integrators**: Provide explicit, time-reversible symplectic integrators (Symplectic Euler, Störmer–Verlet, Ruth 3rd-order, Yoshida 4th and 6th-order composition) for separable Hamiltonian systems $\dot{q} = \partial H / \partial p, \dot{p} = -\partial H / \partial q$.
- **FR-5 — Adaptive Embedded Pairs with FSAL and Dense Output**: Provide embedded adaptive Runge–Kutta pairs (Tsitouras 5(4), Dormand–Prince 5(4), Bogacki–Shampine 3(2)) supporting FSAL stage reuse, local truncation error estimation, PI step-size control, and continuous polynomial dense output.
- **FR-6 — Implicit Collocation and Stiff DAE Solvers**: Provide implicit Runge–Kutta schemes (Implicit Euler, Crank–Nicolson, Gauss–Legendre 4th-order, Radau IIA 5th-order, SDIRK) with Newton iteration for stiff differential equations and index-1 differential-algebraic equations.
- **FR-7 — Exact Continuous-to-Discrete LTI Solvers**: Provide exact Matrix Exponential Zero-Order Hold (ZOH) and Bilinear/Tustin transforms for linear continuous state-space models.

#### 2.2 Non-Functional Requirements

- **NFR-1 — Zero Dynamic Memory Allocation**: All integrator algorithms, internal stage vectors, Butcher coefficient tables, and interpolants must operate exclusively within stack memory or caller-provided static storage (`no_alloc`).
- **NFR-2 — Compile-Time Static Shape Checking**: State, input, output, and stage dimensions must be checked and constrained at compile time using the crate's `Dim` and const-generic type system.
- **NFR-3 — Deterministic Execution**: Fixed-step explicit and symplectic integration routines must exhibit bounded, deterministic execution timing without data-dependent heap allocations or variable loop bounds.

#### 2.3 Constraints

- **C-1 — Native Rust Authoring**: Integrators must be authored natively in Rust without external host-to-target C code generation (inheriting C-1/C-2 from [`controls-tools-design.md`](../control-toolboxes/controls-tools-design.md)).
- **C-2 — Storage Subsystem Interoperability**: Integrators must accept state and stage vectors adhering to the crate's `Storage` and `DenseStorage` contracts (`Owned`, `StorageView`, `StorageViewMut`).
- **C-3 — Zero Dynamic Dispatch Overhead**: Stepper and model combinations must fully monomorphize without requiring dynamic trait object dispatch (`dyn`) on inner integration loops.

---

### 3. Technical Overview

Numerical integration approximates the solution to an initial value problem (IVP):
$$\dot{x}(t) = f(t, x(t), u(t)), \quad x(t_0) = x_0$$

#### 3.1 Runge–Kutta Methods and Butcher Tableaus
An $s$-stage Runge–Kutta method is defined by its Butcher tableau (Butcher, 2016):
$$\begin{array}{c|c} c & A \\ \hline & b^T \end{array} \quad \implies \quad k_i = f\left(t_n + c_i h, \, x_n + h \sum_{j=1}^s a_{ij} k_j, \, u(t_n + c_i h)\right), \quad x_{n+1} = x_n + h \sum_{i=1}^s b_i k_i$$
For explicit methods (ERK), $a_{ij} = 0$ for $j \ge i$, allowing sequential evaluation of stages $k_1, \dots, k_s$ (Butcher, 2016; Hairer et al., 1993). For diagonally implicit methods (DIRK/SDIRK), fully implicit methods (IRK, Gauss–Legendre, Radau IIA), and Additive Runge–Kutta (ARK/IMEX) methods, stage evaluations require solving coupled algebraic systems, yielding A-stable and L-stable methods for stiff ODEs and DAEs (Hairer and Wanner, 1996; Kennedy and Carpenter, 2003; Reynolds et al., 2023).

#### 3.2 Adaptive Embedded Pairs & FSAL
Adaptive integrators evaluate two approximations of orders $p$ and $\hat{p} = p-1$ using an embedded weight vector $\hat{b}$ (Dormand and Prince, 1980):
$$\hat{x}_{n+1} = x_n + h \sum_{i=1}^s \hat{b}_i k_i, \quad e_{n+1} = \|x_{n+1} - \hat{x}_{n+1}\|$$
By enforcing the First Same As Last (FSAL) property ($c_s = 1, a_{sj} = b_j$), stage $k_s$ of step $n$ becomes stage $k_1$ of step $n+1$, reducing function evaluations from 7 to 6 in order 5(4) pairs (Dormand and Prince, 1980; Tsitouras, 2011). The step size is updated via Proportional-Integral (PI) control (Hairer et al., 1993; Rackauckas and Nie, 2017):
$$h_{new} = h_n \cdot \min\left(\text{fac}_{max}, \, \max\left(\text{fac}_{min}, \, \text{fac} \cdot \left(\frac{\text{tol}}{e_{n+1}}\right)^{\alpha} \left(\frac{e_n}{e_{n+1}}\right)^\beta\right)\right)$$

#### 3.3 Symplectic Geometric Integrators
For separable Hamiltonian systems $H(q, p) = T(p) + V(q)$ where $\dot{q} = \nabla_p T(p)$ and $\dot{p} = -\nabla_q V(q)$, standard Runge–Kutta methods introduce artificial energy dissipation or growth (Hairer et al., 2006; Ruth, 1983). Symplectic maps preserve the differential 2-form $dp \wedge dq$, bounding energy error within an $O(h^p)$ strip over exponentially long integration intervals (Hairer et al., 2006). Higher-order symplectic integrators are constructed via symmetric composition of 2nd-order stages (Yoshida, 1990):
$$S_{2n}(h) = S_{2n-2}(w_1 h) \circ S_{2n-2}(w_0 h) \circ S_{2n-2}(w_1 h), \quad w_0 = \frac{-2^{1/(2n-1)}}{2 - 2^{1/(2n-1)}}, \quad w_1 = \frac{1}{2 - 2^{1/(2n-1)}}$$

```
                ┌─────────────────────────────────────────────────────────┐
                │   MathFunction / MathFunctionInto / InvertibleFunction  │
                │        (Generic Discrete Mathematical Function)         │
                └────────────────────────────┬────────────────────────────┘
                                             │ implements
               ┌─────────────────────────────┴─────────────────────────────┐
               │                                                           │
┌──────────────▼──────────────┐                             ┌──────────────▼──────────────┐
│       SystemDynamics        │                             │     HamiltonianDynamics     │
│   (Continuous Vector Field) │                             │  (Separable Energy Fields)  │
│    dot(x) = f(t, x, u)      │                             │ dot(q)=T'(p), dot(p)=-V'(q) │
└──────────────┬──────────────┘                             └──────────────┬──────────────┘
               │                                                           │
               │ integrated by                                             │ integrated by
┌──────────────▼───────────────────────────┐                ┌──────────────▼──────────────┐
│            Stepper / AdaptiveStepper     │                │      SymplecticStepper      │
│  - Explicit: Euler, Heun, Ralston, RK4   │                │  - Symplectic Euler (1st)   │
│  - Adaptive: Tsit5, DP54, BS32 (FSAL)    │                │  - Störmer-Verlet (2nd)     │
│  - Implicit: Radau IIA, GL4, SDIRK       │                │  - Ruth (3rd), Yoshida (4th)│
└──────────────────────────────────────────┘                └─────────────────────────────┘
```

---

### 4. Architecture

#### 4.1 Modernized Mathematical Function Trait Hierarchy (`control-rs::math`)

**Proposal (not in evidence)**: Implement `MathFunction`, `MathFunctionInto`, `InvertibleFunction`, and `InvertibleFunctionInto` in `src/math/mod.rs` to clearly express discrete mathematical functions $f(x) = y$ and bijections $f: A \leftrightarrow B$, ensuring that any `InvertibleFunction<A, B>` explicitly implements both forward `MathFunction<A, B>` and inverse `MathFunction<B, A>` mappings:

```rust
/// Pure discrete mathematical function f: Domain -> Codomain.
pub trait MathFunction<Domain, Codomain> {
    /// Evaluates the mathematical function out-of-place: y = f(x).
    fn evaluate(&self, x: Domain) -> Codomain;
}

/// In-place discrete mathematical function f: &Domain -> &mut Codomain.
pub trait MathFunctionInto<Domain, Codomain> {
    /// Evaluates the mathematical function into a caller-provided destination buffer: f(x, &mut y).
    fn evaluate_into(&self, x: &Domain, y: &mut Codomain);
}

/// Invertible mathematical bijection f: A <-> B with forward and inverse mappings.
///
/// An invertible function implements both `MathFunction<A, B>` (forward mapping A -> B)
/// and `MathFunction<B, A>` (inverse mapping B -> A).
pub trait InvertibleFunction<A, B>: MathFunction<A, B> + MathFunction<B, A> {
    /// Evaluates the forward mapping y = f(x).
    #[inline]
    fn evaluate_forward(&self, x: A) -> B {
        <Self as MathFunction<A, B>>::evaluate(self, x)
    }

    /// Evaluates the inverse mapping x = f^{-1}(y).
    #[inline]
    fn evaluate_inverse(&self, y: B) -> A {
        <Self as MathFunction<B, A>>::evaluate(self, y)
    }
}

/// In-place invertible mathematical bijection.
///
/// Implements both `MathFunctionInto<A, B>` (forward in-place)
/// and `MathFunctionInto<B, A>` (inverse in-place).
pub trait InvertibleFunctionInto<A, B>: MathFunctionInto<A, B> + MathFunctionInto<B, A> {
    /// Evaluates the forward mapping into destination buffer `y`.
    #[inline]
    fn evaluate_forward_into(&self, x: &A, y: &mut B) {
        <Self as MathFunctionInto<A, B>>::evaluate_into(self, x, y);
    }

    /// Evaluates the inverse mapping into destination buffer `x`.
    #[inline]
    fn evaluate_inverse_into(&self, y: &B, x: &mut A) {
        <Self as MathFunctionInto<B, A>>::evaluate_into(self, y, x);
    }
}
```

#### 4.2 System Dynamics Trait Abstraction

**Proposal (not in evidence)**: Define `SystemDynamics` and `HamiltonianDynamics` as standard domain wrappers over `MathFunctionInto`:

```rust
/// Standard continuous-time dynamical system dx/dt = f(t, x, u).
pub trait SystemDynamics<T, const NX: usize, const NU: usize> {
    /// Computes state derivative dx/dt into `x_dot` given current time `t`, state `x`, and input `u`.
    fn evaluate_derivative(
        &self,
        t: T,
        x: &Owned<T, NX, 1>,
        u: &Owned<T, NU, 1>,
        x_dot: &mut Owned<T, NX, 1>,
    );
}

/// Separable Hamiltonian dynamical system with generalized coordinates q and momenta p.
pub trait HamiltonianDynamics<T, const NQ: usize> {
    /// Computes velocity dq/dt = dT/dp into `q_dot`.
    fn velocity(&self, p: &Owned<T, NQ, 1>, q_dot: &mut Owned<T, NQ, 1>);
    /// Computes generalized force dp/dt = -dV/dq into `p_dot`.
    fn force(&self, q: &Owned<T, NQ, 1>, p_dot: &mut Owned<T, NQ, 1>);
}
```

#### 4.3 Model Implementations of Function & Dynamics Traits

1. **`StateSpaceCore`**: Implements `SystemDynamics<T, NX, NU>` via its linear $Ax + Bu$ formulation, `MathFunction<(&Owned<T, NX, 1>, &Owned<T, NU, 1>), Owned<T, NX, 1>>`, and `MathFunctionInto<(&Owned<T, NX, 1>, &Owned<T, NU, 1>), Owned<T, NX, 1>>` ([`src/state_space/mod.rs`](../../src/state_space/mod.rs)).
2. **`TransferFunctionCore`**: Implements `MathFunction<Complex<T>, Complex<T>>` for frequency evaluation and converts to observable canonical `StateSpaceCore` for continuous integration.
3. **`MatrixCore`**: Implements `MathFunction<Owned<T, N, 1>, Owned<T, M, 1>>` / `MathFunctionInto<Owned<T, N, 1>, Owned<T, M, 1>>`, and `InvertibleFunction<Owned<T, N, 1>, Owned<T, N, 1>>` / `InvertibleFunctionInto<Owned<T, N, 1>, Owned<T, N, 1>>` when $M = N$ and $\det(A) \neq 0$ (via LU solver).
4. **`PolynomialCore`**: Implements `MathFunction<T, T>` via Horner evaluation.

#### 4.4 Fixed-Step and Symplectic Steppers

**Proposal (not in evidence)**: Implement fixed-step explicit and symplectic steppers parameterizing intermediate stage storage on the stack:

```rust
/// Explicit fixed-step integration algorithm.
pub trait FixedStepper<T, const NX: usize, const NU: usize> {
    /// Steps state `x` forward by interval `dt` in place.
    fn step<Sys: SystemDynamics<T, NX, NU>>(
        &self,
        sys: &Sys,
        t: T,
        x: &mut Owned<T, NX, 1>,
        u: &Owned<T, NU, 1>,
        dt: T,
    );
}

/// Symplectic geometric integration algorithm for Hamiltonian systems.
pub trait SymplecticStepper<T, const NQ: usize> {
    /// Steps coordinates `q` and momenta `p` forward by interval `dt` in place.
    fn step<Sys: HamiltonianDynamics<T, NQ>>(
        &self,
        sys: &Sys,
        q: &mut Owned<T, NQ, 1>,
        p: &mut Owned<T, NQ, 1>,
        dt: T,
    );
}
```

The concrete fixed steppers provided under `src/integrators/explicit/` and `src/integrators/symplectic/` are:
- `Euler`: 1st-order explicit Euler.
- `Heun`: 2nd-order explicit Runge–Kutta (trapezoidal).
- `Ralston`: 2nd-order Runge–Kutta minimizing local truncation error bound.
- `RungeKutta4`: 4th-order classical Runge–Kutta.
- `SymplecticEuler`: 1st-order semi-implicit symplectic map ($p_{n+1} = p_n + h F(q_n), q_{n+1} = q_n + h V(p_{n+1})$) (Hairer et al., 2006).
- `StormerVerlet`: 2nd-order Velocity-Verlet symplectic map (Hairer et al., 2006).
- `Ruth3`: 3rd-order 3-stage explicit canonical symplectic integrator (Ruth, 1983).
- `Yoshida4`: 4th-order symmetric composition symplectic integrator (Yoshida, 1990).
- `Yoshida6`: 6th-order symmetric composition symplectic integrator (Yoshida, 1990).

#### 4.5 Adaptive Embedded Runge–Kutta Pairs & Dense Output

**Proposal (not in evidence)**: Implement adaptive embedded pairs with internal stage array buffers:

```rust
/// Embedded Runge-Kutta pair with error estimation and dense output.
pub struct AdaptiveRungeKutta<T, const NX: usize, const STAGES: usize, const ORDER: usize> {
    a: [[T; STAGES]; STAGES],
    b: [T; STAGES],
    b_hat: [T; STAGES],
    c: [T; STAGES],
    d: [[T; STAGES]; 4], // Dense output polynomial coefficients
}
```

Concrete embedded pairs:
- `Tsitouras54`: Tsitouras 5(4) pair with 7 stages (FSAL $\to$ 6 effective evaluations) and continuous 4th-order dense interpolant (Tsitouras, 2011).
- `DormandPrince54`: Dormand–Prince 5(4) pair (Dopri5) with FSAL error estimation (Dormand and Prince, 1980; Hairer et al., 1993).
- `BogackiShampine32`: Bogacki–Shampine 3(2) pair (BS3) for lower-tolerance, low-cost stepping (Hairer et al., 1993).

#### 4.6 File Structure and Impact

```
src/
├── lib.rs                       # Registers `pub mod integrators;`
├── math/
│   └── mod.rs                   # Modernized `MathFunction`, `MathFunctionInto`, `InvertibleFunction`, `InvertibleFunctionInto`
└── integrators/
    ├── mod.rs                   # Re-exports steppers, traits, and error types
    ├── traits.rs                # `SystemDynamics`, `HamiltonianDynamics`, `FixedStepper`, `AdaptiveStepper`
    ├── explicit.rs              # `Euler`, `Heun`, `Ralston`, `RungeKutta4`
    ├── symplectic.rs            # `SymplecticEuler`, `StormerVerlet`, `Ruth3`, `Yoshida4`, `Yoshida6`
    ├── adaptive.rs              # `AdaptiveRungeKutta`, `Tsitouras54`, `DormandPrince54`, `BogackiShampine32`
    ├── dense.rs                 # `DenseOutput` polynomial interpolant
    ├── implicit.rs              # `ImplicitEuler`, `CrankNicolson`, `GaussLegendre4`, `RadauIIA5`, `Sdirk`
    └── tests/                   # Target and host integration tests
```

---

### 5. Alternatives Considered

#### 5.1 Dynamic Heap-Allocated Vector Solvers vs Statically Sized Storage
- **Alternative**: Adopting dynamic vectors (like `nalgebra::DVector` or `Vec<T>` as in `ode_solvers` and `peroxide`) (Kim, 2024; Renevey, 2024).
- **Tradeoff & Decision**: Rejected. `control-rs` enforces `no_std` and `no_alloc` constraints on embedded microcontrollers (C-1, C-3, NFR-1). Dynamic allocation introduces non-deterministic allocation latency and heap fragmentation. Static sizing with `Dim` verifies stage buffer compatibility at compile time.

#### 5.2 Dynamic Trait Objects (`&dyn SystemDynamics`) vs Generic Monomorphization
- **Alternative**: Accepting `&dyn SystemDynamics` to allow heterogeneous runtime model swapping.
- **Tradeoff & Decision**: Rejected for hot integration loops. Virtual method calls prevent compiler inlining and SIMD auto-vectorization. Monomorphization via generic type parameters (`Sys: SystemDynamics<T, NX, NU>`) allows full inlining of $Ax + Bu$ operations.

#### 5.3 Coupled vs Separable Hamiltonian Coordinates
- **Alternative**: Formulating all Hamiltonian systems as general state vectors $x = [q; p]^T$ using standard Runge–Kutta methods.
- **Tradeoff & Decision**: Rejected. General RK methods are not symplectic for non-linear Hamiltonians and cause steady energy drift. Partitioned $q$ and $p$ representations allow explicit, symplectic evaluation without matrix inversion (Hairer et al., 2006; Ruth, 1983).

---

### 6. Verification & Validation Plan

#### 6.1 Unit Verification Suite
- **Convergence Rate Verification**: Verify theoretical order of convergence $O(h^p)$ by stepping linear test ODE $\dot{x} = -\lambda x$ over varying step sizes $h \in [10^{-1}, 10^{-4}]$ and asserting error slope matches method order:
  - 1st-order: `Euler`, `SymplecticEuler` ($\text{slope} \approx 1.0$).
  - 2nd-order: `Heun`, `Ralston`, `StormerVerlet` ($\text{slope} \approx 2.0$).
  - 3rd-order: `Ruth3`, `BogackiShampine32` ($\text{slope} \approx 3.0$).
  - 4th-order: `RungeKutta4`, `Yoshida4`, `GaussLegendre4` ($\text{slope} \approx 4.0$).
  - 5th-order: `Tsitouras54`, `DormandPrince54`, `RadauIIA5` ($\text{slope} \approx 5.0$).
  - 6th-order: `Yoshida6` ($\text{slope} \approx 6.0$).
- **Symplectic Invariant Test**: Simulate 2D Kepler two-body orbit ($H(q, p) = \frac{1}{2}\|p\|^2 - \frac{1}{\|q\|}$) over $10^5$ cycles. Assert symplectic integrators maintain energy error $|\Delta H / H_0| < 10^{-6}$ without secular drift, whereas classical RK4 exhibits monotonic orbital decay (Hairer et al., 2006).
- **Dense Output Continuity**: Assert that `Tsitouras54` dense output matches 4th-order accuracy at arbitrary points $\theta \in (0, 1)$ between grid points.
- **Stiff Van der Pol Oscillator**: Verify `RadauIIA5` and `Sdirk` successfully integrate the stiff Van der Pol equation ($\mu = 1000$) where explicit methods fail due to step-size collapse (Hairer and Wanner, 1996).

#### 6.2 Validation & Model Integration
- **State-Space Step Validation**: Connect `StateSpaceCore` to `RungeKutta4` and `Tsitouras54` for step-response simulation; compare against analytical matrix exponential $x(t) = e^{At} x_0 + \int_0^t e^{A(t-\tau)} B u d\tau$.
- **Function Trait Round-Trip**: Assert invertible mathematical functions satisfy $f^{-1}(f(x)) \equiv x$ to within working precision for matrix transformations and coordinate rotations.

---

### 7. Performance & Resource Considerations

- **Stack Allocation Footprint**: An $s$-stage Runge–Kutta method for an $N_x$-state system requires $s \times N_x$ floats of temporary stage scratch space. For $N_x = 16$ (e.g., full aircraft dynamics) and $s = 7$ (`Tsitouras54`), scratch space is $7 \times 16 \times 8 = 896$ bytes, easily accommodated on Cortex-M4/M7 stack budgets (16–64 KB).
- **FSAL Optimization**: Reusing stage $k_s$ as stage $k_1$ of the subsequent step saves 14.3% of vector field evaluations in continuous trajectory simulation (Dormand and Prince, 1980; Tsitouras, 2011).

---

### 8. Risks & Open Questions

- **Open Question (Research Query 1)**: Unified cross-domain method under the literal title was not found in published literature; the domain taxonomy combines building blocks from controls (acados, direct collocation), ML (Neural ODEs, HNNs), and scientific computing (SciML, Hairer).
- **Proposal (not in evidence) 1**: Introduction of `MathFunctionInto` and `InvertibleFunction` (with dual `MathFunction<A, B> + MathFunction<B, A>` bound) in `control-rs::math` to handle in-place mutation and bidirectional bijections without heap allocation.
- **Proposal (not in evidence) 2**: Decoupled `SystemDynamics` and `HamiltonianDynamics` trait definitions parameterized over const generic dimensions `NX`, `NU`, `NQ`.
- **Proposal (not in evidence) 3**: Generic `AdaptiveRungeKutta` struct using const-generic stage matrices for compile-time tableau inlining.
- **Proposal (not in evidence) 4**: Dense output interpolant design using 4-coefficient polynomial evaluation per stage.

---

### 9. Development Plan

| Phase | Description | Estimated Effort (1-10) |
|:------|:------------|:------------------------|
| **Phase 1: Math Function Trait Redesign** | Implement `MathFunction`, `MathFunctionInto`, `InvertibleFunction`, and `InvertibleFunctionInto` in `src/math/`, and wire implementations for `MatrixCore`, `StateSpaceCore`, `TransferFunctionCore`, and `PolynomialCore`. | 3 |
| **Phase 2: Dynamics Traits & Explicit Steppers** | Implement `SystemDynamics`, `HamiltonianDynamics`, and fixed-step explicit integrators (`Euler`, `Heun`, `Ralston`, `RungeKutta4`). | 4 |
| **Phase 3: Symplectic Geometric Integrators** | Implement partitioned symplectic integrators (`SymplecticEuler`, `StormerVerlet`, `Ruth3`, `Yoshida4`, `Yoshida6`) and Kepler conservation tests. | 4 |
| **Phase 4: Adaptive Embedded Pairs & Dense Output** | Implement `AdaptiveRungeKutta`, `Tsitouras54`, `DormandPrince54`, `BogackiShampine32`, FSAL stage buffer reuse, PI controller, and `DenseOutput` interpolant. | 5 |
| **Phase 5: Implicit & Collocation Solvers** | Implement `ImplicitEuler`, `CrankNicolson`, `GaussLegendre4`, `RadauIIA5`, and `Sdirk` with stack-allocated Newton solvers. | 6 |

---

### 10. Revision History

| Version | Date | Author | Description |
|:--------|:-----|:-------|:------------|
| 1.0 | August 26, 2026 | @MitchellDScott | Initial draft: integrators subsystem across controls, ML, and scientific computing. |
| 1.1 | August 26, 2026 | @MitchellDScott | Renamed mapping traits to `MathFunction`, `MathFunctionInto`, `InvertibleFunction`, and `InvertibleFunctionInto`, ensuring invertible functions implement dual forward and inverse `MathFunction` bounds. |

---

## References

[1] J. T. Betts, *Practical Methods for Optimal Control and Estimation Using Nonlinear Programming*, 2nd ed. Philadelphia, PA: Society for Industrial and Applied Mathematics, 2010.

[2] R. T. Q. Chen, Y. Rubanova, J. Bettencourt, and D. Duvenaud, "Neural Ordinary Differential Equations," in *Advances in Neural Information Processing Systems 31 (NeurIPS 2018)*, 2018, pp. 6571–6583.

[3] C. Rackauckas and Q. Nie, "DifferentialEquations.jl -- A Performant and Feature-Rich Ecosystem for Solving Differential Equations in Julia," *Journal of Open Research Software*, vol. 5, no. 1, p. 15, 2017, doi: 10.5334/jors.151.

[4] R. Verschueren, G. Frison, D. Kouzoupis, J. Frey, N. van Duijkeren, A. Zanelli, B. Novoselnik, T. Albin, R. Quirynen, and M. Diehl, "acados -- a modular open-source framework for fast embedded optimal control," *Mathematical Programming Computation*, vol. 14, no. 1, pp. 147–183, 2022, doi: 10.1007/s12532-021-00208-8.

[5] S. Greydanus, M. Dzamba, and J. Yosinski, "Hamiltonian Neural Networks," in *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*, 2019, pp. 15353–15363.

[6] E. Hairer, C. Lubich, and G. Wanner, *Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations*, 2nd ed. Berlin, Germany: Springer-Verlag, 2006.

[7] R. D. Ruth, "A Canonical Integration Technique," *IEEE Transactions on Nuclear Science*, vol. 30, no. 4, pp. 2669–2671, 1983, doi: 10.1109/TNS.1983.4332919.

[8] H. Yoshida, "Construction of higher order symplectic integrators," *Physics Letters A*, vol. 150, no. 5–7, pp. 262–268, 1990, doi: 10.1016/0375-9601(90)90092-3.

[9] J. R. Dormand and P. J. Prince, "A family of embedded Runge-Kutta formulae," *Journal of Computational and Applied Mathematics*, vol. 6, no. 1, pp. 19–26, 1980, doi: 10.1016/0771-050X(80)90013-3.

[10] P. Kidger, "On Neural Differential Equations," University of Oxford, Oxford, UK, Rep. no. arXiv:2202.02435, 2022.

[11] C. Tsitouras, "Runge--Kutta pairs of order 5(4) satisfying only the first column simplifying assumption," *Computers & Mathematics with Applications*, vol. 62, no. 2, pp. 770–775, 2011, doi: 10.1016/j.camwa.2011.06.002.

[12] E. Hairer and G. Wanner, *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*, 2nd ed. Berlin, Germany: Springer-Verlag, 1996.

[13] A. C. Hindmarsh, P. N. Brown, K. E. Grant, S. L. Lee, R. Serban, D. E. Shumaker, and C. S. Woodward, "SUNDIALS: Suite of Nonlinear and Differential/Algebraic Equation Solvers," *ACM Transactions on Mathematical Software*, vol. 31, no. 3, pp. 363–396, 2005, doi: 10.1145/1089014.1089020.

[14] J. C. Butcher, *Numerical Methods for Ordinary Differential Equations*, 3rd ed. Chichester, UK: John Wiley & Sons, Ltd, 2016.

[15] E. Hairer, S. P. N{\o}rsett, and G. Wanner, *Solving Ordinary Differential Equations I: Nonstiff Problems*, 2nd ed. Berlin, Germany: Springer-Verlag, 1993.

[16] C. A. Kennedy and M. H. Carpenter, "Additive Runge-Kutta schemes for convection-diffusion-reaction equations," *Applied Numerical Mathematics*, vol. 44, no. 1–2, pp. 139–181, 2003, doi: 10.1016/S0168-9274(02)00138-1.

[17] D. R. Reynolds, D. J. Gardner, C. S. Woodward, and R. Chinomona, "ARKODE: A flexible IVP solver infrastructure for one-step methods," *ACM Transactions on Mathematical Software*, vol. 49, no. 2, pp. 19:1–19:30, 2023, doi: 10.1145/3588970.

[18] T.-G. Kim, *peroxide*: Comprehensive numerical computing library for Rust (Version 0.37.0). [Online]. Available: https://docs.rs/peroxide/latest/peroxide/. Accessed: Aug. 26, 2026.

[19] S. Renevey, *ode_solvers*: Numerical methods for solving ordinary differential equations in Rust (Version 0.6.2). [Online]. Available: https://docs.rs/ode_solvers/0.6.2/ode_solvers/. Accessed: Aug. 26, 2026.
