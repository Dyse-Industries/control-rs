# Symplectic Geometric Integrators (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_26,_2026-blue)
![Status: Draft](https://img.shields.io/badge/status-draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

Conservative mechanical systems, multi-link robotic arms, orbital spacecraft dynamics, and physics-informed machine learning architectures (such as Hamiltonian Neural Networks) are governed by Hamiltonian vector fields (Greydanus et al., 2019; Hairer et al., 2006). Conventional explicit Runge–Kutta integrators do not preserve the symplectic 2-form $dp \wedge dq$, causing artificial energy growth or numerical dissipation over extended simulation horizons (Hairer et al., 2006; Ruth, 1983).

This document establishes the architecture, requirement contract, and validation suite for the structure-preserving symplectic integrators in `control-rs::integrators::symplectic`. The module addresses three core usage scenarios:

1. **Long-Horizon Robotic & Orbital Dynamics**: Integration of conservative mechanical and orbital systems where energy must remain bounded without secular drift across millions of integration steps (Hairer et al., 2006; Ruth, 1983).
2. **Hamiltonian Neural Networks & Physics-Informed ML**: Stepping parameterized continuous Hamiltonians $H_\theta(q, p)$ in deep learning pipelines while enforcing exact phase-space volume conservation and time reversibility (Greydanus et al., 2019).
3. **High-Order Symmetric Composition**: 4th- and 6th-order symplectic integration of separable systems via Yoshida composition without implicit matrix solving (Yoshida, 1990).

---

### 2. Requirements

#### 2.1 Functional Requirements

- **FR-1 — Partitioned Hamiltonian Dynamics Interface**: Provide a `HamiltonianDynamics<T, NQ>` trait separating generalized coordinate velocity $\dot{q} = \partial H / \partial p$ and generalized force $\dot{p} = -\partial H / \partial q$ for separable Hamiltonians $H(q, p) = T(p) + V(q)$.
- **FR-2 — First-Order Symplectic Euler Stepper**: Provide `SymplecticEuler` evaluating the semi-implicit map $p_{n+1} = p_n + h F(q_n)$ and $q_{n+1} = q_n + h V(p_{n+1})$.
- **FR-3 — Second-Order Störmer–Verlet Stepper**: Provide `StormerVerlet` (Velocity-Verlet) evaluating half-step momentum updates $p_{n+1/2} = p_n + \frac{h}{2} F(q_n)$, coordinate updates $q_{n+1} = q_n + h V(p_{n+1/2})$, and final momentum updates $p_{n+1} = p_{n+1/2} + \frac{h}{2} F(q_{n+1})$.
- **FR-4 — Third-Order Ruth Canonical Stepper**: Provide `Ruth3`, a 3-stage explicit canonical symplectic integrator preserving Hamiltonian structure with 3rd-order accuracy.
- **FR-5 — Higher-Order Yoshida Composition Steppers**: Provide `Yoshida4` (4th order) and `Yoshida6` (6th order) constructed via symmetric composition of 2nd-order Störmer–Verlet stages using analytical stage weight coefficients.

#### 2.2 Non-Functional Requirements

- **NFR-1 — Zero Dynamic Memory Allocation**: Coordinate vectors $q$ and momentum vectors $p$ must be stepped in place using caller-provided stack buffers (`no_alloc`).
- **NFR-2 — Symplectic Invariant Preservation**: Numerical flows must strictly satisfy the symplectic condition $M^T J M = J$ where $J = \begin{bmatrix} 0 & I \\ -I & 0 \end{bmatrix}$, ensuring bounded Hamiltonian energy oscillations over long horizons.
- **NFR-3 — Compile-Time Dimension Verification**: Generalized coordinate dimensions $N_q$ must be verified at compile time via `Dim`.

#### 2.3 Constraints

- **C-1 — Native Rust Implementation**: Authored natively in Rust without C code generation ([`controls-tools-design.md`](../control-toolboxes/controls-tools-design.md) C-1).
- **C-2 — Storage Decoupling**: Vector storage must use `control-rs::matrix::Owned` and `DenseStorage` ([`matrix-design.md`](../numerical-models/matrix-design.md)).
- **C-3 — Zero Virtual Dispatch**: Steppers must monomorphize over `Sys: HamiltonianDynamics<T, NQ>`.

---

### 3. Technical Overview

For a separable Hamiltonian $H(q, p) = T(p) + V(q)$, the continuous equations of motion are (Hairer et al., 2006; Ruth, 1983):
$$\dot{q} = \nabla_p T(p) = V(p), \quad \dot{p} = -\nabla_q V(q) = F(q)$$

```
Symplectic Euler:
  p_{n+1} = p_n + h·F(q_n)
  q_{n+1} = q_n + h·V(p_{n+1})

Störmer-Verlet (2nd order):
  p_{n+1/2} = p_n + (h/2)·F(q_n)
  q_{n+1}   = q_n + h·V(p_{n+1/2})
  p_{n+1}   = p_{n+1/2} + (h/2)·F(q_{n+1})

Yoshida Composition (2n-th order):
  S_{2n}(h) = S_{2n-2}(w₁h) ∘ S_{2n-2}(w₀h) ∘ S_{2n-2}(w₁h)
  w₀ = -2^{1/(2n-1)} / (2 - 2^{1/(2n-1)}),  w₁ = 1 / (2 - 2^{1/(2n-1)})
```

Through backward error analysis, a symplectic integrator of order $p$ applied to $H(q, p)$ exactly tracks a modified Hamiltonian $\tilde{H}(q, p) = H(q, p) + h^p H_{p+1}(q, p) + \dots$, guaranteeing that energy error $|H(q_n, p_n) - H(q_0, p_0)|$ remains bounded in an $O(h^p)$ band for exponentially long times $t \le e^{c/h}$ (Hairer et al., 2006).

---

### 4. Architecture

#### 4.1 Hamiltonian Dynamics Trait

**Proposal (not in evidence)**: Define `HamiltonianDynamics` for separable Hamiltonian systems:

```rust
/// Separable Hamiltonian dynamics H(q, p) = T(p) + V(q).
pub trait HamiltonianDynamics<T, const NQ: usize> {
    /// Computes generalized velocity dq/dt = dT/dp into `q_dot`.
    fn velocity(&self, p: &Owned<T, NQ, 1>, q_dot: &mut Owned<T, NQ, 1>);
    /// Computes generalized force dp/dt = -dV/dq into `p_dot`.
    fn force(&self, q: &Owned<T, NQ, 1>, p_dot: &mut Owned<T, NQ, 1>);
}
```

#### 4.2 Symplectic Stepper Implementations

**Proposal (not in evidence)**: Implement concrete symplectic steppers:

```rust
/// Semi-implicit 1st-order Symplectic Euler stepper.
pub struct SymplecticEuler;

/// Störmer-Verlet (Velocity-Verlet) 2nd-order symmetric symplectic stepper.
pub struct StormerVerlet;

/// Ruth 3rd-order 3-stage explicit canonical symplectic stepper.
pub struct Ruth3;

/// Yoshida 4th-order 3-stage symmetric composition symplectic stepper.
pub struct Yoshida4;

/// Yoshida 6th-order 7-stage symmetric composition symplectic stepper.
pub struct Yoshida6;
```

---

### 5. Alternatives Considered

- **Standard Runge–Kutta for Hamiltonians**: Using classical RK4 on state vector $x = [q; p]^T$ (Hairer et al., 2006). Rejected: RK4 is not symplectic, leading to secular orbital decay and energy drift in conservative physical models.
- **Implicit Midpoint Rule**: An implicit 2nd-order symplectic method for non-separable Hamiltonians. Rejected for separable systems due to the cost of nonlinear Newton iterations per step.

---

### 6. Verification & Validation Plan

#### 6.1 Unit Verification Suite
- **Kepler Two-Body Orbit Conservation**: Simulate Kepler 2D orbit ($H(q, p) = \frac{1}{2}\|p\|^2 - \frac{1}{\|q\|}$) over $10^5$ orbital periods with eccentricity $e = 0.6$:
  - Assert `SymplecticEuler`, `StormerVerlet`, `Ruth3`, `Yoshida4`, and `Yoshida6` bound energy error $|\Delta H / H_0| < 10^{-5}$ with zero secular drift.
  - Assert classical `RungeKutta4` exhibits monotonic orbital decay and inward spiraling under the same time step.
- **Order of Convergence Verification**: Step a harmonic oscillator $H(q, p) = \frac{1}{2} p^2 + \frac{1}{2} \omega^2 q^2$ over steps $h \in [0.1, 0.001]$:
  - `SymplecticEuler`: slope $1.00 \pm 0.05$.
  - `StormerVerlet`: slope $2.00 \pm 0.05$.
  - `Ruth3`: slope $3.00 \pm 0.05$.
  - `Yoshida4`: slope $4.00 \pm 0.05$.
  - `Yoshida6`: slope $6.00 \pm 0.05$.

#### 6.2 Validation
- **Hamiltonian Neural Network Trajectory**: Connect an in-crate HNN pendulum model implementing `HamiltonianDynamics` to `Yoshida4` and assert exact time reversibility $S(-h) \circ S(h) = I$.

---

### 7. Performance & Resource Considerations

- **Stack Allocation**: Symplectic steppers require only 1–2 temporary vector buffers of size $N_q$ floats on the stack.
- **Computational Cost**: For separable systems, symplectic steppers require zero matrix inversions and evaluate velocity and force sequentially.

---

### 8. Risks & Open Questions

- **Proposal (not in evidence) 1**: Decoupled `HamiltonianDynamics` trait definition.
- **Proposal (not in evidence) 2**: In-place partitioned coordinate mutation for composition steppers.

---

### 9. Development Plan

| Phase | Description | Estimated Effort (1-10) |
|:------|:------------|:------------------------|
| **Phase 1: Dynamics Trait & Low-Order Steppers** | Implement `HamiltonianDynamics`, `SymplecticEuler`, and `StormerVerlet`. | 2 |
| **Phase 2: High-Order Composition Steppers** | Implement `Ruth3`, `Yoshida4`, and `Yoshida6` composition engines. | 3 |
| **Phase 3: Kepler & Symplectic Verification** | Implement long-horizon Kepler orbit and energy conservation benchmarks. | 3 |

---

### 10. Revision History

| Version | Date | Author | Description |
|:--------|:-----|:-------|:------------|
| 1.0 | August 26, 2026 | @MitchellDScott | Initial draft: symplectic geometric integration subsystem design. |

---

## References

[1] S. Greydanus, M. Dzamba, and J. Yosinski, "Hamiltonian Neural Networks," in *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*, 2019, pp. 15353–15363.

[2] E. Hairer, C. Lubich, and G. Wanner, *Geometric Numerical Integration: Structure-Preserving Algorithms for Ordinary Differential Equations*, 2nd ed. Berlin, Germany: Springer-Verlag, 2006.

[3] R. D. Ruth, "A Canonical Integration Technique," *IEEE Transactions on Nuclear Science*, vol. 30, no. 4, pp. 2669–2671, 1983, doi: 10.1109/TNS.1983.4332919.

[4] H. Yoshida, "Construction of higher order symplectic integrators," *Physics Letters A*, vol. 150, no. 5–7, pp. 262–268, 1990, doi: 10.1016/0375-9601(90)90092-3.
