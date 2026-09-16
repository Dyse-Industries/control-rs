# Classical Control Tools & Firmware Realizations (Design Document)

![Date Badge](https://img.shields.io/badge/Date-September_15,_2026-blue)
![Status](https://img.shields.io/badge/status-approved-brightgreen)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

`classical_tools` provides SISO control system analysis, compensator synthesis,
and real-time firmware execution topologies. Typical usage scenarios include:

- Assessing closed-loop characteristic polynomial stability and right-half plane
  pole counts via Routh-Hurwitz criteria.
- Evaluating root-locus closed-loop pole trajectories over parameterized gain
  sweeps.
- Extracting gain margins ($K_g$), phase margins ($\Phi_m$), delay margins
  ($\tau_m$), and crossover frequencies ($\omega_{gc}, \omega_{pc}$) from
  rational transfer functions.
- Realizing discrete-time transfer functions and polynomials as high-rate
  firmware controllers using Transposed Direct Form II (DF2T) and cascaded
  Second-Order Sections (Biquads / SOS)
- Synthesizing continuous and discrete lead, lag, and lead-lag compensator
  models emitting `TransferFunction` primitives.
- Executing high-rate discrete PID loops directly on scalar signals.

Comparable prior art exists in the broader Rust and Julia control ecosystem —
ControlSystems.jl (JuliaControl, 2026), `control_systems_torbox` (Børve,
2026), and `scirs2-signal` (cool-japan, 2026) — none of which target
`#![no_std]` firmware execution directly.

---

### 2. Requirements

#### 2.1 Functional Requirements

- **FR-1 — Routh-Hurwitz stability assessment**: The module constructs the
  Routh table from polynomial coefficients and returns the number of open
  right-half-plane roots by counting sign changes in the first column.
- **FR-2 — Degenerate Routh array handling**: The Routh algorithm resolves
  all-zero rows and first-column zeros. Auxiliary-polynomial and
  $\epsilon$-perturbation mechanics are a §4 choice.
- **FR-3 — Root locus trajectory computation**: The module evaluates
  closed-loop poles of $D(s) + k \cdot N(s) = 0$ over a caller-supplied gain
  array, writing roots into a pre-allocated buffer. An adaptive sweep over
  $[K_{\min}, K_{\max}]$ may insert sub-steps to bound pole displacement,
  without heap allocation.
- **FR-4 — Frequency-response crossovers and stability margins**: The module
  extracts gain margin, phase margin, and the associated crossover frequencies
  from a transfer-function frequency response.
- **FR-5 — Discrete filter step with bounded state**: A discrete transfer
  function steps in constant memory with a bounded delay-register count.
  Direct-form layout is a §4 choice.
- **FR-6 — High-order filter as cascaded sections**: High-order discrete
  filters ($n \ge 2$) execute as a series of second-order stages so coefficient
  quantization and roundoff do not dominate the response.
- **FR-7 — Anti-windup error-feedback regulation**: Closed-loop PID updates on
  a scalar error include output saturation and anti-windup so the integrator
  does not wind up under clamp.
- **FR-8 — Rational compensator synthesis**: The module constructs continuous
  and discrete PID, lead, lag, and lead-lag networks as native transfer-function
  models.
- **FR-9 — Multiply-accumulate hardware lowering**: Second-order section
  evaluation is formulated so a fused multiply-accumulate can lower to a
  single-cycle instruction on FMA-equipped targets.
- **FR-10 — Step-response transient metrics**: The module extracts rise
  time, peak overshoot, settling time, and steady-state error from a sampled
  SISO step-response trajectory without heap allocation.

#### 2.2 Non-Functional Requirements

- **NFR-1 — Zero Dynamic Allocation**: All sweep drivers, Routh array
  constructions, margin solvers, biquad cascades, and PID step updates must
  execute within `#![no_std]` and `no_alloc` constraints, using caller-allocated
  buffers or stack memory.
- **NFR-2 — Real-Time $\mathcal{O}(1)$ Section Execution**: Each Biquad stage
  and PID update must execute in constant time $\mathcal{O}(1)$ with
  deterministic instruction latency suitable for embedded ISR execution up to 50
  kHz+.
- **NFR-3 — Margin Crossing Precision**: Frequency margin extraction must
  achieve relative frequency crossover precision $\le 10^{-4}$ and phase margin
  precision $\le 0.1^\circ$ via localized interval bisection between sweep
  points.

#### 2.3 Constraints

- **C-1 — Inherited toolbox bounds**: Conforms to `controls-tools-design.md`
  C-1..C-4. Former restated native-authorability, no-codegen, and no_alloc
  paragraphs are withdrawn.
- **C-3 — Generic float support**: Analysis routines, filter realizations, and
  PID are generic over a floating scalar, supporting `f32` and `f64`.
- **C-4 — Offline-only interactive plotting**: Interactive GUI plotting of root
  locus, Bode, Nyquist, and Nichols charts remains an offline host-side utility.

---

### 3. Technical Overview

The module delivers a complete classical control suite organized into three
layers:

1. **Algebraic & Frequency Analysis**: Routh-Hurwitz polynomial stability,
   companion-matrix root locus sweeps, and bisection-refined frequency margin
   extraction.
2. **Discrete Firmware Realization Topologies**: High-assurance execution
   structures that translate rational $H(z)$ transfer functions into numerically
   conditioned difference equations:
    - `Biquad<T>` & `BiquadCascade<T, N>`: Transposed Direct Form II
      second-order sections.
    - `DirectForm2T<T, N>`: Canonical minimal-delay transposed state execution.
3. **Dedicated Real-Time Controllers**: Compact `Pid<T>` scalar controller with
   filtered derivative-on-measurement and clamping anti-windup.

---

### 4. Architecture

#### 4.1. Routh-Hurwitz Stability Engine

Given a characteristic
polynomial $P(s) = a_n s^n + a_{n-1} s^{n-1} + \dots + a_0$:

1. **Table Construction**: Forms array $R \in \mathbb{R}^{(n+1) \times m}$
   where $m = \lceil(n+1)/2\rceil$.
    - Row 0: $[a_n, a_{n-2}, a_{n-4}, \dots]$
    - Row 1: $[a_{n-1}, a_{n-3}, a_{n-5}, \dots]$
    - Subsequent
      rows $i \ge 2$: $R_{i, j} = \frac{R_{i-1, 0} \cdot R_{i-2, j+1} - R_{i-2, 0} \cdot R_{i-1, j+1}}{R_{i-1, 0}}$.
2. **Degenerate Handling**:
    - **First-Column Zero ($R_{i,0} = 0$)**: Replaces $R_{i,0}$ with
      dynamic $\epsilon = 10^{-12}$ to avoid division by zero (Davidson, 2020).
    - **Row of Zeros ($R_{i,j} = 0, \forall j$)**: Forms auxiliary
      polynomial $A(s) = \sum_k R_{i-1, k} s^{n - (i-1) - 2k}$, differentiates
      with respect to $s$, and replaces the zero row with $\frac{d}{ds} A(s)$
      (Davidson, 2020).
3. **Stability Determination**: Counts sign changes in $R_{:, 0}$ to determine
   the number of open right-half plane roots.

#### 4.2. Root Locus Sweep Engine

##### 1. Fixed-Grid Sweep (`sweep`)

Given open-loop transfer function $G(s) = \frac{N(s)}{D(s)}$ and gain
grid $K = [k_0, k_1, \dots, k_M]$:

1. Forms closed-loop characteristic
   polynomials $P_j(s) = D(s) + k_j \cdot N(s)$.
2. Solves complex roots via companion-matrix eigensolves
   (`control_rs::polynomial`), the standard numerical technique for
   polynomial root extraction (NumPy Developers, 2024).
3. Tracks continuous branch trajectories across consecutive gains using greedy
   nearest-neighbor matching, executing internal sub-steps when pole displacement
   exceeds branch separation boundaries to guarantee zero branch swapping.
4. Writes roots into caller-supplied pre-allocated slice `&mut [Complex<T>]` of
   size $(M+1) \times \deg(P)$.

##### 2. Adaptive Sweep with Retained Sub-steps (`sweep_adaptive`)

When sweeping root locus trajectories across a uniform gain parameterization ($\Delta K = \text{const}$), the complex-plane velocity of closed-loop poles:
$$\frac{ds}{dK} = -\frac{N(s)}{D'(s) + K N'(s)}$$
diverges at breakaway and break-in points where $D'(s) + K N'(s) = 0$. For a multiplicity-2 breakaway point, the local displacement scales as $\Delta s \sim \sqrt{\Delta K} \gg \Delta K$. Consequently, uniform gain stepping creates large visual jumps and coarse polygonal chords that skip breakaway junctions.

To trace smooth and visually consistent pole trajectories without dynamic heap allocation, `sweep_adaptive` computes an arc-length-conditioned gain trajectory (Pan and Chao, 1978):

1. **Parameterization & Buffers**: Given gain range $[K_{\min}, K_{\max}]$ and maximum complex-plane pole displacement $\Delta s_{\max}$, the caller supplies pre-allocated buffers `gains_out: &mut [T]` and `roots_out: &mut [Complex<T>]` of capacity $M$ and $M \times \deg(P)$ respectively.
2. **Initial Roots**: Solves closed-loop poles at $K_{\min}$, canonically sorts them by real then imaginary part, and records them at index 0 (`count = 1`).
3. **Adaptive Step Control**: At gain $k_i$ with tracked poles $s_j(k_i)$, evaluates candidate gain $k_{i+1} = \min(k_i + \Delta K, K_{\max})$:
   - Solves candidate roots and matches them to tracked poles via nearest-neighbor assignment.
   - Computes maximum pole displacement $\Delta s = \max_j |s_j(k_{i+1}) - s_j(k_i)|$.
   - **Acceptance Criterion**: The step is accepted if $\Delta s \le \Delta s_{\max}$ and $\Delta s^2 \le \frac{1}{4} \min_{a \ne b} |s_a(k_i) - s_b(k_i)|^2$ (preserving branch identity).
   - **Step Refinement**: If rejected and $\Delta K > \Delta K_{\min}$, $\Delta K$ is halved or scaled down proportionally and re-evaluated.
   - **Singular Safeguard**: A minimum gain step floor $\Delta K_{\min} = \max(\epsilon \cdot (K_{\max} - K_{\min}), 10^{-12})$ prevents infinite stalls at exact branch collisions.
4. **Sub-step Retention & Termination**: Every accepted step is directly written into `gains_out` and `roots_out`. The sweep terminates when either $K = K_{\max}$ is achieved or the output buffer is filled (`count == gains_out.len()`), returning the number of accepted trajectory points written.


#### 4.3. Frequency-Domain Stability Margin Extraction

Given rational transfer function $G(s)$ and frequency sweep
array $\omega \in [\omega_{\min}, \omega_{\max}]$:

1. **Evaluation & Unwrapping**:
   Computes $G(j\omega_i) = M(\omega_i) e^{j \phi(\omega_i)}$ via Horner complex
   polynomial evaluation, unwrapping continuous phase across $[-\pi, \pi]$.
2. **Crossing Detection**:
    - Phase Crossover ($\omega_{pc}$): Detects phase
      crossing $-\pi \pmod{2\pi}$.
    - Gain Crossover ($\omega_{gc}$): Detects magnitude crossing
      unity ($0\text{ dB}$).
3. **Bisection Refinement**: Executes 10-step interval bisection on $G(j\omega)$
   to refine crossing frequencies $(\omega_{pc}, \omega_{gc})$.
4. **Margin Outputs**: Gain Margin $K_g = \frac{1}{|G(j\omega_{pc})|}$, Phase
   Margin $\Phi_m = \pi + \phi(\omega_{gc})$, and Delay
   Margin $\tau_m = \frac{\Phi_m}{\omega_{gc}}$, consistent with gain margin
   being determined at the phase crossover frequency and phase margin at the
   gain crossover frequency (python-control, 2024).

#### 4.4. Firmware Realization Topologies: Transposed Direct Form II & Biquad Cascades

To implement discrete transfer
functions $H(z) = \frac{b_0 + b_1 z^{-1} + \dots + b_n z^{-n}}{1 + a_1 z^{-1} + \dots + a_n z^{-n}}$
on embedded firmware without numerical degradation, the module standardizes on
Transposed Direct Form II sections, which minimize coefficient sensitivity and
avoid internal summing-junction overflow relative to Direct Form I
(Oppenheim and Schafer, 2010):

##### 1. Single Biquad (Second-Order Section, DF2T)

A single 2nd-order
section $H(z) = \frac{b_0 + b_1 z^{-1} + b_2 z^{-2}}{1 + a_1 z^{-1} + a_2 z^{-2}}$
maintaining two state delays ($d_1, d_2$):

Each multiply-accumulate term uses `Float::mul_add` rather than separate
`*`/`+` operators, so the compiler emits a fused multiply-add where the
target provides one (Rust Project, 2026a) — including the Cortex-M7's
single-precision FPU, without any `target_arch`-specific code:

```rust
pub struct Biquad<T> {
    pub b0: T,
    pub b1: T,
    pub b2: T,
    pub a1: T,
    pub a2: T,
    d1: T,
    d2: T,
}

impl<T: Float> Biquad<T> {
    #[inline]
    pub fn update(&mut self, input: T) -> T {
        let output = self.b0.mul_add(input, self.d1);
        self.d1 = self.b1.mul_add(input, self.d2) - self.a1 * output;
        self.d2 = self.b2 * input - self.a2 * output;
        output
    }
    pub fn reset(&mut self) {
        self.d1 = T::zero();
        self.d2 = T::zero();
    }
}
```

##### 2. Cascaded Second-Order Sections (`BiquadCascade<T, NUM_SECTIONS>`)

For higher-order controllers ($n > 2$), the transfer function is factored
into $L = \lceil n/2 \rceil$ cascaded biquads, reducing roundoff noise by
orders of magnitude relative to a single high-order direct-form
implementation (Jackson, 1996):
$$H(z) = \prod_{k=1}^{L} \frac{b_{0,k} + b_{1,k} z^{-1} + b_{2,k} z^{-2}}{1 + a_{1,k} z^{-1} + a_{2,k} z^{-2}}$$

```rust
pub struct BiquadCascade<T, const NUM_SECTIONS: usize> {
    pub sections: [Biquad<T>; NUM_SECTIONS],
}

impl<T: Float, const NUM_SECTIONS: usize> BiquadCascade<T, NUM_SECTIONS> {
    #[inline]
    pub fn update(&mut self, mut signal: T) -> T {
        for section in &mut self.sections {
            signal = section.update(signal);
        }
        signal
    }
    pub fn reset(&mut self) {
        for section in &mut self.sections { section.reset(); }
    }
}
```

```mermaid
---
config:
  layout: dagre
---
flowchart LR
    In["u[k]"]:::input --> BQ1["Biquad 1 (DF2T)"]
    BQ1 --> BQ2["Biquad 2 (DF2T)"]
    BQ2 --> Dots["..."]
    Dots --> BQL["Biquad L (DF2T)"]
    BQL --> Out["y[k]"]:::output
    classDef input fill: #0f172a, stroke: #38bdf8, stroke-width: 2px, color: #f8fafc
    classDef output fill: #064e3b, stroke: #34d399, stroke-width: 2px, color: #ecfdf5
```

##### 3. Arbitrary Order Transposed Direct Form II (`DirectForm2T<T, ORDER>`)

For general discrete filters where direct polynomial execution is preferred,
maintaining state delay vector $d \in \mathbb{R}^{\text{ORDER}}$:
$$y[k] = b_0 u[k] + d_1[k-1]$$
$$d_i[k] = b_i u[k] - a_i y[k] + d_{i+1}[k-1]$$

#### 4.5. `Pid<T>` Controller

For direct single-input single-output loops with nonlinear clamping, the
controller retains the positional form, computing the absolute control signal
at each sample rather than an incremental update (Franklin et al., 1998):

- **State**: `integrator: T, prev_meas: T, filtered_deriv: T, has_run: bool`.
- **Step Algorithm**:
    1. $e[k] = r[k] - y[k]$, $\Delta y[k] = y[k] - y[k-1]$.
    2. Derivative on
       measurement: $D[k] = \frac{T_f}{T_f + \Delta t} D[k-1] - \frac{K_d}{T_f + \Delta t} \Delta y[k]$,
       computed from the measurement rather than the error to avoid
       derivative kick on setpoint steps (Åström and Hägglund, 2006).
    3. Candidate
       integration: $I[k] = I[k-1] + K_i \cdot e[k] \cdot \Delta t$.
    4. Output
       clamping: $u[k] = \text{clamp}(K_p \cdot e[k] + I[k] + D[k], u_{\min}
       , u_{\max})$.
    5. Anti-windup: Uses a discount factor to "forget" old error:  $I[k] =
       \gamma I[k-1]$, where $0 \leq \gamma \leq 1$.

#### 4.6. Rational Compensator Constructors

Provides factories returning `TransferFunction` primitives:

- **Lead / Lag**: $C(s) = K \frac{s + 1/T}{s + 1/(\alpha T)}$, where $\alpha
  < 1$ (lead) or $\alpha > 1$ (lag), with pole/zero placement driven by the
  target phase margin and crossover frequency (Iqbal, 2023).
- **Standard PID**: $C(s) = K_p \left(1 + \frac{1}{s T_i} + \frac{s T_d}{1 +
  s T_f}\right)$ (Åström and Hägglund, 2006).

---

### 5. Alternatives Considered

#### 5.1 Direct Form I vs. Direct Form II Transposed

- *Direct Form I*: Requires $2N$ state delays (separate delays for inputs and
  outputs).
- *Direct Form II Transposed*: Requires only $N$ state delays, eliminates
  internal summing junction overflow risks, and minimizes coefficient
  sensitivity in floating-point operations (Oppenheim & Schafer, 2010).
- *Decision*: Standardize on Transposed Direct Form II for both single biquads
  and general direct forms.

#### 5.2 Single High-Order Difference Equation vs. Cascaded Biquads (SOS)

- *Single high-order difference equation*: Roots of high-degree polynomials are
  notoriously ill-conditioned; small floating-point quantization errors shift
  poles across the unit circle, causing instability.
- *Cascaded Biquads*: Decouples high-order dynamics into independent 2nd-order
  stages, dramatically reducing roundoff noise and coefficient sensitivity (
  Jackson, 1996).
- *Decision*: Provide `BiquadCascade` as the primary recommended firmware
  realization for transfer functions of degree $n \ge 2$.

#### 5.3 Standalone `Pid<T>` vs. `TransferFunction` Execution

- *TransferFunction wrapper*: Clean linear algebra abstraction, but unable to
  handle dynamic saturation clamping, state resets, or derivative-on-measurement
  cleanly.
- *Decision*: Use standalone `Pid<T>` for on-target execution and provide
  `Pid::to_transfer_function()` for linear analysis.

#### 5.4 Routh Degenerate $\epsilon$-Substitution vs. Reverse Polynomial

- *Reverse polynomial ($s \leftarrow 1/z$)*: Replaces $s$ with $1/z$, shifting
  zeros away from the first column. However, it fails when both ends of the
  polynomial have zeros.
- *Decision*: Adopt $\epsilon$-perturbation with dynamic epsilon
  threshold ($\epsilon = 10^{-12}$) as primary, using auxiliary differentiation
  for zero rows, following the standard treatment of both degenerate cases
  (Davidson, 2020).

#### 5.5 Portable `Float::mul_add` vs. ARM-Specific Dispatch

- *CMSIS-DSP FFI*: Linking ARM's C library would call the vendor routine
  directly, but requires an external toolchain dependency and non-Rust code,
  conflicting with C-1 and C-2.
- *`core::arch::arm` DSP intrinsics*: ARM's dual-MAC SIMD instructions operate
  on packed 16-bit integers (`int16x2_t`), not `f32`/`f64` (Rust Project,
  2026b), and remain nightly-only with no stabilization scheduled (Rust
  Project, 2026b; rust-lang, 2023) — inapplicable to the current
  generic-float `Biquad<T: Float>` and unsuitable for a stable-Rust default.
- *Portable `Float::mul_add`*: CMSIS-DSP's own reference `f32` biquad
  implementation computes the same recurrence with plain scalar
  multiply-accumulate arithmetic and no SIMD or intrinsic calls (ARM
  Limited, 2026b); `Float::mul_add` reaches the same hardware acceleration
  portably, without a `target_arch` branch.
- *Decision*: Use `Float::mul_add` unconditionally in `Biquad::update`. No
  `target_arch`-gated ARM path is introduced for the current per-sample
  `f32`/`f64` API.

---

### 6. Verification & Validation

#### 6.1 Approach

This section follows the structure and method catalogue of the project's
verification and validation standard, which distinguishes verification
("was the component built right") from validation ("does it serve its
intended purpose") (NASA, 2023; NASA, 2016; control-rs, 2026).

- Demonstrate that `routh::stability` recovers the correct right-half-plane
  root count, including the first-column-zero and row-of-zeros
  degeneracies.
- Demonstrate that `root_locus::sweep` returns closed-loop poles satisfying
  the swept characteristic equation and matching the open-loop poles at
  zero gain.
- Demonstrate that `margins::stability_margins` locates gain and phase
  crossovers within NFR-3's precision bound, and correctly reports no
  crossing when none exists in the sweep.
- Demonstrate equivalent difference-equation behavior between `Biquad`,
  `BiquadCascade`, and `DirectForm2T` realizations of the same transfer
  function, cross-checked against an independent Direct Form I reference.
- Demonstrate that `Pid<T>` reproduces closed-form P, I, and D responses,
  clamps to `[u_min, u_max]`, and recovers from saturation without
  integrator windup.
- Demonstrate that `compensators::lead`, `lag`, and `lead_lag` place poles
  and zeros as specified, and that cascading two stages multiplies their
  frequency responses.
- Demonstrate that `step_info` recovers rise time, overshoot, and
  settling-time metrics from manufactured well-settled and Type-0 offset
  trajectories.
- Demonstrate that Routh, root locus, margins, PID, and compensators
  produce mutually consistent results when applied to one physical plant.

| Method | Mechanism |
|:-------|:----------|
| Compile-time shape check | Type-level parameter bounds `T: Float` |
| Inspection | Architecture, build, and dependency inspection |
| Requirements-based test | `#[test]` unit tests, `src/classical_tools/tests/*.rs` |
| Metamorphic relation | `#[test]` over series connection of lead and lag stages |
| Doctest | Runnable rustdoc examples |
| Static analysis | `cargo lint`, `cargo clippy-ci`; source inspection of the biquad update |
| Resource usage evaluation | `no_alloc` review of analysis and realization kernels |
| On-target execution | ETS suites under QEMU (`thumbv7em`, `riscv32imac`) |
| Coverage measurement | `cargo coverage` |

Target: statement coverage $\ge90\%$ for `routh`, `root_locus`, `margins`,
`pid`, `compensators`, `realization`, and `step_info`, measured with `cargo coverage`. Excluded:
`Debug` / `Display` implementations on `RouthError`, `RootLocusError`,
`CompensatorError`, and `Margins<T>`; and the `#[cfg(not(test))]` ETS
suite re-export paths in `tests/mod.rs`, exercised instead by on-target
execution (NFR-2).

Validation of `classical_tools` is established by the integration and physical plant suites documented in [`classical-tools-examples-design.md`](classical-tools-examples-design.md):
- **Buck Converter Voltage-Mode Loop** (`control-rs-validation`): averaged small-signal $G_{vd}(s)$ per the state-space averaging model of Erickson and Maksimovic (2001), compensated with lead synthesis and cross-validated back-to-back against SciPy and `ngspice` (small-signal AC, averaged transient, and switched PWM ripple).
- **DC Motor Armature Servo** (`control-rs-validation`): third-order electromechanical servo under transport delay, actuator saturation, and 16-bit encoder quantization, comparing lead vs PID control against SciPy.

Both validation suites conform to the host oracle harness contract (`documentation/vv/oracle-harness-design.md`), execute in continuous integration, and emit unified multi-source result envelopes. Pedagogical copies of the same plants live under `examples/` (`examples/buck_converter.rs`, `examples/dc_motor.rs`). Host latency benches live under `benches/classical_tools.rs`. Neither pedagogical examples nor benches are B2B gates.

#### 6.2 Acceptance

| Claim | Oracle | Measure | Bound | Justification |
|:------|:-------|:--------|:------|:--------------|
| Routh RHP count | Independent Aberth root-solve (`Polynomial::roots`), filtered at `re > 1e-6` | Exact equality of counts | count-exact | The `1e-6` filter was set from the root solver's measured noise on the exact $\pm j$ roots of $s^4+2s^3+2s^2+2s+1$ ($\text{re}\sim10^{-16}$); f64 epsilon is $2.22\times10^{-16}$ (IEEE, 2019), so `1e-6` clears solver noise by nine orders of magnitude without risking a true RHP root going uncounted. |
| Root locus residual | Characteristic polynomial evaluated at the computed root | Absolute residual $\lvert P(\hat s)\rvert$ | $<10^{-6}$ | A fixed absolute bound is used instead of the LAPACK scaled-residual convention $r=\lVert A-LU\rVert/(n\lVert A\rVert\epsilon)$ (Anderson et al., 1994) because the swept companion matrices here are low order ($n\le3$) and well conditioned; $10^{-6}$ still leaves five to six orders of margin above the Aberth solver's f64 convergence floor. |
| Adaptive root locus displacement bound | Maximum adjacent pole displacement $\max_j \lvert s_j(K_{i+1}) - s_j(K_i)\rvert$ across recorded steps | Absolute displacement | $\le \Delta s_{\max} + 10^{-4}$ (or at $\Delta K_{\min}$ floor) | Adaptive sub-stepping constrains step-to-step jumps; $10^{-4}$ is the unit-test slack matching f64 companion-root noise on the recorded mesh, not a claim of $10^{-9}$ geometric exactness. |
| Adaptive root locus residual | Closed-loop characteristic polynomial evaluated at each recorded adaptive root | Absolute residual $\lvert P(\hat s)\rvert$ | $<10^{-6}$ | Demonstrates that every recorded adaptive sub-step satisfies the characteristic equation $D(s) + K N(s) = 0$. |
| Margin crossover frequency | Closed-form analytical crossover ($\omega_{gc}=\sqrt{\sqrt[3]{16}-1}$, $\omega_{pc}=\sqrt3$ for $G(s)=4/(s+1)^3$), method of manufactured solutions (Yeo, 2020) | Absolute error | $10^{-2}$ rad/s (test comparison); $\sim2\times10^{-6}$ rad/s absolute (algorithm) | The bisection loop halves a $\sim0.0025$ rad/s coarse-sweep bracket for 10 iterations, reaching $\sim2\times10^{-6}$ rad/s absolute precision — comfortably inside NFR-3's $10^{-4}$ relative bound. The test's own $10^{-2}$ tolerance is a looser sanity check against the independently-derived analytical crossover, not the algorithm's achieved precision. |
| PID closed-form responses | Analytical P, I, D, and frequency-response formulas evaluated independently of `step` / `to_transfer_function` (method of manufactured solutions, Yeo, 2020) | Absolute error | $10^{-12}$ ($P$-only, saturation recovery, reset); $10^{-9}$ (integral running sum, frequency cross-check); $10^{-3}$ (derivative steady state) | Exact-algebra cases bound at f64 rounding ($10^{-12}$); the derivative filter's steady state is a settling limit reached over a finite simulated window ($5\,\text{s}$ against a $T_f=0.05\,\text{s}$ time constant), so its bound is looser. |
| Realization equivalence | Independent Direct Form I difference equation (Oppenheim and Schafer, 2010) | Absolute error | $10^{-12}$ (cross-structure equivalence); $10^{-9}$ (DC-gain convergence); $10^{-15}$ (reset / identity) | DF2T and Biquad share no code path with the Direct Form I reference; bounds track accumulated f64 rounding over $\le200$ recurrence steps. |
| Compensator pole/zero placement | Closed-form pole/zero location of $K(s+1/T)/(s+1/(\alpha T))$ | Absolute error | $<10^{-9}$ | A degree-1 polynomial resolves via `Polynomial::roots`'s closed-form branch, not the iterative Aberth solver, so error is limited to f64 rounding in one division. |

#### 6.3 Limits

- No property-based (`proptest`) coverage exists over randomized polynomial
  or transfer-function inputs; all cases above are fixed, hand-derived
  representative inputs.
- This plan does not establish `f32` high-order cascade error of cascaded
  second-order sections versus a direct-form realization.
- On-target ETS execution under QEMU and Teensy has not run in firmware
  binaries that link this module; NFR-2 is argued from source inspection
  rather than measured on-target.
- Withdrawn C-2: restated inherited toolbox bounds; covered by C-1.
- Whether `Float::mul_add` lowers to a genuine fused multiply-add
  instruction on target, rather than being scalarized, is unverified;
  FR-9's rationale is architectural (matching the CMSIS-DSP reference
  approach) rather than confirmed by disassembly or cycle count.

---

### 7. Performance & Resource Considerations

- **Biquad Section Footprint**:
    - Memory: 2 scalar state words ($8$ bytes on 32-bit, $16$ bytes on 64-bit
      per section).
    - Computation: 5 multiplications, 4 additions per section ($\sim 8$ cycles
      on ARM Cortex-M7 with single-cycle FPU MAC instructions), matching the
      reference CMSIS-DSP DF2T biquad cost (ARM Limited, 2026b).
- **PID Controller**: Requires 4 words of scalar state ($16$ bytes on
  32-bit, $32$ bytes on 64-bit) and $\sim 15$ arithmetic cycles per update step.
- **Analysis Tools**: Routh array construction requires $\mathcal{O}(n^2)$
  scratch space on stack ($< 1\text{ KB}$ for degree $n \le 16$).

---

### 8. Risks & Open Questions

- **Complex Root Pairing for SOS Factoring**: Automated conversion of arbitrary
  high-order `TransferFunction` into `BiquadCascade` requires sorting and
  pairing complex conjugate poles with their closest zeros to maximize dynamic
  range; whether this factoring helper should be an offline constructor or
  runtime method is unresolved.
- **High-Rate Delta Operator ($\delta$) Extension**: Whether a specialized
  `DeltaBiquad` struct should be introduced in a future phase for sample rates
  exceeding 20 kHz, where shift-operator poles cluster near $z=1$ and become
  numerically ill-conditioned (Middleton and Goodwin, 1990).
- **Block-Oriented SIMD Path**: `Biquad`/`BiquadCascade` process one sample
  per call; a future `update_block(&mut [T])` API processing multiple
  samples per call could benefit from Cortex-M55 Helium/MVE or Cortex-A NEON
  vectorization in a way the current per-sample form cannot. Whether to add
  a `target_arch`-gated block path at that point is unresolved.

---

### 9. Development Plan

| Phase                                      | Description                                                                                                             | Estimated Effort |
|:-------------------------------------------|:------------------------------------------------------------------------------------------------------------------------|:-----------------|
| **Phase 1: Biquad & DF2T Realizations**    | Implement `Biquad<T>`, `BiquadCascade<T, N>`, and `DirectForm2T<T, N>` structs with `update` and `reset` methods.       | 2.0 Days         |
| **Phase 2: Routh-Hurwitz Engine**          | Implement Routh array builder with first-column-zero and row-of-zeros auxiliary polynomial handling.                    | 2.0 Days         |
| **Phase 3: Root Locus & Margins**          | Implement gain-sweep root locus and frequency margin crossing bisection over `TransferFunction`.                        | 2.5 Days         |
| **Phase 4: Standalone PID & Compensators** | Implement `Pid<T>` struct, `lead()`, `lag()`, and   `Pid::to_transfer_function()` model factories.                        | 1.5 Days         |
| **Phase 5: V&V & Target Profiling**        | Numerical equivalence tests, SOS quantization tests including `f32` high-order cascade error versus a direct-form realization, independent host-side oracle cross-validation, ETS QEMU target execution, and a buck-converter validation example. | 2.5 Days         |

---

### 10. Revision History

| Revision | Date              | Author          | Description                                                                                                                                                                                                 |
|:---------|:------------------|:----------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1.0      | September 3, 2026 | @MitchellDScott | Initial draft: classical tools, Routh-Hurwitz, and frequency response margins.                                                                                                                              |
| 1.1      | September 3, 2026 | @MitchellDScott | Added standalone discrete `Pid<T>` controller struct with anti-windup and filtered derivative.                                                                                                              |
| 1.2      | September 3, 2026 | @MitchellDScott | Balanced section weights: expanded Routh-Hurwitz and margin analysis; streamlined PID.                                                                                                                      |
| 1.3      | September 3, 2026 | @MitchellDScott | Added firmware realization topologies: Transposed Direct Form II and Cascaded Biquads.                                                                                                                      |
| 1.4      | September 4, 2026 | @MitchellDScott | Grounded all requirements and architecture claims in research evidence; restored anti-windup citation; added prior-art note; renumbered References as §10.                                                  |
| 1.5      | September 4, 2026 | @MitchellDScott | Added FR-9 and a portable `Float::mul_add` biquad MAC path; rejected `target_arch`-gated ARM SIMD/CMSIS-DSP-FFI dispatch (Alternatives 5.5); flagged a future block-oriented SIMD path as an open question. |
| 1.6      | September 4, 2026 | @MitchellDScott | Restructured §6 (Verification & Validation) to the project's vv-standards.md format (6.1 Objectives through 6.7 Not verified); replaced aspirational bullets with descriptions of the committed unit-test suites and the cases they cover; replaced the disconnected synthetic-transfer-function validation plan with a single RLC-circuit scenario exercising Routh, root locus, margins, PID, and compensators together. |
| 1.7      | September 4, 2026 | @MitchellDScott | Replaced §6.6's series-RLC validation plant with a buck converter's averaged small-signal duty-cycle-to-output-voltage model, so validation exercises a compensator design (crossover placement, phase margin) instead of only open-loop analysis; the plant maps directly to a PWM-timer/ADC firmware loop. |
| 1.8      | September 9, 2026 | @MitchellDScott | Structural hardening: renamed FR-5/6/7/9 to need-named statements, mapped Constraints in §6.2/6.4, standardized reference ordering.                                                                         |
| 1.9      | September 12, 2026 | @MitchellDScott | Added adaptive root locus sweep (`sweep_adaptive`) specification with retained sub-steps and displacement-bounded parameterization (Pan and Chao, 1978) to resolve locus jumps near breakaway singularities. |
| 1.10     | September 12, 2026 | @MitchellDScott | Added FR-10 `step_info`; aligned adaptive $\Delta s$ slack with unit tests; repaired §6.4 locators. |
| 1.11     | September 15, 2026 | @MitchellDScott | Split validation/, examples/, and bench/; relocate plant oracle suites under validation/. |
| 1.12     | September 15, 2026 | @MitchellDScott | Need-named FRs without in-body cites; inherited C-1 pointer; §6.2 catalogue table only; f32 cascade in §9 not 6.7. |
| 1.13     | September 16, 2026 | @MitchellDScott | Retired `vv-standards.md`: reference [22] is now the design template. |

---

## References

[1] JuliaControl, "ControlSystems.jl: A Control Systems Toolbox for Julia,"
*GitHub*, 2026. [Online].
Available: https://github.com/JuliaControl/ControlSystems.jl.
Accessed: Sep. 3, 2026.

[2] T. B{\o}rve, "control\_systems\_torbox," Version 0.1.0, 2026. [Online].
Available: https://docs.rs/control_systems_torbox/latest/control_systems_torbox/.
Accessed: Sep. 3, 2026.

[3] cool-japan, "scirs2-signal: Signal processing and control routines,"
*GitHub*, 2026. [Online]. Available: https://github.com/cool-japan/scirs2.
Accessed: Sep. 3, 2026.

[4] R. R. Shamshiri, "Routh-Hurwitz Stability test," *MATLAB Central File
Exchange*, 2009. [Online].
Available: https://www.mathworks.com/matlabcentral/fileexchange/25956-routh-hurwitz-stability-test.
Accessed: Sep. 3, 2026.

[5] T. Davidson, "Section 4: Stability and the Routh-Hurwitz Condition," *EE
3CL4 Course Notes, McMaster University*, 2020. [Online].
Available: https://www.ece.mcmaster.ca/~davidson/EE3CL4/slides/Routh_Hurwitz_handout.pdf.
Accessed: Sep. 4, 2026.

[6] F. Sagharchi, "Routh-Hurwitz stability criterion," *MATLAB Central File
Exchange*, 2016. [Online].
Available: https://www.mathworks.com/matlabcentral/fileexchange/17483-routh-hurwitz-stability-criterion.
Accessed: Sep. 3, 2026.

[7] NumPy Developers, "numpy.roots," *NumPy Manual*, Version 2.2, 2024.
[Online].
Available: https://numpy.org/doc/2.2/reference/generated/numpy.roots.html.
Accessed: Sep. 4, 2026.

[8] python-control Developers, "control.stability\_margins," *Python Control
Systems Library Documentation*, Version 0.10.2, 2024. [Online].
Available: https://python-control.readthedocs.io/en/0.10.2/generated/control.stability_margins.html.
Accessed: Sep. 4, 2026.

[9] A. V. Oppenheim and R. W. Schafer, *Discrete-Time Signal Processing*, 3rd
ed. Upper Saddle River, NJ, USA: Prentice Hall, 2010.

[10] L. B. Jackson, *Digital Filters and Signal Processing*, 3rd ed. Norwell,
MA,
USA: Kluwer Academic Publishers, 1996.

[12] K. Iqbal, "6.3: Frequency Response Design," in *Introduction to Control
Systems*, LibreTexts, 2023. [Online].
Available: https://eng.libretexts.org/Bookshelves/Industrial_and_Systems_Engineering/Introduction_to_Control_Systems_(Iqbal)/06:_Compensator_Design_with_Frequency_Response_Methods/6.03:_Frequency_Response_Design.
Accessed: Sep. 4, 2026.

[13] ARM Limited, "arm\_biquad\_cascade\_df2T\_f32.c," in
*ARM-software/CMSIS_4*.
[Online].
Available: https://github.com/ARM-software/CMSIS_4/blob/master/CMSIS/DSP_Lib/Source/FilteringFunctions/arm_biquad_cascade_df2T_f32.c.
Accessed: Sep. 4, 2026.

[14] The Rust Project Developers, "f32::mul\_add," *The Rust Standard
Library*. [Online]. Available: https://doc.rust-lang.org/std/primitive.f32.html.
Accessed: Sep. 4, 2026.

[15] G. F. Franklin, J. D. Powell, and M. L. Workman, *Digital Control of
Dynamic Systems*, 3rd ed. Menlo Park, CA, USA: Addison-Wesley, 1998.

[16] The Rust Project Developers, "\_\_smlad in core::arch::arm," *Rust
Standard Library Documentation*. [Online].
Available: https://doc.rust-lang.org/core/arch/arm/fn.__smlad.html.
Accessed: Sep. 4, 2026.

[17] rust-lang, "Tracking Issue for 32-bit ARM DSP intrinsics (\#117237)," in
*rust-lang/rust*. [Online].
Available: https://github.com/rust-lang/rust/issues/117237.
Accessed: Sep. 4, 2026.

[18] ARM Limited, "arm\_biquad\_cascade\_df2T\_f32: Biquad Cascade IIR Filters
Using a Direct Form II Transposed Structure," *CMSIS-DSP Documentation*, 2026.
[Online].
Available: https://arm-software.github.io/CMSIS_5/DSP/html/group__BiquadCascadeDF2T.html.
Accessed: Sep. 3, 2026.

[19] R. H. Middleton and G. C. Goodwin, *Digital Control and Estimation: A
Unified Approach*. Englewood Cliffs, NJ, USA: Prentice Hall, 1990.

[20] NASA, "SWE-028 - Verification Planning," NASA Software Engineering
Handbook, 2023.

[21] NASA, *NASA Systems Engineering Handbook*, NASA/SP-2016-6105 Rev 2,
Washington, DC, USA, 2016.

[22] control-rs, "Design Document Template," internal document,
documentation/design-template.md, 2026.

[23] S. Segura, G. Fraser, A. B. Sanchez, and A. Ruiz-Cortes, "A Survey on
Metamorphic Testing," *IEEE Trans. Softw. Eng.*, vol. 42, no. 9, pp.
805-824, 2016.

[24] IEEE, "IEEE Standard for Floating-Point Arithmetic," IEEE Std
754-2019, IEEE, New York, NY, USA, 2019.

[25] E. Anderson, J. Dongarra, and S. Ostrouchov, "Installation Guide for
LAPACK," LAPACK Working Note 41, Univ. of Tennessee, Knoxville, TN, USA,
1994.

[26] D. Yeo, "A Summary of Industrial Verification, Validation, and
Uncertainty Quantification Procedures in Computational Fluid Dynamics,"
NISTIR 8298, NIST, Gaithersburg, MD, USA, 2020.

[27] K. Claessen and J. Hughes, "QuickCheck: A Lightweight Tool for Random
Testing of Haskell Programs," in *Proc. 5th ACM SIGPLAN Int. Conf.
Functional Programming*, Montreal, Canada, 2000, pp. 268-279.

[28] R. W. Erickson and D. Maksimovic, *Fundamentals of Power
Electronics*, 2nd ed. Norwell, MA, USA: Kluwer Academic Publishers, 2001.

[29] C. T. Pan and K. S. Chao, "A computer-aided root-locus method," *IEEE
Trans. Automat. Contr.*, vol. 23, no. 5, pp. 856-860, Oct. 1978.

