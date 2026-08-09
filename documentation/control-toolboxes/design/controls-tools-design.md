# Controls Ecosystem Positioning (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_8,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

`control-rs` authors `TransferFunction`, `StateSpace`, `Polynomial`, `Matrix`,
and `Tensor` models directly in Rust against a hardware-agnostic `Storage`
trait, compiling to `no_std`/`no_alloc` targets without an intermediate
export or code-generation step. The `classical_tools`, `modern_tools`,
`robust_tools`, and `nonlinear_tools` modules extend this same primitive set
toward classical, modern, robust, and nonlinear control design.

External embedded-control tooling generally does not share this authoring
model. Research into ETH Zurich's Control Toolbox (CT), acados, FORCES Pro,
CVXGEN, SymForce, PX4, ArduPilot, ros2_control, and Simulink Embedded Coder
(`documentation/control-toolboxes/research/results/controls-tools.json`)
shows a spectrum running from workstation research toolboxes, through
symbolic-design/embedded-codegen pipelines, to native on-target authoring.
This document records where `control-rs` sits on that spectrum and states
the constraint that positioning places on every current and future
`*_tools` module.

---

### 2. Requirements

The following are binding constraints on `classical_tools`, `modern_tools`,
`robust_tools`, `nonlinear_tools`, and any future controls-toolbox module,
derived from the comparative research:

- **Native authorability.** Every controller construct a toolbox module
  exposes (gain computation, filter design, controller structure) must be
  directly expressible as Rust code against `Matrix`/`Polynomial`/
  `TransferFunction`/`StateSpace` and the `Storage` trait, compilable and
  runnable on the target `no_std` device without a separate code-generation
  or export pass. This mirrors the "author natively for the target" pole
  observed in PX4, ArduPilot, and the existing Rust `no_std` control-crate
  ecosystem — `minikalman` (sunsided, 2026), `adskalman` (strawlab, 2026),
  and `pid-ctrl` (pid-ctrl, 2026).
- **No required host-to-target codegen pipeline.** A toolbox module must
  not depend on a MATLAB/Python/Simulink/CasADi-style external modeling
  environment producing C or Rust source that is then vendored into the
  crate. This rules out an acados- or FORCES Pro-style workflow, where the
  control problem is formulated in one environment and a solver is
  generated for another (Verschueren et al., 2021; embotech, 2026).
- **No mandatory dynamic allocation.** Toolbox algorithms must hold to the
  same `no_alloc`, stack/static-storage discipline as `Matrix` and
  `StateSpace`, keeping `control-rs` at the bare-metal, zero-heap execution
  tier described in §4.2, distinct from the RTOS-plus-heap tier PX4 and
  ArduPilot occupy.
- **Compile-time verification takes priority over runtime numerical-
  equivalence pipelines.** A toolbox module should rely on the crate's
  existing Peano-arithmetic const-generic dimension checking to catch
  shape and structural errors at build time, rather than depending on a
  Simulink-style MIL/SIL/PIL sequence as the primary correctness gate
  (MathWorks, 2026).
- **Symbolic/offline tooling, if ever introduced, stays clearly out of the
  runtime path.** Any future design-time helper (e.g., symbolic
  linearization or gain-scheduling table generation) must be scoped as an
  offline, host-side utility analogous to `matrix-design.md`'s existing
  Faddeev-LeVerrier characteristic-polynomial scoping, not as a
  SymForce-style generator whose output becomes a runtime dependency
  (Steiner et al., 2022).

---

### 3. Technical Overview

This document does not scope a new module or API. It states an
architectural constraint — grounded in comparative evidence from nine
external tools and frameworks — that governs how `classical_tools`,
`modern_tools`, `robust_tools`, and `nonlinear_tools` are designed and
implemented. No `src/controls_tools/` module exists or is proposed; the
four sibling toolbox scaffolds already present under `src/` are the actual
subjects of this constraint.

---

### 4. Architecture

#### 4.1. The Workflow Spectrum

Every surveyed tool falls somewhere between two poles: "author natively
for the target" and "design in one environment, generate code for
another."

- **Native-authoring pole**: CT, PX4, ArduPilot, ros2_control, and the
  existing Rust `no_std` control crates (`minikalman`, `adskalman`,
  `pid-ctrl`) write the control law directly against the target's runtime
  abstraction — Eigen/C++ objects for CT (Giftthaler et al., 2018), AP_HAL
  for ArduPilot, or a `Storage`-backed type for `control-rs`.
- **Generate-for-target pole**: Simulink Embedded Coder, acados, FORCES
  Pro, CVXGEN, and SymForce's PX4 EKF2 integration formulate the problem
  in a workstation-class symbolic or modeling environment (CasADi,
  SymPy/SymForce, Simulink) and emit static C for the embedded target
  (Andersson et al., 2019; Verschueren et al., 2021; embotech, 2026;
  Steiner et al., 2022).

`control-rs` sits at the native-authoring pole. `TransferFunction`,
`StateSpace`, `Polynomial`, `Matrix`, and `Tensor` are authored directly in
Rust against the `Storage` trait; its differentiator relative to the other
native-authoring tools is compile-time dimensional verification via
Peano-arithmetic const generics, which the generate-for-target tools obtain
instead through post-hoc SIL/PIL numerical-equivalence testing (MathWorks,
2026).

#### 4.2. Hardware/Runtime Tier Placement

`control-rs` occupies a bare-metal `no_std` execution tier — zero heap
allocation, 1 kHz-10 kHz+ loop frequency — distinct from the workstation
tools surveyed (CT, Drake, Pinocchio, CasADi), the real-time Linux/
middleware tools surveyed (ros2_control, micro-ROS), and the RTOS
microcontroller tools surveyed (PX4, ArduPilot, acados' STM32 target,
FORCES Pro embedded C) that retain a heap allocator or RTOS dependency
(ArduPilot, 2026; Verschueren et al., 2021; embotech, 2026).
`classical_tools`, `modern_tools`, `robust_tools`, and `nonlinear_tools`
inherit this tier placement by construction: they build exclusively on
`Matrix`/`StateSpace`/`TransferFunction`, which already carry the
`no_alloc`, static-storage constraints documented in `matrix-design.md`
and `state-space-design.md`.

#### 4.3. Compile-Time Verification vs. MIL/SIL/PIL/HIL

Simulink-derived and acados/FORCES Pro workflows rely on a MIL -> SIL ->
PIL -> HIL verification sequence to establish numerical equivalence
between the modeling environment and the deployed target (MathWorks,
2026). `control-rs` catches an overlapping but distinct class of errors —
matrix-dimension mismatches, state-space dimension mismatches, storage
layout mismatches — at `rustc` compile time via the crate's existing
Peano-arithmetic `Dim` system, at zero runtime cost, before any MIL/SIL/PIL
pass would run. This is complementary to, not
a replacement for, the crate's own `control-rs-hil` runtime test harness;
it removes an entire class of runtime shape-mismatch and allocation-panic
failure modes ahead of that runtime testing rather than duplicating what
MIL/SIL/PIL already cover.

#### 4.4. Explicitly Decided Against

- **No acados/FORCES Pro/CVXGEN-style embedded-codegen exporter.**
  `control-rs` does not take a symbolically-formulated problem from an
  external modeling environment and emit static C or Rust solver code as
  a build artifact. Solver logic for `modern_tools`/`robust_tools` is
  written directly against `Matrix`/`StateSpace`, matching the native-
  authoring pole rather than the acados/FORCES Pro pole (Verschueren et
  al., 2021; embotech, 2026).
- **No Simulink Embedded Coder-style MIL/SIL/PIL pipeline as a primary
  gate.** `control-rs` does not adopt IEC Certification Kit-style tool
  qualification or a dedicated MIL/SIL/PIL numerical-equivalence pipeline
  as the toolbox modules' main verification mechanism; compile-time
  dimensional verification plus the existing `control-rs-hil` harness
  serve this role instead (MathWorks, 2026).
- **No SymForce-style symbolic-generator dependency for toolbox
  algorithms.** Gain computation, filter synthesis, and controller
  structure in `classical_tools`/`modern_tools`/`robust_tools` are
  hand-authored against `Matrix`/`Polynomial`, not produced by a symbolic
  Python/SymPy front end emitting C++ kernels, as SymForce does for PX4's
  EKF2 (Steiner et al., 2022).

---

### 5. Alternatives

#### 5.1. Adopt a CasADi/acados-Style Symbolic-Formulation-to-Codegen Pipeline

A symbolic front end (CasADi-style `SX`/`MX` expression graphs) generating
static C or Rust solver code was considered as a path to more complex
modern/robust control problems. Rejected: this requires vendoring or
depending on an external symbolic differentiation and codegen toolchain,
conflicting with the crate's minimal-dependency and self-contained
`no_std` audit constraints, and reproduces the acados/FORCES Pro
generate-for-target model this research explicitly evaluated against
(Andersson et al., 2019; Verschueren et al., 2021).

#### 5.2. Adopt a Simulink-Style MIL/SIL/PIL/HIL Verification Pipeline

Formalizing a MIL/SIL/PIL staged pipeline (host simulation -> host-compiled
target code -> target-processor execution -> full hardware loop) modeled
on Embedded Coder's workflow was considered as the toolboxes' primary
verification mechanism (MathWorks, 2026). Rejected as the primary
mechanism: compile-time dimensional verification already eliminates the
shape-mismatch and allocation-panic failure classes MIL/SIL numerical-
equivalence testing is partly aimed at, and the existing `control-rs-hil`
harness already provides target-hardware verification. A lightweight
SIL/PIL-style numerical-equivalence extension to `control-rs-hil` remains
an open question (see §7) rather than a decision made here.

#### 5.3. Narrow Symbolic Codegen for Isolated Kernels (SymForce Precedent)

SymForce's model — hand-authored codebase with a narrow symbolic generator
producing specific dense kernels (e.g., PX4 EKF2's Jacobians) — was
considered as a bounded exception that would not compromise native
authorability elsewhere (Steiner et al., 2022; PX4 Development Team, 2026).
Not adopted at this time: no toolbox module currently has a kernel
complex enough to justify introducing a symbolic-generation dependency,
and doing so would require a design-time-only scoping precedent (as
`matrix-design.md` establishes for Faddeev-LeVerrier) that has not yet
been evaluated for any `*_tools` algorithm. Left open per §7.

---

### 6. Verification & Validation

This document establishes a documented architectural constraint, not a
code change. Verification of this design consists of conformance review:
future `classical_tools`, `modern_tools`, `robust_tools`, and
`nonlinear_tools` design docs and implementations are checked against §2's
requirements (native authorability, no required host-to-target codegen
pipeline, no mandatory dynamic allocation, compile-time verification
priority, offline-only scoping for any symbolic tooling) during their own
design-doc review and code review. There is no separate test suite, CI
gate, or HIL harness to build for this document itself.

---

### 7. Risks & Open Questions

- **SIL/PIL numerical-equivalence gap.** `control-rs` has no analogue to
  Simulink's SIL/PIL numerical-equivalence testing between host-simulated
  and target-executed LTI responses. Value of extending `control-rs-hil` to
  close this gap is unresolved (open question 2 in
  `documentation/control-toolboxes/research/results/controls-tools.json`).
- **SymForce precedent applicability.** Whether SymForce's narrow
  symbolic-kernel-generation model is ever relevant to a future
  `control-rs` Jacobian or linearization workflow is unresolved (open
  question 3 in
  `documentation/control-toolboxes/research/results/controls-tools.json`).
- **OpEn's Rust-codegen pattern.** Whether Optimization Engine's
  Python-formulation-to-generated-Rust-solver pattern (Sopasakis et al.,
  2020) is a viable external integration point if users request
  Python-driven controller synthesis is unresolved and out of scope for
  this document (open question 4 in
  `documentation/control-toolboxes/research/results/controls-tools.json`).
- **Assumption:** this document assumes the four existing `src/`
  toolbox scaffolds (`classical_tools`, `modern_tools`, `robust_tools`,
  `nonlinear_tools`) are the intended scope of "future controls-toolbox
  modules" referenced in §2; this is inferred from the current repository
  structure and is not itself sourced from the research file.

---

### 8. Development Plan

| Task / Feature | Description | Estimated Effort |
|:---|:---|:---|
| Positioning note review | Circulate this document for maintainer review and status update to Approved. | 0.5 Day |
| Requirement propagation | Reference §2's constraints explicitly in the `classical_tools`, `modern_tools`, and `robust_tools` design docs as they are drafted, rather than re-deriving them. | Ongoing, per-module |
| Open-question triage | Decide whether the SIL/PIL harness extension (§7) warrants its own research query before any `modern_tools`/`robust_tools` design doc that would depend on it. | 0.5 Day |

---

### 9. References

[1] sunsided, *minikalman-rs*: no_std Kalman filter with static buffers, f32 and Q16.16 fixed-point support. [Online]. Available: https://github.com/sunsided/minikalman-rs. Accessed: Aug. 8, 2026.

[2] strawlab, *adskalman-rs*: no_std/embedded-capable Kalman filter built on nalgebra, includes Rauch-Tung-Striebel smoothing. [Online]. Available: https://github.com/strawlab/adskalman-rs. Accessed: Aug. 8, 2026.

[3] pid-ctrl, *pid-ctrl*: #![no_std] PID controller with output clamping and measurement-based derivative. [Online]. Available: https://crates.io/crates/pid-ctrl. Accessed: Aug. 8, 2026.

[4] R. Verschueren et al., "acados — a modular open-source framework for fast embedded optimal control," *Mathematical Programming Computation*, 2021. [Online]. Available: https://publications.syscop.de/Verschueren2021.pdf.

[5] embotech AG, "FORCESPRO documentation," *embotech*. [Online]. Available: https://forces.embotech.com/Documentation/. Accessed: Aug. 8, 2026.

[6] MathWorks, "About SIL and PIL Simulations," *Embedded Coder Documentation*. [Online]. Available: https://www.mathworks.com/help/ecoder/ug/about-sil-and-pil-simulations.html. Accessed: Aug. 8, 2026.

[7] P. Steiner et al., "SymForce: Symbolic Computation and Code Generation for Robotics Applications," in *Proc. Robotics: Science and Systems (RSS)*, 2022.

[8] M. Giftthaler, M. Neunert, M. Stauble, and J. Buchli, "The Control Toolbox — An Open-Source C++ Library for Robotics, Optimal and Model Predictive Control," in *Proc. IEEE Int. Conf. Simulation, Modeling, and Programming for Autonomous Robots (SIMPAR)*, 2018, pp. 123–129, doi: 10.1109/SIMPAR.2018.8376281.

[9] J. A. E. Andersson, J. Gillis, G. Horn, J. B. Rawlings, and M. Diehl, "CasADi — A software framework for nonlinear optimization and optimal control," *Mathematical Programming Computation*, 2019.

[10] ArduPilot, *AP_HAL_ChibiOS/hwdef/common/malloc.c*: confirms dynamic heap allocation via a custom multi-region DMA-aware allocator on ArduPilot's ChibiOS bare-metal target. [Online]. Available: https://github.com/ArduPilot/ardupilot/blob/master/libraries/AP_HAL_ChibiOS/hwdef/common/malloc.c. Accessed: Aug. 8, 2026.

[11] PX4 Development Team, "Using the ECL EKF," *PX4 Developer Guide*. [Online]. Available: https://docs.px4.io/main/en/advanced_config/tuning_the_ecl_ekf. Accessed: Aug. 8, 2026.

[12] P. Sopasakis et al., "Open source implementation of PANOC and OpEn," *arXiv preprint*, arXiv:2003.00292, 2020.

---

### 10. Revision History

| Revision | Date | Author | Description of Changes |
|:---|:---|:---|:---|
| 1.0 | August 8, 2026 | @MitchellDScott | Initial draft: positions control-rs against CT, acados, FORCES Pro, CVXGEN, SymForce, PX4/ArduPilot, ros2_control, and Simulink Embedded Coder; states native-authoring/no-required-codegen/no_alloc/compile-time-verification constraints for classical_tools/modern_tools/robust_tools/nonlinear_tools. |
| 1.1 | August 8, 2026 | @MitchellDScott | Removed fabricated "control-rs research, 2026" self-citations (review feedback); replaced with primary-source citations (ArduPilot, minikalman-rs, adskalman-rs, pid-ctrl) or direct references to the research artifact's open-questions list; renumbered References. |
