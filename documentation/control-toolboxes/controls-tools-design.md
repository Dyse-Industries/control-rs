# Controls Ecosystem Positioning & Architecture (Design Document)

![Status: Draft](https://img.shields.io/badge/status-draft-orange)
![Date Badge](https://img.shields.io/badge/Date-September_15,_2026-blue)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

This document is the ecosystem parent for `classical_tools`, `modern_tools`,
`robust_tools`, and `sysid`. It states the jobs those modules share and the
standing constraints they inherit. Algorithm inventories belong in the
sibling designs.

Primary usage scenarios:

- A synthesis routine consumes and emits the crate's numerical models
  without a foreign wrapper.
- A controller or estimator produced by one toolbox is usable in simulation
  or on-target execution without a conversion step.
- A reviewer checks that a sibling design inherits C-1..C-4 rather than
  restating them.

---

### 2. Requirements

#### 2.1 Functional Requirements

- **FR-1 — Native models in and out**: Toolbox algorithms consume and emit
  the crate's numerical models. A wrapper type or foreign memory layout is
  not required to call them.
- **FR-2 — Downstream use without conversion**: A controller or estimator
  synthesized in one toolbox is usable as an input to simulation or to
  another toolbox without a conversion step.

#### 2.2 Non-Functional Requirements

- **NFR-2 — Real-time determinism**: On-target toolbox steps have bounded
  latency and do not wait on non-deterministic operating-system services.
  The former NFR-1 (compile-time shape) is a crate-wide constraint
  discharged in numerical-model designs and is withdrawn here.

#### 2.3 Constraints

These four constraints are the standing text sibling designs point at. They
are not restated in those documents.

- **C-1 — Native authorability**: Every controller and estimator construct
  is expressible directly in Rust and compiles for bare-metal targets without
  an intermediate code-generation pass.
- **C-2 — No host-to-target codegen**: Toolbox modules must not depend on
  external MATLAB, Simulink, Python, or CasADi toolchains emitting vendored C
  or Rust source.
- **C-3 — No dynamic allocation**: Algorithmic kernels adhere to `#![no_std]`
  and `no_alloc`, using caller-provided static buffers, stack allocations, or
  storage views.
- **C-4 — Offline-only symbolic and heavy optimization**: General multi-block
  SDP/LMI solvers and symbolic Jacobians stay host-side and must not be
  runtime dependencies on target.

---

### 6. Verification & Validation

#### 6.1 Approach

This document specifies
inherited bounds; numeric kernels are verified in sibling designs.

- Sibling toolbox designs inherit C-1..C-4 by pointer rather than restating
  them.
- Toolbox public APIs accept and return native numerical models.
- Embedded toolbox builds contain no heap allocation symbols.

| Method | Mechanism |
|:-------|:----------|
| Inspection | Sibling design §2.3 against this document's C-1..C-4 |
| Inspection | Sibling public types consume and emit crate models |
| Static analysis | `cargo clippy-ci` on toolbox modules |
| Resource usage evaluation | `no_alloc` / allocation-symbol review of embedded toolbox builds |
| On-target execution | ETS toolbox steps |
| Coverage measurement | `cargo coverage` |

This document has no executable crate surface. Coverage targets live in the
sibling module designs. Excluded: this markdown file.

Validation is the existence of at least one host plant that consumes a
toolbox output as a native model: the buck-converter and DC-motor suites in
`classical-tools-examples-design.md`. Modern, robust, and sysid plants are
absent.

#### 6.2 Acceptance

| Claim | Oracle | Measure | Bound |
|:------|:-------|:--------|:------|
| Sibling inherit pointer | Each of `classical-tools`, `modern-control`, `robust-control`, `sysid` §2.3 | Presence of a pointer to this document's C-1..C-4 | Exact: pointer present; no restated C-1..C-4 paragraphs |
| Native model I/O | Sibling public signatures | Foreign wrapper types in the API | None |
| No heap in embedded toolbox builds | `cargo clippy-ci` and allocation-symbol review | `alloc::` / `malloc` / `free` in target artifacts | Zero matches |

#### 6.3 Limits

- **On-target toolbox steps**: no modern, robust, or sysid ETS suite is
  linked into firmware. Classical ETS suites exist in source but are not
  the evidence for this parent document.
- **Withdrawn NFR-1 (compile-time shape)**: a crate-wide constraint
  discharged in numerical-model designs, not a quality of this parent.
- **Modern / robust / sysid native I/O at the type level**: those modules
  are stubs, so FR-1/FR-2 are established for classical tools by inspection
  and remain a plan for the stubs.

---

## References

[1] M. Giftthaler, M. Neunert, M. St{\"a}uble, and J. Buchli, "The Control Toolbox --- An Open-Source C++ Library for Robotics, Optimal and Model Predictive Control," in *Proc. IEEE Int. Conf. Simulation, Modeling and Programming for Autonomous Robots (SIMPAR)*, Brisbane, Australia, 2018, pp. 123--129, doi: 10.1109/SIMPAR.2018.8376281.

[2] R. Verschueren et al., "acados --- a modular open-source framework for fast embedded optimal control," *Mathematical Programming Computation*, vol. 14, pp. 147--183, 2021, doi: 10.1007/s12532-021-00209-0.

[3] P. Steiner et al., "SymForce: Symbolic Computation and Code Generation for Robotics Applications," in *Proc. Robotics: Science and Systems (RSS)*, New York, NY, USA, 2022, doi: 10.15607/RSS.2022.XVIII.042.

[4] P. Sopasakis, E. Fresk, and P. Patrinos, "Open source implementation of PANOC and OpEn," arXiv, Rep. no. arXiv:2003.00292, 2020. [Online]. Available: https://arxiv.org/abs/2003.00292.

[5] ArduPilot Development Team, "AP\_HAL\_ChibiOS/hwdef/common/malloc.c," *GitHub*, 2026. [Online]. Available: https://github.com/ArduPilot/ardupilot/blob/master/libraries/AP_HAL_ChibiOS/hwdef/common/malloc.c.

[6] PX4 Development Team, "Using the ECL EKF," *PX4 Developer Guide*, 2026. [Online]. Available: https://docs.px4.io/main/en/advanced_config/tuning_the_ecl_ekf.

---

### 10. Revision History

| Revision | Date              | Author          | Description                                                    |
|:---------|:------------------|:----------------|:---------------------------------------------------------------|
| 1.0      | September 3, 2026 | @MitchellDScott | Initial requirements and verification outline for ecosystem.   |
| 1.1      | September 15, 2026 | @MitchellDScott | Ecosystem jobs; standing C-1..C-4 as the inherit target; full §6. |
