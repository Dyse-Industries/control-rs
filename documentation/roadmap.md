# `control-rs` Roadmap

[Documentation index](README.md)

- [x] **PR1**: Workspace Foundations & Strict Lint Baseline
- [x] **PR2**: ETS Host Extraction & Terminal Dashboard ([#55](https://github.com/Dyse-Industries/control-rs/pull/55); [ets-host](ets-host/ets-host-design.md), [tui](tui/tui-design.md))
- [x] **PR3**: CI Quality Gate Infrastructure (`control-rs-ci`) ([#58](https://github.com/Dyse-Industries/control-rs/pull/58); [ci](ci/ci-design.md))
  - [x] **PR3-1**: Multi-Oracle Differential Test Harness (`control-rs-compare`) ([#59](https://github.com/Dyse-Industries/control-rs/pull/59); [cross-compare](vv/cross-compare-design.md))
  - [ ] **PR3-2**: CI Gate — Performance & Benchmarking (`bench`)
  - [ ] **PR3-3**: CI Gate — Property-Based Testing (`prop-test`) ([ets-prop-test](vv/ets-prop-test-design.md))
  - [ ] **PR3-4**: CI Gate — Requirement Traceability (`trace`) ([requirement-traceability](vv/requirement-traceability-design.md))
- [ ] **PR4**: Classical Control Synthesis & Math Core (`src/classical_tools`)
- [ ] **PR5**: Modern Control Toolbox & State Observers (`src/modern_control`)
- [ ] **PR6**: System Identification (SysID) & Frequency Estimation (`src/sysid`)
- [ ] **PR7**: Safety Validation & Run-Time Assurance (`src/validation`)
- [ ] **PR8**: Hardware Acceleration & Architecture Subprograms (`src/math/subprograms`) ([subprograms](math/subprograms-design.md))
- [ ] **PR9**: CI Workflow Hardening & Multi-Target QEMU Emulation
- [ ] **PR10**: Automated Git Release System & Release Pipeline
- [ ] **PR11**: Flight Examples, Documentation Consolidation & Initial Release (`v0.1.0`)
