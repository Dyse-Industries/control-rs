# ADR-0002: Decouple Core Library Crates from Embedded Test Server Runtime

![Date Badge](https://img.shields.io/badge/Date-November_13,_2026-blue)
![Status Badge](https://img.shields.io/badge/ADR%20Status-Proposed-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

## Context

`control-rs` mathematical models and control toolboxes are developed
`#![no_std]`
with minimal dependencies. Previously, the root `control-rs` crate carried an
optional `ets` feature that pulled in `control-rs-macros` and `control-rs-ets`,
decorating internal unit test modules across `src/` with
`#[cfg_attr(not(test), control_rs_macros::ets_suite)]`. Downstream hardware
targets
and QEMU runners linked these suites by re-exporting internal module paths.

This introduced procedural macro compilation overhead into core library crates,
coupled pure mathematical code to target server runtime types and linker script
fragments (`ets_suites.x`), and conflated host unit testing (`cargo test` via
`libtest`) with embedded hardware execution. Furthermore, design documents like
`classical-control-design.md` carried active traced constraints (C-6 / VC-26.1)
enforcing macro attribute presence on internal test modules. How should
`control-rs`
structure on-target testing without coupling library crates to the target
execution
runtime?

---

## Decision

Decouple the root `control-rs` library crate and all inner mathematical
submodules from `control-rs-ets` and `control-rs-macros`:

1. Remove `feature = "ets"` and the optional dependencies `control-rs-macros`
   and `control-rs-ets` from the root `Cargo.toml`.
2. Strip `#[ets_suite]` attributes and `pub mod suites` re-exports from all
   library modules in `src/`. Internal unit tests remain standard private
   `#[cfg(test)] mod tests` verified on host via `libtest` and `kani`.
3. Relocate on-target test suites and HIL fixtures to dedicated integration
   targets and example binaries (`examples/qemu`, `examples/teensy4`) that
   consume `control-rs` as an external library.
4. Retire `classical-control#C-6` and `classical-control#VC-26.1` in
   `classical-control-design.md` and `.cargo/trace/trace.toml`.

---

## Consequences

- Good: Core library crates remain strictly zero-dependency beyond `libm`,
  eliminating macro compilation overhead for downstream library consumers.
- Good: Host unit tests are isolated from embedded target dispatch mechanics and
  linker sections.
- Good: The core library unit tests are set up as basic cargo tests, no more
  conditional compilation noise or procedural macro clutter.
- Good: On-target test harnesses construct tests against the public API of
  `control-rs` rather than coupling to internal module layouts.
- Bad: On-target firmware images cannot include core library tests, those
  are fixed in `src/` now.
- Follow-up: Retire `classical-control#C-6` and `classical-control#VC-26.1` in
  `documentation/control-toolboxes/classical-control-design.md` and
  `.cargo/trace/trace.toml`.
- Follow-up: Retarget ETS lead and subsystem designs to define suites within
  target integration targets rather than library internals.

---

## Rejected Options

- Retain `feature = "ets"` with optional macros: Preserves macro dependencies
  and conditional compilation noise across pure algorithmic code.
- Provide a no-op macro stub: Hides dependency coupling without addressing the
  architectural conflation of host unit tests and target hardware execution.
