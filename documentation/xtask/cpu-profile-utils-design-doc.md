# CPUProfiler Design Document

![Date Badge](https://img.shields.io/badge/Date-July_18,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Reviewed-yellow)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

### 1. Introduction

ETS requires target-specific hooks to measure performance
metrics (clock cycles, elapsed execution time and stack usage). By defining a
unified `CPUProfiler` trait within the `control-rs-ets::profiler` module,
the server delegates the implementation of primitive hardware
hooks to the end-user or silicon vendor.

---

### 2. Requirements

#### Functional Requirements

- **FR-1 — Cycle Count**: Read a monotonically increasing CPU cycle count (
  `get_cycles`), robust to hardware counter wraparound.
- **FR-2 — Timer**: Report elapsed time in nanoseconds (`get_nanos`).
- **FR-3 — Stack Pointer Read**: Read the active stack pointer (`get_sp`) and
  the stack boundary (`get_stack_end`).
- **FR-4 — Stack Painting/Scanning**: Paint the stack with a sentinel pattern
  and scan for the peak-usage high-water mark.
- **FR-5 — Interrupt Control**: Run a closure with interrupts disabled and
  disable interrupts permanently.
- **FR-6 — Default Implementations**: Provide defaults for targets that do not
  support these features.

#### Non-Functional Requirements

- **NFR-1 — Deterministic Low Overhead**: Cycle and timer hooks must execute
  within a few clock cycles so measurements do not perturb the code under test.

#### Constraints

- **C-1 — Strict `#![no_std]`, Zero Heap**: Consistent with the rest of
  `control-rs-ets`.
- **C-2 — Target Architectures**: ARM Cortex-M (ARMv6/7/8-M) and RISC-V (
  RV32/RV64); ARMv6-M has no DWT cycle counter (§8).

---

### 3. Technical Overview

This project provides hardware specific hooks through a trait to keep the
server hardware-agnostic.

---

### 4. Core Architecture

### 4.1. The CPUProfiler Trait

The core of the abstraction is the `CPUProfiler` trait, defined in
`control-rs-ets::profiler`:

```rust
pub trait CPUProfiler {
    /// Disables interrupts and runs the given closure, returning its result.
    fn disable_interrupts<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        f()
    }

    /// Disables interrupts permanently.
    fn disable_interrupts_permanently(&self) {}

    /// Exits the application/environment using target-specific mechanisms.
    fn exit(&self) -> ! {
        loop {}
    }

    /// Get the current CPU cycle count.
    fn get_cycles(&self) -> u64;

    /// Get the current time in nanoseconds.
    fn get_nanos(&self) -> u64;

    /// Get the current stack pointer.
    fn get_sp(&self) -> usize;

    /// Get the end of the current stack.
    fn get_stack_end(&self) -> usize;

    /// Paints the stack below the given stack pointer.
    unsafe fn paint_stack(&self, sp: usize);

    /// Reads the peak stack usage (in bytes) since the stack was painted, relative to the given stack pointer.
    unsafe fn read_stack_peak(&self, sp: usize) -> u32;

    /// Resets the CPU/system.
    fn reset(&self) -> ! {
        loop {}
    }
}
```

---

### 4.2. Implementations and Mock Profilers

Rather than exposing a default stub struct, target platforms implement
`CPUProfiler` directly. For host-side testing, the codebase provides mock
profilers that return zeroed/dummy metrics:

```rust
pub struct HostCPUProfiler;

impl CPUProfiler for HostCPUProfiler {
    fn exit(&self) -> ! {
        panic!("exit called");
    }
    fn get_cycles(&self) -> u64 { 0 }
    fn get_nanos(&self) -> u64 { 0 }
    fn get_sp(&self) -> usize { 0 }
    fn get_stack_end(&self) -> usize { 0 }
    fn reset(&self) -> ! {
        panic!("reset called");
    }
}
```

---

### 4.3. User implementation

The developer implements `CPUProfiler` for their target platform. For
example, on an ARM Cortex-M architecture:

```rust
struct CortexMProfiler;

impl CPUProfiler for CortexMProfiler {
    fn get_cycles(&self) -> u64 {
        // Read DWT cycle counter
        read_dwt_cyccnt()
    }

    fn get_nanos(&self) -> u64 {
        // Retrieve time in nanoseconds
        read_systick_nanos()
    }

    fn get_sp(&self) -> usize {
        let sp: usize;
        unsafe {
            core::arch::asm!("mov {}, sp", out(reg) sp);
        }
        sp
    }

    fn get_stack_end(&self) -> usize {
        // Retrieve linker script symbol address
        0
    }

    fn disable_interrupts_permanently(&self) {
        cortex_m::interrupt::disable();
    }

    fn reset(&self) -> ! {
        cortex_m::peripheral::SCB::sys_reset();
    }
}
```

`reset()` performs a host-commanded warm reboot (e.g. `Command::TryReset`).
Crash recovery does not use it: ETS's panic path relies on watchdog
starvation for a hard reset, since a soft `SCB` reset leaves peripherals and
active DMA running (see `embedded-test-server-design.md` §4.4).

---

### 5. Alternatives

**Retain ClientClock and TestExecutor**: Previously, this was split into two
interfaces: `ClientClock` (system clock
abstraction) and `TestExecutor` (orchestrating the full execution lifecycle of a
test). However, implementing the full execution logic in `TestExecutor` on the
target side was verbose, repetitive and error-prone.

To simplify target-side integration, these hooks were consolidated into a single
trait: `CPUProfiler`. Users only implement primitive hardware hooks (
retrieving cycle counts, reading time, getting the stack pointer and
identifying stack boundaries). The ETS `Server` implements the generic
execution wrapper, timing calculations and metric telemetry reporting in a
platform-agnostic manner.

**Heap Profiling**: Adding traits for tracking dynamic memory allocation, if
applicable to the targets.

**Multicore Profiling**: Expanding the trait to handle multicore execution and
synchronization metrics.

**Existing Crates**:
[embedded-profiling](https://github.com/TDHolmes/embedded-profiling) is the
only surveyed complete profiling crate in this niche
(`documentation/xtask/research/results/cpu-profiling-utils.json`). Its
`EmbeddedProfiler` trait covers roughly one-fifth of the required surface —
elapsed time via `read_clock()`/snapshots only, with no cycle counting, stack
introspection or interrupt control — and its last release (0.3.0) was
published 2021-12-25. Rejected; implementing the trait in-house allows it to
be tightly coupled with the server.

**`critical-section` for Interrupt Control**: `critical_section::with(|cs| ...)`
is structurally identical to `disable_interrupts<F, R>` and
`CortexMProfiler` already delegates to `cortex_m::interrupt::free`, one of its
backends. Depending on the crate directly (it is already a transitive
dependency) would reduce the interrupt-control surface to a thin wrapper
across Cortex-M and RISC-V instead of per-architecture logic.

**Provided GPIO Implementation**: Implement a profiler that automatically
toggles an output pin when a test starts and stops. Then read this test pin
from the host (or another controller).

**Stack Painting**: The paint-then-scan technique matches
[Rapita RapiTest](https://www.rapitasystems.com/blog/how-measure-stack-usage-through-stack-painting-rapitest),
Zephyr's `k_thread_stack_space_get()` and the FreeRTOS high-water mark — a
recognized industry technique, not an ad hoc invention.

**Static Stack Usage Analysis**: Tools like `cargo-call-stack` evaluate LLVM-IR
to bound maximum stack allocation at compile time. This is complementary to
runtime painting, not a substitute. The lighter-weight `stack-sizes` crate
(per-function numbers, same unstable `-Z emit-stack-sizes` flag) is a narrower
alternative.
[flip-link](https://github.com/knurling-rs/flip-link) addresses a third,
distinct failure mode — link-time overflow protection via a flipped RAM
layout — and is a candidate on Cortex-M (no RISC-V support per its README).

---

### 6. Verification & Validation

Developers must validate their custom hardware hooks using automated
on-target testing. Ideally these will be available to end users.

* **Runner Accuracy**: Execute a mathematically proven delay loop (e.g.,
  utilizing `cortex_m::asm::delay`). Assert that the numerical delta
  recorded by `get_cycles()` correlates linearly with the elapsed time
  calculated by `get_nanos()` based on the configured core clock frequency.
* **Memory Bounds**: Validate that the `paint_stack` and
  `read_stack_peak` implementations respect the physical RAM boundaries defined
  by the compiler's linker script (specifically `_stack_start` and
  `_stack_end`).
* **Interrupt Latency**: Measure the overhead introduced by the
  generic wrapper and the disable_interrupts closure. Quantify the execution
  overhead introduced by the concurrency wrapper to ensure it does not induce
  unacceptable execution jitter.

---

### 7. Performance and Resource Considerations

* **Hook latency**: The `get_cycles()` and `get_nanos()` hooks must execute
  deterministically and as close to zero-overhead as possible (typically within
  a few clock cycles).
* **Closure Execution Overhead**: The `disable_interrupts<F, R>` wrapper
  introduces critical section overhead. Implementations must minimize the setup
  and teardown instructions around the closure to avoid artificially elevating
  interrupt latency or masking real-time deadlines.
* **Scan Time Complexity**: The `read_stack_peak()` method relies on scanning
  memory linearly for sentinel values. Because this operation is $O(N)$ relative
  to the size of the stack, it should be easy to verify.

---

### 8. Risks and Open Questions

* **Register Overflows**: `CortexMProfiler::get_cycles()` casts the 32-bit DWT
  counter to `u64` with no wraparound handling; at typical core clocks it
  wraps within seconds to tens of seconds. `dwt-systick-monotonic`'s `extend`
  technique (compare against last reading, track the high 32 bits, driven by a
  periodic interrupt) is a proven reference implementation; verify against
  `rtic-monotonics` (its actively maintained successor) before committing,
  since `dwt-systick-monotonic` last released in 2022. Open question whether
  ETS's polling cadence is frequent enough to observe every
  wraparound without a dedicated interrupt. On RISC-V, RV32's `mcycle` is also
  32 bits (with a separate `mcycleh` CSR) — whether `RiscvProfiler` combines
  them into a full 64-bit value is unverified.
* **ARMv6-M Support**: `cortex-m` gates `DWT::cycle_count()` behind
  `#[cfg(not(armv6m))]` — Cortex-M0/M0+ have no DWT cycle counter, so
  `CortexMProfiler` as written fails to compile on those cores. Open question
  whether ARMv6-M is in scope: if so, a SysTick-derived fallback (the
  `rtic-monotonics` pattern, lower resolution) is required; if not, record the
  restriction as an explicit non-goal.
* **Stack Bounds Calculation**: How will the user implementation safely
  determine the absolute bottom of the stack. It should be available through
  the linker, but this is a complex integration task.
* **Clock Skew / Power Saving**: If the user needs to run tests in a reduced
  power or low compute state the profiler may become less accurate.
* **Unsafe Code Proliferation**: Users must implement unsafe functions for
  their profiler to work. Link-time protection (`flip-link`) can complement,
  but not remove, the unsafe paint/scan implementations.

---

### 9. Development Plan

| Task / Feature                           | Description                                                                                                     | Status / Effort |
|:-----------------------------------------|:----------------------------------------------------------------------------------------------------------------|:----------------|
| **Step 1: Trait & Target Impls**         | Define `CPUProfiler` and implement `CortexMProfiler`/`RiscvProfiler` in `control-rs-ets::profiler`.             | Shipped         |
| **Step 2: Overflow-Safe Cycle Counting** | Add DWT wraparound handling (extend technique) and verify RV32 `mcycle`/`mcycleh` combination.                  | 0.5 day         |
| **Step 3: ARMv6-M Decision**             | Either add a SysTick-only fallback for Cortex-M0/M0+ or document the restriction as a non-goal.                 | 0.5 day         |
| **Step 4: CI Static Analysis**           | Integrate `cargo-call-stack` with explicit scoping of its SysTick/inline-asm blind spots; evaluate `flip-link`. | 1.0 day         |

---

### 10. Revision History

| Revision | Date           | Author          | Description                                                                                                                           |
|:---------|:---------------|:----------------|:--------------------------------------------------------------------------------------------------------------------------------------|
| 1.0      | July 18, 2026  | @MitchellDScott | Initial specification for the `CPUProfiler` trait, hardware timer abstractions, and benchmark execution.                              |
| 1.1      | August 6, 2026  | @MitchellDScott | Hardware targets: added DWT cycle counting for Cortex-M and CSR `mcycle` for RISC-V with static call-stack analysis.                 |
