# CPUProfiler Design Document

![Date Badge](https://img.shields.io/badge/Date-September_9,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Approved-brightgreen)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

### 1. Introduction

ETS requires target-specific hooks to measure performance
metrics (clock cycles, elapsed execution time and stack usage). By defining a
unified `CPUProfiler` trait within the `control-rs-ets::profiler` module,
the server delegates the implementation of primitive hardware
hooks to the end-user or silicon vendor.

---

### 2. Requirements

#### 2.1 Functional Requirements

- **FR-1 — Execution Duration Measurement**: Read a monotonically increasing CPU cycle count (
  `get_cycles`), robust to hardware counter wraparound.
- **FR-2 — Wall-Clock Timestamp Query**: Report elapsed time in nanoseconds (`get_nanos`).
- **FR-3 — Call-Stack Watermark Inspection**: Read the active stack pointer (`get_sp`) and
  the stack boundary (`get_stack_end`).
- **FR-4 — High-Watermark Stack Profiling**: Paint the stack with a sentinel pattern
  and scan for the peak-usage high-water mark.
- **FR-5 — Atomic Section Profiling**: Run a closure with interrupts disabled and
  disable interrupts permanently.
- **FR-6 — Fallback Platform Provision**: Provide defaults for targets that do not
  support these features.

#### 2.2 Non-Functional Requirements

- **NFR-1 — Deterministic Low Overhead**: Cycle and timer hooks must execute
  within a few clock cycles so measurements do not perturb the code under test.

#### 2.3 Constraints

- **C-1 — Strict `#![no_std]`, Zero Heap**: Consistent with the rest of
  `control-rs-ets`.
- **C-2 — Target Architectures**: ARM Cortex-M (ARMv6/7/8-M) and RISC-V (
  RV32/RV64); ARMv6-M has no DWT cycle counter (§8).

---

### 3. Technical Overview

This project provides hardware specific hooks through a trait to keep the
server hardware-agnostic.

---

### 4. Architecture

#### 4.1. The CPUProfiler Trait

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


The Cortex-M cycle source is the DWT cycle counter, exposed by the `cortex-m`
peripheral access crate as a `cyccnt` register that is absent on Armv6-M
[1]. That absence is why FR-1 is a trait method rather than a
direct register read: a target without DWT supplies its own implementation.

Interrupt control (FR-5) is expressed through the portable critical-section
abstraction, which exists because there is otherwise no universal API across
targets [2].

Stack measurement (FR-4) uses painting, "a runtime technique for estimating the
maximum stack depth a task has reached" [3]. The approach is "very
common in embedded systems and is used by both FreeRTOS and Zephyr RTOS"
[4], and Zephyr exposes the same idea through
`k_thread_stack_space_get`, which reports unused stack when its init-stacks
options are enabled [5].

#### 4.2. Implementations and Mock Profilers

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

#### 4.3. User implementation

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

`reset()` performs a host-commanded warm reboot (for example, `Command::TryReset`).
Crash recovery does not use it: ETS's panic path relies on watchdog
starvation for a hard reset, since a soft `SCB` reset leaves peripherals and
active DMA running (see `embedded-test-server-design.md` §4.4).

---

### 5. Alternatives

* **Adopt an existing profiling crate**: Rejected. Crates such as `cortex-m-funnel`
  and `embedded-profiling` address runtime profiling in `no_std` environments
  [6] via traits requiring a clock read and an output sink [7]. However,
  `embedded-profiling` has not released since version 0.3.0 in December 2021 [8],
  and its trait covers only elapsed time snapshots without cycle counting, stack
  introspection, or interrupt control. Reimplementing the unified `CPUProfiler`
  trait in-house minimizes dependencies and tightly couples lifecycle hooks to ETS.
* **Static stack analysis instead of painting**: Rejected as the primary
  measure, retained as a complement. `cargo-call-stack` is a whole-program stack
  analyzer but relies on experimental `-Z stack-sizes` [9], and inline assembly
  breaks LLVM's stack usage analysis [9]. Compiler-emitted per-function usage
  from `-fstack-usage` does not readily analyze nested call trees [4]. Static
  analysis is therefore delegated to CI (`documentation/ci/static-analyzer-design.md`),
  with runtime stack painting providing empirical high-water marks and link-time
  tools (`flip-link`) providing complementary overflow protection.
* **Tick-based timekeeping**: Rejected. A tickless monotonic avoids periodic
  interrupts to count ticks [10], whereas SysTick-based periodic tick generation
  incurs high interrupt rates [11] that perturb execution duration and jitter
  measurements.
* **Host-side trace tooling**: Rejected as a requirement. External tools such as
  SEGGER SystemView analyze and verify embedded systems from the host [12], but
  impose vendor-specific hardware or debug probes, violating open emulator and
  target independence constraints.
* **Separated clock and test executor traits**: Rejected. Splitting profiling
  into distinct clock and test executor interfaces requires target implementations
  to duplicate test execution and timing math. Consolidating hooks into
  `CPUProfiler` restricts target code to raw hardware reads while the ETS server
  orchestrates execution and telemetry.
* **Direct `critical-section` dependency for interrupt control**: Considered for
  future deduplication. `critical_section::with(|cs| ...)` mirrors
  `disable_interrupts<F, R>` and allows shared implementations across Cortex-M
  and RISC-V.
* **External GPIO pin toggling**: Rejected for automated test telemetry. Toggling
  hardware pins allows external instrumentation (for example, logic analyzers) to measure
  timings, but requires dedicated physical I/O and host capture hardware,
  precluding headless CI and QEMU execution.
* **Dynamic heap and multicore profiling**: Deferred as non-goals for single-core
  deterministic `no_std` / `no_alloc` targets.

### 6. Verification & Validation

#### 6.1 Approach

The implementation must produce evidence that cycle and time sources agree with
each other and with the configured clock, that stack measurement never reads
outside the stack region, and that the profiling wrapper's own overhead is
small enough not to distort what it measures.

| Method | Mechanism |
|:-------|:----------|
| Back-to-back comparison | `get_cycles()` delta against `get_nanos()` across a proven delay loop (`cortex_m::asm::delay`) |
| Requirements-based test | `#[test]` on mock profilers over wraparound, zero-length and maximal intervals |
| Static analysis | Linker-symbol bounds check on `paint_stack` and `read_stack_peak` against `_stack_start` and `_stack_end` |
| Metamorphic relation | Painting then running a known-depth call chain; measured peak must increase monotonically with depth |
| Resource usage evaluation | Overhead of the generic wrapper and the critical-section closure, measured in cycles |
| On-target execution | ETS suites on every supported architecture |
| Static analysis | `cargo clippy-ci`; inspection for allocation |
| Coverage measurement | `cargo coverage` on mock-profiler paths |

Target: 85% line coverage of the host-testable profiler logic, measured with
`cargo coverage`.

Excluded: target-specific register reads, which require hardware or an
emulator; and the painting routine itself, whose effect is observed through the
peak measurement rather than through line execution.

* **On-target suite**: Run the profiling suite on a physical Teensy 4.1 and
  under QEMU for each architecture, confirming cycle, duration and stack
  figures are produced and self-consistent.
* **User implementation**: Implement the trait for a board outside this
  workspace following §4.3 and confirm the ETS reports its telemetry unchanged.

#### 6.2 Acceptance

| Claim | Oracle | Measure | Bound |
|:------|:-------|:--------|:------|
| Cycle and time agreement | `cortex_m::asm::delay(n)` at a known core frequency | Relative error between the cycle delta and the derived nanoseconds | ≤ 1% over intervals ≥ 10,000 cycles |
| Counter wraparound | Mock profiler stepped across the 32-bit boundary | Reported delta | Correct modulo the counter width, never negative |
| Stack bounds | Linker symbols `_stack_start`, `_stack_end` | Addresses touched by paint and scan | All within `[_stack_end, _stack_start]` |
| Peak monotonicity | Call chains of increasing known depth | Reported peak | Non-decreasing with depth |
| Wrapper overhead | Empty measured region | Cycles attributed to the wrapper | ≤ 1% of the shortest measured interval in the suite |

The cycle bound applies only above 10,000 cycles because at shorter intervals
the wrapper overhead is a significant fraction of the measurement, which is the
same reason the overhead itself is bounded.

#### 6.3 Limits

- Absolute accuracy of cycle counts under emulation. QEMU states no cycle or
  timing model, so emulated figures are indicative per
  `documentation/ci/ci-design.md` C-2; only hardware runs measure.
- Stack usage of interrupt handlers and inline assembly. Painting measures what
  ran on the measured stack; static analysis of these paths is explicitly
  unavailable, since inline assembly "breaks LLVM's stack usage analysis"
  [9].
- Behaviour under preemption. Measurements assume the measured region is not
  preempted; nothing detects or corrects for a context switch mid-measurement.
- Cross-architecture comparability. Cycle counts from Cortex-M and RISC-V are
  not claimed to be comparable to each other.
- Long-interval measurement. Nothing bounds error accumulation across intervals
  longer than one counter period.

### 7. Performance and Resource Considerations

* **Hook latency**: The `get_cycles()` and `get_nanos()` hooks must execute
  deterministically and as close to zero-overhead as possible (typically within
  a few clock cycles).
* **Closure Execution Overhead**: The `disable_interrupts<F, R>` wrapper
  introduces critical section overhead. Implementations must minimize the setup
  and teardown instructions around the closure to avoid artificially elevating
  interrupt latency or masking real-time deadlines.
* **Scan Time Complexity**: The `read_stack_peak()` method relies on scanning
  memory linearly for sentinel values. This operation is $O(N)$ relative to the
  size of the stack and executes during post-test reporting to avoid perturbing
  active test execution.

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
* **Stack Bounds Calculation**: Determining the absolute bottom of the stack
  safely requires linker script symbols or target-specific runtime boundaries.
  Board-level implementations must standardize linker symbol exports to prevent
  boundary miscalculation.
* **Clock Skew and Power Saving**: Running tests in reduced-power or
  dynamic-frequency states alters hardware cycle counters and timer prescalers,
  degrading profiling accuracy. Test suites requiring high-precision timing
  must enforce fixed core frequencies.
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
| **Step 4: CI Static Analysis Integration** | Integrate with `control-rs-static-analyzer` (`documentation/ci/static-analyzer-design.md`); evaluate `flip-link` complementary protection. | 1.0 day |

---

### 10. Revision History

| Revision | Date           | Author          | Description                                                                                                          |
|:---------|:---------------|:----------------|:---------------------------------------------------------------------------------------------------------------------|
| 1.0      | July 18, 2026  | @MitchellDScott | Initial specification for the `CPUProfiler` trait, hardware timer abstractions, and benchmark execution.             |
| 1.1      | August 6, 2026 | @MitchellDScott | Hardware targets: added DWT cycle counting for Cortex-M and CSR `mcycle` for RISC-V with static call-stack analysis. |
| 1.2 | September 9, 2026 | @MitchellDScott | Project split: relocated to `documentation/ets/cpu-profiler-design.md`; research pair renamed to match the slug. |
| 1.3      | September 9, 2026 | @MitchellDScott | Citation pass: grounded the DWT, critical-section and stack-painting mechanisms, added four evidence-backed alternatives, restructured §6 from prose into 6.1-6.7 per `vv-standards.md` with numeric acceptance bounds. |
| 1.4      | September 9, 2026 | @MitchellDScott | Structural hardening: numbered §2 subsections 2.1-2.3, reconciled Step 4 with static-analyzer-design, standardized reference ordering. |
| 1.5      | September 9, 2026 | @MitchellDScott | Dropped the author-year / `[n]` mapping table. |

---

## References

[1] rust-embedded, "cortex-m," *rust-embedded/cortex-m*. [Online]. Available: https://docs.rs/cortex-m/latest/src/cortex_m/peripheral/dwt.rs.html. Accessed: Aug. 7, 2026.

[2] rust-embedded, "critical-section," *rust-embedded/critical-section*. [Online]. Available: https://github.com/rust-embedded/critical-section. Accessed: Aug. 7, 2026.

[3] Antoine Colin, "How to measure stack usage through stack painting with RapiTest," *Rapita Systems Blog*. [Online]. Available: https://www.rapitasystems.com/blog/how-measure-stack-usage-through-stack-painting-rapitest. Accessed: Aug. 7, 2026.

[4] Noah Pendleton, "Measuring Stack Usage the Hard Way," *Interrupt (Memfault blog)*. [Online]. Available: https://interrupt.memfault.com/blog/measuring-stack-usage. Accessed: Aug. 7, 2026.

[5] Zephyr Project, "Threads," *Zephyr Project Documentation*. [Online]. Available: https://docs.zephyrproject.org/latest/kernel/services/threads/index.html. Accessed: Aug. 7, 2026.

[6] TDHolmes, "embedded-profiling," *TDHolmes/embedded-profiling*. [Online]. Available: https://github.com/TDHolmes/embedded-profiling. Accessed: Aug. 7, 2026.

[7] TDHolmes, *embedded-profiling* (Version 0.3.0). [Online]. Available: https://docs.rs/embedded-profiling/latest/embedded_profiling/trait.EmbeddedProfiler.html. Accessed: Aug. 7, 2026.

[8] TDHolmes, *embedded-profiling* (Version 0.3.0). [Online]. Available: https://crates.io/api/v1/crates/embedded-profiling. Accessed: Aug. 7, 2026.

[9] japaric, "cargo-call-stack," *japaric/cargo-call-stack*. [Online]. Available: https://github.com/japaric/cargo-call-stack. Accessed: Aug. 7, 2026.

[10] rtic-rs, "dwt-systick-monotonic," *rtic-rs/dwt-systick-monotonic*. [Online]. Available: https://github.com/rtic-rs/dwt-systick-monotonic/blob/master/src/lib.rs. Accessed: Aug. 7, 2026.

[11] rtic-rs, *rtic-monotonics* (Version 2.2.1). [Online]. Available: https://docs.rs/rtic-monotonics/latest/rtic_monotonics/systick/. Accessed: Aug. 7, 2026.

[12] SEGGER Microcontroller GmbH, "What is SystemView?," *SEGGER product documentation*. [Online]. Available: https://www.segger.com/products/development-tools/systemview/technology/what-is-systemview/. Accessed: Aug. 7, 2026.
