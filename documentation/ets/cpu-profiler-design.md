# CPUProfiler Design Document

![Date Badge](https://img.shields.io/badge/Date-July_18,_2026-blue)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)
![Design Status Badge](https://img.shields.io/badge/Design%20Status-Stale-lightgrey)

### 1. Introduction

ETS requires target-specific hooks to measure performance metrics (clock cycles,
elapsed execution time, stack usage and loop timing jitter). By defining a
unified `CPUProfiler` trait within the `control-rs-ets::profiler` module, the
server delegates low-level silicon timer and hardware counter hooks to the
board-support crate or target platform.

Under the unified architecture (ADR-0002), `CPUProfiler` serves two testing
modes:

1. **Atomic Kernel Profiling (Layer 0/1)**: Executes benchmark cases within an
   uninterruptible critical section, taking high-resolution DWT cycle counts and
   scanning stack memory high-water marks.
2. **Lifecycle Task Profiling (Layers 1 & 1/2)**: Monitors continuous real-time
   execution at $\ge 1\ \text{kHz}$. Lightweight timestamp snapshots capture
   per-step execution duration and timing jitter with $\le 1\ \mu\text{s}$
   overhead,
   deferring the $O (N)$ stack memory scan until task teardown.

---

### 2. Requirements

#### 2.1 Functional Requirements

- **FR-1 — Execution Duration Measurement**: Read a monotonically increasing CPU
  cycle count (
  `get_cycles`), robust to hardware counter wraparound.
- **FR-2 — Wall-Clock Timestamp Query**: Report elapsed time in nanoseconds
  (`get_nanos`) and whether the profiler provides a clock (`has_clock`); the
  executive paces `Periodic` runs with it (`ets-executive-design.md` §4.2).
- **FR-3 — Call-Stack Watermark Inspection**: Read the active stack pointer
  (`get_sp`) and
  the stack boundary (`get_stack_end`).
- **FR-4 — High-Watermark Stack Profiling**: Paint the stack with a sentinel
  pattern
  and scan for the peak-usage high-water mark.
- **FR-5 — Atomic Section Profiling**: Run a closure with interrupts disabled
  and
  disable interrupts permanently.
- **FR-6 — Fallback Platform Provision**: Provide defaults for targets that do
  not
  support these features.
- **FR-7 — Lifecycle Step Jitter Profiling**: Measure start-to-finish duration
  for
  individual `LifecycleTask::step()` executions, recording step count, total,
  minimum and maximum duration over the session; the mean is derived on the
  host.

#### 2.2 Non-Functional Requirements

- **NFR-1 — Deterministic Low Overhead**: Cycle and timer hooks must execute
  within a few clock cycles so measurements do not perturb the code under test.
- **NFR-2 — Step Profiling Jitter Invariant**: Per-step timestamp capture during
  lifecycle task execution must introduce $\le 1\ \mu\text{s}$ overhead per
  step.

#### 2.3 Constraints

- **C-1 — Strict `#![no_std]`, Zero Heap**: Consistent with the rest of
  `control-rs-ets`.
- **C-2 — Target Architectures**: ARM Cortex-M (ARMv6-M, ARMv7-M, ARMv8-M)
  and RISC-V (RV32/RV64). ARMv6-M has no DWT cycle counter, so its profiler
  derives cycles and time from SysTick (§8).

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
    /// Board identifier for `Telemetry::TargetInfo`; `0` when unknown.
    fn board_id(&self) -> u16 {
        0
    }

    /// Core clock in hertz for `Telemetry::TargetInfo`; `0` when unknown.
    fn core_clock_hz(&self) -> u32 {
        0
    }

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

    /// FPU bits for `Telemetry::TargetInfo`: bit 0 single, bit 1 double,
    /// derived from the compilation target by default.
    fn fpu_flags(&self) -> u8 {
        let single = u8::from(cfg!(target_abi = "eabihf") || cfg!(target_feature = "f"));
        let double = u8::from(cfg!(target_feature = "d"));
        single | (double << 1)
    }

    /// Get the current CPU cycle count.
    fn get_cycles(&self) -> u64;

    /// Get the current time in nanoseconds.
    fn get_nanos(&self) -> u64;

    /// Whether `get_nanos` advances; profilers without a time source return false.
    fn has_clock(&self) -> bool {
        true
    }

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

`board_id`, `core_clock_hz` and `fpu_flags` feed `Telemetry::TargetInfo`
(`ets-link-design.md` §4.8). `CortexMProfiler` and `RiscvProfiler` return
their configured `clock_frequency` from `core_clock_hz`; no shipped profiler
sets a board ID.

---


The Cortex-M cycle source is the DWT cycle counter, exposed by the `cortex-m`
peripheral access crate as a `cyccnt` register that is absent on Armv6-M
(cortex-m, 2023). That absence is why FR-1 is a trait method rather than a
direct register read: a target without DWT supplies its own implementation.

Interrupt control (FR-5) is expressed through the portable critical-section
abstraction, which exists because there is otherwise no universal API across
targets (rust-embedded, 2024).

Stack measurement (FR-4) uses painting, "a runtime technique for estimating the
maximum stack depth a task has reached" (Colin, 2025). The approach is "very
common in embedded systems and is used by both FreeRTOS and Zephyr RTOS"
(Pendleton, 2023), and Zephyr exposes the same idea through
`k_thread_stack_space_get`, which reports unused stack when its init-stacks
options are enabled (Zephyr Project, 2026b).

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
    fn has_clock(&self) -> bool { false }
    fn get_sp(&self) -> usize { 0 }
    fn get_stack_end(&self) -> usize { 0 }
    fn reset(&self) -> ! {
        panic!("reset called");
    }
    // paint_stack, read_stack_peak, disable_interrupts and
    // disable_interrupts_permanently use the trait's provided defaults (FR-6).
}
```

---

#### 4.3 Reference Implementations

`control-rs-ets::profiler` provides reference implementations for ARM Cortex-M
and RISC-V architectures:

##### 4.3.1 CortexMProfiler (ARM Cortex-M)

Applicable to Cortex-M3/M4/M7 targets (for example, Teensy 4.0/4.1 Cortex-M7,
Seeed XIAO nRF52840 Cortex-M4F, Arduino Uno R4 Minima RA4M1 Cortex-M4):

```rust
pub struct CortexMProfiler {
    clock_frequency: u32,
    ticks: &'static AtomicU32,
}

impl CPUProfiler for CortexMProfiler {
    fn core_clock_hz(&self) -> u32 {
        self.clock_frequency
    }

    fn disable_interrupts<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        cortex_m::interrupt::free(|_| f())
    }

    fn disable_interrupts_permanently(&self) {
        cortex_m::interrupt::disable();
    }

    fn get_cycles(&self) -> u64 {
        cortex_m::peripheral::DWT::cycle_count() as u64
    }

    fn get_nanos(&self) -> u64 {
        // Read SysTick current counter and atomic millisecond tick count
        ...
    }

    fn get_sp(&self) -> usize {
        let sp: usize;
        unsafe {
            core::arch::asm!("mov {}, sp", out(reg) sp, options(nomem, nostack, preserves_flags));
        }
        sp
    }

    fn get_stack_end(&self) -> usize {
        unsafe extern "C" {
            static mut _stack_end: u32;
        }
        core::ptr::addr_of!(_stack_end) as usize
    }

    fn reset(&self) -> ! {
        cortex_m::peripheral::SCB::sys_reset();
    }
}
```

##### 4.3.2 RiscvProfiler (RISC-V Architecture)

Applicable to 32-bit and 64-bit RISC-V targets (for example, ESP32-C6 DevKitC-1
RV32IMAC @ 160 MHz):

```rust
pub struct RiscvProfiler {
    clock_frequency: u32,
}

impl CPUProfiler for RiscvProfiler {
    fn core_clock_hz(&self) -> u32 {
        self.clock_frequency
    }

    fn disable_interrupts<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        unsafe { riscv::interrupt::disable(); }
        let res = f();
        unsafe { riscv::interrupt::enable(); }
        res
    }

    fn disable_interrupts_permanently(&self) {
        unsafe { riscv::interrupt::disable(); }
    }

    fn get_cycles(&self) -> u64 {
        riscv::register::mcycle::read() as u64
    }

    fn get_nanos(&self) -> u64 {
        #[cfg(target_arch = "riscv32")]
        let ticks = riscv::register::time::read64();
        #[cfg(target_arch = "riscv64")]
        let ticks = riscv::register::time::read() as u64;

        (ticks as u64).saturating_mul(1_000_000_000) / (self.clock_frequency as u64)
    }

    fn get_sp(&self) -> usize {
        let sp: usize;
        unsafe {
            core::arch::asm!("mv {}, sp", out(reg) sp, options(nomem, nostack, preserves_flags));
        }
        sp
    }

    fn get_stack_end(&self) -> usize {
        unsafe extern "C" {
            static _profiler_stack_start: u32;
        }
        core::ptr::addr_of!(_profiler_stack_start) as usize
    }

    fn reset(&self) -> ! {
        loop {}
    }
}
```

`reset()` performs a host-commanded warm reboot. ETS's panic path calls it after
the host sends `Command::TryReset`; on Cortex-M it is `SCB::sys_reset`, which
can leave peripherals and active DMA running (`ets-executive-design.md` §4.5).
On RISC-V platforms without architectural reset registers, it halts or triggers
watchdog reboot. A watchdog-starvation hard reset comes from the board watchdog
that the user starts in `#[ets_setup]` and the executive feeds
(`ets-executive-design.md` §4.6).

#### 4.4 Profiling Execution Models

The profiler hooks are coordinated across two distinct execution engines:

##### 4.4.1. Atomic Kernel Engine (Layer 0/1)

For single-shot test cases:

1. **Stack painting**: The stack below the active stack pointer is filled with
   sentinel
   bytes (`paint_stack(sp)`).
2. **Critical section entry**: Hardware interrupts are disabled via
   `disable_interrupts`.
3. **Timestamp capture**: Start cycles and wall-clock nanoseconds are recorded
   (`get_cycles`, `get_nanos`).
4. **Invocation**: The test function is executed to completion.
5. **Duration capture**: Stop cycles and nanoseconds are recorded; elapsed time
   is computed.
6. **Watermark evaluation**: `read_stack_peak(sp)` scans backwards through
   memory to
   find the lowest un-overwritten sentinel, determining peak stack usage.
7. **Telemetry dispatch**: Metrics are packed into the run-statistics message
   (`Telemetry::RunStats`).

##### 4.4.2. Lifecycle Task Engine (Layers 1 & 1/2)

For continuous real-time control loops ($\ge 1\ \text{kHz}$) run by the task
runner (`ets-executive-design.md` §4.2):

1. **No stack painting**: Interrupts stay live during a task, and painting below
   the stack pointer would corrupt interrupt frames. Task stack depth is bounded
   by static analysis (`cargo-call-stack`), not measured by painting.
2. **Step statistics**: Each step reads `get_nanos()` twice. The runner keeps
   `StepStats { count, total_ns, min_ns, max_ns }`; the mean is `total / count`
   on the host, so no per-step division runs on target.
    - Per NFR-2, per-step timestamp capture adds $\le 1\ \mu\text{s}$ overhead,
      safeguarding the 1 ms control deadline.
3. **Task End**: The statistics are emitted in the final run-statistics message.

---

### 5. Alternatives

* **Adopt an existing profiling crate**: Rejected. Crates such as
  `cortex-m-funnel`
  and `embedded-profiling` address runtime profiling in `no_std` environments
  (TDHolmes, 2021a) via traits requiring a clock read and an output sink
  (TDHolmes, 2021b). However,
  `embedded-profiling` has not released since version 0.3.0 in December 2021
  (TDHolmes, 2021c),
  and its trait covers only elapsed time snapshots without cycle counting, stack
  introspection, or interrupt control. Reimplementing the unified `CPUProfiler`
  trait in-house minimizes dependencies and tightly couples lifecycle hooks to
  ETS.
* **Static stack analysis instead of painting**: Rejected as the primary
  measure, retained as a complement. `cargo-call-stack` is a whole-program stack
  analyzer but relies on experimental `-Z stack-sizes` (japaric, 2023), and
  inline assembly
  breaks LLVM's stack usage analysis (japaric, 2023). Compiler-emitted
  per-function usage
  from `-fstack-usage` does not readily analyze nested call trees (Pendleton,
  2023). Runtime
  stack painting therefore provides the empirical high-water mark, with
  link-time tools (`flip-link`) as complementary overflow protection. No
  static stack analysis gate exists (the static-analyzer design and its
  `analyze` gate were withdrawn on 2026-09-14).
* **Tick-based timekeeping**: Rejected. A tickless monotonic avoids periodic
  interrupts to count ticks (rtic-rs, 2022a), whereas SysTick-based periodic
  tick generation
  incurs high interrupt rates (rtic-monotonics, 2026) that perturb execution
  duration and jitter
  measurements.
* **Host-side trace tooling**: Rejected as a requirement. External tools such as
  SEGGER SystemView analyze and verify embedded systems from the host (SEGGER,
  2026), but
  impose vendor-specific hardware or debug probes, violating open emulator and
  target independence constraints.
* **Separated clock and test executor traits**: Rejected. Splitting profiling
  into distinct clock and test executor interfaces requires target
  implementations
  to duplicate test execution and timing math. Consolidating hooks into
  `CPUProfiler` restricts target code to raw hardware reads while the ETS server
  orchestrates execution and telemetry.
* **Direct `critical-section` dependency for interrupt control**: Considered for
  future deduplication. `critical_section::with(|cs| ...)` mirrors
  `disable_interrupts<F, R>` and allows shared implementations across Cortex-M
  and RISC-V.
* **External GPIO pin toggling**: Rejected for automated test telemetry.
  Toggling
  hardware pins allows external instrumentation (for example, logic analyzers)
  to measure
  timings, but requires dedicated physical I/O and host capture hardware,
  precluding headless CI and QEMU execution.
* **Dynamic heap and multicore profiling**: Deferred as non-goals for
  single-core
  deterministic `no_std` / `no_alloc` targets.

### 6. Verification & Validation

#### 6.1 Verification

The implementation must produce evidence that cycle and time sources agree with
each other and with the configured clock, that stack measurement never reads
outside the stack region, and that the profiling wrapper's own overhead is
small enough not to distort what it measures.

| Condition | Requirement | Method                    | Target                                                                               | Criterion                                                                                              |
|:----------|:------------|:--------------------------|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------------------------------------------------|
| VC-1.1    | FR-1        | `libtest`                 | `control_rs_ets::profiler::tests::cycles_agree_with_nanos_over_delay`                | `get_cycles()` delta and `get_nanos()` delta over a proven delay loop agree within the clock tolerance |
| VC-1.2    | FR-1        | `libtest`                 | `control_rs_ets::profiler::tests::cycle_delta_wraps`                                 | Mock profilers yield correct deltas over wraparound, zero-length and maximal intervals                 |
| VC-2.1    | FR-2        | `libtest`                 | `control_rs_ets::profiler::tests::nanos_monotonic`                                   | `get_nanos` is non-decreasing across calls                                                             |
| VC-2.2    | FR-2        | `libtest`                 | `control_rs_ets::profiler::tests::mock_reports_no_clock`                             | The mock profiler reports `has_clock` false and the hardware profilers report true                     |
| VC-3.1    | FR-3        | Static analysis           | `read_stack_peak` and `paint_stack` sources                                          | Every read lies within `_stack_start` to `_stack_end`                                                  |
| VC-4.1    | FR-4        | `libtest`                 | `control_rs_ets::profiler::tests::peak_monotonic_with_depth`                         | Measured peak increases monotonically with call-chain depth after painting                             |
| VC-5.1    | FR-5        | `libtest`                 | `control_rs_ets::profiler::tests::atomic_closure_masks_and_reports`                  | Closure runs with interrupts masked and returns cycles and stack figures                               |
| VC-6.1    | FR-6        | `libtest`                 | `control_rs_ets::profiler::tests::fpu_flags_combine_abi_single_and_double_precision` | Defaults hold for every hard-float, single-precision and double-precision combination                  |
| VC-7.1    | FR-7        | `libtest`                 | `control_rs_ets::profiler::tests::step_jitter_bounds`                                | Start-to-finish duration per step is reported with no gap between steps                                |
| VC-21.1   | NFR-1       | Resource usage evaluation | Wrapper and critical-section closure on target                                       | Overhead in cycles is recorded and a few clock cycles                                                  |
| VC-22.1   | NFR-2       | Resource usage evaluation | Step-jitter suite on target                                                          | Per-step capture overhead is at most 1 µs                                                              |
| VC-31.1   | C-1         | Static analysis           | `cargo ci` (clippy stage); inspection                                                | No allocation and no `std` in the profiler crate paths                                                 |
| VC-32.1   | C-2         | On-target execution       | ETS profiling suite on Teensy 4.1 and QEMU per architecture                          | Cycle, duration and stack figures are produced and self-consistent on every listed architecture        |
| VC-32.2   | C-2         | Compile-time shape check  | `cargo ci` build for `thumbv6m-none-eabi`                                            | The ARMv6-M profiler builds without DWT                                                                |

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

| Claim                    | Oracle                                              | Measure                                                            | Bound                                               |
|:-------------------------|:----------------------------------------------------|:-------------------------------------------------------------------|:----------------------------------------------------|
| Cycle and time agreement | `cortex_m::asm::delay(n)` at a known core frequency | Relative error between the cycle delta and the derived nanoseconds | ≤ 1% over intervals ≥ 10,000 cycles                 |
| Counter wraparound       | Mock profiler stepped across the 32-bit boundary    | Reported delta                                                     | Correct modulo the counter width, never negative    |
| Stack bounds             | Linker symbols `_stack_start`, `_stack_end`         | Addresses touched by paint and scan                                | All within `[_stack_end, _stack_start]`             |
| Peak monotonicity        | Call chains of increasing known depth               | Reported peak                                                      | Non-decreasing with depth                           |
| Wrapper overhead         | Empty measured region                               | Cycles attributed to the wrapper                                   | ≤ 1% of the shortest measured interval in the suite |
| Step profiling overhead  | Per-step timestamp capture                          | Latency added per step                                             | $\le 1\ \mu\text{s}$ (NFR-2)                        |
| Step statistics          | Stepped loop timestamps                             | Count, total, minimum and maximum step duration                    | Exact match against clock deltas                    |

The cycle bound applies only above 10,000 cycles because at shorter intervals
the wrapper overhead is a significant fraction of the measurement, which is the
same reason the overhead itself is bounded.

#### 6.3 Limits

- Absolute accuracy of cycle counts under emulation. QEMU states no cycle or
  timing model, so emulated figures are indicative per
  `documentation/ci/ci-design.md` C-6; only hardware runs measure.
- Stack usage of interrupt handlers and inline assembly. Painting measures what
  ran on the measured stack; static analysis of these paths is explicitly
  unavailable, since inline assembly "breaks LLVM's stack usage analysis"
  (japaric, 2023).
- Behaviour under preemption. Measurements assume the measured region is not
  preempted; nothing detects or corrects for a context switch mid-measurement.
- Cross-architecture comparability. Cycle counts from Cortex-M and RISC-V are
  not claimed to be comparable to each other.
- Long-interval measurement. Nothing bounds error accumulation across intervals
  longer than one counter period.

### 7. Performance & Resource Considerations

* **Hook latency**: The `get_cycles()` and `get_nanos()` hooks must execute
  deterministically and as close to zero-overhead as possible (typically within
  a few clock cycles).
* **Closure Execution Overhead**: The `disable_interrupts<F, R>` wrapper
  introduces critical section overhead. Implementations must minimize the setup
  and teardown instructions around the closure to avoid artificially elevating
  interrupt latency or masking real-time deadlines.
* **Scan Time Complexity**: The `read_stack_peak()` method relies on scanning
  memory linearly for sentinel values. This operation is $O (N)$ relative to the
  size of the stack and executes during post-test reporting or task teardown to
  avoid perturbing active test or simulation execution.

---

### 8. Risks & Open Questions

* **Register Overflows**: `CortexMProfiler::get_cycles()` casts the 32-bit DWT
  counter to `u64` with no wraparound handling; at typical core clocks it
  wraps within seconds to tens of seconds. `dwt-systick-monotonic`'s `extend`
  technique (compare against last reading, track the high 32 bits, driven by a
  periodic interrupt) is a proven reference implementation; verify against
  `rtic-monotonics` (its actively maintained successor) before committing,
  since `dwt-systick-monotonic` last released in 2022. Open question whether
  ETS's polling cadence is frequent enough to observe every
  wraparound without a dedicated interrupt. On RISC-V, RV32's `mcycle` is also
  32 bits (with a separate `mcycleh` CSR); whether `RiscvProfiler` combines
  them into a full 64-bit value is unverified.
* **ARMv6-M Resolution**: `cortex-m` gates `DWT::cycle_count()` behind
  `#[cfg(not(armv6m))]`, so Cortex-M0/M0+ have no DWT cycle counter (cortex-m,
  2023). The ARMv6-M profiler reads SysTick instead, following the
  SysTick monotonic of `rtic-monotonics` (rtic-monotonics, 2026); resolution
  and the wrap limit below are the cost.
* **Clock Wrap While Masked**: `Atomic` cases run with interrupts masked, so a
  clock that extends its counter in an interrupt cannot correct a wrap during a
  case. On SysTick fallbacks (24-bit) a case must stay shorter than one counter
  period (about 350 ms at 48 MHz), or the reported duration is wrong.
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

| Phase                              | Delivers                                                                                                  | Requirements | Effort  | Status   |
|:-----------------------------------|:----------------------------------------------------------------------------------------------------------|:-------------|:--------|:---------|
| 1. Trait and target profilers      | Define `CPUProfiler` and implement `CortexMProfiler` and `RiscvProfiler` in `control-rs-ets::profiler`    | FR-1 to FR-6 | Shipped | Complete |
| 2. Overflow-safe cycle counting    | DWT wraparound handling (extend technique) and RV32 `mcycle`/`mcycleh` combination                        | FR-1, NFR-1  | 0.5 day | Planned  |
| 3. ARMv6-M profiler                | SysTick-derived profiler for Cortex-M0/M0+, built for `thumbv6m-none-eabi`                                | FR-6, C-2    | 0.5 day | Planned  |
| 4. Overflow protection             | Evaluate `flip-link` complementary protection; static-analyzer integration is withdrawn                   | FR-4         | 1.0 day | Planned  |
| 5. Lifecycle step jitter profiling | Step statistics (count, total, min, max) for lifecycle tasks; no stack painting while interrupts are live | FR-7, NFR-2  | 0.5 day | Planned  |

---

### 10. Revision History

| Revision | Date               | Author            | Description                                                                                                                            |
|:---------|:-------------------|:------------------|:---------------------------------------------------------------------------------------------------------------------------------------|
| 1.0      | August 29, 2026    | @MitchellDScott   | `CPUProfiler` trait with DWT and `mcycle` cycle counters.                                                                              |
| 2.0      | September 18, 2026 | @MitchellDScott   | Moved to `documentation/ets/`; citation and structure pass.                                                                            |
| 3.0      | September 25, 2026 | @MitchellDScott   | `TargetInfo` provided methods; static-analyzer references removed.                                                                     |
| 4.0      | October 1, 2026    | @MitchellDScott   | Host-testable `fpu_flags` default.                                                                                                     |
| 2.6      | October 12, 2026   | @MitchellDScott   | Development plan in the standard phase table.                                                                                          |
| 2.7      | October 13, 2026   | Claude Sonnet 5.5 | Header migrated to the Approval and Doc Status badge grammar; Approval set to Stale pending re-review. Affects C-2, FR-2, FR-7, NFR-2. |

---

## References

[1] rust-embedded, "cortex-m," *rust-embedded/cortex-m*. [Online].
Available: https://docs.rs/cortex-m/latest/src/cortex_m/peripheral/dwt.rs.html.
Accessed: Aug. 7, 2026.

[2] rust-embedded, "critical-section,"
*rust-embedded/critical-section*. [Online].
Available: https://github.com/rust-embedded/critical-section. Accessed: Aug. 7,

2026.

[3] Antoine Colin, "How to measure stack usage through stack painting with
RapiTest," *Rapita Systems Blog*. [Online].
Available: https://www.rapitasystems.com/blog/how-measure-stack-usage-through-stack-painting-rapitest.
Accessed: Aug. 7, 2026.

[4] Noah Pendleton, "Measuring Stack Usage the Hard Way," *Interrupt (Memfault
blog)*. [Online].
Available: https://interrupt.memfault.com/blog/measuring-stack-usage. Accessed:
Aug. 7, 2026.

[5] Zephyr Project, "Threads," *Zephyr Project Documentation*. [Online].
Available: https://docs.zephyrproject.org/latest/kernel/services/threads/index.html.
Accessed: Aug. 7, 2026.

[6] TDHolmes, "embedded-profiling," *TDHolmes/embedded-profiling*. [Online].
Available: https://github.com/TDHolmes/embedded-profiling. Accessed: Aug. 7,

2026.

[7] TDHolmes, *embedded-profiling* (Version 0.3.0). [Online].
Available: https://docs.rs/embedded-profiling/latest/embedded_profiling/trait.EmbeddedProfiler.html.
Accessed: Aug. 7, 2026.

[8] TDHolmes, *embedded-profiling* (Version 0.3.0). [Online].
Available: https://crates.io/api/v1/crates/embedded-profiling. Accessed: Aug. 7,

2026.

[9] japaric, "cargo-call-stack," *japaric/cargo-call-stack*. [Online].
Available: https://github.com/japaric/cargo-call-stack. Accessed: Aug. 7, 2026.

[10] rtic-rs, "dwt-systick-monotonic,"
*rtic-rs/dwt-systick-monotonic*. [Online].
Available: https://github.com/rtic-rs/dwt-systick-monotonic/blob/master/src/lib.rs.
Accessed: Aug. 7, 2026.

[11] rtic-rs, *rtic-monotonics* (Version 2.2.1). [Online].
Available: https://docs.rs/rtic-monotonics/latest/rtic_monotonics/systick/.
Accessed: Aug. 7, 2026.

[12] SEGGER Microcontroller GmbH, "What is SystemView?," *SEGGER product
documentation*. [Online].
Available: https://www.segger.com/products/development-tools/systemview/technology/what-is-systemview/.
Accessed: Aug. 7, 2026.
