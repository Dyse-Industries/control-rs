# CPUProfiler Design Document

![Date Badge](https://img.shields.io/badge/Date-July_18,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

## 1. Introduction

The HIL test harness requires target-specific hooks to measure performance
metrics (clock cycles, elapsed execution time, and stack usage). By defining a
unified `CPUProfiler` trait within the `control-rs-hil::profiler` module,
the server delegates the implementation of primitive hardware
hooks to the end-user or silicon vendor.

---

## 2. Requirements

**Profiling**:

1. Cycle count
2. Timer
3. Stack pointer read
4. Stack painting/scanning
5. Interrupt control

The profiler must also provide default implementations for targets that do
not support these features.

---

## 3. Technical Overview

This project provides hardware specific hooks through a trait to keep the
server hardware-agnostic.

---

## 4. Core Architecture

### 4.1. The CPUProfiler Trait

The core of the abstraction is the `CPUProfiler` trait, defined in
`control-rs-hil::profiler`:

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
    ///
    /// # Safety
    /// This function writes directly to the stack memory range. The caller MUST ensure the active stack pointer is valid
    /// and stack bounds are safe.
    unsafe fn paint_stack(&self, sp: usize);

    /// Reads the peak stack usage (in bytes) since the stack was painted, relative to the given stack pointer.
    ///
    /// # Safety
    /// This function reads directly from the stack memory range. The caller MUST ensure stack has been painted.
    unsafe fn read_stack_peak(&self, sp: usize) -> u32;

    /// Resets the CPU/system.
    fn reset(&self) -> ! {
        loop {}
    }
}
```

---

### 4.2. Implementations and Mock Profilers

Rather than exposing a default stub struct, target platforms implement `CPUProfiler` directly. For host-side testing, the codebase provides mock profilers (e.g. `HostCPUProfiler` in tests) that return zeroed/dummy metrics:

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

---

## 5. Alternatives

**Retain ClientClock and TestExecutor**: Previously, this was split into two
interfaces: `ClientClock` (system clock
abstraction) and `TestExecutor` (orchestrating the full execution lifecycle of a
test). However, implementing the full execution logic in `TestExecutor` on the
target side was verbose, repetitive, and error-prone.

To simplify target-side integration, these hooks were consolidated into a single
trait: `CPUProfiler`. Users only implement primitive hardware hooks (
retrieving cycle counts, reading time, getting the stack pointer, and identifying stack boundaries). The core HIL `Server` implements the generic
execution wrapper, timing calculations, and metric telemetry reporting in a
platform-agnostic manner.

**Heap Profiling**: Adding traits for tracking dynamic memory allocation, if
applicable to the targets.

**Multicore Profiling**: Expanding the trait to handle multicore execution and
synchronization metrics.

**Existing Crates**:
[embedded-profiling](https://github.com/TDHolmes/embedded-profiling) Uses a very
similar pattern to accomplish the same thing (with built-in logging). This
crate was not used because it provides a small amount of code, does not meet
this crates test and format standards (it also has not been updated for 4
years). Implementing the profile trait in-house allows it to be tightly
coupled with the server.

**Provided GPIO Implementation**: Implement a profiler that automatically
toggles an output pin when a test starts and stops.

**Rapita Systems**:
[profiling tools](https://www.rapitasystems.com/blog/how-measure-stack-usage-through-stack-painting-rapitest)

**Static Stack Usage Analysis**: Tools like cargo-call-stack evaluate LLVM-IR to
mathematically prove maximum stack allocation during compile-time. This
will be implemented as part of the CI.

---

## 6. Verification & Validation

Developers must validate their custom hardware hooks using automated
on-target testing. Ideally these will be available to end users.

* **Runner Accuracy**: Execute a mathematically proven delay loop (e.g.,
  utilizing `cortex_m::asm::delay`). Assert that the numerical delta
  recorded by `get_cycles()` correlates linearly with the elapsed time
  calculated by `get_nanos()` based on the configured core clock frequency.
* **Memory Bounds**: Validate that the `paint_stack` and
  `read_stack_peak` implementations respect the physical RAM boundaries defined
  by the compiler's linker script (specifically `_stack_start` and _
  `stack_end`).
* **Interrupt Latency**: Measure the overhead introduced by the
  generic wrapper and the disable_interrupts closure. Quantify the execution
  overhead introduced by the concurrency wrapper to ensure it does not induce
  unacceptable execution jitter.

---

## 7. Performance and Resource Considerations

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

## 8. Risks and Open Questions

* **Register Overflows**: Cycle counters and timers are usually integers,
  these will overflow eventually.
* **Stack Bounds Calculation**: How will the user implementation safely
  determine the absolute bottom of the stack. It should be available through
  the linker, but this is a complex integration task.
* **Clock Skew / Power Saving**: If the user needs to run tests in a reduced
  power or low compute state the profiler may become less accurate.
* **Unsafe Code Proliferation**: Users must implement unsafe functions for
  their profiler to work.

---