# CPUProfileUtils Design Document

![Date Badge](https://img.shields.io/badge/Date-May_23,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Needs%20Review-yellow)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

## 1. Context and Objective

The HIL test harness requires target-specific hooks to measure performance
metrics (clock cycles, elapsed execution time, and stack usage).

Previously, this was split into two interfaces: `ClientClock` (system clock
abstraction) and `TestExecutor` (orchestrating the full execution lifecycle of a
test). However, implementing the full execution logic in `TestExecutor` on the
target side was verbose, repetitive, and error-prone.

To simplify target-side integration, these hooks were consolidated into a single
trait: `CPUProfileUtils`. Users only implement primitive hardware hooks (
retrieving cycle counts, reading time, getting the stack pointer, painting the
stack, and checking stack usage). The core HIL `Server` implements the generic
execution wrapper, timing calculations, and metric telemetry reporting in a
platform-agnostic manner.

## 2. Core Mechanics

### 2.1. The CPUProfileUtils Trait

The core of the abstraction is the `CPUProfileUtils` trait, defined in
`control-rs-hil::profiler`:

```rust
pub trait CPUProfileUtils {
    /// Get the current CPU cycle count.
    fn get_cycles(&self) -> u64;

    /// Get the current time in nanoseconds.
    fn get_nanos(&self) -> u64;

    /// Get the current stack pointer.
    fn get_sp(&self) -> usize;

    /// Paints the stack below the given stack pointer.
    ///
    /// # Safety
    /// This writes to the stack memory space. The caller must ensure that the stack pointer is valid
    /// and that the painting bounds do not overwrite any active stack frames or reserved memory.
    unsafe fn paint_stack(&self, sp: usize);

    /// Reads the peak stack usage (in bytes) since the stack was painted, relative to the given stack pointer.
    ///
    /// # Safety
    /// This reads from the stack memory space. The caller must ensure that the stack has been painted
    /// and that memory accesses remain within the valid stack bounds.
    unsafe fn read_stack_peak(&self, sp: usize) -> u32;

    /// Disables interrupts and runs the given closure, returning its result.
    fn disable_interrupts<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R;
}
```

### 2.2. Standard Default Implementation

A standard default implementation, `DefaultCPUProfileUtils`, is provided for
platforms without profiling capabilities. It returns zeroed metrics and
implements `CPUProfileUtils`:

```rust
pub struct DefaultCPUProfileUtils;

impl CPUProfileUtils for DefaultCPUProfileUtils {
    fn get_cycles(&self) -> u64 { 0 }
    fn get_nanos(&self) -> u64 { 0 }
    fn get_sp(&self) -> usize { 0 }
    unsafe fn paint_stack(&self, _sp: usize) {}
    unsafe fn read_stack_peak(&self, _sp: usize) -> u32 { 0 }
}
```

## 3. Usage

### 3.1. Client side (Target MCU)

The developer implements `CPUProfileUtils` on their target platform. For
example, on an ARM Cortex-M architecture:

```rust
struct CortexMProfileUtils;

impl CPUProfileUtils for CortexMProfileUtils {
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

    unsafe fn paint_stack(&self, sp: usize) {
        // Paint stack below `sp` with sentinel values
    }

    unsafe fn read_stack_peak(&self, sp: usize) -> u32 {
        // Scan stack below `sp` for peak usage
        0
    }

    fn disable_interrupts<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        // Cortex-M specific critical section wrapper
        cortex_m::interrupt::free(|_| f())
    }
}
```
