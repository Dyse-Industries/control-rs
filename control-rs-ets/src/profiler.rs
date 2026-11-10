//! Hardware execution abstraction and CPU profiling for ETS tests.
//!
//! # Description
//!
//! This module provides the `CPUProfiler` trait and architecture-specific implementations
//! to measure hardware performance metrics (such as clock cycles, timing and stack consumption)
//! during ETS tests.
//!
//! # Core Concepts
//!
//! - **CPU Profiler Trait**: A target-agnostic interface to retrieve low-level system performance metrics.
//! - **Stack Tracking**: Functions (`paint_stack` and `read_stack_peak`) paint the unused stack area
//!   with sentinel bytes to compute high-water-mark stack usage safely.
//! - **Platform Implementations**: Cortex-M DWT-based and RISC-V CSR-based profiling structures.

/// Maximum number of retries for reading `SysTick` before falling back to whole-millisecond ticks.
pub const MAX_SYSTICK_RETRIES: usize = 8;

/// Target-specific utilities for CPU profiling, stack tracking and interrupt management during ETS tests.
///
/// This trait provides a target-agnostic interface to low-level hardware features, allowing the
/// ETS to execute test functions in a controlled environment, gather cycle counts and
/// safely measure stack usage. By encapsulating architecture-specific inline assembly and memory-mapped
/// I/O operations, this trait helps prevent safety issues in the Server's core logic.
///
/// # Safety
/// Implementors must ensure that architecture-specific profiling operations (like reading CSRs or memory mapped registers)
/// do not cause undefined behavior or violate memory safety.
///
/// # Panics
/// Implementations of this trait do not panic under normal usage, except when resetting or exiting the system.
///
/// # Example
/// ```
/// use control_rs_ets::profiler::CPUProfiler;
///
/// struct MockProfiler;
/// impl CPUProfiler for MockProfiler {
///     fn get_cycles(&self) -> u64 { 100 }
///     fn get_nanos(&self) -> u64 { 1000 }
///     fn get_sp(&self) -> usize { 0 }
///     fn get_stack_end(&self) -> usize { 0 }
/// }
///
/// let profiler = MockProfiler;
/// assert_eq!(profiler.get_cycles(), 100);
/// assert_eq!(profiler.get_nanos(), 1000);
/// ```
pub trait CPUProfiler {
    /// Whether this profiler has a cycle-accurate hardware clock.
    /// Returns false on architectures lacking DWT or cycle counters (for example, ARMv6-M).
    fn has_clock(&self) -> bool {
        true
    }

    /// Board identifier reported in `Telemetry::TargetInfo`; `0` when the
    /// profiler does not know the board.
    fn board_id(&self) -> u16 {
        0
    }

    /// Core clock frequency in hertz reported in `Telemetry::TargetInfo`;
    /// `0` when unknown.
    fn core_clock_hz(&self) -> u32 {
        0
    }

    /// Disables interrupts and runs the given closure, returning its result.
    ///
    /// This ensures that the test function is executed without interruption, preventing context
    /// switches or interrupt service routines (ISRs) from corrupting the timing/cycle measurements.
    ///
    /// # Generic Arguments
    /// * `F` - A closure with no arguments.
    /// * `R` - Return type of the closure.
    ///
    /// # Arguments
    /// * `f` - The execution closure.
    ///
    /// # Returns
    /// * `R` - The value returned by the execution closure.
    fn disable_interrupts<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        f()
    }

    /// Disables interrupts permanently.
    ///
    /// This is typically called during a panic or a critical failure condition to halt all regular
    /// target activity and prevent interrupts from interfering with failure telemetry transmission.
    fn disable_interrupts_permanently(&self) {}

    /// Exits the application/environment using target-specific mechanisms.
    ///
    /// Useful for virtualized targets (like QEMU) to signal termination back to the host machine.
    ///
    /// # Returns
    /// * `!` - This function never returns.
    #[allow(clippy::empty_loop)]
    fn exit(&self) -> ! {
        loop {}
    }

    /// FPU capability bits reported in `Telemetry::TargetInfo`: bit 0 single
    /// precision, bit 1 double precision. The default derives them from the
    /// compilation target.
    fn fpu_flags(&self) -> u8 {
        fpu_flags_from(
            cfg!(target_abi = "eabihf"),
            cfg!(target_feature = "f"),
            cfg!(target_feature = "d"),
        )
    }

    /// Get the current CPU cycle count.
    ///
    /// Used by the Server to calculate the clock cycles spent running a test.
    ///
    /// # Returns
    /// * `u64` - The current clock cycle count.
    fn get_cycles(&self) -> u64;

    /// Get the current time in nanoseconds.
    ///
    /// Used to compute elapsed wall-clock execution time of tests.
    ///
    /// # Returns
    /// * `u64` - The current timestamp in nanoseconds.
    fn get_nanos(&self) -> u64;

    /// Get the current stack pointer.
    ///
    /// Used as a reference point for painting and reading stack usage.
    ///
    /// # Returns
    /// * `usize` - The memory address representing the stack pointer.
    fn get_sp(&self) -> usize;

    /// Get the end of the current stack.
    ///
    /// Returns the address of the bottom (end) of the stack area.
    ///
    /// # Returns
    /// * `usize` - The memory address representing the bottom of the stack area.
    fn get_stack_end(&self) -> usize;

    /// Paints the stack below the given stack pointer.
    ///
    /// Writes a sentinel value (`0xCDCD_CDCD`) to unused stack space from the bottom (end) of the stack
    /// up to a safety threshold below the current stack pointer.
    ///
    /// # Arguments
    /// * `sp` - The active stack pointer.
    ///
    /// # Safety
    ///
    /// This function writes directly to the stack memory range. The caller MUST ensure the following conditions are met:
    ///
    /// * The stack pointer `sp` is valid and represents the active stack pointer.
    /// * The stack bounds returned by `get_stack_end()` map to a valid memory region allocated for the stack.
    /// * The safety margin is sufficient to prevent overwriting active frames.
    /// * The stack memory below `sp` (down to the stack end) is painted with the sentinel value `0xCDCD_CDCD`.
    /// * Current CPU register states and active stack frames remain unmodified.
    /// * The memory range is mutable and can be safely written to using volatile writes without triggering memory faults.
    #[inline(never)] // Crucial for predictable stack frame analysis
    unsafe fn paint_stack(&self, sp: usize) {
        let stack_end_ptr = self.get_stack_end();
        // Leave a 32-byte safety margin below the current stack pointer to avoid
        // overwriting active stack frames (like paint_stack's own frame and return address).
        let limit = sp.saturating_sub(32);

        if limit > stack_end_ptr {
            let mut ptr = stack_end_ptr as *mut u32;
            let limit_ptr = limit as *mut u32;

            while ptr < limit_ptr {
                // SAFETY: The safety invariant of `paint_stack` guarantees that `ptr` to `limit_ptr`
                // falls within the bounds of the allocated stack and does not overlap with any active
                // stack frames. Using `write_volatile` prevents the compiler from optimizing out these writes.
                unsafe {
                    core::ptr::write_volatile(ptr, 0xCDCD_CDCD);
                    ptr = ptr.add(1);
                }
            }
        }
    }

    /// Reads the peak stack usage (in bytes) since the stack was painted, relative to the given stack pointer.
    ///
    /// Scans the stack space from the bottom (end) upward, looking for the first memory address that no longer
    /// contains the sentinel value (`0xCDCD_CDCD`). The difference between `sp` and this address is returned.
    ///
    /// # Arguments
    /// * `sp` - The active stack pointer.
    ///
    /// # Returns
    /// * `u32` - The peak stack usage in bytes.
    ///
    /// # Safety
    ///
    /// This function reads directly from the stack memory range. The caller MUST ensure the following conditions are met:
    ///
    /// * The stack has been previously painted with the `paint_stack` function.
    /// * The stack pointer `sp` matches the original reference point when the stack was painted.
    /// * The stack bounds returned by `get_stack_end()` remain valid.
    /// * Stack contents are unchanged by reading.
    /// * The painted memory is not corrupted or overwritten by other unrelated operations in the meantime.
    #[inline(never)] // Crucial for predictable stack frame analysis
    unsafe fn read_stack_peak(&self, sp: usize) -> u32 {
        let stack_end_ptr = self.get_stack_end();
        if sp <= stack_end_ptr {
            return 0;
        }
        let mut ptr = stack_end_ptr as *const u32;
        let limit_ptr = sp as *const u32;

        while ptr < limit_ptr {
            // SAFETY: The caller guarantees that the memory range between `stack_end_ptr` and `limit_ptr`
            // has been painted and is safe to read. Using `read_volatile` prevents optimization/caching of the reads.
            let is_sentinel =
                unsafe { core::ptr::read_volatile(ptr) == 0xCDCD_CDCD };
            if !is_sentinel {
                break;
            }
            // SAFETY: The bounds check in the loop condition (`ptr < limit_ptr`) ensures that
            // incrementing the pointer by 1 stays within the valid stack memory bounds.
            unsafe {
                ptr = ptr.add(1);
            }
        }

        let lowest_address = ptr as usize;
        sp.checked_sub(lowest_address)
            .and_then(|diff| u32::try_from(diff).ok())
            .unwrap_or(0)
    }

    /// Resets the CPU/system.
    ///
    /// # Returns
    /// * `!` - This function never returns.
    #[allow(clippy::empty_loop)]
    fn reset(&self) -> ! {
        loop {}
    }
}

/// Mock CPU profiler for host-side testing and simulation.
#[derive(Debug, Clone, Copy, Default)]
pub struct HostCPUProfiler;

/// Hardware reading snapshot from ARM Cortex-M `SysTick` down-counter and status.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SysTickReading {
    /// The current 24-bit down-counter value (`SYST_CVR`).
    pub current: u32,
    /// Whether the `SysTick` exception is pending in ICSR (`PENDSTSET`).
    pub pending: bool,
}

impl CPUProfiler for HostCPUProfiler {
    fn get_cycles(&self) -> u64 {
        0
    }

    fn get_nanos(&self) -> u64 {
        0
    }

    fn get_sp(&self) -> usize {
        0
    }

    fn get_stack_end(&self) -> usize {
        0
    }

    unsafe fn paint_stack(&self, _sp: usize) {}

    unsafe fn read_stack_peak(&self, _sp: usize) -> u32 {
        0
    }
}

/// Target-specific implementation of `CPUProfiler` for ARM Cortex-M microcontrollers.
///
/// This structure tracks execution cycles using the ARM DWT (Data Watchpoint and Trace) cycle
/// counter, measures elapsed time using the SysTick timer and inspects the stack using linker symbols.
#[cfg(target_arch = "arm")]
pub struct CortexMProfiler {
    clock_frequency: u32,
    ticks: &'static core::sync::atomic::AtomicU32,
}

#[cfg(target_arch = "arm")]
impl CortexMProfiler {
    /// Creates a new `CortexMProfiler` instance.
    ///
    /// # Arguments
    /// * `clock_frequency` - CPU core clock frequency in Hz.
    /// * `ticks` - A static reference to an atomic millisecond counter updated by SysTick.
    ///
    /// # Returns
    /// * `Self` - The Cortex-M profiler instance.
    #[must_use]
    pub const fn new(
        clock_frequency: u32,
        ticks: &'static core::sync::atomic::AtomicU32,
    ) -> Self {
        Self {
            clock_frequency,
            ticks,
        }
    }
}

#[cfg(target_arch = "arm")]
impl CPUProfiler for CortexMProfiler {
    fn has_clock(&self) -> bool {
        cfg!(target_has_atomic = "ptr")
    }

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
        #[cfg(target_has_atomic = "ptr")]
        {
            cortex_m::peripheral::DWT::cycle_count() as u64
        }
        #[cfg(not(target_has_atomic = "ptr"))]
        {
            0
        }
    }

    fn get_nanos(&self) -> u64 {
        let reload =
            self.clock_frequency.saturating_div(1000).saturating_sub(1);
        for _ in 0..MAX_SYSTICK_RETRIES {
            let ms1 = u64::from(
                self.ticks.load(core::sync::atomic::Ordering::Relaxed),
            );
            let current = cortex_m::peripheral::SYST::get_current();
            let icsr =
                unsafe { core::ptr::read_volatile(0xE000_ED04 as *const u32) };
            cortex_m::asm::isb();
            let ms2 = u64::from(
                self.ticks.load(core::sync::atomic::Ordering::Relaxed),
            );
            if ms1 == ms2 {
                let reading = SysTickReading {
                    current,
                    pending: (icsr & (1 << 26)) != 0,
                };
                return calculate_systick_nanos(
                    ms1,
                    reading,
                    reload,
                    self.clock_frequency,
                );
            }
        }
        let ms =
            u64::from(self.ticks.load(core::sync::atomic::Ordering::Relaxed));
        ms.saturating_mul(1_000_000)
    }

    fn get_sp(&self) -> usize {
        let sp: usize;
        // SAFETY: Reading the active stack pointer `sp` using inline assembly does not write to
        // memory, affect any active stack frames or corrupt processor flags. The assembly options
        // `nomem` and `nostack` verify these constraints.
        unsafe {
            core::arch::asm!("mov {}, sp", out(reg) sp, options(nomem, nostack, preserves_flags));
        }
        sp
    }

    fn get_stack_end(&self) -> usize {
        // SAFETY: The external static variable `_stack_end` is provided by the linker script (generated by
        // `build.rs` as the limit of the FLASH/RAM boundaries). It is safe to take the address of this
        // symbol because it is treated strictly as a numeric boundary address, not as a dereferenced u32 value.
        unsafe extern "C" {
            static mut _stack_end: u32;
        }
        core::ptr::addr_of!(_stack_end) as usize
    }

    fn reset(&self) -> ! {
        cortex_m::peripheral::SCB::sys_reset();
    }
}

/// Target-specific implementation of `CPUProfiler` for RISC-V targets.
///
/// This structure tracks execution cycles using the RISC-V `mcycle` CSR register,
/// measures elapsed time using the `time` CSR register and calculates stack limits using linker-defined variables.
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
pub struct RiscvProfiler {
    clock_frequency: u32,
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
impl RiscvProfiler {
    /// Creates a new `RiscvProfiler` instance.
    ///
    /// # Arguments
    /// * `clock_frequency` - Timer/CSR ticking frequency in Hz.
    ///
    /// # Returns
    /// * `Self` - RISC-V profiler instance.
    #[must_use]
    pub const fn new(clock_frequency: u32) -> Self {
        Self { clock_frequency }
    }
}

#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
impl CPUProfiler for RiscvProfiler {
    fn core_clock_hz(&self) -> u32 {
        self.clock_frequency
    }

    fn disable_interrupts<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        // SAFETY: Disabling and enabling global interrupts via the RISC-V status register (mstatus) is
        // safe because it does not violate memory safety or cause undefined behavior. The state
        // of the global interrupts is restored once the critical section code (the closure `f`) completes.
        unsafe {
            riscv::interrupt::disable();
        }
        let res = f();
        // SAFETY: Re-enabling interrupts is safe to restore normal execution flow.
        unsafe {
            riscv::interrupt::enable();
        }
        res
    }

    fn disable_interrupts_permanently(&self) {
        // SAFETY: Disabling global interrupts permanently is safe and ensures that no ISRs execute
        // during panic/failure recovery.
        unsafe {
            riscv::interrupt::disable();
        }
    }

    fn get_cycles(&self) -> u64 {
        riscv::register::mcycle::read() as u64
    }

    fn get_nanos(&self) -> u64 {
        #[cfg(target_arch = "riscv32")]
        let ticks = riscv::register::time::read64();
        #[cfg(target_arch = "riscv64")]
        let ticks = riscv::register::time::read() as u64;

        (ticks as u64).saturating_mul(1_000_000_000)
            / (self.clock_frequency as u64)
    }

    fn get_sp(&self) -> usize {
        let sp: usize;
        // SAFETY: Reading the active stack pointer `sp` using inline assembly does not write to
        // memory, affect any active stack frames or corrupt processor flags. The assembly options
        // `nomem` and `nostack` verify these constraints.
        unsafe {
            core::arch::asm!("mv {}, sp", out(reg) sp, options(nomem, nostack, preserves_flags));
        }
        sp
    }

    fn get_stack_end(&self) -> usize {
        // SAFETY: The external symbol `_profiler_stack_start` is provided by the RISC-V linker
        // configuration. It is safe to take the address of this static symbol to perform pointer arithmetic
        // because it is treated strictly as a numeric boundary, not as a dereferenced value.
        unsafe extern "C" {
            static _profiler_stack_start: u32;
        }
        core::ptr::addr_of!(_profiler_stack_start) as usize
    }
}

/// FPU capability bits for the given target properties: bit 0 is set for a
/// hard-float ABI or the single-precision feature, bit 1 for double precision.
#[must_use]
const fn fpu_flags_from(
    hard_float_abi: bool,
    single: bool,
    double: bool,
) -> u8 {
    match (hard_float_abi || single, double) {
        (false, false) => 0b00,
        (true, false) => 0b01,
        (false, true) => 0b10,
        (true, true) => 0b11,
    }
}

/// Calculates wall-clock nanoseconds from a `SysTick` reading and millisecond ticks,
/// reconciling pending counter rollovers.
#[must_use]
pub fn calculate_systick_nanos(
    ms: u64,
    reading: SysTickReading,
    reload: u32,
    clock_frequency_hz: u32,
) -> u64 {
    let adjusted_ms = if reading.pending && reading.current > reload / 2 {
        ms.saturating_add(1)
    } else {
        ms
    };
    let cycles = reload.saturating_sub(reading.current);
    let cycles_per_us = u64::from(clock_frequency_hz)
        .checked_div(1_000_000)
        .unwrap_or(1);
    let fractional_us = if cycles_per_us > 0 {
        u64::from(cycles).checked_div(cycles_per_us).unwrap_or(0)
    } else {
        0
    };
    let us = adjusted_ms
        .saturating_mul(1000)
        .saturating_add(fractional_us);
    us.saturating_mul(1000)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestProfiler {
        stack: [u32; 64],
    }

    impl TestProfiler {
        fn new() -> Self {
            Self { stack: [0; 64] }
        }
        fn write_stack(&mut self, index: usize, val: u32) {
            if let Some(slot) = self.stack.get_mut(index) {
                *slot = val;
            }
        }
    }

    impl CPUProfiler for TestProfiler {
        fn get_cycles(&self) -> u64 {
            0
        }
        fn get_nanos(&self) -> u64 {
            0
        }
        fn get_sp(&self) -> usize {
            core::ptr::addr_of!(self.stack[48]) as usize
        }
        fn get_stack_end(&self) -> usize {
            core::ptr::addr_of!(self.stack[0]) as usize
        }
    }

    #[test]
    fn test_default_profiler_methods() {
        let mut profiler = TestProfiler::new();

        let val = profiler.disable_interrupts(|| 42);
        assert_eq!(val, 42);

        profiler.disable_interrupts_permanently();

        let sp = profiler.get_sp();
        unsafe {
            profiler.paint_stack(sp);
        }

        let peak = unsafe { profiler.read_stack_peak(sp) };
        assert_eq!(peak, 32);

        profiler.write_stack(20, 0);

        let peak2 = unsafe { profiler.read_stack_peak(sp) };
        assert_eq!(peak2, 112);
    }

    #[test]
    fn fpu_flags_combine_abi_single_and_double_precision() {
        let cases = [
            (false, false, false, 0b00),
            (true, false, false, 0b01),
            (false, true, false, 0b01),
            (true, true, false, 0b01),
            (false, false, true, 0b10),
            (true, false, true, 0b11),
            (false, true, true, 0b11),
            (true, true, true, 0b11),
        ];
        for (abi, single, double, want) in cases {
            assert_eq!(
                fpu_flags_from(abi, single, double),
                want,
                "{abi} {single} {double}"
            );
        }
    }

    #[test]
    fn the_default_fpu_flags_describe_the_compilation_target() {
        let expected = fpu_flags_from(
            cfg!(target_abi = "eabihf"),
            cfg!(target_feature = "f"),
            cfg!(target_feature = "d"),
        );
        assert_eq!(TestProfiler::new().fpu_flags(), expected);
    }

    #[test]
    fn test_stack_paint_exhaustion_clamp() {
        let profiler = TestProfiler::new();
        let sp = profiler.get_stack_end();
        unsafe {
            profiler.paint_stack(sp);
            let peak = profiler.read_stack_peak(sp);
            assert_eq!(peak, 0);
        }
    }

    #[test]
    fn test_naive_systick_rollover_flaw_and_hardened_resolution() {
        // Setup a 600 MHz clock (for example, Teensy 4 / i.MX RT1062, where 1 ms = 600,000 cycles)
        let clock_freq = 600_000_000u32;
        let reload = clock_freq / 1000 - 1; // 599_999 cycles

        // Sample 1: Just before rollover (at the end of millisecond 10)
        let ms_before = 10u64;
        let current_before = 5u32; // down-counter is nearly at zero (599,994 cycles elapsed)

        // Naive calculation (without pending rollover detection):
        // (ms * 1000 + (reload - current) / cycles_per_us) * 1000
        let naive_before = {
            let cycles = reload - current_before;
            let microsecond = ms_before * 1000
                + u64::from(cycles) / u64::from(clock_freq / 1_000_000);
            microsecond * 1000
        };
        // At 600 MHz, 599,994 cycles / 600 = 999 microseconds.
        // naive_before = (10 * 1000 + 999) * 1000 = 10_999_000 nanoseconds.
        assert_eq!(naive_before, 10_999_000);

        // Sample 2: Immediately after rollover in hardware (counter wrapped to reload - 10,
        // and ICSR PENDSTSET is set), but the software interrupt has not yet incremented ticks.
        let ms_after = 10u64; // Stale! The ISR hasn't executed yet.
        let current_after = reload - 10; // Counter reloaded and has only elapsed 10 cycles.
        let icsr_pending = true;

        let naive_after = {
            let cycles = reload - current_after;
            let microsecond = ms_after * 1000
                + u64::from(cycles) / u64::from(clock_freq / 1_000_000);
            microsecond * 1000
        };
        // At 600 MHz, 10 cycles / 600 = 0 microseconds.
        // naive_after = (10 * 1000 + 0) * 1000 = 10_000_000 nanoseconds.
        assert_eq!(naive_after, 10_000_000);

        // --- NEGATIVE TEST ASSERTION ---
        // Verify that the naive implementation is demonstrably flawed:
        // despite time moving forward in hardware, the computed timestamp went BACKWARDS by 999 microseconds!
        assert!(
            naive_after < naive_before,
            "Flaw confirmed: naive SysTick calculation stepped backwards in time ({naive_after} < {naive_before})"
        );
        let time_regression_ns = naive_before - naive_after;
        assert_eq!(
            time_regression_ns, 999_000,
            "Naive implementation suffered a 999 microsecond backward retrograde glitch"
        );

        // --- HARDENED RESOLUTION ASSERTION ---
        // Verify that calculate_systick_nanos detects the pending rollover and reconciles ms,
        // ensuring strictly monotonic time progression.
        let reading_before = SysTickReading {
            current: current_before,
            pending: false,
        };
        let reading_after = SysTickReading {
            current: current_after,
            pending: icsr_pending,
        };
        let hardened_before = calculate_systick_nanos(
            ms_before,
            reading_before,
            reload,
            clock_freq,
        );
        let hardened_after = calculate_systick_nanos(
            ms_after,
            reading_after,
            reload,
            clock_freq,
        );

        assert_eq!(hardened_before, 10_999_000);
        assert_eq!(hardened_after, 11_000_000); // Correctly accounts for next millisecond (11 ms + 0 us)!
        assert!(
            hardened_after > hardened_before,
            "Hardened implementation must advance monotonically across rollover ({hardened_after} > {hardened_before})"
        );
    }

    #[test]
    fn test_systick_nanos_boundary_cases() {
        let clock_freq = 600_000_000u32;
        let reload = clock_freq / 1000 - 1;

        // Start of millisecond (current == reload)
        let at_start = calculate_systick_nanos(
            5,
            SysTickReading {
                current: reload,
                pending: false,
            },
            reload,
            clock_freq,
        );
        assert_eq!(at_start, 5_000_000);

        // Late in millisecond before rollover (current small, pending false)
        let near_end = calculate_systick_nanos(
            5,
            SysTickReading {
                current: 50,
                pending: false,
            },
            reload,
            clock_freq,
        );
        assert_eq!(near_end, 5_999_000);

        // Rollover occurred after read (current small, pending true):
        // Because current <= reload / 2, it belongs to the pre-rollover interval.
        let wrap_after_read = calculate_systick_nanos(
            5,
            SysTickReading {
                current: 50,
                pending: true,
            },
            reload,
            clock_freq,
        );
        assert_eq!(wrap_after_read, 5_999_000);
    }
}
