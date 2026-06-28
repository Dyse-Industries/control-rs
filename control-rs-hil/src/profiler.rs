//! Hardware execution abstraction and CPU profiling for HIL tests.
//!
//! Provides traits to measure hardware performance metrics (like clock cycles, execution time,
//! and stack space consumption) in a hardware-agnostic manner.

/// Target-specific utilities for CPU profiling, stack tracking, and interrupt management during HIL tests.
///
/// This trait provides a target-agnostic interface to low-level hardware features, allowing the
/// HIL test runner to execute test functions in a controlled environment, gather cycle counts, and
/// safely measure stack usage. By encapsulating architecture-specific inline assembly and memory-mapped
/// I/O operations, this trait helps prevent safety issues in the test runner's core logic.
pub trait CPUProfiler {
    /// Disables interrupts and runs the given closure, returning its result.
    ///
    /// This ensures that the test function is executed without interruption, preventing context
    /// switches or interrupt service routines (ISRs) from corrupting the timing/cycle measurements.
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
    #[allow(clippy::empty_loop)]
    fn exit(&self) -> ! {
        loop {}
    }

    /// Get the current CPU cycle count.
    ///
    /// Used by the test runner to calculate the clock cycles spent running a test.
    fn get_cycles(&self) -> u64;

    /// Get the current time in nanoseconds.
    ///
    /// Used to compute elapsed wall-clock execution time of tests.
    fn get_nanos(&self) -> u64;

    /// Get the current stack pointer.
    ///
    /// Used as a reference point for painting and reading stack usage.
    fn get_sp(&self) -> usize;

    /// Get the end of the current stack.
    ///
    /// Returns the address of the bottom (end) of the stack area.
    fn get_stack_end(&self) -> usize;

    /// Paints the stack below the given stack pointer.
    ///
    /// Writes a sentinel value (`0xCDCD_CDCD`) to unused stack space from the bottom (end) of the stack
    /// up to a safety threshold below the current stack pointer.
    ///
    /// # Safety
    ///
    /// This function writes directly to the memory range representing the stack. The caller must ensure:
    /// 1. The provided stack pointer `sp` is valid and represents the active stack pointer.
    /// 2. The stack limits returned by `get_stack_end()` are accurate and map to a valid memory region allocated for the stack.
    /// 3. The safety margin (currently 32 bytes below `sp`) is sufficient to prevent overwriting any active
    ///    stack frames of callers or local variables.
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
    /// # Safety
    ///
    /// This function reads directly from the stack memory range. The caller must ensure:
    /// 1. The stack has been previously painted with the `paint_stack` function.
    /// 2. The stack pointer `sp` matches the original reference point when the stack was painted.
    /// 3. The stack bounds returned by `get_stack_end()` remain valid.
    #[inline(never)] // Crucial for predictable stack frame analysis
    unsafe fn read_stack_peak(&self, sp: usize) -> u32 {
        let stack_end_ptr = self.get_stack_end();
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
    #[allow(clippy::empty_loop)]
    fn reset(&self) -> ! {
        loop {}
    }
}

/// Target-specific implementation of `CPUProfiler` for ARM Cortex-M microcontrollers.
///
/// This structure tracks execution cycles using the ARM DWT (Data Watchpoint and Trace) cycle
/// counter, measures elapsed time using the SysTick timer, and inspects the stack using linker symbols.
#[cfg(target_arch = "arm")]
pub struct CortexMProfiler {
    clock_frequency: u32,
    ticks: &'static core::sync::atomic::AtomicU32,
}

#[cfg(target_arch = "arm")]
impl CortexMProfiler {
    /// Creates a new `CortexMProfiler` instance.
    ///
    /// # Parameters
    /// * `clock_frequency` - CPU core clock frequency in Hz.
    /// * `ticks` - A static reference to an atomic millisecond counter updated by SysTick.
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
        // SAFETY: 0xE000_1004 is the address of the DWT CYCCNT (Cycle Count) register on Cortex-M
        // processors. Reading this address is safe and has no side effects, provided DWT has been
        // enabled in the core debug control registers (which is handled by `enable_dwt_cycle_counter`).
        unsafe {
            let dwt_cyccnt = 0xE000_1004 as *const u32;
            core::ptr::read_volatile(dwt_cyccnt) as u64
        }
    }

    fn get_nanos(&self) -> u64 {
        let ms = self.ticks.load(core::sync::atomic::Ordering::Relaxed) as u64;
        let current = cortex_m::peripheral::SYST::get_current();
        let reload = self.clock_frequency / 1000 - 1;
        let cycles = reload.saturating_sub(current);
        let us = ms * 1000
            + (cycles as u64) / (self.clock_frequency as u64 / 1_000_000);
        us * 1000
    }

    fn get_sp(&self) -> usize {
        let sp: usize;
        // SAFETY: Reading the active stack pointer `sp` using inline assembly does not write to
        // memory, affect any active stack frames, or corrupt processor flags. The assembly options
        // `nomem` and `nostack` verify these constraints.
        unsafe {
            core::arch::asm!("mov {}, sp", out(reg) sp, options(nomem, nostack, preserves_flags));
        }
        sp
    }

    fn get_stack_end(&self) -> usize {
        // SAFETY: The external static variable `_stack_end` is provided by the linker script (generated by
        // `build.rs` as the limit of the FLASH/RAM boundaries). It is safe to take the address of this
        // symbol because we treat it strictly as a numeric boundary address, not as a dereferenced u32 value.
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
/// measures elapsed time using the `time` CSR register, and calculates stack limits using linker-defined variables.
#[cfg(target_arch = "riscv32")]
pub struct RiscvProfiler;

#[cfg(target_arch = "riscv32")]
impl RiscvProfiler {
    /// Creates a new `RiscvProfiler` instance.
    pub const fn new() -> Self {
        Self
    }
}

#[cfg(target_arch = "riscv32")]
impl CPUProfiler for RiscvProfiler {
    fn disable_interrupts<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        // SAFETY: Disabling and enabling global interrupts via the RISC-V status register (mstatus) is
        // safe because it does not violate memory safety or cause undefined behavior. We restore the state
        // of the global interrupts once the critical section code (the closure `f`) completes.
        unsafe {
            riscv::interrupt::disable();
        }
        let res = f();
        // SAFETY: Re-enabling interrupts is safe as we are restoring normal execution flow.
        unsafe {
            riscv::interrupt::enable();
        }
        res
    }

    fn disable_interrupts_permanently(&self) {
        // SAFETY: Disabling global interrupts permanently is safe and ensures that no ISRs will execute
        // during panic/failure recovery.
        unsafe {
            riscv::interrupt::disable();
        }
    }

    fn get_cycles(&self) -> u64 {
        riscv::register::mcycle::read() as u64
    }

    fn get_nanos(&self) -> u64 {
        let ticks = riscv::register::time::read64();
        ticks * 100
    }

    fn get_sp(&self) -> usize {
        let sp: usize;
        // SAFETY: Reading the active stack pointer `sp` using inline assembly does not write to
        // memory, affect any active stack frames, or corrupt processor flags. The assembly options
        // `nomem` and `nostack` verify these constraints.
        unsafe {
            core::arch::asm!("mv {}, sp", out(reg) sp, options(nomem, nostack, preserves_flags));
        }
        sp
    }

    fn get_stack_end(&self) -> usize {
        // SAFETY: The external symbols `_stack_start` and `_hart_stack_size` are provided by the RISC-V linker
        // configuration. It is safe to take the addresses of these static symbols to perform pointer arithmetic
        // because we treat them strictly as numeric boundaries, not as dereferenced values.
        unsafe extern "C" {
            static _stack_start: u32;
            static _hart_stack_size: u32;
        }
        let stack_start_ptr = core::ptr::addr_of!(_stack_start) as usize;
        let stack_size = core::ptr::addr_of!(_hart_stack_size) as usize;
        stack_start_ptr.saturating_sub(stack_size)
    }
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
}
