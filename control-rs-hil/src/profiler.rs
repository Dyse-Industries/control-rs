//! Hardware execution abstraction and CPU profiling for HIL tests.
//!
//! Provides traits to measure hardware performance metrics (like clock cycles, execution time,
//! and stack space consumption) in a hardware-agnostic manner.

/// Target-specific utilities for CPU profiling and execution wrapping during HIL tests.
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
    #[allow(clippy::empty_loop)]
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
    /// This writes to the stack memory space. The caller must ensure that the stack pointer is valid
    /// and that the painting bounds do not overwrite any active stack frames or reserved memory.
    unsafe fn paint_stack(&self, sp: usize) {
        let stack_end_ptr = self.get_stack_end();
        // Leave a 32-byte safety margin below the current stack pointer to avoid
        // overwriting active stack frames (like paint_stack's own frame and return address).
        let limit = sp.saturating_sub(32);

        if limit > stack_end_ptr {
            let mut ptr = stack_end_ptr as *mut u32;
            let limit_ptr = limit as *mut u32;

            while ptr < limit_ptr {
                unsafe {
                    core::ptr::write_volatile(ptr, 0xCDCD_CDCD);
                    ptr = ptr.add(1);
                }
            }
        }
    }

    /// Reads the peak stack usage (in bytes) since the stack was painted, relative to the given stack pointer.
    ///
    /// # Safety
    /// This reads from the stack memory space. The caller must ensure that the stack has been painted
    /// and that memory accesses remain within the valid stack bounds.
    unsafe fn read_stack_peak(&self, sp: usize) -> u32 {
        let stack_end_ptr = self.get_stack_end();
        let mut ptr = stack_end_ptr as *const u32;
        let limit_ptr = sp as *const u32;

        while ptr < limit_ptr {
            let is_sentinel =
                unsafe { core::ptr::read_volatile(ptr) == 0xCDCD_CDCD };
            if !is_sentinel {
                break;
            }
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
#[cfg(target_arch = "arm")]
pub struct CortexMProfiler {
    clock_frequency: u32,
    ticks: &'static core::sync::atomic::AtomicU32,
}

#[cfg(target_arch = "arm")]
impl CortexMProfiler {
    /// Creates a new `CortexMProfiler` instance.
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

/// Target-specific implementation of `CPUProfiler` for RISC-V targets.
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
        unsafe {
            riscv::interrupt::disable();
        }
        let res = f();
        unsafe {
            riscv::interrupt::enable();
        }
        res
    }

    fn disable_interrupts_permanently(&self) {
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
        unsafe {
            core::arch::asm!("mv {}, sp", out(reg) sp, options(nomem, nostack, preserves_flags));
        }
        sp
    }

    fn get_stack_end(&self) -> usize {
        unsafe extern "C" {
            static _stack_start: u32;
            static _hart_stack_size: u32;
        }
        let stack_start_ptr = core::ptr::addr_of!(_stack_start) as usize;
        let stack_size = core::ptr::addr_of!(_hart_stack_size) as usize;
        stack_start_ptr.saturating_sub(stack_size)
    }
}
