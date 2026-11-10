//! CPU Profiling and Stack Measurement
//!
//! Demonstrates the `control-rs-ets` hardware profiling hooks:
//!
//! 1. `CPUProfiler` trait: abstract interface for target performance metrics
//!    (clock cycles, elapsed wall-clock duration, stack pointer and boundaries,
//!    and interrupt isolation).
//! 2. Stack painting and high-watermark inspection: sentinel word painting
//!    (`0xCDCD_CDCD`) and linear scan for peak stack depth without external probes.

use control_rs_ets::profiler::{CPUProfiler, HostCPUProfiler};

/// Target stack buffer for simulating stack painting and inspection.
struct SimulatedStack {
    memory: [u32; 128],
}

/// A profiler that wraps a simulated stack buffer.
struct SimulatedProfiler<'a> {
    stack: &'a mut SimulatedStack,
}

impl SimulatedStack {
    const fn new() -> Self {
        Self { memory: [0; 128] }
    }
}

impl CPUProfiler for SimulatedProfiler<'_> {
    fn get_cycles(&self) -> u64 {
        0
    }

    fn get_nanos(&self) -> u64 {
        0
    }

    fn get_sp(&self) -> usize {
        // Active stack pointer near the top of the buffer (index 96 of 128)
        core::ptr::addr_of!(self.stack.memory[96]) as usize
    }

    fn get_stack_end(&self) -> usize {
        // Bottom boundary of the stack buffer (index 0)
        core::ptr::addr_of!(self.stack.memory[0]) as usize
    }
}

fn main() {
    println!("=== control-rs: CPU Profiler Example ===");

    demonstrate_host_profiler();
    demonstrate_stack_painting();
}

/// Demonstrates default `HostCPUProfiler` capabilities.
fn demonstrate_host_profiler() {
    println!("\n--- 1. HostCPUProfiler Default Hooks ---");
    let profiler = HostCPUProfiler;
    println!("  has_clock()    : {}", profiler.has_clock());
    println!("  get_cycles()   : {}", profiler.get_cycles());
    println!("  get_nanos()    : {} ns", profiler.get_nanos());
}

/// Demonstrates stack sentinel painting and high-watermark stack measurement.
fn demonstrate_stack_painting() {
    println!("\n--- 2. Stack Painting & High-Watermark Measurement ---");
    let mut stack_storage = SimulatedStack::new();
    let profiler = SimulatedProfiler {
        stack: &mut stack_storage,
    };

    let sp = profiler.get_sp();
    let stack_end = profiler.get_stack_end();
    println!("  Stack Boundary End: {stack_end:#010x}");
    println!("  Active Stack Top  : {sp:#010x}");

    // 1. Paint unused stack memory with sentinel word 0xCDCD_CDCD
    unsafe {
        profiler.paint_stack(sp);
    }
    println!(
        "  Unused stack memory painted with sentinel bytes (0xCDCD_CDCD)."
    );

    // 2. Measure initial baseline peak (relative to SP minus 32-byte safety margin)
    let initial_peak = unsafe { profiler.read_stack_peak(sp) };
    println!("  Initial Baseline Stack Peak: {initial_peak} bytes");

    // 3. Simulate deep function call chain consuming stack space
    // Corrupt sentinel words starting at index 50 (displacing 46 words below index 96)
    profiler.stack.memory[50] = 0x1234_5678;
    let measured_peak = unsafe { profiler.read_stack_peak(sp) };
    println!("  Simulated Call Frame Displacement Peak: {measured_peak} bytes");
}
