//! Hardware profiler and timer ETS test suite for RISC-V targets.
#![cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]

use control_rs_macros::ets_suite;

#[ets_suite]
/// RISC-V hardware profiler and CSR timer verification suite.
pub mod riscv_profiler_test_suite {
    use control_rs_ets::RiscvProfiler;
    use control_rs_ets::profiler::CPUProfiler;

    /// Verifies that the RISC-V cycle counter (`mcycle`) advances monotonically across execution.
    fn test_hardware_riscv_mcycle_monotonic_progression() {
        let profiler = RiscvProfiler::new(10_000_000);
        let start_cycles = profiler.get_cycles();

        // Perform computation to advance cycles
        let mut sum = 0u64;
        for i in 0..1_000u64 {
            sum = sum.wrapping_add(i.wrapping_mul(3));
            core::hint::black_box(sum);
        }

        let end_cycles = profiler.get_cycles();
        assert!(
            end_cycles > start_cycles,
            "RISC-V mcycle must advance across computation: start={start_cycles}, end={end_cycles}"
        );
    }

    /// Verifies that RISC-V timer/nanosecond calculations advance monotonically.
    fn test_hardware_riscv_nanos_monotonic_progression() {
        let profiler = RiscvProfiler::new(10_000_000);
        let mut prev = profiler.get_nanos();
        for _ in 0..100 {
            let curr = profiler.get_nanos();
            assert!(
                curr >= prev,
                "RISC-V timer nanoseconds must be monotonic: curr={curr} < prev={prev}"
            );
            prev = curr;
        }
    }

    /// Verifies that the stack pointer read lies within a reasonable non-zero range.
    fn test_hardware_riscv_stack_pointer_validity() {
        let profiler = RiscvProfiler::new(10_000_000);
        let sp = profiler.get_sp();
        assert!(
            sp > 0x1000,
            "RISC-V stack pointer must be valid: sp={sp:#x}"
        );
    }

    /// Verifies that the reported core clock frequency matches the hardware configuration.
    fn test_hardware_riscv_clock_frequency() {
        let profiler = RiscvProfiler::new(10_000_000);
        assert_eq!(profiler.core_clock_hz(), 10_000_000);
    }
}
