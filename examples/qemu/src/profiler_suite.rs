//! Hardware profiler and timer ETS test suite for ARM Cortex-M.

use control_rs_macros::ets_suite;

#[ets_suite]
/// ARM Cortex-M hardware profiler and timer verification suite.
pub mod profiler_test_suite {
    use control_rs_ets::CortexMProfiler;
    use control_rs_ets::profiler::{
        CPUProfiler, SysTickReading, calculate_systick_nanos,
    };
    use core::sync::atomic::Ordering;

    /// Verifies monotonic time progression from hardware SysTick over thousands of samples
    /// with interrupts enabled so the SysTick ISR services counter rollovers.
    fn test_hardware_systick_monotonic_progression() {
        // ETS server executes tests with interrupts disabled; enable them to test live SysTick ISR servicing.
        unsafe {
            cortex_m::interrupt::enable();
        }

        let profiler =
            CortexMProfiler::new(crate::CLOCK_FREQ, &crate::MILLISECONDS);
        let mut prev = profiler.get_nanos();
        for _ in 0..5_000 {
            let curr = profiler.get_nanos();
            assert!(
                curr >= prev,
                "SysTick hardware time must advance monotonically: curr={curr} < prev={prev}"
            );
            prev = curr;
        }

        cortex_m::interrupt::disable();
    }

    /// Negative test demonstrating that naive SysTick calculation steps backwards across hardware rollover,
    /// while the hardened calculation maintains strict monotonicity.
    fn test_hardware_naive_systick_rollover_flawed_negative_test() {
        let reload = crate::CLOCK_FREQ.saturating_div(1000).saturating_sub(1);
        let cycles_per_us =
            u64::from(crate::CLOCK_FREQ.saturating_div(1_000_000));

        cortex_m::interrupt::free(|_| {
            let ms = u64::from(crate::MILLISECONDS.load(Ordering::Relaxed));

            // Busy wait until counter is close to wrapping (< 500 cycles before zero)
            let mut curr = cortex_m::peripheral::SYST::get_current();
            while curr > 500 {
                curr = cortex_m::peripheral::SYST::get_current();
            }
            let current_before = curr;
            let naive_before = {
                let cycles = u64::from(reload.saturating_sub(current_before));
                let us = ms
                    .saturating_mul(1000)
                    .saturating_add(cycles.saturating_div(cycles_per_us));
                us.saturating_mul(1000)
            };

            // Wait until counter reloads and ICSR PENDSTSET (bit 26) is flagged
            loop {
                curr = cortex_m::peripheral::SYST::get_current();
                let icsr = unsafe {
                    core::ptr::read_volatile(0xE000_ED04 as *const u32)
                };
                let pending = (icsr & (1 << 26)) != 0;
                if pending && curr > reload / 2 {
                    break;
                }
            }
            let current_after = curr;

            // Naive calculation suffers from stale ms because interrupts are masked
            let naive_after = {
                let cycles = u64::from(reload.saturating_sub(current_after));
                let us = ms
                    .saturating_mul(1000)
                    .saturating_add(cycles.saturating_div(cycles_per_us));
                us.saturating_mul(1000)
            };

            // Assert that naive calculation exhibits the backward time regression flaw
            assert!(
                naive_after < naive_before,
                "Naive SysTick on hardware must be flawed across rollover: after={naive_after} >= before={naive_before}"
            );

            // Hardened calculation reconciles the pending rollover bit and advances forward
            let reading_after = SysTickReading {
                current: current_after,
                pending: true,
            };
            let hardened_after = calculate_systick_nanos(
                ms,
                reading_after,
                reload,
                crate::CLOCK_FREQ,
            );
            assert!(
                hardened_after > naive_before,
                "Hardened calculation on hardware must advance monotonically across rollover"
            );
        });
    }

    /// Verifies that CortexMProfiler accurately computes wall-clock time from hardware SysTick
    /// consistent with the underlying millisecond tick counter.
    fn test_hardware_cortex_m_profiler_time_accuracy() {
        let profiler =
            CortexMProfiler::new(crate::CLOCK_FREQ, &crate::MILLISECONDS);
        let t1 = profiler.get_nanos();
        let ms1 = u64::from(crate::MILLISECONDS.load(Ordering::Relaxed));
        // t1 in nanoseconds must be >= ms1 * 1_000_000 and within 2 ms of the tick counter
        assert!(
            t1 >= ms1.saturating_mul(1_000_000),
            "Computed nanos ({t1}) must be >= base millisecond tick ({ms1} ms)"
        );
        assert!(
            t1 <= (ms1 + 2).saturating_mul(1_000_000),
            "Computed nanos ({t1}) must be within 2 ms of base millisecond tick ({ms1} ms)"
        );
    }
}
