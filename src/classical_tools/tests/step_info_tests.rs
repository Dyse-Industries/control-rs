//! # Step Response Transient Metrics Extraction Tests
// Expected values here are hand-derived reference expressions; the `mul_add`
// form clippy suggests obscures the algebra being asserted and is not a
// performance concern in a test oracle.
#![allow(clippy::suboptimal_flops)]
#![allow(clippy::indexing_slicing, clippy::float_cmp)]

#[cfg_attr(not(test), control_rs_macros::ets_suite)]
pub mod step_info_test_suite {
    use crate::classical_tools::step_info::step_info;

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-10
    /// Method: Requirements-based test
    fn test_empty_or_zero_step_returns_defaults() {
        let empty_time: [f64; 0] = [];
        let empty_y: [f64; 0] = [];
        let info = step_info(&empty_time, &empty_y, 0.0, 0.0, 1.0, None);
        assert_eq!(info.rise_time, 0.0);
        assert_eq!(info.settling_time, None);

        let t = [0.0, 1.0, 2.0];
        let y = [5.0, 5.0, 5.0];
        let info_zero = step_info(&t, &y, 0.0, 5.0, 5.0, None);
        assert_eq!(info_zero.rise_time, 0.0);
        assert_eq!(info_zero.steady_state_error, 0.0);
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-10
    /// Method: Requirements-based test
    fn test_step_info_well_settled_response() {
        // Ideal 1st order step response: y(t) = 1 - exp(-t / tau) with tau = 0.1 s
        // Step applied at t = 0.1 s from initial 0.0 to target 1.0
        // Rise time (10% to 90%): ln(9) * tau = 2.197 * 0.1 = 0.220 s
        // 2% settling time: -ln(0.02) * tau = 3.912 * 0.1 = 0.391 s
        let mut time = [0.0; 500];
        let mut y = [0.0; 500];
        let dt = 0.002;
        let tau = 0.1;
        let t_step = 0.1;

        for i in 0..500 {
            let t = (i as f64) * dt;
            time[i] = t;
            if t < t_step {
                y[i] = 0.0;
            } else {
                let elapsed = t - t_step;
                y[i] = 1.0 - libm::exp(-elapsed / tau);
            }
        }

        let info = step_info(&time, &y, t_step, 0.0, 1.0, Some(0.02));
        assert!(
            (info.rise_time - 0.22).abs() < 0.01,
            "Rise time {info_rise_time} vs expected ~0.22",
            info_rise_time = info.rise_time
        );
        assert_eq!(info.peak_overshoot_pct, 0.0);
        assert!(info.settling_time.is_some());
        let ts = info.settling_time.unwrap();
        assert!(
            (ts - 0.39).abs() < 0.01,
            "Settling time {ts} vs expected ~0.39"
        );
        assert!(info.steady_state_error < 0.01);
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-10
    /// Method: Requirements-based test
    fn test_step_info_type_0_offset_unsettled_target() {
        // Type-0 response settling to 5.409 V with target 5.5 V (initial 5.0 V)
        // 2% band of 0.5 V step is 0.01 V = 10 mV.
        // SSE is 91 mV > 10 mV, so settling_time to target must be None,
        // while settling_time_achieved should be a valid positive duration.
        let mut time = [0.0; 300];
        let mut y = [0.0; 300];
        let dt = 0.00001;
        let t_step = 0.0005;

        for i in 0..300 {
            let t = (i as f64) * dt;
            time[i] = t;
            if t < t_step {
                y[i] = 5.0;
            } else {
                let elapsed = t - t_step;
                // Asymptote to 5.409 with an overshoot peak at 5.56
                let exp_decay = libm::exp(-elapsed / 0.0001);
                y[i] = 5.409 + 0.3 * exp_decay * libm::sin(elapsed * 20000.0);
            }
        }

        let info = step_info(&time, &y, t_step, 5.0, 5.5, Some(0.02));
        assert_eq!(
            info.settling_time, None,
            "Target settling time must be None when offset exceeds tolerance band"
        );
        assert!(
            info.settling_time_achieved > 0.0,
            "Achieved settling time must be non-zero"
        );
        let sse = info.steady_state_error;
        assert!((sse - 0.091).abs() < 0.002, "SSE: {sse}");
    }
}
