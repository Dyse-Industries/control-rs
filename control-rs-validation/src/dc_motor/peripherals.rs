//! src/peripherals.rs
//!
//! Peripheral models for the DC motor position control loop:
//! - **Noisy Actuator**: Voltage saturation, circular delay buffer, additive voltage noise.
//! - **Delayed & Noisy Sensor**: 16-bit absolute position encoder quantization,
//!   circular delay buffer, additive angle measurement noise.
//! - **Deterministic PRNG**: Zero-dependency xorshift64* generator with Box-Muller transform.

use std::collections::VecDeque;

/// Resolution of the 16-bit absolute encoder ($2^{16} = 65,536$ counts per revolution).
pub const ENCODER_BITS: u32 = 16;
pub const ENCODER_COUNTS: f64 = 65536.0; // 2^16
pub const ENCODER_QUANTUM: f64 = (2.0 * core::f64::consts::PI) / ENCODER_COUNTS;

/// Lightweight deterministic XorShift64* PRNG for reproducible Gaussian noise without external dependencies.
#[derive(Debug, Clone)]
pub struct DeterministicPrng {
    state: u64,
}

impl DeterministicPrng {
    /// Creates a new PRNG with a non-zero seed.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x853c_49e6_748f_ea9b
            } else {
                seed
            },
        }
    }

    /// Generates next pseudo-random `u64`.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }

    /// Generates uniform float in interval `(0, 1]`.
    pub fn next_f64(&mut self) -> f64 {
        let val = (self.next_u64() >> 11) as f64;
        (val + 1.0) / 9007199254740993.0
    }

    /// Generates zero-mean Gaussian distributed float $\mathcal{N}(0, \sigma^2)$ via Box-Muller transform.
    pub fn next_gaussian(&mut self, sigma: f64) -> f64 {
        if sigma <= 0.0 {
            return 0.0;
        }
        let u1 = self.next_f64();
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * core::f64::consts::PI * u2;
        r * theta.cos() * sigma
    }
}

/// Noisy actuator model with circular delay queue and voltage clamping.
#[derive(Debug, Clone)]
pub struct Actuator {
    delay_steps: usize,
    buffer: VecDeque<f64>,
    noise_sigma_v: f64,
    v_max: f64,
    prng: DeterministicPrng,
}

impl Actuator {
    /// Constructs a new Actuator with the given delay steps, noise standard deviation, and voltage limit.
    #[must_use]
    pub fn new(
        delay_steps: usize,
        noise_sigma_v: f64,
        v_max: f64,
        seed: u64,
    ) -> Self {
        let mut buffer = VecDeque::with_capacity(delay_steps + 1);
        for _ in 0..delay_steps {
            buffer.push_back(0.0);
        }
        Self {
            delay_steps,
            buffer,
            noise_sigma_v,
            v_max,
            prng: DeterministicPrng::new(seed),
        }
    }

    /// Pushes a desired voltage command $u[k]$, applies transport delay, adds voltage noise,
    /// and clamps to $[-V_{\max}, +V_{\max}]$, returning the actual terminal voltage applied.
    pub fn step(&mut self, command_v: f64) -> f64 {
        // Saturation on input command
        let clamped_cmd = command_v.clamp(-self.v_max, self.v_max);

        let delayed_v = if self.delay_steps == 0 {
            clamped_cmd
        } else {
            self.buffer.push_back(clamped_cmd);
            self.buffer.pop_front().unwrap_or(0.0)
        };

        // Add actuator noise (PWM jitter / supply ripple)
        let noise = self.prng.next_gaussian(self.noise_sigma_v);
        (delayed_v + noise).clamp(-self.v_max, self.v_max)
    }

    /// Resets buffer states to zero.
    pub fn reset(&mut self) {
        self.buffer.clear();
        for _ in 0..self.delay_steps {
            self.buffer.push_back(0.0);
        }
    }
}

/// Delayed and noisy 16-bit absolute position encoder sensor.
#[derive(Debug, Clone)]
pub struct EncoderSensor {
    delay_steps: usize,
    buffer: VecDeque<f64>,
    noise_sigma_rad: f64,
    quantum_rad: f64,
    prng: DeterministicPrng,
}

impl EncoderSensor {
    /// Constructs a new 16-bit absolute position encoder sensor.
    #[must_use]
    pub fn new(delay_steps: usize, noise_sigma_rad: f64, seed: u64) -> Self {
        let mut buffer = VecDeque::with_capacity(delay_steps + 1);
        for _ in 0..delay_steps {
            buffer.push_back(0.0);
        }
        Self {
            delay_steps,
            buffer,
            noise_sigma_rad,
            quantum_rad: ENCODER_QUANTUM,
            prng: DeterministicPrng::new(seed),
        }
    }

    /// Quantizes continuous angle $\theta$ to the discrete 16-bit encoder grid.
    #[must_use]
    pub fn quantize(&self, theta_rad: f64) -> f64 {
        (theta_rad / self.quantum_rad).round() * self.quantum_rad
    }

    /// Takes the physical rotor angle $\theta(t)$, applies 16-bit quantization,
    /// adds sensor measurement noise, and routes through the transmission delay buffer.
    pub fn step(&mut self, physical_theta_rad: f64) -> f64 {
        // 1. Quantize through 16-bit encoder
        let quantized = self.quantize(physical_theta_rad);

        // 2. Add measurement / interface noise
        let noisy = quantized + self.prng.next_gaussian(self.noise_sigma_rad);

        // 3. Transport delay buffer
        if self.delay_steps == 0 {
            noisy
        } else {
            self.buffer.push_back(noisy);
            self.buffer.pop_front().unwrap_or(0.0)
        }
    }

    /// Resets buffer states to zero.
    pub fn reset(&mut self) {
        self.buffer.clear();
        for _ in 0..self.delay_steps {
            self.buffer.push_back(0.0);
        }
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::dc_motor::motor::V_MAX;

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-2
    /// Method: Requirements-based test
    fn test_encoder_16bit_quantization_resolution() {
        let encoder = EncoderSensor::new(0, 0.0, 42);

        // Quantum is 2*pi / 65536 ~= 9.587e-5 rad
        assert!((ENCODER_QUANTUM - 9.587379924285257e-5).abs() < 1e-12);

        // Value exactly on quantum
        let q1 = encoder.quantize(ENCODER_QUANTUM);
        assert!((q1 - ENCODER_QUANTUM).abs() < 1e-12);

        // Value at 0.49 * quantum should round down to 0
        let q_down = encoder.quantize(0.49 * ENCODER_QUANTUM);
        assert!(q_down.abs() < 1e-12);

        // Value at 0.51 * quantum should round up to 1 * quantum
        let q_up = encoder.quantize(0.51 * ENCODER_QUANTUM);
        assert!((q_up - ENCODER_QUANTUM).abs() < 1e-12);
    }

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-2
    /// Method: Requirements-based test
    fn test_actuator_delay_queue_behavior() {
        let mut act = Actuator::new(2, 0.0, V_MAX, 123);

        // With 2 delay steps, initial steps should return 0.0
        assert_eq!(act.step(10.0), 0.0);
        assert_eq!(act.step(10.0), 0.0);
        // Step 3 should now output the first 10.0
        assert_eq!(act.step(5.0), 10.0);
        // Step 4 outputs second 10.0
        assert_eq!(act.step(0.0), 10.0);
        // Step 5 outputs 5.0
        assert_eq!(act.step(0.0), 5.0);
    }

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-2
    /// Method: Requirements-based test
    fn test_actuator_voltage_saturation() {
        let mut act = Actuator::new(0, 0.0, 12.0, 456);
        assert_eq!(act.step(15.0), 12.0);
        assert_eq!(act.step(-20.0), -12.0);
        assert_eq!(act.step(8.5), 8.5);
    }

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-2
    /// Method: Requirements-based test
    fn test_sensor_delay_queue_behavior() {
        let mut sensor = EncoderSensor::new(1, 0.0, 789);
        let angle = 1.0;
        let quantized_angle = sensor.quantize(angle);

        assert_eq!(sensor.step(angle), 0.0);
        assert!((sensor.step(angle) - quantized_angle).abs() < 1e-12);
    }

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-2
    /// Method: Requirements-based test
    fn test_deterministic_prng_gaussian_properties() {
        let mut prng = DeterministicPrng::new(1001);
        let n = 10_000;
        let sigma = 0.5;
        let mut sum = 0.0;
        let mut sum_sq = 0.0;

        for _ in 0..n {
            let val = prng.next_gaussian(sigma);
            sum += val;
            sum_sq += val * val;
        }

        let mean = sum / (n as f64);
        let variance = (sum_sq / (n as f64)) - (mean * mean);

        assert!(
            mean.abs() < 0.02,
            "Gaussian mean should be near 0: {}",
            mean
        );
        assert!(
            (variance - sigma * sigma).abs() < 0.02,
            "Gaussian variance should be near {}: {}",
            sigma * sigma,
            variance
        );
    }
}
