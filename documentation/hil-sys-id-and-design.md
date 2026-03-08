## Real-Time Model Tuning and Estimation via HIL: A Rust Developer's Guide

Performing system identification (SysId) entirely offline has a much slower turnaround time.

By combining the **Adaptive Estimator** pattern with the **HIL Test Harness**, `control-rs` allows you to perform
real-time model tuning and parameter estimation directly on the target silicon.

This guide demonstrates how to build an interactive firmware server over Real-Time Transfer (RTT), enabling you to step
a physical actuator, run an estimator and stream the converging parameters back to your host in real-time.

### The SysId Workflow

The standard embedded Rust workflow for HIL tuning with `control-rs` looks like this:

1. **Trigger:** The host machine sends a tuning command (e.g., `'t'`) over the RTT down-channel.
2. **Excite:** The microcontroller applies a known excitation signal (like a PRBS or a simple step voltage) to the
   physical hardware.
3. **Estimate:** The target samples the sensors, feeds the data into the `RecursiveEstimator`, and securely updates the
   internal model covariance using safe, zero-allocation math.
4. **Stream:** The target streams the converging parameters back to the host over the RTT up-channel for visualization
   or verification.

### Implementing the Harness

Below is an example of wrapping a motor tuning routine inside the `hil_test_harness!` macro. This example emphasizes
Rust's ability to gracefully handle mathematical divergence on bare-metal hardware using `ArithmeticResult`.

```rust
#![no_std]
#![no_main]

use cortex_m_rt::entry;
use rtt_target::{rprintln, rtt_init_default, DownChannel};
use panic_probe as _;

// Import our control-rs components
use control_rs::math::ArithmeticError;
use core::marker::PhantomData;

// Assume we have defined a `MotorModel` and initialized a hardware abstraction
// let mut hardware = TractionMotorDriver::new();

fn run_traction_motor_sysid() {
    rprintln!("--- Starting Real-Time Motor SysId ---");

    // 1. Initialize the statically allocated estimator
    // Estimating 2 parameters: Torque Constant (Kt) and Back-EMF (Ke)
    let mut estimator = RecursiveEstimator::<MotorModel, CmsisDspBackend, 2, 1, 2> {
        model: MotorModel::new(),
        estimated_params: [0.01, 0.01], // Initial rough guesses
        covariance_matrix: [[1.0, 0.0], [0.0, 1.0]], // Initial confidence
        _backend: PhantomData,
    };

    let sample_time_ms = 10;

    // 2. Execute the excitation profile (e.g., 100 samples)
    for step in 0..100 {
        // Apply a step input voltage
        let input = [12.0];

        // Read current state (e.g., [Velocity, Current])
        let current_state = hardware.read_sensors();

        hardware.apply_voltage(input[0]);
        hardware.delay_ms(sample_time_ms);

        // Read the resulting state after the applied input
        let next_state = hardware.read_sensors();

        // 3. Update the model parameters online
        match estimator.update_model(&current_state, &input, &next_state) {
            Ok(_) => {
                // 4. Stream converging parameters back to the host
                let [kt, ke] = estimator.estimated_params;
                rprintln!("Step {:03} | Kt: {:.5} | Ke: {:.5}", step, kt, ke);
            }
            Err(ArithmeticError::PrecisionLoss) => {
                // Catch covariance singularity or matrix wind-up before it causes a hard fault
                rprintln!("ERROR: Estimator lost precision at step {}. Excitation may be insufficient.", step);
                break;
            }
            Err(ArithmeticError::DomainViolation) => {
                // Catch NaN propagation from sensor noise
                rprintln!("ERROR: Domain violation (NaN detected) from noisy sensor read.");
                break;
            }
            Err(e) => {
                rprintln!("ERROR: Unhandled math exception: {}", e);
                break;
            }
        }
    }

    // Safely spin down the hardware after tuning
    hardware.apply_voltage(0.0);
    rprintln!("--- SysId Routine Complete ---");
}

// --- Harness Configuration ---

hil_test_harness! {
    setup: || { /* Init system clocks, ADC, PWM */ },
    tests: {
        't' => TuneTractionMotor : run_traction_motor_sysid,
    }
}

```

### The Developer Experience

When you run `cargo hil` and press `'t'` in the terminal, the target immediately begins executing the physical movement
and streaming the data:

```text
--- control-rs HIL Benchmark Server Ready ---
Available Commands:
  [t] -> Run TuneTractionMotor
  [q] -> Finish and enter safe idle state
t
Executing TuneTractionMotor...
--- Starting Real-Time Motor SysId ---
Step 000 | Kt: 0.01520 | Ke: 0.01480
Step 001 | Kt: 0.02105 | Ke: 0.01950
Step 002 | Kt: 0.02450 | Ke: 0.02210
...
Step 099 | Kt: 0.03102 | Ke: 0.03115
--- SysId Routine Complete ---
Done.

```

### Why This Architecture Matters for Embedded Rust

1. **Safety First, Without the Overhead:** Notice that there are no `unwrap()` calls on the estimation math.
   Floating-point math on real-world sensor data is inherently chaotic. Rust's `match` statement forces you to handle
   `ArithmeticError`, ensuring that a mathematically unstable model never silently crashes your firmware or sends
   unbound parameters to an active actuator.
2. **Immediate Turnaround:** You skip the entire cycle of logging CSVs to an SD card, moving them to a PC, parsing them
   in a Python script, and recompiling gains into firmware. The parameters converge on the hardware, using the exact
   same arithmetic implementations that will govern them in production.
3. **Automated MBD Integration:** Because this runs over standard `probe-rs` channels, you can write a simple host-side
   Rust script that triggers the `'t'` command over TCP, parses the `rprintln!` output, and automatically updates a
   configuration `const` in your codebase.