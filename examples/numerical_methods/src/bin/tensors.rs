//! Demonstration of the generic compile-time Tensor struct.
//!
//! This example models a gain-scheduled controller for an autonomous vehicle (e.g., a drone or aircraft).
//! Controller gains (Kp, Ki, Kd) are scheduled over a 3D grid corresponding to operating conditions:
//! Altitude levels (2) x Airspeed levels (3) x PID Gain Types (3).

use control_rs::math::num_types::Const;
use control_rs::tensor::{Tensor, TensorLayout};

fn main() {
    println!("=== Tensor Demonstration (Gain Scheduling Lookup) ===");

    // 1. Define dimensions and layout
    // Shape: [Altitude (2 levels) x Airspeed (3 levels) x Gain Types (3)]
    // Altitude: 0 = Low Altitude, 1 = High Altitude
    // Airspeed: 0 = Slow Speed, 1 = Cruise Speed, 2 = Fast Speed
    // Gain Type: 0 = Kp (Proportional), 1 = Ki (Integral), 2 = Kd (Derivative)
    type GainSchedulerLayout = (Const<2>, Const<3>, Const<3>);

    println!("Gain Scheduler Tensor layout properties:");
    println!("  Rank (dimensions): {}", GainSchedulerLayout::RANK);
    println!("  Size (total elements): {}", GainSchedulerLayout::SIZE);
    println!("  Shape dimensions: {:?}", GainSchedulerLayout::dims());

    // 2. Construction
    // A 3D tensor representing pre-tuned gains at each operating point.
    // Memory layout is column-major: the first dimension (Altitude) varies fastest.
    // Flat index = alt_idx + speed_idx * 2 + gain_idx * 6
    let mut gain_scheduler = Tensor::<f64, 18, GainSchedulerLayout>::new([
        // Kp gains for:
        // [Low Alt, Slow], [High Alt, Slow],
        // [Low Alt, Cruise], [High Alt, Cruise],
        // [Low Alt, Fast], [High Alt, Fast]
        2.0, 2.5, 1.8, 2.2, 1.5, 1.9,
        // Ki gains for the same operating points:
        0.5, 0.6, 0.4, 0.5, 0.3, 0.4,
        // Kd gains for the same operating points:
        0.1, 0.12, 0.08, 0.1, 0.05, 0.07,
    ]);

    // 3. Multi-coordinate Lookup
    // Query gains at Low Altitude (0) and Cruise Speed (1)
    let alt_idx = 0;
    let speed_idx = 1;

    println!(
        "\nQuerying scheduled controller gains at Low Altitude and Cruise Speed:"
    );
    println!("  Kp = {:?}", gain_scheduler.get(&[alt_idx, speed_idx, 0])); // Expected: Some(1.8)
    println!("  Ki = {:?}", gain_scheduler.get(&[alt_idx, speed_idx, 1])); // Expected: Some(0.4)
    println!("  Kd = {:?}", gain_scheduler.get(&[alt_idx, speed_idx, 2])); // Expected: Some(0.08)

    // Out of bounds checking
    println!("\nQuerying out-of-bounds coords [2, 0, 0]:");
    println!("  Result = {:?}", gain_scheduler.get(&[2, 0, 0])); // Expected: None

    // Dimension mismatch checking
    println!("Querying coordinate mismatch [0, 0]:");
    println!("  Result = {:?}", gain_scheduler.get(&[0, 0])); // Expected: None

    // 4. Mutation
    // Suppose we fine-tune the high-altitude, fast speed Kd gain (coords [1, 2, 2]) during testing.
    if let Some(val) = gain_scheduler.get_mut(&[1, 2, 2]) {
        *val = 0.075;
    }
    println!("\nAfter fine-tuning High Altitude, Fast Speed Kd gain:");
    println!("  Kd = {:?}", gain_scheduler.get(&[1, 2, 2])); // Expected: Some(0.075)

    // 5. Tensor Arithmetic
    // Scenario A: Scaling gains. In high-turbulence conditions, we scale down all gains by 80%
    // to prevent over-control and reduce actuator strain.
    let scaled_scheduler = gain_scheduler * 0.8;
    println!(
        "\nScaled gains (80% for high turbulence) at Low Altitude, Cruise Speed:"
    );
    println!(
        "  Kp = {:?}",
        scaled_scheduler.get(&[alt_idx, speed_idx, 0])
    ); // Expected: Some(1.44)

    // Scenario B: Adding offsets. We add bias correction offsets to all gains.
    let bias_scheduler = Tensor::<f64, 18, GainSchedulerLayout>::new([
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, // Kp offsets
        0.02, 0.02, 0.02, 0.02, 0.02, 0.02, // Ki offsets
        0.005, 0.005, 0.005, 0.005, 0.005, 0.005, // Kd offsets
    ]);
    let corrected_scheduler = gain_scheduler + bias_scheduler;
    println!(
        "\nCorrected gains (with bias offset) at Low Altitude, Cruise Speed:"
    );
    println!(
        "  Kp = {:?}",
        corrected_scheduler.get(&[alt_idx, speed_idx, 0])
    ); // Expected: Some(1.9)
}
