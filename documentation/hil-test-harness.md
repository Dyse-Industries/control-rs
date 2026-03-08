## Hardware-In-the-Loop (HIL) Benchmarking

Verifying control algorithms on the host machine is a good start, but actual cycle counts and hardware-specific
behaviors (like DSP intrinsics and floating-point unit quirks) can only be validated on the target silicon.

`control-rs` facilitates a HIL benchmarking workflow using `probe-rs`. By using a custom test harness macro, you
can compile your unit and system tests, flash them to the target, and spin up an interactive Real-Time Transfer (RTT)
server. This allows you to trigger, benchmark, and re-run specific tests on the live hardware without needing to
re-flash the board.

### 1. Cargo Configuration

First, configure `probe-rs` as your default runner for your embedded target and set up a Cargo alias to trigger the HIL
harness.

**`.cargo/config.toml`**

```toml
[target.thumbv7em-none-eabihf] # Replace with your MCU's architecture
# `probe-rs run` will flash the chip and immediately attach an RTT terminal
runner = "probe-rs run --chip STM32F401RETx"

[alias]
# Compiles the test harness and runs `probe-rs` with a specific target.
hil = "run --test hil_benchmarks --release --target thumbv7em-none-eabihf"

```

### 2. The Test Harness Macro

The following macro generates an embedded `main` function that acts as a listening server over the RTT down-channel. The
user manually configures the setup routine and maps keystrokes to specific test functions.

**`tests/hil_benchmarks.rs`**

```rust
#![no_std]
#![no_main]

use cortex_m_rt::entry;
use rtt_target::{rprintln, rtt_init_default, DownChannel};
use panic_probe as _;

/// Generates an interactive RTT server for running benchmarks on-target.
macro_rules! hil_test_harness {
    (
        setup: $setup:path,
        tests: { $($cmd:expr => $test_name:ident : $test_func:path),* $(,)? }
    ) => {
        #[entry]
        fn main() -> ! {
            let channels = rtt_init_default!();
            let mut down_channel = channels.down.0;
            $setup();

            rprintln!("--- control-rs HIL Benchmark Server Ready ---");
            rprintln!("Available Commands:");
            $( rprintln!("  [{}] -> Run {}", $cmd, stringify!($test_name)); )*
            rprintln!("  [q] -> Finish and enter safe idle state");

            let mut buf = [0u8; 1];
            loop {
                // Poll the RTT down-channel for host commands
                if down_channel.read(&mut buf) > 0 {
                    match buf[0] as char {
                        $(
                            $cmd => {
                                rprintln!("Executing {}...", stringify!($test_name));
                                // In a real setup, start the DWT cycle counter here
                                $test_func();
                                // Stop the DWT cycle counter and print results here
                                rprintln!("Done.");
                            }
                        )*
                        'q' => {
                            rprintln!("Tests complete. Jumping to safe no-op loop.");
                            loop { cortex_m::asm::nop(); }
                        }
                        _ => rprintln!("Unknown command."),
                    }
                }
            }
        }
    };
}

```

### 3. Integrating Your Tests

You can now wrap your algorithmic unit tests and system-level validation routines into the harness.

```rust
// --- Your Test Functions ---

fn setup_hardware() {
    // Enable DWT cycle counters, initialize DSP peripherals, etc.
}

fn bench_axpy_controller() {
    use control_rs::math::subprograms::level1::AXPY;
    use core::marker::PhantomData;

    // Using the Controller from the previous example
    let controller = Controller::<CmsisDspBackend> { _marker: PhantomData };
    let mut state = [0.0; 64];
    let input = [1.5; 64];

    let _ = controller.update(&mut state, &input, 0.5);
}

fn bench_kalman_filter() {
    // Execute a simulated step of the Kalman filter
}

// --- Harness Configuration ---

hil_test_harness! {
    setup: setup_hardware,
    tests: {
        '1' => AxpyControllerUpdate : bench_axpy_controller,
        '2' => KalmanFilterStep     : bench_kalman_filter,
    }
}

```

### 4. Running the HIL Server

Trigger the process using the alias defined earlier:

```bash
cargo hil
```

**Output:**

```text
      Erasing ✔ [00:00:00] [#####################] 16.00 KiB/16.00 KiB @ 31.97 KiB/s
  Programming ✔ [00:00:00] [#####################] 16.00 KiB/16.00 KiB @ 40.52 KiB/s
--- control-rs HIL Benchmark Server Ready ---
Available Commands:
  [1] -> Run AxpyControllerUpdate
  [2] -> Run KalmanFilterStep
  [q] -> Finish and enter safe idle state

```

Because `probe-rs` handles the bidirectional RTT stream, you can simply type `1` or `2` into your terminal to execute
and re-execute the benchmarks in real-time, observing the cycle counts directly from the hardware. When you are
finished, typing `q` safely parks the microcontroller in a no-op loop.