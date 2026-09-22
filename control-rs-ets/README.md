# `control-rs-ets`

Target-side `no_std` Embedded Test Server (ETS): transport and profiler
traits, framed command and telemetry protocol, and the server event loop.
[Design](../documentation/ets/embedded-test-server-design.md) · [Overview](../documentation/ets/ets-overview.md) · [Workspace](../README.md)

| Module | Contents | Design |
|:--|:--|:--|
| `comms` | `HostComms`, frame reader, telemetry framing (CRC-16) | [host-comm-design](../documentation/ets/host-comm-design.md) |
| `profiler` | `CPUProfiler` cycle and stack measurement | [cpu-profiler-design](../documentation/ets/cpu-profiler-design.md) |
| `settings` | Host-adjustable atomic test settings | [test-suite-design](../documentation/ets/test-suite-design.md) |
| `server` | `Context` and the event loop | [embedded-test-server-design](../documentation/ets/embedded-test-server-design.md) |

## End-User Example

Implement `HostComms` for the target transport and `CPUProfiler` for its
timer, then pass both to `Context`:

```rust
#![no_std]
#![no_main]

use control_rs_ets::comms::{Command, FrameReader, HostComms, Telemetry, frame_telemetry};
use control_rs_ets::server::Context;
use control_rs_ets::CPUProfiler;
use control_rs_macros::{ets_setup, ets_suite};

// 1. Define target-side communication channel (e.g., UART or Semihosting)
struct UartComms {
    reader: FrameReader,
}

impl HostComms for UartComms {
    type Error = ();

    fn poll_command(&mut self) -> Result<Option<Command>, Self::Error> {
        // Retrieve byte from hardware peripheral non-blockingly
        if let Some(byte) = read_uart_byte() {
            if let Some(payload) = self.reader.handle_byte(byte) {
                if let Ok(cmd) = postcard::from_bytes(payload) {
                    return Ok(Some(cmd));
                }
            }
        }
        Ok(None)
    }

    fn send_telemetry(&mut self, telemetry: &Telemetry<'_>) -> Result<(), Self::Error> {
        let mut buf = [0u8; 512];
        if let Ok(len) = frame_telemetry(telemetry, &mut buf) {
            for &byte in &buf[..len] {
                write_uart_byte(byte);
            }
            Ok(())
        } else {
            Err(())
        }
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

// 2. Define target-side CPU profiling utilities
struct SystemCPUUtils;

impl CPUProfiler for SystemCPUUtils {
    fn get_cycles(&self) -> u64 {
        get_cycles()
    }

    fn get_nanos(&self) -> u64 {
        get_nanoseconds()
    }

    fn get_sp(&self) -> usize {
        get_sp()
    }

    unsafe fn paint_stack(&self, sp: usize) {
        // Hardware stack painting logic
    }

    unsafe fn read_stack_peak(&self, sp: usize) -> u32 {
        // Hardware stack peak scanning logic
        0
    }
}

// Helper stub functions
fn read_uart_byte() -> Option<u8> { None }
fn write_uart_byte(_b: u8) {}
fn get_cycles() -> u64 { 0 }
fn get_nanoseconds() -> u64 { 0 }
fn get_sp() -> usize { 0 }

// 3. Declare a test suite
#[ets_suite]
pub mod pid_control_suite {
    pub static TARGET_HEADING: u32 = 180;

    fn test_step_response() {
        // Test logic using target settings
        assert!(TARGET_HEADING.get() == 180);
    }
}

// 4. Initialize ETS
#[ets_setup]
fn setup() -> Context<UartComms, SystemCPUUtils> {
    Context::new(
        UartComms { reader: FrameReader::new() },
        SystemCPUUtils,
    )
}
```

## QA / Testing

To ensure that the target-side abstractions, serialization and compilation
build correctly, execute the following commands:

### Target Compilation Check

Verify that the crate compiles successfully for the thumbv7em target
architecture (configured in `Cargo.toml` / QEMU profile):

```bash
cargo check --package control-rs-ets --target thumbv7em-none-eabihf
```

### Workspace Unit Tests

Ensure ETS serialization/deserialization logic behaves correctly under host-side
unit testing:

```bash
cargo test --package control-rs-ets
```