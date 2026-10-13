# `control-rs-ets`

Target-side `no_std` Embedded Test Server (ETS): message link and profiler
traits, command and telemetry schema, and the server event loop.
[Design](../documentation/ets/embedded-test-server-design.md) · [Overview](../documentation/ets/ets-overview.md) · [Workspace](../README.md)

| Module | Contents | Design |
|:--|:--|:--|
| `comms` | `Command` and `Telemetry` schema, frame reader and encoder (CRC-16) | [ets-link-design](../documentation/ets/ets-link-design.md) |
| `link` | `Link`, `ByteStream`, `StreamLink` | [ets-link-design](../documentation/ets/ets-link-design.md) |
| `profiler` | `CPUProfiler` cycle and stack measurement | [cpu-profiler-design](../documentation/ets/cpu-profiler-design.md) |
| `settings` | Host-adjustable atomic test settings | [test-suite-design](../documentation/ets/test-suite-design.md) |
| `context`, `run`, `server` | `Context`, `Runner` and `Mode`, the dispatcher | [ets-executive-design](../documentation/ets/ets-executive-design.md) |

## End-User Example

Implement `ByteStream` for the target transport (or `Link` for a packet
transport) and `CPUProfiler` for its timer, then pass both to `Context`:

```rust
#![no_std]
#![no_main]

use control_rs_ets::link::{ByteStream, CountOf, StreamLink};
use control_rs_ets::server::Context;
use control_rs_ets::CPUProfiler;
use control_rs_macros::{ets_setup, ets_suite};

// 1. Define the target-side byte stream (e.g., UART or Semihosting). Reads and
// writes never block: 0 bytes read means nothing is ready and a short write is
// back-pressure. `StreamLink` adds the sync, length and CRC-16 frame.
struct Uart;

impl ByteStream for Uart {
    type Error = ();

    fn read(&mut self, buf: &mut [u8]) -> CountOf<Self> {
        match (buf.first_mut(), read_uart_byte()) {
            (Some(slot), Some(byte)) => {
                *slot = byte;
                Ok(1)
            }
            _ => Ok(0),
        }
    }

    fn write(&mut self, buf: &[u8]) -> CountOf<Self> {
        for &byte in buf {
            write_uart_byte(byte);
        }
        Ok(buf.len())
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
fn setup() -> Context<StreamLink<Uart>, SystemCPUUtils> {
    Context::new(StreamLink::new(Uart), SystemCPUUtils)
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