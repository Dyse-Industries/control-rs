# `control-rs-ets`

Target-side `no_std` Embedded Test Server (ETS): transport and profiler
traits, the message link, the command and telemetry schema, and the server event loop.
[Design](../documentation/ets/embedded-test-server-design.md) · [Overview](../documentation/ets/ets-overview.md) · [Workspace](../README.md)

| Module | Contents | Design |
|:--|:--|:--|
| `wire` | `Command` and `Telemetry` schema, `PROTOCOL_VERSION` | [ets-link-design](../documentation/ets/ets-link-design.md) |
| `link` | `Link`, `ByteStream`, `StreamLink`, `usb` and `can` utilities | [ets-link-design](../documentation/ets/ets-link-design.md) |
| `profiler` | `CPUProfiler` cycle and stack measurement | [cpu-profiler-design](../documentation/ets/cpu-profiler-design.md) |
| `params` | Host-adjustable parameter table (`ParamEntry`) | [test-suite-design](../documentation/ets/test-suite-design.md) |
| `server` | `Context` and the event loop | [embedded-test-server-design](../documentation/ets/embedded-test-server-design.md) |

## End-User Example

Implement `Link` for the target transport (or `ByteStream` and wrap it in
`StreamLink`) and `CPUProfiler` for its timer, then pass both to `Context`:

```rust
#![no_std]
#![no_main]

use control_rs_ets::link::{ByteStream, CountOf, StreamLink};
use control_rs_ets::server::Context;
use control_rs_ets::CPUProfiler;
use control_rs_macros::{ets_setup, ets_suite};

// 1. Define the target-side byte stream (e.g., UART or semihosting)
struct Uart;

impl ByteStream for Uart {
    type Error = ();

    fn read(&mut self, buf: &mut [u8]) -> CountOf<Self> {
        // Read the bytes the peripheral has ready; never wait
        Ok(read_uart(buf))
    }

    fn write(&mut self, buf: &[u8]) -> CountOf<Self> {
        // Write what the peripheral accepts; a short count is back-pressure
        Ok(write_uart(buf))
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
fn read_uart(_buf: &mut [u8]) -> usize { 0 }
fn write_uart(buf: &[u8]) -> usize { buf.len() }
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