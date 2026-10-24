# `esp32c6`

Embedded Test Server (ETS) firmware for the ESP32-C6 DevKitC-1 over native USB-C.
Exposes on-target control and matrix benchmarks, hardware cycle profiling via RISC-V
`mcycle` CSRs, and tunable controller settings via the host TUI or CI runner.
[Design](../../documentation/ets/board-examples-design.md) · [Examples](../README.md) · [Workspace](../../README.md)

---

## Connection

Connect the ESP32-C6 DevKitC-1 to the host using a USB-C cable attached to the **USB** port (native USB Serial/JTAG controller), not the UART port. No external JTAG or USB-UART converter is required.

---

## Target Configuration

| Item | Value |
|:--|:--|
| Architecture | 32-bit RISC-V (`riscv32imac-unknown-none-elf`) |
| Core Clock | 160 MHz (HP core) |
| Transport | USB-C native USB Serial/JTAG (CDC-ACM virtual COM port) via `UsbSerialJtag` |
| Profiler | RISC-V `mcycle` CSR 64-bit cycle counter (`RiscvProfiler`) |
| Stack Profiling | High-water mark scanning between active `sp` and linker boundary `_stack_end` |
| Memory Mapping | Read-only descriptors linked into `RODATA` flash segment (`.ets_test_suites`) |

---

## Firmware

One image carries two suites, discovered like any other suite:

- `esp32c6_control_suite`: on-target checks (gain, matrix multiplication) and the settings `KP` and `KD`, tunable from the host.
- `gain_task_suite`: the lifecycle task shared with the QEMU example (`examples/qemu/src/gain_task.rs`), started from the dashboard in `FreeRunning` mode or driven from a script with `Session::run_task`.

The link is a `StreamLink` over the board transport (`ets-link-design.md`); a panic is reported with `TargetPanic` before the board resets.

---

## Building and Flashing

### 1. Prerequisites

Install the RISC-V target toolchain:

```bash
rustup target add riscv32imac-unknown-none-elf
```

Install `espflash` for flashing over the native USB-C interface:

```bash
cargo install espflash
```

### 2. Build Release Firmware

From the crate root or example directory:

```bash
cd examples/esp32c6
cargo build --release
```

### 3. Flash to Target

Flash the board using `cargo-espflash` or `espflash`:

```bash
cd examples/esp32c6

# Using cargo-espflash:
cargo espflash flash --release --monitor

# Or if the standalone espflash binary is installed:
cargo run --release
```

`espflash` automatically detects the connected serial port (for example `/dev/cu.usbmodem*` on macOS or `/dev/ttyACM*` on Linux), flashes the bootloader, partition table, and application binary with the embedded ESP-IDF App Descriptor, and attaches the serial monitor.

---

## Running the Host TUI

Connect the `control-rs-tui` console to the flashed ESP32-C6 target:

```bash
# On macOS:
cargo tui -- --serial --port /dev/cu.usbmodem1101

# On Linux:
cargo tui -- --serial --port /dev/ttyACM0
```

The TUI displays target discovery (`TargetInfo` reporting architecture, clock frequency 160 MHz, and capabilities), registers test cases in `esp32c6_control_suite`, measures execution cycles with sub-microsecond precision via `mcycle`, and provides real-time gain tuning for `KP` and `KD`.
