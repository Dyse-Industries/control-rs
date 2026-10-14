# `uno_r4`

Embedded Test Server (ETS) firmware for the Arduino Uno R4 Minima (Renesas RA4M1).
Exposes on-target discrete state-space simulation, ARM Cortex-M SysTick execution profiling,
and runtime tunable controller gains via the host TUI or CI runner.
[Design](../../documentation/ets/board-examples-design.md) · [Examples](../README.md) · [Workspace](../../README.md)

---

## Connection

Connect the Arduino Uno R4 Minima to the host with a USB-C cable. The onboard Renesas RA4M1
communicates over UART / USB CDC serial at 115,200 baud.

---

## Target Configuration

| Item | Value |
|:--|:--|
| Architecture | ARM Cortex-M4 (`thumbv7em-none-eabi`) |
| Core Clock | 48 MHz |
| Transport | Hardware UART / USB CDC serial via `arduino_uno_r4_hal::serial::Serial` (115,200 baud) |
| Profiler | `CortexMProfiler` with 1 ms `SysTick` interrupt counter |
| Memory Mapping | Linker script `memory.x` provided by `arduino-uno-r4-hal` (256 KB Flash, 32 KB SRAM) |

---

## Firmware

One image carries two suites, discovered like any other suite:

- `uno_r4_control_suite`: on-target checks (gain constraints, state transition) and the settings `KP` and `KI`, tunable from the host.
- `gain_task_suite`: the lifecycle task shared with the QEMU example (`examples/qemu/src/gain_task.rs`), started from the dashboard in `FreeRunning` mode or driven from a script with `Session::run_task`.

The link is a `StreamLink` over the board transport (`ets-link-design.md`); a panic is reported with `TargetPanic` before the board resets.

---

## Building and Flashing

### 1. Prerequisites

Install the `thumbv7em-none-eabi` target toolchain:

```bash
rustup target add thumbv7em-none-eabi
```

Install `bossac` (v1.9.1+) or use the Arduino CLI loader to flash over the native USB DFU/bootloader:

```bash
# On macOS (Homebrew):
brew install bossac

# On Linux (apt):
sudo apt install bossac
```

### 2. Build Release Firmware

From `examples/uno_r4`:

```bash
cd examples/uno_r4
cargo build --release
```

### 3. Convert and Flash to Board

Double-tap the reset button on the Arduino Uno R4 Minima to enter bootloader mode (pulsing LED), then run:

```bash
rust-objcopy -O binary target/thumbv7em-none-eabi/release/uno-r4 uno-r4.bin
bossac -d --port=/dev/ttyACM0 -U -e -w uno-r4.bin -R
```

---

## Running the Host TUI

Connect the `control-rs-tui` console over the serial port:

```bash
# On macOS:
cargo tui -- --serial --port /dev/cu.usbmodem1101 --baud 115200

# On Linux:
cargo tui -- --serial --port /dev/ttyACM0 --baud 115200
```
