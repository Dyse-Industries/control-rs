# `xiao_nrf52840`

Embedded Test Server (ETS) firmware for the Seeed Studio XIAO nRF52840 (Nordic nRF52840).
Exposes on-target discrete state-space simulation, ARM Cortex-M4F SysTick execution profiling,
and runtime tunable filter cutoff / Nyquist verification via the host TUI or CI runner.
[Design](../../documentation/ets/board-examples-design.md) · [Examples](../README.md) · [Workspace](../../README.md)

---

## Connection

Connect the Seeed Studio XIAO nRF52840 to the host with a USB-C cable. The onboard nRF52840
communicates over native Full-Speed USB 2.0 (12 Mbps) CDC-ACM serial.

---

## Target Configuration

| Item | Value |
|:--|:--|
| Architecture | ARM Cortex-M4F with hardware FPU (`thumbv7em-none-eabihf`) |
| Core Clock | 64 MHz |
| Transport | Native Full-Speed USB 2.0 (12 Mbps) CDC-ACM via `nrf52840_hal::usbd::Usbd` |
| Profiler | `CortexMProfiler` with 1 ms `SysTick` interrupt counter and 64 MHz clock |
| Bootloader | Adafruit UF2 + SoftDevice S140 7.3.0 (`FLASH ORIGIN = 0x27000`) |
| Memory Mapping | Linker script `memory.x` (788 KB app flash from `0x27000`, SRAM from `0x20006000`) |

---

## Firmware

One image carries two suites, discovered like any other suite:

- `xiao_nrf52840_suite`: on-target checks (Nyquist criterion, Kalman prediction) and the settings `CUTOFF_FREQ_HZ` and `SAMPLE_RATE_HZ`, tunable from the host.
- `gain_task_suite`: the lifecycle task shared with the QEMU example (`examples/qemu/src/gain_task.rs`), started from the dashboard in `FreeRunning` mode or driven from a script with `Session::run_task`.

The link is a `StreamLink` over the board transport (`ets-link-design.md`); a panic is reported with `TargetPanic` before the board resets.

---

## Building and Flashing

### 1. Prerequisites

Install the `thumbv7em-none-eabihf` target toolchain:

```bash
rustup target add thumbv7em-none-eabihf
```

Install `cargo-hf2` or `uf2conv` for flashing via the UF2 bootloader:

```bash
cargo install hf2-cli
# or Python uf2conv
pip install uf2conv
```

### 2. Build Release Firmware

From `examples/xiao_nrf52840`:

```bash
cd examples/xiao_nrf52840
cargo build --release
```

### 3. Flash to Board

#### Method A: UF2 Drag-and-Drop
Double-tap the reset button on the XIAO nRF52840 to enter UF2 bootloader mode (mounts as a USB storage drive named `XIAO-SENSE` or `Arduino`). Convert the binary to UF2 and copy to the drive:

```bash
rust-objcopy -O binary target/thumbv7em-none-eabihf/release/xiao-nrf52840 xiao.bin
uf2conv xiao.bin -b 0x27000 -c -f 0xADA52840 -o xiao.uf2
cp xiao.uf2 /Volumes/Arduino/
```

#### Method B: HF2 CLI

```bash
hf2 elf target/thumbv7em-none-eabihf/release/xiao-nrf52840
```

---

## Running the Host TUI

Connect the `control-rs-tui` console over the serial port:

```bash
# On macOS:
cargo tui -- --serial --port /dev/cu.usbmodem* --baud 115200

# On Linux:
cargo tui -- --serial --port /dev/ttyACM0 --baud 115200
```
