# teensy4

A demonstration crate implementing the `control-rs` Embedded Test Server (ETS)
test server on physical **Teensy 4.0** hardware over a native USB connection.

## Purpose

The purpose of this example is to show how to integrate the target-side
`control-rs-ets` event loop, clocks and macros into a real bare-metal embedded
microcontroller environment. It exposes a live PID controller tuning suite where
settings (Proportional, Integral and Derivative gains) can be read and set
dynamically by the host TUI or CI runner over a native USB serial connection.

---

## Hardware Connection Setup

Unlike standard UART-based ETS configurations, this example utilizes the Teensy
4.0's native USB controller.

### Wiring Diagram

Connect the Teensy 4.0 directly to your host PC using a standard **Micro-USB** (
or USB-C) cable.

```text
  +------------------+                    +------------------+
  |     Host PC      | <================> |   Teensy 4.0     |
  | (CLI / TUI / CI) |    USB Cable       |  (ETS Test Bed)  |
  +------------------+                    +------------------+
```

No external USB-to-UART serial adapters or custom wiring are needed!

---

## How it Works

1. **Host-Target Communications**: ETS operates as a native **USB CDC
   ACM (virtual COM port) device**. The `TeensyComms` struct polls the USB stack
   dynamically inside `poll_command()`, handling the USB enumeration,
   configuration states and bidirectional packet transmission seamlessly.
2. **Device USB Profile**:
    - **Vendor ID (VID)**: `0x5824`
    - **Product ID (PID)**: `0x27dd`
    - **Manufacturer / Product**: `teensy4`
3. **System Clock**: We configure the ARM Cortex-M `SysTick` exception to tick
   every 1 ms. The `TeensyClock` struct uses this tick counter and the current
   SysTick register countdown to provide microsecond-accurate timekeeping (
   `now_us()`).
4. **Status Indicator**: The onboard LED on **Pin 13** turns on solid when the
   ETS is successfully initialized and ready to communicate with the
   host.

---

```
# UDEV Rules for Teensy boards, http://www.pjrc.com/teensy/
#
# The latest version of this file may be found at:
#   http://www.pjrc.com/teensy/00-teensy.rules

ATTRS{idVendor}=="16c0", ATTRS{idProduct}=="04*", ENV{ID_MM_DEVICE_IGNORE}="1", ENV{ID_MM_PORT_IGNORE}="1"
ATTRS{idVendor}=="16c0", ATTRS{idProduct}=="04[789a]*", ENV{MTP_NO_PROBE}="1"
KERNEL=="ttyACM*", ATTRS{idVendor}=="16c0", ATTRS{idProduct}=="04*", MODE:="0666", RUN:="/bin/stty -F /dev/%k raw -echo", SYMLINK+="teensy"
KERNEL=="hidraw*", ATTRS{idVendor}=="16c0", ATTRS{idProduct}=="04*", MODE:="0666"
SUBSYSTEMS=="usb", ATTRS{idVendor}=="16c0", ATTRS{idProduct}=="04*", MODE:="0666"
KERNEL=="hidraw*", ATTRS{idVendor}=="1fc9", ATTRS{idProduct}=="013*", MODE:="0666"
SUBSYSTEMS=="usb", ATTRS{idVendor}=="1fc9", ATTRS{idProduct}=="013*", MODE:="0666"
```

---

## Building and Running the Example

### Prerequisites

Ensure you have the Rust `thumbv7em-none-eabihf` target toolchain installed:

```bash
rustup target add thumbv7em-none-eabihf
```

For flashing the binary to the Teensy, install the `teensy_loader_cli` utility:

```bash
# On Ubuntu/Debian
sudo apt-get install teensy-loader-cli
```

Alternatively, you can use the official graphical Teensy Loader GUI.

### 1. Build the Binary

Navigate to the example directory and build the package:

```bash
cd examples/teensy4
cargo build --release
```

### 2. Convert ELF to Hex

Generate the `.hex` image file required by the Teensy bootloader:

```bash
rust-objcopy -O ihex target/thumbv7em-none-eabihf/release/teensy4 teensy4.hex
```

### 3. Flash to Teensy

Press the program button on the Teensy 4 board and run:

```bash
teensy_loader_cli -w -v --mcu=TEENSY40 teensy4.hex
```

The status LED on Pin 13 will light up, indicating that the USB ETS is
active and waiting for a connection from the host.

### Alternative: Run/Flash via Cargo Alias

Alternatively, you can compile, convert and flash the Teensy in one step from
the workspace root:

```bash
cargo run
```

Press the program button on the Teensy 4 board when prompted to initiate
flashing.

---

## Host-Side Testing & TUI Verification

Once the Teensy 4.0 is flashed and plugged into the host PC:

### 1. Identify the USB CDC Serial Port

Check the device path assigned by the host operating system:

- **Linux**: `/dev/ttyACM0` (or `/dev/ttyACM1`, etc.)
- **macOS**: `/dev/tty.usbmodem101` (or similar)
- **Windows**: `COM3` (or check Device Manager for the virtual COM port index)

### 2. Launch the TUI

Run the cargo alias from the workspace root (defaults to `/dev/ttyACM0` and
`115200` baud):

```bash
cargo teensy
```

If your Teensy is assigned to a different serial port path (e.g.
`/dev/ttyACM1`), pass it as an argument:

```bash
cargo teensy /dev/ttyACM1
```

The TUI header will automatically update to display:

- **TARGET**: `Teensy 4.0 (Cortex-M7)`
- **LINK**: `USB CDC (/dev/ttyACM0)` (or your specified port)

You can now interactively trigger test cases (by selecting them and pressing
`Enter`), change PID parameters and inspect the real-time console telemetry
logs streamed back from the Teensy.