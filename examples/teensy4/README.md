# `teensy4`

Embedded Test Server firmware for Teensy 4.0 over native USB. Exposes a PID
gain-tuning suite whose settings the host TUI or CI runner reads and writes.
[Design](../../documentation/ets/embedded-test-server-design.md) · [Examples](../README.md) · [Workspace](../../README.md)

---

## Connection

Connect the board to the host with a USB cable. The ETS uses the board's native
USB controller; no UART adapter is required.

---

## Target Configuration

| Item                   | Value                                                                                            |
|:-----------------------|:-------------------------------------------------------------------------------------------------|
| Transport              | USB CDC ACM (virtual COM port), polled by `TeensyComms::poll_command()`                          |
| USB VID / PID          | `0x16c0` / `0x0413`                                                                              |
| Manufacturer / Product | `teensy4`                                                                                        |
| Clock                  | `SysTick` at 1 ms; `TeensyClock::now_us()` adds the SysTick countdown for microsecond resolution |
| Ready indicator        | LED on pin 13, solid once the ETS is initialized                                                 |

---

## Building and Running the Example

### Prerequisites

Ensure you have the Rust `thumbv7em-none-eabihf` target toolchain installed:

```bash
rustup target add thumbv7em-none-eabihf
```

For flashing the binary to the Teensy, install the `teensy_loader_cli` utility:

```bash
# On Ubuntu/Debian (apt)
sudo apt-get install teensy-loader-cli

# On macOS (Homebrew)
brew install teensy_loader_cli
```

Alternatively, you can use the official graphical Teensy Loader GUI.

On Linux, install the PJRC udev rules (`/etc/udev/rules.d/00-teensy.rules`) and
reload `udev`:

```bash
# Ensure directory exists and install the PJRC udev rules
sudo mkdir -p /etc/udev/rules.d
curl -fsSL https://www.pjrc.com/teensy/00-teensy.rules | sudo tee /etc/udev/rules.d/00-teensy.rules > /dev/null

# Reload and trigger udev rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Alternatively, save the rules to `/etc/udev/rules.d/00-teensy.rules` directly:

```udev
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

#### 3.1. Install Teensy Loader CLI

```bash
#!/usr/bin/env bash
set -e

echo "==> Preparing to build teensy_loader_cli from source..."

# Check for Xcode Command Line Tools
if ! xcode-select -p &>/dev/null; then
    echo "==> Installing Xcode Command Line Tools..."
    xcode-select --install
fi

# Clone repository if not already present in the current directory
if [ ! -d "teensy_loader_cli" ]; then
    echo "==> Cloning teensy_loader_cli repository..."
    git clone https://github.com/PaulStoffregen/teensy_loader_cli.git
fi

cd teensy_loader_cli

echo "==> Configuring Makefile for macOS IOKit (disabling USE_LIBUSB)..."
# Comment out any active USE_LIBUSB declarations to default to IOKit
if [[ "$OSTYPE" == "darwin"* ]]; then
    sed -i '' 's/^[[:space:]]*USE_LIBUSB/#USE_LIBUSB/' Makefile || true
else
    sed -i 's/^[[:space:]]*USE_LIBUSB/#USE_LIBUSB/' Makefile || true
fi

echo "==> Compiling binary..."
make clean
make

echo "==> Installing binary to /usr/local/bin..."
sudo cp teensy_loader_cli /usr/local/bin/
sudo chmod +x /usr/local/bin/teensy_loader_cli

echo "==> Installation successful!"
```

Press the program button on the Teensy 4 board when prompted to initiate
flashing.

---

## Host-Side Testing & TUI Verification

Once the Teensy 4.0 is flashed and plugged into the host PC:

### 1. Identify the USB CDC Serial Port

Check the device path assigned by the host operating system:

- **Linux**: `/dev/teensy` (symlink created by udev rules) or `/dev/ttyACM0` (or
  `/dev/ttyACM1`, etc.)
- **macOS**: `/dev/cu.usbmodem101` (or similar). Use the `cu.` node, not
  `tty.`: the `tty.` node waits for carrier detect, so writes to the board time
  out (`transport error: I/O failure sending command: timed out`).
- **Windows**: `COM3` (or check Device Manager for the virtual COM port index)

### 2. Launch the TUI

Run the cargo alias from the workspace root (defaults to `/dev/ttyACM0` and
`115200` baud):

```bash
cargo teensy
```

If your Teensy is assigned to a different serial port path (such as
`/dev/teensy` or `/dev/ttyACM1`), pass it as an argument:

```bash
cargo teensy /dev/teensy
```

The TUI header will automatically update to display:

- **TARGET**: `Teensy 4.0 (Cortex-M7)`
- **LINK**: `USB CDC (/dev/ttyACM0)` (or your specified port)

You can now interactively trigger test cases (by selecting them and pressing
`Enter`), change PID parameters and inspect the real-time console telemetry
logs streamed back from the Teensy.