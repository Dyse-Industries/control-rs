# QEMU ETS firmware

Bare-metal Embedded Test Server (ETS) runners for virtual ARM Cortex-M7 and
RISC-V targets. This package is validation firmware. It is not a subprogram
backend and does not link CMSIS-DSP or NMSIS-DSP.

The crate is its own `[workspace]`. Run commands from this directory unless
using the root alias `cargo qemu`, which launches the host TUI against this
firmware.

## Prerequisites

```bash
rustup target add thumbv7em-none-eabihf thumbv7em-none-eabi \
    riscv32imac-unknown-none-elf riscv64gc-unknown-none-elf
```

QEMU system emulators on `PATH`: `qemu-system-arm`, `qemu-system-riscv32`,
`qemu-system-riscv64`.

## Run

Interactive TUI from the repository root:

```bash
cargo qemu
```

Headless, using the aliases in `.cargo/config.toml` (default target is
`thumbv7em-none-eabihf`):

```bash
cd examples/qemu
cargo arm-hf      # Cortex-M7 hard-float, machine mps2-an500
cargo arm-sf      # Cortex-M7 soft-float
cargo riscv32     # rv32imac, machine virt
cargo riscv64     # rv64gc, machine virt
```

Each alias is `cargo run --bin <name> --release` plus the matching
`--target`. The runner in `.cargo/config.toml` starts QEMU with semihosting
and passes the built kernel.

## Layout

| Binary | Source |
|:-------|:-------|
| `control-rs-qemu-thumbv7em-none-eabihf` | `src/thumbv7em-none-eabihf.rs` |
| `control-rs-qemu-thumbv7em-none-eabi` | `src/thumbv7em-none-eabi.rs` |
| `control-rs-qemu-riscv32imac-unknown-none-elf` | `src/riscv32imac-unknown-none-elf.rs` |
| `control-rs-qemu-riscv64gc-unknown-none-elf` | `src/riscv64gc-unknown-none-elf.rs` |

`build.rs` emits `memory.x` and `ets_suites.x` into `OUT_DIR`. Suite
discovery uses the `.ets_test_suites` section, not a hardcoded test list.
