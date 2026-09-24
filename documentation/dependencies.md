# Dependency registry

Every external tool the `control-rs` workspace uses, in one place. Each row
names the tool, the version CI uses, what needs it and how to install it.
Crate dependencies resolve through Cargo and are not listed.
[Workspace](../README.md) · [Development Guide](development-guide.md)

---

## Install everything

Run from the workspace root. Omit the lines for areas you do not work on;
the [registry](#registry) says which area needs each tool.

### Linux (`apt-get`)

```bash
# Rust toolchain, components and targets
rustup toolchain install stable --component rustfmt,clippy,llvm-tools
rustup toolchain install 1.89.0
rustup target add thumbv7em-none-eabihf thumbv7em-none-eabi \
    riscv32imac-unknown-none-elf riscv64gc-unknown-none-elf

# Cargo subcommands
cargo install cargo-binstall
cargo binstall -y cargo-deny cargo-geiger cargo-semver-checks \
    cargo-tarpaulin cargo-mutants cargo-binutils

# System packages
sudo apt-get update
sudo apt-get install -y libudev-dev qemu-system valgrind clang teensy-loader-cli

# Vale, pinned to the CI version
curl -fsSL https://github.com/vale-cli/vale/releases/download/v3.22.0/vale_3.22.0_Linux_64-bit.tar.gz \
    | tar -xz -C ~/.cargo/bin vale
vale --config=.vale.ini sync

# Python oracles
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r control-rs-verification/python3/requirements.txt
```

### macOS (`brew`)

```bash
# Rust toolchain, components and targets
rustup toolchain install stable --component rustfmt,clippy,llvm-tools
rustup toolchain install 1.89.0
rustup target add thumbv7em-none-eabihf thumbv7em-none-eabi \
    riscv32imac-unknown-none-elf riscv64gc-unknown-none-elf

# Cargo subcommands
cargo install cargo-binstall
cargo binstall -y cargo-deny cargo-geiger cargo-semver-checks \
    cargo-tarpaulin cargo-mutants cargo-binutils

# System packages
brew install qemu llvm teensy_loader_cli python@3.12

# Vale, pinned to the CI version (use macOS_64-bit on Intel)
curl -fsSL https://github.com/vale-cli/vale/releases/download/v3.22.0/vale_3.22.0_macOS_arm64.tar.gz \
    | tar -xz -C ~/.cargo/bin vale
vale --config=.vale.ini sync

# Python oracles
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r control-rs-verification/python3/requirements.txt
```

`valgrind` and `libudev-dev` have no macOS install. The `valgrind` gate runs
in CI on Linux, and the serial transport uses the native macOS API.

### Check the installation

```bash
cargo gate --list   # every gate declared in gate.toml
cargo ci -v         # runs every gate; a gate whose tool is missing fails
```

---

## Registry

### Rust toolchain

| Dependency | Version | Needed for | Install |
|:--|:--|:--|:--|
| Rust | `1.89.0` minimum; CI also tests `stable` and `beta` | Everything | `rustup toolchain install stable` |
| `rustfmt`, `clippy` | toolchain | `fmt`, `clippy` gates | `rustup component add rustfmt clippy` |
| `llvm-tools` | toolchain | `rust-objcopy` for the Teensy runner, `cargo size` and `cargo objdump` in `control-rs-macros` | `rustup component add llvm-tools` |
| `thumbv7em-none-eabihf`, `thumbv7em-none-eabi` | toolchain | `examples/qemu`, `examples/teensy4`, `examples/subprograms/thumbv7em`, ETS | `rustup target add ...` |
| `riscv32imac-unknown-none-elf`, `riscv64gc-unknown-none-elf` | toolchain | `examples/qemu`, `examples/subprograms/riscv32imac`, ETS | `rustup target add ...` |

### Cargo subcommands

| Dependency | Version | Needed for | Install |
|:--|:--|:--|:--|
| `cargo-binstall` | latest | Installing the rows below as prebuilt binaries | `cargo install cargo-binstall` |
| `cargo-deny` | latest | `deny` gate | `cargo binstall -y cargo-deny` |
| `cargo-geiger` | latest | `geiger` gate | `cargo binstall -y cargo-geiger` |
| `cargo-semver-checks` | latest | `semver` gate | `cargo binstall -y cargo-semver-checks` |
| `cargo-tarpaulin` | latest | `coverage` gate, `cargo coverage` | `cargo binstall -y cargo-tarpaulin` |
| `cargo-mutants` | latest | `mutants` and `mutants-*` gates | `cargo binstall -y cargo-mutants` |
| `cargo-binutils` | latest | `rust-objcopy`, `cargo size`, `cargo objdump` | `cargo binstall -y cargo-binutils` |

### System packages

| Dependency | Version | Needed for | Linux | macOS |
|:--|:--|:--|:--|:--|
| `vale` | `3.22.0` | `vale` gate | release tarball, then `vale --config=.vale.ini sync` | release tarball, then `vale --config=.vale.ini sync` |
| QEMU (`qemu-system-arm`, `qemu-system-riscv32`, `qemu-system-riscv64`) | distribution | `cargo qemu`, virtual ETS, `examples/subprograms` runners | `apt-get install qemu-system` | `brew install qemu` |
| `valgrind` | distribution | `valgrind` gate, `cargo valgrind` | `apt-get install valgrind` | Linux only |
| `libudev-dev` | distribution | Serial transport in `control-rs-ets-host` | `apt-get install libudev-dev` | not needed |
| `clang` | distribution | C sources in `examples/subprograms/thumbv7em` | `apt-get install clang` | `brew install llvm` |
| `teensy_loader_cli` | distribution | Flashing `examples/teensy4` | `apt-get install teensy-loader-cli` | `brew install teensy_loader_cli` |
| Netlib CBLAS and BLAS (`libcblas`, `libblas`) | distribution | `examples/subprograms/x86_64` with `--features cblas` only | any package providing both libraries on the linker path | not needed; `aarch64` uses `--features accelerate` and the system framework |

On Linux, flashing a Teensy also needs the udev rules described in
[`examples/teensy4`](../examples/teensy4/README.md). A RISC-V GCC is optional
for `examples/subprograms/riscv32imac`; without it the crate uses portable
Rust stand-ins.

### Python

| Dependency | Version | Needed for | Install |
|:--|:--|:--|:--|
| Python | `3.12`, virtual environment at `.venv` | `cross-compare` gate, `cargo compare` | `python3.12 -m venv .venv` |
| Oracle packages (`h5py`, `numpy`, `scipy`, `jax`, `python-flint`, `tensorflow`, `onnx`, `onnxruntime`) | minimums in [`requirements.txt`](../control-rs-verification/python3/requirements.txt) | Reference oracles in `control-rs-verification` | `pip install -r control-rs-verification/python3/requirements.txt` |

---

## Maintaining the registry

- A change that adds a gate to `gate.toml`, an install step to
  `.github/workflows/CI.yml` or a runner to an example's `.cargo/config.toml`
  adds or updates a row here in the same pull request.
- Pinned versions match `.github/workflows/CI.yml`. `vale` is the only tool
  CI pins; the rest install the latest release.
- Project READMEs keep setup steps specific to their crate or board and link
  here for tools.
