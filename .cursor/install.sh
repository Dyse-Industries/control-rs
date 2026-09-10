#!/usr/bin/env bash
#
# Idempotent development-environment bootstrap for control-rs.
#
# Installs the system packages, Rust toolchain (with cross-compilation targets),
# host tooling and Python virtualenv required to build, test and run the full
# `cargo ci` pipeline (host unit tests, clippy, rustfmt, tarpaulin coverage and
# the QEMU-backed embedded test server for ARM Cortex-M and RISC-V), as well as
# the separate numerical-models validation workflow.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Installing system packages (qemu, libudev, python venv, build tools)"
export DEBIAN_FRONTEND=noninteractive
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential \
  pkg-config \
  libudev-dev \
  qemu-system \
  python3.12-venv \
  python3-dev \
  curl \
  ca-certificates

echo "==> Installing Rust stable toolchain with rustfmt + clippy"
# The workspace uses edition 2024 and CI pins a minimum of 1.88.0, so the
# default image's older toolchain is insufficient. Install and default to stable.
rustup toolchain install stable --profile minimal -c rustfmt -c clippy --no-self-update
rustup default stable

echo "==> Adding embedded cross-compilation targets"
rustup target add \
  thumbv7em-none-eabihf \
  thumbv7em-none-eabi \
  riscv32imac-unknown-none-elf \
  riscv64gc-unknown-none-elf

echo "==> Installing cargo-binstall + cargo-tarpaulin (coverage)"
if ! command -v cargo-binstall >/dev/null 2>&1; then
  curl -L --proto '=https' --tlsv1.2 -sSf \
    https://raw.githubusercontent.com/cargo-bins/cargo-binstall/main/install-from-binstall-release.sh | bash
fi
if ! command -v cargo-tarpaulin >/dev/null 2>&1; then
  cargo binstall cargo-tarpaulin -y
fi

echo "==> Setting up Python 3.12 virtualenv for numerical-models validation"
# The development guide places the venv at the repository root (.venv).
if [ ! -x "$REPO_ROOT/.venv/bin/python" ]; then
  python3.12 -m venv "$REPO_ROOT/.venv"
fi
# shellcheck disable=SC1091
source "$REPO_ROOT/.venv/bin/activate"
python -m pip install --upgrade pip
pip install -r examples/numerical-models-validation/python3/requirements.txt
deactivate

echo "==> Warming build cache (workspace debug build)"
cargo build --workspace

echo "==> Development environment ready."
