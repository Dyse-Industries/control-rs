# `control-rs-ci`

Continuous integration runner and quality-gate orchestrator for `control-rs`.

It sequences the gates listed in `[gates]` of `gate.toml`, drives on-target Embedded Test Server (ETS) matrices via [`control-rs-ets-host`](../control-rs-ets-host), enforces bidirectional requirement traceability against design documents, and validates multi-example oracle suites from `validate.toml`.

---

## Config-driven pipeline

`[gates]` key order is execution order. Each value is `skip`, `warn`, or `fail`. Omitted gates are skipped. `--only` runs named gates as `fail` when they were omitted or listed as `skip`. `--up-to` must name a listed gate. `--strict` promotes `warn` to `fail`.

Typical root workspace gates: `clean`, `fmt`, `clippy`, `check`, `deny`, `audit`, `build`, `test`, `coverage`, `trace`. `validate` is omitted at the repo root (Validation CI owns host numerics).

Stage outcomes are aggregated into a dual-tier report:
- Executive summary: `ci-report.md`
- Standalone detailed audit reports: `trace-report.md`, `trace-report.json`, `validate-report.json`, `ets-results.json`

---

## Declarative Configuration (`gate.toml`)

### Workspace Scoping Rule

Every `gate.toml` configuration file strictly applies to the workspace that contains it:

1. **Workspace Isolation**: All items (`[[ets.targets]]`) refer directly to items within that workspace. Targets do **not** specify `manifest_path`.
2. **Root Workspace**: The root repository `gate.toml` manages quality gates for the root workspace members (`.`, `control-rs-ets`, `control-rs-macros`, `control-rs-ets-host`, `control-rs-tui`, `control-rs-ci`). It does not reach into or execute nested crates.
3. **Nested Crates**: Standalone/nested crates (such as `examples/qemu`) are independent workspaces with their own local `gate.toml`, executed via `--manifest-path <path>` or by running within their directory. Host numerics live in the `control-rs-validation` workspace member and are declared in the repository-root `validate.toml`.

#### Root Workspace Configuration (`./gate.toml`)

```toml
[runner]
title = "control-rs"
out_dir = "target/ci"
timeout_secs = 90

[gates]
clean = "fail"
fmt = "fail"
clippy = "fail"
check = "fail"
deny = "fail"
audit = "fail"
build = "fail"
test = "fail"
coverage = "fail"
trace = "fail"
```

#### Nested Workspace Configuration (`examples/qemu/gate.toml`)

Targets refer directly to local workspace binaries without manifest paths:

```toml
[runner]
title = "control-rs-qemu"
out_dir = "target/ci"
timeout_secs = 60

[gates]
clean = "fail"
fmt = "fail"
clippy = "fail"
check = "fail"
build = "fail"
ets = "fail"

[[ets.targets]]
name = "qemu-arm-hf"
target = "thumbv7em-none-eabihf"
bin = "control-rs-qemu-thumbv7em-none-eabihf"
args = ["--release"]
```

---

## Cargo Aliases

Configured in [`.cargo/config.toml`](../.cargo/config.toml):

| Alias | Invocation | Description |
|:---|:---|:---|
| `cargo ci` | `run --package control-rs-ci --bin ci --` | Executes the root `gate.toml` quality-gate pipeline |
| `cargo qemu-ci` | `run --package control-rs-ci --bin ci -- --manifest-path examples/qemu/Cargo.toml --only ets` | Runs QEMU ETS matrix against the `examples/qemu` workspace |
| `cargo teensy-ci` | `run --package control-rs-ci --bin ci -- --only ets --serial --port /dev/ttyACM0` | Runs ETS against physical hardware over serial |
| `cargo validate` | `run --package control-rs-ci --bin validate --` | Standalone 1:1 HDF5 comparison from `validate.toml` |


---

## CLI Options & Gate Controls

```
Usage: ci [SUBCOMMAND] [OPTIONS]

Subcommands:
  ci                         Run workspace continuous integration pipeline (default)
  ets                        Run Embedded Test Server (ETS) target matrix
  validate                   Run host example validation suites

Options:
      --config <PATH>        Path to configuration file [default: gate.toml]
      --manifest-path <PATH> Path to target Cargo.toml or crate directory
      --out-dir <DIR>        Directory for report output
      --title <TITLE>        Report title
      --timeout <SECS>       Per-target wall-clock bound
      --fmt                  Apply formatting instead of checking

Gate Controls:
      --up-to <GATE>         Run listed gates up to the specified name
      --only <GATE>          Run only specified gate(s); omitted names run as fail
      --skip <GATE>          Skip specified gate(s), repeatable
      --strict               Promote warn to fail
      --no-fmt               Skip formatting gate
      --no-clippy            Skip clippy linter gate
      --no-check             Skip cargo check gate
      --no-build             Skip cargo build gate
      --no-clean             Skip cargo clean gate
      --no-test              Skip unit test gate
      --no-cov               Skip code coverage analysis gate
      --no-ets               Skip target execution matrix (ETS) gate
      --no-trace             Skip requirement traceability gate
      --no-validate          Skip host validation suites

Target & Item Filters:
  -t, --target <TRIPLE>      Filter ETS targets to matching cross-compilation triple
  -b, --bin <BIN>            Filter ETS targets to matching binary name
      --serial               Verify a physical target over serial
      --port <PORT>          Serial device path [default: /dev/ttyACM0]
  -h, --help                 Print help
```

### Usage Examples

```bash
# Full CI run
cargo ci

# Fast local iterations up to tests
cargo ci --up-to test

# Run only ETS matrix across all configured QEMU targets
cargo qemu-ci

# Point CI to a nested example crate
cargo run -p control-rs-ci --bin compare -- --name buck-converter
```

---

## Design References

- Continuous Integration Architecture: `documentation/ci/ci-design.md`
- Requirement Traceability Subsystem: `documentation/vv/requirement-traceability-design.md`
- Oracle Validation Harness: `documentation/vv/oracle-harness-design.md`

---

## License

Licensed under MIT OR Apache-2.0.
