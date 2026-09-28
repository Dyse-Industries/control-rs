# `control-rs-ci`

Quality gate runner and report aggregator for the `control-rs` workspace.
Gates, groups and commands are declared in [`.cargo/gate.toml`](../.cargo/gate.toml).
[Design](../documentation/ci/ci-design.md) · [Workspace](../README.md)

## Usage

| Alias | Binary | Runs |
|:--|:--|:--|
| `cargo ci` | `control-rs-ci` | Default gates: `pre` gates, then groups in parallel, then `post` gates |
| `cargo gate <gates>` | `gate` | Selected gates, for example `cargo gate fmt,clippy` |
| `cargo report` | `report` | Aggregates gate artifacts into `ci-report.md` |
| `cargo valgrind --example <name>...` | `valgrind` | Valgrind Memcheck over the named examples; fails without Valgrind |
| `cargo regression` | `regression` | Criterion results against budgets and baselines |
| `cargo ets <targets>` | `ets` | Builds ETS firmware and runs its suites headless, writing `ets-results.json` |
| `cargo trace-reqs` | `trace-reqs` | Checks requirement definitions and verification conditions in configured design documents; writes `reqs.jsonl` |
| `cargo trace-marks` | `trace-marks` | Finds requirement markers in source text and writes `marks.jsonl` |
| `cargo trace-check` | `trace-check` | Derives condition coverage from `reqs.jsonl` and `marks.jsonl` (gate `trace`); writes `trace-report.json` |

```sh
cargo ci --list            # registered gates
cargo ci --group lint      # one group
cargo ci --only fmt,clippy # selected gates
cargo ci --all             # include default = false gates (mutants-*, regression)
cargo ci --skip valgrind   # all but one (for example on macOS)
cargo ci -j 2              # at most two groups at once
cargo ci -v --group verify # stream output as "[verify] cross-compare | ..."
cargo ci --clean           # remove previous artifacts and per-group target dirs
```

Artifacts are written to `target/ci-artifacts` (`[runner].out_dir`).
Gate failures exit 1; usage and configuration errors exit 2.
Required tools per gate: [Development Guide](../documentation/development-guide.md#prerequisites).

## License

MIT OR Apache-2.0.
