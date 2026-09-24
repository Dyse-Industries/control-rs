# `control-rs-ci`

Quality gate runner and report aggregator for the `control-rs` workspace.
Gates, groups and commands are declared in [`.cargo/gate.toml`](../.cargo/gate.toml).
[Design](../documentation/ci/ci-design.md) · [Workspace](../README.md)

## Usage

| Alias | Binary | Runs |
|:--|:--|:--|
| `cargo ci` | `control-rs-ci` | All gates, groups in parallel, exclusive gates last |
| `cargo gate <gates>` | `gate` | Selected gates, for example `cargo gate fmt,clippy` |
| `cargo report` | `report` | Aggregates gate artifacts into `ci-report.md` |
| `cargo valgrind` | `valgrind` | Valgrind Memcheck over the workspace examples (Linux) |
| `cargo regression` | `regression` | Criterion results against budgets and baselines |

```sh
cargo ci --list            # registered gates
cargo ci --group lint      # one group
cargo ci --only fmt,clippy # selected gates
cargo ci --skip mutants    # all but one
cargo ci -v --group verify # stream output as "[verify] cross-compare | ..."
cargo ci --clean           # remove previous artifacts and per-group target dirs
```

Artifacts are written to `target/ci-artifacts` (`[runner].out_dir`).
Required tools per gate: [Development Guide](../documentation/development-guide.md#prerequisites).

## License

MIT OR Apache-2.0.
