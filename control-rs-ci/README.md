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
| `cargo trace-check` | `trace-check` | Derives condition status from `reqs.jsonl` and the result logs (gate `trace`); writes `trace-report.json` |
| `cargo trace` | `gate` | `trace-reqs`, then `trace`, in one run |

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

## Requirement tracing

`cargo trace` runs `trace-reqs` then `trace` as gates, so the check always
reads a fresh `reqs.jsonl`. Configuration is
[`.cargo/trace/trace.toml`](../.cargo/trace/trace.toml);
the rules are in the
[design](../documentation/vv/requirement-traceability-design.md).

- **Requirements.** Each `FR-n`, `NFR-n` or `C-n` in a document listed in
  `files` needs a `VC-x.y` row naming its method and, for an automated
  method, the fully qualified test or proof harness in the `Target` cell.
  `trace` fails a condition whose target is absent from, or failed in, its
  result log, so run the tests (`cargo gate test,kani`) before tracing.
- **Decisions.** With a `[decisions]` table, every `ADR-NNNN` cited in a
  traced document must have a record under `documentation/adr/`, and an
  `Approved` document may cite only `Accepted` decisions.
- **Adopting a document.** Add condition rows to its Verification table,
  then add its path to `files`.

Each defect prints as `path:line: message`:

```text
documentation/math/fixed-num-design.md:739: decision ADR-0009 is cited but has no decision record
documentation/math/fixed-num-design.md:739: Approved document cites decision ADR-0001 with status Proposed; it may cite only accepted decisions
```

## License

MIT OR Apache-2.0.
