# `control-rs-ets-host`

Host-side counterpart to the [`control-rs-ets`](../control-rs-ets) Embedded
Test Server. It owns the connection to a target, the wire framing and the
command/telemetry session that drives test discovery and execution. It renders
nothing and runs no quality gates.

| Module    | Responsibility |
|:----------|:---------------|
| `target`  | Target descriptors, QEMU architecture entries, ELF path resolution and build |
| `bridge`  | `ETSBridge` construction, reader threads, `BridgeMessage` stream |
| `session` | Discovery and run-queue state machine, panic detection, reset sequence |
| `runner`  | `run_headless_ets(target, timeout) -> Result<RunRecord, HostError>` |

Two transports: a spawned subprocess (QEMU under `cargo run`) and a USB CDC
serial port. Both resolve to the same reader/writer pair, so `session` is
transport-agnostic.

The interactive consumer is [`control-rs-tui`](../control-rs-tui). Headless
execution is `run_headless_ets`; automated quality gates run through
[`control-rs-ci`](../control-rs-ci).

Design: `documentation/ets-host/ets-host-design.md`.

## License

MIT OR Apache-2.0.
