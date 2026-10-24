# `control-rs-tui`

Interactive terminal console for driving
[`control-rs-ets`](../control-rs-ets) targets over
[`control-rs-ets-host`](../control-rs-ets-host).
[Design](../documentation/ets/tui-design.md) · [Workspace](../README.md)

The dashboard shows target metadata, a hierarchical suite tree with cases and
lifecycle tasks, per-case cycle, duration and peak-stack telemetry, a setting
console, fault and link status, and a live target log. A worker thread owns the
session, so drawing never blocks a run. Execution is driven by single-key
shortcuts (`f` filter, `r` run all, `p` settings, `m` task mode, `[` `]` release
period, `s` stop, `d` describe, `q` quit). `Enter` starts the selected case or
task, or edits the selected setting. A task starts in `FreeRunning`, `Periodic`
or, when the target reports a release source, `Triggered`; `Lockstep` needs a
host-side plant and runs from a script through `Session::run_task`.

```sh
cargo tui -- --manifest-path examples/qemu/Cargo.toml --target thumbv7em-none-eabihf --release
cargo tui -- --serial --port /dev/ttyACM0
```

Framing, CRC verification, telemetry decoding, panic detection and the
reset/reconnect sequence belong to `control-rs-ets-host`. This binary holds
layout and input handling only, which is what keeps the terminal stack out of
every headless dependency closure.

## License

MIT OR Apache-2.0.
