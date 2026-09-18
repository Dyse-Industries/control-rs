# `control-rs-tui`

Interactive terminal console for driving
[`control-rs-ets`](../control-rs-ets) targets over
[`control-rs-ets-host`](../control-rs-ets-host).

The dashboard shows target metadata, a hierarchical suite and case tree, per-case
cycle, duration and peak-stack telemetry, and a live target log. Execution is
driven by single-key shortcuts (`f` filter, `r` run all, `s` stop, `q` quit).

```sh
cargo tui -- --manifest-path examples/qemu/Cargo.toml --target thumbv7em-none-eabihf --release
cargo tui -- --serial --port /dev/ttyACM0
```

Framing, CRC verification, telemetry decoding, panic detection and the
reset/reconnect sequence belong to `control-rs-ets-host`. This binary holds
layout and input handling only, which is what keeps the terminal stack out of
every headless dependency closure.

Design: `documentation/tui/tui-design.md`.

## License

MIT OR Apache-2.0.
