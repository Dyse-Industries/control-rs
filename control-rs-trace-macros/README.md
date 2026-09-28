# `control-rs-trace-macros`

`#[req]` marks an item with the requirements it verifies.

[Design](../documentation/vv/requirement-traceability-design.md) · [Workspace](../README.md)

```rust
use control_rs_trace_macros::req;

#[req("storage#FR-3")]
#[test]
fn packed_value_out_of_bounds_is_none() { /* ... */ }
```

The attribute checks that each argument is a qualified ID `<doc>#<id>` and
returns the item unchanged. It writes nothing: `cargo trace-marks` finds
markers by reading source text and writes them to `marks.jsonl`.

## License

MIT OR Apache-2.0.
