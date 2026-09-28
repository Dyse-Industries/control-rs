//! `#[req]` on real items: each marked item behaves as if unmarked.

use control_rs_trace_macros::req;

/// A marked type keeps its fields and derives.
#[req("requirement-traceability#VC-15.1")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Marked {
    /// A value that survives the attribute.
    value: u8,
}

/// A marked function keeps its body.
#[req("requirement-traceability#VC-15.1")]
const fn marked_function() -> u8 {
    7
}

#[req("requirement-traceability#VC-12.1")]
#[test]
fn marked_items_are_unchanged() {
    assert_eq!(marked_function(), 7);
    let marked = Marked { value: 3 };
    assert_eq!(marked, Marked { value: 3 });
    assert_eq!(marked.value, 3);
}
