//! Structural guard: every gated path must exist in its suite's payload.
//!
//! The comparator projects `results/<suite>.rust.h5` onto `GATED_PATHS`. A path
//! that is not in the payload is silently dropped, so the container ships with
//! fewer datasets than the oracle writes and the suite compares less than it
//! claims to. This test fails instead.

// Harness code: the tests abort loudly on a malformed payload and index fixed
// paths that the gated schema guarantees.
#![allow(
    clippy::indexing_slicing,
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::arithmetic_side_effects
)]

use serde_json::Value;

/// Resolve a `group/dataset` path against a payload.
fn resolve<'a>(payload: &'a Value, path: &str) -> Option<&'a Value> {
    path.split('/')
        .try_fold(payload, |node, segment| node.get(segment))
}

/// Assert every gated path of one suite resolves to a value.
fn assert_gated(suite: &str, payload: &Value, gated: &[&str]) {
    assert!(!gated.is_empty(), "{suite}: no gated paths declared");
    for path in gated {
        let node = resolve(payload, path);
        assert!(
            node.is_some(),
            "{suite}: gated path `{path}` is not in the payload"
        );
    }
}

#[test]
/// # Verification
/// Method: Structural test over each suite's declared gate set.
fn gated_paths_exist_in_payloads() {
    use control_rs_validation as v;

    assert_gated("matrix", &v::matrix::payload(), v::matrix::GATED_PATHS);
    assert_gated(
        "polynomial",
        &v::polynomial::payload(),
        v::polynomial::GATED_PATHS,
    );
    assert_gated(
        "state_space",
        &v::state_space::payload(),
        v::state_space::GATED_PATHS,
    );
    assert_gated(
        "transfer_function",
        &v::transfer_function::payload(),
        v::transfer_function::GATED_PATHS,
    );
    assert_gated("tensor", &v::tensor::payload(), v::tensor::GATED_PATHS);
}
