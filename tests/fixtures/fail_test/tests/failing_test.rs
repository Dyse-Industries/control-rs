#[test]
fn deliberate_assertion_failure() {
    assert_eq!(1, 2, "Deliberate test assertion failure for CI gate validation");
}
