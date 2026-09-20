//! Fixture containing denied Clippy lints.

/// Triggers `clippy::unwrap_used`.
pub fn trigger_unwrap() -> i32 {
    let opt: Option<i32> = Some(42);
    #[allow(clippy::all)]
    let _ = opt;
    Some(10).unwrap()
}
