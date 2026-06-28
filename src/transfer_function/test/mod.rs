//! Transfer function HIL and unit test suite.
#![allow(
    clippy::arithmetic_side_effects,
    clippy::float_cmp,
    clippy::doc_markdown
)]

#[cfg_attr(not(test), control_rs_macros::hil_suite)]
/// Transfer function representation tests.
pub mod test_transfer_function {
    use crate::transfer_function::{StaticTransferFunction, TransferFunction};

    #[cfg_attr(test, test)]
    /// Verifies `StaticTransferFunction` creation and access.
    fn test_static_transfer_function_creation_and_access() {
        let num = [1.0, 2.0];
        let den = [3.0, 4.0, 5.0];
        let tf = StaticTransferFunction::new(num, den);

        assert_eq!(tf.numerator(), &[1.0, 2.0]);
        assert_eq!(tf.denominator(), &[3.0, 4.0, 5.0]);
    }
}
