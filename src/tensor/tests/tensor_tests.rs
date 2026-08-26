//! # Tensor Unit and Verification Tests
#![allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::similar_names,
    clippy::unwrap_used,
    clippy::items_after_statements,
    clippy::cast_precision_loss,
    clippy::float_cmp,
    clippy::approx_constant
)]

#[cfg_attr(not(test), control_rs_macros::ets_suite)]
pub mod tensor_test_suite {
    use crate::assert_almost_eq;
    use crate::tensor::{
        Activation, ArrayTensor, Quantized, Relu, TableActivation,
    };

    #[cfg_attr(test, test)]
    fn test_tensor_indexing_and_storage() {
        let t = ArrayTensor::<f32, 2, 3>::from_raw([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]);
        assert_eq!(t.get(&[0, 0]), Some(&1.0));
        assert_eq!(t.get(&[1, 0]), Some(&2.0));
        assert_eq!(t.get(&[0, 1]), Some(&3.0));
        assert_eq!(t.get(&[1, 1]), Some(&4.0));
        assert_eq!(t.get(&[0, 2]), Some(&5.0));
        assert_eq!(t.get(&[1, 2]), Some(&6.0));
    }

    #[cfg_attr(test, test)]
    fn test_tensor_grid_interpolation() {
        // 2D grid: f(x, y) = [[0, 2], [4, 6]]
        let grid = ArrayTensor::<f32, 2, 2>::from_raw([[0.0, 4.0], [2.0, 6.0]]);
        // Center point at (0.5, 0.5): (0 + 2 + 4 + 6) / 4 = 3.0
        let val = grid.interpolate(&[0.5, 0.5]);
        assert_almost_eq!(val, 3.0, 1e-6);

        // Exact corners
        assert_almost_eq!(grid.interpolate(&[0.0, 0.0]), 0.0, 1e-6);
        assert_almost_eq!(grid.interpolate(&[1.0, 0.0]), 4.0, 1e-6);
        assert_almost_eq!(grid.interpolate(&[0.0, 1.0]), 2.0, 1e-6);
        assert_almost_eq!(grid.interpolate(&[1.0, 1.0]), 6.0, 1e-6);
    }

    #[cfg_attr(test, test)]
    fn test_quantized_scalar_operations() {
        // Q7 format: 1.0 = 128 (overflows i8, so 0.5 = 64, 0.25 = 32)
        type Q7 = Quantized<i8, 7>;

        let a = Q7::quantize(0.5);
        let b = Q7::quantize(0.25);
        assert_eq!(a.raw(), 64);
        assert_eq!(b.raw(), 32);

        // Sum: 0.5 + 0.25 = 0.75
        let sum = a + b;
        assert_almost_eq!(sum.dequantize(), 0.75, 0.01);

        // Product: 0.5 * 0.25 = 0.125
        let prod = a * b;
        assert_almost_eq!(prod.dequantize(), 0.125, 0.01);
    }

    #[cfg_attr(test, test)]
    fn test_activations() {
        let relu = Relu;
        assert_eq!(relu.apply(3.5f32), 3.5f32);
        assert_eq!(relu.apply(-2.0f32), 0.0f32);

        let table = TableActivation {
            breakpoints: [-1.0f32, 0.0, 1.0],
            values: [-1.0f32, 0.0, 1.0],
        };
        assert_almost_eq!(table.apply(0.5f32), 0.5f32, 1e-6);
        assert_almost_eq!(table.apply(-0.5f32), -0.5f32, 1e-6);
    }
}
