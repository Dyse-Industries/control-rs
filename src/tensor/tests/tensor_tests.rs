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

    #[cfg_attr(test, test)]
    fn test_tensor_contract() {
        use crate::matrix::Owned;
        let a = ArrayTensor::<f64, 2, 3>::from_fn(|idx| {
            let r = idx[0];
            let c = idx[1];
            (r * 3 + c) as f64
        });
        let b = ArrayTensor::<f64, 3, 2>::from_fn(|idx| {
            (idx[0] + idx[1] * 2) as f64
        });
        let mut out = ArrayTensor::<f64, 2, 2>::zero();
        a.contract_into(&b, &mut out);

        let ma = Owned::<f64, 2, 3>::from_storage(*a.buffer());
        let mb = Owned::<f64, 3, 2>::from_storage(*b.buffer());
        let gemm = &ma * &mb;
        for i in 0..2 {
            for j in 0..2 {
                assert_almost_eq!(
                    out.get(&[i, j]).copied().unwrap(),
                    gemm.get(i, j).copied().unwrap(),
                    1e-12
                );
            }
        }

        let t = a.permute([1, 0]);
        assert_eq!(t.get(&[0, 1]), a.get(&[1, 0]));
        let back = t.permute([1, 0]);
        assert_eq!(back.get(&[1, 2]), a.get(&[1, 2]));

        let sum = &a + &a;
        assert_almost_eq!(sum.get(&[0, 0]).copied().unwrap(), 0.0, 1e-12);
        let scaled = &a * 2.0;
        assert_almost_eq!(scaled.get(&[1, 0]).copied().unwrap(), 6.0, 1e-12);
    }

    #[cfg_attr(test, test)]
    fn test_shape4d_and_view() {
        use crate::tensor::{ArrayTensor4D, Shape4D, TensorLayout};
        assert_eq!(Shape4D::<1, 2, 2, 2>::SIZE, 8);
        let t4 = ArrayTensor4D::<f64, 1, 1, 1, 1, 1>::from_storage([3.0]);
        assert_eq!(t4.get(&[0, 0, 0, 0]), Some(&3.0));
        let mut grid =
            ArrayTensor::<f64, 2, 2>::from_raw([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(grid.view().get(&[0, 0]), Some(&1.0));
        {
            let mut view = grid.view_mut();
            if let Some(v) = view.get_mut(&[0, 0]) {
                *v = 9.0;
            }
        }
        assert_eq!(grid.get(&[0, 0]), Some(&9.0));
        let m = grid.slice_matrix();
        assert_eq!(m.get(0, 0), Some(&9.0));
    }

    #[cfg_attr(test, test)]
    #[allow(clippy::too_many_lines)]
    fn test_tensor_shape_interpolate_and_table_edges() {
        use crate::tensor::{
            ArrayTensor3D, FlatBuffer, Shape1D, Shape3D, Shape4D, TensorLayout,
        };
        assert!(Shape1D::<2>::flat_offset(&[9]).is_none());
        assert!(Shape1D::<2>::flat_offset(&[]).is_none());
        let mut d1 = [0usize; 1];
        Shape1D::<4>::dims(&mut d1);
        assert_eq!(d1[0], 4);
        Shape1D::<4>::dims(&mut []);

        assert!(Shape3D::<2, 2, 2>::flat_offset(&[0, 0]).is_none());
        assert!(Shape3D::<2, 2, 2>::flat_offset(&[0, 0, 9]).is_none());
        assert_eq!(Shape3D::<2, 2, 2>::flat_offset(&[1, 1, 1]), Some(7));
        let mut d3 = [0usize; 3];
        Shape3D::<2, 3, 4>::dims(&mut d3);
        assert_eq!(d3, [2, 3, 4]);
        Shape3D::<2, 3, 4>::dims(&mut [0usize; 1]);

        assert!(Shape4D::<2, 2, 2, 2>::flat_offset(&[0, 0, 0]).is_none());
        assert!(Shape4D::<1, 1, 1, 1>::flat_offset(&[0, 0, 0, 1]).is_none());
        assert_eq!(Shape4D::<1, 1, 1, 1>::flat_offset(&[0, 0, 0, 0]), Some(0));
        let mut d4 = [0usize; 4];
        Shape4D::<1, 2, 3, 4>::dims(&mut d4);
        assert_eq!(d4, [1, 2, 3, 4]);
        Shape4D::<1, 2, 3, 4>::dims(&mut [0usize; 2]);

        let t3 = ArrayTensor3D::<f64, 2, 2, 2, 8>::from_storage([0.0; 8]);
        assert!(t3.slice_matrix(&[9]).is_none());
        assert!(t3.slice_matrix(&[]).is_none());
        assert!(t3.slice_matrix(&[0]).is_some());

        let grid = ArrayTensor::<f64, 2, 2>::from_raw([[1.0, 2.0], [3.0, 4.0]]);
        let lo = grid.interpolate(&[-1.0, -1.0]);
        assert_almost_eq!(lo, 1.0, 1e-12);
        let hi = grid.interpolate(&[9.0, 9.0]);
        assert_almost_eq!(hi, 4.0, 1e-12);
        assert!(!grid.buffer().is_empty());
        assert!(!grid.buffer().as_ptr().is_null());
        let _ = grid.into_buffer();

        let empty = TableActivation::<f64, 0> {
            breakpoints: [],
            values: [],
        };
        assert_eq!(empty.apply(1.0), 0.0);
        let dup = TableActivation::<f64, 2> {
            breakpoints: [0.0, 0.0],
            values: [1.0, 2.0],
        };
        assert_eq!(dup.apply(0.0), 1.0);
        let clamp_hi = TableActivation::<f64, 2> {
            breakpoints: [0.0, 1.0],
            values: [3.0, 5.0],
        };
        assert_eq!(clamp_hi.apply(-1.0), 3.0);
        assert_eq!(clamp_hi.apply(2.0), 5.0);

        // Test tensor subtraction and pointers
        let t_a = ArrayTensor::<f64, 2, 2>::from_raw([[2.0, 4.0], [6.0, 8.0]]);
        let t_b = ArrayTensor::<f64, 2, 2>::from_raw([[1.0, 2.0], [3.0, 4.0]]);
        let diff = &t_a - &t_b;
        assert_almost_eq!(diff.get(&[0, 0]).copied().unwrap(), 1.0, 1e-12);
        assert_almost_eq!(diff.get(&[1, 1]).copied().unwrap(), 4.0, 1e-12);

        use crate::math::storage::RowArrayStorage;
        use crate::tensor::FlatBufferMut;
        let mut row_buf =
            RowArrayStorage::<f64, 2, 2>::from_array([[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(FlatBuffer::len(&row_buf), 4);
        assert_eq!(FlatBuffer::as_slice(&row_buf).len(), 4);
        assert!(!FlatBuffer::as_ptr(&row_buf).is_null());
        assert!(!FlatBufferMut::as_mut_ptr(&mut row_buf).is_null());

        let arr = [1.0f64, 2.0, 3.0];
        assert_eq!(FlatBuffer::len(&arr), 3);
        assert_eq!(FlatBuffer::as_slice(&arr), &[1.0, 2.0, 3.0]);

        let mut grid_for_view =
            ArrayTensor::<f64, 2, 2>::from_raw([[1.0, 2.0], [3.0, 4.0]]);
        let v = grid_for_view.view();
        assert_eq!(FlatBuffer::len(v.buffer()), 4);
        assert_eq!(FlatBuffer::as_slice(v.buffer()).len(), 4);

        let vm = grid_for_view.view_mut();
        let mut fvm = vm.into_buffer();
        assert_eq!(FlatBuffer::len(&fvm), 4);
        assert_eq!(FlatBuffer::as_slice(&fvm).len(), 4);
        assert_eq!(FlatBufferMut::as_mut_slice(&mut fvm).len(), 4);
    }

    #[cfg_attr(test, test)]
    fn test_quantization_roundtrip_half_lsb() {
        type Q7 = Quantized<i8, 7>;
        let step = 2.0_f64.powi(-7);
        let half = step / 2.0;
        let samples = [
            0.0_f64,
            0.25,
            -0.5,
            1.0 / 3.0,
            core::f64::consts::FRAC_PI_4,
            -0.75,
            Q7::MAX.dequantize(),
            Q7::MIN.dequantize(),
        ];
        let mut max_err = 0.0_f64;
        for &x in &samples {
            let err = (x - Q7::quantize(x).dequantize()).abs();
            max_err = max_err.max(err);
            assert!(
                err <= half,
                "Q7 round-trip |{x} - dequant(quant)|={err} exceeds {half}"
            );
        }
        assert!(max_err <= half);
    }

    #[cfg_attr(test, test)]
    fn test_quantization_monotonicity() {
        type Q7 = Quantized<i8, 7>;
        let pairs = [
            (0.3_f64, 0.1),
            (0.5, -0.5),
            (0.8, 0.799),
            (-0.1, -0.9),
            (Q7::MAX.dequantize(), Q7::MIN.dequantize()),
        ];
        for (x, y) in pairs {
            assert!(x > y);
            let qx = Q7::quantize(x);
            let qy = Q7::quantize(y);
            assert!(
                qx >= qy,
                "quant({x})={:?} < quant({y})={:?}",
                qx.raw(),
                qy.raw()
            );
        }
    }
}

#[cfg(test)]
mod tensor_property_tests {
    use crate::tensor::Quantized;
    use proptest::prelude::*;

    type Q7 = Quantized<i8, 7>;

    proptest! {
        /// Round-trip error never exceeds half the Q7 step on the closed
        /// representable interval.
        #[test]
        fn prop_quantization_roundtrip_half_lsb(
            x in -1.0_f64..Q7::MAX.dequantize(),
        ) {
            let half = 2.0_f64.powi(-7) / 2.0;
            let err = (x - Q7::quantize(x).dequantize()).abs();
            prop_assert!(
                err <= half,
                "Q7 round-trip {err} exceeds {half} for x={x}"
            );
        }

        /// $x > y$ implies $\mathrm{quant}(x) \ge \mathrm{quant}(y)$.
        #[test]
        fn prop_quantization_monotonicity(
            x in -4.0_f64..4.0,
            y in -4.0_f64..4.0,
        ) {
            prop_assume!(x > y);
            prop_assert!(Q7::quantize(x) >= Q7::quantize(y));
        }
    }
}
