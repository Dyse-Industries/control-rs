//! # Tensor Unit and Verification Tests
// Expected values here are hand-derived reference expressions; the `mul_add`
// form clippy suggests obscures the algebra being asserted and is not a
// performance concern in a test oracle.
#![allow(clippy::suboptimal_flops)]

#[cfg_attr(not(test), control_rs_macros::ets_suite)]
pub mod tensor_test_suite {
    use crate::assert_almost_eq;
    use crate::math::storage::RowArrayStorage;
    use crate::tensor::{
        Activation, ArrayTensor, ArrayTensor3D, ArrayTensor4D, Axes2D,
        FlatBufferMut, Quantized, Relu, TableActivation,
    };

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: tensor-design#FR-1
    /// Method: Requirements-based test
    fn test_tensor_indexing_and_storage() {
        let t = ArrayTensor::<f32, 2, 3>::from_cols([
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
    /// # Verification
    /// Trace: tensor-design#FR-2
    /// Method: Requirements-based test
    fn test_tensor_grid_interpolation() {
        // 2D grid: f(x, y) = [[0, 2], [4, 6]]
        let grid =
            ArrayTensor::<f32, 2, 2>::from_cols([[0.0, 4.0], [2.0, 6.0]]);
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
    /// # Verification
    /// Trace: tensor-design#FR-4
    /// Method: Requirements-based test
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
    /// # Verification
    /// Trace: tensor-design#FR-5
    /// Method: Requirements-based test
    fn test_activations() {
        let relu = Relu;
        assert_almost_eq!(relu.apply(3.5f32), 3.5f32);
        assert_almost_eq!(relu.apply(-2.0f32), 0.0f32);

        let table = TableActivation {
            breakpoints: [-1.0f32, 0.0, 1.0],
            values: [-1.0f32, 0.0, 1.0],
        };
        assert_almost_eq!(table.apply(0.5f32), 0.5f32, 1e-6);
        assert_almost_eq!(table.apply(-0.5f32), -0.5f32, 1e-6);
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: tensor-design#FR-3
    /// Method: Requirements-based test
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

        let t = a.permute(Axes2D::Transpose);
        assert_eq!(t.get(&[0, 1]), a.get(&[1, 0]));
        let back = t.permute(Axes2D::Transpose);
        assert_eq!(back.get(&[1, 2]), a.get(&[1, 2]));

        let sum = &a + &a;
        assert_almost_eq!(sum.get(&[0, 0]).copied().unwrap(), 0.0, 1e-12);
        let scaled = &a * 2.0;
        assert_almost_eq!(scaled.get(&[1, 0]).copied().unwrap(), 6.0, 1e-12);
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: tensor-design#FR-1
    /// Method: Requirements-based test
    fn test_shape4d_and_view() {
        use crate::tensor::{ArrayTensor4D, Shape4D, TensorLayout};
        assert_eq!(Shape4D::<1, 2, 2, 2>::SIZE, 8);
        let t4 = ArrayTensor4D::<f64, 1, 1, 1, 1, 1>::from_storage([3.0]);
        assert_eq!(t4.get(&[0, 0, 0, 0]), Some(&3.0));
        let mut grid =
            ArrayTensor::<f64, 2, 2>::from_cols([[1.0, 2.0], [3.0, 4.0]]);
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
    /// # Verification
    /// Trace: tensor-design#FR-2
    /// Method: Requirements-based test
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

        let grid =
            ArrayTensor::<f64, 2, 2>::from_cols([[1.0, 2.0], [3.0, 4.0]]);
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
        assert_almost_eq!(empty.apply(1.0), 0.0);
        let dup = TableActivation::<f64, 2> {
            breakpoints: [0.0, 0.0],
            values: [1.0, 2.0],
        };
        assert_almost_eq!(dup.apply(0.0), 1.0);
        let clamp_hi = TableActivation::<f64, 2> {
            breakpoints: [0.0, 1.0],
            values: [3.0, 5.0],
        };
        assert_almost_eq!(clamp_hi.apply(-1.0), 3.0);
        assert_almost_eq!(clamp_hi.apply(2.0), 5.0);

        // Test tensor subtraction and pointers
        let t_a = ArrayTensor::<f64, 2, 2>::from_cols([[2.0, 4.0], [6.0, 8.0]]);
        let t_b = ArrayTensor::<f64, 2, 2>::from_cols([[1.0, 2.0], [3.0, 4.0]]);
        let diff = &t_a - &t_b;
        assert_almost_eq!(diff.get(&[0, 0]).copied().unwrap(), 1.0, 1e-12);
        assert_almost_eq!(diff.get(&[1, 1]).copied().unwrap(), 4.0, 1e-12);

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
            ArrayTensor::<f64, 2, 2>::from_cols([[1.0, 2.0], [3.0, 4.0]]);
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
    /// # Verification
    /// Trace: tensor-design#FR-4
    /// Method: Requirements-based test
    fn test_quantization_roundtrip_half_lsb() {
        type Q7 = Quantized<i8, 7>;
        let step = 1.0 / 128.0;
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
    /// # Verification
    /// Trace: tensor-design#FR-4
    /// Method: Requirements-based test
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

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: tensor-design#FR-1
    /// Method: Requirements-based test
    fn test_tensor_permute_square_matrix() {
        let square =
            ArrayTensor::<f32, 2, 2>::from_cols([[1.0, 2.0], [3.0, 4.0]]);

        // 1. Square matrix identity permutation [0, 1]
        let id = square.permute(Axes2D::Identity);
        assert_eq!(id.get(&[0, 0]), Some(&1.0));
        assert_eq!(id.get(&[1, 0]), Some(&2.0));
        assert_eq!(id.get(&[0, 1]), Some(&3.0));
        assert_eq!(id.get(&[1, 1]), Some(&4.0));

        // 2. Square matrix transpose permutation [1, 0]
        let tr = square.permute(Axes2D::Transpose);
        assert_eq!(tr.get(&[0, 0]), Some(&1.0));
        assert_eq!(tr.get(&[1, 0]), Some(&3.0));
        assert_eq!(tr.get(&[0, 1]), Some(&2.0));
        assert_eq!(tr.get(&[1, 1]), Some(&4.0));
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: tensor-design#FR-1
    /// Method: Requirements-based test
    fn test_tensor_permute_rectangular_matrix() {
        // Rectangular matrix transpose permutation [1, 0]
        let rect = ArrayTensor::<f32, 2, 3>::from_cols([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]);
        let rect_tr = rect.permute(Axes2D::Transpose);
        assert_eq!(rect_tr.get(&[0, 0]), Some(&1.0));
        assert_eq!(rect_tr.get(&[1, 0]), Some(&3.0));
        assert_eq!(rect_tr.get(&[2, 0]), Some(&5.0));
        assert_eq!(rect_tr.get(&[0, 1]), Some(&2.0));
        assert_eq!(rect_tr.get(&[1, 1]), Some(&4.0));
        assert_eq!(rect_tr.get(&[2, 1]), Some(&6.0));
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: tensor-design#FR-1
    /// Method: Requirements-based test
    fn test_axes2d_conversions_and_helpers() {
        assert_eq!(Axes2D::try_from([0, 1]), Ok(Axes2D::Identity));
        assert_eq!(Axes2D::try_from([1, 0]), Ok(Axes2D::Transpose));
        assert_eq!(Axes2D::try_from(0), Ok(Axes2D::Identity));
        assert_eq!(Axes2D::try_from(1), Ok(Axes2D::Transpose));
        assert_eq!(Axes2D::from(false), Axes2D::Identity);
        assert_eq!(Axes2D::from(true), Axes2D::Transpose);
        assert_eq!(Axes2D::Identity.as_array(), [0, 1]);
        assert_eq!(Axes2D::Transpose.as_array(), [1, 0]);
        assert!(Axes2D::Identity.is_identity());
        assert!(!Axes2D::Identity.is_transpose());
        assert!(Axes2D::Transpose.is_transpose());
        assert!(!Axes2D::Transpose.is_identity());
    }

    #[test]
    #[should_panic(expected = "invalid permutation axes")]
    fn _test_tensor_permute_invalid_axes_panics() {
        let _ = Axes2D::try_from([0, 0]).expect("invalid permutation axes");
    }

    #[test]
    #[should_panic(expected = "invalid permutation axes")]
    fn _test_tensor_permute_non_square_identity_panic() {
        let rect = ArrayTensor::<f32, 2, 3>::from_cols([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]);
        let _ = rect.permute(Axes2D::Identity);
    }

    /// Rank-2 constructors agree, and each accessor round-trips its own
    /// nesting.
    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: tensor-design#FR-1
    /// Method: Requirements-based test
    fn test_rank2_array_constructors() {
        let rows = [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let cols = [[1.0f32, 4.0], [2.0, 5.0], [3.0, 6.0]];

        let from_rows = ArrayTensor::<f32, 2, 3>::from_rows(rows);
        let from_cols = ArrayTensor::<f32, 2, 3>::from_cols(cols);

        assert_eq!(from_rows.to_rows(), rows);
        assert_eq!(from_cols.to_cols(), cols);
        assert_eq!(from_rows.to_cols(), cols);
        assert_eq!(from_rows.get(&[1, 2]), Some(&6.0));
        assert_eq!(from_rows.as_slice(), from_cols.as_slice());
    }

    /// Rank-3 and rank-4 flat constructors index by
    /// $i_0 + i_1 D_0 + i_2 D_0 D_1 (+ i_3 D_0 D_1 D_2)$.
    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: tensor-design#FR-1
    /// Method: Requirements-based test
    fn test_rank3_and_rank4_array_constructors() {
        let t3 = ArrayTensor3D::<f32, 2, 2, 2, 8>::from_array([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        ]);
        assert_eq!(t3.get(&[0, 0, 0]), Some(&1.0));
        assert_eq!(t3.get(&[1, 0, 1]), Some(&6.0));
        assert_eq!(t3.get(&[1, 1, 1]), Some(&8.0));
        assert_eq!(t3.get(&[2, 0, 0]), None);

        let plane = t3.slice_matrix(&[1]).unwrap();
        assert_eq!(plane.get(0, 0), Some(&5.0));

        let t4 = ArrayTensor4D::<f32, 2, 1, 2, 1, 4>::from_array([
            1.0, 2.0, 3.0, 4.0,
        ]);
        assert_eq!(t4.get(&[1, 0, 1, 0]), Some(&4.0));
        assert_eq!(t4.get(&[0, 0, 1, 0]), Some(&3.0));
    }

    /// Multilinear interpolation is exact for an affine field, so
    /// $f(i,j) = 3i + 5j + 1$ must be reproduced at every interior point.
    ///
    /// The existing centre-point check uses equal weights, which a swap of
    /// `frac` and `1 - frac` also satisfies. An affine field at an asymmetric
    /// point pins each corner weight individually.
    ///
    /// # Verification
    /// Trace: tensor-design#FR-2
    /// Method: Requirements-based test
    #[cfg_attr(test, test)]
    fn test_interpolate_is_exact_on_an_affine_field() {
        // f(i, j) = 3i + 5j + 1 over a 2x2 grid.
        let grid =
            ArrayTensor::<f64, 2, 2>::from_cols([[1.0, 4.0], [6.0, 9.0]]);

        for &(i, j) in &[
            (0.0_f64, 0.0_f64),
            (0.3, 0.8),
            (0.25, 0.75),
            (1.0, 0.5),
            (0.5, 1.0),
            (0.9, 0.1),
        ] {
            let want = 3.0 * i + 5.0 * j + 1.0;
            assert_almost_eq!(grid.interpolate(&[i, j]), want, 1e-12);
        }
    }

    /// An asymmetric interior point on an asymmetric grid: swapping the upper
    /// and lower corner weights changes the answer.
    ///
    /// # Verification
    /// Trace: tensor-design#FR-2
    /// Method: Requirements-based test
    #[cfg_attr(test, test)]
    fn test_interpolate_weights_are_not_symmetric() {
        // f(0,0)=0, f(1,0)=4, f(0,1)=2, f(1,1)=6.
        let grid =
            ArrayTensor::<f64, 2, 2>::from_cols([[0.0, 4.0], [2.0, 6.0]]);
        // 0.75*0.25*0 + 0.25*0.25*4 + 0.75*0.75*2 + 0.25*0.75*6 = 2.5
        assert_almost_eq!(grid.interpolate(&[0.25, 0.75]), 2.5, 1e-12);
        // The mirrored point is a different value.
        assert_almost_eq!(grid.interpolate(&[0.75, 0.25]), 3.5, 1e-12);
    }

    /// A non-square grid indexes each axis with its own extent, so a
    /// transposed stride would land on the wrong corner.
    ///
    /// # Verification
    /// Trace: tensor-design#FR-2
    /// Method: Requirements-based test
    #[cfg_attr(test, test)]
    fn test_interpolate_non_square_grid() {
        // 2 rows, 3 columns, f(i, j) = 10i + j.
        let grid = ArrayTensor::<f64, 2, 3>::from_cols([
            [0.0, 10.0],
            [1.0, 11.0],
            [2.0, 12.0],
        ]);
        assert_almost_eq!(grid.interpolate(&[0.0, 2.0]), 2.0, 1e-12);
        assert_almost_eq!(grid.interpolate(&[1.0, 2.0]), 12.0, 1e-12);
        assert_almost_eq!(grid.interpolate(&[0.5, 1.5]), 6.5, 1e-12);
        // Clamping uses each axis's own maximum, not a shared one.
        assert_almost_eq!(grid.interpolate(&[5.0, 5.0]), 12.0, 1e-12);
        assert_almost_eq!(grid.interpolate(&[0.0, 5.0]), 2.0, 1e-12);
    }

    /// Rank 3 exercises all eight corners of the multilinear sum.
    ///
    /// # Verification
    /// Trace: tensor-design#FR-2
    /// Method: Requirements-based test
    #[cfg_attr(test, test)]
    fn test_interpolate_rank_three_affine() {
        // f(i, j, k) = i + 2j + 4k over a 2x2x2 grid, column-major flat.
        let mut buf = [0.0_f64; 8];
        for k in 0..2 {
            for j in 0..2 {
                for i in 0..2 {
                    let v = i as f64 + 2.0 * j as f64 + 4.0 * k as f64;
                    buf[i + 2 * j + 4 * k] = v;
                }
            }
        }
        let t = ArrayTensor3D::<f64, 2, 2, 2, 8>::from_storage(buf);
        assert_almost_eq!(t.interpolate(&[0.0, 0.0, 0.0]), 0.0, 1e-12);
        assert_almost_eq!(t.interpolate(&[1.0, 1.0, 1.0]), 7.0, 1e-12);
        assert_almost_eq!(t.interpolate(&[0.5, 0.5, 0.5]), 3.5, 1e-12);
        assert_almost_eq!(t.interpolate(&[0.25, 0.5, 0.75]), 4.25, 1e-12);
    }

    /// ReLU is `max(0, x)`: the boundary at zero belongs to the zero branch,
    /// and the identity branch must not scale or shift its input.
    ///
    /// # Verification
    /// Trace: tensor-design#FR-5
    /// Method: Requirements-based test
    #[cfg_attr(test, test)]
    fn test_relu_boundary_and_identity() {
        let relu = Relu;
        assert_almost_eq!(relu.apply(0.0_f64), 0.0, 1e-12);
        assert_almost_eq!(relu.apply(-1e-12_f64), 0.0, 1e-12);
        assert_almost_eq!(relu.apply(1e-12_f64), 1e-12, 1e-24);
        assert_almost_eq!(relu.apply(1234.5_f64), 1234.5, 1e-12);
        assert_almost_eq!(relu.apply(-1234.5_f64), 0.0, 1e-12);
        // Not the identity, and not a constant.
        assert!((relu.apply(-3.0_f64) - -3.0).abs() > 1.0);
        assert!((relu.apply(3.0_f64) - relu.apply(2.0_f64)).abs() > 0.5);
    }

    /// The table activation interpolates linearly inside each segment and
    /// saturates outside the breakpoint range.
    ///
    /// The segments here have unequal widths and slopes, so a wrong segment
    /// index or a swapped endpoint gives a different value.
    ///
    /// # Verification
    /// Trace: tensor-design#FR-5
    /// Method: Requirements-based test
    #[cfg_attr(test, test)]
    fn test_table_activation_segments_and_saturation() {
        // Breakpoints at -2, 0, 1, 5 with values 0, 4, 5, 1.
        // Slopes: 2 on [-2,0], 1 on [0,1], -1 on [1,5].
        let t = TableActivation::<f64, 4> {
            breakpoints: [-2.0, 0.0, 1.0, 5.0],
            values: [0.0, 4.0, 5.0, 1.0],
        };

        // Exactly on each breakpoint.
        assert_almost_eq!(t.apply(-2.0), 0.0, 1e-12);
        assert_almost_eq!(t.apply(0.0), 4.0, 1e-12);
        assert_almost_eq!(t.apply(1.0), 5.0, 1e-12);
        assert_almost_eq!(t.apply(5.0), 1.0, 1e-12);

        // Inside each segment, at its own slope.
        assert_almost_eq!(t.apply(-1.0), 2.0, 1e-12);
        assert_almost_eq!(t.apply(-0.5), 3.0, 1e-12);
        assert_almost_eq!(t.apply(0.25), 4.25, 1e-12);
        assert_almost_eq!(t.apply(3.0), 3.0, 1e-12);
        assert_almost_eq!(t.apply(4.0), 2.0, 1e-12);

        // Saturation outside the range holds the end values.
        assert_almost_eq!(t.apply(-100.0), 0.0, 1e-12);
        assert_almost_eq!(t.apply(100.0), 1.0, 1e-12);

        // A single breakpoint is a constant function.
        let one = TableActivation::<f64, 1> {
            breakpoints: [3.0],
            values: [7.0],
        };
        assert_almost_eq!(one.apply(-5.0), 7.0, 1e-12);
        assert_almost_eq!(one.apply(3.0), 7.0, 1e-12);
        assert_almost_eq!(one.apply(9.0), 7.0, 1e-12);
    }

    /// `to_rows` reads the tensor out in row-major order, which is the
    /// transpose of the column-major storage it is built from.
    ///
    /// # Verification
    /// Trace: tensor-design#FR-1
    /// Method: Requirements-based test
    #[cfg_attr(test, test)]
    fn test_to_rows_matches_element_access() {
        // Columns [1,2,3], [4,5,6] -> rows [1,4], [2,5], [3,6].
        let t = ArrayTensor::<f64, 3, 2>::from_cols([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]);
        let rows = t.to_rows();
        assert_eq!(rows, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);

        // Every entry agrees with indexed access, so the walk order is right.
        for (i, row) in rows.iter().enumerate() {
            for (j, v) in row.iter().enumerate() {
                assert_almost_eq!(*v, *t.get(&[i, j]).unwrap(), 1e-12);
            }
        }

        // A 1xN tensor stays a single row, an Nx1 becomes N rows of one.
        let wide = ArrayTensor::<f64, 1, 3>::from_cols([[7.0], [8.0], [9.0]]);
        assert_eq!(wide.to_rows(), [[7.0, 8.0, 9.0]]);
        let tall = ArrayTensor::<f64, 3, 1>::from_cols([[7.0, 8.0, 9.0]]);
        assert_eq!(tall.to_rows(), [[7.0], [8.0], [9.0]]);
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
        /// # Verification
        /// Trace: tensor-design#FR-4
        /// Method: Property-based test
        fn prop_quantization_roundtrip_half_lsb(
            x in -1.0_f64..Q7::MAX.dequantize(),
        ) {
            let half = (1.0 / 128.0) / 2.0;
            let err = (x - Q7::quantize(x).dequantize()).abs();
            prop_assert!(
                err <= half,
                "Q7 round-trip {err} exceeds {half} for x={x}"
            );
        }

        /// $x > y$ implies $\mathrm{quant}(x) \ge \mathrm{quant}(y)$.
        #[test]
        /// # Verification
        /// Trace: tensor-design#FR-4
        /// Method: Property-based test
        fn prop_quantization_monotonicity(
            x in -4.0_f64..4.0,
            y in -4.0_f64..4.0,
        ) {
            prop_assume!(x > y);
            prop_assert!(Q7::quantize(x) >= Q7::quantize(y));
        }
    }
}
