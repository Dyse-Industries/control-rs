#![allow(dead_code)]
#![allow(clippy::arithmetic_side_effects)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::indexing_slicing)]
#![allow(clippy::panic)]
#![allow(clippy::missing_const_for_fn)]
#[cfg_attr(all(not(test), not(feature = "std")), control_rs_macros::hil_suite)]
/// HIL and unit test suite for tensors.
pub mod tensor_tests {
    use crate::math::num_types::Const;
    use crate::tensor::{Tensor, TensorLayout};

    #[cfg_attr(test, test)]
    fn test_tensor_dims_traits() {
        type Shape2D = (Const<2>, Const<3>);
        type Shape3D = (Const<2>, Const<2>, Const<2>);

        assert_eq!(<Const<3> as TensorLayout>::RANK, 1);
        assert_eq!(<Const<3> as TensorLayout>::SIZE, 3);
        assert_eq!(<Const<3> as TensorLayout>::dims(), &[3]);

        assert_eq!(Shape2D::RANK, 2);
        assert_eq!(Shape2D::SIZE, 6);
        assert_eq!(Shape2D::dims(), &[2, 3]);

        assert_eq!(Shape3D::RANK, 3);
        assert_eq!(Shape3D::SIZE, 8);
        assert_eq!(Shape3D::dims(), &[2, 2, 2]);
    }

    #[cfg_attr(test, test)]
    fn test_tensor_construction_and_indexing() {
        let mut tensor =
            Tensor::<i32, 8, (Const<2>, Const<2>, Const<2>)>::new([
                1, 2, 3, 4, 5, 6, 7, 8,
            ]);

        assert_eq!(tensor.rank(), 3);
        assert_eq!(tensor.shape(), &[2, 2, 2]);

        assert_eq!(tensor.get(&[0, 0, 0]), Some(&1));
        assert_eq!(tensor.get(&[1, 0, 0]), Some(&2));
        assert_eq!(tensor.get(&[0, 1, 0]), Some(&3));
        assert_eq!(tensor.get(&[1, 1, 0]), Some(&4));
        assert_eq!(tensor.get(&[0, 0, 1]), Some(&5));
        assert_eq!(tensor.get(&[1, 1, 1]), Some(&8));

        assert_eq!(tensor.get(&[2, 0, 0]), None);
        assert_eq!(tensor.get(&[0, 0]), None);
        assert_eq!(tensor.get(&[0, 0, 0, 0]), None);

        if let Some(val) = tensor.get_mut(&[0, 1, 1]) {
            *val = 42;
        }
        assert_eq!(tensor.get(&[0, 1, 1]), Some(&42));
    }

    #[cfg_attr(test, test)]
    fn test_tensor_arithmetic() {
        let mut t1 =
            Tensor::<f32, 4, (Const<2>, Const<2>)>::new([1.0, 2.0, 3.0, 4.0]);
        let t2 = Tensor::<f32, 4, (Const<2>, Const<2>)>::new([
            10.0, 20.0, 30.0, 40.0,
        ]);

        t1 += t2;
        assert_eq!(t1.as_slice(), &[11.0, 22.0, 33.0, 44.0]);

        t1 -= t2;
        assert_eq!(t1.as_slice(), &[1.0, 2.0, 3.0, 4.0]);

        let t3 = t1 + t2;
        assert_eq!(t3.as_slice(), &[11.0, 22.0, 33.0, 44.0]);

        let t4 = t3 - t2;
        assert_eq!(t4.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[cfg_attr(test, test)]
    fn test_tensor_scaling() {
        let mut tensor = Tensor::<f32, 6, (Const<2>, Const<3>)>::new([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        ]);

        tensor *= 2.0;
        assert_eq!(tensor.as_slice(), &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);

        tensor /= 2.0;
        assert_eq!(tensor.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let t2 = tensor * 3.0;
        assert_eq!(t2.as_slice(), &[3.0, 6.0, 9.0, 12.0, 15.0, 18.0]);

        let t3 = t2 / 3.0;
        assert_eq!(t3.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[cfg_attr(test, test)]
    fn test_tensor_empty_and_compile_fail() {
        // Statically validated sizes
    }
}
