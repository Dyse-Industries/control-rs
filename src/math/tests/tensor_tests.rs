#![allow(clippy::items_after_statements)]

use crate::math::{
    num_types::Const,
    tensor::{Tensor, TensorLayout},
};

#[test]
fn test_tensor_dims_traits() {
    // 1D shape Const<3>
    assert_eq!(<Const<3> as TensorLayout>::RANK, 1);
    assert_eq!(<Const<3> as TensorLayout>::SIZE, 3);
    assert_eq!(<Const<3> as TensorLayout>::dims(), &[3]);

    // 2D shape (Const<2>, Const<3>)
    type Shape2D = (Const<2>, Const<3>);
    assert_eq!(Shape2D::RANK, 2);
    assert_eq!(Shape2D::SIZE, 6);
    assert_eq!(Shape2D::dims(), &[2, 3]);

    // 3D shape (Const<2>, Const<2>, Const<2>)
    type Shape3D = (Const<2>, Const<2>, Const<2>);
    assert_eq!(Shape3D::RANK, 3);
    assert_eq!(Shape3D::SIZE, 8);
    assert_eq!(Shape3D::dims(), &[2, 2, 2]);
}

#[test]
fn test_tensor_construction_and_indexing() {
    // 3D tensor of shape 2 x 2 x 2
    // elements: 1, 2, 3, 4, 5, 6, 7, 8
    let mut tensor = Tensor::<i32, 8, (Const<2>, Const<2>, Const<2>)>::new([
        1, 2, 3, 4, 5, 6, 7, 8,
    ]);

    assert_eq!(tensor.rank(), 3);
    assert_eq!(tensor.shape(), &[2, 2, 2]);

    // Test indexing
    // In column-major layout, coordinate indices:
    // coord [0, 0, 0] -> flat index 0
    // coord [1, 0, 0] -> flat index 1
    // coord [0, 1, 0] -> flat index 2
    // coord [1, 1, 0] -> flat index 3
    // coord [0, 0, 1] -> flat index 4
    // ...
    assert_eq!(tensor.get(&[0, 0, 0]), Some(&1));
    assert_eq!(tensor.get(&[1, 0, 0]), Some(&2));
    assert_eq!(tensor.get(&[0, 1, 0]), Some(&3));
    assert_eq!(tensor.get(&[1, 1, 0]), Some(&4));
    assert_eq!(tensor.get(&[0, 0, 1]), Some(&5));
    assert_eq!(tensor.get(&[1, 1, 1]), Some(&8));

    // Test out of bounds indexing
    assert_eq!(tensor.get(&[2, 0, 0]), None);
    assert_eq!(tensor.get(&[0, 0]), None); // dimension mismatch
    assert_eq!(tensor.get(&[0, 0, 0, 0]), None); // dimension mismatch

    // Test mutable indexing
    if let Some(val) = tensor.get_mut(&[0, 1, 1]) {
        *val = 42;
    }
    assert_eq!(tensor.get(&[0, 1, 1]), Some(&42));
}

#[test]
fn test_tensor_arithmetic() {
    let mut t1 =
        Tensor::<f32, 4, (Const<2>, Const<2>)>::new([1.0, 2.0, 3.0, 4.0]);
    let t2 =
        Tensor::<f32, 4, (Const<2>, Const<2>)>::new([10.0, 20.0, 30.0, 40.0]);

    // Addition
    t1 += t2;
    assert_eq!(t1.as_slice(), &[11.0, 22.0, 33.0, 44.0]);

    // Subtraction
    t1 -= t2;
    assert_eq!(t1.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn test_tensor_scaling() {
    let mut tensor = Tensor::<f32, 6, (Const<2>, Const<3>)>::new([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
    ]);

    // MulAssign
    tensor *= 2.0;
    assert_eq!(tensor.as_slice(), &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);

    // DivAssign
    tensor /= 2.0;
    assert_eq!(tensor.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}
