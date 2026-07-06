//! Demonstration of the generic compile-time Tensor struct.

use control_rs::math::num_types::Const;
use control_rs::tensor::{Tensor, TensorLayout};

fn main() {
    println!("=== Tensor Demonstration ===");

    // 1. Defining shape and layout traits
    // Dims define dimension lengths.
    // 3D tensor of shape 2 x 2 x 2 (total elements = 8)
    type Shape3D = (Const<2>, Const<2>, Const<2>);

    println!("TensorLayout Shape3D properties:");
    println!("  Rank: {}", Shape3D::RANK);
    println!("  Size: {}", Shape3D::SIZE);
    println!("  Dimensions: {:?}", Shape3D::dims());

    // 2. Construction
    // A 3D tensor initialized with elements 1..=8
    let mut tensor = Tensor::<f64, 8, Shape3D>::new([
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    ]);

    // 3. Multi-coordinate indexing
    // Coordinates use column-major ordering (first dimension varies fastest).
    // coords [0, 0, 0] -> flat index 0
    // coords [1, 0, 0] -> flat index 1
    // coords [0, 1, 0] -> flat index 2
    // coords [0, 0, 1] -> flat index 4
    println!("Coordinate indexing:");
    println!("  tensor[0, 0, 0] = {:?}", tensor.get(&[0, 0, 0])); // 1.0
    println!("  tensor[1, 0, 0] = {:?}", tensor.get(&[1, 0, 0])); // 2.0
    println!("  tensor[0, 1, 0] = {:?}", tensor.get(&[0, 1, 0])); // 3.0
    println!("  tensor[0, 0, 1] = {:?}", tensor.get(&[0, 0, 1])); // 5.0

    // Out of bounds checking
    println!(
        "  Out of bounds coords [2, 0, 0] = {:?}",
        tensor.get(&[2, 0, 0])
    ); // None
    println!(
        "  Dimension mismatch coords [0, 0] = {:?}",
        tensor.get(&[0, 0])
    ); // None

    // Mutation
    if let Some(val) = tensor.get_mut(&[1, 1, 1]) {
        *val = 80.0;
    }
    println!("  Mutated tensor[1, 1, 1] = {:?}", tensor.get(&[1, 1, 1])); // 80.0

    // 4. Arithmetic (Addition / Subtraction / Scaling)
    let t1 = Tensor::<f64, 4, (Const<2>, Const<2>)>::new([1.0, 2.0, 3.0, 4.0]);
    let t2 =
        Tensor::<f64, 4, (Const<2>, Const<2>)>::new([10.0, 20.0, 30.0, 40.0]);

    // Addition / Subtraction
    let t_sum = t1 + t2;
    println!("(t1 + t2) elements: {:?}", t_sum.as_slice());

    // Scaling
    let t_scaled = t1 * 3.0;
    println!("(t1 * 3.0) elements: {:?}", t_scaled.as_slice());
}
