//! Demonstration of the generic compile-time Matrix struct and specialized structures.

use control_rs::matrix::{
    LowerTriangular, Matrix, RowVector, SquareMatrix, Symmetric,
    UpperTriangular, Vector,
};

fn main() {
    println!("=== Matrix Demonstration ===");

    // 1. Creation and layout
    // Matrix::new takes an array of arrays representing columns (column-major order).
    // Matrix A: 2 rows, 3 columns:
    // [ 1.0, 3.0, 5.0 ]
    // [ 2.0, 4.0, 6.0 ]
    let mut a = Matrix::new([
        [1.0, 2.0], // Column 0
        [3.0, 4.0], // Column 1
        [5.0, 6.0], // Column 2
    ]);

    println!("Matrix A dimensions: {}x{}", a.rows(), a.cols());
    println!("Element A[0, 1] (row 0, col 1): {:?}", a.get(0, 1)); // Should be 3.0
    println!("Element A[1, 2] (row 1, col 2): {:?}", a.get(1, 2)); // Should be 6.0

    // Mutation
    if let Some(val) = a.get_mut(1, 2) {
        *val = 60.0;
    }
    println!("Mutated element A[1, 2]: {:?}", a.get(1, 2));

    // 2. Types Aliases
    let _sq: SquareMatrix<f64, 3> =
        Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let _vec: Vector<f64, 3> = Matrix::new([[1.0, 2.0, 3.0]]); // 3x1 Column Vector
    let _row: RowVector<f64, 3> = Matrix::new([[1.0], [2.0], [3.0]]); // 1x3 Row Vector

    // 3. Matrix Arithmetic (Addition / Subtraction / Scaling)
    let m1 = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    let m2 = Matrix::new([[10.0, 20.0], [30.0, 40.0]]);

    let m_sum = m1 + m2;
    println!("(m1 + m2) columns: {:?}", m_sum);

    let m_scaled = m1 * 2.0;
    println!("(m1 * 2.0) columns: {:?}", m_scaled);

    // 4. Matrix-Matrix Multiplication (GEMM)
    // A: 2x3, B: 3x2. Multiplication yields C: 2x2.
    // Memory layout and calculations are handled via standard Level 3 BLAS GEMM.
    let mat_a = Matrix::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]); // 2x3
    let mat_b = Matrix::new([[7.0, 8.0, 1.0], [9.0, 10.0, 2.0]]); // 3x2

    let mat_c = mat_a * mat_b; // 2x2
    println!("Matrix Multiplication A * B = C:");
    println!("  C rows: {}, cols: {}", mat_c.rows(), mat_c.cols());
    println!("  C[0, 0]: {:?}", mat_c.get(0, 0)); // Should be 36.0
    println!("  C[1, 1]: {:?}", mat_c.get(1, 1)); // Should be 70.0

    // 5. Matrix-Vector Multiplication
    // Vector is just a 3x1 Matrix.
    let mat_v: Vector<f64, 3> = Matrix::new([[2.0, 1.0, -1.0]]);
    let mat_y = mat_a * mat_v; // 2x1 Column Vector
    println!("Matrix-Vector Multiplication A * v = y:");
    println!("  y dimensions: {}x{}", mat_y.rows(), mat_y.cols());
    println!("  y[0, 0]: {:?}", mat_y.get(0, 0)); // 1*2 + 3*1 + 5*(-1) = 0.0
    println!("  y[1, 0]: {:?}", mat_y.get(1, 0)); // 2*2 + 4*1 + 6*(-1) = 2.0

    // 6. Specialized wrappers (Upper, Lower, Symmetric)
    println!("Specialized Matrix Wrappers:");
    let raw = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

    let ut = UpperTriangular::new(raw).unwrap();
    println!("  UpperTriangular [0, 0]: {:?}", ut.get(0, 0)); // 1.0
    println!("  UpperTriangular [0, 1]: {:?}", ut.get(0, 1)); // 4.0
    println!("  UpperTriangular [2, 0]: {:?}", ut.get(2, 0)); // 3.0

    let lt = LowerTriangular::new(raw).unwrap();
    println!("  LowerTriangular [1, 0]: {:?}", lt.get(1, 0)); // 2.0

    let mut sym = Symmetric::new(raw).unwrap();
    sym.set(0, 1, 42.0).unwrap(); // Updates both [0, 1] and [1, 0] to maintain symmetry
    println!("  Symmetric [0, 1]: {:?}", sym.get(0, 1)); // 42.0
    println!("  Symmetric [1, 0]: {:?}", sym.get(1, 0)); // 42.0
}
