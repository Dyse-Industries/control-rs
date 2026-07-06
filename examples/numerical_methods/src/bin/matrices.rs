//! Demonstration of the generic compile-time Matrix struct and specialized structures.
//!
//! This example simulates a discretized physical mass-spring-damper system using state-space equations:
//! x_{k+1} = (A - B * K) * x_k + B * (k_r * r)

use control_rs::matrix::{
    LowerTriangular, Matrix, RowVector, SquareMatrix, Symmetric,
    UpperTriangular, Vector,
};

fn main() {
    println!("=== Matrix Demonstration (State-Space Simulation) ===");

    // 1. Define physical system matrices
    // Let's assume a discretized mass-spring-damper system with sample time dt = 0.1s.
    // State vector: x = [position, velocity]^T
    // State transition matrix A (2x2):
    // [ 1.0,   0.1 ]
    // [ -0.2,  0.9 ]
    // (Note: Matrix::new takes an array of columns (column-major order))
    let a: SquareMatrix<f64, 2> = Matrix::new([
        [1.0, -0.2], // Column 0
        [0.1, 0.9],  // Column 1
    ]);

    // Input coupling matrix B (2x1 column vector):
    // [ 0.0 ]
    // [ 0.1 ]
    let b: Vector<f64, 2> = Matrix::new([
        [0.0, 0.1], // Column 0
    ]);

    println!("System Matrix A (dimensions: {}x{}):", a.rows(), a.cols());
    println!("  A[0, 0] = {:?}", a.get(0, 0));
    println!("  A[1, 0] = {:?}", a.get(1, 0));
    println!("Input Matrix B (dimensions: {}x{}):", b.rows(), b.cols());
    println!("  B[0, 0] = {:?}", b.get(0, 0));
    println!("  B[1, 0] = {:?}", b.get(1, 0));

    // 2. Define Controller Parameters
    // State feedback gain matrix K (1x2 row vector):
    // [ 2.0, 0.5 ]
    let k: RowVector<f64, 2> = Matrix::new([
        [2.0], // Column 0
        [0.5], // Column 1
    ]);

    // Feedforward scaling factor (scalar)
    let k_r = 2.0;

    // Reference signal (step input of 1.0)
    let r = 1.0;

    // 3. Compute Closed-loop system dynamics: A_cl = A - B * K
    // Multiplication of B (2x1) and K (1x2) yields a (2x2) matrix.
    let bk = b * k;
    let a_cl = a - bk;

    println!("\nClosed-loop system matrix A_cl = A - B * K:");
    println!("  A_cl[0, 0] = {:?}", a_cl.get(0, 0)); // Expected: 1.0
    println!("  A_cl[1, 0] = {:?}", a_cl.get(1, 0)); // Expected: -0.4 (since -0.2 - 0.1 * 2.0)
    println!("  A_cl[0, 1] = {:?}", a_cl.get(0, 1)); // Expected: 0.1
    println!("  A_cl[1, 1] = {:?}", a_cl.get(1, 1)); // Expected: 0.85 (since 0.9 - 0.1 * 0.5)

    // Pre-calculate scaled input term: B_scaled = B * (k_r * r)
    let b_scaled = b * (k_r * r);

    // 4. Run State-space Simulation
    // Initial state x_0 = [0.0, 0.0]^T (at rest)
    let mut x: Vector<f64, 2> = Matrix::new([
        [0.0, 0.0], // Column 0
    ]);

    println!("\nSimulating closed-loop response to step input:");
    for step in 0..11 {
        // Calculate control input: u_k = -K * x_k + k_r * r
        // k * x yields a 1x1 matrix.
        let kx_mat = k * x;
        let kx = kx_mat.get(0, 0).copied().unwrap_or(0.0);
        let u = -kx + (k_r * r);

        println!(
            "Step {:2}: Position = {:8.4}, Velocity = {:8.4}, Control Input (u) = {:8.4}",
            step,
            x.get(0, 0).copied().unwrap_or(0.0),
            x.get(1, 0).copied().unwrap_or(0.0),
            u
        );

        // State update: x_{k+1} = A_cl * x_k + B_scaled
        x = a_cl * x + b_scaled;
    }

    // 5. Specialized wrappers (Upper, Lower, Symmetric)
    // In control systems, covariance matrices (Kalman filters) or cost matrices (LQR)
    // are symmetric. Triangular matrices are common in numerical matrix decompositions (Cholesky).
    println!("\n=== Specialized Matrix Wrappers ===");
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
