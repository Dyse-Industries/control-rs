//! State-space demo. Copy this file and point `Store` / `Blas` at your backends.
//!
//! `Store::from_array` uses **column-major** literals (each inner array is one
//! column). NumPy `[[a, b], [c, d]]` is `[[a, c], [b, d]]` in Rust.

use crate::{ABS_F64, native_artifact, owned_to_rows, print_matrix, save};
use control_rs::math::num_types::Const;
use control_rs::math::storage::ArrayStorage;
use control_rs::math::subprograms::DefaultBlas;
use control_rs::matrix::Matrix;
use control_rs::state_space::ArrayStateSpace;
use serde_json::json;

/// Swap this for a custom dense backend on the system matrices.
type Store<const R: usize, const C: usize> = ArrayStorage<f64, R, C>;
/// Swap this for a hardware BLAS (`Gemm`).
type Blas = DefaultBlas;
type Mat<const R: usize, const C: usize> =
    Matrix<f64, Const<R>, Const<C>, Store<R, C>>;

pub fn main() {
    println!("=== State-Space Numerical Model Example ===");

    let a_c = Mat::<2, 2>::from_storage(Store::from_array([
        [0.0, -4.0],
        [1.0, -0.8],
    ]));
    let b_c = Mat::<2, 1>::from_storage(Store::from_array([[0.0, 1.0]]));
    let c_c = Mat::<1, 2>::from_storage(Store::from_array([[1.0], [0.0]]));
    let d_c = Mat::<1, 1>::zero();

    let sys_c = ArrayStateSpace::continuous(a_c, b_c, c_c, d_c);

    println!("\n--- Continuous-Time System ---");
    print_matrix("A_c", &sys_c.a());
    print_matrix("B_c", &sys_c.b());
    print_matrix("C_c", &sys_c.c());
    print_matrix("D_c", &sys_c.d());

    let x_test = Mat::<2, 1>::from_storage(Store::from_array([[1.0, 0.5]]));
    let u_test = Mat::<1, 1>::zero();
    let (x_dot, y_test) = sys_c.derivative(&x_test, &u_test);

    print_matrix("x_dot at [1.0, 0.5]^T", &x_dot);
    let y_val = y_test.get(0, 0).copied().unwrap_or(0.0);
    println!("y at [1.0, 0.5]^T: {y_val:.6}");

    let dt = 0.05;
    let sys_d = sys_c.to_discrete_zoh(dt);

    println!("\n--- Discrete-Time System (ZOH, Ts = {dt}s) ---");
    print_matrix("A_d", &sys_d.a());
    print_matrix("B_d", &sys_d.b());
    print_matrix("C_d", &sys_d.c());
    print_matrix("D_d", &sys_d.d());

    let num_steps = 20;
    let mut x_k = Mat::<2, 1>::zero();
    let u_step = Mat::<1, 1>::from_fn(|_, _| 1.0);
    let mut step_x1 = [0.0_f64; 20];
    let mut step_x2 = [0.0_f64; 20];
    let mut step_y = [0.0_f64; 20];

    println!("\n--- 20-Step Unit Step Trajectory ---");
    println!(
        "{:<6}{:<16}{:<16}{:<16}",
        "Step", "x_1 (pos)", "x_2 (vel)", "y (output)"
    );
    for k in 0..num_steps {
        let pos = x_k.get(0, 0).copied().unwrap_or(0.0);
        let vel = x_k.get(1, 0).copied().unwrap_or(0.0);
        let (x_next, y_k) = sys_d.step(&x_k, &u_step);
        let y_out = y_k.get(0, 0).copied().unwrap_or(0.0);
        step_x1[k] = pos;
        step_x2[k] = vel;
        step_y[k] = y_out;
        println!("{k:<6}{pos:<16.8}{vel:<16.8}{y_out:<16.8}");

        let mut ax = Mat::<2, 1>::zero();
        let mut bu = Mat::<2, 1>::zero();
        sys_d
            .a()
            .mul_into_with::<Blas, Const<1>, _, _>(&x_k, &mut ax);
        sys_d
            .b()
            .mul_into_with::<Blas, Const<1>, _, _>(&u_step, &mut bu);
        let x_pred: Mat<2, 1> = &ax + &bu;
        for i in 0..2 {
            let got = x_next.get(i, 0).copied().unwrap();
            let expected = x_pred.get(i, 0).copied().unwrap();
            assert!(
                (got - expected).abs() <= ABS_F64,
                "step {k} x[{i}]: {got} vs {expected}"
            );
        }
        x_k = x_next;
    }

    let t =
        Mat::<2, 2>::from_storage(Store::from_array([[1.0, 0.0], [1.0, 1.0]]));
    let sys_transformed = sys_d.similarity_transform(&t).expect("similarity");
    println!("\n--- Transformed System (T = [[1, 1], [0, 1]]) ---");
    print_matrix("A_tilde", &sys_transformed.a());
    print_matrix("B_tilde", &sys_transformed.b());
    print_matrix("C_tilde", &sys_transformed.c());

    let t_axis: Vec<f64> = (0..20).map(|k| f64::from(k) * dt).collect();
    let values = json!({
        "X_DOT": owned_to_rows(&x_dot),
        "Y_TEST": y_val,
        "AD": owned_to_rows(&sys_d.a()),
        "BD": owned_to_rows(&sys_d.b()),
        "STEP_X1": step_x1,
        "STEP_X2": step_x2,
        "STEP_Y": step_y,
        "A_TILDE": owned_to_rows(&sys_transformed.a()),
        "B_TILDE": owned_to_rows(&sys_transformed.b()),
        "C_TILDE": owned_to_rows(&sys_transformed.c()),
    });
    let series = json!({
        "step_y": { "x": t_axis, "y": step_y },
        "step_x1": { "x": t_axis, "y": step_x1 },
        "step_x2": { "x": t_axis, "y": step_x2 },
    });
    save(
        "results/state_space/native.json",
        &native_artifact("state_space", values, series),
    );
}
