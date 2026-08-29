//! State-space demo. Copy this file and point `Store` / `Blas` at your backends.
//!
//! `Store::from_array` uses **column-major** literals (each inner array is one
//! column). NumPy `[[a, b], [c, d]]` is `[[a, c], [b, d]]` in Rust.

use crate::suite::{
    case_inputs, emit_stdout, json_f64, json_f64_vec, json_usize,
    require_usize, storage_from_rows,
};
use crate::{
    ABS_F64, native_artifact, owned_to_rows, print_matrix, time_kernel,
    timing_entry,
};
use control_rs::math::num_types::Const;
use control_rs::math::storage::ArrayStorage;
use control_rs::math::subprograms::DefaultBlas;
use control_rs::matrix::Matrix;
use control_rs::state_space::ArrayStateSpace;
use serde_json::{Value, json};

/// Swap this for a custom dense backend on the system matrices.
type Store<const R: usize, const C: usize> = ArrayStorage<f64, R, C>;
/// Swap this for a hardware BLAS (`Gemm`).
type Blas = DefaultBlas;
type Mat<const R: usize, const C: usize> =
    Matrix<f64, Const<R>, Const<C>, Store<R, C>>;

const NUM_STEPS: usize = 200;
const STIFF_STEPS: usize = 200;

fn run_traj(
    sys_d: &ArrayStateSpace<f64, 2, 1, 1>,
    n: usize,
    x0: [f64; 2],
    u: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut x_k =
        Mat::<2, 1>::from_storage(Store::from_array([[x0[0], x0[1]]]));
    let u_k = Mat::<1, 1>::from_fn(|_, _| u);
    let mut traj_x1 = vec![0.0_f64; n];
    let mut traj_x2 = vec![0.0_f64; n];
    let mut traj_y = vec![0.0_f64; n];
    for k in 0..n {
        let pos = x_k.get(0, 0).copied().unwrap_or(0.0);
        let vel = x_k.get(1, 0).copied().unwrap_or(0.0);
        let (x_next, y_k) = sys_d.step(&x_k, &u_k);
        let y_out = y_k.get(0, 0).copied().unwrap_or(0.0);
        traj_x1[k] = pos;
        traj_x2[k] = vel;
        traj_y[k] = y_out;
        x_k = x_next;
    }
    (traj_x1, traj_x2, traj_y)
}

fn vec2(v: &Value) -> [f64; 2] {
    let x = json_f64_vec(v);
    [x[0], x[1]]
}

pub fn run(suite: &Value) {
    eprintln!("=== State-Space Numerical Model Example ===");

    let tut = case_inputs(suite, "state_space.host.tutorial_plant");
    require_usize(tut, "n_steps", NUM_STEPS);
    let a_c = Mat::<2, 2>::from_storage(storage_from_rows(&tut["A"]));
    let b_c = Mat::<2, 1>::from_storage(storage_from_rows(&tut["B"]));
    let c_c = Mat::<1, 2>::from_storage(storage_from_rows(&tut["C"]));
    let d_c = Mat::<1, 1>::from_storage(storage_from_rows(&tut["D"]));

    let sys_c = ArrayStateSpace::continuous(a_c, b_c, c_c, d_c);

    eprintln!("\n--- Continuous-Time System ---");
    print_matrix("A_c", &sys_c.a());
    print_matrix("B_c", &sys_c.b());
    print_matrix("C_c", &sys_c.c());
    print_matrix("D_c", &sys_c.d());

    let x0_test = vec2(&tut["x_test"]);
    let x_test = Mat::<2, 1>::from_storage(Store::from_array([[
        x0_test[0], x0_test[1],
    ]]));
    let u_test = Mat::<1, 1>::from_fn(|_, _| json_f64(&tut["u_test"]));
    let (x_dot, y_test) = sys_c.derivative(&x_test, &u_test);

    print_matrix("x_dot at [1.0, 0.5]^T", &x_dot);
    let y_val = y_test.get(0, 0).copied().unwrap_or(0.0);
    eprintln!("y at [1.0, 0.5]^T: {y_val:.6}");

    let dt = json_f64(&tut["Ts"]);
    let sys_d = sys_c.to_discrete_zoh(dt);

    eprintln!("\n--- Discrete-Time System (ZOH, Ts = {dt}s) ---");
    print_matrix("A_d", &sys_d.a());
    print_matrix("B_d", &sys_d.b());
    print_matrix("C_d", &sys_d.c());
    print_matrix("D_d", &sys_d.d());

    let (step_x1, step_x2, step_y) = run_traj(
        &sys_d,
        NUM_STEPS,
        vec2(&tut["step_x0"]),
        json_f64(&tut["step_u"]),
    );
    let (free_x1, free_x2, _) = run_traj(
        &sys_d,
        NUM_STEPS,
        vec2(&tut["free_x0"]),
        json_f64(&tut["free_u"]),
    );
    eprintln!("\n--- {NUM_STEPS}-Step Unit Step Trajectory ---");
    for k in [0, 1, 2, NUM_STEPS - 1] {
        eprintln!(
            "k={k}: x1={:.8} x2={:.8} y={:.8}",
            step_x1[k], step_x2[k], step_y[k]
        );
    }
    eprintln!("\n--- {NUM_STEPS}-Step Free Response (x0=[1.0, 0.5], u=0) ---");
    for k in [0, 1, 2, NUM_STEPS - 1] {
        eprintln!("k={k}: x1={:.8} x2={:.8}", free_x1[k], free_x2[k]);
    }
    for k in 0..3 {
        let mut ax = Mat::<2, 1>::zero();
        let mut bu = Mat::<2, 1>::zero();
        let x_k = Mat::<2, 1>::from_storage(Store::from_array([[
            step_x1[k], step_x2[k],
        ]]));
        let u_step = Mat::<1, 1>::from_fn(|_, _| json_f64(&tut["step_u"]));
        sys_d
            .a()
            .mul_into_with::<Blas, Const<1>, _, _>(&x_k, &mut ax);
        sys_d
            .b()
            .mul_into_with::<Blas, Const<1>, _, _>(&u_step, &mut bu);
        let x_pred: Mat<2, 1> = &ax + &bu;
        let (x_next, _) = sys_d.step(&x_k, &u_step);
        for i in 0..2 {
            let got = x_next.get(i, 0).copied().unwrap();
            let expected = x_pred.get(i, 0).copied().unwrap();
            assert!(
                (got - expected).abs() <= ABS_F64,
                "step {k} x[{i}]: {got} vs {expected}"
            );
        }
    }

    let t = Mat::<2, 2>::from_storage(storage_from_rows(&tut["T"]));
    let sys_transformed = sys_d.similarity_transform(&t).expect("similarity");
    eprintln!("\n--- Transformed System (T = [[1, 1], [0, 1]]) ---");
    print_matrix("A_tilde", &sys_transformed.a());
    print_matrix("B_tilde", &sys_transformed.b());
    print_matrix("C_tilde", &sys_transformed.c());

    let stiff = case_inputs(suite, "state_space.host.stiff_zoh");
    require_usize(stiff, "n_steps", STIFF_STEPS);
    eprintln!("\n--- Stiff plant A=diag(-200, -0.5), Ts=0.01 ---");
    let a_s = Mat::<2, 2>::from_storage(storage_from_rows(&stiff["A"]));
    let b_s = Mat::<2, 1>::from_storage(storage_from_rows(&stiff["B"]));
    let c_s = Mat::<1, 2>::from_storage(storage_from_rows(&stiff["C"]));
    let d_s = Mat::<1, 1>::from_storage(storage_from_rows(&stiff["D"]));
    let sys_s = ArrayStateSpace::continuous(a_s, b_s, c_s, d_s);
    let dt_s = json_f64(&stiff["Ts"]);
    let sys_sd = sys_s.to_discrete_zoh(dt_s);
    print_matrix("A_d stiff", &sys_sd.a());
    print_matrix("B_d stiff", &sys_sd.b());
    let (_, _, stiff_y) = run_traj(
        &sys_sd,
        STIFF_STEPS,
        vec2(&stiff["x0"]),
        json_f64(&stiff["u"]),
    );

    let zoh_iters = json_usize(&tut["zoh_iters"]).unwrap_or(20) as u32;
    let step_iters = json_usize(&tut["step_iters"]).unwrap_or(20) as u32;
    let zoh_ns = time_kernel(zoh_iters, || {
        sys_c
            .to_discrete_zoh(core::hint::black_box(dt))
            .a()
            .get(0, 0)
            .copied()
    });
    let step_ns = time_kernel(step_iters, || {
        let (_, _, y) = run_traj(
            &sys_d,
            NUM_STEPS,
            vec2(&tut["step_x0"]),
            json_f64(&tut["step_u"]),
        );
        y.last().copied()
    });
    eprintln!("ZOH min ns ({zoh_iters} iters): {zoh_ns}");
    eprintln!("step-loop min ns ({step_iters} iters): {step_ns}");

    let t_axis: Vec<f64> = (0..NUM_STEPS).map(|k| k as f64 * dt).collect();
    let t_stiff: Vec<f64> = (0..STIFF_STEPS).map(|k| k as f64 * dt_s).collect();
    let values = json!({
        "X_DOT": owned_to_rows(&x_dot),
        "Y_TEST": y_val,
        "AD": owned_to_rows(&sys_d.a()),
        "BD": owned_to_rows(&sys_d.b()),
        "STEP_X1": step_x1,
        "STEP_X2": step_x2,
        "STEP_Y": step_y,
        "FREE_X1": free_x1,
        "FREE_X2": free_x2,
        "A_TILDE": owned_to_rows(&sys_transformed.a()),
        "B_TILDE": owned_to_rows(&sys_transformed.b()),
        "C_TILDE": owned_to_rows(&sys_transformed.c()),
        "STIFF_AD": owned_to_rows(&sys_sd.a()),
        "STIFF_BD": owned_to_rows(&sys_sd.b()),
        "STIFF_Y": stiff_y,
    });
    let series = json!({
        "step_y": { "x": t_axis, "y": step_y },
        "stiff_y": { "x": t_stiff, "y": stiff_y },
        "free_x1": { "x": t_axis, "y": free_x1 },
        "free_x2": { "x": t_axis, "y": free_x2 },
    });
    let metrics = json!({});
    let timings = json!({
        "zoh": timing_entry(zoh_iters, zoh_ns),
        "step": timing_entry(step_iters, step_ns),
    });
    emit_stdout(&native_artifact(
        "state_space",
        "rust",
        values,
        series,
        metrics,
        timings,
    ));
}
