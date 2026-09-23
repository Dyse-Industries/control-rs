//! Criterion benchmarks for execution latency jitter and determinism.
//!
//! Evaluates timing variance and jitter distributions for:
//! 1. 32x32 Hilbert LU solve latency
//! 2. 8x8 EKF state covariance update recursion
//! 3. State-space discrete-time pendulum step response

#![allow(
    missing_docs,
    clippy::arbitrary_source_item_ordering,
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::suboptimal_flops,
    clippy::too_many_lines
)]

use std::hint::black_box;

use control_rs::matrix::{LuDecomposition, Owned};
use control_rs::state_space::ArrayStateSpace;
use criterion::{Criterion, criterion_group, criterion_main};

fn bench_hilbert_solve_jitter(c: &mut Criterion) {
    const N: usize = 32;
    let h = Owned::<f64, N, N>::from_fn(|i, j| 1.0 / ((i + j + 1) as f64));
    let b = Owned::<f64, N, 1>::from_fn(|_, _| 1.0);
    let lu =
        LuDecomposition::decompose(h).expect("Hilbert LU decomposition failed");

    c.bench_function("jitter/hilbert_32x32_solve", |bencher| {
        let mut x = b;
        bencher.iter(|| {
            lu.solve_mut(black_box(&mut x))
                .expect("Hilbert solve failed");
            black_box(x);
        });
    });
}

fn bench_ekf_update_jitter(c: &mut Criterion) {
    const N: usize = 8;
    let p_0 = Owned::<f64, N, N>::from_fn(|i, j| {
        let diff = (i as f64) - (j as f64);
        (-0.25 * diff * diff).exp() + if i == j { 0.1 } else { 0.0 }
    });

    let h = Owned::<f64, N, N>::identity();
    let k = Owned::<f64, N, N>::from_fn(|i, j| if i == j { 0.4 } else { 0.01 });
    let r = Owned::<f64, N, N>::from_fn(|i, j| if i == j { 0.05 } else { 0.0 });

    let kh = &k * &h;
    let eye = Owned::<f64, N, N>::identity();
    let i_minus_kh = &eye - &kh;

    let k_t = k.transpose();
    let kr = &k * &r;
    let krk_t = &kr * &k_t;
    let i_minus_kh_t = i_minus_kh.transpose();

    let mut p_current = p_0;

    c.bench_function("jitter/ekf_8x8_covariance_update", |bencher| {
        bencher.iter(|| {
            let p_temp = &i_minus_kh * black_box(&p_current);
            let p_update1 = &p_temp * &i_minus_kh_t;
            p_current = &p_update1 + &krk_t;
            black_box(&p_current);
        });
    });
}

fn bench_step_response_jitter(c: &mut Criterion) {
    // Inverted pendulum linearized discrete-time model
    let a = Owned::<f64, 2, 2>::from_row_arrays([[1.0, 0.05], [-0.1, 0.96]]);
    let b = Owned::<f64, 2, 1>::from_row_arrays([[0.0], [0.05]]);
    let c_mat = Owned::<f64, 1, 2>::from_row_arrays([[1.0, 0.0]]);
    let d = Owned::<f64, 1, 1>::scalar(0.0);

    let sys_d = ArrayStateSpace::continuous(a, b, c_mat, d);
    let mut x_k = Owned::<f64, 2, 1>::zero();
    let u_k = Owned::<f64, 1, 1>::scalar(1.0);

    c.bench_function("jitter/state_space_step_response", |bencher| {
        bencher.iter(|| {
            let (x_next, y_k) = sys_d.step(black_box(&x_k), black_box(&u_k));
            x_k = x_next;
            black_box(y_k);
        });
    });
}

criterion_group!(
    benches,
    bench_hilbert_solve_jitter,
    bench_ekf_update_jitter,
    bench_step_response_jitter
);
criterion_main!(benches);
