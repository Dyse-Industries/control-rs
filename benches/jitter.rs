//! Criterion benchmarks for execution latency jitter and determinism.
//!
//! Evaluates timing variance and jitter distributions for:
//! 1. 32x32 Hilbert LU solve latency
//! 2. 8x8 EKF state covariance update recursion
//! 3. State-space discrete-time pendulum step response

#![allow(missing_docs)]

use std::hint::black_box;

use control_rs::math::num_types::{Const, Dim};
use control_rs::matrix::{LuDecomposition, Owned};
use control_rs::state_space::ArrayStateSpace;
use criterion::{Criterion, criterion_group, criterion_main};

/// Converts a matrix index to `f64` (exact for every index here).
fn index_f64(i: usize) -> f64 {
    f64::from(u32::try_from(i).unwrap_or(u32::MAX))
}

/// `a * b`, computed as the `&Matrix * &Matrix` operator does: a zeroed
/// output filled by the default BLAS backend.
fn matmul<const M: usize, const K: usize, const P: usize>(
    a: &Owned<f64, M, K>,
    b: &Owned<f64, K, P>,
) -> Owned<f64, M, P>
where
    Const<M>: Dim,
    Const<K>: Dim,
    Const<P>: Dim,
{
    let mut out = Owned::<f64, M, P>::zero();
    a.mul_into(b, &mut out);
    out
}

/// Element-wise `a + sign * b`, as the `+`/`-` operators compute it.
fn axpy<const R: usize, const C: usize>(
    a: &Owned<f64, R, C>,
    sign: f64,
    b: &Owned<f64, R, C>,
) -> Owned<f64, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    let mut out = Owned::<f64, R, C>::zero();
    for ((o, &x), &y) in out
        .as_mut_slice()
        .iter_mut()
        .zip(a.as_slice())
        .zip(b.as_slice())
    {
        *o = sign.mul_add(y, x);
    }
    out
}

fn bench_hilbert_solve_jitter(c: &mut Criterion) {
    const N: usize = 32;
    let h = Owned::<f64, N, N>::from_fn(|i, j| {
        1.0 / (index_f64(i) + index_f64(j) + 1.0)
    });
    let b = Owned::<f64, N, 1>::from_fn(|_, _| 1.0);
    let Ok(lu) = LuDecomposition::decompose(h) else {
        eprintln!("Hilbert LU decomposition failed; benchmark skipped");
        return;
    };

    c.bench_function("jitter/hilbert_32x32_solve", |bencher| {
        let mut x = b;
        bencher.iter(|| {
            let solved = lu.solve_mut(black_box(&mut x));
            black_box(solved.is_ok());
            black_box(x);
        });
    });
}

fn bench_ekf_update_jitter(c: &mut Criterion) {
    const N: usize = 8;
    let p_0 = Owned::<f64, N, N>::from_fn(|i, j| {
        let diff = index_f64(i) - index_f64(j);
        (-0.25 * diff * diff).exp() + if i == j { 0.1 } else { 0.0 }
    });

    let h = Owned::<f64, N, N>::identity();
    let k = Owned::<f64, N, N>::from_fn(|i, j| if i == j { 0.4 } else { 0.01 });
    let r = Owned::<f64, N, N>::from_fn(|i, j| if i == j { 0.05 } else { 0.0 });

    let kh = matmul(&k, &h);
    let eye = Owned::<f64, N, N>::identity();
    let i_minus_kh = axpy(&eye, -1.0, &kh);

    let k_t = k.transpose();
    let kr = matmul(&k, &r);
    let krk_t = matmul(&kr, &k_t);
    let i_minus_kh_t = i_minus_kh.transpose();

    let mut p_current = p_0;

    c.bench_function("jitter/ekf_8x8_covariance_update", |bencher| {
        bencher.iter(|| {
            let p_temp = matmul(&i_minus_kh, black_box(&p_current));
            let p_update = matmul(&p_temp, &i_minus_kh_t);
            p_current = axpy(&p_update, 1.0, &krk_t);
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

// Changes within 15 % are noise, not regressions (`cargo regression`).
criterion_group! {
    name = benches;
    config = Criterion::default().noise_threshold(0.15);
    targets = bench_hilbert_solve_jitter,
        bench_ekf_update_jitter,
        bench_step_response_jitter
}

criterion_main!(benches);
