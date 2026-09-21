//! Criterion benchmarks for algorithmic scaling across numerical kernels.
//!
//! Evaluates asymptotic scaling across dimensions for matrix inversion ($O(N^3)$),
//! state-space discretization/controllability/observability, tensor contraction,
//! and polynomial evaluation.

#![allow(
    missing_docs,
    clippy::arbitrary_source_item_ordering,
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::large_stack_frames,
    clippy::suboptimal_flops,
    clippy::too_many_lines
)]

use std::hint::black_box;

use control_rs::math::num_types::{Const, Dim};
use control_rs::matrix::{LuDecomposition, Owned};
use control_rs::polynomial::ArrayPolynomial;
use control_rs::state_space::ArrayStateSpace;
use control_rs::tensor::ArrayTensor;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

/// Diagonally dominant matrix generator for stable LU factorization.
fn dominant_matrix<const N: usize>() -> Owned<f64, N, N>
where
    Const<N>: Dim,
{
    Owned::<f64, N, N>::from_fn(|i, j| {
        if i == j {
            2.0 * ((i + 1) as f64)
        } else {
            0.5 / ((i + j + 1) as f64)
        }
    })
}

macro_rules! bench_matrix_inversion_dim {
    ($group:expr, $($N:literal),+ $(,)?) => {
        $({
            let a = dominant_matrix::<$N>();
            $group.bench_with_input(
                BenchmarkId::new("dim", $N),
                &$N,
                |b, _| {
                    b.iter(|| {
                        let lu = LuDecomposition::decompose(black_box(a))
                            .expect("LU decompose failed");
                        lu.inverse().expect("LU inverse failed")
                    });
                },
            );
        })+
    };
}

fn bench_matrix_inversion_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_inversion_scaling");
    bench_matrix_inversion_dim!(group, 2, 4, 8, 16, 32, 64);
    group.finish();
}

macro_rules! bench_state_space_dim {
    ($group:expr, $($N:literal),+ $(,)?) => {
        $({
            let a = Owned::<f64, $N, $N>::from_fn(|i, j| {
                if i == j {
                    -0.5 * ((i + 1) as f64)
                } else {
                    0.1 / ((i + j + 1) as f64)
                }
            });
            let b = Owned::<f64, $N, 1>::from_fn(|i, _| 1.0 / ((i + 1) as f64));
            let c = Owned::<f64, 1, $N>::from_fn(|_, j| 1.0 / ((j + 1) as f64));
            let d = Owned::<f64, 1, 1>::scalar(0.0);
            let sys_c = ArrayStateSpace::continuous(a, b, c, d);

            $group.bench_with_input(
                BenchmarkId::new("zoh_dim", $N),
                &$N,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(&sys_c).to_discrete_zoh(black_box(0.05))
                    });
                },
            );

            $group.bench_with_input(
                BenchmarkId::new("ctrb_dim", $N),
                &$N,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(&sys_c).controllability_matrix::<$N>()
                    });
                },
            );

            $group.bench_with_input(
                BenchmarkId::new("obsv_dim", $N),
                &$N,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(&sys_c).observability_matrix::<$N>()
                    });
                },
            );
        })+
    };
}

fn bench_state_space_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_space_scaling");
    bench_state_space_dim!(group, 2, 4, 8, 16, 32, 64, 128);
    group.finish();
}

macro_rules! bench_tensor_contract_dim {
    ($group:expr, $($N:literal),+ $(,)?) => {
        $({
            let a = ArrayTensor::<f32, $N, $N>::from_fn(|idx| {
                (idx[0] as f32 * 0.5 + idx[1] as f32 * 0.3).sin() * 10.0
            });
            let b = ArrayTensor::<f32, $N, $N>::from_fn(|idx| {
                (idx[0] as f32 * 0.3 - idx[1] as f32 * 0.4).cos() * 5.0
            });
            let mut out = ArrayTensor::<f32, $N, $N>::zero();

            $group.bench_with_input(
                BenchmarkId::new("dim", $N),
                &$N,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(&a).contract_into(black_box(&b), black_box(&mut out));
                    });
                },
            );
        })+
    };
}

fn bench_tensor_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_contraction_scaling");
    bench_tensor_contract_dim!(group, 4, 8, 16, 32, 64);
    group.finish();
}

macro_rules! bench_polynomial_eval_dim {
    ($group:expr, $($deg:literal),+ $(,)?) => {
        $({
            let coeffs: [f64; $deg + 1] = core::array::from_fn(|i| 1.0 / ((i + 1) as f64));
            let poly = ArrayPolynomial::<f64, { $deg + 1 }>::from_coefficients(coeffs);
            $group.bench_with_input(
                BenchmarkId::new("degree", $deg),
                &$deg,
                |bencher, _| {
                    bencher.iter(|| {
                        black_box(&poly).evaluate(black_box(0.5))
                    });
                },
            );
        })+
    };
}

fn bench_polynomial_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("polynomial_evaluation_scaling");
    bench_polynomial_eval_dim!(group, 2, 4, 8, 16, 32, 48);
    group.finish();
}

criterion_group!(
    benches,
    bench_matrix_inversion_scaling,
    bench_state_space_scaling,
    bench_tensor_scaling,
    bench_polynomial_scaling
);
criterion_main!(benches);
