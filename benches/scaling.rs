//! Criterion benchmarks for algorithmic scaling across numerical kernels.
//!
//! Evaluates asymptotic scaling across dimensions for matrix inversion ($O(N^3)$),
//! state-space discretization/controllability/observability, tensor contraction,
//! and polynomial evaluation.

#![allow(missing_docs)]

use std::hint::black_box;

use control_rs::math::num_types::{Const, Dim};
use control_rs::matrix::{LuDecomposition, Owned};
use control_rs::polynomial::ArrayPolynomial;
use control_rs::state_space::ArrayStateSpace;
use control_rs::tensor::ArrayTensor;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

macro_rules! bench_matrix_inversion_dim {
    ($group:expr, $($N:literal),+ $(,)?) => {
        $({
            let a = dominant_matrix::<$N>();
            $group.bench_with_input(
                BenchmarkId::new("dim", $N),
                &$N,
                |b, _| {
                    b.iter(|| {
                        LuDecomposition::decompose(black_box(a))
                            .and_then(|lu| lu.inverse())
                    });
                },
            );
        })+
    };
}

macro_rules! bench_state_space_dim {
    ($group:expr, $($N:literal),+ $(,)?) => {
        $({
            let a = Owned::<f64, $N, $N>::from_fn(|i, j| {
                if i == j {
                    -0.5 * (index_f64(i) + 1.0)
                } else {
                    0.1 / (index_f64(i) + index_f64(j) + 1.0)
                }
            });
            let b = Owned::<f64, $N, 1>::from_fn(|i, _| 1.0 / (index_f64(i) + 1.0));
            let c = Owned::<f64, 1, $N>::from_fn(|_, j| 1.0 / (index_f64(j) + 1.0));
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

macro_rules! bench_tensor_contract_dim {
    ($group:expr, $($N:literal),+ $(,)?) => {
        $({
            let a = ArrayTensor::<f32, $N, $N>::from_fn(|idx| {
                let (i, j) = index2(idx);
                i.mul_add(0.5, j * 0.3).sin() * 10.0
            });
            let b = ArrayTensor::<f32, $N, $N>::from_fn(|idx| {
                let (i, j) = index2(idx);
                i.mul_add(0.3, -(j * 0.4)).cos() * 5.0
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

macro_rules! bench_polynomial_eval_dim {
    ($group:expr, $($deg:literal),+ $(,)?) => {
        $({
            let coeffs: [f64; $deg + 1] = core::array::from_fn(|i| 1.0 / (index_f64(i) + 1.0));
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

/// Converts an index to `f64` (exact for every index here).
fn index_f64(i: usize) -> f64 {
    f64::from(u32::try_from(i).unwrap_or(u32::MAX))
}

/// Converts an index to `f32` (exact for every index here).
fn index_f32(i: usize) -> f32 {
    f32::from(u16::try_from(i).unwrap_or(u16::MAX))
}

/// Row and column of a rank-2 tensor index.
fn index2(idx: &[usize]) -> (f32, f32) {
    match *idx {
        [i, j, ..] => (index_f32(i), index_f32(j)),
        [i] => (index_f32(i), 0.0),
        [] => (0.0, 0.0),
    }
}

/// Diagonally dominant matrix generator for stable LU factorization.
fn dominant_matrix<const N: usize>() -> Owned<f64, N, N>
where
    Const<N>: Dim,
{
    Owned::<f64, N, N>::from_fn(|i, j| {
        if i == j {
            2.0 * (index_f64(i) + 1.0)
        } else {
            0.5 / (index_f64(i) + index_f64(j) + 1.0)
        }
    })
}

fn bench_matrix_inversion_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_inversion_scaling");
    bench_matrix_inversion_dim!(group, 2, 4, 8, 16, 32, 64);
    group.finish();
}

fn bench_state_space_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("state_space_scaling");
    bench_state_space_small(&mut group);
    bench_state_space_large(&mut group);
    group.finish();
}

/// Dimensions 2 to 32, whose systems fit comfortably in one stack frame.
fn bench_state_space_small(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
) {
    bench_state_space_dim!(group, 2, 4, 8, 16, 32);
}

/// Dimension 64, in its own stack frame.
fn bench_state_space_64(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
) {
    bench_state_space_dim!(group, 64);
}

/// Dimension 128, in its own stack frame.
fn bench_state_space_128(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
) {
    bench_state_space_dim!(group, 128);
}

/// Dimensions 64 and 128, one stack frame each.
fn bench_state_space_large(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
) {
    bench_state_space_64(group);
    bench_state_space_128(group);
}

fn bench_tensor_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_contraction_scaling");
    bench_tensor_contract_dim!(group, 4, 8, 16, 32, 64);
    group.finish();
}

fn bench_polynomial_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("polynomial_evaluation_scaling");
    bench_polynomial_eval_dim!(group, 2, 4, 8, 16, 32, 48);
    group.finish();
}

// Changes within 15 % are noise, not regressions (`cargo regression`).
criterion_group! {
    name = benches;
    config = Criterion::default().noise_threshold(0.15);
    targets = bench_matrix_inversion_scaling,
        bench_state_space_scaling,
        bench_tensor_scaling,
        bench_polynomial_scaling
}

criterion_main!(benches);
