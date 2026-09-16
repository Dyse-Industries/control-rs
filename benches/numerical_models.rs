//! Host latency benches for numerical-model kernels.
//!
//! Criterion measures steady-state latency of the same kernels the
//! `control-rs-validation` suites check for numerical agreement. Timing is not
//! a CI fail gate: these benches report, they do not assert.
//!
//! Run with `cargo bench --bench numerical_models`.
#![allow(
    missing_docs,
    clippy::arbitrary_source_item_ordering,
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::suboptimal_flops,
    clippy::too_many_lines
)]

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

use control_rs::math::num_types::{Const, Dim};
use control_rs::matrix::{LuDecomposition, Owned, Symmetric};
use control_rs::polynomial::ArrayPolynomial;
use control_rs::state_space::ArrayStateSpace;
use control_rs::tensor::ArrayTensor;
use control_rs::transfer_function::ArrayTransferFunction;

/// Diagonally dominant fill shared by the GEMM, inverse and scaling kernels.
fn dominant<const N: usize>() -> Owned<f64, N, N>
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

/// Hilbert matrix $H_{ij} = 1 / (i + j + 1)$, ill-conditioned by construction.
fn hilbert<const N: usize>() -> Owned<f64, N, N>
where
    Const<N>: Dim,
{
    Owned::from_fn(|i, j| 1.0 / ((i + j + 1) as f64))
}

fn matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix");

    let a64 = dominant::<64>();
    group.bench_function("gemm_64", |b| {
        b.iter(|| black_box(&a64) * black_box(&a64));
    });

    let a16 = Owned::<f64, 16, 16>::from_fn(|i, j| {
        if i == j {
            5.0 + i as f64
        } else {
            0.2 * ((i + j + 1) as f64)
        }
    });
    let b16 = Owned::<f64, 16, 1>::from_fn(|_, _| 1.0);
    group.bench_function("lu_16_decompose_solve", |b| {
        b.iter(|| {
            let lu = LuDecomposition::decompose(black_box(a16))
                .expect("LU decompose n=16");
            let mut x = b16;
            lu.solve_mut(&mut x).expect("LU solve n=16");
            x
        });
    });

    let a8 = dominant::<8>();
    group.bench_function("inverse_8", |b| {
        b.iter(|| {
            let lu = LuDecomposition::decompose(black_box(a8))
                .expect("LU decompose n=8");
            lu.inverse().expect("LU inverse n=8")
        });
    });

    let spd = Symmetric::<f64, 16>::from_owned(Owned::<f64, 16, 16>::from_fn(
        |i, j| {
            if i == j {
                10.0 + (i as f64)
            } else {
                1.0 / ((i + j + 2) as f64)
            }
        },
    ))
    .expect("symmetric construction");
    let rhs16 = Owned::<f64, 16, 1>::from_fn(|_, _| 1.0);
    group.bench_function("cholesky_16_solve", |b| {
        b.iter(|| {
            let chol = black_box(spd).into_cholesky().expect("Cholesky n=16");
            let mut x = rhs16;
            chol.solve_mut(&mut x).expect("Cholesky solve n=16");
            x
        });
    });

    let a_qr = Owned::<f64, 16, 16>::from_fn(|i, j| {
        if i == j {
            3.0 + (i as f64)
        } else {
            0.1 * ((i + j) as f64)
        }
    });
    group.bench_function("qr_16_decompose", |b| {
        b.iter(|| {
            let mut q = Owned::<f64, 16, 16>::zero();
            let mut a = black_box(a_qr);
            a.qr_decompose_mut(&mut q);
            (a, q)
        });
    });

    // Determinism probe: repeated solves of a fixed 32x32 Hilbert
    // factorization. Criterion's sample distribution replaces the
    // hand-rolled jitter histogram the validation suite used to emit.
    let h32 = hilbert::<32>();
    let lu32 = LuDecomposition::decompose(h32).expect("Hilbert LU n=32");
    let rhs32 = Owned::<f64, 32, 1>::from_fn(|_, _| 1.0);
    group.bench_function("hilbert_32_solve", |b| {
        b.iter(|| {
            let mut x = rhs32;
            lu32.solve_mut(&mut x).expect("Hilbert solve n=32");
            x
        });
    });

    group.finish();
}

/// Registers one `decompose + inverse` bench for a single compile-time size.
macro_rules! inversion_scaling {
    ($group:expr, $($n:literal),+ $(,)?) => {
        $({
            let a = dominant::<$n>();
            $group.bench_with_input(
                BenchmarkId::from_parameter($n),
                &a,
                |b, a| {
                    b.iter(|| {
                        let lu = LuDecomposition::decompose(black_box(*a))
                            .expect("LU decompose");
                        lu.inverse().expect("LU inverse")
                    });
                },
            );
        })+
    };
}

/// $O(N^3)$ scaling sweep of LU inversion.
fn matrix_inversion_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_inversion_scaling");
    inversion_scaling!(group, 2, 4, 8, 16, 32, 64);
    group.finish();
}

fn polynomial(c: &mut Criterion) {
    let mut group = c.benchmark_group("polynomial");

    let coeffs: [f64; 17] = core::array::from_fn(|i| 1.0 / ((i + 1) as f64));
    let poly = ArrayPolynomial::<f64, 17>::from_coefficients(coeffs);
    group.bench_function("horner_deg16", |b| {
        b.iter(|| poly.evaluate(black_box(0.5)));
    });

    let xs: [f64; 128] =
        core::array::from_fn(|i| -1.0 + 2.0 * (i as f64) / 127.0);
    group.bench_function("horner_deg16_batch_128", |b| {
        b.iter(|| {
            let mut acc = 0.0;
            for &xi in &xs {
                acc += poly.evaluate(black_box(xi));
            }
            acc
        });
    });

    group.finish();
}

fn state_space(c: &mut Criterion) {
    let sys = ArrayStateSpace::<f64, 2, 1, 1>::continuous(
        [[0.0, 1.0], [-4.0, -0.8]],
        [[0.0], [1.0]],
        [[1.0, 0.0]],
        [[0.0]],
    )
    .to_discrete_zoh(0.05);
    let u = Owned::<f64, 1, 1>::scalar(0.0);
    let x0 = Owned::<f64, 2, 1>::from_column([0.1, 0.0]);

    c.bench_function("state_space/step_2state", |b| {
        b.iter(|| sys.step(black_box(&x0), black_box(&u)));
    });
}

fn transfer_function(c: &mut Criterion) {
    let mut group = c.benchmark_group("transfer_function");
    let tf =
        ArrayTransferFunction::<f64, 1, 3>::continuous([1.0], [1.0, 0.2, 1.0]);

    group.bench_function("bode_point", |b| {
        b.iter(|| tf.bode_point(black_box(1.0)));
    });

    let omegas: [f64; 1000] =
        core::array::from_fn(|i| 10f64.powf(-2.0 + 4.0 * (i as f64) / 999.0));
    group.bench_function("bode_sweep_1000", |b| {
        b.iter(|| {
            for &w in &omegas {
                black_box(tf.bode_point(black_box(w)));
            }
        });
    });

    group.finish();
}

fn tensor(c: &mut Criterion) {
    let grid = ArrayTensor::<f64, 2, 2>::from_cols([[1.0, 3.0], [2.0, 4.0]]);
    c.bench_function("tensor/interpolate_2x2", |b| {
        b.iter(|| grid.interpolate(black_box(&[0.5, 0.5])));
    });
}

criterion_group!(
    benches,
    matrix,
    matrix_inversion_scaling,
    polynomial,
    state_space,
    transfer_function,
    tensor
);
criterion_main!(benches);
