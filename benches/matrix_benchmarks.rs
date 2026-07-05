#![allow(
    missing_docs,
    clippy::unwrap_used,
    clippy::semicolon_if_nothing_returned,
    clippy::arithmetic_side_effects
)]

use control_rs::matrix::{Matrix, SquareMatrix, Symmetric, UpperTriangular};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

pub struct MatrixBenchmarkSuite;

impl MatrixBenchmarkSuite {
    /// Runs the matrix benchmark suite.
    ///
    /// # Clippy Allow explanation
    /// We allow `clippy::missing_panics_doc` here because this is a benchmark entry point,
    /// and the nested panics/unwraps are intended to enforce setup correctness of benchmarks.
    #[allow(clippy::missing_panics_doc)]
    pub fn run(c: &mut Criterion) {
        // --- Successful Cases ---

        // 1. Matrix Addition
        let mut m1: SquareMatrix<f32, 4> = Matrix::new([[0.0; 4]; 4]);
        let m2: SquareMatrix<f32, 4> = Matrix::new([[1.0; 4]; 4]);
        c.bench_function("matrix_addition_success", |b| {
            b.iter(|| {
                m1 += black_box(&m2);
            });
        });

        // 2. Matrix Subtraction
        let mut m3: SquareMatrix<f32, 4> = Matrix::new([[100.0; 4]; 4]);
        let m4: SquareMatrix<f32, 4> = Matrix::new([[1.0; 4]; 4]);
        c.bench_function("matrix_subtraction_success", |b| {
            b.iter(|| {
                m3 -= black_box(&m4);
            });
        });

        // 3. Symmetric Matrix set (Successful)
        let m_sym = Matrix::new([[0.0; 4]; 4]);
        let mut sym = Symmetric::new(m_sym).unwrap();
        c.bench_function("symmetric_matrix_set_success", |b| {
            b.iter(|| {
                let _ = sym.set(black_box(1), black_box(2), black_box(5.0));
            });
        });

        // 4. Triangular Matrix Mut Write (Successful)
        let m_ut = Matrix::new([[0.0; 4]; 4]);
        let mut ut = UpperTriangular::new(m_ut).unwrap();
        c.bench_function("triangular_matrix_mut_write_success", |b| {
            b.iter(|| {
                if let Some(val) = ut.get_mut(black_box(1), black_box(2)) {
                    *val = black_box(5.0);
                }
            });
        });

        // --- Failing Cases ---

        // 5. Symmetric Matrix Set Out of Bounds (Failure)
        c.bench_function("symmetric_matrix_set_failure", |b| {
            b.iter(|| {
                let res = sym.set(black_box(4), black_box(0), black_box(5.0));
                debug_assert!(res.is_err());
            });
        });

        // 6. Triangular Matrix Mut Write in forbidden region (Failure)
        c.bench_function("triangular_matrix_mut_write_failure", |b| {
            b.iter(|| {
                let res = ut.get_mut(black_box(2), black_box(1));
                debug_assert!(res.is_none());
            });
        });
    }
}

fn run_suite(c: &mut Criterion) {
    MatrixBenchmarkSuite::run(c);
}

criterion_group!(benches, run_suite);
criterion_main!(benches);
