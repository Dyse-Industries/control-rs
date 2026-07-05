#![allow(
    missing_docs,
    clippy::unwrap_used,
    clippy::semicolon_if_nothing_returned,
    clippy::arithmetic_side_effects
)]

use control_rs::math::matrix::{
    LowerTriangular, Matrix, SquareMatrix, Symmetric, UpperTriangular,
};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

fn bench_matrix_addition(c: &mut Criterion) {
    let mut m1: SquareMatrix<f32, 4> = Matrix::new([[0.0; 4]; 4]);
    let m2: SquareMatrix<f32, 4> = Matrix::new([[1.0; 4]; 4]);

    c.bench_function("matrix_addition_nested_array", |b| {
        b.iter(|| {
            m1 += black_box(&m2);
        });
    });
}

fn bench_matrix_subtraction(c: &mut Criterion) {
    let mut m1: SquareMatrix<f32, 4> = Matrix::new([[1000.0; 4]; 4]);
    let m2: SquareMatrix<f32, 4> = Matrix::new([[1.0; 4]; 4]);

    c.bench_function("matrix_subtraction_nested_array", |b| {
        b.iter(|| {
            m1 -= black_box(&m2);
        });
    });
}

fn bench_symmetric_matrix_write(c: &mut Criterion) {
    let m: SquareMatrix<f32, 4> = Matrix::new([[0.0; 4]; 4]);
    let mut sym = Symmetric::new(m).unwrap();

    c.bench_function("symmetric_matrix_set", |b| {
        b.iter(|| {
            let _ = sym.set(black_box(0), black_box(3), black_box(5.0));
            let _ = sym.set(black_box(1), black_box(2), black_box(5.0));
        });
    });
}

fn bench_triangular_matrix_write(c: &mut Criterion) {
    let m1: SquareMatrix<f32, 4> = Matrix::new([[0.0; 4]; 4]);
    let mut ut = UpperTriangular::new(m1).unwrap();

    let m2: SquareMatrix<f32, 4> = Matrix::new([[0.0; 4]; 4]);
    let mut lt = LowerTriangular::new(m2).unwrap();

    c.bench_function("triangular_matrix_mut_write", |b| {
        b.iter(|| {
            *ut.get_mut(black_box(1), black_box(2)).unwrap() = black_box(5.0);
            *lt.get_mut(black_box(2), black_box(1)).unwrap() = black_box(5.0);
        });
    });
}

criterion_group!(
    benches,
    bench_matrix_addition,
    bench_matrix_subtraction,
    bench_symmetric_matrix_write,
    bench_triangular_matrix_write
);
criterion_main!(benches);
