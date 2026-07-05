#![allow(
    missing_docs,
    clippy::unwrap_used,
    clippy::semicolon_if_nothing_returned,
    clippy::arithmetic_side_effects
)]

use control_rs::math::num_types::Const;
use control_rs::tensor::Tensor;
use criterion::{Criterion, black_box, criterion_group, criterion_main};

pub struct TensorBenchmarkSuite;

impl TensorBenchmarkSuite {
    /// Runs the tensor benchmark suite.
    ///
    /// # Clippy Allow explanation
    /// We allow `clippy::missing_panics_doc` here because this is a benchmark suite
    /// where runtime checks are expected to be verified by benchmark harness assertions.
    #[allow(clippy::missing_panics_doc)]
    pub fn run(c: &mut Criterion) {
        // --- Successful Cases ---

        // 1. Tensor Addition
        let mut t1 = Tensor::<f32, 8, (Const<2>, Const<2>, Const<2>)>::new([
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        ]);
        let t2 = Tensor::<f32, 8, (Const<2>, Const<2>, Const<2>)>::new([
            10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,
        ]);
        c.bench_function("tensor_addition_success", |b| {
            b.iter(|| {
                t1 += black_box(t2);
            });
        });

        // 2. Tensor Scaling
        c.bench_function("tensor_scaling_success", |b| {
            b.iter(|| {
                t1 *= black_box(2.0);
            });
        });

        // 3. Coordinate Indexing (Successful)
        c.bench_function("tensor_indexing_success", |b| {
            b.iter(|| {
                let coords = [black_box(1), black_box(0), black_box(1)];
                let val = t1.get(&coords);
                debug_assert!(val.is_some());
            });
        });

        // --- Failing Cases ---

        // 4. Indexing Out of Bounds (Failure)
        c.bench_function("tensor_indexing_failure_out_of_bounds", |b| {
            b.iter(|| {
                let coords = [black_box(2), black_box(0), black_box(0)];
                let val = t1.get(&coords);
                debug_assert!(val.is_none());
            });
        });

        // 5. Indexing Dimension Mismatch (Failure)
        c.bench_function("tensor_indexing_failure_dimension_mismatch", |b| {
            b.iter(|| {
                let coords = [black_box(0), black_box(0)];
                let val = t1.get(&coords);
                debug_assert!(val.is_none());
            });
        });
    }
}

fn run_suite(c: &mut Criterion) {
    TensorBenchmarkSuite::run(c);
}

criterion_group!(benches, run_suite);
criterion_main!(benches);
