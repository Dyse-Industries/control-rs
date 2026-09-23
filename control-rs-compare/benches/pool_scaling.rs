//! Criterion benchmark for intra-dataset parallel chunked comparison scaling.
//!
//! Evaluates speedup and asymptotic scaling across worker thread counts
//! `j = 1..=n_proc` on massive floating-point datasets.

#![allow(missing_docs)]

use std::num::NonZero;

use control_rs_compare::compare::{
    ToleranceSpec, compare_float_arrays_parallel,
};
use criterion::{
    BenchmarkId, Criterion, Throughput, black_box, criterion_group,
    criterion_main,
};

fn bench_pool_asymptotic_scaling(c: &mut Criterion) {
    let n_proc = std::thread::available_parallelism().map_or(4, NonZero::get);

    let tol = ToleranceSpec {
        method: "abs".to_string(),
        bound: 1e-3,
    };

    // Array sizes ranging from chunking threshold (64K) to massive (4M elements)
    let dataset_sizes: [usize; 4] = [65_536, 262_144, 1_048_576, 4_194_304];

    for size in dataset_sizes {
        let mut group =
            c.benchmark_group(format!("chunked_pool_scaling_{size}_elements"));
        let bytes = size.saturating_mul(std::mem::size_of::<f64>());
        group.throughput(Throughput::Bytes(
            u64::try_from(bytes).unwrap_or(u64::MAX),
        ));

        // Generate synthetic floating-point datasets
        let oracle: Vec<f64> = (0..u32::try_from(size).unwrap_or(u32::MAX))
            .map(|i| (f64::from(i) * 0.001).sin())
            .collect();
        let peer: Vec<f64> = oracle.iter().map(|&v| v + 1e-6).collect();

        // Benchmark scaling across j = 1..=n_proc threads
        for threads in 1..=n_proc {
            group.bench_with_input(
                BenchmarkId::new("threads", threads),
                &threads,
                |b, &threads| {
                    b.iter(|| {
                        compare_float_arrays_parallel(
                            black_box(&oracle),
                            black_box(&peer),
                            black_box(&tol),
                            threads,
                        )
                    });
                },
            );
        }
        group.finish();
    }
}

criterion_group!(benches, bench_pool_asymptotic_scaling);
criterion_main!(benches);
