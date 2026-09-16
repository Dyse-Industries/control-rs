//! Host latency benches for classical-tools kernels.
//!
//! Criterion measures steady-state latency of the analysis and compensator
//! kernels exercised by the `dc-motor` and `buck-converter` validation
//! suites. Timing is not a CI fail gate.
//!
//! Run with `cargo bench --bench classical_tools`.
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

use criterion::{Criterion, criterion_group, criterion_main};

use control_rs::classical_tools::compensators::lead;
use control_rs::classical_tools::margins::stability_margins;
use control_rs::classical_tools::pid::Pid;
use control_rs::classical_tools::realization::DirectForm2T;
use control_rs::classical_tools::routh::stability;
use control_rs::polynomial::ArrayPolynomial;
use control_rs::transfer_function::ArrayTransferFunction;

/// Routh-Hurwitz on $(s + 1)^4$, five ascending coefficients.
fn routh(c: &mut Criterion) {
    let poly =
        ArrayPolynomial::<f64, 5>::from_coefficients([1.0, 4.0, 6.0, 4.0, 1.0]);
    c.bench_function("classical_tools/routh_stability_deg4", |b| {
        b.iter(|| {
            stability(black_box(&poly), black_box(1e-12))
                .expect("Routh stability (degree 4)")
        });
    });
}

/// Gain and phase margins over a fixed 256-point frequency grid.
fn margins(c: &mut Criterion) {
    let plant =
        ArrayTransferFunction::<f64, 1, 3>::continuous([1.0], [1.0, 0.4, 1.0]);
    let omegas: [f64; 256] =
        core::array::from_fn(|i| 10f64.powf(-2.0 + 4.0 * (i as f64) / 255.0));
    c.bench_function("classical_tools/stability_margins_256", |b| {
        b.iter(|| stability_margins(black_box(&plant), black_box(&omegas)));
    });
}

/// Lead network synthesis and its discrete realization.
fn compensators(c: &mut Criterion) {
    let mut group = c.benchmark_group("classical_tools");

    group.bench_function("lead_synthesis", |b| {
        b.iter(|| {
            lead(black_box(1.0_f64), black_box(1.0), black_box(0.1))
                .expect("lead alpha < 1")
        });
    });

    let mut df2t = DirectForm2T::<f64, 1>::new(1.0, [0.0], [0.0]);
    group.bench_function("direct_form_2t_update", |b| {
        b.iter(|| df2t.update(black_box(1.0)));
    });

    let mut pid = Pid::new(1.0, 0.1, 0.01, 0.002, -12.0, 12.0, 1.0);
    group.bench_function("pid_step", |b| {
        b.iter(|| pid.step(black_box(1.0), black_box(0.0), black_box(5e-4)));
    });

    group.finish();
}

criterion_group!(benches, routh, margins, compensators);
criterion_main!(benches);
