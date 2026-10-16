//! Criterion benchmarks for the per-sample cost of the classical control
//! execution layer (NFR-4).
//!
//! Each benchmark advances one realization by one sample:
//! 1. Discrete `Pid` step with output limits and clamping
//! 2. Direct Form I and Transposed Direct Form II second-order sections
//! 3. A four-section Direct Form I cascade
//! 4. An order-4 `DirectForm2T`

#![allow(missing_docs)]

use std::hint::black_box;

use control_rs::classical_control::{
    AntiWindup, BiquadCascade, Df1, Df2t, DirectForm2T, Pid, PidParams,
    SectionCoefficients,
};
use criterion::{Criterion, criterion_group, criterion_main};

/// A stable, lightly damped low-pass section.
const SECTION: SectionCoefficients<f64> = SectionCoefficients {
    b0: 0.067,
    b1: 0.134,
    b2: 0.067,
    a1: -1.143,
    a2: 0.413,
};

fn bench_pid_step(c: &mut Criterion) {
    let Ok(mut pid) = Pid::new(PidParams {
        kp: 1.5,
        ki: 0.8,
        kd: 0.4,
        tf: 0.05,
        h: 0.001,
        u_min: -1.0,
        u_max: 1.0,
        anti_windup: AntiWindup::Clamping,
    }) else {
        return;
    };
    let mut y = 0.0_f64;
    c.bench_function("classical_control/pid_step", |b| {
        b.iter(|| {
            let u = pid.step(black_box(0.8), black_box(y));
            y = 0.001f64.mul_add(u - y, y);
            black_box(u)
        });
    });
}

fn bench_sections(c: &mut Criterion) {
    let mut df1 = Df1::from(SECTION);
    c.bench_function("classical_control/df1_update", |b| {
        use control_rs::classical_control::Section;
        b.iter(|| black_box(df1.update(black_box(1.0))));
    });
    let mut df2t = Df2t::from(SECTION);
    c.bench_function("classical_control/df2t_update", |b| {
        use control_rs::classical_control::Section;
        b.iter(|| black_box(df2t.update(black_box(1.0))));
    });
}

fn bench_cascades(c: &mut Criterion) {
    let mut cascade =
        BiquadCascade::<Df1<f64>, 4>::from_coefficients(&[SECTION; 4]);
    c.bench_function("classical_control/df1_cascade_4_update", |b| {
        b.iter(|| black_box(cascade.update(black_box(1.0))));
    });
    let mut direct = DirectForm2T::<f64, 4>::new(
        0.01,
        [0.02, 0.02, 0.01, 0.005],
        [-2.5, 2.4, -1.1, 0.2],
    );
    c.bench_function("classical_control/direct_form_2t_order_4_update", |b| {
        b.iter(|| black_box(direct.update(black_box(1.0))));
    });
}

criterion_group!(benches, bench_pid_step, bench_sections, bench_cascades);
criterion_main!(benches);
