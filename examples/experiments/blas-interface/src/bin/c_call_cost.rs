//! Runtime cost of crossing into C from the storage/subprogram interface,
//! and comparing storage-subprogram design variants.
//!
//! Times:
//! - gemv_dyn: Rust runtime fields with slice indexing.
//! - gemv_storage_trait_4: Design A (generic storage via trait method).
//! - gemv_generic_fn_4: Design C (generic storage function directly).
//! - gemv_nested_4: Design B (statically-sized nested arrays - proposed).
//! - gemv_c: C FFI.
//!
//! Requires the "cffi" feature: `cargo run --bin c_call_cost --features cffi --release`.

use blas_interface::{
    gemv_c, gemv_dyn, gemv_generic_fn_4, gemv_nested_4, gemv_storage_trait_4, Dense4, DynDense,
    MatrixLayout,
};
use std::time::Instant;

const ITERS: u32 = 1_000_000;

fn max_abs_diff(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(l, r)| (l - r).abs())
        .fold(0.0f32, f32::max)
}

fn main() {
    let buf: [f32; 16] = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let x = [1.0f32, 1.0, 1.0, 1.0];
    let s = DynDense {
        buf: buf.as_slice(),
        rows: 4,
        cols: 4,
        lda: 4,
        order: MatrixLayout::ColMajor,
    };
    let dense4 = Dense4(buf.as_slice());
    let nested_buf: [[f32; 4]; 4] = [
        [buf[0], buf[1], buf[2], buf[3]],
        [buf[4], buf[5], buf[6], buf[7]],
        [buf[8], buf[9], buf[10], buf[11]],
        [buf[12], buf[13], buf[14], buf[15]],
    ];

    // Correctness checks
    let mut y_ref = [0.0f32; 4];
    gemv_dyn(&s, &x, &mut y_ref);

    let mut y_c = [0.0f32; 4];
    gemv_c(&s, &x, &mut y_c);
    assert!(max_abs_diff(&y_ref, &y_c) < 1e-6);

    let mut y_trait = [0.0f32; 4];
    gemv_storage_trait_4(&dense4, &x, &mut y_trait);
    assert!(max_abs_diff(&y_ref, &y_trait) < 1e-6);

    let mut y_fn = [0.0f32; 4];
    gemv_generic_fn_4(&dense4, &x, &mut y_fn);
    assert!(max_abs_diff(&y_ref, &y_fn) < 1e-6);

    let mut y_nested = [0.0f32; 4];
    gemv_nested_4(&nested_buf, &x, &mut y_nested);
    assert!(max_abs_diff(&y_ref, &y_nested) < 1e-6);

    println!("All variants verified correct.");

    // Benchmarking
    let mut y = [0.0f32; 4];
    let start = Instant::now();
    for _ in 0..ITERS {
        gemv_dyn(&s, &x, &mut y);
        std::hint::black_box(&y);
    }
    let rust_elapsed = start.elapsed();

    let start = Instant::now();
    for _ in 0..ITERS {
        gemv_storage_trait_4(&dense4, &x, &mut y);
        std::hint::black_box(&y);
    }
    let trait_elapsed = start.elapsed();

    let start = Instant::now();
    for _ in 0..ITERS {
        gemv_generic_fn_4(&dense4, &x, &mut y);
        std::hint::black_box(&y);
    }
    let fn_elapsed = start.elapsed();

    let start = Instant::now();
    for _ in 0..ITERS {
        gemv_nested_4(&nested_buf, &x, &mut y);
        std::hint::black_box(&y);
    }
    let nested_elapsed = start.elapsed();

    let start = Instant::now();
    for _ in 0..ITERS {
        gemv_c(&s, &x, &mut y);
        std::hint::black_box(&y);
    }
    let c_elapsed = start.elapsed();

    let rust_ns = rust_elapsed.as_nanos() as f64 / f64::from(ITERS);
    let trait_ns = trait_elapsed.as_nanos() as f64 / f64::from(ITERS);
    let fn_ns = fn_elapsed.as_nanos() as f64 / f64::from(ITERS);
    let nested_ns = nested_elapsed.as_nanos() as f64 / f64::from(ITERS);
    let c_ns = c_elapsed.as_nanos() as f64 / f64::from(ITERS);

    println!();
    println!("{ITERS} calls, 4x4 f32 GEMV:");
    println!("  gemv_dyn (Rust Runtime fields)           {rust_ns:>8.2} ns/call");
    println!("  gemv_storage_trait_4 (Design A: Trait)   {trait_ns:>8.2} ns/call");
    println!("  gemv_generic_fn_4 (Design C: Fn)         {fn_ns:>8.2} ns/call");
    println!("  gemv_nested_4 (Design B: Proposed)       {nested_ns:>8.2} ns/call");
    println!("  gemv_c (C FFI)                           {c_ns:>8.2} ns/call");
}
