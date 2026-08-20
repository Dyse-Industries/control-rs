//! Cross-crate correctness check for the `blas_interface` codegen variants.
//!
//! This binary is a separate compilation unit from `blas_interface`, so
//! calling its `gemv_*` entry points here does not defeat the opaque-boundary
//! premise the way calling them from within the same crate would: `gemv_*`
//! are plain (non-generic, non-`#[inline]`) functions, so rustc does not
//! inline them across the crate boundary. For the actual codegen
//! measurements, see `src/bin/measure.rs`, which disassembles the
//! `blas_interface` staticlib directly (it has no in-crate caller at all).

use blas_interface::{
    gemv_arr_16, gemv_arr_4, gemv_arr_8, gemv_checked_4, gemv_const_4, gemv_const_xy_4, gemv_dyn,
    gemv_generic_fn_4, gemv_nested_4, gemv_ptr_16, gemv_ptr_4, gemv_ptr_8, gemv_ptr_ab_4,
    gemv_storage_trait_4, Arr16, Arr4, Arr8, Dense4, DynDense, MatrixLayout, Ptr16, Ptr4, Ptr8,
};

fn max_abs_diff<const N: usize>(a: &[f32; N], b: &[f32; N]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(l, r)| (l - r).abs())
        .fold(0.0f32, f32::max)
}

fn report<const N: usize>(name: &str, y: &[f32; N], expected: &[f32; N]) -> bool {
    let diff = max_abs_diff(y, expected);
    let passed = diff < 1e-9;
    let status = if passed { "PASS" } else { "FAIL" };
    println!(
        "{name:<18} y[0] = {:.1}  max|diff| = {diff:.3e}  [{status}]",
        y[0]
    );
    passed
}

/// Column-major packed `N x N` with `buf[j*N + i] = j*N + i + 1`.
fn expected_row_sums<const N: usize>() -> [f32; N] {
    core::array::from_fn(|i| {
        let n = N as f32;
        n * ((i + 1) as f32) + n * n * ((N - 1) as f32) / 2.0
    })
}

fn packed_colmajor<const NN: usize>() -> [f32; NN] {
    core::array::from_fn(|k| (k + 1) as f32)
}

fn ones<const N: usize>() -> [f32; N] {
    [1.0; N]
}

fn main() {
    let mut all_passed = true;

    all_passed &= check_colmajor_4();
    all_passed &= check_rowmajor_4();
    all_passed &= check_padded_lda();
    all_passed &= check_n::<8, 64>("8x8", |buf, x, y| {
        gemv_arr_8(&Arr8(buf), x, y);
    });
    all_passed &= check_n::<8, 64>("8x8 ptr", |buf, x, y| unsafe {
        gemv_ptr_8(&Ptr8(buf.as_slice()), x.as_ptr(), y.as_mut_ptr());
    });
    all_passed &= check_n::<16, 256>("16x16", |buf, x, y| {
        gemv_arr_16(&Arr16(buf), x, y);
    });
    all_passed &= check_n::<16, 256>("16x16 ptr", |buf, x, y| unsafe {
        gemv_ptr_16(&Ptr16(buf.as_slice()), x.as_ptr(), y.as_mut_ptr());
    });

    if all_passed {
        println!("all variants agree");
    } else {
        eprintln!("mismatch detected");
        std::process::exit(1);
    }
}

fn check_colmajor_4() -> bool {
    let buf = packed_colmajor::<16>();
    let x = ones::<4>();
    let expected = expected_row_sums::<4>();
    let mut ok = true;

    let dyn_dense = DynDense {
        buf: buf.as_slice(),
        rows: 4,
        cols: 4,
        lda: 4,
        order: MatrixLayout::ColMajor,
    };
    let mut y = [0.0f32; 4];
    gemv_dyn(&dyn_dense, &x, &mut y);
    ok &= report("gemv_dyn", &y, &expected);

    let dense4 = Dense4(buf.as_slice());
    let mut y = [0.0f32; 4];
    gemv_const_4(&dense4, &x, &mut y);
    ok &= report("gemv_const_4", &y, &expected);

    let mut y = [0.0f32; 4];
    gemv_const_xy_4(&dense4, &x, &mut y);
    ok &= report("gemv_const_xy_4", &y, &expected);

    let mut y = [0.0f32; 4];
    gemv_checked_4(&dense4, &x, &mut y);
    ok &= report("gemv_checked_4", &y, &expected);

    let mut y = [0.0f32; 4];
    gemv_arr_4(&Arr4(&buf), &x, &mut y);
    ok &= report("gemv_arr_4", &y, &expected);

    let ptr4 = Ptr4(buf.as_slice());
    let mut y = [0.0f32; 4];
    unsafe { gemv_ptr_4(&ptr4, x.as_ptr(), y.as_mut_ptr()) };
    ok &= report("gemv_ptr_4", &y, &expected);

    let mut y = [0.0f32; 4];
    gemv_storage_trait_4(&dense4, &x, &mut y);
    ok &= report("gemv_storage_trait_4", &y, &expected);

    let mut y = [0.0f32; 4];
    gemv_generic_fn_4(&dense4, &x, &mut y);
    ok &= report("gemv_generic_fn_4", &y, &expected);

    let nested_buf: [[f32; 4]; 4] = [
        [buf[0], buf[1], buf[2], buf[3]],
        [buf[4], buf[5], buf[6], buf[7]],
        [buf[8], buf[9], buf[10], buf[11]],
        [buf[12], buf[13], buf[14], buf[15]],
    ];
    let mut y = [0.0f32; 4];
    gemv_nested_4(&nested_buf, &x, &mut y);
    ok &= report("gemv_nested_4", &y, &expected);

    let alpha = 2.0f32;
    let beta = 0.5f32;
    let mut y_ab = [1.0f32; 4];
    let expected_ab: [f32; 4] = core::array::from_fn(|i| alpha * expected[i] + beta * y_ab[i]);
    unsafe { gemv_ptr_ab_4(&ptr4, alpha, x.as_ptr(), beta, y_ab.as_mut_ptr()) };
    ok &= report("gemv_ptr_ab_4", &y_ab, &expected_ab);

    ok
}

fn check_rowmajor_4() -> bool {
    // Same logical matrix as packed col-major 1..=16, stored row-major.
    let buf = [
        1.0, 5.0, 9.0, 13.0, 2.0, 6.0, 10.0, 14.0, 3.0, 7.0, 11.0, 15.0, 4.0, 8.0, 12.0, 16.0,
    ];
    let x = ones::<4>();
    let expected = expected_row_sums::<4>();
    let s = DynDense {
        buf: buf.as_slice(),
        rows: 4,
        cols: 4,
        lda: 4,
        order: MatrixLayout::RowMajor,
    };
    let mut y = [0.0f32; 4];
    gemv_dyn(&s, &x, &mut y);
    report("gemv_dyn rm", &y, &expected)
}

fn check_padded_lda() -> bool {
    // 4x4 logical matrix in a col-major buffer with lda=6.
    let mut buf = [0.0f32; 24];
    for j in 0..4 {
        for i in 0..4 {
            buf[j * 6 + i] = (j * 4 + i + 1) as f32;
        }
    }
    let x = ones::<4>();
    let expected = expected_row_sums::<4>();
    let s = DynDense {
        buf: buf.as_slice(),
        rows: 4,
        cols: 4,
        lda: 6,
        order: MatrixLayout::ColMajor,
    };
    let mut y = [0.0f32; 4];
    gemv_dyn(&s, &x, &mut y);
    report("gemv_dyn lda6", &y, &expected)
}

fn check_n<const N: usize, const NN: usize>(
    name: &str,
    gemv: impl FnOnce(&[f32; NN], &[f32; N], &mut [f32; N]),
) -> bool {
    let buf = packed_colmajor::<NN>();
    let x = ones::<N>();
    let expected = expected_row_sums::<N>();
    let mut y = [0.0f32; N];
    gemv(&buf, &x, &mut y);
    report(name, &y, &expected)
}
