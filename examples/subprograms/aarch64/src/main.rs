use core::hint::black_box;
use std::time::Instant;

use control_rs::math::num_types::Const;
use control_rs::math::storage::{
    ArrayStorage, RowArrayStorage, StorageView, Trans,
};
use control_rs::math::subprograms::level1::{Axpy, Dotu, Nrm2, Scal};
use control_rs::math::subprograms::level2::Gemv;
use control_rs::math::subprograms::level3::Gemm;
use control_rs::math::subprograms::DefaultBlas;

#[cfg(feature = "accelerate")]
use aarch64_subprograms::AccelerateBlas;
use aarch64_subprograms::NeonBlas;

/// Level-1 length: 256× 4-wide NEON (and 128× 8-wide AVX2 on the sibling crate).
const N_L1: usize = 1024;
const GV_M: usize = 128;
const GV_N: usize = 128;
const GM_M: usize = 64;
const GM_K: usize = 64;
const GM_N: usize = 64;

const ITERS_L1: u32 = 800;
const ITERS_L2: u32 = 120;
const ITERS_L3: u32 = 40;

fn fill_vec_f32<const N: usize>() -> [f32; N] {
    let mut a = [0.0f32; N];
    for i in 0..N {
        a[i] = (i % 17) as f32 * 0.125 + 0.5;
    }
    a
}

fn fill_vec_f64<const N: usize>() -> [f64; N] {
    let mut a = [0.0f64; N];
    for i in 0..N {
        a[i] = (i % 17) as f64 * 0.125 + 0.5;
    }
    a
}

fn fill_mat_f32<const R: usize, const C: usize>() -> [[f32; C]; R] {
    let mut a = [[0.0f32; C]; R];
    for r in 0..R {
        for c in 0..C {
            a[r][c] = ((r + 3 * c) % 13) as f32 * 0.05 + 0.1;
        }
    }
    a
}

fn fill_mat_f64<const R: usize, const C: usize>() -> [[f64; C]; R] {
    let mut a = [[0.0f64; C]; R];
    for r in 0..R {
        for c in 0..C {
            a[r][c] = ((r + 3 * c) % 13) as f64 * 0.05 + 0.1;
        }
    }
    a
}

fn transpose_f32<const R: usize, const C: usize>(
    a: &[[f32; C]; R],
) -> [[f32; R]; C] {
    let mut t = [[0.0f32; R]; C];
    for r in 0..R {
        for c in 0..C {
            t[c][r] = a[r][c];
        }
    }
    t
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn max_abs_diff_f64(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

fn bench_ns(iters: u32, mut body: impl FnMut()) -> u64 {
    body();
    let t0 = Instant::now();
    for _ in 0..iters {
        body();
    }
    let ns = t0.elapsed().as_nanos();
    (ns / u128::from(iters.max(1))) as u64
}

fn report<T: Copy + PartialOrd + core::fmt::LowerExp>(
    label: &str,
    diff: T,
    limit: T,
    backend_ns: u64,
    default_ns: u64,
) {
    let ratio = if backend_ns == 0 {
        0.0
    } else {
        default_ns as f64 / backend_ns as f64
    };
    println!(
        "  {label}: max diff = {diff:.2e}  backend={backend_ns} ns/call  DefaultBlas={default_ns} ns/call  ({ratio:.2}x) ... OK"
    );
    assert!(diff < limit);
}

fn test_neon_f32() {
    println!("--- Testing NeonBlas (f32) ---");

    let x_arr = fill_vec_f32::<N_L1>();
    let y_init = fill_vec_f32::<N_L1>();
    let x_s = ArrayStorage::<f32, N_L1, 1>::from_array([x_arr]);
    let mut y_s_neon = ArrayStorage::<f32, N_L1, 1>::from_array([y_init]);
    let mut y_s_ref = ArrayStorage::<f32, N_L1, 1>::from_array([y_init]);

    NeonBlas::axpy(2.5, &x_s, &mut y_s_neon);
    DefaultBlas::axpy(2.5, &x_s, &mut y_s_ref);
    let diff = max_abs_diff(y_s_neon.as_slice(), y_s_ref.as_slice());
    let be = bench_ns(ITERS_L1, || {
        NeonBlas::axpy(2.5, &x_s, &mut y_s_neon);
        black_box(y_s_neon.as_slice());
    });
    let re = bench_ns(ITERS_L1, || {
        DefaultBlas::axpy(2.5, &x_s, &mut y_s_ref);
        black_box(y_s_ref.as_slice());
    });
    report(&format!("Axpy (N={N_L1})"), diff, 1e-4, be, re);

    let mut x_scal_neon = ArrayStorage::<f32, N_L1, 1>::from_array([x_arr]);
    let mut x_scal_ref = ArrayStorage::<f32, N_L1, 1>::from_array([x_arr]);
    NeonBlas::scal(3.0, &mut x_scal_neon);
    DefaultBlas::scal(3.0, &mut x_scal_ref);
    let diff_scal = max_abs_diff(x_scal_neon.as_slice(), x_scal_ref.as_slice());
    let be = bench_ns(ITERS_L1, || {
        NeonBlas::scal(3.0, &mut x_scal_neon);
        black_box(x_scal_neon.as_slice());
    });
    let re = bench_ns(ITERS_L1, || {
        DefaultBlas::scal(3.0, &mut x_scal_ref);
        black_box(x_scal_ref.as_slice());
    });
    report(&format!("Scal (N={N_L1})"), diff_scal, 1e-4, be, re);

    let y_dot = ArrayStorage::<f32, N_L1, 1>::from_array([y_init]);
    let dot_neon = NeonBlas::dotu(&x_s, &y_dot);
    let dot_ref = DefaultBlas::dotu(&x_s, &y_dot);
    let diff_dot = (dot_neon - dot_ref).abs();
    let be = bench_ns(ITERS_L1, || {
        black_box(NeonBlas::dotu(&x_s, &y_dot));
    });
    let re = bench_ns(ITERS_L1, || {
        black_box(DefaultBlas::dotu(&x_s, &y_dot));
    });
    report(&format!("Dotu (N={N_L1})"), diff_dot, 1e-3, be, re);

    let nrm_neon = NeonBlas::nrm2(&x_s);
    let nrm_ref = DefaultBlas::nrm2(&x_s);
    let diff_nrm = (nrm_neon - nrm_ref).abs();
    let be = bench_ns(ITERS_L1, || {
        black_box(NeonBlas::nrm2(&x_s));
    });
    let re = bench_ns(ITERS_L1, || {
        black_box(DefaultBlas::nrm2(&x_s));
    });
    report(&format!("Nrm2 (N={N_L1})"), diff_nrm, 1e-3, be, re);

    let a_gemv_data = fill_mat_f32::<GV_M, GV_N>();
    let vx_data = fill_vec_f32::<GV_N>();
    let a_mat = RowArrayStorage::<f32, GV_M, GV_N>::from_array(a_gemv_data);
    let vx = ArrayStorage::<f32, GV_N, 1>::from_array([vx_data]);
    let mut vy_neon =
        ArrayStorage::<f32, GV_M, 1>::from_array([fill_vec_f32::<GV_M>()]);
    let mut vy_ref =
        ArrayStorage::<f32, GV_M, 1>::from_array([fill_vec_f32::<GV_M>()]);

    NeonBlas::gemv(Trans::NoTrans, 1.5, &a_mat, &vx, 0.5, &mut vy_neon);
    DefaultBlas::gemv(Trans::NoTrans, 1.5, &a_mat, &vx, 0.5, &mut vy_ref);
    let diff_gemv = max_abs_diff(vy_neon.as_slice(), vy_ref.as_slice());
    let be = bench_ns(ITERS_L2, || {
        NeonBlas::gemv(Trans::NoTrans, 1.5, &a_mat, &vx, 0.5, &mut vy_neon);
        black_box(vy_neon.as_slice());
    });
    let re = bench_ns(ITERS_L2, || {
        DefaultBlas::gemv(Trans::NoTrans, 1.5, &a_mat, &vx, 0.5, &mut vy_ref);
        black_box(vy_ref.as_slice());
    });
    report(
        &format!("Gemv NoTrans ({GV_M}x{GV_N})"),
        diff_gemv,
        2e-3,
        be,
        re,
    );

    let vx_t =
        ArrayStorage::<f32, GV_M, 1>::from_array([fill_vec_f32::<GV_M>()]);
    let mut vy_t_neon =
        ArrayStorage::<f32, GV_N, 1>::from_array([[0.0f32; GV_N]]);
    let mut vy_t_ref =
        ArrayStorage::<f32, GV_N, 1>::from_array([[0.0f32; GV_N]]);
    NeonBlas::gemv(Trans::Trans, 1.2, &a_mat, &vx_t, 0.0, &mut vy_t_neon);
    DefaultBlas::gemv(Trans::Trans, 1.2, &a_mat, &vx_t, 0.0, &mut vy_t_ref);
    let diff_gemv_t = max_abs_diff(vy_t_neon.as_slice(), vy_t_ref.as_slice());
    let be = bench_ns(ITERS_L2, || {
        NeonBlas::gemv(Trans::Trans, 1.2, &a_mat, &vx_t, 0.0, &mut vy_t_neon);
        black_box(vy_t_neon.as_slice());
    });
    let re = bench_ns(ITERS_L2, || {
        DefaultBlas::gemv(Trans::Trans, 1.2, &a_mat, &vx_t, 0.0, &mut vy_t_ref);
        black_box(vy_t_ref.as_slice());
    });
    report(
        &format!("Gemv Trans ({GV_N}x{GV_M})"),
        diff_gemv_t,
        2e-3,
        be,
        re,
    );

    let a_gemm_data = fill_mat_f32::<GM_M, GM_K>();
    let b_gemm_data = fill_mat_f32::<GM_K, GM_N>();
    let a_storage = RowArrayStorage::<f32, GM_M, GM_K>::from_array(a_gemm_data);
    let b_storage = RowArrayStorage::<f32, GM_K, GM_N>::from_array(b_gemm_data);
    let mut c_neon =
        RowArrayStorage::<f32, GM_M, GM_N>::from_array([[1.0f32; GM_N]; GM_M]);
    let mut c_ref =
        RowArrayStorage::<f32, GM_M, GM_N>::from_array([[1.0f32; GM_N]; GM_M]);

    NeonBlas::gemm(
        Trans::NoTrans,
        Trans::NoTrans,
        1.2,
        &a_storage,
        &b_storage,
        0.8,
        &mut c_neon,
    );
    DefaultBlas::gemm(
        Trans::NoTrans,
        Trans::NoTrans,
        1.2,
        &a_storage,
        &b_storage,
        0.8,
        &mut c_ref,
    );
    let diff_gemm = max_abs_diff(c_neon.as_slice(), c_ref.as_slice());
    let be = bench_ns(ITERS_L3, || {
        NeonBlas::gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            1.2,
            &a_storage,
            &b_storage,
            0.8,
            &mut c_neon,
        );
        black_box(c_neon.as_slice());
    });
    let re = bench_ns(ITERS_L3, || {
        DefaultBlas::gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            1.2,
            &a_storage,
            &b_storage,
            0.8,
            &mut c_ref,
        );
        black_box(c_ref.as_slice());
    });
    report(
        &format!("Gemm NoTrans ({GM_M}x{GM_K} * {GM_K}x{GM_N})"),
        diff_gemm,
        5e-3,
        be,
        re,
    );

    let a_t_mat = RowArrayStorage::<f32, GM_K, GM_M>::from_array(
        transpose_f32(&a_gemm_data),
    );
    let mut c_t_neon =
        RowArrayStorage::<f32, GM_M, GM_N>::from_array([[0.0f32; GM_N]; GM_M]);
    let mut c_t_ref =
        RowArrayStorage::<f32, GM_M, GM_N>::from_array([[0.0f32; GM_N]; GM_M]);
    NeonBlas::gemm(
        Trans::Trans,
        Trans::NoTrans,
        1.0,
        &a_t_mat,
        &b_storage,
        0.0,
        &mut c_t_neon,
    );
    DefaultBlas::gemm(
        Trans::Trans,
        Trans::NoTrans,
        1.0,
        &a_t_mat,
        &b_storage,
        0.0,
        &mut c_t_ref,
    );
    let diff_gemm_t = max_abs_diff(c_t_neon.as_slice(), c_t_ref.as_slice());
    let be = bench_ns(ITERS_L3, || {
        NeonBlas::gemm(
            Trans::Trans,
            Trans::NoTrans,
            1.0,
            &a_t_mat,
            &b_storage,
            0.0,
            &mut c_t_neon,
        );
        black_box(c_t_neon.as_slice());
    });
    let re = bench_ns(ITERS_L3, || {
        DefaultBlas::gemm(
            Trans::Trans,
            Trans::NoTrans,
            1.0,
            &a_t_mat,
            &b_storage,
            0.0,
            &mut c_t_ref,
        );
        black_box(c_t_ref.as_slice());
    });
    report(
        &format!("Gemm Trans A ({GM_M}x{GM_K} * {GM_K}x{GM_N})"),
        diff_gemm_t,
        5e-3,
        be,
        re,
    );
}

fn test_neon_f64() {
    println!("\n--- Testing NeonBlas (f64) ---");

    let x_arr = fill_vec_f64::<N_L1>();
    let y_init = fill_vec_f64::<N_L1>();
    let x_s = ArrayStorage::<f64, N_L1, 1>::from_array([x_arr]);
    let mut y_s_neon = ArrayStorage::<f64, N_L1, 1>::from_array([y_init]);
    let mut y_s_ref = ArrayStorage::<f64, N_L1, 1>::from_array([y_init]);

    NeonBlas::axpy(2.5, &x_s, &mut y_s_neon);
    DefaultBlas::axpy(2.5, &x_s, &mut y_s_ref);
    let diff = max_abs_diff_f64(y_s_neon.as_slice(), y_s_ref.as_slice());
    let be = bench_ns(ITERS_L1, || {
        NeonBlas::axpy(2.5, &x_s, &mut y_s_neon);
        black_box(y_s_neon.as_slice());
    });
    let re = bench_ns(ITERS_L1, || {
        DefaultBlas::axpy(2.5, &x_s, &mut y_s_ref);
        black_box(y_s_ref.as_slice());
    });
    report(&format!("Axpy (f64, N={N_L1})"), diff, 1e-10, be, re);

    let y_dot = ArrayStorage::<f64, N_L1, 1>::from_array([y_init]);
    let dot_neon = NeonBlas::dotu(&x_s, &y_dot);
    let dot_ref = DefaultBlas::dotu(&x_s, &y_dot);
    let diff_dot = (dot_neon - dot_ref).abs();
    let be = bench_ns(ITERS_L1, || {
        black_box(NeonBlas::dotu(&x_s, &y_dot));
    });
    let re = bench_ns(ITERS_L1, || {
        black_box(DefaultBlas::dotu(&x_s, &y_dot));
    });
    report(&format!("Dotu (f64, N={N_L1})"), diff_dot, 1e-9, be, re);

    let nrm_neon = NeonBlas::nrm2(&x_s);
    let nrm_ref = DefaultBlas::nrm2(&x_s);
    let diff_nrm = (nrm_neon - nrm_ref).abs();
    let be = bench_ns(ITERS_L1, || {
        black_box(NeonBlas::nrm2(&x_s));
    });
    let re = bench_ns(ITERS_L1, || {
        black_box(DefaultBlas::nrm2(&x_s));
    });
    report(&format!("Nrm2 (f64, N={N_L1})"), diff_nrm, 1e-9, be, re);

    let a_gemv_data = fill_mat_f64::<GV_M, GV_N>();
    let vx_data = fill_vec_f64::<GV_N>();
    let a_mat = RowArrayStorage::<f64, GV_M, GV_N>::from_array(a_gemv_data);
    let vx = ArrayStorage::<f64, GV_N, 1>::from_array([vx_data]);
    let mut vy_neon =
        ArrayStorage::<f64, GV_M, 1>::from_array([fill_vec_f64::<GV_M>()]);
    let mut vy_ref =
        ArrayStorage::<f64, GV_M, 1>::from_array([fill_vec_f64::<GV_M>()]);
    NeonBlas::gemv(Trans::NoTrans, 1.5, &a_mat, &vx, 0.5, &mut vy_neon);
    DefaultBlas::gemv(Trans::NoTrans, 1.5, &a_mat, &vx, 0.5, &mut vy_ref);
    let diff_gemv = max_abs_diff_f64(vy_neon.as_slice(), vy_ref.as_slice());
    let be = bench_ns(ITERS_L2, || {
        NeonBlas::gemv(Trans::NoTrans, 1.5, &a_mat, &vx, 0.5, &mut vy_neon);
        black_box(vy_neon.as_slice());
    });
    let re = bench_ns(ITERS_L2, || {
        DefaultBlas::gemv(Trans::NoTrans, 1.5, &a_mat, &vx, 0.5, &mut vy_ref);
        black_box(vy_ref.as_slice());
    });
    report(
        &format!("Gemv NoTrans (f64, {GV_M}x{GV_N})"),
        diff_gemv,
        1e-9,
        be,
        re,
    );

    let a_gemm_data = fill_mat_f64::<GM_M, GM_K>();
    let b_gemm_data = fill_mat_f64::<GM_K, GM_N>();
    let a_storage = RowArrayStorage::<f64, GM_M, GM_K>::from_array(a_gemm_data);
    let b_storage = RowArrayStorage::<f64, GM_K, GM_N>::from_array(b_gemm_data);
    let mut c_neon =
        RowArrayStorage::<f64, GM_M, GM_N>::from_array([[1.0f64; GM_N]; GM_M]);
    let mut c_ref =
        RowArrayStorage::<f64, GM_M, GM_N>::from_array([[1.0f64; GM_N]; GM_M]);
    NeonBlas::gemm(
        Trans::NoTrans,
        Trans::NoTrans,
        1.2,
        &a_storage,
        &b_storage,
        0.8,
        &mut c_neon,
    );
    DefaultBlas::gemm(
        Trans::NoTrans,
        Trans::NoTrans,
        1.2,
        &a_storage,
        &b_storage,
        0.8,
        &mut c_ref,
    );
    let diff_gemm = max_abs_diff_f64(c_neon.as_slice(), c_ref.as_slice());
    let be = bench_ns(ITERS_L3, || {
        NeonBlas::gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            1.2,
            &a_storage,
            &b_storage,
            0.8,
            &mut c_neon,
        );
        black_box(c_neon.as_slice());
    });
    let re = bench_ns(ITERS_L3, || {
        DefaultBlas::gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            1.2,
            &a_storage,
            &b_storage,
            0.8,
            &mut c_ref,
        );
        black_box(c_ref.as_slice());
    });
    report(
        &format!("Gemm NoTrans (f64, {GM_M}x{GM_K} * {GM_K}x{GM_N})"),
        diff_gemm,
        1e-8,
        be,
        re,
    );
}

fn test_strided_fallback() {
    println!("\n--- Testing Strided / Non-Contiguous Fallback Path ---");
    let big = fill_mat_f32::<16, 16>();
    let big_mat = RowArrayStorage::<f32, 16, 16>::from_array(big);
    let subview_a = StorageView::<f32, Const<8>, Const<8>>::new_with_strides(
        big_mat.as_slice(),
        16,
        2,
    )
    .unwrap();
    let vx_sub = ArrayStorage::<f32, 8, 1>::from_array([fill_vec_f32::<8>()]);
    let mut vy_sub_neon = ArrayStorage::<f32, 8, 1>::from_array([[0.0f32; 8]]);
    let mut vy_sub_ref = ArrayStorage::<f32, 8, 1>::from_array([[0.0f32; 8]]);

    NeonBlas::gemv(
        Trans::NoTrans,
        1.0,
        &subview_a,
        &vx_sub,
        0.0,
        &mut vy_sub_neon,
    );
    DefaultBlas::gemv(
        Trans::NoTrans,
        1.0,
        &subview_a,
        &vx_sub,
        0.0,
        &mut vy_sub_ref,
    );
    let diff_strided =
        max_abs_diff(vy_sub_neon.as_slice(), vy_sub_ref.as_slice());
    let be = bench_ns(ITERS_L2, || {
        NeonBlas::gemv(
            Trans::NoTrans,
            1.0,
            &subview_a,
            &vx_sub,
            0.0,
            &mut vy_sub_neon,
        );
        black_box(vy_sub_neon.as_slice());
    });
    let re = bench_ns(ITERS_L2, || {
        DefaultBlas::gemv(
            Trans::NoTrans,
            1.0,
            &subview_a,
            &vx_sub,
            0.0,
            &mut vy_sub_ref,
        );
        black_box(vy_sub_ref.as_slice());
    });
    report(
        "Gemv Strided Subview (8x8, stride=2)",
        diff_strided,
        1e-5,
        be,
        re,
    );
}

fn main() {
    println!("=== AArch64 Subprogram Backend Verification ===");
    println!(
        "Shapes: L1 N={N_L1}, Gemv {GV_M}x{GV_N}, Gemm {GM_M}x{GM_K}x{GM_N}. Times averaged over {ITERS_L1}/{ITERS_L2}/{ITERS_L3} iters (L1/L2/L3). Use --release. Not a pass/fail gate."
    );
    test_neon_f32();
    test_neon_f64();
    test_strided_fallback();

    #[cfg(feature = "accelerate")]
    {
        println!("\n--- Testing AccelerateBlas (vecLib CBLAS) ---");
        let x_arr = fill_vec_f32::<N_L1>();
        let y_init = fill_vec_f32::<N_L1>();
        let x_s = ArrayStorage::<f32, N_L1, 1>::from_array([x_arr]);
        let mut y_s_acc = ArrayStorage::<f32, N_L1, 1>::from_array([y_init]);
        let mut y_s_ref = ArrayStorage::<f32, N_L1, 1>::from_array([y_init]);

        AccelerateBlas::axpy(2.5, &x_s, &mut y_s_acc);
        DefaultBlas::axpy(2.5, &x_s, &mut y_s_ref);
        let diff = max_abs_diff(y_s_acc.as_slice(), y_s_ref.as_slice());
        let be = bench_ns(ITERS_L1, || {
            AccelerateBlas::axpy(2.5, &x_s, &mut y_s_acc);
            black_box(y_s_acc.as_slice());
        });
        let re = bench_ns(ITERS_L1, || {
            DefaultBlas::axpy(2.5, &x_s, &mut y_s_ref);
            black_box(y_s_ref.as_slice());
        });
        report(&format!("Accelerate Axpy (N={N_L1})"), diff, 1e-4, be, re);

        let a_data = fill_mat_f32::<GM_M, GM_K>();
        let b_data = fill_mat_f32::<GM_K, GM_N>();
        let a = RowArrayStorage::<f32, GM_M, GM_K>::from_array(a_data);
        let b = RowArrayStorage::<f32, GM_K, GM_N>::from_array(b_data);
        let mut c_acc = RowArrayStorage::<f32, GM_M, GM_N>::from_array(
            [[0.0f32; GM_N]; GM_M],
        );
        let mut c_ref = RowArrayStorage::<f32, GM_M, GM_N>::from_array(
            [[0.0f32; GM_N]; GM_M],
        );
        AccelerateBlas::gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            1.0,
            &a,
            &b,
            0.0,
            &mut c_acc,
        );
        DefaultBlas::gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            1.0,
            &a,
            &b,
            0.0,
            &mut c_ref,
        );
        let diff_g = max_abs_diff(c_acc.as_slice(), c_ref.as_slice());
        let be = bench_ns(ITERS_L3, || {
            AccelerateBlas::gemm(
                Trans::NoTrans,
                Trans::NoTrans,
                1.0,
                &a,
                &b,
                0.0,
                &mut c_acc,
            );
            black_box(c_acc.as_slice());
        });
        let re = bench_ns(ITERS_L3, || {
            DefaultBlas::gemm(
                Trans::NoTrans,
                Trans::NoTrans,
                1.0,
                &a,
                &b,
                0.0,
                &mut c_ref,
            );
            black_box(c_ref.as_slice());
        });
        report(
            &format!("Accelerate Gemm ({GM_M}x{GM_K} * {GM_K}x{GM_N})"),
            diff_g,
            5e-3,
            be,
            re,
        );
    }

    println!("\nAll aarch64 subprogram backend equivalence checks PASSED.");
}
