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

use x86_64_subprograms::Avx2Blas;
#[cfg(feature = "cblas")]
use x86_64_subprograms::CblasBlas;

/// Level-1 length: 128× 8-wide AVX2 (and 256× 4-wide NEON on the sibling crate).
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

fn main() {
    println!("=== x86_64 Subprogram Backend Verification ===");
    println!(
        "Shapes: L1 N={N_L1}, Gemv {GV_M}x{GV_N}, Gemm {GM_M}x{GM_K}x{GM_N}. Times averaged over {ITERS_L1}/{ITERS_L2}/{ITERS_L3} iters (L1/L2/L3). Use --release. Not a pass/fail gate."
    );

    #[cfg(target_arch = "x86_64")]
    {
        let has_avx2 = std::arch::is_x86_feature_detected!("avx2");
        let has_fma = std::arch::is_x86_feature_detected!("fma");
        println!("CPU Feature Detection: AVX2 = {has_avx2}, FMA = {has_fma}");
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("Running on non-x86_64 host; Avx2Blas will delegate to DefaultBlas.");
    }

    println!("\n--- Testing Avx2Blas (f32) ---");

    let x_arr = fill_vec_f32::<N_L1>();
    let y_init = fill_vec_f32::<N_L1>();
    let x_s = ArrayStorage::<f32, N_L1, 1>::from_array([x_arr]);
    let mut y_avx = ArrayStorage::<f32, N_L1, 1>::from_array([y_init]);
    let mut y_ref = ArrayStorage::<f32, N_L1, 1>::from_array([y_init]);

    Avx2Blas::axpy(2.5, &x_s, &mut y_avx);
    DefaultBlas::axpy(2.5, &x_s, &mut y_ref);
    let diff = max_abs_diff(y_avx.as_slice(), y_ref.as_slice());
    let be = bench_ns(ITERS_L1, || {
        Avx2Blas::axpy(2.5, &x_s, &mut y_avx);
        black_box(y_avx.as_slice());
    });
    let re = bench_ns(ITERS_L1, || {
        DefaultBlas::axpy(2.5, &x_s, &mut y_ref);
        black_box(y_ref.as_slice());
    });
    report(&format!("Axpy (f32, N={N_L1})"), diff, 1e-4, be, re);

    let mut x_scal_avx = ArrayStorage::<f32, N_L1, 1>::from_array([x_arr]);
    let mut x_scal_ref = ArrayStorage::<f32, N_L1, 1>::from_array([x_arr]);
    Avx2Blas::scal(3.0, &mut x_scal_avx);
    DefaultBlas::scal(3.0, &mut x_scal_ref);
    let diff_scal = max_abs_diff(x_scal_avx.as_slice(), x_scal_ref.as_slice());
    let be = bench_ns(ITERS_L1, || {
        Avx2Blas::scal(3.0, &mut x_scal_avx);
        black_box(x_scal_avx.as_slice());
    });
    let re = bench_ns(ITERS_L1, || {
        DefaultBlas::scal(3.0, &mut x_scal_ref);
        black_box(x_scal_ref.as_slice());
    });
    report(&format!("Scal (f32, N={N_L1})"), diff_scal, 1e-4, be, re);

    let y_dot = ArrayStorage::<f32, N_L1, 1>::from_array([y_init]);
    let dot_avx = Avx2Blas::dotu(&x_s, &y_dot);
    let dot_ref = DefaultBlas::dotu(&x_s, &y_dot);
    let diff_dot = (dot_avx - dot_ref).abs();
    let be = bench_ns(ITERS_L1, || {
        black_box(Avx2Blas::dotu(&x_s, &y_dot));
    });
    let re = bench_ns(ITERS_L1, || {
        black_box(DefaultBlas::dotu(&x_s, &y_dot));
    });
    report(&format!("Dotu (f32, N={N_L1})"), diff_dot, 1e-3, be, re);

    let nrm_avx = Avx2Blas::nrm2(&x_s);
    let nrm_ref = DefaultBlas::nrm2(&x_s);
    let diff_nrm = (nrm_avx - nrm_ref).abs();
    let be = bench_ns(ITERS_L1, || {
        black_box(Avx2Blas::nrm2(&x_s));
    });
    let re = bench_ns(ITERS_L1, || {
        black_box(DefaultBlas::nrm2(&x_s));
    });
    report(&format!("Nrm2 (f32, N={N_L1})"), diff_nrm, 1e-3, be, re);

    let a_gemv_data = fill_mat_f32::<GV_M, GV_N>();
    let vx_data = fill_vec_f32::<GV_N>();
    let a_mat = RowArrayStorage::<f32, GV_M, GV_N>::from_array(a_gemv_data);
    let vx = ArrayStorage::<f32, GV_N, 1>::from_array([vx_data]);
    let mut vy_avx =
        ArrayStorage::<f32, GV_M, 1>::from_array([fill_vec_f32::<GV_M>()]);
    let mut vy_ref =
        ArrayStorage::<f32, GV_M, 1>::from_array([fill_vec_f32::<GV_M>()]);

    Avx2Blas::gemv(Trans::NoTrans, 1.5, &a_mat, &vx, 0.5, &mut vy_avx);
    DefaultBlas::gemv(Trans::NoTrans, 1.5, &a_mat, &vx, 0.5, &mut vy_ref);
    let diff_gemv = max_abs_diff(vy_avx.as_slice(), vy_ref.as_slice());
    let be = bench_ns(ITERS_L2, || {
        Avx2Blas::gemv(Trans::NoTrans, 1.5, &a_mat, &vx, 0.5, &mut vy_avx);
        black_box(vy_avx.as_slice());
    });
    let re = bench_ns(ITERS_L2, || {
        DefaultBlas::gemv(Trans::NoTrans, 1.5, &a_mat, &vx, 0.5, &mut vy_ref);
        black_box(vy_ref.as_slice());
    });
    report(
        &format!("Gemv NoTrans (f32, {GV_M}x{GV_N})"),
        diff_gemv,
        2e-3,
        be,
        re,
    );

    let vx_t =
        ArrayStorage::<f32, GV_M, 1>::from_array([fill_vec_f32::<GV_M>()]);
    let mut vy_t_avx =
        ArrayStorage::<f32, GV_N, 1>::from_array([[0.0f32; GV_N]]);
    let mut vy_t_ref =
        ArrayStorage::<f32, GV_N, 1>::from_array([[0.0f32; GV_N]]);
    Avx2Blas::gemv(Trans::Trans, 1.2, &a_mat, &vx_t, 0.0, &mut vy_t_avx);
    DefaultBlas::gemv(Trans::Trans, 1.2, &a_mat, &vx_t, 0.0, &mut vy_t_ref);
    let diff_gemv_t = max_abs_diff(vy_t_avx.as_slice(), vy_t_ref.as_slice());
    let be = bench_ns(ITERS_L2, || {
        Avx2Blas::gemv(Trans::Trans, 1.2, &a_mat, &vx_t, 0.0, &mut vy_t_avx);
        black_box(vy_t_avx.as_slice());
    });
    let re = bench_ns(ITERS_L2, || {
        DefaultBlas::gemv(Trans::Trans, 1.2, &a_mat, &vx_t, 0.0, &mut vy_t_ref);
        black_box(vy_t_ref.as_slice());
    });
    report(
        &format!("Gemv Trans (f32, {GV_N}x{GV_M})"),
        diff_gemv_t,
        2e-3,
        be,
        re,
    );

    let a_gemm_data = fill_mat_f32::<GM_M, GM_K>();
    let b_gemm_data = fill_mat_f32::<GM_K, GM_N>();
    let a_s = RowArrayStorage::<f32, GM_M, GM_K>::from_array(a_gemm_data);
    let b_s = RowArrayStorage::<f32, GM_K, GM_N>::from_array(b_gemm_data);
    let mut c_avx =
        RowArrayStorage::<f32, GM_M, GM_N>::from_array([[1.0f32; GM_N]; GM_M]);
    let mut c_ref =
        RowArrayStorage::<f32, GM_M, GM_N>::from_array([[1.0f32; GM_N]; GM_M]);

    Avx2Blas::gemm(
        Trans::NoTrans,
        Trans::NoTrans,
        1.2,
        &a_s,
        &b_s,
        0.8,
        &mut c_avx,
    );
    DefaultBlas::gemm(
        Trans::NoTrans,
        Trans::NoTrans,
        1.2,
        &a_s,
        &b_s,
        0.8,
        &mut c_ref,
    );
    let diff_gemm = max_abs_diff(c_avx.as_slice(), c_ref.as_slice());
    let be = bench_ns(ITERS_L3, || {
        Avx2Blas::gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            1.2,
            &a_s,
            &b_s,
            0.8,
            &mut c_avx,
        );
        black_box(c_avx.as_slice());
    });
    let re = bench_ns(ITERS_L3, || {
        DefaultBlas::gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            1.2,
            &a_s,
            &b_s,
            0.8,
            &mut c_ref,
        );
        black_box(c_ref.as_slice());
    });
    report(
        &format!("Gemm NoTrans (f32, {GM_M}x{GM_K} * {GM_K}x{GM_N})"),
        diff_gemm,
        5e-3,
        be,
        re,
    );

    let a_t = RowArrayStorage::<f32, GM_K, GM_M>::from_array(transpose_f32(
        &a_gemm_data,
    ));
    let mut c_t_avx =
        RowArrayStorage::<f32, GM_M, GM_N>::from_array([[0.0f32; GM_N]; GM_M]);
    let mut c_t_ref =
        RowArrayStorage::<f32, GM_M, GM_N>::from_array([[0.0f32; GM_N]; GM_M]);
    Avx2Blas::gemm(
        Trans::Trans,
        Trans::NoTrans,
        1.0,
        &a_t,
        &b_s,
        0.0,
        &mut c_t_avx,
    );
    DefaultBlas::gemm(
        Trans::Trans,
        Trans::NoTrans,
        1.0,
        &a_t,
        &b_s,
        0.0,
        &mut c_t_ref,
    );
    let diff_gemm_t = max_abs_diff(c_t_avx.as_slice(), c_t_ref.as_slice());
    let be = bench_ns(ITERS_L3, || {
        Avx2Blas::gemm(
            Trans::Trans,
            Trans::NoTrans,
            1.0,
            &a_t,
            &b_s,
            0.0,
            &mut c_t_avx,
        );
        black_box(c_t_avx.as_slice());
    });
    let re = bench_ns(ITERS_L3, || {
        DefaultBlas::gemm(
            Trans::Trans,
            Trans::NoTrans,
            1.0,
            &a_t,
            &b_s,
            0.0,
            &mut c_t_ref,
        );
        black_box(c_t_ref.as_slice());
    });
    report(
        &format!("Gemm Trans A (f32, {GM_M}x{GM_K} * {GM_K}x{GM_N})"),
        diff_gemm_t,
        5e-3,
        be,
        re,
    );

    println!("\n--- Testing Avx2Blas (f64) ---");

    let x_arr64 = fill_vec_f64::<N_L1>();
    let y_init64 = fill_vec_f64::<N_L1>();
    let x_s64 = ArrayStorage::<f64, N_L1, 1>::from_array([x_arr64]);
    let mut y_avx64 = ArrayStorage::<f64, N_L1, 1>::from_array([y_init64]);
    let mut y_ref64 = ArrayStorage::<f64, N_L1, 1>::from_array([y_init64]);

    Avx2Blas::axpy(2.5, &x_s64, &mut y_avx64);
    DefaultBlas::axpy(2.5, &x_s64, &mut y_ref64);
    let diff64 = max_abs_diff_f64(y_avx64.as_slice(), y_ref64.as_slice());
    let be = bench_ns(ITERS_L1, || {
        Avx2Blas::axpy(2.5, &x_s64, &mut y_avx64);
        black_box(y_avx64.as_slice());
    });
    let re = bench_ns(ITERS_L1, || {
        DefaultBlas::axpy(2.5, &x_s64, &mut y_ref64);
        black_box(y_ref64.as_slice());
    });
    report(&format!("Axpy (f64, N={N_L1})"), diff64, 1e-10, be, re);

    let mut x_scal64 = ArrayStorage::<f64, N_L1, 1>::from_array([x_arr64]);
    let mut x_scal_ref64 = ArrayStorage::<f64, N_L1, 1>::from_array([x_arr64]);
    Avx2Blas::scal(3.0, &mut x_scal64);
    DefaultBlas::scal(3.0, &mut x_scal_ref64);
    let diff_scal64 =
        max_abs_diff_f64(x_scal64.as_slice(), x_scal_ref64.as_slice());
    let be = bench_ns(ITERS_L1, || {
        Avx2Blas::scal(3.0, &mut x_scal64);
        black_box(x_scal64.as_slice());
    });
    let re = bench_ns(ITERS_L1, || {
        DefaultBlas::scal(3.0, &mut x_scal_ref64);
        black_box(x_scal_ref64.as_slice());
    });
    report(&format!("Scal (f64, N={N_L1})"), diff_scal64, 1e-10, be, re);

    let y_dot64 = ArrayStorage::<f64, N_L1, 1>::from_array([y_init64]);
    let dot64 = Avx2Blas::dotu(&x_s64, &y_dot64);
    let dot_ref64 = DefaultBlas::dotu(&x_s64, &y_dot64);
    let diff_dot64 = (dot64 - dot_ref64).abs();
    let be = bench_ns(ITERS_L1, || {
        black_box(Avx2Blas::dotu(&x_s64, &y_dot64));
    });
    let re = bench_ns(ITERS_L1, || {
        black_box(DefaultBlas::dotu(&x_s64, &y_dot64));
    });
    report(&format!("Dotu (f64, N={N_L1})"), diff_dot64, 1e-9, be, re);

    let nrm64 = Avx2Blas::nrm2(&x_s64);
    let nrm_ref64 = DefaultBlas::nrm2(&x_s64);
    let diff_nrm64 = (nrm64 - nrm_ref64).abs();
    let be = bench_ns(ITERS_L1, || {
        black_box(Avx2Blas::nrm2(&x_s64));
    });
    let re = bench_ns(ITERS_L1, || {
        black_box(DefaultBlas::nrm2(&x_s64));
    });
    report(&format!("Nrm2 (f64, N={N_L1})"), diff_nrm64, 1e-9, be, re);

    let a_gemv64 = fill_mat_f64::<GV_M, GV_N>();
    let vx64 = fill_vec_f64::<GV_N>();
    let a_mat64 = RowArrayStorage::<f64, GV_M, GV_N>::from_array(a_gemv64);
    let vx_s64 = ArrayStorage::<f64, GV_N, 1>::from_array([vx64]);
    let mut vy_avx64 =
        ArrayStorage::<f64, GV_M, 1>::from_array([fill_vec_f64::<GV_M>()]);
    let mut vy_ref64 =
        ArrayStorage::<f64, GV_M, 1>::from_array([fill_vec_f64::<GV_M>()]);
    Avx2Blas::gemv(Trans::NoTrans, 1.5, &a_mat64, &vx_s64, 0.5, &mut vy_avx64);
    DefaultBlas::gemv(
        Trans::NoTrans,
        1.5,
        &a_mat64,
        &vx_s64,
        0.5,
        &mut vy_ref64,
    );
    let diff_gemv64 =
        max_abs_diff_f64(vy_avx64.as_slice(), vy_ref64.as_slice());
    let be = bench_ns(ITERS_L2, || {
        Avx2Blas::gemv(
            Trans::NoTrans,
            1.5,
            &a_mat64,
            &vx_s64,
            0.5,
            &mut vy_avx64,
        );
        black_box(vy_avx64.as_slice());
    });
    let re = bench_ns(ITERS_L2, || {
        DefaultBlas::gemv(
            Trans::NoTrans,
            1.5,
            &a_mat64,
            &vx_s64,
            0.5,
            &mut vy_ref64,
        );
        black_box(vy_ref64.as_slice());
    });
    report(
        &format!("Gemv NoTrans (f64, {GV_M}x{GV_N})"),
        diff_gemv64,
        1e-9,
        be,
        re,
    );

    let a_gemm64 = fill_mat_f64::<GM_M, GM_K>();
    let b_gemm64 = fill_mat_f64::<GM_K, GM_N>();
    let a_s64 = RowArrayStorage::<f64, GM_M, GM_K>::from_array(a_gemm64);
    let b_s64 = RowArrayStorage::<f64, GM_K, GM_N>::from_array(b_gemm64);
    let mut c_avx64 =
        RowArrayStorage::<f64, GM_M, GM_N>::from_array([[1.0f64; GM_N]; GM_M]);
    let mut c_ref64 =
        RowArrayStorage::<f64, GM_M, GM_N>::from_array([[1.0f64; GM_N]; GM_M]);
    Avx2Blas::gemm(
        Trans::NoTrans,
        Trans::NoTrans,
        1.2,
        &a_s64,
        &b_s64,
        0.8,
        &mut c_avx64,
    );
    DefaultBlas::gemm(
        Trans::NoTrans,
        Trans::NoTrans,
        1.2,
        &a_s64,
        &b_s64,
        0.8,
        &mut c_ref64,
    );
    let diff_gemm64 = max_abs_diff_f64(c_avx64.as_slice(), c_ref64.as_slice());
    let be = bench_ns(ITERS_L3, || {
        Avx2Blas::gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            1.2,
            &a_s64,
            &b_s64,
            0.8,
            &mut c_avx64,
        );
        black_box(c_avx64.as_slice());
    });
    let re = bench_ns(ITERS_L3, || {
        DefaultBlas::gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            1.2,
            &a_s64,
            &b_s64,
            0.8,
            &mut c_ref64,
        );
        black_box(c_ref64.as_slice());
    });
    report(
        &format!("Gemm NoTrans (f64, {GM_M}x{GM_K} * {GM_K}x{GM_N})"),
        diff_gemm64,
        1e-8,
        be,
        re,
    );

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
    let mut vy_sub_avx = ArrayStorage::<f32, 8, 1>::from_array([[0.0f32; 8]]);
    let mut vy_sub_ref = ArrayStorage::<f32, 8, 1>::from_array([[0.0f32; 8]]);

    Avx2Blas::gemv(
        Trans::NoTrans,
        1.0,
        &subview_a,
        &vx_sub,
        0.0,
        &mut vy_sub_avx,
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
        max_abs_diff(vy_sub_avx.as_slice(), vy_sub_ref.as_slice());
    let be = bench_ns(ITERS_L2, || {
        Avx2Blas::gemv(
            Trans::NoTrans,
            1.0,
            &subview_a,
            &vx_sub,
            0.0,
            &mut vy_sub_avx,
        );
        black_box(vy_sub_avx.as_slice());
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

    #[cfg(feature = "cblas")]
    {
        println!("\n--- Testing CblasBlas ---");
        let fresh_x =
            ArrayStorage::<f32, N_L1, 1>::from_array([fill_vec_f32::<N_L1>()]);
        let y0 = fill_vec_f32::<N_L1>();
        let mut fresh_y_cblas = ArrayStorage::<f32, N_L1, 1>::from_array([y0]);
        let mut fresh_y_ref = ArrayStorage::<f32, N_L1, 1>::from_array([y0]);

        CblasBlas::axpy(2.5, &fresh_x, &mut fresh_y_cblas);
        DefaultBlas::axpy(2.5, &fresh_x, &mut fresh_y_ref);
        let diff_cblas =
            max_abs_diff(fresh_y_cblas.as_slice(), fresh_y_ref.as_slice());
        let be = bench_ns(ITERS_L1, || {
            CblasBlas::axpy(2.5, &fresh_x, &mut fresh_y_cblas);
            black_box(fresh_y_cblas.as_slice());
        });
        let re = bench_ns(ITERS_L1, || {
            DefaultBlas::axpy(2.5, &fresh_x, &mut fresh_y_ref);
            black_box(fresh_y_ref.as_slice());
        });
        report(&format!("CBLAS Axpy (N={N_L1})"), diff_cblas, 1e-4, be, re);

        let a = RowArrayStorage::<f32, GM_M, GM_K>::from_array(fill_mat_f32::<
            GM_M,
            GM_K,
        >());
        let b = RowArrayStorage::<f32, GM_K, GM_N>::from_array(fill_mat_f32::<
            GM_K,
            GM_N,
        >());
        let mut c_cblas = RowArrayStorage::<f32, GM_M, GM_N>::from_array(
            [[0.0f32; GM_N]; GM_M],
        );
        let mut c_ref = RowArrayStorage::<f32, GM_M, GM_N>::from_array(
            [[0.0f32; GM_N]; GM_M],
        );
        CblasBlas::gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            1.0,
            &a,
            &b,
            0.0,
            &mut c_cblas,
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
        let diff_g = max_abs_diff(c_cblas.as_slice(), c_ref.as_slice());
        let be = bench_ns(ITERS_L3, || {
            CblasBlas::gemm(
                Trans::NoTrans,
                Trans::NoTrans,
                1.0,
                &a,
                &b,
                0.0,
                &mut c_cblas,
            );
            black_box(c_cblas.as_slice());
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
            &format!("CBLAS Gemm ({GM_M}x{GM_K} * {GM_K}x{GM_N})"),
            diff_g,
            5e-3,
            be,
            re,
        );
    }

    println!("\nAll x86_64 subprogram backend equivalence checks PASSED.");
}
