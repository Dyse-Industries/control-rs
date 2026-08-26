#![no_std]
#![no_main]

use core::hint::black_box;
use core::panic::PanicInfo;
use riscv_rt::entry;
use semihosting::println;

use control_rs::math::num_types::Const;
use control_rs::math::storage::{
    ArrayStorage, Diag, RowArrayStorage, Side, StorageView, Trans, UpLo,
};
use control_rs::math::subprograms::lapack::Potrf;
use control_rs::math::subprograms::level1::{Dotu, Scal};
use control_rs::math::subprograms::level2::Gemv;
use control_rs::math::subprograms::level3::{Gemm, Trsm};
use control_rs::math::subprograms::DefaultBlas;

use riscv32imac_subprograms::NmsisDspBlas;

const N_L1: usize = 64;
const GV: usize = 32;
const GM: usize = 32;
const FACT: usize = 8;
const RHS_N: usize = 2;

const ITERS_L1: u32 = 50;
const ITERS_L2: u32 = 20;
const ITERS_L3: u32 = 8;

fn fill_vec<const N: usize>() -> [f32; N] {
    let mut a = [0.0f32; N];
    let mut i = 0;
    while i < N {
        a[i] = (i % 17) as f32 * 0.125 + 0.5;
        i += 1;
    }
    a
}

fn fill_mat<const R: usize, const C: usize>() -> [[f32; C]; R] {
    let mut a = [[0.0f32; C]; R];
    let mut r = 0;
    while r < R {
        let mut c = 0;
        while c < C {
            a[r][c] = ((r + 3 * c) % 13) as f32 * 0.05 + 0.1;
            c += 1;
        }
        r += 1;
    }
    a
}

fn transpose<const R: usize, const C: usize>(
    a: &[[f32; C]; R],
) -> [[f32; R]; C] {
    let mut t = [[0.0f32; R]; C];
    let mut r = 0;
    while r < R {
        let mut c = 0;
        while c < C {
            t[c][r] = a[r][c];
            c += 1;
        }
        r += 1;
    }
    t
}

fn fill_spd<const N: usize>() -> [[f32; N]; N] {
    let mut a = [[0.0f32; N]; N];
    let mut r = 0;
    while r < N {
        let mut c = 0;
        while c < N {
            a[r][c] = if r == c { 10.0 } else { 1.0 };
            c += 1;
        }
        r += 1;
    }
    a
}

fn fill_upper<const N: usize>() -> [[f32; N]; N] {
    let mut a = [[0.0f32; N]; N];
    let mut r = 0;
    while r < N {
        let mut c = 0;
        while c < N {
            if r == c {
                a[r][c] = 4.0 + r as f32 * 0.5;
            } else if c > r {
                a[r][c] = 1.0;
            }
            c += 1;
        }
        r += 1;
    }
    a
}

fn fill_rhs<const R: usize, const C: usize>() -> [[f32; C]; R] {
    let mut a = [[0.0f32; C]; R];
    let mut r = 0;
    while r < R {
        let mut c = 0;
        while c < C {
            a[r][c] = (r + 2 * c + 1) as f32 * 0.5;
            c += 1;
        }
        r += 1;
    }
    a
}

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    semihosting::process::exit(1);
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    let mut max_d = 0.0f32;
    let mut i = 0;
    while i < a.len() {
        let d = libm::fabsf(a[i] - b[i]);
        if d > max_d {
            max_d = d;
        }
        i += 1;
    }
    max_d
}

fn rdcycle() -> u32 {
    let mut value: u32;
    unsafe {
        core::arch::asm!(
            "rdcycle {}",
            out(reg) value,
            options(nomem, nostack, preserves_flags),
        );
    }
    value
}

fn bench_cy(iters: u32, mut body: impl FnMut()) -> u32 {
    body();
    let start = rdcycle();
    let mut i = 0;
    while i < iters {
        body();
        i += 1;
    }
    rdcycle().wrapping_sub(start) / iters.max(1)
}

fn report(
    label: &str,
    diff: f32,
    limit: f32,
    backend_cy: u32,
    default_cy: u32,
) {
    let ratio = if backend_cy == 0 {
        0.0
    } else {
        f64::from(default_cy) / f64::from(backend_cy)
    };
    println!(
        "  {label}: max diff = {diff:.2e}  backend={backend_cy} cy/call  DefaultBlas={default_cy} cy/call  ({ratio:.2}x) ... OK"
    );
    assert!(diff < limit);
}

#[entry]
fn main() -> ! {
    println!("=== RISC-V 32 NMSIS-DSP Subprogram Verification ===");
    println!(
        "Shapes: L1 N={N_L1}, Gemv {GV}x{GV}, Gemm {GM}x{GM}x{GM}, Potrf/Trsm {FACT}x{FACT}. Times averaged over {ITERS_L1}/{ITERS_L2}/{ITERS_L3} iters (L1/L2/L3). Soft-float QEMU is not a speedup claim."
    );

    {
        let x_arr = fill_vec::<N_L1>();
        let y_arr = fill_vec::<N_L1>();
        let mut x_nmsis = ArrayStorage::<f32, N_L1, 1>::from_array([x_arr]);
        let mut x_ref = ArrayStorage::<f32, N_L1, 1>::from_array([x_arr]);
        let y_s = ArrayStorage::<f32, N_L1, 1>::from_array([y_arr]);

        NmsisDspBlas::scal(2.5, &mut x_nmsis);
        DefaultBlas::scal(2.5, &mut x_ref);
        let diff_scal = max_abs_diff(x_nmsis.as_slice(), x_ref.as_slice());
        let be = bench_cy(ITERS_L1, || {
            NmsisDspBlas::scal(2.5, &mut x_nmsis);
            black_box(x_nmsis.as_slice());
        });
        let re = bench_cy(ITERS_L1, || {
            DefaultBlas::scal(2.5, &mut x_ref);
            black_box(x_ref.as_slice());
        });
        report("Scal (N=64)", diff_scal, 1e-5, be, re);

        let x_dot = ArrayStorage::<f32, N_L1, 1>::from_array([x_arr]);
        let dot_nmsis = NmsisDspBlas::dotu(&x_dot, &y_s);
        let dot_ref = DefaultBlas::dotu(&x_dot, &y_s);
        let diff_dot = libm::fabsf(dot_nmsis - dot_ref);
        let be = bench_cy(ITERS_L1, || {
            black_box(NmsisDspBlas::dotu(&x_dot, &y_s));
        });
        let re = bench_cy(ITERS_L1, || {
            black_box(DefaultBlas::dotu(&x_dot, &y_s));
        });
        report("Dotu (N=64)", diff_dot, 1e-4, be, re);
    }

    {
        let a_mat =
            RowArrayStorage::<f32, GV, GV>::from_array(fill_mat::<GV, GV>());
        let vx = ArrayStorage::<f32, GV, 1>::from_array([fill_vec::<GV>()]);
        let mut vy_nmsis =
            ArrayStorage::<f32, GV, 1>::from_array([[0.0f32; GV]]);
        let mut vy_ref = ArrayStorage::<f32, GV, 1>::from_array([[0.0f32; GV]]);

        NmsisDspBlas::gemv(
            Trans::NoTrans,
            1.0,
            &a_mat,
            &vx,
            0.0,
            &mut vy_nmsis,
        );
        DefaultBlas::gemv(Trans::NoTrans, 1.0, &a_mat, &vx, 0.0, &mut vy_ref);
        let diff_gemv = max_abs_diff(vy_nmsis.as_slice(), vy_ref.as_slice());
        let be = bench_cy(ITERS_L2, || {
            NmsisDspBlas::gemv(
                Trans::NoTrans,
                1.0,
                &a_mat,
                &vx,
                0.0,
                &mut vy_nmsis,
            );
            black_box(vy_nmsis.as_slice());
        });
        let re = bench_cy(ITERS_L2, || {
            DefaultBlas::gemv(
                Trans::NoTrans,
                1.0,
                &a_mat,
                &vx,
                0.0,
                &mut vy_ref,
            );
            black_box(vy_ref.as_slice());
        });
        report("Gemv NoTrans (32x32)", diff_gemv, 2e-4, be, re);

        let vx_t = ArrayStorage::<f32, GV, 1>::from_array([fill_vec::<GV>()]);
        let mut vy_t_nmsis =
            ArrayStorage::<f32, GV, 1>::from_array([[0.0f32; GV]]);
        let mut vy_t_ref =
            ArrayStorage::<f32, GV, 1>::from_array([[0.0f32; GV]]);
        NmsisDspBlas::gemv(
            Trans::Trans,
            1.5,
            &a_mat,
            &vx_t,
            0.0,
            &mut vy_t_nmsis,
        );
        DefaultBlas::gemv(Trans::Trans, 1.5, &a_mat, &vx_t, 0.0, &mut vy_t_ref);
        let diff_gemv_t =
            max_abs_diff(vy_t_nmsis.as_slice(), vy_t_ref.as_slice());
        let be = bench_cy(ITERS_L2, || {
            NmsisDspBlas::gemv(
                Trans::Trans,
                1.5,
                &a_mat,
                &vx_t,
                0.0,
                &mut vy_t_nmsis,
            );
            black_box(vy_t_nmsis.as_slice());
        });
        let re = bench_cy(ITERS_L2, || {
            DefaultBlas::gemv(
                Trans::Trans,
                1.5,
                &a_mat,
                &vx_t,
                0.0,
                &mut vy_t_ref,
            );
            black_box(vy_t_ref.as_slice());
        });
        report("Gemv Trans (32x32)", diff_gemv_t, 2e-4, be, re);
    }

    {
        let a_data = fill_mat::<GM, GM>();
        let b_data = fill_mat::<GM, GM>();
        let a_gemm_mat = RowArrayStorage::<f32, GM, GM>::from_array(a_data);
        let b_gemm_mat = RowArrayStorage::<f32, GM, GM>::from_array(b_data);
        let mut c_nmsis =
            RowArrayStorage::<f32, GM, GM>::from_array([[0.0f32; GM]; GM]);
        let mut c_ref =
            RowArrayStorage::<f32, GM, GM>::from_array([[0.0f32; GM]; GM]);

        NmsisDspBlas::gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            1.0,
            &a_gemm_mat,
            &b_gemm_mat,
            0.0,
            &mut c_nmsis,
        );
        DefaultBlas::gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            1.0,
            &a_gemm_mat,
            &b_gemm_mat,
            0.0,
            &mut c_ref,
        );
        let diff_gemm = max_abs_diff(c_nmsis.as_slice(), c_ref.as_slice());
        let be = bench_cy(ITERS_L3, || {
            NmsisDspBlas::gemm(
                Trans::NoTrans,
                Trans::NoTrans,
                1.0,
                &a_gemm_mat,
                &b_gemm_mat,
                0.0,
                &mut c_nmsis,
            );
            black_box(c_nmsis.as_slice());
        });
        let re = bench_cy(ITERS_L3, || {
            DefaultBlas::gemm(
                Trans::NoTrans,
                Trans::NoTrans,
                1.0,
                &a_gemm_mat,
                &b_gemm_mat,
                0.0,
                &mut c_ref,
            );
            black_box(c_ref.as_slice());
        });
        report("Gemm NoTrans (32x32 * 32x32)", diff_gemm, 5e-4, be, re);

        let a_t_mat =
            RowArrayStorage::<f32, GM, GM>::from_array(transpose(&a_data));
        let mut c_t_nmsis =
            RowArrayStorage::<f32, GM, GM>::from_array([[0.0f32; GM]; GM]);
        let mut c_t_ref =
            RowArrayStorage::<f32, GM, GM>::from_array([[0.0f32; GM]; GM]);
        NmsisDspBlas::gemm(
            Trans::Trans,
            Trans::NoTrans,
            1.0,
            &a_t_mat,
            &b_gemm_mat,
            0.0,
            &mut c_t_nmsis,
        );
        DefaultBlas::gemm(
            Trans::Trans,
            Trans::NoTrans,
            1.0,
            &a_t_mat,
            &b_gemm_mat,
            0.0,
            &mut c_t_ref,
        );
        let diff_gemm_t =
            max_abs_diff(c_t_nmsis.as_slice(), c_t_ref.as_slice());
        let be = bench_cy(ITERS_L3, || {
            NmsisDspBlas::gemm(
                Trans::Trans,
                Trans::NoTrans,
                1.0,
                &a_t_mat,
                &b_gemm_mat,
                0.0,
                &mut c_t_nmsis,
            );
            black_box(c_t_nmsis.as_slice());
        });
        let re = bench_cy(ITERS_L3, || {
            DefaultBlas::gemm(
                Trans::Trans,
                Trans::NoTrans,
                1.0,
                &a_t_mat,
                &b_gemm_mat,
                0.0,
                &mut c_t_ref,
            );
            black_box(c_t_ref.as_slice());
        });
        report("Gemm Trans A (32x32 * 32x32)", diff_gemm_t, 5e-4, be, re);
    }

    {
        let spd_data = fill_spd::<FACT>();
        let mut chol_nmsis =
            RowArrayStorage::<f32, FACT, FACT>::from_array(spd_data);
        let mut chol_ref =
            RowArrayStorage::<f32, FACT, FACT>::from_array(spd_data);

        let res_nmsis = NmsisDspBlas::potrf(UpLo::Lower, &mut chol_nmsis);
        let res_ref = DefaultBlas::potrf(UpLo::Lower, &mut chol_ref);
        assert!(res_nmsis.is_ok() && res_ref.is_ok());
        let diff_chol =
            max_abs_diff(chol_nmsis.as_slice(), chol_ref.as_slice());
        let be = bench_cy(ITERS_L3, || {
            let mut tmp =
                RowArrayStorage::<f32, FACT, FACT>::from_array(spd_data);
            let _ = black_box(NmsisDspBlas::potrf(UpLo::Lower, &mut tmp));
        });
        let re = bench_cy(ITERS_L3, || {
            let mut tmp =
                RowArrayStorage::<f32, FACT, FACT>::from_array(spd_data);
            let _ = black_box(DefaultBlas::potrf(UpLo::Lower, &mut tmp));
        });
        report("Potrf (8x8 SPD)", diff_chol, 1e-4, be, re);

        let u_data = fill_upper::<FACT>();
        let rhs_data = fill_rhs::<FACT, RHS_N>();
        let u_mat = RowArrayStorage::<f32, FACT, FACT>::from_array(u_data);
        let mut b_nmsis =
            RowArrayStorage::<f32, FACT, RHS_N>::from_array(rhs_data);
        let mut b_ref =
            RowArrayStorage::<f32, FACT, RHS_N>::from_array(rhs_data);

        let trsm_nmsis = NmsisDspBlas::trsm(
            Side::Left,
            UpLo::Upper,
            Trans::NoTrans,
            Diag::NonUnit,
            1.0,
            &u_mat,
            &mut b_nmsis,
        );
        let trsm_ref = DefaultBlas::trsm(
            Side::Left,
            UpLo::Upper,
            Trans::NoTrans,
            Diag::NonUnit,
            1.0,
            &u_mat,
            &mut b_ref,
        );
        assert!(trsm_nmsis.is_ok() && trsm_ref.is_ok());
        let diff_trsm = max_abs_diff(b_nmsis.as_slice(), b_ref.as_slice());
        let be = bench_cy(ITERS_L3, || {
            let mut tmp =
                RowArrayStorage::<f32, FACT, RHS_N>::from_array(rhs_data);
            let _ = black_box(NmsisDspBlas::trsm(
                Side::Left,
                UpLo::Upper,
                Trans::NoTrans,
                Diag::NonUnit,
                1.0,
                &u_mat,
                &mut tmp,
            ));
        });
        let re = bench_cy(ITERS_L3, || {
            let mut tmp =
                RowArrayStorage::<f32, FACT, RHS_N>::from_array(rhs_data);
            let _ = black_box(DefaultBlas::trsm(
                Side::Left,
                UpLo::Upper,
                Trans::NoTrans,
                Diag::NonUnit,
                1.0,
                &u_mat,
                &mut tmp,
            ));
        });
        report("Trsm (8x8 solve)", diff_trsm, 1e-4, be, re);
    }

    {
        let big_mat =
            RowArrayStorage::<f32, 8, 8>::from_array(fill_mat::<8, 8>());
        let subview = StorageView::<f32, Const<4>, Const<4>>::new_with_strides(
            big_mat.as_slice(),
            8,
            2,
        )
        .unwrap();
        let vx_sub = ArrayStorage::<f32, 4, 1>::from_array([fill_vec::<4>()]);
        let mut vy_sub_nmsis =
            ArrayStorage::<f32, 4, 1>::from_array([[0.0f32; 4]]);
        let mut vy_sub_ref =
            ArrayStorage::<f32, 4, 1>::from_array([[0.0f32; 4]]);

        NmsisDspBlas::gemv(
            Trans::NoTrans,
            1.0,
            &subview,
            &vx_sub,
            0.0,
            &mut vy_sub_nmsis,
        );
        DefaultBlas::gemv(
            Trans::NoTrans,
            1.0,
            &subview,
            &vx_sub,
            0.0,
            &mut vy_sub_ref,
        );
        let diff_strided =
            max_abs_diff(vy_sub_nmsis.as_slice(), vy_sub_ref.as_slice());
        let be = bench_cy(ITERS_L2, || {
            NmsisDspBlas::gemv(
                Trans::NoTrans,
                1.0,
                &subview,
                &vx_sub,
                0.0,
                &mut vy_sub_nmsis,
            );
            black_box(vy_sub_nmsis.as_slice());
        });
        let re = bench_cy(ITERS_L2, || {
            DefaultBlas::gemv(
                Trans::NoTrans,
                1.0,
                &subview,
                &vx_sub,
                0.0,
                &mut vy_sub_ref,
            );
            black_box(vy_sub_ref.as_slice());
        });
        report(
            "Gemv Strided Subview (4x4, stride=2)",
            diff_strided,
            1e-5,
            be,
            re,
        );
    }

    println!("\nAll RISC-V 32 NMSIS-DSP subprogram checks PASSED.");
    semihosting::process::exit(0);
}
