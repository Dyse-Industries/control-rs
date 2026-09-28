//! Reference-oracle tests for the `DefaultBlas` kernels.
//!
//! Each case compares a kernel against a dense reference computed with plain
//! complex arithmetic, over both vector orientations, every `Trans`/`UpLo`/
//! `Side`/`Diag` combination, and non-trivial `alpha`/`beta`. Inputs place
//! sentinel values in the triangle a kernel must not read, and use non-square
//! operands where the kernel dimensions depend on `Trans`.

use crate::math::complex_num::{Complex, Complex64};
use crate::math::num_types::{Const, Dim};
use crate::math::ops::{SaturatingAdd, SaturatingMul};
use crate::math::storage::{
    ArrayCooStorage, ArrayCscStorage, ArrayCsrStorage, ArraySparseVector,
    ArrayStorage, DenseStorage, DenseStorageMut, Diag, HermitianPackedStorage,
    PackedStorage, PackedStorageMut, Side, StorageInit, SymmetricPackedStorage,
    ToCscStorage, ToCsrStorage, Trans, TriangularPackedStorage, UpLo,
};
use crate::math::subprograms::{
    DefaultBlas,
    lapack::{Getrf, Getrs},
    level1::Iamax,
    level2::{Gemv, Gerc, Hemv, Her, Her2, Symv, Syr, Syr2, Trmv, Trsv},
    level3::{Gemm, Hemm, Her2k, Herk, Symm, Syr2k, Syrk, Trmm, Trsm},
    packed::{Hpmv, Hpr, Hpr2, Spmv, Spr, Spr2, Tpmv, Tpsv},
    sparse::{Cscmv, Csrmv, SpAxpy},
};

const DIAGS: [Diag; 2] = [Diag::Unit, Diag::NonUnit];
const SIDES: [Side; 2] = [Side::Left, Side::Right];
const TRANS: [Trans; 3] = [Trans::NoTrans, Trans::Trans, Trans::ConjTrans];
const UPLOS: [UpLo; 2] = [UpLo::Upper, UpLo::Lower];

type C = Complex64;
type Row = [C; 4];
type V3 = [C; 3];
type VCol = ArrayStorage<C, 3, 1>;
type VRow = ArrayStorage<C, 1, 3>;
type SparseFixture = (Mat, ArrayCooStorage<C, 3, 3, 9>);

trait Vec3: DenseStorageMut<C> {
    fn make(v: V3) -> Self;
    fn read(&self) -> V3;
}

#[derive(Clone, Copy)]
struct Mat {
    r: usize,
    c: usize,
    d: [Row; 4],
}

/// Runs `$f::<X, Y>($args)` over all four vector-orientation pairs.
macro_rules! orient2 {
    ($f:ident($($arg:expr),*)) => {{
        $f::<VCol, VCol>($($arg),*);
        $f::<VCol, VRow>($($arg),*);
        $f::<VRow, VCol>($($arg),*);
        $f::<VRow, VRow>($($arg),*);
    }};
}

macro_rules! orient1 {
    ($f:ident($($arg:expr),*)) => {{
        $f::<VCol>($($arg),*);
        $f::<VRow>($($arg),*);
    }};
}

/// Invokes `$call` with `$sa`/`$sb` bound to correctly shaped operands.
macro_rules! with_rank_stores {
    ($trans:expr, $a:expr, $b:expr, |$sa:ident, $sb:ident| $call:expr) => {
        if $trans == Trans::NoTrans {
            let $sa = to_store::<3, 2>(&$a);
            let $sb = to_store::<3, 2>(&$b);
            $call;
        } else {
            let $sa = to_store::<2, 3>(&$a);
            let $sb = to_store::<2, 3>(&$b);
            $call;
        }
    };
}

impl Mat {
    fn from_fn(r: usize, cc: usize, f: impl Fn(usize, usize) -> C) -> Self {
        let mut m = Self {
            r,
            c: cc,
            d: [[zero(); 4]; 4],
        };
        for i in 0..r {
            for j in 0..cc {
                m.put(i, j, f(i, j));
            }
        }
        m
    }

    fn at(&self, i: usize, j: usize) -> C {
        self.d
            .get(i)
            .and_then(|row| row.get(j))
            .copied()
            .unwrap_or_else(zero)
    }

    fn put(&mut self, i: usize, j: usize, v: C) {
        if let Some(slot) = self.d.get_mut(i).and_then(|row| row.get_mut(j)) {
            *slot = v;
        }
    }

    fn sample(r: usize, cc: usize, salt: f64) -> Self {
        Self::from_fn(r, cc, |i, j| {
            let (i, j) = (fl(i), fl(j));
            c(
                1.0 + 0.7 * i + 1.3 * j + 0.35 * salt + 0.11 * i * j,
                0.5 - 0.4 * i + 0.9 * j + 0.2 * salt - 0.13 * i * j,
            )
        })
    }

    /// Diagonally dominant so triangular solves are well conditioned.
    fn sample_dominant(n: usize, salt: f64) -> Self {
        let m = Self::sample(n, n, salt);
        Self::from_fn(n, n, |i, j| {
            if i == j {
                c(6.0 + fl(i), 1.0 - 0.5 * fl(i))
            } else {
                let v = m.at(i, j);
                c(v.re * 0.3, v.im * 0.3)
            }
        })
    }

    fn t(&self) -> Self {
        Self::from_fn(self.c, self.r, |i, j| self.at(j, i))
    }

    fn h(&self) -> Self {
        Self::from_fn(self.c, self.r, |i, j| self.at(j, i).conj())
    }

    fn op(&self, trans: Trans) -> Self {
        match trans {
            Trans::NoTrans => *self,
            Trans::Trans => self.t(),
            Trans::ConjTrans => self.h(),
        }
    }

    fn mul(&self, o: &Self) -> Self {
        assert_eq!(self.c, o.r);
        Self::from_fn(self.r, o.c, |i, j| {
            (0..self.c)
                .fold(zero(), |acc, k| add(acc, mul(self.at(i, k), o.at(k, j))))
        })
    }

    fn add(&self, o: &Self) -> Self {
        Self::from_fn(self.r, self.c, |i, j| add(self.at(i, j), o.at(i, j)))
    }

    fn scale(&self, s: C) -> Self {
        Self::from_fn(self.r, self.c, |i, j| mul(s, self.at(i, j)))
    }

    /// Keeps the `uplo` triangle and replaces the rest with sentinels.
    fn poisoned(&self, uplo: UpLo) -> Self {
        Self::from_fn(self.r, self.c, |i, j| {
            if in_tri(uplo, i, j) {
                self.at(i, j)
            } else {
                sentinel(i, j)
            }
        })
    }

    /// Full symmetric matrix generated by the `uplo` triangle.
    fn sym_from(&self, uplo: UpLo) -> Self {
        Self::from_fn(self.r, self.c, |i, j| {
            if in_tri(uplo, i, j) {
                self.at(i, j)
            } else {
                self.at(j, i)
            }
        })
    }

    /// Full Hermitian matrix generated by the `uplo` triangle (real diagonal).
    fn herm_from(&self, uplo: UpLo) -> Self {
        Self::from_fn(self.r, self.c, |i, j| {
            if i == j {
                c(self.at(i, j).re, 0.0)
            } else if in_tri(uplo, i, j) {
                self.at(i, j)
            } else {
                self.at(j, i).conj()
            }
        })
    }

    /// `op(T)` where `T` is the `uplo` triangle of `self` with optional unit diagonal.
    fn tri_op(&self, uplo: UpLo, trans: Trans, diag: Diag) -> Self {
        Self::from_fn(self.r, self.c, |i, j| {
            if !in_tri(uplo, i, j) {
                zero()
            } else if diag == Diag::Unit && i == j {
                one()
            } else {
                self.at(i, j)
            }
        })
        .op(trans)
    }

    /// Applies `f` to the entries inside the `uplo` triangle only.
    fn map_tri(&self, uplo: UpLo, f: impl Fn(usize, usize, C) -> C) -> Self {
        Self::from_fn(self.r, self.c, |i, j| {
            if in_tri(uplo, i, j) {
                f(i, j, self.at(i, j))
            } else {
                self.at(i, j)
            }
        })
    }

    fn col(v: &V3) -> Self {
        Self::from_fn(3, 1, |i, _| at3(v, i))
    }

    fn into_vec(self) -> V3 {
        [self.at(0, 0), self.at(1, 0), self.at(2, 0)]
    }
}

impl Vec3 for VCol {
    fn make(v: V3) -> Self {
        let mut s = Self::zeros();
        for (i, val) in v.iter().enumerate() {
            unsafe { s.set_unchecked(i, 0, *val) };
        }
        s
    }

    fn read(&self) -> V3 {
        core::array::from_fn(|i| unsafe { *self.get_unchecked(i, 0) })
    }
}

impl Vec3 for VRow {
    fn make(v: V3) -> Self {
        let mut s = Self::zeros();
        for (i, val) in v.iter().enumerate() {
            unsafe { s.set_unchecked(0, i, *val) };
        }
        s
    }

    fn read(&self) -> V3 {
        core::array::from_fn(|i| unsafe { *self.get_unchecked(0, i) })
    }
}

fn c(re: f64, im: f64) -> C {
    Complex::new(re, im)
}

fn zero() -> C {
    c(0.0, 0.0)
}

fn one() -> C {
    c(1.0, 0.0)
}

fn betas() -> [C; 3] {
    [zero(), one(), c(2.0, -1.0)]
}

fn alpha() -> C {
    c(1.5, -0.5)
}

fn add(a: C, b: C) -> C {
    a.saturating_add(&b)
}

fn mul(a: C, b: C) -> C {
    a.saturating_mul(&b)
}

fn mul3(a: C, b: C, d: C) -> C {
    mul(mul(a, b), d)
}

fn fl(i: usize) -> f64 {
    f64::from(u8::try_from(i).unwrap())
}

fn close(a: C, b: C) -> bool {
    (a.re - b.re).abs() < 1e-9 && (a.im - b.im).abs() < 1e-9
}

fn in_tri(uplo: UpLo, i: usize, j: usize) -> bool {
    match uplo {
        UpLo::Upper => i <= j,
        UpLo::Lower => i >= j,
    }
}

fn sentinel(r: usize, k: usize) -> C {
    c(700.0 + fl(r), -500.0 - fl(k))
}

fn at3(v: &V3, i: usize) -> C {
    v.get(i).copied().unwrap_or_else(zero)
}

fn xv() -> V3 {
    [c(1.0, 0.5), c(-2.0, 1.0), c(0.5, -3.0)]
}

fn yv() -> V3 {
    [c(2.0, -1.0), c(0.25, 3.0), c(-1.0, 1.5)]
}

fn assert_vec(got: &V3, want: &V3, what: &str) {
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(close(*g, *w), "{what}: [{i}] got {g:?} want {w:?}");
    }
}

// ---------------------------------------------------------------------------
// Dense reference matrices (at most 4x4).
// ---------------------------------------------------------------------------

fn assert_mat(got: &Mat, want: &Mat, what: &str) {
    assert_eq!((got.r, got.c), (want.r, want.c), "{what}: shape");
    for i in 0..got.r {
        for j in 0..got.c {
            assert!(
                close(got.at(i, j), want.at(i, j)),
                "{what}: ({i},{j}) got {:?} want {:?}",
                got.at(i, j),
                want.at(i, j)
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Storage bridges.
// ---------------------------------------------------------------------------

fn to_store<const R: usize, const K: usize>(m: &Mat) -> ArrayStorage<C, R, K>
where
    Const<R>: Dim,
    Const<K>: Dim,
{
    let mut s = ArrayStorage::<C, R, K>::zeros();
    for i in 0..R {
        for j in 0..K {
            unsafe { s.set_unchecked(i, j, m.at(i, j)) };
        }
    }
    s
}

fn from_store<const R: usize, const K: usize>(s: &ArrayStorage<C, R, K>) -> Mat
where
    Const<R>: Dim,
    Const<K>: Dim,
{
    Mat::from_fn(R, K, |i, j| unsafe { *s.get_unchecked(i, j) })
}

// ---------------------------------------------------------------------------
// Packed builders.
// ---------------------------------------------------------------------------

fn sym_packed(m: &Mat, uplo: UpLo) -> SymmetricPackedStorage<C, 3, 6> {
    let mut s = SymmetricPackedStorage::<C, 3, 6>::new([zero(); 6], uplo);
    for i in 0..3 {
        for j in 0..3 {
            if in_tri(uplo, i, j) {
                s.set(i, j, m.at(i, j)).unwrap();
            }
        }
    }
    s
}

fn herm_packed(m: &Mat, uplo: UpLo) -> HermitianPackedStorage<C, 3, 6> {
    let mut s = HermitianPackedStorage::<C, 3, 6>::new([zero(); 6], uplo);
    for i in 0..3 {
        for j in 0..3 {
            if in_tri(uplo, i, j) {
                s.set(i, j, m.at(i, j)).unwrap();
            }
        }
    }
    s
}

fn tri_packed(
    m: &Mat,
    uplo: UpLo,
    diag: Diag,
) -> TriangularPackedStorage<C, 3, 6> {
    let mut s =
        TriangularPackedStorage::<C, 3, 6>::new([zero(); 6], uplo, diag);
    for i in 0..3 {
        for j in 0..3 {
            if in_tri(uplo, i, j) && !(diag == Diag::Unit && i == j) {
                s.set(i, j, m.at(i, j)).unwrap();
            }
        }
    }
    s
}

fn packed_dense<P: PackedStorage<C>>(p: &P) -> Mat {
    Mat::from_fn(3, 3, |i, j| p.value_unchecked(i, j))
}

fn assert_packed_tri<P: PackedStorage<C>>(
    p: &P,
    uplo: UpLo,
    want: &Mat,
    what: &str,
) {
    for i in 0..3 {
        for j in 0..3 {
            if in_tri(uplo, i, j) {
                assert!(
                    close(p.value_unchecked(i, j), want.at(i, j)),
                    "{what}: ({i},{j}) got {:?} want {:?}",
                    p.value_unchecked(i, j),
                    want.at(i, j)
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Level 2 dense.
// ---------------------------------------------------------------------------

/// `alpha * product + beta * y0` for a column-vector product.
fn axpby(product: &Mat, beta: C) -> V3 {
    product
        .scale(alpha())
        .add(&Mat::col(&yv()).scale(beta))
        .into_vec()
}

fn gemv_case<X: Vec3, Y: Vec3>(trans: Trans, beta: C) {
    let a = Mat::sample(3, 3, 0.0);
    let mut y = Y::make(yv());
    DefaultBlas::gemv(
        trans,
        alpha(),
        &to_store::<3, 3>(&a),
        &X::make(xv()),
        beta,
        &mut y,
    );
    let ax = a.op(trans).mul(&Mat::col(&xv()));
    assert_vec(&y.read(), &axpby(&ax, beta), "gemv");
}

#[test]
fn test_kernel_gemv_all_variants() {
    for &t in &TRANS {
        for &b in &betas() {
            orient2!(gemv_case(t, b));
        }
    }
}

fn gerc_case<X: Vec3, Y: Vec3>() {
    let a0 = Mat::sample(3, 3, 1.0);
    let mut a = to_store::<3, 3>(&a0);
    DefaultBlas::gerc(alpha(), &X::make(xv()), &Y::make(yv()), &mut a);
    let outer = Mat::col(&xv()).mul(&Mat::col(&yv()).h());
    assert_mat(&from_store(&a), &a0.add(&outer.scale(alpha())), "gerc");
}

#[test]
fn test_kernel_gerc_orientations() {
    orient2!(gerc_case());
}

fn symv_case<X: Vec3, Y: Vec3>(uplo: UpLo, beta: C) {
    let full = Mat::sample(3, 3, 2.0);
    let a = full.poisoned(uplo);
    let mut y = Y::make(yv());
    DefaultBlas::symv(
        uplo,
        alpha(),
        &to_store::<3, 3>(&a),
        &X::make(xv()),
        beta,
        &mut y,
    );
    let ax = full.sym_from(uplo).mul(&Mat::col(&xv()));
    assert_vec(&y.read(), &axpby(&ax, beta), "symv");
}

fn hemv_case<X: Vec3, Y: Vec3>(uplo: UpLo, beta: C) {
    let herm = Mat::sample(3, 3, 2.0).herm_from(uplo);
    // The kernel reads the stored diagonal as-is, so the stored diagonal is real.
    let a = herm.poisoned(uplo);
    let mut y = Y::make(yv());
    DefaultBlas::hemv(
        uplo,
        alpha(),
        &to_store::<3, 3>(&a),
        &X::make(xv()),
        beta,
        &mut y,
    );
    let ax = herm.mul(&Mat::col(&xv()));
    assert_vec(&y.read(), &axpby(&ax, beta), "hemv");
}

#[test]
fn test_kernel_symv_hemv_all_variants() {
    for &u in &UPLOS {
        for &b in &betas() {
            orient2!(symv_case(u, b));
            orient2!(hemv_case(u, b));
        }
    }
}

/// Initial `A` with the `uplo` triangle from `sample` and sentinels elsewhere.
fn tri_init(uplo: UpLo, salt: f64) -> Mat {
    Mat::sample(3, 3, salt).poisoned(uplo)
}

fn syr_case<X: Vec3>(uplo: UpLo) {
    let a0 = tri_init(uplo, 3.0);
    let mut a = to_store::<3, 3>(&a0);
    DefaultBlas::syr(uplo, alpha(), &X::make(xv()), &mut a);
    let x = xv();
    let want = a0.map_tri(uplo, |i, j, v| {
        add(v, mul3(alpha(), at3(&x, i), at3(&x, j)))
    });
    assert_mat(&from_store(&a), &want, "syr");
}

fn syr2_case<X: Vec3, Y: Vec3>(uplo: UpLo) {
    let a0 = tri_init(uplo, 3.0);
    let mut a = to_store::<3, 3>(&a0);
    DefaultBlas::syr2(uplo, alpha(), &X::make(xv()), &Y::make(yv()), &mut a);
    let (x, y) = (xv(), yv());
    let want = a0.map_tri(uplo, |i, j, v| {
        let fwd = mul3(alpha(), at3(&x, i), at3(&y, j));
        let bwd = mul3(alpha(), at3(&y, i), at3(&x, j));
        add(add(v, fwd), bwd)
    });
    assert_mat(&from_store(&a), &want, "syr2");
}

fn her_case<X: Vec3>(uplo: UpLo) {
    let a0 = tri_init(uplo, 4.0);
    let mut a = to_store::<3, 3>(&a0);
    DefaultBlas::her(uplo, 1.5, &X::make(xv()), &mut a);
    let x = xv();
    let want = a0.map_tri(uplo, |i, j, v| {
        add(v, mul3(c(1.5, 0.0), at3(&x, i), at3(&x, j).conj()))
    });
    assert_mat(&from_store(&a), &want, "her");
}

fn her2_case<X: Vec3, Y: Vec3>(uplo: UpLo) {
    let a0 = tri_init(uplo, 4.0);
    let mut a = to_store::<3, 3>(&a0);
    DefaultBlas::her2(uplo, alpha(), &X::make(xv()), &Y::make(yv()), &mut a);
    let (x, y) = (xv(), yv());
    let want = a0.map_tri(uplo, |i, j, v| {
        let fwd = mul3(alpha(), at3(&x, i), at3(&y, j).conj());
        let bwd = mul3(alpha().conj(), at3(&y, i), at3(&x, j).conj());
        add(add(v, fwd), bwd)
    });
    assert_mat(&from_store(&a), &want, "her2");
}

#[test]
fn test_kernel_rank_updates_dense() {
    for &u in &UPLOS {
        orient1!(syr_case(u));
        orient2!(syr2_case(u));
        orient1!(her_case(u));
        orient2!(her2_case(u));
    }
}

fn trmv_case<X: Vec3>(uplo: UpLo, trans: Trans, diag: Diag) {
    let a = tri_init(uplo, 5.0);
    let mut x = X::make(xv());
    DefaultBlas::trmv(uplo, trans, diag, &to_store::<3, 3>(&a), &mut x);
    let want = a.tri_op(uplo, trans, diag).mul(&Mat::col(&xv()));
    assert_vec(&x.read(), &want.into_vec(), "trmv");
}

fn trsv_case<X: Vec3>(uplo: UpLo, trans: Trans, diag: Diag) {
    let a = Mat::sample_dominant(3, 5.0).poisoned(uplo);
    let mut x = X::make(xv());
    DefaultBlas::trsv(uplo, trans, diag, &to_store::<3, 3>(&a), &mut x)
        .unwrap();
    let back = a
        .tri_op(uplo, trans, diag)
        .mul(&Mat::col(&x.read()))
        .into_vec();
    assert_vec(&back, &xv(), "trsv");
}

#[test]
fn test_kernel_trmv_trsv_all_variants() {
    for &u in &UPLOS {
        for &t in &TRANS {
            for &d in &DIAGS {
                orient1!(trmv_case(u, t, d));
                orient1!(trsv_case(u, t, d));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Packed.
// ---------------------------------------------------------------------------

fn spmv_case<X: Vec3, Y: Vec3>(uplo: UpLo, beta: C) {
    let ap = sym_packed(&Mat::sample(3, 3, 6.0), uplo);
    let mut y = Y::make(yv());
    DefaultBlas::spmv(uplo, alpha(), &ap, &X::make(xv()), beta, &mut y);
    let ax = packed_dense(&ap).mul(&Mat::col(&xv()));
    assert_vec(&y.read(), &axpby(&ax, beta), "spmv");
}

fn hpmv_case<X: Vec3, Y: Vec3>(uplo: UpLo, beta: C) {
    let hp = herm_packed(&Mat::sample(3, 3, 6.0).herm_from(uplo), uplo);
    let mut y = Y::make(yv());
    DefaultBlas::hpmv(uplo, alpha(), &hp, &X::make(xv()), beta, &mut y);
    let ax = packed_dense(&hp).mul(&Mat::col(&xv()));
    assert_vec(&y.read(), &axpby(&ax, beta), "hpmv");
}

fn spr_case<X: Vec3>(uplo: UpLo) {
    let m = Mat::sample(3, 3, 7.0);
    let mut ap = sym_packed(&m, uplo);
    DefaultBlas::spr(uplo, alpha(), &X::make(xv()), &mut ap);
    let x = xv();
    let want = m.map_tri(uplo, |i, j, v| {
        add(v, mul3(alpha(), at3(&x, i), at3(&x, j)))
    });
    assert_packed_tri(&ap, uplo, &want, "spr");
}

fn spr2_case<X: Vec3, Y: Vec3>(uplo: UpLo) {
    let m = Mat::sample(3, 3, 7.0);
    let mut ap = sym_packed(&m, uplo);
    DefaultBlas::spr2(uplo, alpha(), &X::make(xv()), &Y::make(yv()), &mut ap);
    let (x, y) = (xv(), yv());
    let want = m.map_tri(uplo, |i, j, v| {
        let fwd = mul3(alpha(), at3(&x, i), at3(&y, j));
        let bwd = mul3(alpha(), at3(&y, i), at3(&x, j));
        add(add(v, fwd), bwd)
    });
    assert_packed_tri(&ap, uplo, &want, "spr2");
}

fn hpr_case<X: Vec3>(uplo: UpLo) {
    let m = Mat::sample(3, 3, 7.0).herm_from(uplo);
    let mut hp = herm_packed(&m, uplo);
    DefaultBlas::hpr(uplo, 1.5, &X::make(xv()), &mut hp);
    let x = xv();
    let want = m.map_tri(uplo, |i, j, v| {
        add(v, mul3(c(1.5, 0.0), at3(&x, i), at3(&x, j).conj()))
    });
    assert_packed_tri(&hp, uplo, &want, "hpr");
}

fn hpr2_case<X: Vec3, Y: Vec3>(uplo: UpLo) {
    let m = Mat::sample(3, 3, 7.0).herm_from(uplo);
    let mut hp = herm_packed(&m, uplo);
    DefaultBlas::hpr2(uplo, alpha(), &X::make(xv()), &Y::make(yv()), &mut hp);
    let (x, y) = (xv(), yv());
    let want = m.map_tri(uplo, |i, j, v| {
        let fwd = mul3(alpha(), at3(&x, i), at3(&y, j).conj());
        let bwd = mul3(alpha().conj(), at3(&y, i), at3(&x, j).conj());
        add(add(v, fwd), bwd)
    });
    assert_packed_tri(&hp, uplo, &want, "hpr2");
}

fn tpmv_case<X: Vec3>(uplo: UpLo, trans: Trans, diag: Diag) {
    let m = Mat::sample(3, 3, 8.0);
    let tp = tri_packed(&m, uplo, diag);
    let mut x = X::make(xv());
    DefaultBlas::tpmv(uplo, trans, diag, &tp, &mut x);
    let want = m.tri_op(uplo, trans, diag).mul(&Mat::col(&xv()));
    assert_vec(&x.read(), &want.into_vec(), "tpmv");
}

fn tpsv_case<X: Vec3>(uplo: UpLo, trans: Trans, diag: Diag) {
    let m = Mat::sample_dominant(3, 8.0);
    let tp = tri_packed(&m, uplo, diag);
    let mut x = X::make(xv());
    DefaultBlas::tpsv(uplo, trans, diag, &tp, &mut x).unwrap();
    let back = m
        .tri_op(uplo, trans, diag)
        .mul(&Mat::col(&x.read()))
        .into_vec();
    assert_vec(&back, &xv(), "tpsv");
}

#[test]
fn test_kernel_packed_all_variants() {
    for &u in &UPLOS {
        for &b in &betas() {
            orient2!(spmv_case(u, b));
            orient2!(hpmv_case(u, b));
        }
        orient1!(spr_case(u));
        orient2!(spr2_case(u));
        orient1!(hpr_case(u));
        orient2!(hpr2_case(u));
        for &t in &TRANS {
            for &d in &DIAGS {
                orient1!(tpmv_case(u, t, d));
                orient1!(tpsv_case(u, t, d));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Level 3 (non-square operands so `Trans` changes the shared dimension).
// ---------------------------------------------------------------------------

/// Order of the square `A` operand of a one-sided product with a 3x2 `B`.
fn side_order(side: Side) -> usize {
    match side {
        Side::Left => 3,
        Side::Right => 2,
    }
}

fn side_product(side: Side, a: &Mat, b: &Mat) -> Mat {
    match side {
        Side::Left => a.mul(b),
        Side::Right => b.mul(a),
    }
}

fn symm_case(side: Side, uplo: UpLo, beta: C) {
    let b = Mat::sample(3, 2, 1.0);
    let c0 = Mat::sample(3, 2, 2.0);
    let n = side_order(side);
    let full = Mat::sample(n, n, 3.0);
    let a = full.poisoned(uplo);
    let mut cm = to_store::<3, 2>(&c0);
    let bs = to_store::<3, 2>(&b);
    match side {
        Side::Left => DefaultBlas::symm(
            side,
            uplo,
            alpha(),
            &to_store::<3, 3>(&a),
            &bs,
            beta,
            &mut cm,
        ),
        Side::Right => DefaultBlas::symm(
            side,
            uplo,
            alpha(),
            &to_store::<2, 2>(&a),
            &bs,
            beta,
            &mut cm,
        ),
    }
    let prod = side_product(side, &full.sym_from(uplo), &b);
    let want = prod.scale(alpha()).add(&c0.scale(beta));
    assert_mat(&from_store(&cm), &want, "symm");
}

fn hemm_case(side: Side, uplo: UpLo, beta: C) {
    let b = Mat::sample(3, 2, 1.0);
    let c0 = Mat::sample(3, 2, 2.0);
    let n = side_order(side);
    let herm = Mat::sample(n, n, 3.0).herm_from(uplo);
    let a = herm.poisoned(uplo);
    let mut cm = to_store::<3, 2>(&c0);
    let bs = to_store::<3, 2>(&b);
    match side {
        Side::Left => DefaultBlas::hemm(
            side,
            uplo,
            alpha(),
            &to_store::<3, 3>(&a),
            &bs,
            beta,
            &mut cm,
        ),
        Side::Right => DefaultBlas::hemm(
            side,
            uplo,
            alpha(),
            &to_store::<2, 2>(&a),
            &bs,
            beta,
            &mut cm,
        ),
    }
    let prod = side_product(side, &herm, &b);
    let want = prod.scale(alpha()).add(&c0.scale(beta));
    assert_mat(&from_store(&cm), &want, "hemm");
}

#[test]
fn test_kernel_symm_hemm_all_variants() {
    for &s in &SIDES {
        for &u in &UPLOS {
            for &b in &betas() {
                symm_case(s, u, b);
                hemm_case(s, u, b);
            }
        }
    }
}

/// Returns `(stored, op-applied)` for a 3x2 (`NoTrans`) or 2x3 (otherwise) factor.
fn gemm_operand(t: Trans, salt: f64, op_rows: usize, op_cols: usize) -> Mat {
    if t == Trans::NoTrans {
        Mat::sample(op_rows, op_cols, salt)
    } else {
        Mat::sample(op_cols, op_rows, salt)
    }
}

fn gemm_case(ta: Trans, tb: Trans, beta: C) {
    let a = gemm_operand(ta, 1.0, 3, 2);
    let b = gemm_operand(tb, 2.0, 2, 3);
    let c0 = Mat::sample(3, 3, 3.0);
    let mut cm = to_store::<3, 3>(&c0);
    match (ta == Trans::NoTrans, tb == Trans::NoTrans) {
        (true, true) => DefaultBlas::gemm(
            ta,
            tb,
            alpha(),
            &to_store::<3, 2>(&a),
            &to_store::<2, 3>(&b),
            beta,
            &mut cm,
        ),
        (true, false) => DefaultBlas::gemm(
            ta,
            tb,
            alpha(),
            &to_store::<3, 2>(&a),
            &to_store::<3, 2>(&b),
            beta,
            &mut cm,
        ),
        (false, true) => DefaultBlas::gemm(
            ta,
            tb,
            alpha(),
            &to_store::<2, 3>(&a),
            &to_store::<2, 3>(&b),
            beta,
            &mut cm,
        ),
        (false, false) => DefaultBlas::gemm(
            ta,
            tb,
            alpha(),
            &to_store::<2, 3>(&a),
            &to_store::<3, 2>(&b),
            beta,
            &mut cm,
        ),
    }
    let want = a.op(ta).mul(&b.op(tb)).scale(alpha()).add(&c0.scale(beta));
    assert_mat(&from_store(&cm), &want, "gemm");
}

#[test]
fn test_kernel_gemm_all_variants() {
    for &ta in &TRANS {
        for &tb in &TRANS {
            for &b in &betas() {
                gemm_case(ta, tb, b);
            }
        }
    }
}

/// Stored `A` for a rank-k update of a 3x3 `C`: 3x2 for `NoTrans`, else 2x3.
fn rank_operand(trans: Trans, salt: f64) -> Mat {
    if trans == Trans::NoTrans {
        Mat::sample(3, 2, salt)
    } else {
        Mat::sample(2, 3, salt)
    }
}

fn syrk_case(uplo: UpLo, trans: Trans, beta: C) {
    let a = rank_operand(trans, 1.0);
    let c0 = tri_init(uplo, 2.0);
    let mut cm = to_store::<3, 3>(&c0);
    with_rank_stores!(trans, a, a, |sa, _sb| DefaultBlas::syrk(
        uplo,
        trans,
        alpha(),
        &sa,
        beta,
        &mut cm
    ));
    let prod = if trans == Trans::NoTrans {
        a.mul(&a.t())
    } else {
        a.t().mul(&a)
    };
    let want = c0.map_tri(uplo, |i, j, v| {
        add(mul(beta, v), mul(alpha(), prod.at(i, j)))
    });
    assert_mat(&from_store(&cm), &want, "syrk");
}

fn herk_case(uplo: UpLo, trans: Trans, beta: f64) {
    let a = rank_operand(trans, 1.0);
    let c0 = tri_init(uplo, 2.0);
    let mut cm = to_store::<3, 3>(&c0);
    with_rank_stores!(trans, a, a, |sa, _sb| DefaultBlas::herk(
        uplo, trans, 1.5, &sa, beta, &mut cm
    ));
    let prod = if trans == Trans::NoTrans {
        a.mul(&a.h())
    } else {
        a.h().mul(&a)
    };
    let want = c0.map_tri(uplo, |i, j, v| {
        add(mul(c(beta, 0.0), v), mul(c(1.5, 0.0), prod.at(i, j)))
    });
    assert_mat(&from_store(&cm), &want, "herk");
}

fn syr2k_case(uplo: UpLo, trans: Trans, beta: C) {
    let a = rank_operand(trans, 1.0);
    let b = rank_operand(trans, 3.0);
    let c0 = tri_init(uplo, 2.0);
    let mut cm = to_store::<3, 3>(&c0);
    with_rank_stores!(trans, a, b, |sa, sb| DefaultBlas::syr2k(
        uplo,
        trans,
        alpha(),
        &sa,
        &sb,
        beta,
        &mut cm
    ));
    let prod = if trans == Trans::NoTrans {
        a.mul(&b.t()).add(&b.mul(&a.t()))
    } else {
        a.t().mul(&b).add(&b.t().mul(&a))
    };
    let want = c0.map_tri(uplo, |i, j, v| {
        add(mul(beta, v), mul(alpha(), prod.at(i, j)))
    });
    assert_mat(&from_store(&cm), &want, "syr2k");
}

fn her2k_case(uplo: UpLo, trans: Trans, beta: f64) {
    let a = rank_operand(trans, 1.0);
    let b = rank_operand(trans, 3.0);
    let c0 = tri_init(uplo, 2.0);
    let mut cm = to_store::<3, 3>(&c0);
    with_rank_stores!(trans, a, b, |sa, sb| DefaultBlas::her2k(
        uplo,
        trans,
        alpha(),
        &sa,
        &sb,
        beta,
        &mut cm
    ));
    let (fwd, bwd) = if trans == Trans::NoTrans {
        (a.mul(&b.h()), b.mul(&a.h()))
    } else {
        (a.h().mul(&b), b.h().mul(&a))
    };
    let prod = fwd.scale(alpha()).add(&bwd.scale(alpha().conj()));
    let want =
        c0.map_tri(uplo, |i, j, v| add(mul(c(beta, 0.0), v), prod.at(i, j)));
    assert_mat(&from_store(&cm), &want, "her2k");
}

#[test]
fn test_kernel_rank_k_all_variants() {
    for &u in &UPLOS {
        for &b in &betas() {
            syrk_case(u, Trans::NoTrans, b);
            syrk_case(u, Trans::Trans, b);
            syr2k_case(u, Trans::NoTrans, b);
            syr2k_case(u, Trans::Trans, b);
        }
        for &b in &[0.0, 1.0, 2.5] {
            herk_case(u, Trans::NoTrans, b);
            herk_case(u, Trans::ConjTrans, b);
            her2k_case(u, Trans::NoTrans, b);
            her2k_case(u, Trans::ConjTrans, b);
        }
    }
}

fn trmm_case(side: Side, uplo: UpLo, trans: Trans, diag: Diag) {
    let b0 = Mat::sample(3, 2, 1.0);
    let n = side_order(side);
    let a = Mat::sample(n, n, 4.0).poisoned(uplo);
    let mut bm = to_store::<3, 2>(&b0);
    match side {
        Side::Left => DefaultBlas::trmm(
            side,
            uplo,
            trans,
            diag,
            alpha(),
            &to_store::<3, 3>(&a),
            &mut bm,
        ),
        Side::Right => DefaultBlas::trmm(
            side,
            uplo,
            trans,
            diag,
            alpha(),
            &to_store::<2, 2>(&a),
            &mut bm,
        ),
    }
    let prod = side_product(side, &a.tri_op(uplo, trans, diag), &b0);
    assert_mat(&from_store(&bm), &prod.scale(alpha()), "trmm");
}

fn trsm_case(side: Side, uplo: UpLo, trans: Trans, diag: Diag) {
    let b0 = Mat::sample(3, 2, 1.0);
    let n = side_order(side);
    let a = Mat::sample_dominant(n, 4.0).poisoned(uplo);
    let mut bm = to_store::<3, 2>(&b0);
    match side {
        Side::Left => DefaultBlas::trsm(
            side,
            uplo,
            trans,
            diag,
            alpha(),
            &to_store::<3, 3>(&a),
            &mut bm,
        ),
        Side::Right => DefaultBlas::trsm(
            side,
            uplo,
            trans,
            diag,
            alpha(),
            &to_store::<2, 2>(&a),
            &mut bm,
        ),
    }
    .unwrap();
    let back =
        side_product(side, &a.tri_op(uplo, trans, diag), &from_store(&bm));
    assert_mat(&back, &b0.scale(alpha()), "trsm");
}

#[test]
fn test_kernel_trmm_trsm_all_variants() {
    for &s in &SIDES {
        for &u in &UPLOS {
            for &t in &TRANS {
                for &d in &DIAGS {
                    trmm_case(s, u, t, d);
                    trsm_case(s, u, t, d);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Sparse.
// ---------------------------------------------------------------------------

fn sparse_fixture() -> SparseFixture {
    let entries = [(0, 0), (0, 2), (1, 1), (2, 0), (2, 1)];
    let base = Mat::sample(3, 3, 9.0);
    let mut coo = ArrayCooStorage::<C, 3, 3, 9>::new();
    let mut dense = Mat::from_fn(3, 3, |_, _| zero());
    for &(i, j) in &entries {
        coo.push(i, j, base.at(i, j)).unwrap();
        dense.put(i, j, base.at(i, j));
    }
    (dense, coo)
}

fn csrmv_case<X: Vec3, Y: Vec3>(beta: C) {
    let (d, coo) = sparse_fixture();
    let csr: ArrayCsrStorage<C, 3, 3, 9, 4> = coo.to_csr().unwrap();
    let mut y = Y::make(yv());
    DefaultBlas::csrmv(alpha(), &csr, &X::make(xv()), beta, &mut y);
    let ax = d.mul(&Mat::col(&xv()));
    assert_vec(&y.read(), &axpby(&ax, beta), "csrmv");
}

fn cscmv_case<X: Vec3, Y: Vec3>(beta: C) {
    let (d, coo) = sparse_fixture();
    let csc: ArrayCscStorage<C, 3, 3, 9, 4> = coo.to_csc().unwrap();
    let mut y = Y::make(yv());
    DefaultBlas::cscmv(alpha(), &csc, &X::make(xv()), beta, &mut y);
    let ax = d.mul(&Mat::col(&xv()));
    assert_vec(&y.read(), &axpby(&ax, beta), "cscmv");
}

fn sp_axpy_case<Y: Vec3>() {
    let (v0, v2) = (c(1.0, 1.0), c(-2.0, 0.5));
    let mut sv = ArraySparseVector::<C, 3, 3>::new();
    sv.push(0, v0).unwrap();
    sv.push(2, v2).unwrap();
    let mut y = Y::make(yv());
    DefaultBlas::sp_axpy(alpha(), &sv, &mut y);
    let base = yv();
    let want = [
        add(at3(&base, 0), mul(alpha(), v0)),
        at3(&base, 1),
        add(at3(&base, 2), mul(alpha(), v2)),
    ];
    assert_vec(&y.read(), &want, "sp_axpy");
}

#[test]
fn test_kernel_sparse_orientations() {
    for &b in &betas() {
        orient2!(csrmv_case(b));
        orient2!(cscmv_case(b));
    }
    orient1!(sp_axpy_case());
}

// ---------------------------------------------------------------------------
// Level 1 ties and LAPACK.
// ---------------------------------------------------------------------------

fn iamax_tie_case<X: Vec3>() {
    let x = X::make([c(1.0, 0.0), c(3.0, 0.0), c(0.0, 3.0)]);
    assert_eq!(DefaultBlas::iamax(&x), 1);
}

#[test]
fn test_kernel_iamax_first_of_ties() {
    orient1!(iamax_tie_case());
}

#[test]
fn test_kernel_getrf_pivot_first_of_ties() {
    let dom = Mat::sample_dominant(3, 1.0);
    // Column 0 has equal magnitudes, so the first row must be chosen as pivot.
    let m = Mat::from_fn(
        3,
        3,
        |i, j| {
            if j == 0 { c(2.0, 0.0) } else { dom.at(i, j) }
        },
    );
    let mut a = to_store::<3, 3>(&m);
    let mut ipiv = [7usize; 3];
    DefaultBlas::getrf(&mut a, &mut ipiv).unwrap();
    assert_eq!(ipiv.first().copied(), Some(0));
}

#[test]
fn test_kernel_getrs_transposed_solves() {
    let a0 = Mat::sample_dominant(3, 2.0);
    let x0 = Mat::col(&xv());
    for &t in &[Trans::Trans, Trans::ConjTrans] {
        let b0 = a0.op(t).mul(&x0);
        let mut lu = to_store::<3, 3>(&a0);
        let mut ipiv = [0usize; 3];
        DefaultBlas::getrf(&mut lu, &mut ipiv).unwrap();
        let mut b = to_store::<3, 1>(&b0);
        DefaultBlas::getrs(t, &lu, &ipiv, &mut b).unwrap();
        assert_mat(&from_store(&b), &x0, "getrs");
    }
}
