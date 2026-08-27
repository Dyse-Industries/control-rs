//! # Polynomial Module
//!
//! Generic, compile-time sized polynomial representation [`Polynomial<T, N, S>`]
//! decoupled from physical memory storage via [`crate::math::storage`].
//!
//! # Coefficient Layout
//! Coefficients are stored in **ascending order of powers**:
//! $$p(x) = c_0 + c_1 x + c_2 x^2 + \dots + c_{N-1} x^{N-1}$$
//! where index `i` maps to the coefficient of $x^i$.
//!
//! # Examples
//!
//! ```rust
//! use control_rs::math::num_types::Const;
//! use control_rs::polynomial::ArrayPolynomial;
//!
//! let p = ArrayPolynomial::<f64, 3>::from_coefficients([1.0, 2.0, 3.0]); // 1 + 2x + 3x^2
//! assert_eq!(p.evaluate(2.0), 1.0 + 2.0 * 2.0 + 3.0 * 4.0); // 1 + 4 + 12 = 17
//! assert_eq!(p.degree(), Some(2));
//! ```
#![allow(
    clippy::arbitrary_source_item_ordering,
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::similar_names,
    clippy::needless_range_loop,
    clippy::type_complexity,
    clippy::doc_markdown,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::option_if_let_else,
    clippy::must_use_candidate,
    clippy::many_single_char_names,
    clippy::collapsible_if,
    clippy::use_self,
    clippy::too_many_arguments,
    clippy::missing_const_for_fn,
    clippy::cast_lossless
)]

#[cfg(any(test, feature = "ets"))]
/// Polynomial module unit tests.
pub mod tests;

use crate::math::complex_num::Complex;
use crate::math::dsp::{Convolution, DefaultDsp};
use crate::math::num_traits::{Float, Scalar, Zero};
use crate::math::num_types::{Const, Dim};
use crate::math::ops::{Add, Neg, Sub};
use crate::math::storage::{
    ArrayStorage, ContiguousStorage, ContiguousStorageMut, DenseStorage,
    DenseStorageMut, Storage, StorageInit, StorageMut, StorageView,
    StorageViewMut,
};
use crate::math::subprograms::{DefaultBlas, level1::Axpy};
use crate::math::{ConversionError, ConversionResult};
use crate::matrix::Owned;
use core::convert::TryFrom;
use core::marker::PhantomData;

/// Errors returned by [`ArrayPolynomial::div_rem`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DivisionError {
    /// The divisor's leading coefficient is exactly zero.
    ZeroLeadingCoefficient,
    /// The divisor's degree exceeds the dividend's degree.
    DegreeMismatch,
}

/// Statically-typed polynomial over coefficient storage backend `S`.
///
/// `N` represents the maximum coefficient capacity (maximum degree $N - 1$).
/// `S` is the underlying storage backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Polynomial<T, N: Dim, S: Storage<T, N, Const<1>>> {
    storage: S,
    _marker: PhantomData<(T, N)>,
}

/// Owning stack-allocated polynomial.
pub type ArrayPolynomial<T, const N: usize> =
    Polynomial<T, Const<N>, ArrayStorage<T, N, 1>>;

/// Read-only borrowed polynomial view.
pub type PolynomialView<'a, T, N> =
    Polynomial<T, N, StorageView<'a, T, N, Const<1>>>;

/// Mutable borrowed polynomial view.
pub type PolynomialViewMut<'a, T, N> =
    Polynomial<T, N, StorageViewMut<'a, T, N, Const<1>>>;

////////////////////////////////////////////////////////////////////////////////
// Constructors
////////////////////////////////////////////////////////////////////////////////

impl<T, const N: usize> ArrayPolynomial<T, N>
where
    Const<N>: Dim,
{
    /// Builds an all-zero polynomial of capacity `N`.
    #[must_use]
    pub const fn zero() -> Self
    where
        T: Zero + Copy,
    {
        Self::from_storage(ArrayStorage::zero())
    }

    /// Builds a polynomial from a fixed-size array of coefficients in ascending power order.
    #[must_use]
    pub const fn from_coefficients(data: [T; N]) -> Self {
        Self::from_storage(ArrayStorage::from_column(data))
    }

    /// Builds a polynomial element-by-element using a generating closure.
    pub fn from_fn(mut f: impl FnMut(usize) -> T) -> Self {
        Self::from_storage(ArrayStorage::<T, N, 1>::from_fn(|i, _| f(i)))
    }

    /// Zero-copy strided view of the coefficient vector.
    #[must_use]
    pub fn view(&self) -> PolynomialView<'_, T, Const<N>> {
        let storage = unsafe {
            StorageView::new_with_strides_unchecked(
                self.storage.as_ptr(),
                self.storage.r_stride(),
                self.storage.c_stride(),
            )
        };
        Polynomial::from_storage(storage)
    }

    /// Zero-copy mutable strided view of the coefficient vector.
    pub fn view_mut(&mut self) -> PolynomialViewMut<'_, T, Const<N>> {
        let storage = unsafe {
            StorageViewMut::new_with_strides_unchecked(
                self.storage.as_mut_ptr(),
                self.storage.r_stride(),
                self.storage.c_stride(),
            )
        };
        Polynomial::from_storage(storage)
    }
}

impl<T: Copy> ArrayPolynomial<T, 1> {
    /// Builds a constant degree-0 polynomial ($p(x) = c_0$).
    #[must_use]
    pub const fn constant(val: T) -> Self {
        Self::from_coefficients([val])
    }
}

impl<T: Copy> ArrayPolynomial<T, 2> {
    /// Builds a linear degree-1 polynomial ($p(x) = c_0 + c_1 x$).
    #[must_use]
    pub const fn line(c0: T, c1: T) -> Self {
        Self::from_coefficients([c0, c1])
    }
}

impl<T, N: Dim, S: Storage<T, N, Const<1>>> Polynomial<T, N, S> {
    /// Wraps a custom storage backend.
    pub const fn from_storage(storage: S) -> Self {
        Self {
            storage,
            _marker: PhantomData,
        }
    }

    /// Borrows the underlying storage backend.
    pub const fn storage(&self) -> &S {
        &self.storage
    }

    /// Mutably borrows the underlying storage backend.
    pub fn storage_mut(&mut self) -> &mut S {
        &mut self.storage
    }

    /// Unwraps the underlying storage backend.
    pub fn into_storage(self) -> S {
        self.storage
    }

    /// Number of coefficient slots allocated (capacity).
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.storage.rows()
    }

    /// Returns a reference to the coefficient of $x^i$, or `None` if `i >= capacity`.
    #[must_use]
    pub fn get(&self, i: usize) -> Option<&T> {
        self.storage.get(i, 0)
    }
}

impl<T, N: Dim, S: StorageMut<T, N, Const<1>>> Polynomial<T, N, S> {
    /// Returns a mutable reference to the coefficient of $x^i$, or `None` if `i >= capacity`.
    pub fn get_mut(&mut self, i: usize) -> Option<&mut T> {
        self.storage.get_mut(i, 0)
    }
}

impl<T, N: Dim, S: Storage<T, N, Const<1>> + ContiguousStorage<T>>
    Polynomial<T, N, S>
{
    /// Exposes a safe contiguous slice view of coefficient memory in ascending power order.
    #[must_use]
    pub fn as_slice(&self) -> &[T] {
        self.storage.as_slice()
    }
}

impl<T, N: Dim, S: StorageMut<T, N, Const<1>> + ContiguousStorageMut<T>>
    Polynomial<T, N, S>
{
    /// Exposes a safe mutable contiguous slice view of coefficient memory.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.storage.as_mut_slice()
    }
}

////////////////////////////////////////////////////////////////////////////////
// Polynomial Queries & Horner Evaluation
////////////////////////////////////////////////////////////////////////////////

impl<T: Scalar + Copy, N: Dim, S: Storage<T, N, Const<1>>> Polynomial<T, N, S> {
    /// Evaluates the true mathematical degree (highest index with non-zero coefficient).
    ///
    /// Returns `None` for the zero polynomial.
    #[must_use]
    pub fn degree(&self) -> Option<usize> {
        let cap = self.capacity();
        for i in (0..cap).rev() {
            if let Some(&c) = self.get(i) {
                if c != T::ZERO {
                    return Some(i);
                }
            }
        }
        None
    }

    /// Returns the leading non-zero coefficient, or `None` if all coefficients are zero.
    #[must_use]
    pub fn leading_coefficient(&self) -> Option<&T> {
        self.degree().and_then(|d| self.get(d))
    }

    /// Checks whether the polynomial is monic (leading coefficient is `T::ONE`).
    #[must_use]
    pub fn is_monic(&self) -> bool {
        self.leading_coefficient().is_some_and(|&c| c == T::ONE)
    }

    /// Evaluates the polynomial at scalar `x` using Horner's method.
    ///
    /// $$p(x) = c_0 + x(c_1 + x(c_2 + \dots))$$
    /// Runs in $\mathcal{O}(N)$ multiplications and additions with optimal backward stability.
    #[must_use]
    pub fn evaluate(&self, x: T) -> T {
        let cap = self.capacity();
        if cap == 0 {
            return T::ZERO;
        }
        let mut result = match self.get(cap - 1) {
            Some(&c) => c,
            None => T::ZERO,
        };
        for i in (0..(cap - 1)).rev() {
            let c = match self.get(i) {
                Some(&coeff) => coeff,
                None => T::ZERO,
            };
            result = result * x + c;
        }
        result
    }

    /// Evaluates the polynomial at complex scalar `x` using complex Horner evaluation.
    #[must_use]
    pub fn evaluate_complex(&self, x: Complex<T>) -> Complex<T> {
        let cap = self.capacity();
        if cap == 0 {
            return Complex::ZERO;
        }
        let mut result = match self.get(cap - 1) {
            Some(&c) => Complex::new(c, T::ZERO),
            None => Complex::ZERO,
        };
        for i in (0..(cap - 1)).rev() {
            let c = match self.get(i) {
                Some(&coeff) => coeff,
                None => T::ZERO,
            };
            result = result * x + Complex::new(c, T::ZERO);
        }
        result
    }
}

////////////////////////////////////////////////////////////////////////////////
// Operator Overloads
////////////////////////////////////////////////////////////////////////////////

impl<'b, T, const N: usize, S, S2> Add<&'b Polynomial<T, Const<N>, S2>>
    for &Polynomial<T, Const<N>, S>
where
    T: Scalar + Copy,
    Const<N>: Dim,
    S: Storage<T, Const<N>, Const<1>>,
    S2: Storage<T, Const<N>, Const<1>>,
{
    type Output = ArrayPolynomial<T, N>;

    fn add(self, rhs: &'b Polynomial<T, Const<N>, S2>) -> Self::Output {
        let mut out = ArrayPolynomial::<T, N>::zero();
        DefaultBlas::axpy(T::ONE, &self.storage, &mut out.storage);
        DefaultBlas::axpy(T::ONE, &rhs.storage, &mut out.storage);
        out
    }
}

impl<'b, T, const N: usize, S, S2> Sub<&'b Polynomial<T, Const<N>, S2>>
    for &Polynomial<T, Const<N>, S>
where
    T: Scalar + Copy,
    Const<N>: Dim,
    S: Storage<T, Const<N>, Const<1>>,
    S2: Storage<T, Const<N>, Const<1>>,
{
    type Output = ArrayPolynomial<T, N>;

    fn sub(self, rhs: &'b Polynomial<T, Const<N>, S2>) -> Self::Output {
        let mut out = ArrayPolynomial::<T, N>::zero();
        DefaultBlas::axpy(T::ONE, &self.storage, &mut out.storage);
        DefaultBlas::axpy(T::ZERO - T::ONE, &rhs.storage, &mut out.storage);
        out
    }
}

impl<T, const N: usize, S> Neg for &Polynomial<T, Const<N>, S>
where
    T: Scalar + Copy,
    Const<N>: Dim,
    S: Storage<T, Const<N>, Const<1>>,
{
    type Output = ArrayPolynomial<T, N>;

    fn neg(self) -> Self::Output {
        let mut out = ArrayPolynomial::<T, N>::zero();
        DefaultBlas::axpy(T::ZERO - T::ONE, &self.storage, &mut out.storage);
        out
    }
}

////////////////////////////////////////////////////////////////////////////////
// Calculus: Derivative & Integral
////////////////////////////////////////////////////////////////////////////////

impl<T: Scalar + Copy, const N: usize> ArrayPolynomial<T, N>
where
    Const<N>: Dim,
{
    /// Computes the exact analytical derivative of the polynomial:
    ///
    /// $$\frac{d}{dx} \left( \sum_{i=0}^{N-1} c_i x^i \right) = \sum_{i=1}^{N-1} i c_i x^{i-1}$$
    #[must_use]
    pub fn derivative(&self) -> ArrayPolynomial<T, N> {
        let mut out = ArrayPolynomial::<T, N>::zero();
        for i in 1..N {
            let factor = {
                let mut f = T::ZERO;
                for _ in 0..i {
                    f = f + T::ONE;
                }
                f
            };
            if let Some(&c) = self.get(i) {
                if let Some(out_c) = out.get_mut(i - 1) {
                    *out_c = factor * c;
                }
            }
        }
        out
    }

    /// Multiplies two polynomials using DSP backend `C`.
    ///
    /// Product capacity $P = N + M - 1$.
    pub fn mul_poly_with<C, const M: usize, const P: usize>(
        &self,
        rhs: &ArrayPolynomial<T, M>,
    ) -> ArrayPolynomial<T, P>
    where
        C: Convolution<T>,
        Const<M>: Dim,
        Const<P>: Dim,
    {
        let mut out = ArrayPolynomial::<T, P>::zero();
        let _ = C::convolve_input(
            self.as_slice(),
            rhs.as_slice(),
            out.as_mut_slice(),
        );
        out
    }

    /// Multiplies two polynomials via [`DefaultDsp`].
    ///
    /// Product capacity $P = N + M - 1$.
    pub fn mul_poly<const M: usize, const P: usize>(
        &self,
        rhs: &ArrayPolynomial<T, M>,
    ) -> ArrayPolynomial<T, P>
    where
        Const<M>: Dim,
        Const<P>: Dim,
    {
        self.mul_poly_with::<DefaultDsp, M, P>(rhs)
    }

    /// Convolution multiply via [`crate::math::dsp::Convolution`] (`matrix-design` sibling DSP path).
    pub fn mul_with_conv<const M: usize, const P: usize>(
        &self,
        rhs: &ArrayPolynomial<T, M>,
    ) -> ArrayPolynomial<T, P>
    where
        Const<M>: Dim,
        Const<P>: Dim,
    {
        self.mul_poly_with::<DefaultDsp, M, P>(rhs)
    }
}

impl<T: Float + Copy, const N: usize> ArrayPolynomial<T, N>
where
    Const<N>: Dim,
{
    /// Computes the exact analytical integral of the polynomial with integration constant `c0`:
    ///
    /// $$\int \left( \sum_{i=0}^{N-2} c_i x^i \right) dx = c_0 + \sum_{i=0}^{N-2} \frac{c_i}{i+1} x^{i+1}$$
    #[must_use]
    pub fn integral(&self, c0: T) -> ArrayPolynomial<T, N> {
        let mut out = ArrayPolynomial::<T, N>::zero();
        if let Some(target) = out.get_mut(0) {
            *target = c0;
        }
        for i in 0..(N.saturating_sub(1)) {
            let divisor = {
                let mut d = T::ZERO;
                for _ in 0..=(i) {
                    d = d + T::ONE;
                }
                d
            };
            if let Some(&c) = self.get(i) {
                if let Some(target) = out.get_mut(i + 1) {
                    *target = c / divisor;
                }
            }
        }
        out
    }

    /// Computes Euclidean polynomial division ($A(x) = Q(x) D(x) + R(x)$).
    ///
    /// Returns `(Quotient, Remainder)` where degree of `Q` is `N - M` (capacity `N - M + 1`)
    /// and degree of `R` is `< M - 1` (capacity `M - 1`).
    ///
    /// # Errors
    /// Returns [`DivisionError::ZeroLeadingCoefficient`] if the divisor is the
    /// zero polynomial, or [`DivisionError::DegreeMismatch`] if $\deg D > \deg A$.
    pub fn div_rem<const M: usize, const Q: usize, const R: usize>(
        &self,
        divisor: &ArrayPolynomial<T, M>,
    ) -> Result<(ArrayPolynomial<T, Q>, ArrayPolynomial<T, R>), DivisionError>
    where
        T: Float,
        Const<M>: Dim,
        Const<Q>: Dim,
        Const<R>: Dim,
    {
        let deg_a = self.degree().unwrap_or(0);
        let deg_b = divisor
            .degree()
            .ok_or(DivisionError::ZeroLeadingCoefficient)?;
        if deg_b > deg_a {
            return Err(DivisionError::DegreeMismatch);
        }

        let mut rem_coeffs = [T::ZERO; N];
        for i in 0..N {
            rem_coeffs[i] = self.get(i).copied().unwrap_or(T::ZERO);
        }

        let mut quot_coeffs = [T::ZERO; Q];
        let b_lead = divisor.get(deg_b).copied().unwrap_or(T::ZERO);
        if b_lead.abs() < T::epsilon() {
            return Err(DivisionError::ZeroLeadingCoefficient);
        }

        if deg_a >= deg_b {
            for i in (deg_b..=deg_a).rev() {
                let factor = rem_coeffs[i] / b_lead;
                let q_idx = i - deg_b;
                if q_idx < Q {
                    quot_coeffs[q_idx] = factor;
                }
                for j in 0..=deg_b {
                    let c_b = divisor.get(j).copied().unwrap_or(T::ZERO);
                    rem_coeffs[i - deg_b + j] =
                        rem_coeffs[i - deg_b + j] - factor * c_b;
                }
            }
        }

        let mut final_rem = [T::ZERO; R];
        for i in 0..R {
            if i < N {
                final_rem[i] = rem_coeffs[i];
            }
        }

        Ok((
            ArrayPolynomial::<T, Q>::from_coefficients(quot_coeffs),
            ArrayPolynomial::<T, R>::from_coefficients(final_rem),
        ))
    }
}

////////////////////////////////////////////////////////////////////////////////
// Companion Matrix & Root Finding
////////////////////////////////////////////////////////////////////////////////

impl<T: Float + Copy, const N: usize> ArrayPolynomial<T, N>
where
    Const<N>: Dim,
{
    /// Constructs the Frobenius companion matrix in Controllable Canonical Form.
    ///
    /// For monic $p(x) = c_0 + c_1 x + \dots + c_{n-1} x^{n-1} + x^n$ of degree $n = N - 1$,
    /// the companion matrix $C \in \mathbb{R}^{n \times n}$ is:
    /// $$C = \begin{bmatrix} 0 & 0 & \dots & 0 & -c_0 \\ 1 & 0 & \dots & 0 & -c_1 \\ 0 & 1 & \dots & 0 & -c_2 \\ \vdots & \vdots & \ddots & \vdots & \vdots \\ 0 & 0 & \dots & 1 & -c_{n-1} \end{bmatrix}$$
    ///
    /// # Errors
    /// Returns [`ConversionError::DimensionMismatch`] if $N < 2$, or [`ConversionError::NonMonicPolynomial`] if the polynomial is not monic.
    pub fn companion_matrix<const DEG: usize>(
        &self,
    ) -> ConversionResult<Owned<T, DEG, DEG>>
    where
        Const<DEG>: Dim,
    {
        if DEG + 1 != N || DEG == 0 {
            return Err(ConversionError::DimensionMismatch);
        }
        let leading = self.get(DEG).copied().unwrap_or(T::ZERO);
        if (leading - T::ONE).abs() > T::epsilon() * (T::ONE + T::ONE) {
            return Err(ConversionError::NonMonicPolynomial);
        }

        let mut comp = Owned::<T, DEG, DEG>::zero();
        // Subdiagonal ones
        for i in 1..DEG {
            if let Some(elem) = comp.get_mut(i, i - 1) {
                *elem = T::ONE;
            }
        }
        // Last column: -c_i
        for i in 0..DEG {
            let c = self.get(i).copied().unwrap_or(T::ZERO);
            if let Some(elem) = comp.get_mut(i, DEG - 1) {
                *elem = T::ZERO - c;
            }
        }
        Ok(comp)
    }
}

impl<T: Float + Copy, const N: usize, const DEG: usize>
    TryFrom<&ArrayPolynomial<T, N>> for Owned<T, DEG, DEG>
where
    Const<N>: Dim,
    Const<DEG>: Dim,
{
    type Error = ConversionError;

    fn try_from(poly: &ArrayPolynomial<T, N>) -> Result<Self, Self::Error> {
        poly.companion_matrix::<DEG>()
    }
}

impl<T: Scalar + Copy> ArrayPolynomial<T, 4> {
    /// Cubic Hermite segment on $t \in [0, 1]$ from endpoints $(p_0, v_0)$, $(p_1, v_1)$.
    #[must_use]
    pub fn cubic(p0: T, p1: T, v0: T, v1: T) -> Self {
        let three = T::ONE + T::ONE + T::ONE;
        let two = T::ONE + T::ONE;
        let dp = p1 - p0;
        let a2 = three * dp - two * v0 - v1;
        let a3 = two * (p0 - p1) + v0 + v1;
        Self::from_coefficients([p0, v0, a2, a3])
    }
}

impl<T: Float + Copy> ArrayPolynomial<T, 6> {
    /// Quintic Hermite segment on $t \in [0, 1]$ from position, velocity, and acceleration.
    #[must_use]
    pub fn quintic(p0: T, p1: T, v0: T, v1: T, a0: T, a1: T) -> Self {
        let two = T::ONE + T::ONE;
        let half = T::ONE / two;
        let three = two + T::ONE;
        let four = two + two;
        let six = three + three;
        let seven = six + T::ONE;
        let eight = four + four;
        let ten = eight + two;
        let fifteen = ten + four + T::ONE;
        let dp = p1 - p0;
        let c0 = p0;
        let c1 = v0;
        let c2 = a0 * half;
        let c3 =
            ten * dp - six * v0 - four * v1 - (three * half) * a0 + half * a1;
        let c4 = (T::ZERO - fifteen) * dp
            + eight * v0
            + seven * v1
            + (three * half) * a0
            - a1;
        let c5 = six * dp - three * v0 - three * v1 - half * a0 + half * a1;
        Self::from_coefficients([c0, c1, c2, c3, c4, c5])
    }
}

impl<T: Float + Copy, const N: usize> ArrayPolynomial<T, N>
where
    Const<N>: Dim,
{
    /// Substitute $s = \frac{2}{T_s}\frac{z-1}{z+1}$ and clear $(z+1)^{N-1}$.
    #[must_use]
    pub fn compose_bilinear(&self, sample_time: T) -> Self {
        let two = T::ONE + T::ONE;
        let k = two / sample_time;
        let n = N.saturating_sub(1);
        let mut out = [T::ZERO; N];
        for deg in 0..N {
            let c = self.get(deg).copied().unwrap_or(T::ZERO);
            if c == T::ZERO {
                continue;
            }
            let scale = {
                let mut s = T::ONE;
                for _ in 0..deg {
                    s = s * k;
                }
                s * c
            };
            let zm1 = expand_linear::<T, N>(deg, T::ONE, T::ZERO - T::ONE);
            let zp1 =
                expand_linear::<T, N>(n.saturating_sub(deg), T::ONE, T::ONE);
            for i in 0..N {
                for j in 0..(N - i) {
                    out[i + j] = out[i + j] + scale * zm1[i] * zp1[j];
                }
            }
        }
        Self::from_coefficients(out)
    }
}

/// Expand $(a z + b)^{\mathrm{power}}$ into ascending coefficients of length `N`.
fn expand_linear<T: Float + Copy, const N: usize>(
    power: usize,
    a: T,
    b: T,
) -> [T; N] {
    let mut coeffs = [T::ZERO; N];
    if N == 0 {
        return coeffs;
    }
    coeffs[0] = T::ONE;
    for _ in 0..power {
        let mut next = [T::ZERO; N];
        for i in 0..N {
            next[i] = next[i] + coeffs[i] * b;
            if i + 1 < N {
                next[i + 1] = next[i + 1] + coeffs[i] * a;
            }
        }
        coeffs = next;
    }
    coeffs
}
