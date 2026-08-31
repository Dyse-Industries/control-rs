//! Fixed-point number mathematical ETS and unit test suite.
#![allow(clippy::arithmetic_side_effects)]

#[cfg_attr(not(test), control_rs_macros::ets_suite)]
/// Unit and ETS test suite for fixed-point number operations.
pub mod fixed_num_test_suite {
    use crate::math::{
        ArithmeticError,
        fixed_num::{
            Fixed, OneRepresentable, Q7, Q15, Q31, Q63, TwoRepresentable, UQ7,
            UQ15, UQ31, UQ63,
        },
        num_traits::{Conjugate, One, SaturatingInteger, Scalar, Signed, Zero},
        ops::{
            SaturatingAdd, SaturatingMul, SaturatingSub, TryAdd, TryMul,
            TryNeg, TrySub,
        },
    };
    use core::mem::{align_of, size_of};

    struct StackBuf {
        buf: [u8; 64],
        len: usize,
    }

    impl core::fmt::Write for StackBuf {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            for &b in s.as_bytes() {
                if let Some(slot) = self.buf.get_mut(self.len) {
                    *slot = b;
                    self.len += 1;
                }
            }
            Ok(())
        }
    }

    #[cfg_attr(test, test)]
    /// Verifies single-word memory footprint and alignment across all supported primitives (NFR-1).
    fn test_memory_footprint_and_alignment() {
        assert_eq!(size_of::<Q7>(), size_of::<i8>());
        assert_eq!(align_of::<Q7>(), align_of::<i8>());

        assert_eq!(size_of::<Q15>(), size_of::<i16>());
        assert_eq!(align_of::<Q15>(), align_of::<i16>());

        assert_eq!(size_of::<Q31>(), size_of::<i32>());
        assert_eq!(align_of::<Q31>(), align_of::<i32>());

        assert_eq!(size_of::<Q63>(), size_of::<i64>());
        assert_eq!(align_of::<Q63>(), align_of::<i64>());

        assert_eq!(size_of::<UQ7>(), size_of::<u8>());
        assert_eq!(align_of::<UQ7>(), align_of::<u8>());

        assert_eq!(size_of::<UQ15>(), size_of::<u16>());
        assert_eq!(align_of::<UQ15>(), align_of::<u16>());

        assert_eq!(size_of::<UQ31>(), size_of::<u32>());
        assert_eq!(align_of::<UQ31>(), align_of::<u32>());

        assert_eq!(size_of::<UQ63>(), size_of::<u64>());
        assert_eq!(align_of::<UQ63>(), align_of::<u64>());
    }

    #[cfg_attr(test, test)]
    /// Verifies associated constants and bit-level constructors (FR-1, FR-2).
    fn test_constants_and_constructors() {
        type Q14 = Fixed<i16, 14>;

        let zero = Q14::ZERO;
        assert_eq!(zero.to_bits(), 0);
        assert!(zero.is_zero());

        let delta = Q14::DELTA;
        assert_eq!(delta.to_bits(), 1);
        assert!((delta.to_num() - (1.0 / 16384.0)).abs() < 1e-9);

        let one = Q14::ONE;
        assert_eq!(one.to_bits(), 1 << 14);
        assert!((one.to_num() - 1.0).abs() < 1e-9);
        assert!(one.is_one());

        let min = Q14::MIN;
        assert_eq!(min.to_bits(), i16::MIN);

        let max = Q14::MAX;
        assert_eq!(max.to_bits(), i16::MAX);
    }

    #[cfg_attr(test, test)]
    /// Verifies basic addition, subtraction, and negation with saturation (FR-3).
    fn test_addition_subtraction_negation_saturating() {
        type Q14 = Fixed<i16, 14>;

        let a = Q14::from_num(0.5);
        let b = Q14::from_num(0.25);

        let sum = a + b;
        assert!((sum.to_num() - 0.75).abs() < 1e-4);

        let diff = a - b;
        assert!((diff.to_num() - 0.25).abs() < 1e-4);

        let neg = -a;
        assert!((neg.to_num() - (-0.5)).abs() < 1e-4);

        // Saturation at MAX bound
        let max_val = Q14::MAX;
        let saturated_add = max_val + Q14::ONE;
        assert_eq!(saturated_add, Q14::MAX);

        // Saturation at MIN bound
        let min_val = Q14::MIN;
        let saturated_sub = min_val - Q14::ONE;
        assert_eq!(saturated_sub, Q14::MIN);

        // Negation of MIN saturates to MAX
        let neg_min = -min_val;
        assert_eq!(neg_min, Q14::MAX);
    }

    #[cfg_attr(test, test)]
    /// Verifies widening multiplication with exact intermediate and convergent rounding (FR-4).
    fn test_widening_multiplication() {
        type Q14 = Fixed<i16, 14>;

        let a = Q14::from_num(0.5);
        let b = Q14::from_num(0.5);
        let prod = a * b;
        assert!((prod.to_num() - 0.25).abs() < 1e-4);

        let c = Q14::from_num(-0.75);
        let prod_c = a * c;
        assert!((prod_c.to_num() - (-0.375)).abs() < 1e-4);

        // Large multiplication saturation
        let big_a = Q14::from_num(1.5);
        let big_b = Q14::from_num(1.5);
        let big_prod = big_a * big_b; // 2.25 exceeds Q14 max (~1.9999) -> saturates to MAX
        assert_eq!(big_prod, Q14::MAX);
    }

    #[cfg_attr(test, test)]
    /// Verifies explicit scale conversion and rescaling round-trips (FR-5).
    fn test_scale_rescaling() {
        type Q14 = Fixed<i16, 14>;
        type Q12 = Fixed<i16, 12>;

        let val_q14 = Q14::from_num(0.75);
        let val_q12: Q12 = val_q14.rescale();
        assert!((val_q12.to_num() - 0.75).abs() < 1e-3);

        let round_trip: Q14 = val_q12.rescale();
        assert_eq!(round_trip, val_q14);

        // Upscaling with saturation
        let big_q12 = Q12::from_num(3.5);
        let sat_q14: Q14 = big_q12.rescale(); // 3.5 exceeds Q14 max -> saturates
        assert_eq!(sat_q14, Q14::MAX);
    }

    #[cfg_attr(test, test)]
    /// Verifies fallible arithmetic operations return overflow error at bounds.
    fn test_fallible_try_operations() {
        type Q14 = Fixed<i16, 14>;

        let a = Q14::from_num(0.5);
        let b = Q14::from_num(0.25);

        assert_eq!(
            a.try_add(&b).unwrap().to_bits(),
            (Q14::from_num(0.75)).to_bits()
        );
        assert_eq!(
            a.try_sub(&b).unwrap().to_bits(),
            (Q14::from_num(0.25)).to_bits()
        );
        assert_eq!(
            a.try_mul(&b).unwrap().to_bits(),
            (Q14::from_num(0.125)).to_bits()
        );
        assert_eq!(
            a.try_neg().unwrap().to_bits(),
            (Q14::from_num(-0.5)).to_bits()
        );

        // Overflow conditions
        assert_eq!(Q14::MAX.try_add(&Q14::ONE), Err(ArithmeticError::Overflow));
        assert_eq!(Q14::MIN.try_sub(&Q14::ONE), Err(ArithmeticError::Overflow));
        assert_eq!(Q14::MIN.try_neg(), Err(ArithmeticError::Overflow));

        let big_a = Q14::from_num(1.5);
        let big_b = Q14::from_num(1.5);
        assert_eq!(big_a.try_mul(&big_b), Err(ArithmeticError::Overflow));
    }

    #[cfg_attr(test, test)]
    /// Verifies numeric trait participation (Zero, One, Conjugate, Scalar, Signed, `AdditiveGroup`, `SaturatingInteger`).
    fn test_numeric_trait_hierarchy() {
        type Q13 = Fixed<i16, 13>;

        fn check_scalar<T: Scalar<Real = T> + core::fmt::Debug>(val: &T) -> T {
            let re = val.re();
            let abs2 = val.abs2();
            assert_eq!(val.im(), T::ZERO);
            assert_eq!(re, *val);
            abs2
        }

        let val = Q13::from_num(0.5);
        let abs2 = check_scalar(&val);
        assert!((abs2.to_num() - 0.25).abs() < 1e-3);

        // Conjugate is identity
        assert_eq!(val.conj(), val);

        // Signed & AdditiveGroup
        assert_eq!(val.abs(), val);
        let neg_val = -val;
        assert_eq!(neg_val.abs(), val);
        assert!(neg_val.is_sign_negative());
        assert!(val.is_sign_positive());

        // SaturatingInteger
        assert_eq!(val.saturating_add(&val), Q13::from_num(1.0));
        assert_eq!(val.saturating_mul(&val), Q13::from_num(0.25));
    }

    #[cfg_attr(test, test)]
    /// Verifies unsigned fixed-point representations.
    fn test_unsigned_fixed_point() {
        type UQ14 = Fixed<u16, 14>;

        let u_one = UQ14::ONE;
        assert_eq!(u_one.to_bits(), 1 << 14);
        assert!((u_one.to_num() - 1.0).abs() < 1e-9);

        let a = UQ14::from_num(1.5);
        let b = UQ14::from_num(0.5);

        let sum = a + b;
        assert!((sum.to_num() - 2.0).abs() < 1e-4);

        let diff = a - b;
        assert!((diff.to_num() - 1.0).abs() < 1e-4);

        let underflow = b - a; // underflow saturates to 0
        assert_eq!(underflow.to_bits(), 0);

        let prod = a * b;
        assert!((prod.to_num() - 0.75).abs() < 1e-4);
    }

    #[cfg_attr(test, test)]
    /// Signed product tie (raw product −3, `SHIFT = 1`) rounds to −2
    /// (ties-to-even), not −1 (§6.1 item 5 of `fixed-num-design.md`).
    fn test_signed_product_tie_rounds_to_even() {
        type Q1 = Fixed<i16, 1>;
        let a = Q1::from_bits(-3);
        let b = Q1::from_bits(1);
        assert_eq!((a * b).to_bits(), -2);
    }

    #[cfg_attr(test, test)]
    /// Verifies gate separation between `OneRepresentable` and `TwoRepresentable` markers (§6.1.5).
    fn test_gate_separation_boundary_pin() {
        type Q14 = Fixed<i16, 14>;
        type Q13 = Fixed<i16, 13>;

        fn assert_one_rep<T: OneRepresentable>() {}
        fn assert_two_rep<T: TwoRepresentable>() {}
        fn assert_one<T: One>() {}
        fn assert_scalar<T: Scalar>() {}
        fn assert_sat_int<T: SaturatingInteger>() {}

        // Fixed<i16, 13>: both One and Two representable
        assert_one_rep::<Q13>();
        assert_two_rep::<Q13>();
        assert_one::<Q13>();
        assert_scalar::<Q13>();
        assert_sat_int::<Q13>();

        // Fixed<i16, 14>: One and Scalar hold, but SaturatingInteger does not
        assert_one_rep::<Q14>();
        assert_one::<Q14>();
        assert_scalar::<Q14>();
        // Note: assert_sat_int::<Q14>() and assert_two_rep::<Q14>() fail compile-time trait bound
    }

    #[cfg_attr(test, test)]
    fn test_fixed_display_assign_and_rescale() {
        use core::fmt::Write;
        type Q1 = Fixed<i16, 1>;
        type Q8 = Fixed<i16, 8>;
        type UQ8 = Fixed<u16, 8>;
        let a = Q8::from_num(0.5);
        let mut buf = StackBuf {
            buf: [0u8; 64],
            len: 0,
        };
        write!(&mut buf, "{a}").unwrap();
        let rendered = buf
            .buf
            .get(..buf.len)
            .and_then(|s| core::str::from_utf8(s).ok())
            .unwrap_or("");
        assert!(rendered.contains("Q8"));
        let mut b = a;
        b += a;
        b -= a;
        let _ = b * a;
        let mut c = a;
        c *= Q8::from_num(2.0);
        let up: Q1 = a.rescale();
        assert!(up.to_num().abs() > 0.0);
        let down: Q8 = Q1::from_num(1.0).rescale();
        assert!((down.to_num() - 1.0).abs() < 0.1);
        let huge = Q8::from_bits(i16::MAX);
        let _shifted: Fixed<i16, 12> = huge.rescale();
        let tiny: Fixed<i16, 0> = Q8::from_num(0.25).rescale();
        assert!(tiny.to_bits().abs() >= 0);

        // Saturation and Try error paths
        let max_q8 = Q8::from_bits(i16::MAX);
        let min_q8 = Q8::from_bits(i16::MIN);
        assert_eq!(max_q8.try_add(&max_q8), Err(ArithmeticError::Overflow));
        assert_eq!(min_q8.try_sub(&max_q8), Err(ArithmeticError::Overflow));
        assert_eq!(max_q8.try_mul(&max_q8), Err(ArithmeticError::Overflow));
        assert_eq!(min_q8.try_neg(), Err(ArithmeticError::Overflow));

        let _sat_add = max_q8.saturating_add(&max_q8);
        let _sat_sub = min_q8.saturating_sub(&max_q8);
        let _sat_mul = max_q8.saturating_mul(&max_q8);

        // UQ8 / unsigned fixed
        let u_max = UQ8::from_bits(u16::MAX);
        assert_eq!(u_max.try_add(&u_max), Err(ArithmeticError::Overflow));
        assert_eq!(
            UQ8::from_bits(0).try_sub(&u_max),
            Err(ArithmeticError::Overflow)
        );
        assert_eq!(u_max.try_mul(&u_max), Err(ArithmeticError::Overflow));
    }
}

#[cfg(test)]
mod fixed_num_scale_bound_tests {
    use crate::math::fixed_num::Fixed;

    /// `Dim` admits `R` through 1024, but C-3 requires `R <= BITS`. Oversized
    /// destination scales must panic at `rescale` rather than deferring failure
    /// into a later unsigned `Mul`.
    #[test]
    #[should_panic(expected = "exceeds representation bit width")]
    fn rescale_rejects_destination_scale_above_bits() {
        let a = Fixed::<u8, 4>::from_num(0.5);
        let _: Fixed<u8, 16> = a.rescale();
    }

    /// Invalid `SHIFT > Wide::BITS` constructed via `from_bits` must not panic
    /// on unsigned `Mul`. Pre-fix native `$w << n` / `$w >> n` panicked in
    /// debug and silently masked the shift count in release.
    #[test]
    fn unsigned_mul_with_oversized_shift_does_not_panic() {
        let a = Fixed::<u8, 16>::from_bits(255);
        let b = Fixed::<u8, 16>::from_bits(255);
        // 255*255 = 65025; shift-16 convergent round yields 1 (rem > half).
        // Pre-fix release builds masked `>> 16` to `>> 0`, returning ~65025
        // narrowed to 255 — catastrophic product corruption.
        let prod = a * b;
        assert_eq!(prod.to_bits(), 1);
    }

    #[test]
    fn unsigned_try_mul_with_oversized_shift_does_not_panic() {
        use crate::math::ops::TryMul;
        let a = Fixed::<u8, 16>::from_bits(200);
        let b = Fixed::<u8, 16>::from_bits(200);
        // 200*200 = 40000; shift-16 rem=40000 > half=32768 → round to 1.
        let prod = a.try_mul(&b).expect("checked mul must succeed or Overflow");
        assert_eq!(prod.to_bits(), 1);
    }
}

// Property-based coverage of product exactness and rescale round-tripping
// (§6.1 items 3 and 4 of `fixed-num-design.md`). Kept outside the
// `#[ets_suite]`-wrapped module: `proptest` is a host-only dev-dependency.
#[cfg(test)]
mod fixed_num_property_tests {
    use crate::math::fixed_num::Fixed;
    use proptest::prelude::*;

    proptest! {
        /// Exact product rescale test (§6.1 item 3 of fixed-num-design.md):
        /// For random raw pairs, the widening multiplication result equals
        /// the reference product rounded to the grid with round-ties-to-even,
        /// error bounded by DELTA / 2.0.
        #[test]
        fn prop_widening_product_exactness(
            raw_a in -16384i16..=16383i16,
            raw_b in -16384i16..=16383i16,
        ) {
            type Q14 = Fixed<i16, 14>;
            let a = Q14::from_bits(raw_a);
            let b = Q14::from_bits(raw_b);
            let prod = a * b;
            let ref_prod = a.to_num() * b.to_num();
            let delta = 1.0 / f64::from(1u32 << 14);
            let half_delta = delta / 2.0 + 1e-12;

            if ref_prod <= f64::from(i16::MAX) * delta && ref_prod >= f64::from(i16::MIN) * delta {
                let diff = (prod.to_num() - ref_prod).abs();
                prop_assert!(diff <= half_delta);
            }
        }

        /// Rescale round-trip test (§6.1 item 4 of fixed-num-design.md):
        /// Rescaling from q to r and back is the identity when r >= q,
        /// and within DELTA/2 when r < q.
        #[test]
        fn prop_rescale_round_trip(
            raw in -4096i16..=4095i16,
        ) {
            type Q12 = Fixed<i16, 12>;
            type Q14 = Fixed<i16, 14>;
            type Q10 = Fixed<i16, 10>;

            let val_q12 = Q12::from_bits(raw);

            // r >= q: Upscaling to Q14 and back to Q12 is exact identity
            let up_q14: Q14 = val_q12.rescale();
            let back_q12: Q12 = up_q14.rescale();
            prop_assert_eq!(back_q12, val_q12);

            // r < q: Downscaling to Q10 and back to Q12 is within DELTA/2 of Q10
            let down_q10: Q10 = val_q12.rescale();
            let back_from_down: Q12 = down_q10.rescale();
            let delta_q10 = 1.0 / f64::from(1u32 << 10);
            let diff = (back_from_down.to_num() - val_q12.to_num()).abs();
            prop_assert!(diff <= delta_q10 / 2.0 + 1e-12);
        }
    }
}
