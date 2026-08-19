//! # Num Types
//!
//! The tests must verify that the compiler accurately translates constant values into their
//! corresponding Peano representations.
//!
//! The test suite must comprehensively cover the operations for addition, subtraction,
//! multiplication, minimum and maximum behaviors.
//!
//! A specialized testing sequence must assert that the memory footprint of these types remains
//! strictly zero bytes, ensuring that compile-time mathematics does not inflate the compiled
//! binary.
//!
//! The static tests must verify the commutative property of this trait, proving that adding
//! dimension A to dimension B results in the exact same type representation as adding B to A.
//!
//! ```compile_fail
//!  fn test_subtraction_underflow() {
//!      let _ = <U2 as DimSub<U5>>::Output::default();
//!  }
//! ```
//!
//! See `num_types`'s own module documentation for the regression
//! characterizing arithmetic past the `U127` ceiling (this suite lives behind
//! `#[cfg(any(test, feature = "hil"))]`, which `rustdoc`'s doctest extraction
//! does not set, so `compile_fail` examples placed here never actually run).
//!
//! ## Functional Requirement Coverage (`num-types-design.md`)
//!
//! - **FR-1** (base dimension trait): dimension value/structure checks.
//! - **FR-2** (type-level arithmetic): addition/subtraction/multiplication/
//!   min/max tests, including the pairwise recursion-limit regressions.
//! - **FR-3** (const-generic bridge): `Const<N>` hoisting, `from_i8` and
//!   the static-concatenation helpers.
//! - **FR-4** (named aliases `U0`..`U127`): exercised incidentally as the
//!   type arguments every other test in this suite uses, but has no test
//!   dedicated to the alias-generation macro itself.
//!

#[cfg_attr(not(test), control_rs_macros::hil_suite)]
pub mod num_type_test_suite {
    use crate::math::num_types::{
        Const, Dim, DimAdd, DimMax, DimMin, DimMul, DimSub, S, U0, U1, U2, U3,
        U4, U5, U6, U10, U12, U15, U32, U63, U64, U125, U127, Z,
    };
    use core::marker::PhantomData;
    use core::mem;

    struct TestStorage<C: Dim>(PhantomData<C>);

    #[cfg_attr(test, test)]
    /// Verifies that Peano representation structs compile down to a
    /// zero-byte memory footprint (NFR-1 of `num-types-design.md`).
    fn test_num_type_zero_byte_footprint() {
        assert_eq!(mem::size_of::<Z>(), 0);
        assert_eq!(mem::size_of::<S<Z>>(), 0);
        assert_eq!(mem::size_of::<S<S<Z>>>(), 0);
    }

    #[cfg_attr(test, test)]
    /// Verifies compile-time translation from standard const-generic values
    /// into Peano types (FR-3 of `num-types-design.md`).
    fn test_num_type_constant_generic_hoisting() {
        let _: <Const<5> as Dim>::PeanoTypeNum = U5::default();
    }

    #[cfg_attr(test, test)]
    /// Verifies `Const::from_i8` accepts both signs, matching on magnitude
    /// (FR-3 of `num-types-design.md`, const-generic bridge).
    fn test_num_type_const_from_i8() {
        let _: Const<5> = Const::from_i8(5);
        let _: Const<5> = Const::from_i8(-5);
        let _: Const<0> = Const::from_i8(0);
        // `i8::MIN.abs()` overflows; `saturating_abs()` clamps to `i8::MAX`.
        let _: Const<127> = Const::from_i8(i8::MIN);
    }

    #[cfg(debug_assertions)]
    #[cfg(test)]
    #[test]
    #[should_panic(expected = "n.saturating_abs() as usize == N")]
    fn _test_num_type_const_from_i8_magnitude_mismatch() {
        let _: Const<5> = Const::from_i8(4);
    }

    #[cfg_attr(test, test)]
    /// Verifies commutativity of type-level addition (A + B == B + A)
    /// (FR-2 of `num-types-design.md`).
    fn test_num_type_addition_commutativity() {
        let _: <U2 as DimAdd<U3>>::Output =
            <<U3 as DimAdd<U2>>::Output>::default();
    }

    #[cfg_attr(test, test)]
    /// Verifies static dimension constants equal their runtime values
    /// (FR-1 of `num-types-design.md`, runtime `usize` exposure).
    fn test_num_type_dimension_values() {
        assert_eq!(U0::USIZE, 0);
        assert_eq!(U1::USIZE, 1);
        assert_eq!(U15::USIZE, 15);
        assert_eq!(U127::USIZE, 127);
    }

    #[cfg_attr(test, test)]
    /// Verifies type-level addition arithmetic at compile-time and runtime
    /// (FR-2 of `num-types-design.md`).
    fn test_num_type_addition() {
        // Compile-time type structure assertions
        let _: U5 = <U2 as DimAdd<U3>>::Output::default();
        let _: U3 = <U0 as DimAdd<U3>>::Output::default();
        let _: U3 = <U3 as DimAdd<U0>>::Output::default();

        // Constant value assertions
        assert_eq!(<<U10 as DimAdd<U5>>::Output as Dim>::USIZE, 15);
    }

    #[cfg_attr(test, test)]
    /// Verifies base cases for type-level addition involving zero (FR-2 of
    /// `num-types-design.md`).
    fn test_num_type_addition_base_cases() {
        let _: U5 = <U5 as DimAdd<U0>>::Output::default();
        let _: U5 = <U0 as DimAdd<U5>>::Output::default();
    }

    #[cfg_attr(test, test)]
    /// Verifies type-level subtraction arithmetic (FR-2 of
    /// `num-types-design.md`).
    fn test_num_type_subtraction() {
        let _: U2 = <U5 as DimSub<U3>>::Output::default();
        let _: U0 = <U5 as DimSub<U5>>::Output::default();
        let _: U5 = <U5 as DimSub<U0>>::Output::default();

        assert_eq!(<<U15 as DimSub<U10>>::Output as Dim>::USIZE, 5);
    }

    #[cfg_attr(test, test)]
    /// Verifies type-level multiplication arithmetic (FR-2 of
    /// `num-types-design.md`).
    fn test_num_type_multiplication() {
        let _: U6 = <U2 as DimMul<U3>>::Output::default();
        let _: U0 = <U5 as DimMul<U0>>::Output::default();
        let _: U0 = <U0 as DimMul<U5>>::Output::default();

        assert_eq!(<<U4 as DimMul<U5>>::Output as Dim>::USIZE, 20);
    }

    #[cfg_attr(test, test)]
    /// Verifies that type-level multiplication recursion limits are
    /// respected by the compiler (FR-2 of `num-types-design.md`).
    fn test_num_type_multiplication_recursion_limit() {
        // `DimMul`'s per-pair envelope is narrower than the U127 single-value
        // ceiling (C-1): each recursive step re-resolves `Dim` on the
        // multiplier, so `DimMul<U1>` overflows the trait solver's recursion
        // limit above U125, not U127 (see C-2). U125 is the largest operand
        // pinned here as a regression against that boundary shrinking further.
        let _: U125 = <U125 as DimMul<U1>>::Output::default();
    }

    #[cfg_attr(test, test)]
    /// Verifies type-level addition at the pairwise recursion boundary (C-2)
    /// (FR-2 of `num-types-design.md`).
    fn test_num_type_addition_recursion_limit() {
        // `DimAdd`'s envelope is asymmetric, not bounded by the operand sum:
        // `U63 + U64` resolves while `U1 + U126` (the same sum) overflows
        // the trait solver — the failing side is pinned as a `compile_fail`
        // doctest in `num_types`. This pins the passing side against the
        // envelope shrinking.
        let _: U127 = <U63 as DimAdd<U64>>::Output::default();
    }

    #[cfg_attr(test, test)]
    /// Verifies type-level maximum resolution (FR-2 of
    /// `num-types-design.md`).
    fn test_num_type_maximum() {
        let _: U5 = <U2 as DimMax<U5>>::Output::default();
        let _: U5 = <U5 as DimMax<U2>>::Output::default();
        let _: U5 = <U5 as DimMax<U5>>::Output::default();

        assert_eq!(<<U12 as DimMax<U4>>::Output as Dim>::USIZE, 12);
    }

    #[cfg_attr(test, test)]
    /// Verifies type-level minimum resolution (FR-2 of
    /// `num-types-design.md`).
    fn test_num_type_minimum() {
        let _: U2 = <U2 as DimMin<U5>>::Output::default();
        let _: U2 = <U5 as DimMin<U2>>::Output::default();
        let _: U5 = <U5 as DimMin<U5>>::Output::default();

        assert_eq!(<<U12 as DimMin<U4>>::Output as Dim>::USIZE, 4);
    }

    #[cfg_attr(test, test)]
    /// Verifies compile-time minimum and maximum bounds resolution on
    /// non-uniform dimensions (FR-2 of `num-types-design.md`).
    fn test_num_type_dynamic_min_max_bounding() {
        // Use an operation to confirm that Max or Min bounds a dimension
        fn assert_bounds<A, B, Max, Min>()
        where
            A: DimMax<B, Output = Max> + DimMin<B, Output = Min>,
            B: Dim,
            Max: Dim,
            Min: Dim,
        {
        }

        // Statically assert that type-level Min and Max traits correctly resolve bounding boxes
        // for non-uniform tensor operations.
        let _: U5 = <U2 as DimMax<U5>>::Output::default();
        let _: U2 = <U2 as DimMin<U5>>::Output::default();

        assert_bounds::<U10, U15, U15, U10>();
        assert_bounds::<U32, U1, U32, U1>();
    }

    fn _concat_static_arrays<const A: usize, const B: usize, C: Dim>(
        _: [f32; A],
        _: [f32; B],
    ) -> TestStorage<C>
    where
        Const<A>: DimAdd<Const<B>, Output = C>,
    {
        TestStorage(PhantomData)
    }

    const fn _concat_static_types<A, B, C>(
        _: TestStorage<A>,
        _: TestStorage<B>,
    ) -> TestStorage<C>
    where
        A: Dim + DimAdd<B, Output = C>,
        B: Dim,
        C: Dim,
    {
        TestStorage(PhantomData)
    }

    #[cfg_attr(test, test)]
    /// Verifies static dimension concatenation matching for arrays and
    /// custom storage wrappers (FR-2 + FR-3 of `num-types-design.md`).
    fn test_num_type_concat_static() {
        let l1: TestStorage<U1> = TestStorage(PhantomData);
        let l2: TestStorage<U2> = TestStorage(PhantomData);
        let l3: TestStorage<U3> = TestStorage(PhantomData);

        let l_test: TestStorage<U3> = _concat_static_types(l1, l2);
        assert_eq!(l3.0, l_test.0);

        let l_array_test: TestStorage<U3> =
            _concat_static_arrays([0.0], [0.0, 0.0]);
        assert_eq!(l3.0, l_array_test.0);
    }
}
