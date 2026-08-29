//! # Num Types
//!
//! Verifies compile-time translation from const-generic values into binary
//! unsigned encodings, and covers addition, subtraction, multiplication,
//! minimum and maximum.
//!
//! Zero-byte footprint of the canonical types is asserted so compile-time
//! mathematics does not inflate the compiled binary.
//!
//! Addition commutativity is checked as type equality: `A + B` and `B + A`
//! resolve to the same canonical type.
//!
//! Subtraction underflow is a `compile_fail` doctest on `num_types` itself
//! (`U2 - U5`). This suite lives behind
//! `#[cfg(any(test, feature = "ets"))]`, which `rustdoc`'s doctest extraction
//! does not set, so `compile_fail` examples placed here never actually run.
//!
//! ## Functional Requirement Coverage (`num-types-design.md`)
//!
//! - **FR-1** (base dimension trait): dimension value/structure checks.
//! - **FR-2** (type-level arithmetic): addition/subtraction/multiplication/
//!   min/max tests, including the `U128 * U128` product pin.
//! - **FR-3** (const-generic bridge): `Const<N>` hoisting and the static
//!   concatenation helpers. No runtime constructor.
//! - **FR-4** (named aliases `U0`..`U1024` plus extras): exercised as the
//!   type arguments every other test in this suite uses.
//!

#[cfg_attr(not(test), control_rs_macros::ets_suite)]
pub mod num_type_test_suite {
    use crate::math::num_types::{
        B0, B1, Const, Dim, DimAdd, DimBitAnd, DimBitOr, DimBitXor, DimMax,
        DimMin, DimMul, DimSub, U0, U1, U2, U3, U4, U5, U6, U7, U8, U10, U12,
        U15, U16, U32, U63, U64, U126, U127, U128, U256, U1024, U16384, UInt,
        UTerm,
    };
    use core::marker::PhantomData;
    use core::mem;

    struct TestStorage<C: Dim>(PhantomData<C>);

    #[cfg_attr(test, test)]
    /// Verifies that canonical unsigned structs compile down to a
    /// zero-byte memory footprint (NFR-1 of `num-types-design.md`).
    fn test_num_type_zero_byte_footprint() {
        assert_eq!(mem::size_of::<UTerm>(), 0);
        assert_eq!(mem::size_of::<UInt<UTerm, B1>>(), 0);
        assert_eq!(mem::size_of::<UInt<UInt<UTerm, B1>, B0>>(), 0);
        assert_eq!(mem::size_of::<Const<5>>(), 0);
    }

    #[cfg_attr(test, test)]
    /// Verifies compile-time translation from standard const-generic values
    /// into canonical unsigned types (FR-3 of `num-types-design.md`).
    fn test_num_type_constant_generic_hoisting() {
        let _: <Const<5> as Dim>::TypeNum = U5::default();
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
        assert_eq!(U128::USIZE, 128);
        assert_eq!(U256::USIZE, 256);
        assert_eq!(U16384::USIZE, 16384);
    }

    #[cfg_attr(test, test)]
    /// Verifies type-level addition arithmetic at compile-time and runtime
    /// (FR-2 of `num-types-design.md`).
    fn test_num_type_addition() {
        let _: U5 = <U2 as DimAdd<U3>>::Output::default();
        let _: U3 = <U0 as DimAdd<U3>>::Output::default();
        let _: U3 = <U3 as DimAdd<U0>>::Output::default();

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
    /// Verifies `U128 * U128` resolves to `U16384` (FR-2 of
    /// `num-types-design.md`).
    fn test_num_type_large_product() {
        let _: U16384 = <U128 as DimMul<U128>>::Output::default();
        assert_eq!(<<U128 as DimMul<U128>>::Output as Dim>::USIZE, 16384);
    }

    #[cfg_attr(test, test)]
    /// Former Peano overflow pairs must compile under binary encoding.
    fn test_num_type_former_peano_overflow_pairs() {
        assert_eq!(<<U1 as DimAdd<U126>>::Output as Dim>::USIZE, 127);
        assert_eq!(<<U126 as DimMul<U1>>::Output as Dim>::USIZE, 126);
        assert_eq!(<<U127 as DimMul<U2>>::Output as Dim>::USIZE, 254);
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
        fn assert_bounds<A, B, Max, Min>()
        where
            A: DimMax<B, Output = Max> + DimMin<B, Output = Min>,
            B: Dim,
            Max: Dim,
            Min: Dim,
        {
        }

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

    #[cfg_attr(test, test)]
    /// Verifies that unnamed C-1 values implement Dim and arithmetic correctly
    /// (C-1 and FR-3 of `num-types-design.md`).
    fn test_num_type_unnamed_c1_constants() {
        assert_eq!(<Const<200> as Dim>::USIZE, 200);
        assert_eq!(<Const<750> as Dim>::USIZE, 750);
        assert_eq!(<Const<1024> as Dim>::USIZE, 1024);
        assert_eq!(
            <<Const<200> as DimAdd<Const<300>>>::Output as Dim>::USIZE,
            500
        );
        assert_eq!(
            <<Const<750> as DimSub<Const<250>>>::Output as Dim>::USIZE,
            500
        );
        assert_eq!(
            <<Const<200> as DimMax<Const<750>>>::Output as Dim>::USIZE,
            750
        );
        assert_eq!(
            <<Const<200> as DimMin<Const<750>>>::Output as Dim>::USIZE,
            200
        );
    }

    #[cfg_attr(test, test)]
    /// Verifies type-level bitwise AND operations and power-of-two invariants
    /// (N & (N - 1) == 0) (FR-5 and V&V §6.1.2 of `num-types-design.md`).
    fn test_num_type_bitwise_and() {
        let _: U0 = <U1 as DimBitAnd<U0>>::Output::default();
        let _: U0 = <U0 as DimBitAnd<U1>>::Output::default();
        let _: U0 = <U2 as DimBitAnd<U1>>::Output::default();
        let _: U0 = <U4 as DimBitAnd<U3>>::Output::default();
        let _: U0 = <U8 as DimBitAnd<U7>>::Output::default();
        let _: U0 = <U16 as DimBitAnd<U15>>::Output::default();
        let _: U0 = <U128 as DimBitAnd<U127>>::Output::default();
        let _: U10 = <U15 as DimBitAnd<U10>>::Output::default();
        let _: U3 = <U7 as DimBitAnd<U3>>::Output::default();

        assert_eq!(
            <<Const<1024> as DimBitAnd<Const<1023>>>::Output as Dim>::USIZE,
            0
        );
    }

    #[cfg_attr(test, test)]
    /// Verifies type-level bitwise OR operations and masking identity
    /// (A | 0 == A) (FR-5 and V&V §6.1.2 of `num-types-design.md`).
    fn test_num_type_bitwise_or() {
        let _: U0 = <U0 as DimBitOr<U0>>::Output::default();
        let _: U5 = <U5 as DimBitOr<U0>>::Output::default();
        let _: U5 = <U0 as DimBitOr<U5>>::Output::default();
        let _: U5 = <U5 as DimBitOr<U5>>::Output::default();
        let _: U15 = <U10 as DimBitOr<U5>>::Output::default();
        let _: U127 = <U64 as DimBitOr<U63>>::Output::default();

        assert_eq!(
            <<Const<1024> as DimBitOr<Const<1>>>::Output as Dim>::USIZE,
            1025
        );
    }

    #[cfg_attr(test, test)]
    /// Verifies type-level bitwise XOR operations, self-cancellation (A ^ A == 0),
    /// and identity (A ^ 0 == A) (FR-5 and V&V §6.1.2 of `num-types-design.md`).
    fn test_num_type_bitwise_xor() {
        let _: U0 = <U0 as DimBitXor<U0>>::Output::default();
        let _: U0 = <U5 as DimBitXor<U5>>::Output::default();
        let _: U0 = <U128 as DimBitXor<U128>>::Output::default();
        let _: U5 = <U5 as DimBitXor<U0>>::Output::default();
        let _: U5 = <U0 as DimBitXor<U5>>::Output::default();
        let _: U5 = <U15 as DimBitXor<U10>>::Output::default();

        assert_eq!(
            <<Const<1024> as DimBitXor<Const<1024>>>::Output as Dim>::USIZE,
            0
        );
    }

    #[cfg_attr(test, test)]
    /// Verifies in-bounds Const arithmetic for boundary values (Const<1024>, Const<16384>)
    /// (V&V §6.1 item 6 of `num-types-design.md`).
    fn test_num_type_in_bounds_const_arithmetic() {
        assert_eq!(U1024::USIZE, 1024);
        assert_eq!(<Const<1024> as Dim>::USIZE, 1024);
        assert_eq!(<Const<16384> as Dim>::USIZE, 16384);
        assert_eq!(
            <<Const<1024> as DimAdd<Const<1024>>>::Output as Dim>::USIZE,
            2048
        );
        assert_eq!(
            <<Const<128> as DimMul<Const<128>>>::Output as Dim>::USIZE,
            16384
        );
        assert_eq!(
            <<Const<100> as DimMul<Const<100>>>::Output as Dim>::USIZE,
            10000
        );
    }
}
