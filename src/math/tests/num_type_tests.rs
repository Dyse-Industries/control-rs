//! # Num Types
//!
//! The tests must verify that the compiler accurately translates constant values into their
//! corresponding Peano representations.
//!
//! The test suite must comprehensively cover the operations for addition, subtraction,
//! multiplication, minimum, and maximum behaviors.
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

#[cfg_attr(not(test), control_rs_macros::hil_suite)]
/// Basic Peano compile-time dimension type bounds and zero-footprint tests.
pub mod num_type_basic {
    use crate::math::num_types::{
        Const, Dim, DimAdd, DimMax, DimMin, DimMul, DimSub, S, U0, U1, U2, U3,
        U4, U5, U6, U10, U12, U15, U32, Z,
    };
    use core::marker::PhantomData;
    use core::mem;

    struct TestStorage<C: Dim>(PhantomData<C>);

    #[cfg_attr(test, test)]
    fn test_num_type_zero_byte_footprint_basic() {
        assert_eq!(mem::size_of::<Z>(), 0);
        assert_eq!(mem::size_of::<S<Z>>(), 0);
        assert_eq!(mem::size_of::<S<S<Z>>>(), 0);
    }

    #[cfg_attr(test, test)]
    fn test_num_type_constant_generic_hoisting_basic() {
        let _: <Const<5> as Dim>::PeanoTypeNum = U5::default();
    }

    #[cfg_attr(test, test)]
    fn test_num_type_addition_commutativity_basic() {
        let _: <U2 as DimAdd<U3>>::Output =
            <<U3 as DimAdd<U2>>::Output>::default();
    }

    #[cfg_attr(test, test)]
    fn test_num_type_dimension_values_basic() {
        assert_eq!(U0::DIM, 0);
        assert_eq!(U1::DIM, 1);
        assert_eq!(U15::DIM, 15);
        assert_eq!(U32::DIM, 32);
    }

    #[cfg_attr(test, test)]
    fn test_num_type_addition_basic() {
        // Compile-time type structure assertions
        let _: U5 = <U2 as DimAdd<U3>>::Output::default();
        let _: U3 = <U0 as DimAdd<U3>>::Output::default();
        let _: U3 = <U3 as DimAdd<U0>>::Output::default();

        // Constant value assertions
        assert_eq!(<<U10 as DimAdd<U5>>::Output as Dim>::DIM, 15);
    }

    #[cfg_attr(test, test)]
    fn test_num_type_addition_base_cases_basic() {
        let _: U5 = <U5 as DimAdd<U0>>::Output::default();
        let _: U5 = <U0 as DimAdd<U5>>::Output::default();
    }

    #[cfg_attr(test, test)]
    fn test_num_type_subtraction_basic() {
        let _: U2 = <U5 as DimSub<U3>>::Output::default();
        let _: U0 = <U5 as DimSub<U5>>::Output::default();
        let _: U5 = <U5 as DimSub<U0>>::Output::default();

        assert_eq!(<<U15 as DimSub<U10>>::Output as Dim>::DIM, 5);
    }

    #[cfg_attr(test, test)]
    fn test_num_type_multiplication_basic() {
        let _: U6 = <U2 as DimMul<U3>>::Output::default();
        let _: U0 = <U5 as DimMul<U0>>::Output::default();
        let _: U0 = <U0 as DimMul<U5>>::Output::default();

        assert_eq!(<<U4 as DimMul<U5>>::Output as Dim>::DIM, 20);
    }

    #[cfg_attr(test, test)]
    fn test_num_type_multiplication_recursion_depth_limit_basic() {
        // Test a multiplication that forces deep recursion in the trait solver
        let _: U32 = <U32 as DimMul<U1>>::Output::default();
    }

    #[cfg_attr(test, test)]
    fn test_num_type_maximum_basic() {
        let _: U5 = <U2 as DimMax<U5>>::Output::default();
        let _: U5 = <U5 as DimMax<U2>>::Output::default();
        let _: U5 = <U5 as DimMax<U5>>::Output::default();

        assert_eq!(<<U12 as DimMax<U4>>::Output as Dim>::DIM, 12);
    }

    #[cfg_attr(test, test)]
    fn test_num_type_minimum_basic() {
        let _: U2 = <U2 as DimMin<U5>>::Output::default();
        let _: U2 = <U5 as DimMin<U2>>::Output::default();
        let _: U5 = <U5 as DimMin<U5>>::Output::default();

        assert_eq!(<<U12 as DimMin<U4>>::Output as Dim>::DIM, 4);
    }

    #[cfg_attr(test, test)]
    fn test_num_type_dynamic_min_max_bounding_basic() {
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
    fn test_num_type_concat_static_basic() {
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
