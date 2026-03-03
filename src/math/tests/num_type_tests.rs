use crate::math::num_types::{
    Const, Dim, DimAdd, DimMax, DimMin, DimMul, DimSub, U0, U1, U2, U3, U4, U5,
    U6, U10, U12, U15, U32,
};
use core::marker::PhantomData;

#[test]
fn test_dimension_values() {
    assert_eq!(U0::DIM, 0);
    assert_eq!(U1::DIM, 1);
    assert_eq!(U15::DIM, 15);
    assert_eq!(U32::DIM, 32);
}

#[test]
fn test_addition() {
    // Compile-time type structure assertions
    let _: U5 = <U2 as DimAdd<U3>>::Output::default();
    let _: U3 = <U0 as DimAdd<U3>>::Output::default();
    let _: U3 = <U3 as DimAdd<U0>>::Output::default();

    // Constant value assertions
    assert_eq!(<<U10 as DimAdd<U5>>::Output as Dim>::DIM, 15);
}

#[test]
fn test_subtraction() {
    let _: U2 = <U5 as DimSub<U3>>::Output::default();
    let _: U0 = <U5 as DimSub<U5>>::Output::default();
    let _: U5 = <U5 as DimSub<U0>>::Output::default();

    assert_eq!(<<U15 as DimSub<U10>>::Output as Dim>::DIM, 5);
}

#[test]
fn test_multiplication() {
    let _: U6 = <U2 as DimMul<U3>>::Output::default();
    let _: U0 = <U5 as DimMul<U0>>::Output::default();
    let _: U0 = <U0 as DimMul<U5>>::Output::default();

    assert_eq!(<<U4 as DimMul<U5>>::Output as Dim>::DIM, 20);
}

#[test]
fn test_maximum() {
    let _: U5 = <U2 as DimMax<U5>>::Output::default();
    let _: U5 = <U5 as DimMax<U2>>::Output::default();
    let _: U5 = <U5 as DimMax<U5>>::Output::default();

    assert_eq!(<<U12 as DimMax<U4>>::Output as Dim>::DIM, 12);
}

#[test]
fn test_minimum() {
    let _: U2 = <U2 as DimMin<U5>>::Output::default();
    let _: U2 = <U5 as DimMin<U2>>::Output::default();
    let _: U5 = <U5 as DimMin<U5>>::Output::default();

    assert_eq!(<<U12 as DimMin<U4>>::Output as Dim>::DIM, 4);
}

struct TestStorage<C: Dim>(PhantomData<C>);
fn concat_static_arrays<const A: usize, const B: usize, C: Dim>(
    _: [f32; A],
    _: [f32; B],
) -> TestStorage<C>
where
    Const<A>: DimAdd<Const<B>, Output = C>,
{
    TestStorage(PhantomData)
}

fn concat_static_types<A, B, C>(
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

#[test]
fn test_concat_static() {
    let l1: TestStorage<U1> = TestStorage(PhantomData);
    let l2: TestStorage<U2> = TestStorage(PhantomData);
    let l3: TestStorage<U3> = TestStorage(PhantomData);

    let l_test: TestStorage<U3> = concat_static_types(l1, l2);
    assert_eq!(l3.0, l_test.0);

    let l_array_test: TestStorage<U3> = concat_static_arrays([0.0], [0.0, 0.0]);
    assert_eq!(l3.0, l_array_test.0);
}
