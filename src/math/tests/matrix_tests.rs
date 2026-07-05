use crate::math::ArithmeticError;
use crate::math::matrix::{
    LowerTriangular, Matrix, MatrixMN, RowVector, SquareMatrix, Symmetric,
    UpperTriangular, Vector,
};

#[test]
fn test_matrix_construction_and_getters() {
    let mut m = Matrix::new([
        [1.0, 2.0], // Column 0: rows 0 and 1
        [3.0, 4.0], // Column 1: rows 0 and 1
        [5.0, 6.0], // Column 2: rows 0 and 1
    ]);
    // 2x3 matrix:
    // [ 1.0, 3.0, 5.0 ]
    // [ 2.0, 4.0, 6.0 ]
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 3);

    assert_eq!(m.get(0, 0), Some(&1.0));
    assert_eq!(m.get(1, 0), Some(&2.0));
    assert_eq!(m.get(0, 1), Some(&3.0));
    assert_eq!(m.get(1, 1), Some(&4.0));
    assert_eq!(m.get(0, 2), Some(&5.0));
    assert_eq!(m.get(1, 2), Some(&6.0));
    assert_eq!(m.get(2, 0), None);
    assert_eq!(m.get(0, 3), None);

    // Mutation
    *m.get_mut(1, 1).unwrap() = 10.0;
    assert_eq!(m.get(1, 1), Some(&10.0));
}

#[test]
fn test_matrix_construction_and_aliases() {
    let m: MatrixMN<f32, 2, 3> =
        Matrix::new([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
    assert_eq!(m.rows(), 2);
    assert_eq!(m.cols(), 3);

    let sq: SquareMatrix<f32, 2> = Matrix::new([[1.0, 2.0], [3.0, 4.0]]);
    assert_eq!(sq.rows(), 2);
    assert_eq!(sq.cols(), 2);

    let v: Vector<f32, 3> = Matrix::new([[1.0, 2.0, 3.0]]);
    assert_eq!(v.rows(), 3);
    assert_eq!(v.cols(), 1);

    let rv: RowVector<f32, 3> = Matrix::new([[1.0], [2.0], [3.0]]);
    assert_eq!(rv.rows(), 1);
    assert_eq!(rv.cols(), 3);
}

#[test]
fn test_matrix_addition_and_subtraction() {
    let mut m1 = Matrix::new([[1, 2], [3, 4]]);
    let m2 = Matrix::new([[10, 20], [30, 40]]);
    m1 += &m2;
    assert_eq!(m1.get(0, 0), Some(&11));
    assert_eq!(m1.get(1, 0), Some(&22));
    assert_eq!(m1.get(0, 1), Some(&33));
    assert_eq!(m1.get(1, 1), Some(&44));

    m1 -= &m2;
    assert_eq!(m1.get(0, 0), Some(&1));
    assert_eq!(m1.get(1, 0), Some(&2));
}

#[test]
fn test_upper_triangular() {
    let m = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let ut_result = UpperTriangular::new(m);
    assert!(ut_result.is_ok());
    let mut ut = ut_result.unwrap();

    // Upper triangle read
    assert_eq!(ut.get(0, 0), Some(&1.0));
    assert_eq!(ut.get(0, 1), Some(&4.0));
    // Lower triangle read (returns storage value)
    assert_eq!(ut.get(1, 0), Some(&2.0));
    assert_eq!(ut.get(2, 0), Some(&3.0));

    // Upper triangle mutation
    assert!(ut.get_mut(0, 1).is_some());
    *ut.get_mut(0, 1).unwrap() = 40.0;
    assert_eq!(ut.get(0, 1), Some(&40.0));

    // Lower triangle mutation forbidden
    assert!(ut.get_mut(1, 0).is_none());
}

#[test]
fn test_lower_triangular() {
    let m = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
    let mut lt = LowerTriangular::new(m).unwrap();

    // Lower triangle read
    assert_eq!(lt.get(1, 0), Some(&2.0));
    assert_eq!(lt.get(2, 1), Some(&6.0));

    // Lower triangle mutation
    assert!(lt.get_mut(1, 0).is_some());
    *lt.get_mut(1, 0).unwrap() = 20.0;
    assert_eq!(lt.get(1, 0), Some(&20.0));

    // Upper triangle mutation forbidden
    assert!(lt.get_mut(0, 1).is_none());
}

#[test]
fn test_symmetric() {
    let m = Matrix::new([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]);
    let mut sym = Symmetric::new(m).unwrap();

    // Mirror set update
    assert!(sym.set(0, 1, 5.0).is_ok());
    assert_eq!(sym.get(0, 1), Some(&5.0));
    assert_eq!(sym.get(1, 0), Some(&5.0));

    // Out of bounds set error
    assert_eq!(
        sym.set(3, 0, 1.0).err(),
        Some(ArithmeticError::DomainViolation)
    );
}
