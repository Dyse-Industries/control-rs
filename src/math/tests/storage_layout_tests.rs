//! Layout oracle for packed structured storage: every owned type and view is
//! swept over all in-range and just-out-of-range indices, and compared with a
//! reference built by enumerating the packing order.

use crate::math::complex_num::{Complex, Complex64};
use crate::math::storage::{
    Diag, DiagonalStorage, DiagonalView, DiagonalViewMut,
    HermitianPackedStorage, HermitianPackedView, HermitianPackedViewMut,
    PackedStorage, PackedStorageMut, StorageError, SymmetricPackedStorage,
    SymmetricPackedView, SymmetricPackedViewMut, TriangularPackedStorage,
    TriangularPackedView, TriangularPackedViewMut, UpLo,
};

/// Order of the matrices under test. Four keeps `j * (j + 1) / 2` and its
/// lower-triangle counterpart distinct from their `%` and `*` variants.
const LEN: usize = 10;

/// Order of the matrices under test. Four keeps `j * (j + 1) / 2` distinct from
/// its `%` and `*` variants.
const N: usize = 4;

const UPLOS: [UpLo; 2] = [UpLo::Upper, UpLo::Lower];

type C = Complex64;

/// A family together with the stored triangle.
#[derive(Clone, Copy)]
struct Layout {
    family: Family,
    uplo: UpLo,
}

/// Which structured matrix a storage represents.
#[derive(Clone, Copy)]
enum Family {
    Diagonal,
    Hermitian,
    Symmetric,
    Triangular(Diag),
}

fn c(re: f64, im: f64) -> C {
    Complex::new(re, im)
}

fn zero() -> C {
    c(0.0, 0.0)
}

fn fl(k: usize) -> f64 {
    f64::from(u8::try_from(k).unwrap())
}

/// Distinct complex data for slot `k`.
fn slot_value(k: usize) -> C {
    c(1.0 + fl(k), 0.25 + 0.5 * fl(k))
}

/// Data for every slot of the largest packed layout; smaller layouts use a prefix.
fn data() -> [C; LEN] {
    core::array::from_fn(slot_value)
}

fn in_triangle(uplo: UpLo, i: usize, j: usize) -> bool {
    match uplo {
        UpLo::Upper => i <= j,
        UpLo::Lower => i >= j,
    }
}

/// Packed position of `(i, j)` by enumerating the column-packed order.
fn slot(uplo: UpLo, i: usize, j: usize) -> Option<usize> {
    let mut position = 0usize;
    for col in 0..N {
        for row in 0..N {
            if !in_triangle(uplo, row, col) {
                continue;
            }
            if (row, col) == (i, j) {
                return Some(position);
            }
            position = position.saturating_add(1);
        }
    }
    None
}

/// The stored slot backing `(i, j)`, mirroring across the diagonal when the
/// entry lies outside the stored triangle.
fn backing_slot(
    family: Family,
    uplo: UpLo,
    i: usize,
    j: usize,
) -> Option<usize> {
    match family {
        Family::Diagonal => (i == j).then_some(i),
        _ => slot(uplo, i, j),
    }
}

/// Expected algebraic entry at `(i, j)`, `None` when out of range.
fn expected_value(
    layout: Layout,
    stored: &[C],
    i: usize,
    j: usize,
) -> Option<C> {
    let Layout { family, uplo } = layout;
    if i >= N || j >= N {
        return None;
    }
    let at = |k: Option<usize>| k.and_then(|k| stored.get(k)).copied();
    Some(match family {
        Family::Diagonal => {
            at(backing_slot(family, uplo, i, j)).unwrap_or_else(zero)
        }
        Family::Symmetric => {
            at(slot(uplo, i, j).or_else(|| slot(uplo, j, i))).unwrap()
        }
        Family::Hermitian => slot(uplo, i, j).map_or_else(
            || at(slot(uplo, j, i)).unwrap().conj(),
            |k| at(Some(k)).unwrap(),
        ),
        Family::Triangular(diag) => {
            if diag == Diag::Unit && i == j {
                c(1.0, 0.0)
            } else {
                at(slot(uplo, i, j)).unwrap_or_else(zero)
            }
        }
    })
}

fn check_reads<S: PackedStorage<C>>(
    s: &S,
    layout: Layout,
    stored: &[C],
    what: &str,
) {
    let Layout { family, uplo } = layout;
    for i in 0..=N.saturating_add(1) {
        for j in 0..=N.saturating_add(1) {
            let want_index = if i < N && j < N {
                backing_slot(family, uplo, i, j).filter(|_| match family {
                    Family::Diagonal => true,
                    _ => in_triangle(uplo, i, j),
                })
            } else {
                None
            };
            assert_eq!(
                s.packed_index(i, j),
                want_index,
                "{what}: packed_index({i},{j})"
            );
            if let Some(k) = want_index {
                assert_eq!(
                    s.packed_index_unchecked(i, j),
                    k,
                    "{what}: packed_index_unchecked({i},{j})"
                );
            }
            let want = expected_value(layout, stored, i, j);
            assert_eq!(s.value(i, j), want, "{what}: value({i},{j})");
            if let Some(w) = want {
                assert_eq!(
                    s.value_unchecked(i, j),
                    w,
                    "{what}: value_unchecked({i},{j})"
                );
            }
        }
    }
}

/// The error `set(i, j, _)` must report, or `None` when it must succeed.
fn expected_set_error(
    family: Family,
    uplo: UpLo,
    i: usize,
    j: usize,
) -> Option<StorageError> {
    if i >= N || j >= N {
        return Some(StorageError::OutOfBounds);
    }
    match family {
        Family::Diagonal if i != j => {
            Some(StorageError::InvalidStructuralInvariant)
        }
        Family::Triangular(Diag::Unit) if i == j => {
            Some(StorageError::ImmutableUnitDiagonal)
        }
        Family::Diagonal => None,
        _ if !in_triangle(uplo, i, j) => {
            Some(StorageError::InvalidStructuralInvariant)
        }
        _ => None,
    }
}

fn check_writes<S: PackedStorageMut<C>>(
    s: &mut S,
    family: Family,
    uplo: UpLo,
    what: &str,
) {
    let mut model = [zero(); LEN];
    for (slot_model, value) in model.iter_mut().zip(s.as_slice()) {
        *slot_model = *value;
    }
    let live = s.as_slice().len();
    for i in 0..=N.saturating_add(1) {
        for j in 0..=N.saturating_add(1) {
            let real = i == j;
            let val =
                c(100.0 + fl(i) * 10.0 + fl(j), if real { 0.0 } else { 0.5 });
            let want = expected_set_error(family, uplo, i, j);
            let got = s.set(i, j, val).err();
            assert_eq!(got, want, "{what}: set({i},{j})");
            if want.is_none() {
                let k = backing_slot(family, uplo, i, j).unwrap();
                if let Some(entry) = model.get_mut(k) {
                    *entry = val;
                }
            }
            assert_eq!(
                Some(s.as_slice()),
                model.get(..live),
                "{what}: contents after set({i},{j})"
            );
        }
    }
    // The unchecked writer targets the same slots.
    for i in 0..N {
        for j in 0..N {
            let stored_here = match family {
                Family::Diagonal => i == j,
                Family::Triangular(Diag::Unit) => {
                    in_triangle(uplo, i, j) && i != j
                }
                _ => in_triangle(uplo, i, j),
            };
            if !stored_here {
                continue;
            }
            let val = c(500.0 + fl(i) * 10.0 + fl(j), 0.0);
            unsafe { s.set_unchecked(i, j, val) };
            let k = backing_slot(family, uplo, i, j).unwrap();
            if let Some(entry) = model.get_mut(k) {
                *entry = val;
            }
            assert_eq!(
                Some(s.as_slice()),
                model.get(..live),
                "{what}: set_unchecked({i},{j})"
            );
        }
    }
}

#[test]
fn diagonal_storage_layout() {
    let all = data();
    let stored = all.get(..N).unwrap();
    let array: [C; N] = core::array::from_fn(slot_value);
    let owned = DiagonalStorage::<C, N>::from_array(array);
    check_reads(
        &owned,
        Layout {
            family: Family::Diagonal,
            uplo: UpLo::Upper,
        },
        stored,
        "DiagonalStorage",
    );
    let mut owned = DiagonalStorage::<C, N>::from_array(array);
    check_writes(&mut owned, Family::Diagonal, UpLo::Upper, "DiagonalStorage");

    let view = DiagonalView::<C, N>::new(stored).unwrap();
    check_reads(
        &view,
        Layout {
            family: Family::Diagonal,
            uplo: UpLo::Upper,
        },
        stored,
        "DiagonalView",
    );
    let mut buf = data();
    let mut view =
        DiagonalViewMut::<C, N>::new(buf.get_mut(..N).unwrap()).unwrap();
    check_reads(
        &view,
        Layout {
            family: Family::Diagonal,
            uplo: UpLo::Upper,
        },
        stored,
        "DiagonalViewMut",
    );
    check_writes(&mut view, Family::Diagonal, UpLo::Upper, "DiagonalViewMut");
}

fn packed_array() -> [C; LEN] {
    core::array::from_fn(slot_value)
}

#[test]
fn symmetric_packed_layout() {
    let stored = data();
    for uplo in UPLOS {
        let owned =
            SymmetricPackedStorage::<C, N, LEN>::new(packed_array(), uplo);
        check_reads(
            &owned,
            Layout {
                family: Family::Symmetric,
                uplo,
            },
            &stored,
            "SymmetricPackedStorage",
        );
        let mut owned =
            SymmetricPackedStorage::<C, N, LEN>::new(packed_array(), uplo);
        check_writes(
            &mut owned,
            Family::Symmetric,
            uplo,
            "SymmetricPackedStorage",
        );

        let view = SymmetricPackedView::<C, N>::new(&stored, uplo).unwrap();
        check_reads(
            &view,
            Layout {
                family: Family::Symmetric,
                uplo,
            },
            &stored,
            "SymmetricPackedView",
        );
        let mut buf = data();
        let mut view =
            SymmetricPackedViewMut::<C, N>::new(&mut buf, uplo).unwrap();
        check_reads(
            &view,
            Layout {
                family: Family::Symmetric,
                uplo,
            },
            &stored,
            "SymmetricPackedViewMut",
        );
        check_writes(
            &mut view,
            Family::Symmetric,
            uplo,
            "SymmetricPackedViewMut",
        );
    }
}

#[test]
fn hermitian_packed_layout() {
    let stored = data();
    for uplo in UPLOS {
        let owned =
            HermitianPackedStorage::<C, N, LEN>::new(packed_array(), uplo);
        check_reads(
            &owned,
            Layout {
                family: Family::Hermitian,
                uplo,
            },
            &stored,
            "HermitianPackedStorage",
        );
        let mut owned =
            HermitianPackedStorage::<C, N, LEN>::new(packed_array(), uplo);
        check_writes(
            &mut owned,
            Family::Hermitian,
            uplo,
            "HermitianPackedStorage",
        );

        let view = HermitianPackedView::<C, N>::new(&stored, uplo).unwrap();
        check_reads(
            &view,
            Layout {
                family: Family::Hermitian,
                uplo,
            },
            &stored,
            "HermitianPackedView",
        );
        let mut buf = data();
        let mut view =
            HermitianPackedViewMut::<C, N>::new(&mut buf, uplo).unwrap();
        check_reads(
            &view,
            Layout {
                family: Family::Hermitian,
                uplo,
            },
            &stored,
            "HermitianPackedViewMut",
        );
        check_writes(
            &mut view,
            Family::Hermitian,
            uplo,
            "HermitianPackedViewMut",
        );
    }
}

#[test]
fn hermitian_diagonals_must_stay_real() {
    for uplo in UPLOS {
        let mut owned =
            HermitianPackedStorage::<C, N, LEN>::new(packed_array(), uplo);
        for i in 0..N {
            assert_eq!(
                owned.set(i, i, c(1.0, 1.0)),
                Err(StorageError::InvalidHermitianDiagonal)
            );
            assert!(owned.set(i, i, c(1.0, 0.0)).is_ok());
        }
    }
}

#[test]
fn triangular_packed_layout() {
    let stored = data();
    for uplo in UPLOS {
        for diag in [Diag::NonUnit, Diag::Unit] {
            let family = Family::Triangular(diag);
            let owned = TriangularPackedStorage::<C, N, LEN>::new(
                packed_array(),
                uplo,
                diag,
            );
            check_reads(
                &owned,
                Layout { family, uplo },
                &stored,
                "TriangularPackedStorage",
            );
            let mut owned = TriangularPackedStorage::<C, N, LEN>::new(
                packed_array(),
                uplo,
                diag,
            );
            check_writes(&mut owned, family, uplo, "TriangularPackedStorage");

            let view =
                TriangularPackedView::<C, N>::new(&stored, uplo, diag).unwrap();
            check_reads(
                &view,
                Layout { family, uplo },
                &stored,
                "TriangularPackedView",
            );
            let mut buf = data();
            let mut view =
                TriangularPackedViewMut::<C, N>::new(&mut buf, uplo, diag)
                    .unwrap();
            check_reads(
                &view,
                Layout { family, uplo },
                &stored,
                "TriangularPackedViewMut",
            );
            check_writes(&mut view, family, uplo, "TriangularPackedViewMut");
        }
    }
}
