//! LU solve of a 2x2 linear system.
//!
//! Run with `cargo run --example matrix`.
#![allow(clippy::expect_used)]

use control_rs::matrix::{LuDecomposition, Owned};

fn fmt2(m: &Owned<f64, 2, 2>) -> String {
    format!(
        "[[{:.4}, {:.4}], [{:.4}, {:.4}]]",
        m.get(0, 0).copied().expect("A00"),
        m.get(0, 1).copied().expect("A01"),
        m.get(1, 0).copied().expect("A10"),
        m.get(1, 1).copied().expect("A11")
    )
}

fn main() {
    let a = Owned::<f64, 2, 2>::from_rows([[3.0, 1.0], [1.0, 2.0]]);
    let b = Owned::<f64, 2, 1>::from_column([1.0, 0.0]);
    println!("A = {},  I = {}", fmt2(&a), fmt2(&Owned::identity()));
    println!(
        "b = [{:.4}, {:.4}]^T",
        b.get(0, 0).copied().expect("b row 0"),
        b.get(1, 0).copied().expect("b row 1")
    );

    let lu = LuDecomposition::decompose(a).expect("LU factorization");
    let mut x = b;
    lu.solve_mut(&mut x).expect("LU solve");
    println!(
        "x = [{:.6}, {:.6}]^T",
        x.get(0, 0).copied().expect("x row 0"),
        x.get(1, 0).copied().expect("x row 1")
    );
}
