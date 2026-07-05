#![allow(
    missing_docs,
    clippy::unwrap_used,
    clippy::semicolon_if_nothing_returned,
    clippy::arithmetic_side_effects
)]

use control_rs::polynomial::{Constant, Line, Polynomial, StaticPolynomial};
use criterion::{Criterion, black_box, criterion_group, criterion_main};

pub struct PolynomialBenchmarkSuite;

impl PolynomialBenchmarkSuite {
    /// Runs the polynomial benchmark suite.
    ///
    /// # Clippy Allow explanation
    /// - We allow `clippy::float_cmp` here to perform precise zero check assertions on evaluated floats.
    /// - We allow `clippy::missing_panics_doc` since this is a benchmark suite where unwraps ensure configuration correctness.
    #[allow(clippy::float_cmp, clippy::missing_panics_doc)]
    pub fn run(c: &mut Criterion) {
        // --- Successful Cases ---

        // 1. Polynomial Evaluation (Constant)
        let poly_const = Constant::new(5.0);
        c.bench_function("poly_eval_constant", |b| {
            b.iter(|| {
                let _ = poly_const.evaluate(black_box(10.0));
            });
        });

        // 2. Polynomial Evaluation (Line)
        let poly_line = Line::new(2.0, 3.0);
        c.bench_function("poly_eval_line", |b| {
            b.iter(|| {
                let _ = poly_line.evaluate(black_box(10.0));
            });
        });

        // 3. Polynomial Evaluation (Quadratic)
        let poly_quad = StaticPolynomial::from_coefficients([1.0, 2.0, 3.0]);
        c.bench_function("poly_eval_quadratic", |b| {
            b.iter(|| {
                let _ = poly_quad.evaluate(black_box(10.0));
            });
        });

        // 4. Polynomial Addition
        let p1 = StaticPolynomial::from_coefficients([1.0, 2.0, 3.0]);
        let p2 = StaticPolynomial::from_coefficients([4.0, 5.0, 6.0]);
        c.bench_function("poly_addition_success", |b| {
            b.iter(|| {
                let _ = black_box(p1) + black_box(p2);
            });
        });

        // 5. Polynomial Multiplication
        let p_m1 = StaticPolynomial::from_coefficients([1.0, 2.0]);
        let p_m2 = StaticPolynomial::from_coefficients([3.0, 4.0]);
        c.bench_function("poly_multiplication_success", |b| {
            b.iter(|| {
                let _prod: StaticPolynomial<f64, 3> =
                    black_box(p_m1).mul_poly(&black_box(p_m2));
            });
        });

        // --- Failing / Edge Cases ---

        // 6. Evaluating Empty Polynomial (Edge Case)
        // Note: Commented out because evaluating a 0-sized polynomial now correctly fails to compile
        // due to the compile-time `NonZeroDim` trait bound on the `Polynomial` implementation.
        // let poly_empty = StaticPolynomial::<f64, 0>::from_coefficients([]);
        // c.bench_function("poly_eval_empty_edge", |b| {
        //     b.iter(|| {
        //         let res = poly_empty.evaluate(black_box(10.0));
        //         debug_assert_eq!(res, 0.0);
        //     });
        // });

        // 7. Finding Degree/Leading Coefficient of Zero Polynomial (Failure/None case)
        let poly_zero = Constant::new(0.0);
        c.bench_function("poly_zero_leading_coeff_failure", |b| {
            b.iter(|| {
                let res = poly_zero.leading_coefficient();
                debug_assert!(res.is_none());
            });
        });
    }
}

fn run_suite(c: &mut Criterion) {
    PolynomialBenchmarkSuite::run(c);
}

criterion_group!(benches, run_suite);
criterion_main!(benches);
