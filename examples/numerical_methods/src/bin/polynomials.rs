//! Demonstration of the generic compile-time Polynomial struct.
use control_rs::polynomial::{Constant, Line, Polynomial};

fn main() {
    println!("=== Polynomial Demonstration ===");

    // 1. Creation from coefficients
    // Coefficients are stored in ascending order: p(x) = c0 + c1*x + c2*x^2 + ...
    // p1(x) = 1 + 2x + 3x^2
    let p1 = Polynomial::from_coefficients([1.0, 2.0, 3.0]);
    println!("p1(x) coefficients (ascending): {:?}", p1.coefficients());
    println!("p1(x) degree: {:?}", p1.degree());
    println!("p1(x) leading coefficient: {:?}", p1.leading_coefficient());

    // Creation from descending order (standard mathematical notation: ax^2 + bx + c)
    // p2(x) = 3x^2 + 2x + 1
    let p2 = Polynomial::from_descending([3.0, 2.0, 1.0]);
    println!("p2(x) coefficients (ascending): {:?}", p2.coefficients());

    // 2. Polynomial Evaluation
    // Evaluates p1(x) at x = 2.0: 1.0 + 2.0*(2.0) + 3.0*(4.0) = 17.0
    let x_val = 2.0;
    println!("p1({}) = {}", x_val, p1.evaluate(x_val));

    // 3. Aliases: Constant and Line
    let c = Constant::new(5.0); // p(x) = 5
    let l = Line::new(2.0, 3.0); // p(x) = 2x + 3
    println!("Constant polynomial evaluation c(10): {}", c.evaluate(10.0));
    println!("Linear polynomial evaluation l(4): {}", l.evaluate(4.0));

    // 4. Arithmetic (Addition / Subtraction / Scaling)
    let p_sum = p1 + p2;
    println!("(p1 + p2)(x) coefficients: {:?}", p_sum.coefficients());

    let p_diff = p1 - p2;
    println!("(p1 - p2)(x) coefficients: {:?}", p_diff.coefficients());

    let p_scaled = p1 * 2.5;
    println!("(p1 * 2.5)(x) coefficients: {:?}", p_scaled.coefficients());

    // 5. Polynomial Multiplication
    // Uses the underlying DSP convolution to compute multiplication.
    // p_a(x) = 1 + 2x (length 2)
    // p_b(x) = 4 + 3x (length 2)
    // p_prod(x) = (1 + 2x)(4 + 3x) = 4 + 11x + 6x^2 (length 3)
    let p_a = Polynomial::from_coefficients([1.0, 2.0]);
    let p_b = Polynomial::from_coefficients([4.0, 3.0]);
    let p_prod: Polynomial<f64, 3> = p_a.mul_poly(&p_b);
    println!("(p_a * p_b)(x) coefficients: {:?}", p_prod.coefficients());

    // 6. Polynomial Derivative
    let dp1 = p1.derivative(); // 2 + 6x
    println!("Derivative of p1(x) coefficients: {:?}", dp1.coefficients());

    // 7. Polynomial Long Division
    // divides A(x) = 3x^2 + 5x + 6 by B(x) = x + 2
    // A(x) = (3x - 1)*(x + 2) + 8
    // Quotient = 3x - 1 (coefficients [-1.0, 3.0])
    // Remainder = 8 (coefficients [8.0])
    let a = Polynomial::from_coefficients([6.0, 5.0, 3.0]);
    let b = Polynomial::from_coefficients([2.0, 1.0]);
    #[allow(clippy::type_complexity)]
    let (q, r): (Polynomial<f64, 2>, Polynomial<f64, 1>) = a.div_rem(&b);
    println!("Polynomial Long Division:");
    println!("  Dividend: 3x^2 + 5x + 6");
    println!("  Divisor: x + 2");
    println!("  Quotient coefficients: {:?}", q.coefficients());
    println!("  Remainder coefficients: {:?}", r.coefficients());
}
