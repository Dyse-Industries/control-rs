//! Demonstration of the generic compile-time Polynomial struct.
//!
//! This example models a classical closed-loop control system where a DC motor (plant)
//! is controlled by a PI controller. The open-loop and closed-loop transfer functions
//! are calculated algebraically using polynomial arithmetic, evaluation, derivatives, and division.

use control_rs::polynomial::{Constant, Line, Polynomial};

fn main() {
    println!("=== Polynomial Demonstration (Classical Control) ===");

    // 1. Model the Plant: G(s) = 3 / (0.5s + 1)
    // In control-rs, polynomials are defined with coefficients in ascending order:
    // P(s) = c0 + c1*s + c2*s^2 + ...
    // Numerator: N_g(s) = 3.0 (degree 0)
    let n_g = Polynomial::from_coefficients([3.0]);
    // Denominator: D_g(s) = 1.0 + 0.5s (degree 1)
    let d_g = Polynomial::from_coefficients([1.0, 0.5]);

    println!(
        "Plant Numerator N_g(s) coefficients: {:?}",
        n_g.coefficients()
    );
    println!(
        "Plant Denominator D_g(s) coefficients: {:?}",
        d_g.coefficients()
    );

    // 2. Model the PI Controller: C(s) = K_p + K_i/s = (K_p * s + K_i) / s
    // Let's set K_p = 2.0 and K_i = 1.0.
    // Numerator: N_c(s) = 1.0 + 2.0s (degree 1)
    let n_c = Polynomial::from_coefficients([1.0, 2.0]);
    // Denominator: D_c(s) = 0.0 + 1.0s (degree 1)
    let d_c = Polynomial::from_coefficients([0.0, 1.0]);

    // 3. Compute Open-loop Transfer Function: G_ol(s) = C(s) * G(s) = N_ol(s) / D_ol(s)
    // N_ol(s) = N_c(s) * N_g(s) -> (1.0 + 2.0s) * 3.0 = 3.0 + 6.0s
    // D_ol(s) = D_c(s) * D_g(s) -> s * (1.0 + 0.5s) = 0.0 + 1.0s + 0.5s^2
    // Polynomial multiplication returns a new polynomial of size OUT = N + M - 1.
    let n_ol: Polynomial<f64, 2> = n_c.mul_poly(&n_g);
    let d_ol: Polynomial<f64, 3> = d_c.mul_poly(&d_g);

    println!("\nOpen-loop System Polynomials:");
    println!("  N_ol(s) = {:?}", n_ol.coefficients()); // Expected: [3.0, 6.0]
    println!("  D_ol(s) = {:?}", d_ol.coefficients()); // Expected: [0.0, 1.0, 0.5]

    // 4. Compute Closed-loop Characteristic Denominator: D_cl(s) = D_ol(s) + N_ol(s)
    // To add polynomials in control-rs, they must be of the same static size.
    // We pad N_ol(s) with a trailing zero to make it size 3: 3.0 + 6.0s + 0.0s^2
    let n_ol_padded = Polynomial::from_coefficients([3.0, 6.0, 0.0]);
    let d_cl: Polynomial<f64, 3> = d_ol + n_ol_padded;

    println!("\nClosed-loop Characteristic Denominator D_cl(s) (D_ol + N_ol):");
    println!("  D_cl(s) = {:?}", d_cl.coefficients()); // Expected: [3.0, 7.0, 0.5]
    println!("  D_cl(s) degree: {:?}", d_cl.degree()); // Expected: Some(2)

    // 5. Evaluate steady-state response at s = 0 (DC Gain)
    // Steady-state tracking gain: T(0) = N_ol(0) / D_cl(0)
    let n_ol_at_0 = n_ol.evaluate(0.0);
    let d_cl_at_0 = d_cl.evaluate(0.0);
    let dc_gain = n_ol_at_0 / d_cl_at_0;
    println!("\nSteady-State Evaluation at s = 0:");
    println!("  N_ol(0) = {}", n_ol_at_0);
    println!("  D_cl(0) = {}", d_cl_at_0);
    println!("  Closed-loop DC Gain T(0) = {}", dc_gain); // Expected: 1.0 (perfect tracking)

    // 6. Polynomial Aliases: Constant and Line
    // Handy wrappers for simple low-degree polynomials
    let c = Constant::new(5.0); // P(s) = 5
    let l = Line::new(2.0, 3.0); // P(s) = 2s + 3
    println!("\nEvaluating static polynomial aliases:");
    println!("  Constant c(10): {}", c.evaluate(10.0));
    println!("  Linear line l(4): {}", l.evaluate(4.0));

    // 7. Polynomial Derivative
    // Differentiate the characteristic polynomial to compute the rate of change
    // of the system's sensitivity dynamics.
    // D_cl(s) = 3.0 + 7.0s + 0.5s^2 -> D_cl'(s) = 7.0 + 1.0s
    let d_cl_prime = d_cl.derivative();
    println!("\nDerivative of Characteristic Polynomial:");
    println!("  D_cl'(s) coefficients: {:?}", d_cl_prime.coefficients());

    // 8. Polynomial Long Division (Proper fraction decomposition)
    // Suppose we have an improper transfer function H(s) = (3s^2 + 5s + 6) / (s + 2).
    // In control design, we decompose this into a direct transmission (quotient)
    // and a strictly proper part (remainder / denominator).
    // H(s) = (3s - 1) + 8 / (s + 2)
    let dividend = Polynomial::from_coefficients([6.0, 5.0, 3.0]); // 3s^2 + 5s + 6
    let divisor = Polynomial::from_coefficients([2.0, 1.0]); // s + 2
    #[allow(clippy::type_complexity)]
    let (q, r): (Polynomial<f64, 2>, Polynomial<f64, 1>) =
        dividend.div_rem(&divisor);

    println!("\nImproper Transfer Function Division:");
    println!("  Dividend H_num(s): 3s^2 + 5s + 6");
    println!("  Divisor H_den(s):  s + 2");
    println!(
        "  Quotient (Direct feedthrough terms):   {:?}",
        q.coefficients()
    ); // Expected: [-1.0, 3.0] -> 3s - 1
    println!(
        "  Remainder (Strictly proper numerator): {:?}",
        r.coefficients()
    ); // Expected: [8.0] -> 8
}
