#!/usr/bin/env python3
"""Polynomial Numerical Prototype Oracle.

Calculates reference polynomial evaluation (real and complex), differentiation,
integration, Euclidean division, and Frobenius companion matrix formulation.
Implemented in pure Python (standard library).
"""


def poly_eval_real(coeffs, x):
    """Horner's method for real evaluation. coeffs: [c0, c1, ..., cN]."""
    res = 0.0
    for c in reversed(coeffs):
        res = res * x + c
    return res


def poly_eval_complex(coeffs, z_re, z_im):
    """Horner's method for complex evaluation z = z_re + j * z_im."""
    res_re, res_im = 0.0, 0.0
    for c in reversed(coeffs):
        # res = res * z + c
        # (res_re + j*res_im) * (z_re + j*z_im) + c
        new_re = (res_re * z_re - res_im * z_im) + c
        new_im = res_re * z_im + res_im * z_re
        res_re, res_im = new_re, new_im
    return res_re, res_im


def poly_deriv(coeffs):
    """Derivative of polynomial. coeffs: [c0, c1, ..., cN]."""
    if len(coeffs) <= 1:
        return [0.0]
    return [float(i * coeffs[i]) for i in range(1, len(coeffs))]


def poly_integ(coeffs, c0=0.0):
    """Indefinite integral of polynomial with integration constant c0."""
    res = [float(c0)]
    for i, c in enumerate(coeffs):
        res.append(float(c) / float(i + 1))
    return res


def poly_mul(p1, p2):
    """Multiply two polynomials p1 and p2."""
    deg1 = len(p1) - 1
    deg2 = len(p2) - 1
    res = [0.0] * (deg1 + deg2 + 1)
    for i, a in enumerate(p1):
        for j, b in enumerate(p2):
            res[i + j] += a * b
    return res


def poly_div_rem(num, den):
    """Euclidean polynomial division: num / den -> (quotient, remainder)."""
    # Remove trailing zeros to find true degrees
    def deg(p):
        d = len(p) - 1
        while d > 0 and abs(p[d]) < 1e-14:
            d -= 1
        return d

    deg_num = deg(num)
    deg_den = deg(den)
    if abs(den[deg_den]) < 1e-14:
        raise ZeroDivisionError("Division by zero polynomial")
    if deg_num < deg_den:
        return [0.0], list(num)

    rem = [float(c) for c in num[: deg_num + 1]]
    deg_quot = deg_num - deg_den
    quot = [0.0] * (deg_quot + 1)

    for i in range(deg_quot, -1, -1):
        lead_rem_idx = i + deg_den
        scale = rem[lead_rem_idx] / den[deg_den]
        quot[i] = scale
        for j in range(deg_den + 1):
            rem[i + j] -= scale * den[j]

    # Remainder size = deg_den
    final_rem = rem[:deg_den] if deg_den > 0 else [0.0]
    return quot, final_rem


def main():
    print("=== Polynomial Numerical Prototype Oracle ===")

    # 1. Real Polynomial: p(x) = 2 - 3x + 4x^2 + x^3
    coeffs = [2.0, -3.0, 4.0, 1.0]
    print("\n--- Polynomial Evaluation & Calculus ---")
    print(f"Coefficients (ascending): {coeffs}")

    x_test = 2.5
    val_real = poly_eval_real(coeffs, x_test)
    print(f"p({x_test}) = {val_real:.10f}")

    s_re, s_im = 1.0, 2.0
    val_c_re, val_c_im = poly_eval_complex(coeffs, s_re, s_im)
    print(f"p({s_re} + {s_im}j) = {val_c_re:.10f} + {val_c_im:.10f}j")

    dp = poly_deriv(coeffs)
    print(f"p'(x) coefficients: {dp}")
    print(f"p'({x_test}) = {poly_eval_real(dp, x_test):.10f}")

    integ = poly_integ(coeffs, c0=5.0)
    print(f"int p(x) dx (c0=5) coefficients: {integ}")
    print(f"int_0^{x_test} p(t) dt + 5.0 = {poly_eval_real(integ, x_test):.10f}")

    # 2. Polynomial Multiplication & Division
    p1 = [1.0, 2.0]  # (1 + 2x)
    p2 = [3.0, 4.0]  # (3 + 4x)
    prod = poly_mul(p1, p2)
    print("\n--- Polynomial Multiplication & Division ---")
    print(f"(1 + 2x) * (3 + 4x) = {prod}")

    quot, rem = poly_div_rem(prod, p1)
    print(f"Quotient of ({prod}) / ({p1}): {quot}")
    print(f"Remainder: {rem}")

    # 3. Monic Polynomial & Companion Matrix
    # Monic p(x) = -6 - 5x + x^2  (roots: 6 and -1)
    p_monic = [-6.0, -5.0, 1.0]
    print("\n--- Monic Companion Matrix ---")
    print(f"Monic p(x) = {p_monic}")

    # C = [[0, 6.0], [1, 5.0]]
    C = [[0.0, -p_monic[0]], [1.0, -p_monic[1]]]
    print(f"Companion Matrix C:")
    for row in C:
        print("  [" + ", ".join(f"{v:8.4f}" for v in row) + "]")


if __name__ == "__main__":
    main()
