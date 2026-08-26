#!/usr/bin/env python3
"""Transfer Function Numerical Prototype Oracle.

Calculates 2nd-order continuous Butterworth low-pass filter frequency response,
Bode magnitude/phase points, series cascade algebra, and controllable canonical form realization.
Implemented in pure Python (standard library).
"""

import math


def complex_div(num_re, num_im, den_re, den_im):
    den_mag2 = den_re * den_re + den_im * den_im
    if den_mag2 == 0.0:
        raise ZeroDivisionError("Division by zero in complex division")
    re = (num_re * den_re + num_im * den_im) / den_mag2
    im = (num_im * den_re - num_re * den_im) / den_mag2
    return re, im


def poly_eval_complex(coeffs, z_re, z_im):
    """Evaluate polynomial with ascending coefficients at complex z."""
    res_re, res_im = 0.0, 0.0
    for c in reversed(coeffs):
        new_re = (res_re * z_re - res_im * z_im) + c
        new_im = res_re * z_im + res_im * z_re
        res_re, res_im = new_re, new_im
    return res_re, res_im


def poly_mul(p1, p2):
    res = [0.0] * (len(p1) + len(p2) - 1)
    for i, a in enumerate(p1):
        for j, b in enumerate(p2):
            res[i + j] += a * b
    return res


def main():
    print("=== Transfer Function Numerical Prototype Oracle ===")

    # 1. Continuous 2nd-Order Low-Pass Butterworth Filter
    # H(s) = \omega_c^2 / (s^2 + \sqrt{2}\omega_c s + \omega_c^2)
    # \omega_c = 10.0 rad/s
    # num(s) = 100.0
    # den(s) = 100.0 + 14.14213562373095 s + 1.0 s^2 (ascending degree)
    omega_c = 10.0
    w_c2 = omega_c**2
    sqrt2_wc = math.sqrt(2.0) * omega_c

    num_coeffs = [w_c2]
    den_coeffs = [w_c2, sqrt2_wc, 1.0]

    print(f"\n--- Butterworth Filter (omega_c = {omega_c} rad/s) ---")
    print(f"Numerator (ascending): {num_coeffs}")
    print(f"Denominator (ascending): {den_coeffs}")

    test_freqs = [0.1, 1.0, 10.0, 100.0]
    print("\n--- Frequency Evaluation ---")
    print(
        f"{'omega (rad/s)':<16}{'Real':<16}{'Imag':<16}{'Mag (abs)':<16}{'Mag (dB)':<16}{'Phase (deg)':<16}"
    )

    for w in test_freqs:
        # s = 0 + j * w
        num_re, num_im = poly_eval_complex(num_coeffs, 0.0, w)
        den_re, den_im = poly_eval_complex(den_coeffs, 0.0, w)
        h_re, h_im = complex_div(num_re, num_im, den_re, den_im)

        mag_abs = math.sqrt(h_re * h_re + h_im * h_im)
        mag_db = 20.0 * math.log10(mag_abs) if mag_abs > 0 else -float("inf")
        phase_deg = math.degrees(math.atan2(h_im, h_re))

        print(
            f"{w:<16.2f}{h_re:<16.8f}{h_im:<16.8f}{mag_abs:<16.8f}{mag_db:<16.8f}{phase_deg:<16.8f}"
        )

    # 2. Series Cascade Interconnection
    # H1(s) = 2 / (2 + s), H2(s) = 5 / (5 + s)
    # H_series(s) = 10 / (10 + 7s + s^2)
    h1_num, h1_den = [2.0], [2.0, 1.0]
    h2_num, h2_den = [5.0], [5.0, 1.0]

    num_ser = poly_mul(h1_num, h2_num)
    den_ser = poly_mul(h1_den, h2_den)

    print("\n--- Series Cascade H1 * H2 ---")
    print(f"H1(s): {h1_num} / {h1_den}")
    print(f"H2(s): {h2_num} / {h2_den}")
    print(f"H_series: {num_ser} / {den_ser}")

    # 3. Controllable Canonical State-Space Realization
    # For H(s) = (2 + 3s) / (4 + 5s + s^2)
    # A = [[0, -4], [1, -5]]
    # B = [[0], [1]]
    # C = [[2, 3]]
    # D = [[0]]
    print("\n--- Controllable Canonical Realization ---")
    print(f"H(s) = (2 + 3s) / (4 + 5s + s^2)")
    A_canon = [[0.0, -4.0], [1.0, -5.0]]
    B_canon = [[0.0], [1.0]]
    C_canon = [[2.0, 3.0]]
    D_canon = [[0.0]]

    print("A:")
    for row in A_canon:
        print("  " + str(row))
    print("B:")
    for row in B_canon:
        print("  " + str(row))
    print("C:")
    for row in C_canon:
        print("  " + str(row))
    print("D:")
    for row in D_canon:
        print("  " + str(row))


if __name__ == "__main__":
    main()
