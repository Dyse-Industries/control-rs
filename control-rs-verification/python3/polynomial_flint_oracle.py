#!/usr/bin/env python3
"""Polynomial reference oracle generating target/verification/polynomial.flint.h5 via python-flint (256-bit)."""

import flint

from h5_writer import get_results_dir, write_h5

flint.ctx.prec = 256


def generate_datasets():
    # P(x) = (x - 2)(x - 3)(x - 5), ascending coefficients.
    p_real = flint.arb_poly([-30, 31, -10, 1])
    p_complex = flint.acb_poly([-30, 31, -10, 1])
    real_val = p_real(flint.arb("2.5"))
    c_val = p_complex(flint.acb(1, 2))

    return {
        "tutorial/p_real": [float(real_val.mid())],
        "tutorial/p_c_re": [float(c_val.real.mid())],
        "tutorial/p_c_im": [float(c_val.imag.mid())],
    }


def main():
    write_h5(get_results_dir() / "polynomial.flint.h5", generate_datasets())


if __name__ == "__main__":
    main()
