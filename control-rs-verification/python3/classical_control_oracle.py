#!/usr/bin/env python3
"""Classical control reference oracle generating target/verification/classical_control.scipy.h5 via SciPy.

Cases mirror control-rs-verification/src/classical_control.rs:
direct-form and cascade outputs (lfilter, sosfilt), factored-cascade
responses against the source (freqz), margins by root bracketing on L(jw)
(brentq) and fixed-point cascades against float64 sosfilt of the same
quantized coefficients.
"""

import numpy as np
from scipy import optimize, signal

from h5_writer import get_results_dir, write_h5

SAMPLES = 256
FREQS = 64


def excitation(scale):
    t = np.arange(SAMPLES, dtype=np.float64)
    return scale * (0.5 + 0.3 * np.sin(0.2 * t) + 0.2 * np.cos(1.3 * t))


def poly(real, pairs):
    """Ascending coefficients of prod (z - r) over real roots and conjugate pairs."""
    c = np.array([1.0])
    for r in real:
        c = np.convolve(c, [-r, 1.0])
    for rho, th in pairs:
        c = np.convolve(c, [rho * rho, -2.0 * rho * np.cos(th), 1.0])
    return c


def biquad(rho, theta, phi, g):
    return [g, -2.0 * g * np.cos(phi), g, 1.0, -2.0 * rho * np.cos(theta), rho * rho]


def cascade_sos():
    return np.array(
        [biquad(0.9, 0.4, 1.5, 0.2), biquad(0.8, 1.1, 2.2, 0.2), biquad(0.6, 2.0, 2.8, 0.2)]
    )


def df2t_output(num, den):
    """lfilter of an ascending-z (num, den) pair padded to equal length."""
    n = max(len(num), len(den))
    num = np.pad(num, (0, n - len(num)))
    den = np.pad(den, (0, n - len(den)))
    return signal.lfilter(num[::-1], den[::-1], excitation(1.0))


def sections_response(num, den):
    """Source response at the cascade frequencies, normalized by its peak."""
    n = max(len(num), len(den))
    num = np.pad(num, (0, n - len(num)))
    den = np.pad(den, (0, n - len(den)))
    w = 0.049 * np.arange(FREQS) + 0.01
    _, h = signal.freqz(num[::-1], den[::-1], worN=w)
    h = h / np.max(np.abs(h))
    return h.real, h.imag


def margins(num, den, m):
    """Crossovers of L(jw) bracketed on 0.01 + 0.05 k and refined by brentq."""
    lw = lambda w: np.polyval(num[::-1], 1j * w) / np.polyval(den[::-1], 1j * w)
    w = 0.05 * np.arange(m) + 0.01
    out = {}
    mag = np.abs(lw(w)) - 1.0
    for k in range(m - 1):
        if np.sign(mag[k]) != np.sign(mag[k + 1]):
            wg = optimize.brentq(lambda x: abs(lw(x)) - 1.0, w[k], w[k + 1], xtol=1e-15, rtol=1e-15)
            out["w_gc"] = wg
            out["phase_margin"] = np.degrees(np.angle(-lw(wg)))
            break
    l = lw(w)
    for k in range(m - 1):
        if np.sign(l[k].imag) != np.sign(l[k + 1].imag) and l[k].real < 0 and l[k + 1].real < 0:
            wp = optimize.brentq(lambda x: lw(x).imag, w[k], w[k + 1], xtol=1e-15, rtol=1e-15)
            out["w_pc"] = wp
            out["gain_margin"] = 1.0 / abs(lw(wp))
            break
    return out


def fixed_output(shift):
    """float64 sosfilt of the cascade quantized to `shift` fractional bits."""
    scale = 2.0**shift
    sos = np.round(cascade_sos() * scale) / scale
    u = np.round(excitation(0.25) * scale) / scale
    return signal.sosfilt(sos, u)


def generate_datasets():
    d = {}
    d["df2t_order1/output"] = df2t_output(poly([-0.5], []), 2.0 * poly([0.8], []))
    d["df2t_order2/output"] = df2t_output(poly([], [(1.0, 1.0)]), poly([], [(0.95, 0.3)]))
    d["df2t_order3/output"] = df2t_output(poly([-1.0, 0.5, 0.1], []), poly([0.7, 0.2, -0.5], []))
    d["df2t_order4/output"] = df2t_output(
        poly([-1.0, 0.2], [(1.0, 2.0)]), poly([0.7, -0.4], [(0.9, 0.6)])
    )
    u = excitation(1.0)
    d["cascade_df1/output"] = signal.sosfilt(cascade_sos(), u)
    d["cascade_df2t/output"] = signal.sosfilt(cascade_sos(), u)

    p = [(0.95, 0.3), (0.8, 1.0), (0.6, 2.0), (0.5, 2.7)]
    z = [(1.0, 0.9), (1.0, 1.7), (0.9, 2.5), (1.2, 0.2)]
    cases = [
        (poly([-1.0], []), poly([0.7], [])),
        (poly([], z[:1]), poly([], p[:1])),
        (poly([], z[:1]), poly([0.7], p[:1])),
        (poly([0.0, -1.0], z[:1]), poly([], p[:2])),
        (poly([-1.0], z[:2]), poly([-0.3], p[:2])),
        (poly([0.5], z[:2]), poly([], p[:3])),
        (poly([-1.0, 0.4], z[:2]), poly([0.2], p[:3])),
        (poly([], z[:4]), poly([], p[:4])),
    ]
    for order, (num, den) in enumerate(cases, start=1):
        re, im = sections_response(num, den)
        d[f"sections_order{order}/re"] = re
        d[f"sections_order{order}/im"] = im

    for case, num, den, m in [
        ("margins_third_order", np.array([4.0]), np.array([1.0, 3.0, 3.0, 1.0]), 61),
        ("margins_integrator", np.array([10.0]), np.array([0.0, 1.0, 1.0]), 201),
    ]:
        for key, val in margins(num, den, m).items():
            d[f"{case}/{key}"] = [val]

    d["df1_q13/fixed"] = fixed_output(13)
    d["df1_q29/fixed"] = fixed_output(29)
    return d


def main():
    out_file = get_results_dir() / "classical_control.scipy.h5"
    write_h5(out_file, generate_datasets())


if __name__ == "__main__":
    main()
