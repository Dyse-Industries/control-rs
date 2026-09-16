# control-rs-validation

Cross-language oracle validation for `control-rs`. Each suite computes an
ill-conditioned kernel in Rust, writes it to `results/<suite>.rust.h5`, and
leaves the comparison against a NumPy/SciPy (or ngspice, harold, flint,
TFLite) peer to `control-rs-ci`.

This crate answers one question: **does `control-rs` accumulate floating-point
error the same way the reference implementations do, on problems where the
error is large enough to see?** Agreement on well-conditioned inputs proves
nothing, so nothing here is well-conditioned.

Three things live elsewhere on purpose:

| Concern | Where | Why |
|:--------|:------|:----|
| Showing the API | [`examples/`](../examples/) | A demo should be readable, not adversarial |
| Measuring latency | [`benches/`](../benches/) | Criterion samples and reports; a validator must be deterministic |
| Library unit tests | `src/**/tests` | They test one function, not two implementations |

---

## Layout

```
src/<suite>.rs          payload builder and container writer
src/bin/<suite>.rs      emitter the validate runner invokes
src/dc_motor/           plant model, peripherals, controllers, simulation
src/buck_converter/     averaged circuit, analysis, simulation
tests/                  Rust-side invariants (no Python, no HDF5)
python3/<suite>_oracle.py   peer implementation
python3/plot_*.py       diagnostic plots (not a gate)
tolerances/*.toml       write-time acceptance bounds
results/                generated containers and plots (gitignored)
```

---

## Running

From the repository root, one suite at a time:

```bash
cargo run -p control-rs-ci --bin compare -- --name matrix --out-dir target/ci
```

Or the Rust emitter alone, which writes the container and nothing else:

```bash
cargo run -p control-rs-validation --bin matrix
```

The Rust-side invariants need no Python:

```bash
cargo test -p control-rs-validation
```

Python dependencies: `pip install -r python3/requirements.txt`. The
`buck-converter` suite additionally needs `ngspice` on `PATH`. Optional peers
(`harold`, `python-flint`) are skipped when not installed; `tensorflow` and
`onnxruntime` are required by the tensor suite.

---

## Suites

| Suite | Kernels | True oracle |
|:------|:--------|:------------|
| `matrix` | Hilbert solve at $\kappa \approx 10^{13}$, monomial Vandermonde on equispaced nodes, explicit inverse residual, Cholesky across a $10^{10}$ eigenvalue spread, QR orthogonality loss | SciPy |
| `polynomial` | Wilkinson $W(x) = \prod_{k=1}^{20}(x-k)$ at its own roots, Horner on a 16-fold root, division by a near-exact factor, convolution across $10^{16}$ of scale | NumPy (flint peer) |
| `state_space` | 2000-step trajectory accumulation, $5 \times 10^{3}$ stiffness through ZOH, similarity transform at $\kappa(T) \approx 10^{8}$, rank-deficient controllability and observability | SciPy (harold peer) |
| `transfer_function` | $\zeta = 0.01$ notch discretized near Nyquist, Nyquist sweep with a free integrator, `f32` cascade vs direct form, quadruple poles $0.01$ apart | SciPy (harold peer) |
| `tensor` | Off-grid interpolation manifold, 16x16 contraction drift, `Quantized<i8, 7>` saturation and rounding | SciPy / ONNX / TFLite |
| `dc-motor` | Armature position servo with actuator delay, noise, and a 16-bit encoder, under a lead compensator and a discrete PID | SciPy |
| `buck-converter` | Averaged CCM plant, margins, compensator realization, and a switching simulation | SciPy / ngspice |

---

## Bounds

`tolerances/*.toml` declares one row per gated dataset. The bound is set by
the conditioning of the kernel, not by preference: a forward error at
$\kappa \approx 10^{13}$ cannot be held to $10^{-12}$, while the matching
backward residual can. Where those two rows disagree by ten orders of
magnitude, that gap is the measurement.

A dataset is gated only when it is a stable basis for comparison. Root lists
(implementation-defined order and conjugate sign) and chaotic quantities such
as the stalling point of Newton's method on a multiple root are emitted as
context and carry no bound.

Each suite's gate set is declared twice on purpose: `GATED_PATHS` in the Rust
module and the key map in the oracle. `tests/gated_schema.rs` fails if a gated
path is missing from its payload; `numerical-models-design.md` §6.3 must
declare exactly the key set in `tolerances/numerical_models.toml`.
