# System Identification & Model Estimation (Design Document)

![Status: Draft](https://img.shields.io/badge/status-draft-orange)
![Date Badge](https://img.shields.io/badge/Date-September_15,_2026-blue)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

`sysid` estimates discrete transfer-function and state-space models from
measured input-output or frequency-response data. Solver names (ARX, IV,
Levy, SK, VF, ERA, MOESP, N4SID, Matrix Pencil) belong in §4.

Primary usage scenarios:

- Recover a linear difference equation from a discrete input-output record.
- Fit a rational model to measured frequency-response points.
- Recover a minimal state-space model from impulse or input-output records.
- Extract modal frequencies and damping from a ring-down transient.

---

### 2. Requirements

#### 2.1 Functional Requirements

- **FR-1 — Difference-equation coefficients from time series**: The module
  estimates discrete autoregressive-exogenous coefficients from $(u[k], y[k])$.
  When output noise is colored, an instrumental-variable refinement is
  available; unconstrained prediction-error minimization is not (C-4).
- **FR-2 — Rational fit to frequency data**: The module estimates numerator
  and denominator coefficients from complex frequency-response samples.
- **FR-3 — Pole-relocated rational fit**: The module fits a rational model
  whose poles are relocated to match frequency data. Initial-pole heuristics
  are a §4 choice.
- **FR-4 — Minimal realization from Markov parameters**: The module extracts
  a minimal discrete state-space model from impulse-response Markov
  parameters.
- **FR-5 — State-space from input-output records**: The module identifies
  multivariable state-space matrices from paired input-output sequences.
- **FR-6 — Modes from ring-down**: The module extracts modal frequencies and
  damping from a transient time signal.
- **FR-7 — Caller-sized Hankel assembly**: Block Hankel matrices with
  caller-chosen block counts are written into caller-allocated storage. The
  module does not grow that storage.

#### 2.2 Non-Functional Requirements

- **NFR-1 — Bounded embedded footprint**: Identification runs in a fixed
  static footprint sized for $K \le 200$ samples and order $n_x \le 6$, with
  no heap allocation.
- **NFR-2 — Optional stable poles**: Vector-fitting and subspace paths may
  project or flip poles so continuous poles lie in the open left-half plane
  and discrete poles lie inside the unit circle. The option is off unless
  requested.

#### 2.3 Constraints

- **C-1 — Inherited toolbox bounds**: Conforms to `controls-tools-design.md`
  C-1..C-4. Former C-2 and C-3 restated those bounds and are withdrawn.
  Identified models are native crate types (parent C-1).
- **C-4 — Offline-only unconstrained prediction-error minimization**:
  Non-convex PEM stays host-side. It is not a target runtime dependency.

---

### 6. Verification & Validation

#### 6.1 Approach

No `sysid` source module
exists; behavioral FRs are unverified until implementations and locators
exist.

- Noise-free ARX recovery matches the generating polynomials exactly.
- Frequency-domain fits recover known poles and zeros within the stated
  relative bound.
- A minimal realization reproduces the generating Markov parameters within
  the stated Frobenius bound.
- Subspace models agree with an independent identification tool on a MIMO
  plant.
- Hankel assembly writes only into caller storage.

| Method | Mechanism |
|:-------|:----------|
| Requirements-based test | Manufactured time-series, frequency, Markov, and ring-down cases |
| Back-to-back comparison | Independent Python / Julia subspace reference |
| Static analysis | `cargo clippy-ci`; source inspection for heap use |
| Inspection | This document against `controls-tools-design.md` C-1..C-4 |
| On-target execution | ETS Hankel and least-squares step |
| Coverage measurement | `cargo coverage` |

Target: 90% statement coverage of the `sysid` module once it exists, measured
with `cargo coverage`. Excluded: `Debug` / `Display`. There is no executable
surface to cover at this revision.

No identification plant exists for this module. Fitness for purpose is judged
only after a host example estimates a model that a toolbox synthesis path
then consumes.

#### 6.2 Acceptance

| Claim | Oracle | Measure | Bound |
|:------|:-------|:--------|:------|
| Noise-free ARX coefficients | Generating discrete polynomials | Exact coefficient equality | Exact |
| Frequency-domain poles and zeros | Known low-pass and non-minimum-phase plants | Relative pole/zero error | $\le 10^{-4}$ |
| Vector-fitting pole relocation | Known resonant / power-system plants | Relative pole error after convergence | $\le 10^{-4}$ |
| ERA Markov reconstruction | Generating Markov sequence | Frobenius residual | $\le 10^{-6}$ |
| Subspace $(A,B,C,D)$ | Independent Python / Julia subspace tool on a MIMO plant | Relative $\ell_{2}$ of simulated outputs | $\le 10^{-4}$ |
| Matrix-pencil modes | Multi-tone exponentially damped synthetics | Absolute frequency and damping error | $\le 10^{-4}$ relative at high SNR |

#### 6.3 Limits

- **FR-1..FR-7, NFR-2**: no `sysid` source module and no `test:` or `oracle:`
  locator exists.
- **On-target Hankel / least-squares latency**: no ETS suite is linked into
  firmware.
- **Colored-noise IV statistical efficiency**: FR-1 offers IV as a
  refinement; consistency proofs are out of scope.
- **Low-SNR matrix-pencil bias**: 6.2 bounds high-SNR synthetics only.
- **Withdrawn C-2, C-3**: restated inherited toolbox bounds; covered by C-1.

---

## References

[1] L. Ljung, "Prediction Error Estimation Methods," Department of Electrical Engineering, Link{\"o}ping University, Link{\"o}ping, Sweden, Rep. no. LiTH-ISY-R-2365, 2001. [Online]. Available: https://www.rt.isy.liu.se/research/reports/2001/2365.pdf.

[2] M. Gilson, "What Has Instrumental Variable Method to Offer for System Identification?," in *Proc. 8th IFAC Int. Conf. Mathematical Modelling (MATHMOD 2015)*, Vienna, Austria, 2015, pp. 176--181, doi: 10.1016/j.ifacol.2015.05.176.

[3] E. C. Levy, "Complex-Curve Fitting," *IRE Transactions on Automatic Control*, vol. 4, no. 1, pp. 37--43, 1959, doi: 10.1109/TAC.1959.6429401.

[4] C. K. Sanathanan and J. Koerner, "Transfer Function Synthesis as a Ratio of Two Complex Polynomials," *IEEE Transactions on Automatic Control*, vol. 8, no. 1, pp. 56--58, 1963, doi: 10.1109/TAC.1963.1105517.

[5] B. Gustavsen and A. Semlyen, "Application of Vector Fitting to State Equation Representation of Transformers for Simulation of Electromagnetic Transients," *IEEE Transactions on Power Delivery*, vol. 13, no. 3, pp. 834--842, 1998, doi: 10.1109/61.686981.

[6] J.-N. Juang and R. S. Pappa, "An Eigensystem Realization Algorithm for Modal Parameter Identification and Model Reduction," *Journal of Guidance, Control, and Dynamics*, vol. 8, no. 5, pp. 620--627, 1985, doi: 10.2514/3.20031.

[7] M. Verhaegen and P. Dewilde, "Subspace Model Identification Part 1. The Output-Error State-Space Model Identification Class of Algorithms," *International Journal of Control*, vol. 56, no. 5, pp. 1187--1210, 1992, doi: 10.1080/00207179208934363.

[8] P. {Van Overschee} and B. {De Moor}, "N4SID: Subspace Algorithms for the Identification of Combined Deterministic-Stochastic Systems," *Automatica*, vol. 30, no. 1, pp. 75--93, 1994, doi: 10.1016/0005-1098(94)90230-5.

[9] Y. Hua and T. K. Sarkar, "Matrix Pencil Method for Estimating Parameters of Exponentially Damped/Undamped Sinusoids in Noise," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 38, no. 5, pp. 814--824, 1990, doi: 10.1109/29.56027.

---

### 10. Revision History

| Revision | Date              | Author          | Description                                                    |
|:---------|:------------------|:----------------|:---------------------------------------------------------------|
| 1.0      | September 3, 2026 | @MitchellDScott | Initial requirements and verification outline for sysid.       |
| 1.1      | September 15, 2026 | @MitchellDScott | Need-named FRs; inherited toolbox pointer; full §6 with missing module listed in 6.7. |
