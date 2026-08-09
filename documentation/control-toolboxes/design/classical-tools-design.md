# Classical Tools (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_8,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

`src/classical_tools/mod.rs` is currently an empty scaffold. Its module doc
commits the module to "root locus, Routh-Hurwitz stability criterion, Bode
plots, Nyquist plots and compensators." This document defines the
architecture that fulfills that commitment on top of `control-rs`'s existing
`Polynomial`, `Matrix`, and `TransferFunction` primitives.

Classical frequency- and root-locus-based design remains the primary design
method for a large share of real control loops. Industrial process control is
dominated by PID: more than 95 percent of industrial control loops are of PID
type, and the majority of those are pure PI (Åström and Hägglund, 2006).
Root-locus and Bode/Nyquist loop-shaping carry the same weight in other
domains — flight control design documented by MathWorks pairs Control System
Toolbox with Simulink Control Design and Aerospace Blockset for longitudinal
motion control (MathWorks, 2026b), and switching-mode power supply
compensator design is a Bode-plot-driven discipline in its own right: an
undercompensated converter shows up as audible noise, output oscillation, and
FET overheating (Analog Devices, 2026), and vendor guidance frames the Bode
plot directly as the tool for assessing whether a design meets its dynamic
control requirements (Dostal, 2021). `classical_tools` targets this same
class of design problem, but as a `no_std`, `no_alloc` numeric core rather
than an interactive workstation application — the distinction that drives
every architectural decision below.

---

### 2. Requirements

#### 2.1. Functional Requirements

- **Root Locus**: Compute the closed-loop pole locations of a `TransferFunction`
  as a function of a swept scalar gain, matching the analysis role MATLAB's
  `rlocus` and Julia's `rlocus` occupy in their respective toolboxes
  (MathWorks, 2026a; JuliaControl, 2026).
- **Routh-Hurwitz Stability Criterion**: Construct the Routh array from a
  characteristic polynomial's coefficients and report the number of
  right-half-plane roots from first-column sign changes, without requiring
  root-finding.
- **Frequency-Domain Analysis**: Generate Bode (magnitude/phase), Nyquist
  (real/imaginary), and Nichols (open-loop gain/phase) response data over a
  caller-supplied frequency sweep. python-control groups these three plus
  root locus and stability margins under a single "Control analysis" feature
  set, separate from its "Control design" functions (python-control, 2026a) —
  `classical_tools` follows the same functional split (§4).
- **Stability Margins**: Compute gain margin, phase margin, and their
  crossover frequencies from frequency-response data, matching the scope of
  `margin`/`sisomargin` in python-control and Julia's `ControlSystems.jl`
  (python-control, 2026a; JuliaControl, 2026).
- **Compensator Design**: Provide PID (parallel and standard forms) and
  lead/lag network constructors that return `TransferFunction` values,
  matching the `pid`/`pidstd`/`leadlink`/`laglink` primitives found in Octave
  and Julia's control toolboxes (Octave-Forge, 2026; JuliaControl, 2026).

#### 2.2. Non-Functional Requirements

- **No Plotting Dependency**: The module must not depend on a plotting or
  GUI library. Every surveyed toolbox assumes an interactive workstation
  host — MATLAB's Control System Designer is an interactive GUI app for
  graphically tuning root locus, Bode, and Nichols views (MathWorks, 2026a);
  python-control requires `matplotlib` directly (python-control, 2026a); the
  closest existing Rust prior art, `control_systems_torbox`, depends on the
  `egui`/`egui_plot` GUI stack (TorBorve, 2026). None of this is compatible
  with `no_std`, so `classical_tools` must expose numeric results only (§4).
- **No Dynamic Allocation**: Sweep outputs (root locus points, frequency
  response samples) are written into caller-provided, statically sized
  buffers, consistent with the crate-wide no-heap rule.
- **Deterministic Core, Bounded Search Elsewhere**: Routh array construction
  and frequency-response evaluation execute in a fixed operation count for a
  given polynomial/transfer-function size. Root locus, which depends on
  per-gain root-finding, inherits whatever convergence behavior the
  underlying companion-matrix eigenvalue solver provides; no additional
  iterative search is introduced at the `classical_tools` layer.

#### 2.3. Constraints

- **Built on Existing Primitives**: `classical_tools` builds on
  `Polynomial` and `TransferFunction`; it does not introduce a parallel
  polynomial or rational-function representation.
- **Capacity Bounds**: Maximum representable system order is bounded by
  `Polynomial`'s 128-element capacity and `Matrix`'s 32×32 companion-matrix
  limit (README.md capacity table), since root locus and margin computation
  route through both.
- **No Fortran/GPL Wrapper Dependencies**: Reference toolboxes lean on a
  SLICOT Fortran wrapper for advanced routines; python-control's own
  `Slycot` wrapper remains GPLv2-licensed pending contributor sign-off to
  relicense under the BSD-3-Clause terms SLICOT itself adopted in December
  2020 (python-control, 2026b). `classical_tools` does not adopt an
  equivalent wrapper — see §5.3.
- **No Interactive Tuning Surface**: An `sisotool`/`pidtune`-equivalent
  interactive design loop is out of scope for this module (§5.4).

---

### 3. Technical Overview

`classical_tools` is a design-time numeric library, not a runtime control
component. It sits alongside `TransferFunction` and consumes it directly:
inputs are `TransferFunction`/`Polynomial` values, outputs are either
`TransferFunction` values (compensator synthesis) or fixed-capacity point
buffers (root locus, frequency response, margins). The module owns no
plotting, rendering, or terminal-UI logic; that is left entirely to
consumers, matching the numeric/graphical split every surveyed toolbox
implements internally but does not expose as a boundary a `no_std` crate can
reuse (§4.1).

---

### 4. Architecture

#### 4.1. Numeric Core vs. Plotting Layer

Every reference toolbox internally separates a numeric computation from its
display: JuliaControl's own command listing puts `margin`, `sisomargin`, and
other analysis functions in a distinct "Analysis" category from
`bodeplot`/`nyquistplot`/`rlocus`/`nicholsplot` in "Plotting" (JuliaControl,
2026); Octave's control package documents `rlocus` itself as "Display root
locus plot of the specified SISO system," and ships a separate interactive
`rlocusx` on top of it (Octave-Forge, 2026). `classical_tools` draws the
same boundary, but makes it a hard module boundary rather than an internal
convention: every public function returns numeric data (points, coefficients,
scalars) written into caller-owned storage. No function in `classical_tools`
renders anything. This is what makes the module viable in `no_std` where the
egui/matplotlib-class dependencies the reference toolboxes rely on for their
plotting layer (TorBorve, 2026; python-control, 2026a) are not available.

#### 4.2. Root Locus

For a fixed gain $k$, the closed-loop characteristic polynomial of
$1 + k \cdot L(s) = 0$ is a `Polynomial` produced from `TransferFunction`'s
existing numerator/denominator algebra. Its roots are found via the
companion-matrix conversion and QR eigenvalue solve `Polynomial` and `Matrix`
already implement for characteristic-polynomial root-finding
(`polynomial-design.md` §4.4.1, `matrix-design.md` §4.10.2). Root locus adds
no new root-finding algorithm; it drives the existing companion-matrix path
once per swept gain value into a caller-provided output buffer sized to
(gain count × system order).

#### 4.3. Routh-Hurwitz Stability Criterion

The Routh array is built directly from a `Polynomial`'s coefficient slice
using the standard tabular recurrence — it requires no root-finding and no
companion-matrix conversion, making it the cheapest and least
dependency-coupled primitive in this module. Stability is read off the
number of sign changes in the first column. This is architecturally
independent of §4.2; nothing about it requires `Matrix`. The classic
degenerate cases (an all-zero row, a zero in the first column with a
nonzero remainder in that row) are known special cases of the tabular
method but are not resolved by evidence gathered for this design — see §8.

#### 4.4. Frequency-Domain Analysis and Margins

Bode, Nyquist, and Nichols outputs are all projections of the same
underlying computation: `TransferFunction::evaluate_complex` evaluated at
$s = j\omega$ (or $z = e^{j\omega T_s}$) over a caller-supplied frequency
buffer (`transfer-function-design.md` §5.2). `classical_tools` adds no new
evaluation primitive here — it adds the sweep driver and the
magnitude/phase/real/imaginary projections consumers need for each plot
family, plus gain- and phase-margin extraction (0 dB and -180° crossing
detection) over the resulting sample buffer. Because margin values are read
from a discrete sweep rather than solved for analytically, margin accuracy
is bounded by sweep resolution; this is flagged as an open question in §8
rather than assumed resolved by a particular interpolation scheme.

#### 4.5. Compensator Synthesis

PID and lead/lag constructors build `TransferFunction` values directly,
mirroring the primitive set Octave and Julia expose (`pid`, `pidstd`,
`leadlink`, `laglink` — Octave-Forge, 2026; JuliaControl, 2026) rather than
introducing a distinct compensator type. A synthesized compensator is
ordinary `TransferFunction` data from that point forward — it composes with
`series`/`parallel`/`feedback` (`transfer-function-design.md` §5.3) with no
special-casing in `classical_tools`.

---

### 5. Alternatives

#### 5.1. Root-Finding Strategy for Root Locus

Companion-matrix QR eigenvalue solving (chosen, §4.2) versus a dedicated
root-locus-specific iterative solver: rejected, since it would duplicate
machinery `Polynomial`/`Matrix` already provide and would introduce a second
root-finding code path with a different convergence and determinism profile
than the one already selected for characteristic-polynomial root-finding
elsewhere in the crate (`polynomial-design.md` §7 rejects
data-dependent-convergence root-finding for the same determinism reason).

#### 5.2. Coupling Numeric Output to a Plotting Layer

Reference toolboxes typically expose plotting and numeric analysis through
the same function (Octave's `rlocus` both computes and displays; MATLAB's
Control System Designer is graphical-first — MathWorks, 2026a). Coupling was
rejected outright: `classical_tools` has no plotting dependency available
under `no_std` (§2.2), so numeric output and any future visualization must
be separate concerns by construction, not by convention.

#### 5.3. Vendored Fortran/SLICOT-Class Numerical Routines

Wrapping a SLICOT-derived routine (as python-control's optional `Slycot`
dependency does) was considered for margin/root-locus-adjacent numerics and
rejected. Beyond the `no_std`/no-FFI mismatch, `Slycot` itself remains
GPLv2-licensed because relicensing requires sign-off from every past
contributor, despite upstream SLICOT moving to BSD-3-Clause in December 2020
(python-control, 2026b) — a licensing posture this crate's minimal-dependency
convention (`CLAUDE.md`) does not want to inherit even indirectly.

#### 5.4. Full Interactive Design Tool (`sisotool`-Equivalent)

Implementing an interactive, MATLAB `sisotool`/`pidtune`-style design loop
was considered and rejected for this module. No open-source toolbox surveyed
offers a `no_std`-compatible equivalent; every one found is workstation- and
plotting-bound (MathWorks, 2026a; TorBorve, 2026). An interactive tuning
surface, if built at all, belongs to a host-side companion tool consuming
`classical_tools`'s numeric output — not to this module.

---

### 6. Verification & Validation

1. **Golden-Value Regression**: Root locus point sets at fixed gains, Routh
   array stability determinations, and gain/phase margin values are checked
   against `python-control` and/or Octave control-package reference outputs
   for a shared set of test transfer functions (`python-control`'s `margin`,
   Octave's `rlocus`/`margin` — python-control, 2026a; Octave-Forge, 2026),
   per the crate's Solvers & Estimators testing standard (`CLAUDE.md`).
2. **Property-Based Testing**: The Routh-array first-column sign-change
   count must equal the number of right-half-plane roots independently
   computed via companion-matrix root-finding, checked with `proptest` over
   randomly generated polynomials, including the degenerate first-column-zero
   and all-zero-row cases flagged in §8.
3. **Cross-Validation of Compensator Synthesis**: PID and lead/lag
   constructors are checked against Octave's `pid`/`pidstd`/`leadlink`/
   `laglink` outputs for equivalent parameterizations (Octave-Forge, 2026).
4. **No HIL Requirement**: `classical_tools` performs no hardware I/O and
   has no runtime target-side component; the crate's HIL mock-harness
   requirement (`CLAUDE.md`) does not apply.

---

### 7. Performance & Resource Considerations

Root locus and frequency-response sweeps are the two functions whose cost
scales with a caller-chosen resolution rather than system order alone: each
sample point in either sweep requires one companion-matrix eigenvalue solve
(root locus) or one complex Horner evaluation pair (frequency response).
Because output buffers are caller-provided rather than internally allocated
(§2.2), sweep resolution is a cost the caller controls explicitly rather than
one the module amortizes or hides.

---

### 8. Risks & Open Questions

- **Limited Rust Prior Art**: `control_systems_torbox`'s `analysis` module
  exposes only `frequency_response` and `system_properties` submodules — no
  root locus or Routh-Hurwitz (TorBorve, 2026). `scirs2-signal` re-exports
  `root_locus`, `nichols_chart`, `nyquist_diagram`, and `stability_margins`
  directly (cool-japan, 2026), making it the closer functional analog, but
  its `Cargo.toml` carries `chrono`, `serde_json`, `num_cpus`, and other
  std-oriented dependencies with no `no_std` marker present (cool-japan,
  2026) — it is not a reusable dependency, only a naming/scope reference.
- **Routh Array Degenerate Cases Unverified**: The classic "row of zeros"
  and "first-column zero" special cases in the Routh array were not
  independently verified against either MATLAB File Exchange submission
  found during research (Shamshiri, 2009; Sagharchi, 2016); both are
  third-party submissions outside any vendor toolbox and are treated as weak,
  uncorroborated evidence rather than a settled implementation reference.
  Octave's control package has no `routh`-named function at all
  (Octave-Forge, 2026), so no vendor-toolbox implementation was available to
  cross-check either. The degenerate-case algorithm must be independently
  derived and verified during implementation, not assumed from a single
  third-party source.
- **Margin Accuracy vs. Sweep Resolution**: §4.4's crossing-detection
  approach to gain/phase margin has no interpolation or refinement strategy
  specified yet; whether a coarse caller-chosen sweep produces acceptable
  margin accuracy, or whether the API needs a documented minimum resolution
  or a follow-up refinement step, is unresolved and not addressed by the
  evidence gathered for this design.
- **No `sisotool`-Equivalent Scoping Decision**: §5.4 defers, rather than
  answers, whether a host-side interactive companion tool should exist at
  all; that remains an open product question outside this design's scope.

---

### 9. Development Plan

| Task / Feature | Description | Estimated Effort |
|:---|:---|:---|
| **Phase 1: Routh-Hurwitz** | Routh array construction over `Polynomial` coefficients, first-column sign-change stability determination, degenerate-case (zero-row, first-column-zero) handling. | 2.5 Days |
| **Phase 2: Root Locus** | Gain-sweep driver over the existing companion-matrix root-finding path; caller-provided output buffer API. | 2.0 Days |
| **Phase 3: Frequency Response & Margins** | Bode/Nyquist/Nichols sweep driver over `TransferFunction::evaluate_complex`; gain/phase margin and crossover-frequency extraction. | 2.5 Days |
| **Phase 4: Compensator Synthesis** | PID (parallel and standard form) and lead/lag `TransferFunction` constructors. | 1.5 Days |
| **Phase 5: Verification** | Golden-value regression vs. python-control/Octave, `proptest` Routh/root-count invariants, compensator cross-validation. | 2.5 Days |

---

### 10. References

1. **Åström, K. J., & Hägglund, T. (2006).** *Advanced PID Control*, Chapter
   1: Introduction. ISA — The Instrumentation, Systems, and Automation
   Society. — PID prevalence statistics motivating compensator-design scope
   (§1, §2.1).
2. **MathWorks. (2026a).** *Control System Toolbox*. [Online]. Available:
   https://www.mathworks.com/products/control.html. Accessed: Aug. 8, 2026.
   — Root locus/Bode/Nichols/compensator design scope and Control System
   Designer's interactive, GUI-based design model (§2.1, §2.2, §4.1, §5.2,
   §5.4).
3. **MathWorks. (2026b).** *Designing a High Angle of Attack Pitch Mode
   Control*. [Online]. Available:
   https://www.mathworks.com/help/simulink/slref/designing-a-high-angle-of-attack-pitch-mode-control.html.
   Accessed: Aug. 8, 2026. — Aerospace flight-control application of
   classical design tooling (§1).
4. **Analog Devices. (2026).** *AN-149: Modeling and Loop Compensation
   Design of Switching Mode Power Supplies*. [Online]. Available:
   https://www.analog.com/en/resources/app-notes/an-149.html. Accessed: Aug.
   8, 2026. — Power-electronics compensator-design domain motivation (§1).
5. **Dostal, F. (2021).** *Power Supply Design: How Bode Plots Can Help You
   Meet the Requirements for Dynamic Control Behavior*. Analog Devices.
   [Online]. Available:
   https://www.analog.com/en/resources/technical-articles/power-supply-design-how-bode-plots-can-help.html.
   Accessed: Aug. 8, 2026. — Corroborating power-electronics Bode-plot use
   case (§1).
6. **python-control. (2026a).** *python-control/python-control*. [Online].
   Available: https://github.com/python-control/python-control. Accessed:
   Aug. 8, 2026. — Feature-set grouping (analysis vs. design), and
   `matplotlib`/`slycot` host-dependency profile (§2.1, §2.2, §4.1, §6).
7. **JuliaControl. (2026).** *ControlSystems.jl README.md*. [Online].
   Available:
   https://github.com/JuliaControl/ControlSystems.jl/blob/master/README.md.
   Accessed: Aug. 8, 2026. — Analysis/Plotting command split and
   `pid`/`leadlink`/`laglink` compensator-primitive naming precedent (§2.1,
   §4.1, §4.5).
8. **Octave-Forge. (2026).** *List of Functions for the 'control' package*.
   [Online]. Available: https://octave.sourceforge.io/control/overview.html.
   Accessed: Aug. 8, 2026. — `rlocus`/`rlocusx`/`margin`/`pid`/`pidstd`
   function set and absence of a `routh`-named function (§2.1, §4.1, §4.5,
   §6, §8).
9. **TorBorve. (2026).** *control_systems_torbox*. [Online]. Available:
   https://docs.rs/control_systems_torbox/latest/control_systems_torbox/.
   Accessed: Aug. 8, 2026. — Existing Rust prior art scope
   (`frequency_response`/`system_properties` only) and its GUI-plotting
   (`egui`/`egui_plot`) dependency profile (§2.2, §4.1, §5.4, §8).
10. **cool-japan. (2026).** *scirs2-signal*. [Online]. Available:
    https://docs.rs/scirs2-signal/latest/scirs2_signal/. Accessed: Aug. 8,
    2026. — Closer Rust functional analog (`root_locus`, `nichols_chart`,
    `stability_margins`) and its non-`no_std` dependency profile (§8).
11. **python-control. (2026b).** *python-control/Slycot*. [Online].
    Available: https://github.com/python-control/Slycot. Accessed: Aug. 8,
    2026. — SLICOT-wrapper GPLv2 licensing status, motivating rejection of
    an equivalent vendored dependency (§2.3, §5.3).
12. **Shamshiri, R. R. (2009).** *Routh-Hurwitz Stability test*. MATLAB
    Central File Exchange (submission #25956). [Online]. Available:
    https://www.mathworks.com/matlabcentral/fileexchange/25956-routh-hurwitz-stability-test.
    Accessed: Aug. 8, 2026. — Third-party, uncorroborated Routh-array
    reference implementation (§8).
13. **Sagharchi, F. (2016).** *Routh-Hurwitz stability criterion*. MATLAB
    Central File Exchange (submission #17483). [Online]. Available:
    https://www.mathworks.com/matlabcentral/fileexchange/17483-routh-hurwitz-stability-criterion.
    Accessed: Aug. 8, 2026. — Second third-party, uncorroborated Routh-array
    reference implementation (§8).

---

### 11. Revision History

| Revision | Date | Author | Description of Changes |
|:---|:---|:---|:---|
| 1.0 | August 8, 2026 | @MitchellDScott | Initial draft: functional scope, numeric-core/no-plotting architecture, root locus and Routh-Hurwitz precedent mapping to existing `Polynomial`/`Matrix` primitives, and open questions on Routh degenerate cases and margin sweep resolution. |
