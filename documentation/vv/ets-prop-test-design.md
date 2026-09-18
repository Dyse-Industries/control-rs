# Range-Valued On-Target Property Tests (Design Document)

![Date Badge](https://img.shields.io/badge/Date-September_8,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

ETS suites today assert numerical properties at the specific points a test
author wrote down. A matrix kernel that passes at three hand-picked operating
points says nothing about the rest of the flight envelope.

Usage scenarios:

- A firmware engineer runs an ETS suite on a Teensy 4 and needs a factorization
  residual bound to hold across an input range, not at sampled points.
- The same property runs unchanged on the host under `cargo ci`, so a range
  failure is reproducible off-target before anyone reaches for hardware.
- A suite author sweeps a kernel across several element types and dimensions,
  and expects each instantiation to report as its own ETS test entry.
- A suite must still fit in flash beside the rest of the firmware image.

---

### 2. Requirements

#### 2.1 Functional Requirements

- **FR-1 — Range-valued numeric substitution**: A numeric value denotes a
  closed range of reals and stands in for a scalar wherever crate algorithms
  are already generic over their element type. Algorithms are not rewritten to
  know about ranges.
- **FR-2 — Sound enclosure**: Evaluating an algorithm on input ranges yields a
  result range containing every result the scalar algorithm would produce for
  any point in those inputs. Without inclusion a passing test proves nothing.
- **FR-3 — Enclosure valid under round-to-nearest**: Rounding error committed
  while computing the enclosure stays inside it, on targets that offer no way
  to change the floating-point rounding mode.
- **FR-4 — Range verdict**: A property reduces its result range to a pass or
  fail on target. A range that straddles the acceptance bound fails; the check
  does not silently narrow to a midpoint.
- **FR-5 — Instantiation as ordinary suite entries**: A bounded set of type and
  dimension instantiations registers as individual ETS test entries, so a
  failure names the instantiation that produced it. The bound is author-supplied,
  as in published bounded-exhaustive tooling (Politano et al., 2024).

#### 2.2 Non-Functional Requirements

- **NFR-1 — Target execution**: The range type compiles and runs on the crate's
  embedded targets under `no_std` with no allocation.
- **NFR-2 — No new dependency**: The capability adds no third-party crate to
  the dependency tree, consistent with the standing rule to minimize
  dependencies.
- **NFR-3 — Pay-per-use footprint**: Suites that do not instantiate a property
  at the range type carry no flash or stack cost for it.

#### 2.3 Constraints

- **C-1 — Elementary arithmetic only**: Ranges support addition, subtraction,
  multiplication, division, and square root. Transcendental enclosures
  (exponential, logarithm, trigonometry) are out of scope, so the range type
  satisfies the crate's scalar-level bounds but not its full floating-point
  bound.
- **C-2 — Binary floating-point elements**: `f32` and `f64` only. Fixed-point
  and complex element types are out of scope.
- **C-3 — Existing suite registration is unchanged**: No new procedural macro
  and no change to how ETS suites are declared or discovered.
- **C-4 — Does not replace host oracle validation**: Cross-checks against host
  oracles conformant to `documentation/vv/oracle-harness-design.md` remain the
  validation layer; this design adds a verification technique beside them.

---

### 3. Technical Overview

The work is a small numeric type plus trait implementations in `src/math`, and
a convention for writing ETS suites against it. The hard part is not volume of
code but floating-point care: every operation must round its endpoints outward,
and the type must be honest about comparisons it cannot decide. Required
expertise is IEEE 754 rounding behavior, interval arithmetic's inclusion
property, and `no_std` constraints on the crate's embedded targets.

---

### 4. Architecture

#### 4.1 The range type

`Interval<T>` holds two endpoints, `lo` and `hi`, with the invariant
`lo <= hi`. A degenerate interval (`lo == hi`) is an exact scalar, which is how
a property mixes fixed constants with swept inputs. Construction from an
inverted or NaN pair is rejected rather than silently repaired, so an enclosure
never begins in a state that cannot be sound.

IEEE Std 1788-2015 specifies interval operations as a layer between hardware
and language and explicitly "does not define any realization of the basic
operations as functions in a programming language" (IEEE, 2015). This design
follows the standard's arithmetic model for the operations in C-1 and does not
claim conformance to it.

#### 4.2 Outward rounding

Each operation computes its endpoints in the default round-to-nearest mode and
then widens: the lower endpoint moves down one representable step, the upper
endpoint moves up one. The steps come from `next_down` and `next_up` on the
primitive float types, which implement IEEE 754 `nextDown`/`nextUp` exactly
(Peters, 2021). They are integer-arithmetic operations on the bit pattern,
which "avoids issues with denormal values potentially flushing to zero during
floating point arithmetic operations on some platforms" (Peters, 2021) — the
reason they behave identically on host and on target.

This is the round-to-nearest emulation of directed rounding that Rump et al.
describe, chosen because "performing a computation using directed roundings may
not be supported by the programming language in use, or it may depend on a
rounding mode change, which usually involves a flush of the processor pipeline"
(Rump et al., 2009). Rust exposes no portable rounding-mode control, so the
question is settled by the language rather than by preference.

Multiplication and division take the extremal combination of endpoint products
before widening, which is what makes sign-crossing inputs safe.

#### 4.3 Trait participation

`Interval<T>` implements the existing `src/math/num_traits` hierarchy up to the
scalar level, which is the bound the crate's linear-algebra kernels already
consume:

| Trait | Status | Note |
|:------|:-------|:-----|
| `Zero`, `One` | implemented | degenerate intervals at 0 and 1 |
| `Add`, `Sub`, `Mul`, `Div` | implemented | outward-rounded per §4.2 |
| `AdditiveGroup`, `Signed` | implemented | `abs` maps to the range of magnitudes |
| `Conjugate` | implemented | identity; intervals here are real |
| `Scalar` | implemented | `Real = Self` |
| `Radical` | implemented | `sqrt` outward-rounded |
| `PartialOrd` | partial | `None` when intervals overlap |
| `Exponential`, `Trig` | not implemented | C-1 |
| `Float` | not implemented | requires `Exponential` and `Trig` |

`PartialOrd` returning `None` for overlapping intervals is the mathematically
honest answer, and it is load-bearing: an algorithm that branches on an
undecidable comparison cannot be evaluated at this type, and the compiler and
the test will say so rather than producing an unsound bound.

A small trait local to the module supplies `next_up`/`next_down` for `f32` and
`f64`. Widening the crate-wide floating-point trait instead would force every
existing implementor, including the fixed-point type, to answer a question that
only binary floats have.

#### 4.4 Suite integration

A property is an ordinary generic function over the crate's scalar bound. A
suite entry is a concrete zero-argument function that instantiates that property
at one element type, one set of dimensions, and one input range, then asserts
the verdict from FR-4. Registration and discovery are the existing mechanism
(C-3): each instantiation is already a plain `fn()`, which is what the suite
attribute registers, so a failure is attributed by test name with no change to
the executor.

Input ranges are chosen by equivalence partitioning and boundary value analysis
(ISO/IEC/IEEE, 2021) — a partition per operating regime, plus the endpoints of
each partition as degenerate intervals.

---

### 5. Alternatives

**Adopt `inari`.** The mature Rust interval library conforms to IEEE Std
1788.1-2017 (inari, 2026), which makes it the obvious candidate. It cannot run
here: its supported CPUs are "x86-64 ... AArch64 (ARM64)" (inari, 2026), and the
crate's ETS targets are `thumbv7em-none-eabi{,hf}` and
`riscv32imac-unknown-none-elf`. Its default feature set also brings GMP and MPFR
through `gmp-mpfr-sys`, and with that feature "target platforms are limited to
those that are supported by the `gmp-mpfr-sys` crate" (inari, 2026). A
dependency-tree check confirms the cost: default features pull ten transitive
crates including a C library build, and disabling them leaves a lean two-crate
tree that still targets only host architectures. It remains a candidate as a
host-side cross-check oracle, which is a separate question from this design.

**Sample the range instead of enclosing it.** Executing a property at every
input up to a bound is bounded exhaustive testing, justified by the small scope
hypothesis: "the majority of software defects can be revealed by relatively
small inputs" (Politano et al., 2024; Coppit et al., 2005). For a continuous
numerical range this degrades to sampling, which proves nothing between
samples, and the sample count multiplies out — "this combinatorial explosion is
an inevitable consequence of relentlessly increasing a uniform depth-limit for
exhaustive testing" (Runciman et al., 2008), a serious cost when each execution
occupies target hardware. Enumeration is still the right tool for the discrete
axis, which is why FR-5 keeps it for type and dimension arguments; where
enumeration of those axes grows too large, covering arrays bound it, since
"testing all 3-way combinations may detect 90% or more of bugs" (Kuhn et al.,
2010).

**Generate instantiations with a parameterized-test macro.** `rstest` expands a
value list into "an independent test for each case" and builds matrix tests as
"the cartesian product of all the values" (rstest, 2026). This is the ergonomic
form of FR-5, but it is a host testing crate and a new dependency, against
NFR-2 and C-3. Hand-written instantiation thunks cost more keystrokes and no
dependency.

**Switch the hardware rounding mode.** Directly setting round-toward-negative
and round-toward-positive gives the tightest enclosure. Rust exposes no portable
control over it, and the mode change "usually involves a flush of the processor
pipeline" (Rump et al., 2009).

**Widen by a relative factor rather than one step.** Scaling endpoints by a
relative rounding-error term is the classical emulation and needs no
`nextUp`/`nextDown`. It is looser: the resulting interval "is twice as wide as
it needed to be, i.e., 4 ulps (units in the last place) instead of 2 ulps"
(Rump et al., 2009). Since `next_up`/`next_down` are available in the core
library, the looser form buys nothing.

---

### 6. Verification & Validation

#### 6.1 Approach

Confirm the range type never reports a sound-looking enclosure that excludes a
reachable scalar result, and that it executes on the ETS targets.

| Method | Mechanism |
|:-------|:----------|
| Requirements-based test | `#[test]` over construction, ordering and each arithmetic operation, including sign-crossing and degenerate inputs |
| Property-based test | `proptest` asserting the inclusion property against sampled scalar evaluations |
| Metamorphic relation | `proptest` over transformed inputs: degenerate intervals reproduce scalar arithmetic, and widening inputs never narrows outputs |
| On-target execution | ETS suite exercising the same properties under QEMU and on physical Teensy 4.1 hardware |
| Static analysis | `cargo clippy-ci` and source inspection for added dependencies and per-use footprint |
| Compile-time shape check | Suite registration compiles unchanged; elementary arithmetic traits only; binary floats only |
| Inspection | Design documentation verification of oracle validation separation |
| Coverage measurement | `cargo coverage` reporting statement coverage for `src/math` |

Target: 90% statement coverage of the range type once it exists, measured
with `cargo coverage`. Excluded: `Debug` / `Display`. There is no executable
surface to cover at this revision.

A range-valued property replaces any sampled test whose kernel is generic over
the scalar bound, requiring the property test to reach the same verdict on the
same inputs. Host oracle cross-validation conformant to
`documentation/vv/oracle-harness-design.md` stays in place as the
independent-implementation check (C-4).

#### 6.2 Acceptance

| Property | Oracle | Measure | Bound |
|:---------|:-------|:--------|:------|
| Enclosure soundness | invariant (inclusion isotonicity) | containment of the scalar result in the interval result | holds for every sampled point, no exceptions |
| Endpoint tightness | closed form | extra width introduced per operation | at most one representable step per endpoint |
| Degenerate agreement | scalar evaluation | endpoint difference from the scalar result | within the widening of the operation sequence |
| Undecidable ordering | closed form | comparison result on overlapping intervals | `None` |
| Target build | toolchain | build status for each ETS target | success |

Floating-point results are compared by containment or by a stated tolerance,
never by exact equality.

#### 6.3 Limits

- **FR-1..FR-5, NFR-1..NFR-3, C-1..C-3**: no range type in `src/math` and no
  `test:` locator exists. Planned methods in 6.2 are the contract once the
  type lands.
- Conformance to IEEE Std 1788-2015 is not claimed.
- Enclosure tightness over long operation sequences is not bounded.

---

### 7. Performance & Resource Considerations

A range value is two scalars, so storage doubles and arithmetic costs roughly
two to four scalar operations plus two bit-pattern steps per result. The steps
are integer operations, not floating-point ones (Peters, 2021). Per NFR-3, cost
is confined to suites that instantiate a property at the range type. The flash
and stack cost of range-valued evaluation relative to scalar evaluation on a
Cortex-M target is a measurement that has not been taken.

---

### 8. Risks & Open Questions

- **Overestimation growth.** Interval arithmetic loses correlation between
  repeated occurrences of the same variable, so enclosures widen through long
  computations. A residual bound that is true pointwise may still fail as an
  enclosure. Whether the crate's kernels stay tight enough to be useful is
  unknown until measured.
- **Undecidable control flow.** Pivoting and other comparison-driven branches
  may be unevaluable at the range type, which would restrict the technique to a
  subset of kernels.
- **Reach of the scalar-level bound.** C-1 excludes algorithms bounded by the
  full floating-point trait. Whether the reachable subset covers enough of the
  numerical models to justify the work is the main open question for review.
- **Reach across `classical_tools`.**
  - `Biquad`, `Pid`, and compensator factories require `Float` (`mul_add`,
    transcendentals), so constraint C-1 excludes them from range-valued property
    testing without transcendental interval extensions.
  - Stability and frequency-domain tools such as `routh` and `margins` satisfy
    the scalar-level bound, but `routh`'s epsilon-perturbation branch and
    `margins`' bisection search are comparison-driven — introducing the
    undecidable control flow risk identified above when intervals bracket zero or
    crossing thresholds.
  - FR-5 on-target execution for `classical_tools` cannot run on embedded
    hardware until firmware image linking expands beyond `math` and `matrix`
    (which are currently the only suites linked in target builds to preserve
    flash headroom).
- **Footprint.** Unmeasured, per §7.

---

### 9. Development Plan

| Task / Feature | Description | Estimated Effort (1-10) |
|:---------------|:------------|:------------------------|
| Phase 1: Range type and rounding | `Interval<T>`, construction invariants, outward-rounded arithmetic and `sqrt`, the local step trait for `f32`/`f64` | 4 |
| Phase 2: Trait integration | Implement the §4.3 trait rows; confirm existing scalar-bounded kernels instantiate at the range type | 3 |
| Phase 3: Suite convention | Verdict helper, instantiation thunks, one ETS suite on both targets | 3 |
| Phase 4: Verification | §6 unit, property, and metamorphic tests; replace one sampled numerical-model test; measure footprint | 4 |

---

### 10. Revision History

| Revision | Date              | Author          | Description                                                                                                                                                                                                         |
|:---------|:------------------|:----------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1.0      | September 8, 2026 | @MitchellDScott | Initial draft.                                                                                                                                                                                                      |
| 1.1      | September 9, 2026 | @MitchellDScott | Structural hardening: converted §6.4 from prose into complete traceability table mapping all requirements, updated badge to standard dialect, mapped catalogue methods in §6.2, standardized revision history. |
| 1.2      | September 15, 2026 | @MitchellDScott | Locator-only §6.4; unimplemented range type listed in 6.7; citations removed from FR bodies. |

---

## References

[1] S. M. Rump, *Computer-assisted Proofs and Self-validating Methods*, in
*Accuracy and Reliability in Scientific Computing*. Philadelphia, PA, USA:
SIAM, 2005. [Online]. Available:
https://www.tuhh.de/ti3/paper/rump/Ru05a.pdf

[2] IEEE, "IEEE Standard for Interval Arithmetic," IEEE Std 1788-2015, New
York, NY, USA, 2015. [Online]. Available:
https://standards.ieee.org/ieee/1788/4431/

[3] S. M. Rump, P. Zimmermann, S. Boldo, and G. Melquiond, "Computing
predecessor and successor in rounding to nearest," *BIT Numer. Math.*, vol. 49,
no. 2, pp. 419–431, 2009, doi: 10.1007/s10543-009-0218-z.

[4] M. Politano, V. Bengolea, F. Molina, N. Aguirre, M. Frias, and P. Ponzio,
"BEAPI: A tool for bounded exhaustive input generation from APIs," *Sci.
Comput. Program.*, no. 103153, 2024, doi: 10.1016/j.scico.2024.103153.

[5] O. Peters, "RFC 3173: float_next_up_down," in *rust-lang/rfcs*, 2021.
[Online]. Available:
https://rust-lang.github.io/rfcs/3173-float-next-up-down.html. Accessed: Sep. 8,
2026.

[6] ISO, IEC, and IEEE, "Software and systems engineering — Software testing —
Part 4: Test techniques," ISO/IEC/IEEE 29119-4:2021, Geneva, Switzerland, 2021.
[Online]. Available: https://www.iso.org/standard/79430.html. Accessed: Sep. 8,
2026.

[7] inari, *inari: Interval Arithmetic Library for Rust* (Version 2.0.0), 2026.
[Online]. Available: https://github.com/unageek/inari. Accessed: Sep. 8, 2026.

[8] D. Coppit, J. Yang, S. Khurshid, W. Le, and K. Sullivan, "Software assurance
by bounded exhaustive testing," *IEEE Trans. Softw. Eng.*, 2005, doi:
10.1109/TSE.2005.52.

[9] C. Runciman, M. Naylor, and F. Lindblad, "SmallCheck and Lazy SmallCheck
automatic exhaustive testing for small values," in *Proc. 1st ACM SIGPLAN Symp.
Haskell*, Victoria, BC, Canada, 2008, pp. 37–48. [Online]. Available:
https://www.cs.york.ac.uk/fp/smallcheck/smallcheck.pdf

[10] D. R. Kuhn, R. N. Kacker, and Y. Lei, "Practical Combinatorial Testing,"
NIST Special Publication 800-142, National Institute of Standards and
Technology, Gaithersburg, MD, USA, Oct. 2010. [Online]. Available:
https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-142.pdf

[11] rstest, *rstest* (Version 0.27.0), la10736/rstest, 2026. [Online].
Available: https://github.com/la10736/rstest. Accessed: Sep. 8, 2026.
