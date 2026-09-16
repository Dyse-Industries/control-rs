# Type/Module Name (Design Document)

![Date Badge](https://img.shields.io/badge/Date-July_12,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@AuthorName-blueviolet)

---

### 1. Introduction

Scope and motivation in a short paragraph, then **Primary usage scenarios**:
3–6 named jobs a user can fail at. File paths, trait trees, and "this module
provides" lists belong in §3/§4, not here.

A requirement that cannot point at a scenario in this section, or at one
inherited standing constraint, is architecture — move it to §4.

```
- Count right-half-plane roots of a characteristic polynomial.
- Advance plant state between controller ticks on a bare-metal target.
```

---

### 2. Requirements

A requirement is a testable user need, not a feature. Derive off the page;
publish only finished IDs under 2.1, 2.2 and 2.3. Classify / Derive / Shape
below are authoring rules, not headings in the finished document.

#### Classify (authoring)

- **FR** — observable behavior the module must provide.
- **NFR** — a quality of that behavior (cost, size, determinism).
- **C** — a bound, inherited decision, or explicit out-of-scope.

The same claim must not appear as both NFR and C. Inherited toolbox or crate
bounds are one pointer ID, not a restated paragraph:

```
- **C-n — Inherited toolbox bounds**: Conforms to
  `controls-tools-design.md` C-1..C-4.
```

Do not publish a deferred FR. Roadmap belongs in §8/§9.

#### Derive (authoring)

1. Restate each §1 scenario as a job.
2. List needs as one-liners and classify each as FR, NFR, or C.
3. Split any one-liner that contains two independently testable claims.
4. Name the ID from the need, not the solution (`Exact product rescale`,
   not `Widening multiply helper`).
5. Body: 1–3 sentences. First states the need. Second bounds it (what is
   not required, or what would go wrong). Do not cite research in the FR
   body; citations live in §4 and the §6.2 bound justification. Do not
   name types, methods, or files unless the need *is* that contract.
6. Order IDs as you would explain them: core capability, then consequences,
   then how this module participates in the rest of the crate. Do not group
   by subsystem.
7. Every ID traces to a §1 scenario or a standing crate constraint (minimize
   dependencies, no panic outside tests, toolbox C-1..C-4, an
   Approved sibling design). If it does not, it is architecture — move it
   to §4.

Keep existing IDs stable. Split only when two claims are independently
testable; append the new ID. A retired ID is listed in §6.3 as withdrawn.

#### Shape (authoring)

```
- **FR-n — Short name**: One to three sentences. First states the need.
  Second bounds it.
```

Same form for `NFR-n` and `C-n`. Typical size: 4–8 FRs, 1–3 NFRs, 2–5 Cs.
Present tense; `must` / `does not`.

**Accept** — one need, named from the need, next ID is a consequence of the
last:

- **FR-3 — Zero-copy strided reinterpretation**: Algorithms require
  reinterpreting underlying storage across submatrix windows, transpositions,
  and vector reversals through signed strides without data copying. Copying
  matrix elements for stride or orientation adjustments would exhaust
  microcontroller stack budgets and introduce runtime allocation.

- **FR-4 — Exact product rescale**: Multiplication forms its product in a
  representation wide enough to hold it before rescaling. A same-width
  multiply would discard the high half of every product.

**Reject** — architecture slogan as the title, nested inventory, several
claims glued with "and", API mandate, or a literature review in the body:

- **FR-1 — Decoupled Storage Subsystems Architecture**: Provide distinct,
  zero-cost storage subsystem contracts across dense strided arrays
  (`DenseStorage<T>` / `Storage<T, R, C>`), packed structured matrices
  (`PackedStorage<T>`), and compressed sparse backends (`SparseStorage<T>`).

- **FR-1 — Continuous & Discrete Algebraic Riccati Solvers**: The module
  must solve the Continuous-Time Algebraic Riccati Equation and the
  Discrete-Time Algebraic Riccati Equation for stabilizing positive
  semi-definite $P$.

Also reject: nested trait/kernel sub-bullets; an FR that is only true after
choosing the §4 design; the same cap restated as both NFR and C; paper
citations in the FR body; `*(deferred)*` IDs. Nested sub-bullets mean the
item is an architecture dump — split or move.

---

### 3. Technical Overview

A brief explanation of the project's scope and the expertise it will require.

---

### 4. Architecture

Describe the implementation in detail, start from a broad scope then focus in
on the specifics. Algorithm choice, type names, and research cites that
justify a design live here.

---

### 5. Alternatives

Describe any architecture/implementation details that were considered but
not chosen.

---

### 6. Verification & Validation

The plan for showing this component correct. Verification asks whether the
component was built right, validation whether it serves its purpose [1], [2].
That split is the frame, not the structure: the structure is the four kinds of
evidence this workspace produces, `test`, `bench`, `example` and
`cross-check`. Declare the ones this component uses and what each establishes.

Code verification, that a numerical algorithm is implemented correctly, is
established by evaluating error against a known solution of high accuracy [3]
and belongs here. Solution verification, the accuracy of one particular
computed result, belongs in the rustdoc `# Errors` and `# Limitations`
contract, not here.

The requirement tracer does not read this section. Association is
`#[req_trace]` (`doc-standards.md` §3.3); status and fully qualified paths are
the tracer report (`vv/requirement-traceability-design.md`). Never write a
test path, a file name, a case count, a requirement-ID discharge column, or a
verified/unverified status in §6.

#### The four kinds (authoring)

| Kind | Mechanism | Established by | Discharges |
|:--|:--|:--|:--|
| `test` | `#[test]` unit and integration tests, `proptest` properties, dual-mode ETS suites, rustdoc doctests including `compile_fail` [5] | `cargo test`, `cargo ci` | FR, NFR or C directly |
| `bench` | criterion benches under `benches/` | `cargo bench` | An NFR only when 6.2 states a budget; otherwise reports |
| `example` | one runnable example per public capability | `cargo run --example`, built by `cargo test` | Nothing alone; fitness evidence |
| `cross-check` | a `control-rs-validation` suite against an independent oracle, compared through a tolerance table | `cargo compare` | A numerical claim only alongside a `test` |

ETS suites are `test`. They execute the component on target against acceptance
criteria this document states, which answers whether it was built right.
Systems-engineering guidance classifies target-platform execution as
validation [4]; that applies to whole-system demonstrations, which in this
crate are examples.

A `cross-check` is a code-to-code comparison. It establishes agreement with a
second implementation, and agreement is only as strong as a reference that is
itself unverified. It ranks below every oracle in 6.2 except a recorded golden
value, and it is never the only evidence for a numerical claim.

#### 6.1 Plan

One row per step. `Kind` is one of the four words. `Step` names the activity
in a noun phrase. `Establishes` is the claim, in the vocabulary of §2.

| Kind | Step | Establishes |
|:--|:--|:--|

Rules:

- A kind the component does not use has no row. Do not write "none".
- No paths, file names, case counts, or requirement IDs. The row says what is
  done and what it shows, not where it lives.
- A step that produces a numerical claim publishes a 6.2 row.
- Coverage is a CI gate, not a step. Name a coverage target here only when
  something must be excluded from it, with the reason.

#### 6.2 Acceptance

Numerical claims only. Omit the subsection when there are none. A criterion
without a bound is not a criterion.

| Claim | Oracle | Measure | Bound |
|:--|:--|:--|:--|

**Oracle**, strongest first. Use the highest that applies:

1. Closed-form or manufactured solution. Choose the answer, derive the input
   that produces it [3]. Preferred for solvers, integrators and estimators.
2. Independent reference implementation, under
   `vv/oracle-harness-design.md`, independent only if it was not derived from
   the Rust source.
3. Metamorphic relation: relate an input change to the expected output change
   where no oracle exists [7].
4. Invariant or algebraic property over generated inputs [6].
5. Recorded golden value. Last resort, and only with provenance: the tool,
   its version and the exact inputs.

**Measure**: exact equality for integer, fixed-point and structural results;
absolute error when the magnitude is known and bounded away from zero;
relative error across orders of magnitude; ULP distance against a
correctly-rounded reference [8]; relative $\ell_2$ for trajectories; interval
enclosure for a declared range; scaled residual ratio for factorizations,
solves and eigenproblems. In Rust, `approx` supplies the matching `assert_*`
forms [9].

**Bound**: justified by conditioning or by a cited backward-error result [11],
never by the output of the code under test. State the backward error the
algorithm is expected to achieve, then bound the forward error by the
condition number, naming the estimator when it is estimated [11]. A
factorization is accepted on a scaled residual, not on the computed factors:

```text
r = ||A - LU|| / (n * ||A|| * eps)      pass if r < tau
```

Reference LAPACK ships `tau = 20.0` [10]. A design adopting it says so; a
design departing from it gives the reason.

A `cross-check` row cites its tolerance-table key
(`nmv.matrix.hilbert_solve.scipy`) rather than restating the number. A child
document citing a parent key still publishes its own row.

Prohibited: `assert_eq!` or `==` on floating-point results; a bound tightened
until the current implementation passes; a golden value with no provenance; an
unexplained constant; a row omitted because the number lives in a parent
table; a `bench` row with no budget.

#### 6.3 Limits

What this plan does not establish, and withdrawn IDs. Scope, not a backlog.
Omit when the plan is complete and no ID has been withdrawn. "Not tested yet"
and "deferred to Phase N" belong in §9.

---

### 7. Performance & Resource Considerations

(optional) - This section is used to describe any requirements based on the
practical nature of the implementation.

---

### 8. Risks & Open Questions

Acknowledge any unspecified details or partial thoughts here.

---

### 9. Development Plan

Include the required implementation tasks:

| Task / Feature | Description | Estimated Effort (1-10) |
|:---------------|:------------|:------------------------|
| Step 1: [...]  | [...]       | [...]                   |

---

### 10. Revision History

---

## References

[1] NASA, "SWE-028 - Verification Planning," NASA Software Engineering
Handbook, 2023.

[2] NASA, *NASA Systems Engineering Handbook*, NASA/SP-2016-6105 Rev 2,
Washington, DC, USA, 2016.

[3] D. Yeo, "A Summary of Industrial Verification, Validation, and Uncertainty
Quantification Procedures in Computational Fluid Dynamics," NISTIR 8298, NIST,
Gaithersburg, MD, USA, 2020.

[4] NASA, "SWE-073 - Platform or Hi-Fidelity Simulations," NASA Software
Engineering Handbook, Ver. D, 2022.

[5] The Rust Project Developers, "Documentation tests," *The rustdoc book*.

[6] K. Claessen and J. Hughes, "QuickCheck: A Lightweight Tool for Random
Testing of Haskell Programs," in *Proc. 5th ACM SIGPLAN Int. Conf. Functional
Programming*, Montreal, Canada, 2000, pp. 268-279.

[7] S. Segura, G. Fraser, A. B. Sanchez and A. Ruiz-Cortes, "A Survey on
Metamorphic Testing," *IEEE Trans. Softw. Eng.*, vol. 42, no. 9, pp. 805-824,
2016.

[8] IEEE, "IEEE Standard for Floating-Point Arithmetic," IEEE Std 754-2019,
IEEE, New York, NY, USA, 2019.

[9] approx, version 0.5.1, crates.io, 2022.

[10] E. Anderson, J. Dongarra and S. Ostrouchov, "Installation Guide for
LAPACK," LAPACK Working Note 41, Univ. of Tennessee, Knoxville, TN, USA, 1994.

[11] E. Anderson et al., *LAPACK Users' Guide*, 3rd ed. Philadelphia, PA, USA:
SIAM, 1999.
