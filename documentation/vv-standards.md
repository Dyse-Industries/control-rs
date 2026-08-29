# Verification & Validation Standards for `control-rs`

These rules define how a design document specifies the experiments, tests and
examples that establish a component is correct. They govern §6 of
`documentation/design-template.md` and the test code that section commits to.
`documentation/doc-standards.md` governs prose and rustdoc; this file governs
evidence.

# 1. Definitions

* **Verification**: confirmation that the component satisfies the requirements
  stated for it. Verification asks whether the component was built right [1].
* **Validation**: confirmation that the component serves its intended purpose
  in its intended environment [1], [2].
* **Code verification**: confirmation that a numerical algorithm is implemented
  correctly, established by evaluating error against a known solution of high
  accuracy [3].
* **Solution verification**: estimation of the numerical accuracy of a
  particular computed result [3].

A numerical component needs both. Code verification is a property of the
implementation and belongs in §6. Solution verification is a property of a
given call and belongs in the rustdoc `# Errors` / `# Limitations` contract.

**Project convention.** ETS suites are verification. They execute a component
on target hardware or an emulator against acceptance criteria the design
states, so they answer whether the component was built right. Published
systems-engineering guidance instead classifies target-platform execution as
validation [4]; that classification applies to whole-system demonstrations,
which in this crate are examples and external projects (§8).

# 2. Required structure of §6

Every design document's §6 contains the following subsections in this order.
Omit a subsection only when the reason is stated in one sentence under its
heading.

| Subsection | Contents |
|:-----------|:---------|
| 6.1 Objectives | What evidence the implementation must produce, one line per objective. |
| 6.2 Methods | The method table (§3). One row per method the component uses. |
| 6.3 Acceptance criteria | The oracle, error measure and numeric bound for each numerical claim (§4, §5). |
| 6.4 Traceability | Requirement ID to method mapping (§6). |
| 6.5 Coverage | Coverage target and stated exclusions (§7). |
| 6.6 Validation | Examples, demos and external use (§8). |
| 6.7 Not verified | Claims the plan does not establish, and why. |

6.7 is mandatory. A plan that lists no gaps is incomplete, not complete.

Section titles may be re-titled or combined when a component's plan is small,
consistent with the tailoring allowance in the source test-documentation
standard [5]. The seven contents above may not be dropped.

# 3. Method catalogue

Each row of the §6.2 method table names one method, the mechanism that
executes it and the requirement IDs it discharges. Methods are drawn from the
following catalogue. Do not invent method names.

| Method | Mechanism | Establishes |
|:-------|:----------|:------------|
| Compile-time shape check | Const-generic dimensions; `compile_fail` doctest [6] | Illegal shapes and units are unrepresentable. A type system is a syntactic method for proving the absence of a class of behaviors [7]. |
| Inspection | `/cr-review design` | The design is internally consistent and traceable. |
| Static analysis | `cargo lint`, `cargo clippy-ci` | Absence of lint-detectable defect patterns. |
| Requirements-based test | `#[test]` unit and integration tests | A named FR holds on chosen inputs. |
| Property-based test | `proptest` [8] | An invariant holds over generated inputs. |
| Doctest | Runnable rustdoc example | The documented contract compiles and runs as written. |
| Back-to-back comparison | `examples/prototypes/<project>/<slug>/` via `/cr-prototype` | Agreement with an independent implementation to a stated tolerance. |
| Metamorphic relation | `#[test]` or `proptest` over transformed inputs | An input-output relation holds where no oracle exists [9]. |
| Resource usage evaluation | Stack analysis, `no_alloc` review, ETS timing | Space and time bounds hold on target. |
| On-target execution | ETS suites under QEMU and Teensy | Behavior holds under real toolchain, ABI and timing. |
| Coverage measurement | `cargo coverage` | The test set exercises the implementation. |

Use the cargo aliases in `documentation/development-guide.md`. Do not hand-roll
the underlying commands in a plan.

# 4. Acceptance criteria

Every numerical claim in §6.3 states three things. A claim missing any of them
is not an acceptance criterion.

1. **Oracle**: what the computed value is compared against (§5).
2. **Error measure**: how the difference is measured.
3. **Bound**: the numeric threshold, and the reason it holds.

## 4.1 Error measures

| Measure | Use when |
|:--------|:---------|
| Exact equality | Integer, fixed-point exact operations and structural results (shape, rank, index, ordering). |
| Absolute error | The expected magnitude is known and bounded away from zero. |
| Relative error | The result spans several orders of magnitude. |
| ULP distance | Comparing against a correctly-rounded reference [10]. |
| Residual test ratio | Factorizations, solves and eigenproblems (§4.2). |

For floating-point comparison in Rust, `approx` supplies `abs_diff_eq!`,
`relative_eq!` and `ulps_eq!` with matching `assert_*` forms [11]. Adopting it
is a per-module dependency decision recorded in that module's design doc.

## 4.2 Residual test ratios

Linear-algebra results are accepted on a scaled residual rather than on the
computed factors. The reference-implementation pattern scales a residual norm
by the problem dimension and machine epsilon, and fails any ratio at or above
a threshold [12]:

```text
r = ||A - LU|| / (n * ||A|| * eps)      pass if r < tau
```

State `tau` in §6.3. The reference LAPACK test programs ship `tau = 20.0` [12].
That value is a host-side convention for a general-purpose library; a design
that adopts it says so, and a design that departs from it gives the reason.

## 4.3 Justifying a bound

A bound is justified by conditioning or by a cited backward-error result, not
by the output of the code under test. An algorithm is backward stable when the
computed result is the exact result for a nearby input [13]; state the backward
error the algorithm is expected to achieve, then bound the forward error by
multiplying it by the condition number of the problem. Where the condition
number is expensive to compute, state the estimator used [13].

## 4.4 Prohibited

* `assert_eq!` or `==` on floating-point results.
* A bound tightened until the current implementation passes.
* Golden values recorded from a run of the code under test.
* An unexplained tolerance constant.

Recording a golden value is permitted only under §5, rule 5.

# 5. Oracles

Choose the highest applicable oracle. State which one §6.3 uses.

1. **Closed-form exact solution.** Where a component solves an equation,
   construct the case by choosing the answer first and deriving the input that
   produces it. This is the method of manufactured solutions [3]. It is the
   preferred oracle for solvers, integrators and estimators.
2. **Independent reference implementation.** A host-side numerical oracle
   written from the design doc by `/cr-prototype`, compared back-to-back. The
   prototype is independent only if it was not derived from the Rust source.
3. **Metamorphic relation.** Where no oracle exists, verify a transformation
   instead of a value: relate an input change to the expected output change
   [9]. Examples: scaling a system and its response, transposing a symmetric
   operand, permuting inputs to an order-invariant reduction.
4. **Invariant or algebraic property.** Conservation, symmetry, idempotence,
   round-trip identity, monotonicity, bounds. Verified with `proptest` over
   generated inputs [8].
5. **Recorded golden value.** Last resort. State the provenance: the tool, its
   version and the exact inputs. A golden value with no stated provenance is
   not evidence.

Determinism of the comparison rests on the arithmetic being reproducible:
results and exceptions are uniquely determined by the input values, the
sequence of operations and the destination formats [10]. A plan that compares
across targets states the precision and the operation order it assumes.

# 6. Traceability

§6.4 carries a table mapping every requirement ID in §2 to the methods that
discharge it.

| Requirement | Method | Artifact |
|:------------|:-------|:---------|
| FR-1 | Property-based test | `tests/<name>.rs::prop_<name>` |
| NFR-2 | Resource usage evaluation | ETS suite `<name>` |

Traceability runs both ways. A requirement with no method is an untested
requirement; a test that discharges no requirement is either an undocumented
requirement or an unnecessary test [14]. Resolve both before the design is
marked `Reviewed`.

# 7. Coverage

State the coverage target and what is excluded from it. Report statement and
branch coverage from `cargo coverage`. Structural coverage evaluates the
completeness of the test set; it does not evidence correctness [15]. Do not
present a coverage percentage as an acceptance criterion for a numerical claim.

Exclusions are named, not implied. Typical exclusions: target-only code paths
measured by ETS instead, and `Debug` / `Display` implementations.

# 8. Validation

Validation artifacts demonstrate the component serves its purpose.

* **Examples**: at least one example per public capability, runnable via
  `cargo test`, using `?` rather than `unwrap()`, `expect()` or `panic!()`, per
  `documentation/doc-standards.md` §5.
* **Demonstrations**: an end-to-end run on target hardware where the component
  participates in a closed loop.
* **External use**: use of the component from a project outside this workspace.

Validation does not substitute for a missing acceptance criterion. A plan whose
only evidence is an example that runs has verified nothing.

# 9. Section skeleton

Copy into §6 of a new design document and fill in.

```markdown
### 6. Verification & Validation

#### 6.1 Objectives

- [What the implementation must demonstrate, one line each.]

#### 6.2 Methods

| Method | Mechanism | Requirements discharged |
|:-------|:----------|:------------------------|
|        |           |                         |

#### 6.3 Acceptance criteria

| Claim | Oracle | Measure | Bound | Justification |
|:------|:-------|:--------|:------|:--------------|
|       |        |         |       |               |

#### 6.4 Traceability

| Requirement | Method | Artifact |
|:------------|:-------|:---------|
|             |        |          |

#### 6.5 Coverage

Target: [statement / branch]. Excluded: [named paths, with reason].

#### 6.6 Validation

- Examples: [...]
- Demonstrations: [...]

#### 6.7 Not verified

- [Claim the plan does not establish, and why.]
```

# 10. Review checklist

`/cr-review <project>/<slug> design` checks §6 against the following. Each
failure is blocking.

* All seven subsections of §2 present, or an omission reason given.
* Every method row names a catalogue method from §3.
* Every numerical claim states oracle, measure and bound.
* No bound is justified by the behavior of the implementation under test.
* No floating-point exact comparison outside the cases permitted in §4.1.
* Every requirement ID in §2 appears in the §6.4 table.
* Every artifact in the §6.4 table discharges a requirement.
* Coverage exclusions are named.
* §6.7 lists at least one gap or states that the plan is exhaustive and why.

# References

[1] NASA, "SWE-028 - Verification Planning," NASA Software Engineering
Handbook, 2023.

[2] NASA, *NASA Systems Engineering Handbook*, NASA/SP-2016-6105 Rev 2,
Washington, DC, USA, 2016.

[3] D. Yeo, "A Summary of Industrial Verification, Validation, and Uncertainty
Quantification Procedures in Computational Fluid Dynamics," NISTIR 8298, NIST,
Gaithersburg, MD, USA, 2020.

[4] NASA, "SWE-073 - Platform or Hi-Fidelity Simulations," NASA Software
Engineering Handbook, Ver. D, 2022.

[5] ISO/IEC/IEEE, "Software and systems engineering - Software testing - Part
3: Test documentation," ISO/IEC/IEEE 29119-3:2021, ISO, Geneva, Switzerland,
2021.

[6] The Rust Project Developers, "Documentation tests," *The rustdoc book*.

[7] B. C. Pierce, *Types and Programming Languages*. Cambridge, MA, USA: MIT
Press, 2002.

[8] K. Claessen and J. Hughes, "QuickCheck: A Lightweight Tool for Random
Testing of Haskell Programs," in *Proc. 5th ACM SIGPLAN Int. Conf. Functional
Programming*, Montreal, Canada, 2000, pp. 268-279.

[9] S. Segura, G. Fraser, A. B. Sanchez and A. Ruiz-Cortes, "A Survey on
Metamorphic Testing," *IEEE Trans. Softw. Eng.*, vol. 42, no. 9, pp. 805-824,
2016.

[10] IEEE, "IEEE Standard for Floating-Point Arithmetic," IEEE Std 754-2019,
IEEE, New York, NY, USA, 2019.

[11] approx, version 0.5.1, crates.io, 2022.

[12] E. Anderson, J. Dongarra and S. Ostrouchov, "Installation Guide for
LAPACK," LAPACK Working Note 41, Univ. of Tennessee, Knoxville, TN, USA, 1994.

[13] E. Anderson et al., *LAPACK Users' Guide*, 3rd ed. Philadelphia, PA, USA:
SIAM, 1999.

[14] NASA, "SWE-072 - Bidirectional Traceability Between Software Test
Procedures and Software Requirements," NASA Software Engineering Handbook,
2017.

[15] ISO, "Road vehicles - Functional safety - Part 6: Product development at
the software level," ISO 26262-6:2018, ISO, Geneva, Switzerland, 2018.
