# Documentation Standards for `control-rs`

These rules establish a rigorous and consistent documentation standard for a
Rust native control systems toolbox intended for safety-critical applications.

§1 to §5 govern rustdoc and ETS documentation in source. §6 governs
Architecture Decision Records (ADRs), the record of decisions at the
workspace, crate or project level. §7 governs design documents. The
process that produces an ADR is in [`CONTRIBUTING.md`](../CONTRIBUTING.md); the design-document skeleton is
[`design-template.md`](design-template.md) and its rules are §7.

# 1. General Etiquette

To maintain a clean, readable and highly maintainable codebase, all
documentation must adhere to the following rules:

* **Module-level Documentation**: Module-level documentation must be written
  using inner doc comments (`//!` or `/*!`) placed at the top of the file.
  Do not use outer doc comments (`///` or `/**`) placed directly above module
  declarations (for example, `pub mod my_module;`). Placing module-level docs at the
  top of the file keeps the module definition clean and ensures the rustdoc
  output is correctly associated with the module content.
* **Language, Tone and Tense**: Write in clear, concise and professional
  English. Avoid marketing jargon, colloquialisms and excessive formalities.
  Focus on providing precise technical information. Always use present tense,
  the second-person imperative for procedural steps and the objective
  third-person for concepts.
* **Formatting and Code Blocks**:
    * All code examples in documentation should compile and pass formatting
      checks.
    * Prefer using the question mark operator `?` for propagating errors in
      examples instead of panic-inducing methods like `.unwrap()` or
      `.expect()` (unless the example is explicitly demonstrating a panic
      scenario).

# 2. Crate level documentation

Every crate must provide comprehensive crate-level documentation at the root of
the crate. This documentation serves as the primary technical overview and
safety manual for the crate.

* **Description**: A clear and concise description of the crate's purpose and
  its role within the control system.
* **Core Concepts**: An explanation of the fundamental principles and algorithms
  implemented in the crate.
* **Usage**: A practical example demonstrating the crate's primary use case.
  This example must be runnable with cargo test.
* **Features**: A list and explanation of all cargo features, especially those
  that alter functionality or introduce
  dependencies.
* **Limitations**: A clear statement of any known limitations, assumptions or
  operational constraints.

# 3. Public API Documentation

All public items (modules, structs, enums, functions, traits and macros) must
be thoroughly documented using ///. The
documentation for each item must follow a consistent and explicit structure.

## 3.1. General Structure

The documentation for every public item should adhere to the following order:

* **Summary**: A brief, one-sentence description of the item's purpose.
* **Detailed Description**: A more in-depth explanation of the item's
  functionality and its intended use.
* **Generic Arguments**: Generic fields that can be passed to the function or
  type.
* **Arguments**: Arguments that a function expects.
* **Returns**: For functions, a clear description of the return value and its
  meaning in all conditions.
* **Errors**: A clear explanation of all possible Err variants returned by the
  item.
* **Safety**: A mandatory section detailing all safety-related aspects.
* **Panics**: An explicit list of all conditions under which the item will
  panic.
* **Example**: At least one runnable example that demonstrates typical usage.

### Example of function/trait docs

```rust
/// Finds the largest index of a non-zero value in a slice.
///
/// This function iterates through a slice in reverse and checks if each value 
/// `.is_zero()`. It will return after the first time the condition is
/// **false**.
///
/// The logical correctness of the result is dependent on the correctness of
/// the `.is_zero()` implementation for type `T`.
///
/// # Generic Arguments
/// * `T` - field type of the array, which must implement [Zero].
///
/// # Arguments
/// * `coefficients` - a slice of `T`.
///
/// # Returns
/// * `Option<usize>`
///     * `Some(index)` - The largest index containing a non-zero value.
///     * `None` - If the slice is empty or all elements are zero.
///
/// # Panics
/// This function does not panic.
///
/// # Safety
/// This function does not use `unsafe` code.
/// # Example
/// ```
/// use control_rs::polynomial::utils::largest_nonzero_index;
/// assert_eq!(largest_nonzero_index::<u8>(&[]), None);
/// assert_eq!(largest_nonzero_index(&[0, 1]), Some(1));
/// assert_eq!(largest_nonzero_index(&[1, 0]), Some(0));
/// assert_eq!(largest_nonzero_index(&[0, 0]), None);
/// assert_eq!(largest_nonzero_index(&[1]), Some(0));
/// ```
```

### Example of implementation docs

There will be many implementations of each trait/function, so don't repeat the
doc strings in every impl. Instead, it is best to leave a comment with keywords
the user can search the docs for.

```rust
////////////////////////////////////////////////////////////////////////////////

/// Implementation of A for type.
impl A for MyType {}

////////////////////////////////////////////////////////////////////////////////

/// Implementation of B for type.
impl B for MyType {}
```

## 3.2 The #Safety Section: A Contract for Critical Code

For any function, method or unsafe block that has safety implications, a
dedicated `#safety` section is mandatory. This section must explicitly detail
the contract the caller must uphold to ensure safe execution.

The `#safety` documentation must state any applicable items from the
list below:

* Pre-conditions: The conditions that must be true before calling the function.
  This includes, but is not limited to,
  the state of hardware, the validity of inputs and the expected configuration
  of the system.
* Post-conditions: The state of the system after the function has executed
  successfully. This describes the expected
  outputs and any side effects.
* Invariants: The properties that are guaranteed to remain unchanged during the
  execution of the function.
* Assumptions: Any assumptions made about the environment or other parts of the
  system.

### Example of #[safety] documentation:

```rust
/// # Safety
///
/// This function directly manipulates hardware registers and must be used with
/// extreme care. The caller MUST ensure the following conditions are met:
/// - The system clock for the peripheral has been enabled.
/// - The provided `base_address` is a valid memory-mapped address for the UART 
///   peripheral.
/// - No other part of the system is concurrently accessing this UART peripheral.
///
/// Failure to adhere to these conditions will result in undefined behavior.
```

# 4. ETS Test Suite Documentation

Embedded Test Server (ETS) test suites are target-side verification suites meant
to run on embedded hardware or emulators. They are internal test utilities and
not part of the public-facing API. Therefore, they are subject to a much more
lightweight and concise documentation standard.

The goal is to provide the minimum information needed to understand the test
logic and the settings it uses.

## 4.1. Structuring ETS Docs

* **Omit Formalities**: Do not include `# Detailed Summary`, `# Generic Args`,
  `# Returns` or `# Example`
  sections in ETS documentation.
* **Suite-Level Docs**: Provide a brief one-sentence or two-sentence description
  using outer doc comments (`///`) directly above the `#[ets_suite]` macro or
  the module definition (since these are typically defined within a main runner
  file or test module).
* **Setting Docs**: For each settings static, write a brief, single-sentence
  description of what the setting controls and its units or limits.
* **Test Functions**: Use a single line/sentence summary description of what
  behavior or condition the test validates.

### Example of ETS suite docs

```rust
/// PID controller on-target ETS test suite.
#[ets_suite]
pub mod teensy_pid_suite {
    use control_rs_ets::settings::{Setting, SettingValue};

    /// Proportional gain setting (dimensionless, scaled by 1000). Controls system error response.
    pub static PROPORTIONAL_GAIN: u32 = 1500;
    /// Integral gain setting (dimensionless, scaled by 1000). Minimizes steady-state tracking error.
    pub static INTEGRAL_GAIN: u32 = 400;

    /// Verifies that Proportional gain strictly exceeds Integral gain.
    fn test_gain_inequality() {
        // ...
    }

    /// Verifies the system response settles within the target error bound under step input.
    fn test_step_response() {
        // ...
    }
}
```

# 5. Examples and Testing

* All examples must be runnable via cargo test.
* Examples should use the ? operator for error handling and avoid unwrap(),
  expect() or panic!() unless demonstrating
  a panic condition.

# 6. Architecture Decision Records

An Architecture Decision Record (ADR) records one decision at the workspace,
crate or project level: a dependency or toolchain policy, a workspace
convention, the default scalar type, a crate boundary, a format or protocol
shared by more than one crate. A choice that stays inside one component is
recorded in that component's design document: §5 Alternatives holds the
tradeoff and §10 Revision History holds the change.

An ADR is not a stage of the design process. It is written when a
workspace-level decision arises, and design documents cite it by number once
it is accepted. The decision to adopt ADRs is itself recorded in
[ADR-0001](adr/0001-adopt-adrs.md).

## 6.1. Location and Naming

* ADRs live in [`documentation/adr/`](adr/README.md), one file per decision,
  named `NNNN-<slug>.md`: a four-digit sequence number and a lowercase,
  hyphenated slug (`0001-adopt-adrs.md`).
* Numbers are assigned in order and never reused. A superseded ADR keeps its
  number and its file.
* The title line is `# ADR-NNNN: <Title>`. The title names the decision in
  the imperative or as a noun phrase: "Adopt ADRs for workspace decisions",
  "`f64` as the default continuous-state scalar".
* [`documentation/adr/README.md`](adr/README.md) lists every ADR with its
  status. Add a row when creating an ADR and update the row on each status
  change.

## 6.2. Structure

Copy [`adr-template.md`](adr-template.md). It follows the Nygard layout and
carries these sections in this order:

* **Badges**: date, status and author, in the same form as a design document.
* **Context**: the forces at play as facts about the workspace, ending in the
  question the ADR answers. One or two paragraphs.
* **Decision**: the chosen option in one or two sentences, justified against
  the forces in Context.
* **Consequences**: `Good`, `Bad` and `Follow-up` bullets. Each follow-up
  names the document, gate or roadmap entry that owns it.
* **Rejected Options**: one line per serious alternative with the reason it
  lost. Omit when only one option was serious.
* **References**: IEEE style, as in a design document.

## 6.3. Status and Lifecycle

| Status       | Badge markdown                                                                    | Set by                                 |
|:-------------|:----------------------------------------------------------------------------------|:---------------------------------------|
| `Proposed`   | `![Status Badge](https://img.shields.io/badge/ADR%20Status-Proposed-orange)`       | Author, on creation                    |
| `Accepted`   | `![Status Badge](https://img.shields.io/badge/ADR%20Status-Accepted-brightgreen)`  | Maintainer, after review               |
| `Deprecated` | `![Status Badge](https://img.shields.io/badge/ADR%20Status-Deprecated-lightgrey)`  | Maintainer, when the decision no longer applies |
| `Superseded` | `![Status Badge](https://img.shields.io/badge/ADR%20Status-Superseded-red)`        | Maintainer, when a newer ADR replaces it |

* A `Proposed` ADR is edited in place through review.
* An `Accepted` ADR is immutable apart from its status badge and one line
  under the badges, `Superseded by [ADR-NNNN](NNNN-<slug>.md).` To change the
  decision, write a new ADR that names the one it supersedes in its Context.
* Tooling and automation never set `Accepted`, `Deprecated` or `Superseded`.

## 6.4. Writing Rules

* One decision per ADR. A proposal that makes two independent choices is two
  ADRs.
* Keep the record to one page. Detail that belongs to one component goes in
  that component's design document.
* Write in the present tense and record the current decision only. The pull
  request holds the history of the draft.
* Design documents link to ADRs; ADRs do not list the documents that cite
  them. A design document cites `ADR-NNNN` in §5 Alternatives and in the
  Revision History row that starts following it, and a search for the number
  finds every citing document.
* An ADR records a decision; it asserts nothing. Any requirement the decision
  creates is written into the design document it constrains, where the
  `trace-reqs` gate checks it has a verification condition
  ([`requirement-traceability-design.md`](vv/requirement-traceability-design.md)).

# 7. Design Documents

The skeleton is [`design-template.md`](design-template.md). It carries no
guidance, so what it contains is only structure. The rules below govern the
content and are for authors and reviewers; they are not part of the
skeleton. The process is in [`CONTRIBUTING.md`](../CONTRIBUTING.md) §3.1.

## 7.1. Header and sections

Set the date badge to today (`September_5,_2026`, day not zero-padded) and
the author badge to a GitHub handle. The status stays `Draft`; only a
maintainer sets `Reviewed` or `Approved` (`CONTRIBUTING.md` §4). Add the
document to its project table in `documentation/README.md`.

Each section has one job:

- **§1** names 2 to 5 jobs a user can succeed or fail at. Every
  requirement in §2 traces to a scenario or an inherited constraint; one
  that traces to neither is architecture and belongs in §4.
- **§4** describes the implementation. Keep Error Handling and Dependencies
  as the last two subsections, omitting Error Handling when the component
  cannot fail and Dependencies when no third-party crate is added. Base each
  dependency row on `cargo tree` (`CONTRIBUTING.md` §5).
- **§5** gives one row per rejected option, stating the reason in terms of
  §2 where possible ("violates C-2").
- **§6.3** names requirements only review can check and configurations the
  plan does not exercise. Planned work belongs in §9.
- **§7** explains how the design meets each bound. A bound that must hold is
  an NFR or C in §2. `no_std` and numeric-type support are decided here, per
  module.
- **§8** names the requirement or section each question affects. A question
  that blocks implementation is resolved before approval.
- **§9** has 3 to 5 phases, not a task list. Every requirement lands in at
  least one phase. Effort is in engineer-days and status is `Planned`,
  `In progress` or `Complete`.
- **§10** has one row per revision, oldest first. Bump the major number for
  a change that needs re-approval and the minor number otherwise. Name every
  requirement ID the revision adds, changes or withdraws.
- **References** are IEEE style, numbered in order of first citation, and
  published works only.

## 7.2. Requirements

The requirement tracer (`vv/requirement-traceability-design.md` §4.1) parses
§2, so the shape of each bullet matters:

- One claim per bullet, written exactly as `- **FR-n — Name**: Text` with an
  em dash. No nested bullets, no suffixed IDs (`NFR-1a`) and no ID families
  other than `FR`, `NFR` and `C`.
- Name the requirement after the need, not the solution.
- IDs are permanent. Append new IDs and never renumber or reuse one. To
  retire an ID, keep the bullet and replace its text:
  `- **FR-4 — Strided reinterpretation (withdrawn)**: Withdrawn in revision 1.3; replaced by FR-9.`
- An inherited bound is one `C-n` that points at its owner:
  `- **C-1 — Inherited numeric types**: Conforms to num-types-design.md C-1.`
- Cite another document's requirement as `<slug>-design.md` FR-n, in code
  format.
- A bound is either an NFR or a C, never both.

## 7.3. Verification

Decompose each requirement into verification conditions (`VC-x.y`), one row
each. `Method` is `libtest`, `kani`, `pytest`, `gtest`, `analysis`,
`inspection` or `review`. A condition with an automated method lists in
`Target` the fully qualified test or harness that discharges it, as the
result log prints it (`crate::module::tests::test_name`,
`crate::module::proofs::harness_name`, `path/to/test_file.py::test_name` or
`TestSuite.TestCase`). Other methods put `—` in `Target`. Conditions may
share a target, and one condition may name more than one target, each in its own
code span. `Criterion` is one declarative sentence with the exact pass
bound, and names no other condition ID or method.

For `kani` conditions the author discharges four obligations:

- **O-1**: the Criterion decision is asserted as
  `assert!(decision == f(c_1, ..., c_n))` or encoded in `#[kani::ensures]`.
- **O-2**: assumptions (`kani::assume` or `#[kani::requires]`) define the
  domain, and review compares them against the requirements.
- **O-3**: `kani::cover!(c_i)` and `kani::cover!(!c_i)` witness each
  predicate value and independence pair, so the proof is not vacuous.
- **O-4**: the loop unwinding bound and dimension limits are stated, with
  unwinding assertions enabled.

MC/DC is evaluated over the input premise predicates of a condition. For
`libtest` the named tests exercise the premise combinations, and for `kani`
the `cover!` witnesses establish them. The `coverage` gate measures line
coverage and enforces no threshold, so a reviewer checks the target stated
in §6.1.

## 7.4. Acceptance

Omit §6.2 when the component makes no quantitative claim. Choose the
strongest oracle that applies:

1. Closed-form or manufactured solution.
2. Independent reference implementation (`cross-check`).
3. Metamorphic relation between input and output changes.
4. Invariant or algebraic property over generated inputs.
5. Recorded golden value, with the tool, version and inputs that produced it.

The measure is exact equality for integer, fixed-point and structural
results, and absolute, relative, ULP or scaled-residual error for floating
point. Justify each bound from conditioning or a cited error analysis, never
from what the current implementation achieves. A `cross-check` row cites its
tolerance entry (`<case>/<signal>` in the suite's tolerance table) instead of
restating the number, and a `bench` row states its budget.

# Resources

* [Meet safe and unsafe](https://doc.rust-lang.org/nomicon/meet-safe-and-unsafe.html)
* [unsafe keyword](https://doc.rust-lang.org/std/keyword.unsafe.html)
* [Unsafe Rust](https://doc.rust-lang.org/book/ch20-01-unsafe-rust.html)
* [Awesome safety critical](https://awesome-safety-critical.readthedocs.io/en/latest/)
* [Rustdoc book](https://doc.rust-lang.org/rustdoc/how-to-write-documentation.html)
* [Documenting Architecture Decisions (Nygard)](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
* [MADR: Markdown Any Decision Records](https://adr.github.io/madr/)