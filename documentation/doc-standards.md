# Documentation Standards for `control-rs`

These rules establish a rigorous and consistent documentation standard for a
Rust native control systems toolbox intended for safety-critical applications.

§1 to §5 govern rustdoc and ETS documentation in source. §6 governs
Architecture Decision Records (ADRs), the record of decisions at the
workspace, crate or project level. The process that produces an ADR is in
[`CONTRIBUTING.md`](../CONTRIBUTING.md); the design-document format is in
[`design-template.md`](design-template.md).

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

# Resources

* [Meet safe and unsafe](https://doc.rust-lang.org/nomicon/meet-safe-and-unsafe.html)
* [unsafe keyword](https://doc.rust-lang.org/std/keyword.unsafe.html)
* [Unsafe Rust](https://doc.rust-lang.org/book/ch20-01-unsafe-rust.html)
* [Awesome safety critical](https://awesome-safety-critical.readthedocs.io/en/latest/)
* [Rustdoc book](https://doc.rust-lang.org/rustdoc/how-to-write-documentation.html)
* [Documenting Architecture Decisions (Nygard)](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
* [MADR: Markdown Any Decision Records](https://adr.github.io/madr/)