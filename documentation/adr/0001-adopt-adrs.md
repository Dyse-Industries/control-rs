# ADR-0001: Adopt ADRs for workspace decisions

![Date Badge](https://img.shields.io/badge/Date-October_2,_2026-blue)
![Status Badge](https://img.shields.io/badge/ADR%20Status-Approved-green)
![Author Badge](https://img.shields.io/badge/Author-@mxscott-blueviolet)

---

## Context

`control-rs` is developed design-first ([
`CONTRIBUTING.md`](../../CONTRIBUTING.md)
§1). Each component's design document states its requirements, architecture
and verification conditions, and the `trace-reqs` and `trace-check` gates in
`control-rs-ci` check that every requirement is discharged by a passing test
or proof harness [1]. Within a component, §5 Alternatives records each choice
and §10 Revision History records each change.

A decision at the workspace, crate or project level has no such home. A
dependency policy, a toolchain bound, a workspace convention or a format
shared between crates is written into whichever design document first met
it, or into a pull request, and the next contributor who meets the question
has no index to find it. How does the workspace record these decisions so
that they are findable and citable, without adding a second source of
requirements beside the design documents?

---

## Decision

Record each workspace, crate or project-level decision as an ADR in
`documentation/adr/`, in the layout of Nygard [2] (`doc-standards.md` §6).
Design documents remain the only source of requirements and the trace gates
the only verification record: an ADR asserts nothing, and a requirement it
creates is written into the design document it constrains. An ADR is not a
stage of the design process; it is written when such a decision arises, and
design documents cite it by number.

---

## Consequences

- Good: a workspace decision has one numbered, indexed record, and a later
  ADR supersedes it instead of editing it.
- Good: design documents cite the ADR in §5 Alternatives instead of arguing
  the choice again.
- Good: the trace gates, their configuration and their inputs are unchanged.
- Good: the file count grows per decision, not per change.
- Bad: contributors learn a second document type and its status lifecycle.
- Follow-up: checking that each cited `ADR-NNNN` exists and has a compatible
  status, owned by the requirement traceability redesign (`roadmap.md`
  PR3-8).

---

## Rejected Options

- Design documents only: a workspace decision is appended to whichever
  document met it first, and nothing indexes it.
- A per-change planning artifact set (OpenSpec [3]: proposal, delta specs,
  design and tasks per change): four or more files per change, a second copy
  of each requirement and no link from a scenario to a test result, so it
  adds no check the trace gates do not already perform.
- The MADR layout [4]: its drivers, options and pros-and-cons sections state
  one decision three times; a one-page record does not need them.

---

## References

[1] `control-rs`, "Requirement Traceability Infrastructure,"
`documentation/vv/requirement-traceability-design.md`, rev. 1.16, Sept. 2026.

[2] M. Nygard, "Documenting Architecture Decisions," *Cognitect Blog*, Nov.
2011. [Online].
Available: https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions.
Accessed: Oct. 1, 2026.

[3] Fission-AI, "OpenSpec Documentation," *openspec.dev*. [Online].
Available: https://openspec.dev/docs/overview. Accessed: Oct. 1, 2026.

[4] MADR project, "Markdown Any Decision Records," *adr.github.io*. [Online].
Available: https://adr.github.io/madr/. Accessed: Oct. 1, 2026.
