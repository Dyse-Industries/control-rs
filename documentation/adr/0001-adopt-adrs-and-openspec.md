# ADR-0001: Adopt ADRs and OpenSpec for design decisions

![Date Badge](https://img.shields.io/badge/Date-October_1,_2026-blue)
![Status Badge](https://img.shields.io/badge/ADR%20Status-Proposed-orange)
![Author Badge](https://img.shields.io/badge/Author-@mxscott-blueviolet)

---

## Context and Problem Statement

`control-rs` is developed design-first ([`CONTRIBUTING.md`](../../CONTRIBUTING.md)
§1). Every component has a design document under `documentation/` that
states its requirements (`FR-n`, `NFR-n`, `C-n`), its architecture and the
verification conditions that discharge each requirement. The `trace-reqs` and
`trace-check` binaries in `control-rs-ci` parse those documents
([`requirement-traceability-design.md`](../vv/requirement-traceability-design.md))
and the `trace-reqs` gate fails the build when a requirement has no
verification condition.

Two kinds of information fit poorly into that record. The first is a decision
whose reach is wider than one component: a dependency policy, a toolchain
bound, a numeric default, a wire-format convention, a process rule. Today such
a decision is written into §5 Alternatives of whichever design document first
met it, or into a pull request discussion, and the next contributor who meets
the same question has no index to find it. The second is the plan for a
change: why the change is wanted now, which behaviors it adds, modifies or
removes, and the ordered work that delivers it. Today that plan lives in the
pull request description and in [`roadmap.md`](../roadmap.md), and the
description leaves the repository when the branch is squash-merged.

Both gaps also affect assistant-driven development (`CONTRIBUTING.md` §8): an
assistant can read a design document, but it cannot recover a decision that
was never written down or a plan that lived in a merged pull request.

How does the workspace record cross-cutting decisions and change plans so that
both are discoverable in the repository, reviewable in a pull request and
usable by people and assistants alike, without disturbing the requirement
traceability pipeline?

---

## Decision Drivers

- A decision is findable without reading every design document or pull request.
- A change is reviewable from one place: intent, behavior deltas and plan.
- The requirement traceability pipeline (`trace-reqs`, `trace-check`) keeps
  parsing the design documents unchanged.
- Plain text is the source of truth; reading or writing a record needs no
  tool (`requirement-traceability-design.md`, principle 1).
- Assistants consume the same artifacts as people, in the same files.
- Small changes carry little ceremony.
- The formats follow a published convention rather than a workspace invention,
  so that contributors arrive knowing them.

---

## Considered Options

- Design documents only, with a `Decisions` appendix added to each.
- Architecture Decision Records (ADRs) in `documentation/adr/` and OpenSpec
  changes in `openspec/`, with design documents retained as the requirement
  and verification record.
- ADRs only, with no change-planning artifact.
- OpenSpec specs replacing design documents outright.

---

## Decision Outcome

Chosen option: "ADRs and OpenSpec changes, with design documents retained",
because it gives each missing kind of information a home with a published
format, and it leaves the artifact that the trace gate depends on untouched.

The three artifacts divide the work as follows.

| Artifact | Answers | Location | Lifecycle |
|:--|:--|:--|:--|
| ADR | Why a cross-cutting decision came out as it did | `documentation/adr/NNNN-<slug>.md` | `Proposed` → `Accepted`; later `Deprecated` or `Superseded`. Immutable once accepted. |
| OpenSpec change | What behavior changes and the plan that delivers it | `openspec/changes/<change>/` | Created → artifacts written → tasks applied → archived into `openspec/specs/` |
| Design document | How a component works and how each requirement is verified | `documentation/<project>/<slug>-design.md` | `Draft` → `Reviewed` → `Approved`, as before |

ADRs follow the MADR layout [2]. OpenSpec changes follow the `spec-driven`
schema [3]: `proposal.md`, delta specs under `specs/`, `design.md` and
`tasks.md`. The format rules are in [`doc-standards.md`](../doc-standards.md)
§6 and §7; the process is in `CONTRIBUTING.md` §1 and §3.

### Consequences

- Good: a decision has one record with a number, a status and an index row,
  and a later ADR supersedes it instead of editing history.
- Good: a change is a directory that a reviewer reads top to bottom, and the
  archive under `openspec/changes/archive/` keeps the plan after merge.
- Good: `openspec/specs/` becomes a readable statement of current behavior,
  which design documents do not provide across components.
- Good: both formats have CLI validation (`openspec validate`) or a published
  template, so review is about content rather than shape.
- Bad: requirement text appears in two places, a design document's §2 and the
  matching `openspec/specs/` file, until the trace tooling reads one of them.
  Each delta-spec requirement cites its design-document ID
  (`doc-standards.md` §7.3) to keep the two aligned.
- Bad: the OpenSpec CLI needs Node.js, a runtime the workspace did not depend
  on before. The CLI is a convenience for validation and archiving; every
  artifact is Markdown that is written and read without it.
- Bad: contributors learn a third artifact type.

Follow-up work this decision creates:

1. Run `openspec init --tools none` at the workspace root and commit the
   `openspec/` directory, in the pull request that opens the first change.
2. Add `openspec` to the `vale` gate arguments in `.cargo/gate.toml` and to
   the path filters in `.github/workflows/CI.yml` once the directory exists.
3. Decide, in the requirement traceability redesign (`roadmap.md` PR3-8),
   whether `openspec/specs/` becomes the tracer's requirement source and the
   design document's §2 is retired.

---

## Pros and Cons of the Options

### Design documents only, with a `Decisions` appendix

- Good: no new artifact type and no new tool.
- Bad: a cross-cutting decision still has no single home; it is appended to
  whichever document met it first, and nothing indexes it.
- Bad: the change plan still leaves the repository at squash-merge.

### ADRs and OpenSpec changes, with design documents retained

- Good: each kind of information has one home with a published format.
- Good: the trace gate and its input documents are unchanged.
- Neutral: the design document's `Development Plan` (§9) overlaps with
  `tasks.md`. The design document keeps the phase-level plan; `tasks.md`
  carries the task-level checklist for one change.
- Bad: requirement text is duplicated between a design document and
  `openspec/specs/` until the trace redesign resolves which is normative.

### ADRs only

- Good: solves the decision log with one new artifact and no tooling.
- Bad: the change plan and behavior deltas still have no durable home.

### OpenSpec specs replacing design documents

- Good: one requirement record, no duplication.
- Bad: `trace-reqs` parses the `- **FR-n — Name**:` and `| VC-x.y |` shapes
  of a design document, not the `### Requirement:` and `#### Scenario:`
  shapes of an OpenSpec spec; the gate would need rewriting before any
  document moved, which is the scope of `roadmap.md` PR3-8, not of this
  record.
- Bad: an OpenSpec spec has no section for architecture, performance budgets
  or acceptance oracles, which the design template carries in §4, §6.2 and
  §7.

---

## References

[1] M. Nygard, "Documenting Architecture Decisions," *Cognitect Blog*, Nov. 2011. [Online]. Available: https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions. Accessed: Oct. 1, 2026.

[2] MADR project, "Markdown Any Decision Records," *adr.github.io*. [Online]. Available: https://adr.github.io/madr/. Accessed: Oct. 1, 2026.

[3] Fission-AI, "OpenSpec Documentation," *openspec.dev*. [Online]. Available: https://openspec.dev/docs/overview. Accessed: Oct. 1, 2026.

[4] `control-rs`, "Requirement Traceability Infrastructure," `documentation/vv/requirement-traceability-design.md`, rev. 1.16, Sept. 2026.
