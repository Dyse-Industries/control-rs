# Type/Module Name (Design Document)

![Date Badge](https://img.shields.io/badge/Date-July_12,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@AuthorName-blueviolet)

---

### 1. Introduction

Brief description of the: motivation, scope, goals and usage scenarios.

---

### 2. Requirements

Derive requirements from §1. A requirement is a testable
user need, not a specific feature. Publish only the
finished IDs; keep derivation scratch off the page.

#### Classify

- **FR** — observable behavior the module must provide.
- **NFR** — a quality of that behavior (cost, size, determinism).
- **C** — a bound, inherited decision, or explicit out-of-scope.

The same claim must not appear as both NFR and C. Use subsections 2.1
Functional Requirements, 2.2 Non-Functional Requirements, and 2.3
Constraints.

#### Derive

Scratch, then publish IDs only:

1. Restate each §1 scenario as a job.
2. List needs as one-liners and classify each as FR, NFR, or C.
3. Split any one-liner that contains two independently testable claims.
4. Name the ID from the need, not the solution (`Exact product rescale`,
   not `Widening multiply helper`).
5. Body: 1–3 sentences. First states the need. Second bounds it (what is
   not required, or what would go wrong). Cite research. Do not name types,
   methods, or files unless the need *is* that contract.
6. Order IDs as you would explain them: core capability, then consequences,
   then how this module participates in the rest of the crate. Do not group
   by subsystem.
7. Every ID traces to a §1 scenario or a standing crate constraint
   (`CLAUDE.md`, toolbox C-1..C-4, an Approved sibling design). If it does
   not, it is architecture — move it to §4.

#### Shape

```
- **FR-n — Short name**: One to three sentences. First states the need.
  Second bounds it.
```

Same form for `NFR-n` and `C-n`. Typical size: 4–8 FRs, 1–3 NFRs, 2–5 Cs.

**Accept** — one need, named from the need, next ID is a consequence of the
last:

- **FR-4 — Exact product rescale**: Multiplication forms its product in a
  representation wide enough to hold it before rescaling. A same-width
  multiply would discard the high half of every product.

**Reject** — architecture slogan as the title, nested inventory, several
claims glued with "and":

- **FR-1 — Decoupled Storage Subsystems Architecture**: Provide distinct,
  zero-cost storage subsystem contracts across dense strided arrays
  (`DenseStorage<T>` / `Storage<T, R, C>`), packed structured matrices
  (`PackedStorage<T>`), and compressed sparse backends (`SparseStorage<T>`).

Also reject: nested trait/kernel sub-bullets; an FR that is only true after
choosing the §4 design; the same cap restated as both NFR and C. Nested
sub-bullets mean the item is an architecture dump — split or move.

---

### 3. Technical Overview

A brief explanation of the project's scope and the expertise it will require.

---

### 4. Architecture

Describe the implementation in detail, start from a broad scope then focus in
on the specifics.

---

### 5. Alternatives

Describe any architecture/implementation details that were considered but
not chosen.

---

### 6. Verification & Validation

1. Define the required steps to verify an implementation (unit tests,
   integration tests, ETS tests, CI tests).

2. Define the required steps to validate an implementation (examples, external
   projects, user demos).

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