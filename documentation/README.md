# `control-rs` documentation

![Date Badge](https://img.shields.io/badge/Date-September_9,_2026-blue)
![Type Badge](https://img.shields.io/badge/Type-Index-lightgrey)

Each subdirectory is a project. A project holds its design documents and a
`research/` directory containing one `<slug>.json` + `<slug>.bib` evidence pair
per design document.

## Standards

| Document | Scope |
|:---|:---|
| `design-template.md` | Authoritative design document structure |
| `doc-standards.md` | Documentation policy for docs and code comments |
| `development-guide.md` | Workspace architecture, cargo aliases, CI workflows |

## Projects

Library and toolbox projects are grouped by subject area:

| Project | Subject |
|:---|:---|
| `math/` | Numeric types, traits, storage and subprograms |
| `numerical-models/` | Matrix, polynomial, tensor, state-space, transfer function |
| `control-toolboxes/` | Classical, modern, robust control and system identification |
| `simulation/` | Integrators |
| `vv/` | Verification and validation tooling |

Test and tooling projects are grouped per crate, because each publishes
separately. Project and slug names drop the `control-rs-` prefix; the Crate
column gives the package each one specifies:

| Project | Crate | Type |
|:---|:---|:---|
| `ets/` | `control-rs-ets` | Target-side Embedded Test Server (`no_std`) |
| `macros/` | `control-rs-macros` | Procedural macros for test discovery |
| `ets-host/` | `control-rs-ets-host` | Host transport, framing and headless runner (lib) |
| `tui/` | `control-rs-tui` | Interactive terminal console (bin) |
| `ci/` | `control-rs-ci` | Repository quality-gate runner (bin) |

`control-rs-xtask` is retired. Its host surfaces were split into
`control-rs-ets-host`, `control-rs-tui` and `control-rs-ci`; the projects
above are where their designs now live.

## Conventions

- Design documents are named `<slug>-design.md`. Project and slug names do
  not carry the `control-rs-` prefix.
- Every design document carries Date, Status and Author badges. `Draft`,
  `Reviewed` and `Approved` are the only status values; `Reviewed` and
  `Approved` are set by a human.
- Non-design documents (overviews, plans, indexes) carry a `Type` badge instead
  of a status badge, so they are not mistaken for gated artifacts.
- Research evidence is quotes-only. Bibliographic fields live in the paired
  `.bib`, joined by `bib_key`.
