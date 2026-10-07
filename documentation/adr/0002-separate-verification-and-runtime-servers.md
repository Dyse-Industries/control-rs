# ADR-0002: Separate the verification server from the runtime server

![Date Badge](https://img.shields.io/badge/Date-October_7,_2026-blue)
![Status Badge](https://img.shields.io/badge/ADR%20Status-Proposed-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

## Context

`control-rs-ets` is the workspace's on-target and emulator verification
server: it discovers suites, runs cases with cycle and stack profiling, and
reports panics, across four QEMU targets and on hardware [1]. It depends on
nothing in `control-rs`. Hardware-in-the-loop testing needs continuous
execution with interrupts live, which the loop suite adds to ETS [2].

Layer 2 products such as the proposed `control-rs-esc` motor driver also need
a production runtime: lifecycle transitions with per-transition results,
goals with feedback, results and cancellation, parameters and link
supervision. These are production concerns rather than test concerns, and
they evolve per product. Layer 1 must not depend on Layer 2. Where does the
runtime live, and how does it relate to ETS?

---

## Decision

Keep two independent servers. `control-rs-ets`, extended with loop suites,
remains the Layer 1 verification server and carries no lifecycle or goal
semantics [2]. A lifecycle and goal runtime server lives in Layer 2 and may
depend on Layer 1, never the reverse. The two meet only in Layer 2 product
crates, whose HIL tests are ETS loop suites that drive the product's
lifecycle node.

---

## Consequences

- Good: ETS profiling and its existing suites are unchanged, and the
  dependency direction holds.
- Good: flight firmware links no test infrastructure, and HIL runs exercise
  the production node's logic with interrupts live.
- Good: the Layer 2 runtime evolves per product without Layer 1 releases.
- Bad: parameters, discovery, framing and host libraries exist twice, once
  per server.
- Bad: loop-suite states and runtime goal and lifecycle results are two
  vocabularies, so each product maps one to the other.
- Bad: HIL binaries use the ETS entrypoint, panic handler and link, so the
  flight panic path and the production link are not exercised by HIL.
- Follow-up: research and design of the runtime server, including whether it
  reuses `control-rs-ets::comms` framing, owned by the Layer 2 workspace.
- Follow-up: the state-to-result mapping, owned by each product design,
  starting with `control-rs-esc`.
- Follow-up: verification of the flight panic path and production link,
  owned by the Layer 2 runtime design.

---

## Rejected Options

- One general server in Layer 1 with ETS rebuilt on it: removes duplication
  but rebuilds ETS and puts a production runtime in the numerical crate.
- One server in Layer 2 with ETS on it: Layer 1 on-target verification would
  depend on Layer 2.
- ETS extended into the production runtime: flight firmware would carry the
  settings registry, black-box panic path and reset wait.
- A host-owned HIL loop over atomic ETS cases: each step costs a host round
  trip and runs with interrupts masked.
- Two servers sharing a transport crate extracted from ETS: deferred to the
  Layer 2 runtime design, taken only if framing duplication proves costly.

---

## References

[1] `control-rs`, "Embedded Test Server,"
`documentation/ets/embedded-test-server-design.md`, Sept. 2026.

[2] `control-rs`, "Loop Suites," `documentation/ets/loop-suite-design.md`,
rev. 1.0, Oct. 2026.
