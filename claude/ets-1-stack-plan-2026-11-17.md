# ETS-1 stack plan

Date: 2026-11-17

Scope is the target runtime that the roadmap calls ETS-1: suites, macros, the profiler, `HostComms` framing, and the lead document `embedded-test-server`. The lead's `link` conditions name requirements in `ets-executive`, and a design enters `.cargo/trace.toml` as a whole, so this track also completes `ets-executive`. Host session, TUI, and board examples stay on the later track (open PRs #155 and #156).

All six documents are Draft, latest revision unsigned, and absent from `.cargo/trace.toml`. None is approved. The November 17 text exists only in local commits `c978b01` and `e4847f9` on `mdsdev/ets-1-suites-profiler-macros-comms-executor`. That branch is the draft source. It is not a merge candidate.

## Link grammar this plan uses

`requirement-traceability-design.md` FR-13 resolves a `link` target as a document name, a requirement (`<doc>#<id>`), or a condition. The condition takes the worst status of what it names. A document target takes the worst status of that document's requirements.

That is wider than a whole-document link. It does not allow a partial merge. A named requirement is Covered only when its document is in `trace.toml` and that requirement's conditions pass. Each document below is still completed by one PR. The lead's requirement-level links are legal. The graph is acyclic: only `embedded-test-server` links outward.

## Sign-off decisions

No design is signed, and no branch below is cut, until these three are settled. They change requirement text. After approval, that text cannot be reworded by a later PR.

1. **Executive contract.** The draft on this branch specifies the interactive command loop: discovery, dispatch, lifecycle, three profile arms, settings, panic capture. The research idea, and the copy of `ets-executive` on `mdsdev/ets-2-lifecycle-session-tui` (`e994acb`), specify a single-job executive (Atomic, Lockstep, FreeRunning, Periodic). That branch also adds `documentation/adr/0002-single-job-executive-for-ets-and-hil.md`. On this branch ADR-0002 is the MulAcc record, and ADR-0003 is the library/ETS split. The single-job record cannot take number 0002.
   - Command-loop: this track implements the written executive. Single-job work stays on the later track under a new ADR number.
   - Single-job: rewrite `ets-executive` and every lead link that names it before any sign-off of those two documents. Suite, macros, profiler, and host-comm still proceed only where their text does not assume the command loop. The executive and lead stacks in this plan wait on that rewrite.

2. **Parameter cell.** `test-suite` §4 specifies `TripleBuffer<T>`, with a single `UnsafeCell<T>` when the target has no byte atomics (ARMv6-M). The executive research specifies a seqlock. The crate still implements `Atomic*Setting`, and the macros still emit those wrappers. The suite PR freezes the requirement. Choose `TripleBuffer` as written, or seqlock, before signing suite, macros, executive FR-5, and lead FR-13.

3. **Emulated-target evidence.** No `virtual-ets` gate exists in `.cargo/gate.toml`. `cpu-profiler` names it from VC-1.3, VC-2.1, VC-3.1, VC-4.2, VC-7.2, VC-8.2, VC-10.1, and VC-13.1. `host-comm` names it from VC-15.1 (C-3). A PR that owns those rows cannot pass the trace gate.
   - Move the rows, before approval, to the later track that runs QEMU and boards. These PRs keep host `libtest` plus the existing `build` and `clippy` gates.
   - Add one gate stack, merged to `main` before the profiler and host-comm PRs. That stack owns the gate and a minimal emulated run. It is a seventh PR, and it sits on the critical path.

A fourth fix does not add a stack, and it blocks host-comm and lead sign-off: `host-comm` §8 says log text that contains `0xAA 0x55` is swallowed as a false frame, while lead VC-12.2 requires that same byte stream to come out as exactly the original text plus exactly the frames. Pick one contract and write it into both documents before either is signed. The test `test_frame_reader_interleaved_log_stream_and_telemetry` is written once, in the host-comm PR. The lead only cites it.

## Track

One track, ETS-1 target runtime. Six stacks, one PR each, each based on `main` after its dependencies have merged. A dependency is a rebase onto `main`, not another layer on the dependency's branch. Nothing here stacks on #154, #155, #156, or #158.

| Stack | PR | Owned doc | Depends on | Reqs | Conds | Parallel with |
|:------|:---|:----------|:-----------|-----:|------:|:--------------|
| Suite | 1 | `test-suite` | decisions 1–3 | 12 | 13 | Profiler |
| Macros | 2 | `macros` | Suite merged | 14 | 16 | Host comm, Profiler |
| Profiler | 3 | `cpu-profiler` | decision 3; the gate stack if one is added | 11 | 20 | Suite |
| Host comm | 4 | `host-comm` | Suite merged; decision 3; log contract | 13 | 20 | Macros, Profiler |
| Executive | 5 | `ets-executive` | Suite, Profiler, Host comm | 12 | 27 | Macros, if those three have merged |
| Lead | 6 | `embedded-test-server` | all five | 19 | 34 | nothing |

Macros is two requirements over the default review budget, host-comm is one, and the lead is over on both counts (19 and 34). Each stays one PR. Macros and host-comm are one algorithm family each. The lead is the integrator: 17 of its 34 conditions are links, and the tests it cites are written by earlier PRs. Splitting any of them after this plan would split a document across PRs. The default budget is tunable; the split has to happen now if one of those three should become two documents.

Commit 1 on each branch is the design document, still Draft, with that stack's decision written in. Sign-off happens on the branch. Implementation commits follow. The PR opens with both. No design-only PR.

## PR 1 — Suite

- **Depends on:** the three decisions. Base `main`.
- **Owns:** `documentation/ets/test-suite-design.md`. FR-1, FR-2, FR-3, FR-4, FR-5, FR-6, FR-7, NFR-1, NFR-2, C-1, C-2, C-3.
- **Links out:** none.
- **Implementation:** `Profile` on `ExecDescriptor`; `ConstantDescriptor` and `constants` on `SuiteDescriptor`; the chosen parameter cell in `control-rs-ets::settings`, replacing `Atomic*Setting` and `dyn Setting`. Phases 2 and 3 of this document (constants, parameter cell) are the unimplemented part. Phases marked Shipped predate that API.
- **Tests this PR adds:** `test_duplicate_constant_names_in_distinct_suites` (VC-6.1), `test_external_globals_are_not_suite_metadata` (VC-7.1).
- **Example:** a host example that builds descriptors and reads constants and settings without the macro. `control-rs-ets/examples/ets_suite.rs` stays with the macros PR, because it is a macro consumer.
- **Before sign-off:** §7 still describes `Atomic*Setting` while §4.2 specifies `TripleBuffer`. §8 still says §4 and §6.1 need that update. Make §4, §6, and §7 describe the chosen cell. Add this document to `trace.toml` in the implementation commits.
- **Gate risk:** `build` and `clippy` exist. The PR passes `cargo ci` and the trace gate on `main`.

What stalls if it does not merge: macros, host-comm, the executive, and the lead.

## PR 2 — Macros

- **Depends on:** Suite on `main`. Rebase when Suite merges. Do not open the implementation commits against an unsigned suite API.
- **Owns:** `documentation/ets/macros-design.md`. FR-1 through FR-9, NFR-1, NFR-2, NFR-3, C-1, C-2.
- **Links out:** none. Code depends on the suite types (`Profile`, `ConstantDescriptor`, the parameter cell).
- **Implementation:** `control-rs-macros` still emits `Atomic*Setting`. Generate the signed cell, `ConstantDescriptor`s, and `#[ets_profile]` (`Full` default, `Unmasked`, `Off`). Keep `#[ets_setup]`, `ets_entrypoint!`, `ets_panic!`, and `ets_exception!`.
- **Tests this PR adds:** `process_test_fn_profile_attributes` (VC-2.2), `process_const_descriptor_matches_supported_types` (VC-3.1), `duplicate_const_names_in_separate_suites_compile` (VC-8.1), `suite_inference_ignores_outer_scope_items` (VC-8.2).
- **Example:** update `control-rs-ets/examples/ets_suite.rs` to the signed API.
- **Gate risk:** `build` and `clippy` exist. Do not add clippy allows.

What stalls if it does not merge: the lead. The executive's host tests use hand-built descriptors and can merge without this PR.

## PR 3 — Profiler

- **Depends on:** decision 3. If a `virtual-ets` gate stack is added, that stack merges first. Otherwise the eight `virtual-ets` rows are moved off this document before sign-off. Base `main`. Parallel with Suite.
- **Owns:** `documentation/ets/cpu-profiler-design.md`. FR-1, FR-2, FR-3, FR-4, FR-7, FR-8, FR-9, NFR-1, NFR-2, C-1, C-2. FR-5 and FR-6 are absent. Leave the numbers.
- **Links out:** none.
- **Implementation:** phase 5 is Planned. `run_profiled` and `run_profiled_unmasked` are not in the crate. `TestMetrics` lives on `server.rs`; this PR moves it to `control-rs-ets::profiler` and returns it from both runners. Existing cycle, SysTick, and stack-paint hooks stay.
- **Tests this PR adds:** `test_run_profiled_composition` (VC-7.3), `test_run_profiled_unmasked_composition` (VC-7.4).
- **Example:** `control-rs-ets/examples/cpu_profiler.rs`, extended to call the composed runners on `HostCPUProfiler`.
- **Gate risk:** the `virtual-ets` rows, until decision 3 is applied. Stack-bounds safety (region, growth, margin, rollover, `has_clock` == false means zero cycles) has to be specific enough in §4 that the host tests and the moved gate rows have a criterion to check. Do not leave that as an open question at sign-off.

What stalls if it does not merge: the executive and the lead. Suite and macros do not.

## PR 4 — Host comm

- **Depends on:** Suite on `main` (`Profile` is a suite type and `TestInfo` carries it). Decision 3 applied the same way as the profiler. The log contract is written into FR-7 and into lead VC-12.2 before sign-off.
- **Owns:** `documentation/ets/host-comm-design.md`. FR-1, FR-2, FR-4, FR-5, FR-6, FR-7, FR-9, NFR-1, NFR-2, NFR-3, C-1, C-2, C-3. FR-3 and FR-8 are absent. Leave the numbers.
- **Links out:** none.
- **Implementation:** `PROTOCOL_VERSION` is `1` in `comms.rs`. The draft bumps it to `2` because `TestInfo` gains `profile`. Regenerate the golden vectors for version 2. Implement the chosen log contract in `FrameReader`.
- **Tests this PR adds:** `test_frame_reader_interleaved_log_stream_and_telemetry` (VC-7.1). The lead's VC-12.2 cites the same test.
- **Cross-check:** the golden vectors are the wire-contract check. No separate example.
- **Gate risk:** VC-15.1 (`virtual-ets`) until decision 3. VC-11.1 and VC-13.1 use `clippy`, which exists; their criteria talk about allocation and `no_std`, which `clippy` does not measure. Narrow those criteria to what `clippy` actually enforces, or move the allocation claim onto a `libtest` this PR runs, before sign-off.

What stalls if it does not merge: the executive and the lead.

## PR 5 — Executive

- **Depends on:** Suite, Profiler, and Host comm on `main`. Macros may still be open. This is the command-loop document. If decision 1 is the single-job contract, this PR does not start from the current draft.
- **Owns:** `documentation/ets/ets-executive-design.md`. FR-1 through FR-7, NFR-1, NFR-2, NFR-3, C-1, C-2.
- **Links out:** none. Code calls `CPUProfiler::run_profiled` / `run_profiled_unmasked`, reads suite descriptors, and writes `HostComms` frames.
- **Implementation:** the development plan marks all three phases Shipped. Revision 1.1 is not in the code: there is no `Profile` dispatch and no `test_server_profile_arms`. `server.rs` still owns `TestMetrics` until PR 3 moves it. This PR dispatches `Full` (masked, `MetricReport`), `Unmasked` (unmasked, `MetricReport`), and `Off` (bare call, no `MetricReport`), and it mutates the parameter cell from PR 1.
- **Tests this PR adds:** `test_server_profile_arms` (VC-4.4). The existing `server` and `util` tests stay here. The lead cites them; it does not retake them.
- **Example:** a host example that discovers one hand-built suite, runs one case on each profile arm, and sets one parameter.
- **Gate risk:** VC-26.1 uses `build`, which exists. The document's own phase table must stop saying Shipped for behavior revision 1.1 added.

What stalls if it does not merge: the lead. Macros does not.

## PR 6 — Lead

- **Depends on:** all five documents on `main` and in `trace.toml`. This is the one PR that completes the integrator. It does not add, split, or reword a requirement in any earlier document.
- **Owns:** `documentation/ets/embedded-test-server-design.md`. FR-1 through FR-15, NFR-1, NFR-2, C-1, C-2. Revision 2.1 mentions a C-3 that is not a live requirement. Leave it out.
- **Links out:** the 17 `link` rows in §6.1. Every target is owned by PR 1–5. No cycles.
- **Implementation:** diagnostic suite (FR-15), including the zero-cost-when-disabled build. Integration behavior the links already require stays in the earlier PRs. This PR adds the diagnostic cases and the two tests below.
- **Tests this PR adds:** `test_server_diagnostic_suite_zero_cost_when_disabled` and `test_server_diagnostic_case_tare` (VC-15.2).
- **Example:** one host run that discovers the diagnostic suite, runs `tare`, and checks the profile arm it requested.
- **Before sign-off:** VC-5.2 is method `gate`, target `build`, and the criterion says the firmware builds and executes. `build` compiles. Either narrow the criterion to compilation, or point execution at a gate this PR can run. Add this document to `trace.toml` last.
- **Also in this PR:** correct `documentation/roadmap.md` so ETS-1 names these six documents and these branches, and so ETS-2 keeps host session, TUI, and the single-job follow-up. Correct this document's phase table, which still calls host-comm ETS-2 and the executive ETS-3.
- **Gate risk:** an unresolved link if any earlier document is missing from `trace.toml`. Size: 19 requirements and 34 conditions, of which 17 are links and 14 `libtest` rows mostly cite tests from PR 3–5.

## Critical path

Decisions, then Suite, then Host comm, then Executive, then Lead.

Profiler runs beside Suite. It joins the critical path when its remaining work (composed runners, and the gate stack if decision 3 adds one) outlasts Suite plus Host comm. Macros runs beside Host comm and must merge before Lead; it is shorter than Host comm plus Executive plus Lead.

While Suite is in sign-off, Profiler is the stack that can be signed and implemented. Host comm and Macros wait on the signed suite API. Executive waits on three merges. Lead waits on all five.

If decision 3 adds a gate stack, the path becomes: decisions, gate, then the longer of Profiler and (Suite then Host comm), then Executive, then Lead.

## Requirement ownership

Each requirement belongs to the one PR that completes its document.

| PR | Requirements |
|:---|:-------------|
| 1 Suite | `test-suite` FR-1, FR-2, FR-3, FR-4, FR-5, FR-6, FR-7, NFR-1, NFR-2, C-1, C-2, C-3 |
| 2 Macros | `macros` FR-1, FR-2, FR-3, FR-4, FR-5, FR-6, FR-7, FR-8, FR-9, NFR-1, NFR-2, NFR-3, C-1, C-2 |
| 3 Profiler | `cpu-profiler` FR-1, FR-2, FR-3, FR-4, FR-7, FR-8, FR-9, NFR-1, NFR-2, C-1, C-2 |
| 4 Host comm | `host-comm` FR-1, FR-2, FR-4, FR-5, FR-6, FR-7, FR-9, NFR-1, NFR-2, NFR-3, C-1, C-2, C-3 |
| 5 Executive | `ets-executive` FR-1, FR-2, FR-3, FR-4, FR-5, FR-6, FR-7, NFR-1, NFR-2, NFR-3, C-1, C-2 |
| 6 Lead | `embedded-test-server` FR-1, FR-2, FR-3, FR-4, FR-5, FR-6, FR-7, FR-8, FR-9, FR-10, FR-11, FR-12, FR-13, FR-14, FR-15, NFR-1, NFR-2, C-1, C-2 |

## Link graph

All edges start at `embedded-test-server` and end at a requirement owned by an earlier PR.

| Lead condition | Targets |
|:---------------|:--------|
| VC-1.1 | `test-suite` FR-1, FR-2, FR-3; `macros` FR-1, FR-2; `ets-executive` FR-2; `host-comm` FR-2 |
| VC-2.1 | `test-suite` FR-2; `ets-executive` FR-1, FR-2; `host-comm` FR-2 |
| VC-3.1 | `ets-executive` FR-3, FR-6; `host-comm` FR-6 |
| VC-4.1 | `test-suite` FR-5; `macros` FR-1 |
| VC-5.1 | `ets-executive` FR-1; `host-comm` FR-1 |
| VC-6.1 | `ets-executive` FR-7; `cpu-profiler` FR-8 |
| VC-7.1 | `cpu-profiler` FR-1, FR-2; `ets-executive` FR-4 |
| VC-8.1 | `cpu-profiler` FR-3, FR-4; `ets-executive` FR-4 |
| VC-9.1 | `cpu-profiler` FR-7; `ets-executive` FR-4 |
| VC-10.1 | `ets-executive` FR-7; `host-comm` FR-5 |
| VC-11.1 | `host-comm` FR-2, NFR-3 |
| VC-12.1 | `host-comm` FR-7 |
| VC-13.1 | `test-suite` FR-4; `macros` FR-4; `ets-executive` FR-5 |
| VC-15.1 | `ets-executive` FR-2; `test-suite` FR-2; `host-comm` FR-2 |
| VC-16.1 | `host-comm` NFR-1; `ets-executive` NFR-3 |
| VC-17.1 | `test-suite` NFR-1; `macros` NFR-1 |
| VC-18.2 | `ets-executive` C-2; `test-suite` NFR-2 |

Lead conditions that are not links: VC-1.2, VC-2.2, VC-3.2, VC-4.2, VC-5.2 (`build`), VC-6.2, VC-7.2, VC-8.2, VC-9.2, VC-10.2, VC-11.2, VC-12.2, VC-13.2, VC-14.1 (`build`), VC-15.2, VC-16.1 is a link, VC-17.2, VC-18.1 (`build`). VC-5.2 is the criterion fix in PR 6.

Code edges that are not links, and still set merge order:

- Macros and Host comm compile against Suite types (`Profile`, descriptors, parameter cell).
- Executive calls Profiler runners and Host comm frames, and reads Suite descriptors.
- `TestMetrics` moves in PR 3, before PR 5.

## Doc fixes required before the owning sign-off

- Six documents are missing from `.cargo/trace.toml`. Each PR adds its own document. PR 6 adds the lead last.
- `test-suite` §7 and §8 disagree with §4. PR 1.
- Log text versus exact recovery. PR 4 and PR 6, same contract.
- `virtual-ets` rows, and host-comm VC-11.1 / VC-13.1 claiming allocation results from `clippy`. PR 3 and PR 4.
- Lead VC-5.2 says `build` executes firmware. PR 6.
- Development-plan rows marked Shipped do not match the crate: no `Profile`, no `ConstantDescriptor`, no `TripleBuffer`, `PROTOCOL_VERSION` is 1, no `run_profiled`. Each PR's design commit corrects its own rows. That is status text, not a requirement change.
- Roadmap branch names (`mdsdev/ets-1-suites-profiler`, `mdsdev/ets-3-target-server`, `mdsdev/ets-6-board-examples`) do not exist. PR 6 updates the roadmap. The live names are below.
- Do not renumber the gaps (profiler FR-5/FR-6, host-comm FR-3/FR-8).

## Where the current branches and commits land

Open stack today:

| PR | Head | Base | What it is |
|:---|:-----|:-----|:-----------|
| #158 | `mdsdev/ets-exclusion` | `main` | Library/ETS split, classical-control test move, qemu and teensy example edits. Own PR. Not part of these stacks. |
| #154 | `mdsdev/ets-1-suites-profiler-macros-comms-executor` at `7f247de` | #158 | One design commit. Design-only. Remote head is not the local branch. |
| #155 | `mdsdev/ets-2-lifecycle-session-tui` | #154 | Host session, TUI, and the single-job ADR/design. Later track. |
| #156 | `mdsdev/ets-3-board-examples` | #155 | Board examples. Later track. |

Local `mdsdev/ets-1-suites-profiler-macros-comms-executor` has diverged from #154. Origin has `7f247de`, which is not in the local history. Local has six commits origin does not, including the only copies of the November 17 designs (`c978b01`, `e4847f9`). Do not force-push this branch onto #154.

Copy design files from the local branch onto each new branch with a per-file checkout. Do not cherry-pick. Do not take these files into the ETS-1 PRs: the classical-control test move, `.github` and gate edits that belong to #158, `examples/qemu`, `examples/teensy4`, `control-rs-ets-host`, `control-rs-tui`. Those belong to #158 or to #155/#156.

Closed PRs #142–#147 and #104–#107 already have no remote heads. Nothing further to delete there.

## Branches to create

Cut from `main`, after #158 merges if both sides touch `trace.toml`. Rebase onto `main` at that merge. Do not parent them on #158.

| Branch | Stack |
|:-------|:------|
| `mdsdev/ets-suite` | PR 1 |
| `mdsdev/ets-macros` | PR 2 |
| `mdsdev/ets-profiler` | PR 3 |
| `mdsdev/ets-host-comm` | PR 4 |
| `mdsdev/ets-executive` | PR 5 |
| `mdsdev/ets-lead` | PR 6 |
| `mdsdev/ets-virtual-gate` | only if decision 3 adds the gate; merges before PR 3 and PR 4 |

Agent-opened PRs target `main` only.

## Branches and PRs to retire

After the design files have been copied off the local branch:

- Close #154. It cannot pass the gates, and its remote commit is stale.
- Delete remote `mdsdev/ets-1-suites-profiler-macros-comms-executor` once #154 is closed and the local designs have been copied. Until that copy exists, that local branch is the only copy of the November 17 text.
- Leave #155 and #156 open only as drafts of the later track. They sit on the design-only base. When PR 6 is on `main`, move their unique files (host session, TUI, board examples, single-job text) onto new branches from `main` by per-file checkout, then close #155 and #156. Do not rebase them onto PR 1–6.
- If decision 1 keeps the command loop, the `ets-executive-design.md` on #155 must not replace the approved executive document. The single-job text becomes a new document and a new ADR number on the later track.

## Check

- Every requirement above has one owning PR.
- Every link target is owned by PR 1–5, which merge before PR 6.
- Each PR can pass `cargo ci` and the trace gate on `main` after the decision-3 rows are moved or the gate exists, and after the `clippy` criteria that do not match `clippy` are narrowed.
- No PR edits a requirement in a document an earlier PR completed.
- No PR is design-only. Commit 1 is the design; the PR also contains the implementation, the tests, and the example or golden vectors.
- Stack depth is one. Cross-stack stalls are listed on PR 1–5.
- No planning branch sits above an implementation branch.
