# Terminal User Interface (tui)

![Date Badge](https://img.shields.io/badge/Date-October_11,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Approved-brightgreen)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

This design document establishes the architecture for `control-rs-tui`, a
standalone Terminal User Interface (TUI) binary designed to control and
monitor on-target Embedded Test Server (ETS) execution. The TUI provides
developers with a dynamic, real-time dashboard displaying system metadata, test
suite namespaces, cycle-level execution statistics and target logs,
facilitating rapid on-target iteration.

---

### 2. Requirements

#### 2.1 Functional Requirements

- **FR-1 — Target Metadata Display**: The TUI must query and display target
  board details, core clock frequency, FPU status and debug link parameters,
  consuming `Telemetry::TargetInfo` from the host bridge.
- **FR-2 — Hierarchical Test Tree**: The interface must present test suites
  and their cases in a tree layout mapping the Rust module namespace. A
  suite's tasks are rows of that suite beside its cases, in the same table
  and selection, so cases and tasks read as one list; a task row shows its
  run state, message, teardown outcome, statistics and input and output
  rates.
- **FR-3 — Hierarchical Telemetry Table**: For each test, the TUI must display
  cycle count (CYCCNT), duration (µs) and peak stack memory usage.
- **FR-4 — Persistent Log Terminal and History**: An integrated console window
  must display real-time debug and system logs streaming from the target. When
  autoscroll is enabled, newly arrived lines scroll to the tail and keep the
  newest entry visible without viewport clipping. Operators can scroll back and
  forward through log history using scroll keys (`PageUp`/`PageDown`), pausing
  autoscroll until scrolled back to the bottom. Operators can toggle scroll lock
  (`l`) to freeze autoscroll and inspect any portion of the log history.
  Operators can adjust the log panel height dynamically (`+`/`-`).
- **FR-5 — Keystroke Controls**: Users must control target execution via
  single-key shortcuts (`f` filter, `r` run all, `s` stop, `d` doc modal, `l`
  scroll lock, `+`/`-` log height, `q` quit). On connect and discovery, the
  dashboard remains idle awaiting operator command (`r` or case execution)
  rather than automatically executing all cases. `r` runs every case and never starts a
  lifecycle task. On a task row, `Enter` starts a free-running run. The single `s`
  key stops target execution uniformly: stopping an active lifecycle task if one
  is running, or stopping run-all and clearing queued cases otherwise.
- **FR-6 — Session liveness**: The dashboard retries discovery until the host
  session reports it complete, and it surfaces target process exit. A QEMU
  boot race or a dead subprocess must not look like an idle empty tree.
- **FR-7 — Item Description and Setting Inspection/Edit**: The operator can inspect
  descriptions of selected suites, test cases, tasks, and settings, and edit live
  settings (`SetSetting`) without leaving the dashboard.
- **FR-8 — Modal Documentation Viewer**: Pressing `d` on a selected item displays
  its formatted documentation in an overlaid modal mini-window,
  dismissed with `Esc`, `Enter`, or `d`.

#### 2.2 Non-Functional Requirements

- **NFR-1 — Diff-only repaint**: An unchanged frame writes nothing to the
  terminal; the write is the buffer diff, and repaint happens only when the
  application asks. A 16 ms / 60 FPS budget is not required.
- **NFR-2 — Metrics Caching**: Test suite results are stored on the host to
  maintain state across target restarts or reconnections.
- **NFR-3 — Non-Intrusive Polling**: The communication driver must poll
  telemetry buffers asynchronously without halting the target CPU.

#### 2.3 Constraints

- **C-1 — Presentation Only Scope**: `control-rs-tui` is strictly a presentation
  frontend; it does not own physical transport, session state machines, framing
  algorithms, or CRC checks, which are wholly delegated to
  `control-rs-ets-host::ETSBridge`.
- **C-2 — Single-Session Lifetime**: The binary manages exactly one active
  `ETSBridge` session at any time.
- **C-3 — Host Environment Constraints**: The crate requires standard terminal
  input/output and ANSI/VT escape sequences, operating under `std`.

---

### 3. Technical Overview

The TUI is implemented as the standalone binary crate `control-rs-tui`. It
runs on the developer's host machine and interfaces with physical
microcontrollers through the `ETSBridge` abstraction provided by
`control-rs-ets-host`.

Rendering is immediate mode: "In `ratatui`, every frame draws the UI anew"
[1], in contrast to a retained-mode toolkit where widgets are
created once and mutated [1]. The repaint cost this implies is
bounded by diffing rather than by redraw: changes accumulate in a current
buffer, and "at the end of each draw pass, the two buffers are compared, and
only the changes between these buffers are written to the terminal, avoiding
any redundant operations" [2]. A full-screen redraw of an
unchanged frame therefore writes nothing. Repaint is also on demand, since
`ratatui` "only updates when you tell it to" [1], which is what
makes the event-driven loop of §7 possible.

The terminal backend is `crossterm`, "a pure-rust, terminal manipulation
library that makes it possible to write cross-platform text-based interfaces"
[3].

```mermaid
flowchart LR
    subgraph Host ["Host Computer"]
        TUI["TUI Dashboard (ratatui)"]
        Bridge["ETSBridge"]
        TUI <==> Bridge
    end

    subgraph MCU ["Target MCU/Emulator"]
        Server["Server"]
    end

    Bridge <==>|" Framed stream (Postcard) "| Server
```

---

### 4. Architecture

#### 4.1. User Interface Layout

The TUI layout is partitioned into three main panels designed to maximize
developer situational awareness:

```text
===============================================================================
 TARGET: Teensy 4.0 (Cortex-M7) | LINK: USB CDC (/dev/ttyACM0)
===============================================================================
 [ RUNNING ] control_rs::math
-------------------------------------------------------------------------------
 NAME                                     CYCLES      TIME       STACK
 ▼ math::storage
   ├─ contiguous_storage_alloc            1,204       2.00µs     32
   └─ noncontiguous_storage_dma           3,410       5.68µs     64

 ▼ math::subprograms::level3
   ├─ gemm_10x10_f32 (soft-float)         84,500      140.8µs    128
   ├─ gemm_10x10_f32 (hard-float)        [ RUN... ]   ---        ---
   └─ gemm_50x50_f32 (hard-float)         PENDING     ---        ---
-------------------------------------------------------------------------------
 [ TARGET LOGS ] (Autoscroll: ON)
 > [INFO] Host connected.
 > [INFO] Discovered 24 tests via procedural macro test registry.
 > [PASS] storage::contiguous_storage_alloc
 ===============================================================================
 (f)ilter | (r)un all | (s)top | (q)uit
```

1. **Header Dashboard**: Displays the link, and from `SessionState::target_info`
   the protocol revision, board ID, core clock (MHz) and FPU class (FR-1).
   A rejected session shows `PROTOCOL MISMATCH` with both revisions. Debug
   link parameters beyond the configured port are not shown.
2. **Hierarchical Metrics Table**: A collapsible tree table showing test
   namespaces, cycle metrics, temporal duration (the target-reported `time_us`,
   passed through unmodified by `control-rs-ets-host`) and peak stack memory usage in
   bytes.
3. **Logs Panel**: A live log terminal streaming output from the target.
   The viewport reserves inner height accounting for the title and bottom
   border so the latest log entry is visible without clipping. When autoscroll
   is on, new entries scroll to the bottom. Operators can scroll back and
   forward through log history using `PageUp`/`PageDown` (FR-4) or toggle scroll
   lock with `l` to pause autoscroll, allowing inspection of the middle or top
   of the log stream.

   A suite's lifecycle task (`../ets/lifecycle-suite-design.md`: its setup, step,
   reset and teardown) appears as a row of that suite, after its cases, in
   the same table and selection: an operator moves from a case to the
   lifecycle task with the same keys, sees them as one list and triggers it
   with `Enter`, the key that runs a selected case. The row is marked `LIFECYCLE`
   in its name column. In
   place of cycles and stack, the row shows the run
   state (`Running`, `Warn`, `Pass`, `Fail`, `Error`, `Aborted`, `TimedOut`,
   `Bounded`), the latest message, the teardown outcome, `steps`, `time_us` and
   the input and output packet rates. `Enter` sends `StartTask` with
   `max_steps = 0` and `lockstep = false`; `s` sends `StopNow`. `r` (run all)
   does not start tasks, because a task run is unbounded. Lockstep runs
   need a `TaskSim` and are started from `control-rs-ets-host`, not the
   console. On a QEMU session `Enter` is refused by the host session
   (`../ets/lifecycle-suite-design.md` C-8) and the refusal is shown in the logs
   panel. A task uses its suite's settings through the FR-7 path; an edit
   during a run calls the task's `reset`.
4. **Footer Action Bar**: Displays available key shortcuts.
5. **Modal Documentation Overlay**: Pressing `d` on a selected row opens a
   centered modal mini-window displaying formatted documentation for the
   selected item (FR-8). The overlay renders on top of the table and log
   panels, and dismisses on `Esc`, `Enter`, or `d`.

#### 4.2. Host-Target ETSBridge Integration

The TUI communicates with target environments via the `ETSBridge`
abstraction provided by the `control-rs-ets-host` library crate. That crate is
the sole transport dependency; `ratatui` and `crossterm` stop at this binary
and never enter a headless dependency closure. Two execution targets are
supported:

* **QEMU Emulator (stdio pipe)**: Spawn a local QEMU virtual machine in a
  background child process by running `cargo run --bin ...` inside
  `examples/qemu`, piping stdin, stdout and stderr. (Semihosting is not used
  for data transport; see §5.)
* **Physical Serial Device**: Opens a connection to hardware (such as the Teensy
  4.0 board) over a USB CDC virtual serial port at a configured baud rate (
  defaulting to 115200) using the `serial2` crate.

#### 4.3. Delegated Transport, Framing and Recovery

Frame construction, CRC verification, telemetry deserialization, panic
detection and the reset/reconnect sequence are owned by `control-rs-ets-host`
and specified in `documentation/ets-host/ets-host-design.md`.
The TUI consumes their results only:

* `BridgeMessage::Telemetry` updates the metrics tree.
* `BridgeMessage::RawConsole` appends to the logs panel.
* A session reported as reset clears the local command queue and re-renders
  the tree from the re-issued discovery response, without tearing down
  terminal state.

The TUI holds no framing constants, no CRC parameters and no serial
configuration beyond the flags it forwards to the bridge constructor.

The loop follows the centralized catching, message passing pattern: events are
polled in one place and dispatched onward [4]. The event loop merges incoming
`ETSBridge` telemetry messages and terminal keystrokes into a single event
processing queue.

---

### 5. Alternatives

* **ARM Semihosting**: Rejected. Semihosting relies on triggering target
  software interrupts that halt the CPU core. This introduces millisecond-level
  latencies that violate real-time constraints and mask control-loop timing
  jitter. Additionally, probe-rs has documented register decoding and caching
  bugs during rapid semihosting trap polling.
* **Standard UART Logging**: Rejected. While it operates asynchronously,
  standard serial output lacks structured data capability, rendering
  hierarchical tables, dynamic status overlays and bidirectional control
  vectors impossible without complex custom parsers.
* **ETSBridge Logging**: The bridge parses all traffic between the server
  and the TUI, so it is the natural place to record a session transcript. That
  capability belongs to `control-rs-ets-host`, not to this binary.
* **Process-per-case result model**: Rejected for this dashboard, and noted
  because it is the dominant model elsewhere. `cargo-nextest` isolates each
  test in its own process so that "memory corruption in one test doesn't cause
  others to behave erratically" and "one test segfaulting does not take down a
  bunch of other tests" [5], and reads that process's standard
  output and error directly [5]. ETS keeps one persistent target
  process across a suite, so isolation comes from the reset sequence rather
  than from process boundaries, and the console stream is shared. The dashboard
  therefore attributes log lines to the running case by ordering, not by
  stream ownership.

---

### 6. Verification & Validation

#### 6.1 Verification

| Condition | Requirement | Method       | Target                                                                                                       | Criterion                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|:----------|:------------|:-------------|:-------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| VC-1.1    | FR-1        | `libtest`    | `control_rs_tui::tui::tests::fr1_fr2_fr3_fr4_scripted_discovery_and_metrics` `control_rs_tui::tui::tests::target_details_report_the_fpu_and_the_mismatch` `control_rs_tui::tui::tests::the_header_counts_passed_and_failed_tests` | Header displays target board details, core clock frequency, FPU status, protocol revision and link parameters |
| VC-2.1    | FR-2        | `libtest`    | `control_rs_tui::tui::tests::fr1_fr2_fr3_fr4_scripted_discovery_and_metrics` `control_rs_tui::tui::tests::test_navigation_and_collapse` `control_rs_tui::tui::tests::rows_follow_the_filter_query_and_collapse_state` `control_rs_tui::tui::tests::the_last_test_closes_the_branch_only_without_settings` `control_rs_tui::tui::tests::a_suite_without_tests_still_shows_its_header` | Interface presents test suites and cases in tree layout mapping the Rust module namespace |
| VC-2.2    | FR-2        | `libtest`    | `control_rs_tui::tui::tests::task_start_stop_keys` `control_rs_tui::tui::tests::a_task_row_shows_state_teardown_and_statistics` `control_rs_tui::tui::tests::a_task_row_colors_its_state` `control_rs_tui::tui::tests::the_task_row_closes_the_branch_only_without_settings` `control_rs_tui::tui::tests::an_idle_tick_leaves_the_rows_alone_and_an_active_one_refreshes_them` | Suite tasks are rows of that suite beside its cases displaying state, message, teardown outcome, statistics and rates |
| VC-3.1    | FR-3        | `libtest`    | `control_rs_tui::tui::tests::fr1_fr2_fr3_fr4_scripted_discovery_and_metrics` `control_rs_tui::tui::tests::durations_change_unit_exactly_at_a_thousand` | Displays cycle count, temporal duration and peak stack memory usage for each test |
| VC-4.1    | FR-4        | `libtest`    | `control_rs_tui::tui::tests::fr1_fr2_fr3_fr4_scripted_discovery_and_metrics` `control_rs_tui::tui::tests::log_widget_inner_height_and_newest_entry_visible` | Displays target logs in real-time, scrolling to tail when autoscroll is enabled without clipping bottom entry |
| VC-4.2    | FR-4        | `libtest`    | `control_rs_tui::tui::tests::log_scrollback_navigation` `control_rs_tui::tui::tests::adjustable_log_window_height` | Operator can scroll through log history with PageUp/PageDown, pause/resume autoscroll with scroll lock (l), and adjust log height with +/- |
| VC-5.1    | FR-5        | `libtest`    | `control_rs_tui::tui::tests::fr5_key_dispatch_and_navigation` `control_rs_tui::tui::tests::the_filter_prompt_applies_edits_and_closes` `control_rs_tui::tui::tests::r_runs_every_test_again_after_a_stop` `control_rs_tui::tui::tests::discovery_does_not_auto_run_all` | Single-key shortcuts control target execution (f filter, r run all, s stop, d doc modal, l scroll lock, q quit), remaining idle on discovery |
| VC-5.2    | FR-5        | `libtest`    | `control_rs_tui::tui::tests::task_start_stop_keys` | Enter starts task execution; s key uniformly stops active tasks or stops queued cases |
| VC-6.1    | FR-6        | `libtest`    | `control_rs_tui::tui::tests::discovery_is_repeated_only_while_incomplete_and_after_half_a_second` `control_rs_tui::tui::tests::test_process_exit_is_surfaced` `control_rs_tui::tui::tests::event_loop::a_target_panic_triggers_a_reattach_attempt` `control_rs_tui::tui::tests::event_loop::a_target_that_exits_is_reported_in_the_log` `control_rs_tui::tui::tests::event_loop::quitting_ends_the_loop_without_reattaching` | Retries discovery until complete and surfaces target process exit or panic |
| VC-7.1    | FR-7        | `libtest`    | `control_rs_tui::tui::tests::test_setting_description_and_edit` `control_rs_tui::tui::tests::escape_cancels_a_setting_edit` `control_rs_tui::tui::tests::boolean_settings_accept_words_and_digits` `control_rs_tui::tui::tests::description_shows_for_suite_test_task_and_setting` | Operator can inspect descriptions of selected suites, tests, tasks and settings, and edit live settings |
| VC-8.1    | FR-8        | `libtest`    | `control_rs_tui::tui::tests::modal_docs_popup_display_and_dismiss` | Modal mini-window renders formatted documentation for the selected item and dismisses with Esc, Enter, or d |
| VC-9.1    | NFR-1       | `review`     | —                                                                                                            | Unchanged frame writes nothing to the terminal via Ratatui buffer diffing |
| VC-10.1   | NFR-2       | `libtest`    | `control_rs_tui::tui::tests::nfr2_cache_survives_target_reset`                                              | Test suite results are stored on the host across target restarts or reconnections |
| VC-11.1   | NFR-3       | `review`     | —                                                                                                            | Target telemetry polled asynchronously without halting target CPU |
| VC-12.1   | C-1         | `review`     | —                                                                                                            | TUI is strictly a presentation frontend delegating transport, framing and CRC to ETSBridge |
| VC-13.1   | C-2         | `review`     | —                                                                                                            | Binary manages exactly one active ETSBridge session |
| VC-14.1   | C-3         | `review`     | —                                                                                                            | Operates under standard terminal I/O in std |

Target: 70% line coverage of `control-rs-tui`, measured with `cargo coverage`.

Excluded: widget draw calls, which are verified by inspection against §4.1
rather than by assertion; and terminal setup and teardown, which require a
real terminal.

* **Hardware integration run**: Flash a benchmark binary containing the
  mathematical subprograms onto physical target hardware and drive a full
  session interactively, including a deliberate panic and recovery.
* **Emulated session**: Drive the same session against a QEMU target and
  confirm the dashboard is indistinguishable apart from the link description.

#### 6.2 Acceptance

| Claim | Oracle | Measure | Bound |
|:------|:-------|:--------|:------|
| Tree reflects discovery | Scripted discovery response | Suites and cases rendered against those sent | Exact match |
| Telemetry table values | Scripted `MetricReport` frames | Cycles, duration and stack rendered against those sent | Exact, no rounding of raw counts |
| Duration derivation | Cycle delta divided by core frequency | Rendered microseconds against the computed value | Within display precision |
| Key bindings | Each bound key | Command emitted | The command named in FR-5 |
| Cache survives reset | Reset injected mid-session | Per-case metrics after re-discovery | Unchanged for completed cases |
| Transport isolation | Source and dependency tree | Framing, CRC or serial code in this crate | None |
| Discovery retry | First `ListSuites` dropped | Suites eventually rendered | Discovery completes |
| Target process exit | Spawned QEMU exits | Dashboard state | Exit surfaced; session not left looking connected |
| Setting description | Key bound to description | Setting text shown | Matches the suite registry |
| Task keys | `Enter` and `s` on a task row | Command emitted | `StartTask { max_steps: 0, lockstep: false }` and `StopNow` for that identifier |
| Task row | Scripted `TaskState`, `TeardownReport` and `TaskStats` frames | State, message, teardown outcome and statistics rendered | Exact match |
| Run all scope | `r` with tasks discovered | `StartTask` emitted | 0 |
| Side by side | Scripted discovery of a suite with 2 cases and 1 task | Rows under the suite and selection order | 3 rows in one table, task after the cases, reachable with the same navigation keys |
| QEMU refusal | `Enter` on a task row in a QEMU session | Frames written and logs panel | 0 frames; refusal logged |

No frame-rate bound is asserted. See §6.3.

#### 6.3 Limits

- **A 16 ms / 60 FPS frame budget is not a requirement of this plan and is
  not measured.** NFR-1 is the diff-only write, discharged by inspection of
  §3. The former 16 ms figure has no source and no measurement method here.
- Terminal emulator compatibility. `crossterm` is cross-platform
  [3], but only the emulators developers happen to use are
  exercised.
- Behaviour at large suite counts. No bound is stated on how many cases
  the tree can render before layout or scrolling degrades.
- Accessibility. Screen-reader behaviour of a full-screen TUI is not
  considered.
- Log attribution under concurrency. Because ETS shares one console stream
  across a suite, a line emitted by a case that has already ended may render
  under its successor; nothing detects or corrects this.

### 7. Performance & Resource Considerations

* **Host CPU Loading**: The TUI repaints only when new telemetry arrives or
  when the user presses a key, rather than polling in a hot loop. This is
  possible because `ratatui` "only updates when you tell it to"
  [1], and cheap because a repaint writes only the buffer diff
  [2]. An idle session with a quiet target therefore costs no host
  CPU.

---

### 8. Risks & Open Questions

* **Maintenance surface**: The dashboard is the largest host-side component by
  line count and the least covered by automated tests, since rendering is
  verified by inspection rather than assertion. The mitigation is to keep the
  binary thin: state and protocol handling belong to `control-rs-ets-host`, and
  this crate holds layout and input handling only.

---

### 9. Development Plan

| Task / Feature                         | Description                                                                                 | Estimated Effort |
|:---------------------------------------|:--------------------------------------------------------------------------------------------|:-----------------|
| **Step 1: Ratatui Interface Skeleton** | Build the terminal UI layout panels using `ratatui` (Header, Tree Table, Logs, Footer).     | 1.0 day          |
| **Step 2: ETSBridge Connection**       | Integrate `ETSBridge` polling channels (QEMU stdio / `serial2`) into the TUI event loop.    | 1.0 day          |
| **Step 3: Bidirectional Controls**     | Implement keystroke handlers and write command packets to the target down-buffer.           | 0.5 day          |
| **Step 4: Session liveness and settings** | Repair: retry `ListSuites` on the same interval as the headless runner; observe process exit; wait the host reset delay before re-open; restore setting description and `SetSetting`. Tests: 6.2 discovery-retry, process-exit, and setting-description rows. | 1.0 day          |
| **Step 5: Tasks**                | Task rows beside cases, `Enter` and `s` bindings (FR-2, FR-5; `../ets/lifecycle-suite-design.md` Phase 4). | 1.0 day          |

---

### 10. Revision History

| Revision | Date           | Author          | Description                                                                                                                 |
|:---------|:---------------|:----------------|:----------------------------------------------------------------------------------------------------------------------------|
| 1.0      | May 24, 2026   | @MitchellDScott | Initial specification for interactive terminal user interface (TUI) test dashboard.                                         |
| 1.1      | July 18, 2026  | @MitchellDScott | Architecture & transport: added Ratatui layout, `ServerBridge` async polling channels, and Teensy 4.1 hardware integration. |
| 1.2      | August 6, 2026 | @MitchellDScott | Protocol alignment: synchronized interactive command schemas with shipped postcard framing and reconnection lifecycles.     |
| 1.3      | September 9, 2026 | @MitchellDScott | Packaging split: promoted TUI to standalone `control-rs-tui` binary crate consuming `control-rs-ets-host::ServerBridge`.   |
| 1.4      | September 9, 2026 | @MitchellDScott | Crate split: relocated to `documentation/tui/`; framing, panic detection and reconnection delegated to `control-rs-ets-host`. |
| 1.5      | September 9, 2026 | @MitchellDScott | Evidence pass: added the research pair and citation layer, grounded the rendering and repaint claims, added the process-per-case alternative, restructured §6 per `vv-standards.md`, and moved the unverified 16 ms budget to 6.7. |
| 1.6      | September 9, 2026 | @MitchellDScott | Hardening pass: numbered §2.1–§2.3 and added §2.3 Constraints (C-1..C-3), mapped TargetInfo in FR-1, corrected bridge diagram link, completed 6.4 traceability, and standardized References. |
| 1.7      | September 15, 2026 | @MitchellDScott | NFR-1 is diff-only repaint; 16 ms budget is not a requirement and is not traced. |
| 1.8      | September 16, 2026 | @MitchellDScott | FR-6 session liveness, FR-7 suite setting inspection and edit; 6.2 discovery/exit/setting rows; §9 Step 4. |
| 1.9      | September 18, 2026 | @MitchellDScott | Packaging alignment: updated host bridge references to `control-rs-ets-host::ETSBridge`. |
| 1.10      | September 24, 2026 | @MitchellDScott | Duration column shows the target-reported `time_us`; §6.3 reference corrected. |
| 1.11     | September 24, 2026 | @MitchellDScott | FR-1 header shows `TargetInfo` protocol, board, clock and FPU, or the protocol mismatch. |
| 1.12     | September 28, 2026 | @MitchellDScott | The event loop takes its input from an injected event source (the terminal is the production source) and the discovery retry decision is a pure function, so the loop is testable without a terminal. |
| 1.13     | October 7, 2026 | @MitchellDScott | FR-2 and FR-5 extended for lifecycle suites (`../ets/lifecycle-suite-design.md` 1.2): `TASK` tree nodes with run state and rates, `Enter` and `s` bindings, `r` excludes lifecycle suites, three §6.2 rows, Step 5. |
| 1.14     | October 7, 2026 | @MitchellDScott | QEMU sessions refuse console-started (free-running) task runs and log the refusal (`../ets/lifecycle-suite-design.md` C-8); one §6.2 row. |
| 1.15     | October 7, 2026 | @MitchellDScott | FR-2: tasks are rows of their suite beside its cases in one table and selection; FR-5 keys act on task rows; side-by-side §6.2 row (`../ets/lifecycle-suite-design.md` 1.7); the row is the suite's lifecycle task, triggered with `Enter` like a case; marker `LIFECYCLE` |
| 1.16     | October 11, 2026 | @MitchellDScott | Unified (s) stop key across run-all and tasks, 1-line log viewport autoscroll fix, item description inspection (FR-7), FR-4 log history navigation, FR-8 modal documentation viewer, formal verification matrix (VC-1.1..VC-14.1), and requirement traceability wiring. |

---

## References

[1] Ratatui contributors, "Rendering," *Ratatui documentation, Concepts*.
[Online]. Available: https://ratatui.rs/concepts/rendering/. Accessed:
Sep. 9, 2026.

[2] Ratatui contributors, *ratatui*: terminal user interface library.
[Online]. Available:
https://docs.rs/ratatui/latest/ratatui/terminal/struct.Terminal.html. Accessed:
Sep. 9, 2026.

[3] crossterm contributors, *crossterm*: pure-Rust cross-platform terminal
manipulation library. [Online]. Available: https://docs.rs/crossterm.
Accessed: Sep. 9, 2026.

[4] Ratatui contributors, "Event handling," *Ratatui documentation, Concepts*.
[Online]. Available: https://ratatui.rs/concepts/event-handling/. Accessed:
Sep. 9, 2026.

[5] nextest contributors, "Why process-per-test," *cargo-nextest
documentation*. [Online]. Available:
https://nexte.st/docs/design/why-process-per-test/. Accessed: Sep. 9, 2026.