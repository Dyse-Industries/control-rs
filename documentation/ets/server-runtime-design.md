# Server Runtime (server-runtime)

![Date Badge](https://img.shields.io/badge/Date-November_10,_2026-blue)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)
![Design Status Badge](https://img.shields.io/badge/Design%20Status-Draft-orange)

---

### 1. Introduction

This document specifies the target-side execution engine and server runtime
component of the ETS. It defines: the interactive command loop, test case
dispatch, the test execution lifecycle state machine, atomic execution context
tracking for fault attribution, zero-allocation panic interception and dynamic
parameter mutation.

---

### 2. Requirements

#### 2.1 Functional Requirements

- **FR-1 - Host Transport & Server Event Loop**: The server context shall flush
  bidirectional communication transport and run a persistent event loop
  indefinitely until terminated by an unrecoverable hardware fault, transport
  error or cooperative reset command.
- **FR-2 - Test Suite Discovery & Case Dispatch**: The server runtime shall
  safely
  reconstruct a static slice of test suite descriptors from raw linker section
  boundary pointers (`.ets_test_suites`), validate dispatch bounds without
  panicking on out-of-range indices and stream suite discovery hierarchy upon
  request.
- **FR-3 - Execution Context & Lifecycle State Tracking**: The server runtime
  shall maintain thread-safe atomic indicators recording the active suite and
  test indices and emit sequenced lifecycle state transitions (`Running`,
  `Passed`, `Failed`) for every executed case.
- **FR-4 - Target Profiling & Resource Measurement**: The server profiling
  bracket shall measure elapsed CPU clock cycles, monotonic execution duration
  in microseconds and peak stack memory consumption under critical section
  isolation, packaging measured metrics into a telemetry frame.
- **FR-5 - Dynamic Suite Configuration & Parameter Tuning**: The server shall
  service dynamic parameter mutation requests over the transport, validating
  suite indices, setting indices, and value types, rejecting type mismatches
  without corrupting memory or terminating the server.
- **FR-6 - Failure Diagnostics & Crash Recovery**: The server failure handler
  shall execute a deterministic, zero-allocation diagnostic capture sequence
  upon unhandled panics or hardware exceptions, format forensic records into
  static buffers without buffer overrun, attribute crashes to active tests and
  coordinate cooperative resets.
- **FR-7 - Target Lifecycle & Hardware Identification**: The server shall query
  platform hardware identity, emit target identification telemetry, service
  cooperative warm resets and support clean virtual environment termination via
  dedicated exit routines.
- **FR-8 - Reserved Diagnostic Suite Zero and Index Offset**: The server runtime
  shall reserve `suite_id = 0` for a built-in static `SuiteDescriptor` in
  read-only memory and map user-registered suites from `.ets_test_suites` to
  `suite_id = 1..=N` across suite discovery, case dispatch, and setting
  mutation. Reserving index `0` in the server lookup guarantees the diagnostic
  suite is present even when no user suites are linked, while consuming zero
  static RAM.
- **FR-9 - Built-in Tare Calibration Case**: The built-in diagnostic suite
  (`suite_id = 0`) shall provide a no-op `tare` executable at `test_id = 0` that
  executes through the standard `run_test` lifecycle and profiling bracket.
  Executing `tare` as a standard `ExecDescriptor` entry ensures the host
  measures the exact indirect call, stack-painting, interrupt-masking, and
  telemetry-framing overhead incurred by user test cases.

#### 2.2 Non-Functional Requirements

- **NFR-1 - Low-Overhead Case Dispatch**: Invoking an empty test case through
  the suite dispatch abstraction must introduce minimal latency (bounded to
  fewer than 20 clock cycles on target architectures) so as not to distort
  micro-benchmark timings.
- **NFR-2 - Microsecond Telemetry Timestamp Precision**: Diagnostic error logs
  and execution metrics must derive timestamps with microsecond resolution
  from monotonic platform nanosecond clocks.
- **NFR-3 - Timing-Critical Zero-Allocation Logging**: The server logging,
  state transition, metric reporting, and discovery streaming paths must format
  and transmit telemetry without invoking dynamic allocation or expensive
  on-target string formatting on the hot execution path.

#### 2.3 Constraints

- **C-1 - Strict `#![no_std]`, Zero Heap**: The target Server crate
  (`control-rs-ets`) must compile on bare-metal targets under `#![no_std]` with
  strictly zero dynamic heap allocation.
- **C-2 - Bounded Flash and RAM Footprint**: The target Server implementation
  (excluding test payloads) must consume under 32 KB Flash memory and under 8 KB
  RAM, with test suite descriptors residing in Flash ROM.

---

### 3. Technical Overview

The `Server` runtime manages interactive target execution, processing commands
from the host, executing dispatched test cases, managing dynamic parameter
updates and capturing forensic panic state.

```mermaid
flowchart TD
    Host["Host Transport Link"] <==> Comms["HostComms Interface"]

    subgraph ServerRuntime ["Server Runtime Engine (control-rs-ets)"]
        direction TB
        Server["Server (server.rs)"]
        Context["Context<C, P>"]
        Platform["Platform (identity constants)"]
        Trackers["CURRENT_SUITE / CURRENT_TEST (AtomicIsize)"]
        Panics["handle_failure / FailureBufWriter (util.rs)"]
        Settings["Dynamic Settings Mutation (settings.rs)"]
        Server --> Context
        Context --> Comms
        Context --> Platform
        Server --> Trackers
        Server --> Panics
        Server --> Settings
    end

    subgraph Peripherals ["Subsystem Dependencies"]
        Profiler["CPUProfiler (timing / IRQ / stack)"]
        Suites["Discovered Suites (.ets_test_suites)"]
    end

    Context --> Profiler
    Server -->|Dispatches| Suites
```

---

### 4. Architecture

#### 4.1. Server Struct and Context Abstraction

Serves: FR-1, FR-1.1, FR-1.2, C-1

To remain decoupled from physical UART/RTT hardware and vendor-specific
profiling registers, the `Server` is generic over communications transport `C`
and CPU profiling utilities `P`:

```rust
pub struct Context<C, P> {
    pub comms: C,
    pub cpu_utils: P,
    pub platform: Platform,
}

pub struct Server<'a, C, P> {
    pub context: Context<C, P>,
    pub suites: &'a [&'static SuiteDescriptor],
}
```

The context fields provide:

- `C: HostComms`: Bidirectional host link providing non-blocking command
  polling, telemetry framing and flushing
  ([host-comm-design.md](host-comm-design.md)).
- `P: CPUProfiler`: Cycle counting, monotonic nanosecond clock samples, stack
  pointer queries, stack painting, scoped interrupt masking
  (`disable_interrupts`),
  permanent interrupt disabling and target lifecycle hooks (`reset`, `exit`)
  ([cpu-profiler-design.md](cpu-profiler-design.md)).
- `platform: Platform`: Target hardware identity constants (`board_id`,
  `core_clock_hz`, `fpu_flags`, §4.6).

#### 4.2. Interactive Event Loop and Dispatch

Serves: FR-1, FR-1.1, FR-1.2, FR-2, FR-2.2, FR-7.2, FR-7.3, FR-8, NFR-1

The server's main loop (`Server::run`) polls non-blocking commands from the host
transport and routes them:

```rust
pub fn run(&mut self) -> ServerResult<C::Error> {
    self.context.comms.flush()?;
    loop {
        match self.context.comms.poll_command()? {
            Some(Command::ListSuites) => self.stream_discovery()?,
            Some(Command::RunExecutable { suite_id, test_id }) => {
                self.dispatch_executable(suite_id, test_id)?;
            }
            Some(Command::SetSetting { suite_id, setting_id, value }) => {
                self.mutate_setting(suite_id, setting_id, value)?;
            }
            Some(Command::TryReset) => self.context.cpu_utils.reset(),
            None => {}
        }
    }
}
```

To provide guaranteed diagnostic and timing tare routines without relying on
linker placement or dynamic registration, the server maintains a static
`DIAGNOSTIC_SUITE` in `.rodata` and resolves suite indices through `resolve_suite`:

```rust
static DIAGNOSTIC_EXECS: [ExecDescriptor; 1] = [ExecDescriptor {
    description: "No-op baseline for profiling tare calibration",
    name: "tare",
    test_fn: tare_noop,
}];

pub static DIAGNOSTIC_SUITE: SuiteDescriptor = SuiteDescriptor {
    description: "Built-in server diagnostic and tare suite",
    executables: &DIAGNOSTIC_EXECS,
    name: "__ets_diagnostics",
    settings: &[],
};

impl<'a, C, P> Server<'a, C, P> {
    #[inline]
    fn resolve_suite(&self, suite_id: u16) -> Option<&'static SuiteDescriptor> {
        match suite_id {
            0 => Some(&DIAGNOSTIC_SUITE),
            n => self.suites.get(usize::from(n.saturating_sub(1))).copied(),
        }
    }
}
```

Dispatch resolves `suite_id` through `resolve_suite(suite_id)`: `suite_id = 0`
maps directly to the statically retained `DIAGNOSTIC_SUITE`, while `suite_id >= 1`
indexes user suites from `.ets_test_suites` via `n.saturating_sub(1)`. Discovery
chains `DIAGNOSTIC_SUITE` at index 0 followed by user suites at `1..=N`. An
out-of-bounds index transmits an error diagnostic frame to the host without
panicking or terminating the server loop.

#### 4.3. Test Execution and Lifecycle State Transitions

Serves: FR-3, FR-3.1, FR-4, FR-4.1, FR-4.2, FR-4.3, FR-4.4, FR-4.5, FR-9, NFR-1, NFR-2

When executing a test case, the server coordinates execution inside a scoped
profiling bracket:

```mermaid
sequenceDiagram
    autonumber
    actor Host as Host (ETSBridge)
    participant Server as Target Server (Server::run_test)
    participant Ind as Context Trackers (CURRENT_SUITE / CURRENT_TEST)
    participant Prof as CPUProfiler (cpu_utils)
    participant Test as Test Case (fn())

    Host ->> Server: Command::RunExecutable { suite_id, test_id } [δ_rx]
    Server ->> Host: Telemetry::TestStateChange (TestState::Running) & flush() [δ_tx_run]
    Server ->> Ind: set_active(suite_id), set_active(test_id) [δ_ctx_set]
    Server ->> Prof: sp = get_sp(), paint_stack(sp) [δ_paint]

    Server ->> +Prof: disable_interrupts(closure) [δ_irq_mask]
    Note over Prof,Test: Critical Section (IRQs Masked)
    Prof ->> Prof: c_0 = get_cycles() [δ_cyc]
    Prof ->> Prof: t_0 = get_nanos() [δ_ns]
    Prof ->> +Test: Invoke exec.test_fn() [δ_dispatch]
    Note right of Test: Execute body [δ_fn]
    Test -->> -Prof: Return
    Prof ->> Prof: t_1 = get_nanos() [δ_ns]
    Prof ->> Prof: c_1 = get_cycles() [δ_cyc]
    Prof -->> -Server: Restore IRQs [δ_irq_unmask]

    Server ->> Prof: stack_peak = read_stack_peak(sp) [δ_scan]
    Server ->> Server: Δc = c_1 - c_0, Δt_us = (t_1 - t_0) / 1000 [δ_calc]
    Server ->> Ind: set_idle(), set_idle() [δ_ctx_idle]
    Server ->> Host: Telemetry::TestStateChange (TestState::Passed) [δ_tx_pass]
    Server ->> Host: Telemetry::MetricReport { Δc, Δt_us, stack_peak } & flush() [δ_tx_metric]
```

**Figure 4.1**: Sequenced test execution lifecycle and profiling sample points
(`c_0`, `t_0`, `t_1`, `c_1`) with stage duration indicators (`δ_*`).

1. Transmits `TestState::Running` and flushes transport ($\delta_{\text{tx,run}}$).
2. Marks active indicators (`CURRENT_SUITE.set_active(s)`,
   `CURRENT_TEST.set_active(t)`, $\delta_{\text{ctx,set}}$).
3. Queries stack pointer (`get_sp`) and paints stack sentinel words
   (`0xCDCD_CDCD`, $\delta_{\text{paint}}$).
4. Masks interrupts via `disable_interrupts` ($\delta_{\text{irq,mask}}$),
   sampling start cycles $c_0$ ($\delta_{\text{cyc}}$) and nanoseconds $t_0$
   ($\delta_{\text{ns}}$).
5. Invokes the case function pointer (`fn()`, $\delta_{\text{dispatch}} + \delta_{\text{fn}}$).
6. Samples end nanoseconds $t_1$ ($\delta_{\text{ns}}$) and cycles $c_1$
   ($\delta_{\text{cyc}}$), restoring interrupt masks ($\delta_{\text{irq,unmask}}$).
7. Scans painted memory (`read_stack_peak`, $\delta_{\text{scan}}$) to calculate
   stack byte displacement and computes
   $\Delta c = c_1 - c_0 = \delta_{\text{cyc}} + 2\,\delta_{\text{ns}} + \delta_{\text{dispatch}} + \delta_{\text{fn}}$
   and $\Delta t_{\text{ns}} = t_1 - t_0 = \delta_{\text{ns}} + \delta_{\text{dispatch}} + \delta_{\text{fn}}$
   ($\delta_{\text{calc}}$).
8. Clears indicators (`set_idle()`, $\delta_{\text{ctx,idle}}$).
9. Transmits `TestState::Passed` ($\delta_{\text{tx,pass}}$) and
   `Telemetry::MetricReport` ($\delta_{\text{tx,metric}}$).
10. Executing `Command::RunExecutable { suite_id: 0, test_id: 0 }` runs the
    built-in diagnostic `tare` no-op (`tare_noop`), where $\delta_{\text{fn}} = 0$,
    measuring the exact server and profiler tare baseline:
    $\Delta c_{\text{empty}} = \delta_{\text{cyc}} + 2\,\delta_{\text{ns}} + \delta_{\text{dispatch}}$
    and $\Delta t_{\text{ns,empty}} = \delta_{\text{ns}} + \delta_{\text{dispatch}}$.

#### 4.4. Atomic Execution Context Tracking and Fault Attribution

Serves: FR-3, FR-3.2, FR-6.3, FR-6.4

To attribute unhandled panics or CPU faults to the specific running test case,
global atomic indicators maintain active execution context:

```rust
pub struct TestIndexIndicator {
    index: AtomicIsize,
}

impl TestIndexIndicator {
    pub const fn new() -> Self;
    pub fn get(&self) -> Option<usize>;
    pub fn set_active(&self, index: usize);
    pub fn set_idle(&self);
}

pub static CURRENT_SUITE: TestIndexIndicator = TestIndexIndicator::new();
pub static CURRENT_TEST: TestIndexIndicator = TestIndexIndicator::new();
```

- When active, `index` holds the non-negative index ($\ge 0$).
- When idle or between tests, `index` holds negative sentinel `-1`.
- On panic, `handle_failure` inspects `get()`: if `Some`, it transmits
  `TestState::Failed` for that specific case; if `None`, it records a general
  firmware fault without case attribution.

#### 4.5. Forensic Panic Capture and State Recovery

Serves: FR-6, FR-6.1, FR-6.2, FR-6.3, FR-6.4, FR-6.5, FR-6.6, C-1

When an assertion fails or a hardware exception occurs, `handle_failure`
executes
a zero-allocation diagnostic capture sequence:

```rust
pub fn handle_failure<C: HostComms, P: CPUProfiler>(
    ctx: &mut Context<C, P>,
    info: &core::panic::PanicInfo<'_>,
) -> ! {
    let mut buffer = [0u8; 256];
    let mut writer = FailureBufWriter::new(&mut buffer);
    // Format file, line, and panic message safely into static buffer
    let _ = write!(writer, "{}", info);

    // Attributed state transition if active test
    if let (Some(s), Some(t)) = (CURRENT_SUITE.get(), CURRENT_TEST.get()) {
        let _ = ctx.comms.send_telemetry(&Telemetry::TestStateChange {
            suite_id: s,
            test_id: t,
            state: TestState::Failed,
        });
    }

    // Emit forensic telemetry frame
    let _ = ctx.comms.send_telemetry(&Telemetry::Panic {
        message: writer.as_str(),
    });
    let _ = ctx.comms.flush();

    ctx.cpu_utils.disable_interrupts_permanently();

    // Await cooperative reset or trigger fallback
    loop {
        if let Ok(Some(Command::TryReset)) = ctx.comms.poll_command() {
            ctx.cpu_utils.reset();
        }
    }
}
```

`FailureBufWriter` guarantees zero dynamic heap allocation and enforces strict
capacity limits, cleanly truncating messages that exceed 256 bytes without
panicking or writing out of bounds.

#### 4.6. Platform Struct: Hardware Identity

Serves: FR-7, FR-7.1

`Platform` is a pure data struct containing immutable target board constants:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Platform {
    pub board_id: u16,
    pub core_clock_hz: u32,
    pub fpu_flags: u8,
}

impl Platform {
    pub const fn derive_fpu_flags() -> u8 {
        let single = u8::from(cfg!(target_abi = "eabihf") || cfg!(target_feature = "f"));
        let double = u8::from(cfg!(target_feature = "d"));
        single | (double << 1)
    }
}
```

The server queries `Platform` during discovery to emit `Telemetry::TargetInfo`.

#### 4.7. Dynamic Parameter Mutation and Type Checking

Serves: FR-5, FR-5.1, FR-5.2, NFR-3

The server provides dynamic parameter tuning (`settings.rs`) without stopping
the execution loop:

1. Validates `suite_id < suites.len()`.
2. Validates `setting_id < suite.settings.len()`.
3. Verifies that the requested `SettingValue` matches the registered parameter's
   type (`SettingType`).
4. On type match, applies the updated value using atomic or interior-mutable
   descriptors and echoes the updated state via `Telemetry::SettingUpdate`.
5. On type mismatch or invalid index, emits an error diagnostic telemetry log
   without corrupting setting memory or terminating the server.

---

### 5. Alternatives Considered

#### 5.1. Debug-Probe-Driven Execution (`probe-rs` + semihosting)

* Executing test dispatch via ARM semihosting traps.
* *Rejected*: Semihosting halts the processor core during I/O operations,
  destroying real-time timing determinism.

#### 5.2. Heap-Allocated Panic Formatting (`alloc::format!`)

* Formatting panic strings via dynamic heap allocation.
* *Rejected*: Bare-metal targets run `#![no_std]` without an active global
  allocator; heap allocations during panic handling risk secondary panics on
  OOM.

---

### 6. Verification & Validation

#### 6.1 Verification

| Condition | Requirement | Method    | Target                                                                                     | Criterion                                                                                                                         |
|:----------|:------------|:----------|:-------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------|
| VC-1.1    | FR-1        | `libtest` | `control_rs_ets::server::tests::test_server_ok_to_reset`                                   | Persistent event loop executes until cooperative reset command is received; FR-1 holds iff both conditions hold                   |
| VC-1.2    | FR-1        | `libtest` | `control_rs_ets::server::tests::test_server_poll_command_error`                            | Polling errors from transport are returned as `Err` and not swallowed                                                             |
| VC-2.1    | FR-2        | `libtest` | `control_rs_ets::util::tests::test_get_suites_fn`                                          | Test suites are safely reconstructed from linker boundary pointers; FR-2 holds iff all conditions hold                            |
| VC-2.2    | FR-2        | `libtest` | `control_rs_ets::server::tests::test_server_out_of_bounds`                                 | Invalid suite or test index emits diagnostic error and aborts without panicking                                                   |
| VC-2.3    | FR-2        | `libtest` | `control_rs_ets::server::tests::test_server_discovery`                                     | Discovery command streams target info, suites, executables, and settings                                                          |
| VC-3.1    | FR-3        | `libtest` | `control_rs_ets::server::tests::test_server_run_test`                                      | Test execution emits `Running` and `Passed` lifecycle transitions; FR-3 holds iff both conditions hold                            |
| VC-3.2    | FR-3        | `libtest` | `control_rs_ets::server::tests::test_index_indicator`                                      | Thread-safe atomic indicators track active suite and test indices and reset to idle                                               |
| VC-4.1    | FR-4        | `libtest` | `control_rs_ets::server::tests::profiling_reports_elapsed_cycles_and_microseconds`         | Execution bracket reports non-zero elapsed cycles, microsecond duration, and stack peak; FR-4 holds iff all conditions hold       |
| VC-4.2    | FR-4        | `libtest` | `control_rs_ets::server::tests::test_server_metric_report_flush_error`                     | Metric report transmission flushes telemetry frames to host transport                                                             |
| VC-4.3    | FR-4        | `libtest` | `control_rs_ets::server::tests::test_host_cpu_profile_utils`                               | Profiler execution bracket exercises cycle and duration measurement hooks                                                         |
| VC-5.1    | FR-5        | `libtest` | `control_rs_ets::server::tests::test_server_set_setting`                                   | Valid setting mutation updates parameter value and echoes state; FR-5 holds iff both conditions hold                              |
| VC-5.2    | FR-5        | `libtest` | `control_rs_ets::server::tests::test_server_set_setting_type_mismatch`                     | Incompatible value type is rejected with error without mutating state                                                             |
| VC-6.1    | FR-6        | `libtest` | `control_rs_ets::util::tests::test_failure_buf_writer_fn`                                  | Diagnostic capture formats forensic payload into static buffer without heap allocation; FR-6 holds iff all conditions hold        |
| VC-6.2    | FR-6        | `libtest` | `control_rs_ets::util::tests::the_buffer_writer_fills_its_buffer_exactly_and_refuses_more` | Formatting routine rejects writes exceeding buffer capacity without writing past boundary                                         |
| VC-6.3    | FR-6        | `libtest` | `control_rs_ets::util::tests::test_handle_failure_comms_active`                            | Crash during active test marks active case as failed                                                                              |
| VC-6.4    | FR-6        | `libtest` | `control_rs_ets::util::tests::test_handle_failure_comms_idle`                              | Crash in idle state suppresses case failure transitions                                                                           |
| VC-6.5    | FR-6        | `libtest` | `control_rs_ets::util::tests::test_handle_exception_fn`                                    | Exception handler captures location and coordinates cooperative reset                                                             |
| VC-6.6    | FR-6        | `libtest` | `control_rs_ets::util::tests::test_handle_failure_comms_disabled`                          | Failure handler safely completes target reset when transport is non-functional                                                    |
| VC-7.1    | FR-7        | `libtest` | `control_rs_ets::server::tests::test_server_discovery`                                     | Target info telemetry emits protocol version, board ID, clock frequency, and FPU capabilities; FR-7 holds iff all conditions hold |
| VC-7.2    | FR-7        | `libtest` | `control_rs_ets::server::tests::test_server_ok_to_reset`                                   | Cooperative reset command triggers target reset sequence                                                                          |
| VC-7.3    | FR-7        | `libtest` | `control_rs_ets::server::tests::test_server_exit`                                          | Exit routine closes communications before invoking target exit hook                                                               |
| VC-8.1    | FR-8        | `libtest` | `control_rs_ets::server::tests::test_server_diagnostic_suite_discovery`                | Discovery always emits the built-in diagnostic suite at `suite_id = 0` followed by user suites at indices `1..=N`, even when the user suite slice is empty; FR-8 holds iff both conditions hold |
| VC-8.2    | FR-8        | `libtest` | `control_rs_ets::server::tests::test_server_user_suite_offset_dispatch_and_settings`   | Case dispatch and setting mutation resolve `suite_id = 0` to the built-in diagnostic suite, index `self.suites` via `suite_id.saturating_sub(1)` for `suite_id >= 1`, and report out-of-bounds indices without panicking |
| VC-9.1    | FR-9        | `libtest` | `control_rs_ets::server::tests::test_server_diagnostic_suite_tare`                     | `Command::RunExecutable { suite_id: 0, test_id: 0 }` executes the built-in `tare` no-op case and emits `Running`, `Passed`, and `MetricReport` telemetry frames; FR-9 holds iff both conditions hold |
| VC-9.2    | FR-9        | `libtest` | `control_rs_ets::server::tests::test_server_tare_matches_empty_user_test`              | Executing `(suite_id: 0, test_id: 0)` and executing an empty user test at `suite_id = 1` invoke identical `CPUProfiler` hook sequences, context tracker transitions, and metric reports |
| VC-10.1   | NFR-1       | `libtest` | `control_rs_ets::server::tests::test_server_run_test`                                      | Indirect case dispatch via function pointer introduces low latency                                                                |
| VC-10.2   | NFR-2       | `libtest` | `control_rs_ets::server::tests::log_timestamps_are_in_microseconds`                        | Diagnostic error logs and execution metrics derive timestamps with microsecond resolution                                         |
| VC-10.3   | NFR-3       | `libtest` | `control_rs_ets::util::tests::the_buffer_writer_fills_its_buffer_exactly_and_refuses_more` | Formatting and logging operate with zero dynamic heap allocations on hot path                                                     |
| VC-26.1   | C-1         | `gate`    | `build`                                                                                    | `control-rs-ets` compiles cleanly for bare-metal targets under `#![no_std]`                                                       |
| VC-26.2   | C-2         | `libtest` | `control_rs_ets::tests::test_flash_descriptor_sizes`                                       | Descriptors and runtime state structures reside within bounded memory footprints                                                  |

#### 6.2 Acceptance

| Claim                       | Oracle             | Measure                               | Bound                    |
|:----------------------------|:-------------------|:--------------------------------------|:-------------------------|
| Discovery completeness      | Source annotations | Suites and cases reported to host     | Exact set equality       |
| Persistent execution        | Suite of *n* cases | Target restarts during suite          | 0, absent injected fault |
| Cycle telemetry attribution | Case boundaries    | Reports attributed to running case    | 1:1, no orphans          |
| Panic capture               | Injected panic     | Forensic record received before reset | Received, with location  |
| Dynamic setting mutation    | Update command     | Setting value on target               | Mutated and echoed       |
| Allocation freedom          | Disassembly        | Dynamic allocator symbols             | 0                        |

#### 6.3 Limits

- **Cooperative polling loop (`L-1`)**: Test execution inside
  `disable_interrupts`
  cannot be preempted by `Command::TryReset`; infinite loops in test cases
  require
  hardware watchdog timer reset or external reset line.
- **Fixed 256-byte panic buffer (`L-2`)**: `FailureBufWriter` formats payloads
  into a static 256-byte stack buffer; longer messages truncate cleanly at 256
  bytes.

---

### 7. Performance & Resource Considerations

- **Zero Allocations**: All formatting and dispatch routines operate strictly on
  stack or static buffers.
- **Microsecond Clock Precision**: Profiling durations derive directly from
  hardware timer nanoseconds converted to microseconds.
- **Footprint**: Compiled server runtime logic contributes fewer than 12 KB
  Flash
  and under 2 KB RAM overhead.

---

### 8. Risks & Open Questions

- **Watchdog Integration**: Currently, runaway test loops rely on host session
  timeouts; future hardware watchdog integration will enforce automated target
  reboots.

---

### 9. Development Plan

| Phase                            | Delivers                                         | Requirements          | Effort | Status  |
|:---------------------------------|:-------------------------------------------------|:----------------------|:-------|:--------|
| 1. Server Event Loop & Dispatch  | Core polling loop and case dispatch abstraction  | FR-1, FR-2, FR-8, NFR-1 | 3 days | Shipped |
| 2. Lifecycle & Profiling Bracket | State machine, atomic context tracking, metrics  | FR-3, FR-4, FR-9, NFR-2 | 3 days | Shipped |
| 3. Panic Interception & Settings | Forensic buffer writer, crash recovery, settings | FR-5, FR-6, FR-7, C-1 | 4 days | Shipped |

---

### 10. Revision History

| Revision | Date              | Author          | Description                                                              |
|:---------|:------------------|:----------------|:-------------------------------------------------------------------------|
| 1.0      | November 10, 2026 | @MitchellDScott | Initial component design doc extracted from lead subsystem architecture. |

---

## References

[1] embedded-test, "embedded_test," *docs.rs*. [Online].
Available: https://docs.rs/embedded-test/latest/embedded_test/. Accessed: Aug.
7, 2026.

[2] Noah Pendleton, "Measuring Stack Usage the Hard Way," *Interrupt (Memfault
blog)*. [Online].
Available: https://interrupt.memfault.com/blog/measuring-stack-usage. Accessed:
Aug. 7, 2026.
