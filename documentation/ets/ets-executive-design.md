# ETS Executive (ets-executive)

![Date Badge](https://img.shields.io/badge/Date-November_10,_2026-blue)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)
![Design Status Badge](https://img.shields.io/badge/Design%20Status-Draft-orange)

---

### 1. Introduction

This document specifies the target-side test executive and execution engine
component of the ETS. It defines: the interactive command loop, test case
dispatch, the test execution lifecycle state machine, atomic execution context
tracking for fault attribution, zero-allocation panic interception and dynamic
parameter mutation.

---

### 2. Requirements

#### 2.1 Functional Requirements

- **FR-1 - Host Transport & Event Loop**: The executive context shall flush
  bidirectional communication transport and run a persistent event loop
  indefinitely until terminated by an unrecoverable hardware fault, transport
  error or cooperative reset command.
- **FR-2 - Test Suite Discovery & Case Dispatch**: The ETS executive shall
  safely reconstruct a static slice of test suite descriptors from raw linker
  section boundary pointers (`.ets_test_suites`), validate dispatch bounds
  without panicking on out-of-range indices and stream suite discovery hierarchy
  upon request.
- **FR-3 - Execution Context & Lifecycle State Tracking**: The ETS executive
  shall maintain thread-safe atomic indicators recording the active suite and
  test indices and emit sequenced lifecycle state transitions (`Running`,
  `Passed`, `Failed`) for every executed case.
- **FR-4 - Target Profiling & Resource Measurement**: The executive profiling
  bracket shall dispatch execution based on the case's profile arm: measuring
  elapsed CPU clock cycles, monotonic execution duration in nanoseconds and
  peak
  stack memory consumption under critical section isolation for `Profile::Full`,
  under unmasked interrupts for `Profile::Unmasked`, and executing bare `f()`
  without metric collection for `Profile::Off`. When metrics are measured, they
  are
  transmitted in a `Telemetry::MetricReport` frame following
  `TestState::Passed`;
  for `Profile::Off`, `MetricReport` transmission is omitted.
- **FR-5 - Dynamic Suite Configuration & Parameter Tuning**: The executive shall
  service dynamic parameter mutation requests over the transport, validating
  suite indices, setting indices and value types, rejecting type mismatches
  without corrupting memory or terminating the executive.
- **FR-6 - Failure Diagnostics & Crash Recovery**: The executive failure handler
  shall execute a deterministic, zero-allocation diagnostic capture sequence
  upon unhandled panics or hardware exceptions, format forensic records into
  static buffers without buffer overrun, attribute crashes to active tests and
  coordinate cooperative resets.
- **FR-7 - Target Lifecycle & Hardware Identification**: The executive shall
  query platform hardware identity, emit target identification telemetry,
  service cooperative warm resets and support clean virtual environment
  termination via dedicated exit routines.

#### 2.2 Non-Functional Requirements

- **NFR-1 - Low-Overhead Case Dispatch**: Invoking an empty test case through
  the suite dispatch abstraction must introduce minimal latency (bounded to
  fewer than 20 clock cycles on target architectures) so as not to distort
  micro-benchmark timings.
- **NFR-2 - Nanosecond Telemetry Timestamp Precision**: Diagnostic error logs
  and execution metrics must report timestamps and durations with nanosecond resolution
  from monotonic platform nanosecond clocks.
- **NFR-3 - Timing-Critical Zero-Allocation Logging**: The executive logging,
  state transition, metric reporting, and discovery streaming paths must format
  and transmit telemetry without invoking dynamic allocation or expensive
  on-target string formatting on the hot execution path.

#### 2.3 Constraints

- **C-1 - Strict `#![no_std]`, Zero Heap**: The target executive crate
  (`control-rs-ets`) must compile on bare-metal targets under `#![no_std]` with
  strictly zero dynamic heap allocation.
- **C-2 - Bounded Flash and RAM Footprint**: The target executive implementation
  (excluding test payloads) must operate within the Flash and RAM ceilings
  specified in §7, with test suite descriptors residing in Flash ROM.

---

### 3. Technical Overview

The ETS executive (`Server`) manages interactive target execution, processing
commands
from the host, executing dispatched test cases, managing dynamic parameter
updates and capturing forensic panic state.

```mermaid
flowchart TD
    Host["Host Transport Link"] <==> Comms["HostComms Interface"]

    subgraph ETSExecutive ["ETS Executive (control-rs-ets)"]
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

Serves: FR-1, FR-1.1, FR-1.2, FR-2, FR-2.2, FR-7.2, FR-7.3, NFR-1

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

Dispatch resolves `suite_id` through `self.suites.get(usize::from(suite_id))`
and `test_id` through `suite.executables.get(usize::from(test_id))`.
Discovery chains target info followed by all suites and executable descriptors
registered in `self.suites`. An out-of-bounds suite or test index transmits an
error diagnostic frame to the host without panicking or terminating the server
loop.

#### 4.3. Test Execution and Lifecycle State Transitions

Serves: FR-3, FR-3.1, FR-4, FR-4.1, FR-4.2, FR-4.3, FR-4.4, FR-4.5, NFR-1,
NFR-2

The server's execution engine transitions through discrete operational states
governed by incoming host commands, active test profiling brackets and
fault-interception handlers:

```mermaid
stateDiagram-v2
    [*] --> Idle

    state Idle {
        [*] --> Polling
        Polling --> Polling: None
    }

    Idle --> StreamingDiscovery: ListSuites command
    StreamingDiscovery --> Idle: Descriptors streamed
    Idle --> MutatingSetting: SetSetting command
    MutatingSetting --> Idle: Value applied or type mismatch
    Idle --> RunningTest: RunExecutable valid index
    Idle --> Idle: RunExecutable out of bounds

    state RunningTest {
        [*] --> Initializing: TestState Running emitted
        Initializing --> ProfiledExecution: Paint stack and mask IRQs
        ProfiledExecution --> Evaluating: Invoke test function
        Evaluating --> Finalizing: Test returns
        Finalizing --> [*]: Scan stack and clear trackers
    }

    RunningTest --> Reporting: Normal completion
    Reporting --> Idle: Passed verdict and metrics emitted
    RunningTest --> FaultAttribution: Panic or exception while active
    Idle --> FaultCapture: Panic or exception while idle

    state FaultAttribution {
        [*] --> FormatForensic: Read active indicators
        FormatForensic --> EmitCrashTelemetry: Emit Failed and TargetPanic
    }

    FaultAttribution --> Resetting: Flush transport
    FaultCapture --> Resetting: Emit unlinked TargetPanic
    Idle --> Resetting: TryReset command
    Resetting --> [*]: Target reboot
```

**Figure 4.1**: Target executive lifecycle state machine showing command loop
modes, dispatch guards, atomic context tracking and fault attribution paths.

Within the `RunningTest` state, the server coordinates execution across the
three profiling
arms (`Profile::Full`, `Profile::Unmasked`, `Profile::Off`) using the composed
primitives provided
by `CPUProfiler`:

```rust
let metrics = match exec.profile {
Profile::Full     => Some( self .context.cpu_utils.run_profiled(exec.test_fn)),
Profile::Unmasked => Some( self .context.cpu_utils.run_profiled_unmasked(exec.test_fn)),
Profile::Off      => { (exec.test_fn)(); None }
};

self .context.comms.send_telemetry( & Telemetry::TestStateChange {
suite_id,
test_id,
state: TestState::Passed,
}) ?;

if let Some(m) = metrics {
self .context.comms.send_telemetry( & Telemetry::MetricReport {
suite_id,
test_id,
cycles: m.cycles,
time_ns: m.time_ns,
stack_peak: m.stack_peak,
}) ?;
}
```

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
    alt Profile::Full
        Server ->> + Prof: run_profiled(test_fn) [paint, disable_interrupts, run_clock, scan]
        Prof ->> + Test: Invoke test_fn()
        Test -->> - Prof: Return
        Prof -->> - Server: TestMetrics { cycles, time_ns, stack_peak }
    else Profile::Unmasked
        Server ->> + Prof: run_profiled_unmasked(test_fn) [paint, run_clock, scan]
        Prof ->> + Test: Invoke test_fn()
        Test -->> - Prof: Return
        Prof -->> - Server: TestMetrics { cycles, time_ns, stack_peak }
    else Profile::Off
        Server ->> + Test: Invoke test_fn()
        Test -->> - Server: Return
    end
    Server ->> Ind: set_idle(), set_idle() [δ_ctx_idle]
    Server ->> Host: Telemetry::TestStateChange (TestState::Passed) [δ_tx_pass]
    opt If metrics present (Full or Unmasked)
        Server ->> Host: Telemetry::MetricReport { cycles, time_ns, stack_peak } [δ_tx_metric]
    end
    Server ->> Host: flush()
```

**Figure 4.2**: Profiling dispatch bracket across `Full`, `Unmasked`, and `Off`
arms.

1. Transmits `TestState::Running` and flushes transport
   ($\delta_{\text{tx,run}}$).
2. Marks active indicators (`CURRENT_SUITE.set_active(s)`,
   `CURRENT_TEST.set_active(t)`, $\delta_{\text{ctx,set}}$).
3. Dispatches case execution to the profiler:
    - `Profile::Full`: paints stack, masks interrupts, samples clock before and
      after `test_fn()`, restores interrupts, and scans stack peak.
    - `Profile::Unmasked`: paints stack, leaves interrupts active, samples clock
      before and after `test_fn()`, and scans stack peak (including ISR
      execution time).
    - `Profile::Off`: executes `test_fn()` directly with no stack painting,
      clock sampling, or metric calculation.
4. Clears indicators (`set_idle()`, $\delta_{\text{ctx,idle}}$).
5. Transmits `TestState::Passed` ($\delta_{\text{tx,pass}}$) for all completing
   cases.
6. For `Full` and `Unmasked`, transmits `Telemetry::MetricReport`
   ($\delta_{\text{tx,metric}}$); for `Off`, metric transmission is omitted.
7. Tare baseline: both measured arms run the identical machine code inside
   `run_clock` between cycle samples, preserving a single tare calibration case.

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
4. On type match, applies the updated value using `TripleBuffer<T>`
   interior-mutable descriptors and echoes the updated state via
   `Telemetry::SettingUpdate`.
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

| Condition | Requirement | Method    | Target                                                                                     | Criterion                                                                                                                                                                                                                                                                                |
|:----------|:------------|:----------|:-------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| VC-1.1    | FR-1        | `libtest` | `control_rs_ets::server::tests::test_server_ok_to_reset`                                   | Persistent event loop executes until cooperative reset command is received; FR-1 holds iff both conditions hold                                                                                                                                                                          |
| VC-1.2    | FR-1        | `libtest` | `control_rs_ets::server::tests::test_server_poll_command_error`                            | Polling errors from transport are returned as `Err` and not swallowed                                                                                                                                                                                                                    |
| VC-2.1    | FR-2        | `libtest` | `control_rs_ets::util::tests::test_get_suites_fn`                                          | Test suites are safely reconstructed from linker boundary pointers; FR-2 holds iff all conditions hold                                                                                                                                                                                   |
| VC-2.2    | FR-2        | `libtest` | `control_rs_ets::server::tests::test_server_out_of_bounds`                                 | Invalid suite or test index emits diagnostic error and aborts without panicking                                                                                                                                                                                                          |
| VC-2.3    | FR-2        | `libtest` | `control_rs_ets::server::tests::test_server_discovery`                                     | Discovery command streams target info, suites, executables, and settings                                                                                                                                                                                                                 |
| VC-3.1    | FR-3        | `libtest` | `control_rs_ets::server::tests::test_server_run_test`                                      | Test execution emits `Running` and `Passed` lifecycle transitions; FR-3 holds iff both conditions hold                                                                                                                                                                                   |
| VC-3.2    | FR-3        | `libtest` | `control_rs_ets::server::tests::test_index_indicator`                                      | Thread-safe atomic indicators track active suite and test indices and reset to idle                                                                                                                                                                                                      |
| VC-4.1    | FR-4        | `libtest` | `control_rs_ets::server::tests::profiling_reports_elapsed_cycles_and_nanoseconds`          | Execution bracket reports non-zero elapsed cycles, nanosecond duration, and stack peak; FR-4 holds iff all conditions hold                                                                                                                                |
| VC-4.2    | FR-4        | `libtest` | `control_rs_ets::server::tests::test_server_metric_report_flush_error`                     | Metric report transmission flushes telemetry frames to host transport                                                                                                                                                                                                                    |
| VC-4.3    | FR-4        | `libtest` | `control_rs_ets::server::tests::test_host_cpu_profile_utils`                               | Profiler execution bracket exercises cycle and duration measurement hooks                                                                                                                                                                                                                |
| VC-4.4    | FR-4        | `libtest` | `control_rs_ets::server::tests::test_server_profile_arms`                                  | 3-row table-driven test verifying executive dispatch across `Full`, `Unmasked`, and `Off`: `Full` runs with masked interrupts and sends MetricReport; `Unmasked` runs with unmasked interrupts and sends MetricReport; `Off` runs bare and sends no MetricReport while emitting `Passed` |
| VC-5.1    | FR-5        | `libtest` | `control_rs_ets::server::tests::test_server_set_setting`                                   | Valid setting mutation updates parameter value and echoes state; FR-5 holds iff both conditions hold                                                                                                                                                                                     |
| VC-5.2    | FR-5        | `libtest` | `control_rs_ets::server::tests::test_server_set_setting_type_mismatch`                     | Incompatible value type is rejected with error without mutating state                                                                                                                                                                                                                    |
| VC-6.1    | FR-6        | `libtest` | `control_rs_ets::util::tests::test_failure_buf_writer_fn`                                  | Diagnostic capture formats forensic payload into static buffer without heap allocation; FR-6 holds iff all conditions hold                                                                                                                                                               |
| VC-6.2    | FR-6        | `libtest` | `control_rs_ets::util::tests::the_buffer_writer_fills_its_buffer_exactly_and_refuses_more` | Formatting routine rejects writes exceeding buffer capacity without writing past boundary                                                                                                                                                                                                |
| VC-6.3    | FR-6        | `libtest` | `control_rs_ets::util::tests::test_handle_failure_comms_active`                            | Crash during active test marks active case as failed                                                                                                                                                                                                                                     |
| VC-6.4    | FR-6        | `libtest` | `control_rs_ets::util::tests::test_handle_failure_comms_idle`                              | Crash in idle state suppresses case failure transitions                                                                                                                                                                                                                                  |
| VC-6.5    | FR-6        | `libtest` | `control_rs_ets::util::tests::test_handle_exception_fn`                                    | Exception handler captures location and coordinates cooperative reset                                                                                                                                                                                                                    |
| VC-6.6    | FR-6        | `libtest` | `control_rs_ets::util::tests::test_handle_failure_comms_disabled`                          | Failure handler safely completes target reset when transport is non-functional                                                                                                                                                                                                           |
| VC-7.1    | FR-7        | `libtest` | `control_rs_ets::server::tests::test_server_discovery`                                     | Target info telemetry emits protocol version, board ID, clock frequency, and FPU capabilities; FR-7 holds iff all conditions hold                                                                                                                                                        |
| VC-7.2    | FR-7        | `libtest` | `control_rs_ets::server::tests::test_server_ok_to_reset`                                   | Cooperative reset command triggers target reset sequence                                                                                                                                                                                                                                 |
| VC-7.3    | FR-7        | `libtest` | `control_rs_ets::server::tests::test_server_exit`                                          | Exit routine closes communications before invoking target exit hook                                                                                                                                                                                                                      |
| VC-10.1   | NFR-1       | `libtest` | `control_rs_ets::server::tests::test_server_run_test`                                      | Indirect case dispatch via function pointer introduces low latency                                                                                                                                                                                                                       |
| VC-10.2   | NFR-2       | `libtest` | `control_rs_ets::server::tests::log_timestamps_are_in_nanoseconds`                         | Diagnostic error logs and execution metrics derive timestamps with nanosecond resolution                                                                                                                                                                  |
| VC-10.3   | NFR-3       | `libtest` | `control_rs_ets::util::tests::the_buffer_writer_fills_its_buffer_exactly_and_refuses_more` | Formatting and logging operate with zero dynamic heap allocations on hot path                                                                                                                                                                                                            |
| VC-26.1   | C-1         | `gate`    | `build`                                                                                    | `control-rs-ets` compiles cleanly for bare-metal targets under `#![no_std]`                                                                                                                                                                                                              |
| VC-26.2   | C-2         | `libtest` | `control_rs_ets::tests::test_flash_descriptor_sizes`                                       | Descriptors and runtime state structures reside within bounded memory footprints                                                                                                                                                                                                         |

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
- **Fixed panic buffer (`L-2`)**: `FailureBufWriter` formats payloads into a
  static stack buffer sized per §7; longer messages truncate cleanly at buffer
  capacity without panicking.

---

### 7. Performance & Resource Considerations

| Limit | Formula or value | Class |
|:--|:--|:--|
| Core Flash ceiling | $\le 32$ KB | Chosen |
| Executive Flash share | $\le 12$ KB | Chosen |
| Core static RAM, zero suites | $\le 2$ KB | Chosen |
| Core peak stack | $\le 2$ KB | Chosen |
| Runtime RAM ceiling | $\le 8$ KB | Chosen |
| Suite capacity | 65,536 IDs (65,535 user suites + `diagnostics` at 0) | Fixed |
| Idle poll interval | $\le 100\ \mu\text{s}$ | Chosen |
| Fault record transmission start | $\le 1\ \text{ms}$ | Chosen |
| Fault target reset or halt | $\le 1\ \text{s}$ | Chosen |
| Software reset latency | $\le 10\ \text{ms}$ | Chosen |
| Target ready post-reset | $\le 1\ \text{s}$ | Chosen |
| Emulator exit latency | $\le 100\ \text{ms}$ | Chosen |

- **Footprint budget**: 12 KB executive + 20 KB buffer/profiler/comms fits within 32 KB Flash; 2 KB static RAM + 2 KB stack leaves 4 KB headroom inside 8 KB ceiling.
- **Latency budgets**: 100 µs idle poll balances responsiveness with CPU load; 1 s timeout bounds unacknowledged crash recovery.
- **Nanosecond precision**: All telemetry timestamps and durations are reported in integer nanoseconds (`time_ns`).

*All executive routines operate on statically allocated memory with zero dynamic heap allocation.*

---

### 8. Risks & Open Questions

- **Watchdog Integration**: Currently, runaway test loops rely on host session
  timeouts; future hardware watchdog integration will enforce automated target
  reboots.

---

### 9. Development Plan

| Phase                              | Delivers                                         | Requirements          | Effort | Status  |
|:-----------------------------------|:-------------------------------------------------|:----------------------|:-------|:--------|
| 1. Executive Event Loop & Dispatch | Core polling loop and case dispatch abstraction  | FR-1, FR-2, NFR-1     | 3 days | Shipped |
| 2. Lifecycle & Profiling Bracket   | State machine, atomic context tracking, metrics  | FR-3, FR-4, NFR-2     | 3 days | Shipped |
| 3. Panic Interception & Settings   | Forensic buffer writer, crash recovery, settings | FR-5, FR-6, FR-7, C-1 | 4 days | Shipped |

---

### 10. Revision History

| Revision | Date              | Author          | Description                                                                                                                                            |
|:---------|:------------------|:----------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1.0      | November 10, 2026 | @MitchellDScott | Initial component design doc extracted from lead subsystem architecture.                                                                               |
| 2.0 | November 18, 2026 | Antigravity | Report durations in nanoseconds, relocate panic partition sentence to host-comm, and specify footprint in §7. Affects C-1, C-2, FR-1, FR-2, FR-3, FR-4, FR-5, FR-6, FR-7, NFR-1, NFR-2, NFR-3. |

---

## References

None.
