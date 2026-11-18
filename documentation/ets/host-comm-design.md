# HostComms Trait and Wire Protocol (host-comm)

![Date Badge](https://img.shields.io/badge/Date-August_29,_2026-blue)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)
![Design Status Badge](https://img.shields.io/badge/Design%20Status-Draft-orange)

---

### 1. Introduction

The `HostComms` trait provides a unified, hardware-agnostic communication
abstraction
for target firmware operating in resource-constrained environments. This allows
the
Embedded Test Server (ETS) running on-target to separate physical transport
mechanics (such as UART, USB CDC, TCP or debug probes) from test execution and
telemetry dispatch.
Host-side demultiplexing, session management and decoding are owned by
`control-rs-ets-host::Session` (`ets-host-design.md`), while user presentation
is handled by host dashboards or CLI frontends.

---

### 2. Requirements

#### 2.1 Functional Requirements

- **FR-1 — Target Transport Abstraction**: The target firmware must expose a
  hardware-agnostic trait (`HostComms`) to poll commands, transmit telemetry,
  and flush data without coupling test execution logic to physical peripheral
  hardware.
- **FR-2 — Bidirectional Framed Packet Exchange**: The system must support
  host-to-target command packets and target-to-host telemetry packets over a
  shared byte stream using compact binary framing (`0xAA 0x55` sync header,
  2-byte big-endian length, `postcard` payload and 2-byte CRC-16 checksum).
- **FR-4 — Explicit Wire-Contract Revision**: The protocol carries an explicit
  `PROTOCOL_VERSION` revision identifier in target telemetry that a host checks
  before acting on any payload, preventing misinterpretation of schema changes.
- **FR-5 — Target Hardware and Environment Telemetry**: The target must transmit
  hardware identity and environment metadata—including board identifier, core
  clock frequency and FPU capability flags—enabling the host to interpret
  timing and platform capabilities without local out-of-band assumptions.
- **FR-6 — Catastrophic Failure and Panic Capture**: The target wire schema must
  represent panics and hardware exceptions (`PanicReport` with source file, line
  number, and diagnostic message) for transmission as prioritized telemetry
  before halting or rebooting.
- **FR-7 — Multiplexed Firmware Log Stream**: A `HostComms` implementation may
  carry the firmware's own log output on the same physical communication link,
  and the host separates that text from telemetry frames. ETS defines no log
  API.
- **FR-9 — Wire Contract Golden Vectors**: The wire protocol defines stable,
  append-only `postcard` schemas for `Command` and `Telemetry`, verified against
  checked-in golden byte vectors.

#### 2.2 Non-Functional Requirements

- **NFR-1 — Low Latency & Determinism**: Telemetry serialization and framing
  execute deterministically in bounded single-pass operations
  without unbounded loops, introducing $\le 10\ \mu\text{s}$ jitter into the
  nominal 1 kHz control loop deadline.
- **NFR-2 — Bounded Stack & Predictable Footprint**: Serialization and framing
  paths operate within bounded, pre-allocated stack envelopes without recursion
  or dynamic heap allocation.
- **NFR-3 — Stream Resynchronization & Framing Integrity**: Corrupted or
  truncated bytes before or between sync headers are discarded cleanly, and
  `FrameReader` resynchronizes automatically on the next valid sync header.

#### 2.3 Constraints

- **C-1 - Strict `#![no_std]` and Zero Heap Allocation**: The target firmware
  library must compile under `#![no_std]` with strictly zero dynamic heap
  allocation.
- **C-2 — Frame Size Envelope**: Maximum payload length is bounded at
  `MAX_PAYLOAD_SIZE = 512` bytes (`MAX_FRAME_SIZE = 518` bytes) to prevent
  buffer
  overruns on memory-constrained microcontrollers.
- **C-3 — Physical Interface Scope**: Target transport implementations are
  restricted to standard embedded peripherals (UART, USB CDC, Ethernet MAC/PHY)
  or hardware debug probes (SWD/JTAG).

---

### 3. Technical Overview

The `HostComms` architecture focuses on the **target-side (microcontroller)**
execution environment:

1. **Target-side Trait & Drivers**: A zero-cost abstraction trait (`HostComms`)
   and associated concrete peripheral drivers (UART DMA, USB CDC, TCP)
   that pack telemetry data, frame packets and poll for inbound commands
   without dynamic
   allocation.
2. **Wire Framing & Postcard Payloads**: Compact binary framing (`0xAA 0x55`
   sync header, length,
   payload, CRC-16) wrapping `postcard`-serialized Rust enum
   schemas [1], [3], [4].
3. **Serial Monitor Text Compatibility**: Built-in ASCII parsing (`FromStr`) and
   formatting (`Display`)
   for commands and telemetry directly on the enum schemas, allowing standard
   serial terminal interaction.
4. **Host Counterpart**: The target-side stream is consumed on the PC host by
   `control-rs-ets-host::Session` (`ets-host-design.md`), which
   feeds decoded events to host frontends or automated CI runners.

```mermaid
graph TD
    subgraph "Target MCU (Interactive Server)"
        SRV[Server Event Loop]

        subgraph Middleware Trait
            HC_TRAIT[HostComms]
        end

        subgraph Concrete Implementations
            UART[UartDriver / USB CDC]
            ETH[TcpDriver]
            CUST[CustomDriver]
        end

        SRV <-->|" poll_command() \n send_telemetry() \n flush() "| HC_TRAIT
        HC_TRAIT <-.-> UART
        HC_TRAIT <-.-> ETH
        HC_TRAIT <-.-> CUST
    end

    UART <-->|" Serial Port / USB CDC "| SESS
    ETH <-->|" TCP Socket "| SESS
    CUST <-->|" Custom Stream "| SESS

    subgraph "Host Environment"
        SESS["control-rs-ets-host (Session)"]
        UI["control-rs-tui (TUI)"]
        SESS -->|Events & Logs| UI
        UI -->|Commands| SESS
    end
```

---

### 4. Architecture

#### 4.1. Middleware Trait

Serves: FR-1

The core of the abstraction is the `HostComms` trait. Implementations of this
trait encapsulate the hardware-specific details of reading and writing bytes:

```rust
pub trait HostComms {
    /// The error type associated with transport failures.
    type Error;

    /// Closes the communication interface (for example, signaling semihosting exit).
    fn close(&mut self) {}

    /// Closes the communication interface with a failure or error status.
    fn close_on_failure(&mut self) {}

    /// Flush any pending buffered data out to the physical interface.
    fn flush(&mut self) -> SendResult<Self::Error>;

    /// Read incoming bytes and try to parse a Command.
    ///
    /// This should be non-blocking.
    fn poll_command(&mut self) -> PollResult<Self::Error>;

    /// Send a telemetry message to the host.
    fn send_telemetry(
        &mut self,
        telemetry: &Telemetry<'_>,
    ) -> SendResult<Self::Error>;
}
```

The associated type `Error` allows concrete drivers to bubble up
hardware-specific failures (for example, framing errors, overflow flags or
socket
disconnects) to the calling Server loop.

#### 4.2. Command Schema & Binary Serialization

Serves: FR-2, FR-4, FR-5, FR-6, FR-9, C-1, C-2

Commands originating from the host are defined by a strict, shared Rust schema
to ensure structural alignment:

```rust
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Command {
    ListSuites,
    RunExecutable {
        suite_id: u16,
        test_id: u16,
    },
    SetSetting {
        setting_id: u16,
        suite_id: u16,
        value: SettingValue,
    },
    TryReset,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Telemetry<'a> {
    DiscoveryComplete,
    Log(LogMessage<'a>),
    MetricReport(MetricReport),
    SettingInfo(SettingInfo<'a>),
    SuiteInfo(SuiteInfo<'a>),
    TargetPanic(PanicReport<'a>),
    TestInfo(TestInfo<'a>),
    TestStateChange(TestStateChange),
    TargetInfo {
        protocol_version: u8,
        board_id: u16,
        core_clock_hz: u32,
        fpu_flags: u8,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct TestInfo<'a> {
    pub description: &'a str,
    pub name: &'a str,
    pub suite_id: u16,
    pub test_id: u16,
    pub profile: Profile,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MetricReport {
    pub cycles: u64,
    pub stack_peak: u32,
    pub suite_id: u16,
    pub test_id: u16,
    pub time_us: u64,
}
```

`TestInfo` carries the `Profile` arm (`Full = 0`, `Unmasked = 1`, `Off = 2`) so
the host
knows during discovery whether to expect a subsequent `MetricReport` and whether
the numbers
include interrupt handlers. `MetricReport` retains its full schema: metrics are
all present (emitted for `Full` and `Unmasked`) or all absent (omitted for
`Off`, which emits only
`TestStateChange(TestState::Passed)`).

To minimize parsing overhead and memory usage in a `#![no_std]` environment, the
architecture uses the `postcard` crate [1].

Postcard achieves a minimal wire footprint through:

* **Varints**: Variable-length integer encoding that compresses smaller
  numbers into fewer bytes [1].
* **Endianness**: Explicit Little-Endian byte ordering matching native
  microcontroller architectures (ARM Cortex-M, RISC-V).
* **Zero Allocation**: Serialization directly into pre-allocated static
  buffers [1].

##### 4.2.1. Discriminant Stability and `PROTOCOL_VERSION`

`postcard` encodes a tagged union as a `varint(u32)` discriminant followed by
the variant value [1], [2]. Variant order and field order are therefore part of
the wire contract: inserting a variant ahead of an existing one renumbers every
later discriminant.

FR-4 and FR-5 are discharged by `PROTOCOL_VERSION` and `Telemetry::TargetInfo`,
both in `control-rs-ets::comms`. `PROTOCOL_VERSION` is a `u8` constant in
`control-rs-ets::comms` incremented whenever `Command` or `Telemetry` gains,
loses, or reorders a variant, or whenever a variant's payload changes shape.
With the addition of `profile: Profile` to `TestInfo`, `PROTOCOL_VERSION` is
incremented to `2`.
The target transmits `Telemetry::TargetInfo` carrying `protocol_version`,
`board_id`, `core_clock_hz`, and `fpu_flags` (from `Platform`,
`embedded-test-server-design.md` §4.6) as the first frame of discovery. A host
that receives a `protocol_version` value other than its own compiled constant
terminates the session immediately and reports the mismatch; it does not attempt
to decode further frames. `control-rs-ets-host` implements the host half of this
check, while `control-rs-tui` consumes the board and clock metadata to populate
the header.

- Variants are appended, never inserted or reordered.
- Existing variant payload fields and their order are immutable.

#### 4.3. Direct Text Line Compatibility

Serves: FR-2, NFR-2

Standard string formatting on a microcontroller is computationally and
space-expensive,
bloating target flash memory with static templates and consuming excessive CPU
cycles
to interpolate values. In this baseline, log messages are captured as structured
byte
or string slices within `postcard`-serialized enums with no on-target string
formatting (see `embedded-test-server-design.md` §7). Deferred formatting via
external compression
crates is deferred to future work (see §6.3):

- `Command::from_str` parses plain human commands:
    - `list` -> `Command::ListSuites`
    - `run <suite> <test>` -> `Command::RunExecutable`
    - `set <suite> <param> <val>` -> `Command::SetSetting`
    - `reset` -> `Command::TryReset`
- `Telemetry::fmt` writes human-readable ASCII lines:
    - `INFO: <suite_name> test_count=<N>`
    - `PASS: <suite_id>.<test_id> (cycles=<C>, time_us=<T>, stack=<S>)`
    - `PANIC: <file>:<line>: <message>`

#### 4.4. Packet Framing & Consistency

Serves: FR-2, NFR-1, NFR-2, NFR-3, C-2

Serial channels like UART operate as continuous byte streams without packet
boundaries. To segment individual commands and telemetry logs, the system
uses a custom framed binary packet structure with Sync Headers and CRC-16
integrity validation:

* **Sync Header**: 2 bytes (`0xAA 0x55`) used to identify the beginning of a
  frame.
* **Payload Length**: 2 bytes (big-endian) defining the size of the serialized
  postcard payload.
* **Postcard Payload**: The serialized data block (limited to a maximum of 512
  bytes).
* **Integrity Check**: A 2-byte (big-endian) CRC-16 checksum (calculated using
  `CRC_16_IBM_SDLC` [3], [7]) appended at the end of the frame payload, computed
  strictly over the `postcard` payload slice (`payload[..len]`).
* **Error Detection Scope & Bounds**: By the burst-detection theorem [6, Theorem 2],
  CRC-16 detects all burst errors of length $b \le 16\text{ bits}$. For random bit
  inversions over payloads up to 512 bytes (4,096 bits), the 0x1021 polynomial
  guarantees a minimum Hamming Distance of 4 ($HD \ge 4$), detecting all 1-, 2-, and
  3-bit errors [5]. Header errors outside the CRC span are detected and rejected by
  reader state validation: byte sequence desynchronization in the sync header resets
  the reader to `WaitStart1`, and corrupted length prefixes that exceed the buffer
  ceiling ($> 512\text{ B}$) immediately abort the frame.
* **Single-Pass Decoding**: The host `FrameReader` processes incoming streams
  statefully byte-by-byte, checking for the sync header, validating lengths and
  confirming the CRC-16 checksum before deserializing. Stream bytes received
  outside a frame while waiting in `WaitStart1` are forwarded to host log output
  (printed by the runner and displayed in the TUI log pane) while the reader
  continues decoding telemetry frames.

#### 4.5. Target Peripheral Driver Integration

Serves: FR-1, FR-7, C-3

Target board implementations implement the `HostComms` trait by wrapping
platform-specific peripheral drivers (such as UART DMA, USB CDC or semihosting
byte streams):

- Non-blocking command polling (`poll_command`) reads available bytes from the
  underlying peripheral and feeds them through `FrameReader`.
- Telemetry transmission (`send_telemetry`) serializes the packet via
  `postcard`,
  applies sync headers (`0xAA 0x55`), length prefix and CRC-16 checksum and
  writes
  to the transport buffer.
- Buffer flushing (`flush`) ensures pending transmissions complete before the
  target enters a low-power wait or begins time-critical execution. A
  `HostComms`
  implementation may carry the firmware's own log output on the same link, and
  the host separates that text from telemetry frames. ETS defines no log API.
  Draining log buffers occurs inside `flush()`, which the server runtime calls
  on every loop iteration and after test execution.

#### 4.6. Host Interoperability

Serves: FR-4, FR-5, C-3

On the host side, `control-rs-ets-host::ETSBridge`
(`../ets-host/ets-host-design.md`) consumes the byte stream generated by
`HostComms`. Standard environments use `serial2` for direct serial (UART/USB
CDC) port communication and a spawned subprocess for QEMU (`ets-host-design.md`
FR-2). RTT buffer polling over SWD/JTAG (for example
through `probe-rs`) is not an `ETSBridge` transport; `ets-host-design.md` §5
rejects it as the primary link. The target-side `HostComms` implementation
remains agnostic to which host mechanism reads its framed stream.

---

### 5. Alternatives

* **Text-Based Serialization (JSON, XML)**: Considered for ease of debugging but
  rejected as primary wire format due to bandwidth and allocation overhead.
  Text compatibility is instead provided through zero-allocation `FromStr` and
  `Display`
  methods on the native enum types.
* **Protobuf & FlatBuffers**: Protobuf requires a memory-allocation-heavy
  deserialization step, which is unacceptable under strict `#![no_std]`.
  Postcard was selected for its native Rust type mapping, zero-allocation design
  and variable-length integer compression [1].
* **Standard Printing (`printf`)**: Rejected for primary telemetry because it
  stores
  large static format string templates in Flash and consumes excessive CPU
  cycles.

---

### 6. Verification & Validation

#### 6.1 Verification

| Condition | Requirement | Method    | Target                                                                                                                                                                                                                     | Criterion                                                                                                                         |
|:----------|:------------|:----------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------|
| VC-1.1    | FR-1        | `libtest` | `control_rs_ets::comms::tests::test_host_comms_defaults`                                                                                                                                                                   | `HostComms` trait methods (`poll_command`, `send_telemetry`, `flush`, `close`, `close_on_failure`) execute cleanly on drivers     |
| VC-2.1    | FR-2        | `libtest` | `control_rs_ets::comms::tests::test_frame_encoder_payload_and_command`, `control_rs_ets::comms::tests::test_frame_telemetry_round_trip_and_corruption`, `control_rs_ets::comms::tests::test_frame_reader_invalid_checksum` | All variants round-trip through `FrameEncoder` and `FrameReader`; corrupted frames are rejected                                   |
| VC-2.2    | FR-2        | `libtest` | `control_rs_ets::comms::tests::test_frame_telemetry_byte_layout`                                                                                                                                                           | State transition telemetry packets serialize with exact byte layout and CRC-16 checksum verification                              |
| VC-2.3    | FR-2        | `libtest` | `control_rs_ets::comms::tests::test_frame_telemetry_length_accounting_across_variants`                                                                                                                                     | Telemetry payload variants serialize within wire format length bounds without truncation                                          |
| VC-2.4    | FR-2        | `libtest` | `control_rs_ets::comms::tests::the_length_field_is_big_endian_across_both_bytes`                                                                                                                                           | Frame header length fields serialize in big-endian byte order with CRC-16 integrity verification                                  |
| VC-4.1    | FR-4        | `libtest` | `control_rs_ets::comms::tests::test_golden_wire_vector_target_info`                                                                                                                                                        | `Telemetry::TargetInfo` carries `PROTOCOL_VERSION` at a fixed discriminant so version mismatches are detectable                   |
| VC-5.1    | FR-5        | `libtest` | `control_rs_ets::comms::tests::test_golden_wire_vector_target_info`                                                                                                                                                        | `Telemetry::TargetInfo` serializes `protocol_version`, `board_id`, `core_clock_hz`, and `fpu_flags`                               |
| VC-6.1    | FR-6        | `libtest` | `control_rs_ets::comms::tests::test_golden_wire_vectors_telemetry`                                                                                                                                                         | `Telemetry::TargetPanic` captures source file, line number, and message and matches its golden wire encoding                      |
| VC-6.2    | FR-6        | `libtest` | `control_rs_ets::comms::tests::test_frame_telemetry_round_trip_and_corruption`                                                                                                                                             | Panic telemetry frame serialization round-trips through postcard encoding with CRC-16 checksum integrity                          |
| VC-7.1    | FR-7        | `libtest` | `control_rs_ets::comms::tests::test_frame_reader_interleaved_log_stream_and_telemetry`                                                                                                                                     | Byte stream mixing raw firmware log text and valid telemetry frames separates into the exact expected text and telemetry payloads |
| VC-9.1    | FR-9        | `libtest` | `control_rs_ets::comms::tests::test_golden_wire_vectors_telemetry`, `control_rs_ets::comms::tests::test_golden_wire_vector_list_suites`                                                                                    | Serialized byte representations of `Telemetry` and `Command::ListSuites` match checked-in golden vectors                          |
| VC-9.2    | FR-9        | `libtest` | `control_rs_ets::comms::tests::test_golden_wire_vector_run_executable`                                                                                                                                                     | `Command::RunExecutable` wire representation matches its golden byte vector                                                       |
| VC-9.3    | FR-9        | `libtest` | `control_rs_ets::comms::tests::test_golden_wire_vector_try_reset`                                                                                                                                                          | `Command::TryReset` wire representation matches its golden byte vector                                                            |
| VC-10.1   | NFR-1       | `libtest` | `control_rs_ets::comms::tests::test_frame_telemetry_byte_layout`                                                                                                                                                           | Single-pass framing and CRC-16 calculation execute in bounded linear time over the payload slice                                  |
| VC-11.1   | NFR-2       | `gate`    | `clippy`                                                                                                                                                                                                                   | Serialization and frame decoding allocate zero heap memory and operate within bounded stack space                                 |
| VC-12.1   | NFR-3       | `libtest` | `control_rs_ets::comms::tests::test_frame_reader_resync_after_orphan_start_byte`, `control_rs_ets::comms::tests::test_frame_reader_invalid_start_bytes`                                                                    | `FrameReader` resynchronizes on valid sync bytes after encountering arbitrary garbage or orphan start bytes                       |
| VC-13.1   | C-1         | `gate`    | `clippy`                                                                                                                                                                                                                   | Code compiles under `#![no_std]` without external allocator                                                                       |
| VC-14.1   | C-2         | `libtest` | `control_rs_ets::comms::tests::a_maximum_size_payload_frames_and_decodes`, `control_rs_ets::comms::tests::framing_needs_room_for_the_whole_frame`                                                                          | Enforces `MAX_PAYLOAD_SIZE = 512` and `MAX_FRAME_SIZE = 518` bounds                                                               |
| VC-14.2   | C-2         | `libtest` | `control_rs_ets::comms::tests::an_oversized_payload_is_refused`, `control_rs_ets::comms::tests::test_frame_telemetry_rejects_short_buffers`                                                                                | Payloads exceeding `MAX_PAYLOAD_SIZE` or buffers shorter than the frame length are refused before framing                         |
| VC-15.1   | C-3         | `gate`    | `virtual-ets`                                                                                                                                                                                                              | Target firmware streams framed `HostComms` telemetry over serial/semihosting links across QEMU targets                            |

Target: 90% line coverage of the `comms` module, measured with
`cargo coverage`.

#### 6.2 Acceptance

| Claim                     | Oracle                                             | Measure                             | Bound         |
|:--------------------------|:---------------------------------------------------|:------------------------------------|:--------------|
| Round-trip fidelity       | Serializer output re-read by deserializer          | Field equality across every variant | Exact         |
| Golden vector stability   | Checked-in bytes per variant                       | Byte equality                       | Exact         |
| Corrupted frame rejection | Frame with flipped payload byte                    | Frames delivered upward             | 0             |
| Version mismatch refused  | Discovery response with foreign `PROTOCOL_VERSION` | Session outcome                     | Refused       |
| Serialization stack cost  | Stack measurement across encode path               | Peak bytes                          | Within budget |
| Allocation freedom        | Source inspection                                  | Allocator symbols on transport path | 0             |

#### 6.3 Limits

- Compatibility across protocol revisions. Cross-revision compatibility is not
  supported; differing `PROTOCOL_VERSION` values result in session termination.
- Hardware watchdog integration. Target reset operates cooperatively via
  `Command::TryReset`.

### 7. Performance & Resource Considerations

* **Buffer Bounds**: Message buffers are sized to `MAX_FRAME_SIZE = 518` bytes,
  fitting comfortably in small embedded RAM targets without dynamic heap.
* **Deterministic Single-Pass Framing**: Framing and CRC-16 calculations execute
  in bounded linear time over static buffers without memory allocation, ensuring
  minimal jitter on active execution loops.

---

### 8. Risks & Open Questions

* **Line Noise**: Asynchronous serial links may experience noise bursts. The
  combination
  of sync headers and CRC-16 ensures corrupted packets are dropped rather than
  executed.
* **Interleaved Log Stream Separation**: A `HostComms` implementation may carry
  firmware log text and telemetry packets across a single shared stream. If raw
  log text contains the two-byte sequence `0xAA 0x55`, CRC-16 verification
  rejects the false frame start and the reader resynchronizes cleanly, ensuring
  an interleaved byte stream emerges as exactly the original log text and exactly
  the valid telemetry frames.

---

### 9. Development Plan

| Phase                             | Delivers                                                                                  | Requirements                                         | Effort | Status  |
|:----------------------------------|:------------------------------------------------------------------------------------------|:-----------------------------------------------------|:-------|:--------|
| 1. Core Serialization & Framing   | Postcard schemas, sync-header framing, and CRC-16 verification in `control-rs-ets::comms` | FR-1, FR-2, FR-7, NFR-1, NFR-2, NFR-3, C-1, C-2, C-3 | 2 days | Shipped |
| 2. Wire Contract & Golden Vectors | `PROTOCOL_VERSION`, `TargetInfo`, `TargetPanic`, and golden byte vectors                  | FR-4, FR-5, FR-6, FR-9                               | 2 days | Shipped |

---

### 10. Revision History

| Revision | Date              | Author          | Description                                                                                                                                                        |
|:---------|:------------------|:----------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1.0      | August 29, 2026   | @MitchellDScott | `HostComms` transport trait, telemetry framing and the `serial2` host transport.                                                                                   |
| 2.0      | November 8, 2026  | @MitchellDScott | Requirement traceability matrix, PROTOCOL_VERSION and TargetInfo wire contract, and golden wire vectors. Affects C-2, FR-2, FR-3, FR-6, FR-9, NFR-1, NFR-2, NFR-3. |
| 3.0 | November 18, 2026 | Antigravity | Clarify CRC-16 payload coverage, burst error and Hamming distance detection bounds. Affects FR-7. |

---

## References

[1] J. Munns, "Wire format," *The postcard wire specification*, 2026. [Online].
Available: https://postcard.jamesmunns.com/wire-format. Accessed: Sep. 9, 2026.

[2] Serde contributors, "Enum representations," *Serde documentation*, 2026.
[Online]. Available: https://serde.rs/enum-representations.html. Accessed:
Sep. 9, 2026.

[3] crc-rs contributors, "*crc*: Rust implementation of CRC with support of
various standards," 2026. [Online]. Available: https://docs.rs/crc. Accessed:
Sep. 9, 2026.

[4] L. L. Peterson and B. S. Davie, *Computer Networks: A Systems Approach*, 6th
ed., 2024. [Online].
Available: https://book.systemsapproach.org/. Accessed: Jun. 27, 2026.

[5] P. Koopman, "CRC Polynomial Zoo: 16-bit CRCs," Carnegie Mellon University, 2026.
[Online]. Available: https://users.ece.cmu.edu/~koopman/crc/crc16.html. Accessed:
Nov. 18, 2026.

[6] W. W. Peterson and D. T. Brown, "Cyclic codes for error detection," *Proc. IRE*,
vol. 49, no. 1, pp. 228-235, Jan. 1961, doi: 10.1109/JRPROC.1961.287814.

[7] G. Cook, "Catalogue of parametrised CRC algorithms: Legend," SourceForge, 2026.
[Online]. Available: https://reveng.sourceforge.io/crc-catalogue/legend.htm.
Accessed: Nov. 18, 2026.
