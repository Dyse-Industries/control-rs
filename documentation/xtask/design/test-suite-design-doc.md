# Exportable Test Suites

![Date Badge](https://img.shields.io/badge/Date-May_23,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Needs%20Review-yellow)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

## 1. Context & Objective

This design establishes the structural requirements for suites, ensuring
seamless integration with the distributed test discovery mechanism and the
`hil_suite` runners defined within the `xtask` architecture. Functionally, this
structure acts as the adapter layer, enabling the runner to reliably parse,
discover, and interact with each individual suite, specifically serving the
distributed test discovery mechanism and `xtask` runners.

## 2. Distributed Test Discovery

Using custom linker sections in `no_std` embedded Rust allows tests and
benchmarks to be discovered automatically at compile/link time. By avoiding
standard desktop runtime registries, this pattern provides:

* **Zero Boilerplate:** Developers can declare test or benchmark suites across
  multiple files without maintaining a central registry.
* **Zero Runtime Overhead:** The suite registry is built entirely during
  linking.
* **ROM Efficiency:** Test descriptors are stored directly in Flash memory.

### 2.1. Build Script Injection and Linker Section Mechanics

To facilitate this process, a `build.rs` script is used to "inject" suite
descriptions into a known location in memory. This script parses
project files and generates the necessary linkable assets.

The build script generates `memory.x` linker script fragments that define the
custom `.hil_test_suites` section where the test registry is placed. These
fragments must be included in the end-user's build via
`#[link_section = ".hil_test_suites"]`.

To ensure the custom test registry sections aren't silently discarded by the
linker's garbage collection during a `--release` build, we must explicitly
retain them. This is typically achieved by adding `--gc-keep-exported` to the
linker flags or utilizing `KEEP()` directives in the generated linker scripts
for the `.hil_test_suites` section.

## 3. Execution Lifecycle & Telemetry

The runner's interaction with the suites during a session is managed through a
simple state-machine logic.

### State Tracking

Mirroring standard `cargo test` idioms, test functions are expected to `panic!`
upon failure. The runner evaluates test status by recording start and end
timestamps for each execution:

* **Pending**: No start timestamp is recorded.
* **Running/Failed**: A start timestamp is recorded without a corresponding end
  timestamp. The runner uses other mechanisms to notify the user of the
  failure.
* **Passed**: Both start and end timestamps are successfully recorded.

## 4. Architectural Implementation

The underlying architecture relies on explicit structs and traits designed for
a bare-metal environment to build the suite registry.

### 4.1 The SuiteDescriptor

The `SuiteDescriptor` struct is the primary bridge between the test runner and
the test suite. It is placed in a special
`.hil_test_suites` linker section, allowing the runner to discover and aggregate
all available suites. It holds pointers to the suite's executables and its
configurable settings.

```rust
// The descriptor struct itself
pub struct SuiteDescriptor {
    pub name: &'static str,
    pub executables: &'static [ExecDescriptor],
    pub settings: &'static [&'static dyn Setting],
}

// Example of a manually constructed suite for discovery
#[link_section = ".hil_test_suites"]
#[used]
static SUITE_DESCRIPTOR: SuiteDescriptor = SuiteDescriptor {
    name: "device_connectivity_suite",
    executables: &[/* ... test descriptors ... */],
    settings: &[
        &CONNECTION_TIMEOUT_MS, // These are trait objects
        &MAX_RETRIES,
    ],
};
```

### 4.2 Type-Safe Wrappers & Concurrency

To enable safe interior mutability for settings without using `static mut`,
robust, type-safe atomic wrappers are used. For bare-metal contexts,
`core::sync::atomic` types provide a zero-cost, `unsafe`-free abstraction.

```rust
// A wrapper for a u32 setting
pub struct AtomicU32Setting {
    name: &'static str,
    value: AtomicU32,
}

impl AtomicU32Setting {
    pub const fn new(name: &'static str, initial_value: u32) -> Self {
        Self {
            name,
            value: AtomicU32::new(initial_value),
        }
    }
}

impl Setting for AtomicU32Setting {
    fn name(&self) -> &'static str { self.name }
    fn expected_type(&self) -> SettingType { SettingType::U32 }

    fn get(&self) -> SettingValue {
        SettingValue::U32(self.value.load(Ordering::Relaxed))
    }

    fn set(&self, value: SettingValue) -> Result<(), &'static str> {
        if let SettingValue::U32(v) = value {
            self.value.store(v, Ordering::Relaxed);
            Ok(())
        } else {
            Err("Type mismatch: expected U32")
        }
    }
}

// Similar wrappers are defined for AtomicU8Setting, etc.
```

### 4.3 Core Traits (Setting)

Finally, dynamic dispatch via vtables allows the runner to interact with a
heterogeneous collection of setting types through a common interface. The core
`Setting` trait encapsulates the required behavior (getting/setting), while
`SettingType` and `SettingValue` enums provide a type-safe way to communicate
values.

```rust
use core::sync::atomic::{AtomicU32, AtomicU8, Ordering};

// The data structure used for communication with the runner/host
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SettingValue {
    U8(u8),
    U32(u32),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SettingType {
    U8,
    U32,
}

// The core trait all settings must implement.
// `Sync` is required so the implementors can be placed in `static` memory.
pub trait Setting: Sync {
    fn name(&self) -> &'static str;
    fn expected_type(&self) -> SettingType;
    fn get(&self) -> SettingValue;
    fn set(&self, value: SettingValue) -> Result<(), &'static str>;
}
```