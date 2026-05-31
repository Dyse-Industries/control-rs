# Macros Design Document

**Implementation Order:** 4
**Estimated Time:** 3 days

![Date Badge](https://img.shields.io/badge/Date-May_24,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Needs%20Review-yellow)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

## 1. Context & Objective

The objective of `control-rs-macros` is to provide procedural macros (
`#[hil_suite]`, `#[hil_setup]`)
for distributed test discovery. Because `control-rs` will use these macros, they
are separated from the HIL and CI runners, but they remain closely related
and are kept within the same workspace. These macros automate the construction
of the underlying test suite architecture.

## 2. Developer Experience (The API)

### Declaring a Suite

A suite is declared by annotating a module with the `#[hil_suite]` procedural
macro. All functions within the module will be automatically discovered and
registered by the procedural macro to be used by the underlying distributed test
discovery mechanism.

```rust
#[hil_suite]
pub mod device_connectivity_suite {
    // Configurable settings exposed to the user
    pub static CONNECTION_TIMEOUT_MS: u32 = 5000;
    pub static MAX_RETRIES: u8 = 3;

    fn test_device_discovery() {
        // ... test logic ...
    }
}
```

The tests and benchmarks will then be automatically aggregated into
`SuiteDescriptor`s.

Furthermore, the build script supports an environment variable overriding
feature, allowing users to dynamically modify the location in memory that
descriptors will be stored.

### Configurable Settings Translation

Suites expose configurable settings—mutable parameters that users can adjust
between test iterations. These are declared as simple `static` variables within
the suite module. The macro handles the underlying complexity of making them
safely mutable at runtime by translating simple user declarations (e.g.,
`static MY_SETTING: u32 = 10;`)
into robust, type-safe atomic wrappers (like `AtomicU32Setting`).

### Macro-Generated Registration Code

The clean, declarative API is enabled by macro-generated code. The procedural
macro generates the `SuiteDescriptor` struct and places it in the appropriate
linker section for the test suite registry.

```rust
// Conceptual macro-generated code
pub mod device_connectivity_suite {
    // ...

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
}
```

## 3. Execution Context and Setup

To ensure maximum compatibility across different development boards and
architectures, the `#[hil_setup]` function initializes and returns a `Context`
object. This manages test execution and communication:

* **Settings:** Reconfigurable parameters (e.g., test durations, verbosity) that
  allow dynamic adjustments without recompiling.
* **HostComms:** A generic trait acting as a middleware layer for
  firmware-to-host communications.
* **ClientClock:** A generic trait that allows users to configure different
  hardware timers for the runner. The context expects a timer or clock from the
  user's setup function to enable precise, hardware-specific performance metrics
  and benchmarking.

### 3.1. Panic Handling and Entrypoint Generation

The `#[hil_setup]` macro serves a dual purpose: it automatically creates the
standard `main()` entrypoint that calls the user's setup code and launches the
HIL server, and it also registers a custom panic handler. This ensures that any
test or benchmark failures resulting in a panic are caught gracefully. The panic
handler intercepts the error, serializes the panic message and location using
the `HostComms`, and transmits it back to the host TUI, preventing the MCU
from silently locking up and allowing the host to retain the state across
restarts.

## 4. Usage Example

```rust
use teensy4_bsp::hal::timer::Blocking;
use control_rs::macros::{hil_suite, hil_setup};
use control_rs::hil::{Command, Context, Settings, HostComms, ClientClock, LogMessage};

// A custom algorithm the user wants to benchmark alongside control-rs
#[hil_suite]
pub mod my_drone_benchmarks {
    use super::*;

    fn my_custom_drone_math() { /* ... */ }
}

#[hil_setup]
fn setup() -> Context {
    let board = teensy4_bsp::board::t41();
    // ... custom clock config ...

    Context {
        comms: MyComms { /* ... */ },
        timer: MyTimer { /* ... */ },
        settings: Settings { iterations: 1000, verbose: true },
    }
}
```