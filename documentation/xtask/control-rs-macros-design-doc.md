# Procedural Macros for Distributed Test Discovery (Design Document)

![Date Badge](https://img.shields.io/badge/Date-July_18,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Reviewed-yellow)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

Bare-metal embedded systems lack traditional operating system loaders, making
dynamic test discovery and runtime test registration difficult. Procedural
macros provide a compile-time solution by automatically analyzing source code
and generating the underlying test registry metadata.

This design document establishes the architecture for `control-rs-macros`, a
procedural macro library containing `#[hil_suite]` and `#[hil_setup]`. These
macros enable developers to declare hardware-in-the-loop (HIL) tests and
benchmarks directly in their modules with zero boilerplate. The macro generates
all registration hooks and places them in custom linker sections, permitting
automated discovery without a centralized registry.

---

### 2. Requirements

#### Functional Requirements

- **FR-1 — Distributed Module Annotation**: Developers must declare a suite by tagging a module with `#[hil_suite]`.
- **FR-2 — Automatic Registration**: The macro must automatically identify non-underscore-prefixed functions within the module and generate test descriptors.
- **FR-3 — Type-Safe Settings Translation**: Any static variable declared inside the `#[hil_suite]` module must be translated into a thread-safe atomic setting structure.
- **FR-4 — Entrypoint & Setup Generation**: The `#[hil_setup]` macro must generate the `main()` entrypoint, call the user's hardware init code, and instantiate the execution context.
- **FR-5 — Custom Panic Redirection**: The macro-generated entrypoint must register a custom panic handler that routes test panics through the host communications layer.

#### Non-Functional Requirements

- **NFR-1 — Zero Runtime Allocation**: All generated code, setup contexts, and panic handlers must operate strictly without a heap.
- **NFR-2 — Linker Retention Safety**: Generated descriptors must survive aggressive linker garbage collection (`--gc-sections`) in release builds.
- **NFR-3 — Rust 2024/2027 Compliance**: Generated structures must eliminate `static mut` usage in favor of type-safe atomic wrappers and interior mutability.

#### Constraints

- **C-1 — Strict `#![no_std]` Compatibility**: Code emitted by the macros must compile on bare-metal targets without standard library support.
- **C-2 — Target Independency**: The macros must emit target-agnostic Rust code that delegates hardware specifics to the user-defined profiler.

---

### 3. Technical Overview

The procedural macros are contained within the `control-rs-macros` library. This
library depends on standard compiler crates (`proc-macro`, `syn`, `quote`) and
operates as a compiler plugin. It interacts with other workspace crates to form
the test registry:

```mermaid
flowchart TD
    subgraph Host ["Host Compiler"]
        Macros["control-rs-macros (syn/quote)"]
    end

    subgraph Target ["Target Codebase"]
        UserCode["User Module + #[hil_suite]"]
        HIL["control-rs-hil (SuiteDescriptor)"]
        LinkerScript["hil_suites.x (KEEP Section)"]
    end

    Macros -->|Parses & Generates| UserCode
    UserCode -->|Implements| HIL
    UserCode -->|Emits to Linker| LinkerScript
```

---

### 4. Core Architecture

#### 4.1. `#[hil_suite]` Module Parsing and Code Generation

When the compiler encounters `#[hil_suite]` on a module, the macro performs the
following transformations:

1. **Test Identification**: It traverses the module's items, locating all
   functions. For each function, it generates an `ExecDescriptor` containing the
   function's name, description, and function pointer.
2. **Settings Translation**: It traverses static variables. To eradicate
   `static mut` (deprecated in Rust 2024 and prohibited in 2027), the macro
   converts standard type declarations into atomic settings. For example:
   ```rust
   // User declaration
   pub static MAX_RETRIES: u32 = 3;
   ```
   Is parsed and translated into:
   ```rust
   // Generated code
   pub static MAX_RETRIES: AtomicU32Setting = AtomicU32Setting::new(
       "MAX_RETRIES",
       "User-defined test parameter",
       3
   );
   ```
3. **Registry Emission**: The macro emits a static `SuiteDescriptor` that
   references the array of `ExecDescriptor`s and the list of settings. This
   descriptor is annotated with the link section attribute to ensure it is
   placed in the test registry:
    ```rust
    #[unsafe(link_section = ".hil_test_suites")]
    #[used]
    pub static SUITE_DESCRIPTOR_PTR: &::control_rs_hil::SuiteDescriptor = &SUITE_DESCRIPTOR;
    ```

#### 4.2. Linker Garbage Collection Mitigation

A critical issue in bare-metal Rust is that LLD linker passes the
`--gc-sections` flag by default. Because the application logic does not directly
call the static `SuiteDescriptor` variables, the linker views them as dead code
and strips them during compilation.

To prevent this, the workspace employs a multi-tiered retention architecture:

1. **`#[used]`**: Generated static variables use attributes directing LLVM to
   keep the symbol in the object file.
2. **Linker Script KEEP Directive**: The `control-rs-hil` crate packages a
   custom linker script snippet (`hil_suites.x`) that includes a `KEEP`
   directive for the `.hil_test_suites` section:
   ```ld
   KEEP(*(.hil_test_suites))
   ```
3. **Build Script Linker Argument Injection**: Target binary examples (like QEMU
   and Teensy 4) include a `build.rs` or `.cargo/config.toml` that injects the
   script to the linker command line:
   ```rust
   println!("cargo:rustc-link-arg=-Thil_suites.x");
   ```
   This prevents end-users from needing to manually manage linker configuration
   files.

#### 4.3. `#[hil_setup]` Entrypoint and Panic Handling

The `#[hil_setup]` macro is applied to the user's hardware initialization
function. It replaces the function with the primary entrypoint:

1. **`main` Function Wrapper**: The macro emits the standard
   `#[no_mangle] pub extern "C" fn main() -> !` entrypoint.
2. **Setup Call**: It calls the user's custom setup function to initialize clock
   registers, configure peripherals (UART, SPI, DMA), and return the execution
   `Context`.
3. **Panic Hook Registration**: It registers a custom panic handler. This
   handler intercepts any panics, serializes the panic message and location, and
   writes them directly to the `HostComms` interface:
    ```rust
    #[cfg(target_os = "none")]
    #[panic_handler]
    fn panic(info: &::core::panic::PanicInfo) -> ! {
        let mut msg_buf = [0u8; 128];
        let pos = {
            let mut writer = ::control_rs_hil::util::FailureBufWriter { buf: &mut msg_buf, pos: 0 };
            let _ = ::core::fmt::write(&mut writer, format_args!("{}", info.message()));
            writer.pos
        };
        let msg = ::core::str::from_utf8(&msg_buf[..pos]).unwrap_or("panic occurred");

        let file = info.location().map_or("unknown", |l| l.file());
        let line = info.location().map_or(0, |l| l.line());

        let server_ptr = HIL_SERVER.load(::core::sync::atomic::Ordering::Acquire);
        unsafe {
            if !server_ptr.is_null() {
                let server = &mut *server_ptr;
                let comms_ok = server.context.comms_lock.try_lock();
                ::control_rs_hil::util::handle_failure(
                    &mut server.context,
                    msg,
                    file,
                    line,
                    comms_ok,
                );
            } else {
                loop {
                    ::core::hint::spin_loop();
                }
            }
        }
    }
    ```
   This prevents the target from locking up silently and ensures the host TUI
   displays the failure.

---

### 5. Alternatives

* **Manual Registration Array**: Developers could manually declare a global
  array of function pointers. This is rejected due to high maintenance overhead,
  boilerplate, and the risk of developer error when adding new tests.
* **`linkme::distributed_slice`**: Rejected. The mechanism is architecturally
  identical to the hand-rolled section scheme, but official platform support
  covers OS-hosted targets only (Linux, macOS, Windows, FreeBSD, OpenBSD,
  illumos); bare-metal Cortex-M/RISC-V support is uncorroborated by any primary
  source. Adoption would also surrender project-owned control of the
  `.hil_test_suites` section name and the `hil_suites.x`/`build.rs` linker
  coordination.
* **`inventory` (ctor-based registration)**: Rejected. Registration relies on
  life-before-main constructors invoked by an OS loader (ELF `.init_array`,
  Mach-O `mod_init_func`, PE TLS callbacks), which do not exist on a
  loader-less bare-metal target. Unsupported platforms silently register
  nothing rather than failing to compile.
* **Nightly `custom_test_frameworks`**: Rejected. This requires unstable
  compiler flags and nightly toolchains, which violates the strict reliability
  and safety-certification goals of `control-rs`.
* **Standard `#[test]` Harness (libtest)**: Rejected. It depends on `std`
  components like threads and dynamic memory, which are unavailable in
  bare-metal targets.
* **Global Compiler `-C link-dead-code` Flag**: Rejected. While it prevents test
  registry deletion, it also disables dead code elimination for the entire
  project, leading to bloated binaries that exceed the MCU's Flash capacity.

---

### 6. Verification & Validation

#### 6.1. Verification Plan

- **Procedural Macro Tests**: Implement compiler tests using `trybuild` to
  verify that the macro correctly handles valid code structures and rejects
  invalid constructs (e.g. non-static variable declarations inside a suite
  module) with clean error messages. Compile-fail cases must cover
  `#[hil_setup]`
  misuse (wrong return type, missing `Context` generics) and assert spanned
  `syn::Error` diagnostics rather than opaque proc-macro panics.

#### 6.2. Validation Plan

- **Hardware Integration Test**: Compile the `teensy4` board tests using the
  macros and execute them using the host-side `xtask`/`ServerBridge` to
  validate that all tests are discovered and that settings can be modified
  dynamically at runtime.

---

### 7. Performance & Resource Considerations

* **Static Flash Storage**: Since the macro places all descriptors in
  `.hil_test_suites` marked as read-only, they reside entirely in Flash (ROM)
  and consume zero RAM during idle state.
* **Compiler Timing**: To maintain fast build times, the macro relies on minimal
  syn features and avoids complex, recursive macro expansion paths.

---

### 8. Risks & Open Questions

* **Linker Target Differences**: Different targets (e.g., MSP430 or custom
  architectures) might require variations of the linker script arguments. The
  build script must detect the target architecture and adapt the link flags
  accordingly.
* **Rust compiler updates**: Shifts in compiler syntax/AST structure in future
  Rust editions could disrupt the syn-based parser. Pinning dependencies in
  Cargo.toml mitigates this, supplemented by periodic dependency-tree audits:
  the workspace lockfile already carries syn 1.x (via TUI dependencies) and
  syn 2.x side by side, so version fragmentation is a live condition rather
  than a hypothetical.
* **`#[hil_setup]` Error Diagnostics**: The return-type and generic-extraction
  checks currently abort via `panic!`/`.expect()`, which surface as an opaque
  "proc-macro panicked" diagnostic. They must be migrated to the
  `syn::Error::new_spanned(...).to_compile_error()` pattern already used by
  `#[hil_suite]`.
* **Per-Test Opt-Out**: Test exclusion is limited to underscore-prefixed
  function names. `defmt-test` and `embedded-test` demonstrate attribute-based
  `#[ignore]`/`#[cfg]` opt-out in the same whole-module-rewrite macro shape;
  open question whether to add an equivalent attribute to the functional
  requirements.
* **syn Feature Scope**: The `extra-traits` feature only adds Debug/Eq/Hash
  impls that the transformation logic does not appear to use. A feature audit
  should confirm whether it can be dropped to narrow the proc-macro2/syn API
  surface.
* **`linkme` Re-Evaluation**: Ruling `linkme` in or out definitively requires
  exercising it against a real bare-metal target (QEMU or Teensy 4), since its
  official platform list conflicts with unofficial claims of embedded support.

---

### 9. Development Plan

Steps 1–4 are implemented in `control-rs-macros/src/lib.rs`; discovery via
`.hil_test_suites` is exercised by the HIL Server (see
`hil-server-design-doc.md`). Step 5 covers remaining hardening.

| Task / Feature                                 | Description                                                                                                                      | Status / Effort |
|:-----------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------|:----------------|
| **Step 1: syn Parser Implementation**          | Implement parsing logic for `#[hil_suite]` modules and `#[hil_setup]` functions.                                                 | Shipped         |
| **Step 2: AST Code Generation**                | Develop codegen templates for `SuiteDescriptor` outputs and atomic settings translations.                                        | Shipped         |
| **Step 3: Linker Integration**                 | Write the `build.rs` layout injection code and build the `hil_suites.x` linker script file.                                      | Shipped         |
| **Step 4: Panic Handler Codegen**              | Implement code generation for the custom bare-metal panic handler in `#[hil_setup]`.                                             | Shipped         |
| **Step 5: Diagnostics & Ergonomics Hardening** | Migrate `#[hil_setup]` panics to spanned `syn::Error`s, audit the `extra-traits` feature, evaluate a per-test opt-out attribute. | 1.0 day         |

---

### 10. Revision History

| Revision | Date | Author | Description |
|:---------|:-----|:-------|:-------------|
| 1.0 | May 24, 2026 | @MitchellDScott | Initial design of procedural macros for test discovery. |
| 1.1 | July 18, 2026 | @MitchellDScott | Restructured to design-template standard; added linker GC mitigation and panic-redirection findings. |
| 1.2 | August 6, 2026 | @MitchellDScott | Incorporated build-vs-adopt research; marked Steps 1-4 shipped; unified linker script name to `hil_suites.x`. |