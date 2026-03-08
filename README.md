# control-rs

`control-rs` is a `no_std` Rust library for numerical modeling, control synthesis and real-time execution. It targets autonomous systems and bare-metal embedded platforms.

### Data-Driven Model-Based Design

`control-rs` enables a complete Model-Based Design (MBD) pipeline. System identification, controller tuning, HITL verification and production firmware development are possible with stable Cargo tooling.

## Features

*   **Static Dimensions:** Storage dimensions are calculated at compile-time. No heap allocation; zero-cost bounds checking.
*   **Robust Arithmetic:** Strict algebraic traits (`Scalar`, `Ring`, `Field`) and fallible operations (`try_add`, `try_mul`) prevent undefined behavior.
*   **Backend-Agnostic BLAS:** BLAS operations are generic traits. Hardware backends (e.g., ARM NEON, CMSIS-DSP) are injected at compile-time.

## Verification

The project enforces strict `clippy` lints, tracks coverage with `tarpaulin` and analyzes binary size with `binutils`.

## Installation

```toml
[dependencies]
control-rs = { git = "https://github.com/Dyse-Industries/control-rs.git" }
```

## License

Licensed under the MIT license.