# CI

![Date Badge](https://img.shields.io/badge/Date-May_24,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Needs%20Review-yellow)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

## 1. Context & Objective

Traditionally, embedded hardware libraries treat benchmarks,
Hardware-in-the-Loop (HIL) tests, and Continuous Integration (CI) test matrices
as internal, closed-source chores.

The objective of this design is to provide testing and benchmarking
infrastructure as part of the `control-rs` development tools available to
end-users. This allows users to test `control-rs` against their specific
hardware and compare it side-by-side with their custom algorithms.

## 2. Architectural Overview

To provide a seamless experience, `control-rs` acts as an umbrella crate,
re-exporting the necessary tooling components so users only need a single
dependency.

* **`control-rs`**: Implementations of types, algorithms and sub-programs.
* **`control-rs-xtask`**: Workspace tools and code generators.
    * **`hil`**: Interactive runner to execute tests and benchmarks on target
      hardware.

## 3. Usage

### Testing

To run the CI tests, you can use the following command:
`cargo xtask ci`