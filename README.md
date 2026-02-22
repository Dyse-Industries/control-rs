# control-rs

`control-rs` is a native Rust library for numerical modeling and synthesis.
It is intended for developing robotics, autonomous vehicles, UAVs and other real-time embedded
systems that rely on advanced control algorithms.

```text
control-rs/
├── Cargo.toml               (Main manifest & Workspace root)
├── build.rs                 (Executes the codegen)
├── src/
│   └── num_types.rs         (Includes the generated math)
└── codegen/                 (Your internal utility crate)
    ├── Cargo.toml
    └── src/
        └── lib.rs           (The generation logic)
```