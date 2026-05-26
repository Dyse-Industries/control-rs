//! Testing and benchmarking infrastructure types

#[repr(C)]
pub struct TestDescriptor {
    pub name: &'static str,
    pub test_fn: fn(),
}