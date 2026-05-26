// CI test runner

extern crate std;
use crate::testing::TestDescriptor;

pub mod binutil;
pub mod bloat;
pub mod common;
pub mod coverage;
pub mod lint;
pub mod test;

pub fn start(tests: &[TestDescriptor], benchmarks: &[TestDescriptor]) {
    // In a real implementation this would automatically execute all tests
    // and benchmarks, format the results, and communicate them out.
}