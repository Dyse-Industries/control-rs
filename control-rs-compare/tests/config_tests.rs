//! Integration and unit tests for compare.toml deserialization.

use control_rs_compare::config::CompareConfigFile;

#[test]
fn test_config_deserialization() {
    let toml_str = r#"
[compare]
title = "Test Suite"
out_dir = "custom_results"
timeout_secs = 60
strict = true

suites = [
    "examples/buck-converter",
]

[[suite]]
name = "inlined_test"
true_oracle = "scipy"

[[suite.variants]]
name = "rust"
type = "rust_bin"
manifest_path = "Cargo.toml"
bin = "test_bin"
output_file = "results/test.rust.h5"

[[suite.variants]]
name = "scipy"
type = "python_script"
script = "test.py"
output_file = "results/test.scipy.h5"
"#;

    let config: CompareConfigFile = toml::from_str(toml_str).unwrap();
    assert_eq!(config.compare.title, "Test Suite");
    assert_eq!(config.compare.out_dir, "custom_results");
    assert_eq!(config.compare.suites.len(), 1);
    assert_eq!(config.inlined_suites.len(), 1);
    let inlined = config.inlined_suites.first().unwrap();
    assert_eq!(inlined.name, "inlined_test");
    assert_eq!(inlined.variants.len(), 2);
}
