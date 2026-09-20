//! Integration tests for parallel CI execution and barrier synchronization.

#![allow(
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::type_complexity
)]

use control_rs_ci::config::{ExecutionConfig, GateConfig};

#[test]
fn test_execution_config_defaults() {
    let config = ExecutionConfig::default();
    assert!(config.parallel);
    assert!(
        config
            .exclusive_gates
            .contains(&"cross-compare".to_string())
    );
    assert!(config.exclusive_gates.contains(&"valgrind".to_string()));
    assert!(config.exclusive_gates.contains(&"mutants".to_string()));
    assert!(config.lanes.contains_key("cargo"));
    assert!(config.lanes.contains_key("audit"));
    assert!(config.lanes.contains_key("static"));
}

#[test]
fn test_execution_config_toml_deserialization() {
    let toml_str = r#"
        [execution]
        parallel = true
        exclusive_gates = ["cross-compare", "custom-exclusive"]

        [execution.lanes]
        cargo = ["fmt", "clippy", "build", "test"]
        audit = ["deny", "geiger", "semver"]
        static = ["metrics", "git", "vale"]
    "#;

    let gate_cfg: GateConfig = toml::from_str(toml_str).unwrap();
    let exec = gate_cfg.execution;
    assert!(exec.parallel);
    assert_eq!(exec.exclusive_gates.len(), 2);
    assert!(
        exec.exclusive_gates
            .contains(&"custom-exclusive".to_string())
    );
    assert_eq!(exec.lanes.len(), 3);
}
