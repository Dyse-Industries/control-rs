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
    assert!(config.exclusive_gates.is_empty());
    assert!(config.groups.is_empty());
}

#[test]
fn test_execution_config_toml_deserialization() {
    let toml_str = r#"
        [execution]
        parallel = true
        exclusive_gates = ["cross-compare", "custom-exclusive"]

        [execution.groups]
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
    assert_eq!(exec.groups.len(), 3);

    // Test backwards compatibility alias [execution.lanes]
    let legacy_toml = r#"
        [execution]
        [execution.lanes]
        cargo = ["fmt", "build"]
    "#;
    let legacy_cfg: GateConfig = toml::from_str(legacy_toml).unwrap();
    assert_eq!(legacy_cfg.execution.groups.len(), 1);
    assert!(legacy_cfg.execution.groups.contains_key("cargo"));
}

#[test]
fn test_exclusive_group_normalization() {
    let toml_str = r#"
        [execution]
        parallel = true
        exclusive_gates = ["cross-compare"]

        [execution.groups]
        cargo = ["fmt"]
        exclusive = ["geiger", "valgrind"]
    "#;

    let mut gate_cfg: GateConfig = toml::from_str(toml_str).unwrap();
    gate_cfg.execution.normalize();
    assert_eq!(gate_cfg.execution.groups.len(), 1);
    assert!(!gate_cfg.execution.groups.contains_key("exclusive"));
    assert_eq!(gate_cfg.execution.exclusive_gates.len(), 3);
    assert!(
        gate_cfg
            .execution
            .exclusive_gates
            .contains(&"cross-compare".to_string())
    );
    assert!(
        gate_cfg
            .execution
            .exclusive_gates
            .contains(&"geiger".to_string())
    );
    assert!(
        gate_cfg
            .execution
            .exclusive_gates
            .contains(&"valgrind".to_string())
    );
}

#[test]
fn test_unassigned_gate_routes_to_exclusive() {
    let toml_str = r#"
        [execution]
        parallel = true
        exclusive_gates = ["cross-compare"]

        [execution.groups]
        cargo = ["fmt", "clippy"]
    "#;

    let gate_cfg: GateConfig = toml::from_str(toml_str).unwrap();
    let active = vec!["fmt", "metrics", "cross-compare"];

    let (exclusive, concurrent): (Vec<_>, Vec<_>) =
        active.into_iter().partition(|g| {
            gate_cfg.execution.exclusive_gates.iter().any(|e| e == *g)
                || !gate_cfg
                    .execution
                    .groups
                    .values()
                    .any(|members| members.iter().any(|m| m == *g))
        });

    assert_eq!(exclusive, vec!["metrics", "cross-compare"]);
    assert_eq!(concurrent, vec!["fmt"]);
}
