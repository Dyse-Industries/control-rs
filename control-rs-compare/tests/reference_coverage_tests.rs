//! Signal coverage: a peer may omit a signal only where the oracle marks it `missing_ok.<peer>`.

#![allow(clippy::unwrap_used)]

use std::fs;
use std::path::{Path, PathBuf};

use control_rs_compare::compare::{ComparatorOptions, run_comparison};
use control_rs_compare::config::{
    CompareGeneralConfig, MasterPlan, SuiteConfig, VariantConfig,
};
use hdf5_pure::{AttrValue, FileBuilder};

type Signal<'a> = (&'a str, &'a [f64]);

fn write_container(path: &Path, signals: &[Signal<'_>]) {
    write_oracle(path, signals, &[]);
}

/// Writes a container whose datasets named in `missing_ok` carry `missing_ok.alt = 1`.
fn write_oracle(path: &Path, signals: &[Signal<'_>], missing_ok: &[&str]) {
    let mut b = FileBuilder::new();
    for (name, data) in signals {
        let ds = b.create_dataset(name);
        ds.with_f64_data(data);
        if missing_ok.contains(name) {
            ds.set_attr("missing_ok.alt", AttrValue::I64(1));
        }
    }
    fs::write(path, b.finish().unwrap()).unwrap();
}

fn variant(name: &str, kind: &str) -> VariantConfig {
    VariantConfig {
        name: name.to_string(),
        r#type: kind.to_string(),
        manifest_path: None,
        bin: None,
        script: None,
        command: None,
        output_file: format!("{name}.h5"),
        optional: false,
    }
}

fn plan(variants: Vec<VariantConfig>) -> MasterPlan {
    MasterPlan {
        general: CompareGeneralConfig::default(),
        suites: vec![SuiteConfig {
            name: "cov".to_string(),
            true_oracle: "scipy".to_string(),
            tolerance_table: None,
            signals: None,
            variants,
        }],
    }
}

fn results_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir()
        .join(format!("control-rs-compare-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn verdict(dir: &Path, plan: &MasterPlan) -> String {
    let opts = ComparatorOptions {
        results_dir: dir.to_path_buf(),
        suite_filter: None,
        oracle_override: None,
        signals: None,
        strict: true,
        quiet: true,
        num_threads: Some(1),
    };
    run_comparison(Some(plan), &opts).unwrap().summary.verdict
}

fn variants() -> Vec<VariantConfig> {
    vec![
        variant("rust", "rust_bin"),
        variant("scipy", "python_script"),
        variant("alt", "python_script"),
    ]
}

#[test]
fn annotated_missing_signal_is_skipped() {
    let dir = results_dir("annotated");
    write_oracle(
        &dir.join("cov.scipy.h5"),
        &[("a", &[1.0]), ("b", &[2.0])],
        &["b"],
    );
    write_container(&dir.join("cov.rust.h5"), &[("a", &[1.0]), ("b", &[2.0])]);
    write_container(&dir.join("cov.alt.h5"), &[("a", &[1.0])]);
    assert_eq!(verdict(&dir, &plan(variants())), "Pass");
}

#[test]
fn unannotated_missing_signal_fails() {
    let dir = results_dir("unannotated");
    write_container(&dir.join("cov.scipy.h5"), &[("a", &[1.0]), ("b", &[2.0])]);
    write_container(&dir.join("cov.rust.h5"), &[("a", &[1.0]), ("b", &[2.0])]);
    write_container(&dir.join("cov.alt.h5"), &[("a", &[1.0])]);
    assert_eq!(verdict(&dir, &plan(variants())), "Fail");
}

#[test]
fn annotation_applies_only_to_named_peer() {
    let dir = results_dir("otherpeer");
    write_oracle(
        &dir.join("cov.scipy.h5"),
        &[("a", &[1.0]), ("b", &[2.0])],
        &["b"],
    );
    write_container(&dir.join("cov.rust.h5"), &[("a", &[1.0])]);
    write_container(&dir.join("cov.alt.h5"), &[("a", &[1.0])]);
    assert_eq!(verdict(&dir, &plan(variants())), "Fail");
}

#[test]
fn mismatch_on_provided_signal_fails() {
    let dir = results_dir("mismatch");
    write_oracle(
        &dir.join("cov.scipy.h5"),
        &[("a", &[1.0]), ("b", &[2.0])],
        &["b"],
    );
    write_container(&dir.join("cov.rust.h5"), &[("a", &[1.0]), ("b", &[2.0])]);
    write_container(&dir.join("cov.alt.h5"), &[("a", &[5.0])]);
    assert_eq!(verdict(&dir, &plan(variants())), "Fail");
}

#[test]
fn peer_without_oracle_signals_fails() {
    let dir = results_dir("empty");
    write_oracle(&dir.join("cov.scipy.h5"), &[("a", &[1.0])], &["a"]);
    write_container(&dir.join("cov.rust.h5"), &[("a", &[1.0])]);
    write_container(&dir.join("cov.alt.h5"), &[("z", &[1.0])]);
    assert_eq!(verdict(&dir, &plan(variants())), "Fail");
}
