//! Unit tests for typed numerical comparison algorithms, recursive dataset discovery,
//! and tolerance discovery from external tables and HDF5 attributes.

use control_rs_compare::compare::{
    SignalTolerancePolicy, ToleranceSpec, compare_dataset,
    compare_float_arrays, compare_float_arrays_parallel, discover_datasets,
    resolve_signal_tolerances,
};
use control_rs_compare::config::ToleranceTable;
use hdf5_pure::{AttrValue, File, FileBuilder};

/// Error type shared by the fallible helpers and tests.
type TestError = Box<dyn std::error::Error>;

/// Result of a test or a helper assertion.
type TestResult = Result<(), TestError>;

/// Serialized container bytes, or the builder error.
type ContainerBytes = Result<Vec<u8>, TestError>;

/// Tolerance policy, or the container error.
type PolicyResult = Result<SignalTolerancePolicy, TestError>;

/// Dataset paths a discovery case expects, in sorted order.
type DatasetPaths = Vec<&'static str>;

/// `(key, value)` attributes set on a dataset.
type AttrList<'a> = Vec<(&'a str, AttrValue)>;

/// Builds one discovery-case container.
type Builder = fn() -> ContainerBytes;

struct DiscoveryTestCase {
    name: &'static str,
    builder_fn: Builder,
    expected: DatasetPaths,
}

/// Asserts that method `idx` of `policy` is `method` with a bit-exact
/// `bound`: bounds are parsed from TOML or attributes, so equality is exact.
fn assert_method(
    policy: &SignalTolerancePolicy,
    idx: usize,
    method: &str,
    bound: f64,
) -> TestResult {
    let spec = policy
        .methods
        .get(idx)
        .ok_or_else(|| format!("policy has no method {idx}"))?;
    assert_eq!(spec.method, method);
    assert_eq!(
        spec.bound.to_bits(),
        bound.to_bits(),
        "bound {} != {bound}",
        spec.bound
    );
    Ok(())
}

/// A container holding dataset `sig` = `[1.0, 2.0]` with `attrs`.
fn attributed_file(attrs: AttrList<'_>) -> Result<File, TestError> {
    let mut b = FileBuilder::new();
    let ds = b.create_dataset("sig");
    ds.with_f64_data(&[1.0, 2.0]);
    for (key, value) in attrs {
        ds.set_attr(key, value);
    }
    Ok(File::from_bytes(b.finish()?)?)
}

/// Tolerance policy that `attrs` on dataset `sig` resolve to for `peer`.
fn attribute_policy(attrs: AttrList<'_>, peer: &str) -> PolicyResult {
    let file = attributed_file(attrs)?;
    let ds = file.dataset("sig")?;
    Ok(resolve_signal_tolerances(Some(&ds), "sig", peer, None))
}

#[test]
fn test_abs_comparison() {
    let o = vec![1.0, 2.0, 3.0];
    let p_pass = vec![1.0001, 1.9999, 3.0005];
    let p_fail = vec![1.002, 2.0, 3.0];

    let tol = ToleranceSpec {
        method: "abs".to_string(),
        bound: 1e-3,
    };

    let res_pass = compare_float_arrays(&o, &p_pass, &tol);
    assert_eq!(res_pass.verdict, "pass");
    assert!(res_pass.observed <= 1e-3);

    let res_fail = compare_float_arrays(&o, &p_fail, &tol);
    assert_eq!(res_fail.verdict, "fail");
    assert!(res_fail.observed > 1e-3);
}

#[test]
fn test_rel_comparison() {
    let o = vec![100.0, 200.0];
    let p_pass = vec![100.5, 199.0];
    let p_fail = vec![105.0, 200.0];

    let tol = ToleranceSpec {
        method: "rel".to_string(),
        bound: 1e-2,
    };

    assert_eq!(compare_float_arrays(&o, &p_pass, &tol).verdict, "pass");
    assert_eq!(compare_float_arrays(&o, &p_fail, &tol).verdict, "fail");
}

#[test]
fn test_rms_comparison() {
    let o = vec![1.0, 2.0, 3.0, 4.0];
    let p = vec![1.0001, 2.0001, 3.0001, 4.0001];

    let tol = ToleranceSpec {
        method: "rms".to_string(),
        bound: 1e-3,
    };

    let res = compare_float_arrays(&o, &p, &tol);
    assert_eq!(res.verdict, "pass");
    assert!(res.observed < 1e-3);
}

#[test]
fn test_matrix_norm_comparison() {
    let o = vec![1.0, 0.0, 0.0, 1.0];
    let p_pass = vec![1.01, 0.01, 0.0, 0.99];
    let p_fail = vec![1.1, 0.0, 0.0, 1.0];

    let tol = ToleranceSpec {
        method: "matrix_norm".to_string(),
        bound: 0.05,
    };

    assert_eq!(compare_float_arrays(&o, &p_pass, &tol).verdict, "pass");
    assert_eq!(compare_float_arrays(&o, &p_fail, &tol).verdict, "fail");
}

#[test]
fn test_nan_detection() {
    let o = vec![1.0, f64::NAN, 3.0];
    let p = vec![1.0, 2.0, 3.0];

    let tol = ToleranceSpec::default();
    let res = compare_float_arrays(&o, &p, &tol);
    assert_eq!(res.verdict, "fail");
    assert!(res.details.as_deref().unwrap().contains("NaN"));
}

// =========================================================================
// Parameterized Dataset Discovery Tests
// =========================================================================

fn build_empty_container() -> ContainerBytes {
    Ok(FileBuilder::new().finish()?)
}

fn build_flat_root_datasets() -> ContainerBytes {
    let mut b = FileBuilder::new();
    b.create_dataset("gamma").with_f64_data(&[3.0]);
    b.create_dataset("alpha").with_f64_data(&[1.0]);
    b.create_dataset("beta").with_f64_data(&[2.0]);
    Ok(b.finish()?)
}

fn build_deeply_nested_hierarchy() -> ContainerBytes {
    let mut b = FileBuilder::new();
    let mut g_a = b.create_group("a");
    g_a.create_dataset("leaf3").with_f64_data(&[3.0]);

    let mut g_b = g_a.create_group("b");
    let mut g_c = g_b.create_group("c");
    g_c.create_dataset("leaf2").with_f64_data(&[2.0]);

    let mut g_d = g_c.create_group("d");
    let mut g_e = g_d.create_group("e");
    g_e.create_dataset("leaf1").with_f64_data(&[1.0]);

    g_d.add_group(g_e.finish());
    g_c.add_group(g_d.finish());
    g_b.add_group(g_c.finish());
    g_a.add_group(g_b.finish());
    b.add_group(g_a.finish());

    Ok(b.finish()?)
}

fn build_multi_branch_complex_tree() -> ContainerBytes {
    let mut b = FileBuilder::new();
    b.create_dataset("status").with_f64_data(&[1.0]);

    let mut g_mat = b.create_group("matrix");
    g_mat.create_dataset("b").with_f64_data(&[2.0]);
    g_mat.create_dataset("a").with_f64_data(&[1.0]);
    b.add_group(g_mat.finish());

    let mut g_trans = b.create_group("transient");
    g_trans.create_dataset("v_out").with_f64_data(&[3.3]);
    g_trans.create_dataset("i_load").with_f64_data(&[0.5]);

    let mut g_stg1 = g_trans.create_group("stage1");
    g_stg1.create_dataset("v_node").with_f64_data(&[1.2]);
    g_trans.add_group(g_stg1.finish());

    b.add_group(g_trans.finish());

    let mut g_tensor = b.create_group("tensor");
    g_tensor.create_dataset("state").with_f64_data(&[0.0, 1.0]);
    b.add_group(g_tensor.finish());

    Ok(b.finish()?)
}

fn build_empty_groups_ignored() -> ContainerBytes {
    let mut b = FileBuilder::new();
    let g_empty1 = b.create_group("empty_g1");
    b.add_group(g_empty1.finish());

    let mut g2 = b.create_group("g2");
    let g_sub_empty = g2.create_group("empty_sub");
    g2.add_group(g_sub_empty.finish());
    g2.create_dataset("valid_ds").with_f64_data(&[42.0]);
    b.add_group(g2.finish());

    Ok(b.finish()?)
}

fn build_underscore_metadata_and_internal_excluded() -> ContainerBytes {
    let mut b = FileBuilder::new();
    let mut g_meta = b.create_group("_meta");
    g_meta.create_dataset("timestamp").with_f64_data(&[123.0]);
    g_meta.create_dataset("git_sha").with_f64_data(&[456.0]);
    b.add_group(g_meta.finish());

    let mut g_hidden = b.create_group("_hidden");
    g_hidden.create_dataset("secret").with_f64_data(&[789.0]);
    b.add_group(g_hidden.finish());

    let mut g_sig = b.create_group("signals");
    g_sig
        .create_dataset("_internal_state")
        .with_f64_data(&[0.0]);
    g_sig.create_dataset("public_output").with_f64_data(&[1.0]);

    let mut g_sig_hidden = g_sig.create_group("_private_sub");
    g_sig_hidden.create_dataset("sub_val").with_f64_data(&[2.0]);
    g_sig.add_group(g_sig_hidden.finish());

    b.add_group(g_sig.finish());

    Ok(b.finish()?)
}

#[test]
fn test_parameterized_dataset_discovery() {
    let test_cases = [
        DiscoveryTestCase {
            name: "empty_container",
            builder_fn: build_empty_container,
            expected: vec![],
        },
        DiscoveryTestCase {
            name: "flat_root_datasets",
            builder_fn: build_flat_root_datasets,
            expected: vec!["alpha", "beta", "gamma"],
        },
        DiscoveryTestCase {
            name: "deeply_nested_hierarchy",
            builder_fn: build_deeply_nested_hierarchy,
            expected: vec!["a/b/c/d/e/leaf1", "a/b/c/leaf2", "a/leaf3"],
        },
        DiscoveryTestCase {
            name: "multi_branch_complex_tree",
            builder_fn: build_multi_branch_complex_tree,
            expected: vec![
                "matrix/a",
                "matrix/b",
                "status",
                "tensor/state",
                "transient/i_load",
                "transient/stage1/v_node",
                "transient/v_out",
            ],
        },
        DiscoveryTestCase {
            name: "empty_groups_ignored",
            builder_fn: build_empty_groups_ignored,
            expected: vec!["g2/valid_ds"],
        },
        DiscoveryTestCase {
            name: "underscore_metadata_and_internal_excluded",
            builder_fn: build_underscore_metadata_and_internal_excluded,
            expected: vec!["signals/public_output"],
        },
    ];

    for tc in test_cases {
        let bytes = (tc.builder_fn)().unwrap();
        let file = File::from_bytes(bytes).unwrap_or_else(|e| {
            panic!("Failed to parse HDF5 for '{}': {e}", tc.name)
        });
        let discovered = discover_datasets(&file);
        assert_eq!(
            discovered, tc.expected,
            "Discovery mismatch for test case '{}'",
            tc.name
        );
    }
}

// =========================================================================
// Parameterized Tolerance Discovery Tests
// =========================================================================

#[test]
fn test_tolerance_discovery_from_table() -> TestResult {
    let toml_str = r#"
[signals."matrix/a"]
method = "matrix_norm"
bound = 0.05
policy = "all_of"

[signals."matrix/a".peer_bounds]
matlab = 0.10
python = 0.02

[signals."transient/v_out"]
policy = "any_of"
methods = [
    { type = "rel", bound = 0.01 },
    { type = "rms", bound = 0.005 }
]
"#;
    let table: ToleranceTable = toml::from_str(toml_str).unwrap();

    // 1. Single method with default peer
    let pol_default =
        resolve_signal_tolerances(None, "matrix/a", "rust", Some(&table));
    assert_eq!(pol_default.policy, "all_of");
    assert_eq!(pol_default.methods.len(), 1);
    assert_method(&pol_default, 0, "matrix_norm", 0.05)?;

    // 2. Peer override: `matlab`
    let pol_matlab =
        resolve_signal_tolerances(None, "matrix/a", "matlab", Some(&table));
    assert_method(&pol_matlab, 0, "matrix_norm", 0.10)?;

    // 3. Peer override: python
    let pol_python =
        resolve_signal_tolerances(None, "matrix/a", "python", Some(&table));
    assert_method(&pol_python, 0, "matrix_norm", 0.02)?;

    // 4. Multi-method with `any_of` policy
    let pol_trans = resolve_signal_tolerances(
        None,
        "transient/v_out",
        "rust",
        Some(&table),
    );
    assert_eq!(pol_trans.policy, "any_of");
    assert_eq!(pol_trans.methods.len(), 2);
    assert_method(&pol_trans, 0, "rel", 0.01)?;
    assert_method(&pol_trans, 1, "rms", 0.005)?;
    Ok(())
}

/// Single measure and bound, with a peer override.
#[test]
fn test_tolerance_attrs_single_measure_with_peer_override() -> TestResult {
    let attrs = || {
        vec![
            ("measure", AttrValue::String("matrix_norm".to_string())),
            ("bound", AttrValue::F64(0.01)),
            ("bound.matlab", AttrValue::F64(0.05)),
        ]
    };
    let pol = attribute_policy(attrs(), "rust")?;
    assert_eq!(pol.methods.len(), 1);
    assert_method(&pol, 0, "matrix_norm", 0.01)?;

    let pol_matlab = attribute_policy(attrs(), "matlab")?;
    assert_method(&pol_matlab, 0, "matrix_norm", 0.05)?;
    Ok(())
}

/// Multi-method JSON attribute with `any_of` policy.
#[test]
fn test_tolerance_attrs_multi_method_json() -> TestResult {
    let pol = attribute_policy(
        vec![
            (
                "methods",
                AttrValue::String(
                    r#"[{"type": "rms", "bound": 0.002}, {"type": "rel", "bound": 0.05}]"#
                        .to_string(),
                ),
            ),
            ("policy", AttrValue::String("any_of".to_string())),
        ],
        "rust",
    )?;
    assert_eq!(pol.policy, "any_of");
    assert_eq!(pol.methods.len(), 2);
    assert_method(&pol, 0, "rms", 0.002)?;
    assert_method(&pol, 1, "rel", 0.05)?;
    Ok(())
}

/// Fallback default when neither attributes nor a table apply.
#[test]
fn test_tolerance_attrs_fallback_default() -> TestResult {
    let pol = attribute_policy(vec![], "rust")?;
    assert_eq!(pol.policy, "all_of");
    assert_eq!(pol.methods.len(), 1);
    assert_method(&pol, 0, "abs", 1e-4)?;
    Ok(())
}

/// Single `method` attribute with an `f32` bound.
#[test]
fn test_tolerance_attrs_method_with_f32_bound() -> TestResult {
    let pol = attribute_policy(
        vec![
            ("method", AttrValue::String("rel".to_string())),
            ("bound", AttrValue::F32(0.005_f32)),
        ],
        "rust",
    )?;
    assert_eq!(pol.methods.len(), 1);
    assert_method(&pol, 0, "rel", f64::from(0.005_f32))?;
    Ok(())
}

/// A `bound` attribute alone defaults the method to `abs`.
#[test]
fn test_tolerance_attrs_bound_only() -> TestResult {
    let pol = attribute_policy(vec![("bound", AttrValue::I32(1))], "rust")?;
    assert_eq!(pol.methods.len(), 1);
    assert_method(&pol, 0, "abs", 1.0)?;
    Ok(())
}

/// An external table takes precedence over dataset attributes.
#[test]
fn test_tolerance_table_precedes_attrs() -> TestResult {
    let file = attributed_file(vec![
        ("measure", AttrValue::String("matrix_norm".to_string())),
        ("bound", AttrValue::F64(0.01)),
    ])?;
    let ds = file.dataset("sig")?;
    let table_override: ToleranceTable = toml::from_str(
        r#"
[signals.sig]
method = "rms"
bound = 0.0001
"#,
    )
    .unwrap();

    let pol = resolve_signal_tolerances(
        Some(&ds),
        "sig",
        "rust",
        Some(&table_override),
    );
    assert_eq!(pol.methods.len(), 1);
    assert_method(&pol, 0, "rms", 0.0001)?;
    Ok(())
}

#[test]
fn test_explicit_signals_filtering() {
    let mut b_oracle = FileBuilder::new();
    let mut g_oracle = b_oracle.create_group("matrix");
    g_oracle.create_dataset("a").with_f64_data(&[1.0, 2.0]);
    g_oracle.create_dataset("b").with_f64_data(&[3.0, 4.0]);
    g_oracle.create_dataset("c").with_f64_data(&[5.0, 6.0]);
    b_oracle.add_group(g_oracle.finish());
    let o_bytes = b_oracle.finish().unwrap();
    let o_file = File::from_bytes(o_bytes).unwrap();

    let mut b_peer = FileBuilder::new();
    let mut g_peer = b_peer.create_group("matrix");
    g_peer.create_dataset("a").with_f64_data(&[1.0, 2.0]);
    // peer is missing matrix/c
    g_peer.create_dataset("b").with_f64_data(&[3.0, 4.0]);
    b_peer.add_group(g_peer.finish());
    let p_bytes = b_peer.finish().unwrap();
    let p_file = File::from_bytes(p_bytes).unwrap();

    // When explicitly providing only ["matrix/a"], matrix/c is never compared and succeeds
    let tol = ToleranceSpec::default();
    let finding_a = compare_dataset(&o_file, &p_file, "matrix/a", &tol);
    assert_eq!(finding_a.verdict, "pass");

    // When missing dataset is probed, reports failure
    let finding_c = compare_dataset(&o_file, &p_file, "matrix/c", &tol);
    assert_eq!(finding_c.verdict, "fail");
    assert!(
        finding_c
            .details
            .as_deref()
            .unwrap()
            .contains("missing in peer")
    );
}

// =========================================================================
// Parameterized Parallel Chunked Evaluation Tests
// =========================================================================

#[test]
fn test_parameterized_parallel_chunked_equivalence() {
    let sizes = [10_usize, 100, 65_536, 131_072, 262_144];
    let thread_counts = [1_usize, 2, 4, 8, 16];
    let methods = ["abs", "rel", "rms", "matrix_norm"];

    for &n in &sizes {
        let oracle: Vec<f64> = (0..u32::try_from(n).unwrap())
            .map(|i| (f64::from(i) * 0.001).sin())
            .collect();
        let peer_pass: Vec<f64> = oracle
            .iter()
            .map(|&v| 1e-6f64.mul_add((v * 0.5).cos(), v))
            .collect();
        let mut peer_fail = peer_pass.clone();
        if let Some(v) = peer_fail.get_mut(n / 2) {
            *v += 0.5;
        }

        for &method in &methods {
            let tol = ToleranceSpec {
                method: method.to_string(),
                bound: 1e-3,
            };

            let seq_pass = compare_float_arrays(&oracle, &peer_pass, &tol);
            let seq_fail = compare_float_arrays(&oracle, &peer_fail, &tol);

            for &threads in &thread_counts {
                let par_pass = compare_float_arrays_parallel(
                    &oracle, &peer_pass, &tol, threads,
                );
                let par_fail = compare_float_arrays_parallel(
                    &oracle, &peer_fail, &tol, threads,
                );

                assert_eq!(
                    seq_pass.verdict, par_pass.verdict,
                    "Verdict mismatch on passing test: size={n}, threads={threads}, method={method}"
                );
                assert_eq!(
                    seq_fail.verdict, par_fail.verdict,
                    "Verdict mismatch on failing test: size={n}, threads={threads}, method={method}"
                );

                let diff_pass = (seq_pass.observed - par_pass.observed).abs();
                assert!(
                    diff_pass < 1e-12,
                    "Observed value drift in pass: seq={}, par={}, size={}, threads={}, method={}",
                    seq_pass.observed,
                    par_pass.observed,
                    n,
                    threads,
                    method
                );

                let diff_fail = (seq_fail.observed - par_fail.observed).abs();
                assert!(
                    diff_fail < 1e-12,
                    "Observed value drift in fail: seq={}, par={}, size={}, threads={}, method={}",
                    seq_fail.observed,
                    par_fail.observed,
                    n,
                    threads,
                    method
                );
            }
        }
    }
}

#[test]
fn test_parameterized_parallel_chunked_nan_inf_fail_closed() {
    let n = 131_072_usize;
    let thread_counts = [1_usize, 2, 4, 8];
    let bad_indices = [0_usize, n / 4, n / 2, 3 * n / 4, n - 1];

    let tol = ToleranceSpec::default();

    for &bad_idx in &bad_indices {
        for &threads in &thread_counts {
            // Test NaN injection
            let oracle = vec![1.0_f64; n];
            let mut peer_nan = vec![1.0_f64; n];
            *peer_nan.get_mut(bad_idx).unwrap() = f64::NAN;

            let res_nan = compare_float_arrays_parallel(
                &oracle, &peer_nan, &tol, threads,
            );
            assert_eq!(
                res_nan.verdict, "fail",
                "NaN at {bad_idx} must fail with {threads} threads"
            );
            assert!(
                res_nan
                    .details
                    .as_deref()
                    .unwrap()
                    .contains("NaN or Infinite")
            );

            // Test Infinity injection
            let mut peer_inf = vec![1.0_f64; n];
            *peer_inf.get_mut(bad_idx).unwrap() = f64::INFINITY;

            let res_inf = compare_float_arrays_parallel(
                &oracle, &peer_inf, &tol, threads,
            );
            assert_eq!(
                res_inf.verdict, "fail",
                "Inf at {bad_idx} must fail with {threads} threads"
            );
            assert!(
                res_inf
                    .details
                    .as_deref()
                    .unwrap()
                    .contains("NaN or Infinite")
            );
        }
    }
}

#[test]
fn test_parameterized_parallel_chunked_length_mismatch() {
    let o = vec![1.0_f64; 100_000];
    let p = vec![1.0_f64; 99_999];
    let tol = ToleranceSpec::default();

    let res = compare_float_arrays_parallel(&o, &p, &tol, 4);
    assert_eq!(res.verdict, "fail");
    assert!(res.details.as_deref().unwrap().contains("Length mismatch"));
}
