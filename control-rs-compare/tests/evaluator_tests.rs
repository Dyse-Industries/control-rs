//! Unit tests for typed numerical comparison algorithms, recursive dataset discovery,
//! and tolerance discovery from external tables and HDF5 attributes.

#![allow(
    clippy::arbitrary_source_item_ordering,
    clippy::float_cmp,
    clippy::indexing_slicing,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::unwrap_used
)]

use control_rs_compare::compare::{
    ToleranceSpec, compare_dataset, compare_float_arrays,
    compare_float_arrays_parallel, discover_datasets,
    resolve_signal_tolerances,
};
use control_rs_compare::config::ToleranceTable;
use hdf5_pure::{AttrValue, File, FileBuilder};

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

struct DiscoveryTestCase {
    name: &'static str,
    builder_fn: fn() -> Vec<u8>,
    expected: Vec<&'static str>,
}

#[test]
fn test_parameterized_dataset_discovery() {
    let test_cases = vec![
        DiscoveryTestCase {
            name: "empty_container",
            builder_fn: || {
                let builder = FileBuilder::new();
                builder.finish().unwrap()
            },
            expected: vec![],
        },
        DiscoveryTestCase {
            name: "flat_root_datasets",
            builder_fn: || {
                let mut b = FileBuilder::new();
                b.create_dataset("gamma").with_f64_data(&[3.0]);
                b.create_dataset("alpha").with_f64_data(&[1.0]);
                b.create_dataset("beta").with_f64_data(&[2.0]);
                b.finish().unwrap()
            },
            expected: vec!["alpha", "beta", "gamma"],
        },
        DiscoveryTestCase {
            name: "deeply_nested_hierarchy",
            builder_fn: || {
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

                b.finish().unwrap()
            },
            expected: vec!["a/b/c/d/e/leaf1", "a/b/c/leaf2", "a/leaf3"],
        },
        DiscoveryTestCase {
            name: "multi_branch_complex_tree",
            builder_fn: || {
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

                b.finish().unwrap()
            },
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
            builder_fn: || {
                let mut b = FileBuilder::new();
                let g_empty1 = b.create_group("empty_g1");
                b.add_group(g_empty1.finish());

                let mut g2 = b.create_group("g2");
                let g_sub_empty = g2.create_group("empty_sub");
                g2.add_group(g_sub_empty.finish());
                g2.create_dataset("valid_ds").with_f64_data(&[42.0]);
                b.add_group(g2.finish());

                b.finish().unwrap()
            },
            expected: vec!["g2/valid_ds"],
        },
        DiscoveryTestCase {
            name: "underscore_metadata_and_internal_excluded",
            builder_fn: || {
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

                b.finish().unwrap()
            },
            expected: vec!["signals/public_output"],
        },
    ];

    for tc in test_cases {
        let bytes = (tc.builder_fn)();
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
fn test_tolerance_discovery_from_table() {
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
    assert_eq!(pol_default.methods[0].method, "matrix_norm");
    assert!((pol_default.methods[0].bound - 0.05).abs() < 1e-9);

    // 2. Peer override: `matlab`
    let pol_matlab =
        resolve_signal_tolerances(None, "matrix/a", "matlab", Some(&table));
    assert_eq!(pol_matlab.methods[0].bound, 0.10);

    // 3. Peer override: python
    let pol_python =
        resolve_signal_tolerances(None, "matrix/a", "python", Some(&table));
    assert_eq!(pol_python.methods[0].bound, 0.02);

    // 4. Multi-method with `any_of` policy
    let pol_trans = resolve_signal_tolerances(
        None,
        "transient/v_out",
        "rust",
        Some(&table),
    );
    assert_eq!(pol_trans.policy, "any_of");
    assert_eq!(pol_trans.methods.len(), 2);
    assert_eq!(pol_trans.methods[0].method, "rel");
    assert_eq!(pol_trans.methods[0].bound, 0.01);
    assert_eq!(pol_trans.methods[1].method, "rms");
    assert_eq!(pol_trans.methods[1].bound, 0.005);
}

#[test]
fn test_tolerance_discovery_from_hdf5_attributes() {
    // 1. Single measure & bound with peer override
    let mut b1 = FileBuilder::new();
    let ds1 = b1.create_dataset("sig1");
    ds1.with_f64_data(&[1.0, 2.0]);
    ds1.set_attr("measure", AttrValue::String("matrix_norm".to_string()));
    ds1.set_attr("bound", AttrValue::F64(0.01));
    ds1.set_attr("bound.matlab", AttrValue::F64(0.05));
    let bytes1 = b1.finish().unwrap();
    let file1 = File::from_bytes(bytes1).unwrap();
    let ds1_obj = file1.dataset("sig1").unwrap();

    let pol1 = resolve_signal_tolerances(Some(&ds1_obj), "sig1", "rust", None);
    assert_eq!(pol1.methods.len(), 1);
    assert_eq!(pol1.methods[0].method, "matrix_norm");
    assert_eq!(pol1.methods[0].bound, 0.01);

    let pol1_matlab =
        resolve_signal_tolerances(Some(&ds1_obj), "sig1", "matlab", None);
    assert_eq!(pol1_matlab.methods[0].bound, 0.05);

    // 2. Multi-method JSON attribute with `any_of` policy
    let mut b2 = FileBuilder::new();
    let ds2 = b2.create_dataset("sig2");
    ds2.with_f64_data(&[1.0, 2.0]);
    ds2.set_attr(
        "methods",
        AttrValue::String(
            r#"[{"type": "rms", "bound": 0.002}, {"type": "rel", "bound": 0.05}]"#
                .to_string(),
        ),
    );
    ds2.set_attr("policy", AttrValue::String("any_of".to_string()));
    let bytes2 = b2.finish().unwrap();
    let file2 = File::from_bytes(bytes2).unwrap();
    let ds2_obj = file2.dataset("sig2").unwrap();

    let pol2 = resolve_signal_tolerances(Some(&ds2_obj), "sig2", "rust", None);
    assert_eq!(pol2.policy, "any_of");
    assert_eq!(pol2.methods.len(), 2);
    assert_eq!(pol2.methods[0].method, "rms");
    assert_eq!(pol2.methods[0].bound, 0.002);
    assert_eq!(pol2.methods[1].method, "rel");
    assert_eq!(pol2.methods[1].bound, 0.05);

    // 3. Fallback default when no attributes or table
    let mut b3 = FileBuilder::new();
    b3.create_dataset("sig3").with_f64_data(&[1.0]);
    let bytes3 = b3.finish().unwrap();
    let file3 = File::from_bytes(bytes3).unwrap();
    let ds3_obj = file3.dataset("sig3").unwrap();

    let pol3 = resolve_signal_tolerances(Some(&ds3_obj), "sig3", "rust", None);
    assert_eq!(pol3.policy, "all_of");
    assert_eq!(pol3.methods.len(), 1);
    assert_eq!(pol3.methods[0].method, "abs");
    assert_eq!(pol3.methods[0].bound, 1e-4);

    // 4. Single 'method' attribute + F32 bound
    let mut b4 = FileBuilder::new();
    let ds4 = b4.create_dataset("sig4");
    ds4.with_f64_data(&[1.0]);
    ds4.set_attr("method", AttrValue::String("rel".to_string()));
    ds4.set_attr("bound", AttrValue::F32(0.005_f32));
    let bytes4 = b4.finish().unwrap();
    let file4 = File::from_bytes(bytes4).unwrap();
    let ds4_obj = file4.dataset("sig4").unwrap();

    let pol4 = resolve_signal_tolerances(Some(&ds4_obj), "sig4", "rust", None);
    assert_eq!(pol4.methods.len(), 1);
    assert_eq!(pol4.methods[0].method, "rel");
    assert!((pol4.methods[0].bound - 0.005).abs() < 1e-6);

    // 5. 'bound' attribute only (defaults to abs)
    let mut b5 = FileBuilder::new();
    let ds5 = b5.create_dataset("sig5");
    ds5.with_f64_data(&[1.0]);
    ds5.set_attr("bound", AttrValue::I32(1));
    let bytes5 = b5.finish().unwrap();
    let file5 = File::from_bytes(bytes5).unwrap();
    let ds5_obj = file5.dataset("sig5").unwrap();

    let pol5 = resolve_signal_tolerances(Some(&ds5_obj), "sig5", "rust", None);
    assert_eq!(pol5.methods.len(), 1);
    assert_eq!(pol5.methods[0].method, "abs");
    assert_eq!(pol5.methods[0].bound, 1.0);

    // 6. External table takes precedence over dataset attributes
    let table_override: ToleranceTable = toml::from_str(
        r#"
[signals.sig1]
method = "rms"
bound = 0.0001
"#,
    )
    .unwrap();

    let pol_precedence = resolve_signal_tolerances(
        Some(&ds1_obj),
        "sig1",
        "rust",
        Some(&table_override),
    );
    assert_eq!(pol_precedence.methods.len(), 1);
    assert_eq!(pol_precedence.methods[0].method, "rms");
    assert_eq!(pol_precedence.methods[0].bound, 0.0001);
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
        #[allow(clippy::cast_precision_loss)]
        let oracle: Vec<f64> =
            (0..n).map(|i| (i as f64 * 0.001).sin()).collect();
        let peer_pass: Vec<f64> = oracle
            .iter()
            .map(|&v| 1e-6f64.mul_add((v * 0.5).cos(), v))
            .collect();
        let mut peer_fail = peer_pass.clone();
        if n > 0 {
            peer_fail[n / 2] += 0.5;
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
            peer_nan[bad_idx] = f64::NAN;

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
            peer_inf[bad_idx] = f64::INFINITY;

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
