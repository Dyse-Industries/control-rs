//! Spawn suite validators and plotters, then compare result JSON.

#![allow(clippy::type_complexity)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio, exit};

use control_rs_numerical_model_examples::compare::compare_slug;
use control_rs_numerical_model_examples::suite::load_suite;
use serde_json::Value;

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn usage() -> ! {
    eprintln!("usage: validate <suites/ | suites/<slug>.json>");
    exit(2);
}

fn collect_suite_paths(arg: &Path) -> Vec<PathBuf> {
    if arg.is_dir() {
        let mut paths: Vec<PathBuf> = fs::read_dir(arg)
            .unwrap_or_else(|e| {
                eprintln!("read {}: {e}", arg.display());
                exit(1);
            })
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
            .collect();
        paths.sort();
        if paths.is_empty() {
            eprintln!("no .json suites in {}", arg.display());
            exit(1);
        }
        paths
    } else if arg.is_file() {
        vec![arg.to_path_buf()]
    } else {
        eprintln!("not a file or directory: {}", arg.display());
        exit(1);
    }
}

fn sibling_bin(name: &str) -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_else(|e| {
        eprintln!("current_exe: {e}");
        exit(1);
    });
    exe.parent().unwrap_or(Path::new(".")).join(name)
}

fn running_release() -> bool {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.to_str().map(str::to_string))
        .is_some_and(|p| p.contains("/release/") || p.contains("\\release\\"))
}

fn host_vendor() -> &'static str {
    if cfg!(target_vendor = "apple") {
        "apple"
    } else {
        "other"
    }
}

fn when_matches(entry: &Value) -> bool {
    let Some(when) = entry.get("when") else {
        return true;
    };
    if let Some(want) = when["target_vendor"].as_str()
        && want != host_vendor()
    {
        return false;
    }
    true
}

fn is_optional(entry: &Value) -> bool {
    entry["optional"].as_bool() == Some(true)
}

fn validator_features(entry: &Value) -> Vec<String> {
    entry["features"]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn collect_active_validators(suite_paths: &[PathBuf]) -> Vec<Value> {
    let mut out = Vec::new();
    for path in suite_paths {
        let suite = load_suite(path);
        let Some(validators) = suite["validators"].as_array() else {
            continue;
        };
        for v in validators {
            if when_matches(v) {
                out.push(v.clone());
            }
        }
    }
    out
}

/// `cargo run --bin validate` does not rebuild sibling validator bins.
fn rebuild_validators(entries: &[Value]) -> Vec<String> {
    let mut skipped = Vec::new();
    let mut groups: Vec<(Vec<String>, Vec<(String, bool)>)> = Vec::new();
    for v in entries {
        let Some(bin) = v["bin"].as_str() else {
            continue;
        };
        let features = validator_features(v);
        let optional = is_optional(v);
        if let Some((_, bins)) = groups.iter_mut().find(|(f, _)| *f == features)
        {
            if !bins.iter().any(|(b, _)| b == bin) {
                bins.push((bin.to_string(), optional));
            }
        } else {
            groups.push((features, vec![(bin.to_string(), optional)]));
        }
    }

    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let manifest = crate_root().join("Cargo.toml");
    for (features, bins) in groups {
        if bins.is_empty() {
            continue;
        }
        let mut cmd = Command::new(&cargo);
        cmd.arg("build")
            .arg("--manifest-path")
            .arg(&manifest)
            .arg("--quiet");
        if running_release() {
            cmd.arg("--release");
        }
        if !features.is_empty() {
            cmd.arg("--features").arg(features.join(","));
        }
        for (bin, _) in &bins {
            cmd.arg("--bin").arg(bin);
        }
        cmd.current_dir(crate_root());
        let status = cmd.status().unwrap_or_else(|e| {
            eprintln!("failed to spawn cargo build: {e}");
            exit(1);
        });
        if !status.success() {
            let all_optional = bins.iter().all(|(_, o)| *o);
            if all_optional {
                eprintln!(
                    "optional cargo build failed (features {:?}); skipping {}",
                    features,
                    bins.iter()
                        .map(|(b, _)| b.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                for (bin, _) in bins {
                    skipped.push(bin);
                }
            } else {
                eprintln!("cargo build of validator bins failed");
                exit(1);
            }
        }
    }
    skipped
}

fn stdout_preview(text: &str) -> String {
    let t = text.trim_start();
    let take = t.chars().take(80).collect::<String>();
    if t.is_empty() {
        "<empty>".to_string()
    } else {
        take.replace('\n', "\\n")
    }
}

fn spawn_validator(
    entry: &Value,
    suite_path: &Path,
    cwd: &Path,
) -> Result<(String, Value), String> {
    let source = entry["source"]
        .as_str()
        .ok_or_else(|| "validator missing source".to_string())?
        .to_string();
    let mut cmd = if let Some(bin) = entry["bin"].as_str() {
        let path = sibling_bin(bin);
        if !path.is_file() {
            return Err(format!(
                "missing sibling bin {} (run cargo build --release)",
                path.display()
            ));
        }
        let mut c = Command::new(&path);
        c.arg(suite_path);
        c
    } else if let Some(argv) = entry["argv"].as_array() {
        let args: Vec<String> = argv
            .iter()
            .map(|v| v.as_str().unwrap_or("").to_string())
            .collect();
        if args.is_empty() {
            return Err("validator argv is empty".to_string());
        }
        let mut c = Command::new(&args[0]);
        c.args(&args[1..]).arg(suite_path);
        c
    } else {
        return Err("validator needs argv or bin".to_string());
    };
    cmd.current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());
    let out = cmd
        .output()
        .map_err(|e| format!("{source}: spawn failed: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "{source}: exited {}",
            out.status.code().unwrap_or(-1)
        ));
    }
    let text = String::from_utf8(out.stdout)
        .map_err(|e| format!("{source}: stdout is not UTF-8: {e}"))?;
    let doc: Value = serde_json::from_str(text.trim()).map_err(|e| {
        format!(
            "{source}: invalid JSON on stdout: {e} (got: {})",
            stdout_preview(&text)
        )
    })?;
    Ok((source, doc))
}

fn spawn_plotter(
    entry: &Value,
    results_dir: &Path,
    cwd: &Path,
) -> Result<(), String> {
    let argv = entry["argv"].as_array().ok_or("plot missing argv")?;
    let args: Vec<String> = argv
        .iter()
        .map(|v| v.as_str().unwrap_or("").to_string())
        .collect();
    if args.is_empty() {
        return Err("plot argv is empty".to_string());
    }
    let status = Command::new(&args[0])
        .args(&args[1..])
        .arg(results_dir)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .status()
        .map_err(|e| format!("plot spawn failed: {e}"))?;
    if !status.success() {
        return Err(format!(
            "plot {} exited {}",
            args[0],
            status.code().unwrap_or(-1)
        ));
    }
    Ok(())
}

fn run_suite(
    suite_path: &Path,
    cwd: &Path,
    skipped_bins: &[String],
) -> Result<(), Vec<String>> {
    let suite = load_suite(suite_path);
    let slug = suite["slug"].as_str().unwrap_or("unknown").to_string();
    let results_dir = cwd.join("results").join(&slug);
    fs::create_dir_all(&results_dir)
        .map_err(|e| vec![format!("mkdir: {e}")])?;

    let validators =
        suite["validators"].as_array().cloned().unwrap_or_default();
    if validators.is_empty() {
        return Err(vec![format!("{slug}: no validators")]);
    }

    let mut artifacts = Vec::new();
    let mut spawn_errs = Vec::new();
    for v in &validators {
        if !when_matches(v) {
            let src = v["source"].as_str().unwrap_or("?");
            eprintln!("skip {src}: when clause not matched on this host");
            continue;
        }
        if let Some(bin) = v["bin"].as_str()
            && skipped_bins.iter().any(|b| b == bin)
        {
            let src = v["source"].as_str().unwrap_or("?");
            eprintln!("skip {src}: optional bin `{bin}` was not built");
            continue;
        }
        match spawn_validator(v, suite_path, cwd) {
            Ok((source, doc)) => {
                let dest = results_dir.join(format!("{source}.json"));
                let text =
                    serde_json::to_string_pretty(&doc).unwrap_or_default();
                if let Err(e) = fs::write(&dest, format!("{text}\n")) {
                    spawn_errs.push(format!("write {}: {e}", dest.display()));
                } else {
                    eprintln!("wrote {}", dest.display());
                }
                artifacts.push(doc);
            }
            Err(e) => {
                if is_optional(v) {
                    let src = v["source"].as_str().unwrap_or("?");
                    eprintln!("skip {src}: {e}");
                } else {
                    spawn_errs.push(e);
                }
            }
        }
    }
    if !spawn_errs.is_empty() {
        return Err(spawn_errs);
    }

    let mut failed = false;
    let mut errs = Vec::new();
    if let Err(e) = compare_slug(&slug, &artifacts) {
        errs.extend(e);
        failed = true;
    }

    let plots = suite["plots"].as_array().cloned().unwrap_or_default();
    for p in &plots {
        if let Err(e) = spawn_plotter(p, &results_dir, cwd) {
            errs.push(e);
            failed = true;
        }
    }

    if failed { Err(errs) } else { Ok(()) }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        usage();
    }
    let cwd = crate_root();
    let arg = Path::new(&args[1]);
    let arg = if arg.is_absolute() {
        arg.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| cwd.clone())
            .join(arg)
    };
    let paths = collect_suite_paths(&arg);
    let skipped_bins = rebuild_validators(&collect_active_validators(&paths));
    let mut failed = false;
    for path in &paths {
        eprintln!("=== {} ===", path.display());
        match run_suite(path, &cwd, &skipped_bins) {
            Ok(()) => eprintln!("{}: PASS", path.display()),
            Err(errs) => {
                failed = true;
                for e in errs {
                    eprintln!("error: {e}");
                }
                eprintln!("{}: FAIL", path.display());
            }
        }
    }
    if failed {
        exit(1);
    }
}
