//! Load a suite JSON file and bind host-case inputs.

use std::path::Path;
use std::process::exit;

use control_rs::math::num_types::{Const, Dim};
use control_rs::math::storage::{ArrayStorage, StorageInit};
use serde_json::Value;

/// Helper to print to stderr and exit 1.
fn fatal(msg: &str) -> ! {
    eprintln!("{msg}");
    exit(1);
}

/// Load a suite document from `path`. Exits 1 on I/O or parse failure.
#[must_use]
pub fn load_suite(path: &Path) -> Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        fatal(&format!("failed to read {}: {e}", path.display()))
    });
    serde_json::from_str(&text).unwrap_or_else(|e| {
        fatal(&format!("failed to parse {}: {e}", path.display()))
    })
}

/// CLI wrapper: require `<suite.json>`, check `slug`, run `body`.
pub fn run_cli(slug: &str, body: fn(&Value)) {
    let mut args = std::env::args();
    let bin = args.next().unwrap_or_else(|| "bin".to_string());

    let Some(path) = args.next() else {
        eprintln!("usage: {bin} <suite.json>");
        exit(2);
    };

    // Catch cases where too many arguments are provided
    if args.next().is_some() {
        eprintln!("usage: {bin} <suite.json>");
        exit(2);
    }

    let suite = load_suite(Path::new(&path));
    let got = suite["slug"].as_str().unwrap_or("");
    if got != slug {
        fatal(&format!("suite slug `{got}` does not match `{slug}`"));
    }
    body(&suite);
}

/// `inputs` object for case `id`. Exits 1 if missing.
#[must_use]
pub fn case_inputs<'a>(suite: &'a Value, id: &str) -> &'a Value {
    suite["cases"]
        .as_array()
        .unwrap_or_else(|| fatal("suite has no cases array"))
        .iter()
        .find(|c| c["id"].as_str() == Some(id))
        .map(|c| &c["inputs"])
        .unwrap_or_else(|| fatal(&format!("missing case `{id}`")))
}

/// Require `inputs[key]` to equal a compiled size.
pub fn require_usize(inputs: &Value, key: &str, expected: usize) {
    let n = json_usize(&inputs[key]).unwrap_or_else(|| {
        fatal(&format!("inputs.{key} is not a non-negative integer"))
    });
    if n != expected {
        fatal(&format!("inputs.{key}={n} != compiled {expected}"));
    }
}

/// Parse a JSON number as `usize`.
#[must_use]
pub fn json_usize(v: &Value) -> Option<usize> {
    v.as_u64().map(|n| n as usize).or_else(|| {
        v.as_f64()
            .filter(|x| x.is_finite() && *x >= 0.0 && x.fract() == 0.0)
            .map(|x| x as usize)
    })
}

/// JSON number as `f64`.
#[must_use]
pub fn json_f64(v: &Value) -> f64 {
    v.as_f64()
        .unwrap_or_else(|| fatal(&format!("expected number, got {v}")))
}

/// Flat `f64` vector from a JSON array of numbers.
#[must_use]
pub fn json_f64_vec(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| fatal(&format!("expected number array, got {v}")))
        .iter()
        .map(json_f64)
        .collect()
}

/// Row-major matrix from JSON `[[...], ...]`.
#[must_use]
pub fn json_rows(v: &Value) -> Vec<Vec<f64>> {
    v.as_array()
        .unwrap_or_else(|| {
            fatal(&format!("expected row-major matrix, got {v}"))
        })
        .iter()
        .map(json_f64_vec)
        .collect()
}

/// Dense storage from a row-major JSON matrix via logical `(i, j)` coordinates.
#[must_use]
pub fn storage_from_rows_init<S, const R: usize, const C: usize>(v: &Value) -> S
where
    S: StorageInit<f64, Const<R>, Const<C>>,
    Const<R>: Dim,
    Const<C>: Dim,
{
    let rows = json_rows(v);
    if rows.len() != R {
        fatal(&format!("matrix has {} rows, expected {R}", rows.len()));
    }
    for (i, row) in rows.iter().enumerate() {
        if row.len() != C {
            fatal(&format!("row {i} has {} cols, expected {C}", row.len()));
        }
    }
    S::from_fn(|i, j| rows[i][j])
}

/// Column-major `ArrayStorage` from a row-major JSON matrix.
#[must_use]
pub fn storage_from_rows<const R: usize, const C: usize>(
    v: &Value,
) -> ArrayStorage<f64, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    let rows = json_rows(v);
    if rows.len() != R {
        fatal(&format!("matrix has {} rows, expected {R}", rows.len()));
    }
    let mut cols = [[0.0_f64; R]; C];
    for (i, row) in rows.iter().enumerate() {
        if row.len() != C {
            fatal(&format!("row {i} has {} cols, expected {C}", row.len()));
        }
        for (j, &val) in row.iter().enumerate() {
            cols[j][i] = val;
        }
    }
    ArrayStorage::from_array(cols)
}

/// Length-`N` column storage from a JSON vector.
#[must_use]
pub fn col_array<const N: usize>(v: &Value) -> [f64; N] {
    let vals = json_f64_vec(v);
    let Ok(arr) = vals.try_into() else {
        fatal(&format!("vector length mismatch, expected {N}"));
    };
    arr
}

/// Length-`N` column storage from a JSON vector.
#[must_use]
pub fn storage_from_col<const N: usize>(v: &Value) -> ArrayStorage<f64, N, 1>
where
    Const<N>: Dim,
{
    ArrayStorage::from_array([col_array::<N>(v)])
}

/// Pretty-print a JSON artifact to stdout and flush (stdout is a pipe under `validate`).
pub fn emit_stdout(doc: &Value) {
    use std::io::{Write, stdout};
    let text = serde_json::to_string_pretty(doc).expect("serialize json");
    let mut out = stdout().lock();
    writeln!(out, "{text}").expect("write json to stdout");
    out.flush().expect("flush json stdout");
}
