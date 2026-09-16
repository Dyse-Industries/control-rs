//! Single Source of Truth (SSOT) for cross-validation tolerance bounds.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Optional interval enclosure `[min, max]` when measure is `"interval"`.
pub type Interval = [f64; 2];

/// Parsed table plus the TOML filenames it was merged from.
pub type LoadedTable = (ToleranceTable, Vec<String>);

/// A tolerance bound for a single verified metric or operation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToleranceBound {
    /// Subject name (e.g. "`buck_converter`", "`dc_motor`", "matrix").
    pub subject: String,
    /// Human-readable operation or metric name.
    pub operation: String,
    /// Primary or secondary oracle library name (e.g. "scipy", "ngspice", "jax").
    pub oracle_library: String,
    /// Measurement metric: "abs" | "rel" | "`rel_l2`" | "residual" | "exact" | "interval" | "lt".
    #[serde(default = "default_measure")]
    pub measure: String,
    /// Absolute or relative numeric error bound.
    #[serde(default)]
    pub bound: f64,
    /// Optional interval enclosure [min, max] when measure is "interval".
    #[serde(default)]
    pub interval: Option<Interval>,
    /// Optional justification or engineering rationale.
    #[serde(default)]
    pub justification: Option<String>,
    /// False when the paired oracle transcribes the same closed form as
    /// the Rust implementation (transcription check, not independent
    /// confirmation).
    #[serde(default = "default_independent")]
    pub independent: bool,
}

/// A parsed collection of tolerance bounds keyed by unique token.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ToleranceTable {
    /// Map of unique metric tokens to tolerance bounds.
    #[serde(default)]
    pub tolerances: HashMap<String, ToleranceBound>,
}

impl ToleranceTable {
    /// Parse tolerance table from TOML string.
    ///
    /// # Errors
    ///
    /// Returns the [`toml::de::Error`] raised when `s` is not a tolerance table.
    pub fn from_toml(s: &str) -> Result<Self, toml::de::Error> {
        toml::from_str(s)
    }

    /// Load tolerance table from a file path.
    ///
    /// # Errors
    ///
    /// Returns an error string when `path` cannot be read or does not parse as
    /// a tolerance table.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let content = fs::read_to_string(path).map_err(|e| {
            format!("Failed to read tolerance file '{}': {e}", path.display())
        })?;
        Self::from_toml(&content).map_err(|e| {
            format!("Failed to parse tolerance file '{}': {e}", path.display())
        })
    }

    /// Retrieve a bound by its unique key.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&ToleranceBound> {
        self.tolerances.get(key)
    }

    /// Number of declared keys.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tolerances.len()
    }

    /// Whether the table contains no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tolerances.is_empty()
    }

    /// Find the tolerance directory for an example crate.
    #[must_use]
    pub fn locate(
        example_dir: &Path,
        repo_root: &Path,
    ) -> Option<std::path::PathBuf> {
        let candidates = [
            example_dir.join("tolerances"),
            example_dir.join("src/tolerances"),
            repo_root.join("tolerances"),
        ];
        candidates.into_iter().find(|p| p.is_dir())
    }

    /// The declared key set.
    #[must_use]
    pub fn key_set(&self) -> std::collections::BTreeSet<String> {
        self.tolerances.keys().cloned().collect()
    }

    /// C-2: assert this table and its owning design document's §6.3 table
    /// declare exactly the same key set.
    ///
    /// Drift in either direction is a second declaration of a bound, which C-2
    /// forbids. `doc_text` is the document's contents, not a path: the gate
    /// resolves paths from configuration so that `control-rs-ci` carries no
    /// reference to any example or design document of its own.
    ///
    /// # Errors
    /// Returns a message naming the keys that differ, or reporting that the
    /// document declares no matching rows at all.
    pub fn check_doc_drift(
        &self,
        doc_text: &str,
        prefixes: &[String],
    ) -> Result<(), String> {
        let refs: Vec<&str> = prefixes.iter().map(String::as_str).collect();
        let documented = doc_table_keys(doc_text, &refs);
        if documented.is_empty() {
            return Err(format!(
                "design document declares no tolerance rows matching {refs:?}"
            ));
        }
        let declared = self.key_set();
        if declared == documented {
            return Ok(());
        }
        let only_toml: Vec<&String> =
            declared.difference(&documented).collect();
        let only_doc: Vec<&String> = documented.difference(&declared).collect();
        Err(format!(
            "tolerance keys drifted from the design document: {} only in TOML \
             {only_toml:?}, {} only in the document {only_doc:?}",
            only_toml.len(),
            only_doc.len()
        ))
    }

    /// Load and merge every `*.toml` table in `dir`.
    ///
    /// Duplicate keys across files are an error.
    ///
    /// # Errors
    /// Returns a message if the directory cannot be read, a file cannot be
    /// parsed, or two tables declare the same key.
    pub fn load_dir(dir: &Path) -> Result<LoadedTable, String> {
        let mut table = Self::default();
        let mut files = Vec::new();

        let entries = fs::read_dir(dir).map_err(|e| {
            format!("Failed to read tolerance directory {}: {e}", dir.display())
        })?;

        let mut paths: Vec<std::path::PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.is_file()
                    && p.extension().and_then(|e| e.to_str()) == Some("toml")
            })
            .collect();
        paths.sort();

        for path in paths {
            let parsed = Self::from_file(&path)?;
            for (key, bound) in parsed.tolerances {
                if table.tolerances.contains_key(&key) {
                    return Err(format!(
                        "Tolerance key '{key}' is declared in more than one \
                         table under {}",
                        dir.display()
                    ));
                }
                table.tolerances.insert(key, bound);
            }
            files.push(
                path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or_default()
                    .to_string(),
            );
        }

        Ok((table, files))
    }
}

const fn default_independent() -> bool {
    true
}

fn default_measure() -> String {
    "abs".to_string()
}

/// Extract the leading backticked key column of a markdown table, keeping rows
/// whose key carries one of `prefixes`.
///
/// Takes document text rather than a path. Nothing in `control-rs-ci` names a
/// design document or an example crate; the gate supplies both from
/// configuration.
#[must_use]
pub fn doc_table_keys(
    doc_text: &str,
    prefixes: &[&str],
) -> std::collections::BTreeSet<String> {
    let mut keys = std::collections::BTreeSet::new();
    for line in doc_text.lines() {
        let Some(rest) = line.trim_start().strip_prefix('|') else {
            continue;
        };
        let Some(rest) = rest.trim_start().strip_prefix('`') else {
            continue;
        };
        let Some(end) = rest.find('`') else {
            continue;
        };
        let Some(key) = rest.get(..end) else {
            continue;
        };
        if prefixes.iter().any(|p| key.starts_with(p)) {
            keys.insert(key.to_string());
        }
    }
    keys
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = r#"
[tolerances."demo.plant.natural_frequency"]
subject = "demo"
operation = "natural frequency"
oracle_library = "scipy"
measure = "rel"
bound = 1e-6

[tolerances."demo.plant.damping"]
subject = "demo"
operation = "damping ratio"
oracle_library = "scipy"
measure = "rel"
bound = 1e-6

[tolerances."demo.root_locus.poles_re"]
subject = "demo"
operation = "root locus real parts"
oracle_library = "scipy"
measure = "abs"
bound = 1e-9
"#;

    #[test]
    /// # Verification
    /// Trace: oracle-harness#C-2
    /// Method: Requirements-based test
    fn test_parse_tolerance_table() {
        let table =
            ToleranceTable::from_toml(FIXTURE).expect("fixture must parse");
        assert_eq!(table.len(), 3);
        assert!(table.get("demo.plant.natural_frequency").is_some());
        assert!(table.get("demo.root_locus.poles_re").is_some());
        assert!(table.get("demo.absent").is_none());
        assert!(!table.is_empty());
    }

    #[test]
    /// # Verification
    /// Trace: oracle-harness#C-2
    /// Method: Requirements-based test
    fn test_doc_table_keys_extracts_backticked_column() {
        let doc = "\
| Key | Measure | Bound |\n\
|:--|:--|--:|\n\
| `demo.plant.natural_frequency` | rel | 1e-6 |\n\
| `demo.plant.damping` | rel | 1e-6 |\n\
| `other.thing` | rel | 1e-6 |\n\
not a table row\n";
        let keys = doc_table_keys(doc, &["demo."]);
        assert_eq!(keys.len(), 2);
        assert!(keys.contains("demo.plant.damping"));
        assert!(!keys.contains("other.thing"));
    }

    /// C-2: the tolerance TOML and the owning design document's §6.3 table must
    /// declare exactly the same key set. Drift in either direction is a second
    /// declaration of a bound, which C-2 forbids. The gate runs this against
    /// configured paths; these fixtures verify the comparison itself.
    #[test]
    /// # Verification
    /// Trace: oracle-harness#C-2
    /// Method: Requirements-based test
    fn test_doc_drift_detects_both_directions() {
        let table =
            ToleranceTable::from_toml(FIXTURE).expect("fixture must parse");
        let prefixes = vec!["demo.".to_string()];

        let matching = "\
| `demo.plant.natural_frequency` | rel | 1e-6 |\n\
| `demo.plant.damping` | rel | 1e-6 |\n\
| `demo.root_locus.poles_re` | abs | 1e-9 |\n";
        assert!(table.check_doc_drift(matching, &prefixes).is_ok());

        // A bound declared in the TOML but absent from the document.
        let missing_row = "\
| `demo.plant.natural_frequency` | rel | 1e-6 |\n\
| `demo.plant.damping` | rel | 1e-6 |\n";
        let err = table
            .check_doc_drift(missing_row, &prefixes)
            .expect_err("undocumented key must fail");
        assert!(err.contains("demo.root_locus.poles_re"), "{err}");

        // A bound documented but absent from the TOML.
        let extra_row = "\
| `demo.plant.natural_frequency` | rel | 1e-6 |\n\
| `demo.plant.damping` | rel | 1e-6 |\n\
| `demo.root_locus.poles_re` | abs | 1e-9 |\n\
| `demo.ghost.bound` | abs | 1e-9 |\n";
        let err = table
            .check_doc_drift(extra_row, &prefixes)
            .expect_err("undeclared key must fail");
        assert!(err.contains("demo.ghost.bound"), "{err}");

        // A document with no matching rows is a failure, not a vacuous pass.
        let err = table
            .check_doc_drift("| `other.thing` | rel | 1e-6 |", &prefixes)
            .expect_err("no matching rows must fail");
        assert!(err.contains("no tolerance rows matching"), "{err}");
    }

    #[test]
    /// # Verification
    /// Trace: oracle-harness#C-2
    /// Method: Requirements-based test
    fn test_load_dir_rejects_duplicate_keys() {
        let dir = std::env::temp_dir().join("control-rs-ci-tolerance-dup");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("temp dir");
        fs::write(dir.join("a.toml"), FIXTURE).expect("write a");
        fs::write(dir.join("b.toml"), FIXTURE).expect("write b");
        let err = ToleranceTable::load_dir(&dir)
            .expect_err("a key declared twice must fail");
        assert!(err.contains("more than one"), "{err}");

        let _ = fs::remove_file(dir.join("b.toml"));
        let (table, files) =
            ToleranceTable::load_dir(&dir).expect("single table must load");
        assert_eq!(table.len(), 3);
        assert_eq!(files, vec!["a.toml".to_string()]);
        let _ = fs::remove_dir_all(&dir);
    }
}
