//! Pure-Rust HDF5 container writer utilizing `hdf5-pure`.
//!
//! Emits structured `.rust.h5` and `.scipy.h5` control-rs-verification containers containing
//! nested group hierarchies and signal tolerance metadata.

#![allow(
    missing_docs,
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::indexing_slicing,
    clippy::unwrap_used
)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use hdf5_pure::{AttrValue, FileBuilder};

/// Builder for creating and serializing HDF5 containers in pure Rust.
#[derive(Debug, Default)]
pub struct H5Writer {
    root_datasets: BTreeMap<String, Vec<f64>>,
    groups: BTreeMap<String, BTreeMap<String, Vec<f64>>>,
    tolerances: BTreeMap<String, (String, f64)>,
}

impl H5Writer {
    /// Creates a new, empty container writer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a 1D or flattened numeric dataset at `path` (for example, `"matrix/a"` or `"v_out"`).
    pub fn add_dataset(&mut self, path: &str, data: &[f64]) {
        let clean = path.trim_start_matches('/');
        if let Some((group, ds)) = clean.split_once('/') {
            self.groups
                .entry(group.to_string())
                .or_default()
                .insert(ds.to_string(), data.to_vec());
        } else {
            self.root_datasets.insert(clean.to_string(), data.to_vec());
        }
    }

    /// Attaches tolerance metadata to a signal path.
    pub fn set_tolerance(&mut self, path: &str, measure: &str, bound: f64) {
        let clean = path.trim_start_matches('/').to_string();
        self.tolerances.insert(clean, (measure.to_string(), bound));
    }

    /// Writes the container out to the target file path.
    ///
    /// # Errors
    /// Returns an error string if serialization or filesystem write fails.
    pub fn write_to_file(&self, path: &Path) -> Result<(), String> {
        let mut b = FileBuilder::new();

        // 1. Root datasets
        for (name, data) in &self.root_datasets {
            let ds = b.create_dataset(name);
            ds.with_f64_data(data);
            if let Some((measure, bound)) = self.tolerances.get(name) {
                ds.set_attr("measure", AttrValue::String(measure.clone()));
                ds.set_attr("bound", AttrValue::F64(*bound));
            }
        }

        // 2. Nested group datasets
        for (group_name, datasets) in &self.groups {
            let mut g = b.create_group(group_name);
            for (ds_name, data) in datasets {
                let full_path = format!("{group_name}/{ds_name}");
                let ds = g.create_dataset(ds_name);
                ds.with_f64_data(data);
                if let Some((measure, bound)) = self.tolerances.get(&full_path)
                {
                    ds.set_attr("measure", AttrValue::String(measure.clone()));
                    ds.set_attr("bound", AttrValue::F64(*bound));
                }
            }
            b.add_group(g.finish());
        }

        let bytes = b
            .finish()
            .map_err(|e| format!("FileBuilder serialization failed: {e:?}"))?;

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create parent dir: {e}"))?;
        }

        fs::write(path, bytes).map_err(|e| {
            format!("Failed to write container {}: {e}", path.display())
        })?;

        Ok(())
    }
}

/// Resolves the absolute path to the workspace `target/verification/` directory.
#[must_use]
pub fn results_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).parent().map_or_else(
        || PathBuf::from("target/verification"),
        |root| root.join("target").join("verification"),
    )
}
