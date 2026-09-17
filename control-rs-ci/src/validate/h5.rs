//! Per-variant HDF5 files conforming to `oracle-harness-design.md`.

use rust_hdf5::types::{H5Type, VarLenUnicode};
use rust_hdf5::{H5Dataset, H5File, H5Group, ReadNumeric};
use std::fs;
use std::path::{Path, PathBuf};

use super::comparator::{PeerPath, compare_h5_files};
use super::envelope::{ContainerValidationReport, FreshnessPolicy};
use super::tolerance::{Interval, ToleranceBound};

/// Datasets discovered by a container traversal, excluding `/_meta`.
pub type DiscoveredDatasets = Vec<DiscoveredDataset>;

type GroupAndName = (H5Group, String);

/// Nested JSON object map used when projecting gated paths.
pub type JsonObject = serde_json::Map<String, serde_json::Value>;

/// Per-peer bound override (`bound.<variant>`).
pub type PeerBound = (String, f64);

/// Dataset shape together with flattened element storage.
pub type ShapeAndData<T> = (Vec<usize>, Vec<T>);

/// Result of reading a typed dataset's shape and values.
pub type DatasetRead<T> = Result<ShapeAndData<T>, String>;

type IntervalRead = Result<Option<Interval>, String>;
type JsonMatrix = Result<Vec<f64>, String>;
type PeerBoundsRead = Result<Vec<PeerBound>, String>;

/// Metadata describing a discovered numerical dataset within an HDF5 file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredDataset {
    /// Full HDF5 path (e.g. `/plant/natural_frequency_rad_s`).
    pub path: String,
    /// Signal path without the leading slash.
    pub signal: String,
    /// Dataset dimensions/shape.
    pub shape: Vec<usize>,
}

/// Tolerance attributes attached to a true-oracle dataset.
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetTolerance {
    /// Measurement metric: `abs` | `rel` | `rel_l2` | `residual` | `exact` | `interval` | `lt`.
    pub measure: String,
    /// Scalar bound (default for every peer).
    pub bound: f64,
    /// Optional interval enclosure when `measure` is `interval`.
    pub interval: Option<Interval>,
    /// False when the peer transcribes the same closed form.
    pub independent: bool,
    /// Optional per-peer bound overrides (`bound.<variant>`).
    pub peer_bounds: Vec<PeerBound>,
}

/// One-variant HDF5 wrapper.
pub struct H5Container {
    file: H5File,
}

/// Parsed variant file from a suite glob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SuiteH5File {
    /// Variant name parsed from the filename suffix.
    pub variant: String,
    /// Absolute or relative path to the file.
    pub path: PathBuf,
}

impl DatasetTolerance {
    /// Bound used when comparing against `peer`.
    #[must_use]
    pub fn bound_for(&self, peer: &str) -> f64 {
        self.peer_bounds
            .iter()
            .find(|(v, _)| v == peer)
            .map_or(self.bound, |(_, b)| *b)
    }

    /// In-memory tolerance row for comparator evaluation.
    #[must_use]
    pub fn as_bound(&self, peer: &str) -> ToleranceBound {
        ToleranceBound {
            subject: String::new(),
            operation: String::new(),
            oracle_library: peer.to_string(),
            measure: self.measure.clone(),
            bound: self.bound_for(peer),
            interval: self.interval,
            justification: None,
            independent: self.independent,
        }
    }
}

impl H5Container {
    /// Creates a new HDF5 file at `path` (truncating any existing file).
    ///
    /// # Errors
    ///
    /// Returns an error string when the parent directory cannot be created or
    /// the file cannot be opened for writing.
    pub fn create(path: &Path) -> Result<Self, String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                format!(
                    "Failed to create directory '{}': {e}",
                    parent.display()
                )
            })?;
        }
        let file = H5File::create(path).map_err(|e| {
            format!("Failed to create HDF5 file at '{}': {e}", path.display())
        })?;
        Ok(Self { file })
    }

    /// Opens an existing HDF5 file in read-only mode.
    ///
    /// # Errors
    ///
    /// Returns an error string when `path` cannot be opened as an HDF5 file.
    pub fn open(path: &Path) -> Result<Self, String> {
        let file = H5File::open(path).map_err(|e| {
            format!("Failed to open HDF5 file at '{}': {e}", path.display())
        })?;
        Ok(Self { file })
    }

    /// Flushes and closes the file.
    ///
    /// # Errors
    ///
    /// Returns an error string when buffers cannot be flushed.
    pub fn close(self) -> Result<(), String> {
        self.file
            .close()
            .map_err(|e| format!("Failed to close HDF5 file: {e}"))
    }

    fn ensure_parent_group(&self, path: &str) -> Result<GroupAndName, String> {
        let clean_path = path.trim_matches('/');
        let parts: Vec<&str> = clean_path.split('/').collect();
        if parts.is_empty() || parts.iter().all(|p| p.is_empty()) {
            return Err("Cannot create dataset with empty path".to_string());
        }

        let Some((last, group_parts)) = parts.split_last() else {
            return Err("Cannot create dataset with empty path".to_string());
        };
        let dataset_name = (*last).to_string();

        let mut current_group = self.file.root_group();
        for &segment in group_parts {
            if segment.is_empty() {
                continue;
            }
            let existing_groups = current_group.group_names().map_err(|e| {
                format!(
                    "Failed to inspect groups in '{}': {e}",
                    current_group.name()
                )
            })?;

            if existing_groups.iter().any(|g| g == segment) {
                current_group = current_group.group(segment).map_err(|e| {
                    format!("Failed to open sub-group '{segment}': {e}")
                })?;
            } else {
                current_group =
                    current_group.create_group(segment).map_err(|e| {
                        format!("Failed to create sub-group '{segment}': {e}")
                    })?;
            }
        }

        Ok((current_group, dataset_name))
    }

    /// Writes a 1D numerical dataset at `path`.
    ///
    /// # Errors
    ///
    /// Returns an error string when the dataset cannot be allocated or written.
    pub fn write_dataset_1d<T: H5Type>(
        &self,
        path: &str,
        data: &[T],
    ) -> Result<(), String> {
        let (group, name) = self.ensure_parent_group(path)?;
        let ds = group
            .new_dataset::<T>()
            .shape([data.len()])
            .create(&name)
            .map_err(|e| {
                format!("Failed to create 1D dataset '{path}': {e}")
            })?;
        ds.write_raw(data)
            .map_err(|e| format!("Failed to write data to '{path}': {e}"))
    }

    /// Writes a 2D numerical dataset at `path`.
    ///
    /// # Errors
    ///
    /// Returns an error string when `data.len()` does not match `shape`.
    pub fn write_dataset_2d<T: H5Type>(
        &self,
        path: &str,
        shape: [usize; 2],
        data: &[T],
    ) -> Result<(), String> {
        let [rows, cols] = shape;
        if rows.checked_mul(cols) != Some(data.len()) {
            return Err(format!(
                "Data length {} does not match 2D shape [{rows}, {cols}]",
                data.len(),
            ));
        }
        let (group, name) = self.ensure_parent_group(path)?;
        let ds = group
            .new_dataset::<T>()
            .shape(shape)
            .create(&name)
            .map_err(|e| {
                format!("Failed to create 2D dataset '{path}': {e}")
            })?;
        ds.write_raw(data)
            .map_err(|e| format!("Failed to write 2D data to '{path}': {e}"))
    }

    /// Writes true-oracle tolerance attributes on the dataset at `path`.
    ///
    /// # Errors
    ///
    /// Returns an error string when the dataset cannot be opened or an
    /// attribute cannot be written.
    pub fn write_dataset_tolerance(
        &self,
        path: &str,
        spec: &DatasetTolerance,
    ) -> Result<(), String> {
        let ds = self.file.dataset_writer(path).map_err(|e| {
            format!("Failed to open dataset '{path}' for attributes: {e}")
        })?;
        write_measure_attr(&ds, path, &spec.measure)?;
        write_f64_attr(&ds, path, "bound", spec.bound)?;
        write_interval_attr(&ds, path, spec.interval)?;
        write_independent_attr(&ds, path, spec.independent)?;
        write_peer_bound_attrs(&ds, path, &spec.peer_bounds)?;
        Ok(())
    }

    /// Reads true-oracle tolerance attributes from `path`.
    ///
    /// # Errors
    ///
    /// Returns an error string when required attributes are missing.
    pub fn read_dataset_tolerance(
        &self,
        path: &str,
    ) -> Result<DatasetTolerance, String> {
        let ds = self.file.dataset(path).map_err(|e| {
            format!("Failed to open dataset '{path}' for attributes: {e}")
        })?;
        let names = ds.attr_names().map_err(|e| {
            format!("Failed to list attributes on '{path}': {e}")
        })?;
        check_required_tolerance_attrs(path, &names)?;
        let measure = read_measure(&ds, path)?;
        if measure == "interval" && !has_attr(&names, "interval") {
            return Err(format!(
                "dataset '{path}' measure=interval missing required attribute 'interval'"
            ));
        }
        Ok(DatasetTolerance {
            measure,
            bound: read_bound_value(&ds, path, &names)?,
            interval: read_interval_value(&ds, path, &names)?,
            independent: read_independent_flag(&ds, path, &names)?,
            peer_bounds: read_peer_bounds(&ds, path, &names)?,
        })
    }

    fn join_path(prefix: &str, key: &str) -> String {
        let p = prefix.trim_end_matches('/');
        if p.is_empty() || p == "/" {
            format!("/{key}")
        } else if p.starts_with('/') {
            format!("{p}/{key}")
        } else {
            format!("/{p}/{key}")
        }
    }

    /// Recursively writes a numeric JSON object under `prefix`.
    ///
    /// # Errors
    ///
    /// Returns an error string when a node is not an object or numeric array.
    pub fn write_json_group(
        &self,
        prefix: &str,
        value: &serde_json::Value,
    ) -> Result<(), String> {
        match value {
            serde_json::Value::Object(map) => {
                for (k, v) in map {
                    let subpath = Self::join_path(prefix, k);
                    self.write_json_group(&subpath, v)?;
                }
                Ok(())
            }
            serde_json::Value::Array(arr) => {
                write_json_array(self, prefix, arr)
            }
            serde_json::Value::Number(_) => {
                let f = json_leaf_f64(value, prefix)?;
                self.write_dataset_1d(prefix, &[f])
            }
            _ => Err(format!(
                "non-numeric JSON value at '{prefix}' cannot be written as an HDF5 dataset"
            )),
        }
    }

    /// Writes the numeric subset of `value` under `prefix`.
    ///
    /// # Errors
    ///
    /// Returns an error string when nothing numeric remains.
    pub fn write_numeric_json_group(
        &self,
        prefix: &str,
        value: &serde_json::Value,
    ) -> Result<(), String> {
        let numeric = retain_numeric_json(value).ok_or_else(|| {
            format!("no numeric datasets to write under '{prefix}'")
        })?;
        self.write_json_group(prefix, &numeric)
    }

    /// Traverses datasets, skipping `/_meta`.
    ///
    /// # Errors
    ///
    /// Returns an error string when the hierarchy cannot be walked.
    pub fn traverse_datasets(&self) -> Result<DiscoveredDatasets, String> {
        let mut discovered = Vec::new();
        let root = self.file.root_group();
        self.traverse_group_recursive(&root, &mut discovered)?;
        discovered.sort_by(|a, b| a.path.cmp(&b.path));
        Ok(discovered)
    }

    fn traverse_group_recursive(
        &self,
        group: &H5Group,
        discovered: &mut DiscoveredDatasets,
    ) -> Result<(), String> {
        if group.name() == "/_meta" || group.name().starts_with("/_meta/") {
            return Ok(());
        }

        let datasets = group.dataset_names().map_err(|e| {
            format!("Failed to read dataset names in '{}': {e}", group.name())
        })?;

        for ds_name in datasets {
            let full_path = if group.name() == "/" {
                format!("/{ds_name}")
            } else {
                format!("{}/{}", group.name(), ds_name)
            };
            if full_path == "/_meta" || full_path.starts_with("/_meta/") {
                continue;
            }
            let ds = self.file.dataset(&full_path).map_err(|e| {
                format!("Failed to open dataset '{full_path}': {e}")
            })?;
            let signal = full_path.trim_start_matches('/').to_string();
            discovered.push(DiscoveredDataset {
                path: full_path,
                signal,
                shape: ds.shape().clone(),
            });
        }

        let sub_groups = group.group_names().map_err(|e| {
            format!("Failed to read group names in '{}': {e}", group.name())
        })?;

        for sub_name in sub_groups {
            if group.name() == "/" && sub_name == "_meta" {
                continue;
            }
            let sub = group.group(&sub_name).map_err(|e| {
                format!("Failed to open sub-group '{sub_name}': {e}")
            })?;
            self.traverse_group_recursive(&sub, discovered)?;
        }

        Ok(())
    }

    /// Reads raw `f64` data and shape from a dataset at `path`.
    ///
    /// # Errors
    ///
    /// Returns an error string when `path` is not an `f64` dataset.
    pub fn read_dataset_f64(&self, path: &str) -> DatasetRead<f64> {
        self.read_dataset_raw::<f64>(path)
    }

    /// Reads raw typed data and shape from a dataset at `path`.
    ///
    /// # Errors
    ///
    /// Returns an error string when the element type does not match `T`.
    pub fn read_dataset_raw<T: H5Type + ReadNumeric>(
        &self,
        path: &str,
    ) -> DatasetRead<T> {
        let ds = self
            .file
            .dataset(path)
            .map_err(|e| format!("Failed to open dataset '{path}': {e}"))?;
        let shape = ds.shape();
        let data = ds.read_raw::<T>().map_err(|e| {
            format!("Failed to read raw data from '{path}': {e}")
        })?;
        Ok((shape, data))
    }

    /// Writes gated signals at `/` and the numeric payload under `/_meta`.
    ///
    /// # Errors
    ///
    /// Returns an error string when a gated path is missing or I/O fails.
    pub fn write_variant_file(
        path: &Path,
        payload: &serde_json::Value,
        gated_paths: &[&str],
    ) -> Result<(), String> {
        if gated_paths.is_empty() {
            return Err("gated path set is empty".to_string());
        }
        let gated = project_json_paths(payload, gated_paths)?;
        let container = Self::create(path)?;
        container.write_numeric_json_group("/", &gated)?;
        if let Some(numeric) = retain_numeric_json(payload) {
            container.write_numeric_json_group("/_meta", &numeric)?;
        }
        container.close()
    }
}

fn check_freshness(
    path: &Path,
    freshness: Option<&FreshnessPolicy>,
    errors: &mut Vec<String>,
) {
    let Some(policy) = freshness else {
        return;
    };
    let Some(floor) = policy.written_after else {
        return;
    };
    match fs::metadata(path).and_then(|m| m.modified()) {
        Ok(modified) if modified >= floor => {}
        Ok(_) => errors.push(format!(
            "Container {} predates this run and is stale",
            path.display()
        )),
        Err(e) => errors.push(format!(
            "Cannot determine modification time of {}: {e}",
            path.display()
        )),
    }
}

fn check_required_tolerance_attrs(
    path: &str,
    names: &[String],
) -> Result<(), String> {
    if !has_attr(names, "measure") {
        return Err(format!(
            "dataset '{path}' missing required attribute 'measure'"
        ));
    }
    if !has_attr(names, "bound") && !has_attr(names, "interval") {
        return Err(format!(
            "dataset '{path}' missing required attribute 'bound' or 'interval'"
        ));
    }
    Ok(())
}

/// Discovers `results/<name>.<variant>.h5` under `example_dir`.
#[must_use]
pub fn discover_suite_h5_files(
    example_dir: &Path,
    name: &str,
) -> Vec<SuiteH5File> {
    let results_dir = example_dir.join("results");
    let mut files = Vec::new();
    let Ok(entries) = fs::read_dir(results_dir) else {
        return files;
    };
    let prefix = format!("{name}.");
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file()
            || path.extension().and_then(|e| e.to_str()) != Some("h5")
        {
            continue;
        }
        let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        if let Some(variant) = stem.strip_prefix(&prefix)
            && !variant.is_empty()
            && !variant.contains('.')
        {
            files.push(SuiteH5File {
                variant: variant.to_string(),
                path,
            });
        }
    }
    files.sort_by(|a, b| a.variant.cmp(&b.variant));
    files
}

fn flatten_json_matrix(
    prefix: &str,
    arr: &[serde_json::Value],
    cols: usize,
) -> JsonMatrix {
    let mut data = Vec::new();
    for (i, row) in arr.iter().enumerate() {
        let r = row.as_array().ok_or_else(|| {
            format!(
                "non-numeric JSON value at '{prefix}[{i}]' cannot be written as an HDF5 dataset"
            )
        })?;
        if r.len() != cols {
            return Err(format!(
                "jagged array at '{prefix}': row 0 has {cols} columns, row {i} has {}",
                r.len()
            ));
        }
        for (j, item) in r.iter().enumerate() {
            data.push(json_leaf_f64(item, &format!("{prefix}[{i}][{j}]"))?);
        }
    }
    Ok(data)
}

fn has_attr(names: &[String], name: &str) -> bool {
    names.iter().any(|n| n == name)
}

/// Inserts `leaf` at slash-separated `path` inside a JSON object map.
fn insert_json_path(
    root: &mut JsonObject,
    path: &str,
    leaf: serde_json::Value,
) -> Result<(), String> {
    let mut parts = path.split('/').filter(|s| !s.is_empty()).peekable();
    if parts.peek().is_none() {
        return Err("empty JSON path".to_string());
    }
    let mut current = root;
    while let Some(seg) = parts.next() {
        if parts.peek().is_none() {
            current.insert(seg.to_string(), leaf);
            return Ok(());
        }
        let entry = current
            .entry(seg.to_string())
            .or_insert_with(|| serde_json::Value::Object(JsonObject::new()));
        let serde_json::Value::Object(map) = entry else {
            return Err(format!(
                "path '{path}' collides with a non-object at '{seg}'"
            ));
        };
        current = map;
    }
    Err(format!("failed to insert JSON path '{path}'"))
}

fn json_leaf_f64(value: &serde_json::Value, path: &str) -> Result<f64, String> {
    value.as_f64().ok_or_else(|| {
        format!(
            "non-numeric JSON value at '{path}' cannot be written as an HDF5 dataset"
        )
    })
}

/// Builds a nested JSON object containing only `paths` from `value`.
///
/// # Errors
///
/// Returns an error string when a requested path is missing.
pub fn project_json_paths(
    value: &serde_json::Value,
    paths: &[&str],
) -> Result<serde_json::Value, String> {
    let mut root = JsonObject::new();
    for path in paths {
        let mut cursor = value;
        for seg in path.split('/').filter(|s| !s.is_empty()) {
            cursor = cursor.get(seg).ok_or_else(|| {
                format!("gated path '{path}' is missing from the payload")
            })?;
        }
        insert_json_path(&mut root, path, cursor.clone())?;
    }
    Ok(serde_json::Value::Object(root))
}

fn read_bound_value(
    ds: &H5Dataset,
    path: &str,
    names: &[String],
) -> Result<f64, String> {
    if !has_attr(names, "bound") {
        return Ok(0.0);
    }
    ds.attr("bound")
        .map_err(|e| format!("Failed to open bound on '{path}': {e}"))?
        .read_numeric::<f64>()
        .map_err(|e| format!("Failed to read bound on '{path}': {e}"))
}

fn read_independent_flag(
    ds: &H5Dataset,
    path: &str,
    names: &[String],
) -> Result<bool, String> {
    if !has_attr(names, "independent") {
        return Ok(true);
    }
    Ok(ds
        .attr("independent")
        .map_err(|e| format!("Failed to open independent on '{path}': {e}"))?
        .read_numeric::<u8>()
        .map_err(|e| format!("Failed to read independent on '{path}': {e}"))?
        != 0)
}

fn read_interval_value(
    ds: &H5Dataset,
    path: &str,
    names: &[String],
) -> IntervalRead {
    if !has_attr(names, "interval") {
        return Ok(None);
    }
    let vals = ds
        .attr("interval")
        .map_err(|e| format!("Failed to open interval on '{path}': {e}"))?
        .read_numeric_as::<f64>()
        .map_err(|e| format!("Failed to read interval on '{path}': {e}"))?;
    let lo = vals.first().copied();
    let hi = vals.get(1).copied();
    match (lo, hi) {
        (Some(a), Some(b)) => Ok(Some([a, b])),
        _ => Err(format!(
            "dataset '{path}' interval attribute must have 2 elements"
        )),
    }
}

fn read_measure(ds: &H5Dataset, path: &str) -> Result<String, String> {
    ds.attr("measure")
        .map_err(|e| format!("Failed to open measure on '{path}': {e}"))?
        .read_string()
        .map_err(|e| format!("Failed to read measure on '{path}': {e}"))
}

fn read_peer_bounds(
    ds: &H5Dataset,
    path: &str,
    names: &[String],
) -> PeerBoundsRead {
    let mut peer_bounds = Vec::new();
    for name in names {
        if let Some(peer) = name.strip_prefix("bound.")
            && !peer.is_empty()
        {
            let v = ds
                .attr(name)
                .map_err(|e| format!("Failed to open {name} on '{path}': {e}"))?
                .read_numeric::<f64>()
                .map_err(|e| {
                    format!("Failed to read {name} on '{path}': {e}")
                })?;
            peer_bounds.push((peer.to_string(), v));
        }
    }
    Ok(peer_bounds)
}

/// Drops non-numeric JSON leaves so a mixed payload can be written.
#[must_use]
pub fn retain_numeric_json(
    value: &serde_json::Value,
) -> Option<serde_json::Value> {
    match value {
        serde_json::Value::Number(n) => {
            Some(serde_json::Value::Number(n.clone()))
        }
        serde_json::Value::Array(arr) => {
            if arr.is_empty() {
                return None;
            }
            let mut out = Vec::new();
            for item in arr {
                match retain_numeric_json(item) {
                    Some(v) if v.is_number() || v.is_array() => out.push(v),
                    _ => return None,
                }
            }
            Some(serde_json::Value::Array(out))
        }
        serde_json::Value::Object(map) => {
            let mut out = JsonObject::new();
            for (k, v) in map {
                if let Some(kept) = retain_numeric_json(v) {
                    out.insert(k.clone(), kept);
                }
            }
            if out.is_empty() {
                None
            } else {
                Some(serde_json::Value::Object(out))
            }
        }
        _ => None,
    }
}

/// Validates globbed suite files against the true oracle.
#[must_use]
pub fn validate_suite_files(
    example_dir: &Path,
    suite_name: &str,
    oracle: &str,
    freshness: Option<&FreshnessPolicy>,
) -> ContainerValidationReport {
    let mut errors = Vec::new();
    let files = discover_suite_h5_files(example_dir, suite_name);
    for f in &files {
        check_freshness(&f.path, freshness, &mut errors);
    }

    let sources: Vec<String> =
        files.iter().map(|f| f.variant.clone()).collect();
    let Some(oracle_file) = files.iter().find(|f| f.variant == oracle) else {
        errors.push(format!(
            "true-oracle file results/{suite_name}.{oracle}.h5 is missing"
        ));
        return ContainerValidationReport {
            file_path: example_dir.join("results").display().to_string(),
            subject: suite_name.to_string(),
            sources,
            comparisons: Vec::new(),
            is_valid: false,
            errors,
        };
    };
    if files.len() < 2 {
        errors.push(format!(
            "suite '{suite_name}' produced fewer than two results/{suite_name}.*.h5 files"
        ));
    }

    let peers: Vec<PeerPath> = files
        .iter()
        .filter(|f| f.variant != oracle)
        .map(|f| (f.variant.clone(), f.path.clone()))
        .collect();

    let (comparisons, compare_res) =
        compare_h5_files(&oracle_file.path, oracle, &peers);
    if let Err(cmp_errors) = compare_res {
        errors.extend(cmp_errors);
    }

    ContainerValidationReport {
        file_path: oracle_file.path.display().to_string(),
        subject: suite_name.to_string(),
        sources,
        comparisons,
        is_valid: errors.is_empty(),
        errors,
    }
}

fn write_f64_attr(
    ds: &H5Dataset,
    path: &str,
    name: &str,
    value: f64,
) -> Result<(), String> {
    ds.new_attr::<f64>()
        .shape(())
        .create(name)
        .map_err(|e| format!("Failed to create {name} attr on '{path}': {e}"))?
        .write_numeric(&value)
        .map_err(|e| format!("Failed to write {name} attr on '{path}': {e}"))
}

fn write_independent_attr(
    ds: &H5Dataset,
    path: &str,
    independent: bool,
) -> Result<(), String> {
    ds.new_attr::<u8>()
        .shape(())
        .create("independent")
        .map_err(|e| {
            format!("Failed to create independent attr on '{path}': {e}")
        })?
        .write_numeric(&u8::from(independent))
        .map_err(|e| {
            format!("Failed to write independent attr on '{path}': {e}")
        })
}

fn write_interval_attr(
    ds: &H5Dataset,
    path: &str,
    interval: Option<Interval>,
) -> Result<(), String> {
    let Some([lo, hi]) = interval else {
        return Ok(());
    };
    ds.new_attr::<f64>()
        .shape([2])
        .create("interval")
        .map_err(|e| {
            format!("Failed to create interval attr on '{path}': {e}")
        })?
        .write_array(&[lo, hi])
        .map_err(|e| format!("Failed to write interval attr on '{path}': {e}"))
}

fn write_json_array(
    container: &H5Container,
    prefix: &str,
    arr: &[serde_json::Value],
) -> Result<(), String> {
    if arr.is_empty() {
        return Err(format!(
            "empty array at '{prefix}' cannot be written as an HDF5 dataset"
        ));
    }
    if arr.first().is_some_and(serde_json::Value::is_array) {
        let rows = arr.len();
        let cols = arr
            .first()
            .and_then(serde_json::Value::as_array)
            .map(std::vec::Vec::len)
            .ok_or_else(|| format!("jagged array at '{prefix}'"))?;
        let data = flatten_json_matrix(prefix, arr, cols)?;
        container.write_dataset_2d(prefix, [rows, cols], &data)
    } else {
        let mut data = Vec::new();
        for (i, item) in arr.iter().enumerate() {
            data.push(json_leaf_f64(item, &format!("{prefix}[{i}]"))?);
        }
        container.write_dataset_1d(prefix, &data)
    }
}

fn write_measure_attr(
    ds: &H5Dataset,
    path: &str,
    measure: &str,
) -> Result<(), String> {
    ds.new_attr::<VarLenUnicode>()
        .shape(())
        .create("measure")
        .map_err(|e| format!("Failed to create measure attr on '{path}': {e}"))?
        .write_string(measure)
        .map_err(|e| format!("Failed to write measure attr on '{path}': {e}"))
}

fn write_peer_bound_attrs(
    ds: &H5Dataset,
    path: &str,
    peer_bounds: &[PeerBound],
) -> Result<(), String> {
    for (peer, bound) in peer_bounds {
        let attr_name = format!("bound.{peer}");
        write_f64_attr(ds, path, &attr_name, *bound)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, SystemTime};

    fn spec(measure: &str, bound: f64) -> DatasetTolerance {
        DatasetTolerance {
            measure: measure.to_string(),
            bound,
            interval: None,
            independent: true,
            peer_bounds: Vec::new(),
        }
    }

    #[test]
    /// # Verification
    /// Trace: oracle-harness#FR-4
    /// Method: Requirements-based test
    fn test_h5_container_creation_and_traversal() {
        let temp_dir =
            std::env::temp_dir().join("control_rs_ci_test_h5_traversal");
        let _ = std::fs::remove_dir_all(&temp_dir);
        let h5_path = temp_dir.join("suite.rust.h5");

        let container = H5Container::create(&h5_path).expect("Create failed");
        container
            .write_dataset_1d("/transient/v_out", &[1.0, 2.0, 3.0])
            .expect("write 1d");
        container
            .write_dataset_1d("/_meta/plot_only", &[9.0])
            .expect("write meta");
        container.close().expect("Close failed");

        let read_container = H5Container::open(&h5_path).expect("Open failed");
        let discovered =
            read_container.traverse_datasets().expect("Traverse failed");
        assert_eq!(discovered.len(), 1);
        assert_eq!(discovered.first().unwrap().signal, "transient/v_out");
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    /// # Verification
    /// Trace: oracle-harness#FR-7
    /// Method: Requirements-based test
    fn test_interval_measure_requires_interval_attr() {
        let temp_dir =
            std::env::temp_dir().join("control_rs_ci_test_h5_interval_attr");
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        let path = temp_dir.join("suite.scipy.h5");
        let container = H5Container::create(&path).unwrap();
        container.write_dataset_1d("/step", &[0.5]).unwrap();
        container
            .write_dataset_tolerance("/step", &spec("interval", 1.0))
            .unwrap();
        container.close().unwrap();
        let read = H5Container::open(&path).unwrap();
        let err = read.read_dataset_tolerance("/step").unwrap_err();
        assert!(
            err.contains("interval"),
            "expected interval-required error, got {err}"
        );
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    /// # Verification
    /// Trace: oracle-harness#FR-6
    /// Method: Requirements-based test
    fn test_suite_glob_ignores_other_files() {
        let temp_dir =
            std::env::temp_dir().join("control_rs_ci_test_suite_glob");
        let _ = std::fs::remove_dir_all(&temp_dir);
        let results = temp_dir.join("results");
        std::fs::create_dir_all(&results).unwrap();
        let rust = results.join("matrix.rust.h5");
        let scipy = results.join("matrix.scipy.h5");
        let other = results.join("polynomial.rust.h5");
        let ignored = results.join("other.h5");
        for p in [&rust, &scipy, &other, &ignored] {
            let c = H5Container::create(p).unwrap();
            c.write_dataset_1d("/step", &[1.0]).unwrap();
            if p == &scipy {
                c.write_dataset_tolerance("/step", &spec("abs", 1e-5))
                    .unwrap();
            }
            c.close().unwrap();
        }

        let found = discover_suite_h5_files(&temp_dir, "matrix");
        let variants: Vec<_> =
            found.iter().map(|f| f.variant.as_str()).collect();
        assert_eq!(variants, vec!["rust", "scipy"]);
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_rust_as_oracle() {
        let temp_dir =
            std::env::temp_dir().join("control_rs_ci_test_rust_oracle");
        let _ = fs::remove_dir_all(&temp_dir);
        let results = temp_dir.join("results");
        fs::create_dir_all(&results).unwrap();
        let oracle = results.join("suite.rust.h5");
        let peer = results.join("suite.scipy.h5");
        {
            let c = H5Container::create(&oracle).unwrap();
            c.write_dataset_1d("/step", &[1.0]).unwrap();
            c.write_dataset_tolerance("/step", &spec("abs", 1e-5))
                .unwrap();
            c.close().unwrap();
        }
        {
            let c = H5Container::create(&peer).unwrap();
            c.write_dataset_1d("/step", &[1.0]).unwrap();
            c.close().unwrap();
        }
        let report = validate_suite_files(&temp_dir, "suite", "rust", None);
        assert!(report.is_valid, "{:?}", report.errors);
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_gate_side_verdict_re_derivation() {
        let temp_dir =
            std::env::temp_dir().join("control_rs_ci_test_gate_side");
        let _ = fs::remove_dir_all(&temp_dir);
        let results = temp_dir.join("results");
        fs::create_dir_all(&results).unwrap();

        let oracle = results.join("test.scipy.h5");
        let peer = results.join("test.rust.h5");
        {
            let c = H5Container::create(&oracle).unwrap();
            c.write_dataset_1d("/step", &[1.0, 2.0, 3.0]).unwrap();
            c.write_dataset_tolerance("/step", &spec("abs", 1e-5))
                .unwrap();
            c.close().unwrap();
        }
        {
            let c = H5Container::create(&peer).unwrap();
            c.write_dataset_1d("/step", &[1.0, 2.0, 3.000_000_1])
                .unwrap();
            c.close().unwrap();
        }

        let report = validate_suite_files(&temp_dir, "test", "scipy", None);
        assert!(
            report.is_valid,
            "Expected clean report: {:?}",
            report.errors
        );
        assert_eq!(report.comparisons.len(), 1);
        assert_eq!(report.comparisons.first().unwrap().verdict, "pass");
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_container_freshness() {
        let temp_dir =
            std::env::temp_dir().join("control_rs_ci_test_freshness");
        let _ = fs::remove_dir_all(&temp_dir);
        let results = temp_dir.join("results");
        fs::create_dir_all(&results).unwrap();
        let oracle = results.join("fresh.scipy.h5");
        let peer = results.join("fresh.rust.h5");
        for p in [&oracle, &peer] {
            let c = H5Container::create(p).unwrap();
            c.write_dataset_1d("/val", &[1.0]).unwrap();
            if p == &oracle {
                c.write_dataset_tolerance("/val", &spec("abs", 1.0))
                    .unwrap();
            }
            c.close().unwrap();
        }

        let future_time = SystemTime::now() + Duration::from_secs(3600);
        let freshness = FreshnessPolicy {
            head_commit: None,
            written_after: Some(future_time),
        };
        let report =
            validate_suite_files(&temp_dir, "fresh", "scipy", Some(&freshness));
        assert!(!report.is_valid);
        assert!(
            report
                .errors
                .iter()
                .any(|e| e.contains("predates this run and is stale"))
        );
        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_missing_true_oracle_file_fails() {
        let temp_dir =
            std::env::temp_dir().join("control_rs_ci_test_missing_oracle");
        let _ = fs::remove_dir_all(&temp_dir);
        let results = temp_dir.join("results");
        fs::create_dir_all(&results).unwrap();
        let peer = results.join("miss.rust.h5");
        let c = H5Container::create(&peer).unwrap();
        c.write_dataset_1d("/val", &[1.0]).unwrap();
        c.close().unwrap();

        let report = validate_suite_files(&temp_dir, "miss", "scipy", None);
        assert!(!report.is_valid);
        assert!(report.errors.iter().any(|e| e.contains("true-oracle file")));
        let _ = fs::remove_dir_all(&temp_dir);
    }
}
