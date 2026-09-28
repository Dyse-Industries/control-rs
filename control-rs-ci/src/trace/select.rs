//! File selection by roots and suffixes.
//!
//! A root that is a file is selected. A root that is a directory selects
//! every file below it whose name ends in one of the suffixes. Symbolic links
//! are never followed, so a walk cannot loop. Paths are relative to the
//! working directory, with `/` separators.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::error::{GateError, GateResult};

/// Selected files, relative to the working directory.
pub type Selection = Vec<String>;

/// Selects the files under `roots`, relative to `base`, sorted and without
/// repeats.
///
/// # Errors
/// `GateError::Config` naming a root that is missing or is neither a file nor
/// a directory; `GateError::Io` if a directory cannot be read.
pub fn select(
    base: &Path,
    roots: &[String],
    suffixes: &[String],
) -> GateResult<Selection> {
    let mut found = BTreeSet::new();
    for root in roots.iter().map(|r| r.trim_end_matches('/')) {
        let kind = fs::symlink_metadata(base.join(root)).map(|m| m.file_type());
        match kind {
            Ok(kind) if kind.is_file() => drop(found.insert(root.to_owned())),
            Ok(kind) if kind.is_dir() => {
                walk(base, root, suffixes, &mut found)?;
            }
            _ => {
                return Err(GateError::Config {
                    path: PathBuf::from(root),
                    message: "root is not a file or directory".to_owned(),
                });
            }
        }
    }
    Ok(found.into_iter().collect())
}

/// Adds the files below `dir` whose names end in a suffix to `found`.
fn walk(
    base: &Path,
    dir: &str,
    suffixes: &[String],
    found: &mut BTreeSet<String>,
) -> GateResult<()> {
    for entry in fs::read_dir(base.join(dir))? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let path = if dir == "." {
            name
        } else {
            format!("{dir}/{name}")
        };
        let kind = entry.file_type()?;
        if kind.is_dir() {
            walk(base, &path, suffixes, found)?;
        } else if kind.is_file() && suffixes.iter().any(|s| path.ends_with(s)) {
            found.insert(path);
        }
    }
    Ok(())
}
