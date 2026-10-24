//! Path-gated skip: the check that runs before each gate (FR-22 to FR-24).

use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Environment variable that forces every gate to run or skip.
pub const ALL_GATES_VAR: &str = "CONTROL_RS_CI_GATES";

/// Base ref of the comparison when neither `--base` nor `[runner] base` is set.
pub const DEFAULT_BASE: &str = "origin/main";

/// Head ref of the comparison when neither `--head` nor `[runner] head` is set.
pub const DEFAULT_HEAD: &str = "HEAD";

/// Prefix of the per-gate variable; the gate name follows in upper case with
/// `-` written as `_`.
pub const GATE_VAR_PREFIX: &str = "CONTROL_RS_CI_GATE_";

/// Files changed between two refs, or `None` when the diff cannot be computed.
pub type ChangedFiles = Option<Vec<String>>;

/// A diff to match against, when one could be computed.
pub type ChangedRef<'a> = Option<&'a [String]>;

/// Looks up an environment variable by name.
pub type Lookup<'a> = &'a dyn Fn(&str) -> Option<String>;

/// What one override variable holds.
pub type Read = Result<Option<Override>, String>;

/// An override and the variable it came from.
pub type Forced = Option<(Override, String)>;

/// The `diff-trigger` entries that apply to one gate.
#[derive(Debug, Clone, Copy)]
pub struct Triggers<'a> {
    /// The gate's own entries. A gate with none always runs.
    pub own: &'a [String],
    /// Whether a changed file matches the `[runner]` list. Computed once per
    /// invocation, not per gate.
    pub global_hit: bool,
}

/// Summary of audited input states for a gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditedInputs {
    /// Git commit SHA at HEAD, if in a git repository.
    pub head: Option<String>,
    /// Git tree hash of HEAD, if in a git repository.
    pub tree_hash: Option<String>,
    /// Deterministic content hash across triggered input paths.
    pub content_hash: String,
}

/// A caller's forced decision for a gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Override {
    /// Run the gate whatever changed.
    Run,
    /// Skip the gate whatever changed.
    Skip,
}

/// Outcome of the check for one gate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// Execute the gate.
    Run,
    /// Record the gate as skipped, for the stated reason.
    Skip(String),
}

/// Name of the per-gate variable for `gate`.
#[must_use]
pub fn gate_var(gate: &str) -> String {
    let name = gate.to_ascii_uppercase().replace('-', "_");
    format!("{GATE_VAR_PREFIX}{name}")
}

/// Reads one override variable. An unset or empty variable defers.
///
/// # Errors
/// Returns a message naming the variable when its value is neither `run` nor
/// `skip`.
fn read_override(var: &str, lookup: Lookup<'_>) -> Read {
    match lookup(var).as_deref().map(str::trim) {
        None | Some("") => Ok(None),
        Some("run") => Ok(Some(Override::Run)),
        Some("skip") => Ok(Some(Override::Skip)),
        Some(other) => Err(format!("{var}={other}: expected `run` or `skip`")),
    }
}

/// The override for `gate`: its own variable, else the global one (FR-24).
///
/// # Errors
/// Returns a message when a consulted variable holds another value.
pub fn override_for(gate: &str, lookup: Lookup<'_>) -> Result<Forced, String> {
    for var in [gate_var(gate), ALL_GATES_VAR.to_string()] {
        if let Some(found) = read_override(&var, lookup)? {
            return Ok(Some((found, var)));
        }
    }
    Ok(None)
}

/// Whether `file` equals `entry` or lies below it. Entries are not globs.
#[must_use]
pub fn matches(entry: &str, file: &str) -> bool {
    let entry = entry.trim_end_matches('/');
    !entry.is_empty()
        && file
            .strip_prefix(entry)
            .is_some_and(|rest| rest.is_empty() || rest.starts_with('/'))
}

/// Paths in `git diff -z --name-status` output; a rename or copy lists both
/// (FR-22). `-z` leaves every path unquoted, whatever characters it holds.
#[must_use]
pub fn parse_name_status(output: &str) -> Vec<String> {
    let mut fields = output.split('\0').filter(|f| !f.is_empty());
    let mut paths = Vec::new();
    while let Some(status) = fields.next() {
        let count = if status.starts_with(['R', 'C']) { 2 } else { 1 };
        paths.extend(fields.by_ref().take(count).map(str::to_string));
    }
    paths
}

/// Files changed between the merge-base of `base` and `head`, and `head`.
///
/// Returns `None` when a ref or the merge-base cannot be resolved (FR-23).
#[must_use]
pub fn changed_files(root: &Path, base: &str, head: &str) -> ChangedFiles {
    let git = |args: &[&str]| {
        let out = Command::new("git")
            .current_dir(root)
            .args(args)
            .output()
            .ok()?;
        out.status
            .success()
            .then(|| String::from_utf8_lossy(&out.stdout).into_owned())
    };
    let merge_base = git(&["merge-base", base, head])?;
    let diff =
        git(&["diff", "-z", "--name-status", "-M", merge_base.trim(), head])?;
    Some(parse_name_status(&diff))
}

/// Whether any changed file matches an entry of the `[runner]` list.
#[must_use]
pub fn global_hit(global: &[String], changed: &[String]) -> bool {
    global
        .iter()
        .any(|entry| changed.iter().any(|file| matches(entry, file)))
}

/// Decides whether `gate` runs.
///
/// An override wins. Otherwise a gate with a `diff-trigger` is skipped when no entry
/// of `changed` matches one of them; a gate without one, or a `changed`
/// of `None` (the diff could not be computed), runs.
///
/// # Errors
/// Returns a message when an override variable holds an invalid value.
pub fn should_run(
    gate: &str,
    triggers: Triggers<'_>,
    changed: ChangedRef<'_>,
    lookup: Lookup<'_>,
) -> Result<Decision, String> {
    if let Some((forced, var)) = override_for(gate, lookup)? {
        return Ok(match forced {
            Override::Run => Decision::Run,
            Override::Skip => Decision::Skip(format!("{var}=skip")),
        });
    }
    let Some(changed) = changed else {
        return Ok(Decision::Run);
    };
    let hit = triggers.global_hit
        || triggers
            .own
            .iter()
            .any(|entry| changed.iter().any(|file| matches(entry, file)));
    if triggers.own.is_empty() || hit {
        Ok(Decision::Run)
    } else {
        Ok(Decision::Skip(
            "no changed file under its diff-trigger".to_string(),
        ))
    }
}

/// Scans trigger paths to obtain the content hash, commit SHA, and tree hash.
#[must_use]
pub fn audit_inputs(root: &Path, triggers: &[String]) -> AuditedInputs {
    let (head, tree_hash) =
        if root.join(".git").exists() && !triggers.is_empty() {
            let git = |args: &[&str]| {
                let out = Command::new("git")
                    .current_dir(root)
                    .args(args)
                    .output()
                    .ok()?;
                out.status.success().then(|| {
                    String::from_utf8_lossy(&out.stdout).trim().to_string()
                })
            };
            (
                git(&["rev-parse", "HEAD"]),
                git(&["rev-parse", "HEAD^{tree}"]),
            )
        } else {
            (None, None)
        };

    let content_hash = compute_content_hash(root, triggers);
    AuditedInputs {
        head,
        tree_hash,
        content_hash,
    }
}

fn collect_trigger_files(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_file() {
        out.push(path.to_path_buf());
    } else if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let p = entry.path();
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            // Skip hidden files and target directories.
            if name_str.starts_with('.') || name_str == "target" {
                continue;
            }
            collect_trigger_files(&p, out);
        }
    }
}

/// Recursively computes the deterministic content hash for trigger paths.
#[must_use]
pub fn compute_content_hash(root: &Path, triggers: &[String]) -> String {
    let paths_to_check: Vec<&str> = if triggers.is_empty() {
        vec![
            "src",
            "tests",
            "benches",
            "examples",
            "Cargo.toml",
            "Cargo.lock",
        ]
    } else {
        triggers.iter().map(String::as_str).collect()
    };

    let mut files = Vec::new();
    for path_str in paths_to_check {
        let p = root.join(path_str);
        collect_trigger_files(&p, &mut files);
    }

    files.sort();

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for path in files {
        if let Ok(rel) = path.strip_prefix(root) {
            rel.hash(&mut hasher);
        } else {
            path.hash(&mut hasher);
        }
        if let Ok(bytes) = std::fs::read(&path) {
            bytes.hash(&mut hasher);
        }
    }
    format!("{:016x}", hasher.finish())
}

/// Verifies whether `<gate>.log` is fresh with respect to its triggered inputs.
#[must_use]
pub fn is_log_fresh(
    root: &Path,
    triggers: &[String],
    log_path: &Path,
    recorded_fp: &crate::gate::InputFingerprint,
) -> bool {
    if !log_path.is_file() {
        return false;
    }

    // 1. Fast Path: Git commit matches and working tree is clean for triggered paths
    let git_repo = root.join(".git").exists();
    if let Some(recorded_commit) =
        recorded_fp.head_commit.as_ref().filter(|_| git_repo)
    {
        let mut status_args = vec!["status", "--porcelain", "-z", "--"];
        let trigger_refs: Vec<&str> =
            triggers.iter().map(String::as_str).collect();
        status_args.extend(&trigger_refs);
        let status_cmd = Command::new("git")
            .current_dir(root)
            .args(&status_args)
            .output();
        let is_clean =
            status_cmd.is_ok_and(|s| s.status.success() && s.stdout.is_empty());
        if is_clean {
            let rev_cmd = Command::new("git")
                .current_dir(root)
                .args(["rev-parse", "HEAD"])
                .output();
            if let Ok(rev) = rev_cmd {
                let current_head =
                    String::from_utf8_lossy(&rev.stdout).trim().to_string();
                if &current_head == recorded_commit {
                    return true;
                }
            }
        }
    }

    // 2. Deterministic content hash check across triggered inputs
    let current_content_hash = compute_content_hash(root, triggers);
    current_content_hash == recorded_fp.content_hash
}
