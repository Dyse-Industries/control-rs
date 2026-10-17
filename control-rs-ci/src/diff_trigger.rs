//! Path-gated skip: the check that runs before each gate (FR-22 to FR-24).

use std::path::Path;
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
