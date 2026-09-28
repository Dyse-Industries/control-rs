//! Configuration data models for `gate.toml`.

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use serde::de::{MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::error::{GateError, GateResult};

/// Wall-clock bound, in seconds, for a gate without `timeout_secs` when
/// `[runner] timeout_secs` is also absent.
pub const DEFAULT_TIMEOUT_SECS: u64 = 90;

/// Group names the runner reserves for the exclusive stages and for
/// addressing both of them at once.
pub const RESERVED_GROUP_NAMES: [&str; 3] = ["pre", "post", "exclusive"];

/// Execution policy for a quality gate.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum GatePolicy {
    /// Gate failure blocks the entire pipeline.
    #[default]
    Fail,
    /// Gate executes; failure reports a warning but does not block.
    Warn,
    /// Gate is disabled: it never executes, and an explicit selection
    /// records `Verdict::Skipped`.
    Skip,
}

/// Runner-wide execution settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RunnerConfig {
    /// Pipeline or workspace title.
    #[serde(default = "default_title")]
    pub title: String,
    /// Output artifact directory path.
    #[serde(default = "default_out_dir")]
    pub out_dir: PathBuf,
    /// Default execution timeout in seconds.
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
}

/// Gates that run alone, one at a time, on the calling thread.
///
/// `pre` gates run before any group starts; a `pre` gate that fails aborts
/// the run. `post` gates run after every group has joined.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExclusiveConfig {
    /// Gates run before the concurrent groups, in declared order.
    #[serde(default)]
    pub pre: Vec<String>,
    /// Gates run after the concurrent groups, in declared order.
    #[serde(default)]
    pub post: Vec<String>,
}

/// Gate names in declared order.
pub type GateNames = Vec<String>;

/// One declared group: its name and gates.
pub type GroupEntry = (String, GateNames);

/// A borrowed group: its name and gates.
pub type GroupRef<'a> = (&'a str, &'a [String]);

/// Concurrent execution groups in declaration order.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Groups(Vec<GroupEntry>);

/// Execution schedule and group partitioning.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionConfig {
    /// Bounded group concurrency limit. Defaults to the number of declared groups.
    #[serde(default)]
    pub max_jobs: Option<usize>,
    /// Exclusive gates run before (`pre`) and after (`post`) the groups.
    #[serde(default)]
    pub exclusive: ExclusiveConfig,
    /// Concurrent execution groups: group name to its gates, in declaration order.
    #[serde(default)]
    pub groups: Groups,
}

/// Generic declarative gate definition parsed from `gate.toml`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GateDefinition {
    /// Command string to execute (for example, `"cargo fetch"`, `"cargo fmt"`, `"vale"`).
    pub command: String,
    /// Optional additional command arguments.
    #[serde(default)]
    pub args: Vec<String>,
    /// Optional human-readable description for reports.
    #[serde(default)]
    pub description: Option<String>,
    /// Optional environment variable overrides.
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Optional execution mode / policy for this gate (for example, `"fail"`, `"warn"`, `"skip"`).
    #[serde(default)]
    pub mode: Option<GatePolicy>,
    /// Whether an unfiltered run selects this gate. A `false` gate runs only
    /// when named by `--only`, `--group` or `--all`, under its `mode`.
    #[serde(default = "default_true")]
    pub default: bool,
    /// Working directory, relative to the workspace root.
    #[serde(default)]
    pub cwd: Option<PathBuf>,
    /// Optional execution timeout in seconds for this specific gate.
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    /// Exit codes reported as `Verdict::Skipped`.
    #[serde(default)]
    pub skip_exit_codes: Vec<i32>,
}

/// Top-level workspace quality gate configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GateConfig {
    /// General runner configuration.
    #[serde(default)]
    pub runner: RunnerConfig,
    /// Concurrent groups and exclusive stages.
    #[serde(default)]
    pub execution: ExecutionConfig,
    /// Declarative gate definitions (for example, `[fetch]`, `[fmt]`, `[clippy]`).
    #[serde(flatten)]
    pub gate_definitions: HashMap<String, GateDefinition>,
}

/// Where a gate runs within a pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage<'a> {
    /// Exclusive, before the groups.
    Pre,
    /// In the named concurrent group.
    Group(&'a str),
    /// Exclusive, after the groups. Gates listed nowhere also run here.
    Post,
}

impl Default for RunnerConfig {
    fn default() -> Self {
        Self {
            title: default_title(),
            out_dir: default_out_dir(),
            timeout_secs: default_timeout_secs(),
        }
    }
}

impl Groups {
    /// Builds groups from `(name, gates)` pairs, keeping their order.
    #[must_use]
    pub const fn new(groups: Vec<GroupEntry>) -> Self {
        Self(groups)
    }

    /// Gates of the named group, or `None` if it is not declared.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&GateNames> {
        self.0
            .iter()
            .find(|(group, _)| group == name)
            .map(|(_, gates)| gates)
    }

    /// Groups in declaration order.
    pub fn iter(&self) -> impl Iterator<Item = GroupRef<'_>> {
        self.0
            .iter()
            .map(|(name, gates)| (name.as_str(), gates.as_slice()))
    }

    /// Group names in declaration order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.0.iter().map(|(name, _)| name.as_str())
    }

    /// Number of declared groups.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether no group is declared.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl Serialize for Groups {
    fn serialize<S: Serializer>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (name, gates) in &self.0 {
            map.serialize_entry(name, gates)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for Groups {
    fn deserialize<D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Self, D::Error> {
        /// Collects table entries in the order the document lists them.
        struct OrderedGroups;

        impl<'de> Visitor<'de> for OrderedGroups {
            type Value = Groups;

            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("a table of group names to gate lists")
            }

            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<Groups, A::Error> {
                let mut groups = Vec::new();
                while let Some(entry) =
                    map.next_entry::<String, Vec<String>>()?
                {
                    groups.push(entry);
                }
                Ok(Groups(groups))
            }
        }

        deserializer.deserialize_map(OrderedGroups)
    }
}

impl GateDefinition {
    /// Constructs a new `GateDefinition`.
    #[must_use]
    pub fn new(command: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            command: command.into(),
            args,
            description: None,
            env: HashMap::new(),
            mode: None,
            default: true,
            cwd: None,
            timeout_secs: None,
            skip_exit_codes: Vec::new(),
        }
    }

    /// Resolves the execution mode/policy for this gate, defaulting to [`GatePolicy::Fail`].
    #[must_use]
    pub fn mode(&self) -> GatePolicy {
        self.mode.unwrap_or(GatePolicy::Fail)
    }
}

impl GateConfig {
    /// Loads and validates a `GateConfig` from a TOML file.
    ///
    /// # Errors
    /// Returns `GateError::Config` if the file is missing, unreadable,
    /// malformed or fails [`GateConfig::validate`].
    pub fn load_from_path(path: &Path) -> GateResult<Self> {
        let content =
            fs::read_to_string(path).map_err(|e| config_error(path, &e))?;
        Self::parse(&content, path)
    }

    /// Parses and validates `gate.toml` text; `path` names it in errors.
    ///
    /// # Errors
    /// Returns `GateError::Config` if the text is malformed or fails
    /// [`GateConfig::validate`].
    pub fn parse(content: &str, path: &Path) -> GateResult<Self> {
        let config: Self =
            toml::from_str(content).map_err(|e| config_error(path, &e))?;
        config.validate(path)?;
        Ok(config)
    }

    /// Checks the schedule: every scheduled gate is defined, no gate is
    /// scheduled twice, no group uses a reserved name and `max_jobs` is
    /// positive.
    ///
    /// # Errors
    /// Returns `GateError::Config` naming the first violation.
    pub fn validate(&self, path: &Path) -> GateResult<()> {
        let exec = &self.execution;
        if exec.max_jobs == Some(0) {
            return Err(config_error(
                path,
                &"max_jobs must be greater than zero",
            ));
        }
        let mut scheduled = BTreeSet::new();
        let lists =
            std::iter::once(("exclusive.pre", exec.exclusive.pre.as_slice()))
                .chain(exec.groups.iter())
                .chain(std::iter::once((
                    "exclusive.post",
                    exec.exclusive.post.as_slice(),
                )));
        for (list, gates) in lists {
            if RESERVED_GROUP_NAMES.contains(&list) {
                return Err(config_error(
                    path,
                    &format!("group name '{list}' is reserved"),
                ));
            }
            for gate in gates {
                if !self.gate_definitions.contains_key(gate) {
                    return Err(config_error(
                        path,
                        &format!(
                            "'{list}' lists '{gate}', which has no [{gate}] table"
                        ),
                    ));
                }
                if !scheduled.insert(gate.as_str()) {
                    return Err(config_error(
                        path,
                        &format!("gate '{gate}' is scheduled more than once"),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Policy of the named gate; `fail` when undefined or unset.
    #[must_use]
    pub fn policy_for(&self, gate_name: &str) -> GatePolicy {
        self.gate_definitions
            .get(gate_name)
            .map_or(GatePolicy::Fail, GateDefinition::mode)
    }

    /// Resolves the gate definition for a named quality gate from `gate.toml`.
    ///
    /// Returns `None` if the corresponding `[<gate_name>]` table is missing.
    #[must_use]
    pub fn gate_def(&self, gate_name: &str) -> Option<&GateDefinition> {
        self.gate_definitions.get(gate_name)
    }

    /// Stage in which the named gate runs.
    #[must_use]
    pub fn stage_of(&self, gate_name: &str) -> Stage<'_> {
        let exec = &self.execution;
        if exec.exclusive.pre.iter().any(|g| g == gate_name) {
            return Stage::Pre;
        }
        exec.groups
            .iter()
            .find(|(_, gates)| gates.iter().any(|g| g == gate_name))
            .map_or(Stage::Post, |(name, _)| Stage::Group(name))
    }

    /// Every defined gate in pipeline order: `exclusive.pre`, the groups in
    /// declaration order, `exclusive.post`, then unscheduled gates by name.
    #[must_use]
    pub fn pipeline_order(&self) -> Vec<&str> {
        let exec = &self.execution;
        let mut order: Vec<&str> = exec
            .exclusive
            .pre
            .iter()
            .chain(exec.groups.iter().flat_map(|(_, gates)| gates))
            .chain(&exec.exclusive.post)
            .map(String::as_str)
            .collect();
        let mut rest: Vec<&str> = self
            .gate_definitions
            .keys()
            .map(String::as_str)
            .filter(|name| !order.contains(name))
            .collect();
        rest.sort_unstable();
        order.extend(rest);
        order
    }

    /// Gates addressed by a group name: a declared group, `pre`, `post`, or
    /// `exclusive` for both exclusive stages.
    #[must_use]
    pub fn group_members(&self, name: &str) -> Option<GateNames> {
        let exclusive = &self.execution.exclusive;
        match name {
            "pre" => Some(exclusive.pre.clone()),
            "post" => Some(exclusive.post.clone()),
            "exclusive" => Some(
                exclusive
                    .pre
                    .iter()
                    .chain(&exclusive.post)
                    .cloned()
                    .collect(),
            ),
            _ => self.execution.groups.get(name).cloned(),
        }
    }
}

fn config_error(path: &Path, message: &dyn fmt::Display) -> GateError {
    GateError::Config {
        path: path.to_path_buf(),
        message: message.to_string(),
    }
}

const fn default_true() -> bool {
    true
}

fn default_title() -> String {
    "control-rs".to_string()
}

fn default_out_dir() -> PathBuf {
    PathBuf::from("target/ci-artifacts")
}

const fn default_timeout_secs() -> u64 {
    DEFAULT_TIMEOUT_SECS
}

#[cfg(test)]
mod tests {
    use super::*;

    const PATH: &str = "gate.toml";

    fn parse(text: &str) -> GateResult<GateConfig> {
        GateConfig::parse(text, Path::new(PATH))
    }

    #[test]
    fn groups_keep_declaration_order() {
        let config = parse(
            "[execution.groups]\nzeta = [\"z\"]\nalpha = [\"a\"]\nmid = [\"m\"]\n\
             [z]\ncommand = \"true\"\n[a]\ncommand = \"true\"\n[m]\ncommand = \"true\"\n",
        )
        .unwrap();
        let names: Vec<_> = config.execution.groups.names().collect();
        assert_eq!(names, ["zeta", "alpha", "mid"]);
        assert_eq!(config.pipeline_order(), ["z", "a", "m"]);
    }

    #[test]
    fn pipeline_order_is_pre_groups_post_then_rest() {
        let config = parse(
            "[execution.exclusive]\npre = [\"fetch\"]\npost = [\"bench\"]\n\
             [execution.groups]\nlint = [\"fmt\"]\n\
             [fetch]\ncommand = \"true\"\n[fmt]\ncommand = \"true\"\n\
             [bench]\ncommand = \"true\"\n[loose]\ncommand = \"true\"\n",
        )
        .unwrap();
        assert_eq!(config.pipeline_order(), ["fetch", "fmt", "bench", "loose"]);
        assert_eq!(config.stage_of("fetch"), Stage::Pre);
        assert_eq!(config.stage_of("fmt"), Stage::Group("lint"));
        assert_eq!(config.stage_of("bench"), Stage::Post);
        assert_eq!(config.stage_of("loose"), Stage::Post);
        assert_eq!(
            config.group_members("exclusive"),
            Some(vec!["fetch".to_string(), "bench".to_string()])
        );
    }

    #[test]
    fn missing_file_is_an_error() {
        let err =
            GateConfig::load_from_path(Path::new("/nonexistent/gate.toml"));
        assert!(matches!(err, Err(GateError::Config { .. })));
    }

    #[test]
    fn invalid_schedules_are_rejected() {
        let cases = [
            ("[execution.groups]\na = [\"x\"]\n", "has no [x] table"),
            (
                "[execution.groups]\na = [\"x\"]\nb = [\"x\"]\n[x]\ncommand = \"true\"\n",
                "scheduled more than once",
            ),
            (
                "[execution.exclusive]\npost = [\"x\"]\n[execution.groups]\na = [\"x\"]\n[x]\ncommand = \"true\"\n",
                "scheduled more than once",
            ),
            (
                "[execution.groups]\npre = [\"x\"]\n[x]\ncommand = \"true\"\n",
                "reserved",
            ),
            ("[execution]\nmax_jobs = 0\n", "greater than zero"),
        ];
        for (text, expected) in cases {
            let message = parse(text).unwrap_err().to_string();
            assert!(message.contains(expected), "{message}");
        }
    }

    #[test]
    fn unknown_keys_are_rejected() {
        for text in [
            "[x]\ncommand = \"true\"\ntimout_secs = 5\n",
            "[execution]\nparallel = true\n",
            "[runner]\ntimeout = 5\n",
            "[execution.exclusive]\nbefore = []\n",
        ] {
            assert!(parse(text).is_err(), "{text}");
        }
    }
}
