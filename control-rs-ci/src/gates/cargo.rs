//! Quality gate wrapping standard Cargo subcommands (fmt, clippy, check, build, test, clean).

use std::process::Command;

use crate::error::GateError;
use crate::quality_gate::{GateContext, GateOutcome, QualityGate, Verdict};

/// Built-in Cargo subcommand quality gate.
#[derive(Debug, Clone)]
pub struct CargoArgvGate {
    name: String,
    description: &'static str,
    args: Vec<String>,
}

impl CargoArgvGate {
    /// Constructs a new `CargoArgvGate` with given gate name, description, and cargo arguments.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: &'static str,
        args: Vec<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description,
            args,
        }
    }

    /// Gate verifying code formatting (`cargo fmt --all -- --check`).
    #[must_use]
    pub fn fmt() -> Self {
        Self::new(
            "fmt",
            "Verifies codebase formatting conformity with rustfmt",
            vec![
                "fmt".to_string(),
                "--all".to_string(),
                "--".to_string(),
                "--check".to_string(),
            ],
        )
    }

    /// Gate executing workspace compiler lints (`cargo clippy --workspace --all-targets`).
    #[must_use]
    pub fn clippy() -> Self {
        Self::new(
            "clippy",
            "Executes Clippy linter across workspace targets with deny warnings",
            vec![
                "clippy".to_string(),
                "--workspace".to_string(),
                "--all-targets".to_string(),
                "--".to_string(),
                "-D".to_string(),
                "warnings".to_string(),
            ],
        )
    }

    /// Gate verifying type checking (`cargo check --workspace --all-targets`).
    #[must_use]
    pub fn check() -> Self {
        Self::new(
            "check",
            "Performs compiler type checking without full codegen",
            vec![
                "check".to_string(),
                "--workspace".to_string(),
                "--all-targets".to_string(),
            ],
        )
    }

    /// Gate verifying workspace build compilation (`cargo build --workspace --all-targets`).
    #[must_use]
    pub fn build() -> Self {
        Self::new(
            "build",
            "Compiles all workspace library, binary, test, and example targets",
            vec![
                "build".to_string(),
                "--workspace".to_string(),
                "--all-targets".to_string(),
            ],
        )
    }

    /// Gate executing workspace automated test suites (`cargo test --workspace`).
    #[must_use]
    pub fn test() -> Self {
        Self::new(
            "test",
            "Executes all host unit and integration test suites",
            vec!["test".to_string(), "--workspace".to_string()],
        )
    }

    /// Gate cleaning build cache (`cargo clean`).
    #[must_use]
    pub fn clean() -> Self {
        Self::new(
            "clean",
            "Cleans workspace target build cache",
            vec!["clean".to_string()],
        )
    }
}

impl QualityGate for CargoArgvGate {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &'static str {
        self.description
    }

    fn command_display(&self) -> String {
        format!("`cargo {}`", self.args.join(" "))
    }

    fn execute(&self, ctx: &GateContext) -> Result<GateOutcome, GateError> {
        let log_path = ctx.log_path(&self.name);
        let mut cmd = Command::new("cargo");
        cmd.current_dir(&ctx.workspace_root);
        cmd.args(&self.args);

        let (status, duration) =
            self.spawn_and_log(&mut cmd, &log_path, ctx.default_timeout)?;

        let exit_code = status.code();
        let verdict = if status.success() {
            Verdict::Pass
        } else {
            Verdict::Fail
        };

        let summary = if status.success() {
            Some(format!("{} succeeded cleanly", self.name))
        } else {
            Some(format!(
                "{} failed with exit code {}",
                self.name,
                exit_code.unwrap_or(-1)
            ))
        };

        let log_file = log_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("gate.log")
            .to_string();

        let outcome = GateOutcome {
            gate: self.name.clone(),
            verdict,
            exit_code,
            duration_secs: duration,
            summary,
            log_file,
            raw_artifact: None,
        };

        let _ = outcome.save_to_dir(&ctx.out_dir)?;
        Ok(outcome)
    }
}
