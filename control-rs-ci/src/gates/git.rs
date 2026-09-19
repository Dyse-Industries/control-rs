//! Git repository hygiene gate (working tree status, commit message validation).

use std::fs::{self, File};
use std::io::Write;
use std::process::Command;
use std::time::Instant;

use crate::config::GitConfig;
use crate::error::GateError;
use crate::quality_gate::{GateContext, GateOutcome, QualityGate, Verdict};

/// Built-in Git repository and commit message hygiene gate.
#[derive(Debug, Clone)]
pub struct GitHygieneGate {
    config: GitConfig,
}

impl GitHygieneGate {
    /// Constructs a new `GitHygieneGate` with the given configuration.
    #[must_use]
    pub const fn new(config: GitConfig) -> Self {
        Self { config }
    }

    fn check_commit_subject(&self, subject: &str) -> Vec<String> {
        let mut issues = Vec::new();
        let trimmed = subject.trim();

        if trimmed.is_empty() {
            issues.push("Empty commit message".to_string());
            return issues;
        }

        let lower = trimmed.to_lowercase();
        let is_merge_or_initial = lower.starts_with("merge ")
            || lower.starts_with("initial ")
            || lower.starts_with("release");

        if is_merge_or_initial {
            return issues;
        }

        // Check header length
        if trimmed.chars().count() > self.config.max_header_length {
            issues.push(format!(
                "Header length ({} chars) exceeds limit of {} chars",
                trimmed.chars().count(),
                self.config.max_header_length
            ));
        }

        // Check disallowed patterns
        for pat in &self.config.disallowed_patterns {
            let pat_lower = pat.to_lowercase();
            if lower == pat_lower
                || lower.starts_with(&format!("{pat_lower}:"))
                || lower.starts_with(&format!("{pat_lower} "))
                || lower.contains(&format!(" {pat_lower} "))
                || lower.ends_with(&format!(" {pat_lower}"))
            {
                issues.push(format!(
                    "Contains disallowed placeholder pattern: '{pat}'"
                ));
            }
        }

        // Check conventional commit pattern if enabled
        if self.config.enforce_conventional_commits {
            let valid_prefixes = [
                "feat", "fix", "docs", "style", "refactor", "perf", "test",
                "build", "ci", "chore", "revert",
            ];
            let is_conventional = valid_prefixes.iter().any(|&prefix| {
                if let Some(rest) = lower.strip_prefix(prefix) {
                    if rest.starts_with(':') {
                        return true;
                    }
                    if rest.starts_with('(') {
                        if let Some(end_idx) = rest.find(')') {
                            if let Some(after) =
                                rest.get(end_idx.saturating_add(1)..)
                            {
                                if after.starts_with(':')
                                    || after.starts_with("!:")
                                {
                                    return true;
                                }
                            }
                        }
                    }
                    if rest.starts_with("!:") {
                        return true;
                    }
                }
                false
            });

            if !is_conventional {
                issues.push(format!(
                    "Does not conform to Conventional Commits (e.g. 'feat:', 'fix(scope):', etc.): '{trimmed}'"
                ));
            }
        }

        issues
    }
}

impl QualityGate for GitHygieneGate {
    fn name(&self) -> &str {
        "git"
    }

    fn description(&self) -> &'static str {
        "Audits repository working tree state and branch commit message hygiene"
    }

    fn command_display(&self) -> String {
        "`git status --porcelain && git log -n 20`".to_string()
    }

    fn execute(&self, ctx: &GateContext) -> Result<GateOutcome, GateError> {
        let start = Instant::now();
        let log_path = ctx.log_path(self.name());

        if let Some(parent) = log_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut log_file = File::create(&log_path)?;
        writeln!(log_file, "=== Control-RS Git Hygiene Audit ===")?;

        let mut violations = Vec::new();

        // 1. Check working tree status
        let status_output = Command::new("git")
            .current_dir(&ctx.workspace_root)
            .args(["status", "--porcelain"])
            .output();

        match status_output {
            Ok(output) => {
                let status_str = String::from_utf8_lossy(&output.stdout);
                let dirty_files: Vec<&str> = status_str
                    .lines()
                    .map(str::trim)
                    .filter(|l| !l.is_empty())
                    .collect();
                if !dirty_files.is_empty() {
                    writeln!(
                        log_file,
                        "\n--- Working Tree Status (Dirty Files: {}) ---",
                        dirty_files.len()
                    )?;
                    for file in &dirty_files {
                        writeln!(log_file, "  {file}")?;
                    }
                    if self.config.require_clean_working_tree {
                        violations.push(format!("Working tree contains {} uncommitted/untracked files", dirty_files.len()));
                    }
                } else {
                    writeln!(log_file, "\nWorking tree is completely clean.")?;
                }
            }
            Err(e) => {
                writeln!(
                    log_file,
                    "Warning: Failed to execute 'git status': {e}"
                )?;
            }
        }

        // 2. Audit recent commits
        let log_output = Command::new("git")
            .current_dir(&ctx.workspace_root)
            .args(["log", "-n", "20", "--pretty=format:%h %s"])
            .output();

        match log_output {
            Ok(output) => {
                let commits_str = String::from_utf8_lossy(&output.stdout);
                let lines: Vec<&str> = commits_str
                    .lines()
                    .map(str::trim)
                    .filter(|l| !l.is_empty())
                    .collect();
                writeln!(
                    log_file,
                    "\n--- Audited Recent Commits ({}) ---",
                    lines.len()
                )?;

                for line in &lines {
                    let mut parts = line.splitn(2, ' ');
                    let hash = parts.next().unwrap_or("???????");
                    let subject = parts.next().unwrap_or("");

                    let issues = self.check_commit_subject(subject);
                    if issues.is_empty() {
                        writeln!(log_file, "  [OK]    {} - {}", hash, subject)?;
                    } else {
                        writeln!(log_file, "  [ISSUE] {} - {}", hash, subject)?;
                        for issue in &issues {
                            writeln!(log_file, "          -> {}", issue)?;
                            violations.push(format!(
                                "Commit {} issue: {}",
                                hash, issue
                            ));
                        }
                    }
                }
            }
            Err(e) => {
                writeln!(
                    log_file,
                    "Warning: Failed to execute 'git log': {e}"
                )?;
            }
        }

        let duration = start.elapsed().as_secs_f64();

        let (verdict, summary) = if violations.is_empty() {
            (
                Verdict::Pass,
                Some("Git working tree and commit messages clean".to_string()),
            )
        } else {
            (
                Verdict::Fail,
                Some(format!(
                    "{} git hygiene violation(s) detected",
                    violations.len()
                )),
            )
        };

        let outcome = GateOutcome {
            gate: self.name().to_string(),
            verdict,
            exit_code: if violations.is_empty() {
                Some(0)
            } else {
                Some(1)
            },
            duration_secs: duration,
            summary,
            log_file: "git.log".to_string(),
            raw_artifact: None,
        };

        let _ = outcome.save_to_dir(&ctx.out_dir)?;
        Ok(outcome)
    }
}
