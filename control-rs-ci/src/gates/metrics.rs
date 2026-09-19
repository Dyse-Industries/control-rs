//! Project codebase metrics gate (source lines, comment lines, directory byte footprints).

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::config::MetricsConfig;
use crate::error::GateError;
use crate::quality_gate::{GateContext, GateOutcome, QualityGate, Verdict};

/// Metrics breakdown per file type / category.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LineCounts {
    /// Number of files.
    pub files: usize,
    /// Number of executable/declarative code lines.
    pub code_lines: usize,
    /// Number of doc comments and inline comments.
    pub comment_lines: usize,
    /// Number of blank lines.
    pub blank_lines: usize,
    /// Total lines (code + comment + blank).
    pub total_lines: usize,
}

/// Directory footprint in bytes and file count.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DirectoryFootprint {
    /// Total size in bytes.
    pub bytes: u64,
    /// Total number of files.
    pub files: usize,
}

/// Raw codebase metrics data dumped to `metrics-raw.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetricsReport {
    /// Aggregate line metrics across all scanned files.
    pub total_lines: LineCounts,
    /// Detailed line breakdown by file extension (for example, "rs", "md", "toml").
    pub by_extension: BTreeMap<String, LineCounts>,
    /// Directory size footprints by relative directory path.
    pub directories: BTreeMap<String, DirectoryFootprint>,
}

/// Built-in codebase metrics quality gate.
#[derive(Debug, Clone)]
pub struct MetricsGate {
    config: MetricsConfig,
}

impl MetricsGate {
    /// Constructs a new `MetricsGate` with the given configuration.
    #[must_use]
    pub const fn new(config: MetricsConfig) -> Self {
        Self { config }
    }

    fn scan_directory(
        &self,
        dir_path: &Path,
        rel_prefix: &str,
        report: &mut MetricsReport,
    ) -> Result<DirectoryFootprint, GateError> {
        let mut dir_footprint = DirectoryFootprint::default();

        if !dir_path.exists() {
            return Ok(dir_footprint);
        }

        let entries = fs::read_dir(dir_path)?;
        for entry in entries {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;

            if file_type.is_dir() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with('.')
                    || name == "target"
                    || name == "node_modules"
                {
                    continue;
                }
                let sub_prefix = if rel_prefix.is_empty() {
                    name
                } else {
                    format!("{rel_prefix}/{name}")
                };
                let sub_footprint =
                    self.scan_directory(&path, &sub_prefix, report)?;
                dir_footprint.bytes =
                    dir_footprint.bytes.saturating_add(sub_footprint.bytes);
                dir_footprint.files =
                    dir_footprint.files.saturating_add(sub_footprint.files);
            } else if file_type.is_file() {
                let metadata = entry.metadata()?;
                let bytes = metadata.len();
                dir_footprint.bytes = dir_footprint.bytes.saturating_add(bytes);
                dir_footprint.files = dir_footprint.files.saturating_add(1);

                self.count_file_lines(&path, report)?;
            }
        }

        if !rel_prefix.is_empty() {
            report
                .directories
                .insert(rel_prefix.to_string(), dir_footprint.clone());
        }

        Ok(dir_footprint)
    }

    fn count_file_lines(
        &self,
        path: &Path,
        report: &mut MetricsReport,
    ) -> Result<(), GateError> {
        let ext = path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("other")
            .to_string();

        let Ok(file) = File::open(path) else {
            return Ok(());
        };
        let reader = BufReader::new(file);

        let mut code: usize = 0;
        let mut comment: usize = 0;
        let mut blank: usize = 0;

        for line_res in reader.lines() {
            let Ok(line) = line_res else {
                continue;
            };
            let trimmed = line.trim();
            if trimmed.is_empty() {
                blank = blank.saturating_add(1);
            } else if trimmed.starts_with("//")
                || trimmed.starts_with("/*")
                || trimmed.starts_with('*')
                || trimmed.starts_with('#')
            {
                comment = comment.saturating_add(1);
            } else {
                code = code.saturating_add(1);
            }
        }

        let total = code.saturating_add(comment).saturating_add(blank);

        // Update overall total
        report.total_lines.files = report.total_lines.files.saturating_add(1);
        report.total_lines.code_lines =
            report.total_lines.code_lines.saturating_add(code);
        report.total_lines.comment_lines =
            report.total_lines.comment_lines.saturating_add(comment);
        report.total_lines.blank_lines =
            report.total_lines.blank_lines.saturating_add(blank);
        report.total_lines.total_lines =
            report.total_lines.total_lines.saturating_add(total);

        // Update extension bucket
        let ext_entry = report.by_extension.entry(ext).or_default();
        ext_entry.files = ext_entry.files.saturating_add(1);
        ext_entry.code_lines = ext_entry.code_lines.saturating_add(code);
        ext_entry.comment_lines =
            ext_entry.comment_lines.saturating_add(comment);
        ext_entry.blank_lines = ext_entry.blank_lines.saturating_add(blank);
        ext_entry.total_lines = ext_entry.total_lines.saturating_add(total);

        Ok(())
    }
}

impl QualityGate for MetricsGate {
    fn name(&self) -> &str {
        "metrics"
    }

    fn description(&self) -> &'static str {
        "Measures workspace codebase line metrics and directory byte footprints"
    }

    fn command_display(&self) -> String {
        "`control-rs-ci metrics`".to_string()
    }

    fn execute(&self, ctx: &GateContext) -> Result<GateOutcome, GateError> {
        let start = Instant::now();
        let log_path = ctx.log_path(self.name());
        let raw_artifact_path = ctx.raw_artifact_path(self.name());

        if let Some(parent) = log_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let mut report = MetricsReport::default();

        let tracked = if self.config.tracked_dirs.is_empty() {
            vec!["src".to_string()]
        } else {
            self.config.tracked_dirs.clone()
        };

        for dir_name in &tracked {
            let dir_path = ctx.workspace_root.join(dir_name);
            let _ = self.scan_directory(&dir_path, dir_name, &mut report)?;
        }

        // Write raw JSON artifact
        let raw_file = File::create(&raw_artifact_path)?;
        serde_json::to_writer_pretty(raw_file, &report)?;

        // Write human-readable log
        let mut log_file = File::create(&log_path)?;
        writeln!(
            log_file,
            "=== Control-RS Codebase Metrics ===\n\
             Scanned Directories: {}\n\
             Total Files:         {}\n\
             Source Code Lines:   {}\n\
             Comment Lines:       {}\n\
             Blank Lines:         {}\n\
             Total Lines:         {}\n",
            tracked.join(", "),
            report.total_lines.files,
            report.total_lines.code_lines,
            report.total_lines.comment_lines,
            report.total_lines.blank_lines,
            report.total_lines.total_lines
        )?;

        writeln!(log_file, "--- Lines By Extension ---")?;
        for (ext, counts) in &report.by_extension {
            writeln!(
                log_file,
                ".{:<6} | {:>4} files | {:>6} code | {:>6} comment | {:>6} blank | {:>6} total",
                ext,
                counts.files,
                counts.code_lines,
                counts.comment_lines,
                counts.blank_lines,
                counts.total_lines
            )?;
        }

        writeln!(log_file, "\n--- Directory Byte Footprints ---")?;
        for (dir, foot) in &report.directories {
            writeln!(
                log_file,
                "{:<30} | {:>4} files | {:>10} bytes ({:.2} KiB)",
                dir,
                foot.files,
                foot.bytes,
                foot.bytes as f64 / 1024.0
            )?;
        }

        let duration = start.elapsed().as_secs_f64();

        // Check bounds
        let (verdict, summary) = if let Some(max) = self.config.max_source_lines
        {
            if report.total_lines.code_lines > max {
                (
                    Verdict::Fail,
                    Some(format!(
                        "Source lines ({}) exceed max bound ({})",
                        report.total_lines.code_lines, max
                    )),
                )
            } else {
                (
                    Verdict::Pass,
                    Some(format!(
                        "{} files, {} code lines ({:.1} KiB scanned)",
                        report.total_lines.files,
                        report.total_lines.code_lines,
                        report
                            .directories
                            .values()
                            .map(|d| d.bytes)
                            .sum::<u64>() as f64
                            / 1024.0
                    )),
                )
            }
        } else {
            (
                Verdict::Pass,
                Some(format!(
                    "{} files, {} code lines",
                    report.total_lines.files, report.total_lines.code_lines
                )),
            )
        };

        let outcome = GateOutcome {
            gate: self.name().to_string(),
            verdict,
            exit_code: Some(0),
            duration_secs: duration,
            summary,
            log_file: "metrics.log".to_string(),
            raw_artifact: Some("metrics-raw.json".to_string()),
        };

        let _ = outcome.save_to_dir(&ctx.out_dir)?;
        Ok(outcome)
    }
}
