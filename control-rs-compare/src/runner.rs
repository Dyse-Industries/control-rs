//! Multi-language variant execution runner and process orchestrator.

use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anstyle::AnsiColor;

use crate::config::{MasterPlan, SuiteConfig, VariantConfig};
use crate::error::HarnessError;

/// Cargo status verb style (bright green bold), matching `control-rs-ci`.
const HEADER: anstyle::Style = AnsiColor::BrightGreen.on_default().bold();

/// Cargo `warning:` prefix style (bright yellow bold), matching `control-rs-ci`.
const WARNING: anstyle::Style = AnsiColor::BrightYellow.on_default().bold();

/// Execution options and runtime filters for the variant runner.
#[derive(Debug, Clone)]
pub struct RunnerOptions {
    /// Workspace root directory.
    pub workspace_root: PathBuf,

    /// Target results directory.
    pub out_dir: PathBuf,

    /// Execution timeout per variant.
    pub timeout: Duration,

    /// If true, suppresses non-error output.
    pub quiet: bool,
}

impl Default for RunnerOptions {
    fn default() -> Self {
        Self {
            workspace_root: PathBuf::from("."),
            out_dir: PathBuf::from("results"),
            timeout: Duration::from_secs(120),
            quiet: false,
        }
    }
}

/// Executes all variants declared across all suites in the master plan.
///
/// # Errors
/// Returns `HarnessError` if mandatory variant execution or process spawning fails.
pub fn execute_master_plan(
    plan: &MasterPlan,
    options: &RunnerOptions,
    suite_filter: &[String],
) -> Result<(), HarnessError> {
    std::fs::create_dir_all(&options.out_dir)?;

    let python_bin = resolve_python_runtime(&options.workspace_root);

    for suite in &plan.suites {
        if !suite_filter.is_empty() && !suite_filter.contains(&suite.name) {
            continue;
        }

        execute_suite(suite, &python_bin, options)?;
    }

    Ok(())
}

fn execute_suite(
    suite: &SuiteConfig,
    python_bin: &Path,
    options: &RunnerOptions,
) -> Result<(), HarnessError> {
    for variant in &suite.variants {
        if !options.quiet {
            anstream::println!(
                "{HEADER}     Running{HEADER:#} {}/{} ({})",
                suite.name,
                variant.name,
                variant.r#type
            );
        }

        let res = execute_variant(suite, variant, python_bin, options);
        if let Err(e) = res {
            if variant.optional {
                anstream::eprintln!(
                    "{WARNING}warning{WARNING:#}: optional variant `{}/{}` failed: {e}",
                    suite.name,
                    variant.name
                );
            } else {
                return Err(e);
            }
        }
    }

    Ok(())
}

fn execute_variant(
    suite: &SuiteConfig,
    variant: &VariantConfig,
    python_bin: &Path,
    options: &RunnerOptions,
) -> Result<(), HarnessError> {
    let mut cmd = variant_command(variant, python_bin)?;
    cmd.current_dir(&options.workspace_root);
    cmd.stdout(Stdio::inherit());
    cmd.stderr(Stdio::inherit());

    if let Some(py_dir) = python_bin.parent() {
        let current_path = env::var("PATH").unwrap_or_default();
        cmd.env("PATH", format!("{}:{current_path}", py_dir.display()));
        if let Some(venv_root) = py_dir.parent() {
            cmd.env("VIRTUAL_ENV", venv_root);
        }
    }
    cmd.env("PYTHON", python_bin);

    let execution_error = |message: String| HarnessError::Execution {
        suite: suite.name.clone(),
        variant: variant.name.clone(),
        message,
    };

    let start = Instant::now();
    let mut child = cmd.spawn().map_err(|e| {
        execution_error(format!("Failed to spawn process: {e}"))
    })?;

    let poll_interval = Duration::from_millis(50);
    loop {
        match child.try_wait() {
            Ok(Some(status)) if status.success() => break,
            Ok(Some(status)) => {
                return Err(execution_error(format_exit_status(status)));
            }
            Ok(None) if start.elapsed() > options.timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(HarnessError::Timeout {
                    suite: suite.name.clone(),
                    variant: variant.name.clone(),
                    timeout_secs: options.timeout.as_secs_f64(),
                });
            }
            Ok(None) => thread::sleep(poll_interval),
            Err(e) => {
                let _ = child.kill();
                return Err(execution_error(format!(
                    "Process polling error: {e}"
                )));
            }
        }
    }

    // Verify output container was produced
    let out_file_path = options.workspace_root.join(&variant.output_file);
    if !out_file_path.exists() && !variant.optional {
        return Err(execution_error(format!(
            "Expected output container '{}' was not created",
            variant.output_file
        )));
    }

    Ok(())
}

/// Builds the process that runs `variant`, before workspace-wide settings.
fn variant_command(
    variant: &VariantConfig,
    python_bin: &Path,
) -> Result<Command, HarnessError> {
    let config_error = |message: String| HarnessError::Config {
        path: PathBuf::from(&variant.output_file),
        message,
    };
    match variant.r#type.as_str() {
        "rust_bin" => {
            let mut c = Command::new("cargo");
            c.arg("run");
            if let Some(manifest) = &variant.manifest_path {
                c.arg("--manifest-path").arg(manifest);
            }
            if let Some(bin) = &variant.bin {
                c.arg("--bin").arg(bin);
            }
            // The child runs in the suite directory, so a relative
            // `CARGO_TARGET_DIR` inherited from the gate would resolve to a
            // second target tree. Anchor it to this process's directory.
            if let Some(dir) =
                env::var_os("CARGO_TARGET_DIR").map(PathBuf::from)
                && dir.is_relative()
                && let Ok(cwd) = env::current_dir()
            {
                c.env("CARGO_TARGET_DIR", cwd.join(dir));
            }
            Ok(c)
        }
        "python_script" => {
            let script = variant.script.as_ref().ok_or_else(|| {
                config_error(format!(
                    "Python variant '{}' missing 'script' specification",
                    variant.name
                ))
            })?;
            let mut c = Command::new(python_bin);
            c.arg(script);
            Ok(c)
        }
        "command" => {
            let command_str = variant.command.as_ref().ok_or_else(|| {
                config_error(format!(
                    "Command variant '{}' missing 'command' specification",
                    variant.name
                ))
            })?;
            let mut c = Command::new("sh");
            c.arg("-c").arg(command_str);
            Ok(c)
        }
        unknown => {
            Err(config_error(format!("Unknown variant type '{unknown}'")))
        }
    }
}

fn format_exit_status(status: ExitStatus) -> String {
    if let Some(code) = status.code() {
        return format!("Process exited with status code {code}");
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return format!("Process terminated by signal {signal}");
        }
    }
    "Process terminated without an exit code".to_string()
}

/// Resolves Python runtime per C-1 hierarchy:
/// 1. `PYTHON` environment variable
/// 2. Active `VIRTUAL_ENV`
/// 3. Crate-root `.venv`
/// 4. Local `.venv`
/// 5. System `PATH` (`python3`)
#[must_use]
pub fn resolve_python_runtime(workspace_root: &Path) -> PathBuf {
    // 1. PYTHON environment variable
    if let Ok(py) = env::var("PYTHON") {
        let p = PathBuf::from(py);
        if p.exists() {
            return p;
        }
    }

    // 2. Active VIRTUAL_ENV
    if let Ok(venv) = env::var("VIRTUAL_ENV") {
        let p = PathBuf::from(venv).join("bin").join("python3");
        if p.exists() {
            return p;
        }
    }

    // 3. Crate-root .venv
    let crate_root_venv =
        workspace_root.join(".venv").join("bin").join("python3");
    if crate_root_venv.exists() {
        return crate_root_venv;
    }

    // 4. Local .venv
    let local_venv = PathBuf::from(".venv/bin/python3");
    if local_venv.exists() {
        return local_venv;
    }

    // 5. System PATH
    PathBuf::from("python3")
}

#[cfg(test)]
#[cfg(unix)]
mod tests {
    use std::os::unix::process::ExitStatusExt;
    use std::process::ExitStatus;

    use super::format_exit_status;

    #[test]
    fn test_format_exit_status_reports_signal_number() {
        // Raw wait status 9: terminated by SIGKILL.
        assert_eq!(
            format_exit_status(ExitStatus::from_raw(9)),
            "Process terminated by signal 9"
        );
    }

    #[test]
    fn test_format_exit_status_reports_exit_code() {
        // Raw wait status 0x0100: exited with code 1.
        assert_eq!(
            format_exit_status(ExitStatus::from_raw(0x0100)),
            "Process exited with status code 1"
        );
    }
}
