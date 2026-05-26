use control_rs_xtask::ci::common::TaskResult;
use regex::Regex;
use std::time::Instant;

pub fn run() -> TaskResult {
    println!("Running lint (fmt + clippy)...");
    let start_lint = Instant::now();
    let ansi_escape = Regex::new(r"\x1B\[[0-9;]*[mK]").unwrap();

    let fmt_output = crate::cargo!("fmt", "--all", "--", "--check");
    let fmt_str = String::from_utf8_lossy(&fmt_output.stdout);
    let fmt_str = ansi_escape.replace_all(&fmt_str, "").to_string();
    let fmt_errors = fmt_str.matches("Diff in").count();

    let clippy_output = crate::cargo!("clippy", "--", "-D", "warnings");
    let clippy_str = String::from_utf8_lossy(&clippy_output.stderr);
    let clippy_str = ansi_escape.replace_all(&clippy_str, "").to_string();
    let clippy_errors = clippy_str.matches("error:").count();

    let time = start_lint.elapsed().as_secs_f32();
    let success = fmt_output.status.success() && clippy_output.status.success();

    TaskResult {
        success,
        time,
        fmt_errors,
        clippy_errors,
        fmt_str,
        clippy_str,
        ..Default::default()
    }
}
