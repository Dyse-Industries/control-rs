use control_rs_xtask::ci::common::TaskResult;
use regex::Regex;
use std::time::Instant;

pub fn run() -> TaskResult {
    println!("Running tarpaulin coverage...");
    let start_time = Instant::now();
    let tarpaulin_output = crate::cargo!(
        "tarpaulin",
        "--verbose",
        "--color",
        "never",
        "--out",
        "Html",
    );

    // Tarpaulin mixes stdout and stderr, combine them for parsing
    let log_str = format!(
        "{}\n{}",
        String::from_utf8_lossy(&tarpaulin_output.stderr),
        String::from_utf8_lossy(&tarpaulin_output.stdout)
    );

    let time = start_time.elapsed().as_secs_f32();
    let success = tarpaulin_output.status.success();

    let re_coverage =
        Regex::new(r"(\d+\.\d+)% coverage, (\d+)/(\d+) lines covered").unwrap();

    let (percent, cov_lines, tot_lines) =
        if let Some(caps) = re_coverage.captures(&log_str) {
            (
                caps[1].to_string(),
                caps[2].to_string(),
                caps[3].to_string(),
            )
        } else {
            ("0.00".to_string(), "0".to_string(), "0".to_string())
        };

    TaskResult {
        success,
        time,
        percent,
        cov_lines,
        tot_lines,
        log_str,
        ..Default::default()
    }
}
