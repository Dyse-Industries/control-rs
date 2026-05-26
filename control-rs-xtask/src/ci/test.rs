use crate::ci::common::TaskResult;
use regex::Regex;
use std::time::Instant;

pub fn run() -> TaskResult {
    println!("Running tests...");
    let start_test = Instant::now();
    let test_output = crate::cargo!("test");

    let log_str = String::from_utf8_lossy(&test_output.stdout).to_string();
    let time = start_test.elapsed().as_secs_f32();
    let success = test_output.status.success();

    let re_passed = Regex::new(r"(\d+) passed").unwrap();
    let re_failed = Regex::new(r"(\d+) failed").unwrap();
    let re_ignored = Regex::new(r"(\d+) ignored").unwrap();

    let passed: usize = re_passed
        .captures_iter(&log_str)
        .filter_map(|c| c[1].parse::<usize>().ok())
        .sum();
    let failed: usize = re_failed
        .captures_iter(&log_str)
        .filter_map(|c| c[1].parse::<usize>().ok())
        .sum();
    let ignored: usize = re_ignored
        .captures_iter(&log_str)
        .filter_map(|c| c[1].parse::<usize>().ok())
        .sum();

    TaskResult {
        success,
        time,
        passed,
        failed,
        ignored,
        log_str,
        ..Default::default()
    }
}