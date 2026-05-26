use control_rs_xtask::ci::common::TaskResult;
use regex::Regex;
use std::collections::HashMap;
use std::time::Instant;

pub fn run(target_examples: &HashMap<String, Vec<String>>) -> TaskResult {
    println!("Running bloat check...");
    let start_time = Instant::now();

    control_rs_xtask::ci::common::require_cargo_subcommand(
        "bloat",
        "cargo-bloat",
    );

    let mut report = crate::section_header!(
        2,
        "Top 10 Largest Functions by Size (per Example)"
    );

    let mut bloat_success = true;

    for (target, examples) in target_examples {
        if examples.is_empty() {
            continue;
        }

        control_rs_xtask::ci::common::require_rustup_target(target);

        for example_name in examples {
            let bloat_output = crate::cargo!(
                "bloat",
                "--profile",
                "ci",
                "--target",
                target,
                "--example",
                example_name,
                "--message-format",
                "json",
                "-n",
                "50",
            );

            if !bloat_output.status.success() {
                eprintln!(
                    "cargo bloat failed for example '{}': {}",
                    example_name,
                    String::from_utf8_lossy(&bloat_output.stderr)
                );
                bloat_success = false;
                continue; // Try the next example
            }

            let bloat_json = String::from_utf8_lossy(&bloat_output.stdout);

            let re_func = Regex::new(
                r#"(?x)
                \{
                "name"\s*:\s*"(?P<name>[^"]+)"\s*,
                \s*
                "size"\s*:\s*(?P<size>\d+)
            "#,
            )
            .expect("Failed to compile JSON parse regex");

            report.push_str(&crate::section_header!(
                3,
                &format!("Example: `{}` (Target: `{}`)", example_name, target)
            ));
            report.push_str(&crate::table_header!(
                ["Function", "Size (bytes)"],
                [":---", "---:"]
            ));

            for caps in re_func.captures_iter(&bloat_json).take(10) {
                let name = &caps["name"];
                let size = &caps["size"];

                // Escape pipe characters in function names to not break the Markdown table.
                let escaped_name = name.replace('|', "\\|");
                report.push_str(&crate::table_row!(
                    &format!("`{}`", escaped_name),
                    size
                ));
            }
            report.push('\n');
        }
    }

    let time = start_time.elapsed().as_secs_f32();

    TaskResult {
        success: bloat_success,
        time,
        report,
        ..Default::default()
    }
}