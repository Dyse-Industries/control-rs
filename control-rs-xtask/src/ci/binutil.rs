use crate::ci::common::TaskResult;
use regex::Regex;
use std::collections::HashMap;
use std::process::Command;
use std::time::Instant;

pub fn run(target_examples: &HashMap<String, Vec<String>>) -> TaskResult {
    println!("Running binutils size check...");
    let start_time = Instant::now();

    // 1. Tool Verification
    let llvm_tools_output = Command::new("rustup")
        .args(["component", "list", "--installed"])
        .output()
        .expect("Failed to run rustup component list");
    let installed_components =
        String::from_utf8_lossy(&llvm_tools_output.stdout);

    if !installed_components.contains("llvm-tools") {
        println!(
            "'llvm-tools-preview' not found. Installing via 'rustup component add llvm-tools-preview'..."
        );
        let status = Command::new("rustup")
            .args(["component", "add", "llvm-tools-preview"])
            .status()
            .expect("Failed to install llvm-tools-preview");

        if !status.success() {
            eprintln!("Failed to install 'llvm-tools-preview'");
            std::process::exit(1);
        }
    }

    crate::ci::common::require_cargo_subcommand("size", "cargo-binutils");

    let mut report = String::new();
    report.push_str(&crate::section_header!(2, "Binutils Size Report"));
    report.push_str(&crate::table_header!(
        [
            "Example",
            "Target",
            "`.text`",
            "`.rodata`",
            "`.data`",
            "`.bss`",
            "Total Flash",
            "Total RAM"
        ],
        [
            ":---", ":---", "---:", "---:", "---:", "---:", "---:", "---:"
        ]
    ));

    let mut binutil_success = true;

    // Regex to match the section output from cargo size -A
    // e.g. .text           408     134218332
    let re_section = Regex::new(r"^(?P<section>\.[a-zA-Z_]+)\s+(?P<size>\d+)")
        .expect("Failed to compile regex");

    for (target, examples) in target_examples {
        if examples.is_empty() {
            continue;
        }

        crate::ci::common::require_rustup_target(target);

        for example_name in examples {
            let size_output = crate::cargo!(
                "size",
                "--profile",
                "ci",
                "--target",
                target,
                "--example",
                example_name,
                "--",
                "-A"
            );

            if !size_output.status.success() {
                eprintln!(
                    "cargo size failed for example '{}': {}",
                    example_name,
                    String::from_utf8_lossy(&size_output.stderr)
                );
                binutil_success = false;
                continue;
            }

            let output_str = String::from_utf8_lossy(&size_output.stdout);

            let mut text_size = 0;
            let mut rodata_size = 0;
            let mut data_size = 0;
            let mut bss_size = 0;

            for line in output_str.lines() {
                if let Some(caps) = re_section.captures(line.trim()) {
                    let section = &caps["section"];
                    let size: usize = caps["size"].parse().unwrap_or(0);

                    match section {
                        ".text" => text_size = size,
                        ".rodata" => rodata_size = size,
                        ".data" => data_size = size,
                        ".bss" => bss_size = size,
                        _ => {} // Ignore other sections
                    }
                }
            }

            let total_flash = text_size + rodata_size + data_size;
            let total_ram = data_size + bss_size;

            report.push_str(&crate::table_row!(
                &format!("`{}`", example_name),
                &format!("`{}`", target),
                text_size,
                rodata_size,
                data_size,
                bss_size,
                &format!("**{}**", total_flash),
                &format!("**{}**", total_ram)
            ));
        }
    }

    report.push('\n');
    let time = start_time.elapsed().as_secs_f32();

    TaskResult {
        success: binutil_success,
        time,
        report,
        ..Default::default()
    }
}