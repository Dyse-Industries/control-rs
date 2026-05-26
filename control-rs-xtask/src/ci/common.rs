use std::collections::HashMap;
use std::env;
use std::fs;
use std::process::{Command, exit};

// --- Task Result Types ---

#[derive(Default)]
pub struct TaskResult {
    pub success: bool,
    pub time: f32,
    pub fmt_errors: usize,
    pub clippy_errors: usize,
    pub passed: usize,
    pub failed: usize,
    pub ignored: usize,
    pub percent: String,
    pub cov_lines: String,
    pub tot_lines: String,
    pub fmt_str: String,
    pub clippy_str: String,
    pub log_str: String,
    pub report: String,
}

// --- Macros ---

#[macro_export]
macro_rules! cargo {
    ($($arg:expr),* $(,)?) => {
        std::process::Command::new("cargo")
            .args([$($arg),*])
            .output()
            .expect("Failed to run cargo command")
    };
}

#[macro_export]
macro_rules! section_header {
    ($level:expr, $title:expr) => {
        format!("\n{} {}\n\n", "#".repeat($level), $title)
    };
}

#[macro_export]
macro_rules! table_header {
    ([$($col:expr),*], [$($align:expr),*]) => {{
        let cols = [$($col.to_string()),*];
        let aligns = [$($align.to_string()),*];
        let mut header = format!("| {} |\n", cols.join(" | "));
        header.push_str(&format!("|{}|\n", aligns.join("|")));
        header
    }};
}

#[macro_export]
macro_rules! table_row {
    ($($col:expr),*) => {
        format!("| {} |\n", [$($col.to_string()),*].join(" | "))
    };
}

// --- Common Functions ---

pub fn require_cargo_subcommand(subcommand: &str, install_package: &str) {
    let check = Command::new("cargo")
        .args([subcommand, "--version"])
        .output();
    if check.is_err() || !check.unwrap().status.success() {
        println!(
            "'{}' not found. Installing via 'cargo install'...",
            install_package
        );
        let status = Command::new("cargo")
            .args(["install", install_package])
            .status()
            .unwrap_or_else(|_| {
                panic!("Failed to install {}", install_package)
            });

        if !status.success() {
            eprintln!("Failed to install '{}'", install_package);
            exit(1);
        }
    }
}

pub fn require_rustup_target(target: &str) {
    let target_output = Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .expect("Failed to run rustup target list");
    let installed_targets = String::from_utf8_lossy(&target_output.stdout);

    if !installed_targets.contains(target) {
        eprintln!(
            "Target '{}' is not installed. Please run `rustup target add {}`.",
            target, target
        );
        exit(1);
    }
}

pub fn collect_versions() -> String {
    let mut section = String::new();
    section.push_str(&crate::section_header!(3, "System Information"));
    section.push_str(&crate::table_header!(
        ["Component", "Version"],
        [":---", ":---"]
    ));

    let os_info = format!("{} {}", env::consts::OS, env::consts::ARCH);
    section.push_str(&crate::table_row!("OS", os_info));

    fn get_version(cmd: &str, args: &[&str]) -> String {
        Command::new(cmd)
            .args(args)
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "Not found".to_string())
    }

    section.push_str(&crate::table_row!(
        "rustc",
        get_version("rustc", &["--version"])
    ));
    section.push_str(&crate::table_row!(
        "cargo",
        get_version("cargo", &["--version"])
    ));
    section.push_str(&crate::table_row!(
        "rustfmt",
        get_version("cargo", &["fmt", "--version"])
    ));
    section.push_str(&crate::table_row!(
        "clippy",
        get_version("cargo", &["clippy", "--version"])
    ));
    section.push_str(&crate::table_row!(
        "tarpaulin",
        get_version("cargo", &["tarpaulin", "--version"])
    ));
    section.push_str(&crate::table_row!(
        "bloat",
        get_version("cargo", &["bloat", "--version"])
    ));

    section.push_str(
        "\n<details>\n<summary>Dependency Tree</summary>\n\n```text\n",
    );
    section.push_str(&get_version("cargo", &["tree"]));
    section.push_str("\n```\n</details>\n");

    section
}

pub fn collect_examples(
    known_targets: &[&str],
) -> HashMap<String, Vec<String>> {
    let mut map: HashMap<String, Vec<String>> = HashMap::new();
    let mut common_examples = Vec::new();

    for &target in known_targets {
        map.insert(target.to_string(), Vec::new());
    }

    if let Ok(entries) = fs::read_dir("examples/ci") {
        for entry in entries.flatten() {
            if let Ok(file_type) = entry.file_type() {
                if file_type.is_dir() {
                    let dir_name =
                        entry.file_name().to_string_lossy().to_string();
                    if let Ok(files) = fs::read_dir(entry.path()) {
                        for file in files.flatten() {
                            let path = file.path();
                            if path.is_file()
                                && path
                                    .extension()
                                    .map_or(false, |ext| ext == "rs")
                            {
                                if let Some(example_name) =
                                    path.file_stem().and_then(|s| s.to_str())
                                {
                                    if dir_name == "common" {
                                        common_examples
                                            .push(example_name.to_string());
                                    } else {
                                        map.entry(dir_name.clone())
                                            .or_default()
                                            .push(example_name.to_string());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (_, examples) in map.iter_mut() {
        examples.extend(common_examples.clone());
    }

    for examples in map.values_mut() {
        examples.sort();
        examples.dedup();
    }

    map
}
