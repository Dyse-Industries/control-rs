//! Main entrypoint for the `control-rs-tui` dashboard binary.

#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::equatable_if_let,
    clippy::indexing_slicing,
    clippy::map_unwrap_or,
    clippy::missing_const_for_fn,
    clippy::multiple_crate_versions,
    clippy::needless_pass_by_ref_mut,
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::unused_self
)]

use std::env;
use std::process::exit;

use control_rs_ets_host::ServerBridge;
use control_rs_ets_host::target::{Target, build_target_elf, parse_targets};

mod tui;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("Usage: control-rs-tui [OPTIONS]");
        println!();
        println!("Options:");
        println!("  -t, --target <TRIPLE[:BIN]> Target triple and binary");
        println!(
            "      --manifest-path <PATH>  Path to target Cargo.toml or crate directory"
        );
        println!("      --bin <BIN>             Binary name");
        println!(
            "      --serial                Connect to a physical target over serial"
        );
        println!("      --port <PORT>           Serial device path");
        println!(
            "      --baud <BAUD>           Serial baud rate [default: 115200]"
        );
        println!("      --release               Build target in release mode");
        println!("  -h, --help                  Print help");
        return;
    }

    let mut targets = match parse_targets(&args) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {e}");
            exit(1);
        }
    };

    let target = if targets.is_empty() {
        Target::qemu_arm()
    } else if targets.len() > 1 {
        eprintln!("error: Multiple targets or 'all' is not supported for TUI");
        exit(1);
    } else if let Some(t) = targets.pop() {
        t
    } else {
        Target::qemu_arm()
    };

    let elf_path = match &target {
        Target::Subprocess(sub) => match build_target_elf(sub) {
            Ok(path) => path,
            Err(e) => {
                eprintln!("error: {e}");
                exit(1);
            }
        },
        Target::Serial { .. } => String::new(),
    };

    let elf_opt = if elf_path.is_empty() {
        None
    } else {
        Some(elf_path.as_str())
    };

    let bridge = match ServerBridge::new(target.clone(), elf_opt, false) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: failed to start bridge: {e}");
            exit(1);
        }
    };

    if let Err(e) = tui::run_tui(bridge, &target, &elf_path) {
        eprintln!("error: TUI error: {e}");
        exit(1);
    }
}
