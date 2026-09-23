//! Interactive terminal console for `control-rs-ets` targets.
//!
//! ```text
//! cargo tui [qemu [arm|arm-sf|riscv32|riscv64] | teensy [PORT]] [OPTIONS]
//! ```
//!
//! Run `cargo tui -- --help` for options. Transport, framing and session
//! state live in `control-rs-ets-host`; this binary holds layout and input
//! handling.

use std::env;
use std::process::exit;

use control_rs_ets_host::ETSBridge;
use control_rs_ets_host::target::{Target, parse_targets};

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

    let bridge = match ETSBridge::new(target.clone(), false) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: failed to start bridge: {e}");
            exit(1);
        }
    };

    if let Err(e) = tui::run_tui(bridge, &target) {
        eprintln!("error: TUI error: {e}");
        exit(1);
    }
}
