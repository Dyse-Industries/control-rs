//! Interactive terminal console for `control-rs-ets` targets.
//!
//! ```text
//! cargo tui [qemu [arm|arm-sf|riscv32|riscv64] | teensy [PORT]] [OPTIONS]
//! ```
//!
//! Run `cargo tui -- --help` for options. Links, the session and run objects live in
//! `control-rs-ets-host`; this binary holds layout and input handling. A worker thread owns the
//! session, so drawing never blocks a run.

use std::env;
use std::io::stdout;
use std::process::exit;
use std::time::Duration;

use control_rs_ets_host::ETSBridge;
use control_rs_ets_host::session::SessionOptions;
use control_rs_ets_host::target::{Target, parse_targets};
use crossterm::event::{self, Event};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode,
    enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use crate::app::App;
use crate::worker::Worker;

mod app;
mod keys;
mod render;
mod worker;

/// Discovery is repeated at this interval until the target answers.
const DISCOVERY_RETRY: Duration = Duration::from_millis(500);

/// How long one pass of the event loop waits for a key.
const TICK: Duration = Duration::from_millis(30);

/// Result of the interactive event loop.
type TuiResult = Result<(), Box<dyn std::error::Error>>;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print_help();
        return;
    }
    let target = match parse_targets(&args) {
        Ok(targets) if targets.len() > 1 => {
            eprintln!(
                "error: Multiple targets or 'all' is not supported for TUI"
            );
            exit(1);
        }
        Ok(mut targets) => targets.pop().unwrap_or_else(Target::qemu_arm),
        Err(e) => {
            eprintln!("error: {e}");
            exit(1);
        }
    };
    let bridge = match ETSBridge::new(target, false) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: failed to start bridge: {e}");
            exit(1);
        }
    };
    if let Err(e) = run(bridge) {
        eprintln!("error: TUI error: {e}");
        exit(1);
    }
}

fn print_help() {
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
}

/// Sets up the terminal, runs the event loop and restores the terminal.
fn run(bridge: ETSBridge) -> TuiResult {
    let mut app = App::new(
        bridge.target_info().to_string(),
        bridge.link_info().to_string(),
    );
    let mut worker = Worker::spawn(
        bridge,
        SessionOptions {
            discovery_timeout: DISCOVERY_RETRY,
            ..SessionOptions::default()
        },
    );
    enable_raw_mode()?;
    let mut out = stdout();
    execute!(out, EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(out))?;
    let result = event_loop(&mut terminal, &mut app, &worker);
    worker.shutdown();
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}

/// Draws when something changed, applies worker updates and keys until the operator quits.
fn event_loop<B>(
    terminal: &mut Terminal<B>,
    app: &mut App,
    worker: &Worker,
) -> TuiResult
where
    B: ratatui::backend::Backend,
    B::Error: 'static,
{
    while !app.quit {
        app.drain(&worker.updates);
        if app.dirty {
            terminal.draw(|f| render::draw(f, app))?;
            app.dirty = false;
        }
        match event::poll(TICK)?.then(event::read).transpose()? {
            Some(Event::Key(key)) => {
                if let Some(request) = app.on_key(key) {
                    worker.submit(request);
                }
            }
            Some(Event::Resize(..)) => app.dirty = true,
            _ => {}
        }
    }
    Ok(())
}
