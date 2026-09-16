//! Standalone requirement traceability CLI tool for `control-rs` (`cargo trace` / `trace`).

fn main() {
    let args: Vec<String> = std::env::args().collect();
    std::process::exit(control_rs_ci::cli::trace::main_impl(&args));
}
