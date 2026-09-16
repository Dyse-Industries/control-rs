//! Standalone quality report aggregator tool (`cargo report` / `report`).

fn main() {
    let args: Vec<String> = std::env::args().collect();
    std::process::exit(control_rs_ci::cli::report::main_impl(&args));
}
