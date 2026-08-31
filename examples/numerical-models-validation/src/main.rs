//! src/main.rs
//!
//! Orchestrates cross-language numerical model validation by calling each
//! model validator's `run()` function directly in-process.

mod matrix_validation;
mod polynomial_validation;
mod state_space_validation;
mod tensor_validation;
mod transfer_function_validation;

fn main() {
    println!();
    println!("--- [1/5] Matrix Validation ---");
    matrix_validation::run();
    println!();

    println!("--- [2/5] Polynomial Validation ---");
    polynomial_validation::run();
    println!();

    println!("--- [3/5] State-Space Validation ---");
    state_space_validation::run();
    println!();

    println!("--- [4/5] Transfer Function Validation ---");
    transfer_function_validation::run();
    println!();

    println!("--- [5/5] Tensor Validation ---");
    tensor_validation::run();
    println!();
}
