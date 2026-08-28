//! Run every native JSON generator.

fn main() {
    control_rs_numerical_model_examples::matrix::main();
    control_rs_numerical_model_examples::polynomial::main();
    control_rs_numerical_model_examples::state_space::main();
    control_rs_numerical_model_examples::transfer_function::main();
    control_rs_numerical_model_examples::tensor::main();
}
