use codegen::Scope;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// Helper function to generate and write a single dimension math trait implementation.
#[allow(clippy::too_many_arguments)]
fn write_dim_op_impl<W: Write>(
    writer: &mut W,
    trait_name: &str,
    method_name: &str,
    n: usize,
    m: usize,
    out_val: usize,
) {
    let mut scope = Scope::new();
    let impl_block = scope.new_impl(format!("Const<{n}>"));

    impl_block.impl_trait(format!("{trait_name}<Const<{m}>>"));
    impl_block.associate_type("Output", format!("Const<{out_val}>"));

    impl_block
        .new_fn(method_name)
        .arg_self()
        .arg("_", format!("Const<{m}>"))
        .ret("Self::Output")
        .line("Const");

    writeln!(writer, "#[allow(clippy::use_self)]\n{}", scope.to_string())
        .expect("Failed to write trait implementation");
}

pub fn generate_const_math(out_dir: &str, max_dim: usize) {
    let dest_path = Path::new(out_dir).join("generated_dim_ops.rs");
    let file = File::create(&dest_path)
        .expect("Failed to create generated_dim_ops.rs");

    let mut f = BufWriter::new(file);

    for n in 0..=max_dim {
        for m in 0..=max_dim {
            // Addition: prevents n + m > usize::MAX
            if let Some(out_val) = n.checked_add(m) {
                write_dim_op_impl(&mut f, "DimAdd", "dimadd", n, m, out_val);
            }

            // Subtraction: prevents n - m < 0
            if let Some(out_val) = n.checked_sub(m) {
                write_dim_op_impl(&mut f, "DimSub", "dimsub", n, m, out_val);
            }

            // Multiplication: prevents n * m > usize::MAX
            if let Some(out_val) = n.checked_mul(m) {
                write_dim_op_impl(&mut f, "DimMul", "dimmul", n, m, out_val);
            }

            // Division: natively prevents division by zero (m == 0)
            if let Some(out_val) = n.checked_div(m) {
                write_dim_op_impl(&mut f, "DimDiv", "dimdiv", n, m, out_val);
            }

            // Max & Min cannot fail or overflow, so they remain unconditional
            write_dim_op_impl(&mut f, "DimMax", "dimmax", n, m, n.max(m));
            write_dim_op_impl(&mut f, "DimMin", "dimmin", n, m, n.min(m));
        }
    }
}
