use std::fs::File;
use std::io::Write;
use std::path::Path;

pub fn generate_const_math(out_dir: &str, max_dim: usize) {
    let dest_path = Path::new(out_dir).join("generated_dim_ops.rs");
    let mut f = File::create(&dest_path)
        .expect("Failed to create generated_dim_math.rs");

    for n in 0..=max_dim {
        for m in 0..=max_dim {
            // Addition
            writeln!(f, "impl DimAdd<Const<{m}>> for Const<{n}> {{ type Output = Const<{}>; #[inline(always)] fn dimadd(self, _: Const<{m}>) -> Self::Output {{ Const }} }}", n + m).unwrap();
            // Subtraction (prevent underflow)
            if n >= m {
                writeln!(f, "impl DimSub<Const<{m}>> for Const<{n}> {{ type Output = Const<{}>; #[inline(always)] fn dimsub(self, _: Const<{m}>) -> Self::Output {{ Const }} }}", n - m).unwrap();
            }
            // Multiplication
            writeln!(f, "impl DimMul<Const<{m}>> for Const<{n}> {{ type Output = Const<{}>; #[inline(always)] fn dimmul(self, _: Const<{m}>) -> Self::Output {{ Const }} }}", n * m).unwrap();
            // Division (prevent divide by zero)
            if m > 0 {
                writeln!(f, "impl DimDiv<Const<{m}>> for Const<{n}> {{ type Output = Const<{}>; #[inline(always)] fn dimdiv(self, _: Const<{m}>) -> Self::Output {{ Const }} }}", n / m).unwrap();
            }
            // Max
            writeln!(f, "impl DimMax<Const<{m}>> for Const<{n}> {{ type Output = Const<{}>; #[inline(always)] fn dimmax(self, _: Const<{m}>) -> Self::Output {{ Const }} }}", n.max(m)).unwrap();
            // Min
            writeln!(f, "impl DimMin<Const<{m}>> for Const<{n}> {{ type Output = Const<{}>; #[inline(always)] fn dimmin(self, _: Const<{m}>) -> Self::Output {{ Const }} }}", n.min(m)).unwrap();
        }
    }
}
