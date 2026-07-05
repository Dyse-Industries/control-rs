use crate::polynomial::polynomial::StaticPolynomial;

/// A constant polynomial: $p(x) = c$.
pub type Constant<T> = StaticPolynomial<T, 1>;

/// A linear polynomial: $p(x) = a x + b$.
pub type Line<T> = StaticPolynomial<T, 2>;
