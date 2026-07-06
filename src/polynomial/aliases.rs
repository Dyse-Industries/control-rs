use crate::polynomial::polynomial::Polynomial;

/// A constant polynomial: $p(x) = c$.
pub type Constant<T> = Polynomial<T, 1>;

/// A linear polynomial: $p(x) = a x + b$.
pub type Line<T> = Polynomial<T, 2>;
