use super::*;

mod dyadic;
mod triadic;

pub use dyadic::{schoenhage_dyadic_mul, SchoenhageDyadicDomain};
pub use triadic::{schoenhage_triadic_mul, SchoenhageTriadicDomain};
