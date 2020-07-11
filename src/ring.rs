use alga::general::{Field, Multiplicative, TwoSidedInverse};

pub trait RingInvertible: Sized {
    fn try_invert(input: Self) -> Option<Self>;
}

impl<T> RingInvertible for T
where
    T: Field,
{
    fn try_invert(input: Self) -> Option<Self> {
        Some(<Self as TwoSidedInverse<Multiplicative>>::two_sided_inverse(&input))
    }
}
