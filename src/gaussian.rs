use rand::{RngCore, CryptoRng};

pub trait GaussianDistribution {
    type Integer;
    fn sample<R: RngCore + CryptoRng>() -> Self::Integer;
}

pub struct BaseGaussianSampler<I> {
    b: usize,
    c: I,
}


