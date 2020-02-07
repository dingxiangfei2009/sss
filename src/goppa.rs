use alga::general::Field;
use failure::Fail;
use num::{One, Zero};
use rand::{CryptoRng, RngCore};

use crate::{
    field::{ArbitraryElement, FiniteField},
    Coord, Polynomial,
};

pub struct Goppa<F> {
    points: Vec<F>,
    g_poly: Polynomial<F>,
    k: usize,
    n: usize,
    t: usize,
}

impl<F: Clone + FiniteField + ArbitraryElement> Goppa<F> {
    pub fn generate<R: CryptoRng + RngCore>(
        rng: &mut R,
        correction_level: usize,
        size: usize,
    ) -> Self {
        let n = size;
        let t = correction_level;
        let zero = F::zero();
        assert!(t >= 2);
        assert!(F::degree_extension() * t < n);
        let k = n - F::degree_extension() * t;
        loop {
            let g = Polynomial(
                (0..t)
                    .map(|_| F::arbitrary(rng))
                    .chain(Some(F::one()))
                    .collect(),
            );
            let mut generators = vec![];
            // hint: this loop always terminates since t < n <= |F|
            while generators.len() < size {
                let a = F::arbitrary(rng);
                if generators.iter().any(|x| x != &a) {
                    continue;
                }
                match g.eval_at(a.clone()) {
                    Coord(_, y) if y == zero => continue,
                    _ => generators.push(a),
                }
            }
        }
    }
}
