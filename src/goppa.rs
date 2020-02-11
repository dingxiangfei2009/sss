//! Goppa code (Niederreiter)

use std::{iter::repeat_with, marker::PhantomData};

use alga::general::{Field, Multiplicative, TwoSidedInverse};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use num::{One, Zero};
use rand::{CryptoRng, RngCore};
use rayon::prelude::*;

use crate::{
    adapter::Int,
    field::{ArbitraryElement, FiniteField},
    galois::ExtensionTower,
    linalg::{gaussian_elimination, mat_vec_mul},
    Coord, EuclideanDomain, Polynomial,
};

pub struct GoppaDecoder<F, T> {
    points: Vec<F>,
    g_poly: Polynomial<F>,
    k: usize,
    n: usize,
    t: usize,
    _p: PhantomData<fn() -> T>,
}

pub struct GoppaEncoder<F> {
    parity_check: Array2<F>,
}

impl<F> GoppaEncoder<F>
where
    F: FiniteField + Clone,
{
    pub fn encode(&self, msg: &[F]) -> Vec<F> {
        mat_vec_mul(&self.parity_check, ArrayView1::from(msg)).to_vec()
    }
}

fn generialized_sugiyama<F>(
    syndrome: Polynomial<F>,
    generator: Polynomial<F>,
) -> (Polynomial<F>, Polynomial<F>)
where
    F: Field + Clone,
{
    let t = generator.degree();
    assert!(syndrome.degree() < t, "check your math");
    let max_r = t / 2;
    let mut a = generator;
    let mut b = syndrome;
    let (mut u, mut v) = (Polynomial::zero(), Polynomial::one());
    while b.degree() >= max_r {
        let (q, r) = a.div_with_rem(b.clone());
        let p = u - v.clone() * q;
        u = v;
        v = p;
        a = b;
        b = r;
    }
    (v, b)
}

impl<F, T> GoppaDecoder<F, T>
where
    F: FiniteField + Clone + Sync,
    T: ExtensionTower<Super = F>,
    T::Bottom: Send,
{
    fn invert_factor_poly(&self, alpha: F) -> Polynomial<F> {
        let Coord(_, g_alpha) = self.g_poly.eval_at(alpha.clone());
        let g_ = self.g_poly.clone() - Polynomial(vec![g_alpha.clone()]);
        let (q, r) = g_.div_with_rem(Polynomial(vec![-alpha, F::one()]));
        assert!(
            r.is_zero(),
            "factor polynomial should be invertible in the quotient ring, check your math"
        );
        -q * <F as TwoSidedInverse<Multiplicative>>::two_sided_inverse(&g_alpha)
    }
    pub fn decode(&self, sense: Vec<T::Bottom>) -> (Vec<T::Bottom>, Vec<T::Bottom>) {
        assert_eq!(sense.len(), self.n, "code length mismatch");
        // syndrome
        let mut s = Polynomial::zero();
        for (c, alpha) in sense.iter().zip(&self.points) {
            s = s + self.invert_factor_poly(alpha.clone()) * T::into_super(c.clone());
        }
        // solve key equation
        let (sigma, omega) = generialized_sugiyama(s, self.g_poly.clone());
        let sigma_ = sigma.clone().formal_derivative();
        let errs: Vec<_> = self.points.par_iter().enumerate().filter_map(|(loc, alpha)| {
            let Coord(_, e) = sigma.eval_at(alpha.clone());
            if e.is_zero() {
                None
            } else {
                let Coord(_, p) = omega.eval_at(alpha.clone());
                let Coord(_, q) = sigma_.eval_at(alpha.clone());
                Some((loc, T::try_into_bottom(p / q).expect("check your math")))
            }
        }).collect();
        let mut error = vec![T::Bottom::zero(); self.n];
        let mut code = sense;
        for (err_loc, err) in errs {
            error[err_loc] = err.clone();
            code[err_loc] -= err;
        }
        (code, error)
    }
}

pub fn generate<F, T, R>(
    rng: &mut R,
    correction_level: usize,
    size: usize,
) -> (GoppaEncoder<T::Bottom>, GoppaDecoder<F, T>)
where
    F: FiniteField + Clone + ArbitraryElement,
    T: ExtensionTower<Super = F>,
    T::Bottom: Send + Sync,
    R: CryptoRng + RngCore,
{
    let n = size;
    let t = correction_level;
    assert!(t >= 2);
    let t_ = F::degree_extension::<Int>().assert_usize() * t;
    assert!(t_ < n);
    let k = n - t_;
    let m = T::degree_extension::<Int>().assert_usize();
    loop {
        let g = Polynomial(
            repeat_with(|| F::arbitrary(rng))
                .take(t)
                .chain(Some(F::one()))
                .collect(),
        );
        let mut alphas = vec![];
        let mut gs = vec![];
        // hint: this loop always terminates since t < n <= |F|
        while alphas.len() < n {
            let a = F::arbitrary(rng);
            if alphas.iter().any(|x| x != &a) {
                continue;
            }
            let Coord(_, y) = g.eval_at(a.clone());
            if y.is_zero() {
                continue;
            } else {
                alphas.push(a);
                gs.push(<F as TwoSidedInverse<Multiplicative>>::two_sided_inverse(
                    &y,
                ));
            }
        }
        let alphas_arr = Array1::from(alphas.clone());
        let gs = Array1::from(gs);
        let mut h = Array2::from_shape_vec((0, n), vec![]).expect("shape should be correct");
        let mut r = gs.clone();
        for _ in 0..t {
            let q: Vec<_> = r.iter().cloned().flat_map(T::to_vec).collect();
            let q = ArrayView2::from_shape((n, m), &q)
                .expect("shape should be correct")
                .t()
                .to_owned();
            h = stack![Axis(1), h, q];
            r = r * alphas_arr.clone();
        }
        // let h = Array2::from_shape_vec((t, n), h).expect("shape should be correct");
        let h = gaussian_elimination(h);
        // expand and gaussian eliminate
        let bottom_one = T::Bottom::one();
        h.slice(s![.., ..m * t])
            .axis_iter(Axis(0))
            .into_par_iter()
            .enumerate()
            .any(|(i, r)| {
                r.axis_iter(Axis(0))
                    .into_par_iter()
                    .enumerate()
                    .any(|(j, a)| {
                        let a = a.into_scalar();
                        i == j && *a != bottom_one || i != j && !T::Bottom::is_zero(a)
                    })
            });
        let h = h.slice(s![.., ..m * t]).to_owned();

        let encoder = GoppaEncoder { parity_check: h };
        let decoder = GoppaDecoder {
            points: alphas,
            g_poly: g,
            k,
            n,
            t,
            _p: PhantomData,
        };
        return (encoder, decoder);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_decode() {
    }
}
