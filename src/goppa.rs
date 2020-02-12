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
    pow, Coord, EuclideanDomain, Polynomial,
};

pub struct GoppaDecoder<F, T> {
    points: Vec<F>,
    g_poly: Polynomial<F>,
    _p: PhantomData<fn() -> T>,
}

pub struct GoppaEncoder<F, T> {
    t: usize,
    parity_check: Array2<F>,
    _p: PhantomData<fn() -> T>,
}

impl<F, T> GoppaEncoder<F, T>
where
    F: FiniteField + Clone,
    T: ExtensionTower<Bottom = F>,
{
    pub fn encode(&self, msg: &[F]) -> Vec<F> {
        let m = T::degree_extension::<Int>().assert_usize();
        let h = stack![Axis(1), Array2::eye(m * self.t), self.parity_check];
        mat_vec_mul(&h, ArrayView1::from(msg)).to_vec()
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

fn invert_factor_poly<F: Field + Clone>(g_poly: &Polynomial<F>, alpha: F) -> Polynomial<F> {
    let Coord(_, g_alpha) = g_poly.eval_at(alpha.clone()); // g(a)
    let g_ = g_poly.clone() - Polynomial(vec![g_alpha.clone()]); // g(z) - g(a)
    let (q, r) = g_.div_with_rem(Polynomial(vec![-alpha, F::one()])); // (g(z) - g(a)) / (z - a)
    assert!(
        r.is_zero(),
        "factor polynomial should be invertible in the quotient ring, check your math"
    );
    -q * <F as TwoSidedInverse<Multiplicative>>::two_sided_inverse(&g_alpha)
}

impl<F, T> GoppaDecoder<F, T>
where
    F: FiniteField + Clone + Sync,
    T: ExtensionTower<Super = F>,
    T::Bottom: Send,
{
    pub fn decode(&self, sense: Vec<T::Bottom>) -> (Vec<T::Bottom>, Vec<T::Bottom>) {
        let n = self.points.len();
        let m = T::degree_extension::<Int>().assert_usize();
        let t = self.g_poly.degree();
        assert_eq!(sense.len(), t * m, "code length mismatch");
        // syndrome
        let mut s = Polynomial::zero();
        for (c, alpha) in sense.iter().zip(&self.points) {
            s = s + invert_factor_poly(&self.g_poly, alpha.clone()) * T::into_super(c.clone());
        }
        // solve key equation
        let (sigma, omega) = generialized_sugiyama(s, self.g_poly.clone());
        let sigma_ = sigma.clone().formal_derivative();
        let errs: Vec<_> = self
            .points
            .par_iter()
            .enumerate()
            .filter_map(|(loc, alpha)| {
                let Coord(_, e) = sigma.eval_at(alpha.clone());
                if e.is_zero() {
                    let Coord(_, p) = omega.eval_at(alpha.clone());
                    let Coord(_, q) = sigma_.eval_at(alpha.clone());
                    Some((loc, T::try_into_bottom(p / q).expect("check your math")))
                } else {
                    None
                }
            })
            .collect();
        let mut error = vec![T::Bottom::zero(); n];
        let mut code = sense;
        code.resize(n, T::Bottom::zero());
        for (err_loc, err) in errs {
            error[err_loc] = err.clone();
            code[err_loc] -= err;
        }
        (code, error)
    }
}

pub struct BinaryGoppaDecoder<F, T> {
    points: Vec<F>,
    g_poly: Polynomial<F>,
    _p: PhantomData<fn() -> T>,
}

impl<F, T> BinaryGoppaDecoder<F, T>
where
    F: FiniteField + Clone + Sync,
    T: ExtensionTower<Super = F>,
    T::Bottom: Send,
{
    pub fn from_decoder(dec: GoppaDecoder<F, T>) -> Self {
        let char = T::Bottom::characteristic::<Int>().assert_usize();
        assert_eq!(char, 2, "the field should have characteristic 2");
        Self {
            points: dec.points,
            g_poly: dec.g_poly,
            _p: PhantomData,
        }
    }
    pub fn decode(&self, sense: Vec<T::Bottom>) -> (Vec<T::Bottom>, Vec<T::Bottom>) {
        let n = self.points.len();
        let m = T::degree_extension::<Int>().assert_usize();
        let t = self.g_poly.degree();
        assert_eq!(sense.len(), t * m, "code length mismatch");
        // syndrome
        let mut s = Polynomial::zero();
        let g_sqr = pow(self.g_poly.clone(), 2);
        for (c, alpha) in sense.iter().zip(&self.points) {
            s = s + invert_factor_poly(&g_sqr, alpha.clone()) * T::into_super(c.clone());
        }
        // solve key equation
        let (sigma, omega) = generialized_sugiyama(s, g_sqr);
        let sigma_ = sigma.clone().formal_derivative();
        let errs: Vec<_> = self
            .points
            .par_iter()
            .enumerate()
            .filter_map(|(loc, alpha)| {
                let Coord(_, e) = sigma.eval_at(alpha.clone());
                if e.is_zero() {
                    let Coord(_, p) = omega.eval_at(alpha.clone());
                    let Coord(_, q) = sigma_.eval_at(alpha.clone());
                    Some((loc, T::try_into_bottom(p / q).expect("check your math")))
                } else {
                    None
                }
            })
            .collect();
        let mut error = vec![T::Bottom::zero(); n];
        let mut code = sense;
        code.resize(n, T::Bottom::zero());
        for (err_loc, err) in errs {
            error[err_loc] = err.clone();
            code[err_loc] -= err;
        }
        (code, error)
    }
}

pub fn goppa<F, T>(
    g: Polynomial<F>,
    alphas: Vec<F>,
) -> Option<(GoppaEncoder<T::Bottom, T>, GoppaDecoder<F, T>)>
where
    F: FiniteField + Clone,
    T: ExtensionTower<Super = F>,
    T::Bottom: Send + Sync,
{
    let n = alphas.len();
    let t = g.degree();
    let m = T::degree_extension::<Int>().assert_usize();

    let mut gs = vec![];
    for alpha in &alphas {
        let Coord(_, y) = g.eval_at(alpha.clone());
        if y.is_zero() {
            return None;
        }
        gs.push(<F as TwoSidedInverse<Multiplicative>>::two_sided_inverse(
            &y,
        ))
    }
    let alphas_arr = Array1::from(alphas.clone());
    let gs = Array1::from(gs);
    let mut h = vec![];
    let mut r = gs.clone();
    for _ in 0..t {
        let q: Vec<_> = r.iter().cloned().flat_map(T::to_vec).collect();
        let q = ArrayView2::from_shape((n, m), &q).expect("shape should be correct");
        let q = q.t();
        h.extend(q.iter().cloned());
        r = r * alphas_arr.clone();
    }
    let h = Array2::from_shape_vec((m * t, n), h).expect("shape should be correct");
    let h = gaussian_elimination(h);
    // expand and gaussian eliminate
    let bottom_one = T::Bottom::one();
    if h.slice(s![.., ..m * t])
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
        })
    {
        return None;
    }
    let h = h.slice(s![.., m * t..]).to_owned();

    let encoder = GoppaEncoder {
        parity_check: h,
        t,
        _p: PhantomData,
    };
    let decoder = GoppaDecoder {
        points: alphas,
        g_poly: g,
        _p: PhantomData,
    };
    Some((encoder, decoder))
}

pub fn generate<F, T, R>(
    rng: &mut R,
    correction_level: usize,
    size: usize,
) -> (GoppaEncoder<T::Bottom, T>, GoppaDecoder<F, T>)
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
    loop {
        let g = Polynomial(
            repeat_with(|| F::arbitrary(rng))
                .take(t)
                .chain(Some(F::one()))
                .collect(),
        );
        let mut alphas = vec![];
        // hint: this loop always terminates since t < n <= |F|
        while alphas.len() < n {
            let a = F::arbitrary(rng);
            if alphas.iter().any(|x| x == &a) {
                continue;
            }
            let Coord(_, y) = g.eval_at(a.clone());
            if y.is_zero() {
                continue;
            } else {
                alphas.push(a);
            }
        }
        if let Some(res) = goppa(g, alphas) {
            return res;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::arr2;
    use rand::{distributions::Distribution, rngs::OsRng};

    use crate::{
        field::F2,
        galois::{
            BottomField, FiniteExtensionTower, GF65536NTower, MonicPolynomial, PolynomialExtension,
            GF65536N,
        },
    };

    #[test]
    fn encode_decode_small() {
        let n = 80;
        let t = 5;
        let u = rand::distributions::uniform::Uniform::from(0..n);
        let (enc, dec) = generate::<GF65536N, GF65536NTower, _>(&mut OsRng, t, n);
        let mut x = vec![0; n];
        for _ in 0..t {
            x[u.sample(&mut OsRng)] = 1;
        }
        let x: Vec<_> = x.into_iter().map(F2).collect();
        let y = enc.encode(&x);
        let dec = BinaryGoppaDecoder::from_decoder(dec);
        let (code, error) = dec.decode(y);
        println!("code={:?}", code);
        println!("error={:?}", error);
        assert_eq!(error, x);
    }

    #[test]
    fn encode_decode_large() {
        let n = 6960;
        let t = 94;
        let u = rand::distributions::uniform::Uniform::from(0..n);
        let (enc, dec) = generate::<GF65536N, GF65536NTower, _>(&mut OsRng, t, n);
        let mut x = vec![0; n];
        for _ in 0..t {
            x[u.sample(&mut OsRng)] = 1;
        }
        let x: Vec<_> = x.into_iter().map(F2).collect();
        let y = enc.encode(&x);
        let dec = BinaryGoppaDecoder::from_decoder(dec);
        let (code, error) = dec.decode(y);
        println!("code={:?}", code);
        println!("error={:?}", error);
        assert_eq!(error, x);
    }

    struct F2Poly4;
    impl MonicPolynomial<F2> for F2Poly4 {
        fn degree() -> usize {
            4
        }
        fn repr() -> Polynomial<F2> {
            Polynomial(vec![F2(1), F2(0), F2(0), F2(1), F2(1)])
        }
    }
    type F16 = PolynomialExtension<F2, F2Poly4>;
    type F16Tower = FiniteExtensionTower<F16, FiniteExtensionTower<F2, BottomField>>;

    #[test]
    #[ignore]
    fn f256_goppa_example() {
        let g = Polynomial(vec![F16::root(), F16::one(), F16::one()]);
        let mut alphas = vec![F16::zero()];
        let mut alpha = F16::one();
        for _ in 0..15 {
            alphas.push(alpha.clone());
            alpha *= F16::root();
        }
        let (enc, dec) = goppa::<_, F16Tower>(g, alphas).unwrap();
        let expected_h = arr2(&[
            [F2(1), F2(0), F2(0), F2(0), F2(0), F2(1), F2(1), F2(1)],
            [F2(0), F2(1), F2(1), F2(1), F2(0), F2(0), F2(0), F2(1)],
            [F2(1), F2(1), F2(1), F2(1), F2(1), F2(0), F2(1), F2(1)],
            [F2(0), F2(1), F2(0), F2(1), F2(1), F2(1), F2(0), F2(1)],
            [F2(0), F2(0), F2(0), F2(1), F2(1), F2(1), F2(1), F2(0)],
            [F2(1), F2(1), F2(0), F2(0), F2(1), F2(1), F2(1), F2(0)],
            [F2(1), F2(0), F2(1), F2(1), F2(0), F2(1), F2(0), F2(0)],
            [F2(0), F2(1), F2(1), F2(0), F2(0), F2(1), F2(1), F2(0)],
        ]);
        assert_eq!(enc.parity_check, expected_h);
        let e = vec![
            F2(1),
            F2(0),
            F2(0),
            F2(0),
            F2(0),
            F2(0),
            F2(0),
            F2(0),
            F2(0),
            F2(0),
            F2(0),
            F2(0),
            F2(0),
            F2(0),
            F2(0),
            F2(0),
        ];
        let v = enc.encode(&e);
        let expected_v = vec![F2(1), F2(0), F2(0), F2(0), F2(0), F2(0), F2(0), F2(0)];
        assert_eq!(v, expected_v);
        let (_, e_) = dec.decode(v);
        assert_eq!(e_, e);
    }
}
