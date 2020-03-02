//! Goppa code (Niederreiter)
use std::{iter::repeat_with, marker::PhantomData};

use alga::general::{Field, Multiplicative, TwoSidedInverse};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis};
use num::{One, Zero};
use rand::{CryptoRng, Rng, RngCore};
use rayon::prelude::*;

use crate::{
    adapter::Int,
    field::{ArbitraryElement, FiniteField, PrimeSubfield, F2},
    galois::ExtensionTower,
    int_inj,
    linalg::{gaussian_elimination, mat_vec_mul},
    poly::MultipointEvaluator,
    pow, Coord, EuclideanDomain, Polynomial,
};

#[derive(Clone)]
pub struct GoppaDecoder<F, T, M> {
    points: Vec<F>,
    g_poly: Polynomial<F>,
    multipoint_eval: M,
    invert_factors: Vec<Polynomial<F>>,
    _p: PhantomData<fn() -> T>,
}

#[derive(Clone)]
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
                                                                 // compute (g(z) - g(a)) / (z - a)
                                                                 // specialized since the divisor has degree one
    let Polynomial(mut g_) = g_;
    let mut q_last = g_.pop().expect("at least one coefficient; check your math");
    let mut q = vec![];
    for g in g_.into_iter().rev() {
        q.push(q_last.clone());
        q_last = g - alpha.clone() * q_last;
    }
    assert!(q_last.is_zero());
    q.reverse();
    let q = Polynomial::from(q);
    -q * <F as TwoSidedInverse<Multiplicative>>::two_sided_inverse(&g_alpha)
}

impl<F, T, M> GoppaDecoder<F, T, M>
where
    F: FiniteField + Clone + Sync + Send,
    T: ExtensionTower<Super = F>,
    T::Bottom: Send + Sync,
    M: MultipointEvaluator<F>,
{
    pub fn decode(&self, sense: Vec<T::Bottom>) -> (Vec<T::Bottom>, Vec<T::Bottom>) {
        let n = self.points.len();
        let m = T::degree_extension::<Int>().assert_usize();
        let t = self.g_poly.degree();
        assert_eq!(sense.len(), t * m, "code length mismatch");
        // syndrome
        let s = sense
            .par_iter()
            .zip(&self.invert_factors)
            .map(|(c, a)| a.clone() * T::into_super(c.clone()))
            .reduce(Polynomial::zero, |a, b| a + b);
        // solve key equation
        let (sigma, omega) = generialized_sugiyama(s, self.g_poly.clone());
        let sigma_ = sigma.clone().formal_derivative();
        let errs: Vec<_> = self
            .multipoint_eval
            .eval(sigma)
            .into_par_iter()
            .enumerate()
            .flat_map({
                let points = &self.points;
                move |(loc, e)| {
                    if e.is_zero() {
                        let Coord(_, p) = omega.eval_at(points[loc].clone());
                        let Coord(_, q) = sigma_.eval_at(points[loc].clone());
                        Some((loc, T::try_into_bottom(p / q).expect("check your math")))
                    } else {
                        None
                    }
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

pub struct BinaryGoppaDecoder<F, T, M> {
    points: Vec<F>,
    g_poly: Polynomial<F>,
    g_sqr: Polynomial<F>,
    invert_factors: Vec<Polynomial<F>>,
    multipoint_eval: M,
    _p: PhantomData<fn() -> T>,
}

impl<F, T, M> BinaryGoppaDecoder<F, T, M>
where
    F: FiniteField + Clone + Send + Sync,
    T: ExtensionTower<Super = F, Bottom = F2>,
    M: MultipointEvaluator<F>,
{
    pub fn from_decoder(dec: GoppaDecoder<F, T, M>) -> Self {
        let GoppaDecoder {
            points,
            g_poly,
            multipoint_eval,
            invert_factors,
            ..
        } = dec;
        let g_sqr = pow(g_poly.clone(), 2);
        let invert_factors = invert_factors
            .into_par_iter()
            .zip(&points)
            .map(|(poly, alpha)| poly.clone() * poly * Polynomial(vec![-alpha.clone(), F::one()]))
            .collect();
        Self {
            points,
            g_poly,
            g_sqr,
            multipoint_eval,
            invert_factors,
            _p: PhantomData,
        }
    }
    pub fn decode(&self, sense: Vec<T::Bottom>) -> (Vec<T::Bottom>, Vec<T::Bottom>) {
        let n = self.points.len();
        let m = T::degree_extension::<Int>().assert_usize();
        let t = self.g_poly.degree();
        assert_eq!(sense.len(), t * m, "code length mismatch");
        // syndrome
        let s = sense
            .par_iter()
            .zip(&self.invert_factors)
            .map(|(c, a)| a.clone() * T::into_super(c.clone()))
            .reduce(Polynomial::zero, |a, b| a + b);
        // solve key equation
        let (sigma, omega) = generialized_sugiyama(s, self.g_sqr.clone());
        let sigma_ = sigma.clone().formal_derivative();
        let errs: Vec<_> = self
            .multipoint_eval
            .eval(sigma)
            .into_par_iter()
            .enumerate()
            .flat_map({
                let points = &self.points;
                move |(loc, e)| {
                    if e.is_zero() {
                        let Coord(_, p) = omega.eval_at(points[loc].clone());
                        let Coord(_, q) = sigma_.eval_at(points[loc].clone());
                        Some((loc, T::try_into_bottom(p / q).expect("check your math")))
                    } else {
                        None
                    }
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

pub fn goppa<F, T, M>(
    g: Polynomial<F>,
    alphas: Vec<F>,
) -> Option<(GoppaEncoder<T::Bottom, T>, GoppaDecoder<F, T, M>)>
where
    F: FiniteField + Clone + Send + Sync,
    T: ExtensionTower<Super = F>,
    T::Bottom: Send + Sync,
    M: MultipointEvaluator<F>,
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
    let invert_factors = alphas
        .par_iter()
        .map(|alpha| invert_factor_poly(&g, alpha.clone()))
        .collect();

    let encoder = GoppaEncoder {
        parity_check: h,
        t,
        _p: PhantomData,
    };
    let decoder = GoppaDecoder {
        multipoint_eval: M::prepare(alphas.clone()),
        points: alphas,
        g_poly: g,
        invert_factors,
        _p: PhantomData,
    };
    Some((encoder, decoder))
}

fn sample<R: RngCore + CryptoRng, T: Clone>(
    n: Int,
    k: usize,
    rng: &mut R,
    convert: impl Fn(Int) -> T,
    admissible: impl Fn(T) -> bool,
) -> Vec<T> {
    let mut v: Vec<_> = vec![];
    let mut i = Int::zero();
    while v.len() < k {
        let val = convert(i.clone());
        if admissible(val.clone()) {
            v.push(val)
        }
        i += Int::one();
    }
    let mut w: f64 = (rng.gen::<f64>().ln() / (k as f64)).exp();
    while i < n {
        i += Int::from(
            rug::Integer::from_f64(rng.gen::<f64>().ln() / (1. - w).ln())
                .expect("within representable integers"),
        ) + Int::from(1u8);
        while i < n {
            let val = convert(i.clone());
            if admissible(val.clone()) {
                v[rng.gen_range(0, k)] = val;
                w *= (rng.gen::<f64>().ln() / (k as f64)).exp();
                break;
            } else {
                i += Int::one();
            }
        }
    }
    v
}

pub fn generate<F, T, M, R>(
    rng: &mut R,
    correction_level: usize,
    size: usize,
) -> (GoppaEncoder<T::Bottom, T>, GoppaDecoder<F, T, M>)
where
    F: FiniteField + Clone + ArbitraryElement + Send + Sync,
    T: ExtensionTower<Super = F>,
    T::Bottom: Send + Sync + PrimeSubfield,
    R: CryptoRng + RngCore,
    M: MultipointEvaluator<F>,
{
    let n = size;
    let t = correction_level;
    assert!(t >= 2);
    let t_ = F::degree_extension::<Int>().assert_usize() * t;
    assert!(t_ < n);
    let bases = T::basis_elements_over_bottom();
    let char: Int = T::Bottom::characteristic();
    loop {
        let g = Polynomial(
            repeat_with(|| F::arbitrary(rng))
                .take(t)
                .chain(Some(F::one()))
                .collect(),
        );
        // hint: this loop always terminates since t < n <= |F|
        let alphas = sample(
            F::field_size::<Int>(),
            n,
            rng,
            |mut x| {
                let mut basis = bases.iter();
                let mut val = F::zero();
                while !x.is_zero() {
                    let basis = basis.next().expect("within range").clone();
                    val += basis * T::into_super(int_inj(x.clone() % char.clone()));
                    x /= char.clone();
                }
                val
            },
            |x| !g.eval_at(x).1.is_zero(),
        );
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
        artin::GF65536NPreparedMultipointEvalVZG,
        field::F2,
        galois::{
            BottomField, FiniteExtensionTower, GF65536NTower, MonicPolynomial, PolynomialExtension,
            GF65536N,
        },
        poly::MultiPointEvalTable,
    };

    #[test]
    fn encode_decode_small() {
        let n = 80;
        let t = 5;
        let u = rand::distributions::uniform::Uniform::from(0..n);
        let (enc, dec) = generate::<GF65536N, GF65536NTower, GF65536NPreparedMultipointEvalVZG, _>(
            &mut OsRng, t, n,
        );
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
        let (enc, dec) = generate::<GF65536N, GF65536NTower, GF65536NPreparedMultipointEvalVZG, _>(
            &mut OsRng, t, n,
        );
        let dec = BinaryGoppaDecoder::from_decoder(dec);
        for _ in 0..4 {
            let mut x = vec![0; n];
            for _ in 0..t {
                x[u.sample(&mut OsRng)] = 1;
            }
            println!("x={:?}", x);
            let x: Vec<_> = x.into_iter().map(F2).collect();
            let y = enc.encode(&x);
            let (code, error) = dec.decode(y);
            println!("code={:?}", code);
            println!("error={:?}", error);
            assert_eq!(error, x);
            eprintln!("done")
        }
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
        let (enc, dec) = goppa::<_, F16Tower, MultiPointEvalTable<_>>(g, alphas).unwrap();
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

mod ser {
    use super::*;

    use serde::{de::Error as _, Deserialize, Deserializer, Serialize, Serializer};

    #[derive(Serialize, Deserialize)]
    pub struct GoppaEncoderRepr<F> {
        t: usize,
        parity_check: Vec<F>,
    }

    #[derive(Serialize, Deserialize)]
    pub struct GoppaDecoderRepr<F> {
        points: Vec<F>,
        g: Polynomial<F>,
        invert_factors: Option<Vec<Polynomial<F>>>,
    }

    impl<F, T> Serialize for GoppaEncoder<F, T>
    where
        F: Clone + Serialize,
    {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            // follow the logical order
            let parity_check = self.parity_check.iter().cloned().collect();
            let repr = GoppaEncoderRepr {
                t: self.t,
                parity_check,
            };
            repr.serialize(serializer)
        }
    }

    impl<F, T, M> Serialize for GoppaDecoder<F, T, M>
    where
        F: Clone + Serialize,
    {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let repr = GoppaDecoderRepr {
                points: self.points.clone(),
                g: self.g_poly.clone(),
                invert_factors: Some(self.invert_factors.clone()),
            };
            repr.serialize(serializer)
        }
    }

    impl<'a, F, T> Deserialize<'a> for GoppaEncoder<F, T>
    where
        F: FiniteField + Clone + Deserialize<'a>,
        T: ExtensionTower<Bottom = F>,
    {
        fn deserialize<D: Deserializer<'a>>(deserializer: D) -> Result<Self, D::Error> {
            let GoppaEncoderRepr { t, parity_check } =
                GoppaEncoderRepr::<F>::deserialize(deserializer)?;
            let m = T::degree_extension::<Int>().assert_usize();
            let parity_check_len = parity_check.len();
            let s = parity_check_len / m / t;
            let parity_check = Array2::from_shape_vec((m * t, s), parity_check).map_err(|_| {
                let s = (s * m * t).to_string();
                let s = &s as &str;
                D::Error::invalid_length(parity_check_len, &s)
            })?;
            Ok(Self {
                parity_check,
                t,
                _p: PhantomData,
            })
        }
    }

    impl<'a, F, T, M> Deserialize<'a> for GoppaDecoder<F, T, M>
    where
        F: Clone + Deserialize<'a> + Send + Sync + Field,
        M: MultipointEvaluator<F>,
    {
        fn deserialize<D: Deserializer<'a>>(deserializer: D) -> Result<Self, D::Error> {
            let GoppaDecoderRepr {
                points,
                g,
                invert_factors,
            } = GoppaDecoderRepr::deserialize(deserializer)?;
            let invert_factors = match invert_factors {
                Some(fs) => fs,
                None => points
                    .par_iter()
                    .map(|alpha: &F| invert_factor_poly(&g, alpha.clone()))
                    .collect(),
            };
            Ok(Self {
                multipoint_eval: M::prepare(points.clone()),
                points,
                g_poly: g,
                invert_factors,
                _p: PhantomData,
            })
        }
    }
}
