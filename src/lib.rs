#![type_length_limit = "2000000"]

#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

#[macro_use]
extern crate derive_more;

use std::{
    fmt::{Display, Formatter, Result as FmtResult},
    iter::repeat,
    ops::{Add, BitAnd, Div, Mul, MulAssign, Neg, Shr, Sub},
};

use alga::general::{Additive, Field, Identity};
use num::{
    traits::{One, Pow, Zero},
    BigUint, ToPrimitive,
};
use rand::RngCore;
use serde::{Deserialize, Serialize};

pub mod adapter;
pub mod conv;
pub mod facts;
pub mod field;
pub mod fourier;
pub mod galois;
pub mod gaussian;
pub mod goppa;
pub mod lattice;
pub mod lfsr;
pub mod linalg;
pub mod merkle;
pub mod reed_solomon;

pub use crate::field::{ArbitraryElement, FiniteField, FinitelyGenerated, GF2561DG2};

pub trait EuclideanDomain<Degree: Ord>: Zero {
    /// Euclidean measure of this domain
    fn degree(&self) -> Degree;

    /// Division with respect to this euclidean measure to give `(quotient, remainder)`,
    /// so that the remainder iss either zero, or the degree of the remainder is less
    /// than the degree of the divisor
    fn div_with_rem(self, other: Self) -> (Self, Self);

    fn gcd(mut self, mut other: Self) -> Self
    where
        Self: Clone,
    {
        if self.degree() < other.degree() {
            std::mem::swap(&mut self, &mut other);
        }
        if other.is_zero() {
            return self;
        }
        let (mut a, mut b) = (self, other);
        loop {
            let (_, r) = a.div_with_rem(b.clone());
            if r.is_zero() {
                return b;
            }
            a = b;
            b = r;
        }
    }

    fn extended_gcd<G, H>(self, other: Self) -> (G, H, Self)
    where
        Self: Clone + One + Sub<Output = Self>,
        G: Clone + Zero + One + Mul<Self, Output = G> + Sub<Output = G>,
        H: Clone + Zero + One + Mul<Self, Output = H> + Sub<Output = H>,
    {
        if self.degree() < other.degree() {
            let (r, s, d) = other.extended_gcd(self);
            return (s, r, d);
        }
        if other.is_zero() {
            return (G::one(), H::zero(), self);
        }
        let mut s = G::one();
        let mut t = H::zero();
        let mut u = G::zero();
        let mut v = H::one();
        let mut a = self;
        let mut b = other;
        loop {
            let (q, r) = a.div_with_rem(b.clone());
            let i = s - u.clone() * q.clone();
            let j = t - v.clone() * q.clone();
            s = u;
            t = v;
            u = i;
            v = j;
            a = b;
            b = r;
            if b.is_zero() {
                return (s, t, a);
            }
        }
    }
}

macro_rules! impl_euclidean_domain_int {
    () => {};
    ($t:ty) => {
        impl_euclidean_domain_int! {$t,}
    };
    ($t:ty, $($ts:tt),*) => {
        impl EuclideanDomain<$t> for $t {
            fn degree(&self) -> Self {
                *self
            }
            fn div_with_rem(self, other: Self) -> (Self, Self) {
                let d = self / other;
                (d, self - d * other)
            }
        }
        impl_euclidean_domain_int!($($ts),*);
    };
}

impl_euclidean_domain_int! {
    usize, u8, u16, u32, u64, u128
}

/// Univariate polynomial ring over a field `T`
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Polynomial<T>(pub Vec<T>);

impl<T: Display> Display for Polynomial<T> {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        let mut first = true;
        for (i, a) in self.0.iter().enumerate() {
            if first {
                first = false;
            } else {
                write!(f, " + ")?;
            }
            write!(f, "{} x^{}", a, i)?;
        }
        Ok(())
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct Coord<T>(pub T, pub T);

impl<T: Zero> Polynomial<T> {
    pub fn new(coeffs: impl IntoIterator<Item = T>) -> Self {
        let mut coeffs = coeffs.into_iter().collect();
        truncate_high_degree_zeros(&mut coeffs);
        Self(coeffs)
    }
}

impl<T: Clone + Zero> Add for Polynomial<T> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let (Polynomial(left), Polynomial(right)) = (self, other);
        let max = std::cmp::max(left.len(), right.len());
        let left = left.into_iter().chain(repeat(T::zero()));
        let right = right.into_iter().chain(repeat(T::zero()));
        Polynomial::new(left.zip(right).map(|(a, b)| a + b).take(max))
    }
}

impl<T> Sub for Polynomial<T>
where
    T: Clone + Zero + Sub<Output = T>,
{
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        let (Polynomial(left), Polynomial(right)) = (self, other);
        let max = std::cmp::max(left.len(), right.len());
        let left = left.into_iter().chain(repeat(T::zero()));
        let right = right.into_iter().chain(repeat(T::zero()));
        Polynomial::new(left.zip(right).map(|(a, b)| a - b).take(max))
    }
}

impl<T> Zero for Polynomial<T>
where
    T: Zero + Clone,
{
    fn zero() -> Self {
        Polynomial(vec![T::zero()])
    }

    fn is_zero(&self) -> bool {
        assert!(!self.0.is_empty());
        self.0.iter().all(|a| a.is_zero())
    }
}

impl<T> One for Polynomial<T>
where
    T: One + Zero + Clone,
{
    fn one() -> Self {
        Polynomial(vec![T::one()])
    }
}

impl<T> Default for Polynomial<T>
where
    T: Zero + Clone,
{
    fn default() -> Self {
        Self::zero()
    }
}

impl<T> EuclideanDomain<usize> for Polynomial<T>
where
    T: Field + Clone,
{
    fn degree(&self) -> usize {
        self.0.len() - 1
    }

    fn div_with_rem(self, other: Self) -> (Self, Self) {
        Polynomial::div(self, other)
    }
}

impl<T> Polynomial<T>
where
    T: Field + Clone,
{
    #[allow(clippy::should_implement_trait)] // REASON: it is not sensible to impl std::ops::Div if data like Polynomial is not a field element
    pub fn div(mut self, mut divisor: Self) -> (Self, Self) {
        truncate_high_degree_zeros(&mut self.0);
        truncate_high_degree_zeros(&mut divisor.0);
        assert!(!divisor.is_zero());

        let (Polynomial(mut a), Polynomial(d)) = (self, divisor);
        if a.len() < d.len() {
            return (Polynomial(vec![T::zero()]), Polynomial(a));
        }
        let mut quot = Vec::with_capacity(a.len() - d.len() + 1);
        let d_deg = d.len() - 1;
        assert!(a.len() >= d.len());
        for i in (0..a.len() - d_deg).rev() {
            // this is safe because the divisor `d` is not zero
            let q = a[i + d_deg].clone() / d[d_deg].clone();
            for j in 0..=d_deg {
                a[i + j] -= d[j].clone() * q.clone();
            }
            quot.push(q);
        }
        quot.reverse();
        (Polynomial(quot), Polynomial::new(a))
    }

    pub fn formal_derivative(mut self) -> Self {
        if self.0.len() > 1 {
            let mut n = T::one();
            self.0.remove(0);
            for a in &mut self.0 {
                *a *= n.clone();
                n += T::one();
            }
            self
        } else {
            Self(vec![T::zero()])
        }
    }

    pub fn is_zero(&self) -> bool {
        assert!(!self.0.is_empty());
        self.0.iter().all(|c| c.is_zero())
    }

    pub fn is_one(&self) -> bool {
        assert!(!self.0.is_empty());
        self.0[0] == T::one() && self.0.iter().skip(1).all(Zero::is_zero)
    }

    pub fn eval_at(&self, x: T) -> Coord<T> {
        let mut y = <T as Identity<Additive>>::identity();
        for a in self.0.iter().rev() {
            y *= x.clone();
            y += a.clone();
        }
        Coord(x, y)
    }

    /// Newton's method for reconstructing polynomials from sufficient set of graph points
    #[allow(clippy::many_single_char_names)] // REASON: match symbol names with textbooks and papers
    pub fn from_coords(c: &[Coord<T>]) -> Self
    where
        T: Eq,
    {
        let n = c.len();
        let mut w: Vec<_> = c.iter().map(|c| vec![c.1.clone()]).collect();
        let zero = <T as Identity<Additive>>::identity();
        for round in 2..=n {
            let mut w_ = vec![];
            for i in 0..=n - round {
                let j = i + (round - 1);
                let mut v = vec![zero.clone()];
                v.extend_from_slice(&w[i]);
                for (k, w) in w[i].iter().enumerate() {
                    v[k] -= c[j].0.clone() * w.clone();
                }
                for (k, w) in w[i + 1].iter().enumerate() {
                    v[k + 1] -= w.clone();
                    v[k] += c[i].0.clone() * w.clone();
                }
                let scale = c[i].0.clone() - c[j].0.clone();
                w_.push(v.into_iter().map(|w| w / scale.clone()).collect());
            }
            w = w_;
        }
        // remove zeroed high degrees
        let mut w = w
            .drain(0..1)
            .next()
            .expect("interpolation should return exactly one polynomial");
        truncate_high_degree_zeros(&mut w);
        Polynomial(w)
    }
}

impl<T> Mul for Polynomial<T>
where
    T: Mul<Output = T> + Zero + Clone,
{
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let (Self(left), Self(right)) = (self, other);
        #[allow(clippy::suspicious_arithmetic_impl)]
        // REASON: use of plus operator here is sensible
        let mut r = vec![T::zero(); left.len() + right.len()];
        #[allow(clippy::suspicious_arithmetic_impl)] // REASON: use of operators here is sensible
        for (i, left) in left.into_iter().enumerate() {
            for (r, r_) in r[i..]
                .iter_mut()
                .zip(right.iter().cloned().map(|right| left.clone() * right))
            {
                *r = r.clone() + r_;
            }
        }
        Polynomial::new(r)
    }
}

impl<T> Mul<T> for Polynomial<T>
where
    T: Mul<Output = T> + Zero + Clone,
{
    type Output = Self;
    fn mul(mut self, a: T) -> Self::Output {
        for x in &mut self.0 {
            *x = x.clone() * a.clone();
        }
        Polynomial::new(self.0)
    }
}

impl<T> MulAssign for Polynomial<T>
where
    T: Mul<Output = T> + Zero + Clone,
{
    fn mul_assign(&mut self, rhs: Self) {
        let lhs = self.clone();
        *self = lhs * rhs;
    }
}

impl<T> Div<T> for Polynomial<T>
where
    T: Div<Output = T> + Zero + Clone,
{
    type Output = Self;
    fn div(mut self, a: T) -> Self::Output {
        for x in &mut self.0 {
            *x = x.clone() / a.clone();
        }
        Polynomial::new(self.0)
    }
}

impl<T> Neg for Polynomial<T>
where
    T: Neg<Output = T>,
{
    type Output = Self;
    fn neg(self) -> Self {
        Self(self.0.into_iter().map(|x| -x).collect())
    }
}

/// Given a lossy graph of a polynomial of degree `threshold - 1`, with at most
/// `(c.len() - threshold + 1) / 2` errors, recover the original polynomial
#[allow(clippy::many_single_char_names)] // REASON: match symbol names with textbooks and papers
pub fn error_correct<T>(c: &[Coord<T>], threshold: usize) -> Option<(Polynomial<T>, Vec<usize>)>
where
    T: Field + Clone + Eq + Send + Sync,
{
    let n = c.len();
    if threshold < 1 || n < threshold {
        return None;
    }

    // the degree of the original polynomial
    let k = threshold - 1;
    let max_errors = (n - k) / 2;
    let x_pows: Vec<_> = c
        .iter()
        .map(|Coord(x, _)| {
            let mut pows = vec![];
            let mut pow = T::one();
            for _ in 0..n {
                pows.push(pow.clone());
                pow *= x.clone();
            }
            pows
        })
        .collect();
    let yx_pows: Vec<Vec<_>> = x_pows
        .iter()
        .zip(c.iter())
        .map(|(pows, Coord(_, y))| pows.iter().map(|p| p.clone() * y.clone()).collect())
        .collect();

    for e in (0..=max_errors).rev() {
        use rayon::prelude::*;

        let m: Vec<_> = (0..n)
            .into_par_iter()
            .flat_map(|i| {
                yx_pows[i]
                    .par_iter()
                    .take(e)
                    .cloned()
                    .chain(x_pows[i].par_iter().take(n - e).map(|x| -x.clone()))
                    .chain(Some(-yx_pows[i][e].clone()))
            })
            .collect();
        let m = ndarray::Array::from_shape_vec((n, n + 1), m).expect("shape should be correct");
        if let Some(coeffs) = linalg::solve(m) {
            let mut coeffs = coeffs.to_vec();
            let es = Polynomial(coeffs.drain(0..e).chain(Some(T::one())).collect());
            let qs = Polynomial(coeffs);
            let (fs, rs) = qs.div(es.clone());
            if fs.0.len() < k {
                None?;
            } else if rs.is_zero() {
                let errors: Vec<_> = c
                    .into_par_iter()
                    .enumerate()
                    .filter_map(|(i, Coord(x, _))| {
                        let Coord(_, y) = es.eval_at(x.clone());
                        if y.is_zero() {
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .collect();
                return Some((fs, errors));
            }
        }
    }

    None
}

pub fn truncate_high_degree_zeros<T: Zero>(w: &mut Vec<T>) {
    let mut zeroed = w.len();
    while zeroed > 1 && w[zeroed - 1].is_zero() {
        zeroed -= 1;
    }
    w.truncate(zeroed);
}

fn normal_basis_test_polynomial<F>(deg_extension: usize) -> Vec<F>
where
    F: Field,
{
    let mut p = vec![F::zero(); deg_extension + 1];
    p[deg_extension] = F::one();
    p[0] -= F::one();
    p
}

pub fn find_normal_basis<R, F>(rng: &mut R) -> F
where
    R: rand::RngCore,
    F: FiniteField + ArbitraryElement + Clone,
{
    loop {
        let alpha = F::arbitrary(rng);
        if alpha.is_zero() {
            continue;
        }
        if test_normal_basis(alpha.clone(), F::degree_extension()) {
            return alpha;
        }
    }
}

fn assert_subfield<F: FiniteField>(deg_ext: usize) {
    assert_eq!(
        F::degree_extension() % deg_ext,
        0,
        "finite field GF({}^{}) is not a subfield of GF({}^{})",
        F::characteristic(),
        deg_ext,
        F::characteristic(),
        F::degree_extension()
    );
}

pub fn test_normal_basis<F>(alpha: F, deg_ext: usize) -> bool
where
    F: FiniteField + Clone,
{
    assert_subfield::<F>(deg_ext);
    let g: Polynomial<F> = Polynomial::new(normal_basis_test_polynomial(deg_ext));
    let mut beta = alpha.clone();
    let mut p = vec![];
    for _ in 0..deg_ext {
        p.push(beta.clone());
        beta = pow(beta, F::characteristic());
    }
    assert!(alpha == beta, "field element is not in the subfield");
    p.reverse();
    let p = Polynomial::new(p);
    let gcd = g.clone().gcd(p);
    !gcd.is_zero() && gcd.degree() == 0
}

pub fn search_normal_basis<F, G>(deg_ext: usize) -> F
where
    F: FiniteField + Clone + FinitelyGenerated<G>,
{
    assert_subfield::<F>(deg_ext);

    // the following division will produce zero remainder after the last
    // assertion
    let exp: BigUint = (BigUint::from(F::characteristic()).pow(F::degree_extension()) - 1u8)
        / (BigUint::from(F::characteristic()).pow(deg_ext) - 1u8);

    let gamma = pow(<F as FinitelyGenerated<G>>::generator(), exp);
    let mut alpha = gamma.clone();
    let mut c = BigUint::from(F::characteristic()).pow(deg_ext) - 1u8;
    while !c.is_zero() {
        if test_normal_basis(alpha.clone(), deg_ext) {
            return alpha;
        }
        c -= 1u8;
        alpha *= gamma.clone();
    }
    unreachable!("expecting finding a normal basis before this point")
}

// NOTE: as of Rust 1.27.0, typeck cannot deduce that
// `<E as for<'a> BitAnd<'a E>>::Output: Zero`
// even when it is the case
pub fn pow<F, E>(mut x: F, mut exp: E) -> F
where
    E: BitAnd<Output = E> + Shr<usize, Output = E> + From<u8> + Zero + Clone,
    F: Mul<Output = F> + One + Clone,
{
    let mut p = x;
    x = F::one();
    let bit = E::from(1);
    while !exp.is_zero() {
        if !(exp.clone() & bit.clone()).is_zero() {
            x = x * p.clone();
        }
        p = p.clone() * p;
        exp = exp >> 1;
    }
    x
}

pub fn compute_cyclotomic_cosets<F: FiniteField>(n: usize) -> Vec<Vec<usize>> {
    let mut q = BigUint::from(1u8);
    let mut p_pows = vec![];
    for _ in 0..F::degree_extension() {
        p_pows.push(q.clone());
        q *= F::characteristic();
        q %= n;
    }
    q += n - 1;
    q %= n;
    assert!(q.is_zero(), "{} is not zero", q);

    // TODO: maybe use ufds here?
    let mut cosets = vec![vec![0]];
    let mut s = std::collections::BTreeSet::new();
    for i in 1..n {
        if s.contains(&i.into()) {
            continue;
        }
        let mut coset = vec![];
        for p in &p_pows {
            let j = (p.clone() * i) % n;
            if !s.insert(j.clone()) {
                break;
            }
            coset.push(j.to_usize().expect("size should fit"));
        }
        cosets.push(coset);
    }
    cosets
}

pub fn uniform_sample<T, R>(rng: &mut R, limit: T) -> T
where
    R: RngCore,
    T: Zero + One + Ord + Clone + Div<Output = T>,
{
    let mut low = T::zero();
    let mut high = limit;
    let one = T::one();
    let two = one.clone() + one.clone();
    let mut bit_pool = crate::gaussian::BitPool::new();
    while low.clone() + one.clone() < high {
        let mid = (low.clone() + high.clone()) / two.clone();
        if bit_pool.take_bit(rng) {
            low = mid;
        } else {
            high = mid;
        }
    }
    low
}

#[cfg(test)]
mod tests;
