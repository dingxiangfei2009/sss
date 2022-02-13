//#![type_length_limit = "3000000"]

#[macro_use]
extern crate derive_more;

use std::{
    cmp::max,
    fmt::{Display, Formatter, Result as FmtResult},
    iter::repeat_with,
    mem::{replace, swap},
    ops::{Add, BitAnd, Div, DivAssign, Mul, MulAssign, Neg, Shr, Sub},
};

use alga::general::{Additive, Field, Identity, Ring};
use num::{
    bigint::Sign,
    traits::{One, Zero},
    BigInt, BigUint,
};
use rand::RngCore;
use serde::{Deserialize, Serialize};

#[macro_use]
pub mod array;

pub mod adapter;
pub mod artin;
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
pub mod mceliece;
pub mod merkle;
pub mod poly;
pub mod primes;
pub mod reed_solomon;
pub mod ring;
pub mod ser;
pub mod unsafe_field;

use crate::field::nat_inj;
pub use crate::{
    adapter::Int,
    field::int_inj,
    field::{ArbitraryElement, FiniteField, FinitelyGenerated, GF2561DG2},
    ring::RingInvertible,
};

/// A [Euclidean Domain](https://en.wikipedia.org/wiki/Euclidean_domain) that has a well-defined
/// Euclidean function over non-zero elements of the ring.
pub trait EuclideanDomain<Degree: Ord>: Zero + One + Sub<Output = Self> {
    /// Euclidean measure of this domain
    fn degree(&self) -> Degree;

    /// Division with respect to this euclidean measure to give `(quotient, remainder)`,
    /// so that the remainder is either zero, or the degree of the remainder is less
    /// than the degree of the divisor
    fn div_with_rem(self, other: Self) -> (Self, Self);

    fn rem(self, other: Self) -> Self {
        self.div_with_rem(other).1
    }

    fn div(self, other: Self) -> Self {
        self.div_with_rem(other).0
    }

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
        Self: Clone,
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

impl EuclideanDomain<BigInt> for BigInt {
    fn degree(&self) -> BigInt {
        self.clone()
    }
    fn div_with_rem(self, other: Self) -> (Self, Self) {
        assert_eq!(other.sign(), Sign::Plus);
        let rem = self.clone() % other.clone();
        let quot = self / other;
        (quot, rem)
    }
}

/// Univariate polynomial ring over a field `T`
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
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
/// A pair of value and usually a result of evaluating a polynomial at a point
pub struct Coord<T>(pub T, pub T);

impl<T: Zero> Polynomial<T> {
    pub fn new(coeffs: impl IntoIterator<Item = T>) -> Self {
        let coeffs: Vec<_> = coeffs.into_iter().collect();
        Self::from(coeffs)
    }
    pub fn coeff(&self, deg: usize) -> T
    where
        T: Clone,
    {
        if deg < self.0.len() {
            self.0[deg].clone()
        } else {
            T::zero()
        }
    }
}

impl<T: Zero> From<Vec<T>> for Polynomial<T> {
    fn from(mut v: Vec<T>) -> Self {
        if v.is_empty() {
            v.push(T::zero());
        } else {
            truncate_high_degree_zeros(&mut v);
        }
        Polynomial(v)
    }
}

impl<T: Zero> Add for Polynomial<T> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        if self.is_zero() {
            return other;
        }
        if other.is_zero() {
            return self;
        }
        let (Polynomial(left), Polynomial(right)) = (self, other);
        let (mut left, right) = if left.len() < right.len() {
            (right, left)
        } else {
            (left, right)
        };
        let mut t = T::zero();
        for (i, b) in right.into_iter().enumerate() {
            swap(&mut left[i], &mut t);
            t = t + b;
            swap(&mut left[i], &mut t);
        }
        Polynomial::from(left)
    }
}

impl<T> Sub for Polynomial<T>
where
    T: Zero + Sub<Output = T>,
{
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        if other.is_zero() {
            return self;
        }
        let (Polynomial(mut left), Polynomial(right)) = (self, other);
        while left.len() < right.len() {
            left.push(T::zero());
        }
        let mut t = T::zero();
        for (a, b) in left.iter_mut().zip(right) {
            swap(&mut t, a);
            t = t - b;
            swap(&mut t, a);
        }
        Polynomial::from(left)
    }
}

impl<T> Zero for Polynomial<T>
where
    T: Zero,
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
    T: One + Zero + Sub<Output = T> + Clone,
{
    fn one() -> Self {
        Polynomial(vec![T::one()])
    }
}

impl<T> Default for Polynomial<T>
where
    T: Zero,
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
        Polynomial::divide_by(self, other)
    }
}

impl<T: Zero> Polynomial<T> {
    pub fn mul_pow_x(mut self, pow: usize) -> Self {
        if self.is_zero() {
            self
        } else {
            Polynomial::new(repeat_with(T::zero).take(pow).chain(self.0.drain(..)))
        }
    }

    pub fn is_zero(&self) -> bool {
        assert!(!self.0.is_empty());
        self.0.iter().all(|c| c.is_zero())
    }

    pub fn degree(&self) -> usize {
        self.0.len() - 1
    }
}

impl<T> From<T> for Polynomial<T> {
    fn from(x: T) -> Self {
        Polynomial(vec![x])
    }
}

impl<T> Polynomial<T>
where
    T: Ring + RingInvertible + Clone,
{
    pub fn truncate_upto_deg(&self, deg: usize) -> Self {
        if self.degree() >= deg {
            Polynomial::new(self.0[..deg].to_vec())
        } else {
            self.clone()
        }
    }

    pub fn inv_mod_x_pow(self, target: usize) -> Self {
        let mut t = target.next_power_of_two();
        t >>= 1;
        let mut j = 2;
        let g = RingInvertible::try_invert(
            self.0[0].clone(),
        ).expect("only monic polynomials, whose leading coefficient is a unit, is applicable to Newton's method of inversion");
        let mut g = Polynomial(vec![g]);
        let two = T::one() + T::one();
        while t > 0 {
            let e = if t >> 1 > 0 { j } else { target };
            let g_ = g.clone() * g.clone();
            let g_ = g_.truncate_upto_deg(e);
            let g_ = g_ * self.truncate_upto_deg(e);
            let g_ = g_.truncate_upto_deg(e);
            let g_ = g * two.clone() - g_;
            let g_ = g_.truncate_upto_deg(e);
            g = g_;
            t >>= 1;
            j <<= 1;
        }
        g
    }

    /// Dividing with another polynomial with remainders
    pub fn divide_by(mut self, mut b: Self) -> (Self, Self) {
        truncate_high_degree_zeros(&mut self.0);
        truncate_high_degree_zeros(&mut b.0);
        assert!(!b.is_zero());

        let n = self.degree();
        let m = b.degree();
        if n < m {
            return (Self::zero(), self);
        }
        let Polynomial(mut divisor) = b.clone();
        divisor.reverse();
        let divisor = Polynomial::new(divisor).inv_mod_x_pow(n - m + 1);

        let Polynomial(mut dividend) = self.clone();
        dividend.reverse();
        let Polynomial(mut quotient) = divisor * Polynomial::new(dividend);
        quotient.resize(n - m + 1, T::zero());
        quotient.reverse();
        let quotient = Polynomial::new(quotient);
        let remainder = self - quotient.clone() * b;
        (quotient, remainder)
    }
}

impl<T> Polynomial<T>
where
    T: Field + Clone,
{
    /// A formal derivative of the polynomial.
    /// The definition is $\dv{x}\sum_n a_n x^n = \sum_n n a_n x^{n-1}$.
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

    /// Test if this polynomial is a unit
    pub fn is_one(&self) -> bool {
        assert!(!self.0.is_empty());
        self.0[0] == T::one() && self.0.iter().skip(1).all(Zero::is_zero)
    }

    /// Evaluate this polynomial at a point.
    /// This transformation also makes a F-polynomial an endomorphism of any `U` that is an `F`-module.
    pub fn eval_at<U>(&self, x: U) -> Coord<U>
    where
        U: Clone + Mul<Output = U> + Mul<T, Output = U> + From<T> + Zero,
    {
        let mut y = U::zero();
        for a in self.0.iter().rev() {
            y = y * x.clone();
            y = y + U::from(a.clone());
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

impl<T> Polynomial<T>
where
    T: Mul<Output = T> + Zero + Sub<Output = T> + Clone,
{
    #[inline]
    fn tabular_aux(a: &mut [T], a_top: usize, b: &[T]) {
        let len_b = b.len();
        for i in (0..a_top).rev() {
            let coeff_a = replace(&mut a[i], T::zero());
            for j in 0..len_b {
                let a = &mut a[i + j];
                *a = a.clone() + coeff_a.clone() * b[j].clone();
            }
        }
    }

    #[inline]
    fn tabular_multiplication(Self(mut a): Self, Self(b): Self) -> Self {
        let len_a = a.len();
        let len_b = b.len();
        debug_assert!(len_a > 0 && len_b > 0);
        let len_r = len_a + len_b - 1;
        a.resize_with(len_r, T::zero);
        Self::tabular_aux(&mut a[..], len_a, &b);
        truncate_high_degree_zeros(&mut a);
        Self(a)
    }

    #[inline]
    fn k_truncate_higher_degree_zeros(a: &[T]) -> &[T] {
        for j in (0..a.len()).rev() {
            if !a[j].is_zero() {
                return &a[..j + 1];
            }
        }
        &a[..1]
    }

    #[inline]
    fn k_add_lower_and_higher_halves<'a>(a: &'a [T], m: usize, r: &'a mut [T]) -> &'a [T] {
        let (mut a, mut b) = a.split_at(m);
        if a.len() < b.len() {
            swap(&mut a, &mut b);
        }
        let len_a = a.len();
        let len_b = b.len();
        for i in 0..len_b {
            r[i] = a[i].clone() + b[i].clone();
        }
        for i in len_b..len_a {
            r[i] = a[i].clone();
        }
        Self::k_truncate_higher_degree_zeros(&r[..len_a])
    }

    fn karatsuba_rec<'a>(mut a: &'a [T], mut b: &'a [T], c: &'a mut [T], scratch: &'a mut [T]) {
        a = Self::k_truncate_higher_degree_zeros(a);
        b = Self::k_truncate_higher_degree_zeros(b);
        if a.len() < b.len() {
            swap(&mut a, &mut b);
        }
        let len_a = a.len();
        let len_b = b.len();
        if len_a < 16 || len_b < 16 {
            c[..len_a].clone_from_slice(a);
            c[len_a..].fill_with(T::zero);
            Self::tabular_aux(c, len_a, b);
            return;
        }
        debug_assert!(len_b > 0);
        let n = len_a;
        let m = n / 2;
        let m_ = n - m;
        c.fill_with(T::zero);
        if m < len_b {
            let len_r = m_;
            let len_s = m.max(len_b - m);
            let (rs, scratch) = scratch.split_at_mut(len_r + len_s);
            let (r, s) = rs.split_at_mut(len_r);
            let r = Self::k_add_lower_and_higher_halves(a, m, r);
            let s = Self::k_add_lower_and_higher_halves(b, m, s);
            Self::karatsuba_rec(r, s, &mut c[m..][..len_r + len_s - 1], scratch);
        } else {
            let len_r = m_;
            let len_s = len_b;
            let (r, scratch) = scratch.split_at_mut(len_r);
            let r = Self::k_add_lower_and_higher_halves(a, m, r);
            let s = b;
            Self::karatsuba_rec(r, s, &mut c[m..][..len_r + len_s - 1], scratch);
        }
        {
            let a = Self::k_truncate_higher_degree_zeros(&a[..m]);
            let b = Self::k_truncate_higher_degree_zeros(&b[..m.min(len_b)]);
            let len_q = a.len() + b.len() - 1;
            let (q, scratch) = scratch.split_at_mut(len_q);
            Self::karatsuba_rec(a, b, q, scratch);
            let q = Self::k_truncate_higher_degree_zeros(&q);
            for i in 0..q.len() {
                c[i] = c[i].clone() + q[i].clone();
            }
            let c = &mut c[m..];
            for i in 0..q.len() {
                c[i] = c[i].clone() - q[i].clone();
            }
        }
        if m < len_b {
            let a = &a[m..];
            let b = &b[m..];
            let len_p = a.len() + b.len() - 1;
            let (p, scratch) = scratch.split_at_mut(len_p);
            Self::karatsuba_rec(a, b, p, scratch);
            let p = Self::k_truncate_higher_degree_zeros(&p[..len_p]);
            let c = &mut c[m..];
            for i in 0..p.len() {
                c[i] = c[i].clone() - p[i].clone();
            }
            let c = &mut c[m..];
            for i in 0..p.len() {
                c[i] = c[i].clone() + p[i].clone();
            }
        }
    }

    fn karatsuba_multiplication(Self(mut a): Self, Self(mut b): Self) -> Self {
        let len_a = a.len();
        let len_b = b.len();
        let len_max = max(len_a, len_b);
        let len_r = len_a + len_b - 1;
        let mut r = vec![T::zero(); len_r];
        let mut scratch = vec![T::zero(); len_max.checked_mul(3).expect("input too large")];
        Self::karatsuba_rec(&mut a, &mut b, &mut r, &mut scratch);
        r.resize_with(len_r, T::zero);
        truncate_high_degree_zeros(&mut r);
        Self(r)
    }
}

impl<T> Mul for Polynomial<T>
where
    T: Mul<Output = T> + Zero + Sub<Output = T> + Clone,
{
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        if self.is_zero() || other.is_zero() {
            return Polynomial::zero();
        }
        if self.0.len() < 64 && other.0.len() < 64 {
            Self::tabular_multiplication(self, other)
        } else {
            Self::karatsuba_multiplication(self, other)
            // let (Self(mut left), Self(mut right)) = (self, other);
            // // karatsuba
            // let n = max(left.len(), right.len());
            // let m = max(n / 2, 1);
            // let left_high: Polynomial<T> = if left.len() < m {
            //     Polynomial::zero()
            // } else {
            //     Polynomial::from(left.split_off(m))
            // };
            // let right_high: Polynomial<T> = if right.len() < m {
            //     Polynomial::zero()
            // } else {
            //     Polynomial::from(right.split_off(m))
            // };

            // let left_low = Polynomial::from(left);
            // let right_low = Polynomial::from(right);
            // let high_pdt: Polynomial<_> = left_high.clone() * right_high.clone();
            // let low_pdt: Polynomial<_> = left_low.clone() * right_low.clone();
            // let mid: Polynomial<_> = (left_low + left_high) * (right_low + right_high)
            //     - high_pdt.clone()
            //     - low_pdt.clone();
            // let r_high = high_pdt.mul_pow_x(m * 2);
            // let r_mid = mid.mul_pow_x(m);
            // r_high + r_mid + low_pdt
        }
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
    T: Mul<Output = T> + Zero + Sub<Output = T> + Clone,
{
    fn mul_assign(&mut self, rhs: Self) {
        let lhs = std::mem::take(self);
        *self = lhs * rhs;
    }
}

impl<T> MulAssign<T> for Polynomial<T>
where
    T: MulAssign + Zero + Clone,
{
    fn mul_assign(&mut self, rhs: T) {
        for x in self.0.iter_mut() {
            *x *= rhs.clone();
        }
        truncate_high_degree_zeros(&mut self.0);
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

impl<T> DivAssign<T> for Polynomial<T>
where
    T: DivAssign + Zero + Clone,
{
    fn div_assign(&mut self, rhs: T) {
        for x in self.0.iter_mut() {
            *x /= rhs.clone()
        }
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
            let (fs, rs) = qs.divide_by(es.clone());
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

pub(crate) fn truncate_high_degree_zeros<T: Zero>(w: &mut Vec<T>) {
    // let mut zeroed = w.len();
    // while zeroed > 1 && w[zeroed - 1].is_zero() {
    //     zeroed -= 1;
    // }
    // w.truncate(zeroed);
    while w.len() > 1 && w[w.len() - 1].is_zero() {
        w.pop();
    }
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

/// Randomized search for a normal basis of `F`.
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
        if test_normal_basis(alpha.clone(), F::degree_extension::<Int>().assert_usize()) {
            return alpha;
        }
    }
}

fn assert_subfield<F: FiniteField>(deg_ext: usize) {
    let f_deg_ext: Int = F::degree_extension();
    let char: Int = F::characteristic();
    assert_eq!(
        f_deg_ext.clone() % Int::from(deg_ext),
        Int::zero(),
        "finite field GF({}^{}) is not a subfield of GF({}^{})",
        char,
        deg_ext,
        char,
        f_deg_ext,
    );
}

/// Test if `alpha` generates a normal basis of `F`.
pub fn test_normal_basis<F>(alpha: F, deg_ext: usize) -> bool
where
    F: FiniteField + Clone,
{
    assert_subfield::<F>(deg_ext);
    let g: Polynomial<F> = Polynomial::new(normal_basis_test_polynomial(deg_ext));
    let mut beta = alpha.clone();
    let mut p = vec![];
    let q: Int = F::characteristic();
    for _ in 0..deg_ext {
        p.push(beta.clone());
        beta = pow(beta, q.clone());
    }
    assert!(alpha == beta, "field element is not in the subfield");
    p.reverse();
    let p = Polynomial::new(p);
    let gcd = g.clone().gcd(p);
    !gcd.is_zero() && gcd.degree() == 0
}

/// Deterministically search for a generator of normal basis of `F`.
/// There is at least one normal basis for any finite field `F`.
pub fn search_normal_basis<F, G>(deg_ext: usize) -> F
where
    F: FiniteField + Clone + FinitelyGenerated<G>,
{
    assert_subfield::<F>(deg_ext);

    // the following division will produce zero remainder after the last
    // assertion
    let q: Int = F::characteristic();
    let f_deg_ext: Int = F::degree_extension();
    let a = nat_inj::<BigUint, Int>(pow(q.clone(), f_deg_ext) - Int::one());
    let b = nat_inj::<BigUint, Int>(pow(q.clone(), deg_ext) - Int::one());
    let exp = a / b;

    let gamma = pow(<F as FinitelyGenerated<G>>::generator(), exp);
    let mut alpha = gamma.clone();
    let mut c = nat_inj::<BigUint, Int>(pow(q, deg_ext) - Int::one());
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
/// Raise `x` to the power `exp`.
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

/// Compute a cyclotomic coset of size `n` in `F`.
pub fn compute_cyclotomic_cosets<F: FiniteField>(n: usize) -> Vec<Vec<usize>> {
    let mut q = Int::one();
    let mut p_pows = vec![];
    let char: Int = F::characteristic();
    for _ in 0..F::degree_extension() {
        p_pows.push(q.clone());
        q *= char.clone();
        let n: Int = int_inj(n);
        q %= n;
    }
    q += int_inj(n - 1);
    q %= int_inj::<Int, _>(n);
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
            let j = (p.clone() * int_inj::<Int, _>(i)) % int_inj::<Int, _>(n);
            if !s.insert(j.clone()) {
                break;
            }
            coset.push(j.assert_usize());
        }
        cosets.push(coset);
    }
    cosets
}

/// Uniformly sample an element in the ring `R`, with a precision of `limit` bits.
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
