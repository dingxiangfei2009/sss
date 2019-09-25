#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

#[cfg(test)]
#[macro_use]
extern crate derive_more;

use std::{
    fmt::{Display, Formatter, Result as FmtResult},
    ops::{Add, BitAnd, Mul, ShrAssign, Sub},
};

use alga::general::{Additive, Field, Identity};
use num::{
    traits::{One, Pow, Zero},
    BigUint, ToPrimitive,
};

pub mod field;
pub mod lfsr;
pub mod linalg;

pub use crate::field::{ArbitraryElement, FiniteField, FinitelyGenerated, GF2561DG2};

pub trait EuclideanDomain<Degree: Ord>: Zero {
    /// Euclidean measure of this domain
    fn degree(&self) -> Degree;

    /// Division with respect to this euclidean measure, so that the remainder
    /// is either zero, or the degree of the remainder is less than the degree
    /// of the divisor
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

    fn extended_gcd(self, other: Self) -> (Self, Self, Self)
    where
        Self: Clone + One + Mul<Output = Self> + Sub<Output = Self>,
    {
        if self.degree() < other.degree() {
            let (r, s, d) = other.extended_gcd(self);
            return (s, r, d);
        }
        if other.is_zero() {
            return (Self::one(), Self::zero(), self);
        }
        let mut s = Self::one();
        let mut t = Self::zero();
        let mut u = Self::zero();
        let mut v = Self::one();
        let mut a = self;
        let mut b = other;
        loop {
            let (q, r) = a.div_with_rem(b.clone());
            let i = s - q.clone() * u.clone();
            let j = t - q.clone() * v.clone();
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

/// Univariate polynomial ring over a field `T`
#[derive(Clone, PartialEq, Debug)]
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
    pub fn new(mut coeffs: Vec<T>) -> Self {
        truncate_high_degree_zeros(&mut coeffs);
        Self(coeffs)
    }
}

impl<T: Add<Output = T> + Clone + Zero> Add for Polynomial<T> {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        use std::iter::repeat;

        let (Polynomial(left), Polynomial(right)) = (self, other);
        let max = std::cmp::max(left.len(), right.len());
        let left = left.into_iter().chain(repeat(T::zero()));
        let right = right.into_iter().chain(repeat(T::zero()));
        Polynomial::new(left.zip(right).map(|(a, b)| a + b).take(max).collect())
    }
}

impl<T> Zero for Polynomial<T>
where
    T: Zero + Add<Output = T> + Clone,
{
    fn zero() -> Self {
        Polynomial(vec![T::zero()])
    }

    fn is_zero(&self) -> bool {
        assert!(self.0.len() > 0);
        self.0.iter().all(|a| a.is_zero())
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
            if a[i + d_deg].is_zero() {
                quot.push(T::zero());
            } else {
                // this is safe because the divisor `d` is not zero
                let q = a[i + d_deg].clone() / d[d_deg].clone();
                for j in 0..=d_deg {
                    a[i + j] -= d[j].clone() * q.clone();
                }
                quot.push(q);
            }
        }
        quot.reverse();
        (Polynomial(quot), Polynomial::new(a))
    }

    pub fn formal_derivative(mut self) -> Self {
        if self.0.len() > 1 {
            let mut n = T::one();
            self.0.remove(0);
            for i in 0..self.0.len() {
                self.0[i] *= n.clone();
                n += T::one();
            }
            self
        } else {
            Self(vec![T::zero()])
        }
    }

    pub fn is_zero(&self) -> bool {
        assert!(self.0.len() > 0);
        self.0.iter().all(|c| c.is_zero())
    }

    pub fn is_one(&self) -> bool {
        assert!(self.0.len() > 0);
        self.0[0] == T::one() && self.0.iter().skip(1).all(Zero::is_zero)
    }

    pub fn into_coord(&self, x: T) -> Coord<T> {
        let mut y = <T as Identity<Additive>>::identity();
        for a in self.0.iter().rev() {
            y *= x.clone();
            y += a.clone();
        }
        Coord(x, y)
    }

    pub fn from_coords(c: &[Coord<T>]) -> Self
    where
        T: Eq,
    {
        let n = c.len();
        let mut w: Vec<_> = c.into_iter().map(|c| vec![c.1.clone()]).collect();
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

/// Given a lossy graph of a polynomial of degree `threshold - 1`, with at most
/// `(c.len() - threshold + 1) / 2` errors, recover the original polynomial
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
        .into_iter()
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
        .zip(c.into_iter())
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
                        let Coord(_, y) = es.into_coord(x.clone());
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
        if test_normal_basis(alpha.clone(), F::DEGREE_EXTENSION) {
            return alpha;
        }
    }
}

fn assert_subfield<F: FiniteField>(deg_ext: usize) {
    assert_eq!(
        F::DEGREE_EXTENSION % deg_ext,
        0,
        "finite field GF({}^{}) is not a subfield of GF({}^{})",
        F::CHARACTERISTIC,
        deg_ext,
        F::CHARACTERISTIC,
        F::DEGREE_EXTENSION
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
        beta = pow(beta, F::CHARACTERISTIC);
    }
    assert!(alpha == beta, "field element is not in the subfield");
    p.reverse();
    let p = Polynomial::new(p);
    let gcd = g.clone().gcd(p);
    !gcd.is_zero() && gcd.0.len() == 1
}

pub fn search_normal_basis<F, G>(deg_ext: usize) -> F
where
    F: FiniteField + Clone + FinitelyGenerated<G>,
{
    assert_subfield::<F>(deg_ext);

    // the following division will produce zero remainder after the last
    // assertion
    let exp: BigUint = (BigUint::from(F::CHARACTERISTIC).pow(F::DEGREE_EXTENSION) - 1u8)
        / (BigUint::from(F::CHARACTERISTIC).pow(deg_ext) - 1u8);

    let gamma = pow(<F as FinitelyGenerated<G>>::GENERATOR, exp);
    let mut alpha = gamma.clone();
    let mut c = BigUint::from(F::CHARACTERISTIC).pow(deg_ext) - 1u8;
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
    E: BitAnd,
    E: for<'a> BitAnd<&'a E, Output = E>,
    E: ShrAssign<usize> + From<u8> + Zero + Clone,
    F: Field + Clone,
{
    let mut p = x;
    x = F::one();
    let bit = E::from(1);
    while !exp.is_zero() {
        if !(exp.clone() & &bit).is_zero() {
            x *= p.clone();
        }
        p *= p.clone();
        exp >>= 1;
    }
    x
}

pub fn compute_cyclotomic_cosets<F: FiniteField>(n: usize) -> Vec<Vec<usize>> {
    let mut q = BigUint::from(1u8);
    let mut p_pows = vec![];
    for _ in 0..F::DEGREE_EXTENSION {
        p_pows.push(q.clone());
        q *= F::CHARACTERISTIC;
        q %= n;
    }
    q += n - 1;
    q %= n;
    assert!(q.is_zero(), "{} is not zero", q);

    // TODO: maybe use ufds here?
    let mut cosets = vec![vec![0]];
    let mut s = std::collections::BTreeSet::new();
    for i in 1..n {
        let mut coset = vec![];
        for p in &p_pows {
            let j = (p.clone() * i) % n;
            if !s.insert(j.clone()) {
                break;
            }
            coset.push(j.to_usize().expect("size should fit"));
        }
        if !coset.is_empty() {
            cosets.push(coset);
        }
    }
    cosets
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::field::{GF2561D, GF2561D_NORMAL_BASIS};

    #[quickcheck]
    fn it_works(mut u: Vec<GF2561D>) {
        if u.len() == 0 {
            u = vec![GF2561D(0)];
        }
        if u.len() > 254 {
            u = u.drain(0..254).collect();
        }
        truncate_high_degree_zeros(&mut u);
        let threshold = u.len();
        let p = Polynomial(u.clone());

        let q: Vec<_> = (1u8..=u.len() as u8 + 1)
            .map(GF2561D)
            .map(|x| p.into_coord(x))
            .take(threshold)
            .collect();
        let r = Polynomial::from_coords(q.as_slice());
        assert_eq!(u.as_slice(), r.0.as_slice());

        let q: Vec<_> = (1u8..=u.len() as u8 + 1)
            .map(GF2561D)
            .map(|x| p.into_coord(x))
            .collect();
        let r = Polynomial::from_coords(q.as_slice());
        assert_eq!(u.as_slice(), r.0.as_slice());
    }

    use alga::general::{AbstractMagma, Additive, Identity, Multiplicative, TwoSidedInverse};
    use alga_derive::Alga;
    use approx::{AbsDiffEq, RelativeEq};
    use num::{rational::BigRational, BigInt, One};
    use quickcheck::{Arbitrary, Gen};
    use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

    #[derive(Copy, Clone, Debug, Display, Alga, PartialEq, Eq)]
    #[alga_traits(Field(Additive, Multiplicative))]
    pub struct F7(u8);

    impl F7 {
        pub const fn new(x: u8) -> Self {
            F7((x % 7 + 7) % 7)
        }
    }

    impl quickcheck::Arbitrary for F7 {
        fn arbitrary<G: quickcheck::Gen>(g: &mut G) -> Self {
            Self::new(u8::arbitrary(g))
        }
    }

    impl Zero for F7 {
        fn zero() -> Self {
            F7(0)
        }

        fn is_zero(&self) -> bool {
            self.0 == 0
        }
    }

    impl One for F7 {
        fn one() -> Self {
            F7(1)
        }
    }

    impl Identity<Additive> for F7 {
        fn identity() -> Self {
            Self::zero()
        }
    }

    impl Identity<Multiplicative> for F7 {
        fn identity() -> Self {
            Self::one()
        }
    }

    impl AbstractMagma<Additive> for F7 {
        fn operate(&self, right: &Self) -> Self {
            F7((self.0 + right.0) % 7)
        }
    }

    impl AbstractMagma<Multiplicative> for F7 {
        fn operate(&self, right: &Self) -> Self {
            F7(self.0 * right.0 % 7)
        }
    }

    impl TwoSidedInverse<Additive> for F7 {
        fn two_sided_inverse(&self) -> Self {
            F7((7 - self.0) % 7)
        }
    }

    impl TwoSidedInverse<Multiplicative> for F7 {
        fn two_sided_inverse(&self) -> Self {
            let F7(mut x) = self;
            assert!(x != 0);
            let mut y = x;
            x *= x;
            x %= 7;
            x *= x;
            x %= 7;
            y *= x;
            y %= 7;
            F7(y)
        }
    }

    impl Mul for F7 {
        type Output = Self;
        fn mul(self, other: Self) -> Self {
            <Self as AbstractMagma<Multiplicative>>::operate(&self, &other)
        }
    }

    impl MulAssign for F7 {
        fn mul_assign(&mut self, other: Self) {
            *self = *self * other;
        }
    }

    impl Div for F7 {
        type Output = Self;
        fn div(self, other: Self) -> Self {
            <Self as AbstractMagma<Multiplicative>>::operate(
                &self,
                &<Self as TwoSidedInverse<Multiplicative>>::two_sided_inverse(&other),
            )
        }
    }

    impl DivAssign for F7 {
        fn div_assign(&mut self, other: Self) {
            *self = *self / other;
        }
    }

    impl Add for F7 {
        type Output = Self;
        fn add(self, other: Self) -> Self {
            <Self as AbstractMagma<Additive>>::operate(&self, &other)
        }
    }

    impl AddAssign for F7 {
        fn add_assign(&mut self, other: Self) {
            *self = *self + other;
        }
    }

    impl Sub for F7 {
        type Output = Self;
        fn sub(self, other: Self) -> Self {
            <Self as AbstractMagma<Additive>>::operate(
                &self,
                &<Self as TwoSidedInverse<Additive>>::two_sided_inverse(&other),
            )
        }
    }

    impl SubAssign for F7 {
        fn sub_assign(&mut self, other: Self) {
            *self = *self - other;
        }
    }

    impl Neg for F7 {
        type Output = Self;
        fn neg(self) -> Self {
            <Self as TwoSidedInverse<Additive>>::two_sided_inverse(&self)
        }
    }

    #[derive(
        Alga,
        PartialEq,
        Eq,
        Clone,
        Debug,
        Add,
        AddAssign,
        Mul,
        MulAssign,
        Div,
        DivAssign,
        Sub,
        SubAssign,
        Neg,
        Display,
    )]
    #[alga_traits(Field(Additive, Multiplicative))]
    #[alga_quickcheck]
    pub struct Frac(pub BigRational);

    impl Arbitrary for Frac {
        fn arbitrary<G: Gen>(g: &mut G) -> Self {
            Frac(BigRational::new(
                u64::arbitrary(g).into(),
                std::cmp::max(1, u64::arbitrary(g)).into(),
            ))
        }
    }

    impl AbsDiffEq for Frac {
        type Epsilon = Self;
        fn default_epsilon() -> Self::Epsilon {
            Frac(BigRational::zero())
        }
        fn abs_diff_eq(&self, other: &Self, _: Self::Epsilon) -> bool {
            self == other
        }
    }

    impl RelativeEq for Frac {
        fn default_max_relative() -> Self::Epsilon {
            Frac(BigRational::zero())
        }
        fn relative_eq(&self, other: &Self, _: Self::Epsilon, _: Self::Epsilon) -> bool {
            self == other
        }
    }

    impl Zero for Frac {
        fn zero() -> Self {
            <Self as Identity<Additive>>::identity()
        }

        fn is_zero(&self) -> bool {
            self == &Self::zero()
        }
    }

    impl One for Frac {
        fn one() -> Self {
            <Self as Identity<Multiplicative>>::identity()
        }
    }

    impl std::ops::Mul<Frac> for BigRational {
        type Output = BigRational;
        fn mul(self, other: Frac) -> Self::Output {
            self * other.0
        }
    }

    impl std::ops::Div<Frac> for BigRational {
        type Output = BigRational;
        fn div(self, other: Frac) -> Self::Output {
            self / other.0
        }
    }

    impl std::ops::DivAssign<Frac> for BigRational {
        fn div_assign(&mut self, other: Frac) {
            *self = self.clone() / other.0
        }
    }

    impl std::ops::MulAssign<Frac> for BigRational {
        fn mul_assign(&mut self, other: Frac) {
            *self = self.clone() * other.0
        }
    }

    impl Identity<Additive> for Frac {
        fn identity() -> Self {
            Frac(BigRational::zero())
        }
    }

    impl Identity<Multiplicative> for Frac {
        fn identity() -> Self {
            Frac(BigRational::one())
        }
    }

    impl AbstractMagma<Additive> for Frac {
        fn operate(&self, right: &Self) -> Self {
            Frac(self.0.clone() + right.0.clone())
        }
    }

    impl AbstractMagma<Multiplicative> for Frac {
        fn operate(&self, right: &Self) -> Self {
            Frac(self.0.clone() * right.0.clone())
        }
    }

    impl TwoSidedInverse<Additive> for Frac {
        fn two_sided_inverse(&self) -> Self {
            Frac(BigRational::zero() - self.0.clone())
        }
    }

    impl TwoSidedInverse<Multiplicative> for Frac {
        fn two_sided_inverse(&self) -> Self {
            Frac(BigRational::one() / self.0.clone())
        }
    }
    #[test]
    fn error_correction1() {
        let p = Polynomial(vec![1, 2, 3].into_iter().map(F7).collect());
        let c: Vec<_> = vec![(0, 1), (1, 5), (2, 3), (3, 6), (4, 3), (5, 2), (6, 2)]
            .into_iter()
            .map(|(x, y)| Coord(F7(x), F7(y)))
            .collect();
        assert_eq!((p, vec![1, 4]), error_correct(&c, 3).unwrap());
    }

    #[test]
    fn error_correction2() {
        let p = Polynomial(vec![5, 0, 4, 1].into_iter().map(F7).collect());
        let c: Vec<_> = vec![(1, 3), (2, 1), (3, 5), (4, 0), (5, 6), (6, 1)]
            .into_iter()
            .map(|(x, y)| Coord(F7(x), F7(y)))
            .collect();
        assert_eq!((p, vec![]), error_correct(&c, 4).unwrap());
    }

    #[test]
    fn error_correction3() {
        let p = Polynomial(vec![0, 1, 2, 3, 4].into_iter().map(F7).collect());
        let mut c: Vec<_> = vec![0, 1, 2, 3, 4, 5, 6]
            .into_iter()
            .map(F7)
            .map(|x| p.into_coord(x))
            .collect();
        c[0].1 += F7(1);
        assert_eq!((p, vec![0]), error_correct(&c, 5).unwrap());
    }

    #[quickcheck]
    fn error_correction_quickcheck(mut u: Vec<GF2561D>, mut replaces: Vec<(usize, GF2561D)>) {
        if u.len() == 0 {
            u = vec![Zero::zero()];
        }
        if u.len() > 253 {
            u = u.drain(0..253).collect();
        }
        truncate_high_degree_zeros(&mut u);
        let threshold = u.len();
        let p = Polynomial(u.clone());
        replaces = replaces
            .drain(0..std::cmp::min((255 - threshold) / 2, replaces.len()))
            .filter(|(_, s)| !s.is_zero())
            .collect();
        let mut q: Vec<_> = (1..=255)
            .map(GF2561D)
            .map(|x| p.into_coord(x))
            .take(threshold + 2 * replaces.len())
            .collect();
        for (replace, _) in replaces.iter_mut() {
            *replace %= q.len();
        }
        replaces.sort_by_key(|(r, _)| *r);
        replaces.dedup_by_key(|(r, _)| *r);
        for (replace, with) in replaces.iter() {
            q[*replace].1 += *with;
        }
        eprintln!("correcting {:?}", q);
        let (p_, e) = error_correct(&q, threshold).unwrap();
        assert_eq!(p, p_, "wrong decoding");
        assert_eq!(
            replaces.iter().map(|(r, _)| *r).collect::<Vec<_>>(),
            e,
            "wrong syndromes"
        );
    }

    #[test]
    fn gf2561d_normal_basis() {
        let mut rng = rand::thread_rng();
        let base: GF2561D = find_normal_basis(&mut rng);
        eprintln!("found a basis for GF2561D: {}", base);
    }

    #[test]
    fn gf2561d_gamma_is_normal_basis() {
        let gamma = GF2561D_NORMAL_BASIS;
        let mut g = vec![GF2561D::zero(); GF2561D::DEGREE_EXTENSION + 1];
        g[GF2561D::DEGREE_EXTENSION] = GF2561D::one() / GF2561D(0b00010011);
        g[0] -= GF2561D::one() / GF2561D(0b00010011);
        let g = Polynomial::new(g);

        let mut beta = gamma.clone();
        let mut p = vec![];
        for _ in 0..GF2561D::DEGREE_EXTENSION {
            p.push(beta.clone());
            beta = pow(beta, GF2561D::CHARACTERISTIC);
        }
        p.reverse();
        let p = Polynomial::new(p);
        eprintln!("betas={}", p);
        let d = g.gcd(p.clone());
        eprintln!("gcd={}", d);
        assert!(!d.is_zero() && d.0.len() == 1, "gcd {} should be 1", d);

        // explicitly test linear independence
        let Polynomial(p) = p;
        let mut h = std::collections::HashSet::new();
        for i in 0..=255u8 {
            let mut x = GF2561D::zero();
            let mut m = 1;
            for j in 0..8 {
                if m & i > 0 {
                    x += p[j];
                }
                m <<= 1;
            }
            assert!(h.insert(x));
        }
    }

    impl EuclideanDomain<u32> for u32 {
        fn degree(&self) -> Self {
            *self
        }

        fn div_with_rem(mut self, other: Self) -> (Self, Self) {
            let q = self / other;
            self -= q * other;
            (q, self)
        }
    }

    #[quickcheck]
    fn gcd(a: u32, b: u32) {
        let d = u32::gcd(a, b);
        eprintln!("({}, {})={}", a, b, d);
        assert_eq!(a % d, 0);
        assert_eq!(b % d, 0);
    }

    #[test]
    fn polynomial_gcd() {
        let p = Polynomial::new(
            vec![-1, -1, 2]
                .into_iter()
                .map(BigInt::from)
                .map(BigRational::from_integer)
                .map(Frac)
                .collect(),
        );
        let q = Polynomial::new(
            vec![3, 8, 4]
                .into_iter()
                .map(BigInt::from)
                .map(BigRational::from_integer)
                .map(Frac)
                .collect(),
        );
        let d = q.clone().gcd(p.clone());
        let (_, r) = p.div_with_rem(d.clone());
        assert!(r.is_zero());
        let (_, r) = q.div_with_rem(d);
        assert!(r.is_zero());
    }

    fn gf2561d_normal_basis_conversion(gamma: GF2561D, deg_ext: usize) {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};

        assert_subfield::<GF2561D>(deg_ext);
        let basis: Vec<_> = {
            let mut gamma = gamma;
            let mut basis = vec![];
            for _ in 0..deg_ext {
                basis.push(gamma);
                gamma = pow(gamma, GF2561D::CHARACTERISTIC);
            }
            basis
        };
        let basis: Vec<_> = basis
            .into_iter()
            .flat_map(|GF2561D(beta)| (0..deg_ext).map(move |i| GF2561D((beta >> i) & 1u8)))
            .chain(vec![GF2561D::zero(); deg_ext])
            .collect();
        let basis = ndarray::ArrayView::from_shape((deg_ext + 1, deg_ext), &basis)
            .expect("shape should be correct");
        let inv: Vec<_> = (0..deg_ext)
            .into_par_iter()
            .flat_map(|i| {
                let mut basis = basis.t().to_owned();
                basis[[i, deg_ext]] = GF2561D::one();
                crate::linalg::solve(basis)
                    .expect("should have a solution")
                    .to_vec()
            })
            .collect();
        let inv = ndarray::ArrayView::from_shape((deg_ext, deg_ext), &inv)
            .expect("shape should be correct")
            .t()
            .to_owned();
        println!("{}", inv);
    }

    #[test]
    fn gf2561d_gamma_normal_basis_has_conversion() {
        gf2561d_normal_basis_conversion(
            crate::field::GF2561D_NORMAL_BASIS,
            GF2561D::DEGREE_EXTENSION,
        );
    }

    #[test]
    fn gf2561d_15th_cyclotomic_coset() {
        assert_eq!(
            vec![
                vec![0],
                vec![1, 2, 4, 8],
                vec![3, 6, 12, 9],
                vec![5, 10],
                vec![7, 14, 13, 11],
            ],
            compute_cyclotomic_cosets::<GF2561D>(15)
        );
    }

    #[test]
    fn gf2561d_subfield_16_normal_basis() {
        let gamma = search_normal_basis::<GF2561D, GF2561DG2>(4);
        eprintln!("gamma={}", gamma);
        assert_eq!(pow(gamma.clone(), 16), gamma);
        assert_eq!(gamma, GF2561D(0b00001010));
        gf2561d_normal_basis_conversion(gamma, 4);
    }

    #[test]
    fn gf2561d_subfield_4_normal_basis() {
        let gamma = search_normal_basis::<GF2561D, GF2561DG2>(2);
        eprintln!("gamma={}", gamma);
        assert_eq!(pow(gamma.clone(), 4), gamma);
        assert_eq!(gamma, GF2561D(0b11010110));
        gf2561d_normal_basis_conversion(gamma, 2);
    }
}
