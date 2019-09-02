#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

#[cfg(test)]
#[macro_use]
extern crate derive_more;

use alga::general::{Additive, Field, Identity};
use num::traits::Zero;

pub mod field;
pub mod lfsr;
pub mod linalg;

/// Univariate polynomial ring over a field `T`
#[derive(Clone, PartialEq, Debug)]
pub struct Polynomial<T>(pub Vec<T>);
#[derive(Clone, PartialEq, Debug)]
pub struct Coord<T>(pub T, pub T);

impl<T> Polynomial<T>
where
    T: Field,
{
    pub fn new(mut coeffs: Vec<T>) -> Self {
        truncate_high_degree_zeros(&mut coeffs);
        Self(coeffs)
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
        if self.0.len() < divisor.0.len() {
            return (Polynomial(vec![T::zero()]), self);
        }
        let divisor = divisor;
        let mut quot = vec![];
        let d_deg = divisor.0.len() - 1;
        assert!(self.0.len() >= divisor.0.len());
        for i in (0..self.0.len() - d_deg).rev() {
            if self.0[i + d_deg].is_zero() {
                quot.push(T::zero());
            } else {
                // this is safe because `divisor` is not zero
                let q = self.0[i + d_deg].clone() / divisor.0[d_deg].clone();
                for j in 0..=d_deg {
                    self.0[i + j] -= divisor.0[j].clone() * q.clone();
                }
                quot.push(q);
            }
        }
        quot.reverse();
        truncate_high_degree_zeros(&mut self.0);
        (Polynomial(quot), self)
    }

    pub fn is_zero(&self) -> bool {
        assert!(self.0.len() > 0);
        self.0.iter().all(|c| c.is_zero())
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
    T: Field + Clone + Eq + Send + Sync + std::fmt::Debug,
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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::field::GF2561D;

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
    use num::One;
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
}
