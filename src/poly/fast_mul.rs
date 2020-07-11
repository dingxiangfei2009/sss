use super::*;

use std::marker::PhantomData;

use alga::general::Ring;
use num::traits::One;

use super::schoenhage::schoenhage_triadic_mul;
use crate::{
    field::{int_inj, GF2561D},
    ring::RingInvertible,
    truncate_high_degree_zeros,
};

/// Fast multiplication method for polynomials with data of type `T`
pub trait FastMulMethod<T> {
    fn try_fast_mul_poly(a: &[T], b: &[T]) -> Option<Vec<T>>;
}

#[derive(Debug)]
pub struct FastMul<T, M> {
    pub data: Polynomial<T>,
    _p: PhantomData<fn() -> M>,
}

impl<T: Clone, M> Clone for FastMul<T, M> {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            _p: PhantomData,
        }
    }
}

impl<T: PartialEq, M> PartialEq for FastMul<T, M> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<T, M> FastMul<T, M>
where
    T: Ring + RingInvertible + Clone,
    M: FastMulMethod<T>,
{
    pub fn new(data: Polynomial<T>) -> Self {
        Self {
            data,
            _p: PhantomData,
        }
    }
    pub fn truncate_upto_deg(&self, deg: usize) -> Self {
        if self.data.degree() >= deg {
            Self::new(Polynomial::new(self.data.0[..deg].iter().cloned()))
        } else {
            self.clone()
        }
    }

    pub fn inv_mod_x_pow(self, target: usize) -> Self {
        let mut t = target.next_power_of_two();
        t >>= 1;
        let mut j = 2;
        let g = RingInvertible::try_invert(
            self.data.0[0].clone(),
        ).expect("only monic polynomials, whose leading coefficient is a unit, is applicable to Newton's method of inversion");
        let mut g = Self::new(Polynomial(vec![g]));
        let two: T = int_inj(2);
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
}
impl<T, M> EuclideanDomain<usize> for FastMul<T, M>
where
    T: Ring + RingInvertible + Clone,
    M: FastMulMethod<T>,
{
    fn degree(&self) -> usize {
        self.data.degree()
    }

    fn div_with_rem(mut self, mut b: Self) -> (Self, Self) {
        truncate_high_degree_zeros(&mut self.data.0);
        truncate_high_degree_zeros(&mut b.data.0);
        assert!(!b.is_zero());

        let n = self.degree();
        let m = b.degree();
        if n < m {
            return (Self::zero(), self);
        }

        let Polynomial(mut divisor) = b.data.clone();
        divisor.reverse();
        let divisor = Self::new(Polynomial::new(divisor)).inv_mod_x_pow(n - m + 1);

        let Polynomial(mut dividend) = self.data.clone();
        dividend.reverse();
        let quotient = divisor * Self::new(Polynomial::new(dividend));
        let Polynomial(mut quotient) = quotient.data;
        quotient.resize(n - m + 1, T::zero());
        quotient.reverse();
        let quotient = Self::new(Polynomial::new(quotient));
        let remainder = self - quotient.clone() * b;
        (quotient, remainder)
    }
}

impl<F, M> PolynomialLike<F> for FastMul<F, M>
where
    F: Ring + RingInvertible + Clone,
    M: FastMulMethod<F>,
{
    fn from_polynomial(data: Polynomial<F>) -> Self {
        Self {
            data,
            _p: PhantomData,
        }
    }

    fn into_polynomial(self) -> Polynomial<F> {
        self.data
    }

    fn from_vec(p: Vec<F>) -> Self {
        Self {
            data: Polynomial::from(p),
            _p: PhantomData,
        }
    }
}

impl<F, M> Mul<F> for FastMul<F, M>
where
    F: Ring + Clone,
{
    type Output = Self;
    fn mul(mut self, other: F) -> Self {
        self.data = self.data * other;
        self
    }
}

impl<F, M> Mul for FastMul<F, M>
where
    F: Ring + Clone,
    M: FastMulMethod<F>,
{
    type Output = Self;
    fn mul(mut self, other: Self) -> Self {
        if let Some(data) = M::try_fast_mul_poly(&self.data.0, &other.data.0) {
            Self {
                data: Polynomial::new(data),
                _p: PhantomData,
            }
        } else {
            self.data = self.data * other.data;
            self
        }
    }
}

impl<F, M> Zero for FastMul<F, M>
where
    F: Ring + Clone,
{
    fn is_zero(&self) -> bool {
        self.data.is_zero()
    }
    fn zero() -> Self {
        Self {
            data: <_>::zero(),
            _p: PhantomData,
        }
    }
}

impl<F, M> Add for FastMul<F, M>
where
    F: Ring + Clone,
{
    type Output = Self;
    fn add(mut self, other: Self) -> Self {
        self.data = self.data + other.data;
        self
    }
}

impl<F, M> Sub for FastMul<F, M>
where
    F: Ring + Clone,
{
    type Output = Self;
    fn sub(mut self, other: Self) -> Self {
        self.data = self.data - other.data;
        self
    }
}

impl<F, M> Neg for FastMul<F, M>
where
    F: Ring + Clone,
{
    type Output = Self;
    fn neg(mut self) -> Self {
        self.data = -self.data;
        self
    }
}

impl<F, M> One for FastMul<F, M>
where
    F: Ring + Clone,
    M: FastMulMethod<F>,
{
    fn one() -> Self {
        Self {
            data: <_>::one(),
            _p: PhantomData,
        }
    }
}

/// Since 3 is a unit in `GF256`, we can moderately accelerate multiplication
/// of polynomials of this field
#[derive(Debug)]
pub struct GF2561DFastMulMethod;

impl FastMulMethod<GF2561D> for GF2561DFastMulMethod {
    fn try_fast_mul_poly(a: &[GF2561D], b: &[GF2561D]) -> Option<Vec<GF2561D>> {
        let Polynomial(r) = schoenhage_triadic_mul(
            Polynomial::new(a.iter().cloned()),
            Polynomial::new(b.iter().cloned()),
        );
        Some(r)
    }
}

pub type GF2561DFastMul = FastMul<GF2561D, GF2561DFastMulMethod>;

#[cfg(test)]
mod tests {
    use super::*;

    #[quickcheck]
    fn div_with_rem(a: Vec<GF2561D>, b: Vec<GF2561D>) {
        let a = GF2561DFastMul::from_vec(a);
        let b = GF2561DFastMul::from_vec(b);
        if b.is_zero() {
            return;
        }
        let (q, r) = a.clone().div_with_rem(b.clone());
        assert_eq!(q * b + r, a);
    }

    #[test]
    fn div_with_rem_fixed() {
        let a = Polynomial(vec![
            GF2561D(221),
            GF2561D(249),
            GF2561D(77),
            GF2561D(212),
            GF2561D(158),
            GF2561D(60),
            GF2561D(115),
            GF2561D(185),
            GF2561D(135),
            GF2561D(167),
            GF2561D(124),
            GF2561D(127),
            GF2561D(200),
            GF2561D(138),
            GF2561D(205),
            GF2561D(130),
            GF2561D(111),
            GF2561D(0),
            GF2561D(236),
        ]);
        let b = Polynomial(vec![
            GF2561D(49),
            GF2561D(103),
            GF2561D(246),
            GF2561D(7),
            GF2561D(180),
            GF2561D(209),
            GF2561D(126),
            GF2561D(54),
            GF2561D(75),
            GF2561D(153),
            GF2561D(210),
            GF2561D(197),
            GF2561D(131),
            GF2561D(237),
            GF2561D(57),
            GF2561D(229),
            GF2561D(181),
            GF2561D(117),
        ]);
        let c = GF2561DFastMul::from_polynomial(a.clone());
        let d = GF2561DFastMul::from_polynomial(b.clone());
        let (q, r) = c.div_with_rem(d.clone());
        let m = q.clone() * d;
        let m = m.into_polynomial();
        let q = q.into_polynomial();
        let r = r.into_polynomial();
        let (q_, r_) = a.div_with_rem(b.clone());
        let m_ = q_.clone() * b.clone();
        assert_eq!(m, m_, "q={:?}, b={:?}", q, b);
        assert_eq!(q, q_);
        assert_eq!(r, r_);
    }
}
