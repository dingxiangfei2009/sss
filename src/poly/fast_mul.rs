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
