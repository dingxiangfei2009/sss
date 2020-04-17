//! Unsafe *arbitrary* **prime** field definition
//!
//! Note that arithmetic properties needs to be tested by users,
//! or otherwise operations may panic

use std::{
    cmp::PartialEq,
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use alga::general::{
    AbstractField, AbstractGroup, AbstractGroupAbelian, AbstractLoop, AbstractMagma,
    AbstractMonoid, AbstractQuasigroup, AbstractRing, AbstractRingCommutative, AbstractSemigroup,
    Additive, Identity, Multiplicative, TwoSidedInverse,
};
use num::{BigInt, One, Zero};
use rand::RngCore;

use crate::EuclideanDomain;

enum Prime<'a, V> {
    Unknown,
    Borrowed(&'a V),
}

impl<'a, V> Clone for Prime<'a, V> {
    fn clone(&self) -> Self {
        use Prime::*;
        match self {
            Unknown => Unknown,
            Borrowed(p) => Borrowed(p),
        }
    }
}

impl<'a, V> Copy for Prime<'a, V> {}

impl<'a, V> Prime<'a, V> {
    fn value(&self) -> &V {
        use Prime::*;
        match self {
            Unknown => panic!("unknown prime"),
            Borrowed(p) => p,
        }
    }
    fn merge(&self, other: &Self) -> Self
    where
        V: PartialEq + Clone,
    {
        use Prime::*;
        match (self, other) {
            (Unknown, Unknown) => Unknown,
            (Borrowed(a), Borrowed(b)) if a == b => Borrowed(a),
            (p, Unknown) => p.clone(),
            (Unknown, p) => p.clone(),
            _ => panic!("unmatched primes"),
        }
    }
}

impl<'a, 'b, V> PartialEq<Prime<'b, V>> for Prime<'a, V>
where
    V: PartialEq,
{
    fn eq(&self, other: &Prime<'b, V>) -> bool {
        use Prime::*;
        match (self, other) {
            (Unknown, _) | (_, Unknown) => true,
            (Borrowed(a), Borrowed(b)) => a == b,
        }
    }
}

impl<'a, V> Hash for Prime<'a, V>
where
    V: Hash,
{
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.value().hash(h)
    }
}

pub struct UnsafeFp<'a, V, D> {
    value: V,
    prime: Prime<'a, V>,
    _p: PhantomData<fn() -> D>,
}

impl<'a, V, D> UnsafeFp<'a, V, D>
where
    V: Clone + EuclideanDomain<D>,
    D: Ord,
{
    pub fn new(value: V, prime: &'a V) -> Self {
        Self::new_intern(value, Prime::Borrowed(prime))
    }

    fn new_intern(value: V, prime: Prime<'a, V>) -> Self {
        use Prime::*;
        Self {
            value: match &prime {
                Unknown => value,
                Borrowed(p) => (value.rem(V::clone(p)) + V::clone(p)).rem(V::clone(p)),
            },
            prime,
            _p: PhantomData,
        }
    }

    pub fn into_inner(self) -> V {
        self.value
    }

    pub fn inner(&self) -> &V {
        &self.value
    }

    pub fn uniform_sample<R>(rng: &mut R, prime: &'a V) -> Self
    where
        R: RngCore,
        V: Ord + Clone + Div<Output = V>,
        D: Ord,
    {
        Self::new(crate::uniform_sample(rng, prime.clone()), prime)
    }

    pub fn with_prime(self, prime: &'a V) -> Self {
        Self::new(self.into_inner(), prime)
    }
}

impl<'a, V, D> PartialEq for UnsafeFp<'a, V, D>
where
    V: Clone + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.prime == other.prime && self.value == other.value
    }
}

impl<'a, V, D> Hash for UnsafeFp<'a, V, D>
where
    V: Hash,
{
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.value.hash(h);
        self.prime.hash(h);
    }
}

impl<'a, V, D> Clone for UnsafeFp<'a, V, D>
where
    V: Clone,
{
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            prime: self.prime.clone(),
            _p: PhantomData,
        }
    }
}

impl<'a, V, D> Debug for UnsafeFp<'a, V, D>
where
    V: Clone + Debug,
{
    fn fmt(&self, formatter: &mut Formatter) -> FmtResult {
        self.value.fmt(formatter)
    }
}

impl<'a, V, D> Display for UnsafeFp<'a, V, D>
where
    V: Clone + Display,
{
    fn fmt(&self, formatter: &mut Formatter) -> FmtResult {
        self.value.fmt(formatter)
    }
}

impl<'a, V, D> AddAssign for UnsafeFp<'a, V, D>
where
    Self: Add<Output = Self> + Clone,
{
    fn add_assign(&mut self, other: Self) {
        *self = self.clone() + other;
    }
}

impl<'a, V, D> Add for UnsafeFp<'a, V, D>
where
    V: PartialEq + Clone + EuclideanDomain<D>,
    D: Ord,
{
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        let prime = self.prime.merge(&other.prime);
        Self::new_intern(self.into_inner() + other.into_inner(), prime)
    }
}

impl<'a, V, D> Zero for UnsafeFp<'a, V, D>
where
    V: PartialEq + Clone + EuclideanDomain<D>,
    D: Ord,
{
    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    fn zero() -> Self {
        Self {
            value: V::zero(),
            prime: Prime::Unknown,
            _p: PhantomData,
        }
    }
}

impl<'a, V, D> Identity<Additive> for UnsafeFp<'a, V, D>
where
    Self: Zero,
{
    fn identity() -> Self {
        Self::zero()
    }
}

impl<'a, V, D> TwoSidedInverse<Additive> for UnsafeFp<'a, V, D>
where
    V: PartialEq + Clone + EuclideanDomain<D>,
    D: Ord,
{
    fn two_sided_inverse(&self) -> Self {
        Self::new_intern(V::zero() - self.inner().clone(), self.prime.clone())
    }
}

impl<'a, V, D> Neg for UnsafeFp<'a, V, D>
where
    Self: TwoSidedInverse<Additive>,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        TwoSidedInverse::<Additive>::two_sided_inverse(&self)
    }
}

impl<'a, V, D> Sub for UnsafeFp<'a, V, D>
where
    Self: Neg<Output = Self> + Add<Output = Self>,
{
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        self + -other
    }
}

impl<'a, V, D> SubAssign for UnsafeFp<'a, V, D>
where
    Self: Neg<Output = Self> + Add<Output = Self> + Clone,
{
    fn sub_assign(&mut self, other: Self) {
        *self = self.clone() - other;
    }
}

impl<'a, V, D> One for UnsafeFp<'a, V, D>
where
    V: Clone + PartialEq + One + EuclideanDomain<D>,
    D: Ord,
{
    fn one() -> Self {
        Self::new_intern(V::one(), Prime::Unknown)
    }
}

impl<'a, V, D> Identity<Multiplicative> for UnsafeFp<'a, V, D>
where
    Self: One,
{
    fn identity() -> Self {
        Self::one()
    }
}

impl<'a, V, D> Mul for UnsafeFp<'a, V, D>
where
    V: EuclideanDomain<D> + Clone + PartialEq,
    D: Ord,
{
    type Output = Self;
    fn mul(self, other: Self) -> Self::Output {
        let prime = self.prime.merge(&other.prime);
        Self::new_intern(self.into_inner() * other.into_inner(), prime)
    }
}

impl<'a, V, D> MulAssign for UnsafeFp<'a, V, D>
where
    Self: Mul<Output = Self> + Clone,
{
    fn mul_assign(&mut self, other: Self) {
        *self = self.clone() * other;
    }
}

impl<'a, V, D> Div for UnsafeFp<'a, V, D>
where
    Self: TwoSidedInverse<Multiplicative> + Mul<Output = Self>,
{
    type Output = Self;

    #[allow(clippy::suspicious_arithmetic_impl)] // REASON: division by multiplying multiplicative inverse
    fn div(self, other: Self) -> Self::Output {
        let inv = <_ as TwoSidedInverse<Multiplicative>>::two_sided_inverse(&other);
        self * inv
    }
}

impl<'a, V, D> DivAssign for UnsafeFp<'a, V, D>
where
    Self: Div<Output = Self> + Clone,
{
    fn div_assign(&mut self, other: Self) {
        *self = self.clone() / other;
    }
}

impl<'a, V, D> TwoSidedInverse<Multiplicative> for UnsafeFp<'a, V, D>
where
    V: Clone + EuclideanDomain<D> + PartialEq,
    D: Ord,
{
    fn two_sided_inverse(&self) -> Self {
        let Self { value, .. } = self;
        assert!(!self.value.is_zero());
        let (_, value, d) = self
            .prime
            .value()
            .clone()
            .extended_gcd::<V, V>(value.clone());
        assert!(d.is_one(), "modulus is not a prime number");
        Self {
            value,
            prime: self.prime,
            _p: PhantomData,
        }
    }
}

impl<'a, V, D> AbstractMagma<Additive> for UnsafeFp<'a, V, D>
where
    Self: Add<Output = Self> + Clone,
{
    #[inline]
    fn operate(&self, right: &Self) -> Self {
        self.clone() + right.clone()
    }
}

impl<'a, V, D> AbstractMagma<Multiplicative> for UnsafeFp<'a, V, D>
where
    Self: Mul<Output = Self> + Clone,
{
    #[inline]
    fn operate(&self, right: &Self) -> Self {
        self.clone() * right.clone()
    }
}

impl<'a, V, D> AbstractField for UnsafeFp<'a, V, D>
where
    V: Clone + EuclideanDomain<D> + PartialEq,
    D: Ord,
{
}

impl<'a, V, D> AbstractRingCommutative for UnsafeFp<'a, V, D>
where
    V: Clone + EuclideanDomain<D> + PartialEq,
    D: Ord,
{
}

impl<'a, V, D> AbstractRing for UnsafeFp<'a, V, D>
where
    V: Clone + EuclideanDomain<D> + PartialEq,
    D: Ord,
{
}

impl<'a, V, D> AbstractGroupAbelian<Additive> for UnsafeFp<'a, V, D>
where
    V: Clone + EuclideanDomain<D> + PartialEq,
    D: Ord,
{
}

impl<'a, V, D> AbstractGroupAbelian<Multiplicative> for UnsafeFp<'a, V, D>
where
    V: Clone + EuclideanDomain<D> + PartialEq,
    D: Ord,
{
}

impl<'a, V, D> AbstractGroup<Additive> for UnsafeFp<'a, V, D>
where
    V: Clone + EuclideanDomain<D> + PartialEq,
    D: Ord,
{
}

impl<'a, V, D> AbstractGroup<Multiplicative> for UnsafeFp<'a, V, D>
where
    V: Clone + EuclideanDomain<D> + PartialEq,
    D: Ord,
{
}

impl<'a, V, D> AbstractMonoid<Additive> for UnsafeFp<'a, V, D>
where
    V: Clone + EuclideanDomain<D> + PartialEq,
    D: Ord,
{
}

impl<'a, V, D> AbstractMonoid<Multiplicative> for UnsafeFp<'a, V, D>
where
    V: Clone + EuclideanDomain<D> + PartialEq,
    D: Ord,
{
}

impl<'a, V, D> AbstractLoop<Additive> for UnsafeFp<'a, V, D>
where
    V: Clone + EuclideanDomain<D> + PartialEq,
    D: Ord,
{
}

impl<'a, V, D> AbstractLoop<Multiplicative> for UnsafeFp<'a, V, D>
where
    V: Clone + EuclideanDomain<D> + PartialEq,
    D: Ord,
{
}

impl<'a, V, D> AbstractSemigroup<Additive> for UnsafeFp<'a, V, D>
where
    V: Clone + EuclideanDomain<D> + PartialEq,
    D: Ord,
{
}

impl<'a, V, D> AbstractSemigroup<Multiplicative> for UnsafeFp<'a, V, D>
where
    V: Clone + EuclideanDomain<D> + PartialEq,
    D: Ord,
{
}

impl<'a, V, D> AbstractQuasigroup<Additive> for UnsafeFp<'a, V, D>
where
    V: Clone + EuclideanDomain<D> + PartialEq,
    D: Ord,
{
}

impl<'a, V, D> AbstractQuasigroup<Multiplicative> for UnsafeFp<'a, V, D>
where
    V: Clone + EuclideanDomain<D> + PartialEq,
    D: Ord,
{
}

pub type BigIntUnsafeFp<'a> = UnsafeFp<'a, BigInt, BigInt>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let prime = BigInt::from(7);
        let a = BigIntUnsafeFp::new(BigInt::from(3), &prime);
        let b = BigIntUnsafeFp::new(BigInt::from(4), &prime);
        assert_eq!(a.clone() + b.clone(), BigIntUnsafeFp::zero());
        assert_eq!(a / b, BigIntUnsafeFp::new(BigInt::from(6), &prime))
    }
}
