use std::{
    fmt::Debug,
    hash::{Hash, Hasher},
    ops::{Div, DivAssign, Mul, Rem, Shr},
};

use num::{One, Zero};
use rug::{integer::Order, Integer};
use serde::{Deserialize, Serialize};

use crate::EuclideanDomain;

#[derive(
    Add,
    AddAssign,
    BitAnd,
    BitOr,
    Clone,
    Debug,
    Div,
    DivAssign,
    Eq,
    Mul,
    MulAssign,
    Neg,
    Ord,
    PartialEq,
    PartialOrd,
    Shl,
    ShlAssign,
    Sub,
    SubAssign,
    Rem,
    Serialize,
    Deserialize,
    Display,
)]
pub struct Int(pub(crate) Integer);

const POLY_INT_ORDER: Order = Order::LsfLe;

impl Int {
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let mut int = Integer::from(0);
        int.assign_digits(bytes, POLY_INT_ORDER);
        Self(int)
    }
    pub fn into_bytes(&self) -> Vec<u8> {
        let Int(x) = self;
        let mut data = vec![0; x.significant_digits::<u8>() as usize];
        x.write_digits(&mut data, POLY_INT_ORDER);
        data
    }
}

impl Hash for Int {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.0.hash(h)
    }
}

impl<T> From<T> for Int
where
    Integer: From<T>,
{
    fn from(x: T) -> Self {
        Self(x.into())
    }
}

impl Zero for Int {
    fn zero() -> Self {
        Integer::from(0).into()
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl Mul<Int> for Integer {
    type Output = Integer;
    fn mul(self, other: Int) -> Self::Output {
        self * other.0
    }
}

impl Div<Int> for Integer {
    type Output = Integer;
    fn div(self, other: Int) -> Self::Output {
        self / other.0
    }
}

impl DivAssign<Int> for Integer {
    fn div_assign(&mut self, other: Int) {
        *self /= other.0
    }
}

impl Rem<Int> for Integer {
    type Output = Integer;
    fn rem(self, other: Int) -> Self::Output {
        self % other.0
    }
}

impl One for Int {
    fn one() -> Self {
        Integer::from(1).into()
    }
}

impl EuclideanDomain<Int> for Int {
    fn degree(&self) -> Self {
        self.clone()
    }

    fn div_with_rem(self, other: Self) -> (Self, Self) {
        let (q, r) = self.0.div_rem(other.0);
        (q.into(), r.into())
    }
}

impl Shr<usize> for Int {
    type Output = Self;
    fn shr(self, bits: usize) -> Self {
        Self(self.0 >> (bits as u32))
    }
}

impl Shr<u32> for Int {
    type Output = Self;
    fn shr(self, bits: u32) -> Self {
        Self(self.0 >> bits)
    }
}
