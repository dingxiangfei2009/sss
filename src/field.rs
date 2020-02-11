use std::{
    fmt::{Display, Formatter, Result as FmtResult},
    hash::{Hash, Hasher},
    marker::PhantomData,
    ops::{
        Add, AddAssign, BitAnd, BitOr, Div, DivAssign, Mul, MulAssign, Neg, Rem, Shl, Shr, Sub,
        SubAssign,
    },
};

use alga::general::{AbstractMagma, Additive, Identity, Multiplicative, TwoSidedInverse};
use alga_derive::Alga;
use num::traits::{One, Zero};
use rand::RngCore;
use serde::{de::Deserializer, ser::Serializer, Deserialize, Serialize};

use crate::{adapter::Int, pow, EuclideanDomain};

pub trait ConstructibleNumber:
    Clone
    + Zero
    + One
    + Mul<Output = Self>
    + Div<Output = Self>
    + Rem<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Shl<u32, Output = Self>
    + Shr<u32, Output = Self>
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + Neg<Output = Self>
    + Ord
    + Eq
{
}

impl<T> ConstructibleNumber for T where
    T: Clone
        + Zero
        + One
        + Mul<Output = T>
        + Div<Output = T>
        + Rem<Output = T>
        + Add<Output = T>
        + Sub<Output = T>
        + Shl<u32, Output = T>
        + Shr<u32, Output = T>
        + BitAnd<Output = T>
        + BitOr<Output = T>
        + Neg<Output = T>
        + Ord
        + Eq
{
}

pub trait FiniteField: alga::general::Field {
    fn characteristic<T: ConstructibleNumber>() -> T;
    fn degree_extension<T: ConstructibleNumber>() -> T;

    // TODO: requires the base to have the same characteristics and degree of extension to be 1
    /// Scalar type, with respect to some finite set of
    /// generators of this finite field
    type Scalar: FiniteField;

    fn to_vec(&self) -> Vec<Self::Scalar>;
    fn from_scalar(scalar: Self::Scalar) -> Self;

    /// Apply Frobenius Endomorphism from the base field
    fn frobenius_base(self) -> Self;
    /// Auxillary Frobenius map for arbitrary type
    fn field_size<T: ConstructibleNumber>() -> T;
    /// Try to lower the element into the subfield
    fn try_lower(self) -> Option<Self::Scalar>;
}

pub trait FinitelyGenerated<G> {
    fn generator() -> Self;
}

pub fn int_inj<F, N>(mut n: N) -> F
where
    F: Zero + One + Add<Output = F> + Clone + Neg<Output = F>,
    N: Rem<Output = N> + DivAssign + Clone + One + Zero + Ord,
{
    let neg = n < N::zero();
    let mut f = F::zero();
    let mut f_ = F::one();
    let n_one = N::one();
    let n_two = n_one.clone() + n_one.clone();
    while !n.is_zero() {
        if !(n.clone() % n_two.clone()).is_zero() {
            f = f.clone() + f_.clone();
        }
        f_ = f_.clone() + f_.clone();
        n /= n_two.clone();
    }
    if neg {
        -f
    } else {
        f
    }
}

pub const GF2561D_NORMAL_BASIS: GF2561D = GF2561D(0b1110_0111);
pub const GF2561D_NORMAL_BASIS_SET: &[GF2561D] = &[
    GF2561D(0b1110_0111), // alpha^(2^0)
    GF2561D(0b1011_1111), // alpha^(2^1)
    GF2561D(0b0010_1111), // alpha^(2^2)
    GF2561D(0b0010_0001), // alpha^(2^3)
    GF2561D(0b0111_0101), // alpha^(2^4)
    GF2561D(0b1011_0101), // alpha^(2^5)
    GF2561D(0b0110_1011), // alpha^(2^6)
    GF2561D(0b1111_1100), // alpha^(2^7)
];

pub trait ArbitraryElement {
    fn arbitrary<R: rand::RngCore>(rng: &mut R) -> Self;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Alga, Hash)]
#[alga_traits(Field(Additive, Multiplicative))]
pub struct F2(pub u8);

/// GF(2^8) with quotient 1 + x^2 + x^3 + x^4 + x^8
#[derive(Clone, Copy, PartialEq, Eq, Debug, Alga, Hash)]
#[alga_traits(Field(Additive, Multiplicative))]
pub struct GF2561D(pub u8);

pub struct GF2561DG2;

impl FiniteField for GF2561D {
    fn characteristic<T: ConstructibleNumber>() -> T {
        int_inj(2)
    }
    fn degree_extension<T: ConstructibleNumber>() -> T {
        int_inj(8)
    }
    type Scalar = F2;

    fn to_vec(&self) -> Vec<Self::Scalar> {
        let deg_ext: Int = Self::degree_extension();
        let deg_ext = deg_ext.assert_usize();
        let mut v = Vec::with_capacity(deg_ext);
        let GF2561D(mut x) = *self;
        for _ in 0..Self::degree_extension() {
            v.push(F2(x & 1));
            x >>= 1;
        }
        assert_eq!(x, 0);
        v
    }

    fn from_scalar(F2(x): F2) -> Self {
        GF2561D(x & 1)
    }

    fn frobenius_base(self) -> Self {
        pow(self, 2)
    }

    fn field_size<T: ConstructibleNumber>() -> T {
        let sz: T = int_inj(2);
        pow(sz, 8)
    }

    fn try_lower(self) -> Option<Self::Scalar> {
        if self.0 == 0 || self.0 == 1 {
            Some(F2(self.0))
        } else {
            None
        }
    }
}

impl FiniteField for F2 {
    fn characteristic<T: ConstructibleNumber>() -> T {
        int_inj(2)
    }
    fn degree_extension<T: ConstructibleNumber>() -> T {
        T::one()
    }
    type Scalar = Self;

    fn to_vec(&self) -> Vec<Self::Scalar> {
        vec![F2(self.0)]
    }

    fn from_scalar(s: Self) -> Self {
        s
    }

    fn frobenius_base(self) -> Self {
        self
    }

    fn field_size<T: ConstructibleNumber>() -> T {
        int_inj(2)
    }

    fn try_lower(self) -> Option<Self> {
        Some(self)
    }
}

impl Display for F2 {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{}", self.0)
    }
}

impl FinitelyGenerated<GF2561DG2> for GF2561D {
    fn generator() -> GF2561D {
        GF2561D(2)
    }
}

impl ArbitraryElement for GF2561D {
    fn arbitrary<R: rand::RngCore>(rng: &mut R) -> Self {
        use rand::distributions::Distribution;

        let u = rand::distributions::uniform::Uniform::from(0..=255);
        GF2561D(u.sample(rng))
    }
}

pub const ZERO: GF2561D = GF2561D(0);
pub const ONE: GF2561D = GF2561D(1);

pub struct Tables {
    pub exp: [u8; 256],
    pub log: [u8; 256],
}

pub static TABLES: Tables = Tables {
    exp: [
        1, 2, 4, 8, 16, 32, 64, 128, 29, 58, 116, 232, 205, 135, 19, 38, 76, 152, 45, 90, 180, 117,
        234, 201, 143, 3, 6, 12, 24, 48, 96, 192, 157, 39, 78, 156, 37, 74, 148, 53, 106, 212, 181,
        119, 238, 193, 159, 35, 70, 140, 5, 10, 20, 40, 80, 160, 93, 186, 105, 210, 185, 111, 222,
        161, 95, 190, 97, 194, 153, 47, 94, 188, 101, 202, 137, 15, 30, 60, 120, 240, 253, 231,
        211, 187, 107, 214, 177, 127, 254, 225, 223, 163, 91, 182, 113, 226, 217, 175, 67, 134, 17,
        34, 68, 136, 13, 26, 52, 104, 208, 189, 103, 206, 129, 31, 62, 124, 248, 237, 199, 147, 59,
        118, 236, 197, 151, 51, 102, 204, 133, 23, 46, 92, 184, 109, 218, 169, 79, 158, 33, 66,
        132, 21, 42, 84, 168, 77, 154, 41, 82, 164, 85, 170, 73, 146, 57, 114, 228, 213, 183, 115,
        230, 209, 191, 99, 198, 145, 63, 126, 252, 229, 215, 179, 123, 246, 241, 255, 227, 219,
        171, 75, 150, 49, 98, 196, 149, 55, 110, 220, 165, 87, 174, 65, 130, 25, 50, 100, 200, 141,
        7, 14, 28, 56, 112, 224, 221, 167, 83, 166, 81, 162, 89, 178, 121, 242, 249, 239, 195, 155,
        43, 86, 172, 69, 138, 9, 18, 36, 72, 144, 61, 122, 244, 245, 247, 243, 251, 235, 203, 139,
        11, 22, 44, 88, 176, 125, 250, 233, 207, 131, 27, 54, 108, 216, 173, 71, 142, 1,
    ],
    log: [
        0, 0, 1, 25, 2, 50, 26, 198, 3, 223, 51, 238, 27, 104, 199, 75, 4, 100, 224, 14, 52, 141,
        239, 129, 28, 193, 105, 248, 200, 8, 76, 113, 5, 138, 101, 47, 225, 36, 15, 33, 53, 147,
        142, 218, 240, 18, 130, 69, 29, 181, 194, 125, 106, 39, 249, 185, 201, 154, 9, 120, 77,
        228, 114, 166, 6, 191, 139, 98, 102, 221, 48, 253, 226, 152, 37, 179, 16, 145, 34, 136, 54,
        208, 148, 206, 143, 150, 219, 189, 241, 210, 19, 92, 131, 56, 70, 64, 30, 66, 182, 163,
        195, 72, 126, 110, 107, 58, 40, 84, 250, 133, 186, 61, 202, 94, 155, 159, 10, 21, 121, 43,
        78, 212, 229, 172, 115, 243, 167, 87, 7, 112, 192, 247, 140, 128, 99, 13, 103, 74, 222,
        237, 49, 197, 254, 24, 227, 165, 153, 119, 38, 184, 180, 124, 17, 68, 146, 217, 35, 32,
        137, 46, 55, 63, 209, 91, 149, 188, 207, 205, 144, 135, 151, 178, 220, 252, 190, 97, 242,
        86, 211, 171, 20, 42, 93, 158, 132, 60, 57, 83, 71, 109, 65, 162, 31, 45, 67, 216, 183,
        123, 164, 118, 196, 23, 73, 236, 127, 12, 111, 246, 108, 161, 59, 82, 41, 157, 85, 170,
        251, 96, 134, 177, 187, 204, 62, 90, 203, 89, 95, 176, 156, 169, 160, 81, 11, 245, 22, 235,
        122, 117, 44, 215, 79, 174, 213, 233, 230, 231, 173, 232, 116, 214, 244, 234, 168, 80, 88,
        175,
    ],
};

impl Identity<Additive> for GF2561D {
    #[inline]
    fn identity() -> Self {
        ZERO
    }
}

impl Identity<Multiplicative> for GF2561D {
    #[inline]
    fn identity() -> Self {
        ONE
    }
}

impl AbstractMagma<Additive> for GF2561D {
    #[inline]
    fn operate(&self, right: &Self) -> Self {
        match (self, right) {
            (GF2561D(left), GF2561D(right)) => GF2561D(left ^ right),
        }
    }
}

impl AbstractMagma<Multiplicative> for GF2561D {
    #[inline]
    fn operate(&self, right: &Self) -> Self {
        match (*self, *right) {
            (GF2561D(0), GF2561D(_)) => GF2561D(0),
            (GF2561D(_), GF2561D(0)) => GF2561D(0),
            (GF2561D(left), GF2561D(right)) => {
                let log = TABLES.log[left as usize] as u16 + TABLES.log[right as usize] as u16;
                if log < 255 {
                    GF2561D(TABLES.exp[log as usize])
                } else {
                    GF2561D(TABLES.exp[(log - 255) as usize])
                }
            }
        }
    }
}

impl TwoSidedInverse<Additive> for GF2561D {
    #[inline]
    fn two_sided_inverse(&self) -> Self {
        *self
    }
}

impl TwoSidedInverse<Multiplicative> for GF2561D {
    #[inline]
    fn two_sided_inverse(&self) -> Self {
        match *self {
            GF2561D(0) => panic!("divide by zero"),
            GF2561D(x) => {
                let log = 255 - TABLES.log[x as usize] as u16;
                GF2561D(TABLES.exp[log as usize])
            }
        }
    }
}

impl Add for GF2561D {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        <Self as AbstractMagma<Additive>>::operate(&self, &other)
    }
}

impl AddAssign for GF2561D {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl Sub for GF2561D {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        <Self as AbstractMagma<Additive>>::operate(&self, &other)
    }
}

impl SubAssign for GF2561D {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl Neg for GF2561D {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self
    }
}

impl Mul for GF2561D {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        <Self as AbstractMagma<Multiplicative>>::operate(&self, &other)
    }
}

impl MulAssign for GF2561D {
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl Div for GF2561D {
    type Output = Self;
    #[inline]
    fn div(self, other: Self) -> Self {
        <Self as AbstractMagma<Multiplicative>>::operate(
            &self,
            &<Self as TwoSidedInverse<Multiplicative>>::two_sided_inverse(&other),
        )
    }
}

impl DivAssign for GF2561D {
    #[inline]
    fn div_assign(&mut self, other: Self) {
        *self = *self / other;
    }
}

impl Zero for GF2561D {
    #[inline]
    fn zero() -> Self {
        <Self as Identity<Additive>>::identity()
    }
    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

impl One for GF2561D {
    #[inline]
    fn one() -> Self {
        <Self as Identity<Multiplicative>>::identity()
    }
}

impl Display for GF2561D {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{:08b}", self.0)
    }
}

impl Identity<Additive> for F2 {
    #[inline]
    fn identity() -> Self {
        F2(0)
    }
}

impl Identity<Multiplicative> for F2 {
    #[inline]
    fn identity() -> Self {
        F2(1)
    }
}

impl AbstractMagma<Additive> for F2 {
    #[inline]
    fn operate(&self, right: &Self) -> Self {
        match (self, right) {
            (F2(left), F2(right)) => F2(left ^ right),
        }
    }
}

impl AbstractMagma<Multiplicative> for F2 {
    #[inline]
    fn operate(&self, right: &Self) -> Self {
        match (*self, *right) {
            (F2(x), F2(y)) => F2(x & y),
        }
    }
}

impl TwoSidedInverse<Additive> for F2 {
    #[inline]
    fn two_sided_inverse(&self) -> Self {
        *self
    }
}

impl TwoSidedInverse<Multiplicative> for F2 {
    #[inline]
    fn two_sided_inverse(&self) -> Self {
        match *self {
            F2(0) => panic!("divide by zero"),
            F2(x) => F2(x),
        }
    }
}

impl Add for F2 {
    type Output = Self;
    #[inline]
    fn add(self, other: Self) -> Self {
        <Self as AbstractMagma<Additive>>::operate(&self, &other)
    }
}

impl AddAssign for F2 {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl Sub for F2 {
    type Output = Self;
    #[inline]
    fn sub(self, other: Self) -> Self {
        <Self as AbstractMagma<Additive>>::operate(&self, &other)
    }
}

impl SubAssign for F2 {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = *self - other;
    }
}

impl Neg for F2 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        self
    }
}

impl Mul for F2 {
    type Output = Self;
    #[inline]
    fn mul(self, other: Self) -> Self {
        <Self as AbstractMagma<Multiplicative>>::operate(&self, &other)
    }
}

impl MulAssign for F2 {
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl Div for F2 {
    type Output = Self;
    #[inline]
    fn div(self, other: Self) -> Self {
        <Self as AbstractMagma<Multiplicative>>::operate(
            &self,
            &<Self as TwoSidedInverse<Multiplicative>>::two_sided_inverse(&other),
        )
    }
}

impl DivAssign for F2 {
    #[inline]
    fn div_assign(&mut self, other: Self) {
        *self = *self / other;
    }
}

impl Zero for F2 {
    #[inline]
    fn zero() -> Self {
        <Self as Identity<Additive>>::identity()
    }
    #[inline]
    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

impl One for F2 {
    #[inline]
    fn one() -> Self {
        <Self as Identity<Multiplicative>>::identity()
    }
}

pub struct Fp<P, V = u64, D = V> {
    value: V,
    _p: PhantomData<fn() -> (P, D)>,
}

#[derive(Default)]
pub struct FpVisitor<V>(PhantomData<fn() -> V>);

impl<P, V: Serialize, D> Serialize for Fp<P, V, D> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.value.serialize(s)
    }
}

impl<'a, P, V, D> Deserialize<'a> for Fp<P, V, D>
where
    P: PrimeModulo<V>,
    V: Deserialize<'a> + Zero + EuclideanDomain<D>,
    D: Ord,
{
    fn deserialize<De: Deserializer<'a>>(d: De) -> Result<Self, De::Error> {
        Ok(Self::new(V::deserialize(d)?))
    }
}

impl<P, V, D> Fp<P, V, D> {
    /// Inject onto a Fp value
    pub fn new(value: V) -> Self
    where
        P: PrimeModulo<V>,
        V: Zero + EuclideanDomain<D>,
        D: Ord,
    {
        let (_, value) = value.div_with_rem(P::divisor());
        let value = value + P::divisor();
        let (_, value) = value.div_with_rem(P::divisor());
        Self {
            value,
            _p: PhantomData,
        }
    }

    /// Project out the "inner" value
    pub fn into_inner(self) -> V {
        self.value
    }

    pub fn inner(&self) -> &V {
        &self.value
    }

    pub fn uniform_sample<R>(rng: &mut R) -> Self
    where
        P: PrimeModulo<V>,
        R: RngCore,
        V: Zero + One + Ord + Clone + Div<Output = V> + EuclideanDomain<D>,
        D: Ord,
    {
        Self::new(crate::uniform_sample(rng, P::divisor()))
    }
}

impl<P, V, D> PartialEq for Fp<P, V, D>
where
    V: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<P, V, D> Hash for Fp<P, V, D>
where
    V: Hash,
{
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.value.hash(h)
    }
}

impl<P, V, D> Clone for Fp<P, V, D>
where
    V: Clone,
{
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
            _p: PhantomData,
        }
    }
}

impl<P, V, D> Copy for Fp<P, V, D> where V: Copy {}

impl<P, V, D> std::fmt::Debug for Fp<P, V, D>
where
    V: std::fmt::Debug,
{
    fn fmt(&self, formatter: &mut Formatter) -> FmtResult {
        self.value.fmt(formatter)
    }
}

impl<P, V, D> Display for Fp<P, V, D>
where
    V: Display,
{
    fn fmt(&self, formatter: &mut Formatter) -> FmtResult {
        self.value.fmt(formatter)
    }
}

pub trait PrimeModulo<V> {
    fn divisor() -> V;
}

impl<P, V, D> Zero for Fp<P, V, D>
where
    P: PrimeModulo<V>,
    V: Zero + Clone + EuclideanDomain<D>,
    D: Ord,
{
    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    fn zero() -> Self {
        Self {
            value: V::zero(),
            _p: PhantomData,
        }
    }
}

impl<P, V, D> Identity<Additive> for Fp<P, V, D>
where
    Self: Zero,
{
    fn identity() -> Self {
        Self::zero()
    }
}

impl<P, V, D> Neg for Fp<P, V, D>
where
    P: PrimeModulo<V>,
    V: Ord + Clone + Sub<Output = V> + EuclideanDomain<D>,
    D: Ord,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        TwoSidedInverse::<Additive>::two_sided_inverse(&self)
    }
}

impl<P, V, D> Add for Fp<P, V, D>
where
    P: PrimeModulo<V>,
    V: Add<Output = V> + Clone + EuclideanDomain<D>,
    D: Ord,
{
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        Self::new(self.into_inner() + other.into_inner())
    }
}

impl<P, V, D> AddAssign for Fp<P, V, D>
where
    Self: Add<Output = Self> + Clone,
{
    fn add_assign(&mut self, other: Self) {
        *self = self.clone() + other;
    }
}

impl<P, V, D> Sub for Fp<P, V, D>
where
    Self: Neg<Output = Self> + Add<Output = Self>,
{
    type Output = Self;
    fn sub(self, other: Self) -> Self::Output {
        self + -other
    }
}

impl<P, V, D> SubAssign for Fp<P, V, D>
where
    Self: Neg<Output = Self> + Add<Output = Self> + Clone,
{
    fn sub_assign(&mut self, other: Self) {
        *self = self.clone() - other;
    }
}

impl<P, V, D> TwoSidedInverse<Additive> for Fp<P, V, D>
where
    P: PrimeModulo<V>,
    V: Sub<Output = V> + EuclideanDomain<D> + Clone,
    D: Ord,
{
    fn two_sided_inverse(&self) -> Self {
        Self::new(P::divisor() - self.value.clone())
    }
}

impl<P, V, D> One for Fp<P, V, D>
where
    P: PrimeModulo<V>,
    V: One + EuclideanDomain<D>,
    D: Ord,
{
    fn one() -> Self {
        Self::new(V::one())
    }
}

impl<P, V, D> Identity<Multiplicative> for Fp<P, V, D>
where
    Self: One,
{
    fn identity() -> Self {
        Self::one()
    }
}

impl<P, V, D> Mul for Fp<P, V, D>
where
    P: PrimeModulo<V>,
    V: One + EuclideanDomain<D>,
    D: Ord,
{
    type Output = Self;
    fn mul(self, other: Self) -> Self::Output {
        Self::new(self.into_inner() * other.into_inner())
    }
}

impl<P, V, D> MulAssign for Fp<P, V, D>
where
    Self: Mul<Output = Self> + Clone,
{
    fn mul_assign(&mut self, other: Self) {
        *self = self.clone() * other;
    }
}

impl<P, V, D> Div for Fp<P, V, D>
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

impl<P, V, D> DivAssign for Fp<P, V, D>
where
    Self: Div<Output = Self> + Clone,
{
    fn div_assign(&mut self, other: Self) {
        *self = self.clone() / other;
    }
}

impl<P, V, D> TwoSidedInverse<Multiplicative> for Fp<P, V, D>
where
    P: PrimeModulo<V>,
    V: Zero + One + Clone + EuclideanDomain<D> + Sub<Output = V> + Eq,
    D: Ord,
{
    fn two_sided_inverse(&self) -> Self {
        let Self { value, .. } = self;
        assert!(!self.value.is_zero());
        let (_, value, d) = P::divisor().extended_gcd::<V, V>(value.clone());
        assert!(d.is_one(), "modulus is not a prime number");
        Self {
            value,
            _p: PhantomData,
        }
    }
}

impl<P, V, D> AbstractMagma<Additive> for Fp<P, V, D>
where
    Self: Add<Output = Self> + Clone,
{
    #[inline]
    fn operate(&self, right: &Self) -> Self {
        self.clone() + right.clone()
    }
}

impl<P, V, D> AbstractMagma<Multiplicative> for Fp<P, V, D>
where
    Self: Mul<Output = Self> + Clone,
{
    #[inline]
    fn operate(&self, right: &Self) -> Self {
        self.clone() * right.clone()
    }
}

impl<P, V, D> alga::general::AbstractField for Fp<P, V, D>
where
    P: PrimeModulo<V>,
    V: One + Zero + Clone + EuclideanDomain<D> + Eq + Sub<Output = V>,
    D: Ord,
{
}

impl<P, V, D> alga::general::AbstractRingCommutative for Fp<P, V, D>
where
    P: PrimeModulo<V>,
    V: One + Zero + Clone + EuclideanDomain<D> + Eq + Sub<Output = V>,
    D: Ord,
{
}

impl<P, V, D> alga::general::AbstractRing for Fp<P, V, D>
where
    P: PrimeModulo<V>,
    V: One + Zero + Clone + EuclideanDomain<D> + Eq + Sub<Output = V>,
    D: Ord,
{
}

impl<P, V, D> alga::general::AbstractGroupAbelian<Additive> for Fp<P, V, D>
where
    P: PrimeModulo<V>,
    V: One + Zero + Clone + EuclideanDomain<D> + Eq + Sub<Output = V>,
    D: Ord,
{
}

impl<P, V, D> alga::general::AbstractGroupAbelian<Multiplicative> for Fp<P, V, D>
where
    P: PrimeModulo<V>,
    V: One + Zero + Clone + EuclideanDomain<D> + Eq + Sub<Output = V>,
    D: Ord,
{
}

impl<P, V, D> alga::general::AbstractGroup<Additive> for Fp<P, V, D>
where
    P: PrimeModulo<V>,
    V: One + Zero + Clone + EuclideanDomain<D> + Eq + Sub<Output = V>,
    D: Ord,
{
}

impl<P, V, D> alga::general::AbstractGroup<Multiplicative> for Fp<P, V, D>
where
    P: PrimeModulo<V>,
    V: One + Zero + Clone + EuclideanDomain<D> + Eq + Sub<Output = V>,
    D: Ord,
{
}

impl<P, V, D> alga::general::AbstractMonoid<Additive> for Fp<P, V, D>
where
    P: PrimeModulo<V>,
    V: One + Zero + Clone + EuclideanDomain<D> + Eq + Sub<Output = V>,
    D: Ord,
{
}

impl<P, V, D> alga::general::AbstractMonoid<Multiplicative> for Fp<P, V, D>
where
    P: PrimeModulo<V>,
    V: One + Zero + Clone + EuclideanDomain<D> + Eq + Sub<Output = V>,
    D: Ord,
{
}

impl<P, V, D> alga::general::AbstractLoop<Additive> for Fp<P, V, D>
where
    P: PrimeModulo<V>,
    V: One + Zero + Clone + EuclideanDomain<D> + Eq + Sub<Output = V>,
    D: Ord,
{
}

impl<P, V, D> alga::general::AbstractLoop<Multiplicative> for Fp<P, V, D>
where
    P: PrimeModulo<V>,
    V: One + Zero + Clone + EuclideanDomain<D> + Eq + Sub<Output = V>,
    D: Ord,
{
}

impl<P, V, D> alga::general::AbstractSemigroup<Additive> for Fp<P, V, D>
where
    P: PrimeModulo<V>,
    V: One + Zero + Clone + EuclideanDomain<D> + Eq + Sub<Output = V>,
    D: Ord,
{
}

impl<P, V, D> alga::general::AbstractSemigroup<Multiplicative> for Fp<P, V, D>
where
    P: PrimeModulo<V>,
    V: One + Zero + Clone + EuclideanDomain<D> + Eq + Sub<Output = V>,
    D: Ord,
{
}

impl<P, V, D> alga::general::AbstractQuasigroup<Additive> for Fp<P, V, D>
where
    P: PrimeModulo<V>,
    V: One + Zero + Clone + EuclideanDomain<D> + Eq + Sub<Output = V>,
    D: Ord,
{
}

impl<P, V, D> alga::general::AbstractQuasigroup<Multiplicative> for Fp<P, V, D>
where
    P: PrimeModulo<V>,
    V: One + Zero + Clone + EuclideanDomain<D> + Eq + Sub<Output = V>,
    D: Ord,
{
}

impl<P, V, D> ndarray::ScalarOperand for Fp<P, V, D> where Self: 'static + Clone {}

#[cfg(test)]
mod tests {
    use super::*;

    use alga::general::{AbstractGroupAbelian, AbstractQuasigroup, AbstractRing};

    use quickcheck::{Arbitrary, Gen};

    impl Arbitrary for GF2561D {
        fn arbitrary<G: Gen>(g: &mut G) -> Self {
            GF2561D(u8::arbitrary(g))
        }
    }

    #[quickcheck]
    fn prop_mul_and_add_are_distributive(args: (GF2561D, GF2561D, GF2561D)) -> bool {
        GF2561D::prop_mul_and_add_are_distributive(args)
    }

    #[quickcheck]
    fn prop_mul_is_commutative(args: (GF2561D, GF2561D)) -> bool {
        <GF2561D as AbstractGroupAbelian<Multiplicative>>::prop_is_commutative(args)
    }

    #[quickcheck]
    fn prop_mul_inv_is_latin_square(args: (GF2561D, GF2561D)) -> bool {
        args.0.is_zero()
            || args.1.is_zero()
            || <GF2561D as AbstractQuasigroup<Multiplicative>>::prop_inv_is_latin_square(args)
    }

    impl<P, V, D> Arbitrary for Fp<P, V, D>
    where
        P: 'static + PrimeModulo<V>,
        V: 'static + Ord + Clone + Send + Sync + Arbitrary + EuclideanDomain<D>,
        D: 'static + Ord,
    {
        fn arbitrary<G: Gen>(g: &mut G) -> Self {
            Fp::new(V::arbitrary(g))
        }
    }
}
