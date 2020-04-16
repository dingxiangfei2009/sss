/// More Galois Field constructions and arithmetics
use std::{
    any::type_name,
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    iter::repeat,
    iter::repeat_with,
    marker::PhantomData,
    mem::take,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use alga::general::{
    AbstractField, AbstractGroup, AbstractGroupAbelian, AbstractLoop, AbstractMagma,
    AbstractMonoid, AbstractQuasigroup, AbstractRing, AbstractRingCommutative, AbstractSemigroup,
    Additive, Field, Identity, Multiplicative, TwoSidedInverse,
};
use ndarray::Array2;
use num::{One, Zero};
use rand::RngCore;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::{
    adapter::Int,
    field::{int_inj, ArbitraryElement, ConstructibleNumber, FiniteField, F2, GF2561D},
    linalg::solve,
    pow, Coord, EuclideanDomain, Polynomial,
};

pub fn compute_normal_basis_multiplication_table<F>(
    base_generator: F,
) -> (Vec<F::Scalar>, Vec<Vec<F::Scalar>>)
where
    F: Clone + FiniteField,
    F::Scalar: Send + Sync,
{
    let m = F::degree_extension::<Int>().assert_usize();
    // compute normal basis elements
    let mut betas = vec![];
    let beta = base_generator.clone();
    let mut beta_ = beta.clone();
    for _ in 0..m {
        betas.push(beta_.to_vec());
        beta_ = beta_.frobenius_base();
    }

    // suffice to compute beta^(1+q^i) where q=|F|, i <= floor(m / 2)
    let mut table = vec![];
    beta_ = beta.clone();
    for _ in 0..=m / 2 {
        let alpha = beta_.clone() * beta.clone();
        let alpha = alpha.to_vec();
        let mat: Vec<_> = betas.iter().flatten().cloned().chain(alpha).collect();
        table.push(
            solve(
                Array2::from_shape_vec((m + 1, m), mat)
                    .expect("shape should be correct")
                    .t()
                    .to_owned(),
            )
            .expect("basis are linearly independent")
            .to_vec(),
        );
        beta_ = beta_.frobenius_base();
    }
    let mat: Vec<_> = betas
        .iter()
        .flatten()
        .cloned()
        .chain(vec![F::Scalar::one()])
        .chain(vec![F::Scalar::zero(); m - 1])
        .collect();
    let one = solve(
        Array2::from_shape_vec((m + 1, m), mat)
            .expect("shape should be correct")
            .t()
            .to_owned(),
    )
    .expect("basis are linearly independent")
    .to_vec();
    (one, table)
}

/*
0 1 2 3 4 5 6 7
1 0 0 0 0 1 0 0
1 0 0 1 0 0 0 0

0 1 2 3 4 5 6
1 0 0 0 0 1 0
1 0 1 0 0 0 0
*/

pub trait MonicPolynomial<F>
where
    F: Field,
{
    fn degree() -> usize;
    fn repr() -> Polynomial<F>;
    fn reduce(poly: Polynomial<F>) -> Polynomial<F> {
        if EuclideanDomain::degree(&poly) < Self::degree() {
            poly
        } else {
            let m = Self::repr();
            let (_, r) = poly.div_with_rem(m);
            r
        }
    }
    fn reduce_mul(mut a: Polynomial<F>, b: Polynomial<F>) -> Polynomial<F> {
        if EuclideanDomain::degree(&a) + EuclideanDomain::degree(&b) < Self::degree() {
            let c = a * b;
            debug_assert!(c.degree() < Self::degree());
            c
        } else {
            let Polynomial(b) = b;
            let mut c = Polynomial::zero();
            for b in b {
                c = c + a.clone() * b;
                a = Self::reduce(a * Polynomial(vec![F::zero(), F::one()]));
            }
            debug_assert!(c.degree() < Self::degree());
            c
        }
    }
    fn inv(a: Polynomial<F>) -> Option<Polynomial<F>> {
        let (b, _, g): (Polynomial<F>, Polynomial<F>, Polynomial<F>) =
            Self::reduce(a.clone()).extended_gcd(Self::repr());
        if !g.is_zero() && g.degree() == 0 {
            let r = Self::reduce(b / g.0[0].clone());
            debug_assert!(r.degree() < Self::degree());
            Some(r)
        } else {
            None
        }
    }
}

pub struct PolynomialExtension<F, P> {
    data: Polynomial<F>,
    _p: PhantomData<fn() -> P>,
}

impl<F, P> Serialize for PolynomialExtension<F, P>
where
    F: Serialize,
{
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.data.serialize(s)
    }
}

impl<'a, F, P> Deserialize<'a> for PolynomialExtension<F, P>
where
    F: Deserialize<'a>,
{
    fn deserialize<D: Deserializer<'a>>(d: D) -> Result<Self, D::Error> {
        let data = Polynomial::<F>::deserialize(d)?;
        Ok(Self {
            data,
            _p: PhantomData,
        })
    }
}

impl<F, P> Debug for PolynomialExtension<F, P>
where
    F: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        f.debug_struct(type_name::<Self>())
            .field("data", &self.data)
            .finish()
    }
}

impl<F, P> Display for PolynomialExtension<F, P>
where
    F: Display,
{
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{}({})", type_name::<Self>(), self.data)
    }
}

impl<F, P> Eq for PolynomialExtension<F, P> where F: Eq {}

impl<F, P> PartialEq for PolynomialExtension<F, P>
where
    F: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<F, P> Clone for PolynomialExtension<F, P>
where
    F: Clone,
{
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            _p: PhantomData,
        }
    }
}

impl<F, P> Add for PolynomialExtension<F, P>
where
    F: Field + Clone,
    P: MonicPolynomial<F>,
{
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            data: self.data + other.data,
            _p: PhantomData,
        }
    }
}

impl<F, P> Sub for PolynomialExtension<F, P>
where
    F: Field + Clone,
    P: MonicPolynomial<F>,
{
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self {
            data: self.data - other.data,
            _p: PhantomData,
        }
    }
}

impl<F, P> Zero for PolynomialExtension<F, P>
where
    F: Field + Clone,
    P: MonicPolynomial<F>,
{
    fn zero() -> Self {
        Self {
            data: Polynomial::zero(),
            _p: PhantomData,
        }
    }
    fn is_zero(&self) -> bool {
        self.data.is_zero()
    }
}

impl<F, P> Mul for PolynomialExtension<F, P>
where
    F: Field + Clone,
    P: MonicPolynomial<F>,
{
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let data = P::reduce_mul(self.data, other.data);
        debug_assert!(data.degree() < P::degree());
        Self {
            data,
            _p: PhantomData,
        }
    }
}

impl<F, P> One for PolynomialExtension<F, P>
where
    F: Field + Clone,
    P: MonicPolynomial<F>,
{
    fn one() -> Self {
        Self {
            data: Polynomial::one(),
            _p: PhantomData,
        }
    }
}

impl<F, P> MulAssign for PolynomialExtension<F, P>
where
    F: Field + Clone,
    P: MonicPolynomial<F>,
{
    fn mul_assign(&mut self, rhs: Self) {
        let data = take(&mut self.data);
        self.data = P::reduce_mul(data, rhs.data);
        debug_assert!(self.data.degree() < P::degree());
    }
}

impl<F, P> AddAssign for PolynomialExtension<F, P>
where
    F: Field + Clone,
    P: MonicPolynomial<F>,
{
    fn add_assign(&mut self, other: Self) {
        let data = take(&mut self.data);
        self.data = data + other.data;
        debug_assert!(self.data.degree() < P::degree());
    }
}

impl<F, P> SubAssign for PolynomialExtension<F, P>
where
    F: Field + Clone,
    P: MonicPolynomial<F>,
{
    fn sub_assign(&mut self, other: Self) {
        let data = take(&mut self.data);
        self.data = data - other.data;
        debug_assert!(self.data.degree() < P::degree());
    }
}

impl<F, P> Neg for PolynomialExtension<F, P>
where
    F: Field + Clone,
    P: MonicPolynomial<F>,
{
    type Output = Self;
    fn neg(self) -> Self {
        let data = -self.data;
        debug_assert!(data.degree() < P::degree());
        Self {
            data,
            _p: PhantomData,
        }
    }
}

impl<F, P> Div for PolynomialExtension<F, P>
where
    F: Field + Clone,
    P: MonicPolynomial<F>,
{
    type Output = Self;
    fn div(self, other: Self) -> Self {
        self * TwoSidedInverse::<Multiplicative>::two_sided_inverse(&other)
    }
}

impl<F, P> DivAssign for PolynomialExtension<F, P>
where
    F: Field + Clone,
    P: MonicPolynomial<F>,
{
    fn div_assign(&mut self, other: Self) {
        let data = take(&mut self.data);
        self.data = P::reduce_mul(
            data,
            TwoSidedInverse::<Multiplicative>::two_sided_inverse(&other).data,
        );
        debug_assert!(self.data.degree() < P::degree());
    }
}

impl<F, P> AbstractMagma<Additive> for PolynomialExtension<F, P>
where
    F: Field,
    P: MonicPolynomial<F>,
{
    #[inline]
    fn operate(&self, right: &Self) -> Self {
        self.clone() + right.clone()
    }
}

impl<F, P> AbstractMagma<Multiplicative> for PolynomialExtension<F, P>
where
    F: Field,
    P: MonicPolynomial<F>,
{
    #[inline]
    fn operate(&self, right: &Self) -> Self {
        self.clone() * right.clone()
    }
}

impl<F, P> TwoSidedInverse<Additive> for PolynomialExtension<F, P>
where
    F: Field,
    P: MonicPolynomial<F>,
{
    fn two_sided_inverse(&self) -> Self {
        Self {
            data: -self.data.clone(),
            _p: PhantomData,
        }
    }
}

impl<F, P> TwoSidedInverse<Multiplicative> for PolynomialExtension<F, P>
where
    F: Field,
    P: MonicPolynomial<F>,
{
    fn two_sided_inverse(&self) -> Self {
        assert!(!self.data.is_zero(), "division by zero");
        Self {
            data: P::inv(self.data.clone()).expect("monic polynomial should be irreducible"),
            _p: PhantomData,
        }
    }
}

impl<F, P> Identity<Additive> for PolynomialExtension<F, P>
where
    F: Field,
    P: MonicPolynomial<F>,
{
    fn identity() -> Self {
        Self {
            data: Polynomial::zero(),
            _p: PhantomData,
        }
    }
}

impl<F, P> Identity<Multiplicative> for PolynomialExtension<F, P>
where
    F: Field,
    P: MonicPolynomial<F>,
{
    fn identity() -> Self {
        Self {
            data: Polynomial::one(),
            _p: PhantomData,
        }
    }
}

impl<F, P> AbstractField for PolynomialExtension<F, P>
where
    F: Field,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractRingCommutative for PolynomialExtension<F, P>
where
    F: Field,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractRing for PolynomialExtension<F, P>
where
    F: Field,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractGroupAbelian<Additive> for PolynomialExtension<F, P>
where
    F: Field,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractGroupAbelian<Multiplicative> for PolynomialExtension<F, P>
where
    F: Field,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractGroup<Additive> for PolynomialExtension<F, P>
where
    F: Field,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractGroup<Multiplicative> for PolynomialExtension<F, P>
where
    F: Field,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractMonoid<Additive> for PolynomialExtension<F, P>
where
    F: Field,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractMonoid<Multiplicative> for PolynomialExtension<F, P>
where
    F: Field,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractLoop<Additive> for PolynomialExtension<F, P>
where
    F: Field,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractLoop<Multiplicative> for PolynomialExtension<F, P>
where
    F: Field,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractSemigroup<Additive> for PolynomialExtension<F, P>
where
    F: Field,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractSemigroup<Multiplicative> for PolynomialExtension<F, P>
where
    F: Field,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractQuasigroup<Additive> for PolynomialExtension<F, P>
where
    F: Field,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractQuasigroup<Multiplicative> for PolynomialExtension<F, P>
where
    F: Field,
    P: MonicPolynomial<F>,
{
}

impl<F, P> Mul<F> for PolynomialExtension<F, P>
where
    F: Field + Clone,
    P: MonicPolynomial<F>,
{
    type Output = Self;
    fn mul(self, a: F) -> Self {
        Self {
            data: self.data * a,
            _p: PhantomData,
        }
    }
}

impl<F, P> FiniteField for PolynomialExtension<F, P>
where
    F: FiniteField + Clone,
    P: MonicPolynomial<F>,
{
    type Scalar = F;
    fn characteristic<T: ConstructibleNumber>() -> T {
        F::characteristic()
    }
    fn degree_extension<T: ConstructibleNumber>() -> T {
        int_inj(P::degree())
    }
    fn to_vec(&self) -> Vec<F> {
        let mut data = self.data.clone().0;
        data.resize(Self::degree_extension::<Int>().assert_usize(), F::zero());
        data
    }
    fn from_scalar(scalar: Self::Scalar) -> Self {
        Self {
            data: Polynomial(vec![scalar]),
            _p: PhantomData,
        }
    }
    fn frobenius_base(self) -> Self {
        let Self { data, .. } = self;
        let x = Self::root();
        let q: Int = Self::Scalar::field_size();
        let x_q = pow(x, q);
        let mut x_q_pow = Self::one();
        let mut r = Self::zero();
        for a in data.0 {
            r = r + x_q_pow.clone() * a;
            x_q_pow = x_q_pow * x_q.clone();
        }
        r
    }
    fn field_size<T: ConstructibleNumber>() -> T {
        crate::pow(Self::Scalar::field_size(), Self::degree_extension::<Int>())
    }

    fn try_lower(self) -> Option<Self::Scalar> {
        if self.data.degree() > 0 {
            None
        } else {
            let Polynomial(mut data) = self.data;
            let mut it = data.drain(..);
            Some(it.next().expect("polynomial should not be empty"))
        }
    }
    fn basis_elements() -> Vec<Self> {
        let mut v = vec![];
        for i in 0..P::degree() {
            v.push(Self::from_poly(Polynomial::new(
                repeat(F::zero()).take(i).chain(Some(F::one())),
            )))
        }
        v
    }
}

impl<F, P> PolynomialExtension<F, P>
where
    F: Field,
    P: MonicPolynomial<F>,
{
    pub fn root() -> Self {
        Self {
            data: P::reduce(Polynomial(vec![F::zero(), F::one()])),
            _p: PhantomData,
        }
    }
    pub fn from_poly(poly: Polynomial<F>) -> Self {
        Self {
            data: P::reduce(poly),
            _p: PhantomData,
        }
    }
}

impl<F, P> ArbitraryElement for PolynomialExtension<F, P>
where
    F: Field + ArbitraryElement,
    P: MonicPolynomial<F>,
{
    fn arbitrary<R: RngCore>(rng: &mut R) -> Self {
        Self::from_poly(Polynomial::new(
            repeat_with(|| F::arbitrary(rng)).take(P::degree()),
        ))
    }
}

pub fn artin_candidate<F: FiniteField + Clone, P: MonicPolynomial<F>>(
    candidate: F,
) -> PolynomialExtension<F, P> {
    type E<F, P> = PolynomialExtension<F, P>;
    let f = P::repr();
    let Coord(_, v) = f.eval_at(candidate.clone());
    let alpha = E::root();
    let Polynomial(f_) = f.formal_derivative();
    let f_ = Polynomial::new(f_.into_iter().map(E::from_scalar));
    let Coord(_, v2) = f_.eval_at(alpha.clone());
    let v3 = (E::from_scalar(candidate) - alpha) * v2;
    E::from_scalar(v) / v3
}

#[allow(non_camel_case_types)]
pub struct GF2561D_P2;

impl MonicPolynomial<GF2561D> for GF2561D_P2 {
    fn repr() -> Polynomial<GF2561D> {
        Polynomial(vec![GF2561D(255), GF2561D(1), GF2561D(1)])
    }
    fn degree() -> usize {
        2
    }
}

pub type GF65536P = PolynomialExtension<GF2561D, GF2561D_P2>;

pub fn normal_basis_test_polynomial<F>(deg_extension: usize) -> Vec<F>
where
    F: Field,
{
    let mut p = vec![F::zero(); deg_extension];
    p.push(F::one());
    p[0] -= F::one();
    p
}

pub fn test_normal_basis<F>(alpha: F) -> bool
where
    F: FiniteField + Clone,
{
    let deg_ext: Int = F::degree_extension();
    let deg_ext = deg_ext.assert_usize();
    let q: Int = F::Scalar::field_size();
    let g = Polynomial::new(normal_basis_test_polynomial(deg_ext));
    let mut beta = alpha.clone();
    let mut p = vec![];
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

/// Galois field extension over a subfield, through a normal basis
pub struct NormalBasisExtension<F, B> {
    data: Vec<F>,
    _p: PhantomData<fn() -> B>,
}

impl<F, B> Serialize for NormalBasisExtension<F, B>
where
    F: Serialize,
{
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.data.serialize(s)
    }
}

impl<'a, F, B> Deserialize<'a> for NormalBasisExtension<F, B>
where
    F: Deserialize<'a>,
{
    fn deserialize<D: Deserializer<'a>>(d: D) -> Result<Self, D::Error> {
        let data = Vec::deserialize(d)?;
        Ok(Self {
            data,
            _p: PhantomData,
        })
    }
}

impl<F, B> Debug for NormalBasisExtension<F, B>
where
    F: Debug,
{
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        f.debug_struct(type_name::<Self>())
            .field("data", &self.data)
            .finish()
    }
}

impl<F, B> Display for NormalBasisExtension<F, B>
where
    F: Display,
{
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "{}[", type_name::<Self>())?;
        for x in &self.data {
            write!(f, " {}", x)?;
        }
        write!(f, "]")
    }
}

impl<F, B> Clone for NormalBasisExtension<F, B>
where
    F: Clone,
{
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            _p: PhantomData,
        }
    }
}

impl<F, B> NormalBasisExtension<F, B> {
    fn new(data: impl IntoIterator<Item = F>) -> Self
    where
        B: NormalBasis<F>,
    {
        let data: Vec<_> = data.into_iter().collect();
        assert_eq!(data.len(), B::degree_extension());
        Self {
            data,
            _p: PhantomData,
        }
    }
    fn shift_right(mut self, n: usize) -> Self
    where
        B: NormalBasis<F>,
    {
        let n = n % B::degree_extension();
        self.data.rotate_right(n);
        self
    }
}

pub trait NormalBasis<F> {
    fn degree_extension() -> usize;
    fn multiplier(a: usize) -> Vec<F>;
    fn unit() -> F;

    fn mul(a: usize) -> Vec<F> {
        let m = Self::degree_extension();
        assert!(a < m);
        if a > m / 2 {
            let b = m - a;
            let mut p = Self::multiplier(b);
            assert_eq!(p.len(), m);
            p.rotate_left(b);
            p
        } else {
            Self::multiplier(a)
        }
    }
}

impl<F, B> PartialEq for NormalBasisExtension<F, B>
where
    F: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<F, B> Eq for NormalBasisExtension<F, B> where F: Eq {}

impl<F, B> Add for NormalBasisExtension<F, B>
where
    F: FiniteField,
{
    type Output = Self;
    fn add(mut self, other: Self) -> Self {
        for (a, b) in self.data.iter_mut().zip(other.data) {
            *a += b;
        }
        self
    }
}

impl<F, B> Sub for NormalBasisExtension<F, B>
where
    F: FiniteField,
{
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self {
            data: self
                .data
                .into_iter()
                .zip(other.data)
                .map(|(a, b)| a - b)
                .collect(),
            _p: PhantomData,
        }
    }
}

impl<F, B> Zero for NormalBasisExtension<F, B>
where
    F: FiniteField,
    B: NormalBasis<F>,
{
    fn zero() -> Self {
        Self {
            data: vec![F::zero(); B::degree_extension()],
            _p: PhantomData,
        }
    }
    fn is_zero(&self) -> bool {
        self.data.iter().all(|x| x.is_zero())
    }
}

impl<F, B> TwoSidedInverse<Additive> for NormalBasisExtension<F, B>
where
    F: FiniteField + Clone,
    B: NormalBasis<F>,
{
    fn two_sided_inverse(&self) -> Self {
        Self::new(self.data.iter().map(|x| -x.clone()))
    }
}

impl<F, B> Mul<F> for NormalBasisExtension<F, B>
where
    F: FiniteField + Clone,
    B: NormalBasis<F>,
{
    type Output = Self;
    fn mul(self, other: F) -> Self {
        Self::new(self.data.into_iter().map(|x| x * other.clone()))
    }
}

impl<F, B> Mul for NormalBasisExtension<F, B>
where
    F: FiniteField + Clone,
    B: NormalBasis<F>,
{
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        let mut x = Self::zero();
        let n = self.data.len();
        for j in 0..n {
            let mut m = B::mul(j);
            for i in 0..n - j {
                if j == 0 {
                    x += Self::new(m.clone()) * (self.data[i].clone() * other.data[i].clone());
                } else {
                    x += Self::new(m.clone())
                        * (self.data[i].clone() * other.data[i + j].clone()
                            + self.data[i + j].clone() * other.data[i].clone());
                }
                m.rotate_right(1);
            }
        }
        x
    }
}

impl<F, B> TwoSidedInverse<Multiplicative> for NormalBasisExtension<F, B>
where
    F: FiniteField + Clone,
    B: NormalBasis<F>,
{
    fn two_sided_inverse(&self) -> Self {
        let mut m = B::degree_extension();
        assert!(m > 1);
        let mut n = 0;
        let mut t = 1;
        let a = self.clone();
        let mut b = a.clone().shift_right(1);
        let mut c = b.clone();
        while m > 0 {
            if m & 1 > 0 {
                c = c * b.clone().shift_right(n);
                n += t;
            }
            b = b.clone() * b.shift_right(t);
            m >>= 1;
            t <<= 1;
        }
        assert_eq!(n, B::degree_extension());
        let c_ = (c.clone() * a)
            .try_lower()
            .expect("should be a subfield element");
        let c_ = Self::new(vec![B::unit() / c_; B::degree_extension()]);
        c * c_
    }
}

impl<F, B> Div for NormalBasisExtension<F, B>
where
    F: FiniteField + Clone,
    B: NormalBasis<F>,
{
    type Output = Self;
    fn div(self, other: Self) -> Self {
        self * <_ as TwoSidedInverse<Multiplicative>>::two_sided_inverse(&other)
    }
}

impl<F, B> AddAssign for NormalBasisExtension<F, B>
where
    F: FiniteField,
{
    fn add_assign(&mut self, other: Self) {
        for (a, b) in self.data.iter_mut().zip(other.data) {
            *a += b;
        }
    }
}

impl<F, B> SubAssign for NormalBasisExtension<F, B>
where
    F: FiniteField,
{
    fn sub_assign(&mut self, other: Self) {
        for (a, b) in self.data.iter_mut().zip(other.data) {
            *a -= b;
        }
    }
}

impl<F, B> MulAssign for NormalBasisExtension<F, B>
where
    F: FiniteField + Clone,
    B: NormalBasis<F>,
{
    fn mul_assign(&mut self, other: Self) {
        *self = self.clone() * other;
    }
}

impl<F, B> DivAssign for NormalBasisExtension<F, B>
where
    F: FiniteField + Clone,
    B: NormalBasis<F>,
{
    fn div_assign(&mut self, other: Self) {
        *self = self.clone() / other;
    }
}

impl<F, B> Neg for NormalBasisExtension<F, B>
where
    F: FiniteField,
{
    type Output = Self;
    fn neg(mut self) -> Self {
        let data = take(&mut self.data).into_iter().map(|x| -x).collect();
        self.data = data;
        self
    }
}

impl<F, B> One for NormalBasisExtension<F, B>
where
    F: FiniteField + Clone,
    B: NormalBasis<F>,
{
    fn one() -> Self {
        Self::new(vec![B::unit(); B::degree_extension()])
    }
}

impl<F, B> AbstractMagma<Additive> for NormalBasisExtension<F, B>
where
    F: FiniteField,
    B: NormalBasis<F>,
{
    #[inline]
    fn operate(&self, right: &Self) -> Self {
        self.clone() + right.clone()
    }
}

impl<F, B> AbstractMagma<Multiplicative> for NormalBasisExtension<F, B>
where
    F: FiniteField,
    B: NormalBasis<F>,
{
    #[inline]
    fn operate(&self, right: &Self) -> Self {
        self.clone() * right.clone()
    }
}

impl<F, B> Identity<Additive> for NormalBasisExtension<F, B>
where
    F: FiniteField,
    B: NormalBasis<F>,
{
    fn identity() -> Self {
        Self::new(vec![F::zero(); B::degree_extension()])
    }
}

impl<F, B> Identity<Multiplicative> for NormalBasisExtension<F, B>
where
    F: FiniteField,
    B: NormalBasis<F>,
{
    fn identity() -> Self {
        Self::new(vec![B::unit(); B::degree_extension()])
    }
}

impl<F, B> AbstractField for NormalBasisExtension<F, B>
where
    F: FiniteField,
    B: NormalBasis<F>,
{
}

impl<F, B> AbstractRingCommutative for NormalBasisExtension<F, B>
where
    F: FiniteField,
    B: NormalBasis<F>,
{
}

impl<F, B> AbstractRing for NormalBasisExtension<F, B>
where
    F: FiniteField,
    B: NormalBasis<F>,
{
}

impl<F, B> AbstractGroupAbelian<Multiplicative> for NormalBasisExtension<F, B>
where
    F: FiniteField,
    B: NormalBasis<F>,
{
}

impl<F, B> AbstractGroup<Multiplicative> for NormalBasisExtension<F, B>
where
    F: FiniteField,
    B: NormalBasis<F>,
{
}

impl<F, B> AbstractLoop<Multiplicative> for NormalBasisExtension<F, B>
where
    F: FiniteField,
    B: NormalBasis<F>,
{
}

impl<F, B> AbstractQuasigroup<Multiplicative> for NormalBasisExtension<F, B>
where
    F: FiniteField,
    B: NormalBasis<F>,
{
}

impl<F, B> AbstractMonoid<Multiplicative> for NormalBasisExtension<F, B>
where
    F: FiniteField,
    B: NormalBasis<F>,
{
}

impl<F, B> AbstractSemigroup<Multiplicative> for NormalBasisExtension<F, B>
where
    F: FiniteField,
    B: NormalBasis<F>,
{
}

impl<F, B> AbstractGroupAbelian<Additive> for NormalBasisExtension<F, B>
where
    F: FiniteField,
    B: NormalBasis<F>,
{
}

impl<F, B> AbstractGroup<Additive> for NormalBasisExtension<F, B>
where
    F: FiniteField,
    B: NormalBasis<F>,
{
}

impl<F, B> AbstractLoop<Additive> for NormalBasisExtension<F, B>
where
    F: FiniteField,
    B: NormalBasis<F>,
{
}

impl<F, B> AbstractQuasigroup<Additive> for NormalBasisExtension<F, B>
where
    F: FiniteField,
    B: NormalBasis<F>,
{
}

impl<F, B> AbstractMonoid<Additive> for NormalBasisExtension<F, B>
where
    F: FiniteField,
    B: NormalBasis<F>,
{
}

impl<F, B> AbstractSemigroup<Additive> for NormalBasisExtension<F, B>
where
    F: FiniteField,
    B: NormalBasis<F>,
{
}

impl<F, B> FiniteField for NormalBasisExtension<F, B>
where
    F: FiniteField + Clone,
    B: NormalBasis<F>,
{
    type Scalar = F;
    fn characteristic<T: ConstructibleNumber>() -> T {
        F::characteristic()
    }
    fn degree_extension<T: ConstructibleNumber>() -> T {
        int_inj(B::degree_extension())
    }
    fn to_vec(&self) -> Vec<F> {
        self.data.clone()
    }
    fn from_scalar(scalar: Self::Scalar) -> Self {
        Self {
            data: vec![B::unit() * scalar; B::degree_extension()],
            _p: PhantomData,
        }
    }
    fn frobenius_base(self) -> Self {
        self.shift_right(1)
    }
    fn field_size<T: ConstructibleNumber>() -> T {
        crate::pow(Self::Scalar::field_size(), B::degree_extension())
    }
    fn try_lower(mut self) -> Option<Self::Scalar> {
        for x in &self.data {
            if x != &self.data[0] {
                return None;
            }
        }
        Some(self.data.pop().expect("vec should not be empty"))
    }
    fn basis_elements() -> Vec<Self> {
        let deg_ext = B::degree_extension();
        (0..deg_ext)
            .map(|i| {
                let mut data = vec![F::zero(); deg_ext];
                data[i] = F::one();
                Self {
                    data,
                    _p: PhantomData,
                }
            })
            .collect()
    }
}

impl<F, B> ArbitraryElement for NormalBasisExtension<F, B>
where
    F: FiniteField + ArbitraryElement,
    B: NormalBasis<F>,
{
    fn arbitrary<R: RngCore>(rng: &mut R) -> Self {
        Self::new(repeat_with(|| F::arbitrary(rng)).take(B::degree_extension()))
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug)]
pub struct GF2561D_N2;

impl NormalBasis<GF2561D> for GF2561D_N2 {
    fn unit() -> GF2561D {
        GF2561D(1)
    }
    fn degree_extension() -> usize {
        2
    }
    fn multiplier(a: usize) -> Vec<GF2561D> {
        lazy_static::lazy_static! {
            static ref TABLE: Vec<Vec<GF2561D>> = vec![
                vec![GF2561D(248), GF2561D(249)],
                vec![GF2561D(249), GF2561D(249)],
            ];
        }
        TABLE[a].clone()
    }
}

pub type GF65536N = NormalBasisExtension<GF2561D, GF2561D_N2>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn irreducible_poly_deg_2_over_gf2561d() {
        let p = Polynomial::new(vec![GF2561D(2), GF2561D(3), GF2561D(1)]); // 00000001 x^2 + 00000001 x^1 + 00000010 x^0
        let mut x_superpower = Polynomial::new(vec![GF2561D(0), GF2561D(1)]);
        for _ in 0..16 {
            x_superpower *= x_superpower.clone();
            let (_, r) = x_superpower.div_with_rem(p.clone());
            x_superpower = r;
        }
        x_superpower = x_superpower + Polynomial::new(vec![GF2561D(0), GF2561D(1)]);
        let p = x_superpower.gcd(p);
        println!("{}", p);
    }

    #[test]
    #[ignore]
    fn as_form_forbidden_const() {
        let mut h = std::collections::BTreeSet::new();
        for i in 0..=255u8 {
            let GF2561D(x) = pow(GF2561D(i), 2) - GF2561D(i);
            h.insert(x);
        }
        for x in h {
            println!("{}", x);
        }
    }

    #[test]
    fn normal_basis_of_gf65536p() {
        let alpha = GF2561D(2);
        let beta: GF65536P = artin_candidate(alpha);
        if test_normal_basis(beta.clone()) {
            println!(
                "yay, artin_candidate({}) works as a normal basis of GF65536",
                alpha
            );
        }
        let (one, table) = compute_normal_basis_multiplication_table(beta);
        println!("one=");
        for x in one {
            print!(" {:?}", x);
        }
        println!("");
        for (i, x) in table.into_iter().enumerate() {
            print!("{}:", i);
            for y in x {
                print!(" {:?}", y)
            }
            println!("")
        }
    }
}

#[cfg(test)]
mod gf65536p_tests {
    use super::*;

    use quickcheck::{Arbitrary, Gen};

    impl<F, P> Arbitrary for PolynomialExtension<F, P>
    where
        F: Arbitrary + FiniteField,
        P: MonicPolynomial<F> + 'static,
    {
        fn arbitrary<G: Gen>(g: &mut G) -> Self {
            Self {
                data: Polynomial::new((0..P::degree()).map(|_| F::arbitrary(g))),
                _p: PhantomData,
            }
        }
    }

    #[quickcheck]
    fn prop_mul_and_add_are_distributive(args: (GF65536P, GF65536P, GF65536P)) -> bool {
        GF65536P::prop_mul_and_add_are_distributive(args)
    }

    #[quickcheck]
    fn prop_mul_is_commutative(args: (GF65536P, GF65536P)) -> bool {
        <GF65536P as AbstractGroupAbelian<Multiplicative>>::prop_is_commutative(args)
    }

    #[quickcheck]
    fn prop_mul_inv_is_latin_square(args: (GF65536P, GF65536P)) -> bool {
        args.0.is_zero()
            || args.1.is_zero()
            || <GF65536P as AbstractQuasigroup<Multiplicative>>::prop_inv_is_latin_square(args)
    }
}

pub trait ExtensionTower {
    type Super: FiniteField;
    type Bottom: FiniteField;
    fn to_vec(f: Self::Super) -> Vec<Self::Bottom>;
    fn degree_extension<T: ConstructibleNumber>() -> T;
    fn try_into_bottom(x: Self::Super) -> Option<Self::Bottom>;
    fn into_super(x: Self::Bottom) -> Self::Super;
    fn basis_elements_over_bottom() -> Vec<Self::Super>;
}

#[derive(Clone, Copy, Debug)]
pub struct FiniteExtensionTower<A, B> {
    _p: PhantomData<fn() -> (A, B)>,
}

#[derive(Clone, Copy, Debug)]
pub struct BottomField;

impl<A> ExtensionTower for FiniteExtensionTower<A, BottomField>
where
    A: FiniteField<Scalar = A>,
{
    type Super = A;
    type Bottom = A;
    fn to_vec(f: Self::Super) -> Vec<Self::Bottom> {
        vec![f]
    }
    fn degree_extension<T: ConstructibleNumber>() -> T {
        T::one()
    }
    fn try_into_bottom(x: A) -> Option<A> {
        Some(x)
    }
    fn into_super(x: A) -> A {
        x
    }
    fn basis_elements_over_bottom() -> Vec<A> {
        vec![A::one()]
    }
}

impl<A, B> ExtensionTower for FiniteExtensionTower<A, B>
where
    A: FiniteField<Scalar = B::Super>,
    B: ExtensionTower,
{
    type Super = A;
    type Bottom = B::Bottom;
    fn to_vec(f: Self::Super) -> Vec<Self::Bottom> {
        f.to_vec().into_iter().flat_map(|x| B::to_vec(x)).collect()
    }
    fn degree_extension<T: ConstructibleNumber>() -> T {
        B::degree_extension::<T>() * A::degree_extension()
    }
    fn try_into_bottom(x: Self::Super) -> Option<Self::Bottom> {
        let x = x.try_lower()?;
        B::try_into_bottom(x)
    }
    fn into_super(x: Self::Bottom) -> Self::Super {
        Self::Super::from_scalar(B::into_super(x))
    }
    fn basis_elements_over_bottom() -> Vec<A>
    where
        A: Clone,
        B::Bottom: Clone,
    {
        let mut bs = vec![];
        let lower_bases = B::basis_elements_over_bottom();
        for b in A::basis_elements() {
            for b_ in &lower_bases {
                bs.push(b.clone() * A::from_scalar(b_.clone()));
            }
        }
        bs
    }
}

pub type GF65536NTower = FiniteExtensionTower<
    GF65536N,
    FiniteExtensionTower<GF2561D, FiniteExtensionTower<F2, BottomField>>,
>;

#[cfg(test)]
mod gf65536n_tests {
    use super::*;

    use quickcheck::{Arbitrary, Gen};

    impl<F, B> Arbitrary for NormalBasisExtension<F, B>
    where
        F: Arbitrary + FiniteField,
        B: NormalBasis<F> + 'static,
    {
        fn arbitrary<G: Gen>(g: &mut G) -> Self {
            Self::new((0..B::degree_extension()).map(|_| F::arbitrary(g)))
        }
    }

    #[quickcheck]
    fn prop_mul_and_add_are_distributive(args: (GF65536N, GF65536N, GF65536N)) -> bool {
        GF65536N::prop_mul_and_add_are_distributive(args)
    }

    #[quickcheck]
    fn prop_mul_is_commutative(args: (GF65536N, GF65536N)) -> bool {
        <GF65536N as AbstractGroupAbelian<Multiplicative>>::prop_is_commutative(args)
    }

    #[quickcheck]
    fn prop_mul_inv_is_latin_square((a, b): (GF65536N, GF65536N)) -> bool {
        println!("testing {:?} {:?}", a, b);
        a.is_zero()
            || b.is_zero()
            || <GF65536N as AbstractQuasigroup<Multiplicative>>::prop_inv_is_latin_square((a, b))
    }

    #[quickcheck]
    fn prop_proper_field((a, b): (GF65536N, GF65536N)) -> bool {
        println!("testing {:?} {:?}", a, b);
        a.is_zero() || b.is_zero() || !(a * b).is_zero()
    }

    #[test]
    fn prop_ext_tower()
    where
        GF65536NTower: ExtensionTower,
    {
    }
}
