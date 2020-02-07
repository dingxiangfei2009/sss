/// More Galois Field constructions and arithmetics
use std::{
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
use serde::{Deserialize, Serialize};

use crate::{
    adapter::Int,
    field::{ConstructibleNumber, FiniteField, GF2561D},
    linalg::solve,
    pow, Coord, EuclideanDomain, Polynomial,
};

pub fn compute_normal_basis_multiplication_table<F>(base_generator: F) -> Vec<Vec<F::Scalar>>
where
    F: Clone + FiniteField,
    F::Scalar: Send + Sync,
{
    let m = F::degree_extension();
    // compute normal basis elements
    let mut betas = vec![];
    let beta = base_generator.clone();
    let mut beta_ = beta.clone();
    for _ in 0..m {
        betas.push(beta_.to_vec());
        beta_ = beta_.frobenius_base();
    }

    // suffice to compute beta^(1+q^i) where q=|F|, i < floor(m / 2)
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
    table
}

/*
0 1 2 3 4 5 6 7
1 0 0 0 0 1 0 0
1 0 0 1 0 0 0 0
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
            a * b
        } else {
            let Polynomial(b) = b;
            let mut c = <_>::default();
            for b in b {
                c = c + a.clone() * b;
                a = Self::reduce(a * Polynomial(vec![F::zero(), F::one()]));
            }
            c
        }
    }
    fn inv(a: Polynomial<F>) -> Polynomial<F> {
        let (a, _, g): (Polynomial<F>, Polynomial<F>, Polynomial<F>) =
            Self::reduce(a).extended_gcd(Self::repr());
        assert!(!g.is_zero(), "division by zero polynomial");
        assert!(g.degree() == 0, "deg g={} is not zero", g.degree());
        Self::reduce(a / g.0[0].clone())
    }
    fn gcd(a: Polynomial<F>) -> (Polynomial<F>, Polynomial<F>, Polynomial<F>) {
        Self::reduce(a).extended_gcd(Self::repr())
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PolynomialExtension<F, P> {
    data: Polynomial<F>,
    _p: PhantomData<fn() -> P>,
}

impl<F, P> Eq for PolynomialExtension<F, P> where F: FiniteField {}

impl<F, P> PartialEq for PolynomialExtension<F, P>
where
    F: FiniteField,
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
    F: Zero + Clone,
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
    F: Zero + Clone + Sub<Output = F>,
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
    F: Zero + Clone,
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
    F: FiniteField + Clone,
    P: MonicPolynomial<F>,
{
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self {
            data: P::reduce_mul(self.data, other.data),
            _p: PhantomData,
        }
    }
}

impl<F, P> One for PolynomialExtension<F, P>
where
    F: FiniteField + Clone,
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
    F: FiniteField + Clone,
    P: MonicPolynomial<F>,
{
    fn mul_assign(&mut self, rhs: Self) {
        let data = take(&mut self.data);
        self.data = P::reduce_mul(data, rhs.data);
    }
}

impl<F, P> AddAssign for PolynomialExtension<F, P>
where
    F: FiniteField + Clone,
    P: MonicPolynomial<F>,
{
    fn add_assign(&mut self, other: Self) {
        let data = take(&mut self.data);
        self.data = data + other.data;
    }
}

impl<F, P> SubAssign for PolynomialExtension<F, P>
where
    F: FiniteField + Clone,
    P: MonicPolynomial<F>,
{
    fn sub_assign(&mut self, other: Self) {
        let data = take(&mut self.data);
        self.data = data - other.data;
    }
}

impl<F, P> Neg for PolynomialExtension<F, P>
where
    F: FiniteField + Clone,
    P: MonicPolynomial<F>,
{
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            data: -self.data,
            _p: PhantomData,
        }
    }
}

impl<F, P> Div for PolynomialExtension<F, P>
where
    F: FiniteField + Clone,
    P: MonicPolynomial<F>,
{
    type Output = Self;
    fn div(self, other: Self) -> Self {
        self * TwoSidedInverse::<Multiplicative>::two_sided_inverse(&other)
    }
}

impl<F, P> DivAssign for PolynomialExtension<F, P>
where
    F: FiniteField + Clone,
    P: MonicPolynomial<F>,
{
    fn div_assign(&mut self, other: Self) {
        let data = take(&mut self.data);
        self.data = data * TwoSidedInverse::<Multiplicative>::two_sided_inverse(&other).data;
    }
}

impl<F, P> AbstractMagma<Additive> for PolynomialExtension<F, P>
where
    F: FiniteField,
    P: MonicPolynomial<F>,
{
    #[inline]
    fn operate(&self, right: &Self) -> Self {
        self.clone() + right.clone()
    }
}

impl<F, P> AbstractMagma<Multiplicative> for PolynomialExtension<F, P>
where
    F: FiniteField,
    P: MonicPolynomial<F>,
{
    #[inline]
    fn operate(&self, right: &Self) -> Self {
        self.clone() * right.clone()
    }
}

impl<F, P> TwoSidedInverse<Additive> for PolynomialExtension<F, P>
where
    F: FiniteField,
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
    F: FiniteField,
    P: MonicPolynomial<F>,
{
    fn two_sided_inverse(&self) -> Self {
        Self {
            data: P::inv(self.data.clone()),
            _p: PhantomData,
        }
    }
}

impl<F, P> Identity<Additive> for PolynomialExtension<F, P>
where
    F: FiniteField,
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
    F: FiniteField,
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
    F: FiniteField,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractRingCommutative for PolynomialExtension<F, P>
where
    F: FiniteField,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractRing for PolynomialExtension<F, P>
where
    F: FiniteField,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractGroupAbelian<Additive> for PolynomialExtension<F, P>
where
    F: FiniteField,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractGroupAbelian<Multiplicative> for PolynomialExtension<F, P>
where
    F: FiniteField,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractGroup<Additive> for PolynomialExtension<F, P>
where
    F: FiniteField,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractGroup<Multiplicative> for PolynomialExtension<F, P>
where
    F: FiniteField,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractMonoid<Additive> for PolynomialExtension<F, P>
where
    F: FiniteField,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractMonoid<Multiplicative> for PolynomialExtension<F, P>
where
    F: FiniteField,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractLoop<Additive> for PolynomialExtension<F, P>
where
    F: FiniteField,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractLoop<Multiplicative> for PolynomialExtension<F, P>
where
    F: FiniteField,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractSemigroup<Additive> for PolynomialExtension<F, P>
where
    F: FiniteField,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractSemigroup<Multiplicative> for PolynomialExtension<F, P>
where
    F: FiniteField,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractQuasigroup<Additive> for PolynomialExtension<F, P>
where
    F: FiniteField,
    P: MonicPolynomial<F>,
{
}

impl<F, P> AbstractQuasigroup<Multiplicative> for PolynomialExtension<F, P>
where
    F: FiniteField,
    P: MonicPolynomial<F>,
{
}

impl<F, P> Mul<F> for PolynomialExtension<F, P>
where
    F: FiniteField + Clone,
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
    fn characteristic() -> usize {
        F::characteristic()
    }
    fn degree_extension() -> usize {
        P::degree()
    }
    fn to_vec(&self) -> Vec<F> {
        let mut data = self.data.clone().0;
        data.resize(Self::degree_extension(), F::zero());
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
        crate::pow(Self::Scalar::field_size(), Self::degree_extension())
    }
}

impl<F, P> PolynomialExtension<F, P>
where
    F: FiniteField,
    P: MonicPolynomial<F>,
{
    pub fn root() -> Self {
        Self {
            data: P::reduce(Polynomial(vec![F::zero(), F::one()])),
            _p: PhantomData,
        }
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
#[derive(Debug)]
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

pub fn test_normal_basis<F, M, N>(alpha: F) -> bool
where
    F: FiniteField<M, N> + Clone,
    N: Into<usize>,
{
    let deg_ext = F::degree_extension().into();
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

pub struct NormalBasisExtension<F, B> {
    data: Vec<F>,
    _p: PhantomData<fn() -> B>,
}

#[cfg(test)]
mod tests {
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
        for (i, x) in compute_normal_basis_multiplication_table(beta)
            .into_iter()
            .enumerate()
        {
            print!("{}:", i);
            for y in x {
                print!(" {:?}", y)
            }
            println!("")
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
