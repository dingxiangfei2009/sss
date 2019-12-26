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
        .map(|x| p.eval_at(x))
        .take(threshold)
        .collect();
    let r = Polynomial::from_coords(q.as_slice());
    assert_eq!(u.as_slice(), r.0.as_slice());

    let q: Vec<_> = (1u8..=u.len() as u8 + 1)
        .map(GF2561D)
        .map(|x| p.eval_at(x))
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
        .map(|x| p.eval_at(x))
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
        .map(|x| p.eval_at(x))
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
    let mut g = vec![GF2561D::zero(); GF2561D::degree_extension() + 1];
    g[GF2561D::degree_extension()] = GF2561D::one() / GF2561D(0b00010011);
    g[0] -= GF2561D::one() / GF2561D(0b00010011);
    let g = Polynomial::new(g);

    let mut beta = gamma.clone();
    let mut p = vec![];
    for _ in 0..GF2561D::degree_extension() {
        p.push(beta.clone());
        beta = pow(beta, GF2561D::characteristic());
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
            .map(Frac),
    );
    let q = Polynomial::new(
        vec![3, 8, 4]
            .into_iter()
            .map(BigInt::from)
            .map(BigRational::from_integer)
            .map(Frac),
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
            gamma = pow(gamma, GF2561D::characteristic());
        }
        basis
    };
    type Scalar = <GF2561D as FiniteField>::Scalar;
    let basis: Vec<_> = basis
        .into_iter()
        .flat_map(|beta| beta.to_vec())
        .chain(vec![Scalar::zero(); deg_ext])
        .collect();
    let basis = ndarray::ArrayView::from_shape((deg_ext + 1, deg_ext), &basis)
        .expect("shape should be correct");
    let inv: Vec<_> = (0..deg_ext)
        .into_par_iter()
        .flat_map(|i| {
            let mut basis = basis.t().to_owned();
            basis[[i, deg_ext]] = Scalar::one();
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
        GF2561D::degree_extension(),
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
fn gf2561d_255th_cyclotomic_coset() {
    assert_eq!(
        vec![
            vec![0],
            vec![1, 2, 4, 8, 16, 32, 64, 128],
            vec![3, 6, 12, 24, 48, 96, 192, 129],
            vec![5, 10, 20, 40, 80, 160, 65, 130],
            vec![7, 14, 28, 56, 112, 224, 193, 131],
            vec![9, 18, 36, 72, 144, 33, 66, 132],
            vec![11, 22, 44, 88, 176, 97, 194, 133],
            vec![13, 26, 52, 104, 208, 161, 67, 134],
            vec![15, 30, 60, 120, 240, 225, 195, 135],
            vec![17, 34, 68, 136],
            vec![19, 38, 76, 152, 49, 98, 196, 137],
            vec![21, 42, 84, 168, 81, 162, 69, 138],
            vec![23, 46, 92, 184, 113, 226, 197, 139],
            vec![25, 50, 100, 200, 145, 35, 70, 140],
            vec![27, 54, 108, 216, 177, 99, 198, 141],
            vec![29, 58, 116, 232, 209, 163, 71, 142],
            vec![31, 62, 124, 248, 241, 227, 199, 143],
            vec![37, 74, 148, 41, 82, 164, 73, 146],
            vec![39, 78, 156, 57, 114, 228, 201, 147],
            vec![43, 86, 172, 89, 178, 101, 202, 149],
            vec![45, 90, 180, 105, 210, 165, 75, 150],
            vec![47, 94, 188, 121, 242, 229, 203, 151],
            vec![51, 102, 204, 153],
            vec![53, 106, 212, 169, 83, 166, 77, 154],
            vec![55, 110, 220, 185, 115, 230, 205, 155],
            vec![59, 118, 236, 217, 179, 103, 206, 157],
            vec![61, 122, 244, 233, 211, 167, 79, 158],
            vec![63, 126, 252, 249, 243, 231, 207, 159],
            vec![85, 170],
            vec![87, 174, 93, 186, 117, 234, 213, 171],
            vec![91, 182, 109, 218, 181, 107, 214, 173],
            vec![95, 190, 125, 250, 245, 235, 215, 175],
            vec![111, 222, 189, 123, 246, 237, 219, 183],
            vec![119, 238, 221, 187],
            vec![127, 254, 253, 251, 247, 239, 223, 191]
        ],
        compute_cyclotomic_cosets::<GF2561D>(255)
    )
}

#[test]
fn gf2561d_17th_cyclotomic_coset() {
    assert_eq!(
        vec![
            vec![0],
            vec![1, 2, 4, 8, 16, 15, 13, 9],
            vec![3, 6, 12, 7, 14, 11, 5, 10]
        ],
        compute_cyclotomic_cosets::<GF2561D>(17)
    )
}

#[test]
fn gf2561d_subfield_16_normal_basis() {
    let subfield_deg_ext = 4;
    let gamma = search_normal_basis::<GF2561D, GF2561DG2>(subfield_deg_ext);
    println!("gamma={}", gamma);
    let mut beta = gamma;
    for _ in 0..subfield_deg_ext {
        print!("{} ", beta);
        beta = pow(beta, GF2561D::characteristic());
    }
    println!();
    assert_eq!(beta, gamma);
    assert_eq!(gamma, GF2561D(0b00001010));
}

#[test]
fn gf2561d_subfield_4_normal_basis() {
    let subfield_deg_ext = 2;
    let gamma = search_normal_basis::<GF2561D, GF2561DG2>(subfield_deg_ext);
    println!("gamma={}", gamma);
    let mut beta = gamma;
    for _ in 0..subfield_deg_ext {
        print!("{} ", beta);
        beta = pow(beta, GF2561D::characteristic());
    }
    println!();
    assert_eq!(beta, gamma);
    assert_eq!(gamma, GF2561D(0b11010110));
}

#[test]
fn usize_euclid() {
    use crate::conv::R;
    let (s, t, r) = usize::extended_gcd::<R, R>(240, 46);
    assert_eq!(s, R(-9));
    assert_eq!(t, R(47));
    assert_eq!(r, 2);
}
