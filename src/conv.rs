use std::{
    ops::{Add, Deref, Mul, Neg, Sub},
    sync::Arc,
};

use alga::general::Field;
use ndarray::{Array1, Array2, ArrayView2, Axis, Zip};
use num::{One, Zero};

use crate::{
    linalg::{mat_mat_mul, mat_vec_mul},
    EuclideanDomain,
};

pub fn toeplitz_d(n: usize) -> Array2<i8> {
    assert!(n > 1 && n & 1 > 0);
    let k = n >> 1;
    stack![
        Axis(0),
        stack![Axis(1), -Array2::eye(k), Array2::zeros((k, k + 1))],
        Array2::ones((1, n)),
        stack![Axis(1), Array2::zeros((k, k + 1)), -Array2::eye(k)]
    ]
}

#[allow(clippy::many_single_char_names)] // REASON: match up with symbols in the wu2012 paper
pub fn toeplitz_e(n: usize) -> (usize, Array2<i8>) {
    assert!(n > 0);
    let k = n >> 1;
    if n == 1 {
        (1, Array2::eye(1))
    } else if n & 1 > 0 {
        let (m, e) = toeplitz_e(k);
        let d = toeplitz_d(n);
        let e = stack![
            Axis(1),
            stack![Axis(0), e, Array2::zeros((1, m)), e],
            stack![Axis(0), e, Array2::zeros((k + 1, m))],
            d,
            stack![Axis(0), Array2::zeros((k + 1, m)), e]
        ];
        let m = 3 * m + n;
        assert_eq!((n, m), e.dim());
        (m, e)
    } else {
        let (m, e) = toeplitz_e(k);
        let e = stack![
            Axis(1),
            stack![Axis(0), e, e],
            stack![Axis(0), e, Array2::zeros((k, m))],
            stack![Axis(0), Array2::zeros((k, m)), e]
        ];
        let m = 3 * m;
        assert_eq!((n, m), e.dim());
        (m, e)
    }
}

#[allow(clippy::many_single_char_names)] // REASON: match up with symbols in the wu2012 paper
pub fn toeplitz_gh<F>(n: usize) -> (usize, Array2<F>, Array2<i8>)
where
    F: Zero + One + Clone + Sub<Output = F> + Mul<Output = F> + Neg<Output = F>,
{
    assert!(n > 0);
    let k = n >> 1;
    if n == 1 {
        (1, Array2::eye(1), Array2::eye(1))
    } else if n & 1 > 0 {
        let (m, g, h) = toeplitz_gh::<F>(k);
        let d_ = toeplitz_d(n);
        let d = d_.map(|x| {
            if *x > 0 {
                F::one()
            } else if *x < 0 {
                -F::one()
            } else {
                F::zero()
            }
        });
        #[allow(non_snake_case)]
        let q__1 = stack![Axis(1), Array2::eye(n - 2), Array2::zeros((n - 2, n + 1))];
        let q_0 = stack![
            Axis(1),
            Array2::zeros((n - 2, k + 1)),
            Array2::eye(n - 2),
            Array2::zeros((n - 2, k + 1))
        ];
        let q_1 = stack![Axis(1), Array2::zeros((n - 2, n + 1)), Array2::eye(n - 2)];
        let mut g_0 = Array2::zeros((m, 2 * n - 1));
        let mut g_1 = Array2::zeros((m, 2 * n - 1));
        for i in 0..m {
            if let Some(Some(j)) =
                h.index_axis(Axis(0), i)
                    .indexed_iter()
                    .fold(None, |s, (j, h)| match s {
                        None if *h != 0 => Some(Some(j)),
                        Some(_) if *h != 0 => Some(None),
                        s => s,
                    })
            {
                g_0[[i, 2 * k - j - 1]] = F::one();
                g_1[[i, 3 * k - j]] = F::one();
            }
        }
        let g = stack![
            Axis(0),
            mat_mat_mul(&g, &q_0),
            mat_mat_mul(&g, &(q__1 - q_0.clone())) + g_0,
            stack![Axis(1), Array2::zeros((n, k)), d, Array2::zeros((n, k))],
            mat_mat_mul(&g, &(q_1 - q_0)) + g_1
        ];
        let mut d_ = d_.t();
        d_.invert_axis(Axis(1));
        let h = stack![
            Axis(0),
            stack![Axis(1), h.clone(), Array2::zeros((m, 1)), h.clone()],
            stack![Axis(1), Array2::zeros((m, k + 1)), h.clone()],
            d_,
            stack![Axis(1), h.clone(), Array2::zeros((m, k + 1))]
        ];
        let m = 3 * m + n;
        assert_eq!((m, 2 * n - 1), g.dim());
        assert_eq!((m, n), h.dim());
        (m, g, h)
    } else {
        let (m, g, h) = toeplitz_gh::<F>(k);
        #[allow(non_snake_case)]
        let p__1 = stack![Axis(1), Array2::eye(n - 1), Array2::zeros((n - 1, n))];
        let p_0 = stack![
            Axis(1),
            Array2::zeros((n - 1, k)),
            Array2::eye(n - 1),
            Array2::zeros((n - 1, k))
        ];
        let p_1 = stack![Axis(1), Array2::zeros((n - 1, n)), Array2::eye(n - 1)];
        let g = stack![
            Axis(0),
            mat_mat_mul(&g, &p_0),
            mat_mat_mul(&g, &(p__1 - p_0.clone())),
            mat_mat_mul(&g, &(p_1 - p_0))
        ];
        let h = stack![
            Axis(0),
            stack![Axis(1), h, h],
            stack![Axis(1), Array2::zeros((m, k)), h],
            stack![Axis(1), h, Array2::zeros((m, k))]
        ];
        let m = 3 * m;
        assert_eq!((m, 2 * n - 1), g.dim());
        assert_eq!((m, n), h.dim());
        (m, g, h)
    }
}

pub fn toeplitz_unit_mul<'a, F>(u: impl Into<ArrayView2<'a, i8>>, v: Array1<F>) -> Array1<F>
where
    F: 'a + Zero + Clone + Sub<Output = F> + Mul<Output = F>,
{
    u.into()
        .axis_iter(Axis(0))
        .map(|r| {
            Zip::from(r).and(&v).fold(F::zero(), |a, &u, v| {
                if u > 0 {
                    a + v.clone()
                } else if u < 0 {
                    a - v.clone()
                } else {
                    a
                }
            })
        })
        .collect()
}

pub trait LinearOperator<F> {
    fn op(&self, vec: &[F]) -> Vec<F>;
    fn source_dim(&self) -> usize;
    fn dest_dim(&self) -> usize;
}

pub struct GenericLinearOperator<F>(Box<dyn LinearOperator<F>>);

impl<F, T> LinearOperator<F> for T
where
    T: Deref,
    T::Target: LinearOperator<F>,
{
    fn op(&self, vec: &[F]) -> Vec<F> {
        <T::Target as LinearOperator<F>>::op(self.deref(), vec)
    }

    fn source_dim(&self) -> usize {
        <T::Target as LinearOperator<F>>::source_dim(self.deref())
    }

    fn dest_dim(&self) -> usize {
        <T::Target as LinearOperator<F>>::dest_dim(self.deref())
    }
}

pub trait BilinearAlgorithm<F: Send + Sync + Clone>
where
    F: 'static + Add<Output = F> + Mul<Output = F>,
{
    type A: LinearOperator<F> + Send + Sync;
    type B: 'static + LinearOperator<F> + Send + Sync;
    type C: 'static + LinearOperator<F> + Send + Sync;

    /// construction of the bilinear form algorithm
    fn make_algorithm(&self) -> (Self::A, Self::B, Self::C);

    fn apply(&self, x: &[F]) -> Box<dyn Send + Sync + Fn(&[F]) -> Vec<F>> {
        let (a, b, c) = self.make_algorithm();
        assert_eq!(a.dest_dim(), b.dest_dim());
        let x = a.op(x);
        Box::new(move |y| {
            let y = b.op(y);
            assert_eq!(x.len(), y.len());
            let z: Vec<_> = x
                .iter()
                .cloned()
                .zip(y.into_iter())
                .map(|(x, y)| x * y)
                .collect();
            c.op(&z)
        })
    }
}

/// type-erased bilinear algorithm
#[derive(Clone)]
pub struct GenericBilinearAlgorithm<F> {
    a: Arc<Box<dyn Send + Sync + LinearOperator<F>>>,
    b: Arc<Box<dyn Send + Sync + LinearOperator<F>>>,
    c: Arc<Box<dyn Send + Sync + LinearOperator<F>>>,
}

impl<F: Send + Sync + Clone> BilinearAlgorithm<F> for GenericBilinearAlgorithm<F>
where
    F: 'static + Add<Output = F> + Mul<Output = F>,
{
    type A = Arc<Box<dyn Send + Sync + LinearOperator<F>>>;
    type B = Arc<Box<dyn Send + Sync + LinearOperator<F>>>;
    type C = Arc<Box<dyn Send + Sync + LinearOperator<F>>>;
    fn make_algorithm(&self) -> (Self::A, Self::B, Self::C) {
        (
            Arc::clone(&self.a),
            Arc::clone(&self.b),
            Arc::clone(&self.c),
        )
    }
}

impl<F: Send + Sync + Clone> GenericBilinearAlgorithm<F> {
    pub fn new<A>(algo: &A) -> Self
    where
        A: 'static + BilinearAlgorithm<F>,
        F: 'static + Add<Output = F> + Mul<Output = F>,
    {
        let (a, b, c) = algo.make_algorithm();
        Self {
            a: Arc::new(Box::new(a) as Box<dyn Send + Sync + LinearOperator<F>>),
            b: Arc::new(Box::new(b) as Box<dyn Send + Sync + LinearOperator<F>>),
            c: Arc::new(Box::new(c) as Box<dyn Send + Sync + LinearOperator<F>>),
        }
    }
}

pub struct TensorProduct<A, B>(A, B);

impl<A, B, F> LinearOperator<F> for TensorProduct<A, B>
where
    A: LinearOperator<F>,
    B: LinearOperator<F>,
    F: Clone + Add<Output = F> + Mul<Output = F>,
{
    fn op(&self, vec: &[F]) -> Vec<F> {
        assert_eq!(vec.len(), self.source_dim());
        let m = self.0.source_dim();
        let n = self.1.source_dim();
        let mat = ArrayView2::from_shape((m, n), vec)
            .expect("shape should be correct")
            .axis_iter(Axis(1))
            .flat_map(|c| self.0.op(&c.to_vec()))
            .collect();
        let m = self.0.dest_dim();
        let mat: Vec<_> = Array2::from_shape_vec((n, m), mat)
            .expect("shape should be correct")
            .axis_iter(Axis(1))
            .flat_map(|c| self.1.op(&c.to_vec()))
            .collect();
        let n = self.1.dest_dim();
        assert_eq!(mat.len(), m * n);
        mat
    }

    fn source_dim(&self) -> usize {
        self.0.source_dim() * self.1.source_dim()
    }

    fn dest_dim(&self) -> usize {
        self.0.dest_dim() * self.1.dest_dim()
    }
}

pub struct LinearProduct<A, B>(A, B);

impl<A, B, F> LinearOperator<F> for LinearProduct<A, B>
where
    A: LinearOperator<F>,
    B: LinearOperator<F>,
    F: Clone + Add<Output = F> + Mul<Output = F>,
{
    fn op(&self, vec: &[F]) -> Vec<F> {
        assert_eq!(self.1.source_dim(), vec.len());
        assert_eq!(self.1.dest_dim(), self.0.source_dim());
        self.0.op(&self.1.op(vec))
    }

    fn source_dim(&self) -> usize {
        self.1.source_dim()
    }

    fn dest_dim(&self) -> usize {
        self.0.dest_dim()
    }
}

pub struct AgarwalCooley<A1, A2> {
    n1: usize,
    n2: usize,
    a1: A1,
    a2: A2,
    q: Permutation,
}

impl<A1, A2> AgarwalCooley<A1, A2> {
    #[allow(clippy::many_single_char_names)] // REASON: match up with symbols in the textbook
    pub fn new(n1: usize, a1: A1, n2: usize, a2: A2) -> Self {
        assert!(n1 > 1 && n2 > 1);
        let (R(s), R(t), r) = usize::extended_gcd(n1, n2);
        assert_eq!(r, 1);
        let (s, t) = (
            s.rem_euclid(n2 as isize) as usize,
            t.rem_euclid(n1 as isize) as usize,
        );
        let e1 = t * n2; // e1 mod n1 = 1, e1 mod n2 = 0
        let e2 = s * n1; // e2 mod n1 = 0, e2 mod n2 = 1

        let mut q = Vec::with_capacity(n1 * n2);
        let n = n1 * n2;
        q.resize(n, 0);
        for i in 0..n1 {
            for j in 0..n2 {
                q[i * n2 + j] = (i * e1 % n + j * e2 % n) % n;
            }
        }
        let q = Permutation(q);
        Self { n1, n2, a1, a2, q }
    }
}

impl<A1, A2, F: Send + Sync> BilinearAlgorithm<F> for AgarwalCooley<A1, A2>
where
    A1: BilinearAlgorithm<F>,
    A2: BilinearAlgorithm<F>,
    F: 'static + Clone + Add<Output = F> + Mul<Output = F>,
{
    type A = LinearProduct<TensorProduct<A1::A, A2::A>, Permutation>;
    type B = LinearProduct<TensorProduct<A1::B, A2::B>, Permutation>;
    type C = LinearProduct<Permutation, TensorProduct<A1::C, A2::C>>;
    fn make_algorithm(&self) -> (Self::A, Self::B, Self::C) {
        let Self { n1, n2, a1, a2, q } = self;
        let (n1, n2) = (*n1, *n2);
        let (a1, b1, c1) = a1.make_algorithm();
        let (a2, b2, c2) = a2.make_algorithm();
        assert_eq!(a1.source_dim(), n1);
        assert_eq!(b2.source_dim(), n2);
        (
            LinearProduct(TensorProduct(a1, a2), q.clone()),
            LinearProduct(TensorProduct(b1, b2), q.clone()),
            LinearProduct(q.invert(), TensorProduct(c1, c2)),
        )
    }
}

#[derive(Clone)]
pub struct Permutation(Vec<usize>);

impl Permutation {
    fn invert(&self) -> Self {
        let n = self.0.len();
        let mut r = Vec::with_capacity(n);
        r.resize(n, None);
        for (i, &j) in self.0.iter().enumerate() {
            assert!(r[j].is_none());
            r[j] = Some(i);
        }
        Permutation(r.into_iter().flatten().collect())
    }
}

impl<F: Clone> LinearOperator<F> for Permutation {
    fn op(&self, vec: &[F]) -> Vec<F> {
        let n = self.0.len();
        assert_eq!(vec.len(), n);
        let mut r = Vec::with_capacity(n);
        r.resize(n, None);
        for (i, &j) in self.0.iter().enumerate() {
            assert!(r[i].is_none());
            r[i] = Some(vec[j].clone())
        }
        r.into_iter().flatten().collect()
    }

    fn source_dim(&self) -> usize {
        self.0.len()
    }

    fn dest_dim(&self) -> usize {
        self.0.len()
    }
}

pub struct ConvToToeplitz(usize);

impl<F: Clone> LinearOperator<F> for ConvToToeplitz {
    fn op(&self, x: &[F]) -> Vec<F> {
        assert_eq!(x.len(), self.0);
        let mut r = x[1..].to_vec();
        r.extend_from_slice(x);
        r
    }

    fn source_dim(&self) -> usize {
        self.0
    }

    fn dest_dim(&self) -> usize {
        2 * self.0 - 1
    }
}

pub struct ToeplitzConv(pub usize);

impl<F: 'static + Field + Send + Sync> BilinearAlgorithm<F> for ToeplitzConv {
    type A = LinearProduct<ToeplitzG<F>, ConvToToeplitz>;
    type B = UnitMatrix;
    type C = UnitMatrix;

    #[allow(clippy::many_single_char_names)] // REASON: match up with symbols in the wu2012 paper
    fn make_algorithm(&self) -> (Self::A, Self::B, Self::C) {
        let Self(n) = *self;
        let (m, e) = toeplitz_e(n);
        let (_, g, h) = toeplitz_gh(n);
        let e = {
            let mut add = vec![];
            let mut sub = vec![];
            for i in 0..n {
                let mut add_ = vec![];
                let mut sub_ = vec![];
                for j in 0..m {
                    let v = e[[i, j]];
                    if v > 0 {
                        add_.push(j)
                    } else if v < 0 {
                        sub_.push(j)
                    }
                }
                add.push(add_);
                sub.push(sub_);
            }
            UnitMatrix {
                add,
                sub,
                source_dim: m,
                dest_dim: n,
            }
        };
        let h = {
            let mut add = vec![];
            let mut sub = vec![];
            for i in 0..m {
                let mut add_ = vec![];
                let mut sub_ = vec![];
                for j in 0..n {
                    let v = h[[i, j]];
                    if v > 0 {
                        add_.push(j)
                    } else if v < 0 {
                        sub_.push(j)
                    }
                }
                add.push(add_);
                sub.push(sub_);
            }
            UnitMatrix {
                add,
                sub,
                source_dim: n,
                dest_dim: m,
            }
        };
        (LinearProduct(ToeplitzG(g), ConvToToeplitz(n)), h, e)
    }
}

pub struct ToeplitzG<F>(Array2<F>);

impl<F: Clone + Zero + Mul<Output = F>> LinearOperator<F> for ToeplitzG<F> {
    fn op(&self, x: &[F]) -> Vec<F> {
        mat_vec_mul(&self.0, x)
    }

    fn source_dim(&self) -> usize {
        self.0.dim().1
    }

    fn dest_dim(&self) -> usize {
        self.0.dim().0
    }
}

pub struct UnitMatrix {
    add: Vec<Vec<usize>>,
    sub: Vec<Vec<usize>>,
    source_dim: usize,
    dest_dim: usize,
}

impl<F: Clone + Zero + Sub<Output = F>> LinearOperator<F> for UnitMatrix {
    fn op(&self, x: &[F]) -> Vec<F> {
        let mut r = Vec::with_capacity(self.dest_dim);
        for i in 0..self.dest_dim {
            let mut add = F::zero();
            let mut sub = F::zero();
            for &idx in &self.add[i] {
                add = add + x[idx].clone();
            }
            for &idx in &self.sub[i] {
                sub = sub + x[idx].clone();
            }
            r.push(add - sub);
        }
        r
    }

    fn source_dim(&self) -> usize {
        self.source_dim
    }

    fn dest_dim(&self) -> usize {
        self.dest_dim
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct R(pub isize);

impl One for R {
    fn one() -> Self {
        R(1)
    }
}

impl Zero for R {
    fn zero() -> Self {
        R(0)
    }
    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl Mul for R {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        R(self.0 * other.0)
    }
}

impl Mul<usize> for R {
    type Output = Self;
    fn mul(self, other: usize) -> Self {
        R(other as isize * self.0)
    }
}

impl Add for R {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        R(self.0 + other.0)
    }
}

impl Add<usize> for R {
    type Output = Self;
    fn add(self, other: usize) -> Self {
        R(other as isize + self.0)
    }
}

impl Sub for R {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        R(self.0 - other.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::{cmp::PartialEq, fmt::Display};

    use quickcheck::{quickcheck, Arbitrary, Gen};

    use crate::{
        field::GF2561D,
        tests::{Frac, F7},
    };

    fn mat_y(n: usize) -> Array2<i8> {
        let mut y = Array2::zeros((n, n));
        for i in 0..n {
            y[[i, n - 1 - i]] = 1;
        }
        y
    }

    quickcheck! {
        fn wu2012_lemma4_5(n: usize) -> bool {
            let n = n % 512 + 1;
            let (_, e) = toeplitz_e(n);
            let (_, _, h) = toeplitz_gh::<F7>(n);
            let y = mat_y(n);
            assert_eq!(e, h.dot(&y).t());
            assert!(h.axis_iter(Axis(0)).all(|r| !r.iter().all(|h| *h == 0)));
            let y_ = Array2::from_shape_vec(
                (n, n),
                h.axis_iter(Axis(0))
                    .filter(|r| r.iter().filter(|h| **h != 0).count() == 1)
                    .flat_map(|r| r.to_vec())
                    .collect(),
            )
            .unwrap();
            y_ == mat_y(n)
        }

        fn wu2012_lemma6(n: usize) -> bool {
            let n = n % 512 + 1;
            let (_, e) = toeplitz_e(n);
            assert!(e.axis_iter(Axis(1)).all(|c| !c.iter().all(|e| *e == 0)));
            true
        }
    }

    #[test]
    fn matrix_mul() {
        for _ in 0..100 {
            let mut g = Gen::new(usize::max_value());
            let n = usize::arbitrary(&mut g) % 100 + 1;
            let m = usize::arbitrary(&mut g) % 100 + 1;
            let p = usize::arbitrary(&mut g) % 100 + 1;
            let a = Array2::from_shape_vec(
                (n, m),
                std::iter::repeat(())
                    .take(n * m)
                    .map(|_| F7::arbitrary(&mut g))
                    .collect(),
            )
            .unwrap();
            let b = Array2::from_shape_vec(
                (m, p),
                std::iter::repeat(())
                    .take(p * m)
                    .map(|_| F7::arbitrary(&mut g))
                    .collect(),
            )
            .unwrap();
            assert_eq!(mat_mat_mul(a.view(), b.view()), a.dot(&b));
        }
    }

    #[test]
    fn it_works() {
        toeplitz_e(10);
        toeplitz_gh::<GF2561D>(10);
        toeplitz_gh::<Frac>(10);
    }

    #[allow(clippy::many_single_char_names)] // REASON: match up with symbols in the wu2012 paper
    fn toeplitz_vec_mul<F>(t: Array1<F>, v: Array1<F>) -> Array1<F>
    where
        F: Zero + One + Clone + Sub<Output = F> + Neg<Output = F>,
    {
        let n = v.dim();
        assert_eq!(2 * n - 1, t.dim());
        let (_, e) = toeplitz_e(n);
        let (_, g, h) = toeplitz_gh::<F>(n);
        eprintln!("g, e, h are generated");
        // u = E . ((G . t) * (H . v))
        let x = Array1::from(mat_vec_mul(&g, &t)) * toeplitz_unit_mul(&h, v);
        toeplitz_unit_mul(&e, x)
    }

    #[allow(clippy::many_single_char_names)] // REASON: match up with symbols in the wu2012 paper
    fn toeplitz_test<F>(g: &mut Gen)
    where
        F: PartialEq + Display + Arbitrary + Zero + One + Clone + Sub<Output = F> + Neg<Output = F>,
    {
        let n = u8::arbitrary(g) as usize + 1;
        eprintln!("size n={}", n);
        let t: Array1<_> = (0..2 * n - 1).map(|_| F::arbitrary(g)).collect();
        let v: Array1<_> = (0..n).map(|_| F::arbitrary(g)).collect();

        let toeplitz = Array2::from_shape_vec(
            (n, n),
            (0..n)
                .flat_map(|i| (0..n).map(|j| t[n - 1 + i - j].clone()).collect::<Vec<_>>())
                .collect(),
        )
        .expect("shape should be square");
        let expected = Array1::from(mat_vec_mul(toeplitz.view(), v.view()));
        eprintln!("evaluating actual");
        let actual = toeplitz_vec_mul(t.clone(), v.clone());
        if expected != actual {
            eprintln!("t={}", t);
            eprintln!("v={}", v);
            eprintln!("toeplitz={}", toeplitz);
            eprintln!("expected={}", expected);
            eprintln!("actual={}", actual);
            panic!()
        }
    }

    #[test]
    fn toeplitz() {
        let mut g = Gen::new(usize::max_value());
        for _ in 0..100 {
            toeplitz_test::<GF2561D>(&mut g)
        }
    }

    #[test]
    fn agarwal_cooley() {
        let mut g = Gen::new(usize::max_value());
        type F = F7;
        for i in 2..20 {
            eprintln!("i={}", i);
            let n = i;
            let m = i + 1;
            eprintln!("n={},m={}", n, m);
            let x: Vec<_> = (0..n * m).map(|_| F::arbitrary(&mut g)).collect();
            let y: Vec<_> = (0..n * m).map(|_| F::arbitrary(&mut g)).collect();
            let algo = ToeplitzConv(n * m);
            let expected = algo.apply(&x)(&y);

            eprintln!("eval actual...");
            let algo = AgarwalCooley::new(n, ToeplitzConv(n), m, ToeplitzConv(m));
            let actual = algo.apply(&x)(&y);
            eprintln!("done");
            assert_eq!(actual, expected, "x={:?},y={:?}", x, y);
        }
    }

    #[test]
    fn agarwal_cooley_fix_x() {
        type F = GF2561D;
        let mut g = Gen::new(usize::max_value());
        let n1 = 3;
        let n2 = 5;
        let n = n1 * n2;
        let m = 17;
        let x: Vec<_> = (0..n * m).map(|_| F::arbitrary(&mut g)).collect();
        let algo = ToeplitzConv(n * m);
        let expected = algo.apply(&x);
        let algo = AgarwalCooley::new(
            n,
            AgarwalCooley::new(n1, ToeplitzConv(n1), n2, ToeplitzConv(n2)),
            m,
            ToeplitzConv(m),
        );
        let actual = algo.apply(&x);

        eprintln!("start");
        for i in 1..100 {
            eprintln!("i={}", i);
            let y: Vec<_> = (0..n * m).map(|_| F::arbitrary(&mut g)).collect();
            assert_eq!(expected(&y), actual(&y))
        }
    }

    #[test]
    fn ac_test() {
        let n = 2;
        let m = 3;
        let x = vec![
            F7::new(6),
            F7::new(1),
            F7::new(6),
            F7::new(2),
            F7::new(6),
            F7::new(6),
        ];
        let y = vec![
            F7::new(4),
            F7::new(3),
            F7::new(6),
            F7::new(4),
            F7::new(4),
            F7::new(6),
        ];
        let expected = vec![
            F7::new(4),
            F7::new(0),
            F7::new(4),
            F7::new(4),
            F7::new(4),
            F7::new(6),
        ];

        let algo = AgarwalCooley::new(n, ToeplitzConv(n), m, ToeplitzConv(m));
        let actual = algo.apply(&x)(&y);
        assert_eq!(actual, expected);
    }
}
