use std::{
    collections::{btree_map::Entry, BTreeMap},
    convert::TryInto,
    fmt::{Debug, Formatter, Result as FmtResult},
    iter::repeat_with,
    marker::PhantomData,
    ops::{Add, Mul, Neg, Sub},
    sync::{Arc, RwLock},
};

use alga::general::Field;
use num::{One, Zero};
use rayon::prelude::*;

use crate::{
    field::{nat_inj, FiniteField, PrimeSubfield, F2},
    galois::{ExtensionTower, GF65536NTower, GF65536N},
    poly::MultipointEvaluator,
    pow, Coord, EuclideanDomain, Int, Polynomial,
};

struct LazyListInner<T, S, G> {
    inner: Vec<T>,
    state: Option<S>,
    generator: G,
}

impl<T, S, G> LazyListInner<T, S, G>
where
    T: Clone,
    G: FnMut(S) -> (S, T),
{
    fn generate(&mut self, idx: usize) -> T {
        let mut state = self.state.take().expect("state must be present");
        while idx >= self.inner.len() {
            let (state_, val) = (self.generator)(state);
            self.inner.push(val);
            state = state_;
        }
        self.state = Some(state);
        self.inner[idx].clone()
    }
}

struct LazyList<T, S, G> {
    inner: Arc<RwLock<LazyListInner<T, S, G>>>,
}

impl<T, S, G> Clone for LazyList<T, S, G> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T, S, G> Debug for LazyList<T, S, G> {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        f.debug_struct("LazyList").finish()
    }
}

impl<T, S, G> LazyList<T, S, G> {
    fn new(state: S, generator: G) -> Self {
        let state = Some(state);
        Self {
            inner: Arc::new(RwLock::new(LazyListInner {
                inner: vec![],
                state,
                generator,
            })),
        }
    }
    fn get(&self, idx: usize) -> T
    where
        T: Clone,
        G: FnMut(S) -> (S, T),
    {
        {
            let inner = self.inner.read().expect("lock poisoned");
            if idx < inner.inner.len() {
                return inner.inner[idx].clone();
            }
        }
        let mut inner = self.inner.write().expect("lock poisoned");
        inner.generate(idx)
    }
}

type MultipointEvalVZGathenLazyList<F> = LazyList<
    SparsePolynomial<F>,
    SparsePolynomial<F>,
    Box<dyn Send + Sync + FnMut(SparsePolynomial<F>) -> (SparsePolynomial<F>, SparsePolynomial<F>)>,
>;

/// Multi-point evaluation strategy, by von zur Gathen
#[derive(Clone, Debug)]
pub struct MultipointEvalVZGathen<F, T>
where
    F: FiniteField,
{
    ss: Vec<SparsePolynomial<F>>,
    s_2pows: Vec<MultipointEvalVZGathenLazyList<F>>,
    s_betas: Vec<Vec<F>>,
    _p: PhantomData<fn() -> T>,
}

/// A sparse representation of polynomials, ordered in ascending powers of X
#[derive(Clone, Debug)]
struct SparsePolynomial<F>(Vec<(usize, F)>);

impl<F> SparsePolynomial<F> {
    fn degree(&self) -> usize {
        self.0.last().expect("should not be empty").0
    }
}

impl<F> Mul<F> for SparsePolynomial<F>
where
    F: Mul<F, Output = F> + Clone,
{
    type Output = Self;

    fn mul(mut self, rhs: F) -> Self::Output {
        for (_, c) in &mut self.0 {
            *c = c.clone() * rhs.clone();
        }
        self
    }
}

impl<F> SparsePolynomial<F>
where
    F: Mul<F, Output = F> + Add<F, Output = F> + Zero + Clone + One,
{
    fn eval_at(&self, x: F) -> F {
        let mut r = F::zero();
        for (e, c) in &self.0 {
            r = r + pow(x.clone(), *e) * c.clone();
        }
        r
    }

    /// compute p(x)^2
    fn sqr(&self) -> Self {
        let Self(s) = self;
        if s.len() == 1 && s[0].1.is_zero() {
            return Self(vec![(0, F::zero())]);
        }
        let mut r = BTreeMap::new();
        for (i, ci) in s {
            r.insert(
                i.checked_mul(2).expect("overflowing exponent"),
                ci.clone() * ci.clone(),
            );
        }
        let two: F = nat_inj(2u8);
        if two.is_zero() {
            return Self(r.into_iter().collect());
        }
        for i_ in 0..s.len() {
            let (i, ci) = &s[i_];
            for j_ in i_ + 1..s.len() {
                let (j, cj) = &s[j_];
                let c = two.clone() * ci.clone() * cj.clone();
                if c.is_zero() {
                    continue;
                }
                match r.entry(i.checked_add(*j).expect("overflowing exponent")) {
                    Entry::Vacant(e) => {
                        e.insert(c);
                    }
                    Entry::Occupied(mut e) => {
                        let c_ = e.get_mut();
                        *c_ = c_.clone() + c;
                        if c_.is_zero() {
                            e.remove();
                        }
                    }
                }
            }
        }
        Self(r.into_iter().collect())
    }
}

impl<F> SparsePolynomial<F>
where
    F: FiniteField + Clone,
{
    fn pow_q<T>(&self) -> Self
    where
        T: ExtensionTower<Super = F>,
    {
        let mut s = self.0.clone();
        let q = T::Bottom::field_size::<usize>();
        for (e, c) in &mut s {
            *e = e.checked_mul(q).expect("overflowing exponent");
            *c = pow(c.clone(), q);
        }
        Self(s)
    }
}

impl<F> Sub<Self> for SparsePolynomial<F>
where
    F: Sub<F, Output = F> + Neg<Output = F> + Zero,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let Self(lhs) = self;
        let Self(rhs) = rhs;
        let mut r = Vec::with_capacity(lhs.len() + rhs.len());
        let mut lhs = lhs.into_iter().fuse().peekable();
        let mut rhs = rhs.into_iter().fuse().peekable();
        loop {
            match (lhs.peek(), rhs.peek()) {
                (Some((eleft, _)), Some((eright, _))) => {
                    if eleft < eright {
                        let (e, c) = lhs.next().expect("there must be next element");
                        if !c.is_zero() {
                            r.push((e, c))
                        }
                    } else if eleft > eright {
                        let (e, c) = rhs.next().expect("there must be next element");
                        if !c.is_zero() {
                            r.push((e, -c))
                        }
                    } else {
                        let (_, cleft) = lhs.next().expect("there must be next element");
                        let (e, cright) = rhs.next().expect("there must be next element");
                        let c = cleft - cright;
                        if !c.is_zero() {
                            r.push((e, c))
                        }
                    }
                }
                (Some(_), _) => {
                    let (e, c) = lhs.next().expect("there must be next element");
                    if !c.is_zero() {
                        r.push((e, c))
                    }
                }
                (_, Some(_)) => {
                    let (e, c) = rhs.next().expect("there must be next element");
                    if !c.is_zero() {
                        r.push((e, -c))
                    }
                }
                _ => return Self(r),
            }
        }
    }
}

struct PolynomialSlice<'a, F>(&'a [F]);

impl<'a, F> PolynomialSlice<'a, F>
where
    F: Zero,
{
    fn new(p: &'a [F]) -> Self {
        if p.is_empty() {
            return Self(p);
        }
        let mut j = 0;
        for (i, c) in p.iter().enumerate().rev() {
            if !c.is_zero() {
                j = i;
                break;
            }
        }
        Self(&p[..=j])
    }
}

impl<F, T> MultipointEvalVZGathen<F, T>
where
    F: FiniteField + Clone + Send + Sync,
    T: ExtensionTower<Super = F>,
    T::Bottom: Clone + Eq + Send + Sync,
{
    pub fn new() -> Self {
        let bases = T::basis_elements_over_bottom();
        let mut s = SparsePolynomial(vec![(1, F::one())]); // s_0 = x
        let mut ss = vec![s.clone()];
        let q = T::Bottom::field_size::<usize>();
        assert!(q > 0);
        let mut s_betas = vec![bases.clone()];
        for (j, basis) in bases.iter().enumerate() {
            let t = pow(s.eval_at(basis.clone()), q - 1); // NOTE: `t` is not necessarily `1`
            s = s.pow_q::<T>() - s * t;
            s_betas.push(
                bases[j + 1..]
                    .iter()
                    .map(|b| s.eval_at(b.clone()))
                    .collect(),
            );
            ss.push(s.clone());
        }
        let s_2pows = ss
            .iter()
            .cloned()
            .map(|s| {
                LazyList::new(
                    s,
                    Box::new(|s: SparsePolynomial<F>| (s.sqr(), s))
                        as Box<dyn Send + Sync + FnMut(_) -> _>,
                )
            })
            .collect();
        Self {
            ss,
            s_betas,
            s_2pows,
            _p: PhantomData,
        }
    }

    /// returns (remainder, quotient)
    fn sparse_div_with_rem<'a>(
        p: &'a mut [F],
        divisor: &SparsePolynomial<F>,
    ) -> (&'a mut [F], &'a mut [F]) {
        let deg_divisor = divisor.degree();
        let n = p.len() - 1;
        if n < deg_divisor {
            return p.split_at_mut(p.len());
        }
        for i in (deg_divisor..=n).rev() {
            let mut itr_d = divisor.0.iter().rev();
            let (deg, lead) = itr_d.next().unwrap();
            debug_assert_eq!(*deg, deg_divisor);
            let quot = p[i].clone() / lead.clone();
            p[i] = quot.clone();
            for (j, c) in itr_d {
                debug_assert!(j < deg);
                let c_ = &mut p[i - deg_divisor + j];
                *c_ = c_.clone() - c.clone() * quot.clone();
            }
        }
        p.split_at_mut(deg_divisor)
    }

    fn taylor_expansion_aux<'a>(
        p: &'a mut [F],
        m: usize,
        d_2pow: &MultipointEvalVZGathenLazyList<F>,
    ) -> Vec<PolynomialSlice<'a, F>> {
        if p.is_empty() {
            return vec![PolynomialSlice(&[])];
        }
        let divisor = d_2pow.get(m);
        let (r, q) = Self::sparse_div_with_rem(p, &divisor);
        if m > 0 {
            let mut r = Self::taylor_expansion_aux(r, m - 1, d_2pow);
            assert!(r.len() <= 1 << m);
            r.resize_with(1 << m, || PolynomialSlice(&[]));
            r.extend(Self::taylor_expansion_aux(q, m - 1, d_2pow));
            r
        } else {
            vec![PolynomialSlice::new(r), PolynomialSlice::new(q)]
        }
    }

    fn taylor_expansion<'a>(
        p: &'a mut Polynomial<F>,
        d_2pow: &MultipointEvalVZGathenLazyList<F>,
    ) -> Vec<PolynomialSlice<'a, F>> {
        let d = d_2pow.get(0);
        let n = p.degree();
        let mut m = 0;
        let mut deg_div = d.degree();
        while deg_div < n {
            m += 1;
            deg_div <<= 1;
        }
        let Polynomial(p) = p;
        Self::taylor_expansion_aux(p, m, d_2pow)
    }

    fn eval_at<'a>(
        &'a self,
        mut f: Polynomial<F>,
        suffix: Vec<T::Bottom>,
        node: &'a TreeNode<T::Bottom>,
    ) -> Vec<(usize, F)> {
        let k = self.ss.len() - 1;
        let i = k - suffix.len();
        if i == 0 {
            // eval the polynomial at 0
            let x = f.0.drain(..1).next().expect("at least one coefficient");
            node.idx.iter().map(|&idx| (idx, x.clone())).collect()
        } else {
            let taylor = Self::taylor_expansion(&mut f, &self.s_2pows[i - 1]);
            let mut s_beta = F::zero();
            for (j, d) in suffix.iter().enumerate() {
                s_beta += T::into_super(d.clone()) * self.s_betas[i - 1][suffix.len() - j].clone();
            }
            // eval taylor expansion at s_i = c
            let s_i_beta = self.s_betas[i - 1][0].clone();
            node.children
                .par_iter()
                .flat_map(|child| {
                    let c = child.value.clone();
                    let omega = s_beta.clone() + s_i_beta.clone() * T::into_super(c);
                    let mut omega_pow = F::one();
                    let f = {
                        let mut g =
                            vec![F::zero(); taylor.iter().map(|x| x.0.len()).max().unwrap_or(0)];
                        for PolynomialSlice(p) in &taylor {
                            for (p, g) in p.iter().zip(&mut g) {
                                *g = g.clone() + p.clone() * omega_pow.clone();
                            }
                            omega_pow *= omega.clone();
                        }
                        Polynomial::new(g)
                    };
                    self.eval_at(
                        f,
                        suffix
                            .iter()
                            .cloned()
                            .chain(Some(child.value.clone()))
                            .collect(),
                        child,
                    )
                })
                .collect()
        }
    }

    fn eval_from_node(&self, mut f: Polynomial<F>, n: usize, node: &TreeNode<T::Bottom>) -> Vec<F> {
        let mut result = vec![F::zero(); n];
        let (f, _) = Self::sparse_div_with_rem(&mut f.0, &self.ss[self.ss.len() - 1]);
        let f = Polynomial::from(f.to_vec());
        for (i, r) in self.eval_at(f, vec![], node) {
            result[i] = r;
        }
        result
    }

    pub fn eval(&self, f: Polynomial<F>, points_: Vec<F>) -> Vec<F> {
        let n = points_.len();
        let mut points = TreeNode::new(T::Bottom::zero());
        for (i, p) in points_.into_iter().enumerate() {
            points.insert(T::to_vec(p), i)
        }
        self.eval_from_node(f, n, &points)
    }
}

pub struct GF65536NPreparedMultipointEvalVZG {
    tree: TreeNode<F2>,
    num_points: usize,
}

impl MultipointEvaluator<GF65536N> for GF65536NPreparedMultipointEvalVZG {
    fn eval(&self, f: Polynomial<GF65536N>) -> Vec<GF65536N> {
        MEVZG_GF65536N.eval_from_node(f, self.num_points, &self.tree)
    }
    fn prepare(points: Vec<GF65536N>) -> Self {
        let num_points = points.len();
        let mut tree = TreeNode::new(F2::zero());
        for (i, p) in points.into_iter().enumerate() {
            tree.insert(GF65536NTower::to_vec(p), i)
        }
        Self { num_points, tree }
    }
}

struct TreeNode<F> {
    value: F,
    idx: Vec<usize>,
    children: Vec<TreeNode<F>>,
}

impl<F> TreeNode<F> {
    fn new(value: F) -> Self {
        Self {
            value,
            idx: vec![],
            children: vec![],
        }
    }
    fn insert(&mut self, mut path: Vec<F>, idx: usize)
    where
        F: Eq,
    {
        if let Some(value) = path.pop() {
            for child in &mut self.children {
                if child.value == value {
                    return child.insert(path, idx);
                }
            }
            let mut node = Self::new(value);
            node.idx = vec![idx];
            node.insert(path, idx);
            self.children.push(node);
        }
    }
}

fn lsb_usize(x: usize) -> usize {
    use std::num::Wrapping;
    let Wrapping(b) = Wrapping(x) & -Wrapping(x);
    b
}

fn taylor_expansion_aux<F, S, G>(
    p: Polynomial<F>,
    m: usize,
    d_pow: &LazyList<Polynomial<F>, S, G>,
) -> Vec<Polynomial<F>>
where
    F: Field + Clone,
    G: FnMut(S) -> (S, Polynomial<F>),
{
    let n = p.degree();
    if n < m {
        return vec![p];
    }
    let mut t = (n / m + 1) / 2;
    let mut modulo = Polynomial::<F>::one();
    loop {
        let b = lsb_usize(t);
        if b > 0 {
            modulo *= d_pow.get(b.trailing_zeros() as usize);
            t -= b;
        } else {
            break;
        }
    }
    let (q, r) = p.div_with_rem(modulo);
    let mut lower = taylor_expansion_aux(r, m, d_pow);
    assert!(lower.len() <= t + 1);
    lower.resize(t + 1, Polynomial::zero());
    let right = taylor_expansion_aux(q, m, d_pow);
    lower.extend(right);
    lower
}

/// Taylor expansion of `p` at `d`.
pub fn taylor_expansion<F>(p: Polynomial<F>, d: Polynomial<F>) -> Vec<Polynomial<F>>
where
    F: Field + Clone,
{
    let m = d.degree();
    assert_ne!(m, 0, "no tayler expansion of polynomials at 1");
    let d_pow_2 = LazyList::new(
        d,
        Box::new(|d: Polynomial<F>| {
            let next_2pow = pow(d.clone(), 2);
            (next_2pow, d)
        }) as Box<dyn Send + Sync + FnMut(_) -> _>,
    );
    taylor_expansion_aux(p, m, &d_pow_2)
}

lazy_static::lazy_static! {
    pub static ref MEVZG_GF65536N: Arc<MultipointEvalVZGathen<GF65536N, GF65536NTower>> = Arc::new(MultipointEvalVZGathen::new());
}

pub fn build_cyclotomic_polynomial<F>(mut idx: usize) -> Polynomial<F>
where
    F: Field + Clone,
{
    {
        let idx: u32 = idx.try_into().unwrap();
        assert!(idx < std::u32::MAX);
    }
    let mut primes = crate::primes::PRIMEU16.iter();
    let mut phi = Polynomial(vec![-F::one(), F::one()]);
    while idx > 1 {
        let p = match primes.next() {
            Some(p) => *p as usize,
            None => idx,
        };
        if idx % p == 0 {
            let mut m = 0;
            while idx % p == 0 {
                idx /= p;
                m += 1;
            }
            let Coord(_, phi_) = phi.eval_at(Polynomial(
                repeat_with(F::zero).take(p).chain(Some(F::one())).collect(),
            ));
            let (q, r) = phi_.div_with_rem(phi);
            assert!(r.is_zero());
            if m > 1 {
                let pow = pow(p, m - 1);
                phi = q
                    .eval_at(Polynomial(
                        repeat_with(F::zero)
                            .take(pow)
                            .chain(Some(F::one()))
                            .collect(),
                    ))
                    .1;
            } else {
                phi = q;
            }
        }
    }
    phi
}

fn compose<F>(p: Polynomial<F>, r: Polynomial<F>) -> Polynomial<F>
where
    F: PrimeSubfield + Clone,
{
    let char = F::characteristic::<Int>().assert_usize();
    let d = p.degree();
    let mut c = 1;
    let mut n = 0;
    while c * char <= d {
        c *= char;
        n += 1;
    }
    if n > 0 {
        let Polynomial(p) = p;
        let Coord(_, r_) = r.eval_at(Polynomial(
            repeat_with(F::zero).take(c).chain(Some(F::one())).collect(),
        ));
        let mut q: Polynomial<F> = Polynomial::zero();
        for c in p.chunks(c).rev() {
            let q_ = compose(Polynomial::from(c.to_vec()), r.clone());
            q = q * r_.clone() + q_;
        }
        q
    } else {
        p.eval_at(r).1
    }
}

/// [untested] Build Artin-Schreier tower, where the initial field extension is defined by the `initial` equation.
pub fn build_tower<F>(initial: Polynomial<F>, height: usize) -> Polynomial<F>
where
    F: PrimeSubfield + Clone,
{
    let char = F::characteristic::<Int>().assert_usize();
    {
        let char: u32 = char.try_into().unwrap();
        assert!(char < std::u32::MAX / 2, "prime subfield is too big, sorry");
    }
    let d = initial.degree();
    {
        assert!(d > 1);
        assert!(!initial.coeff(initial.degree() - 1).is_zero(), "initial element has a zero trace; replace initial defining polynomial f(x) with f(x-1)");
    }
    let cyclotomic_index = char * 2 - 1;
    let cyclotomic_poly = build_cyclotomic_polynomial::<F>(cyclotomic_index);
    let r = Polynomial(
        Some(-F::one())
            .into_iter()
            .chain(repeat_with(F::zero))
            .take(char)
            .chain(Some(F::one()))
            .collect(),
    );
    let mut q = compose(initial, r.clone());
    for i in 1..height {
        if i == 1 && char == 2 && d % 2 == 1 {
            q = compose(q, r.clone());
            continue;
        }
        let mut x_pow = Polynomial::one();
        let mut f: Polynomial<Polynomial<F>> = Polynomial::one();
        for _ in 0..cyclotomic_index {
            let f_ = {
                let mut x = Polynomial::<F>::one();
                let mut coeffs = vec![];
                for f in q.0.clone() {
                    coeffs.push(x.clone() * f);
                    let (_, r) = (x * x_pow.clone()).div_with_rem(cyclotomic_poly.clone());
                    x = r;
                }
                Polynomial::new(coeffs)
            };
            let Polynomial(mut f_) = f * f_;
            for f in &mut f_ {
                let (_, r) = std::mem::take(f).div_with_rem(cyclotomic_poly.clone());
                *f = r;
            }
            f = Polynomial::from(f_);
            let (_, r) = x_pow.mul_pow_x(1).div_with_rem(cyclotomic_poly.clone());
            x_pow = r;
        }
        let f: Polynomial<F> = {
            let mut coeffs = vec![];
            for (i, c) in f.0.into_iter().enumerate() {
                if i % cyclotomic_index == 0 {
                    assert_eq!(c.degree(), 0);
                    coeffs.extend(c.0);
                } else {
                    assert!(c.is_zero())
                }
            }
            Polynomial::from(coeffs)
        };
        q = compose(f, r.clone());
    }
    q
}

#[cfg(test)]
mod tests;
