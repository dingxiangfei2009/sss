use std::{
    convert::TryInto,
    fmt::{Debug, Formatter, Result as FmtResult},
    iter::repeat_with,
    marker::PhantomData,
    sync::{Arc, RwLock},
};

use alga::general::Field;
use num::{One, Zero};
use rayon::prelude::*;

use crate::{
    field::{FiniteField, PrimeSubfield, F2},
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

/// Multi-point evaluation strategy, by von zur Gathen
#[derive(Clone, Debug)]
pub struct MultipointEvalVZGathen<F, T>
where
    F: FiniteField,
{
    ss: Vec<Polynomial<F>>,
    s_2pows: Vec<
        LazyList<
            Polynomial<F>,
            Polynomial<F>,
            Box<dyn Send + Sync + FnMut(Polynomial<F>) -> (Polynomial<F>, Polynomial<F>)>,
        >,
    >,
    s_betas: Vec<Vec<F>>,
    _p: PhantomData<fn() -> T>,
}

impl<F, T> MultipointEvalVZGathen<F, T>
where
    F: FiniteField + Clone + Send + Sync,
    T: ExtensionTower<Super = F>,
    T::Bottom: Clone + Eq + Send + Sync,
{
    pub fn new() -> Self {
        let bases = T::basis_elements_over_bottom();
        let mut s = Polynomial(vec![F::zero(), F::one()]);
        let mut ss = vec![s.clone()];
        let q = T::Bottom::field_size::<Int>().assert_usize();
        let mut s_betas = vec![bases.clone()];
        for (j, basis) in bases.iter().enumerate() {
            let Coord(_, t) = s.eval_at(basis.clone());
            let t = pow(t, q - 1);
            s = pow(s.clone(), q) - s * t;
            s_betas.push(
                bases[j + 1..]
                    .iter()
                    .map(|b| s.eval_at(b.clone()).1)
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
                    Box::new(|s: Polynomial<F>| {
                        let next_2pow = pow(s.clone(), 2);
                        (next_2pow, s)
                    }) as Box<dyn Send + Sync + FnMut(_) -> _>,
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
            let taylor = taylor_expansion_aux(f, self.ss[i - 1].degree(), &self.s_2pows[i - 1]);
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
                    let mut g = Polynomial::zero();
                    for p in &taylor {
                        g = g + p.clone() * omega_pow.clone();
                        omega_pow *= omega.clone();
                    }
                    self.eval_at(
                        g,
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

    fn eval_from_node(&self, f: Polynomial<F>, n: usize, node: &TreeNode<T::Bottom>) -> Vec<F> {
        let mut result = vec![F::zero(); n];
        let (_, f) = f.div_with_rem(self.ss[self.ss.len() - 1].clone());
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
            {
                x_pow.mul_pow_x(1);
                let (_, r) = x_pow.div_with_rem(cyclotomic_poly.clone());
                x_pow = r;
            }
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
