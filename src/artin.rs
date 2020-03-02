use std::{marker::PhantomData, sync::Arc};

use alga::general::Field;
use num::{One, Zero};
use rayon::prelude::*;

use crate::{
    field::{FiniteField, F2},
    galois::{ExtensionTower, GF65536NTower, GF65536N},
    poly::MultipointEvaluator,
    pow, Coord, EuclideanDomain, Int, Polynomial,
};

#[derive(Clone, Debug)]
pub struct MultipointEvalVZGathen<F, T>
where
    F: FiniteField,
{
    ss: Vec<Polynomial<F>>,
    s_2pows: Vec<Vec<Polynomial<F>>>,
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
        let s_2pows = {
            let k = q.next_power_of_two();
            ss.par_iter()
                .map(|s| {
                    let mut d_pow_2 = vec![];
                    let mut d_pow = s.clone();
                    for _ in 0..=k.trailing_zeros() {
                        d_pow_2.push(d_pow.clone());
                        d_pow = pow(d_pow, 2);
                    }
                    d_pow_2
                })
                .collect()
        };
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

fn taylor_expansion_aux<F>(
    p: Polynomial<F>,
    m: usize,
    d_pow: &[Polynomial<F>],
) -> Vec<Polynomial<F>>
where
    F: Field + Clone,
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
            modulo *= d_pow[b.trailing_zeros() as usize].clone();
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

pub fn taylor_expansion<F>(p: Polynomial<F>, d: Polynomial<F>) -> Vec<Polynomial<F>>
where
    F: Field + Clone,
{
    let n = p.degree();
    let m = d.degree();
    assert_ne!(m, 0, "no tayler expansion of polynomials at 1");
    let k = (n / m).next_power_of_two();
    let mut d_pow_2 = vec![];
    let mut d_pow = d;
    for _ in 0..=k.trailing_zeros() {
        d_pow_2.push(d_pow.clone());
        d_pow = pow(d_pow, 2);
    }
    taylor_expansion_aux(p, m, &d_pow_2)
}

lazy_static::lazy_static! {
    pub static ref MEVZG_GF65536N: Arc<MultipointEvalVZGathen<GF65536N, GF65536NTower>> = Arc::new(MultipointEvalVZGathen::new());
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{int_inj, tests::Frac};

    #[test]
    fn taylor_exp() {
        let p = Polynomial(vec![Frac::one(), -int_inj::<Frac, _>(2), Frac::one()]);
        let d = Polynomial(vec![-Frac::one(), Frac::one()]);
        assert_eq!(
            taylor_expansion(p, d),
            vec![
                Polynomial(vec![Frac::zero()]),
                Polynomial(vec![Frac::zero()]),
                Polynomial(vec![Frac::one()]),
            ]
        );
        let p = Polynomial(vec![Frac::zero(), -int_inj::<Frac, _>(2), Frac::one()]);
        let d = Polynomial(vec![-Frac::one(), Frac::one()]);
        assert_eq!(
            taylor_expansion(p, d),
            vec![
                Polynomial(vec![-Frac::one()]),
                Polynomial(vec![Frac::zero()]),
                Polynomial(vec![Frac::one()]),
            ]
        );
        let p: Vec<Frac> = (0..9).map(|x| int_inj(x)).collect();
        let p = Polynomial::new(p);
        let d: Vec<Frac> = [1, 2, 1, 4].iter().map(|&x| int_inj(x)).collect();
        let d = Polynomial::new(d);
        let expected = vec![
            Polynomial(vec![
                -int_inj::<Frac, _>(533) / int_inj::<Frac, _>(1024),
                -int_inj::<Frac, _>(315) / int_inj::<Frac, _>(512),
                int_inj::<Frac, _>(259) / int_inj::<Frac, _>(1024),
            ]),
            Polynomial(vec![
                int_inj::<Frac, _>(789) / int_inj::<Frac, _>(1024),
                int_inj::<Frac, _>(227) / int_inj::<Frac, _>(256),
                -int_inj::<Frac, _>(35) / int_inj::<Frac, _>(64),
            ]),
            Polynomial(vec![
                -Frac::one() / int_inj::<Frac, _>(4),
                int_inj::<Frac, _>(3) / int_inj::<Frac, _>(16),
                Frac::one() / int_inj::<Frac, _>(2),
            ]),
        ];
        assert_eq!(taylor_expansion(p, d), expected);
    }

    fn u16_to_gf65536n(xs: Vec<u16>) -> Vec<GF65536N> {
        let bases = GF65536NTower::basis_elements_over_bottom();
        xs.into_iter()
            .map(|mut x| {
                let mut y = GF65536N::zero();
                let mut bases = bases.iter();
                while x > 0 {
                    let basis = bases.next().unwrap();
                    if x & 1 > 0 {
                        y += basis.clone();
                    }
                    x >>= 1;
                }
                y
            })
            .collect()
    }

    #[quickcheck]
    fn vz_gathen(mut points: Vec<u16>, f: Vec<u16>) {
        points.sort();
        points.dedup();
        if points.len() > core::u16::MAX as usize {
            return;
        }
        if f.len() > core::u16::MAX as usize {
            return;
        }
        let points = u16_to_gf65536n(points);
        let f = Polynomial::new(u16_to_gf65536n(f));
        let result = MEVZG_GF65536N.eval(f.clone(), points.clone());
        assert_eq!(result.len(), points.len());
        for (r, p) in result.iter().zip(&points) {
            assert_eq!(r, &f.clone().eval_at(p.clone()).1);
        }
    }
}
