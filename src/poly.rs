use std::ops::{Add, Mul, Neg, Sub};

use alga::general::Field;
use num::traits::Zero;

use crate::{Coord, EuclideanDomain, Polynomial};

pub mod fast_mul;
pub mod schoenhage;

/// Multi-point evaluators of polynomials over a fixed set of points
pub trait MultipointEvaluator<F> {
    fn prepare(points: Vec<F>) -> Self;
    fn eval(&self, p: Polynomial<F>) -> Vec<F>;
}

/// A performant multi-point polynomaial evaluator by recursive quotients
pub struct MultiPointEvalTable<F> {
    n: usize,
    p: Polynomial<F>,
    left: Option<Box<Self>>,
    right: Option<Box<Self>>,
}

impl<F> MultiPointEvalTable<F>
where
    F: Field + Clone,
{
    /// Construct the pre-computation table over the points
    pub fn build(points: &[F]) -> Self {
        match points {
            [] => Self {
                n: 0,
                p: Polynomial(vec![F::one()]),
                left: None,
                right: None,
            },
            [x] => {
                let p = Polynomial(vec![-x.clone(), F::one()]);
                Self {
                    n: 1,
                    p,
                    left: None,
                    right: None,
                }
            }
            points => {
                let n = points.len();
                let (left, right) = points.split_at(n / 2);
                let m_left = Self::build(left);
                let m_right = Self::build(right);
                Self {
                    n,
                    p: m_left.p.clone() * m_right.p.clone(),
                    left: Some(Box::new(m_left)),
                    right: Some(Box::new(m_right)),
                }
            }
        }
    }

    /// Evaluate the polynomial over the fixed set of points using the pre-computed table
    pub fn eval(&self, f: Polynomial<F>) -> Vec<F> {
        let (_, r) = f.div_with_rem(self.p.clone());
        if r.degree() > 0 {
            let mut v = self
                .left
                .as_ref()
                .expect("check the degree")
                .eval(r.clone());
            v.extend(self.right.as_ref().expect("check the degree").eval(r));
            v
        } else {
            let Coord(_, y) = r.eval_at(F::zero());
            vec![y; self.n]
        }
    }
}

impl<F> MultipointEvaluator<F> for MultiPointEvalTable<F>
where
    F: Field + Clone,
{
    fn prepare(points: Vec<F>) -> Self {
        Self::build(&points)
    }
    fn eval(&self, f: Polynomial<F>) -> Vec<F> {
        MultiPointEvalTable::eval(self, f)
    }
}

pub trait PolynomialLike<T>:
    EuclideanDomain<usize>
    + Add<Output = Self>
    + Mul<Output = Self>
    + Mul<T, Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
    + Clone
{
    fn from_polynomial(p: Polynomial<T>) -> Self;
    fn into_polynomial(self) -> Polynomial<T>;
    fn from_vec(p: Vec<T>) -> Self;
}

impl<F> PolynomialLike<F> for Polynomial<F>
where
    F: Clone + Field,
{
    fn from_polynomial(p: Polynomial<F>) -> Self {
        p
    }
    fn into_polynomial(self) -> Polynomial<F> {
        self
    }
    fn from_vec(p: Vec<F>) -> Self {
        Polynomial::new(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::field::GF2561D;
    use quickcheck::quickcheck;

    quickcheck! {
        fn multipoint_eval_prop(p: Polynomial<GF2561D>, t: Vec<u8>) -> bool {
            multipoint_eval(p, t);
            true
        }
    }

    fn multipoint_eval(p: Polynomial<GF2561D>, mut t: Vec<u8>) {
        t.sort();
        t.dedup();
        if t.is_empty() || p.is_zero() {
            return;
        }
        let t: Vec<_> = t.into_iter().map(GF2561D).collect();
        let et = MultiPointEvalTable::build(&t);
        println!("p={:?}", p);
        println!("t={:?}", t);
        assert_eq!(
            et.eval(p.clone()),
            t.into_iter()
                .map(|x| {
                    let Coord(_, y) = p.eval_at(x);
                    y
                })
                .collect::<Vec<_>>()
        );
    }
}
