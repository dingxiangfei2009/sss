use alga::general::Field;

use crate::{Coord, EuclideanDomain, Polynomial};

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
