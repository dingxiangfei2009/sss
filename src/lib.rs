#[cfg(test)]
#[macro_use(quickcheck)]
extern crate quickcheck_macros;

pub mod field;
use alga::general::{Additive, Field, Identity};

#[derive(Clone, PartialEq)]
pub struct Polynomial<T>(pub Vec<T>);
#[derive(Clone, PartialEq)]
pub struct Coord<T>(pub T, pub T);

impl<T> Polynomial<T> {
    pub fn into_coord(&self, x: T) -> Coord<T>
    where
        T: Field + Clone,
    {
        let mut y = <T as Identity<Additive>>::identity();
        for a in self.0.iter().rev() {
            y *= x.clone();
            y += a.clone();
        }
        Coord(x, y)
    }

    pub fn from_coords(c: &[Coord<T>]) -> Self
    where
        T: Field + Clone + Eq,
    {
        let n = c.len();
        let mut w: Vec<_> = c.into_iter().map(|c| vec![c.1.clone()]).collect();
        let zero = <T as Identity<Additive>>::identity();
        for round in 2..=n {
            let mut w_ = vec![];
            for i in 0..=n - round {
                let j = i + (round - 1);
                let mut v = vec![zero.clone()];
                v.extend_from_slice(&w[i]);
                for (k, w) in w[i].iter().enumerate() {
                    v[k] -= c[j].0.clone() * w.clone();
                }
                for (k, w) in w[i + 1].iter().enumerate() {
                    v[k + 1] -= w.clone();
                    v[k] += c[i].0.clone() * w.clone();
                }
                let scale = c[i].0.clone() - c[j].0.clone();
                w_.push(v.into_iter().map(|w| w / scale.clone()).collect());
            }
            w = w_;
        }
        // remove zeroed high degrees
        let mut w = w
            .drain(0..1)
            .next()
            .expect("interpolation should return exactly one polynomial");
        truncate_high_degree_zeros(&mut w, zero);
        Polynomial(w)
    }
}

fn truncate_high_degree_zeros<T: Eq>(w: &mut Vec<T>, zero: T) {
    let mut zeroed = w.len();
    while zeroed > 1 && w[zeroed - 1] == zero {
        zeroed -= 1;
    }
    w.truncate(zeroed);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::GF2561D;
    use num_traits::Zero;

    #[quickcheck]
    fn it_works(mut u: Vec<GF2561D>) {
        if u.len() == 0 {
            u = vec![GF2561D(0)];
        }
        if u.len() > 254 {
            u = u.drain(0..254).collect();
        }
        truncate_high_degree_zeros(&mut u, GF2561D::zero());
        let threshold = u.len();
        let p = Polynomial(u.clone());

        let q: Vec<_> = (1u8..=u.len() as u8 + 1)
            .map(GF2561D)
            .map(|x| p.into_coord(x))
            .take(threshold)
            .collect();
        let r = Polynomial::from_coords(q.as_slice());
        assert_eq!(u.as_slice(), r.0.as_slice());

        let q: Vec<_> = (1u8..=u.len() as u8 + 1)
            .map(GF2561D)
            .map(|x| p.into_coord(x))
            .collect();
        let r = Polynomial::from_coords(q.as_slice());
        assert_eq!(u.as_slice(), r.0.as_slice());
    }
}
