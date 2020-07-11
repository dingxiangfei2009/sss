use super::*;

use alga::general::Ring;
use itertools::{izip, Itertools};
use num::traits::{One, Zero};
use rayon::{join, prelude::*};

use crate::{field::int_inj, ring::RingInvertible, Polynomial};

use std::mem::swap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LevelData {
    Level(usize),
    Undetermined,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchoenhageDyadicDomain<R> {
    level: LevelData,
    data: Polynomial<R>,
}

impl<R> SchoenhageDyadicDomain<R>
where
    R: 'static + Ring + RingInvertible + Clone + Send + Sync,
{
    pub fn reduce(&mut self, level: usize) {
        let n = 1 << level;
        self.level = LevelData::Level(level);
        if self.data.degree() < n {
            return;
        }
        let mut data = Polynomial::zero();
        swap(&mut data, &mut self.data);
        let Polynomial(mut data) = data;
        if data.len() >= 2 * n {
            for chunk in data.split_off(2 * n).into_iter().chunks(2 * n).into_iter() {
                for (i, x) in chunk.into_iter().enumerate() {
                    data[i] += x;
                }
            }
        }
        for (i, x) in data.split_off(n).into_iter().enumerate() {
            data[i] -= x;
        }
        let mut data = Polynomial::new(data);
        swap(&mut data, &mut self.data);
    }

    fn mul_unity_root_pow(self, power: usize) -> Self {
        let (level, mut data) = if let Self {
            data: Polynomial(data),
            level: LevelData::Level(level),
        } = self
        {
            (level, data)
        } else {
            panic!("modulus is unknown")
        };
        let m = 1 << level;
        data.resize(2 * m, R::zero());
        data.rotate_right(power % (2 * m));
        for (i, excess) in data.split_off(m).into_iter().enumerate() {
            data[i] -= excess;
        }
        Self {
            data: Polynomial::new(data),
            level: LevelData::Level(level),
        }
    }

    fn fibers(self) -> Vec<Self> {
        let (k, data) = if let Self {
            data: Polynomial(data),
            level: LevelData::Level(level),
        } = self
        {
            (level, data)
        } else {
            panic!("modulus is unknown")
        };

        let n = 1 << k;
        let m = 1 << (k / 2);
        let t = n / m;
        let mut fibers: Vec<_> = data
            .into_iter()
            .chunks(m)
            .into_iter()
            .map(|f| Self {
                data: Polynomial::new(f),
                level: LevelData::Level(k / 2 + 1),
            })
            .collect();
        debug_assert!(fibers.len() <= t);
        fibers.resize_with(t, || {
            let mut x = Self::zero();
            x.reduce(k / 2 + 1);
            x
        });
        fibers
    }

    fn schoenhage(self, other: Self) -> Self {
        let k = match (self.level, other.level) {
            (LevelData::Level(level), LevelData::Level(level_)) if level == level_ => level,
            _ => panic!("modulus is unknown"),
        };
        let n = 1 << k;
        let m = 1 << (k / 2);
        let t = n / m;
        let (f, g) = join(move || self.fibers(), move || other.fibers());
        let f = f
            .into_par_iter()
            .enumerate()
            .map(|(i, f)| f.mul_unity_root_pow(if m == t { i * 2 } else { i }))
            .collect();
        let g = g
            .into_par_iter()
            .enumerate()
            .map(|(i, g)| g.mul_unity_root_pow(if m == t { i * 2 } else { i }))
            .collect();
        let h: Vec<_> = convolute(f, g, m * 2)
            .into_par_iter()
            .enumerate()
            .map(|(i, h)| h.mul_unity_root_pow(if m == t { 4 * m - 2 * i } else { 4 * m - i }))
            .collect();

        let mut data = vec![R::zero(); 2 * m + 2 * n];
        for (
            i,
            Self {
                data: Polynomial(a),
                ..
            },
        ) in h.into_iter().enumerate()
        {
            for (j, a) in a.into_iter().enumerate() {
                data[i * m + j] += a;
            }
        }
        let mut data = Self {
            data: Polynomial::new(data),
            level: LevelData::Undetermined,
        };
        data.reduce(k);

        data
    }
}

fn convolute<R>(
    f: Vec<SchoenhageDyadicDomain<R>>,
    g: Vec<SchoenhageDyadicDomain<R>>,
    m: usize,
) -> Vec<SchoenhageDyadicDomain<R>>
where
    R: 'static + Ring + RingInvertible + Clone + Send + Sync,
{
    debug_assert!(f.len() == g.len());
    let n = f.len();
    let n_inv = R::try_invert(int_inj(n)).expect("n should be invertible");
    let (f, g) = join(
        move || dyadic_transform(f, m * 2),
        move || dyadic_transform(g, m * 2),
    );
    let mut h = dyadic_transform(
        f.into_par_iter().zip(g).map(|(f, g)| f * g).collect(),
        m * 2,
    );
    h[1..].reverse();
    h.into_par_iter().map(|h| h * n_inv.clone()).collect()
}

fn dyadic_transform<R>(
    f: Vec<SchoenhageDyadicDomain<R>>,
    order: usize,
) -> Vec<SchoenhageDyadicDomain<R>>
where
    R: 'static + Ring + RingInvertible + Clone + Send + Sync,
{
    let n = f.len();
    if n == 1 {
        return f;
    }
    let q = order / n;
    let n_half = n / 2;
    assert_eq!(n % 2, 0);
    assert_eq!(order % n, 0, "order={}, n={}", order, n);
    let r_0: Vec<_> = (0..n_half)
        .map(|j| f[j].clone() + f[j + n_half].clone())
        .collect();
    let r_1: Vec<_> = (0..n_half)
        .map(|j| (f[j].clone() - f[j + n_half].clone()).mul_unity_root_pow(q * j))
        .collect();
    let (r_0, r_1) = join(
        move || dyadic_transform(r_0, order),
        move || dyadic_transform(r_1, order),
    );
    let mut r = vec![];
    for (r_0, r_1) in izip!(r_0, r_1) {
        r.push(r_0);
        r.push(r_1);
    }
    r
}

impl<R> Zero for SchoenhageDyadicDomain<R>
where
    R: 'static + Ring + RingInvertible + Clone + Send + Sync,
{
    fn is_zero(&self) -> bool {
        self.data.is_zero()
    }
    fn zero() -> Self {
        Self {
            data: Polynomial::zero(),
            level: LevelData::Undetermined,
        }
    }
}

impl<R> Add for SchoenhageDyadicDomain<R>
where
    R: 'static + Ring + RingInvertible + Clone + Send + Sync,
{
    type Output = Self;
    fn add(mut self, mut other: Self) -> Self {
        match (self.level, other.level) {
            (LevelData::Level(level), LevelData::Level(level_)) if level == level_ => {}
            (LevelData::Undetermined, LevelData::Level(level)) => {
                self.reduce(level);
            }
            (LevelData::Level(level), LevelData::Undetermined) => {
                other.reduce(level);
            }
            (LevelData::Undetermined, LevelData::Undetermined) => {}
            _ => panic!("level mismatch"),
        }
        self.data = self.data + other.data;
        self
    }
}

impl<R> Sub for SchoenhageDyadicDomain<R>
where
    R: 'static + Ring + RingInvertible + Clone + Send + Sync,
{
    type Output = Self;
    fn sub(mut self, mut other: Self) -> Self {
        match (self.level, other.level) {
            (LevelData::Level(level), LevelData::Level(level_)) if level == level_ => {}
            (LevelData::Undetermined, LevelData::Level(level)) => {
                self.reduce(level);
            }
            (LevelData::Level(level), LevelData::Undetermined) => {
                other.reduce(level);
            }
            (LevelData::Undetermined, LevelData::Undetermined) => {}
            _ => panic!("level mismatch"),
        }
        self.data = self.data - other.data;
        self
    }
}

impl<R> Mul<R> for SchoenhageDyadicDomain<R>
where
    R: Ring,
{
    type Output = Self;
    fn mul(mut self, other: R) -> Self {
        self.data = self.data * other;
        self
    }
}

const SCHOENHAGE_DYADIC_CUTOFF_LEVEL: usize = 3;

impl<R> Mul for SchoenhageDyadicDomain<R>
where
    R: 'static + Ring + RingInvertible + Clone + Send + Sync,
{
    type Output = Self;
    fn mul(mut self, mut other: Self) -> Self {
        let level = match (self.level, other.level) {
            (LevelData::Level(level), LevelData::Level(level_)) if level == level_ => level,
            (LevelData::Undetermined, LevelData::Level(level)) => {
                self.reduce(level);
                level
            }
            (LevelData::Level(level), LevelData::Undetermined) => {
                other.reduce(level);
                level
            }
            (LevelData::Undetermined, LevelData::Undetermined) => {
                self.data = self.data * other.data;
                return self;
            }
            _ => panic!("level mismatch"),
        };
        if level < SCHOENHAGE_DYADIC_CUTOFF_LEVEL {
            self.data = self.data * other.data;
            self.reduce(level);
            self
        } else {
            self.schoenhage(other)
        }
    }
}

impl<R> One for SchoenhageDyadicDomain<R>
where
    R: 'static + Ring + RingInvertible + Clone + Send + Sync,
{
    fn one() -> Self {
        Self {
            data: Polynomial::one(),
            level: LevelData::Undetermined,
        }
    }
}

pub fn schoenhage_dyadic_mul<R>(f: Polynomial<R>, g: Polynomial<R>) -> Polynomial<R>
where
    R: 'static + Ring + RingInvertible + Clone + Send + Sync,
{
    let n = f.degree() + g.degree();
    let n = n
        .checked_add(1)
        .and_then(usize::checked_next_power_of_two)
        .expect("polynomial too large");
    let level = n.trailing_zeros() as _;
    let mut f = SchoenhageDyadicDomain {
        data: f,
        level: LevelData::Undetermined,
    };
    f.reduce(level);
    let mut g = SchoenhageDyadicDomain {
        data: g,
        level: LevelData::Undetermined,
    };
    g.reduce(level);
    let SchoenhageDyadicDomain { data, .. } = f * g;
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::F7;

    use std::iter::{once, repeat};

    use alga::general::{Multiplicative, TwoSidedInverse};
    use rand::{rngs::OsRng, Rng, RngCore};

    type R = F7;
    type D = SchoenhageDyadicDomain<R>;

    #[test]
    fn unity_root_pow() {
        let x = D {
            data: Polynomial::new(vec![R::one()]),
            level: LevelData::Level(3),
        };
        let x = x.mul_unity_root_pow(4);
        let y = D {
            data: Polynomial::new(repeat(R::zero()).take(4).chain(once(R::one()))),
            level: LevelData::Level(3),
        };
        assert_eq!(x, y);
        let x = x.mul_unity_root_pow(4);
        let y = D {
            data: Polynomial::new(once(-R::one())),
            level: LevelData::Level(3),
        };
        assert_eq!(x, y);
        let mut unit = D::one();
        unit.reduce(3);
        assert_eq!(x.mul_unity_root_pow(8), unit);

        let x = D {
            data: Polynomial::new(vec![R::one()]),
            level: LevelData::Level(4),
        };
        let x = x.mul_unity_root_pow(8);
        let y = D {
            data: Polynomial::new(repeat(R::zero()).take(8).chain(once(R::one()))),
            level: LevelData::Level(4),
        };
        assert_eq!(x, y);
        let x = x.mul_unity_root_pow(8);
        let y = D {
            data: Polynomial::new(once(-R::one())),
            level: LevelData::Level(4),
        };
        assert_eq!(x, y);
        let mut unit = D::one();
        unit.reduce(4);
        assert_eq!(x.mul_unity_root_pow(16), unit);
    }

    #[test]
    fn fourier() {
        let k = 5;
        let n = 1 << k;
        let data: Vec<_> = (0..n)
            .map(|i| {
                let x: R = int_inj(i);
                D {
                    data: Polynomial::<R>::new((1..=n).map(int_inj)) * x,
                    level: LevelData::Level(k),
                }
            })
            .collect();
        let transformed = dyadic_transform(data.clone(), 2 * n);
        assert_eq!(data.len(), transformed.len());
        for (i, transformed) in transformed.iter().enumerate() {
            let mut r = D::zero();
            for (j, data) in data.iter().enumerate() {
                r = r + data.clone().mul_unity_root_pow(2 * i * j);
            }
            assert_eq!(*transformed, r, "mismatch at index {}", i);
        }
        let mut data_ = dyadic_transform(transformed, 2 * n);
        data_[1..].reverse();
        let n_inv = <R as TwoSidedInverse<Multiplicative>>::two_sided_inverse(&int_inj(n));
        let data_: Vec<_> = data_.into_par_iter().map(|x| x * n_inv.clone()).collect();
        assert_eq!(data_, data);
    }

    #[test]
    fn convolution_test() {
        let k = 5;
        let m = 1 << k;
        println!("k={} m={}", k, m);
        let f: Vec<_> = (0..m)
            .map(|_| D {
                data: Polynomial::<R>::new((1..=m).map(int_inj)),
                level: LevelData::Level(k),
            })
            .collect();
        let g: Vec<_> = (0..m)
            .map(|_| D {
                data: Polynomial::<R>::new((1..=m).rev().map(int_inj)),
                level: LevelData::Level(k),
            })
            .collect();
        let h = convolute(f.clone(), g.clone(), 2 * m);
        for (i, h) in h.into_iter().enumerate() {
            let mut r = D::zero();
            for (j, f) in f.iter().enumerate() {
                let f = f.clone();
                let g = g[(m + i - j) % m].clone();
                r = r + f * g;
            }
            assert_eq!(h, r);
        }
    }

    #[test]
    fn it_works() {
        let k = 4;
        let n = 1 << (k / 2);
        let f_orig = Polynomial::new((0..n).map(|_| int_inj(OsRng.gen::<u8>())));
        let f = D {
            data: f_orig.clone(),
            level: LevelData::Level(k),
        };
        let g_orig = Polynomial::new((0..n).map(|_| int_inj(OsRng.gen::<u8>())));
        let g = D {
            data: g_orig.clone(),
            level: LevelData::Level(k),
        };
        println!("{:?} {:?}", f_orig, g_orig);
        let D { data: h, .. } = f * g;
        let h_ = f_orig * g_orig;
        assert_eq!(h, h_);
    }

    #[test]
    fn mul_test() {
        let f_orig = Polynomial::<R>::new((0..800).map(|_| int_inj(OsRng.next_u32())));
        let g_orig = Polynomial::<R>::new((0..800).map(|_| int_inj(OsRng.next_u32())));
        let h = schoenhage_dyadic_mul(f_orig.clone(), g_orig.clone());
        assert_eq!(h, f_orig * g_orig);
    }
}
