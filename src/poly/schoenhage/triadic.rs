use super::*;

use std::{
    any::{Any, TypeId},
    collections::HashMap,
    marker::PhantomData,
    mem::swap,
    sync::{Arc, RwLock},
};

use alga::general::Ring;
use itertools::{izip, Itertools};
use lazy_static::lazy_static;
use num::traits::{One, Zero};
use once_cell::sync::Lazy;
use rayon::{join, prelude::*};

use crate::{field::int_inj, pow, ring::RingInvertible, Polynomial};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LevelData {
    Level(usize),
    Undetermined,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchoenhageTriadicDomain<R> {
    level: LevelData,
    data: Polynomial<R>,
}

lazy_static! {
    static ref TYPE_MAP: RwLock<HashMap<TypeId, Box<dyn Any + Send + Sync>>> = <_>::default();
}

fn storage<T, S, F>(init: F) -> Arc<S>
where
    T: 'static,
    S: 'static + Send + Sync,
    F: FnOnce() -> S,
{
    let r = TYPE_MAP.read().unwrap();
    let id = TypeId::of::<T>();
    if let Some(r) = r.get(&id) {
        Arc::clone(r.downcast_ref().unwrap())
    } else {
        drop(r);
        let mut r = TYPE_MAP.write().unwrap();
        let r = r
            .entry(id)
            .or_insert_with(|| Box::new(Arc::new(init())) as _);
        let r = r.downcast_ref().unwrap();
        Arc::clone(r)
    }
}

struct SchoenhageTriadicFFTCache<R> {
    m: usize,
    m_level: usize,
    t: usize,
    _p: PhantomData<fn() -> R>,
}

const SCHOENHAGE_TRIADIC_CUTOFF_LEVEL: usize = 3;

impl<R: Ring + Clone> SchoenhageTriadicFFTCache<R> {
    fn new(level: usize) -> Self {
        if level < SCHOENHAGE_TRIADIC_CUTOFF_LEVEL {
            panic!("level is too low; should use other algorithms for the modulo operation");
        }
        if level % 2 == 0 {
            let m_level = level / 2;
            let m = pow(3, m_level);
            Self {
                m,
                t: m,
                m_level,
                _p: PhantomData,
            }
        } else {
            let m_level = (level + 1) / 2;
            let t_level = level - m_level;
            let m = pow(3, m_level);
            let t = pow(3, t_level);
            Self {
                m,
                t,
                m_level,
                _p: PhantomData,
            }
        }
    }
}

struct SchoenhageTriadicDomainCache<R> {
    cache: Box<
        [Lazy<
            SchoenhageTriadicFFTCache<R>,
            Box<dyn Send + Sync + FnOnce() -> SchoenhageTriadicFFTCache<R>>,
        >],
    >,
}

impl<R: Ring + Clone> Default for SchoenhageTriadicDomainCache<R> {
    fn default() -> Self {
        let mut k = 0;
        let mut curr_len = 2usize;
        let mut cache = vec![];
        loop {
            cache.push(Lazy::new(
                Box::new(move || SchoenhageTriadicFFTCache::new(k)) as _,
            ));
            if let Some(new_len) = curr_len.checked_mul(3) {
                curr_len = new_len;
                k += 1;
            } else {
                break;
            }
        }
        Self {
            cache: cache.into_boxed_slice(),
        }
    }
}

impl<R> SchoenhageTriadicDomain<R>
where
    R: 'static + Ring + RingInvertible + Clone + Send + Sync,
{
    pub fn reduce(&mut self, level: usize) {
        let n = pow(3, level);
        self.level = LevelData::Level(level);
        if self.data.degree() < 2 * n {
            return;
        }
        let mut data = Polynomial::zero();
        swap(&mut data, &mut self.data);
        let Polynomial(mut data) = data;
        if data.len() >= 3 * n {
            for chunk in data.split_off(3 * n).into_iter().chunks(3 * n).into_iter() {
                for (i, x) in chunk.into_iter().enumerate() {
                    data[i] += x;
                }
            }
        }
        for (i, x) in data.split_off(2 * n).into_iter().enumerate() {
            data[i] -= x.clone();
            data[i + n] -= x;
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
        let m = pow(3, level);
        data.resize(3 * m, R::zero());
        data.rotate_right(power % (3 * m));
        for (i, excess) in data.split_off(2 * m).into_iter().enumerate() {
            data[i] -= excess.clone();
            data[i + m] -= excess;
        }
        Self {
            data: Polynomial::new(data),
            level: LevelData::Level(level),
        }
    }

    fn fibers(self) -> (Vec<Self>, Vec<Self>) {
        let (level, data) = if let Self {
            data: Polynomial(data),
            level: LevelData::Level(level),
        } = self
        {
            (level, data)
        } else {
            panic!("modulus is unknown")
        };
        let cache = storage::<Self, SchoenhageTriadicDomainCache<R>, _>(|| Default::default());
        let SchoenhageTriadicFFTCache { m, t, m_level, .. } = &*cache.cache[level];
        let mut fibers: Vec<_> = data
            .into_iter()
            .chunks(*m)
            .into_iter()
            .map(|f_| Self {
                data: Polynomial::new(f_),
                level: LevelData::Level(*m_level),
            })
            .collect();
        debug_assert!(fibers.len() <= 2 * t);
        fibers.resize_with(2 * t, || {
            let mut x = Self::zero();
            x.reduce(*m_level);
            x
        });
        let subring = |j: usize| {
            let mut r = vec![];
            for (a, b) in fibers[..*t]
                .iter()
                .cloned()
                .zip(fibers[*t..].iter().cloned())
            {
                r.push(a + b.mul_unity_root_pow(j * m));
            }
            debug_assert!(r.len() == *t);
            r
        };
        join(|| subring(1), || subring(2))
    }

    fn schoenhage(self, other: Self) -> Self {
        let cache = storage::<Self, SchoenhageTriadicDomainCache<R>, _>(|| Default::default());
        let three = int_inj(3);
        let three_inv = R::try_invert(three).expect("3 must be a unit");
        let level = match (self.level, other.level) {
            (LevelData::Level(level), LevelData::Level(level_)) if level == level_ => level,
            _ => panic!("modulus is unknown"),
        };
        let SchoenhageTriadicFFTCache { m, t, .. } = &*cache.cache[level];
        debug_assert!(m == t || *m == 3 * t);
        let ((f_1, f_2), (g_1, g_2)) = join(move || self.fibers(), move || other.fibers());
        let f_1 = f_1
            .into_par_iter()
            .enumerate()
            .map(|(i, f)| f.mul_unity_root_pow(if m == t { i } else { i * 3 }))
            .collect();
        let f_2 = f_2
            .into_par_iter()
            .enumerate()
            .map(|(i, f)| f.mul_unity_root_pow(if m == t { i * 2 } else { i * 6 }))
            .collect();
        let g_1 = g_1
            .into_par_iter()
            .enumerate()
            .map(|(i, g)| g.mul_unity_root_pow(if m == t { i } else { i * 3 }))
            .collect();
        let g_2 = g_2
            .into_par_iter()
            .enumerate()
            .map(|(i, g)| g.mul_unity_root_pow(if m == t { i * 2 } else { i * 6 }))
            .collect();

        let h_1 = convolute(f_1, g_1, *m);
        let h_2 = convolute(f_2, g_2, *m);
        let h_1: Vec<_> = h_1
            .into_par_iter()
            .enumerate()
            .map(|(i, h)| h.mul_unity_root_pow(if m == t { 3 * m - i } else { 3 * m - 3 * i }))
            .collect();
        let h_2: Vec<_> = h_2
            .into_par_iter()
            .enumerate()
            .map(|(i, h)| h.mul_unity_root_pow(if m == t { 3 * m - 2 * i } else { 3 * m - 6 * i }))
            .collect();

        let h_3: Vec<_> = h_1
            .par_iter()
            .cloned()
            .zip(h_2.par_iter().cloned())
            .map(|(h_1, h_2)| h_2 - h_1)
            .collect();
        let h_4: Vec<_> = h_1
            .into_par_iter()
            .zip(h_2.into_par_iter())
            .map(|(h_1, h_2)| h_1.mul_unity_root_pow(2 * m) - h_2.mul_unity_root_pow(*m))
            .chain(h_3)
            .collect();
        let h_4: Vec<_> = h_4
            .into_par_iter()
            .map(|h| {
                h.clone().mul_unity_root_pow(*m) * int_inj::<R, _>(2) * three_inv.clone()
                    + h * three_inv.clone()
            })
            .collect();

        let n = pow(3, level) * 2;
        let mut data = vec![R::zero(); 2 * m + 2 * n];
        for (i, a) in h_4.into_iter().enumerate() {
            let Self {
                data: Polynomial(a),
                ..
            } = a;
            for (j, a) in a.into_iter().enumerate() {
                data[i * m + j] += a;
            }
        }
        let mut data = Self {
            data: Polynomial::new(data),
            level: LevelData::Undetermined,
        };
        data.reduce(level);

        data
    }
}

// essentially cooley-tukey
// HACK: there is an outstanding issue with overflowing
// `usize` in the multiplications below
fn triadic_transform<R>(
    f: Vec<SchoenhageTriadicDomain<R>>,
    order: usize,
) -> Vec<SchoenhageTriadicDomain<R>>
where
    R: 'static + Ring + RingInvertible + Clone + Send + Sync,
{
    let n = f.len();
    if n == 1 {
        return f;
    }
    let n_3rd = n / 3;
    let q = order / n;
    assert_eq!(n % 3, 0);
    assert_eq!(order % n, 0);
    // o = w ^ q
    // xi = o ^ (n / 3) = w ^ (q * n / 3) = w ^ (order / 3)
    let r_0: Vec<_> = (0..n_3rd)
        .map(|j| f[j].clone() + f[j + n_3rd].clone() + f[j + n_3rd * 2].clone())
        .collect();
    let r_1: Vec<_> = (0..n_3rd)
        .map(|j| {
            (f[j].clone()
                + f[j + n_3rd].clone().mul_unity_root_pow(order / 3)
                + f[j + n_3rd * 2].clone().mul_unity_root_pow(order / 3 * 2))
            .mul_unity_root_pow(j * q)
        })
        .collect();
    let r_2: Vec<_> = (0..n_3rd)
        .map(|j| {
            (f[j].clone()
                + f[j + n_3rd].clone().mul_unity_root_pow(order / 3 * 2)
                + f[j + n_3rd * 2].clone().mul_unity_root_pow(order / 3 * 4))
            .mul_unity_root_pow(j * q * 2)
        })
        .collect();
    let (r_0, (r_1, r_2)) = join(
        || triadic_transform(r_0, order),
        || {
            join(
                || triadic_transform(r_1, order),
                || triadic_transform(r_2, order),
            )
        },
    );

    let mut r = vec![];
    for (r_0, r_1, r_2) in izip!(r_0, r_1, r_2) {
        r.push(r_0);
        r.push(r_1);
        r.push(r_2);
    }

    r
}

fn convolute<R>(
    f: Vec<SchoenhageTriadicDomain<R>>,
    g: Vec<SchoenhageTriadicDomain<R>>,
    m: usize,
) -> Vec<SchoenhageTriadicDomain<R>>
where
    R: 'static + Ring + RingInvertible + Clone + Send + Sync,
{
    debug_assert!(f.len() == g.len());
    let n = f.len();
    let n_inv = R::try_invert(int_inj(n)).expect("n should be invertible");
    let (f, g) = join(
        move || triadic_transform(f, m * 3),
        move || triadic_transform(g, m * 3),
    );
    let mut h = triadic_transform(
        f.into_par_iter().zip(g).map(|(f, g)| f * g).collect(),
        m * 3,
    );
    h[1..].reverse();
    h.into_par_iter().map(|h| h * n_inv.clone()).collect()
}

impl<R> Mul<R> for SchoenhageTriadicDomain<R>
where
    R: Ring,
{
    type Output = Self;
    fn mul(mut self, other: R) -> Self {
        self.data = self.data * other;
        self
    }
}

impl<R> Mul for SchoenhageTriadicDomain<R>
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
        if level < SCHOENHAGE_TRIADIC_CUTOFF_LEVEL {
            self.data = self.data * other.data;
            self.reduce(level);
            self
        } else {
            self.schoenhage(other)
        }
    }
}

impl<R> Add for SchoenhageTriadicDomain<R>
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

impl<R> Sub for SchoenhageTriadicDomain<R>
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

impl<R> One for SchoenhageTriadicDomain<R>
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

impl<R> Zero for SchoenhageTriadicDomain<R>
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

pub fn schoenhage_triadic_mul<R>(f: Polynomial<R>, g: Polynomial<R>) -> Polynomial<R>
where
    R: 'static + Ring + RingInvertible + Clone + Send + Sync,
{
    let n_ = f.degree() + g.degree();
    let mut n = 2;
    let mut level = 0;
    while n <= n_ {
        level += 1;
        n *= 3;
    }
    let mut f = SchoenhageTriadicDomain {
        data: f,
        level: LevelData::Undetermined,
    };
    f.reduce(level);
    let mut g = SchoenhageTriadicDomain {
        data: g,
        level: LevelData::Undetermined,
    };
    g.reduce(level);
    let SchoenhageTriadicDomain { data, .. } = f * g;
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::field::GF2561D;

    use std::iter::{once, repeat};

    use alga::general::{Multiplicative, TwoSidedInverse};
    use rand::{rngs::OsRng, RngCore};

    type R = GF2561D;
    type D = SchoenhageTriadicDomain<R>;

    #[test]
    fn unity_root_pow() {
        let x = D {
            data: Polynomial::new(vec![R::one()]),
            level: LevelData::Level(4),
        };
        // 2n = 2 * 3^4 = 2 * 81 = 162
        let x = x.mul_unity_root_pow(161);
        let y = D {
            data: Polynomial::new(repeat(R::zero()).take(161).chain(once(R::one()))),
            level: LevelData::Level(4),
        };
        assert_eq!(x, y);
        let x = x.mul_unity_root_pow(1);
        let y = D {
            data: Polynomial::new(
                once(R::one())
                    .chain(repeat(R::zero()).take(80))
                    .chain(once(R::one())),
            ),
            level: LevelData::Level(4),
        };
        assert_eq!(x, y);
        let mut unit = D::one();
        unit.reduce(4);
        assert_eq!(x.mul_unity_root_pow(81), unit);
    }

    #[test]
    fn fourier() {
        let k = 2;
        let n = pow(3, k);
        let data: Vec<_> = (0..n)
            .map(|i| D {
                data: Polynomial::new((1..=n as u8).map(GF2561D)) * GF2561D(i as u8 + 1),
                level: LevelData::Level(k),
            })
            .collect();
        let transformed = triadic_transform(data.clone(), 3 * n);
        assert_eq!(data.len(), transformed.len());
        for (i, transformed) in transformed.iter().enumerate() {
            let mut r = D::zero();
            for (j, data) in data.iter().enumerate() {
                r = r + data.clone().mul_unity_root_pow(3 * i * j);
            }
            assert_eq!(*transformed, r, "mismatch at index {}", i);
        }
        let mut data_ = triadic_transform(transformed, 3 * n);
        data_[1..].reverse();
        let n_inv = <R as TwoSidedInverse<Multiplicative>>::two_sided_inverse(&int_inj(n));
        let data_: Vec<_> = data_.into_par_iter().map(|x| x * n_inv.clone()).collect();
        assert_eq!(data_, data);

        let k = 3;
        let m = pow(3, k);
        let t = m / 3;
        let data: Vec<_> = (0..t)
            .map(|i| D {
                data: Polynomial::new((1..=m as u8).map(GF2561D)) * GF2561D(i as u8 + 1),
                level: LevelData::Level(k),
            })
            .collect();
        let transformed = triadic_transform(data.clone(), 3 * m);
        assert_eq!(data.len(), transformed.len());
        for (i, transformed) in transformed.iter().enumerate() {
            let mut r = D::zero();
            for (j, data) in data.iter().enumerate() {
                r = r + data.clone().mul_unity_root_pow(9 * i * j);
            }
            assert_eq!(*transformed, r, "mismatch at index {}", i);
        }
        let mut data_ = triadic_transform(transformed, 3 * m);
        data_[1..].reverse();
        let n_inv = <R as TwoSidedInverse<Multiplicative>>::two_sided_inverse(&int_inj(t));
        let data_: Vec<_> = data_.into_par_iter().map(|x| x * n_inv.clone()).collect();
        assert_eq!(data_, data);
    }

    #[test]
    fn convolution_test() {
        let k = 2;
        let m = pow(3, k);
        let f: Vec<_> = (1..=m)
            .map(|_| D {
                data: {
                    let mut data = vec![0; m];
                    OsRng.fill_bytes(&mut data);
                    Polynomial::new(data.into_iter().map(GF2561D))
                },
                level: LevelData::Level(k),
            })
            .collect();
        let g: Vec<_> = (1..=m)
            .map(|_| D {
                data: {
                    let mut data = vec![0; m];
                    OsRng.fill_bytes(&mut data);
                    Polynomial::new(data.into_iter().map(GF2561D))
                },
                level: LevelData::Level(k),
            })
            .collect();
        let h = convolute(f.clone(), g.clone(), m);
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
        let n = pow(3, k);
        let mut f_orig = vec![0; n];
        OsRng.fill_bytes(&mut f_orig);
        let f_orig = Polynomial::new(f_orig.into_iter().map(GF2561D));
        let f = D {
            data: f_orig.clone(),
            level: LevelData::Level(k),
        };
        let mut g_orig = vec![0; n];
        OsRng.fill_bytes(&mut g_orig);
        let g_orig = Polynomial::new(g_orig.into_iter().map(GF2561D));
        let g = D {
            data: g_orig.clone(),
            level: LevelData::Level(k),
        };
        let D { data: h, .. } = f * g;
        let h_ = f_orig * g_orig;
        assert_eq!(h, h_);
    }

    #[test]
    fn mul_test() {
        let mut f_orig = vec![0; 800];
        OsRng.fill_bytes(&mut f_orig);
        let f_orig = Polynomial::new(f_orig.into_iter().map(|x| GF2561D(x as u8)));

        let mut g_orig = vec![0; 800];
        OsRng.fill_bytes(&mut g_orig);
        let g_orig = Polynomial::new(g_orig.into_iter().map(|x| GF2561D(x as u8)));
        let h = schoenhage_triadic_mul(f_orig.clone(), g_orig.clone());
        assert_eq!(h, f_orig * g_orig);
    }

    #[test]
    fn fast_mul_fixed() {
        let f_orig = Polynomial(vec![GF2561D(236), GF2561D(34)]);
        let g_orig = Polynomial(vec![
            GF2561D(49),
            GF2561D(103),
            GF2561D(246),
            GF2561D(7),
            GF2561D(180),
            GF2561D(209),
            GF2561D(126),
            GF2561D(54),
            GF2561D(75),
            GF2561D(153),
            GF2561D(210),
            GF2561D(197),
            GF2561D(131),
            GF2561D(237),
            GF2561D(57),
            GF2561D(229),
            GF2561D(181),
            GF2561D(117),
        ]);
        let h = schoenhage_triadic_mul(f_orig.clone(), g_orig.clone());
        assert_eq!(h, f_orig * g_orig);
    }
}
