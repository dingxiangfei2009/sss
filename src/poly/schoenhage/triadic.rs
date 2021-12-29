use super::*;

use std::{
    any::{Any, TypeId},
    cmp::max,
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

use crate::{field::int_inj, pow, ring::RingInvertible, Polynomial};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LevelData {
    Level(usize),
    Undetermined,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SchoenhageTriadicDomain<R> {
    level: LevelData,
    data: Vec<R>,
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
        let data = &mut self.data;
        if data.len() <= 2 * n {
            data.resize(3 * n, R::zero());
            return;
        }
        if data.len() > 3 * n {
            for chunk in data.split_off(3 * n).into_iter().chunks(3 * n).into_iter() {
                for (i, x) in chunk.into_iter().enumerate() {
                    data[i] += x;
                }
            }
        }
        data.resize(3 * n, R::zero());
        let (lower, upper) = data.split_at_mut(2 * n);
        for (i, x) in upper.iter_mut().enumerate() {
            let mut y = R::zero();
            swap(x, &mut y);
            lower[i] -= y.clone();
            lower[i + n] -= y;
        }
    }

    fn mul_unity_root_pow(mut self, power: usize) -> Self {
        self.mut_mul_unity_root_pow(power);
        self
    }

    fn mut_mul_unity_root_pow(&mut self, power: usize) {
        let (level, data) = if let Self {
            ref mut data,
            level: LevelData::Level(ref level),
        } = self
        {
            (level, data)
        } else {
            panic!("modulus is unknown")
        };
        let m = pow(3, *level);
        data.resize(3 * m, R::zero());
        data.rotate_right(power % (3 * m));
        let (lower, upper) = data.split_at_mut(2 * m);
        for (i, x) in upper.iter_mut().enumerate() {
            let mut y = R::zero();
            swap(x, &mut y);
            lower[i] -= y.clone();
            lower[i + m] -= y;
        }
    }

    fn fibers(self) -> (Vec<Self>, Vec<Self>) {
        let (level, data) = if let Self {
            data,
            level: LevelData::Level(level),
        } = self
        {
            (level, data)
        } else {
            panic!("modulus is unknown")
        };
        let cache = storage::<Self, SchoenhageTriadicDomainCache<R>, _>(|| Default::default());
        let SchoenhageTriadicFFTCache { m, t, m_level, .. } = &*cache.cache[level];
        let mut fibers = Vec::with_capacity(2 * t);
        for data_ in data.into_iter().chunks(*m).into_iter() {
            let mut data = Vec::with_capacity(3 * m);
            data.extend(data_);
            let mut data = Self {
                data,
                level: LevelData::Undetermined,
            };
            data.reduce(*m_level);
            fibers.push(data);
        }
        debug_assert!(fibers.len() <= 2 * t);
        fibers.resize_with(2 * t, || {
            let mut x = Self::zero();
            x.reduce(*m_level);
            x
        });
        let subring = |j: usize| {
            let mut r = Vec::with_capacity(*t);
            for (a, b) in fibers[..*t].iter().zip(fibers[*t..].iter()) {
                r.push(a.clone() + b.clone().mul_unity_root_pow(j * m));
            }
            debug_assert!(r.len() == *t);
            r
        };
        (subring(1), subring(2))
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
        let (mut f_1, mut f_2) = self.fibers();
        let (mut g_1, mut g_2) = other.fibers();

        for (i, f) in f_1.iter_mut().enumerate() {
            f.mut_mul_unity_root_pow(if m == t { i } else { i * 3 });
        }
        for (i, f) in f_2.iter_mut().enumerate() {
            f.mut_mul_unity_root_pow(if m == t { i * 2 } else { i * 6 });
        }
        for (i, g) in g_1.iter_mut().enumerate() {
            g.mut_mul_unity_root_pow(if m == t { i } else { i * 3 });
        }
        for (i, g) in g_2.iter_mut().enumerate() {
            g.mut_mul_unity_root_pow(if m == t { i * 2 } else { i * 6 });
        }

        let mut h_1 = convolute(f_1, g_1, *m);
        let mut h_2 = convolute(f_2, g_2, *m);
        for (i, h) in h_1.iter_mut().enumerate() {
            h.mut_mul_unity_root_pow(if m == t { 3 * m - i } else { 3 * m - i * 3 });
        }
        for (i, h) in h_2.iter_mut().enumerate() {
            h.mut_mul_unity_root_pow(if m == t { 3 * m - i * 2 } else { 3 * m - i * 6 });
        }

        let h_3 = h_1
            .clone()
            .into_iter()
            .zip(h_2.clone())
            .map(|(h_1, h_2)| h_2 - h_1);
        let h_4 = h_1
            .into_iter()
            .zip(h_2)
            .map(|(h_1, h_2)| h_1.mul_unity_root_pow(2 * m) - h_2.mul_unity_root_pow(*m))
            .chain(h_3)
            .map(|h| {
                h.clone().mul_unity_root_pow(*m) * int_inj::<R, _>(2) * three_inv.clone()
                    + h * three_inv.clone()
            });

        let n = pow(3, level) * 2;
        let mut data = vec![R::zero(); 2 * m + 2 * n];
        for (i, Self { data: a, .. }) in h_4.enumerate() {
            for (j, a) in a.into_iter().enumerate() {
                data[i * m + j] += a;
            }
        }
        let mut data = Self {
            data,
            level: LevelData::Undetermined,
        };
        data.reduce(level);

        data
    }
}

impl<R> SchoenhageTriadicDomain<R>
where
    R: Ring,
{
    fn mul_by_scalar(&mut self, y: R) {
        for x in &mut self.data {
            *x *= y.clone();
        }
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
    let mut r_0 = Vec::with_capacity(n_3rd);
    let mut r_1 = Vec::with_capacity(n_3rd);
    let mut r_2 = Vec::with_capacity(n_3rd);
    for j in 0..n_3rd {
        r_0.push(f[j].clone() + f[j + n_3rd].clone() + f[j + n_3rd * 2].clone());
        r_1.push(
            (f[j].clone()
                + f[j + n_3rd].clone().mul_unity_root_pow(order / 3)
                + f[j + n_3rd * 2].clone().mul_unity_root_pow(order / 3 * 2))
            .mul_unity_root_pow(j * q),
        );
        r_2.push(
            (f[j].clone()
                + f[j + n_3rd].clone().mul_unity_root_pow(order / 3 * 2)
                + f[j + n_3rd * 2].clone().mul_unity_root_pow(order / 3 * 4))
            .mul_unity_root_pow(j * q * 2),
        );
    }
    let r_0 = triadic_transform(r_0, order);
    let r_1 = triadic_transform(r_1, order);
    let r_2 = triadic_transform(r_2, order);

    let mut r = Vec::with_capacity(n);
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
    let mut f = triadic_transform(f, m * 3);
    let g = triadic_transform(g, m * 3);
    let mut temp = Zero::zero();
    for (f, g) in f.iter_mut().zip(g) {
        swap(&mut temp, f);
        temp = temp * g;
        swap(&mut temp, f);
    }
    let mut h = triadic_transform(f, m * 3);
    h[1..].reverse();
    for h in &mut h {
        h.mul_by_scalar(n_inv.clone());
    }
    h
}

impl<R> Mul<R> for SchoenhageTriadicDomain<R>
where
    R: Ring,
{
    type Output = Self;
    fn mul(mut self, y: R) -> Self {
        self.mul_by_scalar(y);
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
                let p = Polynomial::new(self.data);
                let q = Polynomial::new(other.data);
                let Polynomial(r) = p * q;
                self.data = r;
                return self;
            }
            _ => panic!("level mismatch"),
        };
        if level < SCHOENHAGE_TRIADIC_CUTOFF_LEVEL {
            let p = Polynomial::new(self.data);
            let q = Polynomial::new(other.data);
            let Polynomial(r) = p * q;
            self.data = r;
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
        let n = max(self.data.len(), other.data.len());
        self.data.resize(n, R::zero());
        other.data.resize(n, R::zero());
        for (x, y) in self.data.iter_mut().zip(other.data) {
            *x += y;
        }
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
        let n = max(self.data.len(), other.data.len());
        self.data.resize(n, R::zero());
        other.data.resize(n, R::zero());
        for (x, y) in self.data.iter_mut().zip(other.data) {
            *x -= y;
        }
        self
    }
}

impl<R> One for SchoenhageTriadicDomain<R>
where
    R: 'static + Ring + RingInvertible + Clone + Send + Sync,
{
    fn one() -> Self {
        Self {
            data: vec![R::one()],
            level: LevelData::Undetermined,
        }
    }
}

impl<R> Zero for SchoenhageTriadicDomain<R>
where
    R: 'static + Ring + RingInvertible + Clone + Send + Sync,
{
    fn is_zero(&self) -> bool {
        self.data.iter().all(Zero::is_zero)
    }
    fn zero() -> Self {
        Self {
            data: vec![R::zero()],
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
        data: f.0,
        level: LevelData::Undetermined,
    };
    f.reduce(level);
    let mut g = SchoenhageTriadicDomain {
        data: g.0,
        level: LevelData::Undetermined,
    };
    g.reduce(level);
    let SchoenhageTriadicDomain { data, .. } = f * g;
    Polynomial::from(data)
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
        let mut x = D::one();
        x.reduce(4);
        // 2n = 2 * 3^4 = 2 * 81 = 162
        let x = x.mul_unity_root_pow(161);
        let mut y = D {
            data: repeat(R::zero()).take(161).chain(once(R::one())).collect(),
            level: LevelData::Undetermined,
        };
        y.reduce(4);
        assert_eq!(x, y);
        let x = x.mul_unity_root_pow(1);
        let mut y = D {
            data: once(R::one())
                .chain(repeat(R::zero()).take(80))
                .chain(once(R::one()))
                .collect(),
            level: LevelData::Undetermined,
        };
        y.reduce(4);
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
            .map(|i| {
                let mut x = D {
                    data: (1..=n as u8).map(GF2561D).collect(),
                    level: LevelData::Undetermined,
                };
                x.reduce(k);
                x = x * GF2561D(i as u8 + 1);
                x
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
        let data_: Vec<_> = data_.into_iter().map(|x| x * n_inv.clone()).collect();
        assert_eq!(data_, data);

        let k = 3;
        let m = pow(3, k);
        let t = m / 3;
        let data: Vec<_> = (0..t)
            .map(|i| {
                let mut x = D {
                    data: (1..=m as u8).map(GF2561D).collect(),
                    level: LevelData::Undetermined,
                };
                x.reduce(k);
                x = x * GF2561D(i as u8 + 1);
                x
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
        let data_: Vec<_> = data_.into_iter().map(|x| x * n_inv.clone()).collect();
        assert_eq!(data_, data);
    }

    #[test]
    fn convolution_test() {
        let k = 2;
        let m = pow(3, k);
        let f: Vec<_> = (1..=m)
            .map(|_| {
                let mut x = D {
                    data: {
                        let mut data = vec![0; m];
                        OsRng.fill_bytes(&mut data);
                        data.into_iter().map(GF2561D).collect()
                    },
                    level: LevelData::Undetermined,
                };
                x.reduce(k);
                x
            })
            .collect();
        let g: Vec<_> = (1..=m)
            .map(|_| {
                let mut x = D {
                    data: {
                        let mut data = vec![0; m];
                        OsRng.fill_bytes(&mut data);
                        data.into_iter().map(GF2561D).collect()
                    },
                    level: LevelData::Undetermined,
                };
                x.reduce(k);
                x
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
        let f_orig: Vec<_> = f_orig.into_iter().map(GF2561D).collect();
        let mut f = D {
            data: f_orig.clone(),
            level: LevelData::Undetermined,
        };
        f.reduce(k);
        let mut g_orig = vec![0; n];
        OsRng.fill_bytes(&mut g_orig);
        let g_orig: Vec<_> = g_orig.into_iter().map(GF2561D).collect();
        let mut g = D {
            data: g_orig.clone(),
            level: LevelData::Undetermined,
        };
        g.reduce(k);
        let D { data: h, .. } = f * g;
        let h = Polynomial::new(h);
        let h_ = Polynomial::new(f_orig) * Polynomial::new(g_orig);
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
