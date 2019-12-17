use std::{borrow::Borrow, marker::PhantomData, thread::spawn};

use cached::cached;
use crossbeam::channel::{bounded, Receiver, Select};
use lazy_static::lazy_static;
use rand::{CryptoRng, RngCore};
use rug::{float::Constant, ops::Pow, Float, Integer};

pub struct BaseSampler<T> {
    levels: Vec<(Vec<usize>, Vec<(usize, usize)>)>,
    values: Vec<T>,
}

pub struct BitPool {
    counter: u8,
    bits: u64,
}

impl BitPool {
    pub fn new() -> Self {
        Self {
            counter: 0,
            bits: 0,
        }
    }
    pub fn take_bit<R: RngCore>(&mut self, rng: &mut R) -> bool {
        if self.counter > 0 {
            let bit = self.bits & 1;
            self.bits >>= 1;
            self.counter -= 1;
            bit > 0
        } else {
            self.bits = rng.next_u64();
            self.counter = 64;
            self.take_bit(rng)
        }
    }
}

fn upper_bound<T: Ord>(a: impl AsRef<[T]>, key: impl Borrow<T>) -> Option<usize> {
    let a = a.as_ref();
    let key = key.borrow();
    let mut low = 0;
    let mut high = a.len();
    loop {
        if low + 2 < high {
            let mid = (low + high) / 2;
            if &a[mid] > key {
                high = mid + 1;
            } else {
                low = mid + 1;
            }
        } else if low + 1 < high {
            if &a[low] > key {
                return Some(low);
            } else if &a[low + 1] > key {
                return Some(low + 1);
            } else {
                return None;
            }
        } else if &a[low] > key {
            return Some(low);
        } else {
            return None;
        }
    }
}

impl<T> BaseSampler<T> {
    pub fn sample<'a, R: RngCore + CryptoRng>(&'a self, rng: &mut R) -> &'a T {
        let mut bit_pool = BitPool::new();
        let mut choice = 0;
        let mut value = None;
        for (index, blocks) in &self.levels {
            let random_bit = bit_pool.take_bit(rng) as usize;
            choice = 2 * choice + 1 - random_bit;
            if let Some(block_index) = upper_bound(&index, choice) {
                let mut selection = 0;
                for i in 0..block_index {
                    choice -= blocks[i].1;
                    selection += blocks[i].0 + blocks[i].1;
                }
                selection += blocks[block_index].0 + choice + 1;
                if let None = value {
                    value = Some(selection);
                } else {
                    choice = 0;
                }
            } else {
                choice -= index[index.len() - 1];
            }
        }
        value
            .map(|idx| &self.values[idx])
            .unwrap_or(&self.values[0])
    }

    pub fn generate<P>(
        support: impl IntoIterator<Item = T>,
        pdf: impl Fn(&T) -> P,
        precision: u32,
    ) -> Self
    where
        Float: From<P>,
    {
        let mut distri = vec![];
        let mut values = vec![];
        let mut total = Float::new(precision);
        for x in support {
            let p = Float::from(pdf(&x));
            distri.push(p.clone());
            values.push(x);
            total += p;
        }
        let precision_mask = Integer::from(1) << precision;
        let distribution: Vec<_> = distri
            .into_iter()
            .map(|p| {
                (p / total.clone() * precision_mask.clone())
                    .to_integer()
                    .expect("probability should be positive")
            })
            .collect();

        let mut levels = vec![];
        for pos in (0..precision).rev() {
            let mut blocks = vec![];
            let mut block = (0, 0);
            let mut ones = false;
            for x in &distribution {
                let bit = x.get_bit(pos);
                if bit ^ ones {
                    if ones {
                        blocks.push(block);
                        block = (1, 0);
                    } else {
                        block.1 += 1;
                    }
                    ones ^= true;
                } else {
                    if ones {
                        block.1 += 1;
                    } else {
                        block.0 += 1;
                    }
                }
            }
            blocks.push(block);
            let mut index = vec![];
            let mut acc = 0;
            for &(_, ones) in &blocks {
                acc += ones;
                index.push(acc);
            }
            levels.push((index, blocks));
        }

        Self { levels, values }
    }
}

pub const BASE_SAMPLER_B: usize = 2;
/// k=ceil(30/log_2(b))
pub const PRECISION_K: u32 = 30;
pub const PRECISION: u32 = 128;
pub const ETA: usize = 6;

fn compute_s_z_i(i: u32, s_z_i: impl Fn(u32) -> (Float, Integer)) -> (Float, Integer) {
    assert!(i > 0);
    if i > 1 {
        let (mut s, _) = s_z_i(i - 1);
        let z = s.clone() * Float::with_val(PRECISION, Constant::Pi).sqrt()
            / Float::with_val(PRECISION, ETA);
        let z = z.floor().to_integer().expect("no integer representation");
        let s_ = z.clone().square()
            + std::cmp::max(Integer::from(1), (z.clone() - Integer::from(1)).square());
        s *= s_.sqrt();
        (s, z)
    } else {
        (S_0.clone(), 0.into())
    }
}

cached! {
    S_Z_I;
    fn s_z_i(i: u32) -> (Float, Integer) = {
        compute_s_z_i(i, s_z_i)
    }
}

pub struct GenericSampler;

impl GenericSampler {
    fn sample_i<R: RngCore + CryptoRng>(rng: &mut R, i: u32) -> Integer {
        if i > 0 {
            let x_1 = Self::sample_i(rng, i - 1);
            let x_2 = Self::sample_i(rng, i - 1);
            let (_, z_i) = s_z_i(i);
            &z_i * x_1 + std::cmp::max(Integer::from(1), z_i - Integer::from(1)) * x_2
        } else {
            BASE_SAMPLERS[0].sample(rng).into()
        }
    }

    fn sample_c<R: RngCore + CryptoRng>(rng: &mut R, c: Integer, k: u32) -> Integer {
        if k == 0 {
            c
        } else {
            let g = BASE_SAMPLERS[c.mod_u(BASE_SAMPLER_B as u32) as usize].sample(rng)
                / Integer::from(BASE_SAMPLER_B).pow(k);
            g + Self::sample_c(rng, c / Integer::from(BASE_SAMPLER_B), k - 1)
        }
    }

    pub fn sample<R: RngCore + CryptoRng>(rng: &mut R, c: Float, s: Float, max: u32) -> Integer {
        let precision = B_MASK.significant_bits() as u32 * 2;
        let (s_max, _) = s_z_i(max);
        assert!(
            s > *S_BAR && &s / S_BAR.clone() <= &s_max / Float::with_val(4, ETA),
            "s/s_bar({}) should be in range between 1 and {}, s_bar={}",
            &s / S_BAR.clone(),
            &s_max / Float::with_val(4, ETA),
            &*S_BAR,
        );
        let x = Self::sample_i(rng, max);

        let scale_k = Float::with_val(
            std::cmp::max(PRECISION, precision + x.significant_bits() as u32),
            s.square_ref(),
        ) - Float::with_val(precision, S_BAR.square_ref());
        let scale_k = scale_k.sqrt() / s_max;

        let x = scale_k * x;
        let c = x + Float::with_val(precision, c);
        let c = c * &*B_MASK;
        let c_int = c
            .clone()
            .floor()
            .to_integer()
            .expect("no integer representation");
        let c = Self::random_round(rng, c - &c_int) + c_int;
        Self::sample_c(rng, c, PRECISION_K)
    }

    fn random_round<R: RngCore + CryptoRng>(rng: &mut R, mut a: Float) -> Integer {
        assert!(a >= 0 && a <= 1, "a({}) not in range", a);
        let mut bit_pool = BitPool::new();
        let mut b: Float = 1 - a.clone();
        let mut d = 0;
        loop {
            d = 2 * d + 1 - bit_pool.take_bit(rng) as u32;
            if d > Float::with_val(4, b.trunc_ref()) {
                d -= 1;
            } else {
                return 0.into();
            }
            if d > Float::with_val(4, a.trunc_ref()) {
                d -= 1;
            } else {
                return 1.into();
            }
            a *= 2;
            b *= 2;
            a.fract_mut();
            b.fract_mut();
        }
    }
}

/// *NOTE*: The normal distribution function is exp(-pi * x^2). To use the standard normal distribution,
/// scale the standard deviation by (2 * pi)^-1
pub fn make_base_samplers() -> Vec<BaseSampler<Integer>> {
    let sigma = Float::with_val(PRECISION, 13.57);

    (0..BASE_SAMPLER_B)
        .map(|c| {
            let centre = Float::with_val(PRECISION, c) / Float::with_val(PRECISION, BASE_SAMPLER_B);
            make_gaussian_sampler(centre, sigma.clone())
        })
        .collect()
}

pub fn make_gaussian_sampler(mean: Float, sigma: Float) -> BaseSampler<Integer> {
    let pi = Float::with_val(PRECISION, Constant::Pi);
    let sqrt_2_pi = (pi * 2u8).sqrt();
    let cut = sqrt_2_pi * ETA as f64 * sigma.clone();
    let cut_bound = cut
        .ceil()
        .to_integer()
        .expect("cannot represent the tail cut limit")
        .to_i64()
        .expect("parameter is too large, so that the memo table is too big");
    let pdf = |x: &Integer| (-((x - mean.clone()) / &sigma).square() / 2u8).exp();
    let support = (-cut_bound..cut_bound).map(|x| {
        ((Float::with_val(PRECISION, x) + &mean).floor())
            .to_integer()
            .expect("value has no finite representation")
    });
    BaseSampler::generate(support, pdf, PRECISION)
}

lazy_static! {
    static ref B_MASK: Integer = Integer::from(BASE_SAMPLER_B).pow(PRECISION_K);
    static ref S_0: Float = Float::with_val(PRECISION, 13.57);
    static ref S_BAR: Float = {
        let precision = 2 * B_MASK.significant_bits() as u32;
        let xs: Vec<_> = (0..PRECISION_K)
            .map(|i| {
                Float::with_val(precision, BASE_SAMPLER_B)
                    .pow(2 * i)
                    .recip()
            })
            .collect();
        S_0.clone() * Float::with_val(precision, Float::sum(xs.iter())).sqrt()
    };
    pub static ref BASE_SAMPLERS: Vec<BaseSampler<Integer>> = make_base_samplers();
}

#[derive(Clone)]
pub struct ParallelGenericSampler<R>(Receiver<Integer>, PhantomData<fn() -> R>);

impl<R> ParallelGenericSampler<R> {
    pub fn new(rngs: Vec<R>, buffer_size: usize, mean: Float, sigma: Float, max: u32) -> Self
    where
        R: CryptoRng + RngCore + Send + 'static,
    {
        let mut rxs: Vec<_> = rngs
            .into_iter()
            .map(move |mut rng| {
                let mean = mean.clone();
                let sigma = sigma.clone();
                let (tx, rx) = bounded(buffer_size);
                spawn(move || loop {
                    if let Err(_) = tx.send(GenericSampler::sample(
                        &mut rng,
                        mean.clone(),
                        sigma.clone(),
                        max,
                    )) {
                        break;
                    }
                });
                rx
            })
            .collect();
        let (tx, rx) = bounded(buffer_size);
        spawn(move || loop {
            let mut select = Select::new();
            for rx in &rxs {
                select.recv(rx);
            }
            let op = select.select();
            let index = op.index();
            match op.recv(&rxs[index]) {
                Ok(x) => {
                    if let Err(_) = tx.send(x) {
                        break;
                    }
                }
                Err(_) => {
                    rxs.remove(index);
                }
            }
        });
        Self(rx, PhantomData)
    }

    pub fn sample(&self) -> Integer {
        self.0
            .recv()
            .expect("parallel sampler working in other threads disappeared")
    }
}
