//! RLWE Key Exchange with the following parameter
//! Key size n = 2 ^ 12 = 4096
//! Security level kappa <= 0.97 * n = 3973
//! Field: GF(40961) whose characteristic q ~ n ^ 3
//! f = x ^ (2 ^ 12) + 1
//! sigma = 4, such that n >> sigma > omega(sqrt(log_2(n))) > 1
//! t = 2 is a small prime, such that t << n << q
//! q = 3 * 2 ^ 36 + 1 = 206158430209
//! gamma = sigma = 4

use std::{
    mem::{transmute, MaybeUninit},
    pin::Pin,
    sync::Arc,
};

use lazy_static::lazy_static;
use ndarray::{Array1, Array2};
use num::Zero;
use rand::{CryptoRng, RngCore};
use rand_distr::{Distribution, Normal, Uniform};

use crate::{
    field::{int_inj, FiniteField, F206158430209 as F},
    fourier::{cooley_tukey, naive, F206158430209G11_UNITY_ROOT as UNITY_ROOT},
};

/// key size n = 2^12
pub const KEY_SIZE: usize = 4096;

/// stddev alpha
const SIGMA_A: f64 = 4.;
/// stddev beta
const SIGMA_B: f64 = 4.;
/// stddev gamma
const SIGMA_G: f64 = 4.;

pub struct Init([F; KEY_SIZE]);

impl Init {
    pub fn new<R: CryptoRng + RngCore>(rng: &mut R) -> Self {
        let mut init = [F::zero(); KEY_SIZE];
        for x in &mut init[..] {
            *x = int_inj(Uniform::from(0..<F as FiniteField>::CHARACTERISTIC as u128).sample(rng))
        }
        Self(init)
    }
}

type FFTOP = dyn Send + Sync + Fn(Vec<F>) -> Vec<F>;

type ArcFFTOP = Pin<Arc<FFTOP>>;

fn construct_fft_2_12() -> ArcFFTOP {
    fn clone_fn(
        f: &Arc<impl Send + Sync + Fn(Vec<F>) -> Vec<F>>,
    ) -> impl Sync + Fn(Vec<F>) -> Vec<F> {
        let f = Arc::clone(f);
        move |x| f(x)
    }

    let fft_2_1 = Arc::new(naive(UNITY_ROOT.clone().subgroup(2)));
    let fft_2_2 = Arc::new(cooley_tukey(
        2,
        2,
        UNITY_ROOT.clone().subgroup(4),
        clone_fn(&fft_2_1),
        clone_fn(&fft_2_1),
    ));
    let fft_2_3 = Arc::new(cooley_tukey(
        2,
        4,
        UNITY_ROOT.clone().subgroup(8),
        clone_fn(&fft_2_1),
        clone_fn(&fft_2_2),
    ));
    let fft_2_6 = Arc::new(cooley_tukey(
        8,
        8,
        UNITY_ROOT.clone().subgroup(64),
        clone_fn(&fft_2_3),
        clone_fn(&fft_2_3),
    ));
    Arc::pin(cooley_tukey(
        64,
        64,
        UNITY_ROOT.clone().subgroup(KEY_SIZE),
        clone_fn(&fft_2_6),
        clone_fn(&fft_2_6),
    ))
}

fn vec_to_poly(a: Vec<F>) -> Poly {
    let mut r: [MaybeUninit<F>; KEY_SIZE] = [MaybeUninit::uninit(); KEY_SIZE];
    for (r, a) in r.iter_mut().zip(a.into_iter()) {
        unsafe { r.as_mut_ptr().write(a) }
    }
    unsafe { transmute(r) }
}

lazy_static! {
    static ref F_DISTR_A: Normal<f64> =
        { Normal::new(0., SIGMA_A as f64).expect("stddev should be correct") };
    static ref F_DISTR_B: Normal<f64> =
        { Normal::new(0., SIGMA_B as f64).expect("stddev should be correct") };
    static ref FFT_2_12: ArcFFTOP = construct_fft_2_12();
    static ref PHI: Array1<F> = {
        let mut phi = vec![];
        let g = UNITY_ROOT.clone().subgroup(2 * KEY_SIZE).root;
        let mut a = g;
        for _ in 0..KEY_SIZE {
            phi.push(a);
            a *= g;
        }
        Array1::from(phi)
    };
    static ref PHI_INV: Array1<F> = {
        use alga::general::{Multiplicative, TwoSidedInverse};
        PHI.iter()
            .map(|x| <F as TwoSidedInverse<Multiplicative>>::two_sided_inverse(x))
            .collect()
    };
    static ref F_CONST_2: F = int_inj(2);
}

type Poly = [F; KEY_SIZE];

pub struct PrivateKey {
    s: Poly,
    e: Poly,
}

pub struct PublicKey(Poly);

fn poly_mul_mod(a: Poly, b: Poly) -> Poly {
    vec_to_poly(
        (Array1::from(
            Array1::from(FFT_2_12((Array1::from(a.to_vec()) * PHI.clone()).to_vec()))
                * Array1::from(FFT_2_12(b.to_vec())),
        ) * PHI_INV.clone())
        .to_vec(),
    )
}

fn poly_add(mut a: Poly, b: Poly) -> Poly {
    for (a, b) in a.iter_mut().zip(b.to_vec().into_iter()) {
        *a += b
    }
    a
}

pub fn keygen<R: CryptoRng + RngCore>(rng: &mut R, init: &Init) -> (PrivateKey, PublicKey) {
    let mut s = [F::zero(); KEY_SIZE];
    for x in &mut s[..] {
        *x = int_inj(Uniform::from(0..<F as FiniteField>::CHARACTERISTIC as u128).sample(rng))
    }
    let Init(init) = init;
    let mut e = [F::zero(); KEY_SIZE];
    for x in &mut e[..] {
        let v = F_DISTR_A.sample(rng).round() as i128;
        if v < 0 {
            *x = -int_inj::<F, _>(-v)
        } else {
            *x = int_inj(v)
        }
    }
    let public = poly_add(poly_mul_mod(init.clone(), s.clone()), {
        let mut e = e.clone();
        for x in &mut e[..] {
            *x *= F_CONST_2.clone();
        }
        e
    });
    (PrivateKey { s, e }, PublicKey(public))
    // let s = toeplitz_unit_mul(
    //     &*E,
    //     Array1::from(mat_vec_mul(g, &init_)) * toeplitz_unit_mul(h, s),
    // );
    // let e = Array1::from(
    //     (0..KEY_SIZE)
    //         .map(|_| -> F {
    //             let x = F_DISTR.sample(rng).round() as i128;
    //             if x < 0 {
    //                 -int_inj::<F, _>(-x)
    //             } else {
    //                 int_inj(x)
    //             }
    //         })
    //         .collect::<Vec<_>>(),
    // );

    // let mut public = [F::zero(); KEY_SIZE];
    // public.copy_from_slice(&(s.clone() + e * *T_PRIME).to_vec());
    // let mut private = [F::zero(); KEY_SIZE];
    // private.copy_from_slice(&s.to_vec());
    // (PrivateKey(private), PublicKey(public))
}
