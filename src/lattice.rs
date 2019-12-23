//! RLWE Key Exchange with the following parameter
//! Key size n = 2 ^ 12 = 4096
//! Security level kappa <= 0.97 * n = 3973
//! Field: GF(40961) whose characteristic q ~ n ^ 3
//! f = x ^ (2 ^ 12) + 1
//! sigma = 4, such that n >> sigma > omega(sqrt(log_2(n))) > 1
//! t = 2 is a small prime, such that t << n << q
//! q = 273 * 2 ^ 72 + 1
//! gamma = sigma = 4

use std::{
    fmt::{Debug, Display, Formatter, Result as FmtResult},
    iter::repeat_with,
    marker::PhantomData,
    mem::{transmute, MaybeUninit},
    ops::{AddAssign, Deref, DerefMut, Div, DivAssign, Mul, Rem, ShlAssign},
    pin::Pin,
    sync::Arc,
};

use lazy_static::lazy_static;
use ndarray::Array1;
use num::{One, Zero};
use rand::{CryptoRng, RngCore, SeedableRng};
use rug::{Float, Integer};
use serde::{
    de::{Deserializer, Error as DeserializeError, SeqAccess, Visitor},
    ser::{SerializeSeq, Serializer},
    Deserialize, Serialize,
};

use crate::{
    field::{int_inj, FiniteField, Fp, PrimeModulo},
    fourier::{cooley_tukey, naive, UnityRoot},
    gaussian::{make_gaussian_sampler, BaseSampler, ParallelGenericSampler, PRECISION},
    EuclideanDomain,
};

#[derive(
    Add,
    AddAssign,
    BitAnd,
    Clone,
    Debug,
    Div,
    DivAssign,
    Eq,
    From,
    Mul,
    MulAssign,
    Neg,
    Ord,
    PartialEq,
    PartialOrd,
    Shl,
    ShlAssign,
    Sub,
    SubAssign,
    Rem,
    Serialize,
    Deserialize,
    Display,
)]
pub struct Int(Integer);

impl From<u16> for Int {
    fn from(x: u16) -> Self {
        Self(x.into())
    }
}

impl Zero for Int {
    fn zero() -> Self {
        Integer::from(0).into()
    }

    fn is_zero(&self) -> bool {
        self.0 == 0
    }
}

impl Mul<Int> for Integer {
    type Output = Integer;
    fn mul(self, other: Int) -> Self::Output {
        self * other.0
    }
}

impl Div<Int> for Integer {
    type Output = Integer;
    fn div(self, other: Int) -> Self::Output {
        self / other.0
    }
}

impl DivAssign<Int> for Integer {
    fn div_assign(&mut self, other: Int) {
        *self /= other.0
    }
}

impl Rem<Int> for Integer {
    type Output = Integer;
    fn rem(self, other: Int) -> Self::Output {
        self % other.0
    }
}

impl One for Int {
    fn one() -> Self {
        Integer::from(1).into()
    }
}

impl EuclideanDomain<Int> for Int {
    fn degree(&self) -> Self {
        self.clone()
    }

    fn div_with_rem(self, other: Self) -> (Self, Self) {
        let (q, r) = self.0.div_rem(other.0);
        (q.into(), r.into())
    }
}

pub struct Prime273_72;

impl<V> PrimeModulo<V> for Prime273_72
where
    V: From<u16> + ShlAssign<u32> + AddAssign<V>,
{
    fn divisor() -> V {
        let mut v = V::from(273);
        v <<= 72;
        v += V::from(1);
        v
    }
}

impl FiniteField<Int, usize> for F {
    fn characteristic() -> Int {
        Prime273_72::divisor()
    }
    fn degree_extension() -> usize {
        1
    }
    type Scalar = Self;
    fn to_vec(&self) -> Vec<Self::Scalar> {
        vec![self.clone()]
    }
    fn from_scalar(scalar: Self::Scalar) -> Self {
        scalar
    }
}

fn t<F: FiniteField<M, N>, M, N>() {}

lazy_static! {
    static ref P273_2_72: Integer = { Integer::from(273) << 72 + 1 };
    static ref T: () = { t::<F, Int, usize>() };
}

pub type F = Fp<Prime273_72, Int, Int>;

/// key size n = 2^12
pub const KEY_SIZE: usize = 4096;

lazy_static! {
    /// stddev alpha
    static ref SIGMA_A: Float = {
        Float::with_val(256, 3.397)
    };
    /// stddev beta
    static ref SIGMA_B: Float = {
        &*SIGMA_A * Float::with_val(256, Float::u_exp(1, 28))
    };
    /// stddev gamma
    static ref SIGMA_G: Float = {
        Float::with_val(256, 256.495)
    };
}

#[derive(Clone)]
pub struct Poly([F; KEY_SIZE]);

impl Debug for Poly {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "[{:?}", self[0])?;
        for a in &self[1..] {
            write!(f, ", {:?}", a)?;
        }
        write!(f, "]")
    }
}

impl Display for Poly {
    fn fmt(&self, f: &mut Formatter) -> FmtResult {
        write!(f, "[{}", self[0])?;
        for a in &self[1..] {
            write!(f, ", {}", a)?;
        }
        write!(f, "]")
    }
}

impl PartialEq for Poly {
    fn eq(&self, other: &Self) -> bool {
        self.iter().zip(other.iter()).all(|(a, b)| a == b)
    }
}

impl Deref for Poly {
    type Target = [F; KEY_SIZE];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Poly {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl Serialize for Poly {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(KEY_SIZE))?;
        for e in self.0.iter() {
            seq.serialize_element(e)?;
        }
        seq.end()
    }
}

struct PolyVisitor;
impl<'a> Visitor<'a> for PolyVisitor {
    type Value = Poly;
    fn expecting(&self, fmt: &mut Formatter) -> FmtResult {
        write!(fmt, "polynomial in requested field")
    }

    fn visit_seq<V: SeqAccess<'a>>(self, mut seq: V) -> Result<Self::Value, V::Error> {
        let mut v = vec![];
        for _ in 0..KEY_SIZE {
            v.push(
                seq.next_element()?
                    .ok_or(V::Error::custom("incorrect length"))?,
            )
        }
        Ok(vec_to_poly(v))
    }
}

impl<'a> Deserialize<'a> for Poly {
    fn deserialize<D: Deserializer<'a>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_seq(PolyVisitor)
    }
}

fn poly_to_bytes(poly: &Poly) -> Vec<u8> {
    poly[..]
        .iter()
        .flat_map(|x| {
            let Int(x) = x.inner();
            let mut data = vec![0; x.significant_digits::<u8>() as usize];
            x.write_digits(&mut data, rug::integer::Order::LsfLe);
            data
        })
        .collect()
}

fn generate_poly(mut f: impl FnMut(usize) -> F) -> Poly {
    let mut init: [MaybeUninit<F>; KEY_SIZE] = unsafe { MaybeUninit::uninit().assume_init() };
    for (i, x) in init[..].iter_mut().enumerate() {
        unsafe { x.as_mut_ptr().write(f(i)) }
    }
    unsafe { transmute(init) }
}

#[derive(Clone, Debug, Serialize, PartialEq, Deserialize)]
pub struct Init(Poly);

impl Init {
    pub fn new<R: CryptoRng + RngCore>(rng: &mut R) -> Self {
        Self(generate_poly(|_| F::uniform_sample(rng)))
    }
}

type FFTOP = dyn Send + Sync + Fn(Vec<F>) -> Vec<F>;

type ArcFFTOP = Pin<Arc<FFTOP>>;

lazy_static! {
    static ref UNITY_ROOT_8192: UnityRoot<F> = {
        UnityRoot {
            order: 8192, // 2 ^ 13
            root: F::new(Int(Integer::from(578560602093191843790318u128))),
        }
    };
}

fn construct_fft_2_12() -> ArcFFTOP {
    fn clone_fn(
        f: &Arc<impl Send + Sync + Fn(Vec<F>) -> Vec<F>>,
    ) -> impl Sync + Fn(Vec<F>) -> Vec<F> {
        let f = Arc::clone(f);
        move |x| f(x)
    }

    let fft_2_1 = Arc::new(naive(UNITY_ROOT_8192.clone().subgroup(2)));
    let fft_2_2 = Arc::new(cooley_tukey(
        2,
        2,
        UNITY_ROOT_8192.clone().subgroup(4),
        clone_fn(&fft_2_1),
        clone_fn(&fft_2_1),
    ));
    let fft_2_3 = Arc::new(cooley_tukey(
        2,
        4,
        UNITY_ROOT_8192.clone().subgroup(8),
        clone_fn(&fft_2_1),
        clone_fn(&fft_2_2),
    ));
    let fft_2_6 = Arc::new(cooley_tukey(
        8,
        8,
        UNITY_ROOT_8192.clone().subgroup(64),
        clone_fn(&fft_2_3),
        clone_fn(&fft_2_3),
    ));
    Arc::pin(cooley_tukey(
        64,
        64,
        UNITY_ROOT_8192.clone().subgroup(KEY_SIZE),
        clone_fn(&fft_2_6),
        clone_fn(&fft_2_6),
    ))
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PrivateKey(Poly);

fn vec_to_poly(a: Vec<F>) -> Poly {
    assert_eq!(a.len(), KEY_SIZE);
    let mut a = a.into_iter();
    generate_poly(|_| a.next().expect("length mismatch"))
}

lazy_static! {
    static ref F_DISTR_A: BaseSampler<Integer> =
        make_gaussian_sampler(Float::with_val(PRECISION, 0), SIGMA_A.clone());
    static ref F_DISTR_G: BaseSampler<Integer> =
        make_gaussian_sampler(Float::with_val(PRECISION, 0), SIGMA_G.clone());
    static ref FFT_2_12: ArcFFTOP = construct_fft_2_12();
    static ref PHI: Array1<F> = {
        let mut phi = vec![];
        let g = UNITY_ROOT_8192.root.clone();
        let mut a = F::one();
        for _ in 0..KEY_SIZE {
            phi.push(a.clone());
            a *= g.clone();
        }
        Array1::from(phi)
    };
    static ref PHI_INV: Array1<F> = {
        use alga::general::{Multiplicative, TwoSidedInverse};
        let mut phi = vec![];
        let g = <F as TwoSidedInverse<Multiplicative>>::two_sided_inverse(&UNITY_ROOT_8192.root);
        let mut a = F::one();
        for _ in 0..KEY_SIZE {
            phi.push(a.clone());
            a *= g.clone();
        }
        Array1::from(phi)
    };
    static ref F_CONST_2: F = int_inj(2);
    static ref F_CENTRE: Integer = {
        let mut v: Integer = <Prime273_72 as PrimeModulo<Integer>>::divisor() - 1;
        v.div_exact_mut(&Integer::from(2));
        v
    };
    static ref F_E_LOWER: Integer = {
        let (v, _) =
            <Prime273_72 as PrimeModulo<Integer>>::divisor().div_rem_floor(Integer::from(4));
        <Prime273_72 as PrimeModulo<Integer>>::divisor() - v
    };
    static ref F_E_UPPER: Integer = {
        let (v, _) =
            <Prime273_72 as PrimeModulo<Integer>>::divisor().div_rem_round(Integer::from(4));
        v
    };
    static ref F_KEY_SIZE: F = int_inj(KEY_SIZE);
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PublicKey(Poly);

fn poly_mul_mod(a: Poly, b: Poly) -> Poly {
    let a = FFT_2_12((Array1::from(a.to_vec()) * PHI.clone()).to_vec());
    let b = FFT_2_12((Array1::from(b.to_vec()) * PHI.clone()).to_vec());
    let c = Array1::from(a) * Array1::from(b);
    let mut c = FFT_2_12(c.to_vec());
    c[1..].reverse();
    let c = Array1::from(c) * PHI_INV.clone();
    vec_to_poly(
        c.to_vec()
            .into_iter()
            .map(|x| x / F_KEY_SIZE.clone())
            .collect(),
    )
}

fn poly_add(mut a: Poly, b: Poly) -> Poly {
    for (a, b) in a.iter_mut().zip(b.to_vec().into_iter()) {
        *a += b
    }
    a
}

pub fn keygen<R: CryptoRng + RngCore>(rng: &mut R, Init(init): &Init) -> (PrivateKey, PublicKey) {
    let s = generate_poly(|_| int_inj(Int(F_DISTR_A.sample(rng).clone())));
    let e = generate_poly(|_| int_inj(Int(F_DISTR_A.sample(rng).clone() * 2)));

    let public = poly_add(poly_mul_mod(init.clone(), s.clone()), e);
    (PrivateKey(s), PublicKey(public))
}

#[derive(Clone, Serialize, Deserialize)]
pub struct SessionKeyPart(Poly);
#[derive(Clone, Serialize, Deserialize)]
pub struct SessionKeyPartR(Poly);

pub struct SessionKeyPartParallelSampler<R>(ParallelGenericSampler<R>);

impl SessionKeyPart {
    pub fn parallel_sampler<R>() -> SessionKeyPartParallelSampler<R>
    where
        R: CryptoRng + RngCore + SeedableRng + Send + 'static,
    {
        SessionKeyPartParallelSampler(ParallelGenericSampler::new(
            repeat_with(|| R::from_entropy()).take(32).collect(),
            1024,
            Float::with_val(PRECISION, 0),
            SIGMA_B.clone(),
            5,
        ))
    }
    pub fn generate<R>(
        par_sampler: &SessionKeyPartParallelSampler<R>,
        Init(init): &Init,
    ) -> (Self, SessionKeyPartR) {
        let r = vec_to_poly(
            (0..KEY_SIZE)
                .map(|_| {
                    let v = par_sampler.0.sample();
                    int_inj(Int(v))
                })
                .collect(),
        );
        let f = vec_to_poly(
            (0..KEY_SIZE)
                .map(|_| {
                    let v = par_sampler.0.sample();
                    int_inj(Int(v))
                })
                .collect(),
        );
        (
            Self(poly_add(poly_mul_mod(init.clone(), r.clone()), {
                let mut f = f;
                for x in &mut f[..] {
                    *x *= F_CONST_2.clone();
                }
                f
            })),
            SessionKeyPartR(r),
        )
    }

    pub fn into_bytes(&self) -> Vec<u8> {
        poly_to_bytes(&self.0)
    }
}

pub fn hash_ident_session_key_part<R, Id, F>(id: Id, hasher: F) -> Poly
where
    Id: Into<Vec<u8>>,
    R: SeedableRng + CryptoRng + RngCore,
    F: Fn(&[u8]) -> R::Seed,
{
    let mut v = b"key exchange information between Anke and Boris: ".to_vec();
    v.extend(id.into());
    v.extend(b" and that is all we have :D");
    let mut rng = R::from_seed(hasher(&v));
    vec_to_poly(
        (0..KEY_SIZE)
            .map(|_| int_inj(Int(F_DISTR_G.sample(&mut rng).clone())))
            .collect(),
    )
}

pub fn cha(v: &F) -> bool {
    let v = &v.inner().0;
    *v <= *F_E_UPPER || *v >= *F_E_LOWER
}

pub fn mod_2(v: &F, w: bool) -> bool {
    let v = if w {
        v.clone() + F::new(Int(F_CENTRE.clone()))
    } else {
        v.clone()
    };
    v.into_inner().0.is_odd()
}

pub struct Anke;
pub struct Boris;

#[derive(Clone)]
pub struct AnkePublic<'a, 'b>(pub &'a [u8], pub &'b PublicKey);
#[derive(Clone)]
pub struct BorisPublic<'a, 'b>(pub &'a [u8], pub &'b PublicKey);
#[derive(Clone)]
pub struct AnkeIdentity<'a>(pub &'a PrivateKey);
#[derive(Clone)]
pub struct BorisIdentity<'a>(pub &'a PrivateKey);
#[derive(Clone)]
pub struct AnkeSessionKeyPart<'a>(pub &'a SessionKeyPart);
#[derive(Clone)]
pub struct BorisSessionKeyPart<'a>(pub &'a SessionKeyPart);
#[derive(Clone)]
pub struct AnkeSessionKeyPartR<'a>(pub &'a SessionKeyPartR);
#[derive(Clone)]
pub struct BorisSessionKeyPartR<'a>(pub &'a SessionKeyPartR);

pub struct SessionKeyPartMix<P>(Poly, PhantomData<fn() -> P>);
pub struct SessionKeyPartMixParallelSampler<R, P>(
    ParallelGenericSampler<R>,
    PhantomData<fn() -> P>,
);

struct KeyExchange<'a, 'b, 'c, 'd> {
    anke_data: &'a [u8],
    boris_data: &'b [u8],
    anke_key_part: &'c SessionKeyPart,
    boris_key_part: Option<&'d SessionKeyPart>,
}

impl From<KeyExchange<'_, '_, '_, '_>> for Vec<u8> {
    fn from(kex: KeyExchange) -> Self {
        let mut r = vec![];
        r.extend_from_slice(b"<anke data>");
        r.extend(kex.anke_data);
        r.extend_from_slice(b"<anke key part>");
        r.extend(kex.anke_key_part.into_bytes());
        r.extend_from_slice(b"<boris data>");
        r.extend(kex.boris_data);
        if let Some(boris_key_part) = kex.boris_key_part {
            r.extend_from_slice(b"<boris key part>");
            r.extend(boris_key_part.into_bytes())
        }
        r
    }
}

impl SessionKeyPartMix<Anke> {
    pub fn generate<
        H: CryptoRng + RngCore + SeedableRng,
        R: CryptoRng + RngCore,
        S: Fn(&[u8]) -> H::Seed,
    >(
        seeder: S,
        sampler: &SessionKeyPartMixParallelSampler<R, Anke>,
        AnkePublic(anke_data, _): AnkePublic,
        BorisPublic(boris_data, PublicKey(boris_public)): BorisPublic,
        AnkeSessionKeyPart(anke_key_part): AnkeSessionKeyPart,
        BorisSessionKeyPart(boris_key_part): BorisSessionKeyPart,
        AnkeIdentity(PrivateKey(anke_private_key)): AnkeIdentity,
        AnkeSessionKeyPartR(SessionKeyPartR(anke_random)): AnkeSessionKeyPartR,
    ) -> (Self, Poly, Poly) {
        let c = hash_ident_session_key_part::<H, _, _>(
            KeyExchange {
                anke_data: anke_data,
                boris_data: boris_data,
                anke_key_part: anke_key_part,
                boris_key_part: None,
            },
            &seeder,
        );
        let d = hash_ident_session_key_part::<H, _, _>(
            KeyExchange {
                anke_data: anke_data,
                boris_data: boris_data,
                anke_key_part: anke_key_part,
                boris_key_part: Some(boris_key_part),
            },
            &seeder,
        );
        let g = generate_poly(|_| {
            let v = sampler.0.sample() * 2;
            int_inj(Int(v))
        });
        let part_mix = Self(
            poly_add(
                g,
                poly_mul_mod(
                    poly_add(
                        poly_mul_mod(boris_public.clone(), d.clone()),
                        boris_key_part.0.clone(),
                    ),
                    poly_add(
                        poly_mul_mod(anke_private_key.clone(), c.clone()),
                        anke_random.clone(),
                    ),
                ),
            ),
            PhantomData,
        );
        (part_mix, c, d)
    }
}

impl SessionKeyPartMix<Boris> {
    pub fn generate<
        H: CryptoRng + RngCore + SeedableRng,
        R: CryptoRng + RngCore,
        S: Fn(&[u8]) -> H::Seed,
    >(
        seeder: S,
        sampler: &SessionKeyPartMixParallelSampler<R, Boris>,
        AnkePublic(anke_data, PublicKey(anke_public)): AnkePublic,
        BorisPublic(boris_data, _): BorisPublic,
        AnkeSessionKeyPart(anke_key_part): AnkeSessionKeyPart,
        BorisSessionKeyPart(boris_key_part): BorisSessionKeyPart,
        BorisIdentity(PrivateKey(boris_private_key)): BorisIdentity,
        BorisSessionKeyPartR(SessionKeyPartR(boris_random)): BorisSessionKeyPartR,
    ) -> (Self, Poly, Poly) {
        let c = hash_ident_session_key_part::<H, _, _>(
            KeyExchange {
                anke_data: anke_data,
                boris_data: boris_data,
                anke_key_part: anke_key_part,
                boris_key_part: None,
            },
            &seeder,
        );
        let d = hash_ident_session_key_part::<H, _, _>(
            KeyExchange {
                anke_data: anke_data,
                boris_data: boris_data,
                anke_key_part: anke_key_part,
                boris_key_part: Some(boris_key_part),
            },
            &seeder,
        );
        let g = generate_poly(|_| {
            let v = sampler.0.sample() * 2;
            int_inj(Int(v))
        });
        let part_mix = Self(
            poly_add(
                g,
                poly_mul_mod(
                    poly_add(
                        poly_mul_mod(anke_public.clone(), c.clone()),
                        anke_key_part.0.clone(),
                    ),
                    poly_add(
                        poly_mul_mod(boris_private_key.clone(), d.clone()),
                        boris_random.clone(),
                    ),
                ),
            ),
            PhantomData,
        );
        (part_mix, c, d)
    }
}

pub struct Reconciliator([bool; KEY_SIZE]);

impl Serialize for Reconciliator {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use bitvec::prelude::*;
        let mut v = BitVec::<LittleEndian, u64>::new();
        v.extend(self.0.iter().copied());
        v.serialize(s)
    }
}

impl<'a> Deserialize<'a> for Reconciliator {
    fn deserialize<D: Deserializer<'a>>(d: D) -> Result<Self, D::Error> {
        use bitvec::prelude::*;
        let v = BitVec::<LittleEndian, u64>::deserialize(d)?;
        if v.len() == KEY_SIZE {
            let mut r: [MaybeUninit<bool>; KEY_SIZE] =
                unsafe { MaybeUninit::uninit().assume_init() };
            for (r, v) in r.iter_mut().zip(&v) {
                unsafe { r.as_mut_ptr().write(*v) }
            }
            Ok(Self(unsafe { transmute(r) }))
        } else {
            Err(D::Error::custom("length mismatch"))
        }
    }
}

#[derive(Clone)]
pub struct SharedKey([bool; KEY_SIZE]);

impl std::cmp::PartialEq for SharedKey {
    fn eq(&self, other: &Self) -> bool {
        let mut r = true;
        for (x, y) in self.0.iter().zip(other.0.iter()) {
            r &= x == y;
        }
        r
    }
}

impl std::hash::Hash for SharedKey {
    fn hash<H: std::hash::Hasher>(&self, hasher: &mut H) {
        for x in self.0.iter() {
            x.hash(hasher)
        }
    }
}

impl Debug for SharedKey {
    fn fmt(&self, fmt: &mut Formatter) -> FmtResult {
        writeln!(fmt, "[")?;
        for c in self.0.chunks(256) {
            for &x in c {
                if x {
                    write!(fmt, "1")?;
                } else {
                    write!(fmt, "0")?;
                }
            }
            writeln!(fmt, "")?;
        }
        write!(fmt, "]")
    }
}

impl<P> SessionKeyPartMix<P> {
    pub fn parallel_sampler<R>() -> SessionKeyPartMixParallelSampler<R, P>
    where
        R: CryptoRng + RngCore + SeedableRng + Send + 'static,
    {
        SessionKeyPartMixParallelSampler(
            ParallelGenericSampler::new(
                repeat_with(|| R::from_entropy()).take(32).collect(),
                1024,
                Float::with_val(PRECISION, 0),
                SIGMA_B.clone(),
                5,
            ),
            PhantomData,
        )
    }

    pub fn reconciliator(&self) -> Reconciliator {
        let mut r: [MaybeUninit<bool>; KEY_SIZE] = unsafe { MaybeUninit::uninit().assume_init() };
        for (w, r) in self.0.iter().map(|k| cha(k)).zip(r[..].iter_mut()) {
            unsafe { r.as_mut_ptr().write(w) }
        }
        unsafe { transmute(r) }
    }

    pub fn reconciliate(&self, Reconciliator(r): &Reconciliator) -> SharedKey {
        let mut k: [MaybeUninit<bool>; KEY_SIZE] = unsafe { MaybeUninit::uninit().assume_init() };
        for (bit, k) in r
            .iter()
            .zip(self.0.iter())
            .map(|(&r, k)| mod_2(k, r))
            .zip(&mut k[..])
        {
            unsafe { k.as_mut_ptr().write(bit) }
        }
        SharedKey(unsafe { transmute(k) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use quickcheck::{Arbitrary, Gen};
    use rand::rngs::StdRng;
    use sha2::{Digest, Sha256};

    use crate::Polynomial;

    #[test]
    fn key_ex_works() {
        let session_key_sampler = SessionKeyPart::parallel_sampler::<StdRng>();
        let session_key_mix_sampler_anke = SessionKeyPartMix::<Anke>::parallel_sampler::<StdRng>();
        let session_key_mix_sampler_boris =
            SessionKeyPartMix::<Boris>::parallel_sampler::<StdRng>();
        let mut rng = StdRng::from_entropy();
        let init = Init::new(&mut rng);
        let (anke_pri, anke_pub) = keygen(&mut rng, &init);
        let (boris_pri, boris_pub) = keygen(&mut rng, &init);
        eprintln!("keygen done");
        let (anke_key_part, anke_random) = SessionKeyPart::generate(&session_key_sampler, &init);
        let (boris_key_part, boris_random) = SessionKeyPart::generate(&session_key_sampler, &init);
        let seeder = |v: &[u8]| {
            let digest = Sha256::digest(v);
            let mut seed = [0u8; 32];
            seed[..].copy_from_slice(&digest);
            seed
        };
        eprintln!("generate anke key part");
        let (anke_part_mix, anke_c, anke_d) =
            SessionKeyPartMix::<Anke>::generate::<rand_chacha::ChaChaRng, _, _>(
                seeder,
                &session_key_mix_sampler_anke,
                AnkePublic(&[], &anke_pub),
                BorisPublic(&[], &boris_pub),
                AnkeSessionKeyPart(&anke_key_part),
                BorisSessionKeyPart(&boris_key_part),
                AnkeIdentity(&anke_pri),
                AnkeSessionKeyPartR(&anke_random),
            );
        eprintln!("generate boris key part");
        let (boris_part_mix, boris_c, boris_d) =
            SessionKeyPartMix::<Boris>::generate::<rand_chacha::ChaChaRng, _, _>(
                seeder,
                &session_key_mix_sampler_boris,
                AnkePublic(&[], &anke_pub),
                BorisPublic(&[], &boris_pub),
                AnkeSessionKeyPart(&anke_key_part),
                BorisSessionKeyPart(&boris_key_part),
                BorisIdentity(&boris_pri),
                BorisSessionKeyPartR(&boris_random),
            );

        assert_eq!(P(anke_c.clone()), P(boris_c.clone()));
        assert_eq!(P(anke_d.clone()), P(boris_d.clone()));
        let Init(a) = init;
        let shared = poly_mul_mod(
            poly_mul_mod(a, poly_add(poly_mul_mod(anke_c, anke_pri.0), anke_random.0)),
            poly_add(poly_mul_mod(anke_d, boris_pri.0), boris_random.0),
        );
        eprintln!("shared_part={:?}", P(shared));
        let error = vec_to_poly(
            anke_part_mix
                .0
                .to_vec()
                .into_iter()
                .zip(boris_part_mix.0.to_vec())
                .map(|(a, b)| (a - b) / int_inj(2))
                .collect(),
        );
        let q: Int = Prime273_72::divisor();
        let q_8 = q.clone() / 8;
        let q_7_8 = q_8.clone() * 7;
        let mut err_cnt = 0;
        let mut locations = vec![];
        for (location, error) in error.iter().enumerate() {
            let error = error.inner();
            if *error > q_8 && *error < q_7_8 {
                err_cnt += 1;
                locations.push(location);
            }
        }
        assert_eq!(err_cnt, 0);

        let reconsiliator = boris_part_mix.reconciliator();
        let anke_key = boris_part_mix.reconciliate(&reconsiliator);
        let boris_key = anke_part_mix.reconciliate(&reconsiliator);
        assert_eq!(anke_key, boris_key);
        eprintln!("shared key: {:?}", anke_key);
    }

    #[derive(Clone)]
    struct P(Poly);

    impl Arbitrary for Int {
        fn arbitrary<G: Gen>(g: &mut G) -> Self {
            Int(Integer::from(g.next_u64()))
        }
    }

    impl Arbitrary for P {
        fn arbitrary<G: Gen>(g: &mut G) -> Self {
            P(vec_to_poly(
                (0..KEY_SIZE)
                    .map(|_| <F as Arbitrary>::arbitrary(g))
                    .collect(),
            ))
        }
    }

    impl Debug for P {
        fn fmt(&self, fmt: &mut Formatter) -> FmtResult {
            write!(fmt, "[")?;
            for x in self.0.iter() {
                write!(fmt, "{:?} ", x)?
            }
            write!(fmt, "]")
        }
    }

    impl std::cmp::PartialEq for P {
        fn eq(&self, other: &Self) -> bool {
            self.0.iter().zip(other.0.iter()).all(|(a, b)| a == b)
        }
    }

    #[quickcheck]
    #[ignore]
    fn poly_mul_mod_test(P(a): P, P(b): P) {
        let c_actual = P(poly_mul_mod(a.clone(), b.clone()));

        let a = Polynomial::new(a.to_vec());
        let b = Polynomial::new(b.to_vec());
        let c = a * b;
        let m = Polynomial::new(
            Some(F::one())
                .into_iter()
                .chain(repeat_with(F::zero).take(KEY_SIZE - 1))
                .chain(Some(F::one())),
        );
        let (_, Polynomial(c_expected)) = c.div_with_rem(m);
        let c_expected = P(vec_to_poly(c_expected));

        assert_eq!(c_actual, c_expected);
    }

    #[test]
    fn serde() {
        let mut rng = StdRng::from_entropy();
        let init = Init::new(&mut rng);
        let se = serde_json::to_string(&init).unwrap();
        let init_: Init = serde_json::from_str(&se).unwrap();
        assert_eq!(init, init_);
    }
}
