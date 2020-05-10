use std::marker::PhantomData;

use digest::Digest;
use num::{One, Zero};
use rand::{seq::SliceRandom, CryptoRng, RngCore};
use serde::{Deserialize, Serialize};

use crate::{
    field::F2,
    galois::{GF65536NTower, GF65536N},
    goppa::{BinaryGoppaDecoder, GoppaDecoder, GoppaEncoder},
    poly::MultipointEvaluator,
    ser::{bitvec_to_u8, u8_to_bitvec},
};

#[derive(Serialize, Deserialize)]
pub struct McElieceKEM65536PublicKey {
    key: GoppaEncoder<F2, GF65536NTower>,
}

pub struct McElieceKEM65536PrivateKey<M> {
    decoder: BinaryGoppaDecoder<GF65536N, GF65536NTower, M>,
}

#[derive(Deref)]
pub struct McElieceSessionKey(pub Vec<u8>);

#[derive(Serialize, Deserialize)]
pub struct McElieceCiphertext<H> {
    pub c_0: Vec<u8>,
    pub c_1: Vec<u8>,
    #[serde(skip)]
    _p: PhantomData<H>,
}

impl<H> McElieceCiphertext<H> {
    pub fn new(c_0: Vec<u8>, c_1: Vec<u8>) -> Self {
        Self {
            c_0,
            c_1,
            _p: PhantomData,
        }
    }
}

impl McElieceKEM65536PublicKey {
    pub fn new(key: GoppaEncoder<F2, GF65536NTower>) -> Self {
        Self { key }
    }
    pub fn encapsulate<H, R>(&self, rng: &mut R) -> (McElieceSessionKey, McElieceCiphertext<H>)
    where
        H: Digest,
        R: RngCore + CryptoRng,
    {
        let t = self.key.weight();
        let n = self.key.size();
        let mut idx: Vec<_> = (0..n).collect();
        idx.shuffle(rng);
        let mut secret = vec![F2::zero(); n];
        for &idx in &idx[0..t] {
            secret[idx] = F2::one();
        }
        let c_0 = self.key.encode(&secret);
        let c_0 = bitvec_to_u8(&c_0);
        let secret_bits = bitvec_to_u8(&secret);
        let c_1 = {
            let mut h = H::new();
            h.input([2, 0]);
            h.input(&secret_bits);
            h.result().to_vec()
        };
        let k = {
            let mut h = H::new();
            h.input([1, 0]);
            h.input(&secret_bits);
            h.input(&c_0);
            h.input([0]);
            h.input(&c_1);
            h.result()
        };
        (
            McElieceSessionKey(k.to_vec()),
            McElieceCiphertext {
                c_0,
                c_1,
                _p: PhantomData,
            },
        )
    }
}

impl<M> McElieceKEM65536PrivateKey<M>
where
    M: MultipointEvaluator<GF65536N>,
{
    pub fn new(dec: GoppaDecoder<GF65536N, GF65536NTower, M>) -> Option<Self> {
        let decoder = BinaryGoppaDecoder::from_decoder(dec);
        Some(Self { decoder })
    }

    pub fn decapsulate<H>(&self, ctxt: McElieceCiphertext<H>) -> McElieceSessionKey
    where
        H: Digest,
    {
        let McElieceCiphertext { c_0: c_0_, c_1, .. } = ctxt;
        let c_0 = u8_to_bitvec(&c_0_);
        let mut sense = vec![F2::zero(); self.decoder.code_len()];
        for (sense, c_0) in sense.iter_mut().zip(&c_0) {
            *sense = *c_0;
        }
        let mut b = 1;
        let (_, mut error) = self.decoder.decode(sense);
        if error.iter().filter(|F2(x)| *x > 0).count() != self.decoder.weight() {
            error = vec![F2::zero(); self.decoder.size()];
            b = 0;
        }
        let secret = error;
        let mut secret_bits = bitvec_to_u8(&secret);

        {
            let mut h = H::new();
            h.input([2, 0]);
            h.input(&secret_bits);
            if h.result().to_vec() != c_1 {
                secret_bits = vec![0];
                b = 0;
            }
        };
        {
            let mut h = H::new();
            h.input([b, 0]);
            h.input(&secret_bits);
            h.input(&c_0_);
            h.input([0]);
            h.input(&c_1);
            McElieceSessionKey(h.result().to_vec())
        }
    }
}
