//! Serialization formatting and routines

use bitvec::{order::Lsb0, slice::AsBits, vec::BitVec};
use num::{One, Zero};

use crate::{
    adapter::Int,
    field::{FiniteField, F2},
    galois::ExtensionTower,
};

/// Reinterpret a vector of `F2` into a vector of `u32`.
/// The bit order is least significant bit, where the most significant
/// zero bits are truncated
pub fn bitvec_to_u8(v: &[F2]) -> Vec<u8> {
    let mut bv: BitVec<Lsb0, u8> = BitVec::new();
    bv.resize(v.len(), false);
    for (i, F2(x)) in v.into_iter().enumerate() {
        if *x > 0 {
            bv.set(i, true);
        }
    }
    bv.shrink_to_fit();
    bv.into_vec()
}

pub fn u8_to_bitvec(v: &[u8]) -> Vec<F2> {
    let slice = v.bits::<Lsb0>();
    slice
        .iter()
        .map(|&b| if b { F2::one() } else { F2::zero() })
        .collect()
}

pub fn binary_field_to_bitvec_u32<F, T>(x: F) -> Vec<u32>
where
    F: FiniteField + Clone,
    T: ExtensionTower<Super = F, Bottom = F2>,
{
    let m = T::degree_extension::<Int>().assert_usize();
    let mut bv: BitVec<Lsb0, u32> = BitVec::new();
    bv.resize(m, false);
    for (i, F2(x)) in T::to_vec(x).into_iter().enumerate() {
        if x > 0 {
            bv.set(i, true);
        }
    }
    bv.shrink_to_fit();
    bv.into_vec()
}

pub fn bitvec_u32_to_binary_field<F, T>(x: Vec<u32>) -> F
where
    F: FiniteField + Clone,
    T: ExtensionTower<Super = F, Bottom = F2>,
{
    let bases = T::basis_elements_over_bottom();
    let bv: BitVec<Lsb0, u32> = BitVec::from_vec(x);
    let mut x = F::zero();
    for (&a, b) in bv.iter().zip(bases.into_iter()) {
        if a {
            x += b;
        }
    }
    x
}
