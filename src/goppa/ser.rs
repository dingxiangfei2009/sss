use super::*;

use bitvec::{order::Lsb0, vec::BitVec};
use serde::{de::Error as _, Deserialize, Deserializer, Serialize, Serializer};

pub struct BinaryPacked<T>(pub T);

#[derive(Serialize, Deserialize)]
struct BinaryPackedGoppaEncoder {
    len: usize,
    data: Vec<u32>,
    t: usize,
}

impl<T> Serialize for BinaryPacked<GoppaEncoder<F2, T>> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut bv: BitVec<Lsb0, u32> = BitVec::new();
        let len = self.0.parity_check.len();
        bv.resize(len, false);
        for (i, F2(x)) in self.0.parity_check.iter().enumerate() {
            if *x > 0 {
                bv.set(i, true);
            }
        }
        bv.shrink_to_fit();
        let s = BinaryPackedGoppaEncoder {
            t: self.0.t,
            len,
            data: bv.into_vec(),
        };
        s.serialize(serializer)
    }
}

impl<'a, T> Deserialize<'a> for BinaryPacked<GoppaEncoder<F2, T>>
where
    T: ExtensionTower<Bottom = F2>,
{
    fn deserialize<D: Deserializer<'a>>(deserializer: D) -> Result<Self, D::Error> {
        let BinaryPackedGoppaEncoder { t, data, len } =
            BinaryPackedGoppaEncoder::deserialize(deserializer)?;
        let m = T::degree_extension::<Int>().assert_usize();
        let s = len / m / t;
        let parity_check: BitVec<Lsb0, _> = BitVec::from_vec(data);
        let parity_check: Vec<_> = parity_check
            .iter()
            .map(|&x| if x { F2(1) } else { F2(0) })
            .collect();
        let parity_check = Array2::from_shape_vec((m * t, s), parity_check).map_err(|_| {
            let s = (s * m * t).to_string();
            let s = &s as &str;
            D::Error::invalid_length(len, &s)
        })?;
        Ok(Self(GoppaEncoder {
            parity_check,
            t,
            _p: PhantomData,
        }))
    }
}

#[derive(Serialize, Deserialize)]
struct BinaryPackedGoppaDecoder {
    points: Vec<Vec<u32>>,
    g: Vec<Vec<u32>>,
    invert_factors: Option<Vec<Vec<Vec<u32>>>>,
}

fn binary_field_to_bitvec_u32<F, T>(m: usize, x: F) -> Vec<u32>
where
    F: FiniteField + Clone,
    T: ExtensionTower<Super = F, Bottom = F2>,
{
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

fn bitvec_u32_to_binary_field<F, T>(x: Vec<u32>) -> F
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

impl<F, T, M> Serialize for BinaryPacked<GoppaDecoder<F, T, M>>
where
    F: FiniteField + Clone,
    T: ExtensionTower<Super = F, Bottom = F2>,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let m = T::degree_extension::<Int>().assert_usize();
        let points = self
            .0
            .points
            .iter()
            .cloned()
            .map(|p| binary_field_to_bitvec_u32::<F, T>(m, p))
            .collect();
        let g = self
            .0
            .g_poly
            .0
            .iter()
            .cloned()
            .map(|g| binary_field_to_bitvec_u32::<F, T>(m, g))
            .collect();
        let invert_factors = Some(
            self.0
                .invert_factors
                .iter()
                .cloned()
                .map(|Polynomial(p)| {
                    p.into_iter()
                        .map(|p| binary_field_to_bitvec_u32::<F, T>(m, p))
                        .collect()
                })
                .collect(),
        );
        let s = BinaryPackedGoppaDecoder {
            points,
            g,
            invert_factors,
        };
        s.serialize(serializer)
    }
}

impl<'a, F, T, M> Deserialize<'a> for BinaryPacked<GoppaDecoder<F, T, M>>
where
    F: Clone + Deserialize<'a> + Send + Sync + FiniteField,
    T: ExtensionTower<Super = F, Bottom = F2>,
    M: MultipointEvaluator<F>,
{
    fn deserialize<D: Deserializer<'a>>(deserializer: D) -> Result<Self, D::Error> {
        let BinaryPackedGoppaDecoder {
            points,
            g,
            invert_factors,
        } = BinaryPackedGoppaDecoder::deserialize(deserializer)?;
        let points: Vec<_> = points
            .into_iter()
            .map(bitvec_u32_to_binary_field::<_, T>)
            .collect();
        let g_poly = Polynomial::new(g.into_iter().map(bitvec_u32_to_binary_field::<_, T>));
        let invert_factors = match invert_factors {
            Some(fs) => fs
                .into_iter()
                .map(|fs| Polynomial::new(fs.into_iter().map(bitvec_u32_to_binary_field::<_, T>)))
                .collect(),
            None => points
                .par_iter()
                .map(|alpha: &F| invert_factor_poly(&g_poly, alpha.clone()))
                .collect(),
        };
        Ok(Self(GoppaDecoder {
            multipoint_eval: M::prepare(points.clone()),
            points,
            g_poly,
            invert_factors,
            _p: PhantomData,
        }))
    }
}

#[derive(Serialize, Deserialize)]
struct GoppaEncoderRepr<F> {
    t: usize,
    parity_check: Vec<F>,
}

#[derive(Serialize, Deserialize)]
struct GoppaDecoderRepr<F> {
    points: Vec<F>,
    g: Polynomial<F>,
    invert_factors: Option<Vec<Polynomial<F>>>,
}

impl<F, T> Serialize for GoppaEncoder<F, T>
where
    F: Clone + Serialize,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        // follow the logical order
        let parity_check = self.parity_check.iter().cloned().collect();
        let repr = GoppaEncoderRepr {
            t: self.t,
            parity_check,
        };
        repr.serialize(serializer)
    }
}

impl<F, T, M> Serialize for GoppaDecoder<F, T, M>
where
    F: Clone + Serialize,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let repr = GoppaDecoderRepr {
            points: self.points.clone(),
            g: self.g_poly.clone(),
            invert_factors: Some(self.invert_factors.clone()),
        };
        repr.serialize(serializer)
    }
}

impl<'a, F, T> Deserialize<'a> for GoppaEncoder<F, T>
where
    F: FiniteField + Clone + Deserialize<'a>,
    T: ExtensionTower<Bottom = F>,
{
    fn deserialize<D: Deserializer<'a>>(deserializer: D) -> Result<Self, D::Error> {
        let GoppaEncoderRepr { t, parity_check } =
            GoppaEncoderRepr::<F>::deserialize(deserializer)?;
        let m = T::degree_extension::<Int>().assert_usize();
        let parity_check_len = parity_check.len();
        let s = parity_check_len / m / t;
        let parity_check = Array2::from_shape_vec((m * t, s), parity_check).map_err(|_| {
            let s = (s * m * t).to_string();
            let s = &s as &str;
            D::Error::invalid_length(parity_check_len, &s)
        })?;
        Ok(Self {
            parity_check,
            t,
            _p: PhantomData,
        })
    }
}

impl<'a, F, T, M> Deserialize<'a> for GoppaDecoder<F, T, M>
where
    F: Clone + Deserialize<'a> + Send + Sync + Field,
    M: MultipointEvaluator<F>,
{
    fn deserialize<D: Deserializer<'a>>(deserializer: D) -> Result<Self, D::Error> {
        let GoppaDecoderRepr {
            points,
            g,
            invert_factors,
        } = GoppaDecoderRepr::deserialize(deserializer)?;
        let invert_factors = match invert_factors {
            Some(fs) => fs,
            None => points
                .par_iter()
                .map(|alpha: &F| invert_factor_poly(&g, alpha.clone()))
                .collect(),
        };
        Ok(Self {
            multipoint_eval: M::prepare(points.clone()),
            points,
            g_poly: g,
            invert_factors,
            _p: PhantomData,
        })
    }
}
