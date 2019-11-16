use std::{
    collections::BTreeSet,
    iter::{once, repeat},
    ops::Deref,
    pin::Pin,
    sync::Arc,
};

use alga::general::Field;
use failure::Fail;
use num::{One, Zero};

use crate::{field::FiniteField, fourier::UnityRoot, pow, Coord, EuclideanDomain, Polynomial};

pub type FFTOp<F> = Pin<Arc<dyn Send + Sync + Fn(Vec<F>) -> Vec<F>>>;

pub struct ReedSolomon<F> {
    root: UnityRoot<F>,
    data_len: usize,
    generator: Polynomial<F>,
    fft: FFTOp<F>,
    correction_level: usize,
}

#[derive(Fail, Debug)]
pub enum DecodeError {
    #[fail(display = "too many errors found, expect at most {}", _0)]
    TooManyError(usize),
    #[fail(display = "code has wrong size, expect {}, found {}", expect, found)]
    CodeLength { expect: usize, found: usize },
    #[fail(display = "data has wrong size, expect {}, found {}", expect, found)]
    DataLength { expect: usize, found: usize },
    #[fail(
        display = "unexpected erasure position {}, expect at most {}",
        found, expect
    )]
    Erasure { expect: usize, found: usize },
}

#[derive(Clone, Debug)]
pub struct DecodeResult<F> {
    output: Vec<F>,
    error_positions: Vec<usize>,
}

impl<F> DecodeResult<F> {
    pub fn output(&self) -> &[F] {
        &self.output
    }

    pub fn error_positions(&self) -> &[usize] {
        &self.error_positions
    }
}

impl<F> Deref for DecodeResult<F> {
    type Target = [F];
    fn deref(&self) -> &Self::Target {
        &self.output()
    }
}

#[allow(clippy::many_single_char_names)] // REASON: match symbol names with textbooks and papers
fn sugiyama<F: FiniteField + Clone>(
    syndrome: Polynomial<F>,
    t: usize,
    rho: usize,
) -> (Polynomial<F>, Polynomial<F>) {
    let mut a = Polynomial::new(repeat(F::zero()).take(t * 2).chain(once(F::one())));
    let mut b = syndrome;
    assert!(a.degree() >= b.degree(), "check your math");
    let d = t + rho / 2;
    let (mut t, mut v) = (Polynomial::zero(), Polynomial::one());
    while b.degree() >= d {
        let (q, r) = a.div_with_rem(b.clone());
        let j = t - v.clone() * q.clone();
        t = v;
        v = j;
        a = b;
        b = r;
    }
    (v, b)
}

/// Forney's formula
pub fn forney<F: Field + Clone>(
    err_locator: Polynomial<F>,
    err_eval: Polynomial<F>,
    offset: usize,
    root: &UnityRoot<F>,
) -> impl Fn(usize) -> F {
    let err_locator = err_locator.formal_derivative();
    let root = root.root.clone();
    move |x| {
        let x = pow(root.clone(), x);
        let x_inv = F::one() / x.clone();
        let Coord(_, a) = err_eval.eval_at(x_inv.clone());
        let Coord(_, b) = err_locator.eval_at(x_inv.clone());
        -x.clone() * pow(x_inv, offset) * a / b
    }
}

impl<F> ReedSolomon<F>
where
    F: FiniteField + Clone,
{
    pub fn new(correction_level: usize, root: UnityRoot<F>, fft: FFTOp<F>) -> Self {
        let generator_size = 2 * correction_level;
        assert!(root.order > generator_size);
        let data_len = root.order - generator_size;
        let (generator, _, _) = (0..generator_size).fold(
            (Polynomial::one(), root.root.clone(), root.root.clone()),
            |(generator, root, checker), _| {
                (
                    generator * Polynomial(vec![-checker.clone(), F::one()]),
                    root.clone(),
                    checker * root,
                )
            },
        );

        Self {
            data_len,
            root,
            generator,
            fft,
            correction_level,
        }
    }

    pub fn encode(&self, data: &[F]) -> Result<Vec<F>, DecodeError> {
        if data.len() != self.data_len {
            return Err(DecodeError::DataLength {
                expect: self.data_len,
                found: data.len(),
            });
        }
        let Polynomial(p) = Polynomial::new(data.iter().cloned()) * self.generator.clone();
        assert!(p.len() <= self.root.order);
        Ok(p.into_iter()
            .chain(repeat(F::zero()))
            .take(self.root.order)
            .collect())
    }

    pub fn decode(
        &self,
        mut code: Vec<F>,
        erasure: Vec<usize>,
    ) -> Result<DecodeResult<F>, DecodeError> {
        let code_len = code.len();
        if code_len != self.root.order {
            return Err(DecodeError::CodeLength {
                expect: self.root.order,
                found: code.len(),
            });
        }
        let mut erasure_poly = Polynomial::one();

        if erasure.len() > 2 * self.correction_level {
            return Err(DecodeError::TooManyError(self.correction_level));
        }

        for &erasure in &erasure {
            if erasure >= code.len() {
                return Err(DecodeError::Erasure {
                    expect: code_len,
                    found: erasure,
                });
            }
            code[erasure] = F::zero();
            erasure_poly =
                erasure_poly * Polynomial(vec![F::one(), -pow(self.root.root.clone(), erasure)]);
        }

        let v = (self.fft)(code.clone());
        let Polynomial(mut syndrome) =
            Polynomial::new(v[1..=2 * self.correction_level].iter().cloned())
                * erasure_poly.clone();
        let syndrome = Polynomial::new(
            syndrome.drain(..std::cmp::min(syndrome.len(), 2 * self.correction_level + 1)),
        );
        let (err_locator, err_eval) = sugiyama(syndrome, self.correction_level, erasure.len());

        if err_locator.is_zero() {
            return Err(DecodeError::TooManyError(self.correction_level));
        }

        let err_locator = err_locator * erasure_poly;
        let forney = forney(err_locator.clone(), err_eval, 1, &self.root);
        let fft = (self.fft)(
            err_locator
                .0
                .into_iter()
                .chain(repeat(F::zero()))
                .take(self.root.order)
                .collect(),
        );

        let mut error_positions = BTreeSet::new();
        for location in
            fft.into_iter()
                .enumerate()
                .filter_map(|(location, e)| if e.is_zero() { Some(location) } else { None })
        {
            let location = if location > 0 {
                self.root.order - location
            } else {
                0
            };
            let correction = forney(location);
            code[location] = code[location].clone() - correction;
            error_positions.insert(location);
        }
        let error_positions = error_positions.into_iter().collect();
        let (Polynomial(data), rem) = Polynomial::new(code).div_with_rem(self.generator.clone());
        if rem.is_zero() {
            let output = data
                .into_iter()
                .chain(repeat(F::zero()))
                .take(self.data_len)
                .collect();
            Ok(DecodeResult {
                output,
                error_positions,
            })
        } else {
            Err(DecodeError::TooManyError(self.correction_level))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use lazy_static::lazy_static;

    use crate::{
        field::GF2561D,
        fourier::{GF2561DG2_255_FFT, GF2561DG2_UNITY_ROOT},
    };

    lazy_static! {
        static ref RS_255_223: ReedSolomon<GF2561D> =
            ReedSolomon::new(16, GF2561DG2_UNITY_ROOT.clone(), GF2561DG2_255_FFT.clone());
    }

    #[test]
    fn it_works() {
        let rs = &RS_255_223;
        let input: Vec<_> = (0..223).map(GF2561D).collect();
        let code = rs.encode(&input).unwrap();
        let data = rs
            .decode(
                {
                    let mut code = code.clone();
                    code[10] += GF2561D(1);
                    code[11] += GF2561D(1);
                    code[12] += GF2561D(1);
                    code[13] += GF2561D(1);
                    code[14] += GF2561D(1);
                    code[1] += GF2561D(1);
                    code[2] += GF2561D(1);
                    code[3] += GF2561D(1);
                    code[4] += GF2561D(1);
                    code[5] += GF2561D(23);
                    code[6] += GF2561D(1);
                    code[77] += GF2561D(255);
                    code[7] += GF2561D(1);
                    code[8] += GF2561D(1);
                    code[9] += GF2561D(1);
                    code
                },
                vec![44, 88],
            )
            .unwrap();
        assert_eq!(input, &*data);
    }

    #[quickcheck]
    fn encode_decode_error_erasure(
        input: Vec<GF2561D>,
        error: Vec<(u8, GF2561D)>,
        erasure: Vec<u8>,
    ) {
        let input: Vec<_> = input
            .into_iter()
            .chain(repeat(GF2561D::zero()))
            .take(223)
            .collect();
        let mut code = RS_255_223.encode(&input).unwrap();
        let mut pos = std::collections::HashSet::new();
        let mut actual_error = vec![];
        let mut total_check_used = 0;
        for (ep, e) in error {
            if total_check_used < 30 && ep < 255 && !pos.contains(&ep) {
                pos.insert(ep);
                actual_error.push((ep, e));
                code[ep as usize] += e;
                total_check_used += 2;
            }
        }
        let mut actual_erasure = vec![];
        for e in erasure {
            if total_check_used < 32 && e < 255 && !pos.contains(&e) {
                pos.insert(e);
                actual_erasure.push(e as usize);
                code[e as usize] = GF2561D::zero();
                total_check_used += 1;
            }
        }
        let output = RS_255_223
            .decode(code, actual_erasure.clone())
            .unwrap_or_else(|e| {
                panic!(
                    "error: {}, error={:?}, erasure={:?}, input={:?}",
                    e, actual_error, actual_erasure, input
                )
            });
        assert_eq!(input, &*output);
    }
}
