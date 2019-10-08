use std::ops::Div;

use alga::general::Field;
use lazy_static::lazy_static;
use ndarray::{Array1, Array2, ArrayViewMut, Axis};
use num::traits::{One, Zero};

use crate::{
    conv::{int_inj, BilinearAlgorithm, LinearOperator},
    field::{FiniteField, F2},
    linalg::mat_vec_mul,
    pow,
};

#[derive(Clone, Debug)]
pub struct UnityRoot<F> {
    pub order: usize,
    pub root: F,
}

impl<F: Field> UnityRoot<F> {
    /// Compute the cyclotomic subgroup of order m in this group
    pub fn subgroup(self, m: usize) -> Self {
        let Self { order, root } = self;
        let subgroup_order = order / m;
        assert_eq!(order, subgroup_order * m);
        Self {
            root: crate::pow(root, subgroup_order),
            order: m,
        }
    }
}

/// Cooley Tukey fast Fourier Transform where the order of the root of unity
/// decomposes into non-trivial factors
pub fn cooley_tukey<F>(
    n1: usize,
    n2: usize,
    root: UnityRoot<F>,
    transform1: impl Sync + Fn(Vec<F>) -> Vec<F>,
    transform2: impl Sync + Fn(Vec<F>) -> Vec<F>,
) -> impl Fn(Vec<F>) -> Vec<F>
where
    F: Field + Clone + Send + Sync,
{
    use rayon::prelude::*;

    move |mut x: Vec<F>| {
        assert_eq!(n1 * n2, x.len());
        assert_eq!(x.len(), root.order);

        let mut ws = vec![];
        let mut w = F::one();
        for _ in 0..n2 {
            ws.push(w.clone());
            w *= root.root.clone();
        }

        let mut x_view = ArrayViewMut::from_shape((n1, n2), &mut x).expect("shape must be correct");
        x_view
            .axis_iter_mut(Axis(1))
            .into_par_iter()
            .for_each(|mut col| {
                let col_fft = transform1(col.to_owned().to_vec());
                col.iter_mut().zip(col_fft.into_iter()).for_each(|(a, b)| {
                    *a = b;
                });
            });
        x_view
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                let row_fft = transform2(
                    row.iter()
                        .zip(ws.clone().into_iter().map(|x| crate::pow(x, i)))
                        .map(|(a, b)| a.clone() * b)
                        .collect(),
                );
                row.iter_mut()
                    .zip(row_fft.into_iter())
                    .for_each(|(a, b)| *a = b);
            });
        x
    }
}

/// Fourier Transform by definition
pub fn naive<F>(root: UnityRoot<F>) -> impl Fn(Vec<F>) -> Vec<F>
where
    F: Field + Clone + Send + Sync,
{
    move |x: Vec<F>| {
        let n = x.len();
        assert_eq!(n, root.order);

        let (ws_, _) = (0..n).fold((vec![], F::one()), |(mut ws, w), _| {
            ws.push(w.clone());
            (ws, w * root.root.clone())
        });
        let mut ws = vec![F::one(); n];

        let mut ys = vec![];
        for _ in 0..n {
            ys.push(
                ws.iter()
                    .zip(x.iter())
                    .fold(F::zero(), |y, (w, x)| y + w.clone() * x.clone()),
            );
            for (w, w_) in ws.iter_mut().zip(ws_.iter()) {
                *w *= w_.clone();
            }
        }

        ys
    }
}

/// finite field fast Fourier Transform with convolution
pub fn conv<F, BA>(
    assoc: Array2<F::Scalar>,
    cyclotomic_cosets: Vec<Vec<usize>>,
    conv_bilinear_algos: &[&BA],
    normal_basis: &[F],
) -> impl Send + Sync + Fn(Vec<F>) -> Vec<F>
where
    F: 'static + FiniteField + Clone + Send + Sync,
    BA: BilinearAlgorithm<F>,
{
    let p = F::CHARACTERISTIC;
    assert_eq!(cyclotomic_cosets.len(), conv_bilinear_algos.len());
    assert_eq!(cyclotomic_cosets.len(), normal_basis.len());
    let convs: Vec<_> = conv_bilinear_algos
        .into_iter()
        .zip(normal_basis)
        .zip(cyclotomic_cosets.iter().map(|c| c.len()))
        .map(|((algo, basis), size)| {
            let mut x = Vec::with_capacity(size);
            let mut gamma = basis.clone();
            for _ in 0..size {
                x.push(gamma.clone());
                gamma = pow(gamma, p);
            }
            algo.apply(&x)
        })
        .collect();

    let assoc = assoc.map(|a| F::from_scalar(a.clone()));

    move |x: Vec<F>| {
        let mut r = Vec::with_capacity(x.len());
        r.resize_with(x.len(), F::zero);
        mat_vec_mul(
            assoc.view(),
            &convs
                .iter()
                .zip(&cyclotomic_cosets)
                .flat_map(|(conv, coset)| {
                    let mut y: Vec<_> = coset.iter().map(|idx| x[*idx].clone()).collect();
                    y[1..].reverse();
                    conv(&y)
                })
                .collect::<Array1<_>>(),
        )
    }
}

lazy_static! {
    pub static ref GF2561DG2_255_COSETS_CONV: Vec<Vec<usize>> = {
        crate::facts::GF2561D_255_COSETS
            .into_iter()
            .map(|v| v.to_vec())
            .collect()
    };
    pub static ref GF2561DG2_255_ASSOC_CONV: Array2<F2> = {
        Array2::from_shape_vec(
            (255, 255),
            crate::facts::GF2561DG2_255_ASSOC_TABLE
                .into_iter()
                .flat_map(|v| v.into_iter().map(|x| F2(*x)))
                .collect(),
        )
        .expect("shape should be correct")
    };
}

#[derive(Clone)]
pub struct NaiveFourierTransform<A> {
    algo: A,
    size: usize,
    inverse: bool,
}

impl<F, A> LinearOperator<F> for NaiveFourierTransform<A>
where
    A: Fn(Vec<F>) -> Vec<F>,
    F: Clone + One + Zero + Div<Output = F>,
{
    fn op(&self, x: &[F]) -> Vec<F> {
        let mut result = (self.algo)(x.to_vec());
        if self.inverse {
            let n: F = int_inj(self.size);
            result.iter_mut().for_each(|x| *x = x.clone() / n.clone());
            result
        } else {
            result
        }
    }
    fn source_dim(&self) -> usize {
        self.size
    }
    fn dest_dim(&self) -> usize {
        self.size
    }
}

#[derive(Clone)]
pub struct NaiveFourierConv<F, A> {
    algo: A,
    root: UnityRoot<F>,
}

impl<F: Send + Sync + Clone, A> BilinearAlgorithm<F> for NaiveFourierConv<F, A>
where
    F: 'static + Field,
    A: 'static + Clone + Send + Sync + Fn(Vec<F>) -> Vec<F>,
{
    type A = NaiveFourierTransform<A>;
    type B = NaiveFourierTransform<A>;
    type C = NaiveFourierTransform<A>;
    fn make_algorithm(&self) -> (Self::A, Self::B, Self::C) {
        (
            NaiveFourierTransform {
                algo: self.algo.clone(),
                size: self.root.order,
                inverse: false,
            },
            NaiveFourierTransform {
                algo: self.algo.clone(),
                size: self.root.order,
                inverse: false,
            },
            NaiveFourierTransform {
                algo: self.algo.clone(),
                size: self.root.order,
                inverse: true,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;

    use num::traits::One;
    use quickcheck::{Arbitrary, Gen, TestResult, Testable};

    use crate::{
        conv::ToeplitzConv,
        field::{FinitelyGenerated, GF2561D, GF2561DG2},
        linalg::solve,
    };

    #[test]
    fn assoc_255() {
        let order = 255;
        let assoc = compute_assoc_gf2561d(order);
        println!("[");
        for i in 0..order {
            print!("[");
            for j in 0..order {
                print!("{}, ", assoc[[i, j]]);
            }
            println!("],")
        }
        println!("]");
    }

    #[test]
    fn assoc_3() {
        let order = 3;
        let assoc = compute_assoc_gf2561d(order);
        println!("[");
        for i in 0..order {
            print!("[");
            for j in 0..order {
                print!("{}, ", assoc[[i, j]]);
            }
            println!("],")
        }
        println!("]");
    }

    #[test]
    fn assoc_5() {
        let order = 3;
        let assoc = compute_assoc_gf2561d(order);
        println!("[");
        for i in 0..order {
            print!("[");
            for j in 0..order {
                print!("{}, ", assoc[[i, j]]);
            }
            println!("],")
        }
        println!("]");
    }

    #[test]
    fn assoc_17() {
        let order = 17;
        let assoc = compute_assoc_gf2561d(order);
        println!("[");
        for i in 0..order {
            print!("[");
            for j in 0..order {
                print!("{}, ", assoc[[i, j]]);
            }
            println!("],")
        }
        println!("]");
    }

    fn compute_assoc_gf2561d(order: usize) -> Array2<F2> {
        type F = GF2561D;
        type Scalar = <F as FiniteField>::Scalar;
        assert_eq!(255 % order, 0);
        let cosets = crate::compute_cyclotomic_cosets::<F>(order);
        // there are three subfields of interest and correspondingly three normal basis
        let normal_bases = {
            let mut map = HashMap::new();
            map.insert(2, crate::facts::GF2561D_SUB_2_NORMAL_BASIS.to_vec());
            map.insert(4, crate::facts::GF2561D_SUB_4_NORMAL_BASIS.to_vec());
            map.insert(8, crate::facts::GF2561D_SUB_8_NORMAL_BASIS.to_vec());
            map
        };
        let gamma = <F as FinitelyGenerated<GF2561DG2>>::GENERATOR;
        let mut assoc = vec![];
        for j in 0..order {
            let gamma = pow(gamma, j);
            let mut a = vec![Scalar::one()];
            for i in 1..cosets.len() {
                let gamma = pow(gamma, cosets[i][0]).to_vec();
                let subfield_deg_ext = cosets[i].len();
                let mut system: Vec<_> = normal_bases
                    .get(&subfield_deg_ext)
                    .expect("missing bases")
                    .into_iter()
                    .flat_map(|gamma| gamma.to_vec())
                    .collect();
                system.extend(gamma);
                let solution = solve(
                    Array2::from_shape_vec((subfield_deg_ext + 1, F::DEGREE_EXTENSION), system)
                        .expect("shape should be correct")
                        .t()
                        .to_owned(),
                ).expect("normal bases should be linear independent and the root of unity should be in the correct subfield");
                a.extend(solution.into_iter().take(subfield_deg_ext));
            }
            assoc.extend(a.into_iter())
        }
        Array2::from_shape_vec((order, order), assoc).expect("shape should be correct")
    }

    lazy_static! {
        static ref GF2561D_255_FFT: Box<dyn Send + Sync + Fn(Vec<GF2561D>) -> Vec<GF2561D>> = {
            let assoc = GF2561DG2_255_ASSOC_CONV.clone();
            let cosets = GF2561DG2_255_COSETS_CONV.clone();
            let basis: Vec<_> = cosets
                .iter()
                .map(|set| match set.len() {
                    1 => GF2561D::one(),
                    2 => crate::facts::GF2561D_SUB_2_NORMAL_BASIS[0],
                    4 => crate::facts::GF2561D_SUB_4_NORMAL_BASIS[0],
                    8 => crate::facts::GF2561D_SUB_8_NORMAL_BASIS[0],
                    _ => panic!("check your math"),
                })
                .collect();
            let algos: Vec<_> = cosets.iter().map(|set| ToeplitzConv(set.len())).collect();
            let algos: Vec<_> = algos.iter().map(std::convert::identity).collect();
            Box::new(conv(assoc, cosets, &algos, &basis))
        };
        static ref GF2561D_255_NAIVE_FFT: Box<dyn Send + Sync + Fn(Vec<GF2561D>) -> Vec<GF2561D>> = {
            let root = UnityRoot {
                root: <GF2561D as FinitelyGenerated<GF2561DG2>>::GENERATOR,
                order: 255,
            };
            Box::new(naive(root))
        };
    }

    #[test]
    fn fourier_255_quickcheck() {
        let mut quickcheck = quickcheck::QuickCheck::with_gen(quickcheck::StdThreadGen::new(1024))
            .min_tests_passed(100);
        struct Test;
        impl Testable for Test {
            fn result<G: Gen>(&self, g: &mut G) -> TestResult {
                let x: Vec<GF2561D> = Arbitrary::arbitrary(g);
                if x.len() < 255 {
                    TestResult::discard()
                } else {
                    let x: Vec<_> = x.into_iter().take(255).collect();
                    let y = GF2561D_255_FFT(x.clone());
                    let z = GF2561D_255_NAIVE_FFT(x);
                    if y == z {
                        TestResult::passed()
                    } else {
                        TestResult::error(format!("expecting {:?}, got {:?}", z, y))
                    }
                }
            }
        }
        quickcheck.quickcheck(Test)
    }

    #[test]
    fn fourier_255() {
        let x: Vec<_> = (0..255).map(GF2561D).collect();
        let y = GF2561D_255_FFT(x.clone());
        let z = GF2561D_255_NAIVE_FFT(x);
        assert_eq!(y, z)
    }

    lazy_static! {
        static ref GF2561D_NORMAL_BASIS_8_CONV: Box<dyn Send + Sync + Fn(&[GF2561D]) -> Vec<GF2561D>> =
            { ToeplitzConv(8).apply(&crate::facts::GF2561D_SUB_8_NORMAL_BASIS) };
    }

    #[quickcheck]
    fn circulant_mat_mul_is_conv8(
        x: (
            GF2561D,
            GF2561D,
            GF2561D,
            GF2561D,
            GF2561D,
            GF2561D,
            GF2561D,
            GF2561D,
        ),
    ) {
        let (x0, x1, x2, x3, x4, x5, x6, x7) = x;
        let x = vec![x0, x7, x6, x5, x4, x3, x2, x1];
        let actual = GF2561D_NORMAL_BASIS_8_CONV(&x);

        let mut gamma = crate::facts::GF2561D_SUB_8_NORMAL_BASIS;
        let x = vec![x0, x1, x2, x3, x4, x5, x6, x7];
        for actual in &actual {
            assert_eq!(
                *actual,
                gamma
                    .iter()
                    .zip(&x)
                    .fold(GF2561D::zero(), |s, (g, x)| s + *g * *x)
            );
            for gamma in &mut gamma {
                *gamma = pow(*gamma, 2);
            }
        }
    }
}
