use std::ops::Mul;

use alga::general::Field;
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, Axis, Zip};
use num::Zero;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

pub fn mat_vec_mul<'a, 'b, A, B>(
    a: impl Into<ArrayView2<'a, A>>,
    b: impl Into<ArrayView1<'b, B>>,
) -> Vec<B>
where
    A: 'a + 'b + Zero + Clone + Mul<B, Output = B>,
    B: 'a + 'b + Zero + Clone,
{
    let a = a.into();
    let b = b.into();
    assert_eq!(a.dim().1, b.dim());
    a.axis_iter(Axis(0))
        .map(|a| {
            Zip::from(a)
                .and(b)
                .fold(B::zero(), |s, a, b| s + a.clone() * b.clone())
        })
        .collect()
}

pub fn mat_mat_mul<'a, 'b, A, B>(
    a: impl Into<ArrayView2<'a, A>>,
    b: impl Into<ArrayView2<'b, B>>,
) -> Array2<B>
where
    A: 'a + 'b + Zero + Clone + Mul<B, Output = B>,
    B: 'a + 'b + Zero + Clone,
{
    let a = a.into();
    let b = b.into();
    let (n, m) = a.dim();
    let (m_, p) = b.dim();
    assert_eq!(m, m_);
    let c: Vec<_> = b
        .axis_iter(Axis(1))
        .flat_map(|c| mat_vec_mul(a, c))
        .collect();
    Array2::from_shape_vec((p, n), c)
        .expect("shape should be correct")
        .t()
        .to_owned()
}

/// solve a system of linear equation with Gaussian Elimination
pub fn solve<T: Field + Send + Sync>(mut a: Array2<T>) -> Option<Array1<T>> {
    let (equations, unknowns) = a.dim();
    assert!(unknowns > 0, "there needs to be at least one unknown");
    let unknowns = unknowns - 1;
    if unknowns > equations {
        None?;
    }

    for unknown in 0..std::cmp::min(unknowns, equations) {
        if a[[unknown, unknown]].is_zero() {
            let mut a = a.slice_mut(s![unknown.., ..]);
            let mut it = a.axis_iter_mut(Axis(0));
            let this_row = it.next().expect("row must exists");
            let that_row = it.filter(|row| !row[unknown].is_zero()).next()?;
            Zip::from(this_row).and(that_row).par_apply(std::mem::swap);
        }
        let (mut this_equation, mut rest_equations) =
            a.slice_mut(s![unknown.., ..]).split_at(Axis(0), 1);
        let mut this_equation = this_equation.index_axis_mut(Axis(0), 0);
        let scale = this_equation[unknown].clone();
        assert!(!scale.is_zero());
        this_equation.iter_mut().for_each(|a| {
            *a /= scale.clone();
        });

        rest_equations
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut eqn| {
                let scale = eqn[unknown].clone();
                if !scale.is_zero() {
                    eqn.iter_mut()
                        .zip(this_equation.iter())
                        .for_each(|(e, e_)| *e -= e_.clone() * scale.clone());
                }
            });
    }

    // a is in row echelon form now
    for unknown in (1..unknowns).rev() {
        let (mut rest_equations, this_equation) = a
            .slice_mut(s![..unknown + 1, ..])
            .split_at(Axis(0), unknown);
        let this_equation = this_equation.index_axis(Axis(0), 0);
        rest_equations.axis_iter_mut(Axis(0)).for_each(|eqn| {
            let scale = eqn[unknown].clone();
            Zip::from(eqn)
                .and(this_equation)
                .par_apply(|e, e_| *e -= e_.clone() * scale.clone());
        });
    }

    // assert the excessive equations are trivial, i.e. 0=0
    if unknowns >= equations
        || a.slice(s![unknowns + 1.., ..])
            .into_par_iter()
            .all(|x| x.is_zero())
    {
        Some(a.slice(s![.., -1]).map(|x| x.clone()))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::array;
    use num::rational::BigRational;

    use crate::tests::Frac;

    #[test]
    fn it_works() {
        let x = solve(
            array![[1, 1, 1, 4], [0, 1, 1, 3], [0, 2, 1, 2u32]]
                .map(|x| Frac(BigRational::from_integer((*x).into()))),
        )
        .unwrap();
        assert_eq!(
            x,
            array![1, -1, 4].map(|x| Frac(BigRational::from_integer((*x).into())))
        );

        let x = solve(
            array![[1, 1, 1, 10], [1, -1, -1, 4], [5, 1, 2, 8]]
                .map(|x| Frac(BigRational::from_integer((*x).into()))),
        )
        .unwrap();
        assert_eq!(
            x,
            array![7, 33, -30].map(|x| Frac(BigRational::from_integer((*x).into())))
        );

        let x = solve(
            array![[0, 0, 1, 2], [0, 1, 1, 3], [1, 1, 1, 1u32]]
                .map(|x| Frac(BigRational::from_integer((*x).into()))),
        )
        .unwrap();
        assert_eq!(
            x,
            array![-2, 1, 2].map(|x| Frac(BigRational::from_integer((*x).into())))
        );
    }

    #[test]
    fn f7() {
        use crate::tests::F7;
        use ndarray::{arr1, arr2};

        let m = arr2(&[
            [
                F7::new(1),
                F7::new(0),
                F7::new(6),
                F7::new(0),
                F7::new(0),
                F7::new(0),
                F7::new(0),
                F7::new(0),
            ],
            [
                F7::new(5),
                F7::new(5),
                F7::new(6),
                F7::new(6),
                F7::new(6),
                F7::new(6),
                F7::new(6),
                F7::new(2),
            ],
            [
                F7::new(3),
                F7::new(6),
                F7::new(6),
                F7::new(5),
                F7::new(3),
                F7::new(6),
                F7::new(5),
                F7::new(2),
            ],
            [
                F7::new(6),
                F7::new(4),
                F7::new(6),
                F7::new(4),
                F7::new(5),
                F7::new(1),
                F7::new(3),
                F7::new(2),
            ],
            [
                F7::new(3),
                F7::new(5),
                F7::new(6),
                F7::new(3),
                F7::new(5),
                F7::new(6),
                F7::new(3),
                F7::new(1),
            ],
            [
                F7::new(2),
                F7::new(3),
                F7::new(6),
                F7::new(2),
                F7::new(3),
                F7::new(1),
                F7::new(5),
                F7::new(6),
            ],
            [
                F7::new(2),
                F7::new(5),
                F7::new(6),
                F7::new(1),
                F7::new(6),
                F7::new(1),
                F7::new(6),
                F7::new(5),
            ],
        ]);
        let coeffs = solve(m).unwrap();
        assert_eq!(
            coeffs,
            arr1(&[
                F7::new(4),
                F7::new(2),
                F7::new(4),
                F7::new(3),
                F7::new(3),
                F7::new(1),
                F7::new(3)
            ])
        );
    }
}
