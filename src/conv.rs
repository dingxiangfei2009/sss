use std::{
    mem::MaybeUninit,
    ops::{Mul, Neg, Sub},
};

use ndarray::{
    Array, Array1, Array2, ArrayView, ArrayView1, ArrayView2, Axis, ErrorKind, RemoveAxis,
    ShapeError, Zip,
};
use num::{One, Zero};

macro_rules! stack {
    ($axis:expr, $( $array:expr ),+ ) => {
        $crate::conv::stack($axis, &[ $(::ndarray::ArrayView::from(&$array) ),* ]).unwrap()
    }
}

pub fn toeplitz_d(n: usize) -> Array2<i8> {
    assert!(n > 1 && n & 1 > 0);
    let k = n >> 1;
    stack![
        Axis(0),
        stack![Axis(1), -Array2::eye(k), Array2::zeros((k, k + 1))],
        Array2::ones((1, n)),
        stack![Axis(1), Array2::zeros((k, k + 1)), -Array2::eye(k)]
    ]
}

pub fn toeplitz_e(n: usize) -> (usize, Array2<i8>) {
    assert!(n > 0);
    let k = n >> 1;
    if n == 1 {
        (1, Array2::eye(1))
    } else if n & 1 > 0 {
        let (m, e) = toeplitz_e(k);
        let d = toeplitz_d(n);
        let e = stack![
            Axis(1),
            stack![Axis(0), e, Array2::zeros((1, m)), e],
            stack![Axis(0), e, Array2::zeros((k + 1, m))],
            d,
            stack![Axis(0), Array2::zeros((k + 1, m)), e]
        ];
        let m = 3 * m + n;
        assert_eq!((n, m), e.dim());
        (m, e)
    } else {
        let (m, e) = toeplitz_e(k);
        let e = stack![
            Axis(1),
            stack![Axis(0), e, e],
            stack![Axis(0), e, Array2::zeros((k, m))],
            stack![Axis(0), Array2::zeros((k, m)), e]
        ];
        let m = 3 * m;
        assert_eq!((n, m), e.dim());
        (m, e)
    }
}

fn mat_vec_mul<F>(a: ArrayView2<F>, b: ArrayView1<F>) -> Vec<F>
where
    F: Zero + Clone + Mul<Output = F>,
{
    assert_eq!(a.dim().1, b.dim());
    a.axis_iter(Axis(0))
        .map(|r| (r.to_owned() * b.to_owned()).scalar_sum())
        .collect()
}

fn mat_mat_mul<'a, 'b, F>(
    a: impl Into<ArrayView2<'a, F>>,
    b: impl Into<ArrayView2<'b, F>>,
) -> Array2<F>
where
    F: 'a + 'b + Zero + Clone + Mul<Output = F>,
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

fn stack<'a, A, D>(axis: Axis, arrays: &[ArrayView<'a, A, D>]) -> Result<Array<A, D>, ShapeError>
where
    A: Clone,
    D: RemoveAxis,
{
    if arrays.is_empty() {
        return Err(ShapeError::from_kind(ErrorKind::Unsupported));
    }
    let mut res_dim = arrays[0].raw_dim();
    if axis.index() >= res_dim.ndim() {
        return Err(ShapeError::from_kind(ErrorKind::OutOfBounds));
    }
    let common_dim = res_dim.remove_axis(axis);
    if arrays
        .iter()
        .any(|a| a.raw_dim().remove_axis(axis) != common_dim)
    {
        return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
    }

    let stacked_dim = arrays.iter().map(|a| a.len_of(axis)).sum();
    let Axis(ref axis_idx) = axis;
    res_dim.as_array_view_mut()[*axis_idx] = stacked_dim;

    // we can safely use uninitialized values here because they are Copy
    // and we will only ever write to them
    let size = res_dim.size();
    let mut v = Vec::with_capacity(size);
    v.resize_with(size, MaybeUninit::uninit);
    let mut res: Array<MaybeUninit<A>, _> = Array::from_shape_vec(res_dim.clone(), v)?;

    {
        let mut assign_view = res.view_mut();
        let arrays = arrays.into_iter().map(|a| a.mapv(MaybeUninit::new));
        for mut array in arrays {
            let len = array.len_of(axis);
            let (mut front, rest) = assign_view.split_at(axis, len);
            Zip::from(&mut front).and(&mut array).apply(|dst, src|
                       // this is safe, because `MaybeUninit` does not manage `Drop`
                       unsafe { src.as_ptr().copy_to(dst.as_mut_ptr(), 1) });
            assign_view = rest;
        }
    }
    let res = Array::from_shape_vec(
        res_dim,
        res.into_raw_vec()
            .into_iter()
            .map(|x| unsafe { x.assume_init() as A })
            .collect(),
    )?;
    Ok(res)
}

fn mat_y(n: usize) -> Array2<i8> {
    let mut y = Array2::zeros((n, n));
    for i in 0..n {
        y[[i, n - 1 - i]] = 1;
    }
    y
}

pub fn toeplitz_gh<F>(n: usize) -> (usize, Array2<F>, Array2<i8>)
where
    F: Zero + One + Clone + Sub<Output = F> + Mul<Output = F> + Neg<Output = F>,
{
    assert!(n > 0);
    let k = n >> 1;
    if n == 1 {
        (1, Array2::eye(1), Array2::eye(1))
    } else if n & 1 > 0 {
        let (m, g, h) = toeplitz_gh(k);
        let d_ = toeplitz_d(n);
        let d = d_.map(|x| {
            if *x > 0 {
                F::one()
            } else if *x < 0 {
                -F::one()
            } else {
                F::zero()
            }
        });
        #[allow(non_snake_case)]
        let q__1 = stack![Axis(1), Array2::eye(n - 2), Array2::zeros((n - 2, n + 1))];
        let q_0 = stack![
            Axis(1),
            Array2::zeros((n - 2, k + 1)),
            Array2::eye(n - 2),
            Array2::zeros((n - 2, k + 1))
        ];
        let q_1 = stack![Axis(1), Array2::zeros((n - 2, n + 1)), Array2::eye(n - 2)];
        let mut g_0 = Array2::zeros((m, 2 * n - 1));
        let mut g_1 = Array2::zeros((m, 2 * n - 1));
        for i in 0..m {
            if let Some(Some(j)) =
                h.index_axis(Axis(0), i)
                    .indexed_iter()
                    .fold(None, |s, (j, h)| match s {
                        None if *h != 0 => Some(Some(j)),
                        Some(_) if *h != 0 => Some(None),
                        s => s,
                    })
            {
                g_0[[i, 2 * k - j - 1]] = F::one();
                g_1[[i, 3 * k - j]] = F::one();
            }
        }
        let g = stack![
            Axis(0),
            mat_mat_mul(&g, &q_0),
            mat_mat_mul(&g, &(q__1 - q_0.clone())) + g_0,
            stack![Axis(1), Array2::zeros((n, k)), d, Array2::zeros((n, k))],
            mat_mat_mul(&g, &(q_1 - q_0)) + g_1
        ];
        let y = mat_y(n);
        let h = stack![
            Axis(0),
            stack![Axis(1), h.clone(), Array2::zeros((m, 1)), h.clone()],
            stack![Axis(1), Array2::zeros((m, k + 1)), h.clone()],
            mat_mat_mul(d_.t(), &y),
            stack![Axis(1), h.clone(), Array2::zeros((m, k + 1))]
        ];
        let m = 3 * m + n;
        assert_eq!((m, 2 * n - 1), g.dim());
        assert_eq!((m, n), h.dim());
        (m, g, h)
    } else {
        let (m, g, h) = toeplitz_gh(k);
        #[allow(non_snake_case)]
        let p__1 = stack![Axis(1), Array2::eye(n - 1), Array2::zeros((n - 1, n))];
        let p_0 = stack![
            Axis(1),
            Array2::zeros((n - 1, k)),
            Array2::eye(n - 1),
            Array2::zeros((n - 1, k))
        ];
        let p_1 = stack![Axis(1), Array2::zeros((n - 1, n)), Array2::eye(n - 1)];
        let g = stack![
            Axis(0),
            mat_mat_mul(&g, &p_0),
            mat_mat_mul(&g, &(p__1 - p_0.clone())),
            mat_mat_mul(&g, &(p_1 - p_0))
        ];
        let h = stack![
            Axis(0),
            stack![Axis(1), h, h],
            stack![Axis(1), Array2::zeros((m, k)), h],
            stack![Axis(1), h, Array2::zeros((m, k))]
        ];
        let m = 3 * m;
        assert_eq!((m, 2 * n - 1), g.dim());
        assert_eq!((m, n), h.dim());
        (m, g, h)
    }
}

pub fn toeplitz_unit_mul<F>(u: Array2<i8>, v: Array1<F>) -> Array1<F>
where
    F: Zero + Clone + Sub<Output = F> + Mul<Output = F>,
{
    u.axis_iter(Axis(0))
        .map(|r| {
            Zip::from(r).and(&v).fold(F::zero(), |a, u, v| {
                if u > &0 {
                    a + v.clone()
                } else if u < &0 {
                    a - v.clone()
                } else {
                    a
                }
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::{cmp::PartialEq, fmt::Display};

    use quickcheck::{Arbitrary, Gen, StdThreadGen};

    use crate::{
        field::GF2561D,
        tests::{Frac, F7},
    };

    #[quickcheck]
    fn wu2012_lemma4_5(n: usize) {
        let n = n % 512 + 1;
        let (_, e) = toeplitz_e(n);
        let (_, _, h) = toeplitz_gh::<F7>(n);
        let y = mat_y(n);
        assert_eq!(e, h.dot(&y).t());
        assert!(h.axis_iter(Axis(0)).all(|r| !r.iter().all(|h| *h == 0)));
        let y_ = Array2::from_shape_vec(
            (n, n),
            h.axis_iter(Axis(0))
                .filter(|r| r.iter().filter(|h| **h != 0).count() == 1)
                .flat_map(|r| r.to_vec())
                .collect(),
        )
        .unwrap();
        assert_eq!(y_, mat_y(n));
    }

    #[quickcheck]
    fn wu2012_lemma6(n: usize) {
        let n = n % 512 + 1;
        let (_, e) = toeplitz_e(n);
        assert!(e.axis_iter(Axis(1)).all(|c| !c.iter().all(|e| *e == 0)));
    }

    #[test]
    fn matrix_mul() {
        for _ in 0..100 {
            let mut g = StdThreadGen::new(usize::max_value());
            let n = usize::arbitrary(&mut g) % 100 + 1;
            let m = usize::arbitrary(&mut g) % 100 + 1;
            let p = usize::arbitrary(&mut g) % 100 + 1;
            let a = Array2::from_shape_vec(
                (n, m),
                std::iter::repeat(())
                    .take(n * m)
                    .map(|_| F7::arbitrary(&mut g))
                    .collect(),
            )
            .unwrap();
            let b = Array2::from_shape_vec(
                (m, p),
                std::iter::repeat(())
                    .take(p * m)
                    .map(|_| F7::arbitrary(&mut g))
                    .collect(),
            )
            .unwrap();
            assert_eq!(mat_mat_mul(a.view(), b.view()), a.dot(&b));
        }
    }

    #[test]
    fn it_works() {
        toeplitz_e(10);
        toeplitz_gh::<GF2561D>(10);
        toeplitz_gh::<Frac>(10);
    }

    fn toeplitz_vec_mul<F>(t: Array1<F>, v: Array1<F>) -> Array1<F>
    where
        F: Zero + One + Clone + Sub<Output = F> + Neg<Output = F>,
    {
        let n = v.dim();
        assert_eq!(2 * n - 1, t.dim());
        let (_, e) = toeplitz_e(n);
        let (_, g, h) = toeplitz_gh(n);
        eprintln!("g, e, h are generated");
        // u = E . ((G . t) * (H . v))
        let x = Array1::from(mat_vec_mul(g.view(), t.view())) * toeplitz_unit_mul(h, v);
        toeplitz_unit_mul(e, x)
    }

    fn toeplitz_test<F, G>(g: &mut G)
    where
        F: PartialEq + Display + Arbitrary + Zero + One + Clone + Sub<Output = F> + Neg<Output = F>,
        G: Gen,
    {
        let n = u8::arbitrary(g) as usize + 1;
        eprintln!("size n={}", n);
        let t: Array1<_> = (0..2 * n - 1).map(|_| F::arbitrary(g)).collect();
        let v: Array1<_> = (0..n).map(|_| F::arbitrary(g)).collect();

        let toeplitz = Array2::from_shape_vec(
            (n, n),
            (0..n)
                .flat_map(|i| (0..n).map(|j| t[n - 1 + i - j].clone()).collect::<Vec<_>>())
                .collect(),
        )
        .expect("shape should be square");
        let expected = Array1::from(mat_vec_mul(toeplitz.view(), v.view()));
        eprintln!("evaluating actual");
        let actual = toeplitz_vec_mul(t.clone(), v.clone());
        if expected != actual {
            eprintln!("t={}", t);
            eprintln!("v={}", v);
            eprintln!("toeplitz={}", toeplitz);
            eprintln!("expected={}", expected);
            eprintln!("actual={}", actual);
            panic!()
        }
    }

    #[test]
    fn toeplitz() {
        let mut g = StdThreadGen::new(usize::max_value());
        for i in 0..100 {
            eprintln!("test {}", i);
            toeplitz_test::<F7, _>(&mut g)
        }
    }
}
