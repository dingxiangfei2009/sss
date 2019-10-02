use std::ops::Neg;

use alga::general::Field;
use ndarray::{stack, Array2, Axis, LinalgScalar};

pub fn toeplitz_d(n: usize) -> Array2<i8> {
    assert!(n > 1 && n & 1 > 0);
    let k = n >> 1;
    stack![
        Axis(1),
        stack![
            Axis(0),
            -Array2::eye(k),
            Array2::ones((1, k)),
            Array2::zeros((k, k))
        ],
        stack![
            Axis(0),
            Array2::zeros((k, 1)),
            Array2::eye(1),
            Array2::zeros((k, 1))
        ],
        stack![
            Axis(0),
            Array2::zeros((k, k)),
            Array2::ones((1, k)),
            -Array2::eye(k)
        ]
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
            stack![Axis(0), e, Array2::zeros((1, m)), Array2::zeros([k, m])],
            d,
            stack![Axis(0), Array2::zeros((k, m)), Array2::zeros([1, m]), e]
        ];
        (3 * m, e)
    } else {
        let (m, e) = toeplitz_e(k);
        let e = stack![
            Axis(1),
            stack![Axis(0), e, e],
            stack![Axis(0), e, Array2::zeros((k, m))],
            stack![Axis(0), Array2::zeros((k, m)), e]
        ];
        (3 * m + n, e)
    }
}

pub fn toeplitz_gh<F: LinalgScalar + Neg<Output = F>>(n: usize) -> (usize, Array2<F>, Array2<i8>) {
    assert!(n > 0);
    eprintln!("n={}", n);
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
        let q_1 = stack![
            Axis(1),
            Array2::zeros((n - 2, n + 1)),
            Array2::eye(n - 2)
        ];
        let mut g_0 = Array2::zeros((m, 2 * n - 1));
        let mut g_1 = Array2::zeros((m, 2 * n - 1));
        for i in 0..m {
            if let Some(Some(j)) =
                h.index_axis(Axis(0), i)
                    .indexed_iter()
                    .fold(None, |s, (j, h)| match s {
                        None if h > &0i8 => Some(Some(j)),
                        Some(_) if h > &0i8 => Some(None),
                        s => s,
                    })
            {
                g_0[[i, 2 * k - j - 1]] = F::one();
                g_1[[i, 3 * k - j]] = F::one();
            }
        }
        let g_ = stack![
            Axis(0),
            g.clone().dot(&q_0),
            g.clone().dot(&(q__1 - q_0.clone())) + g_0,
            stack![Axis(1), Array2::zeros((n, k)), d, Array2::zeros((n, k))],
            g.dot(&(q_1 - q_0)) + g_1
        ];
        let mut y = Array2::zeros((n, n));
        for i in 0..n {
            y[[i, n - 1 - i]] = 1;
        }
        let h_ = stack![
            Axis(0),
            stack![Axis(1), h.clone(), Array2::zeros((m, 1)), h.clone()],
            stack![Axis(1), Array2::zeros((m, k + 1)), h.clone()],
            d_.t().to_owned().dot(&y),
            stack![Axis(1), h.clone(), Array2::zeros((m, k + 1))]
        ];
        (3 * m + n, g_, h_)
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
        let g_ = stack![
            Axis(0),
            g.clone().dot(&p_0),
            g.clone().dot(&(p__1 - p_0.clone())),
            g.dot(&(p_1 - p_0))
        ];
        let h_ = stack![
            Axis(0),
            stack![Axis(1), h, h],
            stack![Axis(1), Array2::zeros((m, k)), h],
            stack![Axis(1), h, Array2::zeros((m, k))]
        ];
        (3 * m, g_, h_)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::GF2561D;

    #[test]
    fn it_works() {
        toeplitz_e(2);
        toeplitz_gh::<GF2561D>(5);
    }
}
