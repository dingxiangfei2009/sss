pub use alga::general::Field;

use crate::Polynomial;

pub struct LFSR<F> {
    pub connection: Polynomial<F>,
    pub len: usize,
}

pub fn berlekamp_massey<F>(s: &[F]) -> LFSR<F>
where
    F: Field + Clone,
{
    let mut lambda = vec![F::one()];
    let mut gamma = vec![F::zero()];
    let mut l = 0;
    let mut a = vec![-F::one()];
    let mut b = vec![F::one()];
    let mut m = 1;
    for i in 0..s.len() {
        let delta = lambda[0..=l]
            .iter()
            .cloned()
            .zip(s[i - l..=i].iter().cloned().rev())
            .fold(F::zero(), |s, (a, b)| a * b + s);
        if delta.is_zero() {
            m += 1;
        } else if l * 2 <= i {
            l = i + 1 - l;
            let mut a_: Vec<_> = gamma
                .clone()
                .into_iter()
                .map(|g| g / delta.clone())
                .collect();
            let mut b_: Vec<_> = lambda
                .clone()
                .into_iter()
                .map(|l| l / delta.clone())
                .collect();
            crate::truncate_high_degree_zeros(&mut a_);
            crate::truncate_high_degree_zeros(&mut b_);
            {
                let l = std::cmp::max(l + 1, b.len() + m);
                lambda.resize(l, F::zero());
                gamma.resize(l, F::zero());
            }
            assert!(
                lambda.len() >= b.len() + m,
                "expect at least {}, got {}",
                b.len() + m,
                lambda.len()
            );
            assert!(
                gamma.len() >= a.len() + m,
                "expect at least {}, got {}",
                a.len() + m,
                gamma.len()
            );
            lambda[m..]
                .iter_mut()
                .zip(b.clone().into_iter().map(|b| b * -delta.clone()))
                .for_each(|(l, b)| *l += b);
            gamma[m..]
                .iter_mut()
                .zip(a.clone().into_iter().map(|a| a * -delta.clone()))
                .for_each(|(g, a)| *g += a);

            a = a_;
            b = b_;
            m = 1;
        } else {
            lambda.push(F::zero());
            gamma.push(F::zero());
            assert!(
                lambda.len() >= b.len() + m,
                "expect at least {}, got {}",
                b.len() + m,
                lambda.len()
            );
            assert!(
                gamma.len() >= a.len() + m,
                "expect at least {}, got {}",
                a.len() + m,
                gamma.len()
            );
            lambda[m..]
                .iter_mut()
                .zip(b.clone().into_iter().map(|b| b * -delta.clone()))
                .for_each(|(l, b)| *l += b);
            gamma[m..]
                .iter_mut()
                .zip(a.clone().into_iter().map(|a| a * -delta.clone()))
                .for_each(|(l, a)| *l += a);

            m += 1;
        }
    }
    let connection = Polynomial::new(lambda);
    assert!(connection.0.len() <= l + 1);
    LFSR { connection, len: l }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::F7;

    #[quickcheck]
    fn berlekamp_massey_tests(s: Vec<F7>) {
        check(berlekamp_massey(&s), &s);
    }

    fn check<F: Field + std::fmt::Debug>(lfsr: LFSR<F>, s: &[F]) {
        let LFSR {
            connection: lambda,
            len: v,
        } = lfsr;
        let v = v + 1;
        if s.len() < v + 1 {
            return;
        }
        let mut failures = vec![];
        for i in 1..s.len() - v {
            let sum = lambda
                .0
                .clone()
                .into_iter()
                .zip(s[i..i + v].iter().rev().cloned())
                .fold(F::zero(), |s, (a, b)| s + a * b);
            if !sum.is_zero() {
                failures.push((i, sum));
            }
        }
        if !failures.is_empty() {
            panic!(
                "failures={:?}, s={:?}, lambda={:?}, v={}",
                failures, s, lambda, v
            );
        }
    }

    #[test]
    fn it_works() {
        let s = vec![
            F7::new(5),
            F7::new(0),
            F7::new(3),
            F7::new(6),
            F7::new(1),
            F7::new(6),
            F7::new(4),
            F7::new(1),
            F7::new(3),
            F7::new(3),
            F7::new(5),
            F7::new(6),
        ];
        check(berlekamp_massey(&s), &s);
        let s = vec![
            F7::new(1),
            F7::new(6),
            F7::new(2),
            F7::new(6),
            F7::new(6),
            F7::new(1),
            F7::new(6),
            F7::new(1),
            F7::new(6),
            F7::new(1),
            F7::new(6),
            F7::new(1),
        ];
        check(berlekamp_massey(&s), &s);
    }
}
