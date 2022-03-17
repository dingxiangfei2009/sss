use super::*;

use crate::{int_inj, tests::Frac};
use quickcheck::{quickcheck, TestResult};

#[test]
fn taylor_exp() {
    let p = Polynomial(vec![Frac::one(), -int_inj::<Frac, _>(2), Frac::one()]);
    let d = Polynomial(vec![-Frac::one(), Frac::one()]);
    assert_eq!(
        taylor_expansion(p, d),
        vec![
            Polynomial(vec![Frac::zero()]),
            Polynomial(vec![Frac::zero()]),
            Polynomial(vec![Frac::one()]),
        ]
    );
    let p = Polynomial(vec![Frac::zero(), -int_inj::<Frac, _>(2), Frac::one()]);
    let d = Polynomial(vec![-Frac::one(), Frac::one()]);
    assert_eq!(
        taylor_expansion(p, d),
        vec![
            Polynomial(vec![-Frac::one()]),
            Polynomial(vec![Frac::zero()]),
            Polynomial(vec![Frac::one()]),
        ]
    );
    let p: Vec<Frac> = (0..9).map(|x| int_inj(x)).collect();
    let p = Polynomial::new(p);
    let d: Vec<Frac> = [1, 2, 1, 4].iter().map(|&x| int_inj(x)).collect();
    let d = Polynomial::new(d);
    let expected = vec![
        Polynomial(vec![
            -int_inj::<Frac, _>(533) / int_inj::<Frac, _>(1024),
            -int_inj::<Frac, _>(315) / int_inj::<Frac, _>(512),
            int_inj::<Frac, _>(259) / int_inj::<Frac, _>(1024),
        ]),
        Polynomial(vec![
            int_inj::<Frac, _>(789) / int_inj::<Frac, _>(1024),
            int_inj::<Frac, _>(227) / int_inj::<Frac, _>(256),
            -int_inj::<Frac, _>(35) / int_inj::<Frac, _>(64),
        ]),
        Polynomial(vec![
            -Frac::one() / int_inj::<Frac, _>(4),
            int_inj::<Frac, _>(3) / int_inj::<Frac, _>(16),
            Frac::one() / int_inj::<Frac, _>(2),
        ]),
    ];
    assert_eq!(taylor_expansion(p, d), expected);
}

fn u16_to_gf65536n(xs: Vec<u16>) -> Vec<GF65536N> {
    let bases = GF65536NTower::basis_elements_over_bottom();
    xs.into_iter()
        .map(|mut x| {
            let mut y = GF65536N::zero();
            let mut bases = bases.iter();
            while x > 0 {
                let basis = bases.next().unwrap();
                if x & 1 > 0 {
                    y += basis.clone();
                }
                x >>= 1;
            }
            y
        })
        .collect()
}

fn vz_gathen_prop(points: Vec<GF65536N>, f: Polynomial<GF65536N>) {
    let result = MEVZG_GF65536N.eval(f.clone(), points.clone());
    assert_eq!(result.len(), points.len());
    for (r, p) in result.iter().zip(&points) {
        assert_eq!(r, &f.clone().eval_at(p.clone()).1);
    }
}

quickcheck! {
    fn vz_gathen(points: Vec<u16>, f: Vec<u16>) -> TestResult {
        let mut points = points;
        points.sort();
        points.dedup();
        if points.len() > u16::MAX as usize {
            return TestResult::discard();
        }
        if f.len() > u16::MAX as usize {
            return TestResult::discard();
        }
        let points = u16_to_gf65536n(points);
        let f = Polynomial::new(u16_to_gf65536n(f));
        vz_gathen_prop(points, f);
        TestResult::passed()
    }
}

#[test]
fn cyclotomic() {
    assert_eq!(
        Polynomial(
            [1, -1, 0, 1, -1, 1, 0, -1, 1]
                .iter()
                .map(|&x| -> Frac { int_inj(x) })
                .collect()
        ),
        build_cyclotomic_polynomial::<Frac>(15)
    );
    assert_eq!(
        Polynomial(
            [1, 0, -1, 0, 1, 0, -1, 0, 1, 0, -1, 0, 1,]
                .iter()
                .map(|&x| -> Frac { int_inj(x) })
                .collect()
        ),
        build_cyclotomic_polynomial::<Frac>(28)
    );
    assert_eq!(
        Polynomial(
            [
                1, 1, 1, 0, 0, -1, -1, -2, -1, -1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, -1, 0, -1, 0, -1,
                0, -1, 0, -1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, -1, -1, -2, -1, -1, 0, 0, 1, 1, 1,
            ]
            .iter()
            .map(|&x| -> Frac { int_inj(x) })
            .collect(),
        ),
        build_cyclotomic_polynomial::<Frac>(105)
    );
}
