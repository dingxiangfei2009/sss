use alga::general::Field;

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

pub fn cooley_tukey<F>(
    n1: usize,
    n2: usize,
    transform1: impl Sync + Fn(Vec<F>, UnityRoot<F>) -> Vec<F>,
    transform2: impl Sync + Fn(Vec<F>, UnityRoot<F>) -> Vec<F>,
) -> impl Fn(Vec<F>, UnityRoot<F>) -> Vec<F>
where
    F: Field + Clone + Send + Sync,
{
    use ndarray::{ArrayViewMut, Axis};
    use rayon::prelude::*;

    move |mut x: Vec<F>, root: UnityRoot<F>| {
        assert_eq!(n1 * n2, x.len());
        assert_eq!(x.len(), root.order);
        let ur1 = root.clone().subgroup(n1);
        let ur2 = root.clone().subgroup(n2);

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
                let col_fft = transform1(col.to_owned().to_vec(), ur1.clone());
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
                    ur2.clone(),
                );
                row.iter_mut()
                    .zip(row_fft.into_iter())
                    .for_each(|(a, b)| *a = b);
            });
        x
    }
}
