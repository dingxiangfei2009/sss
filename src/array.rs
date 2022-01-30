use std::mem::MaybeUninit;

use ndarray::{Array, ArrayView, Axis, ErrorKind, RemoveAxis, ShapeError, Zip};

/// A variant of `ndarray::stack!` macro that accepts a borrowed `ArrayView`
#[macro_export]
macro_rules! stack {
    ($axis:expr, $( $array:expr ),+ ) => {
        $crate::array::stack($axis, &[ $(::ndarray::ArrayView::from(&$array) ),* ]).unwrap()
    }
}

pub fn stack<'a, A, D>(
    axis: Axis,
    arrays: &[ArrayView<'a, A, D>],
) -> Result<Array<A, D>, ShapeError>
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
        let arrays = arrays.iter().map(|a| a.mapv(MaybeUninit::new));
        for mut array in arrays {
            let len = array.len_of(axis);
            let (mut front, rest) = assign_view.split_at(axis, len);
            Zip::from(&mut front).and(&mut array).for_each(|dst, src|
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
