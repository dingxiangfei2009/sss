//! Serialization formatting and routines
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct GoppaDecoderRepr<F, T> {
    points: Vec<F>,
    g: Polynomial<F>,
}

#[derive(Serialize, Deserialize)]
pub struct GoppaEncoderRepr<F, T> {
    t: usize,
    parity_check: Vec<F>,
}

