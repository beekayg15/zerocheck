use std::marker::PhantomData;

use ark_ec::AffineRepr;

pub struct Hyrax<G: AffineRepr> {
    _affine_data: PhantomData<G>,
}