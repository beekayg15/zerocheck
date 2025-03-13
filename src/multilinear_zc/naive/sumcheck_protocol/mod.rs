use ark_ff::PrimeField;
use std::marker::PhantomData;

pub mod prover;
pub mod verifier;

pub struct IPforSumCheck<F: PrimeField> {
    _marker: PhantomData<F>,
}