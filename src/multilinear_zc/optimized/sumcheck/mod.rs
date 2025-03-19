use ark_ff::PrimeField;
use std::marker::PhantomData;

pub mod prover;
pub mod verifier;

/// Struct to implement the Interactive Protocol for
/// Sum Check. i.e., a function summed over the given
/// boolean hypercube evaluates to a claimed sum
pub struct IPforSumCheck<F: PrimeField> {
    _marker: PhantomData<F>,
}