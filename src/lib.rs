use anyhow::Error;
use ark_ff::PrimeField;

pub mod univariate_zc;
pub trait ZeroCheck<F: PrimeField>: Sized{
    type InputType: Clone;
    type ZeroDomain: Clone;
    type Proof: Clone;

    fn prove<'a> (
        g: Self::InputType,
        h: Self::InputType,
        zero_domain: Self::ZeroDomain
    ) -> Result<Self::Proof, Error>;

    fn verify<'a> (
        g: Self::InputType,
        h: Self::InputType,
        proof: Self::Proof,
        zero_domain: Self::ZeroDomain
    ) -> Result<bool, Error>;
}