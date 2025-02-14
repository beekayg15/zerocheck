use anyhow::Error;
use ark_ff::PrimeField;

/// Import zero check protocol for univariate 
/// polynomials verified using the quotient polynomial
pub mod univariate_zc;

/// Trait for the zero check protocol to prove that 
/// particular function evaluates to zero on a
/// given domain.
pub trait ZeroCheck<F: PrimeField>: Sized{
    /// Type by which input polynomials are provided.
    ///  eg. dense univariate/multilinear polynomials, evaluations
    type InputType: Clone;

    /// Type of the domain, over which the resultant 
    /// function evaluated to zero
    type ZeroDomain: Clone;

    /// Type of the proof used in the protocol
    type Proof: Clone;

    /// function called by the prover to genearte a valid
    /// proof for zero-check protocol
    /// 
    /// Attributes:
    /// g - input polynomial/evalution
    /// h - input polynomial/evalution
    /// zero_domain - domain over which the resulting polynomial evaluates to 0
    /// 
    /// Returns
    /// Proof - valid proof for the zero-check protocol
    fn prove<'a> (
        g: Self::InputType,
        h: Self::InputType,
        zero_domain: Self::ZeroDomain
    ) -> Result<Self::Proof, Error>;

    /// function called by the verifier to check if the proof for the 
    /// zero-check protocol is valid
    /// 
    /// Attributes:
    /// g - input polynomial/evalution
    /// h - input polynomial/evalution
    /// zero_domain - domain over which the resulting polynomial evaluates to 0
    /// proof - proof sent by the prover for the claim
    /// 
    /// Returns
    /// 'true' if the proof is valid, 'false' otherwise
    fn verify<'a> (
        g: Self::InputType,
        h: Self::InputType,
        proof: Self::Proof,
        zero_domain: Self::ZeroDomain
    ) -> Result<bool, Error>;
}

/// Testing the zero-check protocol
#[cfg(test)]
pub mod tests {

}