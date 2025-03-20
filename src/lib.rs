/// Import transcripts structure
pub mod transcripts;

/// Import utility functions
pub mod utils; 

/// Import polynomial commitment schemes
pub mod pcs;

/// Import implemented zerocheck protocols
pub mod zc;

use anyhow::Error;
use ark_ff::PrimeField;

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

    /// Type of Zero-check Parameters used in the protocol
    type ZeroCheckParams: Clone;

    /// Type of Input Parameters to the setup
    type InputParams: Clone;

    /// function to setup the zerocheck protocol
    /// such as setting up the PCS and other
    /// equipments necessary
    fn setup<'a>(
        pp: Self::InputParams
    ) -> Result<Self::ZeroCheckParams, Error>;

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
        zero_params: Self::ZeroCheckParams,
        input_poly: Self::InputType,
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
        zero_params: Self::ZeroCheckParams,
        input_poly: Self::InputType,
        proof: Self::Proof,
        zero_domain: Self::ZeroDomain
    ) -> Result<bool, Error>;
}