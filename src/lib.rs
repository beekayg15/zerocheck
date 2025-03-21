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
    type ZeroCheckParams<'a>: Clone;

    /// Type of Input Parameters to the setup
    type InputParams: Clone;

    /// Type of transcripts to be used to record the interactions
    type Transcripts: Clone;

    /// function to setup the zerocheck protocol
    /// such as setting up the PCS and other
    /// equipments necessary
    fn setup<'a>(
        pp: &'a Self::InputParams
    ) -> Result<Self::ZeroCheckParams<'_>, Error>;

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
        zero_params: &'a Self::ZeroCheckParams<'_>,
        input_poly: &'a Self::InputType,
        zero_domain: &'a Self::ZeroDomain,
        transcript: &'a mut Self::Transcripts, 
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
        zero_params: &'a Self::ZeroCheckParams<'_>,
        input_poly: &'a Self::InputType,
        proof: &'a Self::Proof,
        zero_domain: &'a Self::ZeroDomain,
        transcript: &'a mut Self::Transcripts, 
    ) -> Result<bool, Error>;
}