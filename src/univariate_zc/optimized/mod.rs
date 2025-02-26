// use std::marker::PhantomData;
// use anyhow::{Error, Ok};
// use ark_ff::{FftField, PrimeField};
// use ark_poly::{Evaluations, Radix2EvaluationDomain};
// use ark_ec::pairing::Pairing;

// use crate::ZeroCheck;

// pub mod data_structures;
// use data_structures::*;

// /// Optimized Zero-Check protocol for if a polynomial
// /// f = g^2*h evaluates to 0 over a specific domain
// /// using NTTs and INTTs
// /// 
// /// The inputs g and h are provided as evaluations
// /// and the proof that polynomial f computed as 
// /// mentioned above evaluates to zero, can be given
// /// by proving the existence of a quotient polynomial
// /// q, S.T. f(X) = q(X).z_H(X), where z_H(X) is the 
// /// vanishing polynomial over the zero domain H.
// pub struct OptimizedUnivariateZeroCheck<F: > {
//     _field_data: PhantomData<F>,
// }

// impl<F: FftField + PrimeField, E: Pairing> ZeroCheck<F, E> for OptimizedUnivariateZeroCheck<F> {
//     type InputType = Evaluations<F>;
//     type ZeroDomain = Radix2EvaluationDomain<F>;
//     type Proof = Proof<E>;

//     /// function called by the prover to genearte a valid
//     /// proof for zero-check protocol
//     /// 
//     /// Attributes:
//     /// g - input polynomial evalutions
//     /// h - input polynomial evalutions
//     /// zero_domain - domain over which the resulting polynomial evaluates to 0
//     /// 
//     /// Returns
//     /// Proof - valid proof for the zero-check protocol
//     fn prove<'a> (
//         input_poly: Vec<Self::InputType>,
//         zero_domain: Self::ZeroDomain
//     ) -> Result<Self::Proof, anyhow::Error> {
        
        

//         Ok(Proof{
//             q: g.interpolate(),
//         })
//     }

//     /// function called by the verifier to check if the proof for the 
//     /// zero-check protocol is valid
//     /// 
//     /// Attributes:
//     /// g - input polynomial evalutions
//     /// h - input polynomial evalutions
//     /// zero_domain - domain over which the resulting polynomial evaluates to 0
//     /// proof - proof sent by the prover for the claim
//     /// 
//     /// Returns
//     /// 'true' if the proof is valid, 'false' otherwise
//     fn verify<'a> (
//         _input_poly: Vec<Self::InputType>,
//         _proof: Self::Proof,
//         _zero_domain: Self::ZeroDomain
//     ) -> Result<bool, anyhow::Error> {
        
//         Ok(false)
//     }
// }