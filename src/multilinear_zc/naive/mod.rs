use anyhow::Ok;
use ark_ec::pairing::Pairing;
use ark_ff::{FftField, PrimeField};
use sumcheck_protocol::{verifier::VerifierMsg, IPforSumCheck};
use std::marker::PhantomData;

mod data_structures;
pub use data_structures::*;

mod sumcheck_protocol;

use crate::{utils::get_randomness, ZeroCheck};

#[derive(Clone)]
pub struct NaiveMLZeroCheck<F: PrimeField + FftField, E: Pairing> {
    _field_data: PhantomData<F>,
    _pairing_data: PhantomData<E>
}

impl<F, _E> ZeroCheck<F, _E> for NaiveMLZeroCheck<F, _E>
    where
    F: PrimeField + FftField,
    _E: Pairing 
{
    type InputType = VirtualPolynomial<F>;
    
    // size of the boolean hypercube over which the output polynomial evaluates to 0
    type ZeroDomain = usize;

    type PCS = std::option::Option<F>;
    type Proof = Proof<F>;

    fn prove<'a> (
            input_poly: Self::InputType,
            zero_domain: Self::ZeroDomain
        ) -> Result<Self::Proof, anyhow::Error> {
            
        assert_eq!(
            input_poly.poly_info.num_vars, zero_domain, 
            "Dimensions of boolean hypercube do not match the given polynomials"
        );

        let mut prover_state = IPforSumCheck::prover_init(input_poly);
        let mut verifier_msg = None;
        let mut prover_msgs = vec![];
        let mut inp = vec![];

        for _ in 0..zero_domain {
            let prover_msg = IPforSumCheck::prover_round(
                &mut prover_state, 
                &verifier_msg
            );

            let mut seed = prover_state.challenges.clone();
            seed.extend(prover_msg.evaluations.clone());

            verifier_msg = Some(VerifierMsg{
                challenge: get_randomness(seed, inp.clone())[0]
            });

            inp.extend(prover_msg.evaluations.clone());
            prover_msgs.push(prover_msg);
        }

        Ok(Proof{
            prover_msgs: prover_msgs,
        })
    }

    fn verify<'a> (
            input_poly: Self::InputType,
            proof: Self::Proof,
            zero_domain: Self::ZeroDomain
        ) -> Result<bool, anyhow::Error> {

        assert_eq!(
            input_poly.poly_info.num_vars, zero_domain, 
            "Dimensions of boolean hypercube do not match the given polynomials"
        );

        let mut verifier_state = IPforSumCheck::verifier_init(
            input_poly.poly_info.clone()
        );

        for i in 0..zero_domain {
            let prover_msg = proof.prover_msgs.get(i).expect("proof is incomplete");
            let _verifier_msg = IPforSumCheck::verifier_round(
                (*prover_msg).clone(), 
                &mut verifier_state
            );
        }

        let subclaim = IPforSumCheck::check_n_generate_subclaim(
            verifier_state, 
            F::zero()
        ).unwrap();

        let lhs = subclaim.expected_evaluation;
        let rhs = input_poly.evaluate(subclaim.point);

        Ok(lhs == rhs)
    }
}