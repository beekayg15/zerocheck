use anyhow::Ok;
use ark_ff::{FftField, PrimeField};
use ark_std::{start_timer, end_timer};
use sumcheck_protocol::{verifier::VerifierMsg, IPforSumCheck};
use std::marker::PhantomData;

mod data_structures;
pub use data_structures::*;

mod sumcheck_protocol;

use crate::{utils::get_randomness, ZeroCheck};

/// Optimized Zero-Check protocol for if a polynomial
/// f = sum(product(MLEs)) evaluates to 0
/// over a boolean hypercube of given dimensions
#[derive(Clone)]
pub struct NaiveMLZeroCheck<F: PrimeField + FftField> {
    _field_data: PhantomData<F>,
}

impl<F> ZeroCheck<F> for NaiveMLZeroCheck<F>
    where
    F: PrimeField + FftField,
{
    type InputType = VirtualPolynomial<F>;
    
    // size of the boolean hypercube over which the output polynomial evaluates to 0
    type ZeroDomain = usize;
    type Proof = Proof<F>;
    type ZeroCheckParams = ZeroCheckParams<F>;
    type InputParams = Option<F>;

    fn setup<'a>(
        _pp: Self::InputParams
    ) -> Result<Self::ZeroCheckParams, anyhow::Error> {
        Ok(ZeroCheckParams { 
            _field_data: PhantomData::<F>,
        })
    }

    /// function called by the prover to genearte a valid
    /// proof for zero-check protocol
    ///
    /// Attributes:
    /// input_poly - input polynomial evalutions as a virtual polynomial
    /// zero_domain - number of dimensions of the hypercube over which
    /// output polynomial evaluates to 0
    ///
    /// Returns
    /// Proof - valid proof for the zero-check protocol
    fn prove<'a> (
            _zero_params: Self::ZeroCheckParams,
            input_poly: Self::InputType,
            zero_domain: Self::ZeroDomain
        ) -> Result<Self::Proof, anyhow::Error> {
            
        assert_eq!(
            input_poly.poly_info.num_vars, zero_domain, 
            "Dimensions of boolean hypercube do not match the given polynomials"
        );

        // set up the seed and input required to generate the initial random challenge
        let initial_challenge_timer = start_timer!(|| 
            "computing inital challenge using which f_hat is computed"
        );
        
        let mut init_seed = input_poly.flat_ml_extensions[0].evaluations.clone();
        
        let mut init_inp = vec![];
        for mle in input_poly.flat_ml_extensions.clone() {
            init_inp.extend(mle.evaluations.clone());
        }


        let mut r_point = vec![];
        for _ in 0..zero_domain {
            let r = get_randomness(init_seed.clone(), init_inp.clone())[0];
            r_point.push(r);
            init_seed.push(r);
        }

        end_timer!(initial_challenge_timer);

        // compute f_hat(X) = sum_{B^m} f(X).eq(X, r)
        let compute_f_hat_timer = start_timer!(|| 
            "computing f_hat(X) = sum_{B^m} f(X).eq(X, r)"
        );

        let inp_hat = input_poly.build_f_hat(&r_point).unwrap();

        end_timer!(compute_f_hat_timer);

        // Run sumcheck proving algorithm for #num_var rounds
        let sumcheck_prover_timer = start_timer!(|| format!(
            "running sumcheck proving algorithm for {:?} rounds", zero_domain
        ));

        let mut prover_state = IPforSumCheck::prover_init(inp_hat);
        let mut verifier_msg = None;
        let mut prover_msgs = vec![];
        let mut inp = vec![];

        for _ in 0..zero_domain {
            let prover_msg = IPforSumCheck::prover_round(
                &mut prover_state, 
                &verifier_msg
            );

            // compute the seed and input required to generate the random challenge
            let verifier_challenge_sampling_timer = start_timer!(|| 
                "verifier sampling a random challenge using the transcripts"
            );

            let mut seed = prover_state.challenges.clone();
            seed.extend(prover_msg.evaluations.clone());

            end_timer!(verifier_challenge_sampling_timer);

            verifier_msg = Some(VerifierMsg{
                challenge: get_randomness(seed, inp.clone())[0]
            });

            inp.extend(prover_msg.evaluations.clone());
            prover_msgs.push(prover_msg);
        }

        end_timer!(sumcheck_prover_timer);

        Ok(Proof{
            prover_msgs: prover_msgs,
        })
    }

    /// function called by the verifier to check if the proof for the
    /// zero-check protocol is valid
    ///
    /// Attributes:
    /// input_poly - input polynomial evalutions as a virtual polynomial
    /// zero_domain - number of dimensions of the hypercube over which
    /// output polynomial evaluates to 0
    /// proof - proof sent by the prover for the claim
    ///
    /// Returns
    /// 'true' if the proof is valid, 'false' otherwise
    fn verify<'a> (
            _zero_params: Self::ZeroCheckParams,
            input_poly: Self::InputType,
            proof: Self::Proof,
            zero_domain: Self::ZeroDomain
        ) -> Result<bool, anyhow::Error> {

        // check if the zero domain (dimensions of boolean hypercube) 
        // is same as the number of variables
        assert_eq!(
            input_poly.poly_info.num_vars, zero_domain, 
            "Dimensions of boolean hypercube do not match the given polynomials"
        );

        // set up the seed and input required to generate the initial random challenge
        let initial_challenge_timer = start_timer!(|| 
            "computing inital challenge using which f_hat is computed"
        );
        let mut init_inp = vec![];
        for mle in input_poly.flat_ml_extensions.clone() {
            init_inp.extend(mle.evaluations.clone());
        }

        let mut init_seed = input_poly.flat_ml_extensions[0].evaluations.clone();

        let mut r_point = vec![];
        for _ in 0..zero_domain {
            let r = get_randomness(init_seed.clone(), init_inp.clone())[0];
            r_point.push(r);
            init_seed.push(r);
        }

        end_timer!(initial_challenge_timer);

        // compute f_hat(X) = sum_{B^m} f(X).eq(X, r)
        let compute_f_hat_timer = start_timer!(|| 
            "computing f_hat(X) = sum_{B^m} f(X).eq(X, r)"
        );

        let inp_hat = input_poly.build_f_hat(&r_point).unwrap();

        end_timer!(compute_f_hat_timer);

        // Run sumcheck verifier algorithm for #num_var rounds
        let sumcheck_verifier_timer = start_timer!(|| format!(
            "running sumcheck proving algorithm for {:?} rounds", zero_domain
        ));

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

        end_timer!(sumcheck_verifier_timer);

        let lhs = subclaim.expected_evaluation;
        let rhs = inp_hat.evaluate(subclaim.point);

        // check if the virtual polynomial evaluates to the 
        // given value over the sampled challenges
        let result: bool = lhs == rhs;
        Ok(result)
    }
}

#[cfg(test)]
mod test {
    use ark_bls12_381::Fr;

    use crate::ZeroCheck;

    use super::{rand_zero, NaiveMLZeroCheck};

    #[test]
    fn test_ml_zerocheck() {
        let poly = rand_zero::<Fr>(10, (4, 5), 2);
        let zp = NaiveMLZeroCheck::<Fr>::setup(None).unwrap();

        let proof = NaiveMLZeroCheck::<Fr>::prove(
            zp.clone(),
            poly.clone(), 
            10
        ).unwrap();
        println!("Proof Generated: {:?}", proof);

        let valid = NaiveMLZeroCheck::<Fr>::verify(
            zp, 
            poly, 
            proof, 
            10
        ).unwrap();

        assert!(valid);
    }
}