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

        let mut init_seed = vec![];
        for (coeff, _) in input_poly.products.clone() {
            init_seed.push(coeff);
        }
        
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

        let inp_hat = input_poly.build_f_hat(&r_point).unwrap();

        let mut prover_state = IPforSumCheck::prover_init(inp_hat);
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

        let mut init_seed = vec![];
        for (coeff, _) in input_poly.products.clone() {
            init_seed.push(coeff);
        }
        
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

        let inp_hat = input_poly.build_f_hat(&r_point).unwrap();

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
        let rhs = inp_hat.evaluate(subclaim.point);

        let result: bool = lhs == rhs;
        Ok(result)
    }
}

#[cfg(test)]
mod test {
    use ark_bls12_381::Fr;
    use ark_bls12_381::Bls12_381;

    use crate::ZeroCheck;

    use super::{rand_zero, NaiveMLZeroCheck};

    #[test]
    fn test_ml_zerocheck() {
        let poly = rand_zero::<Fr>(10, (4, 5), 2);
        let proof = NaiveMLZeroCheck::<Fr, Bls12_381>::prove(poly.clone(), 10).unwrap();
        println!("Proof Generated: {:?}", proof);

        let valid = NaiveMLZeroCheck::<Fr, Bls12_381>::verify(
            poly, 
            proof, 
            10
        ).unwrap();

        assert!(valid);
    }
}