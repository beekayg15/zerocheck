use anyhow::Ok;
use ark_ff::FftField;
use ark_ff::PrimeField;
use ark_poly::EvaluationDomain;
use ark_poly::{univariate::DensePolynomial, Evaluations, GeneralEvaluationDomain, Polynomial};
use ark_std::{end_timer, start_timer};
use rayon::prelude::*;
use std::marker::PhantomData;

pub mod data_structures;
use data_structures::*;

use crate::pcs::PolynomialCommitmentScheme;
use crate::transcripts::ZCTranscript;
use crate::ZeroCheck;

pub mod parser;

/// Optimized Zero-Check protocol for if a polynomial
/// f = g*h*s + (1 - s)(g + h) - o evaluates to 0
/// over a specific domain using NTTs and INTTs
///
/// The inputs g, h and s are provided as evaluations
/// and the proof that polynomial f computed as
/// mentioned above evaluates to zero, can be given
/// by proving the existence of a quotient polynomial
/// q, S.T. f(X) = q(X).z_H(X), where z_H(X) is the
/// vanishing polynomial over the zero domain H.
pub struct CustomUnivariateZeroCheck<F, PCS> {
    _field_data: PhantomData<F>,
    _pcs_data: PhantomData<PCS>,
}

impl<F, PCS> ZeroCheck<F> for CustomUnivariateZeroCheck<F, PCS>
where
    F: PrimeField + FftField,
    PCS: PolynomialCommitmentScheme<
        Polynomial = DensePolynomial<F>,
        PolynomialInput = F,
        PolynomialOutput = F,
    >,
{
    type InputType = VirtualEvaluation<F>;
    type ZeroDomain = GeneralEvaluationDomain<F>;
    type Proof = Proof<PCS>;
    type ZeroCheckParams<'a> = ZeroCheckParams<'a, PCS>;
    type InputParams = PCS::PCSParams;
    type Transcripts = ZCTranscript<F>;

    fn setup(pp: &Self::InputParams) -> Result<Self::ZeroCheckParams<'_>, anyhow::Error> {
        let setup_kzg_time =
            start_timer!(|| "Setup KZG10 polynomial commitments global parameters");

        let (ck, vk) = PCS::setup(pp).unwrap();

        end_timer!(setup_kzg_time);

        Ok(ZeroCheckParams { ck, vk })
    }

    /// function called by the prover to genearte a valid
    /// proof for zero-check protocol
    ///
    /// Attributes:
    /// g - input polynomial evalutions
    /// h - input polynomial evalutions
    /// s - input polynomial evalutions
    /// o - input polynomial evalutions
    /// zero_domain - domain over which the resulting polynomial evaluates to 0
    ///
    /// Returns
    /// Proof - valid proof for the zero-check protocol
    fn prove<'a>(
        zero_params: &Self::ZeroCheckParams<'_>,
        input_poly: &Self::InputType,
        zero_domain: &Self::ZeroDomain,
        transcript: &mut Self::Transcripts,
        run_threads: Option<usize>,
        batch_commit_threads: Option<usize>,
        batch_open_threads: Option<usize>,
    ) -> Result<Self::Proof, anyhow::Error> {
        let run_threads = run_threads.unwrap_or(1);
        let batch_commit_threads = batch_commit_threads.unwrap_or(1);
        let batch_open_threads = batch_open_threads.unwrap_or(1);
        let pool_run = rayon::ThreadPoolBuilder::new()
            .num_threads(run_threads)
            .build()
            .unwrap();
        let pool_commit = rayon::ThreadPoolBuilder::new()
            .num_threads(batch_commit_threads)
            .build()
            .unwrap();
        let pool_open = rayon::ThreadPoolBuilder::new()
            .num_threads(batch_open_threads)
            .build()
            .unwrap();

        // compute the vanishing polynomial of the zero domain
        // let z_poly = zero_domain.vanishing_polynomial();
        let z_deg = zero_domain.size();

        let prove_time = start_timer!(|| format!(
            "OptimizedUnivariateZeroCheck::prove, with {z_deg} zero_domain size."
        ));

        let ifft_time = start_timer!(|| "IFFT for g,h,s,o from evaluations to coefficients");

        // compute the polynomials corresponding to g, h, and s using interpolation (IFFT)
        let virtual_polynomial: VirtualPolynomail<F> =
            VirtualPolynomail::new(input_poly.clone(), Some(&pool_run));

        end_timer!(ifft_time);
        let coset_time = start_timer!(|| "Compute coset domain");

        // compute degree of quotient polynomial to
        let f_deg = virtual_polynomial.degree();
        let q_deg = f_deg - z_deg;

        // Compute the coset domain to interpolate q(X)
        let q_domain = GeneralEvaluationDomain::<F>::new(q_deg).unwrap();
        let offset = F::GENERATOR;
        let coset_domain: GeneralEvaluationDomain<F> = q_domain.get_coset(offset).unwrap();

        end_timer!(coset_time);
        let coset_eval_time =
            start_timer!(|| "FFT Compute g,h,s,o,z,q evaluations over coset domain");

        let f_evals = virtual_polynomial.evaluate_over_domain(coset_domain, Some(&pool_run));

        // Find the value of q(X) over the coset domain using threads
        let z_evals = pool_run.install(|| {
            zero_domain
                .vanishing_polynomial()
                .evaluate_over_domain(coset_domain)
                .evals
        });

        // Compute q_evals as f_evals / z_evals
        let q_evals: Vec<F> = f_evals
            .iter()
            .zip(z_evals.iter())
            .map(|(f_eval, z_eval)| {
                if *z_eval == F::zero() {
                    panic!("Zero domain evaluation is zero, cannot compute quotient polynomial");
                }
                *f_eval / *z_eval
            })
            .collect();

        end_timer!(coset_eval_time);
        let ifft_q_time = start_timer!(|| "IFFT for q from evaluations to coefficients");

        // Interpolate q(X) using the evaluations
        let q_coeff = pool_run
            .install(|| Evaluations::from_vec_and_domain(q_evals, coset_domain).interpolate());

        end_timer!(ifft_q_time);
        let commit_time = start_timer!(|| "Commit to (g,h,s,o) polynomials");

        let flatten_polynomials: Vec<DensePolynomial<F>> = virtual_polynomial
            .clone()
            .univariate_polynomials
            .into_par_iter()
            .map(|mle| (*mle.as_ref()).clone())
            .collect();

        // Use the pool_commit thread pool to perform the batch_commit operation
        let input_polynomial_comms = pool_commit
            .install(|| PCS::batch_commit(&zero_params.ck, &flatten_polynomials).unwrap());
        end_timer!(commit_time);
        let commit_q_time = start_timer!(|| "Commit to (q) polynomial");

        let comm_q = pool_commit.install(|| PCS::commit(&zero_params.ck, &q_coeff).unwrap());
        end_timer!(commit_q_time);

        let get_r_eval_time = start_timer!(|| "computing evaluations at challenge, get challenge");

        // Sample a random challenge - Fiat-Shamir
        for comm in input_polynomial_comms.iter() {
            transcript
                .append_serializable_element(
                    b"comm_inp",
                    &PCS::extract_pure_commitment(comm).unwrap(),
                )
                .unwrap();
        }
        // Append q commitment before sampling the challenge to keep transcript in sync with verifier
        transcript
            .append_serializable_element(b"comm_q", &PCS::extract_pure_commitment(&comm_q).unwrap())
            .unwrap();

        let r = transcript.get_and_append_challenge(b"sampling r").unwrap();

        // Collect the evalution of the input polynomials at the challenge
        let mut inp_evals_at_rand = vec![];
        for poly in flatten_polynomials.iter() {
            inp_evals_at_rand.push(poly.evaluate(&r));
        }
        let q_eval_r = q_coeff.evaluate(&r);

        end_timer!(get_r_eval_time);
        let open_time = start_timer!(|| "Batch open the g,h,s,o poly commits at r");

        let mut batch_polynomials = flatten_polynomials.clone();
        let mut batch_comms = input_polynomial_comms.clone();
        batch_polynomials.push(q_coeff.clone());
        batch_comms.push(comm_q.clone());

        // Generate the opening proof that g(r), h(r), s(r), and o(r) are the evaluations of the polynomials
        let mut opening_proofs = pool_open.install(|| {
            PCS::batch_open(&zero_params.ck, &batch_comms, &batch_polynomials, r).unwrap()
        });

        let q_opening_proof = opening_proofs[opening_proofs.len() - 1].clone();
        opening_proofs.pop();

        end_timer!(open_time);
        end_timer!(prove_time);

        // Send the proof with the necessary commitments and opening proofs
        Ok(Proof {
            q_comm: comm_q,
            inp_comms: input_polynomial_comms,
            inp_evals: inp_evals_at_rand,
            inp_openings: opening_proofs,
            q_eval: q_eval_r,
            q_opening: q_opening_proof,
        })
    }

    /// function called by the verifier to check if the proof for the
    /// zero-check protocol is valid
    ///
    /// Attributes:
    /// g - input polynomial evaluations
    /// h - input polynomial evaluations
    /// s - input polynomial evaluations
    /// zero_domain - domain over which the resulting polynomial evaluates to 0
    /// proof - proof sent by the prover for the claim
    ///
    /// Returns
    /// 'true' if the proof is valid, 'false' otherwise
    fn verify<'a>(
        zero_params: &Self::ZeroCheckParams<'_>,
        input_poly: &Self::InputType,
        proof: &Self::Proof,
        zero_domain: &Self::ZeroDomain,
        transcript: &'a mut Self::Transcripts,
    ) -> Result<bool, anyhow::Error> {
        let q_comm = &proof.q_comm;
        let inp_comms = &proof.inp_comms;
        let vk = &zero_params.vk;
        let inp_openings = &proof.inp_openings;
        let inp_evals = &proof.inp_evals;
        let q_opening = &proof.q_opening;
        let q_eval = proof.q_eval;

        // Sample a random challenge - Fiat-Shamir
        for comm in inp_comms.iter() {
            transcript
                .append_serializable_element(
                    b"comm_inp",
                    &PCS::extract_pure_commitment(&comm).unwrap(),
                )
                .unwrap();
        }
        // Append the commitment to q(X) to the transcript
        transcript
            .append_serializable_element(b"comm_q", &PCS::extract_pure_commitment(q_comm).unwrap())
            .unwrap();
        let r = transcript.get_and_append_challenge(b"sampling r").unwrap();

        let mut comms = inp_comms.clone();
        comms.push(q_comm.clone());

        let mut evals = inp_evals.clone();
        evals.push(q_eval);

        let mut openings = inp_openings.clone();
        openings.push(q_opening.clone());

        // check openings to all polynomials
        assert!(
            PCS::batch_check(&vk, &openings, &comms, r, evals).unwrap(),
            "Opening failed at input polynomials"
        );

        // check if q(r) * z_H(r) = g(r).h(r).s(r) + (1 - s(r))(g(r) + h(r))
        let lhs = q_eval * zero_domain.evaluate_vanishing_polynomial(r);
        let rhs = input_poly.evaluate_at_point(r);

        Ok(lhs == rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::pcs::univariate_pcs::kzg::KZG;
    use crate::pcs::univariate_pcs::ligero::Ligero;
    use crate::zc::univariate_zc::custom::parser::prepare_zero_virtual_evaluation_from_string;

    use super::*;
    use ark_bls12_381::Bls12_381;
    use ark_bls12_381::Fr;
    use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
    use ark_std::end_timer;
    use ark_std::start_timer;

    #[test]
    // This function tests the proof and verification combined with parser
    fn test_proof_generation_verification_with_parser() {
        let test_timer = start_timer!(|| "Proof Generation Test");
        let pool_prepare = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        let degree = 1 << 6;
        let input = "g*h*s + (1 - s)*(g + h)";
        let inp_evals =
            prepare_zero_virtual_evaluation_from_string(input, degree, &pool_prepare).unwrap();
        let domain = GeneralEvaluationDomain::<Fr>::new(degree).unwrap();

        let proof_gen_timer = start_timer!(|| "Prove fn called for g, h, zero_domain");

        let max_degree = inp_evals.evals_info.max_multiplicand * degree;

        let zp = CustomUnivariateZeroCheck::<Fr, KZG<Bls12_381>>::setup(&max_degree).unwrap();

        let proof = CustomUnivariateZeroCheck::<Fr, KZG<Bls12_381>>::prove(
            &zp.clone(),
            &inp_evals,
            &domain,
            &mut ZCTranscript::init_transcript(),
            None,
            None,
            None,
        )
        .unwrap();

        end_timer!(proof_gen_timer);

        let verify_timer = start_timer!(|| "Verify fn called for g, h, zero_domain, proof");

        let result = CustomUnivariateZeroCheck::<Fr, KZG<Bls12_381>>::verify(
            &zp,
            &inp_evals,
            &proof,
            &domain,
            &mut ZCTranscript::init_transcript(),
        )
        .unwrap();

        end_timer!(verify_timer);
        assert_eq!(result, true);

        end_timer!(test_timer);
    }

    #[test]
    fn test_proof_generation_verification_custom_uni() {
        let test_timer = start_timer!(|| "Proof Generation Test");
        let pool_prepare = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        let degree = 1 << 6;
        let inp_evals =
            custom_zero_test_case_with_products::<Fr>(degree, 3, vec![3, 2, 2], &pool_prepare);
        let domain = GeneralEvaluationDomain::<Fr>::new(degree).unwrap();

        let proof_gen_timer = start_timer!(|| "Prove fn called for g, h, zero_domain");

        let max_degree = 3 * degree;

        let zp = CustomUnivariateZeroCheck::<Fr, KZG<Bls12_381>>::setup(&max_degree).unwrap();

        let proof = CustomUnivariateZeroCheck::<Fr, KZG<Bls12_381>>::prove(
            &zp.clone(),
            &inp_evals,
            &domain,
            &mut ZCTranscript::init_transcript(),
            None,
            None,
            None,
        )
        .unwrap();

        end_timer!(proof_gen_timer);

        let verify_timer = start_timer!(|| "Verify fn called for g, h, zero_domain, proof");

        let result = CustomUnivariateZeroCheck::<Fr, KZG<Bls12_381>>::verify(
            &zp,
            &inp_evals,
            &proof,
            &domain,
            &mut ZCTranscript::init_transcript(),
        )
        .unwrap();

        end_timer!(verify_timer);
        assert_eq!(result, true);

        end_timer!(test_timer);
    }

    #[test]
    fn test_proof_generation_verification_custom_uni_ligero() {
        let test_timer = start_timer!(|| "Proof Generation Test");
        let pool_prepare = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        let degree = 1 << 5;
        let domain = GeneralEvaluationDomain::<Fr>::new(degree).unwrap();
        let inp_evals =
            custom_zero_test_case_with_products::<Fr>(degree, 3, vec![3, 2, 2], &pool_prepare);

        let proof_gen_timer = start_timer!(|| "Prove fn called for g, h, zero_domain");

        let max_degree = 3 * degree;
        let pp = max_degree;

        let zp = CustomUnivariateZeroCheck::<Fr, Ligero<Fr>>::setup(&pp).unwrap();

        let proof = CustomUnivariateZeroCheck::<Fr, Ligero<Fr>>::prove(
            &zp.clone(),
            &inp_evals,
            &domain,
            &mut ZCTranscript::init_transcript(),
            None,
            None,
            None,
        )
        .unwrap();

        end_timer!(proof_gen_timer);

        let verify_timer = start_timer!(|| "Verify fn called for g, h, zero_domain, proof");

        let result = CustomUnivariateZeroCheck::<Fr, Ligero<Fr>>::verify(
            &zp,
            &inp_evals,
            &proof,
            &domain,
            &mut ZCTranscript::init_transcript(),
        )
        .unwrap();

        end_timer!(verify_timer);
        assert_eq!(result, true);

        end_timer!(test_timer);
    }
}
