use anyhow::Ok;
use ark_ff::PrimeField;
use ark_poly::{DenseMultilinearExtension, Polynomial};
use ark_std::{end_timer, start_timer};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::marker::PhantomData;
use sumcheck::{verifier::VerifierMsg, IPforSumCheck};

mod data_structures;
pub use data_structures::*;

mod sumcheck;

use crate::{pcs::PolynomialCommitmentScheme, transcripts::ZCTranscript, ZeroCheck};

/// Optimized Zero-Check protocol for if a polynomial
/// f = sum(product(MLEs)) evaluates to 0
/// over a boolean hypercube of given dimensions
#[derive(Clone)]
pub struct OptMLZeroCheck<F: PrimeField, PCS: PolynomialCommitmentScheme> {
    _field_data: PhantomData<F>,
    _pcs_data: PhantomData<PCS>,
}

impl<F, PCS> ZeroCheck<F> for OptMLZeroCheck<F, PCS>
where
    F: PrimeField,
    PCS: PolynomialCommitmentScheme<
        Polynomial = DenseMultilinearExtension<F>,
        PolynomialInput = Vec<F>,
        PolynomialOutput = F,
    >,
{
    type InputType = VirtualPolynomial<F>;

    // size of the boolean hypercube over which the output polynomial evaluates to 0
    type ZeroDomain = usize;
    type Proof = Proof<PCS>;
    type ZeroCheckParams<'a> = ZeroCheckParams<'a, PCS>;
    type InputParams = PCS::PCSParams;
    type Transcripts = ZCTranscript<F>;

    fn setup<'a>(pp: &Self::InputParams) -> Result<Self::ZeroCheckParams<'_>, anyhow::Error> {
        let setup_mpc_time = start_timer!(|| "Setup MPC polynomial commitments global parameters");

        let (ck, vk) = PCS::setup(pp).unwrap();

        end_timer!(setup_mpc_time);

        Ok(ZeroCheckParams { vk, ck })
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

        let prover_start =
            start_timer!(|| format!("Prover starts Opt multilinear for 2^{:?}.", zero_domain));
        // compute the commitments to the MLEs in the Virtual Polynomial
        let inp_commitment_timer = start_timer!(|| "commit to (g,h,s,o) input MLEs");

        let flatten_mle_extensions: Vec<DenseMultilinearExtension<_>> = input_poly
            .clone()
            .flat_ml_extensions
            .into_par_iter()
            .map(|mle| (*mle.as_ref()).clone())
            .collect();

        let inp_commitments = pool_commit
            .install(|| PCS::batch_commit(&zero_params.ck, &flatten_mle_extensions))
            .unwrap();

        end_timer!(inp_commitment_timer);

        assert_eq!(
            input_poly.poly_info.num_vars, *zero_domain,
            "Dimensions of boolean hypercube do not match the given polynomials"
        );

        // set up the seed and input required to generate the initial random challenge
        let initial_challenge_timer =
            start_timer!(|| "computing inital challenge using which f_hat is computed");

        let _: Vec<_> = inp_commitments
            .clone()
            .into_iter()
            .map(|comm| {
                transcript
                    .append_serializable_element(b"comm_mle", &comm)
                    .unwrap();
            })
            .collect();

        let r_point: Vec<F> = (0..*zero_domain)
            .into_iter()
            .map(|_| {
                let r = transcript
                    .get_and_append_challenge(b"init_challenge")
                    .unwrap();
                r
            })
            .collect();

        end_timer!(initial_challenge_timer);

        // compute f_hat(X) = sum_{B^m} f(X).eq(X, r)
        let compute_f_hat_timer =
            start_timer!(|| "Build MLE: computing f_hat(X) = sum_{B^m} f(X).eq(X, r)");

        let inp_hat = input_poly.build_f_hat(&r_point).unwrap();

        end_timer!(compute_f_hat_timer);

        // Run sumcheck proving algorithm for #num_var rounds using pool_run threads
        let sumcheck_prover_timer =
            start_timer!(|| format!("running sumcheck proving algorithm for X rounds"));

        let mut prover_state = IPforSumCheck::prover_init(inp_hat);
        let mut verifier_msg = None;
        let mut prover_msgs = vec![];
        let mut challenge = F::zero();

        for _ in 0..*zero_domain {
            let prover_msg = pool_run.install(|| {
                IPforSumCheck::prover_round(&mut prover_state, &verifier_msg, transcript)
            });

            // compute the seed and input required to generate the random challenge
            let verifier_challenge_sampling_timer =
                start_timer!(|| "verifier sampling a random challenge using the transcripts");

            challenge = transcript
                .get_and_append_challenge(b"round_challenge")
                .unwrap();

            end_timer!(verifier_challenge_sampling_timer);

            verifier_msg = Some(VerifierMsg {
                challenge: challenge,
            });
            prover_msgs.push(prover_msg);
        }

        prover_state.challenges.push(challenge);

        end_timer!(sumcheck_prover_timer);

        let opening_proof_timer = start_timer!(|| "batch open proof g,h,s,o input MLEs at r");

        let inp_openings = pool_open.install(|| {
            PCS::batch_open(
                &zero_params.ck,
                &inp_commitments,
                &flatten_mle_extensions,
                prover_state.challenges.clone(),
            )
            .unwrap()
        });

        end_timer!(opening_proof_timer);

        let inp_mle_evaluation_timer =
            start_timer!(|| "computing evaluations of input MLEs at challenges");

        let inp_evals = pool_run.install(|| {
            flatten_mle_extensions
                .clone()
                .into_par_iter()
                .map(|mle| mle.evaluate(&prover_state.challenges))
                .collect::<Vec<_>>()
        });

        end_timer!(inp_mle_evaluation_timer);
        end_timer!(prover_start);

        Ok(Proof {
            prover_msgs: prover_msgs,
            inp_mle_commitments: inp_commitments,
            inp_mle_evals: inp_evals,
            inp_mle_openings: inp_openings,
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
    fn verify<'a>(
        zero_params: &Self::ZeroCheckParams<'_>,
        input_poly: &Self::InputType,
        proof: &Self::Proof,
        zero_domain: &Self::ZeroDomain,
        transcript: &mut Self::Transcripts,
    ) -> Result<bool, anyhow::Error> {
        // check if the zero domain (dimensions of boolean hypercube)
        // is same as the number of variables
        assert_eq!(
            input_poly.poly_info.num_vars, *zero_domain,
            "Dimensions of boolean hypercube do not match the given polynomials"
        );

        let flatten_mle_extensions: Vec<DenseMultilinearExtension<_>> = input_poly
            .clone()
            .flat_ml_extensions
            .into_par_iter()
            .map(|mle| (*mle.as_ref()).clone())
            .collect();

        // set up the seed and input required to generate the initial random challenge
        let initial_challenge_timer =
            start_timer!(|| "computing inital challenge using which f_hat is computed");

        let _: Vec<_> = proof
            .inp_mle_commitments
            .clone()
            .into_iter()
            .map(|comm| {
                transcript
                    .append_serializable_element(b"comm_mle", &comm)
                    .unwrap();
            })
            .collect();

        let r_point: Vec<F> = (0..*zero_domain)
            .into_iter()
            .map(|_| {
                let r = transcript
                    .get_and_append_challenge(b"init_challenge")
                    .unwrap();
                r
            })
            .collect();

        end_timer!(initial_challenge_timer);

        // compute f_hat(X) = sum_{B^m} f(X).eq(X, r)
        let compute_f_hat_timer = start_timer!(|| "computing f_hat(X) = sum_{B^m} f(X).eq(X, r)");

        let inp_hat = input_poly.build_f_hat(&r_point).unwrap();

        end_timer!(compute_f_hat_timer);

        // Run sumcheck verifier algorithm for #num_var rounds
        let sumcheck_verifier_timer = start_timer!(|| format!(
            "running sumcheck proving algorithm for {:?} rounds",
            zero_domain
        ));

        let mut verifier_state = IPforSumCheck::verifier_init(inp_hat.poly_info.clone());

        for i in 0..*zero_domain {
            let prover_msg = proof.prover_msgs.get(i).expect("proof is incomplete");
            let _verifier_msg = IPforSumCheck::verifier_round(
                (*prover_msg).clone(),
                &mut verifier_state,
                transcript,
            );
        }

        let subclaim = IPforSumCheck::check_n_generate_subclaim(verifier_state, F::zero()).unwrap();

        end_timer!(sumcheck_verifier_timer);

        //Check opening proofs of input polynomials
        let _ = (0..flatten_mle_extensions.len()).into_iter().map(|i| {
            let checked = PCS::check(
                &zero_params.vk,
                &proof.inp_mle_openings[i],
                &proof.inp_mle_commitments[i],
                subclaim.point.clone(),
                proof.inp_mle_evals[i],
            )
            .unwrap();
            assert_eq!(checked, true, "invalid opening proof");
        });

        let lhs = subclaim.expected_evaluation;
        let rhs = inp_hat.evaluate(subclaim.point);

        // println!("lhs: {:?}", lhs);
        // println!("rhs: {:?}", rhs);

        // check if the virtual polynomial evaluates to the
        // given value over the sampled challenges
        let result: bool = lhs == rhs;
        Ok(result)
    }
}

#[cfg(test)]
mod test {
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_ec::AffineRepr;
    use ark_ed_on_bls12_381::EdwardsAffine;

    use crate::{
        pcs::multilinear_pcs::{hyrax::Hyrax, mpc::MPC},
        transcripts::ZCTranscript,
        zc::multilinear_zc::optimized::custom_zero_test_case,
        ZeroCheck,
    };

    use super::{rand_zero, OptMLZeroCheck};

    #[test]
    fn test_ml_zerocheck() {
        let poly = rand_zero::<Fr>(10, (4, 5), 2);
        let num_vars = 10;
        let zp = OptMLZeroCheck::<Fr, MPC<Bls12_381>>::setup(&num_vars).unwrap();

        let proof = OptMLZeroCheck::<Fr, MPC<Bls12_381>>::prove(
            &zp.clone(),
            &poly.clone(),
            &10,
            &mut ZCTranscript::init_transcript(),
            None,
            None,
            None,
        )
        .unwrap();
        // println!("Proof Generated: {:?}", proof);

        let valid = OptMLZeroCheck::<Fr, MPC<Bls12_381>>::verify(
            &zp,
            &poly,
            &proof,
            &10,
            &mut ZCTranscript::init_transcript(),
        )
        .unwrap();

        assert!(valid);
    }

    #[test]
    fn test_custom_ml_zerocheck() {
        let poly = custom_zero_test_case::<Fr>(10);
        let num_vars = 10;

        println!("Unique input MLEs: {:?}", poly.flat_ml_extensions.len());

        let zp = OptMLZeroCheck::<Fr, MPC<Bls12_381>>::setup(&num_vars).unwrap();

        let proof = OptMLZeroCheck::<Fr, MPC<Bls12_381>>::prove(
            &zp.clone(),
            &poly.clone(),
            &10,
            &mut ZCTranscript::init_transcript(),
            None,
            None,
            None,
        )
        .unwrap();
        println!("Proof Generated: {:?}", proof);

        let valid = OptMLZeroCheck::<Fr, MPC<Bls12_381>>::verify(
            &zp,
            &poly,
            &proof,
            &10,
            &mut ZCTranscript::init_transcript(),
        )
        .unwrap();

        assert!(valid);
    }

    #[test]
    fn test_custom_ml_zerocheck_hyrax() {
        let num_vars = 20;
        let poly = custom_zero_test_case::<<EdwardsAffine as AffineRepr>::ScalarField>(num_vars);

        println!("Unique input MLEs: {:?}", poly.flat_ml_extensions.len());

        type Fq = <EdwardsAffine as AffineRepr>::ScalarField;

        let zp = OptMLZeroCheck::<Fq, Hyrax<EdwardsAffine>>::setup(&num_vars).unwrap();

        let proof = OptMLZeroCheck::<Fq, Hyrax<EdwardsAffine>>::prove(
            &zp.clone(),
            &poly.clone(),
            &num_vars,
            &mut ZCTranscript::init_transcript(),
            None,
            None,
            None,
        )
        .unwrap();

        // println!("Proof Generated: {:?}", proof);

        let valid = OptMLZeroCheck::<Fq, Hyrax<EdwardsAffine>>::verify(
            &zp,
            &poly,
            &proof,
            &num_vars,
            &mut ZCTranscript::init_transcript(),
        )
        .unwrap();

        assert!(valid);
    }
}
