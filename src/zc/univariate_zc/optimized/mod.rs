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
pub struct OptimizedUnivariateZeroCheck<F, PCS> {
    _field_data: PhantomData<F>,
    _pcs_data: PhantomData<PCS>,
}

impl<F, PCS> ZeroCheck<F> for OptimizedUnivariateZeroCheck<F, PCS>
where
    F: PrimeField + FftField,
    PCS: PolynomialCommitmentScheme<
        Polynomial = DensePolynomial<F>,
        PolynomialInput = F,
        PolynomialOutput = F,
    >,
{
    type InputType = Vec<Evaluations<F>>;
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
        let ghso_coeffs: Vec<_> = pool_run.install(|| {
            input_poly
                .par_iter()
                .enumerate()
                .map(|(i, evals)| (i, evals.clone().interpolate()))
                .collect()
        });
        let mut ghso_sorted_coeffs = ghso_coeffs;
        ghso_sorted_coeffs.sort_by_key(|(i, _)| *i);
        let [g_coeff, h_coeff, s_coeff, o_coeff]: [_; 4] = ghso_sorted_coeffs
            .into_iter()
            .map(|(_, coeff)| coeff)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        end_timer!(ifft_time);
        let coset_time = start_timer!(|| "Compute coset domain");

        // Compute the quotient polynomial q(X) = f(X)/z_H(X) = (g.h.s + (1-s)(g+h))/z_H
        let g_deg = g_coeff.degree();
        let h_deg = h_coeff.degree();
        let s_deg = s_coeff.degree();

        // compute degree of quotient polynomial to
        let f_deg = g_deg + h_deg + s_deg;
        let q_deg = f_deg - z_deg;

        // Compute the coset domain to interpolate q(X)
        let q_domain = GeneralEvaluationDomain::<F>::new(q_deg + 1).unwrap();
        let offset = F::GENERATOR;
        let coset_domain = q_domain.get_coset(offset).unwrap();

        end_timer!(coset_time);
        let coset_eval_time =
            start_timer!(|| "FFT Compute g,h,s,o,z,q evaluations over coset domain");

        // Evaluate the values of g(X), h(X), s(X), and z_h(X) over the coset domain using threads
        let (g_evals, h_evals, s_evals, o_evals, z_evals) = pool_run.install(|| {
            let g_evals = g_coeff.clone().evaluate_over_domain(coset_domain).evals;
            let h_evals = h_coeff.clone().evaluate_over_domain(coset_domain).evals;
            let s_evals = s_coeff.clone().evaluate_over_domain(coset_domain).evals;
            let o_evals = o_coeff.clone().evaluate_over_domain(coset_domain).evals;
            let z_evals = zero_domain
                .vanishing_polynomial()
                .evaluate_over_domain(coset_domain)
                .evals;
            (g_evals, h_evals, s_evals, o_evals, z_evals)
        });

        // Find the value of q(X) over the coset domain using threads
        let q_evals = pool_run.install(|| {
            (0..g_evals.len())
                .into_par_iter()
                .map(|i| {
                    ((g_evals[i] * h_evals[i] * s_evals[i])
                        + (F::one() - s_evals[i]) * (g_evals[i] + h_evals[i])
                        - o_evals[i])
                        / z_evals[i]
                })
                .collect::<Vec<_>>()
        });

        end_timer!(coset_eval_time);
        let ifft_q_time = start_timer!(|| "IFFT for q from evaluations to coefficients");

        // Interpolate q(X) using the evaluations
        let q_coeff = pool_run
            .install(|| Evaluations::from_vec_and_domain(q_evals, coset_domain).interpolate());

        end_timer!(ifft_q_time);
        let commit_time = start_timer!(|| "Commit to (g,h,s,o) polynomials");

        // Use the pool_commit thread pool to perform the batch_commit operation
        let comm_rs = pool_commit.install(|| {
            PCS::batch_commit(
                &zero_params.ck,
                &vec![
                    g_coeff.clone(),
                    h_coeff.clone(),
                    s_coeff.clone(),
                    o_coeff.clone(),
                ],
            )
            .unwrap()
        });

        let comms = comm_rs.clone();
        let [comm_g, comm_h, comm_s, comm_o] = &comms[..] else {
            panic!("commitments mismatch");
        };

        // Collect the commitment to the input polynomials
        let inp_comms = vec![
            comm_g.clone(),
            comm_h.clone(),
            comm_s.clone(),
            comm_o.clone(),
        ];
        end_timer!(commit_time);
        let commit_q_time = start_timer!(|| "Commit to (q) polynomial");

        let comm_q = pool_commit.install(|| PCS::commit(&zero_params.ck, &q_coeff).unwrap());
        let comm_rs = vec![
            comm_g.clone(),
            comm_h.clone(),
            comm_s.clone(),
            comm_o.clone(),
            comm_q.clone(),
        ];
        end_timer!(commit_q_time);

        let get_r_eval_time = start_timer!(|| "computing evaluations at challenge, get challenge");

        // Sample a random challenge - Fiat-Shamir
        transcript
            .append_serializable_element(b"comm_g", &PCS::extract_pure_commitment(comm_g).unwrap())
            .unwrap();
        transcript
            .append_serializable_element(b"comm_h", &PCS::extract_pure_commitment(comm_h).unwrap())
            .unwrap();
        transcript
            .append_serializable_element(b"comm_s", &PCS::extract_pure_commitment(comm_s).unwrap())
            .unwrap();
        transcript
            .append_serializable_element(b"comm_o", &PCS::extract_pure_commitment(comm_o).unwrap())
            .unwrap();
        transcript
            .append_serializable_element(b"comm_q", &PCS::extract_pure_commitment(&comm_q).unwrap())
            .unwrap();
        let r = transcript.get_and_append_challenge(b"sampling r").unwrap();

        // Collect the evalution of the input polynomials at the challenge
        let mut inp_evals_at_rand = vec![];
        inp_evals_at_rand.push(g_coeff.evaluate(&r));
        inp_evals_at_rand.push(h_coeff.evaluate(&r));
        inp_evals_at_rand.push(s_coeff.evaluate(&r));
        inp_evals_at_rand.push(o_coeff.evaluate(&r));
        let q_eval_r = q_coeff.evaluate(&r);

        end_timer!(get_r_eval_time);
        let open_time = start_timer!(|| "Batch open the g,h,s,o,q poly commit at r");

        // Generate the opening proof that g(r), h(r), s(r), and o(r) are the evaluations of the polynomials
        let opening_proofs = pool_open.install(|| {
            PCS::batch_open(
                &zero_params.ck,
                &comm_rs,
                &vec![g_coeff, h_coeff, s_coeff, o_coeff, q_coeff],
                r,
            )
            .unwrap()
        });

        let [g_opening_proof, h_opening_proof, s_opening_proof, o_opening_proof, q_opening_proof] =
            opening_proofs
                .into_iter()
                .map(|proof| proof)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();

        // Collect the opening proofs of the input polynomials
        let inp_opening_proofs = vec![
            g_opening_proof,
            h_opening_proof,
            s_opening_proof,
            o_opening_proof,
        ];

        end_timer!(open_time);
        end_timer!(prove_time);

        // Send the proof with the necessary commitments and opening proofs
        Ok(Proof {
            q_comm: comm_q,
            inp_comms: inp_comms,
            inp_evals: inp_evals_at_rand,
            inp_openings: inp_opening_proofs,
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
        _input_poly: &Self::InputType,
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
        transcript
            .append_serializable_element(
                b"comm_g",
                &PCS::extract_pure_commitment(&inp_comms[0]).unwrap(),
            )
            .unwrap();
        transcript
            .append_serializable_element(
                b"comm_h",
                &PCS::extract_pure_commitment(&inp_comms[1]).unwrap(),
            )
            .unwrap();
        transcript
            .append_serializable_element(
                b"comm_s",
                &PCS::extract_pure_commitment(&inp_comms[2]).unwrap(),
            )
            .unwrap();
        transcript
            .append_serializable_element(
                b"comm_o",
                &PCS::extract_pure_commitment(&inp_comms[3]).unwrap(),
            )
            .unwrap();
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

        let a = inp_evals[0];
        let b = inp_evals[1];
        let c = inp_evals[2];
        let d = inp_evals[3];

        // check if q(r) * z_H(r) = g(r).h(r).s(r) + (1 - s(r))(g(r) + h(r))
        let lhs = q_eval * zero_domain.evaluate_vanishing_polynomial(r);
        let rhs = a * b * c + (F::one() - c) * (a + b) - d;

        Ok(lhs == rhs)
    }
}

#[cfg(test)]
mod tests {
    use crate::pcs::univariate_pcs::kzg::KZG;
    use crate::pcs::univariate_pcs::ligero::Ligero;

    use super::*;
    use ark_bls12_381::Bls12_381;
    use ark_bls12_381::Fr;
    use ark_ff::One;
    use ark_ff::UniformRand;
    use ark_poly::{
        univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Evaluations,
        GeneralEvaluationDomain,
    };
    use ark_std::end_timer;
    use ark_std::start_timer;

    #[test]
    fn test_proof_generation_verification_op_uni() {
        let test_timer = start_timer!(|| "Proof Generation Test");

        let domain = GeneralEvaluationDomain::<Fr>::new(1 << 10).unwrap();

        let rng = &mut ark_std::test_rng();

        let mut rand_g_coeffs = vec![];
        let mut rand_h_coeffs = vec![];
        let mut rand_s_coeffs = vec![];

        for _ in 0..(1 << 10) {
            rand_g_coeffs.push(Fr::rand(rng));
            rand_h_coeffs.push(Fr::rand(rng));
            rand_s_coeffs.push(Fr::rand(rng));
        }

        let g = DensePolynomial::from_coefficients_vec(rand_g_coeffs);
        let h = DensePolynomial::from_coefficients_vec(rand_h_coeffs);
        let s = DensePolynomial::from_coefficients_vec(rand_s_coeffs);

        let evals_over_domain_g: Vec<_> = domain.elements().map(|f| g.evaluate(&f)).collect();

        let evals_over_domain_h: Vec<_> = domain.elements().map(|f| h.evaluate(&f)).collect();

        let evals_over_domain_s: Vec<_> = domain.elements().map(|f| s.evaluate(&f)).collect();

        let evals_over_domain_o: Vec<_> = domain
            .elements()
            .map(|f| {
                g.evaluate(&f) * h.evaluate(&f) * s.evaluate(&f)
                    + (Fr::one() - s.evaluate(&f)) * (g.evaluate(&f) + h.evaluate(&f))
            })
            .collect();

        let g_evals = Evaluations::from_vec_and_domain(evals_over_domain_g, domain);

        let h_evals = Evaluations::from_vec_and_domain(evals_over_domain_h, domain);

        let s_evals = Evaluations::from_vec_and_domain(evals_over_domain_s, domain);

        let o_evals = Evaluations::from_vec_and_domain(evals_over_domain_o, domain);

        let mut inp_evals = vec![];
        inp_evals.push(g_evals);
        inp_evals.push(h_evals);
        inp_evals.push(s_evals);
        inp_evals.push(o_evals);

        let proof_gen_timer = start_timer!(|| "Prove fn called for g, h, zero_domain");

        let max_degree = g.degree() + s.degree() + h.degree();
        let pp = max_degree;

        let zp = OptimizedUnivariateZeroCheck::<Fr, KZG<Bls12_381>>::setup(&pp).unwrap();

        let proof = OptimizedUnivariateZeroCheck::<Fr, KZG<Bls12_381>>::prove(
            &zp.clone(),
            &inp_evals.clone(),
            &domain,
            &mut ZCTranscript::init_transcript(),
            None,
            None,
            None,
        )
        .unwrap();

        end_timer!(proof_gen_timer);

        let verify_timer = start_timer!(|| "Verify fn called for g, h, zero_domain, proof");

        let result = OptimizedUnivariateZeroCheck::<Fr, KZG<Bls12_381>>::verify(
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
    fn test_proof_generation_verification_op_uni_ligero() {
        let test_timer = start_timer!(|| "Proof Generation Test");

        let domain = GeneralEvaluationDomain::<Fr>::new(1 << 5).unwrap();

        let rng = &mut ark_std::test_rng();

        let mut rand_g_coeffs = vec![];
        let mut rand_h_coeffs = vec![];
        let mut rand_s_coeffs = vec![];

        for _ in 0..(1 << 5) {
            rand_g_coeffs.push(Fr::rand(rng));
            rand_h_coeffs.push(Fr::rand(rng));
            rand_s_coeffs.push(Fr::rand(rng));
        }

        let g = DensePolynomial::from_coefficients_vec(rand_g_coeffs);
        let h = DensePolynomial::from_coefficients_vec(rand_h_coeffs);
        let s = DensePolynomial::from_coefficients_vec(rand_s_coeffs);

        let evals_over_domain_g: Vec<_> = domain.elements().map(|f| g.evaluate(&f)).collect();

        let evals_over_domain_h: Vec<_> = domain.elements().map(|f| h.evaluate(&f)).collect();

        let evals_over_domain_s: Vec<_> = domain.elements().map(|f| s.evaluate(&f)).collect();

        let evals_over_domain_o: Vec<_> = domain
            .elements()
            .map(|f| {
                g.evaluate(&f) * h.evaluate(&f) * s.evaluate(&f)
                    + (Fr::one() - s.evaluate(&f)) * (g.evaluate(&f) + h.evaluate(&f))
            })
            .collect();

        let g_evals = Evaluations::from_vec_and_domain(evals_over_domain_g, domain);

        let h_evals = Evaluations::from_vec_and_domain(evals_over_domain_h, domain);

        let s_evals = Evaluations::from_vec_and_domain(evals_over_domain_s, domain);

        let o_evals = Evaluations::from_vec_and_domain(evals_over_domain_o, domain);

        let mut inp_evals = vec![];
        inp_evals.push(g_evals);
        inp_evals.push(h_evals);
        inp_evals.push(s_evals);
        inp_evals.push(o_evals);

        let proof_gen_timer = start_timer!(|| "Prove fn called for g, h, zero_domain");

        let max_degree = g.degree() + s.degree() + h.degree();
        let pp = max_degree;

        let zp = OptimizedUnivariateZeroCheck::<Fr, Ligero<Fr>>::setup(&pp).unwrap();

        let proof = OptimizedUnivariateZeroCheck::<Fr, Ligero<Fr>>::prove(
            &zp.clone(),
            &inp_evals.clone(),
            &domain,
            &mut ZCTranscript::init_transcript(),
            None,
            None,
            None,
        )
        .unwrap();

        end_timer!(proof_gen_timer);

        let verify_timer = start_timer!(|| "Verify fn called for g, h, zero_domain, proof");

        let result = OptimizedUnivariateZeroCheck::<Fr, Ligero<Fr>>::verify(
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
