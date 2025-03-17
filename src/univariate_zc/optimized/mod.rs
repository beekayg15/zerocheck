use crate::utils::*;
use crate::ZeroCheck;
use anyhow::Ok;
use ark_ec::pairing::Pairing;
use ark_ec::AffineRepr;
use ark_ff::One;
use ark_ff::{FftField, PrimeField};
use ark_poly::EvaluationDomain;
use ark_poly::{univariate::DensePolynomial, Evaluations, GeneralEvaluationDomain, Polynomial};
use ark_poly_commit::kzg10::{Powers, VerifierKey, KZG10};
use ark_std::{end_timer, start_timer, test_rng};
use rayon::prelude::*;
use std::marker::PhantomData;

pub mod data_structures;
use data_structures::*;

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
pub struct OptimizedUnivariateZeroCheck<F, E> {
    _field_data: PhantomData<F>,
    _pairing_data: PhantomData<E>
}

impl<F, E> ZeroCheck<F> for OptimizedUnivariateZeroCheck<F, E>
where
    F: PrimeField + FftField,
    E: Pairing
{
    type InputType = Vec<Evaluations<E::ScalarField>>;
    type ZeroDomain = GeneralEvaluationDomain<E::ScalarField>;
    type Proof = Proof<E>;
    type ZeroCheckParams = ZeroCheckParams<E>;
    type InputParams = InputParams;

    fn setup<'a>(
        pp: Self::InputParams
    ) -> Result<Self::ZeroCheckParams, anyhow::Error> {
        let setup_kzg_time =
            start_timer!(|| "Setup KZG10 polynomial commitments global parameters");

        // Setting up the KZG(MSM) Polynomial Commitment Scheme
        let rng = &mut test_rng();
        let params = KZG10::<E, DensePolynomial<E::ScalarField>>::setup(
            pp.max_degree, 
            false, rng
        ).expect("PCS setup failed");

        end_timer!(setup_kzg_time);
        let setup_vk_time = start_timer!(|| "Setup verifier key");

        // Computing the verification key
        let vk: VerifierKey<E> = VerifierKey {
            g: params.powers_of_g[0],
            gamma_g: params.powers_of_gamma_g[&0],
            h: params.h,
            beta_h: params.beta_h,
            prepared_h: params.prepared_h.clone(),
            prepared_beta_h: params.prepared_beta_h.clone(),
        };

        end_timer!(setup_vk_time);
        let powers_of_g_time = start_timer!(|| "Computing the powers of G");

        // Computing the powers of the generator 'G'
        let powers_of_g = params.powers_of_g[..=(pp.max_degree)].to_vec();
        let powers_of_gamma_g = (0..=(pp.max_degree))
            .map(|i| params.powers_of_gamma_g[&i])
            .collect();

        end_timer!(powers_of_g_time);

        Ok(ZeroCheckParams{
            vk,
            powers_of_g,
            powers_of_gamma_g,
        })
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
    fn prove<'a> (
        zero_params: Self::ZeroCheckParams,
        input_poly: Self::InputType,
        zero_domain: Self::ZeroDomain,
    ) -> Result<Self::Proof, anyhow::Error> {
        // compute the vanishing polynomial of the zero domain
        // let z_poly = zero_domain.vanishing_polynomial();
        let z_deg = zero_domain.size();

        let prove_time = start_timer!(|| format!(
            "OptimizedUnivariateZeroCheck::prove, with {z_deg} zero_domain size."
        ));

        let ifft_time = start_timer!(|| "IFFT for g,h,s,o from evaluations to coefficients");
        let g = input_poly[0].clone(); // g_evals
        let h = input_poly[1].clone(); // h_evals
        let s = input_poly[2].clone(); // s_evals
        let o = input_poly[3].clone(); // o_evals

        // compute the polynomials corresponding to g, h, and s using interpolation (IFFT)
        let ghso_coeffs: Vec<_> = [&g, &h, &s, &o]
            .par_iter()
            .enumerate()
            .map(|(i, evals)| (i, (*evals).clone().interpolate()))
            .collect();
        let mut ghso_sorted_coeffs = ghso_coeffs;
        ghso_sorted_coeffs.sort_by_key(|(i, _)| *i);
        let [g_coeff, h_coeff, s_coeff, o_coeff]: [_; 4] = ghso_sorted_coeffs
            .into_iter()
            .map(|(_, coeff)| coeff)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        end_timer!(ifft_time);
        let commit_time = start_timer!(|| "KZG commit to (g,h,s,o) polynomials");

        let powers = Powers {
            powers_of_g: ark_std::borrow::Cow::Owned(zero_params.powers_of_g),
            powers_of_gamma_g: ark_std::borrow::Cow::Owned(zero_params.powers_of_gamma_g),
        };

        // Compute the commitment to the polynomial g(X), h(X), s(X), and o(X)
        let [(comm_g, r_g), (comm_h, r_h), (comm_s, r_s), (comm_o, r_o)] =
            [&g_coeff, &h_coeff, &s_coeff, &o_coeff]
                .par_iter()
                .enumerate()
                .map(|(idx, poly)| {
                    KZG10::<E, DensePolynomial<E::ScalarField>>::commit(
                        &powers, 
                        poly, 
                        None, 
                        None
                    ).expect(format!("Commitment to polynomial {idx}_(X) failed").as_str())
                })
                .collect::<Vec<_>>()
                .try_into()
                .unwrap();

        assert!(!comm_g.0.is_zero(), "Commitment should not be zero");
        assert!(!r_g.is_hiding(), "Commitment should not be hiding");
        assert!(!comm_h.0.is_zero(), "Commitment should not be zero");
        assert!(!r_h.is_hiding(), "Commitment should not be hiding");
        assert!(!comm_s.0.is_zero(), "Commitment should not be zero");
        assert!(!r_s.is_hiding(), "Commitment should not be hiding");
        assert!(!comm_o.0.is_zero(), "Commitment should not be zero");
        assert!(!r_o.is_hiding(), "Commitment should not be hiding");

        // Collect the commitment to the input polynomials
        let inp_comms = vec![comm_g, comm_h, comm_s, comm_o];

        end_timer!(commit_time);
        let get_r_eval_time =
            start_timer!(|| "Get Fiat-Shamir random challenge and evals at challenge");

        // Sample a random challenge - Fiat-Shamir
        let mut inp_rand = h.evals;
        inp_rand.extend(s.evals);
        inp_rand.extend(o.evals);
        let r = get_randomness(g.evals, inp_rand)[0];

        // Collect the evalution of the input polynomials at the challenge
        let mut inp_evals_at_rand = vec![];
        inp_evals_at_rand.push(g_coeff.evaluate(&r));
        inp_evals_at_rand.push(h_coeff.evaluate(&r));
        inp_evals_at_rand.push(s_coeff.evaluate(&r));
        inp_evals_at_rand.push(o_coeff.evaluate(&r));

        end_timer!(get_r_eval_time);
        let open_time = start_timer!(|| "KZG open the g,h,s,o poly commit at r");

        // Generate the opening proof that g(r), h(r), s(r), and o(r) are the evaluations of the polynomials
        let opening_proofs = [
            (&g_coeff, &r_g),
            (&h_coeff, &r_h),
            (&s_coeff, &r_s),
            (&o_coeff, &r_o),
        ]
        .par_iter()
        .enumerate()
        .map(|(idx, (poly, r_poly))| {
            KZG10::<E, DensePolynomial<E::ScalarField>>::open(&powers, poly, r, r_poly)
                .expect(format!("Proof generation failed for {idx}_(X)").as_str())
        })
        .collect::<Vec<_>>();
        let mut opening_proofs_sorted = opening_proofs;
        opening_proofs_sorted.sort_by_key(|(i, _)| *i);
        let [g_opening_proof, h_opening_proof, s_opening_proof, o_opening_proof] =
            opening_proofs_sorted
                .into_iter()
                .map(|(_, proof)| proof)
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

        // Compute the quotient polynomial q(X) = f(X)/z_H(X) = (g.h.s + (1-s)(g+h))/z_H
        let g_deg = g_coeff.degree();
        let h_deg = h_coeff.degree();
        let s_deg = s_coeff.degree();

        // compute degree of quotient polynomial to
        let f_deg = g_deg + h_deg + s_deg;
        let q_deg = f_deg - z_deg;

        let coset_time = start_timer!(|| "Compute coset domain");
        // Compute the coset domain to interpolate q(X)
        let q_domain = GeneralEvaluationDomain::<E::ScalarField>::new(q_deg + 1).unwrap();
        let offset = <E::ScalarField>::GENERATOR;
        let coset_domain = q_domain.get_coset(offset).unwrap();

        end_timer!(coset_time);
        let coset_eval_time = start_timer!(|| "Compute g,h,s,o,z,q evaluations over coset domain");

        // Evaluate the values of g(X), h(X), s(X), and z_h(X) over the coset domain
        let g_evals = g_coeff.evaluate_over_domain(coset_domain).evals;
        let h_evals = h_coeff.evaluate_over_domain(coset_domain).evals;
        let s_evals = s_coeff.evaluate_over_domain(coset_domain).evals;
        let o_evals = o_coeff.evaluate_over_domain(coset_domain).evals;
        let z_evals = zero_domain
            .vanishing_polynomial()
            .evaluate_over_domain(coset_domain)
            .evals;

        // Find the value of q(X) over the coset domain
        let mut q_evals = vec![];
        for i in 0..g_evals.len() {
            q_evals.push(
                ((g_evals[i] * h_evals[i] * s_evals[i])
                    + (<E::ScalarField>::one() - s_evals[i]) * (g_evals[i] + h_evals[i])
                    - o_evals[i])
                    / z_evals[i],
            )
        }

        end_timer!(coset_eval_time);
        let ifft_q_time = start_timer!(|| "IFFT for q from evaluations to coefficients");

        // Interpolate q(X) using the evaluations
        let q_coeff = Evaluations::from_vec_and_domain(q_evals, coset_domain).interpolate();

        end_timer!(ifft_q_time);
        let comm_q_time = start_timer!(|| "KZG commit to q polynomial");

        // Compute the commitment to the polynomial q(X)
        let (comm_q, r_q) =
            KZG10::<E, DensePolynomial<E::ScalarField>>::commit(&powers, &q_coeff, None, None)
                .expect("Commitment to polynomial q(X) failed");

        assert!(!comm_q.0.is_zero(), "Commitment should not be zero");
        assert!(!r_q.is_hiding(), "Commitment should not be hiding");

        end_timer!(comm_q_time);
        let open_q_time = start_timer!(|| "KZG open the q poly commit at r");

        // Generate the opening proof that q(r) = t
        let q_opening_proof =
        KZG10::<E, DensePolynomial<E::ScalarField>>::open(
            &powers, 
            &q_coeff, 
            r, 
            &r_q
        ).expect("Proof generation failed for q(X)");

        end_timer!(open_q_time);
        end_timer!(prove_time);

        // Send the proof with the necessary commitments and opening proofs
        Ok(Proof {
            q_comm: comm_q,
            inp_comms: inp_comms,
            inp_evals: inp_evals_at_rand,
            inp_openings: inp_opening_proofs,
            q_eval: q_coeff.evaluate(&r),
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
    fn verify<'a> (
        zero_params: Self::ZeroCheckParams,
        input_poly: Self::InputType,
        proof: Self::Proof,
        zero_domain: Self::ZeroDomain,
    ) -> Result<bool, anyhow::Error> {
        let g = input_poly[0].clone();
        let h = input_poly[1].clone();
        let s = input_poly[2].clone();
        let o = input_poly[2].clone();

        let q_comm = proof.q_comm;
        let inp_comms = proof.inp_comms;
        let vk = zero_params.vk;
        let inp_openings = proof.inp_openings;
        let inp_evals = proof.inp_evals;
        let q_opening = proof.q_opening;
        let q_eval = proof.q_eval;

        let mut inp_rand = h.evals;
        inp_rand.extend(s.evals);
        inp_rand.extend(o.evals);
        let r = get_randomness(g.evals, inp_rand)[0];

        // check openings to input polynomials
        for i in 0..inp_evals.len() {
            assert!(
                KZG10::<E, DensePolynomial<E::ScalarField>>::check(
                    &vk, 
                    &inp_comms[i], 
                    r, inp_evals[i], 
                    &inp_openings[i]
                ).unwrap(),
                "Opening failed at input polynomial {:?}",
                i + 1
            );
        }

        // check opening to quotient polynomials
        assert!(
            KZG10::<E, DensePolynomial<E::ScalarField>>::check(
                &vk, 
                &q_comm, 
                r, 
                q_eval, 
                &q_opening
            ).unwrap(),
            "Opening failed at quotient polynomial"
        );

        let a = inp_evals[0];
        let b = inp_evals[1];
        let c = inp_evals[2];
        let d = inp_evals[3];

        // check if q(r) * z_H(r) = g(r).h(r).s(r) + (1 - s(r))(g(r) + h(r))
        let lhs = q_eval * zero_domain.evaluate_vanishing_polynomial(r);
        let rhs = a * b * c + (<E::ScalarField>::one() - c) * (a + b) - d;

        Ok(lhs == rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Bls12_381;
    use ark_bls12_381::Fr;
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
        let pp = InputParams{
            max_degree,
        };

        let zp = OptimizedUnivariateZeroCheck::<Fr, Bls12_381>::setup(pp).unwrap();

        let proof = OptimizedUnivariateZeroCheck::<Fr, Bls12_381>::prove(
            zp.clone(),
            inp_evals.clone(), 
            domain
        )
            .unwrap();

        end_timer!(proof_gen_timer);

        println!("Proof Generated");

        let verify_timer = start_timer!(|| "Verify fn called for g, h, zero_domain, proof");

        let result =
            OptimizedUnivariateZeroCheck::<Fr, Bls12_381>::verify(
                zp,
                inp_evals, 
                proof, 
                domain
            ).unwrap();

        end_timer!(verify_timer);

        println!("verification result: {:?}", result);
        assert_eq!(result, true);

        end_timer!(test_timer);
    }
}
