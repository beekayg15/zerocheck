use std::marker::PhantomData;
use anyhow::Ok;
use ark_ff::{FftField, PrimeField};
use ark_poly::EvaluationDomain;
use ark_poly::{
    univariate::DensePolynomial, 
    Evaluations, GeneralEvaluationDomain, Polynomial
};
use ark_ec::pairing::Pairing;
use ark_poly_commit::kzg10::{KZG10, Powers, VerifierKey};
use ark_std::test_rng;
use ark_ff::One;
use ark_ec::AffineRepr;

use crate::ZeroCheck;
use crate::utils::*;

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

impl<F, E> ZeroCheck<F, E> for OptimizedUnivariateZeroCheck<F, E> 
where 
    E: Pairing,
    F: PrimeField + FftField,
{
    type InputType = Evaluations<E::ScalarField>;
    type ZeroDomain = GeneralEvaluationDomain<E::ScalarField>;
    type Proof = Proof<E>;
    type PCS = KZG10<E, DensePolynomial<E::ScalarField>>;

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
        input_poly: Vec<Self::InputType>,
        zero_domain: Self::ZeroDomain
    ) -> Result<Self::Proof, anyhow::Error> {
        
        let g = input_poly[0].clone();
        let h = input_poly[1].clone();
        let s = input_poly[2].clone();
        let o = input_poly[3].clone();

        // compute the polynomials corresponding to g, h, and s using interpolation (IFFT)
        let g_poly = g.clone().interpolate();
        let h_poly = h.clone().interpolate();
        let s_poly = s.clone().interpolate();
        let o_poly = o.clone().interpolate();

        let g_deg = g_poly.degree();
        let h_deg = h_poly.degree();
        let s_deg = s_poly.degree();

        // compute the vanishing polynomial of the zero domain
        // let z_poly = zero_domain.vanishing_polynomial();
        let z_deg = zero_domain.size();

        // compute degree of quotient polynomial to 
        let f_deg = g_deg + h_deg + s_deg;
        let q_deg = f_deg - z_deg;

        // Setting up the KZG(MSM) Polynomial Commitment Scheme
        let rng = &mut test_rng();
        let params = KZG10::<E, DensePolynomial<E::ScalarField>>::setup(
            2 * f_deg, 
            false, 
            rng
        ).expect("PCS setup failed");

        // Computing the verification key
        let vk: VerifierKey<E> = VerifierKey {
            g: params.powers_of_g[0],
            gamma_g: params.powers_of_gamma_g[&0],
            h: params.h,
            beta_h: params.beta_h,
            prepared_h: params.prepared_h.clone(),
            prepared_beta_h: params.prepared_beta_h.clone(),
        };

        // Computing the powers of the generator 'G' 
        let powers_of_g = params.powers_of_g[..= 2 * f_deg].to_vec();
        let powers_of_gamma_g = (0..= 2 * f_deg)
           .map(|i| params.powers_of_gamma_g[&i])
            .collect();
        let powers = Powers {
            powers_of_g: ark_std::borrow::Cow::Owned(powers_of_g),
            powers_of_gamma_g: ark_std::borrow::Cow::Owned(powers_of_gamma_g),
        };

        // Compute the commitment to the polynomial g(X)
        let (comm_g, r_g) = KZG10::<E, DensePolynomial<E::ScalarField>>::commit(
            &powers, 
            &g_poly, 
            None, 
            None
        ).expect("Commitment to polynomail g(X) failed");

        assert!(!comm_g.0.is_zero(), "Commitment should not be zero");
        assert!(!r_g.is_hiding(), "Commitment should not be hiding");

        // Compute the commitment to the polynomial h(X)
        let (comm_h, r_h) = KZG10::<E, DensePolynomial<E::ScalarField>>::commit(
            &powers, 
            &h_poly, 
            None, 
            None
        ).expect("Commitment to polynomail h(X) failed");

        assert!(!comm_h.0.is_zero(), "Commitment should not be zero");
        assert!(!r_h.is_hiding(), "Commitment should not be hiding");

        // Compute the commitment to the polynomial s(X)
        let (comm_s, r_s) = KZG10::<E, DensePolynomial<E::ScalarField>>::commit(
            &powers, 
            &s_poly, 
            None, 
            None
        ).expect("Commitment to polynomail s(X) failed");

        assert!(!comm_s.0.is_zero(), "Commitment should not be zero");
        assert!(!r_s.is_hiding(), "Commitment should not be hiding");

        // Compute the commitment to the polynomial o(X)
        let (comm_o, r_o) = KZG10::<E, DensePolynomial<E::ScalarField>>::commit(
            &powers, 
            &o_poly, 
            None, 
            None
        ).expect("Commitment to polynomail o(X) failed");

        assert!(!comm_o.0.is_zero(), "Commitment should not be zero");
        assert!(!r_o.is_hiding(), "Commitment should not be hiding");

        // Collect the commitment to the input polynomials 
        let mut inp_comms = vec![];
        inp_comms.push(comm_g);
        inp_comms.push(comm_h);
        inp_comms.push(comm_s);
        inp_comms.push(comm_o);

        // Sample a random challenge - Fiat-Shamir
        let mut inp_rand = h.evals;
        inp_rand.extend(s.evals);
        inp_rand.extend(o.evals);
        let r = get_randomness(g.evals ,inp_rand)[0];

        // Collect the evalution of the input polynomials at the challenge
        let mut inp_evals_at_rand = vec![];
        inp_evals_at_rand.push(g_poly.evaluate(&r));
        inp_evals_at_rand.push(h_poly.evaluate(&r));
        inp_evals_at_rand.push(s_poly.evaluate(&r));
        inp_evals_at_rand.push(o_poly.evaluate(&r));

        let mut inp_opening_proofs = vec![];
        
        // Generate the opening proof that g(r) = x
        let g_opening_proof = Self::PCS::open(
            &powers,
            &g_poly,
            r,
            &r_g
        ).expect("Proof generation failed for g(X)");

        // Generate the opening proof that h(r) = y
        let h_opening_proof = Self::PCS::open(
            &powers,
            &h_poly,
            r,
            &r_h
        ).expect("Proof generation failed for h(X)");

        // Generate the opening proof that s(r) = z
        let s_opening_proof = Self::PCS::open(
            &powers,
            &s_poly,
            r,
            &r_s
        ).expect("Proof generation failed for s(X)");

        // Generate the opening proof that s(r) = z
        let o_opening_proof = Self::PCS::open(
            &powers,
            &o_poly,
            r,
            &r_o
        ).expect("Proof generation failed for s(X)");

        // Collect the opening proofs of the input polynomials
        inp_opening_proofs.push(g_opening_proof);
        inp_opening_proofs.push(h_opening_proof);
        inp_opening_proofs.push(s_opening_proof);
        inp_opening_proofs.push(o_opening_proof);

        // Compute the quotient polynomial q(X) = f(X)/z_H(X) = (g.h.s + (1-s)(g+h))/z_H

        // Compute the coset domain to interpolate q(X)
        let q_domain = GeneralEvaluationDomain::<E::ScalarField>::new(q_deg + 1).unwrap();
        let offset = <E::ScalarField>::GENERATOR;
        let coset_domain = q_domain.get_coset(offset).unwrap();

        // Evaluate the values of g(X), h(X), s(X), and z_h(X) over the coset domain
        let g_evals = g_poly.evaluate_over_domain(coset_domain).evals;
        let h_evals = h_poly.evaluate_over_domain(coset_domain).evals;
        let s_evals = s_poly.evaluate_over_domain(coset_domain).evals;
        let o_evals = o_poly.evaluate_over_domain(coset_domain).evals;
        let z_evals = zero_domain.vanishing_polynomial().evaluate_over_domain(coset_domain).evals;

        // Find the value of q(X) over the coset domain
        let mut q_evals = vec![];
        for i in 0..g_evals.len() {
            q_evals.push(
                ((g_evals[i] * h_evals[i] * s_evals[i]) 
                + (<E::ScalarField>::one() - s_evals[i]) * (g_evals[i] + h_evals[i])
                - o_evals[i]) / z_evals[i]
            )
        }

        // Interpolate q(X) using the evaluations
        let q_poly = Evaluations::from_vec_and_domain(q_evals, coset_domain).interpolate();

        // Compute the commitment to the polynomial q(X)
        let (comm_q, r_q) = KZG10::<E, DensePolynomial<E::ScalarField>>::commit(
            &powers, 
            &q_poly, 
            None, 
            None
        ).expect("Commitment to polynomail q(X) failed");

        assert!(!comm_q.0.is_zero(), "Commitment should not be zero");
        assert!(!r_q.is_hiding(), "Commitment should not be hiding");

        // Generate the opening proof that q(r) = t
        let q_opening_proof = Self::PCS::open(
            &powers,
            &q_poly,
            r,
            &r_q
        ).expect("Proof generation failed for q(X)");

        // Send the proof with the necessary commitments and opening proofs
        Ok(Proof{
            q_comm: comm_q,
            inp_comms: inp_comms,
            vk: vk,
            inp_evals: inp_evals_at_rand,
            inp_openings: inp_opening_proofs,
            q_eval: q_poly.evaluate(&r),
            q_opening: q_opening_proof
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
        input_poly: Vec<Self::InputType>,
        proof: Self::Proof,
        zero_domain: Self::ZeroDomain
    ) -> Result<bool, anyhow::Error> {

        let g = input_poly[0].clone();
        let h = input_poly[1].clone();
        let s = input_poly[2].clone();
        let o = input_poly[2].clone();

        let q_comm = proof.q_comm;
        let inp_comms = proof.inp_comms;
        let vk = proof.vk;
        let inp_openings = proof.inp_openings;
        let inp_evals = proof.inp_evals;
        let q_opening = proof.q_opening;
        let q_eval = proof.q_eval;

        let mut inp_rand = h.evals;
        inp_rand.extend(s.evals);
        inp_rand.extend(o.evals);
        let r = get_randomness(g.evals ,inp_rand)[0];

        // check openings to input polynomials
        for i in 0..inp_evals.len() {
            assert!(
                Self::PCS::check(
                    &vk,
                    &inp_comms[i],
                    r,
                    inp_evals[i],
                    &inp_openings[i]
                ).unwrap(),
                "Opening failed at input polynomial {:?}",
                i + 1
            );
        }

        // check opening to quotient polynomials
        assert!(
            Self::PCS::check(
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
    use ark_bls12_381::Fr;
    use ark_bls12_381::Bls12_381;
    use ark_ff::UniformRand;
    use ark_poly::{
        univariate::DensePolynomial, 
        DenseUVPolynomial, EvaluationDomain, 
        Evaluations, GeneralEvaluationDomain
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

        let evals_over_domain_g: Vec<_> = domain
            .elements()
            .map(|f| g.evaluate(&f))
            .collect();

        let evals_over_domain_h: Vec<_> = domain
            .elements()
            .map(|f| h.evaluate(&f))
            .collect();

        let evals_over_domain_s: Vec<_> = domain
            .elements()
            .map(|f| s.evaluate(&f))
            .collect();

        let evals_over_domain_o: Vec<_> = domain
            .elements()
            .map(|f| {
                g.evaluate(&f) * h.evaluate(&f) * s.evaluate(&f) 
                + (Fr::one() - s.evaluate(&f)) * (g.evaluate(&f) + h.evaluate(&f))
            })
            .collect();

        let g_evals = Evaluations::from_vec_and_domain(
            evals_over_domain_g, 
            domain            
        );

        let h_evals = Evaluations::from_vec_and_domain(
            evals_over_domain_h, 
            domain
        );

        let s_evals = Evaluations::from_vec_and_domain(
            evals_over_domain_s, 
            domain
        );

        let o_evals = Evaluations::from_vec_and_domain(
            evals_over_domain_o, 
            domain
        );

        let mut inp_evals = vec![];
        inp_evals.push(g_evals);
        inp_evals.push(h_evals);
        inp_evals.push(s_evals);
        inp_evals.push(o_evals);

        let proof_gen_timer = start_timer!(|| "Prove fn called for g, h, zero_domain");

        let proof = 
            OptimizedUnivariateZeroCheck::<Fr, Bls12_381>::prove(inp_evals.clone(), domain).unwrap();

        end_timer!(proof_gen_timer);
        
        println!("Proof Generated");

        let verify_timer = start_timer!(|| "Verify fn called for g, h, zero_domain, proof");

        let result = OptimizedUnivariateZeroCheck::<Fr, Bls12_381>
            ::verify(inp_evals, proof, domain)
            .unwrap();

        end_timer!(verify_timer);

        println!("verification result: {:?}", result);
        assert_eq!(result, true);

        end_timer!(test_timer);
    }
}