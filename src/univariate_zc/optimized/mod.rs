use std::marker::PhantomData;
use anyhow::Ok;
use ark_ff::{FftField, PrimeField};
use ark_poly::EvaluationDomain;
use ark_poly::{
    univariate::DensePolynomial, 
    Evaluations, GeneralEvaluationDomain, Polynomial,
    DenseUVPolynomial
};
use ark_ec::pairing::Pairing;
use ark_poly_commit::kzg10::{KZG10, Powers, VerifierKey};
use ark_std::test_rng;
use ark_ff::One;
use ark_ec::AffineRepr;
use ark_ff::Zero;

use crate::ZeroCheck;
use crate::utils::*;

pub mod data_structures;
use data_structures::*;

/// Optimized Zero-Check protocol for if a polynomial
/// f = g*h*s + (1 - s)(g + h) evaluates to 0 
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

        // compute the polynomials corresponding to g, h, and s using interpolation (IFFT)
        let g_poly = g.clone().interpolate();
        let h_poly = h.clone().interpolate();
        let s_poly = s.clone().interpolate();

        let g_deg = g_poly.degree();
        let h_deg = h_poly.degree();
        let s_deg = s_poly.degree();

        // compute the vanishing polynomial of the zero domain
        // let z_poly = zero_domain.vanishing_polynomial();

        // let z_deg = z_poly.degree();

        // compute degree of quotient polynomial to 
        let f_deg = g_deg + h_deg + s_deg;
        // let q_deg = f_deg - z_deg;

        println!("degree of f: {:?}", f_deg);

        // compute polynomial commitments to input polynomials
        let rng = &mut test_rng();
        let params = KZG10::<E, DensePolynomial<E::ScalarField>>::setup(
            2 * f_deg, 
            false, 
            rng
        ).expect("PCS setup failed");

        let vk: VerifierKey<E> = VerifierKey {
            g: params.powers_of_g[0],
            gamma_g: params.powers_of_gamma_g[&0],
            h: params.h,
            beta_h: params.beta_h,
            prepared_h: params.prepared_h.clone(),
            prepared_beta_h: params.prepared_beta_h.clone(),
        };

        let powers_of_g = params.powers_of_g[..= 2 * f_deg].to_vec();
        let powers_of_gamma_g = (0..= 2 * f_deg)
           .map(|i| params.powers_of_gamma_g[&i])
            .collect();
        let powers = Powers {
            powers_of_g: ark_std::borrow::Cow::Owned(powers_of_g),
            powers_of_gamma_g: ark_std::borrow::Cow::Owned(powers_of_gamma_g),
        };

        let (comm_g, r_g) = KZG10::<E, DensePolynomial<E::ScalarField>>::commit(
            &powers, 
            &g_poly, 
            None, 
            None
        ).expect("Commitment to polynomail g(X) failed");

        assert!(!comm_g.0.is_zero(), "Commitment should not be zero");
        assert!(!r_g.is_hiding(), "Commitment should not be hiding");

        let (comm_h, r_h) = KZG10::<E, DensePolynomial<E::ScalarField>>::commit(
            &powers, 
            &h_poly, 
            None, 
            None
        ).expect("Commitment to polynomail h(X) failed");

        assert!(!comm_h.0.is_zero(), "Commitment should not be zero");
        assert!(!r_h.is_hiding(), "Commitment should not be hiding");

        let (comm_s, r_s) = KZG10::<E, DensePolynomial<E::ScalarField>>::commit(
            &powers, 
            &s_poly, 
            None, 
            None
        ).expect("Commitment to polynomail s(X) failed");

        assert!(!comm_s.0.is_zero(), "Commitment should not be zero");
        assert!(!r_s.is_hiding(), "Commitment should not be hiding");

        let mut inp_comms = vec![];
        inp_comms.push(comm_g);
        inp_comms.push(comm_h);
        inp_comms.push(comm_s);

        let mut inp_rand = h.evals;
        inp_rand.extend(s.evals);
        let r = get_randomness(g.evals ,inp_rand)[0];

        let mut inp_evals_at_rand = vec![];
        inp_evals_at_rand.push(g_poly.evaluate(&r));
        inp_evals_at_rand.push(h_poly.evaluate(&r));
        inp_evals_at_rand.push(s_poly.evaluate(&r));

        let mut inp_opening_proofs = vec![];
        
        let g_opening_proof = Self::PCS::open(
            &powers,
            &g_poly,
            r,
            &r_g
        ).expect("Proof generation failed for g(X)");

        let h_opening_proof = Self::PCS::open(
            &powers,
            &h_poly,
            r,
            &r_h
        ).expect("Proof generation failed for h(X)");

        let s_opening_proof = Self::PCS::open(
            &powers,
            &s_poly,
            r,
            &r_s
        ).expect("Proof generation failed for s(X)");

        inp_opening_proofs.push(g_opening_proof);
        inp_opening_proofs.push(h_opening_proof);
        inp_opening_proofs.push(s_opening_proof);

        // compute quotient polynomial q(X)
        let unit_poly = DensePolynomial::from_coefficients_vec(
            vec![<E::ScalarField>::one()]
        );

        let f_poly = &g_poly * &h_poly * &s_poly + &(&unit_poly - &s_poly) * &(&g_poly + &h_poly);
        let (q_poly, r_poly) = 
            f_poly
            .divide_by_vanishing_poly(zero_domain);

        // If f evaluates to 0 over the `zero_domain`
        // the vanishing polynomial perfect divides f with remainder r = 0
        assert!(r_poly.is_zero()); 

        let (comm_q, r_q) = KZG10::<E, DensePolynomial<E::ScalarField>>::commit(
            &powers, 
            &q_poly, 
            None, 
            None
        ).expect("Commitment to polynomail q(X) failed");

        assert!(!comm_q.0.is_zero(), "Commitment should not be zero");
        assert!(!r_q.is_hiding(), "Commitment should not be hiding");

        let q_opening_proof = Self::PCS::open(
            &powers,
            &q_poly,
            r,
            &r_q
        ).expect("Proof generation failed for q(X)");

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
    /// g - input polynomial evalutions
    /// h - input polynomial evalutions
    /// s - input polynomial evalutions
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

        let q_comm = proof.q_comm;
        let inp_comms = proof.inp_comms;
        let vk = proof.vk;
        let inp_openings = proof.inp_openings;
        let inp_evals = proof.inp_evals;
        let q_opening = proof.q_opening;
        let q_eval = proof.q_eval;

        let mut inp_rand = h.evals;
        inp_rand.extend(s.evals);
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

        let lhs = q_eval * zero_domain.evaluate_vanishing_polynomial(r);
        let rhs = a * b * c + (<E::ScalarField>::one() - c) * (a + b);

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

        let domain_g = GeneralEvaluationDomain::<Fr>::new(1 << 10).unwrap();
        let domain_h = GeneralEvaluationDomain::<Fr>::new(1 << 10).unwrap();
        let domain_s = GeneralEvaluationDomain::<Fr>::new(1 << 10).unwrap();

        let zero_domain = GeneralEvaluationDomain::<Fr>::new(1 << 7).unwrap();

        let deg_z = 1 << 7;

        let rng = &mut ark_std::test_rng();

        let mut rand_g_coeffs = vec![];
        let mut rand_h_coeffs = vec![];
        let mut rand_s_coeffs = vec![];

        for _ in 1..((1 << 10) - deg_z) {
            rand_g_coeffs.push(Fr::rand(rng));
            rand_h_coeffs.push(Fr::rand(rng));
            rand_s_coeffs.push(Fr::rand(rng));
        }

        let random_poly_g = DensePolynomial::from_coefficients_vec(rand_g_coeffs);
        let random_poly_h = DensePolynomial::from_coefficients_vec(rand_h_coeffs);
        let random_poly_s = DensePolynomial::from_coefficients_vec(rand_s_coeffs);

        let evals_over_domain_g: Vec<_> = domain_g
            .elements()
            .map(|f| (zero_domain.evaluate_vanishing_polynomial(f) * random_poly_g.evaluate(&f)))
            .collect();

        let evals_over_domain_h: Vec<_> = domain_h
            .elements()
            .map(|f| (zero_domain.evaluate_vanishing_polynomial(f) * random_poly_h.evaluate(&f)))
            .collect();

        let evals_over_domain_s: Vec<_> = domain_s
            .elements()
            .map(|f| (zero_domain.evaluate_vanishing_polynomial(f) * random_poly_s.evaluate(&f)))
            .collect();

        let g_evals = Evaluations::from_vec_and_domain(
            evals_over_domain_g, 
            domain_g            
        );

        let h_evals = Evaluations::from_vec_and_domain(
            evals_over_domain_h, 
            domain_h
        );

        let s_evals = Evaluations::from_vec_and_domain(
            evals_over_domain_s, 
            domain_s
        );

        let mut inp_evals = vec![];
        inp_evals.push(g_evals);
        inp_evals.push(h_evals);
        inp_evals.push(s_evals);

        let proof_gen_timer = start_timer!(|| "Prove fn called for g, h, zero_domain");

        let proof = 
            OptimizedUnivariateZeroCheck::<Fr, Bls12_381>::prove(inp_evals.clone(), zero_domain).unwrap();

        end_timer!(proof_gen_timer);
        
        println!("Proof Generated");

        let verify_timer = start_timer!(|| "Verify fn called for g, h, zero_domain, proof");

        let result = OptimizedUnivariateZeroCheck::<Fr, Bls12_381>
            ::verify(inp_evals, proof, zero_domain)
            .unwrap();

        end_timer!(verify_timer);

        println!("verification result: {:?}", result);
        assert_eq!(result, true);

        end_timer!(test_timer);
    }
}