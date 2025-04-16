use ark_poly::{EvaluationDomain, Evaluations, GeneralEvaluationDomain, Polynomial};
use ark_std::marker::PhantomData;
use ark_ff::{
    FftField, PrimeField
};
use ark_ff::Zero;
use anyhow::{Error, Ok};
use ark_std::{end_timer, start_timer};

use crate::transcripts::ZCTranscript;
use crate::ZeroCheck;
use crate::utils::*;

mod data_structures;
pub use data_structures::*;

/// Naive Zero-Check protocol for if a polynomial
/// f = g^2*h evaluates to 0 over a specific domain
/// 
/// The inputs g and h are provided as evaluations
/// and the proof that polynomial f computed as 
/// mentioned above evaluates to zero, can be given
/// by proving the existence of a quotient polynomial
/// q, S.T. f(X) = q(X).z_H(X), where z_H(X) is the 
/// vanishing polynomial over the zero domain H.

#[derive(Clone)]
pub struct NaiveUnivariateZeroCheck<F: PrimeField + FftField> {
    _field_data: PhantomData<F>,
}

/// Zero-Check protocol for univariate polynomials in which the 
/// input polynomials are provided as evalution of the circuit of
/// different inputs, and ZeroDomain is a GeneralEvaluationDomain
impl<F: PrimeField + FftField> ZeroCheck<F> for NaiveUnivariateZeroCheck<F> {
    type InputType = Vec<Evaluations<F>>;
    type Proof = Proof<F>;
    type ZeroDomain = GeneralEvaluationDomain<F>;
    type ZeroCheckParams<'a> = ZeroCheckParams<F>;
    type InputParams = Option<F>;
    type Transcripts = Option<ZCTranscript<F>>;

    fn setup<'a>(
        _pp: &Self::InputParams
    ) -> Result<ZeroCheckParams<F>, Error> {
        let zp = ZeroCheckParams {
            _field_data: PhantomData::<F>,
        };

        Ok(zp)
    }

    /// function called by the prover to genearte a valid
    /// proof for zero-check protocol
    /// 
    /// Attributes:
    /// g - input polynomial evaluations
    /// h - input polynomial evaluations
    /// zero_domain - domain over which the resulting polynomial evaluates to 0
    /// 
    /// Returns
    /// Proof - valid proof for the zero-check protocol
    fn prove<'a> (
            _zero_params: &Self::ZeroCheckParams<'_>,
            input_poly: &Self::InputType,
            zero_domain: &Self::ZeroDomain,
            _transcript: &mut Self::Transcripts,
            _run_threads: Option<usize>,
            _batch_commit_threads: Option<usize>,
            _batch_open_threads: Option<usize>,
        ) -> Result<Self::Proof, Error> {

        let inp_interpolation_time = start_timer!(|| "Interpolating input evaluations");

        let g = input_poly[0].clone();
        let h = input_poly[1].clone();

        // compute the polynomials corresponding to g and h using interpolation (IFFT)
        let g_poly = g.interpolate();
        let h_poly = h.interpolate();

        // println!("deg_g_poly: {:?}", g_poly.degree());
        // println!("deg_h_poly: {:?}", h_poly.degree());

        end_timer!(inp_interpolation_time);

        let f_poly_timer =  start_timer!(|| "interpolating poly. f = g^2 * h");

        // compute the resulting polynomial f = g^2 * h
        let f_poly = &(&g_poly * &g_poly) * &h_poly;

        end_timer!(f_poly_timer);

        let q_poly_timer = start_timer!(|| "Finding quotient polynomail q");
        
        // compute the quotient polynomial q, by dividing f by 
        // the vanishing polynomial over domain `zero_domain`
        let (q_poly, r_poly) = 
            f_poly
            .divide_by_vanishing_poly(*zero_domain);

        end_timer!(q_poly_timer);

        // If f evaluates to 0 over the `zero_domain`
        // the vanishing polynomial perfect divides f with remainder r = 0
        assert!(r_poly.is_zero()); 

        // send the quotient polynomial and the resulting polynomial proof
        let proof = Proof{
            q: q_poly,
            f: f_poly
        };

        Ok(proof)
    }

    /// function called by the verifier to check if the proof for the 
    /// zero-check protocol is valid
    /// 
    /// Attributes:
    /// g - input polynomial evalutions
    /// h - input polynomial evalutions
    /// zero_domain - domain over which the resulting polynomial evaluates to 0
    /// proof - proof sent by the prover for the claim
    /// 
    /// Returns
    /// 'true' if the proof is valid, 'false' otherwise
    fn verify<'a> (
            _zero_params: &Self::ZeroCheckParams<'_>,
            input_poly: &Self::InputType,
            proof: &Self::Proof,
            zero_domain: &Self::ZeroDomain,
            _transcript: &mut Self::Transcripts,
        ) -> Result<bool, anyhow::Error> {
        let g = input_poly[0].clone();
        let h = input_poly[1].clone();

        let q_poly = &proof.q;
        let f_poly = &proof.f;

        // compute the vanishing polynomial over the `zero_domain`
        let z_poly = zero_domain.vanishing_polynomial();

        // sample a random point
        let r = get_randomness(q_poly.coeffs.clone(), f_poly.coeffs.clone());
           
        // evaluate f, z_H, and q at r
        let rhs = q_poly.evaluate(&r[0]) * z_poly.evaluate(&r[0]);
        let lhs = f_poly.evaluate(&r[0]);

        // sample a random evaluation from g and h
        let rand_index = get_random_indices(
            1, 
            q_poly.coeffs.clone(), 
            f_poly.coeffs.clone(), 
            g.evals.len()
        )[0];

        let f_at_rand_index = f_poly.evaluate(&g.domain().element(rand_index));
        let g_sq_times_h = g.evals[rand_index] * g.evals[rand_index] * h.evals[rand_index];

        // check if q(r).z_H(r) == f(r) and if g[rand_index]^2 * h[rand_index] = f(rand_index)
        if lhs == rhs && g_sq_times_h == f_at_rand_index{
            return Ok(true);
        }

        return Ok(false);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::UniformRand;
    use ark_poly::{
        univariate::DensePolynomial, 
        DenseUVPolynomial, EvaluationDomain, 
        Evaluations, GeneralEvaluationDomain
    };
    use ark_std::end_timer;
    use ark_std::start_timer;

    #[test]
    fn test_proof_generation_naive_uni() {
        let test_timer = start_timer!(|| "Proof Generation Test");

        let domain = GeneralEvaluationDomain::<Fr>::new(1 << 15).unwrap();

        // println!("domain size of g: {:?}", domain_g.size());
        // println!("domain size of zero_domain: {:?}", zero_domain.size());

        let evals_over_domain_g: Vec<_> = domain
            .elements()
            .map(|f| domain.evaluate_vanishing_polynomial(f))
            .collect();

        let g_evals = Evaluations::from_vec_and_domain(
            evals_over_domain_g, 
            domain
        );

        let mut rand_coeffs = vec![];

        let rng = &mut ark_std::test_rng();
        for _ in 1..(1 << 15) {
            rand_coeffs.push(Fr::rand(rng));
        }

        let random_poly = DensePolynomial::from_coefficients_vec(rand_coeffs);

        let h_evals = random_poly.evaluate_over_domain(domain);

        let zero_params = 
            NaiveUnivariateZeroCheck::<Fr>::setup(
                &None
            ).unwrap();

        let mut inp_evals = vec![];
        inp_evals.push(g_evals);
        inp_evals.push(h_evals);

        let proof_gen_timer = start_timer!(|| "Prove fn called for g, h, zero_domain");

        let _proof = 
            NaiveUnivariateZeroCheck::<Fr>::prove(
                &zero_params,
                &inp_evals.clone(), 
                &domain,
                &mut None,
                None,
                None,
                None
            ).unwrap();

        end_timer!(proof_gen_timer);
        
        //println!("Proof Generated: {:?}", proof);
        
        println!("Proof Generated");
        end_timer!(test_timer);
    }

    #[test]
    fn test_proof_validation_naive_uni() {
        let test_timer = start_timer!(|| "Proof Generation Test");

        let domain = GeneralEvaluationDomain::<Fr>::new(1 << 15).unwrap();

        let evals_over_domain_g: Vec<_> = domain
            .elements()
            .map(|f| domain.evaluate_vanishing_polynomial(f))
            .collect();

        let g_evals = Evaluations::from_vec_and_domain(
            evals_over_domain_g, 
            domain
        );

        let mut rand_coeffs = vec![];

        let rng = &mut ark_std::test_rng();
        for _ in 1..(1 << 15) {
            rand_coeffs.push(Fr::rand(rng));
        }

        let random_poly = DensePolynomial::from_coefficients_vec(rand_coeffs);

        let h_evals = random_poly.evaluate_over_domain(domain);

        let mut inp_evals = vec![];
        inp_evals.push(g_evals);
        inp_evals.push(h_evals);

        let zero_params = 
            NaiveUnivariateZeroCheck::<Fr>::setup(
                &None
            ).unwrap();

        let proof_gen_timer = start_timer!(|| "Prove fn called for g, h, zero_domain");

        let proof = 
            NaiveUnivariateZeroCheck::<Fr>::prove(
                &zero_params.clone(),
                &inp_evals.clone(), 
                &domain,
                &mut None,
                None,
                None,
                None
            ).unwrap();

        end_timer!(proof_gen_timer);
        
        // println!("Proof Generated: {:?}", proof);

        let verify_timer = start_timer!(|| "Verify fn called for g, h, zero_domain, proof");

        let result = NaiveUnivariateZeroCheck::<Fr>::verify(
            &zero_params,
            &inp_evals, 
            &proof, 
            &domain,
            &mut None
        ).unwrap();

        end_timer!(verify_timer);

        println!("verification result: {:?}", result);
        assert_eq!(result, true);

        end_timer!(test_timer);
    }
}