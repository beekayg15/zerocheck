use zerocheck::{ZeroCheck, univariate_zc::naive::*};

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

fn eg_univar_proof_generation() {
    let test_timer = start_timer!(|| "Proof Generation Test");

    let domain_g = GeneralEvaluationDomain::<Fr>::new(1 << 15).unwrap();
    let domain_h = GeneralEvaluationDomain::<Fr>::new(1 << 15).unwrap();

    let zero_domain = GeneralEvaluationDomain::<Fr>::new(1 << 5).unwrap();

    println!("domain size of g: {:?}", domain_g.size());
    println!("domain size of zero_domain: {:?}", zero_domain.size());

    let evals_over_domain_g: Vec<_> = domain_g
        .elements()
        .map(|f| zero_domain.evaluate_vanishing_polynomial(f))
        .collect();

    let g_evals = Evaluations::from_vec_and_domain(
        evals_over_domain_g, 
        domain_g
    );

    let mut rand_coeffs = vec![];

    let rng = &mut ark_std::test_rng();
    for _ in 1..(1 << 8) {
        rand_coeffs.push(Fr::rand(rng));
    }

    let random_poly = DensePolynomial::from_coefficients_vec(rand_coeffs);

    let h_evals = random_poly.evaluate_over_domain(domain_h);

    let mut inp_evals = vec![];
    inp_evals.push(g_evals);
    inp_evals.push(h_evals);

    let proof_gen_timer = start_timer!(|| "Prove fn called for g, h, zero_domain");

    let proof = 
        NaiveUnivariateZeroCheck::<Fr, Bls12_381>::prove(inp_evals.clone(), zero_domain).unwrap();

    end_timer!(proof_gen_timer);
    
    println!("Proof Generated: {:?}", proof);
    end_timer!(test_timer);
}

fn main() {
    eg_univar_proof_generation();
}

