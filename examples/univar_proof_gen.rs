use zerocheck::{univariate_zc::naive::*, univariate_zc::optimized::*, ZeroCheck};

use ark_bls12_381::Bls12_381;
use ark_bls12_381::Fr;
use ark_ff::UniformRand;
use ark_poly::{
    univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Evaluations,
    GeneralEvaluationDomain, Polynomial,
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

    let g_evals = Evaluations::from_vec_and_domain(evals_over_domain_g, domain_g);

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

fn eg_univar_proof_generation_commit() {
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

    let g_evals = Evaluations::from_vec_and_domain(evals_over_domain_g, domain_g);

    let h_evals = Evaluations::from_vec_and_domain(evals_over_domain_h, domain_h);

    let s_evals = Evaluations::from_vec_and_domain(evals_over_domain_s, domain_s);

    let mut inp_evals = vec![];
    inp_evals.push(g_evals);
    inp_evals.push(h_evals);
    inp_evals.push(s_evals);

    let proof_gen_timer = start_timer!(|| "Prove fn called for g, h, zero_domain");

    let proof =
        OptimizedUnivariateZeroCheck::<Fr, Bls12_381>::prove(inp_evals.clone(), zero_domain)
            .unwrap();

    end_timer!(proof_gen_timer);

    println!("Proof Generated");

    let verify_timer = start_timer!(|| "Verify fn called for g, h, zero_domain, proof");

    let result =
        OptimizedUnivariateZeroCheck::<Fr, Bls12_381>::verify(inp_evals, proof, zero_domain)
            .unwrap();

    end_timer!(verify_timer);

    println!("verification result: {:?}", result);
    assert_eq!(result, true);

    end_timer!(test_timer);
}

fn main() {
    // eg_univar_proof_generation();
    eg_univar_proof_generation_commit();
}
