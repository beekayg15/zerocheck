use zerocheck::univariate_zc::optimized::data_structures::InputParams;
use zerocheck::{
    // univariate_zc::naive::*, 
    univariate_zc::optimized::*, 
    ZeroCheck
};

use ark_bls12_381::Bls12_381;
use ark_bls12_381::Fr;
use ark_ff::{One, UniformRand};
use ark_poly::{
    univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Evaluations,
    GeneralEvaluationDomain, Polynomial,
};
use ark_std::end_timer;
use ark_std::start_timer;

// fn eg_univar_proof_generation() {
//     let test_timer = start_timer!(|| "Proof Generation Test");

//     let domain_g = GeneralEvaluationDomain::<Fr>::new(1 << 15).unwrap();
//     let domain_h = GeneralEvaluationDomain::<Fr>::new(1 << 15).unwrap();

//     let zero_domain = GeneralEvaluationDomain::<Fr>::new(1 << 5).unwrap();

//     println!("domain size of g: {:?}", domain_g.size());
//     println!("domain size of zero_domain: {:?}", zero_domain.size());

//     let evals_over_domain_g: Vec<_> = domain_g
//         .elements()
//         .map(|f| zero_domain.evaluate_vanishing_polynomial(f))
//         .collect();

//     let g_evals = Evaluations::from_vec_and_domain(evals_over_domain_g, domain_g);

//     let mut rand_coeffs = vec![];

//     let rng = &mut ark_std::test_rng();
//     for _ in 1..(1 << 8) {
//         rand_coeffs.push(Fr::rand(rng));
//     }

//     let random_poly = DensePolynomial::from_coefficients_vec(rand_coeffs);

//     let h_evals = random_poly.evaluate_over_domain(domain_h);

//     let mut inp_evals = vec![];
//     inp_evals.push(g_evals);
//     inp_evals.push(h_evals);

//     let proof_gen_timer = start_timer!(|| "Prove fn called for g, h, zero_domain");

//     let proof =
//         NaiveUnivariateZeroCheck::<Fr, Bls12_381>::prove(inp_evals.clone(), zero_domain).unwrap();

//     end_timer!(proof_gen_timer);

//     println!("Proof Generated: {:?}", proof);
//     end_timer!(test_timer);
// }

fn eg_univar_proof_generation_commit() {
    let test_timer = start_timer!(|| "Proof Generation Test");

    let deg_size = 10;
    let domain = GeneralEvaluationDomain::<Fr>::new(1 << deg_size).unwrap();
    // for i in domain.elements() {
    //     println!("domain element: {:?}", i);
    // }

    let rng = &mut ark_std::test_rng();

    let mut rand_g_coeffs = vec![];
    let mut rand_h_coeffs = vec![];
    let mut rand_s_coeffs = vec![];

    for _ in 0..(1 << deg_size) {
        rand_g_coeffs.push(Fr::rand(rng));
        rand_h_coeffs.push(Fr::rand(rng));
        rand_s_coeffs.push(Fr::rand(rng));
    }

    // set g, h, s as random coefficients polynomials
    let g = DensePolynomial::from_coefficients_vec(rand_g_coeffs);
    let h = DensePolynomial::from_coefficients_vec(rand_h_coeffs);
    let s = DensePolynomial::from_coefficients_vec(rand_s_coeffs);

    // ploynomial evaluations over the domain, done by Horner's method O(n)
    let evals_over_domain_g: Vec<Fr> = domain
        .elements()
        .map(|f|
            g.evaluate(&f))
        .collect();
    let evals_over_domain_h: Vec<Fr> = domain.elements().map(|f| h.evaluate(&f)).collect();
    let evals_over_domain_s: Vec<Fr> = domain.elements().map(|f| s.evaluate(&f)).collect();
    let evals_over_domain_o: Vec<Fr> = domain
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

    let max_degree = g.degree() + s.degree() + h.degree();
        let pp = InputParams{
            max_degree,
        };

    let zp = OptimizedUnivariateZeroCheck::<Bls12_381>::setup(pp).unwrap();

    let proof_gen_timer = start_timer!(|| "Prove fn called for g, h, zero_domain");

    let proof =
        OptimizedUnivariateZeroCheck::<Bls12_381>::prove(
            zp.clone(),
            inp_evals.clone(), 
            domain
        ).unwrap();

    end_timer!(proof_gen_timer);

    println!("Proof Generated");

    let verify_timer = start_timer!(|| "Verify fn called for g, h, zero_domain, proof");

    let result =
        OptimizedUnivariateZeroCheck::<Bls12_381>::verify(
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

fn main() {
    // eg_univar_proof_generation();
    eg_univar_proof_generation_commit();
}
