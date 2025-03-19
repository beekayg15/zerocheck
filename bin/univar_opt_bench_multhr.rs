use ark_bls12_381::{Bls12_381, Fr};
use ark_ff::UniformRand;
use ark_poly::{
    univariate::DensePolynomial, DenseUVPolynomial, EvaluationDomain, Evaluations,
    GeneralEvaluationDomain, Polynomial,
};
use ark_std::One;
use ark_std::{end_timer, start_timer};
use clap::Parser;
use rayon::prelude::*;
use std::time::Instant;
use zerocheck::univariate_zc::optimized::data_structures::{InputParams, ZeroCheckParams};
use zerocheck::univariate_zc::optimized::OptimizedUnivariateZeroCheck;
use zerocheck::ZeroCheck;

/// This function prepares the random input evaluations for the prover test.
/// Reuse for the same worksize across multiple repeated tests.
fn prepare_input_evals_domain(
    size: u32,
) -> (
    [Evaluations<Fr>; 4],
    GeneralEvaluationDomain<Fr>,
    ZeroCheckParams<Bls12_381>,
) {
    println!("Preparing input evaluations and domain for 2^{size} work");
    let instant = Instant::now();
    let domain = GeneralEvaluationDomain::<Fr>::new(1 << size).unwrap();

    let rand_g_coeffs: Vec<_> = (0..(1 << size))
        .into_par_iter()
        .map(|_| Fr::rand(&mut ark_std::test_rng()))
        .collect();
    let rand_h_coeffs: Vec<_> = (0..(1 << size))
        .into_par_iter()
        .map(|_| Fr::rand(&mut ark_std::test_rng()))
        .collect();
    let rand_s_coeffs: Vec<_> = (0..(1 << size))
        .into_par_iter()
        .map(|_| Fr::rand(&mut ark_std::test_rng()))
        .collect();

    let g = DensePolynomial::from_coefficients_vec(rand_g_coeffs);
    let h = DensePolynomial::from_coefficients_vec(rand_h_coeffs);
    let s = DensePolynomial::from_coefficients_vec(rand_s_coeffs);

    let max_degree = g.degree() + s.degree() + h.degree();
    let pp = InputParams { max_degree };
    let global_params = OptimizedUnivariateZeroCheck::<Bls12_381>::setup(pp).unwrap();

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

    let inp_evals = [g_evals, h_evals, s_evals, o_evals];
    let duration = instant.elapsed().as_secs_f64();
    println!("Preparing input evaluations and domain for 2^{size} work ....{duration}s");
    return (inp_evals, domain, global_params);
}

/// Benchmark function for the optimized univariate zero check proof generation and verification.
/// `inp_evals` is the input evaluations of g, h, s, o.
/// `domain` is the domain of the evaluations.
/// `size` is the work size exponent (2^size).
fn opt_univariate_zero_check_multithread_benchmark(
    input_evals: &[Evaluations<Fr>; 4],
    domain: GeneralEvaluationDomain<Fr>,
    global_params: ZeroCheckParams<Bls12_381>,
    size: u32,
) -> u128 {
    let test_timer =
        start_timer!(|| format!("Opt Univariate Proof Generation Test for 2^{size} work"));

    let inp_evals = input_evals.to_vec();
    let instant = Instant::now();
    let proof_gen_timer = start_timer!(|| "Prove fn called for g, h, zero_domain");

    let proof = OptimizedUnivariateZeroCheck::<Bls12_381>::prove(
        global_params.clone(),
        inp_evals.clone(),
        domain,
    )
    .unwrap();

    end_timer!(proof_gen_timer);
    let runtime = instant.elapsed();

    // println!("Proof Generated");

    let verify_timer = start_timer!(|| "Verify fn called for g, h, zero_domain, proof");

    let result = OptimizedUnivariateZeroCheck::<Bls12_381>::verify(
        global_params,
        inp_evals,
        proof,
        domain,
    )
    .unwrap();

    end_timer!(verify_timer);

    // println!("verification result: {:?}", result);
    assert_eq!(result, true);

    end_timer!(test_timer);
    return runtime.as_millis();
}

#[derive(Parser, Debug)]
struct Args {
    /// Number of repetitions for each test
    #[arg(long, default_value = "10")]
    repeat: u32,

    /// Minimum work size exponent (2^min_size)
    #[arg(long, default_value = "10")]
    min_size: u32,

    /// Maximum work size exponent (inclusive, 2^max_size)
    #[arg(long, default_value = "20")]
    max_size: u32,

    /// Number of threads to use for prepare input evaluations
    #[arg(long, default_value = "64")]
    prepare_threads: u32,

    /// Number of threads to use to run proof and verify tests
    #[arg(long, default_value = "1")]
    run_threads: u32,
}

fn bench_opt_uni_zc() {
    let args = Args::parse();

    let work_sizes = args.min_size..=args.max_size; // 2 ^ max_size_size

    let (s, tt): (Vec<u32>, Vec<u128>) = (work_sizes)
        .map(|size| {
            let pool_prepare = rayon::ThreadPoolBuilder::new()
                .num_threads(args.prepare_threads as usize)
                .build()
                .unwrap();
            let (input_evals, domain, pp) = pool_prepare.install(|| prepare_input_evals_domain(size));

            let pool_run = rayon::ThreadPoolBuilder::new()
                .num_threads(args.run_threads as usize)
                .build()
                .unwrap();
            let total_runtime: u128 = pool_run.install(|| {
                (0..args.repeat)
                    .map(|repeat_time| {
                        println!("Running test for 2^{} with repeat: {}", size, repeat_time);
                        opt_univariate_zero_check_multithread_benchmark(&input_evals, domain, pp.clone(), size)
                    })
                    .sum()
            });

            (size, total_runtime)
        })
        .unzip();
    println!("size {:?}, total_runtime {:?}", s, tt);
}

fn main() {
    bench_opt_uni_zc();
}
