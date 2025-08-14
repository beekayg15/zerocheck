use ark_bls12_381::{Bls12_381, Fr};
use ark_poly::{EvaluationDomain, GeneralEvaluationDomain};
use ark_std::{end_timer, start_timer};
use clap::Parser;
use std::iter::zip;
use std::time::Instant;
use zerocheck::transcripts::ZCTranscript;
use zerocheck::zc::univariate_zc::custom::{
    data_structures::{VirtualEvaluation, ZeroCheckParams},
    CustomUnivariateZeroCheck,
};
use zerocheck::ZeroCheck;
use zerocheck::{
    pcs::univariate_pcs::{
        kzg::KZG,
        ligero::{Ligero, LigeroPoseidon},
    },
    zc::univariate_zc::custom::parser::prepare_zero_virtual_evaluation_from_string,
};

/// This function prepares the random input evaluations for the prover test.
/// Reuse for the same worksize across multiple repeated tests.
fn prepare_input_evals_domain<'a>(
    size: usize,
    pool_prepare: &rayon::ThreadPool,
    intput_poly: String,
) -> (VirtualEvaluation<Fr>, GeneralEvaluationDomain<Fr>, usize) {
    println!("Preparing input evaluations and domain for 2^{size} work");
    let instant = Instant::now();

    let degree = 1 << size;
    let inp_evals = prepare_zero_virtual_evaluation_from_string(&intput_poly, degree, &pool_prepare).unwrap();
    let domain = GeneralEvaluationDomain::<Fr>::new(degree).unwrap();

    let max_degree = inp_evals.evals_info.max_multiplicand * degree;
    let duration = instant.elapsed().as_secs_f64();
    println!("Preparing input evaluations and domain for 2^{size} work ....{duration}s");
    return (inp_evals, domain, max_degree);
}

/// Benchmark function for the optimized univariate zero check proof generation and verification.
/// `inp_evals` is the input evaluations of g, h, s, o.
/// `domain` is the domain of the evaluations.
/// `size` is the work size exponent (2^size).
fn opt_univ_zc_multhr_benchmark_kzg(
    input_evals: &VirtualEvaluation<Fr>,
    domain: GeneralEvaluationDomain<Fr>,
    global_params: &ZeroCheckParams<KZG<Bls12_381>>,
    size: u32,
    run_threads: Option<usize>,
    batch_commit_threads: Option<usize>,
    batch_open_threads: Option<usize>,
) -> u128 {
    let test_timer =
        start_timer!(|| format!("Opt Univariate Proof Generation Test KZG for 2^{size} work"));

    let instant = Instant::now();
    let proof_gen_timer = start_timer!(|| "Prove fn called for KZG");

    let proof = CustomUnivariateZeroCheck::<Fr, KZG<Bls12_381>>::prove(
        &global_params,
        input_evals,
        &domain,
        &mut ZCTranscript::init_transcript(),
        run_threads,
        batch_commit_threads,
        batch_open_threads,
    )
    .unwrap();

    end_timer!(proof_gen_timer);
    let runtime = instant.elapsed();

    let verify_timer = start_timer!(|| "Verify fn called for KZG");

    let result = CustomUnivariateZeroCheck::<Fr, KZG<Bls12_381>>::verify(
        &global_params,
        input_evals,
        &proof,
        &domain,
        &mut ZCTranscript::init_transcript(),
    )
    .unwrap();

    end_timer!(verify_timer);

    assert_eq!(result, true);

    end_timer!(test_timer);
    return runtime.as_millis();
}

fn opt_univ_zc_multhr_benchmark_ligero(
    input_evals: &VirtualEvaluation<Fr>,
    domain: GeneralEvaluationDomain<Fr>,
    global_params: &ZeroCheckParams<Ligero<Fr>>,
    size: u32,
    run_threads: Option<usize>,
    batch_commit_threads: Option<usize>,
    batch_open_threads: Option<usize>,
) -> u128 {
    let test_timer = start_timer!(|| {
        format!("Opt Univariate Proof Generation Test Ligero for 2^{size} work")
    });

    let instant = Instant::now();
    let proof_gen_timer = start_timer!(|| "Prove fn called for Ligero");

    let proof = CustomUnivariateZeroCheck::<Fr, Ligero<Fr>>::prove(
        &global_params,
        input_evals,
        &domain,
        &mut ZCTranscript::init_transcript(),
        run_threads,
        batch_commit_threads,
        batch_open_threads,
    )
    .unwrap();

    end_timer!(proof_gen_timer);
    let runtime = instant.elapsed();

    let verify_timer = start_timer!(|| "Verify fn called for Ligero");

    let result = CustomUnivariateZeroCheck::<Fr, Ligero<Fr>>::verify(
        &global_params,
        input_evals,
        &proof,
        &domain,
        &mut ZCTranscript::init_transcript(),
    )
    .unwrap();

    end_timer!(verify_timer);

    assert_eq!(result, true);

    end_timer!(test_timer);
    return runtime.as_millis();
}

fn opt_univ_zc_multhr_benchmark_ligero_poseidon(
    input_evals: &VirtualEvaluation<Fr>,
    domain: GeneralEvaluationDomain<Fr>,
    global_params: &ZeroCheckParams<LigeroPoseidon<Fr>>,
    size: u32,
    run_threads: Option<usize>,
    batch_commit_threads: Option<usize>,
    batch_open_threads: Option<usize>,
) -> u128 {
    let test_timer = start_timer!(|| {
        format!("Opt Univariate Proof Generation Test LigeroPoseidon for 2^{size} work")
    });

    let instant = Instant::now();
    let proof_gen_timer = start_timer!(|| "Prove fn called for LigeroPoseidon");

    let proof = CustomUnivariateZeroCheck::<Fr, LigeroPoseidon<Fr>>::prove(
        &global_params,
        input_evals,
        &domain,
        &mut ZCTranscript::init_transcript(),
        run_threads,
        batch_commit_threads,
        batch_open_threads,
    )
    .unwrap();

    end_timer!(proof_gen_timer);
    let runtime = instant.elapsed();

    let verify_timer = start_timer!(|| "Verify fn called for LigeroPoseidon");

    let result = CustomUnivariateZeroCheck::<Fr, LigeroPoseidon<Fr>>::verify(
        &global_params,
        input_evals,
        &proof,
        &domain,
        &mut ZCTranscript::init_transcript(),
    )
    .unwrap();

    end_timer!(verify_timer);

    assert_eq!(result, true);

    end_timer!(test_timer);
    return runtime.as_millis();
}

#[derive(Parser, Debug)]
struct Args {
    /// Number of repetitions for each test
    #[arg(long, default_value = "10")]
    repeat: usize,

    /// Minimum work size exponent (2^min_size)
    #[arg(long, default_value = "10")]
    min_size: usize,

    /// Maximum work size exponent (inclusive, 2^max_size)
    #[arg(long, default_value = "20")]
    max_size: usize,

    /// Number of threads to use for prepare input evaluations
    #[arg(long, default_value = "64")]
    prepare_threads: usize,

    /// Number of threads to use to run proof and verify tests
    #[arg(long, default_value = "1")]
    run_threads: usize,

    // choose between `kzg` and `ligero`
    #[arg(long, default_value = "kzg")]
    poly_commit_scheme: String,

    /// Number of threads to run batch opening
    #[arg(long, default_value = "1")]
    batch_opening_threads: usize,

    /// Number of threads to run batch commit
    #[arg(long, default_value = "1")]
    batch_commit_threads: usize,

    #[arg(long, default_value = "g*h*s + (1 - s)*(g + h)")]
    f: String,
}

fn bench_opt_uni_zc() {
    let args = Args::parse();
    let min_size = args.min_size + (args.min_size % 2); // make it even

    let (sizes, runtimes): (Vec<u32>, Vec<u128>) = (min_size..=args.max_size)
        .step_by(2)
        .map(|size| {
            let pool_prepare = rayon::ThreadPoolBuilder::new()
                .num_threads(args.prepare_threads as usize)
                .build()
                .unwrap();

            let (input_evals, domain, pp) =
                prepare_input_evals_domain(size, &pool_prepare, args.f.clone());

            let total_runtime: u128 = match args.poly_commit_scheme.as_str() {
                "kzg" => {
                    let global_params =
                        CustomUnivariateZeroCheck::<Fr, KZG<Bls12_381>>::setup(&pp).unwrap();
                    (0..args.repeat)
                        .map(|repeat_time| {
                            println!(
                                "Running KZG test for 2^{} with repeat: {}",
                                size, repeat_time
                            );
                            opt_univ_zc_multhr_benchmark_kzg(
                                &input_evals,
                                domain,
                                &global_params,
                                size as u32,
                                Some(args.run_threads as usize),
                                Some(args.batch_commit_threads as usize),
                                Some(args.batch_opening_threads as usize),
                            )
                        })
                        .sum()
                }
                "ligero" => {
                    let global_params =
                        CustomUnivariateZeroCheck::<Fr, Ligero<Fr>>::setup(&pp).unwrap();
                    (0..args.repeat)
                        .map(|repeat_time| {
                            println!(
                                "Running Ligero test for 2^{} with repeat: {}",
                                size, repeat_time
                            );
                            opt_univ_zc_multhr_benchmark_ligero(
                                &input_evals,
                                domain,
                                &global_params,
                                size as u32,
                                Some(args.run_threads as usize),
                                Some(args.batch_commit_threads as usize),
                                Some(args.batch_opening_threads as usize),
                            )
                        })
                        .sum()
                }
                "ligero_poseidon" => {
                    let global_params =
                        CustomUnivariateZeroCheck::<Fr, LigeroPoseidon<Fr>>::setup(&pp).unwrap();
                    (0..args.repeat)
                        .map(|repeat_time| {
                            println!(
                                "Running LigeroPoseidon test for 2^{} with repeat: {}",
                                size, repeat_time
                            );
                            opt_univ_zc_multhr_benchmark_ligero_poseidon(
                                &input_evals,
                                domain,
                                &global_params,
                                size as u32,
                                Some(args.run_threads as usize),
                                Some(args.batch_commit_threads as usize),
                                Some(args.batch_opening_threads as usize),
                            )
                        })
                        .sum()
                }
                _ => panic!("Invalid poly_commit_scheme"),
            };

            (size as u32, total_runtime)
        })
        .unzip();

    // Print results
    for (size, runtime) in zip(sizes, runtimes) {
        println!(
            "Input Polynomial Degree: 2^{:?}\t|| Poly Commit Scheme: {:?}\t|| Avg. Runtime: {:?} ms",
            size,
            args.poly_commit_scheme,
            (runtime as f64) / (args.repeat as f64),
        );
    }
}

fn main() {
    bench_opt_uni_zc();
}
