use ark_bls12_381::{Bls12_381, Fr};
use ark_ec::AffineRepr;
use ark_ed_on_bls12_381::EdwardsAffine;
use clap::Parser;
use std::iter::zip;
use std::time::Instant;
use zerocheck::{
    pcs::multilinear_pcs::{hyrax::Hyrax, kzg::MultilinearKZG, ligero::Ligero, mpc::MPC},
    transcripts::ZCTranscript,
    zc::multilinear_zc::optimized::{custom_zero_test_case, OptMLZeroCheck},
    ZeroCheck,
};

fn test_template_mpc(
    num_vars: usize,
    repeat: u32,
    run_threads: Option<usize>,
    batch_commit_threads: Option<usize>,
    batch_open_threads: Option<usize>,
) -> u128 {
    let instant = Instant::now();

    // Generate a random polynomial.
    // f = ∑_{num_products} rand_coeff*(g1.g2...g_{num_multiplicands_range}), gs are MLEs size 2^num_vars;
    // f = ∑_{i=1..6} rand_coeff*(g_i1·g_i2···g_i{1..=3}).
    // g_ij are MLEs size 2^num_vars, stored in `poly.flat_ml_extensions` (or poly.raw_pointers_lookup_table as (Vec, idx)).
    // (rand_coeff, ij info) are stored in `poly.products`.
    // let poly = rand_zero::<Fr>(num_vars, num_multiplicands_range, num_products);

    let poly = custom_zero_test_case::<Fr>(num_vars);

    let inp_params = 2 * (1 << num_vars);
    let zp = OptMLZeroCheck::<Fr, MPC<Bls12_381>>::setup(&inp_params).unwrap();

    let duration = instant.elapsed().as_millis();
    print!("Polynomial terms: ");
    for i in 0..poly.products.len() {
        print!("{} ", poly.products[i].1.len());
    }
    println!();
    println!("Preparing input evaluations and domain for 2^{num_vars} work ....{duration}ms");

    let proof = (0..repeat)
        .map(|_| {
            OptMLZeroCheck::<Fr, MPC<Bls12_381>>::prove(
                &zp.clone(),
                &poly.clone(),
                &num_vars,
                &mut ZCTranscript::init_transcript(),
                run_threads,
                batch_commit_threads,
                batch_open_threads,
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
        .last()
        .cloned()
        .unwrap();

    let runtime = instant.elapsed();

    let result = OptMLZeroCheck::<Fr, MPC<Bls12_381>>::verify(
        &zp,
        &poly,
        &proof,
        &num_vars,
        &mut ZCTranscript::init_transcript(),
    )
    .unwrap();
    assert_eq!(result, true);
    return runtime.as_millis();
}

fn test_template_hyrax(
    num_vars: usize,
    repeat: u32,
    run_threads: Option<usize>,
    batch_commit_threads: Option<usize>,
    batch_open_threads: Option<usize>,
) -> u128 {
    let instant = Instant::now();

    let poly = custom_zero_test_case::<<EdwardsAffine as AffineRepr>::ScalarField>(num_vars);

    let inp_params = num_vars;
    type Fq = <EdwardsAffine as AffineRepr>::ScalarField;
    let zp = OptMLZeroCheck::<Fq, Hyrax<EdwardsAffine>>::setup(&inp_params).unwrap();

    let duration = instant.elapsed().as_millis();
    print!("Polynomial terms: ");
    for i in 0..poly.products.len() {
        print!("{} ", poly.products[i].1.len());
    }
    println!();
    println!("Preparing input evaluations and domain for 2^{num_vars} work ....{duration}ms");

    let proof = (0..repeat)
        .map(|_| {
            OptMLZeroCheck::<Fq, Hyrax<EdwardsAffine>>::prove(
                &zp.clone(),
                &poly.clone(),
                &num_vars,
                &mut ZCTranscript::init_transcript(),
                run_threads,
                batch_commit_threads,
                batch_open_threads,
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
        .last()
        .cloned()
        .unwrap();

    let runtime = instant.elapsed();

    let result = OptMLZeroCheck::<Fq, Hyrax<EdwardsAffine>>::verify(
        &zp,
        &poly,
        &proof,
        &num_vars,
        &mut ZCTranscript::init_transcript(),
    )
    .unwrap();
    assert_eq!(result, true);
    return runtime.as_millis();
}

fn test_template_kzg(
    num_vars: usize,
    repeat: u32,
    run_threads: Option<usize>,
    batch_commit_threads: Option<usize>,
    batch_open_threads: Option<usize>,
) -> u128 {
    let instant = Instant::now();

    let poly = custom_zero_test_case::<Fr>(num_vars);

    let inp_params = 2 * (1 << num_vars);
    let zp = OptMLZeroCheck::<Fr, MultilinearKZG<Bls12_381>>::setup(&inp_params).unwrap();

    let duration = instant.elapsed().as_millis();
    print!("Polynomial terms: ");
    for i in 0..poly.products.len() {
        print!("{} ", poly.products[i].1.len());
    }
    println!();
    println!("Preparing input evaluations and domain for 2^{num_vars} work ....{duration}ms");

    let proof = (0..repeat)
        .map(|_| {
            OptMLZeroCheck::<Fr, MultilinearKZG<Bls12_381>>::prove(
                &zp.clone(),
                &poly.clone(),
                &num_vars,
                &mut ZCTranscript::init_transcript(),
                run_threads,
                batch_commit_threads,
                batch_open_threads,
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
        .last()
        .cloned()
        .unwrap();

    let runtime = instant.elapsed();

    let result = OptMLZeroCheck::<Fr, MultilinearKZG<Bls12_381>>::verify(
        &zp,
        &poly,
        &proof,
        &num_vars,
        &mut ZCTranscript::init_transcript(),
    )
    .unwrap();
    assert_eq!(result, true);
    return runtime.as_millis();
}

fn test_template_ligero(
    num_vars: usize,
    repeat: u32,
    run_threads: Option<usize>,
    batch_commit_threads: Option<usize>,
    batch_open_threads: Option<usize>,
) -> u128 {
    let instant = Instant::now();

    // Generate a random polynomial for testing
    let poly = custom_zero_test_case::<Fr>(num_vars);

    let inp_params = 2 * (1 << num_vars); // Input parameters for Ligero setup
    let zp = OptMLZeroCheck::<Fr, Ligero<Fr>>::setup(&inp_params).unwrap();

    let duration = instant.elapsed().as_millis();
    print!("Polynomial terms: ");
    for i in 0..poly.products.len() {
        print!("{} ", poly.products[i].1.len());
    }
    println!();
    println!("Preparing input evaluations and domain for 2^{num_vars} work ....{duration}ms");

    // Generate the proof
    let proof = (0..repeat)
        .map(|_| {
            OptMLZeroCheck::<Fr, Ligero<Fr>>::prove(
                &zp.clone(),
                &poly.clone(),
                &num_vars,
                &mut ZCTranscript::init_transcript(),
                run_threads,
                batch_commit_threads,
                batch_open_threads,
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
        .last()
        .cloned()
        .unwrap();

    let runtime = instant.elapsed();

    // Verify the proof
    let result = OptMLZeroCheck::<Fr, Ligero<Fr>>::verify(
        &zp,
        &poly,
        &proof,
        &num_vars,
        &mut ZCTranscript::init_transcript(),
    )
    .unwrap();
    assert_eq!(result, true);

    return runtime.as_millis();
}

#[derive(Parser, Debug)]
struct Args {
    /// Number of repetitions for each test
    #[arg(long, default_value = "10")]
    repeat: u32,

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

    // choose between `hyrax` and `mpc`
    #[arg(long, default_value = "mpc")]
    poly_commit_scheme: String,

    /// Number of threads to run batch opening
    #[arg(long, default_value = "1")]
    batch_opening_threads: usize,

    /// Number of threads to run batch commit
    #[arg(long, default_value = "1")]
    batch_commit_threads: usize,
}

fn bench_opt_mle_zc() {
    let args = Args::parse();
    let repeat = args.repeat;
    let min_size = args.min_size + (args.min_size % 2); // make it even

    let (sizes, runtimes): (Vec<usize>, Vec<u128>) = (min_size..=args.max_size)
        .step_by(2)
        .map(|size| {
            let total_runtime: u128 = match args.poly_commit_scheme.as_str() {
                "mpc" => test_template_mpc(
                    size,
                    repeat,
                    Some(args.run_threads),
                    Some(args.batch_commit_threads),
                    Some(args.batch_opening_threads),
                ),
                "hyrax" => test_template_hyrax(
                    size,
                    repeat,
                    Some(args.run_threads),
                    Some(args.batch_commit_threads),
                    Some(args.batch_opening_threads),
                ),
                "kzg" => test_template_kzg(
                    size,
                    repeat,
                    Some(args.run_threads),
                    Some(args.batch_commit_threads),
                    Some(args.batch_opening_threads),
                ),
                "ligero" => test_template_ligero(
                    size,
                    repeat,
                    Some(args.run_threads),
                    Some(args.batch_commit_threads),
                    Some(args.batch_opening_threads),
                ),
                _ => panic!("Invalid poly_commit_scheme"),
            };
            (size, total_runtime)
        })
        .unzip();

    for (size, runtime) in zip(sizes, runtimes) {
        println!(
            "Input Polynomial Degree: 2^{:?}\t|| Poly Commit Scheme: {:?}\t|| Avg. Runtime: {:?} ms",
            size,
            args.poly_commit_scheme,
            (runtime as f64) / (repeat as f64),
        );
    }
}

fn main() {
    bench_opt_mle_zc();
}
