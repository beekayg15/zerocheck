use ark_bls12_381::Fr;
use clap::Parser;
use std::iter::zip;
use std::time::Instant;
use zerocheck::{
    transcripts::ZCTranscript, zc::multilinear_zc::naive::{rand_zero, NaiveMLZeroCheck}, ZeroCheck
};

fn test_template(
    num_vars: usize,
    num_multiplicands_range: (usize, usize),
    num_products: usize,
    repeat: i32,
) -> u128 {
    let poly = rand_zero::<Fr> (
        num_vars, 
        num_multiplicands_range, 
        num_products
    );

    let zp= NaiveMLZeroCheck::<Fr>::setup(&None).unwrap();

    let instant = Instant::now();

    let proof = (0..repeat)
        .map(|_| {
            NaiveMLZeroCheck::<Fr>::prove(
                &zp.clone(),
                &poly.clone(), 
                &num_vars,
                &mut ZCTranscript::init_transcript()
            ).unwrap()
        })
        .collect::<Vec<_>>()
        .last()
        .cloned()
        .unwrap();

    let runtime = instant.elapsed();

    let result = NaiveMLZeroCheck::<Fr>::verify(
        &zp,
        &poly, 
        &proof, 
        &num_vars,
        &mut ZCTranscript::init_transcript()
    ).unwrap();

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

fn bench_naive_mle_zc() {
    let repeat = 1;
    let max_work_size = 6;

    let (sizes, runtimes): (Vec<usize>, Vec<u128>) = (5..max_work_size)
        .map(|size| {
            let total_runtime: u128 = test_template(size, (6, 7), size, repeat);
            (size, total_runtime)
        })
        .unzip();

    for (size, runtime) in zip(sizes, runtimes) {
        println!(
            "Input Polynomial Degree: 2^{:?}\t|| Avg. Runtime: {:?} ms",
            size,
            (runtime as f64) / (repeat as f64),
        );
    }
}

fn main() {
    bench_naive_mle_zc();
}
