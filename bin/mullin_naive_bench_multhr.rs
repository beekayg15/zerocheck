use ark_bls12_381::Fr;
use clap::Parser;
use std::iter::zip;
use std::time::Instant;
use zerocheck::{
    multilinear_zc::naive::{rand_zero, NaiveMLZeroCheck},
    ZeroCheck,
};

fn test_template(
    num_vars: usize,
    num_multiplicands_range: (usize, usize),
    num_products: usize,
    repeat: u32,
) -> u128 {
    let instant = Instant::now();

    // Generate a random polynomial.
    // f = ∑_{num_products} rand_coeff*(g1.g2...g_{num_multiplicands_range}), gs are MLEs size 2^num_vars;
    // f = ∑_{i=1..6} rand_coeff*(g_i1·g_i2···g_i{1..=3}).
    // g_ij are MLEs size 2^num_vars, stored in `poly.flat_ml_extensions` (or poly.raw_pointers_lookup_table as (Vec, idx)).
    // (rand_coeff, ij info) are stored in `poly.products`.
    let poly = rand_zero::<Fr>(num_vars, num_multiplicands_range, num_products);

    let zp = NaiveMLZeroCheck::<Fr>::setup(None).unwrap();

    let duration = instant.elapsed().as_millis();
    print!("Random polynomial terms: ");
    for i in 0..num_products {
        print!("{} ", poly.products[i].1.len());
    }
    println!();
    println!("Preparing input evaluations and domain for 2^{num_vars} work ....{duration}ms");

    let proof = (0..repeat)
        .map(|_| NaiveMLZeroCheck::<Fr>::prove(zp.clone(), poly.clone(), num_vars).unwrap())
        .collect::<Vec<_>>()
        .last()
        .cloned()
        .unwrap();

    let runtime = instant.elapsed();

    let result = NaiveMLZeroCheck::<Fr>::verify(zp, poly, proof, num_vars).unwrap();

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
    // /// Number of threads to use for prepare input evaluations
    // #[arg(long, default_value = "64")]
    // prepare_threads: usize,

    // /// Number of threads to use to run proof and verify tests
    // #[arg(long, default_value = "1")]
    // run_threads: usize,
}

fn bench_naive_mle_zc() {
    let args = Args::parse();
    let repeat = args.repeat;

    let (sizes, runtimes): (Vec<usize>, Vec<u128>) = (args.min_size..=args.max_size)
        .map(|size| {
            let total_runtime: u128 = test_template(size, (1, 3 + 1), 6, repeat);
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
