use criterion::{
    criterion_group, BenchmarkId,
    Criterion, BatchSize, black_box
};
use ark_bls12_381::Fr;
use ark_ff::UniformRand;
use ark_poly::{
    univariate::DensePolynomial, 
    DenseUVPolynomial, EvaluationDomain, 
    Evaluations, GeneralEvaluationDomain
};
use zerocheck::zc::univariate_zc::naive::NaiveUnivariateZeroCheck;
use zerocheck::ZeroCheck;

fn naive_univariate_zero_check_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("uni_naive_zerocheck");

    for size in [8, 10, 12] {
        let domain = GeneralEvaluationDomain::<Fr>::new(1 << size).unwrap();

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
        for _ in 1..(1 << size) {
            rand_coeffs.push(Fr::rand(rng));
        }

        let random_poly = DensePolynomial::from_coefficients_vec(rand_coeffs);

        let h_evals = random_poly.evaluate_over_domain(domain);

        let mut inp_evals = vec![];
        inp_evals.push(g_evals);
        inp_evals.push(h_evals);

        let zp = NaiveUnivariateZeroCheck::<Fr>::setup(&None).unwrap();

        group.bench_with_input(
            BenchmarkId::new("uni_naive_zerocheck", size), &size, |b, &_size| {
                b.iter_batched(
                    || {
                        inp_evals.clone()
                    },
                    |input_evals| {
                        let _proof = NaiveUnivariateZeroCheck::<Fr>::prove(
                            black_box(&zp.clone()),
                            black_box(&input_evals), 
                            black_box(&domain),
                            &mut None
                        );
                    },
                    BatchSize::LargeInput
                )
            }
        );
    }
    group.finish();
}

criterion_group! (
    name = benchmarks;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = naive_univariate_zero_check_benchmark
);