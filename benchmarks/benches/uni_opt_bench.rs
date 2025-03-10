use criterion::{
    criterion_group, BenchmarkId,
    Criterion, BatchSize, black_box
};
use ark_bls12_381::Fr;
use ark_bls12_381::Bls12_381;
use ark_ff::UniformRand;
use ark_poly::{
    univariate::DensePolynomial, Polynomial,
    DenseUVPolynomial, EvaluationDomain, 
    Evaluations, GeneralEvaluationDomain
};
use zerocheck::univariate_zc::optimized::OptimizedUnivariateZeroCheck;
use zerocheck::ZeroCheck;

fn opt_univariate_zero_check_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("uni_opt_zerocheck");

    for size in [8, 10, 12] {
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

        let g_evals = Evaluations::from_vec_and_domain(
            evals_over_domain_g, 
            domain_g            
        );

        let h_evals = Evaluations::from_vec_and_domain(
            evals_over_domain_h, 
            domain_h
        );

        let s_evals = Evaluations::from_vec_and_domain(
            evals_over_domain_s, 
            domain_s
        );

        let mut inp_evals = vec![];
        inp_evals.push(g_evals);
        inp_evals.push(h_evals);
        inp_evals.push(s_evals);

        group.bench_with_input(
            BenchmarkId::new("uni_opt_zerocheck", size), &size, |b, &_size| {
                b.iter_batched(
                    || {
                        inp_evals.clone()
                    },
                    |input_evals| {
                        let _proof = OptimizedUnivariateZeroCheck::<Fr, Bls12_381>::prove(
                            black_box(input_evals), 
                            black_box(zero_domain)
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
    targets = opt_univariate_zero_check_benchmark
);