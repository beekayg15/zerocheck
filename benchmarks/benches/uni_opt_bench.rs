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
use zerocheck::pcs::univariate_pcs::kzg::KZG;
use zerocheck::transcripts::ZCTranscript;
use zerocheck::zc::univariate_zc::optimized::OptimizedUnivariateZeroCheck;
use zerocheck::ZeroCheck;
use ark_std::One;

fn opt_univariate_zero_check_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("uni_opt_zerocheck");

    for size in [8, 10, 12] {
        let domain = GeneralEvaluationDomain::<Fr>::new(1 << size).unwrap();

        let rng = &mut ark_std::test_rng();

        let mut rand_g_coeffs = vec![];
        let mut rand_h_coeffs = vec![];
        let mut rand_s_coeffs = vec![];

        for _ in 0..(1 << size) {
            rand_g_coeffs.push(Fr::rand(rng));
            rand_h_coeffs.push(Fr::rand(rng));
            rand_s_coeffs.push(Fr::rand(rng));
        }

        let g = DensePolynomial::from_coefficients_vec(rand_g_coeffs);
        let h = DensePolynomial::from_coefficients_vec(rand_h_coeffs);
        let s = DensePolynomial::from_coefficients_vec(rand_s_coeffs);

        let evals_over_domain_g: Vec<_> = domain
            .elements()
            .map(|f| g.evaluate(&f))
            .collect();

        let evals_over_domain_h: Vec<_> = domain
            .elements()
            .map(|f| h.evaluate(&f))
            .collect();

        let evals_over_domain_s: Vec<_> = domain
            .elements()
            .map(|f| s.evaluate(&f))
            .collect();

        let evals_over_domain_o: Vec<_> = domain
            .elements()
            .map(|f| {
                g.evaluate(&f) * h.evaluate(&f) * s.evaluate(&f) 
                + (Fr::one() - s.evaluate(&f)) * (g.evaluate(&f) + h.evaluate(&f))
            })
            .collect();

        let g_evals = Evaluations::from_vec_and_domain(
            evals_over_domain_g, 
            domain            
        );

        let h_evals = Evaluations::from_vec_and_domain(
            evals_over_domain_h, 
            domain
        );

        let s_evals = Evaluations::from_vec_and_domain(
            evals_over_domain_s, 
            domain
        );

        let o_evals = Evaluations::from_vec_and_domain(
            evals_over_domain_o, 
            domain
        );

        let mut inp_evals = vec![];
        inp_evals.push(g_evals);
        inp_evals.push(h_evals);
        inp_evals.push(s_evals);
        inp_evals.push(o_evals);

        let max_degree = g.degree() + s.degree() + h.degree();
        let pp = max_degree;


        let zp = OptimizedUnivariateZeroCheck::<Fr, KZG<Bls12_381>>::setup(&pp).unwrap();

        group.bench_with_input(
            BenchmarkId::new("uni_opt_zerocheck", size), &size, |b, &_size| {
                b.iter_batched(
                    || {
                        inp_evals.clone()
                    },
                    |input_evals| {
                        let _proof = OptimizedUnivariateZeroCheck::<Fr, KZG<Bls12_381>>::prove(
                            black_box(&zp.clone()),
                            black_box(&input_evals), 
                            black_box(&domain),
                            &mut ZCTranscript::init_transcript()
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