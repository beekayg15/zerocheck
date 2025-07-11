#[cfg(test)]
mod tests {
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_poly::univariate::DensePolynomial;
    use ark_poly::{
        DenseUVPolynomial, EvaluationDomain, Evaluations, GeneralEvaluationDomain, Polynomial,
    };
    use ark_std::One;
    use ark_std::UniformRand;
    use ark_std::{end_timer, start_timer};
    use std::time::Instant;
    use zerocheck::pcs::univariate_pcs::kzg::KZG;
    use zerocheck::transcripts::ZCTranscript;
    use zerocheck::zc::univariate_zc::optimized::OptimizedUnivariateZeroCheck;
    use zerocheck::ZeroCheck;

    fn test_template(size: u32) -> u128 {
        let test_timer = start_timer!(|| "Proof Generation Test");

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

        let mut inp_evals = vec![];
        inp_evals.push(g_evals);
        inp_evals.push(h_evals);
        inp_evals.push(s_evals);
        inp_evals.push(o_evals);

        let max_degree = g.degree() + s.degree() + h.degree();
        let pp = max_degree;

        let zp = OptimizedUnivariateZeroCheck::<Fr, KZG<Bls12_381>>::setup(&pp).unwrap();

        let proof_gen_timer = start_timer!(|| "Prove fn called for g, h, zero_domain");

        let instant = Instant::now();

        let proof = OptimizedUnivariateZeroCheck::<Fr, KZG<Bls12_381>>::prove(
            &zp.clone(),
            &inp_evals.clone(),
            &domain,
            &mut ZCTranscript::init_transcript(),
            None, // run_threads
            None, // batch_commit_threads
            None, // batch_open_threads
        )
        .unwrap();

        let runtime = instant.elapsed();

        end_timer!(proof_gen_timer);

        // println!("Proof Generated");

        let verify_timer = start_timer!(|| "Verify fn called for g, h, zero_domain, proof");

        let result = OptimizedUnivariateZeroCheck::<Fr, KZG<Bls12_381>>::verify(
            &zp,
            &inp_evals,
            &proof,
            &domain,
            &mut ZCTranscript::init_transcript(),
        )
        .unwrap();

        end_timer!(verify_timer);

        // println!("verification result: {:?}", result);
        assert_eq!(result, true);

        end_timer!(test_timer);
        return runtime.as_millis();
    }

    #[test]
    fn opt_uni_zc_sample_test() {
        let x = test_template(15);
        println!("Proving time: {:?}", x);
    }

    #[test]
    fn bench_opt_uni_zc() {
        let mut sizes = vec![];
        let mut runtimes = vec![];
        for size in 1..11 {
            let mut total_runtime = 0;
            for _ in 0..5 {
                total_runtime += test_template(size);
            }
            sizes.push(size);
            runtimes.push(total_runtime);

            println!("Test completed for degree: {:?}", 1 << size);
        }

        for i in 0..10 {
            println!(
                "Input Polynomial Degree: 2^{:?}\t|| Avg. Runtime: {:?}",
                sizes[i],
                (runtimes[i] as f32) / 10.0
            );
        }
    }
}
