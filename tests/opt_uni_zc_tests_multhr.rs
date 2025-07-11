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
    use rayon::prelude::*;
    use std::iter::zip;
    use std::time::Instant;
    use zerocheck::pcs::univariate_pcs::kzg::KZG;
    use zerocheck::transcripts::ZCTranscript;
    use zerocheck::zc::univariate_zc::optimized::OptimizedUnivariateZeroCheck;
    use zerocheck::ZeroCheck;

    fn test_template(size: u32, repeat: u32) -> u128 {
        let test_timer = start_timer!(|| "Proof Generation Test");

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

        let inp_evals = vec![g_evals, h_evals, s_evals, o_evals];

        let proof_gen_timer = start_timer!(|| "Prove fn called for g, h, zero_domain");

        let max_degree = g.degree() + s.degree() + h.degree();
        let pp = max_degree;

        let zp = OptimizedUnivariateZeroCheck::<Fr, KZG<Bls12_381>>::setup(&pp).unwrap();

        let instant = Instant::now();

        let proof = (0..repeat)
            .map(|_| {
                OptimizedUnivariateZeroCheck::<Fr, KZG<Bls12_381>>::prove(
                    &zp.clone(),
                    &inp_evals.clone(),
                    &domain,
                    &mut ZCTranscript::init_transcript(),
                    None,
                    None,
                    None,
                )
                .unwrap()
            })
            .collect::<Vec<_>>()
            .last()
            .cloned()
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
        let x = test_template(10, 1);
        println!("Proving time: {:?}", x);
    }

    #[test]
    fn bench_opt_uni_zc() {
        let repeat = 10;
        let max_work_size = 10..16;

        let (sizes, runtimes): (Vec<u32>, Vec<u128>) = (max_work_size)
            .map(|size| {
                let total_runtime: u128 = test_template(size, repeat);
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
}
