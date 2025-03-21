#[cfg(test)]
mod tests {
    use ark_poly::{
        univariate::DensePolynomial, DenseUVPolynomial, 
        GeneralEvaluationDomain, Evaluations, EvaluationDomain
    };
    use ark_std::{end_timer, start_timer};
    use zerocheck::zc::univariate_zc::naive::NaiveUnivariateZeroCheck;
    use ark_bls12_381::Fr;
    use ark_std::UniformRand;
    use std::time::Instant;
    use zerocheck::ZeroCheck;

    fn test_template(size: u32) -> u128 {
        let test_timer = start_timer!(|| "Proof Generation Test");

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
        for _ in 0..(1 << size) {
            rand_coeffs.push(Fr::rand(rng));
        }

        let random_poly = DensePolynomial::from_coefficients_vec(rand_coeffs);

        let h_evals = random_poly.evaluate_over_domain(domain);

        let mut inp_evals = vec![];
        inp_evals.push(g_evals);
        inp_evals.push(h_evals);

        let zp= NaiveUnivariateZeroCheck::<Fr>::setup(&None).unwrap();

        let proof_gen_timer = start_timer!(|| "Prove fn called for g, h, zero_domain");

        let instant = Instant::now();
        
        let proof = 
            NaiveUnivariateZeroCheck::<Fr>::prove(
                &zp.clone(),
                &inp_evals.clone(), 
                &domain,
            &mut None
        ).unwrap();

        let runtime = instant.elapsed();

        end_timer!(proof_gen_timer);
        
        // println!("Proof Generated: {:?}", proof);

        let verify_timer = start_timer!(|| "Verify fn called for g, h, zero_domain, proof");

        let result = NaiveUnivariateZeroCheck::<Fr>
            ::verify(
                &zp, 
                &inp_evals, 
                &proof, 
                &domain,
                &mut None
            ).unwrap();

        end_timer!(verify_timer);

        // println!("verification result: {:?}", result);
        assert_eq!(result, true);

        end_timer!(test_timer);
        return runtime.as_millis();
    }

    #[test]
    fn naive_uni_zc_sample_test() {
        let x = test_template(15);
        println!("Proving time: {:?}", x);
    }

    #[test]
    fn bench_naive_uni_zc() {
        let mut sizes = vec![];
        let mut runtimes = vec![];
        for size in 1..21 {
            let mut total_runtime = 0;
            for _ in 0..10 {
                total_runtime += test_template(size);
            }
            sizes.push(size);
            runtimes.push(total_runtime);

            println!("Test completed for degree: {:?}", 1 << size);
        }

        for i in 0..20 {
            println!("Input Polynomial Degree: 2^{:?}\t|| Avg. Runtime: {:?}", sizes[i], (runtimes[i] as f32)/10.0);
        }
    }
}