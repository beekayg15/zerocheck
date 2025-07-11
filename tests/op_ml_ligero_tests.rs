#[cfg(test)]
mod tests {
    use std::iter::zip;
    use std::time::Instant;

    use ark_bls12_381::Fr;
    use zerocheck::{
        pcs::multilinear_pcs::ligero::Ligero,
        transcripts::ZCTranscript,
        zc::multilinear_zc::optimized::{custom_zero_test_case, OptMLZeroCheck},
        ZeroCheck,
    };

    fn test_template(num_vars: usize, repeat: i32) -> u128 {
        let poly = custom_zero_test_case::<Fr>(num_vars);

        let inp_params = num_vars;

        let zp = OptMLZeroCheck::<Fr, Ligero<Fr>>::setup(&inp_params).unwrap();

        let instant = Instant::now();

        let proof = (0..repeat)
            .map(|_| {
                OptMLZeroCheck::<Fr, Ligero<Fr>>::prove(
                    &zp.clone(),
                    &poly.clone(),
                    &num_vars,
                    &mut ZCTranscript::init_transcript(),
                    None, // run_threads
                    None, // batch_commit_threads
                    None, // batch_open_threads
                )
                .unwrap()
            })
            .collect::<Vec<_>>()
            .last()
            .cloned()
            .unwrap();

        let runtime = instant.elapsed();

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

    #[test]
    fn sample_opt_mle_test() {
        let x = test_template(5, 5);
        println!("Proving time: {:?}", x);
    }

    #[test]
    fn bench_opt_mle_zc() {
        let repeat = 10;
        let max_work_size = 16;

        let (sizes, runtimes): (Vec<usize>, Vec<u128>) = (1..max_work_size)
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
