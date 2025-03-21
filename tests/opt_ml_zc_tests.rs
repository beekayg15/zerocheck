#[cfg(test)]
mod tests { 
    use std::time::Instant;
    use std::iter::zip;

    use ark_bls12_381::{Fr, Bls12_381};
    use zerocheck::{
        pcs::multilinear_pcs::mpc::MPC, transcripts::ZCTranscript, zc::multilinear_zc::optimized::{
            custom_zero_test_case, OptMLZeroCheck
        }, ZeroCheck
    };

    fn test_template(
        num_vars: usize,
        repeat: i32,
    ) -> u128 {
        let poly = custom_zero_test_case::<Fr> (
            num_vars
        );

        let inp_params = num_vars;

        let zp= OptMLZeroCheck::<Bls12_381, MPC<Bls12_381>>::setup(&inp_params).unwrap();

        let instant = Instant::now();

        let proof = (0..repeat)
            .map(|_| {
                OptMLZeroCheck::<Bls12_381, MPC<Bls12_381>>::prove(
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

        let result = OptMLZeroCheck::<Bls12_381, MPC<Bls12_381>>::verify(
            &zp,
            &poly, 
            &proof, 
            &num_vars,
            &mut ZCTranscript::init_transcript()
        ).unwrap();

        assert_eq!(result, true);
        return runtime.as_millis();
    }

    #[test]
    fn sample_naive_mle_test() {
        let x = test_template(15, 5);
        println!("Proving time: {:?}", x);
    }

    #[test]
    fn bench_opt_mle_zc() {
        let repeat = 10;
        let max_work_size = 16;

        let (sizes, runtimes): (Vec<usize>, Vec<u128>) = (1..max_work_size)
            .map(|size| {
                let total_runtime: u128 = test_template(
                    size, 
                    repeat
                );
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