#[cfg(test)]
mod tests { 
    use std::time::Instant;
    use std::iter::zip;

    use ark_bls12_381::Fr;
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
                    &mut ZCTranscript::init_transcript(),
                    None,
                    None,
                    None,
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

    #[test]
    fn sample_naive_mle_test() {
        let x = test_template(15, (5, 6), 2, 10);
        println!("Proving time: {:?}", x);
    }

    #[test]
    fn bench_naive_mle_zc() {
        let repeat = 10;
        let max_work_size = 16;

        let (sizes, runtimes): (Vec<usize>, Vec<u128>) = (1..max_work_size)
            .map(|size| {
                let total_runtime: u128 = test_template(
                    size, 
                    (6, 7),
                    size,
                    repeat);
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