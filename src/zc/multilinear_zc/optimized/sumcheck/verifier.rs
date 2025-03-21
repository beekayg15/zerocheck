use ark_ff::PrimeField;
use ark_std::{end_timer, start_timer};

use crate::{
    transcripts::ZCTranscript, zc::multilinear_zc::optimized::PolynomialInfo
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use super::{prover::ProverMsg, IPforSumCheck};

/// Struct to store the verifier message sent during 
/// eah round of the sumcheck protocol
#[derive(Clone, Debug, CanonicalDeserialize, CanonicalSerialize)]
pub struct VerifierMsg<F: PrimeField> {
    // challenge sampled by the verifier 
    pub(crate) challenge: F,
}

/// Struct to store the state of the sumcheck verifier across rounds
#[derive(Clone, Debug)]
pub struct VerifierState<F: PrimeField> {
    // challenges sampled by the verifier during each round
    pub challenges: Vec<F>,
    // univariate polynomials sent by the prover during each round
    // as evaluations over the hypercube
    pub poly_rcvd: Vec<Vec<F>>,
    // number of variables in the MLEs
    pub num_vars: usize,
    // current round of the interactive sumcheck protocol
    pub round: usize,
    // maximum degree of the variables in the output polynomial
    pub max_multiplicand: usize,
    // indicator to show of the verification of the sumcheck 
    // protocol is completed
    pub finished: bool
}

/// Struct to store the subclaim obtained at the end of the sumcheck protocol
#[derive(Clone, Debug)]
pub struct SubClaim<F: PrimeField> {
    // set of field elements sampled as challenges by the verifier
    pub point: Vec<F>,
    // claimed value to which the output polynomial evaluates to on the point
    pub expected_evaluation: F
}

impl<F: PrimeField> IPforSumCheck<F> {
    /// initialize the verifier state at the start of the sumcheck protocol
    /// 
    /// Attribute:
    /// info: information about the virtual polynomial, i.e., max_degree and num_vars
    /// 
    /// Returns:
    /// the initial verifier state at the start of round 0
    pub fn verifier_init(info: PolynomialInfo<F>) -> VerifierState<F> {
        VerifierState {
            challenges: Vec::with_capacity(info.num_vars),
            poly_rcvd: Vec::with_capacity(info.num_vars),
            num_vars: info.num_vars,
            round: 1,
            max_multiplicand: info.max_multiplicand,
            finished: false,
        }
    }

    /// simulate the operations done by the verifier
    /// during a single round of the sumcheck protocol
    /// 
    /// Attributes
    /// verifier_state: State of the verifier
    /// prover_msg: evaluations sent by the prover
    /// 
    /// Returns
    /// verifier_msg: a random challenge sampled using Fiat-Shamir
    pub fn verifier_round(
        prover_msg: ProverMsg<F>,
        verifier_state: &mut VerifierState<F>,
        transcript: &mut ZCTranscript<F>,
    ) -> Option<VerifierMsg<F>> {
        let verifier_single_round_timer = start_timer!(|| format!(
            "Interative verifier for sumcheck at round {:?}", verifier_state.round
        ));
        
        // check if the verification step is already done
        if verifier_state.finished {
            panic!("Incorrect verifier state: Verifier is already finished.");
        }

        // compute the seed and input required to generate the random challenge
        let verifier_challenge_sampling_timer = start_timer!(|| 
            "verifier sampling a random challenge using the transcripts"
        );

        transcript.append_serializable_element(b"prover_msg", &prover_msg).unwrap();
        let challenge = transcript.get_and_append_challenge(b"round_challenge").unwrap();

        let msg = VerifierMsg{
            challenge: challenge
        };

        end_timer!(verifier_challenge_sampling_timer);

        // Update the verifier state with the new challenge
        verifier_state.challenges.push(msg.challenge);
        verifier_state.poly_rcvd.push(prover_msg.evaluations);

        if verifier_state.round == verifier_state.num_vars {
            verifier_state.finished = true;
        } else {
            verifier_state.round += 1;
        }

        end_timer!(verifier_single_round_timer);
        Some(msg)
    }

    /// simulate the operations done by the verifier
    /// at the end of the sumcheck protocol, i.e., 
    /// reduce the sumcheck to the claim that the given
    /// virtual polynomial evaluates to a given value 
    /// at a point(set of challenges samples by the verifier)
    /// 
    /// Attributes
    /// verifier_state: State of the verifier
    /// asserted_sum: initial claimed sum of the virtual polynomial
    /// over the boolean hypercube
    ///
    /// Returns
    /// subclaim: claim that the given virtual polynomial evaluates 
    /// to a certain value at a given point
    pub fn check_n_generate_subclaim(
        verifier_state: VerifierState<F>,
        asserted_sum: F
    ) -> Result<SubClaim<F>, crate::Error> {
        //  check of the all the round in the sumcheck have been completed
        if !verifier_state.finished {
            panic!("Verifier has not finished.");
        }

        let mut expected = asserted_sum;
        if verifier_state.poly_rcvd.len() != verifier_state.num_vars {
            panic!("insufficient rounds");
        }

        // recursively check if the p_i(0) + p_i(1) = p_{i - 1}(r_{i-1})
        let recursive_sum_check_timer = start_timer!(||
            "recursively check if the p_i(0) + p_i(1) = p_{i - 1}(r_{i-1})"
        );
        for i in 0..verifier_state.num_vars {
            let evaluations = &verifier_state.poly_rcvd[i];

            if evaluations.len() < verifier_state.max_multiplicand + 1 {
                panic!("incorrect number of evaluations");
            }

            let p0 = evaluations[0];
            let p1 = evaluations[1];

            assert_eq!(p0 + p1, expected, "IP for sum-check failed at round {:?}", i);

            let uni_poly_interpolation_timer = start_timer!(|| format!(
                "Interpolating the univariate polnomials at round {:?}",
                i
            ));
            expected = interpolate_uni_poly(evaluations, verifier_state.challenges[i]);

            end_timer!(uni_poly_interpolation_timer);
        }

        end_timer!(recursive_sum_check_timer);

        Ok(SubClaim{
            point: verifier_state.challenges,
            expected_evaluation: expected,
        })
    }
} 

/// function to interpolate the univariate polynomial from
/// the evaluations sent by the prover during each round and
/// find the value of the univariate polynomial at a given point
/// 
/// Attributes:
/// p_i: evaluations of the univariate polynomial
/// point: sampled challenge at which the polynomial should be evaluated
/// 
/// Returns:
/// the evaluation of the univariate polynomial at given point
pub(crate) fn interpolate_uni_poly<F: PrimeField>(
    p_i: &[F],
    eval_at: F
) -> F {
    let len = p_i.len();
    let mut evals = vec![];

    let mut prod = eval_at;
    evals.push(prod);

    let mut check = F::zero();
    for i in 1..len {
        if eval_at == check {
            return p_i[i - 1];
        }
        check += F::one();

        let tmp = eval_at - check;
        evals.push(tmp);
        prod *= tmp;
    }

    if eval_at == check {
        return p_i[len - 1];
    }

    let mut res = F::zero();

    if p_i.len() <= 20 {
        let last_denom = F::from(u64_factorial(len - 1));
        let mut ratio_numerator = 1i64;
        let mut ratio_enumerator = 1u64;

        for i in (0..len).rev() {
            let ratio_numerator_f = if ratio_numerator < 0 {
                -F::from((-ratio_numerator) as u64)
            } else {
                F::from(ratio_numerator as u64)
            };

            res += p_i[i] * prod * F::from(ratio_enumerator)
                / (last_denom * ratio_numerator_f * evals[i]);

            // compute ratio for the next step which is current_ratio * -(len-i)/i
            if i != 0 {
                ratio_numerator *= -(len as i64 - i as i64);
                ratio_enumerator *= i as u64;
            }
        }
    } else if p_i.len() <= 33 {
        let last_denom = F::from(u128_factorial(len - 1));
        let mut ratio_numerator = 1i128;
        let mut ratio_enumerator = 1u128;

        for i in (0..len).rev() {
            let ratio_numerator_f = if ratio_numerator < 0 {
                -F::from((-ratio_numerator) as u128)
            } else {
                F::from(ratio_numerator as u128)
            };

            res += p_i[i] * prod * F::from(ratio_enumerator)
                / (last_denom * ratio_numerator_f * evals[i]);

            // compute ratio for the next step which is current_ratio * -(len-i)/i
            if i != 0 {
                ratio_numerator *= -(len as i128 - i as i128);
                ratio_enumerator *= i as u128;
            }
        }
    } else {
        // since we are using field operations, we can merge
        // `last_denom` and `ratio_numerator` into a single field element.
        let mut denom_up = field_factorial::<F>(len - 1);
        let mut denom_down = F::one();

        for i in (0..len).rev() {
            res += p_i[i] * prod * denom_down / (denom_up * evals[i]);

            // compute denom for the next step is -current_denom * (len-i)/i
            if i != 0 {
                denom_up *= -F::from((len - i) as u64);
                denom_down *= F::from(i as u64);
            }
        }
    }

    res
}

/// compute the factorial(a) = 1 * 2 * ... * a
#[inline]
fn field_factorial<F: PrimeField>(a: usize) -> F {
    let mut res = F::one();
    for i in 1..=a {
        res *= F::from(i as u64);
    }
    res
}

/// compute the factorial(a) = 1 * 2 * ... * a
#[inline]
fn u128_factorial(a: usize) -> u128 {
    let mut res = 1u128;
    for i in 1..=a {
        res *= i as u128;
    }
    res
}

/// compute the factorial(a) = 1 * 2 * ... * a
#[inline]
fn u64_factorial(a: usize) -> u64 {
    let mut res = 1u64;
    for i in 1..=a {
        res *= i as u64;
    }
    res
}

#[cfg(test)]
mod test {
    use crate::zc::multilinear_zc::optimized::sumcheck::verifier::interpolate_uni_poly;
    use ark_poly::{
        univariate::DensePolynomial, 
        DenseUVPolynomial, Polynomial};
    use ark_std::vec::Vec;
    use ark_bls12_381::Fr;
    use ark_ff::UniformRand;

    type F = Fr;

    #[test]
    fn test_interpolation() {
        let mut prng = ark_std::test_rng();

        // test a polynomial with 20 known points, i.e., with degree 19
        let poly = DensePolynomial::<F>::rand(20 - 1, &mut prng);
        let evals = (0..20)
            .map(|i| poly.evaluate(&F::from(i)))
            .collect::<Vec<F>>();
        let query = F::rand(&mut prng);

        assert_eq!(poly.evaluate(&query), interpolate_uni_poly(&evals, query));

        // test a polynomial with 33 known points, i.e., with degree 32
        let poly = DensePolynomial::<F>::rand(33 - 1, &mut prng);
        let evals = (0..33)
            .map(|i| poly.evaluate(&F::from(i)))
            .collect::<Vec<F>>();
        let query = F::rand(&mut prng);

        assert_eq!(poly.evaluate(&query), interpolate_uni_poly(&evals, query));

        // test a polynomial with 64 known points, i.e., with degree 63
        let poly = DensePolynomial::<F>::rand(64 - 1, &mut prng);
        let evals = (0..64)
            .map(|i| poly.evaluate(&F::from(i)))
            .collect::<Vec<F>>();
        let query = F::rand(&mut prng);

        assert_eq!(poly.evaluate(&query), interpolate_uni_poly(&evals, query));

        // test interpolation when we ask for the value at an x-cordinate
        // we are already passing, i.e. in the range 0 <= x < len(values) - 1
        let evals = vec![0, 1, 4, 9]
            .into_iter()
            .map(|i| F::from(i))
            .collect::<Vec<F>>();
        assert_eq!(interpolate_uni_poly(&evals, F::from(3)), F::from(9));
    }
}