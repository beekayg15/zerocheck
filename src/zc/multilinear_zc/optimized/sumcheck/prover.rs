use ark_ff::{batch_inversion, PrimeField};
use ark_poly::{
    DenseMultilinearExtension, 
    MultilinearExtension
};
use ark_std::{cfg_into_iter, end_timer, start_timer, vec::Vec};
use rayon::prelude::{IntoParallelIterator, IntoParallelRefIterator};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

#[cfg(feature = "parallel")]
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};

use crate::{transcripts::ZCTranscript, zc::multilinear_zc::optimized::VirtualPolynomial};

use super::{verifier::VerifierMsg, IPforSumCheck};

/// Struct to store the message to be sent by the prover
/// at the end of each round of the interactive sum-check
/// protocol. The prover sends the evaluations of the univarite 
/// polynomial at points (0..=max_degree)
#[derive(Clone, Debug, CanonicalDeserialize, CanonicalSerialize)]
pub struct ProverMsg<F: PrimeField> {
    // Evaluations of the univariate polynomial at any round
    // evaluated a points (0..=max_degree)
    pub(crate) evaluations: Vec<F>,
}

/// Struct to store the state of the sumcheck prover across rounds
#[derive(Clone, Debug)]
pub struct ProverState<F: PrimeField> {
    // challenges sampled by the verifier during each round
    pub challenges: Vec<F>,
    // list of (indexed) MLEs which are to be multiplied 
    // together along with a coefficient 
    pub products: Vec<(F, Vec<usize>)>,
    // list of dense multilinear extensions of multilinear 
    // polynomials used
    pub flat_ml_extensions: Vec<DenseMultilinearExtension<F>>,
    // number of variables in the MLEs
    pub num_vars: usize,
    // current round of the interactive sumcheck protocol
    pub round: usize,
    // maximum degree of the variables in the output polynomial
    pub max_multiplicand: usize,
    /// points with precomputed barycentric weights for extrapolating smaller
    /// degree uni-polys to `max_degree + 1` evaluations.
    pub extrapolation_aux: Vec<(Vec<F>, Vec<F>)>,
}

impl<F: PrimeField> IPforSumCheck<F> {
    /// initialize the prover state at the start of the sumcheck protocol
    /// 
    /// Attribute:
    /// polynomials: a virtual polynomial that is the function over multiple mle's
    /// 
    /// Returns:
    /// the initial prover state at the start of round 0
    pub fn prover_init(polynomials: VirtualPolynomial<F>)  
        -> ProverState<F>
    {
        // initialize the prover state with necessary
        // information from the virtual polynomial
        let num_variables = polynomials.poly_info.num_vars;
        let degree = polynomials.poly_info.max_multiplicand;

        let flattened_ml_extensions = polynomials
            .flat_ml_extensions
            .par_iter()
            .map(|x| x.as_ref().clone())
            .collect();

        ProverState {
            challenges: Vec::with_capacity(num_variables),
            products: polynomials.products.clone(),
            flat_ml_extensions: flattened_ml_extensions,
            num_vars: num_variables,
            round: 0,
            max_multiplicand: degree,
            extrapolation_aux: (1..polynomials.poly_info.max_multiplicand)
                .map(|degree| {
                    let points = (0..1 + degree as u64).map(F::from).collect::<Vec<_>>();
                    let weights = barycentric_weights(&points);
                    (points, weights)
                })
                .collect(),
        }
    }

    /// simulate the operations done by the prover
    /// during a single round of the sumcheck protocol
    /// 
    /// Attributes
    /// prover_state: State of the prover
    /// verifier_msg: challenge sampled by the verifier
    /// 
    /// Returns
    /// prover_msg: a univariate polynomials as evaluation
    /// over max_degree + 1 points
    pub fn prover_round(
        prover_state: &mut ProverState<F>, 
        verifier_msg: &Option<VerifierMsg<F>>,
        transcripts: &mut ZCTranscript<F>,
    ) -> ProverMsg<F> {
        let prover_single_round_timer = start_timer!(|| format!(
            "Interative prover for sumcheck at round {:?}", prover_state.round
        ));

        let mut flattened_ml_extensions: Vec<DenseMultilinearExtension<F>> = prover_state
            .flat_ml_extensions
            .par_iter()
            .map(|x| x.as_ref().clone())
            .collect();

        if let Some(v_msg) = verifier_msg {
            if prover_state.round == 0 {
                panic!("first round should be prover first.");
            }

            prover_state.challenges.push(v_msg.challenge);

            let i = prover_state.round;
            let r = prover_state.challenges[i - 1];

            // update the multilinear extensions with the challenge
            // sent by the verifier
            let fix_mle_variable_timer = start_timer!(|| format!(
                "fixing variable of the mle's with the challenge {:?}", r
            ));

            flattened_ml_extensions
                .par_iter_mut()
                .for_each(|mle| *mle = mle.fix_variables(&[r]));

            end_timer!(fix_mle_variable_timer);
        } else if prover_state.round > 0 {
            panic!("verifier message is empty");
        }

        // update the round number
        prover_state.round += 1;

        if prover_state.round > prover_state.num_vars {
            panic!("Prover is not active");
        }

        // Find the sum of the virtual polynomial over the remaining dimensions of the boolean hypercube
        let bool_hypercube_sum_timer = start_timer!(|| 
            "Evaluting over the remaining dim. of the boolean hypercube"
        );

        let i = prover_state.round;
        let nv = prover_state.num_vars;
        let degree = prover_state.max_multiplicand;

        let products_list = prover_state.products.clone();
        let mut products_sum = vec![F::zero(); degree + 1];

        // Generate sum for the partial evaluated polynomial:
        // f(r_1, ... r_m,, x_{m+1}... x_n)
        products_list.iter().for_each(|(coefficient, products)| {
            let mut sum: Vec<F> = cfg_into_iter!(0..1 << (nv - i))
                .fold(
                    || {
                        (
                            vec![(F::zero(), F::zero()); products.len()],
                            vec![F::zero(); products.len() + 1],
                    )
                    },
                    |(mut buf, mut acc), b| {
                        buf.iter_mut() // buf of a products (term): [(g_eval_b, g_step_b)],..,[(h_eval_b, h_step_b)];
                            .zip(products.iter())
                            .for_each(|((eval, step), f)| {
                                let table = &flattened_ml_extensions[*f];
                                *eval = table[b << 1];
                                *step = table[(b << 1) + 1] - table[b << 1];
                            });
                        acc[0] += buf.iter().map(|(eval, _)| eval).product::<F>(); // product: f_term0(0,0,0); (b=1)product: f_term0(0,1,0); (b=2)product: f_term0(1,0,0);
                        acc[1..].iter_mut().for_each(|acc| {
                            buf.iter_mut().for_each(|(eval, step)| *eval += step as &_); // buf_eval: [g(0,0,1),h(0,0,1)],...,[g(0,0,4),h(0,0,4)]; (b=1)buf_eval: [g(0,1,1),h(0,1,1)],[g(0,1,4),h(0,1,4)]; (b=2)buf_eval: [g(1,0,1),h(1,0,1)],[g(1,0,4),h(1,0,4)];
                            *acc += buf.iter().map(|(eval, _)| eval).product::<F>(); // product: f_term0(0,0,1),f_term0(0,0,2),f_term0(0,0,4); (b=1)product: f_term0(0,1,1),f_term0(0,1,2),f_term0(0,1,4); (b=2)product: f_term0(1,0,1),f_term0(1,0,2),f_term0(1,0,4);
                        });
                        (buf, acc)
                    } // acc[1] (H_1_term0(1)): f_term0(0,0,1)+f_term0(0,1,1)+f_term0(1,0,1)+f_term0(1,1,1),..acc[4] (H_1_term0(4)): f_term0(0,0,4)+f_term0(0,1,4)+f_term0(1,0,4)+f_term0(1,1,4)
                )
                .map(|(_, partial)| partial) // partial (acc[0..4]): eval points to represent a poly term. size(acc)=deg(term)+1
                .reduce( // save acc[0..4] as sum[0..4]
                    || vec![F::zero(); products.len() + 1],
                    |mut sum, partial| {
                        sum.iter_mut()
                            .zip(partial.iter())
                            .for_each(|(sum, partial)| *sum += partial);
                        sum
                    },
                );
            sum.iter_mut().for_each(|sum| *sum *= coefficient);
            let extraploation = cfg_into_iter!(0..degree - products.len())
                .map(|i| {
                    let (points, weights) = &prover_state.extrapolation_aux[products.len() - 1];
                    let at = F::from((products.len() + 1 + i) as u64);
                    extrapolate(points, weights, &sum, &at)
                })
                .collect::<Vec<_>>();
            products_sum
                .iter_mut()
                .zip(sum.iter().chain(extraploation.iter()))
                .for_each(|(products_sum, sum)| *products_sum += sum); // accumulate Hpoints of a term into products_sum
        });

        prover_state.flat_ml_extensions = flattened_ml_extensions
            .par_iter()
            .map(|x| x.clone())
            .collect();

        end_timer!(bool_hypercube_sum_timer);
        end_timer!(prover_single_round_timer);

        // Send the evalutions on the max_degree + 1 points as the prover message
        let prover_msg = ProverMsg {
            evaluations: products_sum
        };

        transcripts.append_serializable_element(b"prover_msg", &prover_msg).unwrap();

        prover_msg
    }
}

fn barycentric_weights<F: PrimeField>(points: &[F]) -> Vec<F> {
    let mut weights = points
        .iter()
        .enumerate()
        .map(|(j, point_j)| {
            points
                .iter()
                .enumerate()
                .filter(|&(i, _point_i)| (i != j))
                .map(|(_i, point_i)| *point_j - point_i)
                .reduce(|acc, value| acc * value)
                .unwrap_or_else(F::one)
        })
        .collect::<Vec<_>>();
    batch_inversion(&mut weights);
    weights
}

fn extrapolate<F: PrimeField>(points: &[F], weights: &[F], evals: &[F], at: &F) -> F {
    let (coeffs, sum_inv) = {
        let mut coeffs = points.iter().map(|point| *at - point).collect::<Vec<_>>();
        batch_inversion(&mut coeffs);
        coeffs.iter_mut().zip(weights).for_each(|(coeff, weight)| {
            *coeff *= weight;
        });
        let sum_inv = coeffs.iter().sum::<F>().inverse().unwrap_or_default();
        (coeffs, sum_inv)
    };
    coeffs
        .iter()
        .zip(evals)
        .map(|(coeff, eval)| *coeff * eval)
        .sum::<F>()
        * sum_inv
}