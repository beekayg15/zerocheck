use ark_ff::PrimeField;
use ark_poly::{
    DenseMultilinearExtension, 
    MultilinearExtension
};
use ark_std::{cfg_iter_mut, end_timer, start_timer, vec::Vec};
use rayon::prelude::*;

use crate::multilinear_zc::naive::VirtualPolynomial;

use super::{verifier::VerifierMsg, IPforSumCheck};

/// Struct to store the message to be sent by the prover
/// at the end of each round of the interactive sum-check
/// protocol. The prover sends the evaluations of the univarite 
/// polynomial at points (0..=max_degree)
#[derive(Clone, Debug)]
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
            max_multiplicand: degree
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
        verifier_msg: &Option<VerifierMsg<F>>
    ) -> ProverMsg<F> {
        let prover_single_round_timer = start_timer!(|| format!(
            "Interative prover for sumcheck at round {:?}", prover_state.round
        ));

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

            cfg_iter_mut!(prover_state.flat_ml_extensions).for_each(
                |mult_ext| {
                    *mult_ext = mult_ext.fix_variables(&[r]);
                }
            );

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

        let fold_result = ark_std::cfg_into_iter!(0..(1 << (nv - i)), 1 << 10).fold(
            ||{
                (
                    vec![F::zero(); degree + 1], 
                    vec![F::zero(); degree + 1]
                )
            },
            |(mut products_sum, mut product), b| {
                for (coefficient, products) in &prover_state.products {
                    product.fill(*coefficient);
                    for &jth_product in products {
                        let table = &prover_state.flat_ml_extensions[jth_product];
                        let mut start = table[b << 1];
                        let step = table[(b << 1) + 1] - start;
                        for p in product.iter_mut() {
                            *p *= start;
                            start += step;
                        }
                    }
                    for t in 0..degree + 1 {
                        products_sum[t] += product[t];
                    }
                }
                (products_sum, product)
            }
        )
        .map(|(partial, _)| partial)
        .reduce(
            || vec![F::zero(); degree + 1],
            |_, partial| {
                partial
            });

        let products_sum = fold_result;

        end_timer!(bool_hypercube_sum_timer);
        end_timer!(prover_single_round_timer);

        // Send the evalutions on the max_degree + 1 points as the prover message
        ProverMsg {
            evaluations: products_sum
        }
    }
}