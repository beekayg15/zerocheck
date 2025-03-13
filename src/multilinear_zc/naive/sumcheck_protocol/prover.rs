use ark_ff::PrimeField;
use ark_poly::{
    DenseMultilinearExtension, 
    MultilinearExtension
};
use ark_std::{cfg_iter_mut, vec::Vec};

use crate::multilinear_zc::naive::VirtualPolynomial;

use super::{verifier::VerifierMsg, IPforSumCheck};

#[derive(Clone, Debug)]
pub struct ProverMsg<F: PrimeField> {
    pub(crate) evaluations: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct ProverState<F: PrimeField> {
    pub challenges: Vec<F>,
    pub products: Vec<(F, Vec<usize>)>,
    pub flat_ml_extensions: Vec<DenseMultilinearExtension<F>>,
    pub num_vars: usize,
    pub round: usize,
    pub max_multiplicand: usize,
}

impl<F: PrimeField> IPforSumCheck<F> {
    pub fn prover_init(polynomials: VirtualPolynomial<F>)  
        -> ProverState<F>
    {
        let num_variables = polynomials.poly_info.num_vars;
        let degree = polynomials.poly_info.max_multiplicand;

        let flattened_ml_extensions = polynomials
            .flat_ml_extensions
            .iter()
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

    pub fn prover_round(
        prover_state: &mut ProverState<F>, 
        verifier_msg: &Option<VerifierMsg<F>>
    ) -> ProverMsg<F> {
        if let Some(v_msg) = verifier_msg {
            if prover_state.round == 0 {
                panic!("first round should be prover first.");
            }

            prover_state.challenges.push(v_msg.challenge);

            let i = prover_state.round;
            let r = prover_state.challenges[i - 1];

            cfg_iter_mut!(prover_state.flat_ml_extensions).for_each(
                |mult_ext| {
                    *mult_ext = mult_ext.fix_variables(&[r]);
                }
            );
        } else if prover_state.round > 0 {
            panic!("verifier message is empty");
        }

        prover_state.round += 1;

        if prover_state.round > prover_state.num_vars {
            panic!("Prover is not active");
        }

        let i = prover_state.round;
        let nv = prover_state.num_vars;
        let degree = prover_state.max_multiplicand;

        let zeros = (vec![F::zero(); degree + 1], vec![F::zero(); degree + 1]);

        let fold_result = ark_std::cfg_into_iter!(0..(1 << (nv - i)), 1 << 10).fold(
            zeros,
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
        );

        let products_sum = fold_result.0;

        ProverMsg {
            evaluations: products_sum
        }
    }
}