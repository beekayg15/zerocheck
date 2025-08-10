use crate::pcs::PolynomialCommitmentScheme;
use ark_ff::PrimeField;
use ark_poly::univariate::DensePolynomial;
use std::{cmp::max, collections::HashMap, marker::PhantomData, sync::Arc};
use anyhow::{Error, Ok};
use ark_poly::Polynomial;

/// This is the data structure of the proof to be sent to the verifer,
/// to prove that there exists a quotient polynomial q(X), for which,
/// for all x \in Fp, f(x) = q(x).z_H(x), where, z_H(X) is the vanishing
/// polynomial over the domain H, for which prover claims that
/// for x \in H, f(x) = 0.
///  
/// q - stores the commitment to quotient polynomail as Commitment<E>
/// inp_comms - stores the commitment to the input polynomials as Vec<Commitment<E>>
/// vk - stores the verifier key required to check the KZG openings
/// q_opening - stores the opening proof for the evaluation of q(X) at challenge 'r'
/// inp_opening - stores the opening proof for the evaluation of g, h, and s at challenge 'r'
/// inp_evals -stores the evaluation of g, h, and s at challenge 'r'
/// q_eval -stores the evaluation of q(X) at challenge 'r'

#[derive(Clone)]
pub struct Proof<PCS: PolynomialCommitmentScheme> {
    pub(crate) q_comm: PCS::Commitment,
    pub(crate) inp_comms: Vec<PCS::Commitment>,
    pub(crate) q_opening: PCS::OpeningProof,
    pub(crate) inp_openings: Vec<PCS::OpeningProof>,
    pub(crate) inp_evals: Vec<PCS::PolynomialOutput>,
    pub(crate) q_eval: PCS::PolynomialOutput,
}

#[derive(Clone)]
pub struct ZeroCheckParams<'a, PCS: PolynomialCommitmentScheme> {
    pub(crate) ck: PCS::CommitterKey<'a>,
    pub(crate) vk: PCS::VerifierKey,
}

/// Struct to store the necessary information about the virtual polynomial,
/// namely, the number of variables in the MLEs and the max degree of any
/// variable in the virtual polynomial
#[derive(Clone, Debug)]
pub struct PolynomialInfo<F: PrimeField> {
    // maximum degree of the variables in the output polynomial
    pub max_multiplicand: usize,
    // number of variables in the MLEs
    pub num_vars: usize,
    #[doc(hidden)]
    pub phantom: PhantomData<F>,
}

/// Struct to store the sum of products of MLEs
#[derive(Clone, Debug)]
pub struct VirtualPolynomial<F: PrimeField> {
    // information about the virtual polynomial
    pub poly_info: PolynomialInfo<F>,
    // list of (indexed) MLEs which are to be multiplied
    // together along with a coefficient
    pub products: Vec<(F, Vec<usize>)>,
    // list of dense multilinear extensions of multilinear
    // polynomials used
    pub flat_ml_extensions: Vec<Arc<DensePolynomial<F>>>,
    raw_pointers_lookup_table: HashMap<*const DensePolynomial<F>, usize>,
}

impl<F: PrimeField> VirtualPolynomial<F> {
    /// Creates an empty virtual polynomial with `num_variables`
    pub fn new(num_variables: usize) -> Self {
        VirtualPolynomial {
            poly_info: PolynomialInfo {
                max_multiplicand: 0,
                num_vars: num_variables,
                phantom: PhantomData,
            },
            products: Vec::new(),
            flat_ml_extensions: Vec::new(),
            raw_pointers_lookup_table: HashMap::new(),
        }
    }

    /// Add a product of list of multilinear extensions to self
    /// Returns an error if the list is empty, or the MLE has a different
    /// `num_vars` from self
    ///
    /// The MLEs will be multiplied together, and then multiplied by the scalar
    /// `coefficient`
    pub fn add_product(
        &mut self,
        product: impl IntoIterator<Item = Arc<DensePolynomial<F>>>,
        coefficient: F,
    ) {
        let product: Vec<Arc<DensePolynomial<F>>> = product.into_iter().collect();
        let mut indexed_product = Vec::with_capacity(product.len());
        assert!(!product.is_empty());

        self.poly_info.max_multiplicand = max(self.poly_info.max_multiplicand, product.len());

        for m in product {

            let m_ptr: *const DensePolynomial<F> = Arc::as_ptr(&m);

            if let Some(index) = self.raw_pointers_lookup_table.get(&m_ptr) {
                indexed_product.push(*index)
            } else {
                let curr_index = self.flat_ml_extensions.len();
                self.flat_ml_extensions.push(m.clone());
                self.raw_pointers_lookup_table.insert(m_ptr, curr_index);
                indexed_product.push(curr_index);
            }
        }

        self.products.push((coefficient, indexed_product));
    }

    /// Multiple the current VirtualPolynomial by an MLE:
    /// - add the MLE to the MLE list;
    /// - multiple each product by MLE and its coefficient
    ///
    /// Returns an error if the MLE has a different `num_vars` from self
    pub fn mul_dense_polynomial(
        &mut self,
        mle: Arc<DensePolynomial<F>>,
        coefficient: F,
    ) -> Result<(), Error> {

        let mle_ptr: *const DensePolynomial<F> = Arc::as_ptr(&mle);

        let mle_index = match self.raw_pointers_lookup_table.get(&mle_ptr) {
            Some(&p) => p,
            _ => {
                self.raw_pointers_lookup_table
                    .insert(mle_ptr, self.flat_ml_extensions.len());
                self.flat_ml_extensions.push(mle);
                self.flat_ml_extensions.len() - 1
            }
        };

        for (prod_coef, indices) in self.products.iter_mut() {
            // - add the MLE to the MLE list;
            // - multiple each product by MLE and its coefficient.
            indices.push(mle_index);
            *prod_coef *= coefficient;
        }

        self.poly_info.max_multiplicand += 1;
        Ok(())
    }

    /// Evaluate the virtual polynomial at point `point`
    /// Returns an error is point.len() does not match `num_variables`
    pub fn evaluate(&self, point: F) -> F {

        self.products
            .iter()
            .map(|(c, p)| {
                *c * p
                    .iter()
                    .map(|&i| self.flat_ml_extensions[i].evaluate(&point))
                    .product::<F>()
            })
            .sum()
    }
}