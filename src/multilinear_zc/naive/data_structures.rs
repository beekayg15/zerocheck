use ark_ff::PrimeField;
use ark_poly::{
    DenseMultilinearExtension,
    Polynomial
};
use std::{
    collections::HashMap, cmp::max,
    marker::PhantomData, sync::Arc
};

use super::sumcheck_protocol::prover::ProverMsg;

/// This is the data structure of the proof to be sent to the verifer,
/// to prove that the output polynomial f(x') = l(x').r(x').s(x')
/// evaluates to 0 over a given hypercube
#[derive(Clone, Debug)]
pub struct Proof<F: PrimeField> {
    pub(crate) prover_msgs: Vec<ProverMsg<F>>,
}

#[derive(Clone, Debug)]
pub struct PolynomialInfo<F: PrimeField> {
    pub max_multiplicand: usize,
    pub num_vars: usize,
    #[doc(hidden)]
    pub phantom: PhantomData<F>,
} 

#[derive(Clone, Debug)]
pub struct VirtualPolynomial<F: PrimeField> {
    pub poly_info: PolynomialInfo<F>,
    pub products: Vec<(F, Vec<usize>)>,
    pub flat_ml_extensions: Vec<Arc<DenseMultilinearExtension<F>>>,
    raw_pointers_lookup_table: HashMap<*const DenseMultilinearExtension<F>, usize>,
}

impl<F: PrimeField> VirtualPolynomial<F> {
    pub fn new(num_variables: usize) -> Self {
        VirtualPolynomial {
            poly_info: PolynomialInfo {
                max_multiplicand: 0,
                num_vars: num_variables,
                phantom: PhantomData
            },
            products: Vec::new(),
            flat_ml_extensions: Vec::new(),
            raw_pointers_lookup_table: HashMap::new(),
        }
    }

    pub fn add_product(
        &mut self,
        product: impl IntoIterator<Item = Arc<DenseMultilinearExtension<F>>>,
        coefficient: F
    ) {
        let product: Vec<Arc<DenseMultilinearExtension<F>>> = product
            .into_iter()
            .collect();
        let mut indexed_product = Vec::with_capacity(product.len());
        assert!(!product.is_empty());

        self.poly_info.max_multiplicand = max(self.poly_info.max_multiplicand, product.len());

        for m in product {
            assert_eq!(
                m.num_vars, self.poly_info.num_vars,
                "product has a multiplicand with wrong number of variables"
            );

            let m_ptr: *const DenseMultilinearExtension<F> = Arc::as_ptr(&m);

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

    pub fn evaluate(&self, point: Vec<F>) -> F {
        self.products
            .iter()
            .map(|(c, p)| {
                *c * p
                    .iter()
                    .map(|&i| self.flat_ml_extensions[i].evaluate(&point.to_vec()))
                    .product::<F>()
            })
            .sum()
    }
}