use crate::pcs::PolynomialCommitmentScheme;
use anyhow::{Error, Ok};
use ark_ff::PrimeField;
use ark_poly::Polynomial;
use ark_poly::{
    univariate::DensePolynomial, EvaluationDomain, Evaluations, GeneralEvaluationDomain,
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::ThreadPool;
use std::{cmp::max, collections::HashMap, marker::PhantomData, sync::Arc};

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
pub struct EvaluationInfo<F: PrimeField> {
    // maximum degree of the variables in the output polynomial
    pub max_multiplicand: usize,
    #[doc(hidden)]
    pub phantom: PhantomData<F>,
}

/// Struct to store the sum of products of MLEs
#[derive(Clone, Debug)]
pub struct VirtualEvaluation<F: PrimeField> {
    // information about the virtual polynomial
    pub evals_info: EvaluationInfo<F>,
    // list of (indexed) MLEs which are to be multiplied
    // together along with a coefficient
    pub products: Vec<(F, Vec<usize>)>,
    // list of dense multilinear extensions of multilinear
    // polynomials used
    pub univariate_evaluations: Vec<Arc<Evaluations<F>>>,
    raw_pointers_lookup_table: HashMap<*const Evaluations<F>, usize>,
}

impl<F: PrimeField> VirtualEvaluation<F> {
    /// Creates an empty virtual evaluation with `degree`
    pub fn new() -> Self {
        VirtualEvaluation {
            evals_info: EvaluationInfo {
                max_multiplicand: 0,
                phantom: PhantomData,
            },
            products: Vec::new(),
            univariate_evaluations: Vec::new(),
            raw_pointers_lookup_table: HashMap::new(),
        }
    }

    /// Add a product of list of Evaluations to self
    pub fn add_product(
        &mut self,
        product: impl IntoIterator<Item = Arc<Evaluations<F>>>,
        coefficient: F,
    ) {
        let product: Vec<Arc<Evaluations<F>>> = product.into_iter().collect();
        let mut indexed_product = Vec::with_capacity(product.len());
        assert!(!product.is_empty());

        self.evals_info.max_multiplicand = max(self.evals_info.max_multiplicand, product.len());

        for m in product {
            let m_ptr: *const Evaluations<F> = Arc::as_ptr(&m);

            if let Some(index) = self.raw_pointers_lookup_table.get(&m_ptr) {
                indexed_product.push(*index)
            } else {
                let curr_index = self.univariate_evaluations.len();
                self.univariate_evaluations.push(m.clone());
                self.raw_pointers_lookup_table.insert(m_ptr, curr_index);
                indexed_product.push(curr_index);
            }
        }

        self.products.push((coefficient, indexed_product));
    }

    /// Multiple the current VirtualEvaluation by an Evaluation:
    /// - add the Evaluation to the Evaluations list;
    /// - multiply each product by Evaluation and its coefficient
    pub fn mul_evaluations(
        &mut self,
        evals: Arc<Evaluations<F>>,
        coefficient: F,
    ) -> Result<(), Error> {
        let evals_ptr: *const Evaluations<F> = Arc::as_ptr(&evals);

        let evals_index = match self.raw_pointers_lookup_table.get(&evals_ptr) {
            Some(&p) => p,
            _ => {
                self.raw_pointers_lookup_table
                    .insert(evals_ptr, self.univariate_evaluations.len());
                self.univariate_evaluations.push(evals);
                self.univariate_evaluations.len() - 1
            }
        };

        for (prod_coef, indices) in self.products.iter_mut() {
            // - add the Evaluations to the Evaluations list;
            // - multiple each product by Evaluations and its coefficient.
            indices.push(evals_index);
            *prod_coef *= coefficient;
        }

        self.evals_info.max_multiplicand += 1;
        Ok(())
    }

    pub fn evaluate_at_point(&self, point: F) -> F {
        let mut result = F::zero();
        for (coef, indices) in self.products.iter() {
            let mut product = coef.clone();
            for &index in indices {
                product *= <Evaluations<F> as Clone>::clone(&self.univariate_evaluations[index])
                    .interpolate()
                    .evaluate(&point);
            }
            result += product;
        }
        result
    }
}

#[derive(Clone, Debug)]
pub struct PolynomialInfo<F: PrimeField> {
    // maximum degree of the variables in the output polynomial
    pub max_multiplicand: usize,
    // phantom data to ensure that the struct is generic over F
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
    pub univariate_polynomials: Vec<Arc<DensePolynomial<F>>>,
}

impl<F: PrimeField> VirtualPolynomial<F> {
    /// Creates an empty virtual polynomial with `num_variables`
    pub fn new(virtual_evaluation: VirtualEvaluation<F>, pool_run: Option<&ThreadPool>) -> Self {
        let univariate_polynomials: Vec<Arc<DensePolynomial<F>>>;
        if let Some(pool_run) = pool_run {
            univariate_polynomials = pool_run.install(|| {
                virtual_evaluation
                    .univariate_evaluations
                    .into_iter()
                    .map(|eval| Arc::new(<Evaluations<F> as Clone>::clone(&eval).interpolate()))
                    .collect()
            });
        } else {
            univariate_polynomials = virtual_evaluation
                .univariate_evaluations
                .into_iter()
                .map(|eval| Arc::new(<Evaluations<F> as Clone>::clone(&eval).interpolate()))
                .collect();
        }

        VirtualPolynomial {
            poly_info: PolynomialInfo {
                max_multiplicand: virtual_evaluation.evals_info.max_multiplicand,
                phantom: PhantomData,
            },
            products: virtual_evaluation.products,
            univariate_polynomials: univariate_polynomials,
        }
    }

    /// Returns the size (number of coefficients) of the virtual polynomial
    pub fn degree(&self) -> usize {
        self.products
            .iter()
            .map(|(_, indices)| {
                indices
                    .iter()
                    .map(|&index| self.univariate_polynomials[index].degree() + 1)
                    .sum::<usize>()
            })
            .max()
            .unwrap_or(0)
    }

    /// Evaluates the virtual polynomial at a given point
    pub fn evaluate(&self, point: F) -> F {
        let mut result = F::zero();
        for (coef, indices) in &self.products {
            let mut product = coef.clone();
            for &index in indices {
                product *= self.univariate_polynomials[index].evaluate(&point);
            }
            result += product;
        }
        result
    }

    /// Evaluates the virtual polynomial at a given domain
    pub fn evaluate_over_domain(
        &self,
        domain: GeneralEvaluationDomain<F>,
        pool_run: Option<&ThreadPool>,
    ) -> Vec<F> {
        let evals: Vec<Vec<F>>;
        if let Some(pool_run) = pool_run {
            evals = pool_run.install(|| {
                self.univariate_polynomials
                    .iter()
                    .map(|poly| {
                        (*poly.as_ref())
                            .clone()
                            .evaluate_over_domain(domain.clone())
                            .evals
                    })
                    .collect()
            });
        } else {
            evals = self
                .univariate_polynomials
                .iter()
                .map(|poly| {
                    <DensePolynomial<F> as Clone>::clone(&poly)
                        .evaluate_over_domain(domain)
                        .evals
                })
                .collect();
        }

        let mut result = vec![F::zero(); domain.size()];
        for i in 0..domain.size() {
            for (coef, indices) in &self.products {
                let mut product = coef.clone();
                for &index in indices {
                    product *= evals[index][i];
                }
                result[i] += product;
            }
        }
        result
    }
}

// Manually creates a VirtualEvaluation object in the form "ghs + (1-s)(g+h) - o"
// Where "o" is zeroizing polynomial that cancels "ghs + (1-s)(g+h)" to zero.
pub fn custom_zero_test_case<F: PrimeField>(degree: usize) -> VirtualEvaluation<F> {
    let mut poly = VirtualEvaluation::<F>::new();
    let domain = GeneralEvaluationDomain::<F>::new(degree).unwrap();

    let rand_g_evals: Vec<F> = (0..degree)
        .into_par_iter()
        .map(|_| F::rand(&mut ark_std::rand::thread_rng()))
        .collect();
    let rand_h_evals: Vec<F> = (0..degree)
        .into_par_iter()
        .map(|_| F::rand(&mut ark_std::rand::thread_rng()))
        .collect();
    let rand_s_evals: Vec<F> = (0..degree)
        .into_par_iter()
        .map(|_| F::rand(&mut ark_std::rand::thread_rng()))
        .collect();

    let one_minus_s_evals: Vec<F> = (0..degree)
        .into_par_iter()
        .map(|i| F::one() - rand_s_evals[i])
        .collect();

    let g_plus_h_evals: Vec<F> = (0..degree)
        .into_par_iter()
        .map(|i| rand_h_evals[i] + rand_g_evals[i])
        .collect();

    let o_evals: Vec<F> = (0..degree)
        .into_par_iter()
        .map(|i| {
            (rand_g_evals[i] * rand_h_evals[i] * rand_s_evals[i])
                + (one_minus_s_evals[i] * g_plus_h_evals[i])
        })
        .collect();

    let mut p1 = vec![];
    let mut p2 = vec![];
    let mut p3 = vec![];
    let mut p4 = vec![];
    let mut p5 = vec![];
    let mut p6 = vec![];

    let g_evals = Evaluations::from_vec_and_domain(rand_g_evals, domain);

    let h_evals = Evaluations::from_vec_and_domain(rand_h_evals, domain);

    let s_evals = Evaluations::from_vec_and_domain(rand_s_evals, domain);

    let o_evals = Evaluations::from_vec_and_domain(o_evals, domain);

    p1.push(Arc::new(g_evals.clone()));
    p1.push(Arc::new(h_evals.clone()));
    p1.push(Arc::new(s_evals.clone()));

    p2.push(Arc::new(s_evals.clone()));
    p2.push(Arc::new(g_evals.clone()));

    p3.push(Arc::new(s_evals.clone()));
    p3.push(Arc::new(h_evals.clone()));

    p4.push(Arc::new(g_evals.clone()));

    p5.push(Arc::new(h_evals.clone()));

    p6.push(Arc::new(o_evals.clone()));

    poly.add_product(p1, F::from(1));
    poly.add_product(p2, F::from(-1));
    poly.add_product(p3, F::from(-1));
    poly.add_product(p4, F::from(1));
    poly.add_product(p5, F::from(1));
    poly.add_product(p6, F::from(-1));

    poly
}

// Manually creating a VirtualEvalution object that evaluates to 0 any where in the domain.
// For each prod_size, generate a term that has the number of prod_size many polynomials.
// e.g. prod_size = 3: abc
//      prod_size = 2: ab
// In the end, generate a zeroizing polynomial that cancels out the input polynomial to zero.
pub fn custom_zero_test_case_with_products<F: PrimeField>(
    degree: usize,
    num_polys: usize,
    prod_sizes: Vec<usize>,
    pool_prepare: &rayon::ThreadPool,
) -> VirtualEvaluation<F> {
    // Make sure a term only has prod_size up to the number of polynomials
    for i in prod_sizes.iter() {
        assert!(i <= &num_polys);
    }

    let mut inp_evals = VirtualEvaluation::<F>::new();
    let domain = GeneralEvaluationDomain::<F>::new(degree).unwrap();

    // Randomly generate the evalutions for each polynomial
    let mut poly_evals = vec![];
    for _ in 0..num_polys {
        let rand_evals: Vec<F> = pool_prepare.install(|| {
            (0..degree)
                .into_par_iter()
                .map(|_| F::rand(&mut ark_std::rand::thread_rng()))
                .collect()
        });
        poly_evals.push(rand_evals);
    }

    // Manually creating each term while keeping track of zero evaluations.
    let mut zero_evals = vec![F::zero(); degree];
    for i in prod_sizes.iter() {
        let mut prod = vec![];
        let mut prod_evals = vec![F::one(); degree];
        for j in 0..*i {
            prod.push(Arc::new(Evaluations::from_vec_and_domain(
                poly_evals[j].clone(),
                domain,
            )));
            for k in 0..degree {
                prod_evals[k] *= poly_evals[j][k];
            }
        }
        inp_evals.add_product(prod, F::from(1));
        for i in 0..degree {
            zero_evals[i] += prod_evals[i];
        }
    }

    // Adding the zeroizing polynomial that cancels out the input polynomial to zero.
    let zero_evals = pool_prepare.install(|| Evaluations::from_vec_and_domain(zero_evals, domain));
    inp_evals.add_product(vec![Arc::new(zero_evals)], F::from(-1));

    inp_evals
}
