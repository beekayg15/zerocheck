use anyhow::{Error, Ok};
use ark_ec::pairing::Pairing;
use ark_ff::PrimeField;
use ark_poly::{
    DenseMultilinearExtension,
    Polynomial, MultilinearExtension
};
use ark_poly_commit::multilinear_pc::data_structures::{
    Commitment, CommitterKey, Proof as MPCProof, VerifierKey
};
use ark_std::rand::{
    thread_rng, Rng 
};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::{
    collections::HashMap, cmp::max,
    marker::PhantomData, sync::Arc
};

use super::sumcheck::prover::ProverMsg;

#[derive(Clone)]
pub struct ZeroCheckParams<E: Pairing> {
    pub(crate) vk: VerifierKey<E>,
    pub(crate) ck: CommitterKey<E>,
}

#[derive(Clone)]
pub struct InputParams {
    pub num_vars: usize,
}

/// This is the data structure of the proof to be sent to the verifer,
/// to prove that the output polynomial f(x') = l(x').r(x').s(x')
/// evaluates to 0 over a given hypercube
#[derive(Clone, Debug)]
pub struct Proof<E: Pairing> {
    // list of prover message sent during the interactive sumcheck protocol
    pub(crate) prover_msgs: Vec<ProverMsg<E::ScalarField>>,
    pub(crate) inp_mle_commitments: Vec<Commitment<E>>,
    pub(crate) inp_mle_evals: Vec<E::ScalarField>,
    pub(crate) inp_mle_openings: Vec<MPCProof<E>>
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
    pub flat_ml_extensions: Vec<Arc<DenseMultilinearExtension<F>>>,
    raw_pointers_lookup_table: HashMap<*const DenseMultilinearExtension<F>, usize>,
}

impl<F: PrimeField> VirtualPolynomial<F> {
    /// Creates an empty virtual polynomial with `num_variables`
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

    /// Add a product of list of multilinear extensions to self
    /// Returns an error if the list is empty, or the MLE has a different
    /// `num_vars` from self
    ///
    /// The MLEs will be multiplied together, and then multiplied by the scalar
    /// `coefficient`
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

    /// Multiple the current VirtualPolynomial by an MLE:
    /// - add the MLE to the MLE list;
    /// - multiple each product by MLE and its coefficient
    ///
    /// Returns an error if the MLE has a different `num_vars` from self
    pub fn mul_mle(
        &mut self,
        mle: Arc<DenseMultilinearExtension<F>>,
        coefficient: F
    ) -> Result<(), Error> {
        assert_eq!(
            self.poly_info.num_vars, mle.num_vars(),
            "Mismatch in number of variables"
        );

        let mle_ptr: *const DenseMultilinearExtension<F> = Arc::as_ptr(&mle);

        let mle_index = match self.raw_pointers_lookup_table.get(&mle_ptr) {
            Some(&p) => p,
            _ => {
                self.raw_pointers_lookup_table
                    .insert(mle_ptr, self.flat_ml_extensions.len());
                self.flat_ml_extensions.push(mle);
                self.flat_ml_extensions.len() - 1
            },
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
    pub fn evaluate(&self, point: Vec<F>) -> F {
        assert_eq!(
            point.len(), 
            self.poly_info.num_vars, 
            "vector provided does not match number of varibales in MLE"
        );

        self.products
            .iter()
            .map(|(c, p)| {
                *c * p
                    .iter()
                    .map(|&i| self.flat_ml_extensions[i].evaluate(
                        &point.to_vec())
                    )
                    .product::<F>()
            })
            .sum()
    }

    // Input poly f(x) and a random vector r, output
    //      \hat f(x) = \sum_{x_i \in eval_x} f(x_i) eq(x, r)
    // where
    //      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
    //
    // This function is used in ZeroCheck
    pub fn build_f_hat(&self, r: &[F]) -> Result<Self, Error> {

        assert_eq!(
            self.poly_info.num_vars, r.len(),
            "Given vector does not match the number of variables"
        );

        let eq_x_r = build_eq_x_r(r)?;
        let mut res = self.clone();
        res.mul_mle(eq_x_r, F::one())?;

        Ok(res)
    }
}

/// This function build the eq(x, r) polynomial for any given r.
///
/// Evaluate
///      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
/// over r, which is
///      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))
pub fn build_eq_x_r<F: PrimeField> (
    r: &[F],
) -> Result<Arc<DenseMultilinearExtension<F>>, Error> {
    let evals = build_eq_x_r_vec(r)?;
    let mle = DenseMultilinearExtension::from_evaluations_vec(
        r.len(), 
        evals
    );

    Ok(Arc::new(mle))
}

/// This function build the eq(x, r) polynomial for any given r, and output the
/// evaluation of eq(x, r) in its vector form.
///
/// Evaluate
///      eq(x,y) = \prod_i=1^num_var (x_i * y_i + (1-x_i)*(1-y_i))
/// over r, which is
///      eq(x,y) = \prod_i=1^num_var (x_i * r_i + (1-x_i)*(1-r_i))
pub fn build_eq_x_r_vec<F: PrimeField> (r: &[F]) -> Result<Vec<F>, Error> {
    // we build eq(x,r) from its evaluations
    // we want to evaluate eq(x,r) over x \in {0, 1}^num_vars
    // for example, with num_vars = 4, x is a binary vector of 4, then
    //  0 0 0 0 -> (1-r0)   * (1-r1)    * (1-r2)    * (1-r3)
    //  1 0 0 0 -> r0       * (1-r1)    * (1-r2)    * (1-r3)
    //  0 1 0 0 -> (1-r0)   * r1        * (1-r2)    * (1-r3)
    //  1 1 0 0 -> r0       * r1        * (1-r2)    * (1-r3)
    //  ....
    //  1 1 1 1 -> r0       * r1        * r2        * r3
    // we will need 2^num_var evaluations

    let mut eval = Vec::new();
    build_eq_x_r_helper(r, &mut eval)?;

    Ok(eval)
}

/// A helper function to build eq(x, r) recursively.
/// This function takes `r.len()` steps, and for each step it requires a maximum
/// `r.len()-1` multiplications.
fn build_eq_x_r_helper<F: PrimeField> (r: &[F], buf: &mut Vec<F>) -> Result<(), Error> {
    if r.is_empty() {
        assert!(!r.is_empty(), "invalid challenge");
    } else if r.len() == 1 {
        // initializing the buffer with [1-r_0, r_0]
        buf.push(F::one() - r[0]);
        buf.push(r[0]);
    } else {
        build_eq_x_r_helper(&r[1..], buf)?;

        // suppose at the previous step we received [b_1, ..., b_k]
        // for the current step we will need
        // if x_0 = 0:   (1-r0) * [b_1, ..., b_k]
        // if x_0 = 1:   r0 * [b_1, ..., b_k]
        // let mut res = vec![];
        // for &b_i in buf.iter() {
        //     let tmp = r[0] * b_i;
        //     res.push(b_i - tmp);
        //     res.push(tmp);
        // }
        // *buf = res;

        let mut res = vec![F::zero(); buf.len() << 1];
        res.iter_mut().enumerate().for_each(|(i, val)| {
            let bi = buf[i >> 1];
            let tmp = r[0] * bi;
            if i & 1 == 0 {
                *val = bi - tmp;
            } else {
                *val = tmp;
            }
        });
        *buf = res;
    }

    Ok(())
}

/// Sample a random list of multilinear polynomials
/// Returns
/// - the list of polynomials,
/// - its sum of polynomial evaluations over the boolean hypercube
pub fn random_mle_list<F: PrimeField>(
    nv: usize,
    degree: usize
) -> (Vec<Arc<DenseMultilinearExtension<F>>>, F) {
    let mut rng = thread_rng();

    let mut multiplicands = Vec::with_capacity(degree);
    for _ in 0..degree {
        multiplicands.push(Vec::with_capacity(1 << nv))
    }
    let mut sum = F::zero();

    for _ in 0..(1 << nv) {
        let mut product = F::one();

        for e in multiplicands.iter_mut() {
            let val = F::rand(&mut rng);
            e.push(val);
            product *= val;
        }
        sum += product;
    }

    let list = multiplicands
        .into_iter()
        .map(|x| Arc::new(DenseMultilinearExtension::from_evaluations_vec(
            nv, 
            x)))
        .collect();

    (list, sum)
}

// Build a randomize list of mle-s whose sum is zero
pub fn random_zero_mle_list<F: PrimeField> (
    nv: usize,
    degree: usize,
) -> Vec<Arc<DenseMultilinearExtension<F>>> {
    let mut rng = thread_rng();

    let mut multiplicands = Vec::with_capacity(degree);
    for _ in 0..degree {
        multiplicands.push(Vec::with_capacity(1 << nv))
    }
    for _ in 0..(1 << nv) {
        multiplicands[0].push(F::zero());
        for e in multiplicands.iter_mut().skip(1) {
            e.push(F::rand(&mut rng));
        }
    }

    let list = multiplicands
        .into_iter()
        .map(|x| Arc::new(DenseMultilinearExtension::from_evaluations_vec(nv, x)))
        .collect();
    list
}

/// Sample a random virtual polynomial, return the polynomial and its sum
pub fn rand<F: PrimeField>(
    nv: usize,
    num_multiplicands_range: (usize, usize),
    num_products: usize,
) -> Result<(VirtualPolynomial<F>, F), Error> {
    let mut rng = thread_rng();
    let mut sum = F::zero();

    let mut poly = VirtualPolynomial::new(nv);
    for _ in 0..num_products {
        let num_multiplicands =
            rng.gen_range(num_multiplicands_range.0..num_multiplicands_range.1);
        let (product, product_sum) = random_mle_list(nv, num_multiplicands);
        let coefficient = F::rand(&mut rng);
        poly.add_product(product.into_iter(), coefficient);
        sum += product_sum * coefficient;
    }

    Ok((poly, sum))
}

/// Sample a random virtual polynomial that evaluates to zero everywhere
/// over the boolean hypercube.
pub fn rand_zero<F: PrimeField> (
    nv: usize,
    num_multiplicands_range: (usize, usize),
    num_products: usize,
) -> VirtualPolynomial<F> {
    let mut rng = thread_rng();
    let mut poly = VirtualPolynomial::new(nv);
    for _ in 0..num_products {
        let num_multiplicands =
            rng.gen_range(num_multiplicands_range.0..num_multiplicands_range.1);
        let product = random_zero_mle_list(nv, num_multiplicands);
        let coefficient = F::rand(&mut rng);
        poly.add_product(product.into_iter(), coefficient);
    }

    poly
}

/// Sample a random virtual polynomial of the form f = ghs + (1-s)(g+h) - o
/// that evaluates to zero everywhere over the boolean hypercube
pub fn custom_zero_test_case<F: PrimeField> (
    nv: usize
) -> VirtualPolynomial<F> {
    let mut poly = VirtualPolynomial::<F>::new(nv);

    let rand_g_evals: Vec<F> = (0..(1 << nv))
            .into_par_iter()
            .map(|_| F::rand(&mut ark_std::rand::thread_rng()))
            .collect();
    let rand_h_evals: Vec<F> = (0..(1 << nv))
            .into_par_iter()
            .map(|_| F::rand(&mut ark_std::rand::thread_rng()))
            .collect();
    let rand_s_evals: Vec<F> = (0..(1 << nv))
            .into_par_iter()
            .map(|_| F::rand(&mut ark_std::rand::thread_rng()))
            .collect();

    let one_minus_s_evals: Vec<F> = (0..(1 << nv))
        .into_par_iter()
        .map(|i| F::one() - rand_s_evals[i])
        .collect();

    let g_plus_h_evals: Vec<F> = (0..(1 << nv))
        .into_par_iter()
        .map(|i| rand_h_evals[i] + rand_g_evals[i])
        .collect();

    let o_evals: Vec<F> = (0..(1 << nv))
        .into_par_iter()
        .map(|i| (
            rand_g_evals[i] * rand_h_evals[i] * rand_s_evals[i])
            + (one_minus_s_evals[i] * g_plus_h_evals[i])
        )
        .collect();

    let mut p1 = vec![];
    let mut p2 = vec![];
    let mut p3 = vec![];
    let mut p4 = vec![];
    let mut p5 = vec![];
    let mut p6 = vec![];

    let g_mle = Arc::new(DenseMultilinearExtension::<F>::from_evaluations_vec(
        nv, 
        rand_g_evals
    ));

    let h_mle = Arc::new(DenseMultilinearExtension::<F>::from_evaluations_vec(
        nv, 
        rand_h_evals
    ));

    let s_mle = Arc::new(DenseMultilinearExtension::<F>::from_evaluations_vec(
        nv, 
        rand_s_evals
    ));

    let o_mle = Arc::new(DenseMultilinearExtension::<F>::from_evaluations_vec(
        nv, 
        o_evals
    ));


    p1.push(g_mle.clone());
    p1.push(h_mle.clone());
    p1.push(s_mle.clone());

    p2.push(s_mle.clone());
    p2.push(g_mle.clone());

    p3.push(s_mle.clone());
    p3.push(h_mle.clone());

    p4.push(g_mle.clone());

    p5.push(h_mle.clone());

    p6.push(o_mle.clone());

    poly.add_product(p1, F::one());
    poly.add_product(p2, -F::one());
    poly.add_product(p3, -F::one());
    poly.add_product(p4, F::one());
    poly.add_product(p5, F::one());
    poly.add_product(p6, -F::one());

    poly
}