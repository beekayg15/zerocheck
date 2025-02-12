use ark_ff::PrimeField;
use ark_poly::univariate::DensePolynomial;


#[derive(Clone)]
pub struct Proof<F: PrimeField> {
    pub(crate) q: DensePolynomial<F>,
    pub(crate) f: DensePolynomial<F>
}