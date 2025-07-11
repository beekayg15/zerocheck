use ark_ff::PrimeField;
use ark_poly::univariate::DensePolynomial;
use std::marker::PhantomData;

/// This is the data structure of the proof to be sent to the verifer,
/// to prove that there exists a quotient polynomial q(X), for which,
/// for all x \in Fp, f(x) = q(x).z_H(x), where, z_H(X) is the vanishing
/// polynomial over the domain H, for which prover claims that
/// for x \in H, f(x) = 0.
///  
/// q - stores the quotient polynomail as DensePolynomial<F>
/// f - stores the resulting polynomial of (g)^2 * h as DensePolynomial<F>

#[derive(Clone, Debug)]
pub struct Proof<F: PrimeField> {
    pub(crate) q: DensePolynomial<F>,
    pub(crate) f: DensePolynomial<F>,
}

#[derive(Clone)]
pub struct ZeroCheckParams<F: PrimeField> {
    pub _field_data: PhantomData<F>,
}
