use ark_ec::pairing::Pairing;
use ark_poly::univariate::DensePolynomial;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_poly_commit::kzg10::Randomness;

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct Commitment<E: Pairing> {
    pub comm: E::G1Affine, 
    pub(crate) rand: Randomness<<E as Pairing>::ScalarField, DensePolynomial<E::ScalarField>>,
}