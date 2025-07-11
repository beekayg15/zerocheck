use ark_ec::AffineRepr;
use ark_poly_commit::hyrax::{HyraxCommitment, HyraxCommitmentState};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct Commitment<G: AffineRepr> {
    pub(crate) commitment: HyraxCommitment<G>,
    pub(crate) commitment_state: HyraxCommitmentState<G::ScalarField>,
}
