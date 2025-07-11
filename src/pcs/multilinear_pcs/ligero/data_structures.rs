use crate::pcs::multilinear_pcs::ligero::sha256_config::Sha256FieldsToBytesHasher;
use ark_crypto_primitives::sponge::Absorb;
use ark_ff::PrimeField;
use ark_poly_commit::linear_codes::{LinCodePCCommitment, LinCodePCCommitmentState};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use super::merkle_config::MerkleConfig;

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct Commitment<F: PrimeField + Absorb> {
    pub(crate) commitment: LinCodePCCommitment<MerkleConfig<F>>,
    pub(crate) state: LinCodePCCommitmentState<F, Sha256FieldsToBytesHasher<F>>,
}
