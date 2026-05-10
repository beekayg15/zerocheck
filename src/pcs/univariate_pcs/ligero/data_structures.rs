use super::sha256_config::Sha256FieldsToBytesHasher;
use crate::pcs::linear_code_pcs::{LinCodePCCommitment, LinCodePCCommitmentState};
use crate::pcs::univariate_pcs::ligero::FieldsToBytesHasher;
use ark_crypto_primitives::sponge::Absorb;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};

use super::merkle_config::MerkleConfig;

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct Commitment<F: PrimeField + Absorb> {
    pub(crate) commitment: LinCodePCCommitment<MerkleConfig<F>>,
    pub(crate) state: LinCodePCCommitmentState<F, Sha256FieldsToBytesHasher<F>>,
}

#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct CommitmentPoseidon<F: PrimeField + Absorb> {
    pub(crate) commitment: LinCodePCCommitment<MerkleConfig<F>>,
    pub(crate) state: LinCodePCCommitmentState<F, FieldsToBytesHasher<F>>,
}
