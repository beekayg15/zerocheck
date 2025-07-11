use ark_crypto_primitives::{
    crh::{CRHScheme, TwoToOneCRHScheme},
    merkle_tree::{Config, IdentityDigestConverter},
    sponge::Absorb,
};
use std::marker::PhantomData;

use crate::PrimeField;

use super::poseidon_config::{CustomPoseidonHasher, CustomTwoToOneHasher};

#[derive(Debug, Clone)]
pub struct MerkleConfig<F: PrimeField> {
    _field_data: PhantomData<F>,
}

// type CompressH<F> = TwoToOneCRH<F>;

impl<F: PrimeField + Absorb> Config for MerkleConfig<F> {
    type Leaf = Vec<u8>;
    type LeafHash = CustomPoseidonHasher<F>;
    type LeafDigest = <CustomPoseidonHasher<F> as CRHScheme>::Output;
    type InnerDigest = <CustomTwoToOneHasher<F> as TwoToOneCRHScheme>::Output;
    type LeafInnerDigestConverter = IdentityDigestConverter<Self::LeafDigest>;
    type TwoToOneHash = CustomTwoToOneHasher<F>;
}
