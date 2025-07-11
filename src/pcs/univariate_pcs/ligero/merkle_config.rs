use ark_crypto_primitives::{
    crh::{CRHScheme, TwoToOneCRHScheme},
    merkle_tree::{Config, IdentityDigestConverter},
    sponge::Absorb,
};
use std::marker::PhantomData;

use crate::PrimeField;

use super::sha256_config::{Sha256Hasher, Sha256TwoToOneHasher};

#[derive(Debug, Clone)]
pub struct MerkleConfig<F: PrimeField> {
    _field_data: PhantomData<F>,
}

// type CompressH<F> = TwoToOneCRH<F>;

impl<F: PrimeField + Absorb> Config for MerkleConfig<F> {
    type Leaf = Vec<u8>;
    type LeafHash = Sha256Hasher<F>;
    type LeafDigest = <Sha256Hasher<F> as CRHScheme>::Output;
    type InnerDigest = <Sha256TwoToOneHasher<F> as TwoToOneCRHScheme>::Output;
    type LeafInnerDigestConverter = IdentityDigestConverter<Self::LeafDigest>;
    type TwoToOneHash = Sha256TwoToOneHasher<F>;
}
