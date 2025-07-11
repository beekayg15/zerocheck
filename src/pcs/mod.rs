use anyhow::Error;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use std::fmt::Debug;

pub mod multilinear_pcs;
pub mod univariate_pcs;

pub trait PolynomialCommitmentScheme: Clone + Sized {
    type VerifierKey: Clone + Sync;
    type CommitterKey<'a>: Clone + Sync;
    type Commitment: Debug + Clone + CanonicalSerialize + CanonicalDeserialize + Send;
    type OpeningProof: Clone + Send + Debug;
    type PCSParams: Clone;
    type Polynomial: Clone;
    type PolynomialInput: Clone;
    type PolynomialOutput: Clone + PrimeField;

    fn setup<'a>(
        pp: &'a Self::PCSParams,
    ) -> Result<(Self::CommitterKey<'a>, Self::VerifierKey), Error>;

    fn commit<'a>(
        ck: &'a Self::CommitterKey<'_>,
        poly: &'a Self::Polynomial,
    ) -> Result<Self::Commitment, Error>;

    fn batch_commit<'a>(
        ck: &'a Self::CommitterKey<'_>,
        poly: &'a Vec<Self::Polynomial>,
    ) -> Result<Vec<Self::Commitment>, Error>;

    fn open<'a>(
        ck: &'a Self::CommitterKey<'_>,
        comm: &'a Self::Commitment,
        poly: &'a Self::Polynomial,
        point: Self::PolynomialInput,
    ) -> Result<Self::OpeningProof, Error>;

    fn batch_open<'a>(
        ck: &'a Self::CommitterKey<'_>,
        comm: &'a Vec<Self::Commitment>,
        poly: &'a Vec<Self::Polynomial>,
        point: Self::PolynomialInput,
    ) -> Result<Vec<Self::OpeningProof>, Error>;

    fn check<'a>(
        vk: &'a Self::VerifierKey,
        opening_proof: &'a Self::OpeningProof,
        comm: &'a Self::Commitment,
        point: Self::PolynomialInput,
        value: Self::PolynomialOutput,
    ) -> Result<bool, Error>;

    fn batch_check<'a>(
        vk: &'a Self::VerifierKey,
        opening_proof: &'a Vec<Self::OpeningProof>,
        comm: &'a Vec<Self::Commitment>,
        point: Self::PolynomialInput,
        value: Vec<Self::PolynomialOutput>,
    ) -> Result<bool, Error>;

    fn extract_pure_commitment(comm: &Self::Commitment) -> Result<Self::Commitment, Error>;
}
