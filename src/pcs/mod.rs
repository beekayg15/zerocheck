use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use anyhow::Error;
use std::fmt::Debug;

pub mod univariate_pcs;
pub mod multilinear_pcs;

pub trait PolynomialCommitmentScheme: Clone + Sized {
    type VerifierKey: Clone + Sync;
    type CommitterKey<'a>: Clone + Sync;
    type Commitment: Clone + CanonicalSerialize + CanonicalDeserialize + Send + Debug;
    type OpeningProof: Clone + Send + Debug;
    type PCSParams: Clone;
    type Polynomial: Clone;
    type PolynomialInput: Clone;
    type PolynomialOutput: Clone;

    fn setup<'a> (
        pp: &'a Self::PCSParams
    ) -> Result<(Self::CommitterKey<'a>, Self::VerifierKey), Error>;

    fn commit<'a> (
        ck: &'a Self::CommitterKey<'_>,
        poly: &'a Self::Polynomial,
    ) -> Result<Self::Commitment, Error>;

    fn open<'a> (
        ck: &'a Self::CommitterKey<'_>,
        comm: &'a Self::Commitment,
        poly: &'a Self::Polynomial,
        point: Self::PolynomialInput,
    ) -> Result<Self::OpeningProof, Error>;

    fn check<'a> (
        vk: &'a Self::VerifierKey,
        opening_proof: &'a Self::OpeningProof,
        comm: &'a Self::Commitment,
        point: Self::PolynomialInput,
        value: Self::PolynomialOutput,
    ) -> Result<bool, Error>;
}