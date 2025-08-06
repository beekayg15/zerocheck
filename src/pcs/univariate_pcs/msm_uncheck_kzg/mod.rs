use anyhow::Ok;
use ark_ec::pairing::Pairing;
use ark_ec::AffineRepr;
use ark_poly::univariate::DensePolynomial;
use ark_poly_commit::kzg10::{Commitment as KZGCommitment, Powers, Proof, VerifierKey, KZG10};
use ark_std::rand::thread_rng;
use std::marker::PhantomData;

pub mod my_kzg10;
use my_kzg10::fast_commit_unchecked;

pub mod data_structures;
use data_structures::*;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;

use crate::pcs::PolynomialCommitmentScheme;

#[derive(Clone)]
pub struct KZG<E: Pairing> {
    _pairing_data: PhantomData<E>,
}

impl<E: Pairing> PolynomialCommitmentScheme for KZG<E> {
    type CommitterKey<'a> = Powers<'a, E>;
    type VerifierKey = VerifierKey<E>;
    type Commitment = Commitment<E>;
    type OpeningProof = Proof<E>;
    type Polynomial = DensePolynomial<E::ScalarField>;
    type PolynomialInput = E::ScalarField;
    type PolynomialOutput = E::ScalarField;
    type PCSParams = usize;

    fn setup(
        max_degree: &Self::PCSParams,
    ) -> Result<(Self::CommitterKey<'_>, Self::VerifierKey), anyhow::Error> {
        // Setting up the KZG(MSM) Polynomial Commitment Scheme
        let rng = &mut thread_rng();
        let params = KZG10::<E, DensePolynomial<E::ScalarField>>::setup(*max_degree, false, rng)
            .expect("PCS setup failed");

        // Computing the verification key
        let vk = Self::VerifierKey {
            g: params.powers_of_g[0],
            gamma_g: params.powers_of_gamma_g[&0],
            h: params.h,
            beta_h: params.beta_h,
            prepared_h: params.prepared_h.clone(),
            prepared_beta_h: params.prepared_beta_h.clone(),
        };

        // Computing the powers of the generator 'G'
        let powers_of_g = params.powers_of_g[..=(*max_degree)].to_vec();
        let powers_of_gamma_g = (0..=(*max_degree))
            .map(|i| params.powers_of_gamma_g[&i])
            .collect();

        let powers = Powers {
            powers_of_g: ark_std::borrow::Cow::Owned(powers_of_g),
            powers_of_gamma_g: ark_std::borrow::Cow::Owned(powers_of_gamma_g),
        };

        Ok((powers, vk))
    }

    fn commit(
        ck: &Self::CommitterKey<'_>,
        poly: &Self::Polynomial,
    ) -> Result<Self::Commitment, anyhow::Error> {
        // let (comm, r) =
        //     KZG10::<E, DensePolynomial<E::ScalarField>>::commit(&ck, poly, None, None).unwrap();

        let (comm, r) = fast_commit_unchecked(&ck, poly).unwrap();

        assert!(!comm.0.is_zero(), "Commitment should not be zero");
        assert!(!r.is_hiding(), "Commitment should not be hiding");

        Ok(Self::Commitment {
            comm: comm.0,
            rand: r,
        })
    }

    fn batch_commit(
        ck: &Self::CommitterKey<'_>,
        poly: &Vec<Self::Polynomial>,
    ) -> Result<Vec<Self::Commitment>, anyhow::Error> {
        let result: Vec<Self::Commitment> = poly
            .par_iter()
            .map(|p| {
                let (comm, r) =
                    KZG10::<E, DensePolynomial<E::ScalarField>>::commit(&ck, p, None, None)
                        .unwrap();

                assert!(!comm.0.is_zero(), "Commitment should not be zero");
                assert!(!r.is_hiding(), "Commitment should not be hiding");

                Self::Commitment {
                    comm: comm.0,
                    rand: r,
                }
            })
            .collect();

        Ok(result)
    }

    fn open(
        ck: &Self::CommitterKey<'_>,
        comm: &Self::Commitment,
        poly: &Self::Polynomial,
        point: Self::PolynomialInput,
    ) -> Result<Self::OpeningProof, anyhow::Error> {
        let opening_proof =
            KZG10::<E, DensePolynomial<E::ScalarField>>::open(&ck, poly, point, &comm.rand)
                .unwrap();

        Ok(opening_proof)
    }

    fn batch_open<'a>(
        ck: &'a Self::CommitterKey<'_>,
        comm: &'a Vec<Self::Commitment>,
        poly: &'a Vec<Self::Polynomial>,
        point: Self::PolynomialInput,
    ) -> Result<Vec<Self::OpeningProof>, anyhow::Error> {
        let result = poly
            .par_iter()
            .zip(comm)
            .map(|(p, c)| {
                let opening_proof =
                    KZG10::<E, DensePolynomial<E::ScalarField>>::open(&ck, p, point, &c.rand)
                        .unwrap();

                opening_proof
            })
            .collect();

        Ok(result)
    }

    fn check(
        vk: &Self::VerifierKey,
        opening_proof: &Self::OpeningProof,
        comm: &Self::Commitment,
        point: Self::PolynomialInput,
        value: Self::PolynomialOutput,
    ) -> Result<bool, anyhow::Error> {
        let kzg_commitment = KZGCommitment { 0: comm.comm };

        let result = KZG10::<E, DensePolynomial<E::ScalarField>>::check(
            &vk,
            &kzg_commitment,
            point,
            value,
            opening_proof,
        )
        .unwrap();

        Ok(result)
    }

    fn batch_check<'a>(
        vk: &'a Self::VerifierKey,
        opening_proof: &'a Vec<Self::OpeningProof>,
        comm: &'a Vec<Self::Commitment>,
        point: Self::PolynomialInput,
        value: Vec<Self::PolynomialOutput>,
    ) -> Result<bool, anyhow::Error> {
        let result: Vec<bool> = comm
            .par_iter()
            .zip(opening_proof)
            .zip(value)
            .map(|((cm, proof), val)| {
                let kzg_commitment = KZGCommitment { 0: cm.comm };

                let valid = KZG10::<E, DensePolynomial<E::ScalarField>>::check(
                    &vk,
                    &kzg_commitment,
                    point,
                    val,
                    proof,
                )
                .unwrap();

                valid
            })
            .collect();

        Ok(result.into_iter().fold(true, |res, valid| res & valid))
    }

    fn extract_pure_commitment(comm: &Self::Commitment) -> Result<Self::Commitment, anyhow::Error> {
        Ok(comm.clone())
    }
}
