use crate::pcs::univariate_pcs::kzg::data_structures::Commitment as UVKZGCommitment;
use crate::pcs::univariate_pcs::kzg::KZG as UnivariateKZG;
use crate::pcs::PolynomialCommitmentScheme;
use anyhow::Ok;
use ark_ec::pairing::Pairing;
use ark_poly::univariate::DensePolynomial;
use ark_poly::{DenseMultilinearExtension, DenseUVPolynomial};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use ark_poly_commit::kzg10::{Powers, Proof, VerifierKey};

pub mod data_structures;
use data_structures::*;

#[derive(Clone)]
pub struct MultilinearKZG<E: Pairing> {
    pub inner: UnivariateKZG<E>, // Reuse the KZG struct from univariate_pcs
}

impl<E: Pairing> PolynomialCommitmentScheme for MultilinearKZG<E> {
    type CommitterKey<'a> = Powers<'a, E>;
    type VerifierKey = VerifierKey<E>;
    type Commitment = Commitment<E>;
    type OpeningProof = Proof<E>;
    type Polynomial = DenseMultilinearExtension<E::ScalarField>; // ScalarField is ark_ff::PrimeField
    type PolynomialInput = Vec<E::ScalarField>;
    type PolynomialOutput = E::ScalarField;
    type PCSParams = usize;

    /// Setup for the Multilinear KZG Polynomial Commitment Scheme.
    /// This function initializes the KZG parameters for a given maximum degree.
    /// # Arguments
    /// * `max_degree`: 2 ^ `num_var`
    fn setup(
        max_degree: &Self::PCSParams,
    ) -> Result<(Self::CommitterKey<'_>, Self::VerifierKey), anyhow::Error> {
        UnivariateKZG::<E>::setup(max_degree)
    }

    fn commit(
        ck: &Self::CommitterKey<'_>,
        poly: &Self::Polynomial,
    ) -> Result<Self::Commitment, anyhow::Error> {
        // change a DenseMultilinearExtension to a DensePolynomial<E::ScalarField>
        let uv_poly = DensePolynomial::from_coefficients_vec(poly.evaluations.clone());

        let uv_comm = UnivariateKZG::<E>::commit(ck, &uv_poly)?;
        Ok(Commitment {
            comm: uv_comm.comm,
            rand: uv_comm.rand,
        })
    }

    fn open(
        ck: &Self::CommitterKey<'_>,
        comm: &Self::Commitment,
        poly: &Self::Polynomial,
        point: Self::PolynomialInput,
    ) -> Result<Self::OpeningProof, anyhow::Error> {
        let uv_comm = UVKZGCommitment {
            comm: comm.comm.clone(),
            rand: comm.rand.clone(),
        };
        let uv_poly = DensePolynomial::from_coefficients_vec(poly.evaluations.clone());
        let uv_point = point[0];

        UnivariateKZG::<E>::open(ck, &uv_comm, &uv_poly, uv_point)
    }

    fn check(
        vk: &Self::VerifierKey,
        opening_proof: &Self::OpeningProof,
        comm: &Self::Commitment,
        point: Self::PolynomialInput,
        value: Self::PolynomialOutput,
    ) -> Result<bool, anyhow::Error> {
        let uv_comm = UVKZGCommitment {
            comm: comm.comm.clone(),
            rand: comm.rand.clone(),
        };
        let uv_point = point[0];
        UnivariateKZG::<E>::check(vk, opening_proof, &uv_comm, uv_point, value)
    }

    fn batch_commit<'a>(
        ck: &'a Self::CommitterKey<'_>,
        poly: &'a Vec<Self::Polynomial>,
    ) -> Result<Vec<Self::Commitment>, anyhow::Error> {
        let uv_poly: Vec<DensePolynomial<E::ScalarField>> = poly
            .par_iter()
            .map(|p| DensePolynomial::from_coefficients_vec(p.evaluations.clone()))
            .collect();
        let res = UnivariateKZG::<E>::batch_commit(ck, &uv_poly)?;
        Ok(res
            .into_iter()
            .map(|comm| Commitment {
                comm: comm.comm,
                rand: comm.rand,
            })
            .collect())
    }

    fn batch_open<'a>(
        ck: &'a Self::CommitterKey<'_>,
        comm: &'a Vec<Self::Commitment>,
        poly: &'a Vec<Self::Polynomial>,
        point: Self::PolynomialInput,
    ) -> Result<Vec<Self::OpeningProof>, anyhow::Error> {
        let uv_comm: Vec<UVKZGCommitment<E>> = comm
            .par_iter()
            .map(|c| UVKZGCommitment {
                comm: c.comm.clone(),
                rand: c.rand.clone(),
            })
            .collect();
        let uv_poly: Vec<DensePolynomial<E::ScalarField>> = poly
            .par_iter()
            .map(|p| DensePolynomial::from_coefficients_vec(p.evaluations.clone()))
            .collect();
        let uv_point = point[0];
        UnivariateKZG::<E>::batch_open(ck, &uv_comm, &uv_poly, uv_point)
    }

    fn batch_check<'a>(
        vk: &'a Self::VerifierKey,
        opening_proof: &'a Vec<Self::OpeningProof>,
        comm: &'a Vec<Self::Commitment>,
        point: Self::PolynomialInput,
        value: Vec<Self::PolynomialOutput>,
    ) -> Result<bool, anyhow::Error> {
        let uv_comm: Vec<UVKZGCommitment<E>> = comm
            .par_iter()
            .map(|c| UVKZGCommitment {
                comm: c.comm.clone(),
                rand: c.rand.clone(),
            })
            .collect();
        let uv_point = point[0];
        UnivariateKZG::<E>::batch_check(vk, opening_proof, &uv_comm, uv_point, value)
    }
}
