use std::marker::PhantomData;
use anyhow::Ok;
use ark_ec::pairing::Pairing;
use ark_poly::DenseMultilinearExtension;
use ark_poly_commit::multilinear_pc::{data_structures::{Commitment, CommitterKey, Proof, VerifierKey}, MultilinearPC};
use ark_std::rand::thread_rng;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::pcs::PolynomialCommitmentScheme;

#[derive(Clone, Debug)]
pub struct MPC<E: Pairing> {
    _pairing_data: PhantomData<E>,
}

impl<E: Pairing> PolynomialCommitmentScheme for MPC<E> {
    type CommitterKey<'a> = CommitterKey<E>;
    type VerifierKey = VerifierKey<E>;
    type Commitment = Commitment<E>;
    type OpeningProof = Proof<E>;
    type Polynomial = DenseMultilinearExtension<E::ScalarField>;
    type PolynomialInput = Vec<E::ScalarField>;
    type PolynomialOutput = E::ScalarField;
    type PCSParams = usize;

    fn setup(
        num_vars: &Self::PCSParams
    ) -> Result<(Self::CommitterKey<'_>, Self::VerifierKey), anyhow::Error> {
        
        // Setting up the MPC(MSM) Polynomial Commitment Scheme
        let rng = &mut thread_rng();
        let params = MultilinearPC::<E>::setup(*num_vars, rng);

        // Computing the verification key
        let (ck, vk) = MultilinearPC::<E>::trim(
            &params, 
            *num_vars,
        );

        Ok((ck, vk))
    }
    
    fn commit(
        ck: &Self::CommitterKey<'_>,
        poly: &Self::Polynomial,
    ) -> Result<Self::Commitment, anyhow::Error> {
        
        let comm = MultilinearPC::<E>::commit(
            &ck, 
            poly, 
        );

        Ok(comm)
    }
    
    fn open(
        ck: &Self::CommitterKey<'_>,
        _comm: &Self::Commitment,
        poly: &Self::Polynomial,
        point: Self::PolynomialInput,
    ) -> Result<Self::OpeningProof, anyhow::Error> {
        let opening_proof = MultilinearPC::<E>::open(
            &ck, 
            poly, 
            &point
        );

        Ok(opening_proof)
    }
    
    fn check(
        vk: &Self::VerifierKey,
        opening_proof: &Self::OpeningProof,
        comm: &Self::Commitment,
        point: Self::PolynomialInput,
        value: Self::PolynomialOutput,
    ) -> Result<bool, anyhow::Error> {
        let result = MultilinearPC::<E>::check(
            &vk,
            &comm,
            &point,
            value,
            opening_proof
        );

        Ok(result)
    }
    
    fn batch_commit<'a> (
        ck: &'a Self::CommitterKey<'_>,
        poly: &'a Vec<Self::Polynomial>,
    ) -> Result<Vec<Self::Commitment>, anyhow::Error> {
        let result: Vec<Self::Commitment> = poly
            .par_iter()
            .map(|p| {
                let comm = Self::commit(
                    &ck, 
                    p
                ).unwrap();
                
                comm
            })
            .collect();

        Ok(result)
    }
    
    fn batch_open<'a> (
        ck: &'a Self::CommitterKey<'_>,
        comm: &'a Vec<Self::Commitment>,
        poly: &'a Vec<Self::Polynomial>,
        point: Self::PolynomialInput,
    ) -> Result<Vec<Self::OpeningProof>, anyhow::Error> {
        let result = poly
            .par_iter()
            .zip(comm) 
            .map(|(p, c)| {
                let opening_proof = Self::open(
                    &ck, 
                    c, 
                    p, 
                    point.clone()
                ).unwrap();

                opening_proof
            })
            .collect();

        Ok(result)
    }
    
    fn batch_check<'a> (
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
                let valid = Self::check(
                    &vk,
                    proof,
                    cm,
                    point.clone(),
                    val
                ).unwrap();
                
                valid
            })
            .collect();

        Ok(
            result
                .into_iter()
                .fold(
                    true,
                    |res, valid| {
                        res & valid
                    }
                )
        )
    }
}