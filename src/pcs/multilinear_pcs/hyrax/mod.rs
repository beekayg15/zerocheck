use std::marker::PhantomData;
use ark_crypto_primitives::sponge::{poseidon::{PoseidonConfig, PoseidonSponge}, Absorb};
use ark_ec::AffineRepr;
use ark_poly::DenseMultilinearExtension;
use ark_poly_commit::{
    hyrax::{HyraxCommitment, HyraxCommitmentState, HyraxCommitterKey, HyraxPC, HyraxProof, HyraxVerifierKey}, LabeledCommitment, LabeledPolynomial, PolynomialCommitment
};
use ark_std::rand::thread_rng;
use poseidon_config::poseidon_parameters;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator};
use rayon::iter::ParallelIterator;
use ark_crypto_primitives::sponge::CryptographicSponge;

use crate::pcs::PolynomialCommitmentScheme;

pub mod data_structures;
use data_structures::*;

pub mod poseidon_config;

#[derive(Debug, Clone)]
pub struct Hyrax<G: AffineRepr> {
    _affine_data: PhantomData<G>,
}

impl<G> PolynomialCommitmentScheme for Hyrax<G> 
    where 
        G: AffineRepr,
        G::ScalarField: Absorb
{
    type Commitment = Commitment<G>;
    type CommitterKey<'a> = HyraxCommitterKey<G>;
    type VerifierKey = HyraxVerifierKey<G>;
    type OpeningProof = HyraxProof<G>;
    type PCSParams = usize;
    type Polynomial = DenseMultilinearExtension<G::ScalarField>;
    type PolynomialInput = Vec<G::ScalarField>;
    type PolynomialOutput = G::ScalarField;
    
    fn setup<'a> (
        pp: &'a Self::PCSParams
    ) -> Result<(Self::CommitterKey<'a>, Self::VerifierKey), anyhow::Error> {
        let rng = &mut thread_rng();
        let params = HyraxPC::<G, Self::Polynomial>::setup(
            0, 
            Some(*pp), 
            rng
        ).unwrap();

        Ok((params.clone(), params.clone()))
    }
    
    fn commit<'a> (
        ck: &'a Self::CommitterKey<'_>,
        poly: &'a Self::Polynomial,
    ) -> Result<Self::Commitment, anyhow::Error> {
        let lp = LabeledPolynomial::<G::ScalarField, Self::Polynomial>::new(
            "".to_owned(),
            (*poly).clone(),
            None,
            None
        );

        let rng = &mut thread_rng();

        let (comm, comm_state) = HyraxPC::<G, Self::Polynomial>::commit(
            ck, 
            &[lp], 
            Some(rng)
        ).unwrap();

        Ok(Commitment { 
            commitment: comm[0].commitment().clone(),
            commitment_state: comm_state[0].clone()
        })
    }
    
    fn batch_commit<'a> (
        ck: &'a Self::CommitterKey<'_>,
        poly: &'a Vec<Self::Polynomial>,
    ) -> Result<Vec<Self::Commitment>, anyhow::Error> {
        let lp: Vec<LabeledPolynomial<G::ScalarField, Self::Polynomial>> = poly
            .par_iter()
            .map(|p| {
                LabeledPolynomial::<G::ScalarField, Self::Polynomial>::new(
                    "".to_owned(),
                    p.clone(),
                    None,
                    None
                )
            })
            .collect();

        let rng = &mut thread_rng();

        let (comm, comm_state) = HyraxPC::<G, Self::Polynomial>::commit(
            ck, 
            &lp, 
            Some(rng)
        ).unwrap();

        let result = comm
            .par_iter()
            .zip(comm_state)
            .map(|(cm, cm_state)| {
                Commitment{
                    commitment: cm.commitment().clone(),
                    commitment_state: cm_state
                }
            })
            .collect();

        Ok(result)
    }
    
    fn open<'a> (
        _ck: &'a Self::CommitterKey<'_>,
        _comm: &'a Self::Commitment,
        _poly: &'a Self::Polynomial,
        _point: Self::PolynomialInput,
    ) -> Result<Self::OpeningProof, anyhow::Error> {
        todo!()
    }
    
    fn batch_open<'a> (
        ck: &'a Self::CommitterKey<'_>,
        comm: &'a Vec<Self::Commitment>,
        poly: &'a Vec<Self::Polynomial>,
        point: Self::PolynomialInput,
    ) -> Result<Vec<Self::OpeningProof>, anyhow::Error> {
        let rng = &mut thread_rng();
        let poseidon_config: PoseidonConfig<G::ScalarField>  = poseidon_parameters();
        let mut sponge = PoseidonSponge::new(&poseidon_config);

        let lp: Vec<LabeledPolynomial<G::ScalarField, Self::Polynomial>> = poly
            .par_iter()
            .map(|p| {
                LabeledPolynomial::<G::ScalarField, Self::Polynomial>::new(
                    "".to_owned(),
                    p.clone(),
                    None,
                    None
                )
            })
            .collect();

        let lc: Vec<LabeledCommitment<HyraxCommitment<G>>> = comm
            .par_iter()
            .map(|cm| {
                LabeledCommitment::<HyraxCommitment<G>>::new(
                    "".to_owned(),
                    cm.commitment.clone(),
                    None
                )
            })
            .collect();

        let states: Vec<HyraxCommitmentState<G::ScalarField>> = comm
            .par_iter()
            .map(|cm| {
                cm.commitment_state.clone()
            })
            .collect();

        Ok(HyraxPC::<G, Self::Polynomial>::open(
            ck, 
            &lp, 
            &lc, 
            &point, 
            &mut sponge, 
            &states, 
            Some(rng)
        )?)
    }
    
    fn check<'a> (
        _vk: &'a Self::VerifierKey,
        _opening_proof: &'a Self::OpeningProof,
        _comm: &'a Self::Commitment,
        _point: Self::PolynomialInput,
        _value: Self::PolynomialOutput,
    ) -> Result<bool, anyhow::Error> {
        todo!()
    }
    
    fn batch_check<'a> (
        vk: &'a Self::VerifierKey,
        opening_proof: &'a Vec<Self::OpeningProof>,
        comm: &'a Vec<Self::Commitment>,
        point: Self::PolynomialInput,
        value: Vec<Self::PolynomialOutput>,
    ) -> Result<bool, anyhow::Error> {
        let lc: Vec<LabeledCommitment<HyraxCommitment<G>>> = comm
            .par_iter()
            .map(|cm| {
                LabeledCommitment::<HyraxCommitment<G>>::new(
                    "".to_owned(),
                    cm.commitment.clone(),
                    None
                )
            })
            .collect();

        let rng = &mut thread_rng();
        let poseidon_config: PoseidonConfig<G::ScalarField>  = poseidon_parameters();
        let mut sponge = PoseidonSponge::new(&poseidon_config);

        Ok(HyraxPC::<G, Self::Polynomial>::check(
            vk, 
            &lc, 
            &point, 
            value, 
            opening_proof, 
            &mut sponge, 
            Some(rng)
        )?)
    }
}