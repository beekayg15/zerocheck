use std::marker::PhantomData;

use crate::pcs::PolynomialCommitmentScheme;
use anyhow::Ok;
use ark_crypto_primitives::sponge::poseidon::PoseidonConfig;
use ark_crypto_primitives::sponge::poseidon::PoseidonSponge;
use ark_crypto_primitives::sponge::CryptographicSponge;
use ark_crypto_primitives::{
    crh::{CRHScheme, TwoToOneCRHScheme},
    merkle_tree::Config,
    sponge::Absorb,
};
use ark_ff::PrimeField;
use ark_poly::univariate::DensePolynomial;
use ark_poly_commit::linear_codes::UnivariateLigero as UVLigero;
use ark_poly_commit::LabeledCommitment;
use ark_poly_commit::{
    linear_codes::{
        LigeroPCParams, LinCodePCCommitment, LinCodePCCommitmentState, LinCodePCProof,
        LinearCodePCS, LinearEncode,
    },
    LabeledPolynomial, PolynomialCommitment,
};
use ark_std::rand::thread_rng;
use merkle_config::MerkleConfig;
use rayon::iter::ParallelIterator;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator};

pub mod merkle_config;
pub mod sha256_config;
use sha256_config::{Sha256FieldsToBytesHasher, Sha256Sponge};
pub mod poseidon_config;
use poseidon_config::{poseidon_parameters, FieldsToBytesHasher};

pub mod data_structures;
use data_structures::*;

#[derive(Debug, Clone)]
pub struct Ligero<F: PrimeField> {
    _field_data: PhantomData<F>,
}

impl<F> PolynomialCommitmentScheme for Ligero<F>
where
    F: PrimeField + Absorb,
{
    type VerifierKey = LigeroPCParams<F, MerkleConfig<F>, Sha256FieldsToBytesHasher<F>>;
    type CommitterKey<'a> = LigeroPCParams<F, MerkleConfig<F>, Sha256FieldsToBytesHasher<F>>;
    type Commitment = Commitment<F>;
    type OpeningProof = LinCodePCProof<F, MerkleConfig<F>>;
    type PCSParams = usize;
    type Polynomial = DensePolynomial<F>;
    type PolynomialInput = F;
    type PolynomialOutput = F;

    fn setup<'a>(
        _pp: &'a Self::PCSParams,
    ) -> Result<(Self::CommitterKey<'a>, Self::VerifierKey), anyhow::Error> {
        let rng = &mut thread_rng();

        let leaf_hash_param =
            <<MerkleConfig<F> as Config>::LeafHash as CRHScheme>::setup(rng).unwrap();
        let two_to_one_hash_param =
            <<MerkleConfig<F> as Config>::TwoToOneHash as TwoToOneCRHScheme>::setup(rng)
                .unwrap()
                .clone();
        let col_hash_params = <Sha256FieldsToBytesHasher<F> as CRHScheme>::setup(rng).unwrap();

        let params =
            UVLigero::<F, MerkleConfig<F>, Self::Polynomial, Sha256FieldsToBytesHasher<F>>::setup(
                0,
                Some(0),
                rng,
                leaf_hash_param,
                two_to_one_hash_param,
                col_hash_params,
            );

        Ok((params.clone(), params.clone()))
    }

    fn commit<'a>(
        ck: &'a Self::CommitterKey<'_>,
        poly: &'a Self::Polynomial,
    ) -> Result<Self::Commitment, anyhow::Error> {
        let lp = LabeledPolynomial::<F, Self::Polynomial>::new(
            "".to_owned(),
            (*poly).clone(),
            None,
            None,
        );

        let rng = &mut thread_rng();

        let (comm, comm_state) = LinearCodePCS::<
            UVLigero<F, MerkleConfig<F>, Self::Polynomial, Sha256FieldsToBytesHasher<F>>,
            F,
            Self::Polynomial,
            MerkleConfig<F>,
            Sha256FieldsToBytesHasher<F>,
        >::commit(ck, &[lp], Some(rng))
        .unwrap();

        Ok(Self::Commitment {
            commitment: comm[0].commitment().clone(),
            state: comm_state[0].clone(),
        })
    }

    fn batch_commit<'a>(
        ck: &'a Self::CommitterKey<'_>,
        poly: &'a Vec<Self::Polynomial>,
    ) -> Result<Vec<Self::Commitment>, anyhow::Error> {
        let lp: Vec<LabeledPolynomial<F, Self::Polynomial>> = poly
            .par_iter()
            .map(|p| {
                LabeledPolynomial::<F, Self::Polynomial>::new("".to_owned(), p.clone(), None, None)
            })
            .collect();

        let rng = &mut thread_rng();

        let (comm, comm_state) = LinearCodePCS::<
            UVLigero<F, MerkleConfig<F>, Self::Polynomial, Sha256FieldsToBytesHasher<F>>,
            F,
            Self::Polynomial,
            MerkleConfig<F>,
            Sha256FieldsToBytesHasher<F>,
        >::commit(ck, &lp, Some(rng))
        .unwrap();

        let result = comm
            .par_iter()
            .zip(comm_state)
            .map(|(cm, cm_state)| Self::Commitment {
                commitment: cm.commitment().clone(),
                state: cm_state,
            })
            .collect();

        Ok(result)
    }

    fn open<'a>(
        _ck: &'a Self::CommitterKey<'_>,
        _comm: &'a Self::Commitment,
        _poly: &'a Self::Polynomial,
        _point: Self::PolynomialInput,
    ) -> Result<Self::OpeningProof, anyhow::Error> {
        todo!()
    }

    fn batch_open<'a>(
        ck: &'a Self::CommitterKey<'_>,
        comm: &'a Vec<Self::Commitment>,
        poly: &'a Vec<Self::Polynomial>,
        point: Self::PolynomialInput,
    ) -> Result<Vec<Self::OpeningProof>, anyhow::Error> {
        let rng = &mut thread_rng();
        let mut sponge = Sha256Sponge::new();

        let lp: Vec<LabeledPolynomial<F, Self::Polynomial>> = poly
            .par_iter()
            .map(|p| {
                LabeledPolynomial::<F, Self::Polynomial>::new("".to_owned(), p.clone(), None, None)
            })
            .collect();

        let lc: Vec<LabeledCommitment<LinCodePCCommitment<MerkleConfig<F>>>> = comm
            .par_iter()
            .map(|cm| {
                LabeledCommitment::<LinCodePCCommitment<MerkleConfig<F>>>::new(
                    "".to_owned(),
                    cm.commitment.clone(),
                    None,
                )
            })
            .collect();

        let states: Vec<LinCodePCCommitmentState<F, Sha256FieldsToBytesHasher<F>>> =
            comm.par_iter().map(|cm| cm.state.clone()).collect();

        Ok(LinearCodePCS::<
            UVLigero<F, MerkleConfig<F>, Self::Polynomial, Sha256FieldsToBytesHasher<F>>,
            F,
            Self::Polynomial,
            MerkleConfig<F>,
            Sha256FieldsToBytesHasher<F>,
        >::open(
            ck, &lp, &lc, &point, &mut sponge, &states, Some(rng)
        )?)
    }

    fn check<'a>(
        vk: &'a Self::VerifierKey,
        opening_proof: &'a Self::OpeningProof,
        comm: &'a Self::Commitment,
        point: Self::PolynomialInput,
        value: Self::PolynomialOutput,
    ) -> Result<bool, anyhow::Error> {
        let mut vals = vec![];
        let mut proofs = vec![];
        let mut comms = vec![];

        vals.push(value);
        proofs.push(opening_proof.clone());
        comms.push(comm.clone());

        Ok(Self::batch_check(vk, &proofs, &comms, point, vals)?)
    }

    fn batch_check<'a>(
        vk: &'a Self::VerifierKey,
        opening_proof: &'a Vec<Self::OpeningProof>,
        comm: &'a Vec<Self::Commitment>,
        point: Self::PolynomialInput,
        value: Vec<Self::PolynomialOutput>,
    ) -> Result<bool, anyhow::Error> {
        let lc: Vec<LabeledCommitment<LinCodePCCommitment<MerkleConfig<F>>>> = comm
            .par_iter()
            .map(|cm| {
                LabeledCommitment::<LinCodePCCommitment<MerkleConfig<F>>>::new(
                    "".to_owned(),
                    cm.commitment.clone(),
                    None,
                )
            })
            .collect();

        let rng = &mut thread_rng();
        let mut sponge = Sha256Sponge::new();

        Ok(LinearCodePCS::<
            UVLigero<F, MerkleConfig<F>, Self::Polynomial, Sha256FieldsToBytesHasher<F>>,
            F,
            Self::Polynomial,
            MerkleConfig<F>,
            Sha256FieldsToBytesHasher<F>,
        >::check(
            vk,
            &lc,
            &point,
            value,
            opening_proof,
            &mut sponge,
            Some(rng),
        )?)
    }

    fn extract_pure_commitment(comm: &Self::Commitment) -> Result<Self::Commitment, anyhow::Error> {
        Ok(Self::Commitment {
            commitment: comm.commitment.clone(),
            state: LinCodePCCommitmentState::default(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct LigeroPoseidon<F: PrimeField> {
    _field_data: PhantomData<F>,
}

impl<F> PolynomialCommitmentScheme for LigeroPoseidon<F>
where
    F: PrimeField + Absorb,
{
    type VerifierKey = LigeroPCParams<F, MerkleConfig<F>, FieldsToBytesHasher<F>>;
    type CommitterKey<'a> = LigeroPCParams<F, MerkleConfig<F>, FieldsToBytesHasher<F>>;
    type Commitment = CommitmentPoseidon<F>;
    type OpeningProof = LinCodePCProof<F, MerkleConfig<F>>;
    type PCSParams = usize;
    type Polynomial = DensePolynomial<F>;
    type PolynomialInput = F;
    type PolynomialOutput = F;

    fn setup<'a>(
        _pp: &'a Self::PCSParams,
    ) -> Result<(Self::CommitterKey<'a>, Self::VerifierKey), anyhow::Error> {
        let rng = &mut thread_rng();

        let leaf_hash_param =
            <<MerkleConfig<F> as Config>::LeafHash as CRHScheme>::setup(rng).unwrap();
        let two_to_one_hash_param =
            <<MerkleConfig<F> as Config>::TwoToOneHash as TwoToOneCRHScheme>::setup(rng)
                .unwrap()
                .clone();
        let col_hash_params = <FieldsToBytesHasher<F> as CRHScheme>::setup(rng).unwrap();

        let params =
            UVLigero::<F, MerkleConfig<F>, Self::Polynomial, FieldsToBytesHasher<F>>::setup(
                0,
                Some(0),
                rng,
                leaf_hash_param,
                two_to_one_hash_param,
                col_hash_params,
            );

        Ok((params.clone(), params.clone()))
    }

    fn commit<'a>(
        ck: &'a Self::CommitterKey<'_>,
        poly: &'a Self::Polynomial,
    ) -> Result<Self::Commitment, anyhow::Error> {
        let lp = LabeledPolynomial::<F, Self::Polynomial>::new(
            "".to_owned(),
            (*poly).clone(),
            None,
            None,
        );

        let rng = &mut thread_rng();

        let (comm, comm_state) = LinearCodePCS::<
            UVLigero<F, MerkleConfig<F>, Self::Polynomial, FieldsToBytesHasher<F>>,
            F,
            Self::Polynomial,
            MerkleConfig<F>,
            FieldsToBytesHasher<F>,
        >::commit(ck, &[lp], Some(rng))
        .unwrap();

        Ok(Self::Commitment {
            commitment: comm[0].commitment().clone(),
            state: comm_state[0].clone(),
        })
    }

    fn batch_commit<'a>(
        ck: &'a Self::CommitterKey<'_>,
        poly: &'a Vec<Self::Polynomial>,
    ) -> Result<Vec<Self::Commitment>, anyhow::Error> {
        let lp: Vec<LabeledPolynomial<F, Self::Polynomial>> = poly
            .par_iter()
            .map(|p| {
                LabeledPolynomial::<F, Self::Polynomial>::new("".to_owned(), p.clone(), None, None)
            })
            .collect();

        let rng = &mut thread_rng();

        let (comm, comm_state) = LinearCodePCS::<
            UVLigero<F, MerkleConfig<F>, Self::Polynomial, FieldsToBytesHasher<F>>,
            F,
            Self::Polynomial,
            MerkleConfig<F>,
            FieldsToBytesHasher<F>,
        >::commit(ck, &lp, Some(rng))
        .unwrap();

        let result = comm
            .par_iter()
            .zip(comm_state)
            .map(|(cm, cm_state)| Self::Commitment {
                commitment: cm.commitment().clone(),
                state: cm_state,
            })
            .collect();

        Ok(result)
    }

    fn open<'a>(
        _ck: &'a Self::CommitterKey<'_>,
        _comm: &'a Self::Commitment,
        _poly: &'a Self::Polynomial,
        _point: Self::PolynomialInput,
    ) -> Result<Self::OpeningProof, anyhow::Error> {
        todo!()
    }

    fn batch_open<'a>(
        ck: &'a Self::CommitterKey<'_>,
        comm: &'a Vec<Self::Commitment>,
        poly: &'a Vec<Self::Polynomial>,
        point: Self::PolynomialInput,
    ) -> Result<Vec<Self::OpeningProof>, anyhow::Error> {
        let rng = &mut thread_rng();
        let poseidon_config: PoseidonConfig<F> = poseidon_parameters();
        let mut sponge = PoseidonSponge::new(&poseidon_config);

        let lp: Vec<LabeledPolynomial<F, Self::Polynomial>> = poly
            .par_iter()
            .map(|p| {
                LabeledPolynomial::<F, Self::Polynomial>::new("".to_owned(), p.clone(), None, None)
            })
            .collect();

        let lc: Vec<LabeledCommitment<LinCodePCCommitment<MerkleConfig<F>>>> = comm
            .par_iter()
            .map(|cm| {
                LabeledCommitment::<LinCodePCCommitment<MerkleConfig<F>>>::new(
                    "".to_owned(),
                    cm.commitment.clone(),
                    None,
                )
            })
            .collect();

        let states: Vec<LinCodePCCommitmentState<F, FieldsToBytesHasher<F>>> =
            comm.par_iter().map(|cm| cm.state.clone()).collect();

        Ok(LinearCodePCS::<
            UVLigero<F, MerkleConfig<F>, Self::Polynomial, FieldsToBytesHasher<F>>,
            F,
            Self::Polynomial,
            MerkleConfig<F>,
            FieldsToBytesHasher<F>,
        >::open(
            ck, &lp, &lc, &point, &mut sponge, &states, Some(rng)
        )?)
    }

    fn check<'a>(
        vk: &'a Self::VerifierKey,
        opening_proof: &'a Self::OpeningProof,
        comm: &'a Self::Commitment,
        point: Self::PolynomialInput,
        value: Self::PolynomialOutput,
    ) -> Result<bool, anyhow::Error> {
        let mut vals = vec![];
        let mut proofs = vec![];
        let mut comms = vec![];

        vals.push(value);
        proofs.push(opening_proof.clone());
        comms.push(comm.clone());

        Ok(Self::batch_check(vk, &proofs, &comms, point, vals)?)
    }

    fn batch_check<'a>(
        vk: &'a Self::VerifierKey,
        opening_proof: &'a Vec<Self::OpeningProof>,
        comm: &'a Vec<Self::Commitment>,
        point: Self::PolynomialInput,
        value: Vec<Self::PolynomialOutput>,
    ) -> Result<bool, anyhow::Error> {
        let lc: Vec<LabeledCommitment<LinCodePCCommitment<MerkleConfig<F>>>> = comm
            .par_iter()
            .map(|cm| {
                LabeledCommitment::<LinCodePCCommitment<MerkleConfig<F>>>::new(
                    "".to_owned(),
                    cm.commitment.clone(),
                    None,
                )
            })
            .collect();

        let rng = &mut thread_rng();
        let poseidon_config: PoseidonConfig<F> = poseidon_parameters();
        let mut sponge = PoseidonSponge::new(&poseidon_config);

        Ok(LinearCodePCS::<
            UVLigero<F, MerkleConfig<F>, Self::Polynomial, FieldsToBytesHasher<F>>,
            F,
            Self::Polynomial,
            MerkleConfig<F>,
            FieldsToBytesHasher<F>,
        >::check(
            vk,
            &lc,
            &point,
            value,
            opening_proof,
            &mut sponge,
            Some(rng),
        )?)
    }

    fn extract_pure_commitment(comm: &Self::Commitment) -> Result<Self::Commitment, anyhow::Error> {
        Ok(Self::Commitment {
            commitment: comm.commitment.clone(),
            state: LinCodePCCommitmentState::default(),
        })
    }
}