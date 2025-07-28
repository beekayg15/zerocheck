use ark_ec::{pairing::Pairing, PrimeGroup};
use ark_poly::{DenseUVPolynomial};
use ark_poly_commit::{kzg10::{Commitment,  Powers, Randomness}, Error, PCCommitmentState};
use ark_ff::{PrimeField, Zero};
use std::io::{BufWriter, Write};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::io::{BufReader, Read};

use tempfile::NamedTempFile;
use std::io::Seek;


fn msm_offloaded<E: Pairing>(
    bases: &[E::G1],
    scalars: &[E::ScalarField],
) -> E::G1 {
    assert_eq!(bases.len(), scalars.len(), "Mismatched input lengths");

    let mut acc = E::G1::zero();

    for (base, scalar) in bases.iter().zip(scalars) {
        acc += base.mul_bigint(scalar.into_bigint());
    }

    acc
}


/// Returns slices of aligned bases and scalars with leading zeros trimmed from scalars.
/// Ensures the lengths match and panics if not enough bases are available.
pub fn align_msm_inputs<'a, E: Pairing>(
    bases: &'a [E::G1Affine],
    scalars: &'a [E::ScalarField],
) -> (&'a [E::G1Affine], &'a [E::ScalarField]) {
    // Trim leading zeros from scalars
    let num_leading_zeros = scalars.iter().take_while(|c| c.is_zero()).count();
    let trimmed_scalars = &scalars[num_leading_zeros..];

    // Ensure bases are long enough
    let trimmed_bases = &bases[num_leading_zeros..];
    assert!(
        trimmed_bases.len() >= trimmed_scalars.len(),
        "Not enough bases for the trimmed scalars: bases={}, scalars={}",
        trimmed_bases.len(),
        trimmed_scalars.len()
    );

    // Match length exactly
    let aligned_bases = &trimmed_bases[..trimmed_scalars.len()];

    (aligned_bases, trimmed_scalars)
}


const CHUNK_SIZE: usize = 1024; // Tune based on memory and disk throughput

pub fn fast_commit_unchecked<E, P>(
    powers: &Powers<E>,
    polynomial: &P,
) -> Result<(Commitment<E>, Randomness<E::ScalarField, P>), Error>
where
    E: Pairing,
    P: DenseUVPolynomial<E::ScalarField>,
    E::G1: CanonicalSerialize + CanonicalDeserialize,
    E::ScalarField: CanonicalSerialize + CanonicalDeserialize,
{
    println!("Wassup from fast_commit_unchecked (disk offload)!");

    let coeffs = polynomial.coeffs();
    let num_leading_zeros = coeffs.iter().take_while(|c| c.is_zero()).count();
    let mut scalars = &coeffs[num_leading_zeros..];
    let mut bases = &powers.powers_of_g[num_leading_zeros..];

    (bases, scalars) = align_msm_inputs::<E>(bases, scalars);

    assert_eq!(scalars.len(), bases.len());

    // --- Step 1: Write bases and scalars to disk ---
    let mut base_file = NamedTempFile::new().unwrap();
    let mut scalar_file = NamedTempFile::new().unwrap();

    {
        let mut base_writer = BufWriter::new(&mut base_file);
        let mut scalar_writer = BufWriter::new(&mut scalar_file);

        for (b, s) in bases.iter().zip(scalars.iter()) {
            b.serialize_uncompressed(&mut base_writer).unwrap();
            s.serialize_uncompressed(&mut scalar_writer).unwrap();
        }

        base_writer.flush().unwrap();
        scalar_writer.flush().unwrap();
    }

    // Rewind file pointers
    base_file.as_file_mut().rewind().unwrap();
    scalar_file.as_file_mut().rewind().unwrap();

    let mut commitment = E::G1::zero();

    // --- Step 2: Read and multiply in chunks ---
    {
        let mut base_reader = BufReader::new(base_file);
        let mut scalar_reader = BufReader::new(scalar_file);

        let mut base_buf = vec![0u8; E::G1::default().uncompressed_size()];
        let mut scalar_buf = vec![0u8; E::ScalarField::default().uncompressed_size()];

        loop {
            let mut chunk_commit = E::G1::zero();
            let mut read_count = 0;

            for _ in 0..CHUNK_SIZE {
                if base_reader.read_exact(&mut base_buf).is_err() ||
                   scalar_reader.read_exact(&mut scalar_buf).is_err() {
                    break;
                }

                let base = E::G1::deserialize_uncompressed(&*base_buf).unwrap();
                let scalar = E::ScalarField::deserialize_uncompressed(&*scalar_buf).unwrap();
                chunk_commit += base.mul_bigint(scalar.into_bigint());

                read_count += 1;
            }

            if read_count == 0 {
                break; // EOF
            }

            commitment += chunk_commit;
        }
    }

    let randomness = Randomness::<E::ScalarField, P>::empty();
    Ok((Commitment(commitment.into()), randomness))
}
