use ark_ec::{pairing::Pairing, VariableBaseMSM};
use ark_poly::{DenseUVPolynomial};
use ark_poly_commit::{kzg10::{Commitment,  Powers, Randomness}, Error, PCCommitmentState};
use ark_ff::Zero;

pub fn fast_commit_unchecked<E, P>(
    powers: &Powers<E>,
    polynomial: &P,
) -> Result<(Commitment<E>, Randomness<E::ScalarField, P>), Error>
where
    E: Pairing,
    P: DenseUVPolynomial<E::ScalarField>,
{
    // Commit to the polynomial using raw field coeffs (no BigInt conversion)
    let coeffs = polynomial.coeffs();
    let num_leading_zeros = coeffs.iter().take_while(|c| c.is_zero()).count();
    let plain_coeffs = &coeffs[num_leading_zeros..];

    let commitment = E::G1::msm_unchecked(
        &powers.powers_of_g[num_leading_zeros..],
        plain_coeffs,
    );

    let randomness = Randomness::<E::ScalarField, P>::empty();

    // If hiding is requested
    // let random_commitment = if let Some(hiding_degree) = hiding_bound {
    //     let mut rng = rng.ok_or(Error::MissingRng)?;
    //     randomness = Randomness::rand(hiding_degree, false, None, &mut rng);
    //     ark_poly_commit::kzg10::KZG10::<E, P>::check_hiding_bound(
    //         randomness.blinding_polynomial.degree(),
    //         powers.powers_of_gamma_g.len(),
    //     )?;

    //     let blind_coeffs = randomness.blinding_polynomial.coeffs();
    //     E::G1::msm_unchecked(&powers.powers_of_gamma_g, blind_coeffs).into_affine()
    // } else {
    //     E::G1::zero().into_affine()
    // };

    // let final_commitment = commitment + &random_commitment;
    let final_commitment = commitment;
    Ok((Commitment(final_commitment.into()), randomness))
}
