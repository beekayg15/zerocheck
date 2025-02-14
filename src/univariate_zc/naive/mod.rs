use ark_poly::{EvaluationDomain, Evaluations, GeneralEvaluationDomain, Polynomial};
use ark_crypto_primitives::prf::blake2s::Blake2s;
use ark_crypto_primitives::prf::PRF;
use ark_std::marker::PhantomData;
use ark_ff::{
    FftField, PrimeField, BigInteger
};
use anyhow::{Error, Ok};
use ark_ff::Zero;
use crate::ZeroCheck;

mod data_structures;
pub use data_structures::*;

/// Naive Zero-Check protocol for if a polynomial
/// f = g^2*h evaluates to 0 over a specific domain
/// 
/// The inputs g and h are provided as evaluations
/// and the proof that polynomial f computed as 
/// mentioned above evaluates to zero, can be given
/// by proving the existence of a quotient polynomial
/// q, S.T. f(X) = q(X).z_H(X), where z_H(X) is the 
/// vanishing polynomial over the zero domain H.

#[derive(Clone)]
pub struct NaiveUnivariateZeroCheck<F: > {
    _field_data: PhantomData<F>,
}

/// Zero-Check protocol for univariate polynomials in which the 
/// input polynomials are provided as evalution of the circuit of
/// different inputs, and ZeroDomain is a GeneralEvaluationDomain
impl<F: PrimeField + FftField> ZeroCheck<F> for NaiveUnivariateZeroCheck<F> {
    type InputType = Evaluations<F>;
    type Proof = Proof<F>;
    type ZeroDomain = GeneralEvaluationDomain<F>;

    /// function called by the prover to genearte a valid
    /// proof for zero-check protocol
    /// 
    /// Attributes:
    /// g - input polynomial evalutions
    /// h - input polynomial evalutions
    /// zero_domain - domain over which the resulting polynomial evaluates to 0
    /// 
    /// Returns
    /// Proof - valid proof for the zero-check protocol
    fn prove<'a> (
            g: Self::InputType,
            h: Self::InputType,
            zero_domain: Self::ZeroDomain
        ) -> Result<Self::Proof, Error> {

        // compute the polynomials corresponding to g and h using interpolation (IFFT)
        let g_poly = g.interpolate();
        let h_poly = h.interpolate();

        // compute the resulting polynomial f = g^2 * h
        let f_poly = &(&g_poly * &g_poly) * &h_poly;
        
        // compute the quotient polynomial q, by dividing f by 
        // the vanishing polynomial over domain `zero_domain`
        let (q_poly, r_poly) = 
            f_poly
            .divide_by_vanishing_poly(zero_domain)
            .unwrap();

        // If f evaluates to 0 over the `zero_domain`
        // the vanishing polynomial perfect divides f with remainder r = 0
        assert!(r_poly.is_zero()); 

        // send the quotient polynomial and the resulting polynomial proof
        let proof = Proof{
            q: q_poly,
            f: f_poly
        };

        Ok(proof)
    }

    /// function called by the verifier to check if the proof for the 
    /// zero-check protocol is valid
    /// 
    /// Attributes:
    /// g - input polynomial evalutions
    /// h - input polynomial evalutions
    /// zero_domain - domain over which the resulting polynomial evaluates to 0
    /// proof - proof sent by the prover for the claim
    /// 
    /// Returns
    /// 'true' if the proof is valid, 'false' otherwise
    fn verify<'a> (
            g: Self::InputType,
            h: Self::InputType,
            proof: Self::Proof,
            zero_domain: Self::ZeroDomain
        ) -> Result<bool, anyhow::Error> {
        let q_poly = proof.q;
        let f_poly = proof.f;

        // compute the vanishing polynomial over the `zero_domain`
        let z_poly = zero_domain.vanishing_polynomial();

        // sample a random point
        let r = get_randomness(q_poly.coeffs.clone(), f_poly.coeffs.clone());
           
        // evaluate f, z_H, and q at r
        let rhs = q_poly.evaluate(&r[0]) * z_poly.evaluate(&r[0]);
        let lhs = f_poly.evaluate(&r[0]);

        // sample a random evaluation from g and h
        let rand_index = get_random_indices(
            1, 
            q_poly.coeffs.clone(), 
            f_poly.coeffs.clone(), 
            g.evals.len()
        )[0];

        let f_at_rand_index = f_poly.evaluate(&g.domain().element(rand_index));
        let g_sq_times_h = g.evals[rand_index] * g.evals[rand_index] * h.evals[rand_index];

        // check if q(r).z_H(r) == f(r) and if g[rand_index]^2 * h[rand_index] = f(rand_index)
        if lhs == rhs && g_sq_times_h == f_at_rand_index{
            return Ok(true);
        }

        return Ok(false);
    }
}

/// function to convert a vector of field elements to list of bytes
/// 
/// Attributes:
/// bytes - list of bytes(u8 values)
/// 
/// Returns
/// a vec<F> from the list of bytes(u8 values)
fn bytes_to_field_vec<F: PrimeField>(bytes: [u8; 32]) -> Vec<F> {
    
    // Compute the number of bits required to represent an element in field F
    let bits_per_elem = F::MODULUS_BIT_SIZE as usize;

    // Convert bits to bytes, rounding up
    let bytes_per_elem = (bits_per_elem + 7) / 8; 
    let mut result = vec![];

    // Iterate over chunks of the byte array and convert each chunk to a field element
    for chunk in bytes.chunks(bytes_per_elem) {
        // Convert the byte array to a BigInteger and then to a field element
        let elem = F::from_be_bytes_mod_order(chunk);
        result.push(elem);
    }

    result
}

/// function to convert a vector of field elements to list of bytes
/// 
/// Attributes:
/// field_elems - vector of field elements
/// 
/// Returns
/// a list of bytes(u8 values) from the vec<F>
fn field_vec_to_fixed_bytes<F: PrimeField>(field_elems: Vec<F>) -> [u8; 32] {
    let mut byte_vec = Vec::new();

    // Calculate minimum number of elements needed
    let bits_needed = 256; // 32 bytes * 8 bits/byte
    let elements_needed =
        (bits_needed + F::MODULUS_BIT_SIZE as usize - 1) / F::MODULUS_BIT_SIZE as usize;

    // Take only the minimum required elements
    let field_elems_truncated = field_elems.into_iter().take(elements_needed);

    // Convert each field element to bytes using into_bigint().to_le_bytes() and append it to byte_vec
    for elem in field_elems_truncated {
        let elem_bytes = elem.into_bigint().to_bytes_le();
        byte_vec.extend_from_slice(&elem_bytes);
    }

    // Truncate or pad to exactly 32 bytes
    let mut result = [0u8; 32];
    let copy_len = byte_vec.len().min(32);
    result[..copy_len].copy_from_slice(&byte_vec[..copy_len]);

    result
}

/// function to sample a random value in field F
/// 
/// Attributes:
/// seed - seed to the PRF as vec<F>
/// inp - input to the PRF as vec<F>
/// 
/// Returns
/// a random value in the Field F
fn get_randomness<F: PrimeField>(seed: Vec<F>, inp: Vec<F>) -> Vec<F> {
    let seed: [u8; 32] = field_vec_to_fixed_bytes(seed);
    let inp: [u8; 32] = field_vec_to_fixed_bytes(inp);
    bytes_to_field_vec(Blake2s::evaluate(&seed, &inp).unwrap())
}

/// function to generate random indices
/// 
/// Attributes:
/// number_indices - number of random indices to be sampled
/// seed - seed to the PRF as vec<F>
/// inp - input to the PRF as vec<F>
/// max_index - range of the indices to be generated
/// 
/// Returns,
/// `number_indices` random values sampled from [0 ... `max_index - 1`]
/// using the seed and input to the pseudo-random function
fn get_random_indices<F: PrimeField>(
    number_indices: usize, 
    seed: Vec<F>, 
    inp: Vec<F>,
    max_index: usize,
) -> Vec<usize> {
    let seed: [u8; 32] = field_vec_to_fixed_bytes(seed);
    let mut inp: [u8; 32] = field_vec_to_fixed_bytes(inp);
    let mut r:Vec<F> = bytes_to_field_vec(Blake2s::evaluate(&seed, &inp).unwrap());

    let mut indices: Vec<usize> = vec![];
    let mut _bytes= [0u8; 8];

    inp = field_vec_to_fixed_bytes(r.clone());

    for _ in 0 .. number_indices {
        r = bytes_to_field_vec(Blake2s::evaluate(&seed, &inp).unwrap());
        inp = field_vec_to_fixed_bytes(r.clone());
        _bytes.clone_from_slice(&inp[0 .. 8]);
        indices.push(usize::from_be_bytes(_bytes) % max_index);
    }

    indices
}