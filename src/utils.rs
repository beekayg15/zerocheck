use ark_crypto_primitives::prf::Blake2s;
use ark_crypto_primitives::prf::PRF;
use ark_ff::PrimeField;
use ark_ff::BigInteger;

/// function to convert a vector of field elements to list of bytes
/// 
/// Attributes:
/// bytes - list of bytes(u8 values)
/// 
/// Returns
/// a vec<F> from the list of bytes(u8 values)
pub fn bytes_to_field_vec<F: PrimeField>(bytes: [u8; 32]) -> Vec<F> {
    
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
pub fn field_vec_to_fixed_bytes<F: PrimeField>(field_elems: Vec<F>) -> [u8; 32] {
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
pub fn get_randomness<F: PrimeField>(seed: Vec<F>, inp: Vec<F>) -> Vec<F> {
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
pub fn get_random_indices<F: PrimeField>(
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
