use std::marker::PhantomData;
use std::borrow::Borrow;

use ark_crypto_primitives::{
    crh::{CRHScheme, TwoToOneCRHScheme},
    sponge::{Absorb, CryptographicSponge, FieldElementSize},
};
use ark_ff::{PrimeField, BigInteger};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone)]
pub struct Sha256Sponge {
    buffer: Vec<u8>,
    squeeze_counter: u64,
}

impl Sha256Sponge {
    pub fn new() -> Self {
        Self {
            buffer: Vec::new(),
            squeeze_counter: 0,
        }
    }
}

impl CryptographicSponge for Sha256Sponge {
    type Config = ();

    fn new(_config: &Self::Config) -> Self {
        Sha256Sponge::new()
    }

    fn absorb(&mut self, input: &impl Absorb) {
        self.buffer
            .extend_from_slice(&Absorb::to_sponge_bytes_as_vec(input));
    }

    fn squeeze_bytes(&mut self, num_bytes: usize) -> Vec<u8> {
        let mut output = Vec::new();
        while output.len() < num_bytes {
            let mut hasher = Sha256::new();
            hasher.update(&self.buffer);
            hasher.update(self.squeeze_counter.to_le_bytes());
            output.extend_from_slice(&hasher.finalize());
            self.squeeze_counter += 1;
        }
        output.truncate(num_bytes);
        output
    }

    fn squeeze_field_elements<F: PrimeField>(&mut self, num_elements: usize) -> Vec<F> {
        let bytes = self.squeeze_bytes(num_elements * 32);
        bytes
            .chunks(32)
            .map(|chunk| F::from_be_bytes_mod_order(chunk))
            .collect()
    }

    fn squeeze_bits(&mut self, num_bits: usize) -> Vec<bool> {
        let num_bytes = (num_bits + 7) / 8;
        let bytes = self.squeeze_bytes(num_bytes);
        let mut bits = Vec::with_capacity(num_bits);
        for byte in bytes {
            for i in (0..8).rev() {
                if bits.len() == num_bits {
                    break;
                }
                bits.push((byte >> i) & 1 == 1);
            }
            if bits.len() == num_bits {
                break;
            }
        }
        bits
    }

    fn squeeze_field_elements_with_sizes<F: PrimeField>(
        &mut self,
        sizes: &[FieldElementSize],
    ) -> Vec<F> {
        if sizes.is_empty() {
            return Vec::new();
        }

        let mut total_bits = 0usize;
        for size in sizes {
            let num_bits = match size {
                FieldElementSize::Truncated(bits) => *bits,
                FieldElementSize::Full => F::MODULUS_BIT_SIZE as usize,
            };
            total_bits += num_bits;
        }

        let bits = self.squeeze_bits(total_bits);
        let mut bits_window = bits.as_slice();

        let mut output = Vec::with_capacity(sizes.len());
        for size in sizes {
            let num_bits = match size {
                FieldElementSize::Truncated(bits) => *bits,
                FieldElementSize::Full => F::MODULUS_BIT_SIZE as usize,
            };
            let emulated_bits_le: Vec<bool> = bits_window[..num_bits].to_vec();
            bits_window = &bits_window[num_bits..];

            let emulated_bytes = emulated_bits_le
                .chunks(8)
                .map(|bits| {
                    let mut byte = 0u8;
                    for (i, &bit) in bits.iter().enumerate() {
                        if bit {
                            byte |= 1 << i;
                        }
                    }
                    byte
                })
                .collect::<Vec<_>>();

            output.push(F::from_le_bytes_mod_order(&emulated_bytes));
        }

        output
    }

    fn fork(&self, domain: &[u8]) -> Self {
        let mut new_sponge = self.clone();

        let mut input = Absorb::to_sponge_bytes_as_vec(&domain.len());
        input.extend_from_slice(domain);
        new_sponge.absorb(&input);

        new_sponge
    }
}

pub struct Sha256Hasher<F: PrimeField> {
    _field_data: PhantomData<F>,
}

impl<F: PrimeField + Absorb> CRHScheme for Sha256Hasher<F> {
    type Input = Vec<u8>;
    type Output = Vec<u8>;

    type Parameters = ();

    fn setup<R: ark_std::rand::Rng>(
        _r: &mut R,
    ) -> Result<Self::Parameters, ark_crypto_primitives::Error> {
        Ok(())
    }

    fn evaluate<T: std::borrow::Borrow<Self::Input>>(
        _parameters: &Self::Parameters,
        input: T,
    ) -> Result<Self::Output, ark_crypto_primitives::Error> {
        let mut hasher = Sha256::new();
        hasher.update(input.borrow());
        let res = hasher.finalize();
        Ok(res.to_vec())
    }
}

pub struct Sha256TwoToOneHasher<F: PrimeField> {
    _field_data: PhantomData<F>,
}

impl<F: PrimeField + Absorb> TwoToOneCRHScheme for Sha256TwoToOneHasher<F> {
    type Input = Vec<u8>;
    type Output = Vec<u8>;

    type Parameters = ();

    fn setup<R: ark_std::rand::Rng>(
        _r: &mut R,
    ) -> Result<Self::Parameters, ark_crypto_primitives::Error> {
        Ok(())
    }

    fn evaluate<T: Borrow<Self::Input>>(
        parameters: &Self::Parameters,
        left_input: T,
        right_input: T,
    ) -> Result<Self::Output, ark_crypto_primitives::Error> {
        Self::compress(parameters, left_input, right_input)
    }

    fn compress<T: std::borrow::Borrow<Self::Output>>(
        _parameters: &Self::Parameters,
        left_input: T,
        right_input: T,
    ) -> Result<Self::Output, ark_crypto_primitives::Error> {
        let mut hasher = Sha256::new();
        hasher.update(left_input.borrow());
        hasher.update(right_input.borrow());
        let res = hasher.finalize();
        Ok(res.to_vec())
    }
}

pub struct Sha256FieldsToBytesHasher<F: PrimeField> {
    _field_data: PhantomData<F>,
}

impl<F: PrimeField + Absorb> CRHScheme for Sha256FieldsToBytesHasher<F> {
    type Input = Vec<F>;
    type Output = Vec<u8>;
    type Parameters = ();

    fn setup<R: ark_std::rand::Rng>(
        _r: &mut R,
    ) -> Result<Self::Parameters, ark_crypto_primitives::Error> {
        Ok(())
    }

    fn evaluate<T: Borrow<Self::Input>>(
        _parameters: &Self::Parameters,
        input: T,
    ) -> Result<Self::Output, ark_crypto_primitives::Error> {
        let mut hasher = Sha256::new();
        for fe in input.borrow() {
            let bytes = fe.into_bigint().to_bytes_be();
            hasher.update(&bytes);
        }
        let hash = hasher.finalize();
        Ok(hash.to_vec())
    }
}
