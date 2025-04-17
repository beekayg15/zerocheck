use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use merlin::Transcript;
use std::marker::PhantomData;
use ark_std::string::String;
use displaydoc::Display;
use crate::to_bytes;

#[derive(Clone)]
pub struct ZCTranscript<F: PrimeField> {
    transcript: Transcript,
    is_empty: bool,
    _field_data: PhantomData<F>
}

impl<F: PrimeField> ZCTranscript<F> {
    /// Create a new Zero Check transcript.
    pub fn new(label: &'static [u8]) -> Self {
        Self {
            transcript: Transcript::new(label),
            is_empty: true,
            _field_data: PhantomData,
        }
    }

    pub fn init_transcript() -> Self {
        Self::new(b"Initializing ZeroCheck transcript")
    }

    // Append the message to the transcript.
    pub fn append_message(
        &mut self,
        label: &'static [u8],
        msg: &[u8],
    ) -> Result<(), TranscriptError> {
        const MAX_LEN: usize = u32::MAX as usize;

        let mut start = 0;
        while start < msg.len() {
            let end = (start + MAX_LEN).min(msg.len());
            self.transcript.append_message(label, &msg[start..end]);
            start = end;
        }

        self.is_empty = false;
        Ok(())
    }

    // Append the message to the transcript.
    pub fn append_field_element(
        &mut self,
        label: &'static [u8],
        field_elem: &F,
    ) -> Result<(), TranscriptError> {
        self.append_message(label, &to_bytes!(field_elem)?)
    }

    // Append the message to the transcript.
    pub fn append_serializable_element<S: CanonicalSerialize>(
        &mut self,
        label: &'static [u8],
        group_elem: &S,
    ) -> Result<(), TranscriptError> {
        self.append_message(label, &to_bytes!(group_elem)?)
    }

    // Generate the challenge from the current transcript
    // and append it to the transcript.
    //
    // The output field element is statistical uniform as long
    // as the field has a size less than 2^384.
    pub fn get_and_append_challenge(&mut self, label: &'static [u8]) -> Result<F, TranscriptError> {
        //  we need to reject when transcript is empty
        if self.is_empty {
            return Err(TranscriptError::InvalidTranscript(
                "transcript is empty".to_string(),
            ));
        }

        let mut buf = [0u8; 64];
        self.transcript.challenge_bytes(label, &mut buf);
        let challenge = F::from_le_bytes_mod_order(&buf);
        self.append_serializable_element(label, &challenge)?;
        Ok(challenge)
    }

    // Generate a list of challenges from the current transcript
    // and append them to the transcript.
    //
    // The output field element are statistical uniform as long
    // as the field has a size less than 2^384.
    pub fn get_and_append_challenge_vectors(
        &mut self,
        label: &'static [u8],
        len: usize,
    ) -> Result<Vec<F>, TranscriptError> {
        //  we need to reject when transcript is empty
        if self.is_empty {
            return Err(TranscriptError::InvalidTranscript(
                "transcript is empty".to_string(),
            ));
        }

        let mut res = vec![];
        for _ in 0..len {
            res.push(self.get_and_append_challenge(label)?)
        }
        Ok(res)
    }
}

// Errors during Transcription
#[derive(Display, Debug)]
pub enum TranscriptError {
    /// Invalid Transcript: {0}
    InvalidTranscript(String),
    /// An error during (de)serialization: {0}
    SerializationError(ark_serialize::SerializationError),
}

impl From<ark_serialize::SerializationError> for TranscriptError {
    fn from(e: ark_serialize::SerializationError) -> Self {
        Self::SerializationError(e)
    }
}

/// Takes as input a struct, and converts them to a series of bytes. All traits
/// that implement `CanonicalSerialize` can be automatically converted to bytes
/// in this manner.
#[macro_export]
macro_rules! to_bytes {
    ($x:expr) => {{
        let mut buf = ark_std::vec![];
        ark_serialize::CanonicalSerialize::serialize_compressed($x, &mut buf).map(|_| buf)
    }};
}