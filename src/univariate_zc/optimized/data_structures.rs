use ark_poly_commit::kzg10::{Commitment, VerifierKey, Proof as KZGProof};
use ark_ec::pairing::Pairing;

/// This is the data structure of the proof to be sent to the verifer,
/// to prove that there exists a quotient polynomial q(X), for which,
/// for all x \in Fp, f(x) = q(x).z_H(x), where, z_H(X) is the vanishing
/// polynomial over the domain H, for which prover claims that
/// for x \in H, f(x) = 0.
///  
/// q - stores the commitment to quotient polynomail as Commitment<E>
/// inp_comms - stores the commitment to the input polynomials as Vec<Commitment<E>>
/// vk - stores the verifier key required to check the KZG openings
/// q_opening - stores the opening proof for the evaluation of q(X) at challenge 'r'
/// inp_opening - stores the opening proof for the evaluation of g, h, and s at challenge 'r'
/// inp_evals -stores the evaluation of g, h, and s at challenge 'r'
/// q_eval -stores the evaluation of q(X) at challenge 'r'

#[derive(Clone)]
pub struct Proof<E: Pairing> {
    pub(crate) q_comm: Commitment<E>,
    pub(crate) inp_comms: Vec<Commitment<E>>,
    pub(crate) vk: VerifierKey<E>,
    pub(crate) q_opening: KZGProof<E>,
    pub(crate) inp_openings: Vec<KZGProof<E>>,
    pub(crate) inp_evals: Vec<E::ScalarField>,
    pub(crate) q_eval: E::ScalarField
}