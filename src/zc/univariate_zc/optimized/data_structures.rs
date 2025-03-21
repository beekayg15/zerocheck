use crate::pcs::PolynomialCommitmentScheme;

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
pub struct Proof<PCS: PolynomialCommitmentScheme> {
    pub(crate) q_comm: PCS::Commitment,
    pub(crate) inp_comms: Vec<PCS::Commitment>,
    pub(crate) q_opening: PCS::OpeningProof,
    pub(crate) inp_openings: Vec<PCS::OpeningProof>,
    pub(crate) inp_evals: Vec<PCS::PolynomialOutput>,
    pub(crate) q_eval: PCS::PolynomialOutput
}

#[derive(Clone)]
pub struct ZeroCheckParams<'a, PCS: PolynomialCommitmentScheme> {
    pub(crate) ck: PCS::CommitterKey<'a>,
    pub(crate) vk: PCS::VerifierKey,
}