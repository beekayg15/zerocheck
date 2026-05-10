use ark_crypto_primitives::{
    crh::{CRHScheme, TwoToOneCRHScheme},
    merkle_tree::{Config, MerkleTree, Path},
    sponge::{Absorb, CryptographicSponge},
};
use ark_ff::{Field, PrimeField};
use ark_poly::Polynomial;
use ark_poly_commit::{
    linear_codes::{LinCodeParametersInfo, LinearEncode},
    to_bytes, Error, LabeledCommitment, LabeledPolynomial, PCCommitment, PCCommitmentState,
    PCUniversalParams, PolynomialCommitment,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{
    borrow::Borrow, cfg_into_iter, cfg_iter, end_timer, fmt, marker::PhantomData, rand::RngCore,
    start_timer,
};

#[cfg(feature = "parallel")]
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

const FIELD_SIZE_ERROR: &str = "This field is not suitable for the proposed parameters";

#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
struct Matrix<F: Field> {
    n: usize,
    m: usize,
    entries: Vec<Vec<F>>,
}

impl<F: Field> Matrix<F> {
    fn new_from_flat(n: usize, m: usize, entry_list: &[F]) -> Self {
        assert_eq!(
            entry_list.len(),
            n * m,
            "Invalid matrix construction: dimensions are {} x {} but entry vector has {} entries",
            n,
            m,
            entry_list.len()
        );

        let entries = (0..n)
            .map(|row| (0..m).map(|col| entry_list[m * row + col]).collect())
            .collect();

        Self { n, m, entries }
    }

    fn new_from_rows(row_list: Vec<Vec<F>>) -> Self {
        let m = row_list[0].len();

        for row in row_list.iter().skip(1) {
            assert_eq!(
                row.len(),
                m,
                "Invalid matrix construction: not all rows have the same length"
            );
        }

        Self {
            n: row_list.len(),
            m,
            entries: row_list,
        }
    }

    fn rows(&self) -> Vec<Vec<F>> {
        self.entries.clone()
    }

    fn cols(&self) -> Vec<Vec<F>> {
        (0..self.m)
            .map(|col| (0..self.n).map(|row| self.entries[row][col]).collect())
            .collect()
    }

    fn row_mul(&self, v: &[F]) -> Vec<F> {
        assert_eq!(
            v.len(),
            self.n,
            "Invalid row multiplication: vector has {} elements whereas each matrix column has {}",
            v.len(),
            self.n
        );

        cfg_into_iter!(0..self.m)
            .map(|col| {
                inner_product(
                    v,
                    &cfg_into_iter!(0..self.n)
                        .map(|row| self.entries[row][col])
                        .collect::<Vec<F>>(),
                )
            })
            .collect()
    }
}

#[inline]
fn inner_product<F: Field>(v1: &[F], v2: &[F]) -> F {
    ark_std::cfg_iter!(v1)
        .zip(v2)
        .map(|(li, ri)| *li * ri)
        .sum()
}

#[derive(Clone, Debug, Default, CanonicalSerialize, CanonicalDeserialize)]
struct Metadata {
    n_rows: usize,
    n_cols: usize,
    n_ext_cols: usize,
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct LinCodePCCommitment<C: Config> {
    metadata: Metadata,
    root: C::InnerDigest,
}

impl<C: Config> Clone for LinCodePCCommitment<C>
where
    C::InnerDigest: Clone,
{
    fn clone(&self) -> Self {
        Self {
            metadata: self.metadata.clone(),
            root: self.root.clone(),
        }
    }
}

impl<C: Config> fmt::Debug for LinCodePCCommitment<C>
where
    C::InnerDigest: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LinCodePCCommitment")
            .field("metadata", &self.metadata)
            .field("root", &self.root)
            .finish()
    }
}

impl<C: Config> Default for LinCodePCCommitment<C>
where
    C::InnerDigest: Default,
{
    fn default() -> Self {
        Self {
            metadata: Metadata::default(),
            root: C::InnerDigest::default(),
        }
    }
}

impl<C: Config> PCCommitment for LinCodePCCommitment<C>
where
    C::InnerDigest: Clone + Default,
{
    fn empty() -> Self {
        Self::default()
    }

    fn has_degree_bound(&self) -> bool {
        false
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct LinCodePCCommitmentState<F, H>
where
    F: PrimeField,
    H: CRHScheme,
{
    mat: Matrix<F>,
    ext_mat: Matrix<F>,
    leaves: Vec<H::Output>,
}

impl<F, H> Clone for LinCodePCCommitmentState<F, H>
where
    F: PrimeField,
    H: CRHScheme,
    H::Output: Clone,
{
    fn clone(&self) -> Self {
        Self {
            mat: self.mat.clone(),
            ext_mat: self.ext_mat.clone(),
            leaves: self.leaves.clone(),
        }
    }
}

impl<F, H> fmt::Debug for LinCodePCCommitmentState<F, H>
where
    F: PrimeField + fmt::Debug,
    H: CRHScheme,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LinCodePCCommitmentState")
            .field("mat", &self.mat)
            .field("ext_mat", &self.ext_mat)
            .field("leaves_len", &self.leaves.len())
            .finish()
    }
}

impl<F, H> Default for LinCodePCCommitmentState<F, H>
where
    F: PrimeField,
    H: CRHScheme,
{
    fn default() -> Self {
        Self {
            mat: Matrix::default(),
            ext_mat: Matrix::default(),
            leaves: Vec::new(),
        }
    }
}

impl<F, H> PCCommitmentState for LinCodePCCommitmentState<F, H>
where
    F: PrimeField,
    H: CRHScheme,
    H::Output: Clone,
{
    type Randomness = ();

    fn empty() -> Self {
        Self::default()
    }

    fn rand<R: RngCore>(
        _num_queries: usize,
        _has_degree_bound: bool,
        _num_vars: Option<usize>,
        _rng: &mut R,
    ) -> Self::Randomness {
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
struct LinCodePCProofSingle<F, C>
where
    F: PrimeField,
    C: Config,
{
    paths: Vec<Path<C>>,
    v: Vec<F>,
    columns: Vec<Vec<F>>,
}

impl<F, C> Clone for LinCodePCProofSingle<F, C>
where
    F: PrimeField,
    C: Config,
    Path<C>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            paths: self.paths.clone(),
            v: self.v.clone(),
            columns: self.columns.clone(),
        }
    }
}

impl<F, C> fmt::Debug for LinCodePCProofSingle<F, C>
where
    F: PrimeField + fmt::Debug,
    C: Config,
    Path<C>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LinCodePCProofSingle")
            .field("paths", &self.paths)
            .field("v", &self.v)
            .field("columns", &self.columns)
            .finish()
    }
}

impl<F, C> Default for LinCodePCProofSingle<F, C>
where
    F: PrimeField,
    C: Config,
{
    fn default() -> Self {
        Self {
            paths: Vec::new(),
            v: Vec::new(),
            columns: Vec::new(),
        }
    }
}

#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct LinCodePCProof<F, C>
where
    F: PrimeField,
    C: Config,
{
    opening: LinCodePCProofSingle<F, C>,
    well_formedness: Option<Vec<F>>,
}

impl<F, C> Clone for LinCodePCProof<F, C>
where
    F: PrimeField,
    C: Config,
    LinCodePCProofSingle<F, C>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            opening: self.opening.clone(),
            well_formedness: self.well_formedness.clone(),
        }
    }
}

impl<F, C> fmt::Debug for LinCodePCProof<F, C>
where
    F: PrimeField + fmt::Debug,
    C: Config,
    LinCodePCProofSingle<F, C>: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LinCodePCProof")
            .field("opening", &self.opening)
            .field("well_formedness", &self.well_formedness)
            .finish()
    }
}

impl<F, C> Default for LinCodePCProof<F, C>
where
    F: PrimeField,
    C: Config,
{
    fn default() -> Self {
        Self {
            opening: LinCodePCProofSingle::default(),
            well_formedness: None,
        }
    }
}

pub struct LocalLinearCodePCS<L, F, P, C, H>
where
    F: PrimeField,
    C: Config,
    P: Polynomial<F>,
    H: CRHScheme,
    L: LinearEncode<F, C, P, H>,
{
    _phantom: PhantomData<(L, F, P, C, H)>,
}

impl<L, F, P, C, H> LocalLinearCodePCS<L, F, P, C, H>
where
    F: PrimeField,
    C: Config,
    P: Polynomial<F>,
    H: CRHScheme,
    L: LinearEncode<F, C, P, H>,
{
    fn compute_matrices(polynomial: &P, param: &L::LinCodePCParams) -> (Matrix<F>, Matrix<F>) {
        let mut coeffs = L::poly_to_vec(polynomial);
        let (n_rows, n_cols) = param.compute_dimensions(coeffs.len());
        coeffs.resize(n_rows * n_cols, F::zero());

        let mat = Matrix::new_from_flat(n_rows, n_cols, &coeffs);
        let rows = mat.rows();
        let ext_mat = Matrix::new_from_rows(
            cfg_iter!(rows)
                .map(|r| L::encode(r, param).unwrap())
                .collect(),
        );

        (mat, ext_mat)
    }
}

impl<L, F, P, C, H> PolynomialCommitment<F, P> for LocalLinearCodePCS<L, F, P, C, H>
where
    L: LinearEncode<F, C, P, H>,
    F: PrimeField + Absorb,
    P: Polynomial<F>,
    C: Config + 'static,
    Vec<F>: Borrow<<H as CRHScheme>::Input>,
    H::Output: Into<C::Leaf> + Send,
    H::Output: Clone,
    C::InnerDigest: Clone + Default,
    C::Leaf: Sized + Clone + Default + Send + AsRef<C::Leaf>,
    H: CRHScheme + 'static,
{
    type UniversalParams = L::LinCodePCParams;
    type CommitterKey = L::LinCodePCParams;
    type VerifierKey = L::LinCodePCParams;
    type Commitment = LinCodePCCommitment<C>;
    type CommitmentState = LinCodePCCommitmentState<F, H>;
    type Proof = Vec<LinCodePCProof<F, C>>;
    type BatchProof = Vec<Self::Proof>;
    type Error = Error;

    fn setup<R: RngCore>(
        max_degree: usize,
        num_vars: Option<usize>,
        rng: &mut R,
    ) -> Result<Self::UniversalParams, Self::Error> {
        let leaf_hash_param = <C::LeafHash as CRHScheme>::setup(rng).unwrap();
        let two_to_one_hash_param = <C::TwoToOneHash as TwoToOneCRHScheme>::setup(rng)
            .unwrap()
            .clone();
        let col_hash_params = <H as CRHScheme>::setup(rng).unwrap();
        let pp = L::setup::<R>(
            max_degree,
            num_vars,
            rng,
            leaf_hash_param,
            two_to_one_hash_param,
            col_hash_params,
        );
        let real_max_degree = <Self::UniversalParams as PCUniversalParams>::max_degree(&pp);
        if max_degree > real_max_degree || real_max_degree == 0 {
            return Err(Error::InvalidParameters(FIELD_SIZE_ERROR.to_string()));
        }
        Ok(pp)
    }

    fn trim(
        pp: &Self::UniversalParams,
        _supported_degree: usize,
        _supported_hiding_bound: usize,
        _enforced_degree_bounds: Option<&[usize]>,
    ) -> Result<(Self::CommitterKey, Self::VerifierKey), Self::Error> {
        if <Self::UniversalParams as PCUniversalParams>::max_degree(pp) == 0 {
            return Err(Error::InvalidParameters(FIELD_SIZE_ERROR.to_string()));
        }
        Ok((pp.clone(), pp.clone()))
    }

    fn commit<'a>(
        ck: &Self::CommitterKey,
        polynomials: impl IntoIterator<Item = &'a LabeledPolynomial<F, P>>,
        _rng: Option<&mut dyn RngCore>,
    ) -> Result<
        (
            Vec<LabeledCommitment<Self::Commitment>>,
            Vec<Self::CommitmentState>,
        ),
        Self::Error,
    >
    where
        P: 'a,
    {
        let commit_total_timer = start_timer!(|| "LocalLinearCodePCS::commit total");
        let mut commitments = Vec::new();
        let mut states = Vec::new();

        for labeled_polynomial in polynomials {
            let polynomial = labeled_polynomial.polynomial();

            // Step 1: arrange the polynomial into a matrix and encode each row.
            let compute_matrices_timer =
                start_timer!(|| "LocalLinearCodePCS::commit step 1: compute matrices");
            let (mat, ext_mat) = Self::compute_matrices(polynomial, ck);
            let n_rows = mat.n;
            let n_cols = mat.m;
            let n_ext_cols = ext_mat.m;
            end_timer!(compute_matrices_timer);

            // Step 2: hash each encoded matrix column into a Merkle leaf.
            let hash_columns_timer =
                start_timer!(|| "LocalLinearCodePCS::commit step 2: hash encoded columns");
            let ext_mat_cols = ext_mat.cols();
            let leaves: Vec<H::Output> = cfg_into_iter!(ext_mat_cols)
                .map(|col| {
                    H::evaluate(ck.col_hash_params(), col)
                        .map_err(|_| Error::HashingError)
                        .unwrap()
                })
                .collect();
            end_timer!(hash_columns_timer);

            // Step 3: build the Merkle tree and store the commitment/state.
            let merkle_tree_timer = start_timer!(
                || "LocalLinearCodePCS::commit step 3: build Merkle tree and store commitment"
            );
            let state = Self::CommitmentState {
                mat,
                ext_mat,
                leaves,
            };

            let mut leaves: Vec<C::Leaf> =
                state.leaves.clone().into_iter().map(|h| h.into()).collect();

            let col_tree = create_merkle_tree::<C>(
                &mut leaves,
                ck.leaf_hash_param(),
                ck.two_to_one_hash_param(),
            )?;

            let commitment = LinCodePCCommitment {
                metadata: Metadata {
                    n_rows,
                    n_cols,
                    n_ext_cols,
                },
                root: col_tree.root(),
            };

            commitments.push(LabeledCommitment::new(
                labeled_polynomial.label().clone(),
                commitment,
                None,
            ));
            states.push(state);
            end_timer!(merkle_tree_timer);
        }

        end_timer!(commit_total_timer);
        Ok((commitments, states))
    }

    fn open<'a>(
        ck: &Self::CommitterKey,
        _labeled_polynomials: impl IntoIterator<Item = &'a LabeledPolynomial<F, P>>,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        point: &'a P::Point,
        sponge: &mut impl CryptographicSponge,
        states: impl IntoIterator<Item = &'a Self::CommitmentState>,
        _rng: Option<&mut dyn RngCore>,
    ) -> Result<Self::Proof, Self::Error>
    where
        P: 'a,
        Self::CommitmentState: 'a,
        Self::Commitment: 'a,
    {
        let open_total_timer = start_timer!(|| "LocalLinearCodePCS::open total");
        let mut proof_array = Self::Proof::default();

        for (labeled_commitment, state) in commitments.into_iter().zip(states) {
            let commitment = labeled_commitment.commitment();
            let n_rows = commitment.metadata.n_rows;
            let n_cols = commitment.metadata.n_cols;

            let rebuild_tree_timer =
                start_timer!(|| "LocalLinearCodePCS::open step 1: rebuild Merkle tree");
            let Self::CommitmentState {
                mat,
                ext_mat,
                leaves: col_hashes,
            } = state;
            let mut col_hashes: Vec<C::Leaf> =
                col_hashes.clone().into_iter().map(|h| h.into()).collect();

            let col_tree = create_merkle_tree::<C>(
                &mut col_hashes,
                ck.leaf_hash_param(),
                ck.two_to_one_hash_param(),
            )?;
            end_timer!(rebuild_tree_timer);

            let well_formedness_timer = start_timer!(
                || "LocalLinearCodePCS::open step 2: prepare transcript and well-formedness proof"
            );
            let (_, b) = L::tensor(point, n_cols, n_rows);

            sponge.absorb(&to_bytes!(&commitment.root).map_err(|_| Error::TranscriptError)?);

            let well_formedness = if ck.check_well_formedness() {
                let r = sponge.squeeze_field_elements::<F>(n_rows);
                let v = mat.row_mul(&r);

                sponge.absorb(&v);
                Some(v)
            } else {
                None
            };
            end_timer!(well_formedness_timer);

            let point_vec = L::point_to_vec(point.clone());
            sponge.absorb(&point_vec);
            end_timer!(well_formedness_timer);

            let generate_proof_timer =
                start_timer!(|| "LocalLinearCodePCS::open step 3: generate opening proof");
            proof_array.push(LinCodePCProof {
                opening: generate_proof(
                    ck.sec_param(),
                    ck.distance(),
                    &b,
                    mat,
                    ext_mat,
                    &col_tree,
                    sponge,
                )?,
                well_formedness,
            });
            end_timer!(generate_proof_timer);
        }

        end_timer!(open_total_timer);
        Ok(proof_array)
    }

    fn check<'a>(
        vk: &Self::VerifierKey,
        commitments: impl IntoIterator<Item = &'a LabeledCommitment<Self::Commitment>>,
        point: &'a P::Point,
        values: impl IntoIterator<Item = F>,
        proof_array: &Self::Proof,
        sponge: &mut impl CryptographicSponge,
        _rng: Option<&mut dyn RngCore>,
    ) -> Result<bool, Self::Error>
    where
        Self::Commitment: 'a,
    {
        let leaf_hash_param = vk.leaf_hash_param();
        let two_to_one_hash_param = vk.two_to_one_hash_param();

        for (i, (labeled_commitment, value)) in commitments.into_iter().zip(values).enumerate() {
            let proof = &proof_array[i];
            let commitment = labeled_commitment.commitment();
            let n_rows = commitment.metadata.n_rows;
            let n_cols = commitment.metadata.n_cols;
            let n_ext_cols = commitment.metadata.n_ext_cols;
            let root = &commitment.root;
            let t = calculate_t::<F>(vk.sec_param(), vk.distance(), n_ext_cols)?;

            sponge.absorb(&to_bytes!(&commitment.root).map_err(|_| Error::TranscriptError)?);

            let out = if vk.check_well_formedness() {
                if proof.well_formedness.is_none() {
                    return Err(Error::InvalidCommitment);
                }
                let v = proof.well_formedness.as_ref().unwrap();
                let r = sponge.squeeze_field_elements::<F>(n_rows);
                sponge.absorb(v);

                (Some(v), Some(r))
            } else {
                (None, None)
            };

            let point_vec = L::point_to_vec(point.clone());
            sponge.absorb(&point_vec);
            sponge.absorb(&proof.opening.v);

            let indices = get_indices_from_sponge(n_ext_cols, t, sponge)?;

            let col_hashes: Vec<C::Leaf> = proof
                .opening
                .columns
                .iter()
                .map(|c| {
                    H::evaluate(vk.col_hash_params(), c.clone())
                        .map_err(|_| Error::HashingError)
                        .unwrap()
                        .into()
                })
                .collect();

            for (j, (leaf, q_j)) in col_hashes.iter().zip(indices.iter()).enumerate() {
                let path = &proof.opening.paths[j];
                if path.leaf_index != *q_j {
                    return Err(Error::InvalidCommitment);
                }

                path.verify(leaf_hash_param, two_to_one_hash_param, root, leaf.clone())
                    .map_err(|_| Error::InvalidCommitment)?;
            }

            let check_inner_product = |a, b, c| -> Result<(), Error> {
                if inner_product(a, b) != c {
                    return Err(Error::InvalidCommitment);
                }

                Ok(())
            };

            let w = L::encode(&proof.opening.v, vk)?;
            let (a, b) = L::tensor(point, n_cols, n_rows);

            if let (Some(well_formedness), Some(r)) = out {
                let w_well_formedness = L::encode(well_formedness, vk)?;
                for (transcript_index, matrix_index) in indices.iter().enumerate() {
                    check_inner_product(
                        &r,
                        &proof.opening.columns[transcript_index],
                        w_well_formedness[*matrix_index],
                    )?;
                    check_inner_product(
                        &b,
                        &proof.opening.columns[transcript_index],
                        w[*matrix_index],
                    )?;
                }
            } else {
                for (transcript_index, matrix_index) in indices.iter().enumerate() {
                    check_inner_product(
                        &b,
                        &proof.opening.columns[transcript_index],
                        w[*matrix_index],
                    )?;
                }
            }

            if inner_product(&proof.opening.v, &a) != value {
                eprintln!("Function check: claimed value in position {i} does not match the evaluation of the committed polynomial in the same position");
                return Ok(false);
            }
        }

        Ok(true)
    }
}

fn create_merkle_tree<C>(
    leaves: &mut Vec<C::Leaf>,
    leaf_hash_param: &<<C as Config>::LeafHash as CRHScheme>::Parameters,
    two_to_one_hash_param: &<<C as Config>::TwoToOneHash as TwoToOneCRHScheme>::Parameters,
) -> Result<MerkleTree<C>, Error>
where
    C: Config,
    C::Leaf: Default + Clone + Send + AsRef<C::Leaf>,
{
    let next_pow_of_two = leaves.len().next_power_of_two();
    leaves.resize(next_pow_of_two, <C::Leaf>::default());

    MerkleTree::<C>::new(leaf_hash_param, two_to_one_hash_param, leaves)
        .map_err(|_| Error::HashingError)
}

fn generate_proof<F, C, S>(
    sec_param: usize,
    distance: (usize, usize),
    b: &[F],
    mat: &Matrix<F>,
    ext_mat: &Matrix<F>,
    col_tree: &MerkleTree<C>,
    sponge: &mut S,
) -> Result<LinCodePCProofSingle<F, C>, Error>
where
    F: PrimeField + Absorb,
    C: Config,
    S: CryptographicSponge,
{
    let generate_proof_total_timer =
        start_timer!(|| "LocalLinearCodePCS::generate_proof total");

    let row_mul_timer = start_timer!(
        || "LocalLinearCodePCS::generate_proof step 1: compute query count and row product"
    );
    let t = calculate_t::<F>(sec_param, distance, ext_mat.m)?;
    let v = mat.row_mul(b);
    sponge.absorb(&v);
    end_timer!(row_mul_timer);

    let sample_indices_timer =
        start_timer!(|| "LocalLinearCodePCS::generate_proof step 2: sample query indices");
    let indices = get_indices_from_sponge(ext_mat.m, t, sponge)?;
    end_timer!(sample_indices_timer);

    let merkle_paths_timer =
        start_timer!(|| "LocalLinearCodePCS::generate_proof step 3: collect columns and paths");
    let mut queried_columns = Vec::with_capacity(t);
    let mut paths = Vec::with_capacity(t);
    let ext_mat_cols = ext_mat.cols();

    for i in indices {
        queried_columns.push(ext_mat_cols[i].clone());
        paths.push(
            col_tree
                .generate_proof(i)
                .map_err(|_| Error::TranscriptError)?,
        );
    }

    let proof = LinCodePCProofSingle {
        paths,
        v,
        columns: queried_columns,
    };
    end_timer!(merkle_paths_timer);
    end_timer!(generate_proof_total_timer);

    Ok(proof)
}

fn get_indices_from_sponge<S: CryptographicSponge>(
    n: usize,
    t: usize,
    sponge: &mut S,
) -> Result<Vec<usize>, Error> {
    let bytes_to_squeeze = get_num_bytes(n);
    let mut indices = Vec::with_capacity(t);
    for _ in 0..t {
        let bytes = sponge.squeeze_bytes(bytes_to_squeeze);
        sponge.absorb(&bytes);

        let ind = bytes.iter().fold(0, |acc, &x| (acc << 8) + x as usize);
        indices.push(ind % n);
    }
    Ok(indices)
}

#[inline]
fn get_num_bytes(n: usize) -> usize {
    ceil_div((usize::BITS - n.leading_zeros()) as usize, 8)
}

#[inline]
fn ceil_div(x: usize, y: usize) -> usize {
    (x + y - 1) / y
}

#[inline]
fn calculate_t<F: PrimeField>(
    sec_param: usize,
    distance: (usize, usize),
    codeword_len: usize,
) -> Result<usize, Error> {
    let field_bits = F::MODULUS_BIT_SIZE as i32;
    let sec_param = sec_param as i32;

    let residual = codeword_len as f64 / 2.0_f64.powi(field_bits);
    let rhs = (2.0_f64.powi(-sec_param) - residual).log2();
    if !(rhs.is_normal()) {
        return Err(Error::InvalidParameters("For the given codeword length and the required security guarantee, the field is not big enough.".to_string()));
    }
    let nom = rhs - 1.0;
    let denom = (1.0 - 0.5 * distance.0 as f64 / distance.1 as f64).log2();
    if !(denom.is_normal()) {
        return Err(Error::InvalidParameters("The distance is wrong".to_string()));
    }
    let t = (nom / denom).ceil() as usize;
    Ok(if t < codeword_len { t } else { codeword_len })
}
