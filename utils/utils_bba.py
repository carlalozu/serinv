"""Functions for n-block banded arrowhead matrix (bba)

The compressed representation is a tuple of the following arrays:
    - M_diagonal_blocks: 3D array of shape (n_t, diag_blocksize, diag_blocksize)
    - M_lower_diagonal_blocks: 3D array of shape (n_t-1,
        diag_blocksize*n_offdiags_blk, diag_blocksize)
    - M_arrow_bottom_blocks: 3D array of shape (n_t, arrow_blocksize,
        diag_blocksize)
    - M_arrow_tip_block: 2D array of shape (arrow_blocksize, arrow_blocksize)

Where the following parameters are defined:
    - n_t is the number of diagonal blocks in the block banded structure,
    without the arrowhead portion.
    - n_offdiags_blk is the number of off-diagonal blocks in the block banded
    structure.
    - diag_blocksize is the size of the diagonal blocks.
    - arrow_blocksize is the size of the arrowhead blocks.

"""
import numpy as np
SEED = 63


def spd(M_, factor_=2):
    """Makes dense matrix symmetric positive definite."""

    # Make diagonally dominant
    for i in range(M_.shape[0]):
        M_[i, i] = (1 + np.sum(M_[i, :]))*factor_

    # Symmetrize
    M_ = (M_ + M_.conj().T) / 2
    return M_


def dd_bba(
    n_offdiags_blk: int,
    diag_blocksize: int,
    arrow_blocksize: int,
    n_t: int,
    dtype: np.dtype,
):
    """Returns a random, diagonally dominant general, block banded arrowhead
    matrix in compressed format."""

    xp = np
    rc = (1.0 + 1.0j) if dtype == np.complex128 else 1.0

    A_diagonal_blocks = xp.zeros(
        (n_t, diag_blocksize, diag_blocksize),
        dtype=dtype,
    )

    A_lower_diagonal_blocks = xp.zeros(
        (n_t-1, diag_blocksize*n_offdiags_blk, diag_blocksize),
        dtype=dtype,
    )

    A_arrow_bottom_blocks = xp.zeros(
        (n_t, arrow_blocksize, diag_blocksize),
        dtype=dtype,
    )

    A_arrow_tip_block = xp.zeros(
        (arrow_blocksize, arrow_blocksize),
        dtype=dtype,
    )

    A_diagonal_blocks[:, :, :] = (
        rc * xp.random.rand(*A_diagonal_blocks.shape)+1)/2
    A_lower_diagonal_blocks[:, :, :] = (
        rc * xp.random.rand(*A_lower_diagonal_blocks.shape)+1)/2
    A_arrow_bottom_blocks[:, :, :] = (
        rc * xp.random.rand(*A_arrow_bottom_blocks.shape)+1)/2
    A_arrow_tip_block[:, :] = (
        rc * xp.random.rand(*A_arrow_tip_block.shape)+1)/2

    # Make main diagonal symmetric
    for i in range(n_t):
        A_diagonal_blocks[i, :, :] = spd(
            A_diagonal_blocks[i, :, :], factor_=int(np.sqrt(n_t)))

    A_arrow_tip_block[:, :] = spd(
        A_arrow_tip_block[:, :], factor_=int(np.sqrt(n_t)))

    # Remove extra info from
    for i in range(1, n_offdiags_blk):
        A_lower_diagonal_blocks[n_t-1-i:, i*diag_blocksize:, :] = 0.0

    return (A_diagonal_blocks,
            A_lower_diagonal_blocks,
            A_arrow_bottom_blocks,
            A_arrow_tip_block)


def bba_dense_to_arrays(
    M, n_offdiags_blk, diag_blocksize, arrow_blocksize, lower=True
):
    """
    Compress a square matrix with banded and arrowhead structure 
    into a more efficient representation.
    The function handles matrices that have:
    1. Block banded diagonal structure 
    2. Arrowhead pattern in the last few rows and columns
    """

    n_t = (M.shape[0] - arrow_blocksize)//diag_blocksize

    # Initialize compressed storage arrays
    M_diagonal_blocks = np.zeros(
        (n_t, diag_blocksize, diag_blocksize), dtype=M.dtype)

    M_lower_diagonal_blocks = np.zeros(
        (n_t-1, diag_blocksize*n_offdiags_blk, diag_blocksize), dtype=M.dtype)

    M_arrow_bottom_blocks = np.zeros(
        (n_t, arrow_blocksize, diag_blocksize), dtype=M.dtype)

    M_arrow_tip_block = np.zeros(
        (arrow_blocksize, arrow_blocksize), dtype=M.dtype)

    # Extract arrowhead portion
    for i in range(n_t):
        M_arrow_bottom_blocks[i, :, :] = M[
            -arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize
        ]

    if lower:
        M_arrow_tip_block[:, :] = np.tril(
            M[-arrow_blocksize:, -arrow_blocksize:])
    else:
        M_arrow_tip_block[:, :] = M[-arrow_blocksize:, -arrow_blocksize:]

    # Compress the banded portion
    for i in range(n_t):
        # Extract main diagonal

        if lower:
            M_diagonal_blocks[i, :, :] = np.tril(
                M[i*diag_blocksize:(i+1)*diag_blocksize,
                  i*diag_blocksize:(i+1)*diag_blocksize]
            )
        else:
            M_diagonal_blocks[i, :, :] = M[i*diag_blocksize:(i+1)*diag_blocksize,
                                           i*diag_blocksize:(i+1)*diag_blocksize]

        # Extract off diagonal blocks
        for j in range(min(n_offdiags_blk, n_t - i - 1)):
            M_lower_diagonal_blocks[
                i, j*diag_blocksize:(j+1)*diag_blocksize, :
            ] = M[
                (i+j+1)*diag_blocksize:(i+j+2)*diag_blocksize,
                i*diag_blocksize:(i+1)*diag_blocksize
            ]

    return (M_diagonal_blocks, M_lower_diagonal_blocks,
            M_arrow_bottom_blocks, M_arrow_tip_block)


def bba_arrays_to_dense(
    M_diagonal_blocks,
    M_lower_diagonal_blocks,
    M_arrow_bottom_blocks,
    M_arrow_tip_block,
    symmetric=False
):
    """Decompress a square matrix with banded and arrowhead structure into 
    dense format.
    """
    n_t, diag_blocksize, _ = M_diagonal_blocks.shape
    arrow_blocksize = M_arrow_tip_block.shape[0]

    n_offdiags_blk = M_lower_diagonal_blocks.shape[1]//diag_blocksize
    N = diag_blocksize*n_t + arrow_blocksize

    # print(f"Creating new matrix of size ({N}, {N})")
    M = np.zeros((N, N), dtype=M_diagonal_blocks.dtype)

    for i in range(n_t):
        M[
            i*diag_blocksize:(i+1)*diag_blocksize,
            i*diag_blocksize:(i+1) * diag_blocksize
        ] = M_diagonal_blocks[i, :, :]

        for j in range(min(n_offdiags_blk, n_t-i-1)):
            M[
                (i+j+1)*diag_blocksize:(i+j+2)*diag_blocksize,
                i*diag_blocksize:(i+1) * diag_blocksize
            ] = M_lower_diagonal_blocks[
                i, j*diag_blocksize:(j+1)*diag_blocksize, :
            ]

        M[
            -arrow_blocksize:,
            i * diag_blocksize:(i+1)*diag_blocksize
        ] = M_arrow_bottom_blocks[i, :, :]

    M[-arrow_blocksize:, -arrow_blocksize:] = M_arrow_tip_block
    M = np.tril(M)

    if symmetric:
        return (M + M.conj().T) - np.diag(np.diag(M))
    return M
