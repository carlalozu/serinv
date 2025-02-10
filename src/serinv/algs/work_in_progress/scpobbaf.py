# Copyright 2023-2024 ETH Zurich. All rights reserved.

from typing import Tuple
import numpy as np
import scipy.linalg as np_la
from numpy.typing import ArrayLike

try:
    import cupy as cp
    import cupyx.scipy.linalg as cu_la
    from serinv.cupyfix.cholesky_lowerfill import cholesky_lowerfill

    CUPY_AVAIL = True

except ImportError:
    CUPY_AVAIL = False

if CUPY_AVAIL:
    xp = cp
    la = cu_la
    cholesky = cholesky_lowerfill
else:
    xp = np
    la = np_la
    cholesky = np.linalg.cholesky


def scpobbaf(
    A: ArrayLike,
    ndiags: int,
    diag_blocksize: int,
    arrow_blocksize: int,
    overwrite: bool = False,
) -> ArrayLike:
    """Perform the cholesky factorization of a block n-diagonals arrowhead
    matrix. The matrix is assumed to be symmetric positive definite.

    Parameters
    ----------
    A : ArrayLike
        Input matrix to decompose.
    ndiags : int
        Number of diagonals of the matrix.
    diag_blocksize : int
        Blocksize of the diagonals blocks of the matrix.
    arrow_blocksize : int
        Blocksize of the blocks composing the arrowhead.
    overwrite : bool
        If True, the input matrix A is modified in place. Default is False.

    Returns
    -------
    L : ArrayLike
        The cholesky factorization of the matrix.
    """

    if overwrite:
        L = A
    else:
        L = np.copy(A)

    L_inv_temp = np.zeros((diag_blocksize, diag_blocksize))

    n_offdiags_blk = ndiags

    n_diag_blocks = (A.shape[0] - arrow_blocksize) // diag_blocksize
    for i in range(0, n_diag_blocks - 1):
        # L_{i, i} = chol(A_{i, i})
        L[
            i * diag_blocksize: (i + 1) * diag_blocksize,
            i * diag_blocksize: (i + 1) * diag_blocksize,
        ] = la.cholesky(
            L[
                i * diag_blocksize: (i + 1) * diag_blocksize,
                i * diag_blocksize: (i + 1) * diag_blocksize,
            ]
        ).T

        # Temporary storage of re-used triangular solving
        L_inv_temp = la.solve_triangular(
            L[
                i * diag_blocksize: (i + 1) * diag_blocksize,
                i * diag_blocksize: (i + 1) * diag_blocksize,
            ],
            np.eye(diag_blocksize),
            lower=True,
        ).T

        for j in range(1, min(n_offdiags_blk + 1, n_diag_blocks - i)):
            # L_{i+j, i} = A_{i+j, i} @ L_{i, i}^{-T}
            L[
                (i + j) * diag_blocksize: (i + j + 1) * diag_blocksize,
                i * diag_blocksize: (i + 1) * diag_blocksize,
            ] = (
                L[
                    (i + j) * diag_blocksize: (i + j + 1) * diag_blocksize,
                    i * diag_blocksize: (i + 1) * diag_blocksize,
                ]
                @ L_inv_temp
            )
            for k in range(1, j + 1):
                # A_{i+j, i+k} = A_{i+j, i+k} - L_{i+j, i} @ L_{i+k, i}^{T}
                L[
                    (i + j) * diag_blocksize: (i + j + 1) * diag_blocksize,
                    (i + k) * diag_blocksize: (i + k + 1) * diag_blocksize,
                ] = (
                    L[
                        (i + j) * diag_blocksize: (i + j + 1) * diag_blocksize,
                        (i + k) * diag_blocksize: (i + k + 1) * diag_blocksize,
                    ]
                    - L[
                        (i + j) * diag_blocksize: (i + j + 1) * diag_blocksize,
                        i * diag_blocksize: (i + 1) * diag_blocksize,
                    ]
                    @ L[
                        (i + k) * diag_blocksize: (i + k + 1) * diag_blocksize,
                        i * diag_blocksize: (i + 1) * diag_blocksize,
                    ].T
                )

        # Part of the decomposition for the arrowhead structure
        # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
        L[-arrow_blocksize:, i * diag_blocksize: (i + 1) * diag_blocksize] = (
            L[-arrow_blocksize:, i * diag_blocksize: (i + 1) * diag_blocksize]
            @ L_inv_temp
        )

        for k in range(1, min(n_offdiags_blk + 1, n_diag_blocks - i)):
            # A_{ndb+1, i+k} = A_{ndb+1, i+k} - L_{ndb+1, i} @ L_{i+k, i}^{T}
            L[
                -arrow_blocksize:,
                (i + k) * diag_blocksize: (i + k + 1) * diag_blocksize,
            ] = (
                L[
                    -arrow_blocksize:,
                    (i + k) * diag_blocksize: (i + k + 1) * diag_blocksize,
                ]
                - L[-arrow_blocksize:, i *
                    diag_blocksize: (i + 1) * diag_blocksize]
                @ L[
                    (i + k) * diag_blocksize: (i + k + 1) * diag_blocksize,
                    i * diag_blocksize: (i + 1) * diag_blocksize,
                ].T
            )

        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}^{T}
        L[-arrow_blocksize:, -arrow_blocksize:] = (
            L[-arrow_blocksize:, -arrow_blocksize:]
            - L[-arrow_blocksize:, i *
                diag_blocksize: (i + 1) * diag_blocksize]
            @ L[-arrow_blocksize:, i * diag_blocksize: (i + 1) * diag_blocksize].T
        )

    # L_{ndb, ndb} = chol(A_{ndb, ndb})
    L[
        -diag_blocksize - arrow_blocksize: -arrow_blocksize,
        -diag_blocksize - arrow_blocksize: -arrow_blocksize,
    ] = la.cholesky(
        L[
            -diag_blocksize - arrow_blocksize: -arrow_blocksize,
            -diag_blocksize - arrow_blocksize: -arrow_blocksize,
        ]
    ).T

    # L_{ndb+1, nbd} = A_{ndb+1, nbd} @ L_{ndb, ndb}^{-T}
    L[-arrow_blocksize:, -diag_blocksize - arrow_blocksize: -arrow_blocksize] = (
        L[-arrow_blocksize:, -diag_blocksize - arrow_blocksize: -arrow_blocksize]
        @ la.solve_triangular(
            L[
                -diag_blocksize - arrow_blocksize: -arrow_blocksize,
                -diag_blocksize - arrow_blocksize: -arrow_blocksize,
            ],
            np.eye(diag_blocksize),
            lower=True,
        ).T
    )

    # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, ndb} @ L_{ndb+1, ndb}^{T}
    L[-arrow_blocksize:, -arrow_blocksize:] = (
        L[-arrow_blocksize:, -arrow_blocksize:]
        - L[-arrow_blocksize:, -diag_blocksize -
            arrow_blocksize: -arrow_blocksize]
        @ L[-arrow_blocksize:, -diag_blocksize - arrow_blocksize: -arrow_blocksize].T
    )

    # L_{ndb+1, ndb+1} = chol(A_{ndb+1, ndb+1})
    L[-arrow_blocksize:, -arrow_blocksize:] = la.cholesky(
        L[-arrow_blocksize:, -arrow_blocksize:]
    ).T

    # zero out upper triangular part
    L[:] = L * np.tri(*L.shape, k=0)

    return L

# SCPOBBAF in compressed format


def scpobbaf_c(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_arrow_bottom_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    overwrite: bool = False,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Perform the Cholesky factorization of a block n-diagonals arrowhead
    matrix in compressed format. The matrix is assumed to be symmetric positive
    definite.

    Parameters
    ----------
    A_diagonal_blocks : ArrayLike
        Blocks of the main diagonal of the matrix.
    A_lower_diagonal_blocks : ArrayLike
        Blocks of the lower diagonals of the matrix.
    A_arrow_bottom_blocks : ArrayLike
        Blocks of the arrow part.
    A_arrow_tip_block : ArrayLike
        Tip of the arrow, lower right corner.
    overwrite : bool, optional
        If True, the inputs are modified in place. Default is False.

    Returns
    -------
    L_diagonal_blocks : ArrayLike
        The diagonal blocks of the Cholesky factor.
    L_lower_diagonal_blocks : ArrayLike
        The lower diagonal blocks of the Cholesky factor.
    L_arrow_bottom_blocks : ArrayLike
        The arrow bottom blocks of the Cholesky factor.
    L_arrow_tip_block : ArrayLike
        The arrow tip block of the Cholesky factor.
    """

    if overwrite:
        L_diagonal_blocks = A_diagonal_blocks
        L_lower_diagonal_blocks = A_lower_diagonal_blocks
        L_arrow_bottom_blocks = A_arrow_bottom_blocks
        L_arrow_tip_block = A_arrow_tip_block
    else:
        L_diagonal_blocks = xp.copy(A_diagonal_blocks)
        L_lower_diagonal_blocks = xp.copy(A_lower_diagonal_blocks)
        L_arrow_bottom_blocks = xp.copy(A_arrow_bottom_blocks)
        L_arrow_tip_block = xp.copy(A_arrow_tip_block)

    n_diag_blocks, diag_blocksize, _ = L_diagonal_blocks.shape
    # Number of lower diagonals, total bandwidth is n_offdiags_blk*2+1
    n_offdiags_blk = L_lower_diagonal_blocks.shape[1]//diag_blocksize

    for i in range(n_diag_blocks-1):
        # L_{i, i} = chol(A_{i, i})
        L_diagonal_blocks[i, :, :] = cholesky(
            L_diagonal_blocks[i, :, :])

        for j in range(1, min(n_offdiags_blk + 1, n_diag_blocks - i)):
            # L_{i+j, i} = A_{i+j, i} @ L_{i, i}^{-T}
            L_lower_diagonal_blocks[
                i, (j - 1)*diag_blocksize:j*diag_blocksize, :] = (
                    la.solve_triangular(
                        L_diagonal_blocks[i, :, :],
                        L_lower_diagonal_blocks[
                            i, (j - 1)*diag_blocksize:j * diag_blocksize, :].conj().T,
                        lower=True,
                    ).conj().T
            )

            # Update next blocks in row j
            Liji = L_lower_diagonal_blocks[
                i, (j - 1)*diag_blocksize:j*diag_blocksize, :]
            for k in range(1, j):
                # L_{i+j, i+k} = A_{i+j, i+k} - L_{i+j, i} @ L_{i+k, i}^{T}
                L_lower_diagonal_blocks[
                    i + k, (j - k - 1)*diag_blocksize:(j-k)*diag_blocksize, :
                ] -= Liji @ L_lower_diagonal_blocks[
                        i, (k - 1)*diag_blocksize:k*diag_blocksize, :].conj().T

            # Update next diagonal block
            # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ L_{i+1, i}.conj().T
            L_diagonal_blocks[i+j, :, :] -= Liji @ Liji.conj().T

        # Part of the decomposition for the arrowhead structure
        # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
        L_arrow_bottom_blocks[i, :, :] = (
            la.solve_triangular(
                L_diagonal_blocks[i, :, :],
                L_arrow_bottom_blocks[i, :, :].conj().T,
                lower=True,
            ).conj().T
        )

        for k in range(1, min(n_offdiags_blk + 1, n_diag_blocks - i)):
            # L_{ndb+1, i+k} = A_{ndb+1, i+k} - L_{ndb+1, i} @ L_{i+k, i}^{T}
            L_arrow_bottom_blocks[i + k, :, :] -= (
                L_arrow_bottom_blocks[i, :, :]
                @ L_lower_diagonal_blocks[
                    i, (k - 1)*diag_blocksize:k*diag_blocksize, :].conj().T
            )

        # L_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}^{T}
        L_arrow_tip_block[:, :] -= (
            L_arrow_bottom_blocks[i, :, :]
            @ L_arrow_bottom_blocks[i, :, :].conj().T
        )

    # L_{ndb, ndb} = chol(A_{ndb, ndb})
    L_diagonal_blocks[-1, :, :] = cholesky(
        L_diagonal_blocks[-1, :, :])

    # L_{ndb+1, nbd} = A_{ndb+1, nbd} @ L_{ndb, ndb}^{-T}
    L_arrow_bottom_blocks[-1, :, :] = (
        la.solve_triangular(
            L_diagonal_blocks[-1, :, :],
            L_arrow_bottom_blocks[-1, :, :].conj().T,
            lower=True,
        )
        .conj()
        .T
    )

    # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, ndb} @ L_{ndb+1, ndb}^{T}
    L_arrow_tip_block[:, :] -= (
        L_arrow_bottom_blocks[-1, :, :]
        @ L_arrow_bottom_blocks[-1, :, :].conj().T
    )

    # L_{ndb+1, ndb+1} = chol(A_{ndb+1, ndb+1})
    L_arrow_tip_block[:, :] = cholesky(L_arrow_tip_block[:, :])

    return (L_diagonal_blocks, L_lower_diagonal_blocks,
            L_arrow_bottom_blocks, L_arrow_tip_block)
