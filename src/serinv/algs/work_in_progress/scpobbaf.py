# Copyright 2023-2024 ETH Zurich. All rights reserved.

import numpy as np
import scipy.linalg as la
from numpy.typing import ArrayLike


def scpobbaf(
    A: np.ndarray,
    ndiags: int,
    diag_blocksize: int,
    arrow_blocksize: int,
    overwrite: bool = False,
) -> np.ndarray:
    """Perform the cholesky factorization of a block n-diagonals arrowhead
    matrix. The matrix is assumed to be symmetric positive definite.

    Parameters
    ----------
    A : np.ndarray
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
    L : np.ndarray
        The cholesky factorization of the matrix.
    """

    if overwrite:
        L = A
    else:
        L = np.copy(A)

    L_inv_temp = np.zeros((diag_blocksize, diag_blocksize))

    n_offdiags_blk = ndiags - 1

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
) -> np.ndarray:
    """Perform the cholesky factorization of a block n-diagonals arrowhead
    matrix. The matrix is assumed to be symmetric positive definite.

    Parameters
    ----------
    A : np.ndarray
        Input matrix to decompose.
    A_diagonal_blocks : int
        Number of diagonals of the matrix.
    A_lower_diagonal_blocks : int
        Blocksize of the diagonals blocks of the matrix.
    A_arrow_bottom_blocks : int
        Blocksize of the blocks composing the arrowhead.
    A_arrow_tip_block : ArrayLike
        TODO: correct
    overwrite : bool
        If True, the input matrix A is modified in place. Default is False.

    Returns
    -------
    L : np.ndarray
        The cholesky factorization of the matrix.
    """

    if overwrite:
        L_diagonal_blocks = A_diagonal_blocks
        L_lower_diagonal_blocks = A_lower_diagonal_blocks
        L_arrow_bottom_blocks = A_arrow_bottom_blocks
        L_arrow_tip_block = A_arrow_tip_block
    else:
        L_diagonal_blocks = np.copy(A_diagonal_blocks)
        L_lower_diagonal_blocks = np.copy(A_lower_diagonal_blocks)
        L_arrow_bottom_blocks = np.copy(A_arrow_bottom_blocks)
        L_arrow_tip_block = np.copy(A_arrow_tip_block)

    n_diag_blocks, diag_blocksize, _, n_offdiags_blk = L_lower_diagonal_blocks.shape

    L_inv_temp = np.zeros((diag_blocksize, diag_blocksize))

    for i in range(n_diag_blocks-1):
        # L_{i, i} = chol(A_{i, i})
        L_diagonal_blocks[i, :, :] = la.cholesky(
            L_diagonal_blocks[i, :, :]).T

        # Temporary storage of re-used triangular solving
        L_inv_temp = la.solve_triangular(
            L_diagonal_blocks[i, :, :],
            np.eye(diag_blocksize),
            lower=True,
        ).T

        for j in range(min(n_offdiags_blk, n_diag_blocks - i)):
            # L_{i+j, i} = A_{i+j, i} @ L_{i, i}^{-T}
            L_lower_diagonal_blocks[i, :, :, j] = (
                L_lower_diagonal_blocks[i, :, :, j] @ L_inv_temp
            )

            for k in range(1, j+1):
                # L_{i+j, i+k} = A_{i+j, i+k} - L_{i+j, i} @ L_{i+k, i}^{T}
                L_lower_diagonal_blocks[i+k, :, :, j - k] = (
                    L_lower_diagonal_blocks[i+k, :, :, j - k]
                    - L_lower_diagonal_blocks[i, :, :, j]
                    @ L_lower_diagonal_blocks[i, :, :, k].T
                )

        # Part of the decomposition for the arrowhead structure
        # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
        L_arrow_bottom_blocks[i, :, :] = (
            L_arrow_bottom_blocks[i, :, :] @ L_inv_temp)

        for k in range(1, min(n_offdiags_blk, n_diag_blocks - i)):
            # L_{ndb+1, i+k} = A_{ndb+1, i+k} - L_{ndb+1, i} @ L_{i+k, i}^{T}
            L_arrow_bottom_blocks[i + k, :, :] = (
                L_arrow_bottom_blocks[i + k, :, :]
                - L_arrow_bottom_blocks[i, :, :]
                @ L_lower_diagonal_blocks[i, :, :, k].T
            )

        # L_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}^{T}
        L_arrow_tip_block[:, :] = (
            L_arrow_tip_block[:, :]
            - L_arrow_bottom_blocks[i, :, :]
            @ L_arrow_bottom_blocks[i, :, :].T
        )

    # L_{ndb, ndb} = chol(A_{ndb, ndb})
    L_diagonal_blocks[-1, :, :] = la.cholesky(L_diagonal_blocks[-1, :, :]).T

    # L_{ndb+1, nbd} = A_{ndb+1, nbd} @ L_{ndb, ndb}^{-T}
    L_arrow_bottom_blocks[-1, :, :] = (
        L_arrow_bottom_blocks[-1, :, :]
        @ la.solve_triangular(
            L_diagonal_blocks[-1, :, :],
            np.eye(diag_blocksize),
            lower=True,
        ).T
    )

    # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, ndb} @ L_{ndb+1, ndb}^{T}
    L_arrow_tip_block[:, :] = (
        L_arrow_tip_block[:, :]
        - L_arrow_bottom_blocks[-1, :, :]
        @ L_arrow_bottom_blocks[-1, :, :].T
    )

    # L_{ndb+1, ndb+1} = chol(A_{ndb+1, ndb+1})
    L_arrow_tip_block[:, :] = la.cholesky(L_arrow_tip_block[:, :]).T

    return (L_diagonal_blocks, L_lower_diagonal_blocks, L_arrow_bottom_blocks, L_arrow_tip_block)
