# Copyright 2023-2024 ETH Zurich. All rights reserved.

import numpy as np
import scipy.linalg as la
from numpy.typing import ArrayLike


def scpobbasi(
    L: np.ndarray,
    ndiags: int,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> np.ndarray:
    """Perform a selected inversion of a block banded (?) matrix using a
    sequential algorithm on CPU backend.

    Parameters
    ----------
    L : np.ndarray
        The cholesky factorization of the matrix.
    ndiags : int
        Number of diagonals.
    diag_blocksize : int
        Blocksize of the diagonals blocks of the matrix.
    arrow_blocksize : int
        Blocksize of the blocks composing the arrowhead.

    Returns
    -------
    X : np.ndarray
        Selected inversion of the matrix.
    """

    X = np.zeros(L.shape, dtype=L.dtype)
    L_last_blk_inv = np.zeros(
        (arrow_blocksize, arrow_blocksize), dtype=L.dtype)

    L_last_blk_inv = la.solve_triangular(
        L[-arrow_blocksize:, -arrow_blocksize:], np.eye(arrow_blocksize), lower=True
    )
    X[-arrow_blocksize:, -arrow_blocksize:] = L_last_blk_inv.T @ L_last_blk_inv

    L_blk_inv = np.zeros((diag_blocksize, diag_blocksize), dtype=L.dtype)
    n_diag_blocks = (L.shape[0] - arrow_blocksize) // diag_blocksize
    n_offdiags_blk = ndiags // 2
    for i in range(n_diag_blocks - 1, -1, -1):
        # L_blk_inv = L_{i, i}^{-1}
        L_blk_inv = la.solve_triangular(
            L[
                i * diag_blocksize: (i + 1) * diag_blocksize,
                i * diag_blocksize: (i + 1) * diag_blocksize,
            ],
            np.eye(diag_blocksize),
            lower=True,
        )

        # Arrowhead part
        # X_{ndb+1, i} = - X_{ndb+1, ndb+1} L_{ndb+1, i}
        X[-arrow_blocksize:, i * diag_blocksize: (i + 1) * diag_blocksize] = (
            -X[-arrow_blocksize:, -arrow_blocksize:]
            @ L[-arrow_blocksize:, i * diag_blocksize: (i + 1) * diag_blocksize]
        )

        for k in range(i + 1, min(i + n_offdiags_blk + 1, n_diag_blocks), 1):
            # X_{ndb+1, i} = X_{ndb+1, i} - X_{ndb+1, k} L_{k, i}
            X[-arrow_blocksize:, i * diag_blocksize: (i + 1) * diag_blocksize] -= (
                X[-arrow_blocksize:, k *
                    diag_blocksize: (k + 1) * diag_blocksize]
                @ L[
                    k * diag_blocksize: (k + 1) * diag_blocksize,
                    i * diag_blocksize: (i + 1) * diag_blocksize,
                ]
            )

        # X_{ndb+1, i} = X_{ndb+1, i} L_{i, i}^{-1}
        X[-arrow_blocksize:, i * diag_blocksize: (i + 1) * diag_blocksize] = (
            X[-arrow_blocksize:, i * diag_blocksize: (i + 1) * diag_blocksize]
            @ L_blk_inv
        )

        # X_{i, ndb+1} = X_{ndb+1, i}.T
        X[i * diag_blocksize: (i + 1) * diag_blocksize, -arrow_blocksize:] = X[
            -arrow_blocksize:, i * diag_blocksize: (i + 1) * diag_blocksize
        ].T

        # Off-diagonal block part
        for j in range(min(i + n_offdiags_blk, n_diag_blocks - 1), i, -1):
            # Take the effect of the arrowhead part into account
            # X_{j, i} = - X_{ndb+1, j}.T L_{ndb+1, i}
            X[
                j * diag_blocksize: (j + 1) * diag_blocksize,
                i * diag_blocksize: (i + 1) * diag_blocksize,
            ] = (
                -X[-arrow_blocksize:, j *
                    diag_blocksize: (j + 1) * diag_blocksize].T
                @ L[-arrow_blocksize:, i * diag_blocksize: (i + 1) * diag_blocksize]
            )

            for k in range(i + 1, min(i + n_offdiags_blk + 1, n_diag_blocks), 1):
                # The following condition ensure to use the lower elements
                # produced during the selected inversion process. ie. the matrix
                # is symmetric.
                if k > j:

                    # X_{j, i} = X_{j, i} - X_{k, j}.T L_{k, i}
                    X[
                        j * diag_blocksize: (j + 1) * diag_blocksize,
                        i * diag_blocksize: (i + 1) * diag_blocksize,
                    ] -= (
                        X[
                            k * diag_blocksize: (k + 1) * diag_blocksize,
                            j * diag_blocksize: (j + 1) * diag_blocksize,
                        ].T
                        @ L[
                            k * diag_blocksize: (k + 1) * diag_blocksize,
                            i * diag_blocksize: (i + 1) * diag_blocksize,
                        ]
                    )
                else:
                    # X_{j, i} = X_{j, i} - X_{j, k} L_{k, i}
                    X[
                        j * diag_blocksize: (j + 1) * diag_blocksize,
                        i * diag_blocksize: (i + 1) * diag_blocksize,
                    ] -= (
                        X[
                            j * diag_blocksize: (j + 1) * diag_blocksize,
                            k * diag_blocksize: (k + 1) * diag_blocksize,
                        ]
                        @ L[
                            k * diag_blocksize: (k + 1) * diag_blocksize,
                            i * diag_blocksize: (i + 1) * diag_blocksize,
                        ]
                    )

            # X_{j, i} = X_{j, i} L_{i, i}^{-1}
            X[
                j * diag_blocksize: (j + 1) * diag_blocksize,
                i * diag_blocksize: (i + 1) * diag_blocksize,
            ] = (
                X[
                    j * diag_blocksize: (j + 1) * diag_blocksize,
                    i * diag_blocksize: (i + 1) * diag_blocksize,
                ]
                @ L_blk_inv
            )

            # X_{i, j} = X_{j, i}.T
            X[
                i * diag_blocksize: (i + 1) * diag_blocksize,
                j * diag_blocksize: (j + 1) * diag_blocksize,
            ] = X[
                j * diag_blocksize: (j + 1) * diag_blocksize,
                i * diag_blocksize: (i + 1) * diag_blocksize,
            ].T

        # Diagonal block part
        # X_{i, i} = (L_{i, i}^{-T} - X_{ndb+1, i}.T L_{ndb+1, i} - sum_{k=i+1}^{min(i+ndiags/2, n_diag_blocks)} X_{k, i}.T L_{k, i}) L_{i, i}^{-1}

        # X_{i, i} = L_{i, i}^{-T} - X_{ndb+1, i}.T L_{ndb+1, i}
        X[
            i * diag_blocksize: (i + 1) * diag_blocksize,
            i * diag_blocksize: (i + 1) * diag_blocksize,
        ] = (
            L_blk_inv.T
            - X[-arrow_blocksize:, i *
                diag_blocksize: (i + 1) * diag_blocksize].T
            @ L[-arrow_blocksize:, i * diag_blocksize: (i + 1) * diag_blocksize]
        )

        for k in range(i + 1, min(i + n_offdiags_blk + 1, n_diag_blocks), 1):
            # X_{i, i} = X_{i, i} - X_{k, i}.T L_{k, i}
            X[
                i * diag_blocksize: (i + 1) * diag_blocksize,
                i * diag_blocksize: (i + 1) * diag_blocksize,
            ] -= (
                X[
                    k * diag_blocksize: (k + 1) * diag_blocksize,
                    i * diag_blocksize: (i + 1) * diag_blocksize,
                ].T
                @ L[
                    k * diag_blocksize: (k + 1) * diag_blocksize,
                    i * diag_blocksize: (i + 1) * diag_blocksize,
                ]
            )

        # X_{i, i} = X_{i, i} L_{i, i}^{-1}
        X[
            i * diag_blocksize: (i + 1) * diag_blocksize,
            i * diag_blocksize: (i + 1) * diag_blocksize,
        ] = (
            X[
                i * diag_blocksize: (i + 1) * diag_blocksize,
                i * diag_blocksize: (i + 1) * diag_blocksize,
            ]
            @ L_blk_inv
        )

    return X


def scpobbasi_c(
    L_diagonal_blocks: ArrayLike,
    L_lower_diagonal_blocks: ArrayLike,
    L_arrow_bottom_blocks: ArrayLike,
    L_arrow_tip_block: ArrayLike,
    overwrite: bool = False,
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Perform a selected inversion of a block banded matrix by means of its
    Cholesky factor using a sequential algorithm on CPU backend.

    Parameters
    ----------
    L_diagonal_blocks : ArrayLike
        Diagonal blocks of the Cholesky factor.
    L_lower_diagonal_blocks : ArrayLike
        Lower diagonal blocks of the Cholesky factor.
    L_arrow_bottom_blocks : ArrayLike
        Arrow bottom blocks of the Cholesky factor.
    L_arrow_tip_block : ArrayLike
        Arrow tip block of the Cholesky factor.

    Returns
    -------
    X_diagonal_blocks : ArrayLike
        Diagonal blocks of the selected inversion of the matrix.
    X_lower_diagonal_blocks : ArrayLike
        Lower diagonal blocks of the selected inversion of the matrix.
    X_arrow_bottom_blocks : ArrayLike
        Arrow bottom blocks of the selected inversion of the matrix.
    X_arrow_tip_block : ArrayLike
        Arrow tip block of the selected inversion of the matrix.
    """

    if overwrite:
        X_diagonal_blocks = L_diagonal_blocks
        X_lower_diagonal_blocks = L_lower_diagonal_blocks
        X_arrow_bottom_blocks = L_arrow_bottom_blocks
        X_arrow_tip_block = L_arrow_tip_block
    else:
        X_diagonal_blocks = np.copy(L_diagonal_blocks)
        X_lower_diagonal_blocks = np.copy(L_lower_diagonal_blocks)
        X_arrow_bottom_blocks = np.copy(L_arrow_bottom_blocks)
        X_arrow_tip_block = np.copy(L_arrow_tip_block)

    n_diag_blocks, diag_blocksize, _ = X_diagonal_blocks.shape
    arrow_blocksize = X_arrow_tip_block.shape[0]
    # Number of lower diagonals, total bandwidth is n_offdiags_blk*2+1
    n_offdiags_blk = X_lower_diagonal_blocks.shape[1]//diag_blocksize

    L_last_blk_inv = np.zeros(
        (arrow_blocksize, arrow_blocksize),
        dtype=X_arrow_tip_block.dtype
    )

    L_last_blk_inv = la.solve_triangular(
        X_arrow_tip_block[:, :],
        np.eye(arrow_blocksize),
        lower=True
    )
    X_arrow_tip_block[:,:] = L_last_blk_inv.conj().T @ L_last_blk_inv

    L_blk_inv = np.zeros((diag_blocksize, diag_blocksize),
                         dtype=X_arrow_tip_block.dtype)

    for i in range(n_diag_blocks - 1, -1, -1):
        # L_blk_inv = L_{i, i}^{-1}
        L_blk_inv[:,:] = la.solve_triangular(
            X_diagonal_blocks[i, :, :],
            np.eye(diag_blocksize),
            lower=True,
        )

        # Arrowhead part
        # X_{ndb+1, i} = - X_{ndb+1, ndb+1} L_{ndb+1, i}
        X_arrow_bottom_blocks[i, :] = (
            -X_arrow_tip_block[:, :] @ X_arrow_bottom_blocks[i, :]
        )

        for k in range(i + 1, min(i + n_offdiags_blk + 1, n_diag_blocks), 1):
            # X_{ndb+1, i} = X_{ndb+1, i} - X_{ndb+1, k} L_{k, i}
            X_arrow_bottom_blocks[i, :] -= (
                X_arrow_bottom_blocks[k, :] @
                L_lower_diagonal_blocks[
                    i, (k - i - 1) * diag_blocksize: (k - i) * diag_blocksize, :]
            )

        # X_{ndb+1, i} = X_{ndb+1, i} L_{i, i}^{-1}
        X_arrow_bottom_blocks[i, :] = (X_arrow_bottom_blocks[i, :] @ L_blk_inv)

        # Off-diagonal block part
        for j in range(min(i + n_offdiags_blk, n_diag_blocks - 1), i, -1):
            # Take the effect of the arrowhead part into account
            # X_{j, i} = - X_{ndb+1, j}.T L_{ndb+1, i}
            X_lower_diagonal_blocks[
                i, (j - i - 1) * diag_blocksize: (j - i) * diag_blocksize, :
            ] = (
                -X_arrow_bottom_blocks[j, :].conj().T
                @ L_arrow_bottom_blocks[i, :]
            )

            for k in range(i + 1, min(i + n_offdiags_blk + 1, n_diag_blocks), 1):
                # The following condition ensure to use the lower elements
                # produced during the selected inversion process. ie. the matrix
                # is symmetric.
                if k > j:
                    # X_temp = X_{k, j}
                    X_temp = X_lower_diagonal_blocks[
                        j, (k - j - 1) * diag_blocksize: (k - j) * diag_blocksize, :
                    ].conj().T
                elif k < j:
                    # X_temp = X_{j, k}
                    X_temp = X_lower_diagonal_blocks[
                        k, (j - k - 1) * diag_blocksize: (j - k) * diag_blocksize, :
                    ]
                else:  # k == j
                    # X_temp = X_{j, j}
                    X_temp = X_diagonal_blocks[k, :, :].conj().T

                # X_{j, i} = X_{j, i} - X_temp.T L_{k, i}
                X_lower_diagonal_blocks[
                    i, (j - i - 1) * diag_blocksize: (j - i) * diag_blocksize, :
                ] -= (
                    X_temp @ L_lower_diagonal_blocks[
                        i, (k - i - 1) * diag_blocksize: (k - i) * diag_blocksize, :
                    ]
                )

            # X_{j, i} = X_{j, i} L_{i, i}^{-1}
            X_lower_diagonal_blocks[
                i, (j - i - 1) * diag_blocksize: (j - i) * diag_blocksize, :
            ] = (
                X_lower_diagonal_blocks[
                    i, (j - i - 1) * diag_blocksize: (j - i) * diag_blocksize, :
                ] @ L_blk_inv
            )

        # Diagonal block part
        # X_{i, i} = (L_{i, i}^{-T} - X_{ndb+1, i}.T L_{ndb+1, i} -
        # sum_{k=i+1}^{min(i+ndiags/2, n_diag_blocks)} X_{k, i}.T L_{k, i})
        # L_{i, i}^{-1}

        # X_{i, i} = L_{i, i}^{-T} - X_{ndb+1, i}.T L_{ndb+1, i}
        X_diagonal_blocks[i, :, :] = (
            L_blk_inv.conj().T - X_arrow_bottom_blocks[
                i, :].conj().T @ L_arrow_bottom_blocks[i, :]
        )

        for k in range(i + 1, min(i + n_offdiags_blk + 1, n_diag_blocks), 1):
            # X_{i, i} = X_{i, i} - X_{k, i}.T L_{k, i}
            X_diagonal_blocks[i, :, :] -= (
                X_lower_diagonal_blocks[
                    i, (k - i - 1) * diag_blocksize: (k - i) * diag_blocksize, :
                ].conj().T
                @ L_lower_diagonal_blocks[
                    i, (k - i - 1) * diag_blocksize: (k - i) * diag_blocksize, :
                ]
            )

        # X_{i, i} = X_{i, i} L_{i, i}^{-1}
        X_diagonal_blocks[i, :, :] = X_diagonal_blocks[i, :, :] @ L_blk_inv

    return (X_diagonal_blocks, X_lower_diagonal_blocks,
            X_arrow_bottom_blocks, X_arrow_tip_block)
