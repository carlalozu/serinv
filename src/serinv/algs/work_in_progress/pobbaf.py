# Copyright 2023-2024 ETH Zurich. All rights reserved.

try:
    import cupy as cp
    import cupyx.scipy.linalg as cu_la

    from serinv.cupyfix.cholesky_lowerfill import cholesky_lowerfill

    CUPY_AVAIL = True

except ImportError:
    CUPY_AVAIL = False

from typing import Tuple
import numpy as np
import scipy.linalg as np_la
from numpy.typing import ArrayLike


def pobbaf(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_arrow_bottom_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
    device_streaming: bool = False,
    overwrite: bool = False,
):

    if CUPY_AVAIL and cp.get_array_module(A_diagonal_blocks) == np and device_streaming:
        return _streaming_pobbaf(
            A_diagonal_blocks,
            A_lower_diagonal_blocks,
            A_arrow_bottom_blocks,
            A_arrow_tip_block,
        )

    return _pobbaf(
        A_diagonal_blocks,
        A_lower_diagonal_blocks,
        A_arrow_bottom_blocks,
        A_arrow_tip_block,
        overwrite
    )


def _pobbaf(
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
        L_diagonal_blocks = np.copy(A_diagonal_blocks)
        L_lower_diagonal_blocks = np.copy(A_lower_diagonal_blocks)
        L_arrow_bottom_blocks = np.copy(A_arrow_bottom_blocks)
        L_arrow_tip_block = np.copy(A_arrow_tip_block)

    n_diag_blocks, diag_blocksize, _ = L_diagonal_blocks.shape
    # Number of lower diagonals, total bandwidth is n_offdiags_blk*2+1
    n_offdiags_blk = L_lower_diagonal_blocks.shape[1]//diag_blocksize

    for i in range(n_diag_blocks-1):
        # L_{i, i} = chol(A_{i, i})
        L_diagonal_blocks[i, :, :] = np.linalg.cholesky(
            L_diagonal_blocks[i, :, :])

        for j in range(1, min(n_offdiags_blk + 1, n_diag_blocks - i)):
            # L_{i+j, i} = A_{i+j, i} @ L_{i, i}^{-T}
            L_lower_diagonal_blocks[
                i, (j - 1)*diag_blocksize:j*diag_blocksize, :] = (
                    np_la.solve_triangular(
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
            np_la.solve_triangular(
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
    L_diagonal_blocks[-1, :, :] = np.linalg.cholesky(
        L_diagonal_blocks[-1, :, :])

    # L_{ndb+1, nbd} = A_{ndb+1, nbd} @ L_{ndb, ndb}^{-T}
    L_arrow_bottom_blocks[-1, :, :] = (
        np_la.solve_triangular(
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
    L_arrow_tip_block[:, :] = np.linalg.cholesky(L_arrow_tip_block[:, :])

    return (L_diagonal_blocks, L_lower_diagonal_blocks,
            L_arrow_bottom_blocks, L_arrow_tip_block)


def _streaming_pobbaf(
    A_diagonal_blocks: ArrayLike,
    A_lower_diagonal_blocks: ArrayLike,
    A_arrow_bottom_blocks: ArrayLike,
    A_arrow_tip_block: ArrayLike,
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike,]:

    compute_stream = cp.cuda.Stream(non_blocking=True)
    h2d_stream = cp.cuda.Stream(non_blocking=True)
    d2h_stream = cp.cuda.Stream(non_blocking=True)

    h2d_diagonal_events = [cp.cuda.Event(), cp.cuda.Event()]
    h2d_lower_events = [cp.cuda.Event(), cp.cuda.Event()]
    h2d_arrow_events = [cp.cuda.Event(), cp.cuda.Event()]

    d2h_diagonal_events = [cp.cuda.Event(), cp.cuda.Event()]

    compute_diagonal_events = [cp.cuda.Event(), cp.cuda.Event()]
    compute_lower_events = [cp.cuda.Event(), cp.cuda.Event()]
    compute_lower_h2d_events = [cp.cuda.Event(), cp.cuda.Event()]
    compute_arrow_events = [cp.cuda.Event(), cp.cuda.Event()]
    compute_arrow_h2d_events = [cp.cuda.Event(), cp.cuda.Event()]

    # L host aliases
    L_diagonal_blocks = A_diagonal_blocks
    L_lower_diagonal_blocks = A_lower_diagonal_blocks
    L_arrow_bottom_blocks = A_arrow_bottom_blocks
    L_arrow_tip_block = A_arrow_tip_block

    # Device buffers
    A_diagonal_blocks_d = cp.empty(
        (2, *A_diagonal_blocks.shape[1:]), dtype=A_diagonal_blocks.dtype
    )
    A_lower_diagonal_blocks_d = cp.empty(
        (2, *A_diagonal_blocks.shape[1:]), dtype=A_diagonal_blocks.dtype
    )
    A_arrow_bottom_blocks_d = cp.empty(
        (2, *A_arrow_bottom_blocks.shape[1:]), dtype=A_arrow_bottom_blocks.dtype
    )
    A_arrow_tip_block_d = cp.empty_like(A_arrow_tip_block)

    # X Device buffers arrays pointers
    L_diagonal_blocks_d = A_diagonal_blocks_d
    L_lower_diagonal_blocks_d = A_lower_diagonal_blocks_d
    L_arrow_bottom_blocks_d = A_arrow_bottom_blocks_d
    L_arrow_tip_block_d = A_arrow_tip_block_d

    # Forward pass
    # --- C: events + transfers---
    compute_lower_h2d_events[1].record(stream=compute_stream)
    compute_arrow_h2d_events[1].record(stream=compute_stream)
    A_arrow_tip_block_d.set(arr=A_arrow_tip_block[:, :], stream=compute_stream)

    # --- H2D: transfers ---
    A_diagonal_blocks_d[0, :, :].set(
        arr=A_diagonal_blocks[0, :, :], stream=h2d_stream)
    h2d_diagonal_events[0].record(stream=h2d_stream)
    A_arrow_bottom_blocks_d[0, :, :].set(
        arr=A_arrow_bottom_blocks[0, :, :], stream=h2d_stream
    )
    h2d_arrow_events[0].record(stream=h2d_stream)

    n_diag_blocks = A_diagonal_blocks.shape[0]
    if n_diag_blocks > 1:
        A_lower_diagonal_blocks_d[0, :, :].set(
            arr=A_lower_diagonal_blocks[0, :, :], stream=h2d_stream
        )
        h2d_lower_events[0].record(stream=h2d_stream)

    # --- D2H: event ---
    d2h_diagonal_events[1].record(stream=d2h_stream)

    # FOR
    for i in range(n_diag_blocks-1):
        # --- Computations ---
        # L_{i, i} = chol(A_{i, i})
        with compute_stream:
            compute_stream.wait_event(h2d_diagonal_events[i % 2])
            L_diagonal_blocks_d[i % 2, :, :] = cholesky_lowerfill(
                A_diagonal_blocks_d[i % 2, :, :]
            )
            compute_diagonal_events[i % 2].record(stream=compute_stream)

        d2h_stream.wait_event(compute_diagonal_events[i % 2])
        L_diagonal_blocks_d[i % 2, :, :].get(
            out=L_diagonal_blocks[i, :, :],
            stream=d2h_stream,
            blocking=False,
        )
        d2h_diagonal_events[i % 2].record(stream=d2h_stream)

        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}.conj().T
        A_arrow_tip_block_d[:, :] -= (
            L_arrow_bottom_blocks_d[i % 2, :, :]
            @ L_arrow_bottom_blocks_d[i % 2, :, :].conj().T
        )
        compute_arrow_h2d_events[i % 2].record(stream=compute_stream)

    # L_{ndb, ndb} = chol(A_{ndb, ndb})
    with compute_stream:
        compute_stream.wait_event(h2d_diagonal_events[(n_diag_blocks - 1) % 2])
        L_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :] = cholesky_lowerfill(
            A_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :]
        )
        compute_diagonal_events[(n_diag_blocks - 1) %
                                2].record(stream=compute_stream)

    d2h_stream.wait_event(compute_diagonal_events[(n_diag_blocks - 1) % 2])
    L_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :].get(
        out=L_diagonal_blocks[-1, :, :],
        stream=d2h_stream,
        blocking=False,
    )

    # L_{ndb+1, ndb} = A_{ndb+1, ndb} @ L_{ndb, ndb}^{-T}
    with compute_stream:
        compute_stream.wait_event(h2d_arrow_events[(n_diag_blocks - 1) % 2])
        L_arrow_bottom_blocks_d[(n_diag_blocks - 1) % 2, :, :] = (
            cu_la.solve_triangular(
                L_diagonal_blocks_d[(n_diag_blocks - 1) % 2, :, :],
                A_arrow_bottom_blocks_d[(n_diag_blocks - 1) %
                                        2, :, :].conj().T,
                lower=True,
            )
            .conj()
            .T
        )
        compute_arrow_events[(n_diag_blocks - 1) %
                             2].record(stream=compute_stream)

    d2h_stream.wait_event(compute_arrow_events[(n_diag_blocks - 1) % 2])
    L_arrow_bottom_blocks_d[(n_diag_blocks - 1) % 2, :, :].get(
        out=L_arrow_bottom_blocks[-1, :, :],
        stream=d2h_stream,
        blocking=False,
    )

    with compute_stream:
        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, ndb} @ L_{ndb+1, ndb}^{T}
        A_arrow_tip_block_d[:, :] = (
            A_arrow_tip_block_d[:, :]
            - L_arrow_bottom_blocks_d[(n_diag_blocks - 1) % 2, :, :]
            @ L_arrow_bottom_blocks_d[(n_diag_blocks - 1) % 2, :, :].conj().T
        )

        # L_{ndb+1, ndb+1} = chol(A_{ndb+1, ndb+1})
        L_arrow_tip_block_d[:, :] = cholesky_lowerfill(
            A_arrow_tip_block_d[:, :])

        L_arrow_tip_block_d[:, :].get(
            out=L_arrow_tip_block[:, :], stream=compute_stream
        )

    cp.cuda.Device().synchronize()

    return (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_arrow_tip_block,
    )
