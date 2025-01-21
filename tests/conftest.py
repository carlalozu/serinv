# Copyright 2023-2024 ETH Zurich. All rights reserved.

import numpy as np
import pytest
from numpy.typing import ArrayLike

SEED = 63

try:
    import cupy as cp

    CUPY_AVAIL = True
    cp.random.seed(cp.uint64(SEED))

except ImportError:
    CUPY_AVAIL = False


np.random.seed(SEED)


@pytest.fixture(scope="function", autouse=False)
def bta_dense_to_arrays():
    def _bta_dense_to_arrays(
        bta: ArrayLike,
        diagonal_blocksize: int,
        arrowhead_blocksize: int,
        n_diag_blocks: int,
    ):
        """Converts a block tridiagonal arrowhead matrix from a dense representation to arrays of blocks."""
        if CUPY_AVAIL:
            xp = cp.get_array_module(bta)
        else:
            xp = np

        A_diagonal_blocks = xp.zeros(
            (n_diag_blocks, diagonal_blocksize, diagonal_blocksize),
            dtype=bta.dtype,
        )

        A_lower_diagonal_blocks = xp.zeros(
            (n_diag_blocks - 1, diagonal_blocksize, diagonal_blocksize),
            dtype=bta.dtype,
        )
        A_upper_diagonal_blocks = xp.zeros(
            (n_diag_blocks - 1, diagonal_blocksize, diagonal_blocksize),
            dtype=bta.dtype,
        )

        A_arrow_bottom_blocks = xp.zeros(
            (n_diag_blocks, arrowhead_blocksize, diagonal_blocksize),
            dtype=bta.dtype,
        )

        A_arrow_right_blocks = xp.zeros(
            (n_diag_blocks, diagonal_blocksize, arrowhead_blocksize),
            dtype=bta.dtype,
        )

        for i in range(n_diag_blocks):
            A_diagonal_blocks[i, :, :] = bta[
                i * diagonal_blocksize: (i + 1) * diagonal_blocksize,
                i * diagonal_blocksize: (i + 1) * diagonal_blocksize,
            ]
            if i > 0:
                A_lower_diagonal_blocks[i - 1, :, :] = bta[
                    i * diagonal_blocksize: (i + 1) * diagonal_blocksize,
                    (i - 1) * diagonal_blocksize: i * diagonal_blocksize,
                ]
            if i < n_diag_blocks - 1:
                A_upper_diagonal_blocks[i, :, :] = bta[
                    i * diagonal_blocksize: (i + 1) * diagonal_blocksize,
                    (i + 1) * diagonal_blocksize: (i + 2) * diagonal_blocksize,
                ]

            A_arrow_bottom_blocks[i, :, :] = bta[
                -arrowhead_blocksize:,
                i * diagonal_blocksize: (i + 1) * diagonal_blocksize,
            ]

            A_arrow_right_blocks[i, :, :] = bta[
                i * diagonal_blocksize: (i + 1) * diagonal_blocksize,
                -arrowhead_blocksize:,
            ]

        A_arrow_tip_block = bta[-arrowhead_blocksize:, -arrowhead_blocksize:]

        return (
            A_diagonal_blocks,
            A_lower_diagonal_blocks,
            A_upper_diagonal_blocks,
            A_arrow_bottom_blocks,
            A_arrow_right_blocks,
            A_arrow_tip_block,
        )

    return _bta_dense_to_arrays


@pytest.fixture(scope="function", autouse=False)
def bta_arrays_to_dense():
    def _bta_arrays_to_dense(
        A_diagonal_blocks: ArrayLike,
        A_lower_diagonal_blocks: ArrayLike,
        A_upper_diagonal_blocks: ArrayLike,
        A_arrow_bottom_blocks: ArrayLike,
        A_arrow_right_blocks: ArrayLike,
        A_arrow_tip_block: ArrayLike,
    ):
        """Converts arrays of blocks to a block tridiagonal arrowhead matrix in a dense representation."""
        if CUPY_AVAIL:
            xp = cp.get_array_module(A_diagonal_blocks)
        else:
            xp = np

        diagonal_blocksize = A_diagonal_blocks.shape[1]
        arrowhead_blocksize = A_arrow_bottom_blocks.shape[1]
        n_diag_blocks = A_diagonal_blocks.shape[0]

        bta = xp.zeros(
            (
                diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
                diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
            ),
            dtype=A_diagonal_blocks.dtype,
        )

        for i in range(n_diag_blocks):
            bta[
                i * diagonal_blocksize: (i + 1) * diagonal_blocksize,
                i * diagonal_blocksize: (i + 1) * diagonal_blocksize,
            ] = A_diagonal_blocks[i, :, :]
            if i > 0:
                bta[
                    i * diagonal_blocksize: (i + 1) * diagonal_blocksize,
                    (i - 1) * diagonal_blocksize: i * diagonal_blocksize,
                ] = A_lower_diagonal_blocks[i - 1, :, :]
            if i < n_diag_blocks - 1:
                bta[
                    i * diagonal_blocksize: (i + 1) * diagonal_blocksize,
                    (i + 1) * diagonal_blocksize: (i + 2) * diagonal_blocksize,
                ] = A_upper_diagonal_blocks[i, :, :]

            bta[
                -arrowhead_blocksize:,
                i * diagonal_blocksize: (i + 1) * diagonal_blocksize,
            ] = A_arrow_bottom_blocks[i, :, :]

            bta[
                i * diagonal_blocksize: (i + 1) * diagonal_blocksize,
                -arrowhead_blocksize:,
            ] = A_arrow_right_blocks[i, :, :]

        bta[-arrowhead_blocksize:, -arrowhead_blocksize:] = A_arrow_tip_block

        return bta

    return _bta_arrays_to_dense


@pytest.fixture(scope="function", autouse=False)
def bta_symmetrize():
    def _bta_symmetrize(
        bta: ArrayLike,
    ):
        """Symmetrizes a block tridiagonal arrowhead matrix."""

        return (bta + bta.conj().T) / 2

    return _bta_symmetrize


@pytest.fixture(scope="function", autouse=False)
def dd_bta(
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_diag_blocks: int,
    device_array: bool,
    dtype: np.dtype,
):
    """Returns a random, diagonaly dominant general, block tridiagonal arrowhead matrix."""
    xp = cp if device_array and CUPY_AVAIL else np

    DD_BTA = xp.zeros(
        (
            diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
            diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
        ),
        dtype=dtype,
    )

    rc = (1.0 + 1.0j) if dtype == np.complex128 else 1.0

    # Fill the lower arrowhead blocks
    DD_BTA[-arrowhead_blocksize:, :-arrowhead_blocksize] = rc * xp.random.rand(
        arrowhead_blocksize, n_diag_blocks * diagonal_blocksize
    )
    # Fill the right arrowhead blocks
    DD_BTA[:-arrowhead_blocksize, -arrowhead_blocksize:] = rc * xp.random.rand(
        n_diag_blocks * diagonal_blocksize, arrowhead_blocksize
    )

    # Fill the tip of the arrowhead
    DD_BTA[-arrowhead_blocksize:, -arrowhead_blocksize:] = rc * xp.random.rand(
        arrowhead_blocksize, arrowhead_blocksize
    )

    # Fill the diagonal blocks
    for i in range(n_diag_blocks):
        DD_BTA[
            i * diagonal_blocksize: (i + 1) * diagonal_blocksize,
            i * diagonal_blocksize: (i + 1) * diagonal_blocksize,
        ] = rc * xp.random.rand(diagonal_blocksize, diagonal_blocksize) + rc * xp.eye(
            diagonal_blocksize
        )

        # Fill the off-diagonal blocks
        if i > 0:
            DD_BTA[
                i * diagonal_blocksize: (i + 1) * diagonal_blocksize,
                (i - 1) * diagonal_blocksize: i * diagonal_blocksize,
            ] = rc * xp.random.rand(diagonal_blocksize, diagonal_blocksize)

        if i < n_diag_blocks - 1:
            DD_BTA[
                i * diagonal_blocksize: (i + 1) * diagonal_blocksize,
                (i + 1) * diagonal_blocksize: (i + 2) * diagonal_blocksize,
            ] = rc * xp.random.rand(diagonal_blocksize, diagonal_blocksize)

    # Make the matrix diagonally dominant
    for i in range(DD_BTA.shape[0]):
        DD_BTA[i, i] = 1 + xp.sum(DD_BTA[i, :])

    return DD_BTA


@pytest.fixture(scope="function", autouse=False)
def rand_bta(
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_diag_blocks: int,
    device_array: bool,
    dtype: np.dtype,
):
    """Returns a random, diagonaly dominant general, block tridiagonal arrowhead matrix."""
    xp = cp if device_array and CUPY_AVAIL else np

    RAND_BTA = xp.zeros(
        (
            diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
            diagonal_blocksize * n_diag_blocks + arrowhead_blocksize,
        ),
        dtype=dtype,
    )

    rc = (1.0 + 1.0j) if dtype == np.complex128 else 1.0

    # Fill the lower arrowhead blocks
    RAND_BTA[-arrowhead_blocksize:, :-arrowhead_blocksize] = rc * xp.random.rand(
        arrowhead_blocksize, n_diag_blocks * diagonal_blocksize
    )
    # Fill the right arrowhead blocks
    RAND_BTA[:-arrowhead_blocksize, -arrowhead_blocksize:] = rc * xp.random.rand(
        n_diag_blocks * diagonal_blocksize, arrowhead_blocksize
    )

    # Fill the tip of the arrowhead
    RAND_BTA[-arrowhead_blocksize:, -arrowhead_blocksize:] = rc * xp.random.rand(
        arrowhead_blocksize, arrowhead_blocksize
    )

    # Fill the diagonal blocks
    for i in range(n_diag_blocks):
        RAND_BTA[
            i * diagonal_blocksize: (i + 1) * diagonal_blocksize,
            i * diagonal_blocksize: (i + 1) * diagonal_blocksize,
        ] = rc * xp.random.rand(diagonal_blocksize, diagonal_blocksize) + rc * xp.eye(
            diagonal_blocksize
        )

        # Fill the off-diagonal blocks
        if i > 0:
            RAND_BTA[
                i * diagonal_blocksize: (i + 1) * diagonal_blocksize,
                (i - 1) * diagonal_blocksize: i * diagonal_blocksize,
            ] = rc * xp.random.rand(diagonal_blocksize, diagonal_blocksize)

        if i < n_diag_blocks - 1:
            RAND_BTA[
                i * diagonal_blocksize: (i + 1) * diagonal_blocksize,
                (i + 1) * diagonal_blocksize: (i + 2) * diagonal_blocksize,
            ] = rc * xp.random.rand(diagonal_blocksize, diagonal_blocksize)

    return RAND_BTA


@pytest.fixture(scope="function", autouse=False)
def b_rhs(
    n_rhs: int,
    diagonal_blocksize: int,
    arrowhead_blocksize: int,
    n_diag_blocks: int,
    device_array: bool,
    dtype: np.dtype,
):
    """Returns a random right-hand side."""
    xp = cp if device_array and CUPY_AVAIL else np

    rc = (1.0 + 1.0j) if dtype == np.complex128 else 1.0

    B = rc * xp.random.rand(
        diagonal_blocksize * n_diag_blocks + arrowhead_blocksize, n_rhs
    )

    return B


@pytest.fixture(scope="function", autouse=False)
def dd_ba():
    def dd_ba_(
        n_offdiags: int,
        arrowhead_size: int,
        n: int,
        dtype: np.dtype,
    ):
        """Returns a random, diagonaly dominant general, banded arrowhead matrix in
        compressed format."""

        xp = np
        rc = (1.0 + 1.0j) if dtype == np.complex128 else 1.0
        n -= arrowhead_size

        # Declare variables
        A_diagonal = xp.zeros(n, dtype=dtype)
        A_lower_diagonals = xp.zeros((n_offdiags, n-1), dtype=dtype)
        A_arrow_bottom = xp.zeros((arrowhead_size, n), dtype=dtype)
        A_arrow_tip = xp.zeros((arrowhead_size, arrowhead_size), dtype=dtype)

        # Fill with random values
        A_diagonal[:] = (rc * xp.random.rand(*A_diagonal.shape)+1)/2
        A_lower_diagonals[:, :] = (
            rc * xp.random.rand(*A_lower_diagonals.shape)+1)/2
        A_arrow_bottom[:, :] = (rc * xp.random.rand(*A_arrow_bottom.shape)+1)/2
        A_arrow_tip[:, :] = (rc * xp.random.rand(*A_arrow_tip.shape)+1)/2

        # Make diagonally dominant
        for i in range(n):
            A_diagonal[i] = (1 + xp.sum(A_arrow_bottom[:, i]))*2

        for i in range(arrowhead_size):
            A_arrow_tip[i, i] = (1 + xp.sum(A_arrow_bottom[:, i]))*2

        # Remove extra info
        A_lower_diagonals[-n_offdiags:, -n_offdiags:] = np.fliplr(
            np.triu(np.fliplr(A_lower_diagonals[-n_offdiags:, -n_offdiags:])))

        A_arrow_tip[:, :] = np.tril(A_arrow_tip[:, :])

        return (A_diagonal,
                A_lower_diagonals,
                A_arrow_bottom,
                A_arrow_tip)
    return dd_ba_


@pytest.fixture(scope="function", autouse=False)
def compress_ba():
    def compress_ba_(
            M: ArrayLike,
            n_offdiags: int,
            arrowhead_size: int
    ):
        """
        Compress a square matrix with banded and arrowhead structure 
        into a more efficient representation.

        The function handles matrices that have:
        1. A main band around the diagonal with specified bandwidth
        2. An arrowhead pattern in the last few rows and columns
        """

        n = M.shape[0] - arrowhead_size

        # Initialize compressed storage arrays
        M_diagonal = np.zeros(n)
        M_lower_diagonals = np.zeros((n_offdiags, n-1))
        M_arrow_bottom = np.zeros((arrowhead_size, n))
        M_arrow_tip = np.zeros((arrowhead_size, arrowhead_size))

        # Retrieve info for arrowhead
        M_arrow_bottom[:, :] = M[-arrowhead_size:, :-arrowhead_size]
        M_arrow_tip[:, :] = np.tril(M[-arrowhead_size:, -arrowhead_size:])

        # Compress the banded portion
        for i in range(n-1):
            M_diagonal[i] = M[i, i]

            j = min(n_offdiags, n-i-1)
            M_lower_diagonals[:j, i] = M[i+1:i+j+1, i]

        M_diagonal[n-1] = M[n-1, n-1]

        return (M_diagonal,
                M_lower_diagonals,
                M_arrow_bottom,
                M_arrow_tip)
    return compress_ba_

@pytest.fixture(scope="function", autouse=False)
def decompress_ba():
    def decompress_ba_(
        M_diagonal: ArrayLike,
        M_lower_diagonals: ArrayLike,
        M_arrow_bottom: ArrayLike,
        M_arrow_tip: ArrayLike,
        symmetric: bool = True
    ) -> ArrayLike:
        """
        Create dense n-banded arrowhead matrix based on compressed data format.
        """
        # Arrow height, Total matrix dimension (N = a + n)
        n_offdiags = M_lower_diagonals.shape[0]
        n = M_diagonal.shape[0]
        arrowhead_size = M_arrow_tip.shape[0]
        N = n + arrowhead_size

        # Initialize output matrix
        M = np.zeros((N, N))

        # Reinsert bandwidth portion
        for i in range(n-1):
            M[i, i] = M_diagonal[i]

            j = min(n_offdiags, n-i-1)
            M[i+1:i+j+1, i] = M_lower_diagonals[:j, i]

        M[n-1, n-1] = M_diagonal[n-1]

        # Reinsert arrow dense matrix
        M[-arrowhead_size:, :-arrowhead_size] = M_arrow_bottom[:, :]
        M[-arrowhead_size:, -arrowhead_size:] = M_arrow_tip[:, :]

        # Symmetrize
        M = np.tril(M)
        if symmetric:
            M += M.T
            M -= np.diag(np.diag(M))/2

        return M
    return decompress_ba_
