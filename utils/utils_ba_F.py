"""Functions for the cholesky decomposition of a spd matrix algorithm in compressed
format.

The compressed representation is a tuple of the following arrays:
    - M_diagonal: 1D array of shape (n)
    - M_lower_diagonals: 2D array of shape (n_offdiags, n-1)
    - M_arrow_bottom: 2D array of shape (arrowhead_size, n)
    - M_arrow_tip: 2D array of shape (arrowhead_size, arrowhead_size)

Where the following parameters are defined:
    - n is the number of diagonal elements, without the arrowhead portion.
    - n_offdiags is the number of off-diagonals in the block banded
    structure. Total number of diagonals is n_offdiags*2+1.
    - arrowhead_size is the size of the arrowhead blocks.
    - N is the total number of rows in the matrix, N = n + arrowhead_size.

"""
import numpy as np
from numpy.typing import ArrayLike

SEED = 63

try:
    import cupy as cp

    CUPY_AVAIL = True
    cp.random.seed(cp.uint64(SEED))

except ImportError:
    CUPY_AVAIL = False

def dd_ba(
    n_offdiags: int,
    arrowhead_size: int,
    n: int,
    dtype: np.dtype,
    factor: float = 2,
):
    """Returns a random, diagonaly dominant general, banded arrowhead matrix in
    compressed format."""

    xp = cp if CUPY_AVAIL else np
    rc = (1.0 + 1.0j) if dtype == np.complex128 else 1.0
    n -= arrowhead_size

    # Declare variables
    A_diagonal = xp.zeros(n, dtype=dtype, order='F')
    A_lower_diagonals = xp.zeros((n_offdiags, n-1), dtype=dtype, order='F')
    A_arrow_bottom = xp.zeros((arrowhead_size, n), dtype=dtype, order='F')
    A_arrow_tip = xp.zeros((arrowhead_size, arrowhead_size), dtype=dtype, order='F')

    # Fill with random values
    A_diagonal[:] = (rc * xp.random.rand(*A_diagonal.shape)+1)/2
    A_lower_diagonals[:, :] = (
        rc * xp.random.rand(*A_lower_diagonals.shape)+1)/2
    A_arrow_bottom[:, :] = (rc * xp.random.rand(*A_arrow_bottom.shape)+1)/2
    A_arrow_tip[:, :] = (rc * xp.random.rand(*A_arrow_tip.shape)+1)/2

    # Make diagonally dominant
    for i in range(n):
        A_diagonal[i] = (1 + xp.sum(A_arrow_bottom[:, i]))*factor
    A_diagonal[:] = (A_diagonal[:] + A_diagonal[:].conj())/2

    for i in range(arrowhead_size):
        A_arrow_tip[i, i] = (1 + xp.sum(A_arrow_bottom[:, i]))*factor

    # Remove extra info
    A_lower_diagonals[-n_offdiags:, -n_offdiags:] = xp.fliplr(
        xp.triu(xp.fliplr(A_lower_diagonals[-n_offdiags:, -n_offdiags:])))

    A_arrow_tip[:, :] = xp.tril(
        A_arrow_tip[:, :] + A_arrow_tip[:, :].conj().T)/2

    return (A_diagonal,
            A_lower_diagonals,
            A_arrow_bottom,
            A_arrow_tip)


def ba_dense_to_arrays(
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
    xp = cp if CUPY_AVAIL else np

    n = M.shape[0] - arrowhead_size

    # Initialize compressed storage arrays
    M_diagonal = xp.zeros(n, dtype=M.dtype, order='F')
    M_lower_diagonals = xp.zeros((n_offdiags, n-1), dtype=M.dtype, order='F')
    M_arrow_bottom = xp.zeros((arrowhead_size, n), dtype=M.dtype, order='F')
    M_arrow_tip = xp.zeros((arrowhead_size, arrowhead_size), dtype=M.dtype, order='F')

    # Retrieve info for arrowhead
    M_arrow_bottom[:, :] = M[-arrowhead_size:, :-arrowhead_size]
    M_arrow_tip[:, :] = xp.tril(M[-arrowhead_size:, -arrowhead_size:])

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


def ba_arrays_to_dense(
    M_diagonal: ArrayLike,
    M_lower_diagonals: ArrayLike,
    M_arrow_bottom: ArrayLike,
    M_arrow_tip: ArrayLike,
    symmetric: bool = True
) -> ArrayLike:
    """
    Create dense n-banded arrowhead matrix based on compressed data format.
    """
    xp = cp if CUPY_AVAIL else np

    # Arrow height, Total matrix dimension (N = a + n)
    n_offdiags = M_lower_diagonals.shape[0]
    n = M_diagonal.shape[0]
    arrowhead_size = M_arrow_tip.shape[0]
    N = n + arrowhead_size

    # Initialize output matrix
    M = xp.zeros((N, N), dtype=M_diagonal.dtype)
    # print(f"Creating new matrix of size ({N}, {N})")

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
    M = xp.tril(M)
    if symmetric:
        M += M.conj().T
        M -= xp.diag(xp.diag(M))/2

    return M