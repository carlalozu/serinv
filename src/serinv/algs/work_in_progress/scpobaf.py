try:
    import cupy as cp
    from serinv.cupyfix.cholesky_lowerfill import cholesky_lowerfill

    CUPY_AVAIL = True

except ImportError:
    CUPY_AVAIL = False

import numpy as np
from numpy.typing import ArrayLike


if CUPY_AVAIL:
    xp = cp
    cholesky = cholesky_lowerfill
else:
    xp = np
    cholesky = np.linalg.cholesky


def scpobaf(
    A_diagonal: ArrayLike,
    A_lower_diagonals: ArrayLike,
    A_arrow_bottom: ArrayLike,
    A_arrow_tip: ArrayLike,
    overwrite: bool = False
) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Performs Cholesky factorization of a banded arrowhead matrix.

    Parameters
    ----------
    A_diagonal : ArrayLike
        The diagonal elements of the matrix.
    A_lower_diagonals : ArrayLike
        The lower diagonals of the banded part of the matrix in flattened column
        format.
    A_arrow_bottom : ArrayLike
        The bottom part of the arrow in the matrix in flattened column format.
    A_arrow_tip : ArrayLike
        The tip of the arrow in the matrix.
    overwrite : bool, optional
        If True, the input arrays will be overwritten with the result. Default
        is False.

    Returns
    -------
    tuple
        A tuple containing four elements:
        - L_diagonal (ArrayLike): The diagonal elements of the lower triangular
        matrix.
        - L_lower_diagonals (ArrayLike): The lower diagonals of the lower
        triangular matrix.
        - L_arrow_bottom (ArrayLike): The bottom part of the arrow in the lower
        triangular matrix.
        - L_arrow_tip (ArrayLike): The tip of the arrow in the lower triangular
        matrix.
    """
    n_diagonals = A_diagonal.shape[0]
    n_offdiags = A_lower_diagonals.shape[0]

    # Initialize result matrices
    if overwrite:
        L_diagonal = A_diagonal
        L_lower_diagonals = A_lower_diagonals
        L_arrow_bottom = A_arrow_bottom
        L_arrow_tip = A_arrow_tip
    else:
        L_diagonal = xp.copy(A_diagonal)
        L_lower_diagonals = xp.copy(A_lower_diagonals)
        L_arrow_bottom = xp.copy(A_arrow_bottom)
        L_arrow_tip = xp.copy(A_arrow_tip)

    L_i1i1 = xp.zeros((n_offdiags, n_offdiags), dtype=L_diagonal.dtype)
    arrow, _ = L_arrow_bottom.shape
    # Process banded part of the matrix
    FLOPS:int = 0
    for i in range(n_diagonals-1):

        # L_{i, i} = chol(A_{i, i})
        L_diagonal[i] = xp.sqrt(L_diagonal[i])
        FLOPS += 1  # 1 sqrt

        inv_L_ii = 1 / L_diagonal[i].conj()
        FLOPS += 1 # 1 div

        # Update column i of the lower diagonals
        L_lower_diagonals[:-1, i] -= L_i1i1[:, 0].conj().T @ L_i1i1[:, 1:]
        FLOPS += 2 * n_offdiags*(n_offdiags-1)  # matrix product, 1 mult, 1 add

        # L_{i+1, i} = A_{i+1, i} @ L_{i, i}^{-T}
        L_lower_diagonals[:, i] *= inv_L_ii
        FLOPS += n_offdiags  # vector scaling ns

        # data movement, no computational FLOPS
        L_i1i1[:-1, :-1] = L_i1i1[1:, 1:]
        L_i1i1[-1, :] = L_lower_diagonals[:, i]

        # L_{ndb+1, i} = A_{ndb+1, i} @ L_{i, i}^{-T}
        L_arrow_bottom[:, i] *= inv_L_ii
        FLOPS += arrow  # vector scaling nb

        # A_{ndb+1, i+1} = A_{ndb+1, i+1} - L_{ndb+1, i} @ L_{i+1, i}.conj().T
        L_arrow_bottom[:, i+1] -= L_arrow_bottom[:, max(0, i-n_offdiags+1):i+1] @ \
            L_i1i1[max(-n_offdiags, -i)-1:, 0].conj().T
        a, b = L_arrow_bottom[:, max(0, i-n_offdiags+1):i+1].shape
        # FLOPS += 2 * arrow * n_offdiags  # matrix times vector (ns-1)*ns
        FLOPS += 2 * a * b  # matrix times vector (ns-1)*ns

        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ L_{ndb+1, i}.conj().T
        L_arrow_tip[:, :] -= L_arrow_bottom[:, i:i + 1] @ L_arrow_bottom[:, i:i+1].conj().T
        FLOPS += 2 * arrow**2  # outer product, one mult one add

        # Update next diagonal
        # A_{i+1, i+1} = A_{i+1, i+1} - L_{i+1, i} @ L_{i+1, i}.conj().T
        L_diagonal[i+1] -= L_i1i1[:, 0] @ L_i1i1[:, 0].conj().T
        FLOPS += 2 * n_offdiags  # dot product

    # L_{ndb, ndb} = chol(A_{ndb, ndb})
    L_diagonal[-1] = xp.sqrt(L_diagonal[-1])
    FLOPS += 1  # 1 sqrt

    # L_{ndb+1, ndb} = A_{ndb+1, ndb} @ L_{ndb, ndb}^{-T}
    L_arrow_bottom[:, -1] *= (1/L_diagonal[-1].conj())
    FLOPS += 1 # div
    FLOPS += arrow  # vector scaling nb

    # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, ndb} @ L_{ndb+1, ndb}^{T}
    L_arrow_tip[:, :] -= L_arrow_bottom[:, - 1:] @ L_arrow_bottom[:, -1:].conj().T
    FLOPS += 2 * arrow**2  # outer product, one mult one add

    # L_{ndb+1, ndb+1} = chol(A_{ndb+1, ndb+1})
    L_arrow_tip[:, :] = cholesky(L_arrow_tip[:, :])
    FLOPS += 1/3*arrow**3 + 1/2*arrow**2 + 1/6*arrow  # cholesky

    print(FLOPS)

    return L_diagonal, L_lower_diagonals, L_arrow_bottom, L_arrow_tip
