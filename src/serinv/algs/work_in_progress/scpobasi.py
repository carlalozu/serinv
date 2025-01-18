import numpy as np
import scipy.linalg as la
from numpy.typing import ArrayLike


def scpobasi(
    L_diagonal: ArrayLike,
    L_lower_diagonals: ArrayLike,
    L_arrow_bottom: ArrayLike,
    L_arrow_tip: ArrayLike,
    overwrite: bool = False
) -> ArrayLike:
    """Performs the selected inversion of a banded arrowhead matrix given
      its Cholesky factor L in compressed format. Sequential algorithm on CPU backend.

    Parameters
    ----------
    L_diagonal : ArrayLike
        The main diagonal of the lower Cholesky factor.
    L_lower_diagonals : ArrayLike
        The banded part of the lower factor in flattened column format.
    L_arrow_bottom : ArrayLike
        Lower banded arrow part of the Cholesky factor.
    L_arrow_tip : ArrayLike
        The arrow tip of the lower Cholesky factor.
    overwrite : bool, optional
        If True, overwrite the input arrays with the result. Default is False.
        Diagonal of the inverse.

    Returns
    -------
    ArrayLike
        Diagonal of the inverse
    """

    n_offdiags = L_lower_diagonals.shape[0]
    matrix_size = L_diagonal.shape[0]
    arrowhead_size = L_arrow_tip.shape[0]

    # Initialize result matrices
    if overwrite:
        X_diagonal = L_diagonal
        X_lower_diagonals = L_lower_diagonals
        X_arrow_bottom = L_arrow_bottom
        X_arrow_tip = L_arrow_tip
    else:
        X_diagonal = np.copy(L_diagonal)
        X_lower_diagonals = np.copy(L_lower_diagonals)
        X_arrow_bottom = np.copy(L_arrow_bottom)
        X_arrow_tip = np.copy(L_arrow_tip)

    # Arrowhead inversion first
    inv_L_Dndb1 = la.solve_triangular(
        # L_{ndb+1, ndb+1}^{-1}
        L_arrow_tip[:, :], np.eye(arrowhead_size), lower=True)

    inv_L_Dndb = 1/X_diagonal[-1]  # # L_{ndb, ndb}^{-1}
    L_Fndb = np.copy(X_arrow_bottom[:, -1])  # L_{ndb+1, ndb}

    # X_{ndb+1, ndb+1}
    X_arrow_tip[:, :] = inv_L_Dndb1.conj().T @ inv_L_Dndb1

    # X_{ndb+1,ndb}
    X_arrow_bottom[:, -1] = - X_arrow_tip[:, :] @ L_Fndb * inv_L_Dndb
    # X_{ndb,ndb}
    X_diagonal[-1] = inv_L_Dndb**2 - \
        L_Fndb.conj().T @ X_arrow_bottom[:, -1] * inv_L_Dndb

    X_i1i1 = np.zeros((n_offdiags, n_offdiags))
    X_i1i1[0, 0] = X_diagonal[-1]

    # Rest of the matrix
    for i in range(2, matrix_size+1):

        # Adjust for the size of the block E, under the diagonal
        tail = min(i - 1, n_offdiags)

        # Inverse of the L diagonal value i, L_{i, i}^{-1}
        iL_Di = 1/X_diagonal[matrix_size-i]
        # L arrow bottom slice i, L_{ndb+1, i}
        L_Fi = np.copy(X_arrow_bottom[:, matrix_size-i])
        # L lower diagonal slice, L_{i+1, i}
        L_Ei = np.copy(X_lower_diagonals[:tail, matrix_size-i])

        # X arrow bottom slice i+i, X_{ndb+1, i+1}
        X_ndb1_i1 = X_arrow_bottom[:, matrix_size-i+1:matrix_size-i+1+tail]

        # --- Off-diagonal slice part ---
        # X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} -
        #              X_{ndb+1, i+1}^{T} L_{ndb+1, i}) L_{i, i}^{-1}
        X_i1_i = - (X_i1i1[:tail, :tail] @ L_Ei +
                    X_ndb1_i1.conj().T @ L_Fi) * iL_Di
        X_lower_diagonals[:tail, matrix_size-i] = X_i1_i

        X_i1i1[1:, 1:] = X_i1i1[:-1, :-1]
        X_i1i1[1:min(i, n_offdiags), 0] = X_i1_i[:min(i, n_offdiags-1)]
        X_i1i1[0, 1:min(i, n_offdiags)] = X_i1_i[:min(i, n_offdiags-1)]

        # --- Arrowhead part ---
        # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} -
        #                X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X_arrow_bottom[:, matrix_size-i] = - \
            (X_ndb1_i1 @ L_Ei + X_arrow_tip[:, :] @ L_Fi) * iL_Di

        # --- Diagonal value part ---
        # X_{i, i} = (L_{i, i}^{-T} - X_{i+1, i}^{T} L_{i+1, i} -
        #            X_{ndb+1, i}.conj().T L_{ndb+1, i}) L_{i, i}^{-1}
        X_diagonal[matrix_size-i] = (
            iL_Di.conj().T - X_i1_i.conj().T @ L_Ei -
            X_arrow_bottom[:, matrix_size-i].conj().T @ L_Fi
        ) * iL_Di
        X_i1i1[0, 0] = X_diagonal[matrix_size-i]

    return np.concat([X_diagonal, np.diag(X_arrow_tip)])
