import numpy as np
import scipy.linalg as la


def compressed_to_banded(
        M_flatten_cols: np.ndarray,
        M_banded: np.ndarray,
        symmetric: bool = False) -> np.ndarray:
    """
    Convert a flattened banded matrix to a dense banded matrix.

    Parameters
    ----------
    M_flatten_cols : np.ndarray
        A 2D numpy array containing the flattened banded matrix columns.
    symmetric : bool
        Indicate if the input matrix is symmetric. Default is False.

    Returns
    -------
    np.ndarray
        The dense banded matrix.
    """
    # Get dimensions
    column_bandwidth, inner_dim = M_flatten_cols.shape

    # Reinsert flattened columns to dense matrix
    for i in range(inner_dim):
        num_elements = min(column_bandwidth, inner_dim - i)
        M_banded[i:i+num_elements, i] = M_flatten_cols[:num_elements, i]

        if symmetric:
            M_banded[i, i:i+num_elements] = M_flatten_cols[:num_elements, i]

    return M_banded


def scpobasi(L_flatten_cols: np.ndarray, L_flatten_arrow: int) -> np.ndarray:
    """Perform the selected inversion of a banded arrowhead matrix given its Cholesky
    factor L. Sequential algorithm on CPU backend

    Parameters
    ----------
    L_flatten_cols : np.ndarray
        The banded part of the lower factor in flattened column format
    L_flatten_arrow : np.ndarray
        The arrow part of the matrix in flattened column format

    Returns
    -------
    np.ndarray
        Diagonal of the inverse
    """
    # Column bandwidth, Inner matrix dimension
    b, n = L_flatten_cols.shape
    # Arrow height, Total matrix dimension (N = a + n)
    a, _ = L_flatten_arrow.shape

    b -= 1

    # Initialize result matrices
    A_flatten_cols = np.zeros(L_flatten_cols.shape)
    A_flatten_arrow = np.zeros(L_flatten_arrow.shape)

    # Arrowhead inversion first
    inv_L_Dndb1 = la.solve_triangular(
        # L_{ndb+1, ndb+1}^{-1}, size axa
        L_flatten_arrow[:, -a:], np.eye(a), lower=True)

    L_Fndb = L_flatten_arrow[:, -a-1]  # L_{ndb+1, ndb}, size ax1
    inv_L_Dndb = 1/(L_flatten_cols[0, -1])  # # L_{ndb, ndb}^{-1}, size 1

    # X_{ndb+1, ndb+1}, size axa
    A_flatten_arrow[:, -a:] = inv_L_Dndb1.conj().T @ inv_L_Dndb1

    # X_{ndb+1,ndb}, size ax1
    A_flatten_arrow[:, -a-1] = - A_flatten_arrow[:, -a:] @ L_Fndb * inv_L_Dndb
    # X_{ndb,ndb}, size 1
    A_flatten_cols[0, -1] = inv_L_Dndb**2 - L_Fndb.conj().T @ A_flatten_arrow[:, -a-1] * \
        inv_L_Dndb

    X_i1i1 = np.zeros((b, b))
    # Rest of the matrix
    for i in range(2, n+1):

        # Adjust for the size of the block E, under the diagonal
        tail = min(i - 1, b)

        # Inverse of the L diagonal value i, L_{i, i}^{-1}, size 1
        iL_Di = 1/L_flatten_cols[0, -i]
        # L arrow bottom slice i, L_{ndb+1, i}, size ax1
        L_Fi = L_flatten_arrow[:, -a-i]
        # L lower diagonal slice, L_{i+1, i}, size bx1 in most cases
        L_Ei = L_flatten_cols[1:tail+1, -i]

        X_i1i1.fill(0.08)
        # X diagonal block i+1, X_{i+1, i+1}, size bxb
        compressed_to_banded(
            A_flatten_cols[:tail, n-i+1:n-i+1+tail],
            X_i1i1,
            symmetric=True
        )

        # X arrow bottom slice i+i, X_{ndb+1, i+1}, size axb
        X_ndb1_i1 = A_flatten_arrow[:, -i-a+1:-i-a+1+tail]

        # --- Off-diagonal slice part ---
        # X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} -
        #              X_{ndb+1, i+1}^{T} L_{ndb+1, i}) L_{i, i}^{-1}
        # size dx1 in most cases
        X_i1_i = - ( X_i1i1[:tail, :tail] @ L_Ei + X_ndb1_i1.conj().T @ L_Fi) * iL_Di
        A_flatten_cols[1:tail+1, -i] = X_i1_i

        # --- Arrowhead part ---
        # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} -
        #                X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        # size ax1
        A_flatten_arrow[:, -i-a] = - \
            (X_ndb1_i1 @ L_Ei + A_flatten_arrow[:, -a:] @ L_Fi) * iL_Di

        # --- Diagonal value part ---
        # X_{i, i} = (L_{i, i}^{-T} - X_{i+1, i}^{T} L_{i+1, i} -
        #            X_{ndb+1, i}.conj().T L_{ndb+1, i}) L_{i, i}^{-1}
        # size 1
        A_flatten_cols[0, -i] = (iL_Di.conj().T - X_i1_i.conj().T @ L_Ei -
                                 A_flatten_arrow[:, -i-a].conj().T @ L_Fi) * iL_Di

    return np.concatenate([A_flatten_cols[0, :], np.diag(A_flatten_arrow[:, -a:])])
