import numpy as np


def compressed_to_banded(
        M_flatten_cols: np.ndarray,
        M_banded: np.ndarray,
):
    """
    Convert a flattened banded matrix to a dense banded matrix.

    Parameters
    ----------
    M_flatten_cols : np.ndarray
        A 2D numpy array containing the flattened banded matrix columns.
    M_banded : np.ndarray
        The dense banded matrix, in-place.
    """
    # Get dimensions
    # Column bandwidth, Inner matrix dimension
    b, n = M_flatten_cols.shape
    k, N = M_banded.shape
    assert n == N

    # Reinsert flattened columns to dense matrix
    for i in range(n):
        n_ = min(b, k - i)  # Number of elements that fit in this column
        M_banded[i:i+n_, i] = M_flatten_cols[:n_, i]


def scpobaf(
        M_flattened_cols: np.ndarray,
        M_arrow: np.ndarray
) -> tuple:
    """Performs Cholesky factorization of a banded arrowhead matrix.

    Parameters
    ----------
    M_flattened_cols : np.ndarray
        The banded part of the matrix in flattened column format
    M_arrow : np.ndarray
        The arrow part of the matrix in flattened column format

    Returns
    -------
    tuple(np.ndarray, np.ndarray)
        Lower triangular matrices (band and arrow parts) such that A = LL^T
    """
    bandwidth = M_flattened_cols.shape[0] - 1
    arrow_size = M_arrow.shape[0]
    matrix_size = M_flattened_cols.shape[1]

    # Initialize result matrices
    A_flattened_cols = np.zeros(M_flattened_cols.shape)
    A_arrow = np.zeros(M_arrow.shape)

    A_decompressed = np.zeros((bandwidth, bandwidth))
    # Process banded part of the matrix
    for col_idx in range(matrix_size):
        # Define the starting index for the current column
        start_idx = max(col_idx - bandwidth, 0)

        # Extract previous elements needed for computation
        prev_elements = np.flip(
            np.diag(np.fliplr(A_flattened_cols[1:, start_idx:col_idx])))

        # Compute diagonal element
        A_flattened_cols[0, col_idx] = np.sqrt(
            M_flattened_cols[0, col_idx] -
            np.dot(prev_elements, prev_elements.conj())
        )

        A_decompressed.fill(0.0)
        compressed_to_banded(
            np.flipud(np.fliplr(A_flattened_cols[1:, start_idx:col_idx])),
            A_decompressed[1:, :col_idx-start_idx]
)

        # Compute column elements within bandwidth
        A_flattened_cols[1:, col_idx] = (
            M_flattened_cols[1:, col_idx] -
            np.matmul(prev_elements.conj(), np.fliplr(
                np.flipud(A_decompressed[:, :col_idx-start_idx])).T)
        ) / A_flattened_cols[0, col_idx]

        # Compute arrow part
        A_arrow[:, col_idx] = (
            M_arrow[:, col_idx] -
            np.matmul(prev_elements.conj(),
                      A_arrow[:, start_idx:col_idx].T)
        ) / A_flattened_cols[0, col_idx]

    # Process arrow part
    for arrow_idx in range(arrow_size - 1):
        # Compute diagonal elements of arrow part
        A_arrow[arrow_idx, matrix_size + arrow_idx] = np.sqrt(
            M_arrow[arrow_idx, matrix_size + arrow_idx] -
            np.dot(A_arrow[arrow_idx, :matrix_size + arrow_idx],
                   A_arrow[arrow_idx, :matrix_size + arrow_idx])
        )

        # Compute off-diagonal elements of arrow part
        A_arrow[arrow_idx + 1:, matrix_size + arrow_idx] = (
            M_arrow[arrow_idx + 1:, matrix_size + arrow_idx] -
            np.matmul(A_arrow[arrow_idx, :matrix_size + arrow_idx].conj(),
                      A_arrow[arrow_idx + 1:, :matrix_size + arrow_idx].T)
        ) / A_arrow[arrow_idx, matrix_size + arrow_idx]

    # Compute final diagonal element
    A_arrow[-1, -1] = np.sqrt(
        M_arrow[-1, -1] -
        np.dot(A_arrow[-1, :], A_arrow[-1, :].conj())
    )

    return (A_flattened_cols, A_arrow)
