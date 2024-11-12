import numpy as np


def extract_diagonals(matrix, start_row=0, end_row=0):
    """
    Extract diagonals from a matrix using np.diagonal() with offset.

    Parameters:
    matrix : numpy.ndarray
        Input matrix
    start_row : int
        Starting row index for diagonal extraction
    end_row : int
        Ending row index for diagonal extraction

    Returns:
    numpy.ndarray
        Matrix containing extracted diagonals
    """
    n = matrix.shape[1]
    result = np.zeros((matrix.shape[0], matrix.shape[1]))

    if not matrix.shape[1]:
        return result

    # Create intermediate array for storing diagonals
    diags = np.zeros((end_row-start_row, n))

    # Extract diagonals using diagonal() with offset
    for i in range(end_row-start_row):
        diagonal = matrix.diagonal(offset=-i-start_row)
        diags[i, :len(diagonal)] = diagonal

    # Place diagonals in final matrix
    for i in range(n):
        n_ = result[:end_row-start_row+1, i].shape[0]
        result[:end_row-start_row, i] = diags[:, i][:n_]

    return result

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

        # Extract diagonal elements for column compressed storage
        diag_elements = extract_diagonals(
            np.fliplr(A_flattened_cols[1:, start_idx:col_idx]),
            start_row=1,
            end_row=bandwidth
        )

        # Compute column elements within bandwidth
        A_flattened_cols[1:, col_idx] = (
            M_flattened_cols[1:, col_idx] -
            np.matmul(prev_elements.conj(), np.fliplr(diag_elements).T)
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
