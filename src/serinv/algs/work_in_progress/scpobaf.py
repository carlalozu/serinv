import numpy as np
from numpy.typing import ArrayLike


def scpobaf(
    A_diagonal: ArrayLike,
    A_lower_diagonals: ArrayLike,
    A_arrow_bottom: ArrayLike,
    A_arrow_tip: ArrayLike,
    overwrite: bool = False
) -> tuple:
    """Performs Cholesky factorization of a banded arrowhead matrix.

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
    n_offdiags = A_lower_diagonals.shape[0]
    matrix_size = A_diagonal.shape[0]
    arrowhead_size = A_arrow_tip.shape[0]

    # Initialize result matrices
    if overwrite:
        L_diagonal = A_diagonal
        L_lower_diagonals = A_lower_diagonals
        L_arrow_bottom = A_arrow_bottom
        L_arrow_tip = A_arrow_tip
    else:
        L_diagonal = np.copy(A_diagonal)
        L_lower_diagonals = np.copy(A_lower_diagonals)
        L_arrow_bottom = np.copy(A_arrow_bottom)
        L_arrow_tip = np.copy(A_arrow_tip)

    L_temp = np.zeros((n_offdiags+1, n_offdiags+1), dtype=A_diagonal.dtype)

    # Process banded part of the matrix
    for col_idx in range(matrix_size):
        # Define the starting index for the current column
        start_idx = max(col_idx - n_offdiags, 0)

        # Extract previous elements needed for computation
        prev_elements = np.copy(L_temp[0, max(n_offdiags-col_idx+1, 1):])

        # Compute diagonal element
        L_diagonal[col_idx] = np.sqrt(
            L_diagonal[col_idx] -
            np.dot(prev_elements, prev_elements.conj())
        )

        # Compute arrow part
        L_arrow_bottom[:, col_idx] = (
            L_arrow_bottom[:, col_idx] -
            np.matmul(prev_elements.conj(),
                      L_arrow_bottom[:, start_idx:col_idx].T)
        ) / L_diagonal[col_idx]

        if col_idx == matrix_size-1:
            break

        # Compute column elements within bandwidth
        L_lower_diagonals[:, col_idx] = (
            L_lower_diagonals[:, col_idx] -
            np.matmul(
                prev_elements.conj(),
                L_temp[1:, max(n_offdiags-col_idx+1, 1):].T
            )
        ) / L_diagonal[col_idx]

        # Change this last because prev_elements is referencing L_temp
        L_temp[:-1, :-1] = L_temp[1:, 1:]
        L_temp[:-1, -1] = L_lower_diagonals[:, col_idx]

    L_arrow_dot = np.sum(L_arrow_bottom*L_arrow_bottom.conj(), axis=1)
    # Process arrow part
    for arrow_idx in range(arrowhead_size):
        # Compute diagonal elements of arrow part
        L_arrow_tip[arrow_idx, arrow_idx] = np.sqrt(
            L_arrow_tip[arrow_idx, arrow_idx] -
            L_arrow_dot[arrow_idx] -
            np.dot(L_arrow_tip[arrow_idx, :arrow_idx],
                   L_arrow_tip[arrow_idx, :arrow_idx].conj())
        )

        # Compute off-diagonal elements of arrow part
        L_arrow_tip[arrow_idx + 1:, arrow_idx] = (
            L_arrow_tip[arrow_idx + 1:, arrow_idx] -
            np.matmul(
                np.concat([L_arrow_bottom[arrow_idx, :],
                          L_arrow_tip[arrow_idx, :arrow_idx]]).conj(),
                np.concat([L_arrow_bottom[arrow_idx+1:, :],
                          L_arrow_tip[arrow_idx+1:, :arrow_idx]], axis=1).T
            )
        ) / L_arrow_tip[arrow_idx, arrow_idx]

    return L_diagonal, L_lower_diagonals, L_arrow_bottom, L_arrow_tip
