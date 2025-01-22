"""Tests for the selected inversion of a spd banded matrix in compressed format.

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
import pytest

from serinv.algs.work_in_progress.scpobaf import scpobaf
from serinv.algs.work_in_progress.scpobasi import scpobasi


@pytest.mark.mpi_skip()
@pytest.mark.parametrize("arrowhead_size", [2, 3])
@pytest.mark.parametrize("n_offdiags", [1, 2, 3])
@pytest.mark.parametrize("n", [5])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
@pytest.mark.parametrize("overwrite", [True, False])
def test_scpobasi(
    dd_ba,
    ba_arrays_to_dense,
    ba_dense_to_arrays,
    arrowhead_size,
    n_offdiags,
    n,
    dtype,
    overwrite
):
    (
        M_diagonal,
        M_lower_diagonals,
        M_arrow_bottom,
        M_arrow_tip
    ) = dd_ba(
        n_offdiags,
        arrowhead_size,
        n+arrowhead_size,
        dtype,
    )

    M = ba_arrays_to_dense(
        M_diagonal,
        M_lower_diagonals,
        M_arrow_bottom,
        M_arrow_tip,
        symmetric=True
    )

    I_ref = np.linalg.inv(np.copy(M))
    (
        I_ref_diagonal,
        I_ref_lower_diagonals,
        I_ref_arrow_bottom,
        I_ref_arrow_tip
    ) = ba_dense_to_arrays(
        I_ref,
        n_offdiags,
        arrowhead_size
    )

    (
        L_diagonal,
        L_lower_diagonals,
        L_arrow_bottom,
        L_arrow_tip
    ) = scpobaf(
        M_diagonal,
        M_lower_diagonals,
        M_arrow_bottom,
        M_arrow_tip,
        overwrite
    )

    (
        I_diagonal,
        I_lower_diagonals,
        I_arrow_bottom,
        I_arrow_tip
    ) = scpobasi(
        L_diagonal,
        L_lower_diagonals,
        L_arrow_bottom,
        L_arrow_tip,
        overwrite
    )

    assert np.allclose(I_ref_diagonal, I_diagonal)
    assert np.allclose(I_ref_lower_diagonals, I_lower_diagonals)
    assert np.allclose(I_ref_arrow_bottom, I_arrow_bottom)
    assert np.allclose(I_ref_arrow_tip, np.tril(I_arrow_tip))

    assert (I_diagonal.ctypes.data == M_diagonal.ctypes.data) is overwrite
    assert (I_lower_diagonals.ctypes.data == M_lower_diagonals.ctypes.data) is overwrite
    assert (I_arrow_bottom.ctypes.data == M_arrow_bottom.ctypes.data) is overwrite
    assert (I_arrow_tip.ctypes.data == M_arrow_tip.ctypes.data) is overwrite
