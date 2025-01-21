import numpy as np
import pytest

from serinv.algs.work_in_progress.scpobbaf import scpobbaf_c
from serinv.algs.work_in_progress.scpobbasi import scpobbasi_c


@pytest.mark.mpi_skip()
@pytest.mark.parametrize("diagonal_blocksize", [2, 3])
@pytest.mark.parametrize("arrowhead_blocksize", [2, 3])
@pytest.mark.parametrize("n_offdiags_blk", [1, 2, 3])
@pytest.mark.parametrize("n_diag_blocks", [5])
@pytest.mark.parametrize("dtype", [np.float64, np.complex128])
def test_scpobbasi(
    dd_bba,
    bba_arrays_to_dense,
    bba_dense_to_arrays,
    diagonal_blocksize,
    arrowhead_blocksize,
    n_offdiags_blk,
    n_diag_blocks,
    dtype,
):
    # Create matrix in compressed format
    (
        M_diagonal_blocks,
        M_lower_diagonal_blocks,
        M_arrow_bottom_blocks,
        M_arrow_tip_block
    ) = dd_bba(
        n_offdiags_blk,
        diagonal_blocksize,
        arrowhead_blocksize,
        n_diag_blocks,
        dtype,
    )

    M = bba_arrays_to_dense(
        M_diagonal_blocks,
        M_lower_diagonal_blocks,
        M_arrow_bottom_blocks,
        M_arrow_tip_block,
        symmetric=True
    )

    I_ref = np.linalg.inv(np.copy(M))
    (
        I_ref_diagonal_blocks,
        I_ref_lower_diagonal_blocks,
        I_ref_arrow_bottom_blocks,
        I_ref_arrow_tip_block
    ) = bba_dense_to_arrays(
        I_ref,
        n_offdiags_blk,
        diagonal_blocksize,
        arrowhead_blocksize,
        lower=False
    )

    (
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_arrow_tip_block
    ) = scpobbaf_c(
        M_diagonal_blocks,
        M_lower_diagonal_blocks,
        M_arrow_bottom_blocks,
        M_arrow_tip_block,
    )

    (
        I_diagonal_blocks,
        I_lower_diagonal_blocks,
        I_arrow_bottom_blocks,
        I_arrow_tip_block
    ) = scpobbasi_c(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_arrow_tip_block,
        overwrite=False
    )

    assert np.allclose(I_diagonal_blocks, I_ref_diagonal_blocks)
    assert np.allclose(I_lower_diagonal_blocks, I_ref_lower_diagonal_blocks)
    assert np.allclose(I_arrow_bottom_blocks, I_ref_arrow_bottom_blocks)
    assert np.allclose(I_arrow_tip_block, I_ref_arrow_tip_block)

    # Check that the results are not overwritten
    assert L_diagonal_blocks.ctypes.data != \
        I_diagonal_blocks.ctypes.data
    assert L_lower_diagonal_blocks.ctypes.data != \
        I_lower_diagonal_blocks.ctypes.data
    assert L_arrow_bottom_blocks.ctypes.data != \
        I_arrow_bottom_blocks.ctypes.data
    assert L_arrow_tip_block.ctypes.data != \
        I_arrow_tip_block.ctypes.data

    # Check overwrite option
    (
        I_diagonal_blocks,
        I_lower_diagonal_blocks,
        I_arrow_bottom_blocks,
        I_arrow_tip_block
    ) = scpobbasi_c(
        L_diagonal_blocks,
        L_lower_diagonal_blocks,
        L_arrow_bottom_blocks,
        L_arrow_tip_block,
        overwrite=True)

    # Check that the results are overwritten
    assert L_diagonal_blocks.ctypes.data == \
        I_diagonal_blocks.ctypes.data
    assert L_lower_diagonal_blocks.ctypes.data == \
        I_lower_diagonal_blocks.ctypes.data
    assert L_arrow_bottom_blocks.ctypes.data == \
        I_arrow_bottom_blocks.ctypes.data
    assert L_arrow_tip_block.ctypes.data == \
        I_arrow_tip_block.ctypes.data
