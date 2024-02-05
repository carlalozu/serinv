from sdr.utils import matrix_generation
from sdr.lu.lu_decompose import lu_dcmp_tridiag_arrowhead
from sdr.lu.lu_selected_inversion import lu_sinv_tridiag_arrowhead

from sdr.lu.lu_decompose import lu_dcmp_tridiag
from sdr.lu.lu_selected_inversion import lu_sinv_tridiag


import numpy as np
import math
import matplotlib.pyplot as plt
from mpi4py import MPI


def get_partitions_indices(
    n_partitions: int,
    total_size: int,
    partitions_distribution: list = None,
) -> [[], [], []]:
    """Create the partitions start/end indices and sizes for the entire problem.
    If the problem size doesn't match a perfect partitioning w.r.t the distribution,
    partitions will be resized starting from the first one.

    Parameters
    ----------
    n_partitions : int
        Total number of partitions.
    total_size : int
        Total number of blocks in the global matrix. Equal to the sum of the sizes of
        all partitions.
    partitions_distribution : list, optional
        Distribution of the partitions sizes, in percentage. The default is None
        and a uniform distribution is assumed.

    Returns
    -------
    start_blockrows : []
        List of the indices of the first blockrow of each partition in the
        global matrix.
    partition_sizes : []
        List of the sizes of each partition.
    end_blockrows : []
        List of the indices of the last blockrow of each partition in the
        global matrix.

    """

    if n_partitions > total_size:
        raise ValueError(
            "Number of partitions cannot be greater than the total size of the matrix."
        )

    if partitions_distribution is not None:
        if n_partitions != len(partitions_distribution):
            raise ValueError(
                "Number of partitions and number of entries in the distribution list do not match."
            )
        if sum(partitions_distribution) != 100:
            raise ValueError(
                "Sum of the entries in the distribution list is not equal to 100."
            )
    else:
        partitions_distribution = [100 / n_partitions] * n_partitions

    partitions_distribution = np.array(partitions_distribution) / 100

    start_blockrows = []
    partition_sizes = []
    end_blockrows = []

    for i in range(n_partitions):
        partition_sizes.append(math.floor(partitions_distribution[i] * total_size))

    if sum(partition_sizes) != total_size:
        diff = total_size - sum(partition_sizes)
        for i in range(diff):
            partition_sizes[i] += 1

    for i in range(n_partitions):
        start_blockrows.append(sum(partition_sizes[:i]))
        end_blockrows.append(start_blockrows[i] + partition_sizes[i])

    return start_blockrows, partition_sizes, end_blockrows


def extract_partition(
    A_global: np.ndarray,
    start_blockrow: int,
    partition_size: int,
    blocksize: int,
    arrow_blocksize: int,
):
    A_local = np.zeros(
        (partition_size * blocksize, partition_size * blocksize), dtype=A_global.dtype
    )
    A_arrow_bottom = np.zeros(
        (arrow_blocksize, partition_size * arrow_blocksize), dtype=A_global.dtype
    )
    A_arrow_right = np.zeros(
        (partition_size * arrow_blocksize, arrow_blocksize), dtype=A_global.dtype
    )

    stop_blockrow = start_blockrow + partition_size

    A_local = A_global[
        start_blockrow * blocksize : stop_blockrow * blocksize,
        start_blockrow * blocksize : stop_blockrow * blocksize,
    ]
    A_arrow_bottom = A_global[
        -arrow_blocksize:, start_blockrow * blocksize : stop_blockrow * blocksize
    ]
    A_arrow_right = A_global[
        start_blockrow * blocksize : stop_blockrow * blocksize, -arrow_blocksize:
    ]

    return A_local, A_arrow_bottom, A_arrow_right


def extract_bridges(
    A_global: np.ndarray,
    blocksize: int,
    arrow_blocksize: int,
    partition_sizes: list,
) -> [list, list]:
    # Without arrowhead tip
    num_partitions = len(partition_sizes)

    Bridges_lower = []
    Bridges_upper = []

    for i in range(num_partitions - 1):
        start_index = sum(partition_sizes[: i + 1]) * blocksize
        Bridges_lower.append(
            A_global[
                start_index : start_index + blocksize,
                start_index - blocksize : start_index,
            ]
        )
        Bridges_upper.append(
            A_global[
                start_index - blocksize : start_index,
                start_index : start_index + blocksize,
            ]
        )

    return Bridges_lower, Bridges_upper


def top_factorize(
    A_local: np.ndarray,
    A_arrow_bottom: np.ndarray,
    A_arrow_right: np.ndarray,
    blocksize: int,
    arrowhead_blocksize: int,
) -> [np.ndarray, np.ndarray, np.ndarray]:
    LU_local = np.zeros_like(A_local)
    L_arrow_bottom = np.zeros_like(A_arrow_bottom)
    U_arrow_right = np.zeros_like(A_arrow_right)

    nblocks = A_local.shape[0] // blocksize

    for i in range(1, nblocks):
        # L[i, i-1] = A[i, i-1] @ A[i-1, i-1]^(-1)
        LU_local[
            i * blocksize : (i + 1) * blocksize, (i - 1) * blocksize : i * blocksize
        ] = A_local[
            i * blocksize : (i + 1) * blocksize, (i - 1) * blocksize : i * blocksize
        ] @ np.linalg.inv(
            A_local[
                (i - 1) * blocksize : i * blocksize, (i - 1) * blocksize : i * blocksize
            ]
        )
        # LU_local[i * blocksize : (i + 1) * blocksize, (i-1) * blocksize : i * blocksize] = np.linalg.solve(A_local[i * blocksize : (i + 1) * blocksize, (i-1) * blocksize : i * blocksize].T, A_local[(i-1) * blocksize : i * blocksize, (i-1) * blocksize : i * blocksize].T).T

        # U[i-1, i] = A[i-1, i-1]^(-1) @ A[i-1, i]
        LU_local[
            (i - 1) * blocksize : i * blocksize, i * blocksize : (i + 1) * blocksize
        ] = (
            np.linalg.inv(
                A_local[
                    (i - 1) * blocksize : i * blocksize,
                    (i - 1) * blocksize : i * blocksize,
                ]
            )
            @ A_local[
                (i - 1) * blocksize : i * blocksize, i * blocksize : (i + 1) * blocksize
            ]
        )
        # LU_local[(i-1) * blocksize : i * blocksize, i * blocksize : (i+1) * blocksize]   = np.linalg.solve(A_local[(i-1) * blocksize : i * blocksize, (i-1) * blocksize : i * blocksize], A_local[(i-1) * blocksize : i * blocksize, i * blocksize : (i+1) * blocksize])

        # A_{i+1, ndb+1} = A_{i+1, ndb+1} - L_{i+1, i} @ U_{i, ndb+1}

        # A_{ndb+1, ndb+1} = A_{ndb+1, ndb+1} - L_{ndb+1, i} @ U_{i, ndb+1}

        # A[i, i] = A[i, i] - L[i, i-1] @ A[i-1, i]
        A_local[
            i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize
        ] = (
            A_local[
                i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize
            ]
            - LU_local[
                i * blocksize : (i + 1) * blocksize, (i - 1) * blocksize : i * blocksize
            ]
            @ A_local[
                (i - 1) * blocksize : i * blocksize, i * blocksize : (i + 1) * blocksize
            ]
        )

    return A_local, LU_local, L_arrow_bottom, U_arrow_right


def middle_factorize(
    A_local: np.ndarray,
    A_arrow_bottom: np.ndarray,
    A_arrow_right: np.ndarray,
    blocksize: int,
    arrowhead_blocksize: int,
) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    LU_local = np.zeros_like(A_local)
    L_arrow_bottom = np.zeros_like(A_arrow_bottom)
    U_arrow_right = np.zeros_like(A_arrow_right)

    n_blocks = A_local.shape[0] // blocksize

    for i in range(2, n_blocks):
        top = slice(0, blocksize)
        im1 = slice((i - 1) * blocksize, i * blocksize)
        i = slice(i * blocksize, (i + 1) * blocksize)

        A_im1im1_inv = np.linalg.inv(A_local[im1, im1])

        # L[i, i-1] = A[i, i-1] @ A[i-1, i-1]^(-1)
        LU_local[i, im1] = A_local[i, im1] @ A_im1im1_inv

        # L[top, i-1] = A[top, i-1] @ A[i-1, i-1]^(-1)
        LU_local[top, im1] = A_local[top, im1] @ A_im1im1_inv

        # U[i-1, i] = A[i-1, i-1]^(-1) @ A[i-1, i]
        LU_local[im1, i] = A_im1im1_inv @ A_local[im1, i]

        # U[i-1, top] = A[i-1, i-1]^(-1) @ A[i-1, top]
        LU_local[im1, top] = A_im1im1_inv @ A_local[im1, top]

        # A_local[i, i] = A[i, i] - L[i, i-1] @ A_local[i-1, i]
        A_local[i, i] = A_local[i, i] - LU_local[i, im1] @ A_local[im1, i]

        # A_local[top, top] = A[top, top] - L[top, i-1] @ A_local[i-1, top]
        A_local[top, top] = A_local[top, top] - LU_local[top, im1] @ A_local[im1, top]

        # A_local[i, top] = - L[i, i-1] @ A_local[i-1, top]
        A_local[i, top] = -LU_local[i, im1] @ A_local[im1, top]

        # A_local[top, i] = - L[top, i-1] @ A_local[i-1, i]
        A_local[top, i] = -LU_local[top, im1] @ A_local[im1, i]

    return A_local, LU_local, L_arrow_bottom, U_arrow_right


def create_reduced_system(
    A_local,
    A_arrow_bottom,
    A_arrow_right,
    A_global_arrow_tip,
    Bridges_upper,
    Bridges_lower,
    local_arrow_tip_update,
    blocksize,
    arrowhead_blocksize,
):
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Create empty matrix for reduced system -> (2*#process - 1)*blocksize + arrowhead_size
    size_reduced_system = (2 * comm_size - 1) * blocksize + arrow_blocksize
    reduced_system = np.zeros((size_reduced_system, size_reduced_system))
    reduced_system[-arrow_blocksize:, -arrow_blocksize:] = local_arrow_tip_update

    if comm_rank == 0:
        reduced_system[:blocksize, :blocksize] = A_local[-blocksize:, -blocksize:]
        reduced_system[:blocksize, blocksize : 2 * blocksize] = Bridges_upper[comm_rank]

        reduced_system[-arrow_blocksize:, :blocksize] = A_arrow_bottom[:, -blocksize:]
        reduced_system[:blocksize, -arrow_blocksize:] = A_arrow_right[-blocksize:, :]
    else:
        start_index = blocksize + (comm_rank - 1) * 2 * blocksize

        reduced_system[
            start_index : start_index + blocksize, start_index - blocksize : start_index
        ] = Bridges_lower[comm_rank - 1]

        reduced_system[
            start_index : start_index + blocksize, start_index : start_index + blocksize
        ] = A_local[:blocksize, :blocksize]

        reduced_system[
            start_index : start_index + blocksize,
            start_index + blocksize : start_index + 2 * blocksize,
        ] = A_local[:blocksize, -blocksize:]

        reduced_system[
            start_index + blocksize : start_index + 2 * blocksize,
            start_index : start_index + blocksize,
        ] = A_local[-blocksize:, :blocksize]

        reduced_system[
            start_index + blocksize : start_index + 2 * blocksize,
            start_index + blocksize : start_index + 2 * blocksize,
        ] = A_local[-blocksize:, -blocksize:]

        if comm_rank != comm_size - 1:
            reduced_system[
                start_index + blocksize : start_index + 2 * blocksize,
                start_index + 2 * blocksize : start_index + 3 * blocksize,
            ] = Bridges_upper[comm_rank]

        reduced_system[
            -arrow_blocksize:, start_index : start_index + blocksize
        ] = A_arrow_bottom[:, :blocksize]

        reduced_system[
            -arrow_blocksize:, start_index + blocksize : start_index + 2 * blocksize
        ] = A_arrow_bottom[:, -blocksize:]

        reduced_system[
            start_index : start_index + blocksize, -arrow_blocksize:
        ] = A_arrow_right[:blocksize, :]

        reduced_system[
            start_index + blocksize : start_index + 2 * blocksize, -arrow_blocksize:
        ] = A_arrow_right[-blocksize:, :]

    """ plt.matshow(reduced_system)
    plt.title("Reduced system process: " + str(comm_rank))
    plt.show() """

    # Send the reduced_system with MPIallReduce SUM operation
    reduced_system_sum = np.zeros_like(reduced_system)
    comm.Allreduce(
        [reduced_system, MPI.DOUBLE], [reduced_system_sum, MPI.DOUBLE], op=MPI.SUM
    )

    reduced_system_sum[-arrow_blocksize:, -arrow_blocksize:] += A_global_arrow_tip

    """ plt.matshow(reduced_system_sum)
    plt.title("Reduced system process: " + str(comm_rank))
    plt.show() """

    return reduced_system_sum


def inverse_reduced_system(reduced_system, diag_blocksize, arrowhead_blocksize):
    n_diag_blocks = (reduced_system.shape[0] - arrowhead_blocksize) // diag_blocksize

    # For now with blk tridiag
    # Cast the right size
    reduced_system_sliced_to_tridiag = reduced_system[
        : n_diag_blocks * diag_blocksize, : n_diag_blocks * diag_blocksize
    ]

    L_reduced_sliced_to_tridiag, U_reduced_sliced_to_tridiag = lu_dcmp_tridiag(
        reduced_system_sliced_to_tridiag, diag_blocksize
    )
    S_reduced_sliced_to_tridiag = lu_sinv_tridiag(
        L_reduced_sliced_to_tridiag, U_reduced_sliced_to_tridiag, diag_blocksize
    )

    S_reduced = np.zeros_like(reduced_system)
    S_reduced[
        : n_diag_blocks * diag_blocksize, : n_diag_blocks * diag_blocksize
    ] = S_reduced_sliced_to_tridiag

    # Switch when arrowhead
    # L_reduced, U_reduced = lu_dcmp_tridiag_arrowhead(reduced_system, diag_blocksize, arrowhead_blocksize)
    # S_reduced  = lu_sinv_tridiag_arrowhead(L_reduced,  U_reduced, diag_blocksize, arrowhead_blocksize)

    return S_reduced


def update_sinv_reduced_system(
    S_local: np.ndarray,
    S_arrow_bottom: np.ndarray,
    S_arrow_right: np.ndarray,
    reduced_system: np.ndarray,
    Bridges_upper: np.ndarray,
    Bridges_lower: np.ndarray,
    blocksize: int,
    arrow_blocksize: int,
):
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    if comm_rank == 0:
        S_local[-blocksize:, -blocksize:] = reduced_system[:blocksize, :blocksize]

        Bridges_upper[comm_rank] = reduced_system[:blocksize, blocksize : 2 * blocksize]

        S_arrow_bottom[:, -blocksize:] = reduced_system[-arrow_blocksize:, :blocksize]
        S_arrow_right[-blocksize:, :] = reduced_system[:blocksize, -arrow_blocksize:]
    else:
        start_index = blocksize + (comm_rank - 1) * 2 * blocksize

        Bridges_lower[comm_rank - 1] = reduced_system[
            start_index : start_index + blocksize, start_index - blocksize : start_index
        ]

        S_local[:blocksize, :blocksize] = reduced_system[
            start_index : start_index + blocksize, start_index : start_index + blocksize
        ]

        S_local[:blocksize, -blocksize:] = reduced_system[
            start_index : start_index + blocksize,
            start_index + blocksize : start_index + 2 * blocksize,
        ]

        S_local[-blocksize:, :blocksize] = reduced_system[
            start_index + blocksize : start_index + 2 * blocksize,
            start_index : start_index + blocksize,
        ]

        S_local[-blocksize:, -blocksize:] = reduced_system[
            start_index + blocksize : start_index + 2 * blocksize,
            start_index + blocksize : start_index + 2 * blocksize,
        ]

        if comm_rank != comm_size - 1:
            Bridges_upper[comm_rank] = reduced_system[
                start_index + blocksize : start_index + 2 * blocksize,
                start_index + 2 * blocksize : start_index + 3 * blocksize,
            ]

        S_arrow_bottom[:, :blocksize] = reduced_system[
            -arrow_blocksize:, start_index : start_index + blocksize
        ]

        S_arrow_bottom[:, -blocksize:] = reduced_system[
            -arrow_blocksize:, start_index + blocksize : start_index + 2 * blocksize
        ]

        S_arrow_right[:blocksize, :] = reduced_system[
            start_index : start_index + blocksize, -arrow_blocksize:
        ]

        S_arrow_right[-blocksize:, :] = reduced_system[
            start_index + blocksize : start_index + 2 * blocksize, -arrow_blocksize:
        ]

    return S_local, S_arrow_bottom, S_arrow_right


def top_sinv(
    S_local: np.ndarray,
    S_arrow_bottom: np.ndarray,
    S_arrow_right: np.ndarray,
    A_local: np.ndarray,
    A_arrow_bottom: np.ndarray,
    A_arrow_right: np.ndarray,
    LU_local: np.ndarray,
    L_arrow_bottom: np.ndarray,
    U_arrow_right: np.ndarray,
    blocksize: int,
    arrowhead_blocksize: int,
) -> [np.ndarray, np.ndarray, np.ndarray]:
    n_blocks = A_local.shape[0] // blocksize

    # # for now initialize S_local[nblocks, nblocks] = inv(A_local[nblocks, nblocks])
    # #S_local[(n_blocks - 1)*blocksize:n_blocks*blocksize, (n_blocks - 1)*blocksize:n_blocks*blocksize] = np.linalg.inv(A_local[(n_blocks - 1)*blocksize:n_blocks*blocksize, (n_blocks - 1)*blocksize:n_blocks*blocksize])

    for i in range(n_blocks - 1, 0, -1):
        # S_{i, i-1} = - S_{i, i} @ L_{i, i-1}
        S_local[
            i * blocksize : (i + 1) * blocksize, (i - 1) * blocksize : i * blocksize
        ] = (
            -S_local[
                i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize
            ]
            @ LU_local[
                i * blocksize : (i + 1) * blocksize, (i - 1) * blocksize : i * blocksize
            ]
        )

        # S_{i-1, i} = - U_{i-1, i} @ S_{i, i}
        S_local[
            (i - 1) * blocksize : i * blocksize, i * blocksize : (i + 1) * blocksize
        ] = (
            -LU_local[
                (i - 1) * blocksize : i * blocksize, i * blocksize : (i + 1) * blocksize
            ]
            @ S_local[
                i * blocksize : (i + 1) * blocksize, i * blocksize : (i + 1) * blocksize
            ]
        )

        # S_{i-1, i-1} = A_{i-1, i-1}^{-1} - U_{i-1, i} @ S_{i, i-1}
        S_local[
            (i - 1) * blocksize : i * blocksize, (i - 1) * blocksize : i * blocksize
        ] = (
            np.linalg.inv(
                A_local[
                    (i - 1) * blocksize : i * blocksize,
                    (i - 1) * blocksize : i * blocksize,
                ]
            )
            - LU_local[
                (i - 1) * blocksize : i * blocksize, i * blocksize : (i + 1) * blocksize
            ]
            @ S_local[
                i * blocksize : (i + 1) * blocksize, (i - 1) * blocksize : i * blocksize
            ]
        )

    return S_local, S_arrow_bottom, S_arrow_right


def middle_sinv(
    S_local: np.ndarray,
    S_arrow_bottom: np.ndarray,
    S_arrow_right: np.ndarray,
    A_local: np.ndarray,
    A_arrow_bottom: np.ndarray,
    A_arrow_right: np.ndarray,
    LU_local: np.ndarray,
    L_arrow_bottom: np.ndarray,
    U_arrow_right: np.ndarray,
    blocksize: int,
    arrowhead_blocksize: int,
) -> [np.ndarray, np.ndarray, np.ndarray]:
    n_blocks = A_local.shape[0] // blocksize

    top_slice = slice(0, blocksize)
    botm1_slice = slice((n_blocks - 2) * blocksize, (n_blocks - 1) * blocksize)
    bot_slice = slice((n_blocks - 1) * blocksize, n_blocks * blocksize)

    # S_local[bot, bot-1] = - S_local[bot, top] @ L[top, bot-1] - S_local[bot, bot] @ L[bot, bot-1]
    S_local[bot_slice, botm1_slice] = (
        -S_local[bot_slice, top_slice] @ LU_local[top_slice, botm1_slice]
        - S_local[bot_slice, bot_slice] @ LU_local[bot_slice, botm1_slice]
    )

    # S_local[bot-1, bot] = - U[bot-1, bot] @ S_local[bot, bot] - U[bot-1, top] @ S_local[top, bot]
    S_local[botm1_slice, bot_slice] = (
        -LU_local[botm1_slice, bot_slice] @ S_local[bot_slice, bot_slice]
        - LU_local[botm1_slice, top_slice] @ S_local[top_slice, bot_slice]
    )

    for i in range(n_blocks - 2, 0, -1):
        i_slice = slice(i * blocksize, (i + 1) * blocksize)
        ip1_slice = slice((i + 1) * blocksize, (i + 2) * blocksize)

        # S_local[top, i] = - S_local[top, top] @ L[top, i] - S_local[top, i+1] @ L[i+1, i]
        S_local[top_slice, i_slice] = (
            -S_local[top_slice, top_slice] @ LU_local[top_slice, i_slice]
            - S_local[top_slice, ip1_slice] @ LU_local[ip1_slice, i_slice]
        )

        # S_local[i, top] = - U[i, i+1] @ S_local[i+1, top] - U[i, top] @ S_local[top, top]
        S_local[i_slice, top_slice] = (
            -LU_local[i_slice, ip1_slice] @ S_local[ip1_slice, top_slice]
            - LU_local[i_slice, top_slice] @ S_local[top_slice, top_slice]
        )

    for i in range(n_blocks - 2, 1, -1):
        im1_slice = slice((i - 1) * blocksize, i * blocksize)
        i_slice = slice(i * blocksize, (i + 1) * blocksize)
        ip1_slice = slice((i + 1) * blocksize, (i + 2) * blocksize)

        # S_local[i, i] = np.linalg.inv(A_local[i, i]) - U[i, top] @ S_local[top, i] - U[i, i+1] @ S_local[i+1, i]
        S_local[i_slice, i_slice] = (
            np.linalg.inv(A_local[i_slice, i_slice])
            - LU_local[i_slice, top_slice] @ S_local[top_slice, i_slice]
            - LU_local[i_slice, ip1_slice] @ S_local[ip1_slice, i_slice]
        )

        # S_local[i-1, i] = - U[i-1, top] @ S_local[top, i] - U[i-1, i] @ S_local[i, i]
        S_local[im1_slice, i_slice] = (
            -LU_local[im1_slice, top_slice] @ S_local[top_slice, i_slice]
            - LU_local[im1_slice, i_slice] @ S_local[i_slice, i_slice]
        )

        # S_local[i, i-1] = - S_local[i, top] @ L[top, i-1] - S_local[i, i] @ L[i, i-1]
        S_local[i_slice, im1_slice] = (
            -S_local[i_slice, top_slice] @ LU_local[top_slice, im1_slice]
            - S_local[i_slice, i_slice] @ LU_local[i_slice, im1_slice]
        )

    topp1_slice = slice(blocksize, 2 * blocksize)
    topp2_slice = slice(2 * blocksize, 3 * blocksize)

    # S_local[top+1, top+1] = np.linalg.inv(A_local[top+1, top+1]) - U[top+1, top] @ S_local[top, top+1] - U[top+1, top+2] @ S_local[top+2, top+1]
    S_local[topp1_slice, topp1_slice] = (
        np.linalg.inv(A_local[topp1_slice, topp1_slice])
        - LU_local[topp1_slice, top_slice] @ S_local[top_slice, topp1_slice]
        - LU_local[topp1_slice, topp2_slice] @ S_local[topp2_slice, topp1_slice]
    )

    return S_local, S_arrow_bottom, S_arrow_right


def psr_arrowhead(
    A_local: np.ndarray,
    A_arrow_bottom: np.ndarray,
    A_arrow_right: np.ndarray,
    A_global_arrow_tip: np.ndarray,
    Bridges_upper,
    Bridges_lower,
    blocksize: int,
    arrowhead_blocksize: int,
):
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    if comm_rank == 0:
        A_local, LU_local, L_arrow_bottom, U_arrow_right = top_factorize(
            A_local, A_arrow_bottom, A_arrow_right, blocksize, arrowhead_blocksize
        )
    else:
        A_local, LU_local, L_arrow_bottom, U_arrow_right = middle_factorize(
            A_local, A_arrow_bottom, A_arrow_right, blocksize, arrowhead_blocksize
        )

    """ plt.matshow(A_local)
    plt.title("A_local process: " + str(process))

    plt.matshow(LU_local)
    plt.title("LU_local process: " + str(process))
    plt.show() """

    local_arrow_tip_update = np.zeros((arrowhead_blocksize, arrowhead_blocksize))

    reduced_system = create_reduced_system(
        A_local,
        A_arrow_bottom,
        A_arrow_right,
        A_global_arrow_tip,
        Bridges_upper,
        Bridges_lower,
        local_arrow_tip_update,
        blocksize,
        arrowhead_blocksize,
    )

    # plt.matshow(reduced_system)
    # plt.title("reduced_system process: " + str(comm_rank))
    # plt.show()

    reduced_system_inv = inverse_reduced_system(
        reduced_system, diag_blocksize, arrowhead_blocksize
    )

    # plt.matshow(reduced_system_inv)
    # plt.title("reduced_system_inv process: " + str(comm_rank))
    # plt.show()

    S_local = np.zeros_like(A_local)
    S_arrow_bottom = np.zeros_like(A_arrow_bottom)
    S_arrow_right = np.zeros_like(A_arrow_right)

    S_local, S_arrow_bottom, S_arrow_right = update_sinv_reduced_system(
        S_local,
        S_arrow_bottom,
        S_arrow_right,
        reduced_system_inv,
        Bridges_upper,
        Bridges_lower,
        blocksize,
        arrow_blocksize,
    )

    # plt.matshow(reduced_system_inv)
    # plt.title("reduced_system_inv process: " + str(comm_rank))
    # plt.matshow(S_local)
    # plt.title("S_local process: " + str(comm_rank))
    # plt.show()

    # plt.matshow(S_local)
    # plt.title("S_local before process: " + str(comm_rank))

    if comm_rank == 0:
        S_local, S_arrow_bottom, S_arrow_right = top_sinv(
            S_local,
            S_arrow_bottom,
            S_arrow_right,
            A_local,
            A_arrow_bottom,
            A_arrow_right,
            LU_local,
            L_arrow_bottom,
            U_arrow_right,
            blocksize,
            arrowhead_blocksize,
        )

    else:
        S_local, S_arrow_bottom, S_arrow_right = middle_sinv(
            S_local,
            S_arrow_bottom,
            S_arrow_right,
            A_local,
            A_arrow_bottom,
            A_arrow_right,
            LU_local,
            L_arrow_bottom,
            U_arrow_right,
            blocksize,
            arrowhead_blocksize,
        )

    # plt.matshow(S_local)
    # plt.title("S_local after process: " + str(comm_rank))
    # plt.show()

    return S_local, S_arrow_bottom, S_arrow_right


import copy as cp


if __name__ == "__main__":
    nblocks = 30
    diag_blocksize = 3
    arrow_blocksize = 2
    symmetric = False
    diagonal_dominant = True
    seed = 63

    A = matrix_generation.generate_blocktridiag_arrowhead(
        nblocks, diag_blocksize, arrow_blocksize, symmetric, diagonal_dominant, seed
    )

    A_ref = cp.deepcopy(A)
    A_inv_ref = np.linalg.inv(A_ref)

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    n_partitions = comm_size

    start_blockrows, partition_sizes, end_blockrows = get_partitions_indices(
        n_partitions=n_partitions, total_size=nblocks - 1
    )

    Bridges_upper, Bridges_lower = extract_bridges(
        A, diag_blocksize, arrow_blocksize, partition_sizes
    )

    A_arrow_tip = A[-arrow_blocksize:, -arrow_blocksize:]

    A_local, A_arrow_bottom, A_arrow_right = extract_partition(
        A,
        start_blockrows[comm_rank],
        partition_sizes[comm_rank],
        diag_blocksize,
        arrow_blocksize,
    )

    S_local, S_arrow_bottom, S_arrow_right = psr_arrowhead(
        A_local,
        A_arrow_bottom,
        A_arrow_right,
        A_arrow_tip,
        Bridges_upper,
        Bridges_lower,
        diag_blocksize,
        arrow_blocksize,
    )

    A_inv_ref_local, A_ref_arrow_bottom, A_ref_arrow_right = extract_partition(
        A_inv_ref,
        start_blockrows[comm_rank],
        partition_sizes[comm_rank],
        diag_blocksize,
        arrow_blocksize,
    )

    fig, ax = plt.subplots(1, 2)
    ax[0].matshow(A_inv_ref_local)
    ax[0].set_title("A_inv_ref_local process: " + str(comm_rank))
    ax[1].matshow(S_local)
    ax[1].set_title("S_local process: " + str(comm_rank))
    plt.show()
