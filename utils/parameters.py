def calculate_parameters_banded(
        matrix_size: int,
        bandwidth: int,
        arrowhead_width: int
):

    parameters = {
        "matrix_size": matrix_size,
        "arrowhead_blocksize": arrowhead_width,
        "bandwidth": bandwidth
    }

    try:
        assert bandwidth % 2 == 1,  "Bandwidth must be odd"

        diagonal_blocksize_ = 1
        n_offdiags_ = int((bandwidth - 1)/2)

        effective_bandwidth_ = diagonal_blocksize_*(n_offdiags_*2+1)
        parameters["effective_bandwidth"] = effective_bandwidth_

        assert bandwidth <= effective_bandwidth_, \
            "Effective bandwidth is smaller than bandwidth"

        n_t_ = matrix_size - arrowhead_width
        parameters["n_t"] = n_t_

        matrix_size_ = n_t_*diagonal_blocksize_ + arrowhead_width

        parameters["diagonal_blocksize"] = diagonal_blocksize_
        parameters["n_offdiags"] = n_offdiags_

        assert matrix_size_ == matrix_size, "Matrix size must be kept"

        return {
            'parameters': parameters,
            'flag': 1
        }
    except AssertionError as e:
        return {
            'parameters': parameters,
            'flag': 0,
            'error': str(e),
        }


def calculate_parameters_tri_diagonal(
        matrix_size: int,
        bandwidth: int,
        arrowhead_width: int
):

    parameters = {
        "matrix_size": matrix_size,
        "arrowhead_blocksize": arrowhead_width,
        "bandwidth": bandwidth
    }

    try:
        assert bandwidth % 2 == 1,  "Bandwidth must be odd"

        n_offdiags_ = 1
        inner_matrix_size = matrix_size - arrowhead_width

        diagonal_blocksize_ = int((bandwidth-1)/2)
        n_t_ = inner_matrix_size//diagonal_blocksize_

        iters = 0
        while inner_matrix_size % diagonal_blocksize_:
            diagonal_blocksize_ += 1
            n_t_ = inner_matrix_size//diagonal_blocksize_
            iters += 1

        parameters["n_t"] = n_t_
        assert n_t_ > 1, "Matrix not divisible, might be prime"

        effective_bandwidth_ = diagonal_blocksize_*2+1
        parameters["effective_bandwidth"] = effective_bandwidth_

        assert bandwidth <= effective_bandwidth_, \
            "Effective bandwidth is smaller than bandwidth"

        parameters["diagonal_blocksize"] = diagonal_blocksize_
        parameters["n_offdiags"] = n_offdiags_

        matrix_size_ = n_t_*diagonal_blocksize_ + arrowhead_width
        assert matrix_size_ == matrix_size, "Matrix size must be kept"

        return {
            'parameters': parameters,
            'flag': 1
        }
    except AssertionError as e:
        return {
            'parameters': parameters,
            'flag': 0,
            'error': str(e),
        }


def calculate_parameters_n_diagonal(
        matrix_size: int,
        bandwidth: int,
        arrowhead_width: int,
        n_offdiags_: int = 1
):

    parameters = {
        "matrix_size": matrix_size,
        "arrowhead_blocksize": arrowhead_width,
        "bandwidth": bandwidth
    }
    try:
        assert bandwidth % 2 == 1,  "Bandwidth must be odd"

        inner_matrix_size = matrix_size - arrowhead_width

        diagonal_blocksize_ = int(
            (bandwidth-1+2*n_offdiags_-1)/(2*n_offdiags_))
        n_t_ = inner_matrix_size//diagonal_blocksize_

        iters = 0
        while inner_matrix_size % diagonal_blocksize_:
            diagonal_blocksize_ += 1
            n_t_ = inner_matrix_size//diagonal_blocksize_
            iters += 1

        parameters["n_t"] = n_t_
        assert n_t_ > 1, "Matrix not divisible, might be prime"

        matrix_size_ = n_t_*diagonal_blocksize_ + arrowhead_width

        parameters["diagonal_blocksize"] = diagonal_blocksize_
        parameters["n_offdiags"] = n_offdiags_

        assert matrix_size_ == matrix_size, "Matrix size must be kept"

        effective_bandwidth_ = diagonal_blocksize_*n_offdiags_*2+1
        parameters["effective_bandwidth"] = effective_bandwidth_

        assert bandwidth <= effective_bandwidth_, \
            "Effective bandwidth is smaller than bandwidth"

        if bandwidth <= (diagonal_blocksize_*(n_offdiags_-1)*2+1):
            raise AssertionError('Extra block is being used for bandwidth')

        return {
            'parameters': parameters,
            'flag': 1
        }
    except AssertionError as e:
        return {
            'parameters': parameters,
            'flag': 0,
            'error': str(e),
        }
