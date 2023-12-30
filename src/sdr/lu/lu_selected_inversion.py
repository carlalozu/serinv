"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Contains the lu selected inversion routines.

Copyright 2023 ETH Zurich and USI. All rights reserved.
"""


import copy
import numpy as np
import scipy.linalg as la

from sdr.utils import matrix_generation


def invert_LU(LU: np.ndarray) -> np.ndarray:
    """ Invert a triangular matrix.
    
    Parameters
    ----------
    A : np.ndarray
        The triangular matrix to invert.
    lower : bool
        Whether the matrix is lower or upper triangular.
    
    Returns
    -------
    A_inv : np.ndarray
        The inverted triangular matrix.
    """
    n = LU.shape[0]
    
    for i in range(n):
        LU[i, i] = 1 / LU[i, i]
        for j in range(i):
            s = 0
            for k in range(j, i):
                s += LU[k, i] * LU[j, k]
            LU[j,i] = - s * LU[i, i]
    
    for i in range(n):
        for j in range(i):
            s = LU[i, j]
            for k in range(j+1, i):
                s += LU[i, k] * LU[k, j]
            LU[i,j] = - s


    L = np.tril(LU, k=-1) + np.eye(n, dtype=LU.dtype)
    U = np.triu(LU, k=0)    
        
    
    return U @ L




# def sinv_ndiags_greg(
#     in_A: np.ndarray,
#     ndiags: int,
# ) -> np.ndarray:
#     """ Perform the LU factorization of an n-diagonals matrix. The matrix is assumed to be non-singular.

#     Parameters
#     ----------
#     A : np.ndarray
#         Input matrix to decompose.
#     ndiags : int
#         Number of diagonals of the matrix above the main diagonal. The total number of diagonals is 2*ndiags+1.
#     blocksize : int
#         Size of the blocks.
    
#     Returns
#     -------
#     L : np.ndarray
#         Lower factor of the LU factorization of the matrix.
#     U : np.ndarray
#         Upper factor of the LU factorization of the matrix.
#     """


#     A_inv_str = np.zeros(in_A.shape, dtype=in_A.dtype)
#     A = copy.deepcopy(in_A)
#     # in-place LU decomposition without pivoting
#     n = A.shape[0]
#     for k in range(n):
#         A[k, k] = 1 / A[k, k]

#         for i in range(k+1,min(k+1 + ndiags, n)): #(n-1, k, -1)
#             A[i, k] = A[i,k] * A[k,k]
#             for j in range(k+1,min(k+1 + ndiags, n)): #range(n-1, k, -1): 
#                 A[i,j]  -= A[i,k]*A[k,j]      

#         A_inv_str[k,k] = A[k,k] 
#         for i in range(k): #range(max(0, k - ndiags), k): 
#             s1 = A[i, k] * A[i, i]
#             s2 = A[k, i]

#             for j in range(i+1, k):  
#                 s1 += A[j, k] * A[i, j]
#                 s2 += A[k, j] * A[j, i]
#             A[i,k] = - s1 * A[k, k]
#             A[k,i] = - s2


#             A_inv_str[i,k] = A[i,k] 
#             A_inv_str[k,i] = A[k,i] * A[k,k] 
#             for j in range(i):
#                 A_inv_str[i,j] +=  A[i,k]*A[k,j]
#                 A_inv_str[j,i] +=  A[j,k]*A[k,i]
#             A_inv_str[i,i] += A[i,k]*A[k,i] 

#     LU = copy.deepcopy(A)
    

#     U = np.triu(LU, k=0)
#     L = np.tril(LU, k=-1) + np.eye(n, dtype=A.dtype)
#     A_inv = U @ L

#     return U @ L




def sinv_ndiags_greg(
    in_A: np.ndarray,
    ndiags: int,
) -> np.ndarray:
    """ Perform the LU factorization of an n-diagonals matrix. The matrix is assumed to be non-singular.

    Parameters
    ----------
    A : np.ndarray
        Input matrix to decompose.
    ndiags : int
        Number of diagonals of the matrix above the main diagonal. The total number of diagonals is 2*ndiags+1.
    blocksize : int
        Size of the blocks.
    
    Returns
    -------
    L : np.ndarray
        Lower factor of the LU factorization of the matrix.
    U : np.ndarray
        Upper factor of the LU factorization of the matrix.
    """


    A_inv_str = np.zeros(in_A.shape, dtype=in_A.dtype)
    A = copy.deepcopy(in_A)
    n = A.shape[0]
    for k in range(n):
        # BEGIN in-place LU decomposition without pivoting
        A[k, k] = 1 / A[k, k]  # <- this is NOT the part of LU! This is already a first step of INVERSION

        for i in range(k+1,min(k+1 + ndiags, n)): #(n-1, k, -1)
            A[i, k] = A[i,k] * A[k,k]
            for j in range(k+1,min(k+1 + ndiags, n)): #range(n-1, k, -1): 
                A[i,j]  -= A[i,k]*A[k,j]      
        # END in-place LU decomposition without pivoting

        A_inv_str[k,k] = A[k,k]  # <- A will hold  LU inverted. la.tril(A) is L^(-1), 
                                 #  la.triu(A) is U^(-1), A_inv_str is  U^(-1) @ L^(-1)

        # BEGIN invert in-place LU decomposition (two triangular L and U matrices are stored in A)
        for i in range(max(0, k - ndiags), k):  # range(k): #
            s1 = A[i, k] * A[i, i]
            s2 = A[k, i]

            for j in range(i+1, k):  
                s1 += A[j, k] * A[i, j]
                s2 += A[k, j] * A[j, i]
            A[i,k] = - s1 * A[k, k]
            A[k,i] = - s2
            # END invert in-place LU decomposition (two triangular L and U matrices are stored in A)

            # BEGIN matrix-matrix multiplcation of U^(k-1) L^(k-1) (stored in A)
            A_inv_str[i,k] = A[i,k] 
            A_inv_str[k,i] = A[k,i] * A[k,k] 
            for j in range(i):
                A_inv_str[i,j] +=  A[i,k]*A[k,j]
                A_inv_str[j,i] +=  A[j,k]*A[k,i]
            if i == 0:
                print(f"A[i,i] = {A_inv_str[i,i]}, A[i,k] = {A[i,k]}, A[k,i] = {A[k,i]}, res={A_inv_str[i,i] + A[i,k]*A[k,i]}, i={i}, k={k}")
            A_inv_str[i,i] += A[i,k]*A[k,i] 
            # END matrix-matrix multiplcation of U^(k-1) L^(k-1) (stored in A)

    return A_inv_str



def lu_sinv_tridiag(
    L: np.ndarray,
    U: np.ndarray,
    blocksize: int,
) -> np.ndarray:
    """ Perform a selected inversion from a lu decomposed matrix with a
    block tridiagonal structure.
    
    Parameters
    ----------
    L : np.ndarray
        Lower factor of the lu factorization fo the matrix.
    U : np.ndarray
        Upper factor of the lu factorization fo the matrix.
    blocksize : int
        The blocksize of the matrix.
    
    Returns
    -------
    X : np.ndarray
        Selected inversion of the matrix.
    """
    
    X = np.zeros(L.shape, dtype=L.dtype)

    L_blk_inv = np.zeros((blocksize, blocksize), dtype=L.dtype)
    U_blk_inv = np.zeros((blocksize, blocksize), dtype=L.dtype)

    L_blk_inv = la.solve_triangular(L[-blocksize:, -blocksize:], np.eye(blocksize), lower=True)
    U_blk_inv = la.solve_triangular(U[-blocksize:, -blocksize:], np.eye(blocksize), lower=False)
    X[-blocksize:, -blocksize:] = U_blk_inv @ L_blk_inv

    LU = L[-blocksize:, -blocksize:] + U[-blocksize:, -blocksize:] - np.eye(blocksize, dtype=L.dtype)
    # tmp = invert_LU(LU)

    # Aa = matrix_generation.generate_blocktridiag(
    #     5, 2, False, True, 63
    # )
    # Pp, Ll, Uu = la.lu(copy.deepcopy(Aa))
    # Aa_inv = sinv_ndiags_greg(copy.deepcopy(Aa), 3)

    nblocks = L.shape[0] // blocksize
    for i in range(nblocks-2, -1, -1):
        L_blk_inv = la.solve_triangular(L[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize], np.eye(blocksize), lower=True)
        U_blk_inv = la.solve_triangular(U[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize], np.eye(blocksize), lower=False)

        # X_{i+1, i} = -X_{i+1, i+1} L_{i+1, i} L_{i, i}^{-1}
        X[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize] = -X[(i+1)*blocksize:(i+2)*blocksize, (i+1)*blocksize:(i+2)*blocksize] @ L[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize] @ L_blk_inv

        # X_{i, i+1} = -U_{i, i}^{-1} U_{i, i+1} X_{i+1, i+1}
        X[i*blocksize:(i+1)*blocksize, (i+1)*blocksize:(i+2)*blocksize] = -U_blk_inv @ U[i*blocksize:(i+1)*blocksize, (i+1)*blocksize:(i+2)*blocksize] @ X[(i+1)*blocksize:(i+2)*blocksize, (i+1)*blocksize:(i+2)*blocksize]

        # X_{i, i} = (U_{i, i}^{-1} - X_{i, i+1} L_{i+1, i}) L_{i, i}^{-1}
        X[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] = (U_blk_inv - X[i*blocksize:(i+1)*blocksize, (i+1)*blocksize:(i+2)*blocksize] @ L[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize]) @ L_blk_inv

    return X



def lu_sinv_tridiag_arrowhead(
    L: np.ndarray,
    U: np.ndarray,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> np.ndarray:
    """ Perform a selected inversion from a lu decomposed matrix with a
    block tridiagonal arrowhead structure.
    
    Parameters
    ----------
    L : np.ndarray
        Lower factor of the lu factorization fo the matrix.
    U : np.ndarray
        Upper factor of the lu factorization fo the matrix.
    diag_blocksize : int
        Blocksize of the diagonals blocks of the matrix.
    arrow_blocksize : int
        Blocksize of the blocks composing the arrowhead.
    
    Returns
    -------
    X : np.ndarray
        Selected inversion of the matrix.
    """
    
    X = np.zeros(L.shape, dtype=L.dtype)
    
    L_last_blk_inv = np.zeros((arrow_blocksize, arrow_blocksize), dtype=L.dtype)
    U_last_blk_inv = np.zeros((arrow_blocksize, arrow_blocksize), dtype=L.dtype)

    L_last_blk_inv = la.solve_triangular(L[-arrow_blocksize:, -arrow_blocksize:], np.eye(arrow_blocksize), lower=True)
    U_last_blk_inv = la.solve_triangular(U[-arrow_blocksize:, -arrow_blocksize:], np.eye(arrow_blocksize), lower=False)
    
    X[-arrow_blocksize:, -arrow_blocksize:] = U_last_blk_inv @ L_last_blk_inv

    L_blk_inv = np.zeros((diag_blocksize, diag_blocksize), dtype=L.dtype)
    U_blk_inv = np.zeros((diag_blocksize, diag_blocksize), dtype=L.dtype)

    L_blk_inv = la.solve_triangular(L[-arrow_blocksize-diag_blocksize:-arrow_blocksize, -arrow_blocksize-diag_blocksize:-arrow_blocksize], np.eye(diag_blocksize), lower=True)
    U_blk_inv = la.solve_triangular(U[-arrow_blocksize-diag_blocksize:-arrow_blocksize, -arrow_blocksize-diag_blocksize:-arrow_blocksize], np.eye(diag_blocksize), lower=False)

    # X_{ndb+1, ndb} = -X_{ndb+1, ndb+1} L_{ndb+1, ndb} L_{ndb, ndb}^{-1}
    X[-arrow_blocksize:, -arrow_blocksize-diag_blocksize:-arrow_blocksize] = -X[-arrow_blocksize:, -arrow_blocksize:] @ L[-arrow_blocksize:, -arrow_blocksize-diag_blocksize:-arrow_blocksize] @ L_blk_inv

    # X_{ndb, ndb+1} = -U_{ndb, ndb}^{-1} U_{ndb, ndb+1} X_{ndb+1, ndb+1}
    X[-arrow_blocksize-diag_blocksize:-arrow_blocksize, -arrow_blocksize:] = -U_blk_inv @ U[-arrow_blocksize-diag_blocksize:-arrow_blocksize, -arrow_blocksize:] @ X[-arrow_blocksize:, -arrow_blocksize:]

    # X_{ndb, ndb} = (U_{ndb, ndb}^{-1} - X_{ndb, ndb+1} L_{ndb+1, ndb}) L_{ndb, ndb}^{-1}
    X[-arrow_blocksize-diag_blocksize:-arrow_blocksize, -arrow_blocksize-diag_blocksize:-arrow_blocksize] = (U_blk_inv - X[-arrow_blocksize-diag_blocksize:-arrow_blocksize, -arrow_blocksize:] @ L[-arrow_blocksize:, -arrow_blocksize-diag_blocksize:-arrow_blocksize]) @ L_blk_inv

    n_diag_blocks = (L.shape[0]-arrow_blocksize) // diag_blocksize 
    for i in range(n_diag_blocks-2, -1, -1):
        L_blk_inv = la.solve_triangular(L[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize], np.eye(diag_blocksize), lower=True)
        U_blk_inv = la.solve_triangular(U[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize], np.eye(diag_blocksize), lower=False)

        # --- Off-diagonal block part --- 
        # X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} - X_{i+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X[(i+1)*diag_blocksize:(i+2)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] = (-X[(i+1)*diag_blocksize:(i+2)*diag_blocksize, (i+1)*diag_blocksize:(i+2)*diag_blocksize] @ L[(i+1)*diag_blocksize:(i+2)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] - X[(i+1)*diag_blocksize:(i+2)*diag_blocksize, -arrow_blocksize:] @ L[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize]) @ L_blk_inv

        # X_{i, i+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, i+1} - U_{i, i+1} X_{i+1, i+1})
        X[i*diag_blocksize:(i+1)*diag_blocksize, (i+1)*diag_blocksize:(i+2)*diag_blocksize] = U_blk_inv @ (- U[i*diag_blocksize:(i+1)*diag_blocksize, (i+1)*diag_blocksize:(i+2)*diag_blocksize] @ X[(i+1)*diag_blocksize:(i+2)*diag_blocksize, (i+1)*diag_blocksize:(i+2)*diag_blocksize] - U[i*diag_blocksize:(i+1)*diag_blocksize, -arrow_blocksize:] @ X[-arrow_blocksize:, (i+1)*diag_blocksize:(i+2)*diag_blocksize])

        # --- Arrowhead part --- 
        # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize] = (- X[-arrow_blocksize:, (i+1)*diag_blocksize:(i+2)*diag_blocksize] @ L[(i+1)*diag_blocksize:(i+2)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] - X[-arrow_blocksize:, -arrow_blocksize:] @ L[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize]) @ L_blk_inv

        # X_{i, ndb+1} = U_{i, i}^{-1} (- U_{i, i+1} X_{i+1, ndb+1} - U_{i, ndb+1} X_{ndb+1, ndb+1})
        X[i*diag_blocksize:(i+1)*diag_blocksize, -arrow_blocksize:] = U_blk_inv @ (- U[i*diag_blocksize:(i+1)*diag_blocksize, (i+1)*diag_blocksize:(i+2)*diag_blocksize] @ X[(i+1)*diag_blocksize:(i+2)*diag_blocksize, -arrow_blocksize:] - U[i*diag_blocksize:(i+1)*diag_blocksize, -arrow_blocksize:] @ X[-arrow_blocksize:, -arrow_blocksize:]) 

        # --- Diagonal block part --- 
        # X_{i, i} = (U_{i, i}^{-1} - X_{i, i+1} L_{i+1, i} - X_{i, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] = (U_blk_inv - X[i*diag_blocksize:(i+1)*diag_blocksize, (i+1)*diag_blocksize:(i+2)*diag_blocksize] @ L[(i+1)*diag_blocksize:(i+2)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] - X[i*diag_blocksize:(i+1)*diag_blocksize, -arrow_blocksize:] @ L[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize]) @ L_blk_inv

    return X



def lu_sinv_ndiags(
    L: np.ndarray,
    U: np.ndarray,
    ndiags: int,
    blocksize: int,
) -> np.ndarray:
    """ Perform a selected inversion from a lu decomposed matrix with a
    block n-diagonals structure.
    
    Parameters
    ----------
    L : np.ndarray
        Lower factor of the lu factorization fo the matrix.
    U : np.ndarray
        Upper factor of the lu factorization fo the matrix.
    ndiags : int
        Number of diagonals.
    blocksize : int
        Size of the blocks.
    
    Returns
    -------
    X : np.ndarray
        Selected inversion of the matrix.
    """

    X = np.zeros(L.shape, dtype=L.dtype)

    L_blk_inv = np.zeros((blocksize, blocksize), dtype=L.dtype)
    U_blk_inv = np.zeros((blocksize, blocksize), dtype=L.dtype)

    nblocks = L.shape[0] // blocksize
    n_offdiags_blk = ndiags // 2
    for i in range(nblocks-1, -1, -1):
        # L_blk_inv = L_{i, i}^{-1}
        L_blk_inv = la.solve_triangular(L[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize], np.eye(blocksize), lower=True)
        # U_blk_inv = U_{i, i}^{-1}
        U_blk_inv = la.solve_triangular(U[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize], np.eye(blocksize), lower=False)


        # Off-diagonal block part
        for j in range(min(i+n_offdiags_blk, nblocks-1), i, -1):
            for k in range(i+1, min(i+n_offdiags_blk+1, nblocks), 1):
                # X_{j, i} = X_{j, i} - X_{j, k} L_{k, i}
                X[j*blocksize:(j+1)*blocksize, i*blocksize:(i+1)*blocksize] -= X[j*blocksize:(j+1)*blocksize, k*blocksize:(k+1)*blocksize] @ L[k*blocksize:(k+1)*blocksize, i*blocksize:(i+1)*blocksize]

                # X_{i, j} = X_{i, j} - U_{i, k} X_{k, j}
                X[i*blocksize:(i+1)*blocksize, j*blocksize:(j+1)*blocksize] -= U[i*blocksize:(i+1)*blocksize, k*blocksize:(k+1)*blocksize] @ X[k*blocksize:(k+1)*blocksize, j*blocksize:(j+1)*blocksize]

            # X_{j, i} = X_{j, i} L_{i, i}^{-1}
            X[j*blocksize:(j+1)*blocksize, i*blocksize:(i+1)*blocksize] = X[j*blocksize:(j+1)*blocksize, i*blocksize:(i+1)*blocksize] @ L_blk_inv
        
            # X_{i, j} = U_{i, i}^{-1} X_{i, j}
            X[i*blocksize:(i+1)*blocksize, j*blocksize:(j+1)*blocksize] = U_blk_inv @ X[i*blocksize:(i+1)*blocksize, j*blocksize:(j+1)*blocksize]


        # Diagonal block part
        # X_{i, i} = (U_{i, i}^{-1} - sum_{k=i+1}^{min(i+ndiags/2, nblocks)} X_{i, k} L_{k, i}) L_{i, i}^{-1}
        
        # X_{i, i} = U_{i, i}^{-1}
        X[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] = U_blk_inv

        for k in range(i+1, min(i+n_offdiags_blk+1, nblocks), 1):
            # X_{i, i} = X_{i, i} - X_{i, k} L_{k, i}
            X[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] -= X[i*blocksize:(i+1)*blocksize, k*blocksize:(k+1)*blocksize] @ L[k*blocksize:(k+1)*blocksize, i*blocksize:(i+1)*blocksize]

        # X_{i, i} = X_{i, i} L_{i, i}^{-1}
        X[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] = X[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] @ L_blk_inv

    return X



def lu_sinv_ndiags_arrowhead(
    L: np.ndarray,
    U: np.ndarray,
    ndiags: int,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> np.ndarray:
    """ Perform a selected inversion from a lu decomposed matrix with a
    block tridiagonal arrowhead structure.
    
    Parameters
    ----------
    L : np.ndarray
        Lower factor of the lu factorization fo the matrix.
    U : np.ndarray
        Upper factor of the lu factorization fo the matrix.
    ndiags : int
        Number of diagonals.
    diag_blocksize : int
        Blocksize of the diagonals blocks of the matrix.
    arrow_blocksize : int
        Blocksize of the blocks composing the arrowhead.
    
    Returns
    -------
    X : np.ndarray
        Selected inversion of the matrix.
    """

    X = np.zeros(L.shape, dtype=L.dtype)

    L_last_blk_inv = np.zeros((arrow_blocksize, arrow_blocksize), dtype=L.dtype)
    U_last_blk_inv = np.zeros((arrow_blocksize, arrow_blocksize), dtype=L.dtype)

    L_last_blk_inv = la.solve_triangular(L[-arrow_blocksize:, -arrow_blocksize:], np.eye(arrow_blocksize), lower=True)
    U_last_blk_inv = la.solve_triangular(U[-arrow_blocksize:, -arrow_blocksize:], np.eye(arrow_blocksize), lower=False)
    
    X[-arrow_blocksize:, -arrow_blocksize:] = U_last_blk_inv @ L_last_blk_inv

    L_blk_inv = np.zeros((diag_blocksize, diag_blocksize), dtype=L.dtype)
    U_blk_inv = np.zeros((diag_blocksize, diag_blocksize), dtype=L.dtype)

    n_diag_blocks = (L.shape[0]-arrow_blocksize) // diag_blocksize 
    n_offdiags_blk = ndiags // 2
    for i in range(n_diag_blocks-1, -1, -1):
        # L_blk_inv = L_{i, i}^{-1}
        L_blk_inv = la.solve_triangular(L[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize], np.eye(diag_blocksize), lower=True)

        # U_blk_inv = U_{i, i}^{-1}
        U_blk_inv = la.solve_triangular(U[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize], np.eye(diag_blocksize), lower=False)


        # Arrowhead part
        # X_{ndb+1, i} = - X_{ndb+1, ndb+1} L_{ndb+1, i}
        X[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize] = - X[-arrow_blocksize:, -arrow_blocksize:] @ L[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize]

        # X_{i, ndb+1} = - U_{i, ndb+1} X_{ndb+1, ndb+1}
        X[i*diag_blocksize:(i+1)*diag_blocksize, -arrow_blocksize:] = - U[i*diag_blocksize:(i+1)*diag_blocksize, -arrow_blocksize:] @ X[-arrow_blocksize:, -arrow_blocksize:]

        for k in range(i+1, min(i+n_offdiags_blk+1, n_diag_blocks), 1):
            # X_{ndb+1, i} = X_{ndb+1, i} - X_{ndb+1, k} L_{k, i}
            X[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize] -= X[-arrow_blocksize:, k*diag_blocksize:(k+1)*diag_blocksize] @ L[k*diag_blocksize:(k+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize]

            # X_{i, ndb+1} = X_{i, ndb+1} - U_{i, k} X_{k, ndb+1}
            X[i*diag_blocksize:(i+1)*diag_blocksize, -arrow_blocksize:] -= U[i*diag_blocksize:(i+1)*diag_blocksize, k*diag_blocksize:(k+1)*diag_blocksize] @ X[k*diag_blocksize:(k+1)*diag_blocksize, -arrow_blocksize:]

        # X_{ndb+1, i} = X_{ndb+1, i} L_{i, i}^{-1}
        X[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize] = X[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize] @ L_blk_inv

        # X_{i, ndb+1} = U_{i, i}^{-1} X_{i, ndb+1}
        X[i*diag_blocksize:(i+1)*diag_blocksize, -arrow_blocksize:] = U_blk_inv @ X[i*diag_blocksize:(i+1)*diag_blocksize, -arrow_blocksize:]


        # Off-diagonal block part
        for j in range(min(i+n_offdiags_blk, n_diag_blocks-1), i, -1):
            # Take the effect of the arrowhead part into account
            # X_{j, i} = - X_{j, ndb+1} L_{ndb+1, i}
            X[j*diag_blocksize:(j+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] = - X[j*diag_blocksize:(j+1)*diag_blocksize, -arrow_blocksize:] @ L[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize]

            # X_{i, j} = - U_{i, ndb+1} X_{ndb+1, j}
            X[i*diag_blocksize:(i+1)*diag_blocksize, j*diag_blocksize:(j+1)*diag_blocksize] = - U[i*diag_blocksize:(i+1)*diag_blocksize, -arrow_blocksize:] @ X[-arrow_blocksize:, j*diag_blocksize:(j+1)*diag_blocksize]

            for k in range(i+1, min(i+n_offdiags_blk+1, n_diag_blocks), 1):
                # X_{j, i} = X_{j, i} - X_{j, k} L_{k, i}
                X[j*diag_blocksize:(j+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] -= X[j*diag_blocksize:(j+1)*diag_blocksize, k*diag_blocksize:(k+1)*diag_blocksize] @ L[k*diag_blocksize:(k+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize]

                # X_{i, j} = X_{i, j} - U_{i, k} X_{k, j}
                X[i*diag_blocksize:(i+1)*diag_blocksize, j*diag_blocksize:(j+1)*diag_blocksize] -= U[i*diag_blocksize:(i+1)*diag_blocksize, k*diag_blocksize:(k+1)*diag_blocksize] @ X[k*diag_blocksize:(k+1)*diag_blocksize, j*diag_blocksize:(j+1)*diag_blocksize]

            # X_{j, i} = X_{j, i} L_{i, i}^{-1}
            X[j*diag_blocksize:(j+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] = X[j*diag_blocksize:(j+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] @ L_blk_inv
        
            # X_{i, j} = U_{i, i}^{-1} X_{i, j}
            X[i*diag_blocksize:(i+1)*diag_blocksize, j*diag_blocksize:(j+1)*diag_blocksize] = U_blk_inv @ X[i*diag_blocksize:(i+1)*diag_blocksize, j*diag_blocksize:(j+1)*diag_blocksize]


        # Diagonal block part
        # X_{i, i} = (U_{i, i}^{-1} - X_{i, ndb+1} L_{ndb+1, i} - sum_{k=i+1}^{min(i+ndiags/2, n_diag_blocks)} X_{i, k} L_{k, i}) L_{i, i}^{-1}
        
        # X_{i, i} = U_{i, i}^{-1} - X_{i, ndb+1} L_{ndb+1, i}
        X[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] = U_blk_inv - X[i*diag_blocksize:(i+1)*diag_blocksize, -arrow_blocksize:] @ L[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize]

        for k in range(i+1, min(i+n_offdiags_blk+1, n_diag_blocks), 1):
            # X_{i, i} = X_{i, i} - X_{i, k} L_{k, i}
            X[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] -= X[i*diag_blocksize:(i+1)*diag_blocksize, k*diag_blocksize:(k+1)*diag_blocksize] @ L[k*diag_blocksize:(k+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize]

        # X_{i, i} = X_{i, i} L_{i, i}^{-1}
        X[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] = X[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] @ L_blk_inv
   
    return X

