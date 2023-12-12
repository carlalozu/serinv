"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Lisa Gaedke-Merzhaeuser  (lisa.gaedke.merzhaeuser@usi.ch)
@date: 2023-11

Contains the cholesky selected inversion routines.

Copyright 2023 ETH Zurich and USI. All rights reserved.
"""


import numpy as np
import scipy.linalg as la



def chol_sinv_tridiag(
    L: np.ndarray,
    blocksize: int,
) -> np.ndarray:
    """ Perform a selected inversion from a cholesky decomposed matrix with a
    block tridiagonal structure.
    
    Parameters
    ----------
    L : np.ndarray
        The cholesky factorization of the matrix.
    blocksize : int
        The blocksize of the matrix.
    
    Returns
    -------
    X : np.ndarray
        Selected inversion of the matrix.
    """
    
    X = np.zeros(L.shape, dtype=L.dtype)

    nblocks = L.shape[0] // blocksize

    L_blk_inv = np.zeros((blocksize, blocksize), dtype=L.dtype)
    L_blk_inv = la.solve_triangular(L[-blocksize:, -blocksize:], np.eye(blocksize), lower=True)
    X[-blocksize:, -blocksize:] = L_blk_inv.T @ L_blk_inv

    for i in range(nblocks-2, -1, -1):
        L_blk_inv = la.solve_triangular(L[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize], np.eye(blocksize), lower=True)
        
        # X_{i+1, i} = -X_{i+1, i+1} L_{i+1, i} L_{i, i}^{-1}
        X[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize] = -X[(i+1)*blocksize:(i+2)*blocksize, (i+1)*blocksize:(i+2)*blocksize] @ L[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize] @ L_blk_inv

        # X_{i, i+1} = X_{i+1, i}.T
        X[i*blocksize:(i+1)*blocksize, (i+1)*blocksize:(i+2)*blocksize] = X[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize].T

        # X_{i, i} = (L_{i, i}^{-T} - X_{i+1, i}^{T} L_{i+1, i}) L_{i, i}^{-1}
        X[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] = (L_blk_inv.T - X[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize].T @ L[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize]) @ L_blk_inv

    return X



def chol_sinv_tridiag_arrowhead(
    L: np.ndarray,
    diag_blocksize: int,
    arrow_blocksize: int,
) -> np.ndarray:
    """ Perform a selected inversion from a cholesky decomposed matrix with a
    block tridiagonal arrowhead structure.
    
    Parameters
    ----------
    L : np.ndarray
        The cholesky factorization of the matrix.
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
    L_last_blk_inv = la.solve_triangular(L[-arrow_blocksize:, -arrow_blocksize:], np.eye(arrow_blocksize), lower=True)
    X[-arrow_blocksize:, -arrow_blocksize:] = L_last_blk_inv.T @ L_last_blk_inv

    L_blk_inv = np.zeros((diag_blocksize, diag_blocksize), dtype=L.dtype)
    L_blk_inv = la.solve_triangular(L[-arrow_blocksize-diag_blocksize:-arrow_blocksize, -arrow_blocksize-diag_blocksize:-arrow_blocksize], np.eye(diag_blocksize), lower=True)

    # X_{ndb+1, ndb} = -X_{ndb+1, ndb+1} L_{ndb+1, ndb} L_{ndb, ndb}^{-1}
    X[-arrow_blocksize:, -arrow_blocksize-diag_blocksize:-arrow_blocksize] = -X[-arrow_blocksize:, -arrow_blocksize:] @ L[-arrow_blocksize:, -arrow_blocksize-diag_blocksize:-arrow_blocksize] @ L_blk_inv

    # X_{ndb, ndb+1} = X_{ndb+1, ndb}.T
    X[-arrow_blocksize-diag_blocksize:-arrow_blocksize, -arrow_blocksize:] = X[-arrow_blocksize:, -arrow_blocksize-diag_blocksize:-arrow_blocksize].T

    # X_{ndb, ndb} = (L_{ndb, ndb}^{-T} - X_{ndb+1, ndb}^{T} L_{ndb+1, ndb}) L_{ndb, ndb}^{-1}
    X[-arrow_blocksize-diag_blocksize:-arrow_blocksize, -arrow_blocksize-diag_blocksize:-arrow_blocksize] = (L_blk_inv.T - X[-arrow_blocksize:, -arrow_blocksize-diag_blocksize:-arrow_blocksize].T @ L[-arrow_blocksize:, -arrow_blocksize-diag_blocksize:-arrow_blocksize]) @ L_blk_inv

    n_diag_blocks = (L.shape[0]-arrow_blocksize) // diag_blocksize 
    for i in range(n_diag_blocks-2, -1, -1):
        L_blk_inv = la.solve_triangular(L[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize], np.eye(diag_blocksize), lower=True)
        
        # --- Off-diagonal block part --- 
        # X_{i+1, i} = (-X_{i+1, i+1} L_{i+1, i} - X_{ndb+1, i+1}^{T} L_{ndb+1, i}) L_{i, i}^{-1}
        X[(i+1)*diag_blocksize:(i+2)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] = (-X[(i+1)*diag_blocksize:(i+2)*diag_blocksize, (i+1)*diag_blocksize:(i+2)*diag_blocksize] @ L[(i+1)*diag_blocksize:(i+2)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] - X[-arrow_blocksize:, (i+1)*diag_blocksize:(i+2)*diag_blocksize].T @ L[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize]) @ L_blk_inv

        # X_{i, i+1} = X_{i+1, i}.T
        X[i*diag_blocksize:(i+1)*diag_blocksize, (i+1)*diag_blocksize:(i+2)*diag_blocksize] = X[(i+1)*diag_blocksize:(i+2)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize].T

        # --- Arrowhead part --- 
        # X_{ndb+1, i} = (- X_{ndb+1, i+1} L_{i+1, i} - X_{ndb+1, ndb+1} L_{ndb+1, i}) L_{i, i}^{-1}
        X[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize] = (- X[-arrow_blocksize:, (i+1)*diag_blocksize:(i+2)*diag_blocksize] @ L[(i+1)*diag_blocksize:(i+2)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] - X[-arrow_blocksize:, -arrow_blocksize:] @ L[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize]) @ L_blk_inv

        # X_{i, ndb+1} = X_{ndb+1, i}.T
        X[i*diag_blocksize:(i+1)*diag_blocksize, -arrow_blocksize:] = X[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize].T

        # --- Diagonal block part --- 
        # X_{i, i} = (L_{i, i}^{-T} - X_{i+1, i}^{T} L_{i+1, i} - X_{ndb+1, i}.T L_{ndb+1, i}) L_{i, i}^{-1}
        X[i*diag_blocksize:(i+1)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] = (L_blk_inv.T - X[(i+1)*diag_blocksize:(i+2)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize].T @ L[(i+1)*diag_blocksize:(i+2)*diag_blocksize, i*diag_blocksize:(i+1)*diag_blocksize] - X[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize].T @ L[-arrow_blocksize:, i*diag_blocksize:(i+1)*diag_blocksize]) @ L_blk_inv

    return X