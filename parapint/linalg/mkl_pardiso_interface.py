from .base_linear_solver_interface import LinearSolverInterface
from .results import LinearSolverStatus, LinearSolverResults
from .mkl_pardiso import MKLPardisoInterface
from scipy.sparse import isspmatrix_coo, isspmatrix_csr, tril, spmatrix
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
from pyomo.common.timing import HierarchicalTimer
from typing import Union, Tuple, Optional
import numpy as np


class InteriorPointMKLPardisoInterface(LinearSolverInterface):
    def __init__(self, iparm=None):
        self._pardiso = MKLPardisoInterface()
        if iparm is not None:
            for k, v in iparm.items():
                self.set_iparm(k, v)
        self._dim = None
        self._num_status = None

    @classmethod
    def getLoggerName(cls):
        return 'mkl_pardiso'

    def do_symbolic_factorization(
        self, matrix: Union[spmatrix, BlockMatrix], raise_on_error: bool = True
    ) -> LinearSolverResults:
        self._num_status = None
        if not isspmatrix_csr(matrix):
            matrix = matrix.tocsr()
        if not matrix.has_sorted_indices:
            matrix.sort_indices()
        nrows, ncols = matrix.shape
        if nrows != ncols:
            raise ValueError("Matrix must be square")
        self._dim = nrows

        stat = self._pardiso.do_symbolic_factorization(a=matrix.data, ia=matrix.indptr + 1, ja=matrix.indices + 1)
        res = LinearSolverResults()
        if stat == 0:
            res.status = LinearSolverStatus.successful
        else:
            if raise_on_error:
                raise RuntimeError(
                    "Symbolic factorization was not successful; return code: "
                    + str(stat)
                )
            if stat in {-2}:
                res.status = LinearSolverStatus.not_enough_memory
            elif stat in {-7}:
                res.status = LinearSolverStatus.singular
            else:
                res.status = LinearSolverStatus.error
        return res

    def do_numeric_factorization(
        self, matrix: Union[spmatrix, BlockMatrix], raise_on_error: bool = True
    ) -> LinearSolverResults:
        if not isspmatrix_csr(matrix):
            matrix = matrix.tocsr()
        if not matrix.has_sorted_indices:
            matrix.sort_indices()
        nrows, ncols = matrix.shape
        if nrows != ncols:
            raise ValueError("Matrix must be square")
        if nrows != self._dim:
            raise ValueError(
                "Matrix dimensions do not match the dimensions of "
                "the matrix used for symbolic factorization"
            )

        stat = self._pardiso.do_numeric_factorization(a=matrix.data, ia=matrix.indptr + 1, ja=matrix.indices + 1)
        res = LinearSolverResults()
        if stat == 0:
            res.status = LinearSolverStatus.successful
        else:
            if raise_on_error:
                raise RuntimeError(
                    "Numeric factorization was not successful; return code: "
                    + str(stat)
                )
            if stat in {-2}:
                res.status = LinearSolverStatus.not_enough_memory
            elif stat in {-7}:
                res.status = LinearSolverStatus.singular
            else:
                res.status = LinearSolverStatus.error

        self._num_status = res.status

        return res
    
    def increase_memory_allocation(self, factor):
        """
        Increas the memory allocation for factorization. This method should only be called
        if the results status from do_symbolic_factorization or do_numeric_factorization is
        LinearSolverStatus.not_enough_memory.

        Parameters
        ----------
        factor: float
            The factor by which to increase memory allocation. Should be greater than 1.
        """
        # TODO: Determine how to allocate memory using pardiso
        #self._pardiso._mem_factor = factor
        pass

    def do_back_solve(
        self, rhs: Union[np.ndarray, BlockVector], raise_on_error: bool = True
    ) -> Tuple[Optional[Union[np.ndarray, BlockVector]], LinearSolverResults]:
        if self._num_status is None:
            raise RuntimeError('Must call do_numeric_factorization before do_back_solve can be called')
        if self._num_status != LinearSolverStatus.successful:
            raise RuntimeError('Can only call do_back_solve if the numeric factorization was successful.')
        
        if isinstance(rhs, BlockVector):
            _rhs = rhs.flatten()
            result = _rhs
        else:
            result = rhs.copy()

        result = self._pardiso.do_backsolve(result, copy=False)

        if isinstance(rhs, BlockVector):
            _result = rhs.copy_structure()
            _result.copyfrom(result)
            result = _result

        return result

    def set_iparm(self, key, value):
        self._pardiso.set_iparm(key, value)

    def get_iparm(self, key):
        return self._pardiso.get_iparm(key)

    def get_inertia(self):
        if self._num_status is None:
            raise RuntimeError('Must call do_numeric_factorization before inertia can be computed')
        if self._num_status != LinearSolverStatus.successful:
            raise RuntimeError('Can only compute inertia if the numeric factorization was successful.')
        num_negative_eigenvalues = self.get_iparm(23)
        num_positive_eigenvalues = self.get_iparm(22)
        return (num_positive_eigenvalues, num_negative_eigenvalues, 0)
