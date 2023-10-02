from pyomo.contrib.pynumero.sparse.mpi_block_matrix import MPIBlockMatrix
from pyomo.contrib.pynumero.sparse.mpi_block_vector import MPIBlockVector
from parapint.linalg.base_linear_solver_interface import LinearSolverInterface
from parapint.linalg.results import LinearSolverStatus, LinearSolverResults
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import cg, LinearOperator
from mpi4py import MPI
import itertools
from .explicit_schur_complement import _process_sub_results
from typing import Dict, Optional, List
from pyomo.common.timing import HierarchicalTimer

from .mpi_explicit_schur_complement import _gather_results, _BorderMatrix, _get_all_nonzero_elements_in_sc, \
      _get_all_nonzero_elements_in_sc, _process_sub_results

comm: MPI.Comm = MPI.COMM_WORLD
rank: int = comm.Get_rank()
size: int = comm.Get_size()


class MPIImplicitSchurComplementLinearSolver(LinearSolverInterface):
    """

    Solve the system Ax = b.

    A must be block-bordered-diagonal and symmetric::

      K1          transpose(A1)
          K2      transpose(A2)
              K3  transpose(A3)
      A1  A2  A3  Q

    Only the lower diagonal needs supplied.

    Some assumptions are made on the block matrices provided to do_symbolic_factorization and do_numeric_factorization:
      * Q must be owned by all processes
      * K :sub:`i` and A :sub:`i` must be owned by the same process

    Parameters
    ----------
    subproblem_solvers: dict
        Dictionary mapping block index to linear solver
    schur_complement_solver: LinearSolverInterface
        Linear solver to use for factorizing the schur complement

    """
    def __init__(self, subproblem_solvers: Dict[int, LinearSolverInterface],
                 schur_complement_solver: LinearSolverInterface):
        self.subproblem_solvers = subproblem_solvers
        #self.schur_complement_solver = schur_complement_solver
        self.block_dim = 0
        self.block_matrix = None
        self.local_block_indices = list()
        self.schur_complement = coo_matrix((0, 0))
        self.border_matrices: Dict[int, _BorderMatrix] = dict()
        self.sc_data_slices = dict()
        self._preconditioner = None

    def do_symbolic_factorization(self,
                                  matrix: MPIBlockMatrix,
                                  raise_on_error: bool = True,
                                  timer: Optional[HierarchicalTimer] = None) -> LinearSolverResults:
        """
        Perform symbolic factorization. This performs symbolic factorization for each diagonal block and
        collects some information on the structure of the schur complement for sparse communication in
        the numeric factorization phase.

        Parameters
        ----------
        matrix: MPIBlockMatrix
            A Pynumero MPIBlockMatrix. This is the A matrix in Ax=b
        raise_on_error: bool
            If False, an error will not be raised if an error occurs during symbolic factorization. Instead the
            status attribute of the results object will indicate an error ocurred.
        timer: HierarchicalTimer
            A timer for profiling.

        Returns
        -------
        res: LinearSolverResults
            The results object
        """
        if timer is None:
            timer = HierarchicalTimer()

        block_matrix = matrix
        nbrows, nbcols = block_matrix.bshape
        if nbrows != nbcols:
            raise ValueError('The block matrix provided is not square.')
        self.block_dim = nbrows

        # split up the blocks between ranks
        self.local_block_indices = list()
        for ndx in range(self.block_dim - 1):
            if ((block_matrix.rank_ownership[ndx, ndx] == rank) or
                    (block_matrix.rank_ownership[ndx, ndx] == -1 and rank == 0)):
                self.local_block_indices.append(ndx)

        res = LinearSolverResults()
        res.status = LinearSolverStatus.successful
        timer.start('factorize')
        for ndx in self.local_block_indices:
            sub_res = self.subproblem_solvers[ndx].do_symbolic_factorization(matrix=block_matrix.get_block(ndx, ndx),
                                                                             raise_on_error=False)
            _process_sub_results(res, sub_res)
            if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
                break
        timer.stop('factorize')
        res = _gather_results(res)
        if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
            if raise_on_error:
                raise RuntimeError('Symbolic factorization unsuccessful; status: ' + str(res.status))
            else:
                return res

        timer.start('sc_structure')
        self._get_sc_structure(block_matrix=block_matrix, timer=timer)
        timer.stop('sc_structure')

        return res

    def _get_sc_structure(self, block_matrix, timer):
        """
        Parameters
        ----------
        block_matrix: pyomo.contrib.pynumero.sparse.mpi_block_matrix.MPIBlockMatrix
        """
        timer.start('build_border_matrices')
        self.border_matrices = dict()
        for ndx in self.local_block_indices:
            self.border_matrices[ndx] = _BorderMatrix(block_matrix.get_block(self.block_dim - 1, ndx))
        timer.stop('build_border_matrices')
        timer.start('gather_all_nonzero_elements')
        nonzero_rows, nonzero_cols = _get_all_nonzero_elements_in_sc(self.border_matrices)
        timer.stop('gather_all_nonzero_elements')
        timer.start('construct_schur_complement')
        sc_nnz = nonzero_rows.size
        sc_dim = block_matrix.get_row_size(self.block_dim - 1)
        sc_values = np.zeros(sc_nnz, dtype=np.double)
        self.schur_complement = coo_matrix((sc_values, (nonzero_rows, nonzero_cols)), shape=(sc_dim, sc_dim))
        timer.stop('construct_schur_complement')
        timer.start('get_sc_data_slices')
        self.sc_data_slices = dict()
        for ndx in self.local_block_indices:
            self.sc_data_slices[ndx] = dict()
            border_matrix = self.border_matrices[ndx]
            for row_ndx in border_matrix.nonzero_rows:
                self.sc_data_slices[ndx][row_ndx] = np.bitwise_and(nonzero_cols == row_ndx, np.isin(nonzero_rows, border_matrix.nonzero_rows)).nonzero()[0]
        timer.stop('get_sc_data_slices')

    def do_numeric_factorization(self,
                                 matrix: MPIBlockMatrix,
                                 raise_on_error: bool = True,
                                 timer: Optional[HierarchicalTimer] = None) -> LinearSolverResults:
        """
        Perform numeric factorization:
          * perform numeric factorization on each diagonal block
          * form and communicate the Schur-Complement
          * factorize the schur-complement

        This method should only be called after do_symbolic_factorization.

        Parameters
        ----------
        matrix: MPIBlockMatrix
            A Pynumero MPIBlockMatrix. This is the A matrix in Ax=b
        raise_on_error: bool
            If False, an error will not be raised if an error occurs during symbolic factorization. Instead the
            status attribute of the results object will indicate an error ocurred.
        timer: HierarchicalTimer
            A timer for profiling.

        Returns
        -------
        res: LinearSolverResults
            The results object
        """
        if timer is None:
            timer = HierarchicalTimer()

        self.block_matrix = block_matrix = matrix

        res = LinearSolverResults()
        res.status = LinearSolverStatus.successful
        #timer.start('form SC')
        for ndx in self.local_block_indices:
            timer.start('factorize')
            sub_res = self.subproblem_solvers[ndx].do_numeric_factorization(matrix=block_matrix.get_block(ndx, ndx),
                                                                            raise_on_error=False)
            timer.stop('factorize')
            _process_sub_results(res, sub_res)
            if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
                break
        res = _gather_results(res)
        if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
            if raise_on_error:
                raise RuntimeError('Numeric factorization unsuccessful; status: ' + str(res.status))
            else:
                #timer.stop('form SC')
                return res



        return res

    def do_back_solve(self, rhs, timer=None, barrier=None):
        """
        Performs a back solve with the factorized matrix. Should only be called after
        do_numeric_factorixation.

        Parameters
        ----------
        rhs: MPIBlockVector
        timer: HierarchicalTimer

        Returns
        -------
        result: MPIBlockVector
        """
        if timer is None:
            timer = HierarchicalTimer()

        schur_complement_rhs = np.zeros(rhs.get_block(self.block_dim - 1).size, dtype='d')
        for ndx in self.local_block_indices:
            A = self.block_matrix.get_block(self.block_dim-1, ndx)
            contribution = self.subproblem_solvers[ndx].do_back_solve(rhs.get_block(ndx))
            schur_complement_rhs -= A.tocsr().dot(contribution.flatten())
        res = np.zeros(rhs.get_block(self.block_dim - 1).shape[0], dtype='d')
        comm.Allreduce(schur_complement_rhs, res)
        schur_complement_rhs = rhs.get_block(self.block_dim - 1) + res

        result = rhs.copy_structure()

        # TODO: Set tolerance according to rule from paper
        # TODO: LinearSolveroptions  - also relevant for preconditioner
        if barrier is None:
            tol = 1e-8
        else:
            tol = barrier
        coupling, info = self.pcg_schur_solve(schur_complement_rhs=schur_complement_rhs, timer=timer, tol=tol)

        #coupling = self.schur_complement_solver.do_back_solve(schur_complement_rhs)

        for ndx in self.local_block_indices:
            A = self.block_matrix.get_block(self.block_dim-1, ndx)
            result.set_block(ndx, self.subproblem_solvers[ndx].do_back_solve(rhs.get_block(ndx) -
                                                                             A.tocsr().transpose().dot(coupling.flatten())))

        result.set_block(self.block_dim-1, coupling)


        return result

    def get_inertia(self):
        """
        Get the inertia. Should only be called after do_numeric_factorization.

        Returns
        -------
        num_pos: int
            The number of positive eigenvalues of A
        num_neg: int
            The number of negative eigenvalues of A
        num_zero: int
            The number of zero eigenvalues of A
        """
        num_pos = 0
        num_neg = 0
        num_zero = 0

        for ndx in self.local_block_indices:
            _pos, _neg, _zero = self.subproblem_solvers[ndx].get_inertia()
            num_pos += _pos
            num_neg += _neg
            num_zero += _zero

        num_pos = comm.allreduce(num_pos)
        num_neg = comm.allreduce(num_neg)
        num_zero = comm.allreduce(num_zero)

        #_pos, _neg, _zero = self.schur_complement_solver.get_inertia()
        #num_pos += _pos
        #num_neg += _neg
        #num_zero += _zero

        return num_pos, num_neg, num_zero

    def increase_memory_allocation(self, factor):
        """
        Increases the memory allocation of each sub-solver. This method should only be called
        if the results status from do_symbolic_factorization or do_numeric_factorization is
        LinearSolverStatus.not_enough_memory.

        Parameters
        ----------
        factor: float
            The factor by which to increase memory allocation. Should be greater than 1.
        """
        for ndx in self.local_block_indices:
            sub_solver = self.subproblem_solvers[ndx]
            sub_solver.increase_memory_allocation(factor=factor)


    def pcg_schur_solve(self, schur_complement_rhs, timer=None, tol=1e-8):

        if timer is None:
            timer = HierarchicalTimer()

        timer.start('pcg')
        D = self.block_matrix.get_block(self.block_dim-1, self.block_dim-1).tocoo()

        def schur_matvec_mpi(u):
            res = np.zeros_like(u)
            for ndx in self.local_block_indices:
                border_matrix: _BorderMatrix = self.border_matrices[ndx]
                A = border_matrix.csr.transpose()
                #A = self.block_matrix.get_block(self.block_dim-1, ndx).toarray()
                #A = self.block_matrix.get_block(ndx,self.block_dim-1)
                timer.start('dot_product')
                v = A.dot(u)
                timer.stop('dot_product')
                timer.start('back_solve')
                x = self.subproblem_solvers[ndx].do_back_solve(v)
                timer.stop('back_solve')
                timer.start('dot_product')
                y = A.transpose().dot(x)
                timer.stop('dot_product')
                res -= y

            timer.start('communicate')
            res_global = np.empty(res.size)
            comm.Allreduce(res, res_global)
            timer.stop('communicate')
            res_global += D.dot(u)
            return res_global


        if self._preconditioner is None:
            precond_mv = lambda u: u
        else:
            precond_mv = lambda u: self._preconditioner.dot(u)
                #L-BFGS update based on s_iterates, y_iterates


            

        lin_op = LinearOperator(shape=D.shape, matvec=schur_matvec_mpi, dtype=float)

        precond = LinearOperator(shape=D.shape, matvec=precond_mv, dtype=float)

        #iters = 0
        # def nonlocal_iterate(xk):
        #     nonlocal rank
        #     if rank == 0:
        #         nonlocal iters
        #         iters+=1
        from numpy.typing import NDArray
        from collections import deque
        ns: int = D.shape[0]
        mem_len: int = 10
        prev_x = np.zeros_like(schur_complement_rhs)
        #s_iterates: List[NDArray] = list()
        s_iterates = deque(maxlen=mem_len)
        pcg_iter: int = 0
        mem_len: int = 10
        def save_delta_x(xk):
            timer.start('precondition')
            nonlocal prev_x
            nonlocal pcg_iter
            pcg_iter += 1
            delta_x = xk - prev_x
            s_iterates.appendleft(delta_x)
            prev_x = xk
            timer.stop('precondition')

        coupling, info = cg(A=lin_op, b=schur_complement_rhs, tol=1e-8, callback=save_delta_x, maxiter=ns)

        # timer.start('precondition')
        y_iterates = deque([schur_matvec_mpi(s_iterate) for s_iterate in s_iterates])
        sl = s_iterates.popleft()
        yl = y_iterates.popleft()
        M = (sl.T.dot(yl) / yl.T.dot(yl)) * np.eye(ns)
        V = np.empty_like(M)
        while s_iterates and y_iterates:
            s = s_iterates.popleft()
            y = y_iterates.popleft()
            if y.T.dot(s) > 0:
                rho = 1.0 / (y.T.dot(s))
                V = np.eye(ns) - rho * y.dot(s.T)
                M = V.T.dot(M).dot(V) + rho * s.dot(s.T)

        self._preconditioner = M
        # self._y_iterates = y_iterates
        # self._s_iterates = s_iterates
        # timer.stop('precondition')
        #print(f'num_iter: {iters}')
        #print(f'info: {info}')
        timer.stop('pcg')
        return coupling, info