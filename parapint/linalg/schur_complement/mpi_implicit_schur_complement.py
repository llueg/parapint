from pyomo.contrib.pynumero.sparse.mpi_block_matrix import MPIBlockMatrix
from pyomo.contrib.pynumero.sparse.mpi_block_vector import MPIBlockVector
from parapint.linalg.base_linear_solver_interface import LinearSolverInterface
from parapint.linalg.results import LinearSolverStatus, LinearSolverResults
import numpy as np
from numpy.typing import NDArray
import scipy.sparse.linalg 
import scipy.sparse as sps
from scipy.sparse import coo_matrix, csr_matrix
from mpi4py import MPI
import itertools
from .explicit_schur_complement import _process_sub_results
from typing import Dict, Optional, List, Callable
from pyomo.common.timing import HierarchicalTimer
from parapint.linalg.schur_complement.utils import _gather_results, _BorderMatrix, _get_all_nonzero_elements_in_sc_using_ix
from parapint.linalg.iterative.pcg import pcg_solve, PcgOptions, PcgSolution, PcgSolutionStatus, LbfgsInvHessProduct


comm: MPI.Comm = MPI.COMM_WORLD
rank: int = comm.Get_rank()
size: int = comm.Get_size()



class MPISchurComplementUtilMixin:

    def __init__(self, subproblem_solvers: Dict[int, LinearSolverInterface]):
        self.subproblem_solvers: Dict[int, LinearSolverInterface] = subproblem_solvers
        self.block_dim = 0
        self.block_matrix = None
        self.local_block_indices = list()
        self._local_block_indices_for_numeric_factorization = list()
        self.schur_complement = coo_matrix((0, 0))
        self._current_schur_complement = coo_matrix((0, 0))
        self.border_matrices: Dict[int, _BorderMatrix] = dict()
        self.sc_data_slices = dict()
    
    def _get_full_sc_structure(self, block_matrix: MPIBlockMatrix, timer: HierarchicalTimer):
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
        nonzero_rows, nonzero_cols = _get_all_nonzero_elements_in_sc_using_ix(self.border_matrices, self.local_block_indices, self.block_dim - 1)
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

    def _form_full_sc(self, timer) -> None:
        timer.start('form SC')
        self.schur_complement.data = np.zeros(self.schur_complement.data.size, dtype=np.double)
        for ndx in self.local_block_indices:
            border_matrix: _BorderMatrix = self.border_matrices[ndx]
            A = border_matrix.csr
            _rhs = np.zeros(A.shape[1], dtype=np.double)
            solver = self.subproblem_solvers[ndx]
            for row_ndx in border_matrix.nonzero_rows:
                for indptr in range(A.indptr[row_ndx], A.indptr[row_ndx + 1]):
                    col = A.indices[indptr]
                    val = A.data[indptr]
                    _rhs[col] += val
                timer.start('back solve')
                contribution = solver.do_back_solve(_rhs)
                timer.stop('back solve')
                timer.start('dot product')
                contribution = A.dot(contribution)
                timer.stop('dot product')
                self.schur_complement.data[self.sc_data_slices[ndx][row_ndx]] -= contribution[border_matrix.nonzero_rows]
                for indptr in range(A.indptr[row_ndx], A.indptr[row_ndx + 1]):
                    col = A.indices[indptr]
                    val = A.data[indptr]
                    _rhs[col] -= val

        timer.start('communicate')
        sc = np.zeros(self.schur_complement.data.size, dtype=np.double)
        timer.start('Barrier')
        comm.Barrier()
        timer.stop('Barrier')
        timer.start('Allreduce')
        comm.Allreduce(self.schur_complement.data, sc)
        timer.stop('Allreduce')
        self.schur_complement.data = sc
        #self.schur_complement = self.schur_complement + self.block_matrix.get_block(self.block_dim-1, self.block_dim-1).tocoo()
        self._current_schur_complement = self.schur_complement + self.block_matrix.get_block(self.block_dim-1, self.block_dim-1).tocoo()
        timer.stop('communicate')
        timer.stop('form SC')

    def _symbolic_factorize_diag_blocks(self,
                                        block_matrix: MPIBlockMatrix,
                                        timer: HierarchicalTimer
                                        ) -> LinearSolverResults:
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
        return res

    def _numeric_factorize_diag_blocks(self,
                                       block_matrix: MPIBlockMatrix,
                                       timer: HierarchicalTimer
                                       ) -> LinearSolverResults:
        res = LinearSolverResults()
        res.status = LinearSolverStatus.successful

        for ndx in self._local_block_indices_for_numeric_factorization:
            timer.start('factorize')
            sub_res = self.subproblem_solvers[ndx].do_numeric_factorization(matrix=block_matrix.get_block(ndx, ndx),
                                                                            raise_on_error=False)
            timer.stop('factorize')
            _process_sub_results(res, sub_res)
            if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
                break
        res = _gather_results(res)
        return res

    def _symbolic_numeric_factorize_sc(self,
                                       sc,
                                       sc_solver: LinearSolverInterface,
                                       timer
                                       ) -> LinearSolverResults:
            timer.start('factor SC')
            res = sc_solver.do_symbolic_factorization(sc, raise_on_error=False)
            if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
                timer.stop('factor SC')
                return res
            sub_res = sc_solver.do_numeric_factorization(sc)
            _process_sub_results(res, sub_res)
            timer.stop('factor SC')
            return res

class MPIBaseImplicitSchurComplementLinearSolver(LinearSolverInterface, MPISchurComplementUtilMixin):
    """

    Avoids forming S entirely, uses PCG to solve SC system. WIP, focusing on DD preconditioner for now.


    """
    def __init__(self,
                 subproblem_solvers: Dict[int, LinearSolverInterface],
                 options: Dict):
        super().__init__(subproblem_solvers=subproblem_solvers)

        self.pcg_options: PcgOptions = options['pcg']
        self.precond_options: Dict = options['preconditioner']
        self._flag_form_sc = False
        self._flag_factorize_sc = False

    def do_symbolic_factorization(self,
                                  matrix: MPIBlockMatrix,
                                  raise_on_error: bool = True,
                                  timer: Optional[HierarchicalTimer] = None) -> LinearSolverResults:

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

        res = self._symbolic_factorize_diag_blocks(block_matrix, timer)

        if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
            if raise_on_error:
                raise RuntimeError('Symbolic factorization unsuccessful; status: ' + str(res.status))
            else:
                return res
            
        self._get_sc_structure(block_matrix, timer)

        return res

    def _get_sc_structure(self,
                          block_matrix: MPIBlockMatrix,
                          timer: HierarchicalTimer
                          ) -> None:
            if self._flag_form_sc:
                timer.start('sc_structure')
                self._get_full_sc_structure(block_matrix=block_matrix, timer=timer)
                timer.stop('sc_structure')

    def _form_sc_components(self,
                            timer: HierarchicalTimer
                            ) -> None:
        if self._flag_form_sc:
            self._form_full_sc(timer)
    
    def _factorize_sc_components(self,
                                 timer: HierarchicalTimer
                                 ) -> LinearSolverResults:
        if self._flag_factorize_sc:
            res = self._symbolic_numeric_factorize_sc(self._current_schur_complement, self.sc_solver, timer)
        else:
            res = LinearSolverResults()
            res.status = LinearSolverStatus.successful
        return res

    def do_numeric_factorization(self,
                                 matrix: MPIBlockMatrix,
                                 raise_on_error: bool = True,
                                 timer: Optional[HierarchicalTimer] = None
                                 ) -> LinearSolverResults:
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

        # factorize all local blocks
        self._local_block_indices_for_numeric_factorization = self.local_block_indices
        res = self._numeric_factorize_diag_blocks(block_matrix, timer)
 
        if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
            if raise_on_error:
                raise RuntimeError('Numeric factorization unsuccessful; status: ' + str(res.status))
            else:
                return res
            
        self._form_sc_components(timer)

        sub_res = self._factorize_sc_components(timer)

        _process_sub_results(res, sub_res)

        if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
            if raise_on_error:
                raise RuntimeError('Symbolic factorization unsuccessful; status: ' + str(res.status))

        return res

    def do_back_solve(self, rhs, timer=None):
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
        timer.start('back_solve')
        sc_dim = rhs.get_block(self.block_dim - 1).size
        schur_complement_rhs = np.zeros(sc_dim, dtype='d')
        for ndx in self.local_block_indices:
            A = self.block_matrix.get_block(self.block_dim-1, ndx)
            contribution = self.subproblem_solvers[ndx].do_back_solve(rhs.get_block(ndx))
            schur_complement_rhs -= A.tocsr().dot(contribution.flatten())
        res = np.zeros(sc_dim, dtype='d')
        comm.Allreduce(schur_complement_rhs, res)
        schur_complement_rhs = rhs.get_block(self.block_dim - 1) + res

        SC_linop = scipy.sparse.linalg.LinearOperator(shape=(sc_dim, sc_dim),
                                               matvec=lambda u: self._sc_matvec(u, timer),
                                               dtype='d')
        M_linop = scipy.sparse.linalg.LinearOperator(shape=(sc_dim, sc_dim),
                                               matvec=lambda r: self._apply_preconditioner(r),
                                               dtype='d')
        
        pcg_sol: PcgSolution = pcg_solve(A=SC_linop,
                                         b=schur_complement_rhs,
                                         M=M_linop,
                                         pcg_options=self.pcg_options)
        coupling = pcg_sol.x
        print('Number of iterations: ', pcg_sol.num_iterations)
        self._update_preconditioner(pcg_sol)


        result = rhs.copy_structure()
        for ndx in self.local_block_indices:
            A = self.block_matrix.get_block(self.block_dim-1, ndx)
            result.set_block(ndx, self.subproblem_solvers[ndx].do_back_solve(rhs.get_block(ndx) -
                                                                             A.tocsr().transpose().dot(coupling.flatten())))

        result.set_block(self.block_dim-1, coupling)

        timer.stop('back_solve')

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

        # _pos, _neg, _zero = self.schur_complement_solver.get_inertia()
        # num_pos += _pos
        # num_neg += _neg
        # num_zero += _zero

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

    def _sc_matvec(self, u: NDArray, timer) -> NDArray:
        res = np.zeros_like(u)
        for ndx in self.local_block_indices:
            border_matrix: _BorderMatrix = self.border_matrices[ndx]
            A = border_matrix.csr.transpose()
            timer.start('dot_product')
            v = A.dot(u)
            timer.stop('dot_product')
            timer.start('block_back_solve')
            x = self.subproblem_solvers[ndx].do_back_solve(v)
            timer.stop('block_back_solve')
            timer.start('dot_product')
            y = A.transpose().dot(x)
            timer.stop('dot_product')
            res -= y
        timer.start('communicate')
        res_global = np.empty(res.size)
        comm.Allreduce(res, res_global)
        timer.stop('communicate')
        res_global += (self.block_matrix.get_block(self.block_dim-1, self.block_dim-1).tocoo()).dot(u)
        return res_global
    
    def _update_preconditioner(self, pcg_sol: PcgSolution):
        raise NotImplementedError('This method should be implemented in a subclass')
    
    def _apply_preconditioner(self, r: NDArray) -> NDArray:
        raise NotImplementedError('This method should be implemented in a subclass')
    
    @property
    def sc_solver(self) -> LinearSolverInterface:
        raise NotImplementedError('This method should be implemented in a subclass')
    


class MPIAdaptiveImplicitSchurComplementLinearSolver(MPIBaseImplicitSchurComplementLinearSolver):

    def __init__(self,
                 subproblem_solvers: Dict[int, LinearSolverInterface],
                 options: Dict):
        super().__init__(subproblem_solvers=subproblem_solvers, options=options)
        self._flag_form_sc = True
        self._flag_factorize_sc = True
        self._sc_solver: LinearSolverInterface = self.precond_options['schur_complement_solver']
        self._pcg_iter_threshhold = self.precond_options['refactorization_iter_threshhold']
    
    def _update_preconditioner(self, pcg_sol: PcgSolution):
        if pcg_sol.num_iterations > self._pcg_iter_threshhold:
            self._flag_form_sc = True
            self._flag_factorize_sc = True
        else:
            self._flag_form_sc = False
            self._flag_factorize_sc = False
        
    def _apply_preconditioner(self, r: NDArray) -> NDArray:
        return self.sc_solver.do_back_solve(r)

    @property
    def sc_solver(self) -> LinearSolverInterface:
        return self._sc_solver
    

class MPISpiluImplicitSchurComplementLinearSolver(MPIBaseImplicitSchurComplementLinearSolver):

    def __init__(self,
                 subproblem_solvers: Dict[int, LinearSolverInterface],
                 options: Dict):
        super().__init__(subproblem_solvers=subproblem_solvers, options=options)
        self._flag_form_sc = True
        self._flag_factorize_sc = False
        self._spilu_precond: scipy.sparse.linalg.SuperLU = None

    def _update_preconditioner(self, pcg_sol: PcgSolution):
        pass

    def _apply_preconditioner(self, r: NDArray) -> NDArray:
        return self._spilu_precond.solve(r)

    def _factorize_sc_components(self, timer: HierarchicalTimer) -> LinearSolverResults:
        timer.start('SpILU SC')
        self._spilu_precond = scipy.sparse.linalg.spilu(self._current_schur_complement.tocsc())
        timer.stop('SpILU SC')
        res = LinearSolverResults()
        res.status = LinearSolverStatus.successful
        return res

