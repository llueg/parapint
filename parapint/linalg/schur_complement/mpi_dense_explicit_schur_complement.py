from pyomo.contrib.pynumero.sparse.mpi_block_matrix import MPIBlockMatrix
from pyomo.contrib.pynumero.sparse.mpi_block_vector import MPIBlockVector
from parapint.linalg.base_linear_solver_interface import LinearSolverInterface
from parapint.linalg.results import LinearSolverStatus, LinearSolverResults
import numpy as np
from numpy.typing import NDArray
import scipy
from scipy.sparse import coo_matrix, csr_matrix, coo_array
from mpi4py import MPI
import itertools
from .explicit_schur_complement import _process_sub_results
from typing import Dict, Optional, List
from pyomo.common.timing import HierarchicalTimer
from .mpi_explicit_schur_complement import _gather_results, _get_all_nonzero_elements_in_sc, \
      _get_all_nonzero_elements_in_sc, _process_sub_results
from .mpi_distributed_implicit_schur_complement import _BorderMatrix


comm: MPI.Comm = MPI.COMM_WORLD
rank: int = comm.Get_rank()
size: int = comm.Get_size()


class MPIDenseSchurComplementLinearSolver(LinearSolverInterface):
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
        self.schur_complement_solver = schur_complement_solver
        self.block_dim = 0
        self.block_matrix = None
        self.local_block_indices = list()
        #self.schur_complement = coo_matrix((0, 0))
        self.border_matrices: Dict[int, _BorderMatrix] = dict()
        #self.sc_data_slices = dict()

        self.local_schur_complements: Dict[int, NDArray] = dict()

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
        timer.start('block_factorize')
        for ndx in self.local_block_indices:
            sub_res = self.subproblem_solvers[ndx].do_symbolic_factorization(matrix=block_matrix.get_block(ndx, ndx),
                                                                             raise_on_error=False)
            _process_sub_results(res, sub_res)
            if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
                break
        timer.stop('block_factorize')
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


        timer.start('sc_sparsity')
        lsc_components = [self.border_matrices[ndx].nonzero_rows for ndx in self.local_block_indices]
        all_components = comm.allreduce(lsc_components)

        sc_dim = self.border_matrices[self.local_block_indices[0]].csr.shape[0]
        sc = np.zeros((sc_dim, sc_dim), dtype=int)
        for components in all_components:
            sc[np.ix_(components, components)] += 1

        sc = coo_matrix(sc)
        #self.sc = csr_matrix(sc)
        self._sc_col = sc.col
        self._sc_row = sc.row
        nonzero_rows = sc.row
        nonzero_cols = sc.col
        sc_nnz = sc.nnz
        # if rank == 0:
        #     print(f'shape SC: {sc.shape}')
        #     print(f'SC nnz: {sc.nnz}')
        timer.stop('sc_sparsity')
        # timer.start('gather_all_nonzero_elements')
        # nonzero_rows, nonzero_cols = _get_all_nonzero_elements_in_sc(self.border_matrices, timer)
        # timer.stop('gather_all_nonzero_elements')
        # timer.start('construct_schur_complement')
        # sc_nnz = nonzero_rows.size
        # sc_dim = block_matrix.get_row_size(self.block_dim - 1)
        sc_values = np.zeros(sc_nnz, dtype=np.double)
        self.schur_complement = coo_matrix((sc_values, (nonzero_rows, nonzero_cols)), shape=(sc_dim, sc_dim))
        #timer.stop('construct_schur_complement')
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
        
        for ndx in self.local_block_indices:
            timer.start('block_factorize')
            mm = block_matrix.get_block(ndx, ndx)
            sub_res = self.subproblem_solvers[ndx].do_numeric_factorization(matrix=mm,
                                                                            raise_on_error=False)
            if rank == 0:
                print(f'shape A: {mm.shape}')
                print(f'A nnz: {mm.tocoo().nnz}')
            timer.stop('block_factorize')
            _process_sub_results(res, sub_res)
            if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
                break
        res = _gather_results(res)
        timer.start('form_SC')
        if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
            if raise_on_error:
                raise RuntimeError('Numeric factorization unsuccessful; status: ' + str(res.status))
            else:
                timer.stop('form_SC')
                return res

        # in a scipy csr_matrix,
        #     data contains the values
        #     indices contains the column indices
        #     indptr contains the number of nonzeros in the row
        #self.schur_complement.data = np.zeros(self.schur_complement.data.size, dtype=np.double)
        sc_dim = self.border_matrices[self.local_block_indices[0]].csr.shape[0]
        
        for ndx in self.local_block_indices:
            border_matrix: _BorderMatrix = self.border_matrices[ndx]

            local_sc_dim = border_matrix.num_nonzero_rows
            self.local_schur_complements[ndx] = np.zeros((local_sc_dim, local_sc_dim), dtype=np.double)
            Ar = border_matrix._get_reduced_matrix()

            A = border_matrix.csr
            _rhs = np.zeros(A.shape[1], dtype=np.double)
            solver = self.subproblem_solvers[ndx]
            for i, row_ndx in enumerate(border_matrix.nonzero_rows):
                for indptr in range(A.indptr[row_ndx], A.indptr[row_ndx + 1]):
                    col = A.indices[indptr]
                    val = A.data[indptr]
                    _rhs[col] += val
                timer.start('block_back_solve')
                contribution = solver.do_back_solve(_rhs)
                timer.stop('block_back_solve')
                timer.start('dot_product')
                #contribution = A.dot(contribution)
                contribution = Ar.dot(contribution)
                timer.stop('dot_product')
                #self.schur_complement.data[self.sc_data_slices[ndx][row_ndx]] -= contribution[border_matrix.nonzero_rows]
                #dense_sc[row_ndx, border_matrix.nonzero_rows] -= contribution[border_matrix.nonzero_rows]
                timer.start('subtract')
                self.local_schur_complements[ndx][i,:] -= contribution
                timer.stop('subtract')
                
                for indptr in range(A.indptr[row_ndx], A.indptr[row_ndx + 1]):
                    col = A.indices[indptr]
                    val = A.data[indptr]
                    _rhs[col] -= val

            # timer.start('convert_to_coo')
            # self.local_schur_complements[ndx] = coo_array(self.local_schur_complements[ndx])
            # timer.stop('convert_to_coo')

        #timer.start('communicate')
        #assembly_matrices = [(ndx, bm._get_selection_matrix().tocoo()) for ndx, bm in self.border_matrices.items()]
        #local_scs = [(ndx, sc) for ndx, sc in self.local_schur_complements.items()]
        #assembly_matrices = comm.allreduce(assembly_matrices)
        #local_scs = comm.allreduce(local_scs)
        #comm.Allgather(, assembly_matrices)
        #comm.Allgather(, local_scs)

        #timer.stop('communicate')
        # timer.start('assemble_local')
        # #assembly_matrices = dict(assembly_matrices)
        # #local_scs = dict(local_scs)
        # lsc = coo_matrix((sc_dim, sc_dim), dtype=np.double)
        # # for ndx in local_scs.keys():
        # #     sc += assembly_matrices[ndx] @ local_scs[ndx] @ assembly_matrices[ndx].T

        # for ndx in self.local_block_indices:
        #     nm = self.border_matrices[ndx]._get_selection_matrix().tocoo()
        #     lsc += nm @ self.local_schur_complements[ndx] @ nm.T
        #     #lt[np.ix_(self.border_matrices[ndx].nonzero_rows, self.border_matrices[ndx].nonzero_rows)] = self.local_schur_complements[ndx].toarray()
        # timer.stop('assemble_local')

        lsc_components = [(self.local_schur_complements[ndx], self.border_matrices[ndx].nonzero_rows) for ndx in self.local_block_indices]
        timer.start('communicate')
        all_components = comm.allreduce(lsc_components)
        timer.stop('communicate')

        timer.start('assemble_global')
        sc = np.zeros((sc_dim, sc_dim), dtype=np.double)
        #self.sc.data = np.zeros(self.sc.data.size, dtype=np.double)
        for components in all_components:
            self.sc[np.ix_(components[1], components[1])] += components[0]
        timer.stop('assemble_global')

        timer.start('to_coo')
        sc = coo_matrix((sc[self._sc_row, self._sc_col], (self._sc_row, self._sc_col)))
        #sc = coo_matrix(self.sc)
        timer.stop('to_coo')

        #sc = coo_matrix((sc_dim, sc_dim), dtype=np.double)
        # sc = np.zeros((sc_dim, sc_dim), dtype=np.double)
        # timer.start('communicate')
        # comm.Allreduce(lsc.toarray(), sc)
        # #sc = coo_matrix(sc)
        # timer.stop('communicate')
        # timer.start('to_coo')
        # sc = coo_matrix(sc)
        # timer.stop('to_coo')

        # timer.start('zeros')
        # sc = np.zeros(self.schur_complement.data.size, dtype=np.double)
        # timer.stop('zeros')
        # timer.start('Barrier')
        # comm.Barrier()
        # timer.stop('Barrier')
        # timer.start('Allreduce')
        # sc = np.empty_like(dense_sc)
        # comm.Allreduce(dense_sc, sc)
        # timer.stop('Allreduce')
        # self.schur_complement.data = sc
        timer.start('add')
        # sc = self.schur_complement + block_matrix.get_block(self.block_dim-1, self.block_dim-1).tocoo()
        sc += block_matrix.get_block(self.block_dim-1, self.block_dim-1).tocoo()
        timer.stop('add')
        #timer.stop('communicate')
        timer.stop('form_SC')
        #cond = np.linalg.cond(sc.toarray())
        #print(f'SC condition number: {cond}')
        #timer.start('to_coo')
        #sc = coo_matrix(sc)
        #timer.stop('to_coo')
        timer.start('factor_SC')
        sub_res = self.schur_complement_solver.do_symbolic_factorization(sc, raise_on_error=raise_on_error)
        _process_sub_results(res, sub_res)
        if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
            timer.stop('factor_SC')
            return res
        sub_res = self.schur_complement_solver.do_numeric_factorization(sc)
        _process_sub_results(res, sub_res)
        timer.stop('factor_SC')
        return res

    def do_back_solve(self, rhs, timer=None, barrier=None, ip_iter=None):
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

        timer.start('form_rhs')
        schur_complement_rhs = np.zeros(rhs.get_block(self.block_dim - 1).size, dtype='d')
        for ndx in self.local_block_indices:
            A = self.block_matrix.get_block(self.block_dim-1, ndx)
            timer.start('block_back_solve')
            contribution = self.subproblem_solvers[ndx].do_back_solve(rhs.get_block(ndx))
            timer.stop('block_back_solve')
            timer.start('dot_product')
            schur_complement_rhs -= A.tocsr().dot(contribution.flatten())
            timer.stop('dot_product')
        res = np.zeros(rhs.get_block(self.block_dim - 1).shape[0], dtype='d')
        timer.start('communicate')
        timer.start('Allreduce')
        comm.Allreduce(schur_complement_rhs, res)
        timer.stop('Allreduce')
        timer.stop('communicate')
        schur_complement_rhs = rhs.get_block(self.block_dim - 1) + res

        result = rhs.copy_structure()
        timer.stop('form_rhs')
        timer.start('SC_back_solve')
        coupling = self.schur_complement_solver.do_back_solve(schur_complement_rhs)
        timer.stop('SC_back_solve')


        for ndx in self.local_block_indices:
            A = self.block_matrix.get_block(self.block_dim-1, ndx)
            timer.start('block_back_solve')
            result.set_block(ndx, self.subproblem_solvers[ndx].do_back_solve(rhs.get_block(ndx) -
                                                                             A.tocsr().transpose().dot(coupling.flatten())))
            timer.stop('block_back_solve')

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

        _pos, _neg, _zero = self.schur_complement_solver.get_inertia()

        if _neg > 0 or _zero > 0:
            print('WARNING: Schur complement has negative or zero eigenvalues.')
        num_pos += _pos
        num_neg += _neg
        num_zero += _zero

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
        self.schur_complement_solver.increase_memory_allocation(factor=factor)
