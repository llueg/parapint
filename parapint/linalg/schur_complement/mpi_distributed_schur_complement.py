from pyomo.contrib.pynumero.sparse.mpi_block_matrix import MPIBlockMatrix
from pyomo.contrib.pynumero.sparse.mpi_block_vector import MPIBlockVector
from parapint.linalg.base_linear_solver_interface import LinearSolverInterface
from parapint.linalg.results import LinearSolverStatus, LinearSolverResults
import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sps
from scipy.sparse import coo_matrix, csr_matrix
from mpi4py import MPI
import itertools
from .explicit_schur_complement import _process_sub_results
from typing import Dict, Optional, List
from pyomo.common.timing import HierarchicalTimer
from parapint.linalg.schur_complement.utils import _gather_results, _BorderMatrix, _get_all_nonzero_elements_in_sc_using_ix
from parapint.linalg.iterative.pcg import pcg_solve, PcgOptions, PcgSolution, PcgSolutionStatus, LbfgsInvHessProduct
from parapint.linalg.schur_complement.mpi_implicit_schur_complement import MPIBaseImplicitSchurComplementLinearSolver

comm: MPI.Comm = MPI.COMM_WORLD
rank: int = comm.Get_rank()
size: int = comm.Get_size()

# TODO: naming of classes
class MPIASNoOverlapImplicitSchurComplementLinearSolver(MPIBaseImplicitSchurComplementLinearSolver):

    def __init__(self,
                 subproblem_solvers: Dict[int, LinearSolverInterface],
                 local_schur_complinear_solvers: Dict[int, LinearSolverInterface],
                 options: Dict):
        super().__init__(subproblem_solvers=subproblem_solvers, options=options)
        self._flag_form_sc = False
        self._flag_factorize_sc = False

        self.local_schur_complement_solvers: Dict[int, LinearSolverInterface] = local_schur_complinear_solvers
        self.local_schur_complements: Dict[int, NDArray | sps.coo_array] = dict()


    def _check_if_block_indices_local(self, block_indices: List[int]) -> None:
        for i in block_indices:
            assert i in self.local_block_indices, f"Block {i} not local to rank {rank}"

    def _form_sc_components(self,
                            timer: HierarchicalTimer
                            ) -> None:

        for ndx in self._local_block_indices_for_numeric_factorization:
            border_matrix: _BorderMatrix = self.border_matrices[ndx]
            local_sc_dim = border_matrix.num_nonzero_rows
            self.local_schur_complements[ndx] = np.zeros((local_sc_dim, local_sc_dim), dtype=np.double)

            A = border_matrix.csr
            Ar = border_matrix._get_reduced_matrix()
            _rhs = np.zeros(A.shape[1], dtype=np.double)
            solver = self.subproblem_solvers[ndx]
            for i, row_ndx in enumerate(border_matrix.nonzero_rows):
                timer.start('get_rhs')
                for indptr in range(A.indptr[row_ndx], A.indptr[row_ndx + 1]):
                    col = A.indices[indptr]
                    val = A.data[indptr]
                    _rhs[col] += val
                timer.stop('get_rhs')
                timer.start('block_back_solve')
                contribution = solver.do_back_solve(_rhs)
                timer.stop('block_back_solve')
                timer.start('dot_product')
                contribution = Ar.dot(contribution)
                timer.stop('dot_product')
                self.local_schur_complements[ndx][i,:] -= contribution
                timer.start('get_rhs')
                for indptr in range(A.indptr[row_ndx], A.indptr[row_ndx + 1]):
                    col = A.indices[indptr]
                    val = A.data[indptr]
                    _rhs[col] -= val
                timer.stop('get_rhs')
            
            # TODO: Figure out if this is needed, esp. if local regularization is used
            # s_diag = self.block_matrix.get_block(self.block_dim-1, self.block_dim-1).tocoo().diagonal()
            # sk_diag = self.weighting_matrix[border_matrix.nonzero_rows] * s_diag[border_matrix.nonzero_rows]
            # self.local_schur_complements[ndx][np.arange(local_sc_dim), np.arange(local_sc_dim)] += sk_diag
            timer.start('convert_to_coo')
            self.local_schur_complements[ndx] = sps.coo_array(self.local_schur_complements[ndx])
            timer.stop('convert_to_coo')

    def _factorize_sc_components(self,
                                 timer: HierarchicalTimer
                                 ) -> LinearSolverResults:
        res = LinearSolverResults()
        res.status = LinearSolverStatus.successful
        for ndx in self._local_block_indices_for_numeric_factorization:
            sub_res = self.local_schur_complement_solvers[ndx].do_symbolic_factorization(self.local_schur_complements[ndx], raise_on_error=False)
            _process_sub_results(res, sub_res)
            if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
                timer.stop('factor_SC')
                return res
            sub_res = self.local_schur_complement_solvers[ndx].do_numeric_factorization(self.local_schur_complements[ndx], raise_on_error=False)
            _process_sub_results(res, sub_res)
        timer.stop('factor_SC')
        return res

    def _get_sc_structure(self,
                          block_matrix: MPIBlockMatrix,
                          timer: HierarchicalTimer
                          ) -> None:
        timer.start('build_border_matrices')

        self.border_matrices = dict()
        for ndx in self.local_block_indices:
            self.border_matrices[ndx] = _BorderMatrix(block_matrix.get_block(self.block_dim - 1, ndx))

        timer.stop('build_border_matrices')

    def _update_preconditioner(self, pcg_sol: PcgSolution):
        pass
        
    def _apply_preconditioner(self,
                              r: NDArray
                              ) -> NDArray:
        # One level Additive Scwarz preconditioner
        result = np.zeros_like(r)
        for ndx in self.local_block_indices:
            n_mat = self.border_matrices[ndx]._get_selection_matrix()
            r_local = n_mat.transpose().dot(r)
            x_local = self.local_schur_complement_solvers[ndx].do_back_solve(r_local)
            result += n_mat.dot(x_local)

        result = comm.allreduce(result)

        return result

    def get_distributed_intertia(self) -> Dict[int, tuple[int, int, int]]:

        inertia_per_block = dict()

        for ndx in self.local_block_indices:
            _pos, _neg, _zero = self.subproblem_solvers[ndx].get_inertia()
            _pos_sc, _neg_sc, _zero_sc = self.local_schur_complement_solvers[ndx].get_inertia()
            inertia_per_block[ndx] = (_pos + _pos_sc, _neg + _neg_sc, _zero + _zero_sc)

        return inertia_per_block

    def do_numeric_factorization(self,
                                 matrix: MPIBlockMatrix,
                                 raise_on_error: bool = True,
                                 timer: Optional[HierarchicalTimer] = None,
                                 block_indices: Optional[List[int]] = None,
                                 ) -> LinearSolverResults:

        if block_indices is None:
            block_indices = self.local_block_indices
        elif block_indices == []:
            res = LinearSolverResults()
            res.status = LinearSolverStatus.successful
            return res
        else:
            self._check_if_block_indices_local(block_indices)

        self._local_block_indices_for_numeric_factorization = block_indices

        if timer is None:
            timer = HierarchicalTimer()

        self.block_matrix = block_matrix = matrix

        # factorize all local blocks
        
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


class MPIASWithOverlapImplicitSchurComplementLinearSolver(MPIASNoOverlapImplicitSchurComplementLinearSolver):

    #TODO: For inertia correction, figure out if local SC are needed too
    def _form_sc_components(self,
                            timer: HierarchicalTimer
                            ) -> None:
        # For now, slightly inefficient computation of local assembled SCs,
        # by building full SC first
        self._form_full_sc(timer)

        for ndx in self.local_block_indices:
            border_matrix: _BorderMatrix = self.border_matrices[ndx]
            local_sc_dim = border_matrix.num_nonzero_rows
            self.local_schur_complements[ndx] = np.zeros((local_sc_dim, local_sc_dim), dtype=np.double)

            Ar = border_matrix._get_reduced_matrix()
            self.local_schur_complements[ndx] = Ar @ self.schur_complement @ Ar.T
            
            # TODO: Figure out if this is needed, esp. if local regularization is used
            # s_diag = self.block_matrix.get_block(self.block_dim-1, self.block_dim-1).tocoo().diagonal()
            # sk_diag = self.weighting_matrix[border_matrix.nonzero_rows] * s_diag[border_matrix.nonzero_rows]
            # self.local_schur_complements[ndx][np.arange(local_sc_dim), np.arange(local_sc_dim)] += sk_diag
            timer.start('convert_to_coo')
            self.local_schur_complements[ndx] = sps.coo_array(self.local_schur_complements[ndx])
            timer.stop('convert_to_coo')
