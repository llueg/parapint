from pyomo.contrib.pynumero.sparse.mpi_block_matrix import MPIBlockMatrix
from pyomo.contrib.pynumero.sparse.mpi_block_vector import MPIBlockVector
from parapint.linalg.base_linear_solver_interface import LinearSolverInterface
from parapint.linalg.results import LinearSolverStatus, LinearSolverResults
import numpy as np
from numpy.typing import NDArray
import scipy.sparse as sps
from scipy.sparse import coo_matrix, csr_matrix
from mpi4py import MPI
from mpi4py.util import dtlib
import itertools
from .explicit_schur_complement import _process_sub_results
from typing import Dict, Optional, List
from pyomo.common.timing import HierarchicalTimer
from parapint.linalg.schur_complement.utils import _gather_results, _BorderMatrix, _get_all_nonzero_elements_in_sc_using_ix
from parapint.linalg.iterative.pcg import pcg_solve, PcgOptions, PcgSolution, PcgSolutionStatus, LbfgsInvHessProduct
from parapint.linalg.schur_complement.mpi_implicit_schur_complement import MPIBaseImplicitSchurComplementLinearSolver
from parapint.utils import MPIHierarchicalTimer

comm: MPI.Comm = MPI.COMM_WORLD
rank: int = comm.Get_rank()
size: int = comm.Get_size()

# TODO: naming of classes
class MPIASNoOverlapImplicitSchurComplementLinearSolver(MPIBaseImplicitSchurComplementLinearSolver):

    def __init__(self,
                 subproblem_solvers: Dict[int, LinearSolverInterface],
                 local_schur_complement_solvers: Dict[int, LinearSolverInterface],
                 options: Dict):
        super().__init__(subproblem_solvers=subproblem_solvers, options=options)
        self._flag_form_sc = False
        self._flag_factorize_sc = False

        self.local_schur_complement_solvers: Dict[int, LinearSolverInterface] = local_schur_complement_solvers
        self.local_schur_complements: Dict[int, NDArray | sps.coo_array] = dict()
        self.debug = False

    def _check_if_block_indices_local(self, block_indices: List[int]) -> None:
        for i in block_indices:
            assert i in self.local_block_indices, f"Block {i} not local to rank {rank}"

    def _form_sc_components(self,
                            timer: HierarchicalTimer
                            ) -> None:

        timer.start('form_SC')
        #sc_dim = self.block_matrix.get_row_size(self.block_dim - 1)
        #self.weighting_matrix = np.zeros(sc_dim, dtype=np.double)
        for ndx in self._local_block_indices_for_numeric_factorization:
            border_matrix: _BorderMatrix = self.border_matrices[ndx]
            local_sc_dim = border_matrix.num_nonzero_rows
            self.local_schur_complements[ndx] = np.zeros((local_sc_dim, local_sc_dim), dtype=np.double)
            #self.weighting_matrix[self.border_matrices[ndx].nonzero_rows] += 1
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
        timer.start('Barrier')
        comm.Barrier()
        timer.stop('Barrier')
        #self.weighting_matrix = comm.allreduce(self.weighting_matrix)
        #self.weighting_matrix = 1/self.weighting_matrix
        #self.weighting_matrix = np.ones_like(self.weighting_matrix)
        timer.stop('form_SC')
        if self.debug:
            self._form_full_sc(timer)
            assembled_sc = np.zeros_like(self.schur_complement.todense())
            for ndx in self._local_block_indices_for_numeric_factorization:
                border_matrix: _BorderMatrix = self.border_matrices[ndx]
                Nk = border_matrix._get_selection_matrix()
                assembled_sc += Nk @ self.local_schur_complements[ndx] @ Nk.T
            
            assembled_sc = comm.allreduce(assembled_sc)
            diff = np.linalg.norm(assembled_sc - self.schur_complement.todense())
            if diff > 1e-10:
                print(f"Diff in SC: {diff}")
                #raise RuntimeError("Diff in SC")

    def _factorize_sc_components(self,
                                 timer: HierarchicalTimer
                                 ) -> LinearSolverResults:
        res = LinearSolverResults()
        res.status = LinearSolverStatus.successful
        timer.start('factor_SC')
        for ndx in self._local_block_indices_for_numeric_factorization:
            timer.start('symbolic')
            sub_res = self.local_schur_complement_solvers[ndx].do_symbolic_factorization(self.local_schur_complements[ndx], raise_on_error=False)
            _process_sub_results(res, sub_res)
            timer.stop('symbolic')
            if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
                timer.stop('factor_SC')
                return res
            timer.start('numeric')
            sub_res = self.local_schur_complement_solvers[ndx].do_numeric_factorization(self.local_schur_complements[ndx], raise_on_error=False)
            _process_sub_results(res, sub_res)
            timer.stop('numeric')
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

        if self.debug:
            self._get_full_sc_structure(block_matrix, timer)

    def _update_preconditioner(self, pcg_sol: PcgSolution):
        pass
        
    def _apply_preconditioner(self,
                              r: NDArray,
                              timer: HierarchicalTimer=None
                              ) -> NDArray:
        # One level Additive Scwarz preconditioner
        result = np.zeros_like(r)
        
        for ndx in self.local_block_indices:
            #n_mat = self.border_matrices[ndx]._get_selection_matrix()
            #r_local = n_mat.transpose().dot(r)
            r_local = r[self.border_matrices[ndx].nonzero_rows]
            timer.start('local_back_solve')
            x_local = self.local_schur_complement_solvers[ndx].do_back_solve(r_local)
            timer.stop('local_back_solve')
            result[self.border_matrices[ndx].nonzero_rows] += x_local
            #result += n_mat.dot(x_local)

        timer.start('communication')
        comm.Allreduce(MPI.IN_PLACE, result, op=MPI.SUM)
        #comm.Allreduce(MPI.IN_PLACE, result, op=MPI.SUM)
        #res_global = np.empty(result.size)
        #comm.Allreduce(result, res_global)
        timer.stop('communication')

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
        if timer is None:
            timer = HierarchicalTimer()

        timer.start('setup')
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
        timer.stop('setup')
        # factorize all local blocks
        
        res = self._numeric_factorize_diag_blocks(block_matrix, timer)
 
        if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
            if raise_on_error:
                raise RuntimeError('Numeric factorization unsuccessful; status: ' + str(res.status))
            else:
                return res
            
        self._form_sc_components(timer)

        sub_res = self._factorize_sc_components(timer)

        timer.start('process_sub_results')
        _process_sub_results(res, sub_res)
        timer.stop('process_sub_results')

        if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
            if raise_on_error:
                raise RuntimeError('Symbolic factorization unsuccessful; status: ' + str(res.status))

        return res


class MPIASWithOverlapImplicitSchurComplementLinearSolver(MPIASNoOverlapImplicitSchurComplementLinearSolver):

    def __init__(self,
                 subproblem_solvers: Dict[int, LinearSolverInterface],
                 local_schur_complement_solvers: Dict[int, LinearSolverInterface],
                 options: Dict):
        super().__init__(subproblem_solvers=subproblem_solvers,
                         local_schur_complement_solvers=local_schur_complement_solvers,
                         options=options)
        self._local_windows = dict()
        self._windows_allocated = False
        self._neighboring_scs = dict()


    def _get_sc_structure(self,
                          block_matrix: MPIBlockMatrix,
                          timer: HierarchicalTimer
                          ) -> None:
        
        timer.start('build_border_matrices')

        self.border_matrices = dict()
        for ndx in self.local_block_indices:
            self.border_matrices[ndx] = _BorderMatrix(block_matrix.get_block(self.block_dim - 1, ndx))

        self.sc_dim = block_matrix.get_row_size(self.block_dim - 1)

        timer.stop('build_border_matrices')
        self._get_connectivity_info(timer)

        if self.debug:
            self._get_full_sc_structure(block_matrix, timer)


    def _create_windows(self, timer: HierarchicalTimer) -> None:
        timer.start('allocate_windows')
        local_windows = dict()
        datatype = MPI.DOUBLE
        np_dtype = dtlib.to_numpy_dtype(datatype)
        itemsize = datatype.Get_size()
        for ndx in range(self.block_dim - 1):
            if ndx in self.local_block_indices:
                timer.start('allocate_local_window')
                local_sc_dim = self.border_matrices[ndx].num_nonzero_rows
                # local_windows[ndx] = MPI.Win.Create(memory=self.local_schur_complements[ndx].astype(np_dtype),
                #                                     disp_unit=itemsize,
                #                                     info=MPI.INFO_NULL,
                #                                     comm=comm)
                local_windows[ndx] = MPI.Win.Allocate(
                    size=local_sc_dim * local_sc_dim * itemsize,
                    disp_unit=itemsize,
                    #comm=comm,
                    comm=self.local_comms[ndx]
                )
                timer.stop('allocate_local_window')
            elif self.ownership_map[ndx] in self.neighboring_ranks:
                timer.start('allocate_empty_window')
                local_windows[ndx] = MPI.Win.Allocate(
                    size=0,
                    disp_unit=itemsize,
                    comm=self.local_comms[ndx],
                )
                timer.stop('allocate_empty_window')

            else:
                pass
                # local_windows[ndx] = MPI.Win.Create(memory=np.zeros((0,0), dtype=np_dtype),
                #                                     disp_unit=1,
                #                                     info=MPI.INFO_NULL,
                #                                     comm=comm)
                # timer.start('allocate_empty_window')
                # local_windows[ndx] = MPI.Win.Allocate(
                #     size=0,
                #     disp_unit=itemsize,
                #     comm=comm,
                # )
                # timer.stop('allocate_empty_window')
            # timer.start('loop_barrier')
            # comm.barrier()
            # timer.stop('loop_barrier')
        timer.start('Barrier')
        comm.Barrier()
        timer.stop('Barrier')
        self._local_windows = local_windows
        timer.stop('allocate_windows')

    def _put_windows(self, timer: HierarchicalTimer):
        timer.start('put_windows')
        datatype = MPI.DOUBLE
        np_dtype = dtlib.to_numpy_dtype(datatype)
        itemsize = datatype.Get_size()
        for ndx in self.local_block_indices:
            buf = self.local_schur_complements[ndx].astype(np_dtype).reshape(-1)
            #local_sc_dim = self.border_matrices[ndx].num_nonzero_rows
            #self._local_windows[ndx].Lock(rank=rank, lock_type=MPI.LOCK_SHARED)
            rel_rank = self.all_group_ranks[rank, ndx]
            self._local_windows[ndx].Lock(rank=rel_rank)
            self._local_windows[ndx].Put(buf, target_rank=rel_rank)
            self._local_windows[ndx].Unlock(rank=rel_rank)
        #MPI.Win.Unlock()
        timer.start('Barrier')
        comm.Barrier()
        timer.stop('Barrier')
        timer.stop('put_windows')

    def _get_windows(self, timer: HierarchicalTimer):
        timer.start('get_windows')
        datatype = MPI.DOUBLE
        np_dtype = dtlib.to_numpy_dtype(datatype)
        itemsize = datatype.Get_size()
        neighboring_scs = dict()
        for ndx in range(self.block_dim - 1):
            nrank = self.ownership_map[ndx]
            if nrank in self.neighboring_ranks:
                rel_nrank = self.all_group_ranks[nrank, ndx]
                n_sc_dim = len(self.global_rank_to_var_map[nrank])
                buf = np.empty((n_sc_dim, n_sc_dim), dtype=np_dtype).reshape(-1)
                n_sc_window = self._local_windows[ndx]
                #print(f"rank {rank} getting sc from rank {nrank}")
                n_sc_window.Lock(rel_nrank)
                n_sc_window.Get(buf, target_rank=rel_nrank)
                n_sc_window.Unlock(rel_nrank)
                neighboring_scs[nrank] = buf.reshape((n_sc_dim, n_sc_dim))

        # for nrank in self.neighboring_ranks:
        #     rel_nrank = self.all_group_ranks[nrank]
        #     n_sc_dim = len(self.global_rank_to_var_map[nrank])
        #     #neighboring_scs[nrank] = np.zeros((n_sc_dim, n_sc_dim), dtype=np_dtype)
        #     buf = np.empty((n_sc_dim, n_sc_dim), dtype=np_dtype).reshape(-1)
        #     n_sc_window = self._local_windows[nrank]
        #     #print(f"rank {rank} getting sc from rank {nrank}")
        #     n_sc_window.Lock(nrank)
        #     n_sc_window.Get(buf, target_rank=nrank)
        #     n_sc_window.Unlock(nrank)
        #     neighboring_scs[nrank] = buf.reshape((n_sc_dim, n_sc_dim))
            #print(f"rank {rank} got sc from rank {nrank}")
            #comm.Barrier()
        #print(f"rank {rank} got all scs")
        timer.start('Barrier')
        comm.Barrier()
        timer.stop('Barrier')
        self._neighboring_scs = neighboring_scs
        timer.stop('get_windows')

    def _form_sc_components(self,
                            timer: HierarchicalTimer
                            ) -> None:
        timer.start('form_SC')
        #sc_dim = self.block_matrix.get_row_size(self.block_dim - 1)
        #self.weighting_matrix = np.zeros(sc_dim, dtype=np.double)
        timer.start('form_local_SC')
        for ndx in self._local_block_indices_for_numeric_factorization:
            border_matrix: _BorderMatrix = self.border_matrices[ndx]
            local_sc_dim = border_matrix.num_nonzero_rows
            self.local_schur_complements[ndx] = np.zeros((local_sc_dim, local_sc_dim), dtype=np.double)
            #self.weighting_matrix[self.border_matrices[ndx].nonzero_rows] += 1
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
        timer.start('Barrier')
        comm.Barrier()
        timer.stop('Barrier')
        timer.stop('form_local_SC')

        # create mpi windows for local schur complements
        timer.start('communicate')
        if not self._windows_allocated:
            self._create_windows(timer)
            self._windows_allocated = True

        self._put_windows(timer)
        self._get_windows(timer)
        timer.stop('communicate')
        # timer.start('create_windows')
        # local_windows = dict()
        # datatype = MPI.DOUBLE
        # np_dtype = dtlib.to_numpy_dtype(datatype)
        # itemsize = datatype.Get_size()
        # for ndx in range(size):
        #     if ndx in self.local_block_indices:
        #         local_sc_dim = self.border_matrices[ndx].num_nonzero_rows
        #         local_windows[ndx] = MPI.Win.Create(memory=self.local_schur_complements[ndx].astype(np_dtype),
        #                                             disp_unit=itemsize,
        #                                             info=MPI.INFO_NULL,
        #                                             comm=comm)
        #     else:
        #         local_windows[ndx] = MPI.Win.Create(memory=np.zeros((0,0), dtype=np_dtype),
        #                                             disp_unit=1,
        #                                             info=MPI.INFO_NULL,
        #                                             comm=comm)
        # timer.stop('create_windows')

        # timer.start('get_windows')
        # neighboring_scs = dict()
        # for nrank in self.neighboring_ranks:
        #     n_sc_dim = len(self.global_rank_to_var_map[nrank])
        #     neighboring_scs[nrank] = np.zeros((n_sc_dim, n_sc_dim), dtype=np_dtype)
        #     n_sc_window = local_windows[nrank]
        #     #print(f"rank {rank} getting sc from rank {nrank}")
        #     n_sc_window.Lock(nrank)
        #     n_sc_window.Get(neighboring_scs[nrank], target_rank=nrank)
        #     n_sc_window.Unlock(nrank)
        #     #print(f"rank {rank} got sc from rank {nrank}")
        #     #comm.Barrier()
        # #print(f"rank {rank} got all scs")
        # comm.Barrier()
        # #print(f'start free')
        # for ndx in range(size):
        #     local_windows[ndx].Free()
        # timer.stop('get_windows')
        #print('done get windows')

        
        timer.start('assemble_local_SC')
        
        for ndx in self.local_block_indices:
            local_sc = self.local_schur_complements[ndx]
            for nrank in self.neighboring_ranks:
                local_sc_vars = self.border_matrices[ndx].nonzero_rows
                neighboring_sc_vars = self.global_rank_to_var_map[nrank]
                overlapping_vars_nmask = np.isin(neighboring_sc_vars, local_sc_vars)
                if not np.any(overlapping_vars_nmask):
                    continue
                overlapping_vars = neighboring_sc_vars[overlapping_vars_nmask]
                local_ov_idxs = []
                # TODO: can this be achieved without the loop?
                for ov in overlapping_vars:
                    local_ov_idxs.append(np.where(local_sc_vars == ov)[0][0])
                local_ov_idxs = np.array(local_ov_idxs, dtype=np.int64)
                local_sc[np.ix_(local_ov_idxs, local_ov_idxs)] += self._neighboring_scs[nrank][np.ix_(overlapping_vars_nmask, overlapping_vars_nmask)]
            
            self.local_schur_complements[ndx] = sps.coo_array(local_sc)
        
        timer.start('Barrier')
        comm.Barrier()
        timer.stop('Barrier')
        timer.stop('assemble_local_SC')
        #print('done assemble local SC')
        timer.stop('form_SC')

        if self.debug:
            #print('debugging')
            debug_sc = dict()
            self._form_full_sc(timer)
            timer.start('form_SC')
            timer.start('form_local_SC')
            for ndx in self.local_block_indices:
                border_matrix: _BorderMatrix = self.border_matrices[ndx]
                local_sc_dim = border_matrix.num_nonzero_rows
                debug_sc[ndx] = np.zeros((local_sc_dim, local_sc_dim), dtype=np.double)

                Nk = border_matrix._get_selection_matrix(format='coo')
                timer.start('matmul')
                debug_sc[ndx] = Nk.T @ self.schur_complement @ Nk
                timer.stop('matmul')
                
                timer.start('convert_to_coo')
                debug_sc[ndx] = sps.coo_array(debug_sc[ndx])
                timer.stop('convert_to_coo')
            
            timer.stop('form_local_SC')
            timer.stop('form_SC')

            for ndx in self.local_block_indices:
                diff = np.linalg.norm(self.local_schur_complements[ndx].todense() - debug_sc[ndx].todense())
                if np.allclose(self.local_schur_complements[ndx].todense(), debug_sc[ndx].todense()):
                    #print(f"Diff in SC (block {ndx}): {diff}")
                    continue
                print(f"Diff in SC (block {ndx}): {diff}")

    #TODO: For inertia correction, figure out if local SC are needed too
    def _form_sc_components_debug(self,
                            timer: HierarchicalTimer
                            ) -> None:
        # For now, slightly inefficient computation of local assembled SCs,
        # by building full SC first
        self._form_full_sc(timer)
        timer.start('form_SC')
        timer.start('form_local_SC')
        for ndx in self.local_block_indices:
            border_matrix: _BorderMatrix = self.border_matrices[ndx]
            local_sc_dim = border_matrix.num_nonzero_rows
            self.local_schur_complements[ndx] = np.zeros((local_sc_dim, local_sc_dim), dtype=np.double)

            Nk = border_matrix._get_selection_matrix(format='coo')
            timer.start('matmul')
            self.local_schur_complements[ndx] = Nk.T @ self.schur_complement @ Nk
            timer.stop('matmul')
            
            timer.start('convert_to_coo')
            self.local_schur_complements[ndx] = sps.coo_array(self.local_schur_complements[ndx])
            timer.stop('convert_to_coo')
        
        timer.stop('form_local_SC')
        timer.stop('form_SC')


class MPIASDiagOverlapImplicitSchurComplementLinearSolver(MPIBaseImplicitSchurComplementLinearSolver):

    def __init__(self,
                 subproblem_solvers: Dict[int, LinearSolverInterface],
                 local_schur_complement_solvers: Dict[int, LinearSolverInterface],
                 options: Dict):
        super().__init__(subproblem_solvers=subproblem_solvers, options=options)
        self._flag_form_sc = False
        self._flag_factorize_sc = False

        self.local_schur_complement_solvers: Dict[int, LinearSolverInterface] = local_schur_complement_solvers
        self.local_schur_complements: Dict[int, NDArray | sps.coo_array] = dict()
        self.debug = False

    def _check_if_block_indices_local(self, block_indices: List[int]) -> None:
        for i in block_indices:
            assert i in self.local_block_indices, f"Block {i} not local to rank {rank}"

    def _form_sc_components(self,
                            timer: HierarchicalTimer
                            ) -> None:

        timer.start('form_SC')
        #sc_dim = self.block_matrix.get_row_size(self.block_dim - 1)
        #self.weighting_matrix = np.zeros(sc_dim, dtype=np.double)
        sc_diag = np.zeros(self.block_matrix.get_row_size(self.block_dim - 1), dtype=np.double)
        timer.start('form_local_SC')
        for ndx in self._local_block_indices_for_numeric_factorization:
            border_matrix: _BorderMatrix = self.border_matrices[ndx]
            local_sc_dim = border_matrix.num_nonzero_rows
            self.local_schur_complements[ndx] = np.zeros((local_sc_dim, local_sc_dim), dtype=np.double)
            #self.weighting_matrix[self.border_matrices[ndx].nonzero_rows] += 1
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
            sc_diag[border_matrix.nonzero_rows] += np.diag(self.local_schur_complements[ndx])

        timer.start('Barrier')
        comm.Barrier()
        timer.stop('Barrier')
        timer.stop('form_local_SC')

        timer.start('communicate_diag')
        # allredeuce numpy array
        #global_sc_diag = np.zeros_like(sc_diag)
        timer.start('Allreduce')
        comm.Allreduce(MPI.IN_PLACE, sc_diag, op=MPI.SUM)
        timer.stop('Allreduce')
        timer.stop('communicate_diag')
        #sc_diag = comm.Allreduce(sc_diag)
            
            
        for ndx in self._local_block_indices_for_numeric_factorization:
            timer.start('set_global_diag')
            border_matrix: _BorderMatrix = self.border_matrices[ndx]
            local_sc_dim = border_matrix.num_nonzero_rows
            self.local_schur_complements[ndx][np.arange(local_sc_dim), np.arange(local_sc_dim)] = sc_diag[border_matrix.nonzero_rows]
            timer.stop('set_global_diag')
            timer.start('convert_to_coo')
            self.local_schur_complements[ndx] = sps.coo_array(self.local_schur_complements[ndx])
            timer.stop('convert_to_coo')
        
        #self.weighting_matrix = comm.allreduce(self.weighting_matrix)
        #self.weighting_matrix = 1/self.weighting_matrix
        #self.weighting_matrix = np.ones_like(self.weighting_matrix)
        timer.stop('form_SC')
        if self.debug:
            self._form_full_sc(timer)
            assembled_sc = np.zeros_like(self.schur_complement.todense())
            for ndx in self._local_block_indices_for_numeric_factorization:
                border_matrix: _BorderMatrix = self.border_matrices[ndx]
                Nk = border_matrix._get_selection_matrix()
                assembled_sc += Nk @ self.local_schur_complements[ndx] @ Nk.T
            
            assembled_sc = comm.allreduce(assembled_sc)
            diff = np.linalg.norm(assembled_sc - self.schur_complement.todense())
            if diff > 1e-10:
                print(f"Diff in SC: {diff}")
                #raise RuntimeError("Diff in SC")

    def _factorize_sc_components(self,
                                 timer: HierarchicalTimer
                                 ) -> LinearSolverResults:
        res = LinearSolverResults()
        res.status = LinearSolverStatus.successful
        timer.start('factor_SC')
        for ndx in self._local_block_indices_for_numeric_factorization:
            timer.start('symbolic')
            sub_res = self.local_schur_complement_solvers[ndx].do_symbolic_factorization(self.local_schur_complements[ndx], raise_on_error=False)
            _process_sub_results(res, sub_res)
            timer.stop('symbolic')
            if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
                timer.stop('factor_SC')
                return res
            timer.start('numeric')
            sub_res = self.local_schur_complement_solvers[ndx].do_numeric_factorization(self.local_schur_complements[ndx], raise_on_error=False)
            _process_sub_results(res, sub_res)
            timer.stop('numeric')
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

        if self.debug:
            self._get_full_sc_structure(block_matrix, timer)

    def _update_preconditioner(self, pcg_sol: PcgSolution):
        pass
        
    def _apply_preconditioner(self,
                              r: NDArray,
                              timer: HierarchicalTimer=None
                              ) -> NDArray:
        # One level Additive Scwarz preconditioner
        result = np.zeros_like(r)
        
        for ndx in self.local_block_indices:
            n_mat = self.border_matrices[ndx]._get_selection_matrix()
            #r_local = n_mat.transpose().dot(self.weighting_matrix * r)
            r_local = n_mat.transpose().dot(r)
            timer.start('local_back_solve')
            x_local = self.local_schur_complement_solvers[ndx].do_back_solve(r_local)
            timer.stop('local_back_solve')
            #result += self.weighting_matrix * n_mat.dot(x_local)
            result += n_mat.dot(x_local)

        timer.start('communication')
        comm.Allreduce(MPI.IN_PLACE, result, op=MPI.SUM)
        timer.stop('communication')

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

        timer.start('setup')
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
        timer.stop('setup')
        # factorize all local blocks
        
        res = self._numeric_factorize_diag_blocks(block_matrix, timer)
 
        if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
            if raise_on_error:
                raise RuntimeError('Numeric factorization unsuccessful; status: ' + str(res.status))
            else:
                return res
            
        self._form_sc_components(timer)

        sub_res = self._factorize_sc_components(timer)

        timer.start('process_sub_results')
        _process_sub_results(res, sub_res)
        timer.stop('process_sub_results')

        if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
            if raise_on_error:
                raise RuntimeError('Symbolic factorization unsuccessful; status: ' + str(res.status))

        return res
