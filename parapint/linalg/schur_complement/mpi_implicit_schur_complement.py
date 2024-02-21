from __future__ import annotations
from pyomo.contrib.pynumero.sparse.mpi_block_matrix import MPIBlockMatrix
from pyomo.contrib.pynumero.sparse.mpi_block_vector import MPIBlockVector
from parapint.linalg.base_linear_solver_interface import LinearSolverInterface
from parapint.linalg.results import LinearSolverStatus, LinearSolverResults
import numpy as np
from numpy.typing import NDArray
from collections import deque
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import cg, LinearOperator
from mpi4py import MPI
import itertools
from .explicit_schur_complement import _process_sub_results
from typing import Dict, Optional, List, Callable, Tuple
from pyomo.common.timing import HierarchicalTimer
import copy

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
                 schur_complement_solver: LinearSolverInterface, options: Dict):
        self.subproblem_solvers = subproblem_solvers
        #self.schur_complement_solver = schur_complement_solver
        self.block_dim = 0
        self.block_matrix = None
        self.local_block_indices = list()
        #self.schur_complement = coo_matrix((0, 0))
        self.border_matrices: Dict[int, _BorderMatrix] = dict()
        self.sc_data_slices = dict()
        self.schur_complement_solver = schur_complement_solver
        self._diagnostic_flag = options['diagnostic_flag']

        self._factorize_sc_strategy = options['factorize_sc_strategy']['type']
        self._factorize_sc_flag = False
        self._form_sc_flag = self._diagnostic_flag
        self._use_direct_preconditioner_flag = False
        if self._factorize_sc_strategy is not None:
            self._factorize_sc_flag = True
            self._form_sc_flag = True
            self._use_direct_preconditioner_flag = True
            def direct_preconditioner(r):
                return self.schur_complement_solver.do_back_solve(r)
            if self._factorize_sc_strategy == 'adaptive-mu':
                self._prev_mu = np.nan
                H_strategy = 'None'
            elif self._factorize_sc_strategy == 'start':
                self._factorize_sc_threshold = options['factorize_sc_strategy']['threshold']
                H_strategy = 'all'
            elif self._factorize_sc_strategy == 'adaptive':
                self._factorize_sc_threshold = options['factorize_sc_strategy']['threshold']
                H_strategy = 'None'
            else:
                raise NotImplementedError
        else:
            H_strategy = 'all'
            direct_preconditioner = None
        

        if options['preconditioner']['type'] == 'bfgs':
            _preconditioner = BfgsPreqn(H_strategy=H_strategy, diagnostic_flag=self._diagnostic_flag, direct_preconditioner=direct_preconditioner)
        elif options['preconditioner']['type'] == 'lbfgs':
            _preconditioner = LbfgsPreqn(memory=options['preconditioner']['memory'],
                                         strategy=options['preconditioner']['strategy'],
                                         diagnostic_flag=self._diagnostic_flag)
        elif options['preconditioner']['type'] == None:
            _preconditioner = PreqnNone()
        else:
            raise NotImplementedError
        
        self._cg = ConjugateGradientSolver(tol=options['cg']['tol'],
                                           max_iter=options['cg']['max_iter'],
                                           stopping_criteria=options['cg']['stopping_criteria'],
                                           beta_update=options['cg']['beta_update'],
                                           beta_restart=options['cg']['beta_restart'],
                                           preconditioner=_preconditioner)

        self._adjust_sc_tol: bool = options['adjust_sc_tol']
        if self._adjust_sc_tol:
            self._sc_tol_reduction_factor = options['sc_tol_reduction_factor']
        else:
            self._sc_tol_reduction_factor = None
        self._diagnostic_info = dict()
        self._prev_coupling: NDArray = np.empty(0)
        self._initialize_pcg = options['initialize_pcg']

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
        if self._form_sc_flag:
            timer.start('gather_all_nonzero_elements')
            nonzero_rows, nonzero_cols = _get_all_nonzero_elements_in_sc(self.border_matrices, timer)
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
        for ndx in self.local_block_indices:
            timer.start('block_factorize')
            sub_res = self.subproblem_solvers[ndx].do_numeric_factorization(matrix=matrix.get_block(ndx, ndx),
                                                                            raise_on_error=False)
            timer.stop('block_factorize')
            _process_sub_results(res, sub_res)
            if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
                break
        res = _gather_results(res)
        if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
            if raise_on_error:
                raise RuntimeError('Numeric factorization unsuccessful; status: ' + str(res.status))
            else:
                return res
        if self._form_sc_flag:
            timer.start('form_SC')
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
                    timer.start('block_back_solve')
                    contribution = solver.do_back_solve(_rhs)
                    timer.stop('block_back_solve')
                    timer.start('dot_product')
                    contribution = A.dot(contribution)
                    timer.stop('dot_product')
                    self.schur_complement.data[self.sc_data_slices[ndx][row_ndx]] -= contribution[border_matrix.nonzero_rows]
                    for indptr in range(A.indptr[row_ndx], A.indptr[row_ndx + 1]):
                        col = A.indices[indptr]
                        val = A.data[indptr]
                        _rhs[col] -= val

            timer.start('communicate')
            timer.start('zeros')
            sc = np.zeros(self.schur_complement.data.size, dtype=np.double)
            timer.stop('zeros')
            timer.start('Barrier')
            comm.Barrier()
            timer.stop('Barrier')
            timer.start('Allreduce')
            comm.Allreduce(self.schur_complement.data, sc)
            timer.stop('Allreduce')
            self.schur_complement.data = sc
            timer.start('add')
            sc = self.schur_complement + block_matrix.get_block(self.block_dim-1, self.block_dim-1).tocoo()
            timer.stop('add')
            timer.stop('communicate')
            timer.stop('form_SC')
            #cond = np.linalg.cond(sc.toarray())
            #print(f'SC condition number: {cond}')
        if self._form_sc_flag and self._factorize_sc_flag:
            timer.start('factor_SC')
            sub_res = self.schur_complement_solver.do_symbolic_factorization(sc, raise_on_error=raise_on_error)
            _process_sub_results(res, sub_res)
            if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
                timer.stop('factor_SC')
                return res
            sub_res = self.schur_complement_solver.do_numeric_factorization(sc)
            _process_sub_results(res, sub_res)
            timer.stop('factor_SC')
        else:
            pass

        return res

    def do_back_solve(self, rhs, timer=None, barrier=None, ip_iter=None):
        """
        Performs a back solve with the factorized matrix. Should only be called after
        do_numeric_factorization.

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
            contribution  = self.subproblem_solvers[ndx].do_back_solve(rhs.get_block(ndx))
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

        # TODO: Set tolerance according to rule from paper
        if barrier is None or not self._adjust_sc_tol:
            tol = self._cg.tol
        else:
            tol = np.maximum(self._cg.tol, barrier * self._sc_tol_reduction_factor)

        # if ip_iter == 0:
        #     ls = copy.copy(self.schur_complement_solver)
        #     def _precond_0(v):
        #         return ls.do_back_solve(v)
        #     self._tmp_precond = _precond_0
        #     self._facorize_sc_flag = False
        
        if self._initialize_pcg:
            if ip_iter == 0:
                self._prev_coupling = np.zeros_like(schur_complement_rhs)
            x0 = self._prev_coupling
        else:
            x0 = None
            

        timer.start('pcg')
        coupling, info = self.pcg_schur_solve(schur_complement_rhs=schur_complement_rhs, ip_iter=ip_iter,
                                              timer=timer, tol=tol, x0=x0, use_direct_preconditioner=self._use_direct_preconditioner_flag)
        timer.stop('pcg')

        if self._initialize_pcg:
            self._prev_coupling = coupling

        # logging
        pcg_info = dict()
        pcg_info['num_iter'] = info['n_iter']
        if rank == 0:
            print(f'PCG: {info["n_iter"]} iterations')
        if self._diagnostic_flag:
            sc = self.schur_complement + self.block_matrix.get_block(self.block_dim-1, self.block_dim-1).tocoo()
            sc_eigs = np.linalg.eigvals(sc.toarray())
            pcg_info['cond_S'] = np.max(np.abs(sc_eigs)) / np.min(np.abs(sc_eigs))
            hsc = self._cg._preconditioner.H @ sc.toarray()
            psc_eigs = np.linalg.eigvals(hsc)
            pcg_info['cond_HS'] = np.max(np.abs(psc_eigs)) / np.min(np.abs(psc_eigs))
            pcg_info['num_iter'] = info['n_iter']
            pcg_info['residuals'] = info['residuals']
            pcg_info['eig_S'] = sc_eigs
            pcg_info['eig_HS'] = psc_eigs
            ns = rhs.size
            pcg_info['sparsity_S'] = 1 - np.count_nonzero(sc.toarray())/(ns**2)
            pcg_info['sparsity_H'] = 1 - np.count_nonzero(self._cg._preconditioner.H)/(ns**2)
        self._diagnostic_info[ip_iter] = pcg_info

        ns = coupling.size

        if self._factorize_sc_strategy == 'start':
            self._factorize_sc_flag = False
            if info['n_iter'] > ns * self._factorize_sc_threshold:
                self._use_direct_preconditioner_flag = False
        elif self._factorize_sc_strategy == 'adaptive':
            if info['n_iter'] > ns * self._factorize_sc_threshold:
                self._factorize_sc_flag = True
            else:
                self._factorize_sc_flag = False
            self._use_direct_preconditioner_flag = True
        elif self._factorize_sc_strategy == 'adaptive-mu':
            if self._prev_mu == np.nan: #ip_iter = 0
                self._prev_mu = barrier
                self._factorize_sc_flag = False
            if self._prev_mu != barrier:
                self._factorize_sc_flag = True
                self._prev_mu = barrier
            else:
                self._factorize_sc_flag = False
            self._use_direct_preconditioner_flag = True
        
        if self._diagnostic_flag:
            self._form_sc_flag = True
        else:
            self._form_sc_flag = self._factorize_sc_flag
  
        # TODO: Return when negative curvature is detected
        # if self._diagnostic_flag:
        #     _coupling = self.schur_complement_solver.do_back_solve(schur_complement_rhs)

        
        for ndx in self.local_block_indices:
            timer.start('block_back_solve')
            A = self.block_matrix.get_block(self.block_dim-1, ndx)
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

        #num_pos += self.schur_complement.shape[0]
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


    def pcg_schur_solve(self, schur_complement_rhs, ip_iter, timer=None, tol=1e-8, x0=None, use_direct_preconditioner=False):

        if timer is None:
            timer = HierarchicalTimer()

        D = self.block_matrix.get_block(self.block_dim-1, self.block_dim-1).tocoo()

        def schur_matvec_mpi(u: NDArray) -> NDArray:
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
           # print(res.size)
            comm.Allreduce(res, res_global)
            timer.stop('communicate')
            res_global += D.dot(u)
            return res_global

        coupling, info = self._cg.solve(linear_operator=schur_matvec_mpi,
                                        rhs=schur_complement_rhs,
                                        tol=tol,
                                        ip_iter=ip_iter,
                                        x0=x0, timer=timer, use_direct_preconditioner=use_direct_preconditioner)
        

        return coupling, info


class ConjugateGradientSolver:

    def __init__(self,
                 tol: float,
                 max_iter: int,
                 stopping_criteria: str,
                 beta_update: str,
                 beta_restart: bool,
                 preconditioner: Preqn):
        self._preconditioner = preconditioner
        if stopping_criteria == 'inf_res':
            def f_stopping_criteria(r_k, z_k, r_0, rhs):
                return np.max(np.abs(r_k))
        elif stopping_criteria == 'inf_rel_res':
            def f_stopping_criteria(r_k, z_k, r_0, rhs):
                return np.max(np.abs(r_k))/np.max(np.abs(r_0))
        elif stopping_criteria == '2_rel_res':
            def f_stopping_criteria(r_k, z_k, r_0, rhs):
                return np.sqrt(r_k.T.dot(r_k))/np.sqrt(r_0.T.dot(r_0))
        elif stopping_criteria == '2_rel_rhs':
            def f_stopping_criteria(r_k, z_k, r_0, rhs):
                return np.sqrt(r_k.T.dot(r_k))/np.sqrt(rhs.T.dot(rhs))
        elif stopping_criteria == '2_H_res':
            def f_stopping_criteria(r_k, z_k, r_0, rhs):
                return np.sqrt(r_k.T.dot(z_k))
        else:
            raise NotImplementedError
        self._compute_merit = f_stopping_criteria
        self._stopping_criteria = stopping_criteria

        if beta_update == 'FR':
            def f_beta_update(norm1, norm2, r_k, z_prev):
                return norm1 / norm2
        elif beta_update == 'PR':
            def f_beta_update(norm1, norm2, r_k, z_prev):
                return (norm1 - r_k.T.dot(z_prev))/norm2
        else:
            raise NotImplementedError
        self._compute_beta_update = f_beta_update
        self._beta_update = beta_update

        self._beta_restart = beta_restart

        self._max_iter = max_iter
        self.tol = tol
        
    
    def solve(self,
              linear_operator: Callable[[NDArray], NDArray],
              rhs: NDArray,
              tol: float,
              ip_iter: int,
              x0: Optional[NDArray] = None,
              timer: Optional[HierarchicalTimer] = None,
              use_direct_preconditioner: bool = False
              ) -> Tuple[NDArray, Dict]:
        if timer is None:
            timer = HierarchicalTimer()
        
        if x0 is None:
            x_k: NDArray = np.zeros_like(rhs)
        else:
            x_k: NDArray = x0.copy()

        info = dict()
        residuals = list()
        status: int = 1

        k = 0
        timer.start('SC_matvec')
        if x0 is None:
            r_k = - rhs.copy()
        else:
            r_k = linear_operator(x_k) - rhs
        timer.stop('SC_matvec')
        d_k = np.empty_like(rhs)
        w_k = np.empty_like(rhs)
        z_k = np.empty_like(rhs)
        z_k_prev = np.empty_like(rhs)
        n = rhs.size
        #norm_rhs = np.sqrt(rhs.T.dot(rhs))
        norm1 = 0
        #r0 = np.max(np.abs(r_k))
        r0 = r_k.copy()
        for k in range(self._max_iter):
            if self._beta_update == 'PR':
                z_k_prev = z_k.copy()
            # TODO: See if this makes sense generally - intermittently factorizing S - incorporate into preqn?
            # if ip_iter > -1:
            #     z_k = self._preconditioner.preqn(ip_iter, k, r_k, d_k, w_k)
            # else:
            #     zz_k = self._preconditioner.preqn(ip_iter, k, r_k, d_k, w_k)
            #     z_k = self._tmp_precond(r_k)
            timer.start('precondition')
            z_k = self._preconditioner.preqn(ip_iter, k, r_k, d_k, w_k, timer, use_direct_preconditioner)
            timer.stop('precondition')
            norm2 = norm1
            norm1 = r_k.T.dot(z_k)
            res = self._compute_merit(r_k, z_k, r0, rhs)
            residuals.append(res)
            if res < tol:
               status = 0
               break
            if k > 0:
                if np.mod(k, n) != 0 or not self._beta_restart:
                    beta = self._compute_beta_update(norm1, norm2, r_k, z_k_prev)
                else:
                    beta = 0

                d_k = - z_k + beta * d_k
            else:
                d_k = -z_k

            timer.start('SC_matvec')
            w_k = linear_operator(d_k)
            timer.stop('SC_matvec')
            denom = d_k.T.dot(w_k)
            if denom <= 0:
                status = 2
                if rank == 0:
                    print(f'WARNING: PCG: Schur complement has negative or zero eigenvalues (badness: {denom}).')
                info['residuals'] = np.array(residuals)
                info['status'] = status
                info['n_iter'] = k + 1
                return x_k, info
            
            alpha_k = norm1 / denom
            x_k += alpha_k * d_k
            # Recomputing Residual - can help with roundoff
            if np.mod(k, 50) != 0:
                r_k += alpha_k * w_k
            else:
                r_k = linear_operator(x_k) - rhs
            #r_k += alpha_k * w_k

        if res >= tol:
            status = 1
            if rank == 0:
                print(f'WARNING: PCG: Maximum iterations exceeded. Residual: {res}.')

        info['residuals'] = np.array(residuals)
        info['status'] = status
        info['n_iter'] = k + 1
        return x_k, info



class Preqn:
    H: NDArray
    _direct_preconditioner: Optional[Callable] = None

    def preqn(self, ip_iter: int, cg_iter: int, residual: NDArray, d_k: NDArray, w_k:NDArray,
           timer: Optional[HierarchicalTimer] = None, use_direct_preconditioner: bool = False):
        raise NotImplementedError

class PreqnNone(Preqn):

    def preqn(self, ip_iter: int, cg_iter: int, residual: NDArray, d_k: NDArray, w_k:NDArray,
           timer: Optional[HierarchicalTimer] = None, use_direct_preconditioner: bool = False):
        if ip_iter == 0 and cg_iter == 0:
            n = residual.size
            self.H = np.eye(n)
        return residual


class BfgsPreqn(Preqn):

    def __init__(self, H_strategy: str = 'all',
                 diagnostic_flag: bool = False,
                 direct_preconditioner: Optional[Callable] = None):
        self._H_strategy: str = H_strategy
        self._diagnostic_flag: bool = diagnostic_flag
        self.H: NDArray = np.empty(())
        self.H_prev: NDArray = np.empty(())
        self._direct_preconditioner: Callable = direct_preconditioner

    def preqn(self, ip_iter: int, cg_iter: int, residual: NDArray, d_k: NDArray, w_k:NDArray,
           timer: Optional[HierarchicalTimer] = None, use_direct_preconditioner: bool = False):
        if timer is None:
            timer = HierarchicalTimer()
        n = residual.size
        if cg_iter > 0:
            if self._H_strategy == 'all':
                timer.start('form_H')
                y = w_k.reshape(-1,1)
                s = d_k.reshape(-1,1)
                rho = 1/(y.T.dot(s))
                vv = (np.eye(n) - rho * y.dot(s.T))
                self.H = vv.T @ self.H @ vv + rho * s.dot(s.T)
                timer.stop('form_H')
            elif self._H_strategy == 'None':
                pass
            else:
                raise NotImplementedError

        if ip_iter == 0:
            # no preconditioning
            self.H  = np.eye(n)
            if not use_direct_preconditioner:
                return residual

        elif cg_iter == 0:
            self.H_prev = self.H.copy()
            #self.H = np.eye(n)
        
        if not use_direct_preconditioner:
            timer.start('precond_matvec')
            r = self.H_prev.dot(residual)
            timer.stop('precond_matvec')
        else:
            timer.start('SC_back_solve')
            r = self._direct_preconditioner(residual)
            timer.stop('SC_back_solve')

        return r


class LbfgsPreqn(Preqn):

    def __init__(self, memory: int, strategy: str = 'uniform', diagnostic_flag: bool = False):
        #self.options = options
        self._mem_len: int = memory
        self._strategy: str = strategy
        self._diagnostic_flag: bool = diagnostic_flag
        self._Y_STORE = deque(maxlen=self._mem_len)
        self._S_STORE = deque(maxlen=self._mem_len)
        self._RHO_STORE = deque(maxlen=self._mem_len)
        self._Y_STORE_NXT = deque(maxlen=self._mem_len)
        self._S_STORE_NXT = deque(maxlen=self._mem_len)
        self._RHO_STORE_NXT = deque(maxlen=self._mem_len)
        self.H: NDArray = np.empty(())
        
        if self._strategy == 'uniform':
            assert np.mod(self._mem_len, 2) == 0
            self._cycle = 1
            #self._k_indxs = np.zeros((self._mem_len/2))
            self._l = np.arange(1, (self._mem_len/2) + 1)
            self._k_indxs = []
            self._last_s_store = np.empty(())
            self._last_y_store = np.empty(())
            self._last_rho_store = np.nan
        if self._strategy == 'rayleigh':
            self._quotients = []

    def _uniform_sample(self, k: int):
        k_indxs = ((self._mem_len/2) + self._l - 1) * (2 ** self._cycle)
        in_idx = self._l[np.where(k_indxs == k)]
        if len(in_idx) == 0:
            return None, None
        else:
            l_out = (2*in_idx[0] - 1) * (2 ** (self._cycle - 1))
            rem_idx = self._k_indxs.index(int(l_out))
            self._k_indxs.remove(int(l_out))
            self._k_indxs.append(k)
            if in_idx[0] == self._mem_len/2:
                self._cycle += 1
            return k, rem_idx


    def preqn(self, ip_iter: int, cg_iter: int, residual: NDArray, d_k: NDArray, w_k:NDArray,
           timer: Optional[HierarchicalTimer] = None, use_direct_preconditioner: bool = False):
        if timer is None:
            timer = HierarchicalTimer()
        n = residual.size
        if cg_iter > 0:
            if self._strategy == 'last':
                self._Y_STORE_NXT.append(w_k.reshape(-1,1))
                self._S_STORE_NXT.append(d_k.reshape(-1,1))
                self._RHO_STORE_NXT.append(1.0/(w_k.T.dot(d_k)))
            elif self._strategy == 'first':
                if len(self._Y_STORE_NXT) < self._mem_len:
                    self._Y_STORE_NXT.append(w_k.reshape(-1,1))
                    self._S_STORE_NXT.append(d_k.reshape(-1,1))
                    self._RHO_STORE_NXT.append(1.0/(w_k.T.dot(d_k)))
            elif self._strategy == 'uniform':
                if len(self._Y_STORE_NXT) < self._mem_len:
                    self._Y_STORE_NXT.append(w_k.reshape(-1,1))
                    self._S_STORE_NXT.append(d_k.reshape(-1,1))
                    self._RHO_STORE_NXT.append(1.0/(w_k.T.dot(d_k)))
                    self._k_indxs.append(cg_iter - 1)
                else:
                    _, k_out = self._uniform_sample(cg_iter - 1)
                    if k_out is not None:
                        #TODO: Not efficient in deques
                        del self._Y_STORE_NXT[k_out]
                        del self._S_STORE_NXT[k_out]
                        del self._RHO_STORE_NXT[k_out]
                        #del self._k_indxs[k_out]
                        #self._k_indxs.append(cg_iter)
                        self._Y_STORE_NXT.append(w_k.reshape(-1,1))
                        self._S_STORE_NXT.append(d_k.reshape(-1,1))
                        self._RHO_STORE_NXT.append(1.0/(w_k.T.dot(d_k)))

                        self._last_y_store = np.empty(())
                        self._last_s_store = np.empty(())
                        self._last_rho_store = np.nan
                    else:
                        self._last_y_store = w_k.reshape(-1,1)
                        self._last_s_store = d_k.reshape(-1,1)
                        self._last_rho_store = 1.0/(w_k.T.dot(d_k))
            elif self._strategy == 'rayleigh':
                if len(self._Y_STORE_NXT) < self._mem_len:
                    self._Y_STORE_NXT.append(w_k.reshape(-1,1))
                    self._S_STORE_NXT.append(d_k.reshape(-1,1))
                    self._RHO_STORE_NXT.append(1.0/(w_k.T.dot(d_k)))
                    self._quotients.append(d_k.T.dot(w_k)/(d_k.T.dot(d_k)))
                else:
                    q = d_k.T.dot(w_k)/(d_k.T.dot(d_k))
                    min_q = np.min(self._quotients)
                    if q < min_q:
                        max_q_idx = np.argmax(self._quotients)
                        self._Y_STORE_NXT[max_q_idx] = w_k.reshape(-1,1)
                        self._S_STORE_NXT[max_q_idx] = d_k.reshape(-1,1)
                        self._RHO_STORE_NXT[max_q_idx] = 1.0/(w_k.T.dot(d_k))
                        self._quotients[max_q_idx] = q

        if ip_iter == 0:
            # no preconditioning
            if self._diagnostic_flag:
                self.H = np.eye(n)
            return residual
        elif cg_iter == 0:
            if len(self._Y_STORE_NXT) > 0:
                if self._strategy == 'rayleigh':
                    self._quotients.clear()
                if self._strategy == 'uniform':
                    if self._last_rho_store is not np.nan:
                        self._Y_STORE_NXT.append(self._last_y_store)
                        self._S_STORE_NXT.append(self._last_s_store)
                        self._RHO_STORE_NXT.append(self._last_rho_store)
                self._Y_STORE.clear()
                self._S_STORE.clear()
                self._RHO_STORE.clear()

                for _ in range(len(self._Y_STORE_NXT)):
                    self._Y_STORE.append(self._Y_STORE_NXT.popleft())
                    self._S_STORE.append(self._S_STORE_NXT.popleft())
                    self._RHO_STORE.append(self._RHO_STORE_NXT.popleft())
   
                #debug
                if self._diagnostic_flag:
                    timer.start('form H')
                    y: NDArray = self._Y_STORE[-1]
                    s: NDArray = self._S_STORE[-1]
                    rho: NDArray = self._RHO_STORE[-1]
                    H: NDArray = (s.T.dot(y)/y.T.dot(y)) * np.eye(n)
                    for j in range(0, len(self._Y_STORE)-2):
                        y: NDArray = self._Y_STORE[-(j+2)]
                        s: NDArray = self._S_STORE[-(j+2)]
                        rho: NDArray = self._RHO_STORE[-(j+2)]
                        #H = (np.eye(n) - rho * s.dot(y.T)) @ H @ (np.eye(n) - rho * s.dot(y.T)) + rho * s.dot(s.T)
                        vv = (np.eye(n) - rho * y.dot(s.T))
                        H = vv.T @ H @ vv + rho * s.dot(s.T)
                    self.H = H
                    timer.stop('form H')
            else:
                if self._diagnostic_flag:
                    self.H = np.eye(n)
            
            if self._strategy == 'uniform':
                self._cycle = 1
                self._k_indxs = []

            #assert self._Y_STORE.__len__() == 0 and self._S_STORE.__len__() == 0 and self._RHO_STORE.__len__() == 0

        # compute z = M^{-1} r
        timer.start('precond_matvec')
        r2 = self._precond_mv(residual)
        #r2 = self.H @ residual
        timer.stop('precond_matvec')
        return r2
    
    def _precond_mv(self, u: NDArray) -> NDArray:
        q: NDArray = u.copy()
        n = u.size
        bound: int = len(self._RHO_STORE)
        alphas = np.zeros(bound)
        for i in reversed(range(1,bound)):
            alphas[i] = self._RHO_STORE[i] * self._S_STORE[i].T.dot(q)
            q -= alphas[i] * self._Y_STORE[i].flatten()

        gamma: NDArray = (self._S_STORE[bound-1].T.dot(self._Y_STORE[bound-1])/self._Y_STORE[bound-1].T.dot(self._Y_STORE[bound-1]))
        r =  gamma.reshape(()) * q
        for i in range(1, bound):
            beta = self._RHO_STORE[i] * self._Y_STORE[i].T.dot(r)
            r += self._S_STORE[i].flatten() * (alphas[i] - beta)
        
        return r
