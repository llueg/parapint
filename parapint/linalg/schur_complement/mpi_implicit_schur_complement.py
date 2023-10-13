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
                 schur_complement_solver: LinearSolverInterface, diagnostic_flag: bool = False):
        self.subproblem_solvers = subproblem_solvers
        #self.schur_complement_solver = schur_complement_solver
        self.block_dim = 0
        self.block_matrix = None
        self.local_block_indices = list()
        #self.schur_complement = coo_matrix((0, 0))
        self.border_matrices: Dict[int, _BorderMatrix] = dict()
        self.sc_data_slices = dict()
        self._preconditioner = PREQN(options={'memory': 20})
        self._diagnostic_flag = diagnostic_flag
        self.schur_complement_solver = schur_complement_solver
        self._diagnostic_info = dict()

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
        if self._diagnostic_flag:
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
        timer.start('form SC')
        for ndx in self.local_block_indices:
            timer.start('factorize')
            sub_res = self.subproblem_solvers[ndx].do_numeric_factorization(matrix=matrix.get_block(ndx, ndx),
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
                timer.stop('form SC')
                return res
        if self._diagnostic_flag:
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
            timer.stop('form SC')
            #cond = np.linalg.cond(sc.toarray())
            #print(f'SC condition number: {cond}')

            timer.start('factor SC')
            sub_res = self.schur_complement_solver.do_symbolic_factorization(sc, raise_on_error=raise_on_error)
            _process_sub_results(res, sub_res)
            if res.status not in {LinearSolverStatus.successful, LinearSolverStatus.warning}:
                timer.stop('factor SC')
                return res
            sub_res = self.schur_complement_solver.do_numeric_factorization(sc)
            _process_sub_results(res, sub_res)
            timer.stop('factor SC')

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

        timer.start('rhs')
        schur_complement_rhs = np.zeros(rhs.get_block(self.block_dim - 1).size, dtype='d')
        for ndx in self.local_block_indices:
            A = self.block_matrix.get_block(self.block_dim-1, ndx)
            timer.start('back_solve')
            contribution  = self.subproblem_solvers[ndx].do_back_solve(rhs.get_block(ndx))
            timer.stop('back_solve')
            schur_complement_rhs -= A.tocsr().dot(contribution.flatten())
        res = np.zeros(rhs.get_block(self.block_dim - 1).shape[0], dtype='d')
        comm.Allreduce(schur_complement_rhs, res)
        schur_complement_rhs = rhs.get_block(self.block_dim - 1) + res
        timer.stop('rhs')
        result = rhs.copy_structure()

        # TODO: Set tolerance according to rule from paper
        # TODO: LinearSolveroptions  - also relevant for preconditioner
        if barrier is None:
            tol = 1e-8
        else:
            tol = barrier
        timer.start('pcg')
        coupling, info = self.pcg_schur_solve(schur_complement_rhs=schur_complement_rhs, ip_iter=ip_iter,
                                              timer=timer, tol=tol)
        timer.stop('pcg')

        # logging
        if self._diagnostic_flag:
            pcg_info = dict()
            sc = self.schur_complement + self.block_matrix.get_block(self.block_dim-1, self.block_dim-1).tocoo()
            sc_eigs = np.linalg.eigvals(sc.toarray())
            pcg_info['cond_S'] = np.max(np.abs(sc_eigs)) / np.min(np.abs(sc_eigs))
            psc_eigs = np.linalg.eigvals(self._preconditioner.H @ sc.toarray())
            pcg_info['cond_HS'] = np.max(np.abs(psc_eigs)) / np.min(np.abs(psc_eigs))
            pcg_info['num_iter'] = info['n_iter']
            pcg_info['residuals'] = info['residuals']
            self._diagnostic_info[ip_iter] = pcg_info

        # TODO: Return when negative curvature is detected
        # if self._diagnostic_flag:
        #     _coupling = self.schur_complement_solver.do_back_solve(schur_complement_rhs)

        timer.start('block_back_solve')
        for ndx in self.local_block_indices:
            A = self.block_matrix.get_block(self.block_dim-1, ndx)
            result.set_block(ndx, self.subproblem_solvers[ndx].do_back_solve(rhs.get_block(ndx) -
                                                                             A.tocsr().transpose().dot(coupling.flatten())))

        result.set_block(self.block_dim-1, coupling)
        timer.stop('block_back_solve')


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


    def pcg_schur_solve(self, schur_complement_rhs, ip_iter, timer=None, tol=1e-8):

        if timer is None:
            timer = HierarchicalTimer()

        D = self.block_matrix.get_block(self.block_dim-1, self.block_dim-1).tocoo()

        def schur_matvec_mpi(u: NDArray) -> NDArray:
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
           # print(res.size)
            comm.Allreduce(res, res_global)
            timer.stop('communicate')
            res_global += D.dot(u)
            return res_global


        # if self._preconditioner is None:
        #     precond_mv = lambda u: u
        # else:
        #     precond_mv = lambda u: self._preconditioner.dot(u)
        #         #L-BFGS update based on s_iterates, y_iterates


            

        #lin_op = LinearOperator(shape=D.shape, matvec=schur_matvec_mpi, dtype=float)

        # precond = LinearOperator(shape=D.shape, matvec=precond_mv, dtype=float)

        #iters = 0
        # def nonlocal_iterate(xk):
        #     nonlocal rank
        #     if rank == 0:
        #         nonlocal iters
        #         iters+=1
        
        #ns: int = D.shape[0]
        # mem_len: int = 10
        # prev_x = np.zeros_like(schur_complement_rhs)
        # #s_iterates: List[NDArray] = list()
        # s_iterates = deque(maxlen=mem_len)
        # pcg_iter: int = 0
        # mem_len: int = 10
        # def save_delta_x(xk):
        #     timer.start('precondition')
        #     nonlocal prev_x
        #     nonlocal pcg_iter
        #     pcg_iter += 1
        #     delta_x = xk - prev_x
        #     s_iterates.appendleft(delta_x)
        #     prev_x = xk
        #     timer.stop('precondition')

        #coupling, info = cg(A=lin_op, b=schur_complement_rhs, tol=1e-8, callback=save_delta_x, maxiter=ns)
        #coupling, info = cg(A=lin_op, b=schur_complement_rhs, tol=1e-8, maxiter=ns)

        #def precond(u: NDArray) -> NDArray:
        #    return self._preconditioner.preqn(ip_iter, pcg_iter, schur_complement_rhs, u, schur_matvec_mpi(u))
        

        coupling, info = self.cg(linear_operator=schur_matvec_mpi, rhs=schur_complement_rhs, tol=1e-7, maxiter=1000, ip_iter=ip_iter)
        #print(f'ncg_iter: {info["n_iter"]}')
        #print(f'status: {info["status"]}')
        # timer.start('precondition')
        # y_iterates = deque([schur_matvec_mpi(s_iterate) for s_iterate in s_iterates])
        # sl = s_iterates.popleft()
        # yl = y_iterates.popleft()
        # M = (sl.T.dot(yl) / yl.T.dot(yl)) * np.eye(ns)
        # V = np.empty_like(M)
        # while s_iterates and y_iterates:
        #     s = s_iterates.popleft()
        #     y = y_iterates.popleft()
        #     if y.T.dot(s) > 0:
        #         rho = 1.0 / (y.T.dot(s))
        #         V = np.eye(ns) - rho * y.dot(s.T)
        #         M = V.T.dot(M).dot(V) + rho * s.dot(s.T)

        # self._preconditioner = M
        # self._y_iterates = y_iterates
        # self._s_iterates = s_iterates
        # timer.stop('precondition')
        #print(f'num_iter: {iters}')
        #print(f'info: {info}')
        #timer.stop('pcg')
        return coupling, info
    

    def cg(self,
           linear_operator: Callable[[NDArray], NDArray],
           rhs: NDArray,
           tol: float,
           maxiter: int,
           ip_iter: int,
           #preconditioner: Optional[Callable[[NDArray], NDArray]] = None,
           x0: Optional[NDArray] = None,
           timer: Optional[HierarchicalTimer] = None
           ) -> Tuple[NDArray, Dict]:

        if timer is None:
            timer = HierarchicalTimer()
        
        if x0 is None:
            x_k: NDArray = np.zeros_like(rhs)
        else:
            x_k: NDArray = x0.copy()

        # if preconditioner is None:
        #     preconditioner = lambda u: u
        info = dict()
        residuals = list()
        status: int = 1

        #r_k: NDArray = rhs - linear_operator(x_k)
        #old_sq_residual = r_k.T.dot(r_k)
        k = 0
        #timer.start('pcg')
        #p_k = r_k.copy()
        r_k = - rhs.copy()
        x_k = np.zeros_like(rhs)
        #x_k = np.random.rand(rhs.size)
        d_k = np.empty_like(rhs)
        w_k = np.empty_like(rhs)
        z_k = np.empty_like(rhs)
        n = rhs.size
        norm1 = 0
        for k in range(maxiter):
            #z_k = preconditioner(r_k)
            #z_k_prev = z_k
            z_k = self._preconditioner.preqn(ip_iter, k, r_k, d_k, w_k)
            norm2 = norm1
            norm1 = r_k.T.dot(z_k)
            #if np.sqrt(z_k.T.dot(z_k)) < tol:
            residuals.append(np.max(np.abs(r_k)))
            if np.max(np.abs(r_k)) < tol:
               status = 0
               break
            if k > 0:
                # if np.mod(k, n) != 0:
                #     beta = norm1 / norm2 
                # else:
                #     beta = 0
                beta = norm1 / norm2 
                #beta = (norm1 - r_k.T.dot(z_k_prev))/norm2
                d_k = - z_k + beta * d_k
            else:
                d_k = -z_k

            w_k = linear_operator(d_k)
            denom = d_k.T.dot(w_k)
            if denom <= 0:
                status = 2
                #print(f'Negative curvature detected in CG {denom}')
                print('WARNING: Schur complement has negative or zero eigenvalues. (PCG)')
                #break
            
            alpha_k = norm1 / denom
            x_k += alpha_k * d_k
            r_k += alpha_k * w_k

            # if r_k.T.dot(r_k) < tol:
            #     break            
            
            #norm2 = norm1
            #norm1 = r_k.T.dot(z_k)
        # for k in range(maxiter):
        #     if k > 0:
        #         if np.mod(k, n) != 0:
        #             beta = norm1 / norm2 
        #         else:
        #             beta = 0

        #         d_k = - z_k + beta * d_k
        #     w_k = linear_operator(d_k)
        #     denom = d_k.T.dot(w_k)
        #     if denom <= 0:
        #         info = 2
        #         #print(f'Negative curvature detected in CG {denom}')
        #         print('WARNING: Schur complement has negative or zero eigenvalues. (PCG)')
        #         #break
            
        #     alpha_k = norm1 / denom
        #     x_k += alpha_k * d_k
        #     r_k += alpha_k * w_k

        #     z_k = preconditioner(r_k)
        #     if z_k.T.dot(z_k) < tol:
        #         break
        #     norm2 = norm1
        #     norm1 = r_k.T.dot(z_k)

        #timer.stop('pcg')
        info['residuals'] = np.array(residuals)
        info['status'] = status
        info['n_iter'] = k + 1
        return x_k, info
    


class PREQN:

    def __init__(self, options: Dict):
        self.options = options
        self._mem_len = options['memory']
        self._Y_STORE = deque(maxlen=options['memory'])
        self._S_STORE = deque(maxlen=options['memory'])
        self._RHO_STORE = deque(maxlen=options['memory'])
        self._Y_STORE_NXT = deque(maxlen=options['memory'])
        self._S_STORE_NXT = deque(maxlen=options['memory'])
        self._RHO_STORE_NXT = deque(maxlen=options['memory'])
        self.H: NDArray = np.empty(())
        self._strategy: str = 'last'
        if self._strategy == 'uniform':
            assert np.mod(self._mem_len, 2) == 0
            self._cycle = 1

    def preqn(self, ip_iter: int, cg_iter: int, residual: NDArray, d_k: NDArray, w_k:NDArray):
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
                else:
                    # TODO
                    pass


        if ip_iter == 0:
            # no preconditioning
            self.H = np.eye(n)
            return residual
        elif cg_iter == 0:
            if len(self._Y_STORE_NXT) > 0:
                # empty deques
                #self._Y_STORE.clear()
                #self._S_STORE.clear()
                #self._RHO_STORE.clear()
                for _ in range(len(self._Y_STORE_NXT)):
                    self._Y_STORE.append(self._Y_STORE_NXT.pop())
                    self._S_STORE.append(self._S_STORE_NXT.pop())
                    self._RHO_STORE.append(self._RHO_STORE_NXT.pop())
                #debug
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
                # y: NDArray = self._Y_STORE.pop()
                # s: NDArray = self._S_STORE.pop()
                # rho: NDArray = self._RHO_STORE.pop()
                # H: NDArray = (s.T.dot(y)/y.T.dot(y)) * np.eye(n)
                # for j in range(0, int(np.minimum(self.options['memory'] - 1, len(self._Y_STORE)))):
                #     y: NDArray = self._Y_STORE.pop()
                #     s: NDArray = self._S_STORE.pop()
                #     rho: NDArray = self._RHO_STORE.pop()
                #     #H = (np.eye(n) - rho * s.dot(y.T)) @ H @ (np.eye(n) - rho * s.dot(y.T)) + rho * s.dot(s.T)
                #     vv = (np.eye(n) - rho * y.dot(s.T))
                #     H = vv.T @ H @ vv + rho * s.dot(s.T)
                # self.H = H
            else:
                self.H = np.eye(n)

            #assert self._Y_STORE.__len__() == 0 and self._S_STORE.__len__() == 0 and self._RHO_STORE.__len__() == 0

        # compute z = M^{-1} r
        r2 = self._precond_mv(residual)
        #r1 = self.H @ residual
        #r3 = self._precond_mv2(residual)
        #print(np.allclose(r1, r2))
        return r2
    
    def _precond_mv(self, u: NDArray) -> NDArray:
        q: NDArray = u.copy()
        n = u.size
        bound: int = len(self._RHO_STORE)
        alphas = np.zeros(bound)
        for i in reversed(range(bound)):
            alphas[i] = self._RHO_STORE[i] * self._S_STORE[i].T.dot(q)
            q -= alphas[i] * self._Y_STORE[i].flatten()

        gamma: NDArray = (self._S_STORE[bound-1].T.dot(self._Y_STORE[bound-1])/self._Y_STORE[bound-1].T.dot(self._Y_STORE[bound-1]))
        r =  gamma.reshape(()) * q
        #r = np.eye(n) @ q
        #r = (self._RHO_STORE[0] * self._S_STORE[0].dot(self._S_STORE[0].T)) @ q
        for i in range(0, bound):
            beta = self._RHO_STORE[i] * self._Y_STORE[i].T.dot(r)
            r += self._S_STORE[i].flatten() * (alphas[i] - beta)
        
        return r

    def _precond_mv2(self, u: NDArray) -> NDArray:
        q: NDArray = u.copy()
        n = u.size
        bound: int = len(self._RHO_STORE)
        alphas = np.zeros(bound)
        for i in range(bound):
            alphas[i] = self._RHO_STORE[i] * self._S_STORE[i].T.dot(q)
            q -= alphas[i] * self._Y_STORE[i].flatten()

        r = (self._S_STORE[bound-1].T.dot(self._Y_STORE[bound-1])/self._Y_STORE[bound-1].T.dot(self._Y_STORE[bound-1])) * np.eye(n) @ q
        for i in reversed(range(0, bound-1)):
            beta = self._RHO_STORE[i] * self._Y_STORE[i].T.dot(r)
            r += self._S_STORE[i].flatten() * (alphas[i] - beta)
        
        return r


        

