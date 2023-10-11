import parapint
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
from pyomo.contrib.pynumero.sparse.mpi_block_matrix import MPIBlockMatrix
from pyomo.contrib.pynumero.sparse.mpi_block_vector import MPIBlockVector
from parapint.linalg import ScipyInterface
import scipy.sparse as sp
from numpy.random import default_rng
from scipy.sparse import coo_matrix
import numpy as np
from mpi4py import MPI
from pyomo.common.timing import HierarchicalTimer
import time


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

rng = default_rng(11)
n_blocks = 4
block_size = 3
sc_size = 5
rank_by_index = list()
for ndx in range(n_blocks - 1):
    for _rank in range(size):
        if (ndx - _rank) % size == 0:
            rank_by_index.append(_rank)
rank_by_index.append(-1)

A = MPIBlockMatrix(nbrows=n_blocks,
                   nbcols=n_blocks,
                   rank_ownership=[rank_by_index for _ in range(n_blocks)],
                    mpi_comm=comm)

local_A = BlockMatrix(n_blocks, n_blocks)

for k in range(n_blocks - 1):
    d_block = coo_matrix(sp.random(m=block_size,n=block_size, density=0.4, random_state=rng) + (block_size/2) * np.eye(block_size), dtype=np.double)
    local_A.set_block(k, k, d_block)
    if rank_by_index[k] == rank:
        A.set_block(k, k, d_block)

for k in range(n_blocks - 1):
    b_block = coo_matrix(sp.random(m=block_size,n=sc_size, density=0.2, random_state=rng), dtype=np.double)
    local_A.set_block(n_blocks - 1, k, b_block.transpose(copy=True))
    local_A.set_block(k, n_blocks - 1, b_block)
    if rank_by_index[k] == rank:
        A.set_block(n_blocks - 1, k, b_block.transpose(copy=True))
        A.set_block(k, n_blocks - 1, b_block)

s_block = coo_matrix(2 * np.eye(sc_size), dtype=np.double)
A.set_block(n_blocks - 1, n_blocks - 1, s_block)
A.broadcast_block_sizes()
local_A.set_block(n_blocks - 1, n_blocks - 1, s_block)

rhs = MPIBlockVector(nblocks=n_blocks, rank_owner=rank_by_index, mpi_comm=comm)
local_rhs = BlockVector(n_blocks)
for k in range(n_blocks-1):
    r_block = rng.random(size=(block_size))
    local_rhs.set_block(k, r_block)
    if rank_by_index[k] == rank:
        rhs.set_block(k, r_block)

rs_block = rng.random(size=(sc_size))
rhs.set_block(n_blocks - 1, rs_block)
rhs.broadcast_block_sizes()
local_rhs.set_block(n_blocks - 1, rs_block)

x1 = np.linalg.solve(local_A.toarray(), local_rhs.flatten())

sc = local_A.get_block(n_blocks - 1, n_blocks - 1).copy()
for k in range(n_blocks - 1):
    A_k = local_A.get_block(k, n_blocks - 1).toarray()
    W_k = local_A.get_block(k, k).toarray()
    sc -= A_k.T @ np.linalg.inv(W_k) @ (A_k)

eig_sc = np.linalg.eigvals(sc)
pos_sc = np.count_nonzero(eig_sc > 0)
neg_sc = np.count_nonzero(eig_sc < 0)
zero_sc = np.count_nonzero(eig_sc == 0)
inertia_sc = (pos_sc, neg_sc, zero_sc)
print(f'sc inertia: {inertia_sc}')

implicit_timer = HierarchicalTimer()
implicit_sc_solver = parapint.linalg.MPIImplicitSchurComplementLinearSolver(subproblem_solvers={ndx: ScipyInterface(compute_inertia=True) for ndx in range(n_blocks - 1)},
                            schur_complement_solver=ScipyInterface(compute_inertia=True))
implicit_timer.start('symbolic_factorization')
implicit_sc_solver.do_symbolic_factorization(A, timer=implicit_timer)
implicit_timer.stop('symbolic_factorization')
implicit_timer.start('numeric_factorization')
implicit_sc_solver.do_numeric_factorization(A, timer=implicit_timer)
implicit_timer.stop('numeric_factorization')
implicit_timer.start('solve')
x2 = implicit_sc_solver.do_back_solve(rhs, timer=implicit_timer, ip_iter=0)
x2 = implicit_sc_solver.do_back_solve(rhs, timer=implicit_timer, ip_iter=1)
implicit_timer.stop('solve')
implicit_inertia = implicit_sc_solver.get_inertia()
# explicit_timer = HierarchicalTimer()
# explicit_sc_solver = parapint.linalg.MPISchurComplementLinearSolver(subproblem_solvers={ndx: ScipyInterface(compute_inertia=True) for ndx in range(n_blocks - 1)},
#                             schur_complement_solver=ScipyInterface(compute_inertia=True))
# explicit_sc_solver.do_symbolic_factorization(A, timer=explicit_timer)
# explicit_sc_solver.do_numeric_factorization(A, timer=explicit_timer)
# x3 = explicit_sc_solver.do_back_solve(rhs, timer=explicit_timer)
# explicit_inertia = explicit_sc_solver.get_inertia()

#if rank == 0:
#print(f'actual x: {x1}')
#print(f'implicit x: {x2.make_local_copy().flatten()}')
#print(f'explicit x: {x3.make_local_copy().flatten()}')
print(np.allclose(x1, x2.make_local_copy().flatten(), atol=1e-3))



#eig = np.linalg.eigvals(local_A.toarray())
#pos = np.count_nonzero(eig > 0)
#neg = np.count_nonzero(eig < 0)
#zero = np.count_nonzero(eig == 0)
#actual_inertia = (pos, neg, zero)
#print(f'true inertia: {actual_inertia}')
print(f'implicit inertia: {implicit_inertia}')
# print(f'explicit inertia: {explicit_inertia}')

#print(f'preconditioner: \n{sc_solver._preconditioner}')

#print(f'M * rhs: \n{sc_solver._preconditioner * np.array([1, 1], dtype=np.double)}')
#sc_solver.do_numeric_factorization(A)
#x2 = sc_solver.do_back_solve(rhs)
#print(np.allclose(x1, x2.make_local_copy().flatten()))
time.sleep(rank/10)
print(implicit_timer)
# time.sleep(3)
# print(explicit_timer)
