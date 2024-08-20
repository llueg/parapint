from parapint.interfaces.interface import InteriorPointInterface
from abc import ABCMeta
from scipy.sparse import coo_matrix
from pyomo.contrib.pynumero.sparse.mpi_block_vector import MPIBlockVector
from pyomo.contrib.pynumero.sparse.mpi_block_matrix import MPIBlockMatrix
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
import numpy as np
from typing import Dict, Optional, Sequence
from pyomo.common.timing import HierarchicalTimer
from mpi4py import MPI
from .sc_ip_interface import DistStochasticSchurComplementInteriorPointInterface


def _distribute_blocks(num_blocks: int, rank: int, size: int) -> Sequence[int]:
    local_blocks = list()
    for ndx in range(num_blocks):
        if ndx % size == rank:
            local_blocks.append(ndx)
    return local_blocks


def _get_ownership_map(num_blocks: int, size: int) -> Dict[int, int]:
    ownership_map = dict()
    for ndx in range(num_blocks):
        for rank in range(size):
            if ndx % size == rank:
                ownership_map[ndx] = rank
                break
    return ownership_map

class MPIDistStochasticSchurComplementInteriorPointInterface(DistStochasticSchurComplementInteriorPointInterface, metaclass=ABCMeta):
    """
    A class for interfacing with Parapint's interior point algorithm for the parallel solution of
    stochastic optimization problems using Schur-Complement decomposition. Users should inherit from
    this class and, at a minimum, implement the `build_model_for_scenario` method (see
    StochasticSchurComplementInteriorPointInterface.build_model_for_scenario for details).

    Parameters
    ----------
    scenarios: Sequence
        The scenarios for which subproblems need built
    nonanticipative_var_identifiers: Sequence
        Unique identifiers for the first stage variables. Every process should get the
        exact same list in the exact same order.
    """
    def __init__(self,
                 scenarios: Sequence,
                 nonanticipative_var_identifiers: Sequence,
                 comm: MPI.Comm,
                 ownership_map: Optional[Dict] = None):
        """
        This method sets up the coupling matrices and the structure for the kkt system

        Parameters
        ----------
        scenarios: Sequence
            The scenarios for which subproblems need built
        nonanticipative_var_identifiers: Sequence
            Unique identifiers for the first stage variables. Every process should get the
            exact same list in the exact same order.
        comm: MPI.Comm
            The MPI communicator
        ownership_map: Dict
            A dictionary mapping scenario index (i.e., index into scenarios) to rank
        """
        self._num_scenarios: int = len(scenarios)
        self._num_first_stage_vars: int = len(nonanticipative_var_identifiers)
        self._first_stage_var_indices = {identifier: ndx for ndx, identifier in enumerate(nonanticipative_var_identifiers)}
        self._num_first_stage_vars_by_scenario: Dict[int, int] = dict()
        self._nlps: Dict[int, InteriorPointInterface] = dict()  # keys are the scenario indices
        self._scenario_ndx_to_id = dict()
        self._scenario_id_to_ndx = dict()
        self._linking_matrices: Dict[int, coo_matrix] = dict()  # these get multiplied by the primal vars of the corresponding scenario
        self._link_coupling_matrices: Dict[int, coo_matrix] = dict()  # these get multiplied by the coupling variables

        self._comm: MPI.Comm = comm
        self._rank: int = comm.Get_rank()
        self._size: int = comm.Get_size()

        if self._size > self._num_scenarios:
            raise ValueError('Cannot yet handle more processes than scenarios')

        if ownership_map is None:
            self._local_block_indices: Sequence[int] = _distribute_blocks(num_blocks=self._num_scenarios,
                                                                          rank=self._rank,
                                                                          size=self._size)
            self._ownership_map: Dict[int, int] = _get_ownership_map(num_blocks=self._num_scenarios,
                                                                     size=self._size)
        else:
            self._ownership_map = dict(ownership_map)
            self._local_block_indices = list()
            for scenario_ndx, scenario in enumerate(scenarios):
                if self._ownership_map[scenario_ndx] == self._rank:
                    self._local_block_indices.append(scenario_ndx)

        self._primals_lb: MPIBlockVector = self._build_mpi_block_vector(extra_block=True)
        self._primals_ub: MPIBlockVector = self._build_mpi_block_vector(extra_block=True)

        self._ineq_lb: MPIBlockVector = self._build_mpi_block_vector()
        self._ineq_ub: MPIBlockVector = self._build_mpi_block_vector()

        self._init_primals: MPIBlockVector = self._build_mpi_block_vector(extra_block=True)
        self._primals: MPIBlockVector = self._build_mpi_block_vector(extra_block=True)
        self._delta_primals: MPIBlockVector = self._build_mpi_block_vector(extra_block=True)

        self._init_slacks: MPIBlockVector = self._build_mpi_block_vector()
        self._slacks: MPIBlockVector = self._build_mpi_block_vector()
        self._delta_slacks: MPIBlockVector = self._build_mpi_block_vector()

        self._init_duals_eq: MPIBlockVector = self._build_mpi_block_vector()
        self._duals_eq: MPIBlockVector = self._build_mpi_block_vector()
        self._delta_duals_eq: MPIBlockVector = self._build_mpi_block_vector()

        self._init_duals_ineq: MPIBlockVector = self._build_mpi_block_vector()
        self._duals_ineq: MPIBlockVector = self._build_mpi_block_vector()
        self._delta_duals_ineq: MPIBlockVector = self._build_mpi_block_vector()

        self._init_duals_primals_lb: MPIBlockVector = self._build_mpi_block_vector(extra_block=True)
        self._duals_primals_lb: MPIBlockVector = self._build_mpi_block_vector(extra_block=True)
        self._delta_duals_primals_lb: MPIBlockVector = self._build_mpi_block_vector(extra_block=True)

        self._init_duals_primals_ub: MPIBlockVector = self._build_mpi_block_vector(extra_block=True)
        self._duals_primals_ub: MPIBlockVector = self._build_mpi_block_vector(extra_block=True)
        self._delta_duals_primals_ub: MPIBlockVector = self._build_mpi_block_vector(extra_block=True)

        self._init_duals_slacks_lb: MPIBlockVector = self._build_mpi_block_vector()
        self._duals_slacks_lb: MPIBlockVector = self._build_mpi_block_vector()
        self._delta_duals_slacks_lb: MPIBlockVector = self._build_mpi_block_vector()

        self._init_duals_slacks_ub: MPIBlockVector = self._build_mpi_block_vector()
        self._duals_slacks_ub: MPIBlockVector = self._build_mpi_block_vector()
        self._delta_duals_slacks_ub: MPIBlockVector = self._build_mpi_block_vector()

        self._eq_resid: MPIBlockVector = self._build_mpi_block_vector()
        self._ineq_resid: MPIBlockVector = self._build_mpi_block_vector()
        self._grad_objective: MPIBlockVector = self._build_mpi_block_vector(extra_block=True)
        self._jac_eq: MPIBlockMatrix = self._build_mpi_block_matrix(extra_row=False, extra_col=True)
        self._jac_ineq: MPIBlockMatrix = self._build_mpi_block_matrix(extra_row=False, extra_col=True)
        self._kkt: MPIBlockMatrix = self._build_mpi_block_matrix(extra_row=True, extra_col=True)
        self._rhs: MPIBlockVector = self._build_mpi_block_vector(extra_block=True)

        self._setup(scenarios=scenarios)
        self._setup_block_vectors()
        self._setup_jacs()
        self._setup_kkt_and_rhs_structure()

        self._bounds_relaxation_factor = 0
        self.set_bounds_relaxation_factor(self._bounds_relaxation_factor)

    def _build_mpi_block_matrix(self, extra_row: bool = False, extra_col: bool = False) -> MPIBlockMatrix:
        if extra_row:
            nrows = self._num_scenarios + 1
        else:
            nrows = self._num_scenarios
        if extra_col:
            ncols = self._num_scenarios + 1
        else:
            ncols = self._num_scenarios
        rank_ownership = -1 * np.ones((nrows, ncols), dtype=np.int64)
        for ndx in range(self._num_scenarios):
            rank_ownership[ndx, ndx] = self._ownership_map[ndx]
            if extra_row:
                rank_ownership[self._num_scenarios, ndx] = self._ownership_map[ndx]
            if extra_col:
                rank_ownership[ndx, self._num_scenarios] = self._ownership_map[ndx]
        mat = MPIBlockMatrix(nbrows=nrows,
                             nbcols=ncols,
                             rank_ownership=rank_ownership,
                             mpi_comm=self._comm,
                             assert_correct_owners=False)
        return mat

    def _build_mpi_block_vector(self, extra_block: bool = False) -> MPIBlockVector:
        if extra_block:
            n = self._num_scenarios + 1
        else:
            n = self._num_scenarios
        rank_ownership = -1 * np.ones(n, dtype=np.int64)
        for ndx in range(self._num_scenarios):
            rank_ownership[ndx] = self._ownership_map[ndx]
        vec = MPIBlockVector(nblocks=n,
                             rank_owner=rank_ownership,
                             mpi_comm=self._comm,
                             assert_correct_owners=False)
        return vec

    def _setup(self, scenarios: Sequence):
        """
        This method sets up the coupling matrices and the structure for the kkt system
        """
        local_block_indices_set = set(self._local_block_indices)
        for scenario_ndx, scenario_id in enumerate(scenarios):
            if scenario_ndx in local_block_indices_set:
                (pyomo_model,
                 first_stage_vars) = self.build_model_for_scenario(scenario_identifier=scenario_id)
                self._scenario_ndx_to_id[scenario_ndx] = scenario_id
                self._scenario_id_to_ndx[scenario_id] = scenario_ndx
                self._nlps[scenario_ndx] = nlp = InteriorPointInterface(pyomo_model=pyomo_model)
                self._num_first_stage_vars_by_scenario[scenario_ndx] = len(first_stage_vars)
                self._linking_matrices[scenario_ndx] = self._build_linking_matrix(nlp, first_stage_vars)
                self._link_coupling_matrices[scenario_ndx] = self._build_link_coupling_matrix(first_stage_vars)

    def n_primals(self) -> int:
        res = sum(nlp.n_primals() for nlp in self._nlps.values())
        res = self._comm.allreduce(res)
        res += self._total_num_coupling_vars
        return res

    def get_obj_factor(self) -> float:
        return self._nlps[self._local_block_indices[0]].get_obj_factor()

    def evaluate_objective(self) -> float:
        res = super(MPIDistStochasticSchurComplementInteriorPointInterface, self).evaluate_objective()
        res = self._comm.allreduce(res)
        return res

    def n_eq_constraints(self) -> int:
        res = sum(nlp.n_eq_constraints() for nlp in self._nlps.values())
        res += sum(self._num_first_stage_vars_by_scenario.values())
        res = self._comm.allreduce(res)
        return res

    def n_ineq_constraints(self) -> int:
        res = super(MPIDistStochasticSchurComplementInteriorPointInterface, self).n_ineq_constraints()
        res = self._comm.allreduce(res)
        return res

    def evaluate_primal_dual_kkt_rhs(self, timer: HierarchicalTimer = None) -> MPIBlockVector:
        last_block = self._rhs.get_block(self._num_scenarios)
        last_block.fill(0)
        super(MPIDistStochasticSchurComplementInteriorPointInterface, self).evaluate_primal_dual_kkt_rhs()
        last_block = self._rhs.get_block(self._num_scenarios)
        res = np.zeros(last_block.size, dtype=np.double)
        self._comm.Allreduce(last_block, res)
        np.copyto(self._rhs.get_block(self._num_scenarios), res)
        return self._rhs

    @property
    def ownership_map(self) -> Dict[int, int]:
        """
        Returns
        -------
        ownership_map: dict
            This is a map from the time block index to the rank that owns that time block.
        """
        return dict(self._ownership_map)

    @property
    def local_block_indices(self) -> Sequence[int]:
        """
        Returns
        -------
        local_block_indices: list
            The indices of the time blocks owned by the current process.
        """
        return list(self._local_block_indices)
