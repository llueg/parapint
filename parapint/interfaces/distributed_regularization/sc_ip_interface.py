from parapint.interfaces.interface import BaseInteriorPointInterface, InteriorPointInterface
from abc import ABCMeta, abstractmethod
from scipy.sparse import coo_matrix, identity
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
import numpy as np
from typing import Dict, Optional, Union, Tuple, Sequence, Iterable, Any
from pyomo.core.base.block import _BlockData
from pyomo.core.base.var import _GeneralVarData
from pyomo.core.base.constraint import _GeneralConstraintData
from pyomo.common.timing import HierarchicalTimer


class DistStochasticSchurComplementInteriorPointInterface(BaseInteriorPointInterface, metaclass=ABCMeta):
    """
    A class for interfacing with Parapint's interior point algorithm for the serial solution of
    2-stage stochastic optimization problems. This class is primarily for testing purposes. Users should
    favor the MPIStochasticSchurComplementInteriorPointInterface class because it supports
    parallel solution. To utilize this class, create a class which inherits from this class
    and implement the build_model_for_scenario method. If you override the __init__ method
    make sure to call the super class' __init__ method at the end of the derived class'
    __init__ method. See farmer.py in the examples directory for an example.

    Parameters
    ----------
    scenarios: Sequence
        The scenarios for which subproblems need built
    nonanticipative_var_identifiers: Sequence
        Unique identifiers for the first stage variables. Every process should get the
        exact same list in the exact same order.
    """
    def __init__(self, scenarios: Sequence, nonanticipative_var_identifiers: Sequence):
        """
        This method sets up the coupling matrices and the structure for the kkt system

        Parameters
        ----------
        scenarios: Sequence
            The scenarios identifiers for which subproblems need built
        nonanticipative_var_identifiers: Sequence
            Unique identifiers for the first stage variables. Every rank should get the
            exact same list in the exact same order.
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

        self._primals_lb: BlockVector = BlockVector(self._num_scenarios + 1)
        self._primals_ub: BlockVector = BlockVector(self._num_scenarios + 1)

        self._ineq_lb: BlockVector = BlockVector(self._num_scenarios)
        self._ineq_ub: BlockVector = BlockVector(self._num_scenarios)

        self._init_primals: BlockVector = BlockVector(self._num_scenarios + 1)
        self._primals: BlockVector = BlockVector(self._num_scenarios + 1)
        self._delta_primals: BlockVector = BlockVector(self._num_scenarios + 1)

        self._init_slacks: BlockVector = BlockVector(self._num_scenarios)
        self._slacks: BlockVector = BlockVector(self._num_scenarios)
        self._delta_slacks: BlockVector = BlockVector(self._num_scenarios)

        self._init_duals_eq: BlockVector = BlockVector(self._num_scenarios)
        self._duals_eq: BlockVector = BlockVector(self._num_scenarios)
        self._delta_duals_eq: BlockVector = BlockVector(self._num_scenarios)

        self._init_duals_ineq: BlockVector = BlockVector(self._num_scenarios)
        self._duals_ineq: BlockVector = BlockVector(self._num_scenarios)
        self._delta_duals_ineq: BlockVector = BlockVector(self._num_scenarios)

        self._init_duals_primals_lb: BlockVector = BlockVector(self._num_scenarios + 1)
        self._duals_primals_lb: BlockVector = BlockVector(self._num_scenarios + 1)
        self._delta_duals_primals_lb: BlockVector = BlockVector(self._num_scenarios + 1)

        self._init_duals_primals_ub: BlockVector = BlockVector(self._num_scenarios + 1)
        self._duals_primals_ub: BlockVector = BlockVector(self._num_scenarios + 1)
        self._delta_duals_primals_ub: BlockVector = BlockVector(self._num_scenarios + 1)

        self._init_duals_slacks_lb: BlockVector = BlockVector(self._num_scenarios)
        self._duals_slacks_lb: BlockVector = BlockVector(self._num_scenarios)
        self._delta_duals_slacks_lb: BlockVector = BlockVector(self._num_scenarios)

        self._init_duals_slacks_ub: BlockVector = BlockVector(self._num_scenarios)
        self._duals_slacks_ub: BlockVector = BlockVector(self._num_scenarios)
        self._delta_duals_slacks_ub: BlockVector = BlockVector(self._num_scenarios)

        self._eq_resid: BlockVector = BlockVector(self._num_scenarios)
        self._ineq_resid: BlockVector = BlockVector(self._num_scenarios)
        self._grad_objective: BlockVector = BlockVector(self._num_scenarios + 1)
        self._jac_eq: BlockMatrix = BlockMatrix(nbrows=self._num_scenarios, nbcols=self._num_scenarios + 1)
        self._jac_ineq: BlockMatrix = BlockMatrix(nbrows=self._num_scenarios, nbcols=self._num_scenarios + 1)
        self._kkt: BlockMatrix = BlockMatrix(nbrows=self._num_scenarios + 1, nbcols=self._num_scenarios + 1)
        self._rhs: BlockVector = BlockVector(nblocks=self._num_scenarios + 1)

        self._setup(scenarios=scenarios)
        self._setup_block_vectors()
        self._setup_jacs()
        self._setup_kkt_and_rhs_structure()

        self._bounds_relaxation_factor = 0
        self.set_bounds_relaxation_factor(self._bounds_relaxation_factor)

    @abstractmethod
    def build_model_for_scenario(self,
                                 scenario_identifier: Any) -> Tuple[_BlockData, Dict[Any, _GeneralVarData]]:
        """
        This method should be implemented by derived classes.
        This method should return the model for the scenario scenario_id and a dict mapping
        nonanticipative variable identifiers to pyomo variables appearing in the scenario.
        This method will be called once for each scenario.

        Parameters
        ----------
        scenario_identifier: Any
            The scenario

        Returns
        -------
        pyomo_model: pyomo.core.base.block.Block
            The model for the time interval [start_t, end_t].
        nonanticipative_vars: Dict[Any, _GeneralVarData]
            The first stage variables. Keys should be the identifiers passed to the constructor
        """
        pass

    def _setup(self, scenarios: Sequence):
        """
        This method sets up the coupling matrices and the structure for the kkt system
        """
        for scenario_ndx, scenario_id in enumerate(scenarios):
            (pyomo_model,
             first_stage_vars) = self.build_model_for_scenario(scenario_identifier=scenario_id)
            self._scenario_ndx_to_id[scenario_ndx] = scenario_id
            self._scenario_id_to_ndx[scenario_id] = scenario_ndx
            self._nlps[scenario_ndx] = nlp = InteriorPointInterface(pyomo_model=pyomo_model)
            self._num_first_stage_vars_by_scenario[scenario_ndx] = len(first_stage_vars)
            self._linking_matrices[scenario_ndx] = self._build_linking_matrix(nlp, first_stage_vars)
            self._link_coupling_matrices[scenario_ndx] = self._build_link_coupling_matrix(first_stage_vars)

    def _setup_block_vectors(self):
        for ndx, nlp in self._nlps.items():
            self._primals_lb.set_block(ndx, nlp.primals_lb())
            self._primals_ub.set_block(ndx, nlp.primals_ub())

            self._ineq_lb.set_block(ndx, nlp.ineq_lb())
            self._ineq_ub.set_block(ndx, nlp.ineq_ub())

            self._init_primals.set_block(ndx, nlp.init_primals())
            self._primals.set_block(ndx, nlp.init_primals().copy())
            self._delta_primals.set_block(ndx, np.zeros(nlp.n_primals()))

            self._init_slacks.set_block(ndx, nlp.init_slacks())
            self._slacks.set_block(ndx, nlp.init_slacks().copy())
            self._delta_slacks.set_block(ndx, np.zeros(nlp.n_ineq_constraints()))

            self._init_duals_ineq.set_block(ndx, nlp.init_duals_ineq())
            self._duals_ineq.set_block(ndx, nlp.init_duals_ineq().copy())
            self._delta_duals_ineq.set_block(ndx, np.zeros(nlp.n_ineq_constraints()))

            self._init_duals_primals_lb.set_block(ndx, nlp.init_duals_primals_lb())
            self._duals_primals_lb.set_block(ndx, nlp.init_duals_primals_lb().copy())
            self._delta_duals_primals_lb.set_block(ndx, np.zeros(nlp.n_primals()))

            self._init_duals_primals_ub.set_block(ndx, nlp.init_duals_primals_ub())
            self._duals_primals_ub.set_block(ndx, nlp.init_duals_primals_ub().copy())
            self._delta_duals_primals_ub.set_block(ndx, np.zeros(nlp.n_primals()))

            self._init_duals_slacks_lb.set_block(ndx, nlp.init_duals_slacks_lb())
            self._duals_slacks_lb.set_block(ndx, nlp.init_duals_slacks_lb().copy())
            self._delta_duals_slacks_lb.set_block(ndx, np.zeros(nlp.n_ineq_constraints()))

            self._init_duals_slacks_ub.set_block(ndx, nlp.init_duals_slacks_ub())
            self._duals_slacks_ub.set_block(ndx, nlp.init_duals_slacks_ub().copy())
            self._delta_duals_slacks_ub.set_block(ndx, np.zeros(nlp.n_ineq_constraints()))

            self._ineq_resid.set_block(ndx, np.zeros(nlp.n_ineq_constraints()))
            self._grad_objective.set_block(ndx, np.ones(nlp.n_primals()))

        # duals eq, eq resid
        for ndx, nlp in self._nlps.items():
            sub_block = BlockVector(2)
            sub_block.set_block(0, nlp.init_duals_eq())
            sub_block.set_block(1, np.zeros(self._num_first_stage_vars_by_scenario[ndx]))
            self._init_duals_eq.set_block(ndx, sub_block)
            self._duals_eq.set_block(ndx, sub_block.copy())
            self._delta_duals_eq.set_block(ndx, sub_block.copy_structure())
            self._eq_resid.set_block(ndx, sub_block.copy_structure() * 1)

        self._primals_lb.set_block(self._num_scenarios, np.zeros(self._total_num_coupling_vars))
        self._primals_ub.set_block(self._num_scenarios, np.zeros(self._total_num_coupling_vars))
        self._primals_lb.get_block(self._num_scenarios).fill(-np.inf)
        self._primals_ub.get_block(self._num_scenarios).fill(np.inf)

        self._init_primals.set_block(self._num_scenarios, np.zeros(self._total_num_coupling_vars))
        self._primals.set_block(self._num_scenarios, np.zeros(self._total_num_coupling_vars))
        self._delta_primals.set_block(self._num_scenarios, np.zeros(self._total_num_coupling_vars))

        self._init_duals_primals_lb.set_block(self._num_scenarios, np.zeros(self._total_num_coupling_vars))
        self._duals_primals_lb.set_block(self._num_scenarios, np.zeros(self._total_num_coupling_vars))
        self._delta_duals_primals_lb.set_block(self._num_scenarios, np.zeros(self._total_num_coupling_vars))

        self._init_duals_primals_ub.set_block(self._num_scenarios, np.zeros(self._total_num_coupling_vars))
        self._duals_primals_ub.set_block(self._num_scenarios, np.zeros(self._total_num_coupling_vars))
        self._delta_duals_primals_ub.set_block(self._num_scenarios, np.zeros(self._total_num_coupling_vars))

        self._grad_objective.set_block(self._num_scenarios, np.zeros(self._total_num_coupling_vars))

    def _setup_jacs(self):
        self._jac_ineq.set_col_size(self._num_scenarios, self._total_num_coupling_vars)
        for ndx, nlp in self._nlps.items():
            self._jac_ineq.set_row_size(ndx, nlp.n_ineq_constraints())
            self._jac_ineq.set_col_size(ndx, nlp.n_primals())

            sub_block = BlockMatrix(nbrows=2, nbcols=1)
            sub_block.set_row_size(0, nlp.n_eq_constraints())
            sub_block.set_col_size(0, nlp.n_primals())
            sub_block.set_block(1, 0, self._linking_matrices[ndx])
            self._jac_eq.set_block(ndx, ndx, sub_block)

            sub_block = BlockMatrix(nbrows=2, nbcols=1)
            sub_block.set_row_size(0, nlp.n_eq_constraints())
            sub_block.set_col_size(0, self._total_num_coupling_vars)
            sub_block.set_block(1, 0, -self._link_coupling_matrices[ndx])
            self._jac_eq.set_block(ndx, self._num_scenarios, sub_block)

    def _setup_kkt_and_rhs_structure(self):
        # First setup the diagonal blocks
        for ndx, nlp in self._nlps.items():
            sub_kkt = BlockMatrix(nbrows=2, nbcols=2)
            n = nlp.n_primals() + nlp.n_eq_constraints() + 2 * nlp.n_ineq_constraints()
            sub_kkt.set_row_size(0, n)
            sub_kkt.set_col_size(0, n)
            sub_kkt.set_row_size(1, self._num_first_stage_vars_by_scenario[ndx])
            sub_kkt.set_col_size(1, self._num_first_stage_vars_by_scenario[ndx])
            row_1 = BlockMatrix(nbrows=1, nbcols=4)
            row_1.set_row_size(0, self._num_first_stage_vars_by_scenario[ndx])
            row_1.set_col_size(0, nlp.n_primals())
            row_1.set_col_size(1, nlp.n_ineq_constraints())
            row_1.set_col_size(2, nlp.n_eq_constraints())
            row_1.set_col_size(3, nlp.n_ineq_constraints())
            row_1.set_block(0, 0, self._linking_matrices[ndx])
            sub_kkt.set_block(1, 0, row_1)
            sub_kkt.set_block(0, 1, row_1.transpose())
            ptb = identity(self._num_first_stage_vars_by_scenario[ndx], format='coo')
            ptb.data.fill(0)
            sub_kkt.set_block(1, 1, ptb)
            self._kkt.set_block(ndx, ndx, sub_kkt)
            sub_rhs = BlockVector(2)
            sub_rhs.set_block(0, np.zeros(n))
            sub_rhs.set_block(1, np.zeros(self._num_first_stage_vars_by_scenario[ndx]))
            self._rhs.set_block(ndx, sub_rhs)

        # Setup the border blocks
        for ndx, nlp in self._nlps.items():
            nlp = self._nlps[ndx]
            block = BlockMatrix(nbrows=1, nbcols=2)
            n = nlp.n_primals() + nlp.n_eq_constraints() + 2 * nlp.n_ineq_constraints()
            block.set_col_size(0, n)
            block.set_block(0, 1, -self._link_coupling_matrices[ndx].transpose())
            self._kkt.set_block(self._num_scenarios, ndx, block)
            self._kkt.set_block(ndx, self._num_scenarios, block.transpose())

        ptb = identity(self._total_num_coupling_vars, format='coo')
        ptb.data.fill(0)
        self._kkt.set_block(self._num_scenarios, self._num_scenarios, ptb)
        self._rhs.set_block(self._num_scenarios, np.zeros(self._total_num_coupling_vars))

    def _build_linking_matrix(self, nlp: InteriorPointInterface, first_stage_vars: Dict[Any, _GeneralVarData]):
        rows = list()
        cols = list()
        data = list()
        _row = 0
        for var_identifier, var in first_stage_vars.items():
            local_var_ndx = nlp.get_primal_indices([var])[0]
            rows.append(_row)
            cols.append(local_var_ndx)
            data.append(1)
            _row += 1
        linking_matrix = coo_matrix((data, (rows, cols)),
                                    shape=(len(first_stage_vars), nlp.n_primals()),
                                    dtype=np.double)
        return linking_matrix

    def _build_link_coupling_matrix(self, first_stage_vars: Dict[Any, _GeneralVarData]):
        rows = list()
        cols = list()
        data = list()
        _row = 0
        for var_identifier, var in first_stage_vars.items():
            rows.append(_row)
            cols.append(self._first_stage_var_indices[var_identifier])
            data.append(1)
            _row += 1
        link_coupling_matrix = coo_matrix((data, (rows, cols)),
                                          shape=(len(first_stage_vars),
                                                 self._total_num_coupling_vars),
                                          dtype=np.double)
        return link_coupling_matrix

    @property
    def _total_num_coupling_vars(self):
        return self._num_first_stage_vars

    def n_primals(self) -> int:
        """
        Returns
        -------
        n_primals: int
            The number of primal variables
        """
        return sum(nlp.n_primals() for nlp in self._nlps.values()) + self._total_num_coupling_vars

    def nnz_hessian_lag(self) -> int:
        raise NotImplementedError('This is not done yet')

    def get_bounds_relaxation_factor(self) -> float:
        return self._bounds_relaxation_factor

    def set_bounds_relaxation_factor(self, val: float):
        self._bounds_relaxation_factor = val
        for nlp in self._nlps.values():
            nlp.set_bounds_relaxation_factor(val)

    def primals_lb(self) -> BlockVector:
        """
        Returns
        -------
        primals_lb: BlockVector
            The lower bounds for each primal variable. This BlockVector has one block for every time block
            and one block for the coupling variables.
        """
        for ndx, nlp in self._nlps.items():
            self._primals_lb.set_block(ndx, nlp.primals_lb())
        return self._primals_lb

    def primals_ub(self) -> BlockVector:
        """
        Returns
        -------
        primals_ub: BlockVector
            The upper bounds for each primal variable. This BlockVector has one block for every time block
            and one block for the coupling variables.
        """
        for ndx, nlp in self._nlps.items():
            self._primals_ub.set_block(ndx, nlp.primals_ub())
        return self._primals_ub

    def init_primals(self) -> BlockVector:
        """
        Returns
        -------
        init_primals: BlockVector
            The initial values for each primal variable. This BlockVector has one block for every time block
            and one block for the coupling variables.
        """
        return self._init_primals

    def set_primals(self, primals: BlockVector):
        """
        Set the values of the primal variables for evaluation (i.e., the evaluate_* methods).

        Parameters
        ----------
        primals: BlockVector
            The values for each primal variable. This BlockVector should have one block for every time block
            and one block for the coupling variables.
        """
        for ndx, nlp in self._nlps.items():
            nlp.set_primals(primals.get_block(ndx))
            self._primals.set_block(ndx, primals.get_block(ndx))
        self._primals.set_block(self._num_scenarios, primals.get_block(self._num_scenarios))

    def get_primals(self) -> BlockVector:
        """
        Returns
        -------
        primals: BlockVector
            The values for each primal variable. This BlockVector has one block for every time block
            and one block for the coupling variables.
        """
        return self._primals

    def get_obj_factor(self) -> float:
        return self._nlps[0].get_obj_factor()

    def set_obj_factor(self, obj_factor: float):
        for ndx, nlp in self._nlps.items():
            nlp.set_obj_factor(obj_factor)

    def evaluate_objective(self) -> float:
        """
        Returns
        -------
        objective_val: float
            The value of the objective
        """
        return sum(nlp.evaluate_objective() for nlp in self._nlps.values())

    def evaluate_grad_objective(self) -> BlockVector:
        """
        Returns
        -------
        grad_obj: BlockVector
            The gradient of the objective. This BlockVector has one block for every time block
            and one block for the coupling variables.
        """
        for ndx, nlp in self._nlps.items():
            self._grad_objective.set_block(ndx, nlp.evaluate_grad_objective())
        return self._grad_objective

    def n_eq_constraints(self) -> int:
        """
        Returns
        -------
        n_eq_constraints: int
            The number of equality constraints, including the coupling constraints
        """
        return sum(nlp.n_eq_constraints() for nlp in self._nlps.values()) + sum(self._num_first_stage_vars_by_scenario.values())

    def n_ineq_constraints(self) -> int:
        """
        Returns
        -------
        n_ineq_constraints: int
            The number of inequality constraints
        """
        return sum(nlp.n_ineq_constraints() for nlp in self._nlps.values())

    def nnz_jacobian_eq(self) -> int:
        raise NotImplementedError('Not done yet')

    def nnz_jacobian_ineq(self) -> int:
        raise NotImplementedError('Not done yet')

    def ineq_lb(self) -> BlockVector:
        """
        Returns
        -------
        ineq_lb: BlockVector
            The lower bounds for each inequality constraint. This BlockVector has one block for every time block.
        """
        for ndx, nlp in self._nlps.items():
            self._ineq_lb.set_block(ndx, nlp.ineq_lb())
        return self._ineq_lb

    def ineq_ub(self) -> BlockVector:
        """
        Returns
        -------
        ineq_lb: BlockVector
            The lower bounds for each inequality constraint. This BlockVector has one block for every time block.
        """
        for ndx, nlp in self._nlps.items():
            self._ineq_ub.set_block(ndx, nlp.ineq_ub())
        return self._ineq_ub

    def init_duals_eq(self) -> BlockVector:
        """
        Returns
        -------
        init_duals_eq: BlockVector
            The initial values for the duals of the equality constraints, including the coupling constraints.
            This BlockVector has one block for every time block. Each block is itself a BlockVector with
            3 blocks. The first block contains the duals of the equality constraints in the corresponding time
            block. The second block has the duals for the coupling constraints linking the states at the
            beginning of the time block to the coupling variables between the time block and the previous
            time block. The third block has the duals for the coupling constraints linking the states at the
            end of the time block to the coupling variables between the time block and the next time block.
        """
        return self._init_duals_eq

    def init_duals_ineq(self) -> BlockVector:
        """
        Returns
        -------
        init_duals_ineq: BlockVector
            The initial values for the duals of the inequality constraints. This BlockVector has one block for
            every time block.
        """
        return self._init_duals_ineq

    def set_duals_eq(self, duals_eq: BlockVector):
        """
        Parameters
        ----------
        duals_eq: BlockVector
            The values for the duals of the equality constraints, including the coupling constraints.
            This BlockVector has one block for every time block. Each block is itself a BlockVector with
            3 blocks. The first block contains the duals of the equality constraints in the corresponding time
            block. The second block has the duals for the coupling constraints linking the states at the
            beginning of the time block to the coupling variables between the time block and the previous
            time block. The third block has the duals for the coupling constraints linking the states at the
            end of the time block to the coupling variables between the time block and the next time block.
        """
        for ndx, nlp in self._nlps.items():
            sub_block = duals_eq.get_block(ndx)
            nlp.set_duals_eq(sub_block.get_block(0))
            self._duals_eq.get_block(ndx).set_block(0, sub_block.get_block(0))
            self._duals_eq.get_block(ndx).set_block(1, sub_block.get_block(1))

    def set_duals_ineq(self, duals_ineq: BlockVector):
        """
        Parameters
        ----------
        duals_ineq: BlockVector
            The values for the duals of the inequality constraints. This BlockVector has one block for
            every time block.
        """
        for ndx, nlp in self._nlps.items():
            nlp.set_duals_ineq(duals_ineq.get_block(ndx))
            self._duals_ineq.set_block(ndx, duals_ineq.get_block(ndx))

    def get_duals_eq(self) -> BlockVector:
        """
        Returns
        -------
        duals_eq: BlockVector
            The values for the duals of the equality constraints, including the coupling constraints.
            This BlockVector has one block for every time block. Each block is itself a BlockVector with
            3 blocks. The first block contains the duals of the equality constraints in the corresponding time
            block. The second block has the duals for the coupling constraints linking the states at the
            beginning of the time block to the coupling variables between the time block and the previous
            time block. The third block has the duals for the coupling constraints linking the states at the
            end of the time block to the coupling variables between the time block and the next time block.
        """
        return self._duals_eq

    def get_duals_ineq(self) -> BlockVector:
        """
        Returns
        -------
        duals_ineq: BlockVector
            The values for the duals of the inequality constraints. This BlockVector has one block for
            every time block.
        """
        return self._duals_ineq

    def evaluate_eq_constraints(self) -> BlockVector:
        """
        Returns
        -------
        eq_resid: BlockVector
            The residuals of the equality constraints, including the coupling constraints.
            This BlockVector has one block for every time block. Each block is itself a BlockVector with
            3 blocks. The first block contains the residuals of the equality constraints in the corresponding time
            block. The second block has the residuals for the coupling constraints linking the states at the
            beginning of the time block to the coupling variables between the time block and the previous
            time block. The third block has the residuals for the coupling constraints linking the states at the
            end of the time block to the coupling variables between the time block and the next time block.
        """
        for ndx, nlp in self._nlps.items():
            sub_block = BlockVector(2)
            sub_block.set_block(0, nlp.evaluate_eq_constraints())
            sub_block.set_block(1, (self._linking_matrices[ndx] * nlp.get_primals() -
                                    (self._link_coupling_matrices[ndx] *
                                     self._primals.get_block(self._num_scenarios))))
            self._eq_resid.set_block(ndx, sub_block)
        return self._eq_resid

    def evaluate_ineq_constraints(self) -> BlockVector:
        """
        Returns
        -------
        ineq_resid: BlockVector
            The residuals of the inequality constraints. This BlockVector has one block for
            every time block.
        """
        for ndx, nlp in self._nlps.items():
            self._ineq_resid.set_block(ndx, nlp.evaluate_ineq_constraints())
        return self._ineq_resid

    def evaluate_jacobian_eq(self) -> BlockMatrix:
        """
        Returns
        -------
        jac_eq: BlockMatrix
            The jacobian of the equality constraints. The rows have the same structure as the BlockVector
            returned from evaluate_eq_constraints. The columns have the same structure as the BlockVector
            returned from get_primals.
        """
        # diagonal blocks
        for ndx, nlp in self._nlps.items():
            self._jac_eq.get_block(ndx, ndx).set_block(0, 0, nlp.evaluate_jacobian_eq())
        return self._jac_eq

    def evaluate_jacobian_ineq(self) -> BlockMatrix:
        """
        Returns
        -------
        jac_ineq: BlockMatrix
            The jacobian of the inequality constraints. The rows have the same structure as the BlockVector
            returned from evaluate_ineq_constraints. The columns have the same structure as the BlockVector
            returned from get_primals.
        """
        for ndx, nlp in self._nlps.items():
            self._jac_ineq.set_block(ndx, ndx, nlp.evaluate_jacobian_ineq())
        return self._jac_ineq

    def init_slacks(self) -> BlockVector:
        return self._init_slacks

    def init_duals_primals_lb(self) -> BlockVector:
        return self._init_duals_primals_lb

    def init_duals_primals_ub(self) -> BlockVector:
        return self._init_duals_primals_ub

    def init_duals_slacks_lb(self) -> BlockVector:
        return self._init_duals_slacks_lb

    def init_duals_slacks_ub(self) -> BlockVector:
        return self._init_duals_slacks_ub

    def set_slacks(self, slacks: BlockVector):
        for ndx, nlp in self._nlps.items():
            nlp.set_slacks(slacks.get_block(ndx))
            self._slacks.set_block(ndx, slacks.get_block(ndx))

    def set_duals_primals_lb(self, duals: BlockVector):
        for ndx, nlp in self._nlps.items():
            nlp.set_duals_primals_lb(duals.get_block(ndx))
            self._duals_primals_lb.set_block(ndx, duals.get_block(ndx))

    def set_duals_primals_ub(self, duals: BlockVector):
        for ndx, nlp in self._nlps.items():
            nlp.set_duals_primals_ub(duals.get_block(ndx))
            self._duals_primals_ub.set_block(ndx, duals.get_block(ndx))

    def set_duals_slacks_lb(self, duals: BlockVector):
        for ndx, nlp in self._nlps.items():
            nlp.set_duals_slacks_lb(duals.get_block(ndx))
            self._duals_slacks_lb.set_block(ndx, duals.get_block(ndx))

    def set_duals_slacks_ub(self, duals: BlockVector):
        for ndx, nlp in self._nlps.items():
            nlp.set_duals_slacks_ub(duals.get_block(ndx))
            self._duals_slacks_ub.set_block(ndx, duals.get_block(ndx))

    def get_slacks(self) -> BlockVector:
        return self._slacks

    def get_duals_primals_lb(self) -> BlockVector:
        return self._duals_primals_lb

    def get_duals_primals_ub(self) -> BlockVector:
        return self._duals_primals_ub

    def get_duals_slacks_lb(self) -> BlockVector:
        return self._duals_slacks_lb

    def get_duals_slacks_ub(self) -> BlockVector:
        return self._duals_slacks_ub

    def set_barrier_parameter(self, barrier: float):
        for ndx, nlp in self._nlps.items():
            nlp.set_barrier_parameter(barrier)

    def evaluate_primal_dual_kkt_matrix(self, timer: HierarchicalTimer = None) -> BlockMatrix:
        for ndx, nlp in self._nlps.items():
            sub_kkt = nlp.evaluate_primal_dual_kkt_matrix()
            self._kkt.get_block(ndx, ndx).set_block(0, 0, sub_kkt)
        return self._kkt

    def evaluate_primal_dual_kkt_rhs(self, timer: HierarchicalTimer = None) -> BlockVector:
        for ndx, nlp in self._nlps.items():
            sub_rhs = nlp.evaluate_primal_dual_kkt_rhs()
            sub_sub_rhs = sub_rhs.get_block(0)
            sub_sub_rhs -= self._linking_matrices[ndx].transpose().dot(self._duals_eq.get_block(ndx).get_block(1))
            sub_rhs.set_block(0, sub_sub_rhs)
            self._rhs.get_block(ndx).set_block(0, sub_rhs)
            sub_rhs = self._link_coupling_matrices[ndx] * self._primals.get_block(self._num_scenarios) - self._linking_matrices[ndx] * nlp.get_primals()
            self._rhs.get_block(ndx).set_block(1, sub_rhs)
        last_block = 0
        for ndx, nlp in self._nlps.items():
            last_block += self._link_coupling_matrices[ndx].transpose() * self._duals_eq.get_block(ndx).get_block(1)
        self._rhs.set_block(self._num_scenarios, last_block)
        return self._rhs

    def set_primal_dual_kkt_solution(self, sol: BlockVector):
        for ndx, nlp in self._nlps.items():
            nlp.set_primal_dual_kkt_solution(sol.get_block(ndx).get_block(0))
            self._delta_primals.set_block(ndx, nlp.get_delta_primals())
            self._delta_slacks.set_block(ndx, nlp.get_delta_slacks())
            self._delta_duals_eq.get_block(ndx).set_block(0, nlp.get_delta_duals_eq())
            self._delta_duals_ineq.set_block(ndx, nlp.get_delta_duals_ineq())
            self._delta_duals_primals_lb.set_block(ndx, nlp.get_delta_duals_primals_lb())
            self._delta_duals_primals_ub.set_block(ndx, nlp.get_delta_duals_primals_ub())
            self._delta_duals_slacks_lb.set_block(ndx, nlp.get_delta_duals_slacks_lb())
            self._delta_duals_slacks_ub.set_block(ndx, nlp.get_delta_duals_slacks_ub())
            self._delta_duals_eq.get_block(ndx).set_block(1, sol.get_block(ndx).get_block(1))
        self._delta_primals.set_block(self._num_scenarios, sol.get_block(self._num_scenarios))

    def get_delta_primals(self) -> BlockVector:
        return self._delta_primals

    def get_delta_slacks(self) -> BlockVector:
        return self._delta_slacks

    def get_delta_duals_eq(self) -> BlockVector:
        return self._delta_duals_eq

    def get_delta_duals_ineq(self) -> BlockVector:
        return self._delta_duals_ineq

    def get_delta_duals_primals_lb(self) -> BlockVector:
        return self._delta_duals_primals_lb

    def get_delta_duals_primals_ub(self) -> BlockVector:
        return self._delta_duals_primals_ub

    def get_delta_duals_slacks_lb(self) -> BlockVector:
        return self._delta_duals_slacks_lb

    def get_delta_duals_slacks_ub(self) -> BlockVector:
        return self._delta_duals_slacks_ub

    def regularize_equality_gradient(self, kkt: BlockMatrix, coef: float, copy_kkt: bool = True, block_indices: Iterable = None) -> BlockMatrix:
        if copy_kkt:
            kkt = kkt.copy()

        if block_indices == None:
            block_indices = self._nlps.keys()
        elif block_indices == []:
            return kkt
        else:
            block_indices = [i for i in block_indices if i in self._nlps.keys()]

        for ndx in block_indices:
            nlp = self._nlps[ndx]
            nlp.regularize_equality_gradient(kkt=kkt.get_block(ndx, ndx).get_block(0, 0),
                                             coef=coef,
                                             copy_kkt=False)
            # NOTE: This regularization should not be needed, as linking matrices always have full col. rank
            # ptb = coef * identity(self._num_first_stage_vars_by_scenario[ndx], format='coo')
            # kkt.get_block(ndx, ndx).set_block(1, 1, ptb)
        return kkt

    def regularize_hessian(self, kkt: BlockMatrix, coef: float, copy_kkt: bool = True, block_indices: Iterable = None) -> BlockMatrix:
        if copy_kkt:
            kkt = kkt.copy()

        if block_indices == None:
            block_indices = self._nlps.keys()
        elif block_indices == []:
            return kkt
        else:
            block_indices = [i for i in block_indices if i in self._nlps.keys()]

        for ndx in block_indices:
            nlp = self._nlps[ndx]
            nlp.regularize_hessian(kkt=kkt.get_block(ndx, ndx).get_block(0, 0),
                                   coef=coef,
                                   copy_kkt=False)
        #NOTE: Is this regularization needed, if local complicating variables were already regularized above?
        block = kkt.get_block(self._num_scenarios, self._num_scenarios)
        ptb = coef * identity(block.shape[0], format='coo')
        kkt.set_block(self._num_scenarios, self._num_scenarios, ptb)
        return kkt

    def load_primals_into_pyomo_model(self):
        """
        This method takes the current values for the primal variables (those you would get
        from the get_primals() method), and loads them into the corresponding Pyomo variables.
        """
        for ndx, nlp in self._nlps.items():
            nlp.load_primals_into_pyomo_model()

    def pyomo_model(self, scenario_id) -> _BlockData:
        """
        Parameters
        ----------
        scenario_id: Any
            The scenario for which the pyomo model should be returned.

        Returns
        -------
        m: _BlockData
            The pyomo model for the time block corresponding to ndx.
        """
        return self._nlps[self._scenario_id_to_ndx[scenario_id]].pyomo_model()

    def get_pyomo_variables(self, scenario_id) -> Sequence[_GeneralVarData]:
        """
        Parameters
        ----------
        scenario_id: Any
            The scenario for which pyomo variables should be returned

        Returns
        -------
        pyomo_vars: list of _GeneralVarData
            The pyomo variables in the model for the time block corresponding to ndx
        """
        return self._nlps[self._scenario_id_to_ndx[scenario_id]].get_pyomo_variables()

    def get_pyomo_constraints(self, scenario_id) -> Sequence[_GeneralConstraintData]:
        """
        Parameters
        ----------
        scenario_id: Any
            The scenario for which pyomo constraints should be returned

        Returns
        -------
        pyomo_cons: list of _GeneralConstraintData
            The pyomo constraints in the model for the time block corresponding to ndx
        """
        return self._nlps[self._scenario_id_to_ndx[scenario_id]].get_pyomo_constraints()

    def variable_names(self, scenario_id):
        return self._nlps[self._scenario_id_to_ndx[scenario_id]].variable_names()

    def constraint_names(self, scenario_id):
        return self._nlps[self._scenario_id_to_ndx[scenario_id]].constraint_names()

    def get_primal_indices(self, scenario_id: Any, pyomo_variables: Sequence[_GeneralVarData]) -> Sequence[int]:
        """
        Parameters
        ----------
        scenario_id: Any
            The scenario
        pyomo_variables: Sequence of _GeneralVarData
            The pyomo variables for which the indices should be returned

        Returns
        -------
        var_indices: Sequence of int
            The indices of the corresponding pyomo variables. Note that these
            indices correspond to the specified time block, not the overall indices.
            In other words, the indices that are returned are the indices into the
            block within get_primals corresponding to ndx.
        """
        return self._nlps[self._scenario_id_to_ndx[scenario_id]].get_primal_indices(pyomo_variables)

    def get_constraint_indices(self, scenario_id, pyomo_constraints) -> Sequence[int]:
        """
        Parameters
        ----------
        scenario_id: Any
            The scenario
        pyomo_constraints: Sequence of _GeneralConstraintData
            The pyomo constraints for which the indices should be returned

        Returns
        -------
        con_indices: Sequence of int
            The indices of the corresponding pyomo constraints. Note that these
            indices correspond to the specified time block, not the overall indices.
        """
        return self._nlps[self._scenario_id_to_ndx[scenario_id]].get_constraint_indices(pyomo_constraints)