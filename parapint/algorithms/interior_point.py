from pyomo.contrib.pynumero.interfaces.utils import build_bounds_mask, build_compression_matrix
import numpy as np
import logging
import time
from parapint.linalg.results import LinearSolverStatus
from parapint.linalg.iterative.pcg import PcgSolutionStatus
from parapint.linalg.schur_complement.mpi_implicit_schur_complement import MPIBaseImplicitSchurComplementLinearSolver
from pyomo.common.timing import HierarchicalTimer
from parapint.utils import MPIHierarchicalTimer, TimerCollection
import enum
from parapint.interfaces.interface import BaseInteriorPointInterface
from parapint.interfaces.distributed_regularization import MPIDistStochasticSchurComplementInteriorPointInterface, DistStochasticSchurComplementInteriorPointInterface
from parapint.linalg.base_linear_solver_interface import LinearSolverInterface
from parapint.linalg.schur_complement.explicit_schur_complement import SchurComplementLinearSolver
from typing import Optional
from pyomo.common.config import ConfigDict, ConfigValue, PositiveFloat, NonNegativeInt, NonNegativeFloat
from typing import List, Tuple, Dict
import pandas as pd

"""
Interface Requirements
----------------------
1) duals_primals_lb[i] must always be 0 if primals_lb[i] is -inf
2) duals_primals_ub[i] must always be 0 if primals_ub[i] is inf
3) duals_slacks_lb[i] must always be 0 if ineq_lb[i] is -inf
4) duals_slacks_ub[i] must always be 0 if ineq_ub[i] is inf
"""


logger = logging.getLogger(__name__)


class InteriorPointHistory:

    def __init__(self, logger):
        self._header_fmt = {}
        self._history = {}
        self.logger = logger
        self.timer: MPIHierarchicalTimer = None

    # def add_iter_data(self, iter:int, **kwargs):
    #     if iter not in self._history:
    #         self._history[iter] = {}
    #     for arg, width in self._iter_args:
    #         self._history[iter][arg] = kwargs.pop(arg)
        
    def log_header(self, args: List[Tuple[str,int, str]]):
        log_string = ''
        for arg, width, fmt in args:
            #log_string += f'{arg:<{width}}'
            log_string += '{arg:<{width}}'.format(arg=arg, width=width)
            self._header_fmt[arg] = (width, fmt)
        self.logger.info(log_string)

    def log_iter(self, **kwargs):
        log_string = ''
        for arg, (width, fmt) in self._header_fmt.items():
            #log_string += f'{kwargs.get(arg, "na"):<{width}}'
            log_string += '{arg:<{width}{fmt}}'.format(arg=kwargs.get(arg, "na"), width=width, fmt=fmt)
        self.logger.info(log_string)
        iter = kwargs.pop('iter')
        if iter not in self._history:
            self._history[iter] = {}
        for arg in kwargs:
            self._history[iter][arg] = kwargs.get(arg)
    
    def save(self, filename=None):
        if filename is not None:
            df = pd.DataFrame.from_dict(self._history, orient='index')
            df.index.name = 'iter'
            df.to_csv(filename)



class InteriorPointStatus(enum.Enum):
    optimal = 0
    error = 1


class InertiaCorrectionOptions(ConfigDict):
    """
    Attributes
    ----------
    init_coef: float
    factor_increase: float
    factor_decrease: float
    max_coef: float
    """
    def __init__(self,
                 description=None,
                 doc=None,
                 implicit=False,
                 implicit_domain=None,
                 visibility=0):
        super().__init__(description=description,
                         doc=doc,
                         implicit=implicit,
                         implicit_domain=implicit_domain,
                         visibility=visibility)
        self.declare('init_coef', ConfigValue(domain=PositiveFloat))
        self.declare('factor_increase', ConfigValue(domain=PositiveFloat))
        self.declare('factor_decrease', ConfigValue(domain=PositiveFloat))
        self.declare('max_coef', ConfigValue(domain=PositiveFloat))

        self.init_coef = 1e-8
        self.factor_increase = 10
        self.factor_decrease = 1/3
        self.max_coef = 1e9


class LinalgOptions(ConfigDict):
    """
    Attributes
    ----------
    solver: LinearSolverInterface
    reallocation_factor: float
    max_num_reallocations: int
    """
    def __init__(self,
                 description=None,
                 doc=None,
                 implicit=False,
                 implicit_domain=None,
                 visibility=0):
        super().__init__(description=description,
                         doc=doc,
                         implicit=implicit,
                         implicit_domain=implicit_domain,
                         visibility=visibility)
        self.declare('solver', ConfigValue())
        self.declare('reallocation_factor', ConfigValue(domain=PositiveFloat))
        self.declare('max_num_reallocations', ConfigValue(domain=NonNegativeInt))

        self.solver: LinearSolverInterface = None
        self.reallocation_factor = 2
        self.max_num_reallocations = 5


class LineSearchOptions(ConfigDict):
    """
    Attributes
    ----------
    max_iter: int
    disable: bool
    """
    def __init__(self,
                 description=None,
                 doc=None,
                 implicit=False,
                 implicit_domain=None,
                 visibility=0):
        super().__init__(description=description,
                         doc=doc,
                         implicit=implicit,
                         implicit_domain=implicit_domain,
                         visibility=visibility)
        self.declare('max_iter', ConfigValue(domain=NonNegativeInt))
        self.declare('disable', ConfigValue(domain=bool))
        self.declare('step_anyway', ConfigValue(domain=bool))

        self.max_iter: int = 4
        self.disable: bool = True
        self.step_anyway: bool = True


class IPOptions(ConfigDict):
    """
    Attributes
    ----------
    max_iter: int
    tol: float
    init_barrier_parameter: float
    minimum_barrier_parameter: float
    report_timing: bool
    use_inertia_correction: bool
    inertia_correction: InertiaCorrectionOptions
    linalg: LinalgOptions
    line_search: LineSearchOptions
    unified_step: bool
    error_scaling: float
    """
    def __init__(self,
                 description=None,
                 doc=None,
                 implicit=False,
                 implicit_domain=None,
                 visibility=0):
        super().__init__(description=description,
                         doc=doc,
                         implicit=implicit,
                         implicit_domain=implicit_domain,
                         visibility=visibility)
        self.declare('max_iter', ConfigValue(domain=NonNegativeInt))
        self.declare('tol', ConfigValue(domain=PositiveFloat))
        self.declare('init_barrier_parameter', ConfigValue(domain=PositiveFloat))
        self.declare('minimum_barrier_parameter', ConfigValue(domain=PositiveFloat))
        self.declare('barrier_decrease', ConfigValue(domain=PositiveFloat))
        self.declare('report_timing', ConfigValue(domain=bool))
        self.declare('use_inertia_correction', ConfigValue(domain=bool))
        self.declare('inertia_correction', InertiaCorrectionOptions())
        self.declare('linalg', LinalgOptions())
        self.declare('line_search', LineSearchOptions())
        self.declare('unified_step', ConfigValue(domain=bool))
        self.declare('error_scaling', ConfigValue(domain=PositiveFloat))
        self.declare('bounds_relaxation_factor', ConfigValue(domain=NonNegativeFloat))

        self.max_iter = 1000
        self.tol = 1e-8
        self.init_barrier_parameter = 0.1
        self.minimum_barrier_parameter = 1e-9
        self.barrier_decrease = 10
        self.report_timing = False
        self.use_inertia_correction = True
        self.inertia_correction = InertiaCorrectionOptions()
        self.linalg = LinalgOptions()
        self.line_search: LineSearchOptions = LineSearchOptions()
        self.unified_step: bool = False
        self.error_scaling: float = 100
        self.bounds_relaxation_factor: float = 1e-8


def check_convergence(interface, barrier, error_scaling: float, timer=None):
    """
    Parameters
    ----------
    interface: BaseInteriorPointInterface
    barrier: float
    timer: HierarchicalTimer

    Returns
    -------
    primal_inf: float
    dual_inf: float
    complimentarity_inf: float
    """
    if timer is None:
        timer = HierarchicalTimer()

    slacks = interface.get_slacks()
    timer.start('grad obj')
    grad_obj = interface.get_obj_factor() * interface.evaluate_grad_objective()
    timer.stop('grad obj')
    timer.start('jac eq')
    jac_eq = interface.evaluate_jacobian_eq()
    timer.stop('jac eq')
    timer.start('jac ineq')
    jac_ineq = interface.evaluate_jacobian_ineq()
    timer.stop('jac ineq')
    timer.start('eq cons')
    eq_resid = interface.evaluate_eq_constraints()
    timer.stop('eq cons')
    timer.start('ineq cons')
    ineq_resid = interface.evaluate_ineq_constraints() - slacks
    timer.stop('ineq cons')
    primals = interface.get_primals()
    duals_eq = interface.get_duals_eq()
    duals_ineq = interface.get_duals_ineq()
    duals_primals_lb = interface.get_duals_primals_lb()
    duals_primals_ub = interface.get_duals_primals_ub()
    duals_slacks_lb = interface.get_duals_slacks_lb()
    duals_slacks_ub = interface.get_duals_slacks_ub()

    primals_lb = interface.primals_lb()
    primals_ub = interface.primals_ub()
    primals_lb_mod = primals_lb.copy()
    primals_ub_mod = primals_ub.copy()
    primals_lb_mod[np.isneginf(primals_lb)] = 0  # these entries get multiplied by 0
    primals_ub_mod[np.isinf(primals_ub)] = 0  # these entries get multiplied by 0

    ineq_lb = interface.ineq_lb()
    ineq_ub = interface.ineq_ub()
    ineq_lb_mod = ineq_lb.copy()
    ineq_ub_mod = ineq_ub.copy()
    ineq_lb_mod[np.isneginf(ineq_lb)] = 0  # these entries get multiplied by 0
    ineq_ub_mod[np.isinf(ineq_ub)] = 0  # these entries get multiplied by 0

    timer.start('grad_lag_primals')
    grad_lag_primals = grad_obj + jac_eq.transpose() * duals_eq
    grad_lag_primals += jac_ineq.transpose() * duals_ineq
    grad_lag_primals -= duals_primals_lb
    grad_lag_primals += duals_primals_ub
    timer.stop('grad_lag_primals')
    timer.start('grad_lag_slacks')
    grad_lag_slacks = (-duals_ineq -
                       duals_slacks_lb +
                       duals_slacks_ub)
    timer.stop('grad_lag_slacks')
    timer.start('bound resids')
    primals_lb_resid = (primals - primals_lb_mod) * duals_primals_lb - barrier
    primals_ub_resid = (primals_ub_mod - primals) * duals_primals_ub - barrier
    primals_lb_resid[np.isneginf(primals_lb)] = 0
    primals_ub_resid[np.isinf(primals_ub)] = 0
    slacks_lb_resid = (slacks - ineq_lb_mod) * duals_slacks_lb - barrier
    slacks_ub_resid = (ineq_ub_mod - slacks) * duals_slacks_ub - barrier
    slacks_lb_resid[np.isneginf(ineq_lb)] = 0
    slacks_ub_resid[np.isinf(ineq_ub)] = 0
    timer.stop('bound resids')

    if eq_resid.size == 0:
        max_eq_resid = 0
    else:
        max_eq_resid = np.max(np.abs(eq_resid))
    if ineq_resid.size == 0:
        max_ineq_resid = 0
    else:
        max_ineq_resid = np.max(np.abs(ineq_resid))
    primal_inf = max(max_eq_resid, max_ineq_resid)

    max_grad_lag_primals = np.max(np.abs(grad_lag_primals))
    if grad_lag_slacks.size == 0:
        max_grad_lag_slacks = 0
    else:
        max_grad_lag_slacks = np.max(np.abs(grad_lag_slacks))
    dual_inf = max(max_grad_lag_primals, max_grad_lag_slacks)

    if primals_lb_resid.size == 0:
        max_primals_lb_resid = 0
    else:
        max_primals_lb_resid = np.max(np.abs(primals_lb_resid))
    if primals_ub_resid.size == 0:
        max_primals_ub_resid = 0
    else:
        max_primals_ub_resid = np.max(np.abs(primals_ub_resid))
    if slacks_lb_resid.size == 0:
        max_slacks_lb_resid = 0
    else:
        max_slacks_lb_resid = np.max(np.abs(slacks_lb_resid))
    if slacks_ub_resid.size == 0:
        max_slacks_ub_resid = 0
    else:
        max_slacks_ub_resid = np.max(np.abs(slacks_ub_resid))
    complimentarity_inf = max(max_primals_lb_resid, max_primals_ub_resid,
                              max_slacks_lb_resid, max_slacks_ub_resid)

    dual_scaling = ((
        np.abs(duals_eq).sum()
        + np.abs(duals_ineq).sum()
        + np.abs(duals_primals_lb).sum()
        + np.abs(duals_primals_ub).sum()
        + np.abs(duals_slacks_lb).sum()
        + np.abs(duals_slacks_ub).sum()
    ) / (
        duals_eq.size
        + duals_ineq.size
        + np.isfinite(primals_lb).sum()
        + np.isfinite(primals_ub).sum()
        + np.isfinite(ineq_lb).sum()
        + np.isfinite(ineq_ub).sum()
    ))
    dual_scaling = max(error_scaling, dual_scaling) / error_scaling

    compl_scaling = ((
        np.abs(duals_primals_lb).sum()
        + np.abs(duals_primals_ub).sum()
        + np.abs(duals_slacks_lb).sum()
        + np.abs(duals_slacks_ub).sum()
    ) / (
        np.isfinite(primals_lb).sum()
        + np.isfinite(primals_ub).sum()
        + np.isfinite(ineq_lb).sum()
        + np.isfinite(ineq_ub).sum()
    ))
    compl_scaling = max(error_scaling, compl_scaling) / error_scaling

    return primal_inf, dual_inf/dual_scaling, complimentarity_inf/compl_scaling


def line_search(
    interface: BaseInteriorPointInterface,
    barrier: float,
    options: LineSearchOptions,
    delta_primals,
    delta_slacks,
    delta_duals_eq,
    delta_duals_ineq,
    delta_duals_primals_lb,
    delta_duals_primals_ub,
    delta_duals_slacks_lb,
    delta_duals_slacks_ub,
    timer: Optional[HierarchicalTimer] = None,
) -> Optional[float]:
    raise NotImplementedError('This is just a placeholder for a line search')


def numeric_factorization(interface: BaseInteriorPointInterface,
                          kkt,
                          options: IPOptions,
                          inertia_coef,
                          timer: Optional[HierarchicalTimer] = None):
    logger.debug('{reg_iter:<10}{num_realloc:<10}{reg_coef:<10}{pos_eig:<10}'
                 '{neg_eig:<10}{zero_eig:<10}{status:<10}'.format(reg_iter='reg_iter', num_realloc='# realloc',
                                                                  reg_coef='reg_coef', pos_eig='pos_eig',
                                                                  neg_eig='neg_eig', zero_eig='zero_eig',
                                                                  status='status'))
    status, num_realloc = try_factorization_and_reallocation(kkt=kkt,
                                                             linear_solver=options.linalg.solver,
                                                             reallocation_factor=options.linalg.reallocation_factor,
                                                             max_iter=options.linalg.max_num_reallocations,
                                                             symbolic_or_numeric='numeric',
                                                             timer=timer)

    final_inertia_coef = 0

    if not options.use_inertia_correction:
        logger.debug('{reg_iter:<10}{num_realloc:<10}{reg_coef:<10.2e}'
                     '{pos_eig:<10}{neg_eig:<10}{zero_eig:<10}'
                     '{status:<10}'.format(reg_iter=0, num_realloc=num_realloc, reg_coef=final_inertia_coef,
                                           pos_eig=None, neg_eig=None, zero_eig=None, status=str(status)))
        if status != LinearSolverStatus.successful:
            raise RuntimeError('Could not factorize KKT system; linear solver status: ' + str(status))
    else:
        if status not in {LinearSolverStatus.successful, LinearSolverStatus.singular}:
            raise RuntimeError('Could not factorize KKT system; linear solver status: ' + str(status))

        if isinstance(interface, DistStochasticSchurComplementInteriorPointInterface):
            final_inertia_coef = distributed_inertia_check_and_regularization(interface=interface,
                                                                              kkt=kkt,
                                                                              status=status,
                                                                              options=options,
                                                                              linear_solver=options.linalg.solver,
                                                                              inertia_coef=inertia_coef,
                                                                              final_inertia_coef=final_inertia_coef,
                                                                              num_realloc=num_realloc,
                                                                              timer=timer)
        else:
            final_inertia_coef = inertia_check_and_regularization(interface=interface,
                                                                    kkt=kkt,
                                                                    status=status,
                                                                    options=options,
                                                                    inertia_coef=inertia_coef,
                                                                    final_inertia_coef=final_inertia_coef,
                                                                    num_realloc=num_realloc,
                                                                    timer=timer)

    return final_inertia_coef


def inertia_check_and_regularization(interface: BaseInteriorPointInterface,
                                     kkt,
                                     status: LinearSolverStatus,
                                     options: IPOptions,
                                     inertia_coef,
                                     final_inertia_coef,
                                     num_realloc,
                                     timer: Optional[HierarchicalTimer] = None):
    pos_eig, neg_eig, zero_eig = None, None, None
    _iter = 0
    while final_inertia_coef <= options.inertia_correction.max_coef:
        if _iter == 0:
            kkt = kkt.copy()
        if status == LinearSolverStatus.successful:
            pos_eig, neg_eig, zero_eig = options.linalg.solver.get_inertia()
        else:
            pos_eig, neg_eig, zero_eig = None, None, None
        logger.debug('{reg_iter:<10}{num_realloc:<10}{reg_coef:<10.2e}'
                        '{pos_eig:<10}{neg_eig:<10}{zero_eig:<10}'
                        '{status:<10}'.format(reg_iter=_iter, num_realloc=num_realloc, reg_coef=final_inertia_coef,
                                            pos_eig=str(pos_eig), neg_eig=str(neg_eig), zero_eig=str(zero_eig),
                                            status=str(status)))
        if ((neg_eig == interface.n_eq_constraints() + interface.n_ineq_constraints()) and
            (zero_eig == 0) and
            (status == LinearSolverStatus.successful)):
            break
        
        kkt = interface.regularize_equality_gradient(kkt=kkt, coef=-inertia_coef, copy_kkt=False)
        kkt = interface.regularize_hessian(kkt=kkt, coef=inertia_coef, copy_kkt=False)
        status, num_realloc = try_factorization_and_reallocation(kkt=kkt,
                                                                    linear_solver=options.linalg.solver,
                                                                    reallocation_factor=options.linalg.reallocation_factor,
                                                                    max_iter=options.linalg.max_num_reallocations,
                                                                    symbolic_or_numeric='numeric',
                                                                    timer=timer)
        final_inertia_coef = inertia_coef
        inertia_coef *= options.inertia_correction.factor_increase
        _iter += 1
    
    if ((neg_eig != interface.n_eq_constraints() + interface.n_ineq_constraints()) or
            (zero_eig != 0) or
            (status != LinearSolverStatus.successful)):
            raise RuntimeError('Exceeded maximum inertia correciton')

    return final_inertia_coef


def distributed_inertia_check_and_regularization(interface: DistStochasticSchurComplementInteriorPointInterface,
                                                kkt,
                                                status: LinearSolverStatus,
                                                options: IPOptions,
                                                linear_solver: SchurComplementLinearSolver,
                                                inertia_coef,
                                                final_inertia_coef,
                                                num_realloc,
                                                timer: Optional[HierarchicalTimer] = None):
    pos_eig, neg_eig, zero_eig = None, None, None
    _iter = 0
    while final_inertia_coef <= options.inertia_correction.max_coef:
        if _iter == 0:
            kkt = kkt.copy()
        if status == LinearSolverStatus.successful:
            #pos_eig, neg_eig, zero_eig = linear_solver.get_inertia()
            total_inertia, distributed_inertia = linear_solver.get_distributed_inertia()
            pos_eig, neg_eig, zero_eig = total_inertia
        else:
            pos_eig, neg_eig, zero_eig = None, None, None
        logger.debug('{reg_iter:<10}{num_realloc:<10}{reg_coef:<10.2e}'
                        '{pos_eig:<10}{neg_eig:<10}{zero_eig:<10}'
                        '{status:<10}'.format(reg_iter=_iter, num_realloc=num_realloc, reg_coef=final_inertia_coef,
                                            pos_eig=str(pos_eig), neg_eig=str(neg_eig), zero_eig=str(zero_eig),
                                            status=str(status)))
        if ((neg_eig == interface.n_eq_constraints() + interface.n_ineq_constraints()) and
            (zero_eig == 0) and
            (status == LinearSolverStatus.successful)):
            break
        
        blocks_to_be_regularized = []
        for ndx, inertia in distributed_inertia.items():
            if ndx in interface._nlps.keys():
                _pos, _neg, _zero = inertia
                n_eq = interface._nlps[ndx].n_eq_constraints()
                n_ineq = interface._nlps[ndx].n_ineq_constraints()
                if _neg != n_eq + n_ineq or _zero != 0:
                    blocks_to_be_regularized.append(ndx)

        print(f'iter {_iter}, blocks_to_be_regularized: {blocks_to_be_regularized}')
        kkt = interface.regularize_equality_gradient(kkt=kkt, coef=-inertia_coef, copy_kkt=False, block_indices=blocks_to_be_regularized)
        kkt = interface.regularize_hessian(kkt=kkt, coef=inertia_coef, copy_kkt=False, block_indices=blocks_to_be_regularized)
        status, num_realloc = try_factorization_and_reallocation(kkt=kkt,
                                                                linear_solver=linear_solver,
                                                                reallocation_factor=options.linalg.reallocation_factor,
                                                                max_iter=options.linalg.max_num_reallocations,
                                                                symbolic_or_numeric='numeric',
                                                                timer=timer)
        final_inertia_coef = inertia_coef
        inertia_coef *= options.inertia_correction.factor_increase
        _iter += 1
    
    if ((neg_eig != interface.n_eq_constraints() + interface.n_ineq_constraints()) or
            (zero_eig != 0) or
            (status != LinearSolverStatus.successful)):
            raise RuntimeError('Exceeded maximum inertia correciton')

    return final_inertia_coef



def numeric_factorization_and_back_solve(interface: BaseInteriorPointInterface,
                                         kkt,
                                         options: IPOptions,
                                         inertia_coef,
                                         timer: Optional[HierarchicalTimer] = None
                                         ):
    if timer is None:
        timer = HierarchicalTimer()
    logger.debug('{reg_iter:<10}{num_realloc:<10}{reg_coef:<10}{pos_eig:<10}'
                 '{neg_eig:<10}{zero_eig:<10}{status:<10}'.format(reg_iter='reg_iter', num_realloc='# realloc',
                                                                  reg_coef='reg_coef', pos_eig='pos_eig',
                                                                  neg_eig='neg_eig', zero_eig='zero_eig',
                                                                  status='status'))
    timer.start('factorize')
    timer.start('numeric')
    status, num_realloc = try_factorization_and_reallocation(kkt=kkt,
                                                             linear_solver=options.linalg.solver,
                                                             reallocation_factor=options.linalg.reallocation_factor,
                                                             max_iter=options.linalg.max_num_reallocations,
                                                             symbolic_or_numeric='numeric',
                                                             timer=timer)
    timer.stop('numeric')
    timer.stop('factorize')

    final_inertia_coef = 0
    if not options.use_inertia_correction:
        logger.debug('{reg_iter:<10}{num_realloc:<10}{reg_coef:<10.2e}'
                     '{pos_eig:<10}{neg_eig:<10}{zero_eig:<10}'
                     '{status:<10}'.format(reg_iter=0, num_realloc=num_realloc, reg_coef=final_inertia_coef,
                                           pos_eig=None, neg_eig=None, zero_eig=None, status=str(status)))
        if status != LinearSolverStatus.successful:
            raise RuntimeError('Could not factorize KKT system; linear solver status: ' + str(status))
        # do backsolve
        timer.start('back solve')
        assert isinstance(options.linalg.solver, MPIBaseImplicitSchurComplementLinearSolver), 'Expected MPIBaseImplicitSchurComplementLinearSolver'
        solver: MPIBaseImplicitSchurComplementLinearSolver = options.linalg.solver
        delta, pcg_status = solver.do_back_solve(interface.evaluate_primal_dual_kkt_rhs())
        num_pcg_iter = solver._current_pcg_solution.num_iterations
        timer.stop('back solve')
    else:
        if status not in {LinearSolverStatus.successful, LinearSolverStatus.singular}:
            raise RuntimeError('Could not factorize KKT system; linear solver status: ' + str(status))

        pos_eig, neg_eig, zero_eig = None, None, None
        _iter = 0
        while final_inertia_coef <= options.inertia_correction.max_coef:
            if status == LinearSolverStatus.successful:
                pos_eig, neg_eig, zero_eig = options.linalg.solver.get_inertia()
            else:
                pos_eig, neg_eig, zero_eig = None, None, None
            logger.debug('{reg_iter:<10}{num_realloc:<10}{reg_coef:<10.2e}'
                         '{pos_eig:<10}{neg_eig:<10}{zero_eig:<10}'
                         '{status:<10}'.format(reg_iter=_iter, num_realloc=num_realloc, reg_coef=final_inertia_coef,
                                               pos_eig=str(pos_eig), neg_eig=str(neg_eig), zero_eig=str(zero_eig),
                                               status=str(status)))
            if ((neg_eig == interface.n_eq_constraints() + interface.n_ineq_constraints()) and
                (zero_eig == 0) and
                (status == LinearSolverStatus.successful)):
                # do back solve here to check if inertia related error occurs
                assert isinstance(options.linalg.solver, MPIBaseImplicitSchurComplementLinearSolver), 'Expected MPIBaseImplicitSchurComplementLinearSolver'
                solver: MPIBaseImplicitSchurComplementLinearSolver = options.linalg.solver
                timer.start('back solve')
                delta, pcg_status = solver.do_back_solve(interface.evaluate_primal_dual_kkt_rhs())
                num_pcg_iter = solver._current_pcg_solution.num_iterations
                timer.stop('back solve')
                if pcg_status == PcgSolutionStatus.negative_curvature:
                    pass
                else:
                    break
                
            if _iter == 0:
                kkt = kkt.copy()
            kkt = interface.regularize_equality_gradient(kkt=kkt, coef=-inertia_coef, copy_kkt=False)
            kkt = interface.regularize_hessian(kkt=kkt, coef=inertia_coef, copy_kkt=False)
            timer.start('factorize')
            timer.start('numeric')
            status, num_realloc = try_factorization_and_reallocation(kkt=kkt,
                                                                     linear_solver=options.linalg.solver,
                                                                     reallocation_factor=options.linalg.reallocation_factor,
                                                                     max_iter=options.linalg.max_num_reallocations,
                                                                     symbolic_or_numeric='numeric',
                                                                     timer=timer)
            timer.stop('numeric')
            timer.stop('factorize')
            final_inertia_coef = inertia_coef
            inertia_coef *= options.inertia_correction.factor_increase
            _iter += 1

        if ((neg_eig != interface.n_eq_constraints() + interface.n_ineq_constraints()) or
            (zero_eig != 0) or
            (status != LinearSolverStatus.successful)):
            raise RuntimeError('Exceeded maximum inertia correciton')

    return delta, final_inertia_coef, num_pcg_iter



def ip_solve(interface: BaseInteriorPointInterface,
             options: Optional[IPOptions] = None,
             timer: Optional[HierarchicalTimer] = None
             ) -> Tuple[InteriorPointStatus, InteriorPointHistory]:
    """
    Parameters
    ----------
    interface: BaseInteriorPointInterface
        The interior point interface. This object handles the function evaluation,
        building the KKT matrix, and building the KKT right hand side.
    options: IPOptions
    timer: HierarchicalTimer
    """
    if options is None:
        options = IPOptions()

    if timer is None:
        timer = MPIHierarchicalTimer()

    timer.start('IP solve')
    timer.start('init')

    interface.set_bounds_relaxation_factor(options.bounds_relaxation_factor)

    barrier_parameter = options.init_barrier_parameter
    inertia_coef = options.inertia_correction.init_coef
    used_inertia_coef = 0

    t0 = time.time()
    primals = interface.init_primals().copy()
    slacks = interface.init_slacks().copy()
    duals_eq = interface.init_duals_eq().copy()
    duals_ineq = interface.init_duals_ineq().copy()
    duals_primals_lb = interface.init_duals_primals_lb().copy()
    duals_primals_ub = interface.init_duals_primals_ub().copy()
    duals_slacks_lb = interface.init_duals_slacks_lb().copy()
    duals_slacks_ub = interface.init_duals_slacks_ub().copy()

    process_init(primals, interface.primals_lb(), interface.primals_ub())
    process_init(slacks, interface.ineq_lb(), interface.ineq_ub())
    process_init_duals_lb(duals_primals_lb, interface.primals_lb())
    process_init_duals_ub(duals_primals_ub, interface.primals_ub())
    process_init_duals_lb(duals_slacks_lb, interface.ineq_lb())
    process_init_duals_ub(duals_slacks_ub, interface.ineq_ub())

    interface.set_barrier_parameter(barrier_parameter)

    alpha_primal_max = 1
    alpha_dual_max = 1
    alpha = 1

    # logger.info('{_iter:<6}'
    #             '{objective:<11}'
    #             '{primal_inf:<11}'
    #             '{dual_inf:<11}'
    #             '{compl_inf:<11}'
    #             '{barrier:<11}'
    #             '{alpha_p:<11}'
    #             '{alpha_d:<11}'
    #             '{alpha:<11}'
    #             '{reg:<11}'
    #             '{time:<7}'.format(_iter='Iter',
    #                                objective='Objective',
    #                                primal_inf='Prim Inf',
    #                                dual_inf='Dual Inf',
    #                                compl_inf='Comp Inf',
    #                                barrier='Barrier',
    #                                alpha_p='Prim Step',
    #                                alpha_d='Dual Step',
    #                                alpha='LS Step',
    #                                reg='Reg',
    #                                time='Time'))
    
    if isinstance(options.linalg.solver, MPIBaseImplicitSchurComplementLinearSolver):
        aux_logging_vars = [('pcg_iters', 10, '')]
        aux_logging_data = {'pcg_iters': 'na'}
    else:
        aux_logging_vars = []
        aux_logging_data = {}

    logging_vars = [('iter', 6, ''), ('objective', 11, '.2e'), ('primal_inf', 11, '.2e'), ('dual_inf', 11, '.2e'), ('compl_inf', 11, '.2e'),
                    ('barrier', 11, '.2e'), ('alpha_p', 11, '.2e'), ('alpha_d', 11, '.2e'), ('alpha', 11, '.2e'), ('reg', 11, '.2e'), ('time', 7, '.3f')]
    
    ip_history = InteriorPointHistory(logger)
    ip_history.log_header(logging_vars + aux_logging_vars)


    timer.stop('init')
    status = InteriorPointStatus.error

    for _iter in range(options.max_iter):
        interface.set_primals(primals)
        interface.set_slacks(slacks)
        interface.set_duals_eq(duals_eq)
        interface.set_duals_ineq(duals_ineq)
        interface.set_duals_primals_lb(duals_primals_lb)
        interface.set_duals_primals_ub(duals_primals_ub)
        interface.set_duals_slacks_lb(duals_slacks_lb)
        interface.set_duals_slacks_ub(duals_slacks_ub)

        timer.start('convergence check')
        primal_inf, dual_inf, complimentarity_inf = check_convergence(interface=interface, barrier=0, error_scaling=options.error_scaling, timer=timer)
        timer.stop('convergence check')
        objective = interface.evaluate_objective()
        # logger.info('{_iter:<6}'
        #             '{objective:<11.2e}'
        #             '{primal_inf:<11.2e}'
        #             '{dual_inf:<11.2e}'
        #             '{compl_inf:<11.2e}'
        #             '{barrier:<11.2e}'
        #             '{alpha_p:<11.2e}'
        #             '{alpha_d:<11.2e}'
        #             '{alpha:<11.2e}'
        #             '{reg:<11.2e}'
        #             '{time:<7.3f}'.format(_iter=_iter,
        #                                   objective=objective,
        #                                   primal_inf=primal_inf,
        #                                   dual_inf=dual_inf,
        #                                   compl_inf=complimentarity_inf,
        #                                   barrier=barrier_parameter,
        #                                   alpha_p=alpha_primal_max,
        #                                   alpha_d=alpha_dual_max,
        #                                   alpha=alpha,
        #                                   reg=used_inertia_coef,
        #                                   time=time.time() - t0))
        ip_history.log_iter(iter=_iter, objective=objective, primal_inf=primal_inf, dual_inf=dual_inf,
                            compl_inf=complimentarity_inf, barrier=barrier_parameter, alpha_p=alpha_primal_max,
                            alpha_d=alpha_dual_max, alpha=alpha, reg=used_inertia_coef, time=time.time() - t0, **aux_logging_data)

        if max(primal_inf, dual_inf, complimentarity_inf) <= options.tol:
            status = InteriorPointStatus.optimal
            break
        timer.start('convergence check')
        primal_inf, dual_inf, complimentarity_inf = check_convergence(interface=interface,
                                                                      barrier=barrier_parameter,
                                                                      error_scaling=options.error_scaling,
                                                                      timer=timer)
        timer.stop('convergence check')
        if max(primal_inf, dual_inf, complimentarity_inf) \
                <= options.barrier_decrease * barrier_parameter:
            barrier_parameter = max(options.minimum_barrier_parameter,
                                    min(0.5 * barrier_parameter, barrier_parameter ** 1.5))

        interface.set_barrier_parameter(barrier_parameter)
        timer.start('eval')
        timer.start('eval kkt')
        kkt = interface.evaluate_primal_dual_kkt_matrix(timer=timer)
        timer.stop('eval kkt')
        timer.start('eval rhs')
        rhs = interface.evaluate_primal_dual_kkt_rhs(timer=timer)
        timer.stop('eval rhs')
        timer.stop('eval')

        # Factorize linear system

        if _iter == 0:
            timer.start('factorize')
            timer.start('symbolic')
            sym_fact_status, sym_fact_iter = try_factorization_and_reallocation(kkt=kkt,
                                                                                linear_solver=options.linalg.solver,
                                                                                reallocation_factor=options.linalg.reallocation_factor,
                                                                                max_iter=options.linalg.max_num_reallocations,
                                                                                symbolic_or_numeric='symbolic',
                                                                                timer=timer)
            timer.stop('symbolic')
            timer.stop('factorize')
            if sym_fact_status != LinearSolverStatus.successful:
                raise RuntimeError('Could not factorize KKT system; linear solver status: ' + str(sym_fact_status))
        
        if isinstance(options.linalg.solver, MPIBaseImplicitSchurComplementLinearSolver):
            delta, used_inertia_coef, num_pcg_iter = numeric_factorization_and_back_solve(interface=interface,
                                                                            kkt=kkt,
                                                                            options=options,
                                                                            inertia_coef=inertia_coef,
                                                                            timer=timer)
            aux_logging_data['pcg_iters'] = num_pcg_iter
        else:
            timer.start('factorize')
            timer.start('numeric')
            used_inertia_coef = numeric_factorization(interface=interface,
                                                    kkt=kkt,
                                                    options=options,
                                                    inertia_coef=inertia_coef,
                                                    timer=timer)
            inertia_coef = used_inertia_coef * options.inertia_correction.factor_decrease
            if inertia_coef < options.inertia_correction.init_coef:
                inertia_coef = options.inertia_correction.init_coef
            timer.stop('numeric')
            timer.stop('factorize')

            timer.start('back solve')
            delta = options.linalg.solver.do_back_solve(rhs)
            timer.stop('back solve')

        interface.set_primal_dual_kkt_solution(delta)
        timer.start('frac boundary')
        alpha_primal_max, alpha_dual_max = fraction_to_the_boundary(interface=interface, tau=1 - barrier_parameter)
        if options.unified_step:
            tmp = min(alpha_primal_max, alpha_dual_max)
            alpha_primal_max = tmp
            alpha_dual_max = tmp
        timer.stop('frac boundary')

        delta_primals = interface.get_delta_primals()
        delta_slacks = interface.get_delta_slacks()
        delta_duals_eq = interface.get_delta_duals_eq()
        delta_duals_ineq = interface.get_delta_duals_ineq()
        delta_duals_primals_lb = interface.get_delta_duals_primals_lb()
        delta_duals_primals_ub = interface.get_delta_duals_primals_ub()
        delta_duals_slacks_lb = interface.get_delta_duals_slacks_lb()
        delta_duals_slacks_ub = interface.get_delta_duals_slacks_ub()

        delta_primals *= alpha_primal_max
        delta_slacks *= alpha_primal_max
        delta_duals_eq *= alpha_dual_max
        delta_duals_ineq *= alpha_dual_max
        delta_duals_primals_lb *= alpha_dual_max
        delta_duals_primals_ub *= alpha_dual_max
        delta_duals_slacks_lb *= alpha_dual_max
        delta_duals_slacks_ub *= alpha_dual_max

        if options.line_search.disable:
            alpha = 1
        else:
            alpha = line_search(
                interface,
                barrier_parameter,
                options.line_search,
                delta_primals,
                delta_slacks,
                delta_duals_eq,
                delta_duals_ineq,
                delta_duals_primals_lb,
                delta_duals_primals_ub,
                delta_duals_slacks_lb,
                delta_duals_slacks_ub,
                timer,
            )

            if alpha is None:
                logger.warning('line search failed')
                status = InteriorPointStatus.error
                break

        primals += alpha * delta_primals
        slacks += alpha * delta_slacks
        duals_eq += alpha * delta_duals_eq
        duals_ineq += alpha * delta_duals_ineq
        duals_primals_lb += alpha * delta_duals_primals_lb
        duals_primals_ub += alpha * delta_duals_primals_ub
        duals_slacks_lb += alpha * delta_duals_slacks_lb
        duals_slacks_ub += alpha * delta_duals_slacks_ub

    timer.stop('IP solve')
    ip_history.timer = timer
    if options.report_timing:
        print(timer)

    return status, ip_history


def try_factorization_and_reallocation(kkt, linear_solver: LinearSolverInterface, reallocation_factor, max_iter,
                                       symbolic_or_numeric, timer=None):
    if timer is None:
        timer = HierarchicalTimer()

    assert max_iter >= 1
    if symbolic_or_numeric == 'numeric':
        method = linear_solver.do_numeric_factorization
    else:
        assert symbolic_or_numeric == 'symbolic'
        method = linear_solver.do_symbolic_factorization
    for count in range(max_iter):
        res = method(matrix=kkt, raise_on_error=False, timer=timer)
        status = res.status
        if status == LinearSolverStatus.not_enough_memory:
            linear_solver.increase_memory_allocation(reallocation_factor)
        else:
            break
    return status, count


def _fraction_to_the_boundary_helper_lb(tau, x, delta_x, xl):
    delta_x_mod = delta_x.copy()
    delta_x_mod[delta_x_mod == 0] = 1
    alpha = -tau * (x - xl) / delta_x_mod
    alpha[delta_x >= 0] = np.inf
    if alpha.size == 0:
        return 1
    else:
        return min(alpha.min(), 1)


def _fraction_to_the_boundary_helper_ub(tau, x, delta_x, xu):
    delta_x_mod = delta_x.copy()
    delta_x_mod[delta_x_mod == 0] = 1
    alpha = tau * (xu - x) / delta_x_mod
    alpha[delta_x <= 0] = np.inf
    if alpha.size == 0:
        return 1
    else:
        return min(alpha.min(), 1)


def fraction_to_the_boundary(interface, tau):
    """
    Parameters
    ----------
    interface: parapint.interfaces.interface.BaseInteriorPointInterface
    tau: float

    Returns
    -------
    alpha_primal_max: float
    alpha_dual_max: float
    """
    primals = interface.get_primals()
    slacks = interface.get_slacks()
    duals_primals_lb = interface.get_duals_primals_lb()
    duals_primals_ub = interface.get_duals_primals_ub()
    duals_slacks_lb = interface.get_duals_slacks_lb()
    duals_slacks_ub = interface.get_duals_slacks_ub()

    delta_primals = interface.get_delta_primals()
    delta_slacks = interface.get_delta_slacks()
    delta_duals_primals_lb = interface.get_delta_duals_primals_lb()
    delta_duals_primals_ub = interface.get_delta_duals_primals_ub()
    delta_duals_slacks_lb = interface.get_delta_duals_slacks_lb()
    delta_duals_slacks_ub = interface.get_delta_duals_slacks_ub()

    primals_lb = interface.primals_lb()
    primals_ub = interface.primals_ub()
    ineq_lb = interface.ineq_lb()
    ineq_ub = interface.ineq_ub()

    alpha_primal_max_a = _fraction_to_the_boundary_helper_lb(
        tau=tau,
        x=primals,
        delta_x=delta_primals,
        xl=primals_lb)
    alpha_primal_max_b = _fraction_to_the_boundary_helper_ub(
        tau=tau,
        x=primals,
        delta_x=delta_primals,
        xu=primals_ub)
    alpha_primal_max_c = _fraction_to_the_boundary_helper_lb(
        tau=tau,
        x=slacks,
        delta_x=delta_slacks,
        xl=ineq_lb)
    alpha_primal_max_d = _fraction_to_the_boundary_helper_ub(
        tau=tau,
        x=slacks,
        delta_x=delta_slacks,
        xu=ineq_ub)
    alpha_primal_max = min(alpha_primal_max_a, alpha_primal_max_b,
                           alpha_primal_max_c, alpha_primal_max_d)

    _xl = duals_primals_lb.copy()
    _xl.fill(0)
    alpha_dual_max_a = _fraction_to_the_boundary_helper_lb(
        tau=tau,
        x=duals_primals_lb,
        delta_x=delta_duals_primals_lb,
        xl=_xl)
    alpha_dual_max_b = _fraction_to_the_boundary_helper_lb(
        tau=tau,
        x=duals_primals_ub,
        delta_x=delta_duals_primals_ub,
        xl=_xl)
    _xl = duals_slacks_lb.copy()
    _xl.fill(0)
    alpha_dual_max_c = _fraction_to_the_boundary_helper_lb(
        tau=tau,
        x=duals_slacks_lb,
        delta_x=delta_duals_slacks_lb,
        xl=_xl)
    alpha_dual_max_d = _fraction_to_the_boundary_helper_lb(
        tau=tau,
        x=duals_slacks_ub,
        delta_x=delta_duals_slacks_ub,
        xl=_xl)
    alpha_dual_max = min(alpha_dual_max_a, alpha_dual_max_b,
                         alpha_dual_max_c, alpha_dual_max_d)

    return alpha_primal_max, alpha_dual_max


def process_init(x, lb, ub):
    if np.any((ub - lb) < 0):
        raise ValueError(
            'Lower bounds for variables/inequalities should not be larger than upper bounds.')
    if np.any((ub - lb) == 0):
        raise ValueError(
            'Variables and inequalities should not have equal lower and upper bounds.')

    lb_mask = build_bounds_mask(lb)
    ub_mask = build_bounds_mask(ub)

    lb_only = np.logical_and(lb_mask, np.logical_not(ub_mask))
    ub_only = np.logical_and(ub_mask, np.logical_not(lb_mask))
    lb_and_ub = np.logical_and(lb_mask, ub_mask)
    out_of_bounds = ((x >= ub) + (x <= lb))
    out_of_bounds_lb_only = np.logical_and(out_of_bounds, lb_only)
    out_of_bounds_ub_only = np.logical_and(out_of_bounds, ub_only)
    out_of_bounds_lb_and_ub = np.logical_and(out_of_bounds, lb_and_ub)

    cm = build_compression_matrix(out_of_bounds_lb_only)
    x[out_of_bounds_lb_only] = cm * (lb + 1)

    cm = build_compression_matrix(out_of_bounds_ub_only)
    x[out_of_bounds_ub_only] = cm * (ub - 1)

    del cm
    cm1 = build_compression_matrix(lb_and_ub)
    cm2 = build_compression_matrix(out_of_bounds_lb_and_ub)
    x[out_of_bounds_lb_and_ub] = cm2 * (0.5 * cm1.transpose() * (cm1 * lb + cm1 * ub))


def process_init_duals_lb(x, lb):
    x[x <= 0] = 1
    x[np.isneginf(lb)] = 0


def process_init_duals_ub(x, ub):
    x[x <= 0] = 1
    x[np.isinf(ub)] = 0
