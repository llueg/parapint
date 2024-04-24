import numpy as np
import enum
from scipy.sparse.linalg._isolve.utils import make_system
from scipy.optimize import LbfgsInvHessProduct
from typing import Dict
from pyomo.common.config import ConfigDict, ConfigValue, PositiveFloat, NonNegativeInt, NonNegativeFloat, InEnum

class LbfgsSamplingOptions(enum.Enum):
    last = 0
    first = 1
    uniform = 2
    disable = -1

class LbfgsApproxOptions(ConfigDict):

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
        self.declare('m', ConfigValue(domain=NonNegativeInt))
        # TODO: not sure if this is the intended usage
        self.declare('sampling', ConfigValue(domain=InEnum(LbfgsSamplingOptions)))

        self.m = 10
        self.sampling = LbfgsSamplingOptions.disable

class PcgOptions(ConfigDict):

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
        self.declare('a_tol', ConfigValue(domain=PositiveFloat))
        self.declare('r_tol', ConfigValue(domain=PositiveFloat))
        self.declare('lbfgs_sampling_options', ConfigValue(domain=LbfgsApproxOptions))

        self.max_iter = None
        self.a_tol = 1e-6
        self.r_tol = 1e-6
        self.lbfgs_sampling_options = LbfgsApproxOptions()


class LbfgsInvHessCollector:

    def __init__(self, approx_options: LbfgsApproxOptions, dim: int):
        self._sampling_strategy = approx_options.sampling
        self._m = approx_options.m
        self._dim = dim
        self._sk = np.zeros((self._m, self._dim))
        self._yk = np.zeros((self._m, self._dim))
        self._counter = 0

        if approx_options.sampling == LbfgsSamplingOptions.uniform:
            # use first m-1 for uniform samples, last for current
            # from Morales, J.L. and Nocedal, J. (2000). Automatic preconditioning
            # by limited memory quasi-newton updating.
            # SIAM Journal on Optimization, 10(4), 1079-1096.
            assert self._m % 2 == 1
            self._l = np.arange(1, (self._m - 1)/2 + 1)
            self._k_indxs = list(range(0, self._m - 1))
            self._cycle = 1

    def pop_inv_hessian_approx(self):
        assert self._counter > 0
        if self._counter < self._m:
            ret = LbfgsInvHessProduct(self._sk[:self._counter], self._yk[:self._counter])
        else:
            ret = LbfgsInvHessProduct(self._sk, self._yk)
        self._counter = 0
        self._sk = np.zeros((self._m, self._dim))
        self._yk = np.zeros((self._m, self._dim))
        return ret
    
    def update(self, s, y, pcg_iter: int) -> None:
        if pcg_iter < self._m:
            self._sk[pcg_iter] = s
            self._yk[pcg_iter] = y
            self._counter += 1
            return None
        elif self._sampling_strategy == LbfgsSamplingOptions.first:
            pass
        elif self._sampling_strategy == LbfgsSamplingOptions.last:
            self._sk = np.roll(self._sk, -1, axis=0)
            self._yk = np.roll(self._yk, -1, axis=0)
        elif self._sampling_strategy == LbfgsSamplingOptions.uniform:
            _, k_out = self._uniform_sample(pcg_iter)
            if k_out is not None:
                self._sk[k_out] = s
                self._yk[k_out] = y
        # Always keep last iterate
        self._sk[-1] = s
        self._yk[-1] = y
        self._counter += 1
        
    def _uniform_sample(self, k: int):
        k_indxs = (((self._m - 1)/2) + self._l - 1) * (2 ** self._cycle)
        in_idx = self._l[np.where(k_indxs == k)]
        if len(in_idx) == 0:
            return None, None
        else:
            l_out = (2*in_idx[0] - 1) * (2 ** (self._cycle - 1))
            rem_idx = self._k_indxs.index(int(l_out))
            self._k_indxs.remove(int(l_out))
            self._k_indxs.append(k)
            if in_idx[0] == (self._m - 1)/2:
                self._cycle += 1
            return k, rem_idx



def pcg(A, b, x0=None, M=None, pcg_options: PcgOptions = PcgOptions(),
        callback=None):
    """Adaptation of scipy.sparse.linalg.cg

    """
    maxiter = pcg_options.max_iter
    atol = pcg_options.a_tol
    rtol = pcg_options.r_tol
    hess_approx_options = pcg_options.lbfgs_sampling_options
    if hess_approx_options.sampling != LbfgsSamplingOptions.disable:
        inv_hess_collector = LbfgsInvHessCollector(hess_approx_options, len(b))
        collect = lambda s, y, pcg_iter: inv_hess_collector.update(s, y, pcg_iter)
    else:
        collect = lambda s, y, pcg_iter: None

    A, M, x, b, postprocess = make_system(A, M, x0, b)
    bnrm2 = np.linalg.norm(b)

    atol = max(float(atol), float(rtol) * float(bnrm2))

    if bnrm2 == 0:
        return postprocess(b), 0

    n = len(b)

    if maxiter is None:
        maxiter = n*10

    dotprod = np.dot

    matvec = A.matvec
    psolve = M.matvec
    r = b - matvec(x) if x.any() else b.copy()

    rho_prev, p = None, None

    for iteration in range(maxiter):
        if np.linalg.norm(r) < atol:
            return postprocess(x), 0

        z = psolve(r)
        rho_cur = dotprod(r, z)
        if iteration > 0:
            beta = rho_cur / rho_prev
            p *= beta
            p += z
        else:
            p = np.empty_like(r)
            p[:] = z[:]

        q = matvec(p)
        alpha = rho_cur / dotprod(p, q)
        x += alpha*p
        r -= alpha*q
        rho_prev = rho_cur

        collect(alpha*p, -alpha*q, iteration)
        if callback:
            callback(x)

    else:
        return postprocess(x), maxiter
    


