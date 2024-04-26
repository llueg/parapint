import unittest
import parapint
import pytest
from pyomo.contrib.pynumero.sparse import BlockMatrix, BlockVector
from parapint.linalg import ScipyInterface
import scipy.sparse as sps
from scipy.sparse import coo_matrix
import numpy as np
import sys

import parapint.linalg.iterative
import parapint.linalg.iterative.pcg

# Test suite in part taken from sps.linalg._isolve.tests.test_iterative.py

class LinearTestSystem:
    def __init__(self,
                 name,
                 A,
                 b=None,
                 expected_status=None,
                 ):
        self.name = name
        self.A = A
        if b is None:
            self.b = np.arange(A.shape[0], dtype=float)
        else:
            self.b = b
        if expected_status is None:
            self.expected_status = parapint.linalg.iterative.pcg.PcgSolutionStatus.successful
        else:
            self.expected_status = expected_status

def generate_test_systems():
    cases = []
    # Test systems
    N = 40
    data = np.ones((3, N))
    data[0, :] = 2
    data[1, :] = -1
    data[2, :] = -1
    Poisson1D = sps.spdiags(data, [0, -1, 1], N, N, format='csr')
    cases.append(LinearTestSystem('Poisson1D', A=Poisson1D))

    Poisson2D = sps.kronsum(Poisson1D, Poisson1D)
    cases.append(LinearTestSystem('Poisson2D', A=Poisson2D))

    np.random.seed(1234)
    data = np.random.rand(N, N)
    data = np.dot(data.conj(), data.T)
    cases.append(LinearTestSystem('RandomPD', A=data))

    A = np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0,-1, 0],
                [0, 0, 1, 0, 0, 0,-1, 0],
                [0, 0, 0, 1, 0, 0, 0,-1],
                [0, 0, 0, 0, 1, 0, 0,-1],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0,-1,-1, 0, 0, 0, 0, 0],
                [0, 0, 0,-1,-1, 0, 0, 0],
                ], dtype=np.double)
    
    cases.append(LinearTestSystem('SymIndef', A=A,
                                  expected_status=parapint.linalg.iterative.pcg.PcgSolutionStatus.negative_curvature))

    return cases

test_cases = generate_test_systems()

@pytest.fixture(params=test_cases, ids=[x.name for x in test_cases])
def test_case(request):
    return request.param

def test_maxiter(test_case):
    if test_case.expected_status == parapint.linalg.iterative.pcg.PcgSolutionStatus.negative_curvature:
        pytest.skip("Skipped test which terminates at unpredictable iteration count")
    A = test_case.A
    rtol = 1e-12
    b = test_case.b
    x0 = 0 * b

    pcg_options = parapint.linalg.PcgOptions()
    pcg_options.max_iter = 1
    pcg_options.rtol = rtol

    results = parapint.linalg.iterative.pcg.pcg_solve(A=A, b=b, x0=x0, pcg_options=pcg_options)

    assert results.num_iterations == 1


def test_residual(test_case):
    if not test_case.expected_status == parapint.linalg.iterative.pcg.PcgSolutionStatus.successful:
        pytest.skip("Skipped test which is expected to fail")
    A = test_case.A
    rtol = 1e-8
    atol = 1e-8
    b = test_case.b
    x0 = 0 * b

    pcg_options = parapint.linalg.PcgOptions()
    pcg_options.rtol = rtol
    pcg_options.atol = atol

    results = parapint.linalg.iterative.pcg.pcg_solve(A=A, b=b, x0=x0, pcg_options=pcg_options)

    residual = np.linalg.norm(A.dot(results.x) - b)
    assert residual < max(atol, rtol * np.linalg.norm(b))


neg_curvature_cases = [x for x in test_cases
                        if x.expected_status == parapint.linalg.iterative.pcg.PcgSolutionStatus.negative_curvature]
@pytest.mark.parametrize('case', neg_curvature_cases, ids=[x.name for x in neg_curvature_cases])
def test_neg_curvature_detection(case):
    # if not case.expected_status == parapint.linalg.iterative.pcg.PcgSolutionStatus.negative_curvature:
    #     pytest.skip("Skipped test which is expected to fail")
    A = case.A
    b = case.b
    x0 = 0 * b
    pcg_options = parapint.linalg.PcgOptions()
    results = parapint.linalg.iterative.pcg.pcg_solve(A=A, b=b, x0=x0, pcg_options=pcg_options)

    assert results.status == parapint.linalg.iterative.pcg.PcgSolutionStatus.negative_curvature



def test_hess_approx_collection_last():
    lbfgs_approx_options = parapint.linalg.iterative.pcg.LbfgsApproxOptions()
    lbfgs_approx_options.m = 3
    lbfgs_approx_options.sampling = parapint.linalg.iterative.pcg.LbfgsSamplingOptions.last

    n = 20
    hinv_collector = parapint.linalg.iterative.pcg.LbfgsInvHessCollector(approx_options=lbfgs_approx_options, dim=n)
    n_iter = 10
    for pcg_iter in range(1, n_iter):
        s = pcg_iter * np.ones(n)
        y = pcg_iter * np.ones(n) * (1/3)
        hinv_collector.update(s, y)

    hinv_approx = hinv_collector.pop_inv_hessian_approx()

    assert np.allclose(hinv_approx.sk[-1, :], pcg_iter * np.ones(n))
    assert np.allclose(hinv_approx.yk[-1, :], pcg_iter * np.ones(n) * (1/3))
    assert np.allclose(hinv_approx.sk[-2, :], (pcg_iter - 1) * np.ones(n))
    assert np.allclose(hinv_approx.yk[-2, :], (pcg_iter - 1) * np.ones(n) * (1/3))
    assert np.allclose(hinv_approx.sk[-3, :], (pcg_iter - 2) * np.ones(n))
    assert np.allclose(hinv_approx.yk[-3, :], (pcg_iter - 2) * np.ones(n) * (1/3))


def test_hess_approx_collection_first():
    lbfgs_approx_options = parapint.linalg.iterative.pcg.LbfgsApproxOptions()
    lbfgs_approx_options.m = 3
    lbfgs_approx_options.sampling = parapint.linalg.iterative.pcg.LbfgsSamplingOptions.first

    n = 20
    hinv_collector = parapint.linalg.iterative.pcg.LbfgsInvHessCollector(approx_options=lbfgs_approx_options, dim=n)
    n_iter = 10
    for pcg_iter in range(1, n_iter):
        s = pcg_iter * np.ones(n)
        y = pcg_iter * np.ones(n) * (1/3)
        hinv_collector.update(s, y)

    hinv_approx = hinv_collector.pop_inv_hessian_approx()

    assert np.allclose(hinv_approx.sk[0, :], 1 * np.ones(n))
    assert np.allclose(hinv_approx.yk[0, :], 1 * np.ones(n) * (1/3))
    assert np.allclose(hinv_approx.sk[1, :], 2 * np.ones(n))
    assert np.allclose(hinv_approx.yk[1, :], 2 * np.ones(n) * (1/3))
    # last iterate is always collected
    assert np.allclose(hinv_approx.sk[-1, :], pcg_iter * np.ones(n))
    assert np.allclose(hinv_approx.yk[-1, :], pcg_iter * np.ones(n) * (1/3))


def test_hess_approx_collection_uniform():
    lbfgs_approx_options = parapint.linalg.iterative.pcg.LbfgsApproxOptions()
    lbfgs_approx_options.m = 11
    lbfgs_approx_options.sampling = parapint.linalg.iterative.pcg.LbfgsSamplingOptions.uniform

    target_cycle = 2
    exititer = ((lbfgs_approx_options.m - 1) - 1) * (2 ** target_cycle)
    n = 20
    hinv_collector = parapint.linalg.iterative.pcg.LbfgsInvHessCollector(approx_options=lbfgs_approx_options, dim=n)
    # iterate to include exititer calls to collect
    for pcg_iter in range(1, exititer + 1):
        s = pcg_iter * np.ones(n)
        y = pcg_iter * np.ones(n) * (1/3)
        hinv_collector.update(s, y)

    hinv_approx = hinv_collector.pop_inv_hessian_approx()

    # check uniform samples
    print(hinv_collector._k_indxs)
    assert np.allclose(np.array(hinv_collector._k_indxs), np.linspace(0, exititer, lbfgs_approx_options.m - 1, dtype=int)), 'Uniform samples not collected correctly'
    # check that correct samples were stored
    for i in range(lbfgs_approx_options.m - 1):
        assert np.allclose(hinv_approx.sk[i], (hinv_collector._k_indxs[i] + 1) * np.ones(n))
        assert np.allclose(hinv_approx.yk[i], (hinv_collector._k_indxs[i] + 1) * np.ones(n) * (1/3))
    # last iterate is always collected
    assert np.allclose(hinv_approx.sk[-1, :], pcg_iter * np.ones(n))
    assert np.allclose(hinv_approx.yk[-1, :], pcg_iter * np.ones(n) * (1/3))


def test_hess_approx_accuracy(test_case):
    if not test_case.expected_status == parapint.linalg.iterative.pcg.PcgSolutionStatus.successful:
        pytest.skip("Skipped test which is expected to fail")
    A = test_case.A
    rtol = 1e-8
    atol = 1e-8
    b = test_case.b
    x0 = 0 * b

    lbfgs_approx_options = parapint.linalg.iterative.pcg.LbfgsApproxOptions()
    lbfgs_approx_options.m = 11
    lbfgs_approx_options.sampling = parapint.linalg.iterative.pcg.LbfgsSamplingOptions.uniform

    pcg_options = parapint.linalg.PcgOptions()
    pcg_options.lbfgs_approx_options = lbfgs_approx_options
    pcg_options.rtol = rtol
    pcg_options.atol = atol

    results = parapint.linalg.iterative.pcg.pcg_solve(A=A, b=b, x0=x0, pcg_options=pcg_options)
    hess_approx = results.hess_approx.todense()

    true_inv = np.linalg.inv(A.toarray())

    assert np.allclose(hess_approx, true_inv, rtol=1e-2, atol=1e-2)


if __name__ == '__main__':
    # execute pytest for this file
    #pytest.main(['-v', __file__])
    test_hess_approx_collection_uniform()



