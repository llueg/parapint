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



if __name__ == '__main__':
    # execute pytest for this file
    pytest.main(['-v', __file__])



