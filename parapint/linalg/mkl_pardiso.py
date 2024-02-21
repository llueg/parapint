from pyomo.common.fileutils import find_library
from pyomo.contrib.pynumero.linalg.utils import validate_index, validate_value, _NotSet
import numpy.ctypeslib as npct
import numpy as np
import scipy.sparse as sp
import ctypes
import os

PT_INT_TYPE = ctypes.c_int64
INT_TYPE = ctypes.c_int32
NP_INT_TYPE = np.intc

class MKLPardisoInterface(object):
    libname = _NotSet

    @classmethod
    def available(cls):
        if cls.libname is _NotSet:
            cls.libname = find_library('mkl_rt', pathlist=['/home/llueg/software/intel/oneapi/mkl/2024.0/lib'])
        if cls.libname is None:
            return False
        return os.path.exists(cls.libname)

    def __init__(self):
        if not MKLPardisoInterface.available():
            raise RuntimeError('Could not find MKL library.')

        self._dim_cached = None

        self.lib = npct.load_library('libmkl_rt', '/home/llueg/software/intel/oneapi/mkl/2024.0/lib/')
        flags = ('F_CONTIGUOUS', 'ALIGNED')

        array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags=flags)
        array_1d_int = npct.ndpointer(dtype=NP_INT_TYPE, ndim=1, flags=flags)

        # Declare arg and res types of functions:
        self.lib.pardiso.restype = None
        self.lib.pardiso.argtypes = [npct.ndpointer(dtype=np.int64, ndim=1, flags=flags),   # pt
                                     ctypes.POINTER(INT_TYPE),      # maxfct
                                     ctypes.POINTER(INT_TYPE),      # mnum
                                     ctypes.POINTER(INT_TYPE),      # mtype
                                     ctypes.POINTER(INT_TYPE),      # phase
                                     ctypes.POINTER(INT_TYPE),      # n
                                     array_1d_double,          # a
                                     array_1d_int,      # ia
                                     array_1d_int,      # ja
                                     array_1d_int,      # perm
                                     ctypes.POINTER(INT_TYPE),      # nrhs
                                     array_1d_int,      # iparm
                                     ctypes.POINTER(INT_TYPE),      # msglvl
                                     array_1d_double,          # b
                                     array_1d_double,          # x
                                     ctypes.POINTER(INT_TYPE)]      # error
       
        self.lib.pardiso_export.restype = None
        self.lib.pardiso_export.argtypes = [npct.ndpointer(dtype=np.int64, ndim=1, flags=flags),   # pt
                                            array_1d_double,          # values
                                            array_1d_int,      # rows
                                            array_1d_int,      # cols
                                            ctypes.POINTER(INT_TYPE),      # step
                                            array_1d_int,      # iparm
                                            ctypes.POINTER(INT_TYPE)]      # error
       
        self.lib.pardisoinit.restype = None
        self.lib.pardisoinit.argtypes = [npct.ndpointer(dtype=np.int64, ndim=1, flags=flags),   # pt
                                        ctypes.POINTER(INT_TYPE),      # mtype
                                        array_1d_int,      # iparm
                                        ]
        
        # Parameters to pardiso fct
        self._pt = np.zeros(64, dtype=np.int64)
        self._iparm = np.zeros(64, dtype=NP_INT_TYPE)
        self._maxfct = 1
        self._mnum = 1
        self._mtype = -2 # Always assumes real symmetric indefinite matrix
        self._phase = -1
        self._dim = -1
        # CSR Matrix
        self._a = np.zeros(0, dtype=np.double)
        self._ia = np.zeros(0, dtype=NP_INT_TYPE)
        self._ja = np.zeros(0, dtype=NP_INT_TYPE)
        self._perm = np.zeros(0, dtype=NP_INT_TYPE)
        # Constant parameters
        self._nrhs = 1
        self._msglvl = 0
        # Variable input
        self._b = np.zeros(0, dtype=np.double)
        self._x = np.zeros(0, dtype=np.double)
        # Info
        self._error = 0

        # SC variables
        self._sc_dim: int = -1
        self._sc_nnz: int = -1
        self._sc_step: int = -1
        self._sc_a = np.zeros(0, dtype=np.double)
        self._sc_ia = np.zeros(0, dtype=NP_INT_TYPE)
        self._sc_ja = np.zeros(0, dtype=NP_INT_TYPE)
        self._sc = sp.csr_matrix(0)

        self.lib.pardisoinit(self._pt,
                             ctypes.byref(INT_TYPE(self._mtype)),
                             self._iparm)
        self._default_iparm = self._iparm.copy()

    def _set_iparm_default(self):
        iparm = self._default_iparm.copy()
        # iparm[26] = 1
        iparm[9] = 13
        #iparm[1] = 0
        iparm[10] = 1
        iparm[12] = 1
        #iparm[23] = 1
        #iparm[33] = 1
        #iparm[24] = 1
        self._iparm = iparm
    
    def _set_iparm_schur(self):
        iparm = self._default_iparm.copy()
        #iparm[0] = 1
        #iparm[1] = 0
        #iparm[8] = 10
        #iparm[7] = 8
        iparm[9] = 13
        #iparm[23] = 1
        #iparm[34] = 1
        iparm[30] = 0
        iparm[35] = 2
        #iparm[20] = 0
        
        iparm[10] = 1
        iparm[12] = 1
        #iparm[26] = 1
        
        self._iparm = iparm.copy()

    def _call_pardiso(self):
        assert self._a.size == self._ja.size == self._ia[-1] - 1, 'Dimension mismatch in CSR data and row arrays'
        self.lib.pardiso(
                            self._pt,
                            ctypes.byref(INT_TYPE(self._maxfct)),
                            ctypes.byref(INT_TYPE(self._mnum)),
                            ctypes.byref(INT_TYPE(self._mtype)),
                            ctypes.byref(INT_TYPE(self._phase)),
                            ctypes.byref(INT_TYPE(self._dim)),
                            self._a,
                            self._ia,
                            self._ja,
                            self._perm,
                            ctypes.byref(INT_TYPE(self._nrhs)),
                            self._iparm,
                            ctypes.byref(INT_TYPE(self._msglvl)),
                            self._b,
                            self._x,
                            ctypes.byref(INT_TYPE(self._error)),
                        )
        
    
    def _call_pardiso_export(self):
        self.lib.pardiso_export(self._pt,
                                self._sc_a,
                                self._sc_ia,
                                self._sc_ja,
                                ctypes.byref(INT_TYPE(self._sc_step)),
                                self._iparm,
                                ctypes.byref(INT_TYPE(self._error)))

    def __del__(self):
        if self._phase > 0:
            self._phase = -1
            self._call_pardiso()

    def set_iparm(self, i, val):
        validate_index(i, 64, 'iparm')
        validate_value(i, int, 'iparm')
        # NOTE: Use the FORTRAN indexing (documentation: https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2024-0/overview.html)
        # to set iparm values. Internally, the 0-based indexing is used.
        self._iparm[i - 1] = val

    def get_iparm(self, i):
        validate_index(i, 64, 'iparm')
        return self._iparm[i - 1]

    def do_symbolic_factorization(self, a, ia, ja):
        self._a = a.astype(np.double, casting='safe', copy=True)
        self._ia = ia.astype(NP_INT_TYPE, casting='safe', copy=True)
        self._ja = ja.astype(NP_INT_TYPE, casting='safe', copy=True)
        self._dim = ia.size - 1

        self._phase = 11
        self._set_iparm_default()

        self._call_pardiso()

        return self._error

    def do_numeric_factorization(self, a, ia, ja):
        a = a.astype(np.double, casting='safe', copy=True)
        ia = ia.astype(NP_INT_TYPE, casting='safe', copy=True)
        ja = ja.astype(NP_INT_TYPE, casting='safe', copy=True)

        dim = ia.size - 1
        assert dim == self._dim, (
            'Dimension mismatch between symbolic and numeric factorization.'
            'Please re-run symbolic factorization with the correct '
            'dimension.'
        )

        if self._phase < 11:
            raise RuntimeError('Symbolic factorization has not been performed.')
        else:
            if (self._ia != ia).any() or (self._ja != ja).any():
                raise RuntimeError('Symbolic factorization has not been performed on this matrix.')

        self._phase = 22
        self._a = a
        self._set_iparm_default()

        self._call_pardiso()
        
        return self._error


    def do_backsolve(self, rhs, copy=True):
        rhs = rhs.astype(np.double, casting='safe', copy=copy)
        dim = rhs.size
        assert (
            dim == self._dim
        ), 'Dimension mismatch in right hand side. Please correct.'

        if self._phase < 22:
            raise RuntimeError('Numeric factorization has not been performed.')

        self._phase = 33
        self._b = rhs
        self._x = np.zeros_like(self._b)
        self._set_iparm_default()

        self._call_pardiso()

        return self._x
    

    def do_symbolic_factorization_schur(self, a, ia, ja, dim_schur):
        # Assumes Schur Complement in lower right quadrant
        self._a = a.astype(np.double, casting='safe', copy=True)
        self._ia = ia.astype(NP_INT_TYPE, casting='safe', copy=True)
        self._ja = ja.astype(NP_INT_TYPE, casting='safe', copy=True)
        self._dim = ia.size - 1

        self._sc_dim = dim_schur
        self._perm = np.zeros(self._dim, dtype=NP_INT_TYPE)
        self._perm[-self._sc_dim:] = 1

        self._phase = 11

        # Set necesasary parameters
        self._set_iparm_schur()

        self._call_pardiso()

        #assert(self._iparm[35] >= 0)

        self._sc_nnz = self._iparm[35]
        # self._iparm[35] = -2 

        return self._error



    def do_numeric_factorization_schur(self, a, ia, ja):

        a = a.astype(np.double, casting='safe', copy=True)
        ia = ia.astype(NP_INT_TYPE, casting='safe', copy=True)
        ja = ja.astype(NP_INT_TYPE, casting='safe', copy=True)

        dim = ia.size - 1
        assert dim == self._dim, (
            'Dimension mismatch between symbolic and numeric factorization.'
            'Please re-run symbolic factorization with the correct '
            'dimension.'
        )

        if self._phase < 11:
            raise RuntimeError('Symbolic factorization has not been performed.')
        #else:
            # if (self._ia != ia).any() or (self._ja != ja).any():
            #     old_mat = sp.csr_matrix((self._a, self._ja, self._ia), shape=(self._dim, self._dim))
            #     new_mat = sp.csr_matrix((a, ja, ia), shape=(dim, dim))

            #     raise RuntimeError('Symbolic factorization has not been performed on this matrix.')

        self._sc_ia = np.zeros(self._sc_dim + 1, dtype=NP_INT_TYPE)
        self._sc_ja = np.zeros(self._sc_nnz, dtype=NP_INT_TYPE)
        self._sc_a = np.zeros(self._sc_nnz, dtype=np.double)

        self._sc_step = 1
        self._set_iparm_schur()
        #self._iparm[35] = -1

        #self._call_pardiso_export()

        self._phase = 22
        self._a = a
        self._ia = ia
        self._ja = ja
        self._x = np.zeros(self._sc_dim ** 2, dtype=np.double)
        self._call_pardiso()

        self._sc = self._x.copy().reshape(self._sc_dim, self._sc_dim)

        return self._error
    
    def do_backsolve_schur(self, rhs, copy=True):
        # After factorization of 
        # [A   B]
        # [B^T D],
        # This function solves A x = rhs
        # Followed steps from PIPS-IPM C interface, see
        # https://github.com/NCKempke/PIPS-IPMpp/blob/pipsipm/PIPS-IPM/Core/LinearSolvers/PardisoSolver/PardisoSchurSolver/PardisoMKLSchurSolver.C

        assert rhs.size == self._dim - self._sc_dim, (
            ' Dimension of rhs does not match diemnsion of upper left block in factorized matrix.'
        )

        
        # self._iparm[8] = 10
        # self._iparm[9] = 8
        # self._iparm[35] = 2
        self._set_iparm_schur()
        self._iparm[35] = 2
        self._iparm[7] = 0
        self._iparm[9] = 0

        self._b = np.zeros(self._dim, dtype=np.double)
        self._b[:self._dim - self._sc_dim] = rhs
        self._x = np.zeros_like(self._b)

        self._phase = 331

        self._call_pardiso()

        self._phase = 332
        self._b = self._x.copy()
        #self._x = np.zeros_like(self._b)

        self._call_pardiso()

        self._phase = 333
        self._b = self._x.copy()
        self._b[-self._sc_dim:] = 0
        #self._x = np.zeros_like(self._b)

        self._call_pardiso()

        # self._iparm[35] = -2

        res = self._x[:self._dim - self._sc_dim]
        return res






