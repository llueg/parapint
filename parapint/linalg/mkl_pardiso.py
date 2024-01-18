#  ___________________________________________________________________________
#
#  Pyomo: Python Optimization Modeling Objects
#  Copyright (c) 2008-2022
#  National Technology and Engineering Solutions of Sandia, LLC
#  Under the terms of Contract DE-NA0003525 with National Technology and
#  Engineering Solutions of Sandia, LLC, the U.S. Government retains certain
#  rights in this software.
#  This software is distributed under the 3-clause BSD License.
#  ___________________________________________________________________________
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
            cls.libname = find_library('mkl_rt')
        if cls.libname is None:
            return False
        return os.path.exists(cls.libname)

    def __init__(self):
        if not MKLPardisoInterface.available():
            raise RuntimeError('Could not find MKL library.')

        self._dim_cached = None

        self.lib = ctypes.cdll.LoadLibrary(self.libname)

        array_1d_double = npct.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
        array_2d_double = npct.ndpointer(dtype=np.double, ndim=2, flags='CONTIGUOUS')
        array_1d_int = npct.ndpointer(dtype=NP_INT_TYPE, ndim=1, flags='CONTIGUOUS')

        # Declare arg and res types of functions:
        # TODO: Consistent choice array_1d_int ot pointer(INT_TYPE), etc.
        #self.lib.pardiso.restype = ctypes.c_void_p
        self.lib.pardiso.restype = None
        self.lib.pardiso.argtypes = [ctypes.POINTER(PT_INT_TYPE),      # pt
                                     ctypes.POINTER(INT_TYPE),      # maxfct
                                     ctypes.POINTER(INT_TYPE),      # mnum
                                     ctypes.POINTER(INT_TYPE),      # mtype
                                     ctypes.POINTER(INT_TYPE),      # phase
                                     ctypes.POINTER(INT_TYPE),      # n
                                     ctypes.POINTER(None),                # a
                                     #array_1d_double,                     # a 
                                     ctypes.POINTER(INT_TYPE),      # ia
                                     ctypes.POINTER(INT_TYPE),      # ja
                                     ctypes.POINTER(INT_TYPE),      # perm
                                     ctypes.POINTER(INT_TYPE),      # nrhs
                                     ctypes.POINTER(INT_TYPE),      # iparm
                                     ctypes.POINTER(INT_TYPE),      # msglvl
                                     ctypes.POINTER(None),                # b
                                     #array_1d_double,                     # b
                                     ctypes.POINTER(None),                # x
                                    # array_1d_double,                     # x
                                     ctypes.POINTER(INT_TYPE)]      # error
        
        self.lib.pardiso_export.restype = None
        self.lib.pardiso_export.argtypes = [ctypes.POINTER(PT_INT_TYPE),   # pt
                                            ctypes.POINTER(None),          # values
                                            ctypes.POINTER(INT_TYPE),      # rows
                                            ctypes.POINTER(INT_TYPE),      # cols
                                            ctypes.POINTER(INT_TYPE),      # step
                                            ctypes.POINTER(INT_TYPE),      # iparm
                                            ctypes.POINTER(INT_TYPE)]      # error


        # Parameters to pardiso fct
        self._pt = np.zeros(64, dtype=PT_INT_TYPE)
        self._iparm = np.zeros(64, dtype=NP_INT_TYPE)
        # self._iparm[0] = 1
        # self._iparm[1] = 3
        # self._iparm[9] = 8
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
        self._msglvl = 1
        # Variable input
        self._b = np.zeros(0, dtype=np.double)
        self._x = np.zeros(0, dtype=np.double)
        # Info
        self._error = 0

        # Memory Limit
        self._mem_factor = None

        # SC variables
        self._sc_dim: int = -1
        self._sc_nnz: int = -1
        self._sc_step: int = -1
        self._sc_a = np.zeros(0, dtype=np.double)
        self._sc_ia = np.zeros(0, dtype=NP_INT_TYPE)
        self._sc_ja = np.zeros(0, dtype=NP_INT_TYPE)
        self._sc = sp.csr_matrix(0)

    def _call_pardiso(self):
        assert self._a.size == self._ja.size == self._ia[-1] - 1, 'Dimension mismatch in CSR data and row arrays'
        self.lib.pardiso(
                            self._pt.ctypes.data_as(ctypes.POINTER(PT_INT_TYPE)),
                            ctypes.byref(INT_TYPE(self._maxfct)),
                            ctypes.byref(INT_TYPE(self._mnum)),
                            ctypes.byref(INT_TYPE(self._mtype)),
                            ctypes.byref(INT_TYPE(self._phase)),
                            ctypes.byref(INT_TYPE(self._dim)),
                            self._a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            #self._a,
                            self._ia.ctypes.data_as(ctypes.POINTER(INT_TYPE)),
                            self._ja.ctypes.data_as(ctypes.POINTER(INT_TYPE)),
                            self._perm.ctypes.data_as(ctypes.POINTER(INT_TYPE)),
                            ctypes.byref(INT_TYPE(self._nrhs)),
                            self._iparm.ctypes.data_as(ctypes.POINTER(INT_TYPE)),
                            ctypes.byref(INT_TYPE(self._msglvl)),
                            self._b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            self._x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                            ctypes.byref(INT_TYPE(self._error)),
                        )
        
    
    def _call_pardiso_export(self):
        self.lib.pardiso_export(self._pt.ctypes.data_as(ctypes.POINTER(PT_INT_TYPE)),
                                self._sc_a.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                                self._sc_ia.ctypes.data_as(ctypes.POINTER(INT_TYPE)),
                                self._sc_ja.ctypes.data_as(ctypes.POINTER(INT_TYPE)),
                                ctypes.byref(INT_TYPE(self._sc_step)),
                                self._iparm.ctypes.data_as(ctypes.POINTER(INT_TYPE)),
                                ctypes.byref(INT_TYPE(self._error)))

    def __del__(self):
        if self._phase > 0:
            self._phase = -1
            self._call_pardiso()

    def set_iparm(self, i, val):
        validate_index(i, 64, 'iparm')
        validate_value(i, int, 'iparm')
        # NOTE: Use the FORTRAN indexing (same as documentation) to
        # set and access info/cntl arrays from Python, whereas C
        # functions use C indexing. Maybe this is too confusing.
        self._iparm[i - 1] = val
        # TODO: Might be too defesive
        #if val != 0: self._iparm[0] = 1

    def get_iparm(self, i):
        validate_index(i, 64, 'iparm')
        return self._iparm[i - 1]

    def do_symbolic_factorization(self, a, ia, ja):
        self._a = a.astype(np.double, casting='safe', copy=True)
        self._ia = ia.astype(NP_INT_TYPE, casting='safe', copy=True)
        self._ja = ja.astype(NP_INT_TYPE, casting='safe', copy=True)
        self._dim = ia.size - 1

        self._phase = 11

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

        self._call_pardiso()
        #return npct.as_array(b, shape=(dim,))
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
        self._iparm[0] = 1
        self._iparm[35] = -2 # Compute Schur
        # Other params set with Schur - taken from PIPS-IPM C interface
        self._iparm[10] = 0
        self._iparm[12] = 0
        self._iparm[23] = 1


        self._call_pardiso()

        assert(self._iparm[35] >= 0)

        self._sc_nnz = self._iparm[35]

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
        else:
            if (self._ia != ia).any() or (self._ja != ja).any():
                raise RuntimeError('Symbolic factorization has not been performed on this matrix.')

        self._sc_ia = np.zeros(self._sc_dim + 1, dtype=NP_INT_TYPE)
        self._sc_ja = np.zeros(self._sc_nnz, dtype=NP_INT_TYPE)
        self._sc_a = np.zeros(self._sc_nnz, dtype=np.double)

        self._iparm[35] = -2
        self._sc_step = 1

        self._call_pardiso_export()
        print('export done')


        self._phase = 22
        self._a = a
        self._call_pardiso()

        self._sc = sp.csr_matrix((self._sc_a, self._sc_ja, self._sc_ia), shape=(self._sc_dim, self._sc_dim))

        return self._error
    
    def _do_backsolve_schur(self, rhs, copy=True):
        #    // solving phase
        # /* pardiso from mkl does not support same functionality as pardiso-project
        #     *
        #     * pardiso project:
        #     * when computing the schur complement S with factorization matrices we will get
        #     *
        #     * [A11 A12]   [L11 0] [I 0] [U11 U12]
        #     * [A21 A22] = [L12 I] [0 S] [0     I]
        #     *
        #     * a subsequent solve call will then only solve for A11 x1 = b1 instead of the full
        #     * system.
        #     *
        #     * pardiso mkl:
        #     * while the schur complement is the same, the factorization computed, stored and
        #     * used for solve calls is a full factorization. thus pardiso from intel will always
        #     * solve the full system
        #     *
        #     * workaround is to solve
        #     *
        #     * (phase 331)
        #     * [L11   0] [z1] = [b1]
        #     * [L12   I] [z2] = [b2]
        #     *
        #     * (phase 332)
        #     * [I 0] [y1]   [z1]
        #     * [0 S] [y2] = [z2]
        #     *
        #     * (phase 333)
        #     * [U11 U12] [x1]   [y1]
        #     * [0     I] [x2] = [0]
        #     *
        #     */
        # double* z_n = new double[nvec_size];
        # assert(iparm[7] == 0);
        # assert(iparm[35] = -2);

        # // this is necessary for usage of stage = 331/332/333
        # iparm[9] = 0;

        # // HACK: keeping iparm[35] = -2 will, for some reason, not compute the correct result
        # // iparm[35] will be set to -2 after stage 333
        # iparm[35] = 2;

        # phase = 331;
        # pardiso(pt, &maxfct, &mnum, &mtype, &phase, &n, eltsAug, rowptrAug, colidxAug, nullptr, &nrhs, iparm, &msglvl, rhs_n, z_n, &error);
        # assert(error == 0);

        assert rhs.size == self._dim - self._sc_dim

        self._iparm[7] = 0
        self._iparm[9] = 0
        self._iparm[35] = 2

        self._b = np.zeros(self._dim, dtype=np.double)
        self._b[:self._dim - self._sc_dim] = rhs
        self._x = np.zeros_like(self._b)

        self._phase = 331

        self._call_pardiso()

        self._phase = 332
        self._b = self._x.copy()
        self._x = np.zeros_like(self._b)

        self._call_pardiso()

        self._phase = 333
        self._b = self._x.copy()
        self._b[-self._sc_dim:] = 0
        self._x = np.zeros_like(self._b)

        self._call_pardiso()

        self._iparm[35] = -2

        res = self._x[:self._dim - self._sc_dim]
        return res






