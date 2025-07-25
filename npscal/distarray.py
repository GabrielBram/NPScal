from scalapack4py import ScaLAPACK4py
from ctypes import CDLL, POINTER, c_int, c_double, RTLD_GLOBAL
from mpi4py import MPI
from .comms import mpi_bcast_float
import numpy as np

class NPScal():

    def __init__(self, gl_array=None, loc_array=None, descr=None, lib=None):

        from scalapack4py import ScaLAPACK4py
        from ctypes import RTLD_GLOBAL, POINTER, c_int, c_double
        from mpi4py import MPI
        import os

        if ((gl_array is None) and (loc_array is None)):
            raise Exception("No input matrix specified: please specify a global or local array.")

        if ((gl_array is None) and (loc_array is None)):
            raise Exception("Only specify either a global or a local array.")

        if (descr is None) and (not (loc_array is None)):
            raise Exception("descr must be specified to create a distributed array type from a numpy array")

        if isinstance(lib, str):
            self.sl = ScaLAPACK4py(CDLL(libpath, mode=RTLD_GLOBAL))
        else:
            self.sl = lib
        
        self.comm = MPI.COMM_WORLD
        self.ntasks = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        
        if (loc_array is None):
            self.gl_array = gl_array.astype(dtype=np.float64, order='F')
            # Load to set context - descriptor to be replaced for specific array shape
            self.descr = descr
            self.ctx = descr.ctxt
            self.MP, self.NP = self.sl.blacs_gridinfo(self.ctx)[0], self.sl.blacs_gridinfo(self.ctx)[1]            
            self.descr = self.sl.make_blacs_desc(self.ctx, self.gl_m, self.gl_n)
            self.loc_array = self.scatter_to_local()

        if (not (loc_array is None)):
            self.gl_array = None
            self.loc_array = loc_array.astype(dtype=np.float64, order='F')
            self.descr = descr
            self.ctx = self.descr.ctxt
            self.MP, self.NP = self.sl.blacs_gridinfo(self.ctx)[0], self.sl.blacs_gridinfo(self.ctx)[1]

        self.transpose = False    

    @property
    def gl_array(self):
        return self._gl_array

    @gl_array.setter
    def gl_array(self, val):
        self._gl_array = val

    @property
    def gl_m(self):
        if self.gl_array is None:
            return self.descr.m
        else:
            return np.shape(self.gl_array)[0]

    @property
    def gl_n(self):
        if self.gl_array is None:
            return self.descr.n
        else:
            return np.shape(self.gl_array)[0]

    def set_mapping_array(self):
        lr2gr = self.gl_m * [-1]
        lc2gc = self.gl_n * [-1]

        mb = self.descr.mb
        nb = self.descr.nb
        nprow = self.descr.nprow
        npcol = self.descr.npcol
        myrow = self.descr.myrow
        mycol = self.descr.mycol

        self.lr2gr_map = self.gl_m * [-1]
        idx = 0
        for i in range(self.gl_m):
            if ( (i-1)/mb % nprow == myrow ):
               self.lr2gr_map[i] = idx
               idx = idx + 1

        self.lc2gc_map = self.gl_n * [-1]
        idx = 0
        for i in range(self.gl_n):
            if ( (i-1)/nb % npcol == mycol ):
               self.lc2gc_map[i] = idx
               idx = idx + 1
        
    @property
    def loc_array(self):
        return self._loc_array

    @loc_array.setter
    def loc_array(self, val):
        self._loc_array = val

    @property
    def loc_array_ptr(self):
        return self.loc_array.ctypes.data_as(POINTER(c_double))

    @property
    def descr(self):
        return self._descr

    @descr.setter
    def descr(self, val):
        self._descr = val

    def rank_from_rc(self, row, col):
        return row * self.descr.npcol + col

    def prow_from_idx(self, idx):
        return int((idx-1)/self.descr.mb % self.descr.nprow)

    def pcol_from_idx(self, idx):
        return int((idx-1)/self.descr.nb % self.descr.npcol)

    def __matmul__(self, val):
        return None

    def __getitem__(self, val):

        if len(val) != 2:
            raise Exception("You can only grab one value at a time at the moment :(")

        idx1 = val[0]
        idx2 = val[1]

        if (self.lc2gc_map[idx1] != -1 and self.lr2gr_map[idx2] != -1):
            out = self.loc_array[idx1, idx2]
        else:
            out = 0

        root_r = self.prow_from_idx(idx1)
        root_c = self.prow_from_idx(idx1)
        root = self.rank_from_rc(root_r, root_c)
        
        out = mpi_bcast_float(out, root)

        return out

    def __setitem__(self, val):
        return None

    def __add__(self, val):
        return self.loc_array + val.loc_array

    def __radd__(self, val):
        return val.loc_array + self.loc_array

    def __sub__(self, val):
        return self.loc_array - val.loc_array

    def __rsub__(self, val):
        return val.loc_array - self.loc_array

    def __matmul__(self, val):

        if self.transpose:
            transB = "T"
        else:
            transB = "N"
        if val.transpose:
            transA = "T"
        else:
            transA = "N"

        m = self.gl_m
        n = val.gl_n
        k = self.gl_n
        alpha = 1.0
        beta = 1.0
        newmat = SillyDistArray(gl_array=np.zeros((m, n)), descr=self.descr, lib=self.sl)
        desc_a = self.descr
        desc_b = val.descr
        desc_c = newmat.descr

        self.sl.pdgemm(transA, transB, m, n, k,
                       alpha, self.loc_array, 1, 1, desc_a,
                       val.loc_array, 1, 1, desc_b,
                       beta, newmat.loc_array, 1, 1, desc_c)

        return newmat
        
    def __rmatmul__(self, val):

        if self.transpose:
            transA = "T"
            self.transpose = False
        else:
            transA = "N"
        if val.transpose:
            transB = "T"
            val.transpose = False
        else:
            transB = "N"

        m = val.gl_m
        n = self.gl_n
        k = val.gl_n
        alpha = 1.0
        beta = 1.0
        newmat = SillyDistArray(gl_array=np.zeros((m, n)), descr=self.descr, lib=self.sl)
        desc_a = self.descr
        desc_b = val.descr
        desc_c = newmat.descr

        self.sl.pdgemm(transA, transB, m, n, k,
                       alpha, val.loc_array, 1, 1, desc_b,
                       self.loc_array, 1, 1, desc_a,
                       beta, newmat.loc_array, 1, 1, desc_c)

        return newmat

    def __mul__(self, val):
        return self.loc_array * val

    def __rmul__(self, val):
        return self.loc_array * val

    def __div__(self, val):
        return self.loc_array / val

    def __del__(self):
        return None

    def __str__(self):
        return str(self.loc_array)

    def gather_to_global(self):
        self.gl_array = self.sl.gather_numpy(POINTER(c_int)(self.descr), self.loc_array.ctypes.data_as(POINTER(c_double)), (self.gl_m, self.gl_n))

        return self.gl_array

    def scatter_to_local(self):
        gl_array = self.gl_array if self.rank==0 else None

        self.loc_array = np.zeros((self.descr.locrow, self.descr.loccol), dtype=np.float64, order='F')

        self.sl.scatter_numpy(gl_array, POINTER(c_int)(self.descr), self.loc_array.ctypes.data_as(POINTER(c_double)), self.loc_array.dtype)

        return self.loc_array
    
    @property
    def T(self):
        # Unliked numpy, where matrix transposition is pretty simple,
        # it represents a large communication bottleneck for ScaLAPACK
        # As we should only ever really use transposed matrices to perform
        # some kind of operation, we will use it to set a transposition
        # flag, which is passed to the routine and unset it once we're done.
        self.transpose = True
        return self
    

