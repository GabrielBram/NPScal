from scalapack4py import ScaLAPACK4py
from scalapack4py.array_types import nullable_ndpointer, ctypes2ndarray
from ctypes import CDLL, POINTER, c_int, c_double, RTLD_GLOBAL
import ctypes
from mpi4py import MPI
from .comms import mpi_bcast_float, root_print
from .blacs_ctxt_management import CTXT_Register, DESCR_Register
from typing import overload
import numpy as np

class NPScal():

    def __init__(self, gl_array=None, loc_array=None, lib=None, ctxt_tag=None, descr_tag=None):

        from scalapack4py import ScaLAPACK4py
        from ctypes import RTLD_GLOBAL, POINTER, c_int, c_double
        from mpi4py import MPI
        import os

        if ((gl_array is None) and (loc_array is None)):
            raise Exception("No input matrix specified: please specify a global or local array.")

        if ((gl_array is None) and (loc_array is None)):
            raise Exception("Only specify either a global or a local array.")

        if isinstance(lib, str):
            self.sl = ScaLAPACK4py(CDLL(libpath, mode=RTLD_GLOBAL))
        else:
            self.sl = lib

        self.ctxt = CTXT_Register.get_register(ctxt_tag)
        self.descr = DESCR_Register.get_register(descr_tag)

        if isinstance(loc_array, ctypes._Pointer):
            loc_array = ctypes2ndarray(loc_array, (self.descr.locrow, self.descr.loccol)).T
        
        self.comm = MPI.COMM_WORLD
        self.ntasks = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        
        if (loc_array is None):
            self.gl_array = gl_array.astype(dtype=np.float64, order='F')
            self.loc_array = self.scatter_to_local()

        if (not (loc_array is None)):
            self.gl_array = None
            self.loc_array = loc_array.astype(dtype=np.float64, order='F')

        self.set_mapping_array()
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
            if ( int(i/mb % nprow) == myrow ):
               self.lr2gr_map[i] = idx
               idx = idx + 1

        self.lc2gc_map = self.gl_n * [-1]
        idx = 0
        for i in range(self.gl_n):
            if ( int(i/nb % npcol) == mycol ):
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
        return int(idx/self.descr.mb % self.descr.nprow)

    def pcol_from_idx(self, idx):
        return int(idx/self.descr.nb % self.descr.npcol)

    def __getitem__(self, val):
        from npscal.index_utils.npscal_select import select_single_val 
        from npscal.index_utils.npscal_select import select_slice

        if len(val) != 2:
            raise Exception("Please use two indices.")

        if (isinstance(val[0], int) and isinstance(val[1], int)):
            return select_single_val(self, val)

        if (isinstance(val[0], slice) or isinstance(val[1], slice)):
            return select_slice(self, val)

    def __setitem__(self, val, idx: set, val_set: int | float | np.float64) -> None:
        
        if len(idx) != 2:
            raise Exception("Please use two indices.")

        idx1 = idx[0]
        idx2 = idx[1]
        
        if (isinstance(idx1, int) and isinstance(idx2, int)):
            
            lidx1 = npscal.lc2gc_map[idx1]
            lidx2 = npscal.lr2gr_map[idx2]

            if (lidx1 != -1 and lidx2 != -1):
                npscal.loc_array[lidx1, lidx2] = val_set

        if (isinstance(idx1, slice) or isinstance(idx2, slice)):        
            return None
        
    def __add__(self, val):
        new_loc_array = self.loc_array + val.loc_array

        new_descr_tag = self.descr.tag
        new_ctxt_tag = self.ctxt.tag

        add_result = NPScal(loc_array=self.loc_array, ctxt_tag=new_ctxt_tag, descr_tag=new_descr_tag)
        
        return add_result

    def __radd__(self, val):
        return val.loc_array + self.loc_array

    def __sub__(self, val):
        new_loc_array = self.loc_array - val.loc_array

        new_descr_tag = self.descr.tag
        new_ctxt_tag = self.ctxt.tag

        add_result = NPScal(loc_array=self.loc_array, ctxt_tag=new_ctxt_tag, descr_tag=new_descr_tag)
        
        return add_result

    def __rsub__(self, val):
        new_loc_array = val.loc_array - self.loc_array

        new_descr_tag = self.descr.tag
        new_ctxt_tag = self.ctxt.tag

        sub_result = NPScal(loc_array=self.loc_array, ctxt_tag=new_ctxt_tag, descr_tag=new_descr_tag)
        
        return sub_result

    def __matmul__(self, inp_mat):
        from npscal.math_utils.npscal2npscal import matmul
        newmat = matmul(self, inp_mat)
        self.transpose = False
        inp_mat.transpose = False
        return newmat

    def __rmatmul__(self, inp_mat):
        from npscal.math_utils.npscal2npscal import rmatmul
        newmat = rmatmul(self, inp_mat)
        self.transpose = False
        inp_mat.transpose = False
        return newmat

    def __mul__(self, val):
        self.loc_array = self.loc_array * val
        return self

    def __rmul__(self, val):
        self.loc_array = val * self.loc_array
        return self

    def __div__(self, val):
        self.loc_array = self.loc_array / val
        return self

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

    def copy(self):
        import copy

        return copy.copy(self)

