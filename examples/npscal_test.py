scalapack_libpath = "/home/gabrielbramley/Software/FHIaims/_build_modifyH_lib/libaims.250403.scalapack.mpi.so"

if __name__ == "__main__":

    from numpy.random import RandomState
    from scalapack4py import ScaLAPACK4py
    from npscal.distarray import NPScal
    from npscal.utils import find_squarest_grid
    from npscal.blacs_ctxt_management import BLACSContextManager, BLACSDESCRManager, CTXT_Register, DESCR_Register
    
    from ctypes import RTLD_GLOBAL, POINTER, c_int, c_double, CDLL
    import numpy as np
    from mpi4py import MPI
    import os

    sl = ScaLAPACK4py(CDLL(scalapack_libpath, mode=RTLD_GLOBAL))

    ntasks = 4
    n = 500
    dtype=np.float64
    
    c = np.arange(n*n, dtype=dtype).reshape((n,n), order='F') if MPI.COMM_WORLD.rank==0 else None
    #a = ((c.T + c) / 2.0).reshape((n,n), order='F') if MPI.COMM_WORLD.rank==0 else None
    a = np.asfortranarray(c.T) if MPI.COMM_WORLD.rank==0 else None

    MP, NP = find_squarest_grid(ntasks)
    BLACSContextManager("main", MP, NP, sl)
    BLACSDESCRManager("main", "default", sl, n, n)

    ctxt = CTXT_Register.get_register("main")
    descr = DESCR_Register.get_register("default")

    b = np.zeros((descr.locrow, descr.loccol), dtype=dtype, order='F')
    sl.scatter_numpy(a, POINTER(c_int)(descr), b.ctypes.data_as(POINTER(c_double)), b.dtype)
    dist_array1 = NPScal(loc_array=b, ctxt_tag="main", descr_tag="default", lib=sl)

    dist_array1.set_mapping_array()
    dist_array = dist_array1[0:25,0:25]
    dist_array.set_mapping_array()
    
    from npscal.index_utils.npscal_select import diag
    diagonal = diag(dist_array)
    print(diagonal)
