scalapack_libpath = "/home/gabrielbramley/Software/FHIaims/_build_embasi/libaims.250507.scalapack.mpi.so"

if __name__ == "__main__":

    from numpy.random import RandomState
    from scalapack4py import ScaLAPACK4py
    from npscal.distarray import NPScal
    from npscal.utils import find_squarest_grid
    
    from ctypes import RTLD_GLOBAL, POINTER, c_int, c_double, CDLL
    import numpy as np
    from mpi4py import MPI
    import os

    sl = ScaLAPACK4py(CDLL(scalapack_libpath, mode=RTLD_GLOBAL))

    ntasks = 4
    n = 500
    dtype=np.float64
    
    c = np.arange(n*n, dtype=dtype).reshape((n,n), order='F') if MPI.COMM_WORLD.rank==0 else None
    a = ((c.T + c) / 2.0).reshape((n,n), order='F') if MPI.COMM_WORLD.rank==0 else None
    a = np.asfortranarray(a.T) if MPI.COMM_WORLD.rank==0 else None

    MP, NP = find_squarest_grid(ntasks)
    ctx = sl.make_blacs_context(sl.get_default_system_context(), MP, NP)
    descr = sl.make_blacs_desc(ctx, n, n)

    b = np.zeros((descr.locrow, descr.loccol), dtype=dtype, order='F')
    sl.scatter_numpy(a, POINTER(c_int)(descr), b.ctypes.data_as(POINTER(c_double)), b.dtype)
    dist_array1 = NPScal(loc_array=b, descr=descr, lib=sl)

    dist_array1.set_mapping_array()
    print(dist_array1[0,0])

    print(a[0,0]) if MPI.COMM_WORLD.rank==0 else None
