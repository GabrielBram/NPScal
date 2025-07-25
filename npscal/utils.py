from scalapack4py import ScaLAPACK4py
from ctypes import CDLL, POINTER, c_int, c_double, RTLD_GLOBAL
from mpi4py import MPI
import numpy as np

def find_squarest_grid(ntasks):

    if ntasks == 1:
        return 1, 1
    
    factors = []
    for factor in np.arange(1,int(ntasks/2)+1):
        if ntasks%factor == 0:
            factors.append( [int(factor),int(ntasks/factor)] )

    factors = np.array(factors)
    diffs = factors[:,0] - factors[:,1]

    nfactors = np.shape(factors)[0]

    if nfactors == 1:
        return factors[0,0], factors[0,1]
    else:
        idx = np.argwhere(diffs==np.min(np.abs(diffs)))[0][0]
        return factors[idx, 0], factors[idx, 1]
