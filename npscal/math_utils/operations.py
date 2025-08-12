import numpy as np
from npscal.distarray import NPScal
from npscal.blacs_ctxt_management import DESCR_Register, CTXT_Register, BLACSDESCRManager
from npscal.index_utils.npscal_select import diagonal
from npscal.distarray import NPScal

def diag(array, k=0, descr_tag=None, ctxt_tag=None):

    if type(array) is NPScal:
        return diagonal(array)
        
    if type(array) is np.ndarray:
        if descr_tag is None:
            return np.diag(array, k=k)

        if len(np.shape(array)) == 1:
            m = np.size(array)
            n = np.size(array)
            
            descr = DESCR_Register.get_register(descr_tag)
            ctxt = CTXT_Register.get_register(ctxt_tag)
            new_diag_tag = f"newdiag_{ctxt}_{m}_{n}"
            if not(DESCR_Register.check_register(new_diag_tag)):
                descr_diag = BLACSDESCRManager(ctxt_tag, new_diag_tag, descr.lib, m, n, descr.mb, descr.nb, descr.rsrc, descr.csrc, descr.lld)
            else:
                descr_diag = DESCR_Register.get_register(new_diag_tag)

            new_zeros = descr_diag.alloc_zeros(np.float64)
            full_diag_dist = NPScal(loc_array=new_zeros, ctxt_tag=ctxt_tag, descr_tag=new_diag_tag, lib=descr.lib)

            for idx, i in enumerate(array):
                full_diag_dist[idx,idx] = i

            return full_diag_dist

def trace(array, **kwargs):

    if type(array) is NPScal:
        result = diag(array)
        result = np.sum(result)
        
    if type(array) is np.ndarray:
        result = np.trace(array, **kwargs)

    return result
        
def eig(array, vl, vu, b=None, left=False, right=True, overwrite_a=False, overwrite_b=False,
        check_finite=True, homogeneous_eigvals=False):

    if type(array) is NPScal:

        print("NOT IMPLEMENTED")
        
    if type(array) is np.ndarray:
        result = np.linalg.eig(array, b, left, right, overwrite_a, overwrite_b, check_finite,
                               homogeneous_eigvals)

        if left or right:
            return result[0], result[1]
        else:
            return result
    


