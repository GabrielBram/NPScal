import numpy as np
from npscal.distarray import NPScal
from npscal.index_utils.npscal_select import diagonal

def diag(array, k=0):

    if type(array) is NPScal:
        result = diagonal(array)
        
    if type(array) is np.ndarray:
        result = np.diag(array, k=k)

    return result

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
    


