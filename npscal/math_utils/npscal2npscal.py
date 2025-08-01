from npscal.distarray import NPScal
from npscal.blacs_ctxt_management import DESCR_Register, BLACSDESCRManager
import numpy as np

def matmul(self_mat, inp_mat):
    if inp_mat.transpose:
        transB = "T"
    else:
        transB = "N"
    if self_mat.transpose:
        transA = "T"
    else:
        transA = "N"

    m = self_mat.gl_m
    n = inp_mat.gl_n
    k = self_mat.gl_n
    alpha = 1.0
    beta = 1.0

    ctxt_tag = self_mat.ctxt.tag
    
    desc_a = self_mat.descr
    desc_b = inp_mat.descr
    if (m == n and n == k):
        # No need for new descriptor, just use the descriptor of A in A @ B
        desc_c = self_mat.descr
        desc_c_tag = self_mat.descr.tag

    new_loc_array = desc_c.alloc_zeros(dtype=np.float64)

    newmat = NPScal(loc_array=new_loc_array, ctxt_tag=ctxt_tag, descr_tag=desc_c_tag, lib=inp_mat.sl)

    self_mat.sl.pdgemm(transA, transB, m, n, k,
                   alpha, self_mat.loc_array, 1, 1, desc_a,
                   inp_mat.loc_array, 1, 1, desc_b,
                   beta, newmat.loc_array, 1, 1, desc_c)

    return newmat

def rmatmul(self_mat, inp_mat):
        if self_mat.transpose:
            transB = "T"
        else:
            transB = "N"
        if inp_mat.transpose:
            transA = "T"
        else:
            transA = "N"

        m = inp_mat.gl_m
        n = self_mat.gl_n
        k = inp_mat.gl_n
        alpha = 1.0
        beta = 1.0
        newmat = NPScal(gl_array=np.zeros((m, n)), descr=self_mat.descr, lib=self_mat.sl)
        desc_a = self_mat.descr
        desc_b = inp_mat.descr
        desc_c = newmat.descr
        
        self_mat.sl.pdgemm(transA, transB, m, n, k,
                       alpha, inp_mat.loc_array, 1, 1, desc_b,
                       self_mat.loc_array, 1, 1, desc_a,
                       beta, newmat.loc_array, 1, 1, desc_c)

        return newmat
