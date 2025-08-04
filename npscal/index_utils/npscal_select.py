from scalapack4py.array_types import nullable_ndpointer, ctypes2ndarray
from npscal.distarray import NPScal
from npscal.comms import mpi_bcast_float, mpi_bcast_integer
from npscal.blacs_ctxt_management import BLACSDESCRManager, CTXT_Register, DESCR_Register
import numpy as np

def select_single_val(npscal, val, bcast=True):

        # Only want one value wanted - give it back
        # And broadcast to all tasks
        idx1 = val[0]
        idx2 = val[1]
    
        lidx1 = npscal.lr2gr_map[idx1]
        lidx2 = npscal.lc2gc_map[idx2]

        if (lidx1 != -1 and lidx2 != -1):
            out = npscal.loc_array[lidx1, lidx2]
        else:
            out = 0

        root_r = npscal.prow_from_idx(idx1)
        root_c = npscal.pcol_from_idx(idx2)
        root = npscal.rank_from_rc(root_r, root_c)

        if bcast:
            out_val = mpi_bcast_float(out, root)
            return out_val
        else:
            return out

def select_slice(npscal, val):

    # Generates a new instance of NPScal with the
    # desired shape and a new distribution
    #print(val)
        
    if isinstance(val[0], int):
        newstart, newend = val[0], val[0]
        val[0] = slice(newstart, newend, None)
    if isinstance(val[1], int):
        newstart, newend = val[1], val[1]
        val[1] = slice(newstart, newend, None)

    if val[0].start is None:
        gl_row_start = 1
    else:
        gl_row_start = val[0].start + 1
    if val[1].start is None:
        gl_col_start = 1
    else:
        gl_col_start = val[1].start + 1

    if val[0].stop is None:
        gl_row_end = npscal.gl_m
    else:
        gl_row_end = val[0].stop
    if val[1].stop is None:
        gl_col_end = npscal.gl_n
    else:
        gl_col_end = val[1].stop

    new_m = gl_row_end - gl_row_start + 1
    new_n = gl_col_end - gl_col_start + 1

    #print(f"NEW M N: {new_m}, {new_n}")
    #print(f"glrow st end: {gl_row_start}, {gl_row_end}")
    #print(f"glcol st end: {gl_col_start}, {gl_col_end}")

    if new_m == 1 or new_n == 1:
        # We are just using a column or row here - in that case, just return an ndarray
        # Given that the contexts created here are temporary, we do not bother logging
        # them into the context manager
        new_ctx = npscal.sl.make_blacs_context(npscal.sl.get_system_context(npscal.ctxt.ctxt), 1, 1)
        descr_new = npscal.sl.make_blacs_desc(new_ctx, new_m, new_n)
        submatrix = descr_new.alloc_zeros(dtype=np.float64) if (descr_new.myrow==0 and descr_new.mycol==0) else None
        npscal.sl.pdgemr2d(new_m, new_n, npscal.loc_array, gl_row_start, gl_col_start, npscal.descr,
                           submatrix, 1, 1, descr_new, npscal.ctxt.ctxt)

        submatrix = npscal.comm.bcast([submatrix, np.float64], root=0)[0]

        if new_m == 1:
            submatrix = submatrix.reshape(new_n)
        if new_n == 1:
            submatrix = submatrix.reshape(new_m)

    else:
        # Return a new NPScal object with a new distribution
        new_descr_tag = f"matmul_{npscal.ctxt.tag}_{new_m}_{new_n}"
        if not(DESCR_Register.check_register(new_descr_tag)):
            descr_new = BLACSDESCRManager(npscal.ctxt.tag, new_descr_tag, npscal.sl, new_m, new_n)
        else:
            descr_new = DESCR_Register.get_register(new_descr_tag)
                
        submatrix = descr_new.alloc_zeros(dtype=np.float64)
        submatrix = NPScal(loc_array=submatrix, ctxt_tag=npscal.ctxt.tag, descr_tag=new_descr_tag, lib=npscal.sl)

        npscal.sl.pdgemr2d(new_m, new_n, npscal.loc_array, gl_row_start, gl_col_start, npscal.descr,
                           submatrix.loc_array, 1, 1, descr_new, npscal.ctxt.ctxt)
    return submatrix

def diagonal(npscal):
    # Already, the global selection syntax bears some fruit - we
    # no longer have to wrangle with local row/local column
    # indexing.

    diag = np.zeros(npscal.gl_n)
    for i in range(npscal.gl_n):
        diag[i] = npscal[i,i]

    return diag
                
                
