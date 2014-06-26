from __future__ import division
import numpy as np 
cimport numpy as np
cimport cython

np.import_array()

DTYPE = np.float64
DTYPE_2 = np.uint8
ctypedef np.float64_t DTYPE_t
ctypedef np.uint8_t DTYPE2_t

@cython.boundscheck(False)
#@cython.nonecheck(False)
def addIterToMatrix(np.ndarray[DTYPE2_t, ndim=2] assignmentsIter, np.ndarray[DTYPE_t, ndim=1] weights, \
                    np.ndarray[DTYPE_t, ndim=2] inMatrix, int dim):
    cdef:
        int iwalk, ipoint
        int prev_point, point
    for iwalk, walkObj in enumerate(assignmentsIter):
        for ipoint, point in enumerate(walkObj):
            if ipoint == 0:
                prev_point = point
            if ipoint > 0 and point < dim:
                # Here it's now a transition between prev_point -> point
                #print prev_point, point, iwalk
                inMatrix[prev_point][point] += weights[iwalk]
            if point == dim:
                break
    return inMatrix

#def shave_matrix(np.ndarray[DTYPE_t, ndim=2] rawMatrix):
    #cdef ndarray[bool, ndim=1] mask
    #mask = np.ones(rawMatrix.shape[0], dtype=bool)
    #for irow,row in enumerate(rawMatrix):
        #if rawMatrix[irow,:].sum() == 0.0 and rawMatrix[:,irow].sum() == 0.0:
            #mask[irow] = False
    #return rawMatrix[...,mask][mask,...], mask

#def normalize_matrix(shavedMatrix):
    #for irow, row in enumerate(shavedMatrix):
        #total = row.sum()
        #if total != 0.0:
            #shavedMatrix[irow] = row/total
    #return shavedMatrix
#
#def TM_Builder(westH5, assignH5, first_iter, last_iter):
    #bin_labels    = assignH5['bin_labels']
    #dim           = bin_labels.shape[0]
    #tMatrix       = np.zeros((dim,dim))
    ## To build the matrix we need 1) dimensions 2) assignments
    #assignments   = assignH5['assignments']
    #tMatrix       = build_raw_matrix(first_iter, last_iter, westH5, assignments, tMatrix)
    #tMatrix,mask  = shave_matrix(tMatrix)
    #bin_labels    = bin_labels[mask]
    #tMatrix       = normalize_matrix(tMatrix)
    #evals, evecs  = eig(tMatrix)
    #return tMatrix, evals, evecs, bin_labels
