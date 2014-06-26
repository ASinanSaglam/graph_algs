from __future__ import division
from cpython cimport bool 
import numpy as np 
cimport numpy as np
cimport cython

np.import_array()

DTYPE = np.float64
DTYPE_2 = np.uint16
DTYPE_3 = np.uint8
ctypedef np.float64_t DTYPE_t
ctypedef np.uint16_t DTYPE2_t
ctypedef np.uint8_t DTYPE3_t

@cython.boundscheck(False)
#@cython.nonecheck(False)
def addIterToMatrix(np.ndarray[DTYPE2_t, ndim=2, cast=True] assignmentsIter, np.ndarray[DTYPE_t, ndim=1] weights, \
                    np.ndarray[DTYPE_t, ndim=2] inMatrix, int dim):
    cdef:
        int iwalk, ipoint
        DTYPE2_t prev_point, point 
        np.ndarray[DTYPE2_t, ndim=1] walkObj
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

@cython.boundscheck(False)
def shaveMatrix(np.ndarray[DTYPE_t, ndim=2] rawMatrix, np.ndarray[DTYPE3_t, ndim=1, cast=True] mask):
    cdef: 
        np.ndarray[DTYPE_t, ndim=1] row
        int irow
    for irow,row in enumerate(rawMatrix):
        if rawMatrix[irow,:].sum() == 0.0 and rawMatrix[:,irow].sum() == 0.0:
            mask[irow] = False
    return rawMatrix[...,mask][mask,...], mask

@cython.boundscheck(False)
def normalizeMatrix(np.ndarray[DTYPE_t, ndim=2] shavedMatrix):
    cdef:
        int irow
        np.ndarray[DTYPE_t, ndim=1] row
        DTYPE_t total
    for irow, row in enumerate(shavedMatrix):
        total = row.sum()
        if total != 0.0:
            shavedMatrix[irow] = row/total
    return shavedMatrix
