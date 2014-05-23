import h5py 
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

def build_raw_matrix(first_iter, last_iter, westH5, assignments, init_matrix):
    dim = init_matrix.shape[0]
    for iiter in range(first_iter-1, last_iter):
        print "Calculating iteration %i"%(iiter)
        iter_obj = westH5.openFile['iterations']['iter_%08d'%(iiter+1)]
        weights  = iter_obj['seg_index']['weight'][...]
        for iwalk, walkerObj in enumerate(assignments[iiter]):
            for ipoint, point in enumerate(walkerObj):
                if ipoint == 0:
                    prev_point = point
                if ipoint > 0 and point < dim:
                    # Here it's now a transition between prev_point -> point
                    #print prev_point, point, iwalk
                    init_matrix[prev_point][point] += weights[iwalk]
                if point == dim:
                    break
    print init_matrix
    return init_matrix

def shave_matrix(rawMatrix):
    mask = np.ones(rawMatrix.shape[0], dtype=bool)
    for irow,row in enumerate(rawMatrix):
        if rawMatrix[irow,:].sum() == 0.0 and rawMatrix[:,irow].sum() == 0.0:
            mask[irow] = False
    return rawMatrix[...,mask][mask,...], mask

def normalize_matrix(shavedMatrix):
    for irow, row in enumerate(shavedMatrix):
        total = row.sum()
        if total != 0.0:
            shavedMatrix[irow] = row/total
    return shavedMatrix

def TM_Builder(westH5, assignH5, first_iter, last_iter):
    bin_labels    = assignH5['bin_labels']
    dim           = bin_labels.shape[0]
    tMatrix       = np.zeros((dim,dim))
    # To build the matrix we need 1) dimensions 2) assignments
    assignments   = assignH5['assignments']
    tMatrix       = build_raw_matrix(first_iter, last_iter, westH5, assignments, tMatrix)
    tMatrix,mask  = shave_matrix(tMatrix)
    bin_labels    = bin_labels[mask]
    tMatrix       = normalize_matrix(tMatrix)
    evals, evecs  = eig(tMatrix)
    return tMatrix, evals, evecs, bin_labels

class Singleton(type):
    '''A singleton implementation for h5 & assignment files,
    not necessary if you don't care about paralellization'''
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

class H5File:
    '''Thin wrapper around the h5file'''
    __metaclass__ = Singleton
    def __init__(self, h5file): 
        self.openFile = h5py.File(h5file)
        self.locked   = False

    def get_file(self):
        return self.openFile

def build_DiGraph(TM, eigvec, labels):
    attrs = {}
    for ilabel, label in enumerate(labels):
        attrs[label] = eigvec[ilabel]
    DG = nx.DiGraph()
    DG.add_nodes_from(labels, attr_dict=attrs)
    # Now add edges with weights
    ilabels = range(len(labels))
    DG.add_weighted_edges_from([(labels[iu],labels[iv],TM[iu,iv]) for iu in ilabels for iv in ilabels], weight='t_prob')
    return DG
