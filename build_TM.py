import h5py, argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

# TESTING TM BUILDER, NOT THE MAIN FILE, THIS WONT BE A SCRIPT AT THE END
# Currently can build the matrix, find eigvals, eigvectors
# Next step is to fix the code ofc. 

# Parsing time
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", dest="input_file", default="elec_NOHI_assign.h5",
                  help="Input assign.h5 file, output from w_assign") 
parser.add_argument("-w", "--westfile", dest="west_file", default="elec_NOHI.h5",
                  help="West.h5 file of the simulation for the probabilities") 
parser.add_argument("-fi", "--first_iter", dest="first_iter", default=1, type=np.int,
                  help="first iter to calculate the beta values") 
parser.add_argument("-li", "--last_iter", dest="last_iter", default=None, type=np.int,
                  help="last iter to calculate the beta values") 
#parser.add_argument("-e", "--extn", dest="extension", default='png', 
#                  help="Extension to use with matplotlib output") 
#parser.add_argument("-ti", "--title", dest="title", default="K_on vs Q",
#                  help="title of the plot and the files, don't add the extension") 
args = parser.parse_args()

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

def build_raw_matrix(first_iter, last_iter, westH5, assignments, init_matrix):
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
    return rawMatrix[...,mask][mask,...]

def normalize_matrix(shavedMatrix):
    for irow, row in enumerate(shavedMatrix):
        total = row.sum()
        if total != 0.0:
            shavedMatrix[irow] = row/total
    return shavedMatrix

if __name__ == '__main__':
    # The file tree of the assignment file 
    # ['assignments',
    # 'bin_labels',
    # 'labeled_populations',
    # 'npts',
    # 'nsegs',
    # 'state_labels',
    # 'state_map',
    # 'trajlabels']
    ########################################################## 
    westH5     = H5File(args.west_file)
    first_iter = args.first_iter
    if args.last_iter: 
        last_iter = args.last_iter
    else:
        last_iter  = westH5.get_file().attrs['west_current_iteration'] 
    # Main assignment file
    assign_h5   = h5py.File(args.input_file)
    # Bin labels will be the node labels here, as well as the 
    # dimension for the TM
    bin_labels  = assign_h5['bin_labels']
    label_ind   = {}
    dim         = bin_labels.shape[0]
    tMatrix     = np.zeros((dim,dim))
    # To build the matrix we need 1) dimensions 2) assignments
    assignments = assign_h5['assignments']
    tMatrix     = build_raw_matrix(first_iter, last_iter, westH5, assignments, tMatrix)
    tMatrix     = shave_matrix(tMatrix)
    tMatrix     = normalize_matrix(tMatrix)
    evals, evecs = eig(tMatrix)
    print evals, evecs
