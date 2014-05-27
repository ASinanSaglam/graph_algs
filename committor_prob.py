import build_TM
import h5py, argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import eig
from numpy.linalg import solve
import sys
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

if __name__ == '__main__':
    # Let's parse all the neccessary arguments
    H5File     = build_TM.H5File
    TM_Builder = build_TM.TM_Builder 
    DG_Builder = build_TM.build_DiGraph
    westH5     = H5File(args.west_file)
    first_iter = args.first_iter
    if args.last_iter: 
        last_iter = args.last_iter
    else:
        last_iter  = westH5.get_file().attrs['west_current_iteration'] 
    assign_h5   = h5py.File(args.input_file)
    assignments = assign_h5['assignments']
    # Build the matrix, find eigsystem
    TM, eigval, eigvec, labels = TM_Builder(westH5, assign_h5, first_iter, last_iter)
    # First thing, we need some target states, for now let's say 0 and -1 are the tstates
    itstates = [0,-1]
    # Initialize a new matrix
    AbsorbTM = TM[:,:]
    # Set all tstate rows to be zeros
    for index in itstates: 
        AbsorbTM[index] = np.zeros(TM.shape[0])
    # A dictionary for the commitor probs
    CommProbs = {}
    # We need to calculate u(i,tstate) for both tstates so let's loop
    for index in itstates: 
        # Create array in dict for comm probs
        # Need to get the minor of the tstates
        mask        = np.ones(TM.shape[0], dtype=bool)
        mask[index] = np.int(0)
        MinorTM     = TM[...,mask][mask,...]
        CommProbs[index] = np.zeros(MinorTM.shape[0])
        # Now subtract I 
        SubMinTM    = MinorTM - np.identity(MinorTM.shape[0])
        # Now find the vector on the R.H.S.
        RHS         = -1 * (AbsorbTM[:,index][mask])
        solution    = solve(SubMinTM, RHS) 
        CommProbs[index] = solution[:]
    for value in CommProbs.iteritems():
        print value
    print CommProbs[0][:-1] + CommProbs[-1][1:]
    print CommProbs[0][:-1] - CommProbs[-1][1:]
