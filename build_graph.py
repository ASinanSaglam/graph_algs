import build_TM
import h5py, argparse
import numpy as np
import networkx as nx
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
    Builder = TM_Builder(westH5, assign_h5, first_iter, last_iter)
    TM, eigval, eigvec, labels = Builder.run()
    norm_eigvec = eigvec[:,0]/eigvec[:,0].sum()
    DG = DG_Builder(TM, norm_eigvec, labels)
    nx.draw(DG)
    plt.show()
