
'''This script is designed to return the "highest flux path" from one state in a graph to another state, where a "state" is defined as a collection of nodes.  More specifically, it finds the continuous path through which the flow rate from the first state to the second is the greatest.  In our case, we use it to find the "most common" path from the X-ray crystal structure of the villin domain to its NMR structure.  Again, "most common" is taken to mean the single continuous path most likely to be taken from any part of the "X-ray structure state" to the "NMR structure state."  In other words, it should be used to find an "important" transition state.  

To do so, this script must be passed: (1) a transition matrix, (2) the normalized stationary eigenvector for that transition matrix,(3) an array of starting node indices, and (4) an array of ending node indices.

This script then solves for the least cost path between every node in the starting state and every node in the ending state.  The cost function is implemented through a compounded product of the rate (k) values between nodes in the pathway, all multiplied by the equilibrium population of the starting node (giving flux out of that node and to the destination node).  A higher product is lower cost.  This script uses an implementation of Dijkstra's algorithm.'''




cimport cython
import numpy as np
cimport numpy as np
ctypedef np.float_t DTYPE_F
ctypedef np.float64_t DTYPE_F64
ctypedef np.int_t DTYPE_INT
#DTYPE_B = np.uint8_t
from libcpp cimport bool


def hfp(np.ndarray[DTYPE_F, ndim=2] TM, np.ndarray[DTYPE_F64, ndim=1, cast=True] eigvec, np.ndarray[DTYPE_INT, ndim=1, cast=True] starting_node_indices, np.ndarray[DTYPE_INT, ndim=1, cast=True] ending_node_indices):


    # Initialize variables:
    cdef array_len = eigvec.shape[0]
    cdef np.ndarray[DTYPE_F, ndim=1] cost_vector = np.zeros(array_len, dtype=np.float_)
    cdef np.ndarray[np.uint8_t, ndim=1, cast=True] visited 
    visited = np.zeros(eigvec.shape[0],dtype=np.uint8)
    cdef np.uint8_t done = False
    cdef np.uint16_t cnindex = 0
    cdef np.uint16_t old_node_index = 0 
    cdef np.uint16_t number_of_starting_nodes_finished = 0
    cdef np.uint16_t i = 0
    cdef np.uint16_t j = 0
    cdef np.uint16_t k = 0
    #the results array contains the costs is the first column, followed by the path, backwards.  
    cdef np.ndarray[np.uint16_t, ndim=2] path_results = np.zeros( (starting_node_indices.shape[0]*ending_node_indices.shape[0], array_len) , dtype=np.uint16 )
    cdef np.ndarray[DTYPE_F, ndim=1] cost_results = np.zeros( starting_node_indices.shape[0]*ending_node_indices.shape[0])
    print starting_node_indices.shape[0]*ending_node_indices.shape[0]
    
    # Start the algorithm.  First iterate over every starting node.
    for snindex in starting_node_indices:
        print "Working on starting node index %i." % snindex
        # We set the cost vector to zero, the highest cost for any path.
        # The starting node cost is set to its equilibrium population, as determined by the stationary eigenvector
        cost_vector[:] = 0.0 
        cost_vector[snindex] = eigvec[snindex]
        
        # Set the visited list to all False (unvisited):
        visited[:] = False
        
        # Set the current node to the starting node.
        cnindex = snindex
        
        # Set done to false
        done = False
        
        # Iterate over the graph and update all node costs.
        while not done:
            
            # Find the lowest cost (highest value) node:
            for i in range(array_len):
                if not visited[i]:
                    cnindex = i
                
            for i in range(array_len):
                if cost_vector[i] > cost_vector[cnindex] and not visited[i]:
                    cnindex = i
            
            #The current node is now visited
            visited[cnindex] = 1
            
            # Update adjacent node costs
            for i in range(array_len):
                if TM[cnindex][i]*cost_vector[cnindex] > cost_vector[i] and not visited[i]:
                    cost_vector[i] = TM[cnindex][i]*cost_vector[cnindex]
            
            # Update the done variable.  We are done when the visited vector is all True
            if np.all(visited):
                done = True
            
        print "Calculating Paths"       
        #The cost_vector now contains all the information we need, for each ending node.  We now simply extract the path information.
        for i in range(ending_node_indices.shape[0]):
        
          
            # Iterative backwards over the nodes to make the new path
            cnindex = ending_node_indices[i]
            
            # i indexes the the row of the results matrices where we store our answers.
            # We must add to i the number of ending nodes, times the number of starting nodes we have gone through, such that we do not overwrite old data.
            
            i += number_of_starting_nodes_finished*ending_node_indices.shape[0]
            
            # Set the first elements of the current row of results to the cost:
            cost_results[i] = cost_vector[cnindex]
            
            # Set the next element of the current row of results to the ending node index:
            k = 0
            path_results[i][k] = cnindex
            k += 1
            
            
            # Set the done variable to false
            done = False
            
            #Calculate the path
            while not done:
            
                # This variable is used to check if the current node does not update after looping over the matrix.
                # If it does not update, then the path is broken.
                # This is likely it areas where the path starts with zero (or very close to it) starting value
                old_node_index = cnindex
                
                # Now we check to see if each adjacent node is part of the path.
                for j in range(array_len):
                    if cost_vector[j]*TM[j][cnindex] == cost_vector[cnindex]:
                        cnindex = j
                        path_results[i][k] = cnindex
                        k += 1
                        break
                        
                # Check if the node was actually updated, to avoid infinite loops
                if old_node_index == cnindex:
                    print "Broken path on path from node index %i to node index %i, broken on node %i." % (snindex, ending_node_indices[i], cnindex)
                    break
                    
                # Check if the current node is the starting node.  If so, we are done.
                if cnindex == snindex:
                    done = True
        number_of_starting_nodes_finished += 1        
                    
    #Return the array of costs and the array of paths:
    return cost_results, path_results
            
        
               
        
