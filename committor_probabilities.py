#!/usr/bin/env python
'''This is intended as a module.

This module is used to compute committor probabilites.                                             
The idea is as follows:
    From a weighted ensemble simulation, we are given a set of bins, along with segments in each 
bin.  Each segment has a certain weight.  By watching the system evolve over time, we can track the
transition rates between bins.  Normalizing these transition rates gives the average probability 
that a segment in a certain bin will next move to some other bin.  In fact, we need not use the
binning scheme use for the progress coordinates in the propogation steps.  We may use any set of 
bins as can be defined by wassign.
    We can visualize this process as a network of nodes, each of which is a bin. Nodes are connected
by directed edges which denote the probability of moving between bins. Furthermore, we can define
"states" on this set of bins.  A state is a subset of the total set of bins.  For example, we may
define a certain set of bins such that it corresponds to the "folded" or "bound" state of a protein.
    The set of all bins not defined as another state is called the intermediate (I) state.  Assuming
the network is not disjoint (ie, each node is connected to every other node by some path), then a
segment in the intermediate state, moving on a random walk defined by the transition probabilities
between bins, will eventually reach one of the defined states.  Once it reaches the defined state,
we may say the segment "stops" there.  It need not actually "stop" in the defined state in a real 
simulation; in most cases this state will in fact not be a sink. However, for the purposes of this 
analysis, we say it stops in this state, and thus term it an "absorbtion state." If there is only 
one absorbtion state, then any segment in the intermediate state will eventually reach it (given a 
suitable network).
    With two or more absorbtion states, then a segment will sometimes reach one absorbtion state 
(and then "stop"), or sometimes reach another absorbtion state.  The associated probabilities of 
reaching each absorbtion state are known as the committor probabilities.
    If we define absorbtion states such that they correspond to free energy wells, then we can
use committor probabilities to characterize the transition state.  Given two absorbtion states, 
we may look at each node and compare its committor probabilty for the first absorbtion states to
its committor probability for the second absorbtion  state.  If they are equal, then a segment in
this node (bin) is equally likely to move to either absorbtion state; it is thus one way of
defining the transition state.

Calculating the committor probabilities requires a first-step analysis.  Our code implements this analysis as a simple matrix equation.  

Please see the following for more information:
    ** Wales, David J. "Calculating rate constants and committor probabilities for transition networks by graph transformation." The Journal of Chemical Physics. 29 May 2009
    ** Taylor, Howard M. An Introduction to Stochastic Modeling. Third Edition.  Academic Press, New York. 1998.

'''
import numpy as np
from numpy.linalg import solve

class state_info:
    def __init__(self, TM, evals, evecs, labels):
        self.labels = labels
        self.TM = TM
        self.evals = evals
        self.evecs = evecs
        self.states={}
        self.index = {}
        for i in range(len(TM)):
            self.index[i] = labels[i]
            self.index[labels[i]] = i
        self.state_index = {'Intermediate' : [i for i in range(len(TM))] }
    def set_state(self, state_name, state_labels):
        '''Creates a new state.  The state is comprised of bins with labels passed in 
        state_lables (an iteratable).'''
        self.states[state_name] = {'labels' : state_labels}
        index_list = []
        for label in state_labels:
            index_list.append(np.where(self.labels==label)[0][0])
        self.states[state_name]['indices'] = index_list
        self.state_index[state_name] = index_list
        self.intermediate_state_update(index_list)
    def intermediate_state_update(self, indices_now_in_other_state):
        self.state_index['Intermediate'] = [x for x in self.state_index['Intermediate'] if not x in indices_now_in_other_state]
        
def equal_committor_probability_bins(state_info, error_margin):
    'This is the main function for this module.  Given a state_info class, as defined above, and an
error margin, it calculates the bins from which committor probabilites to the non-intermediate 
states are approximately equal.  We define "approximately equal" using the error margin.  For 
example. If there are two states, then a bin should ideally have .5 committor probability to each.
With an error margin of .1, a bin can have probabilities differing as mucn as .4:.6, rather than 
.5:.5.  With three bins, ideal probability is 1/3.  With an error margin of .05 will allow bins with
probabilities as different as, for example about .28333:.33333:.38334.  An error margin at or above 
.5 will return all bins.  This can be useful in many cases.

    This function returns the bin label and committor probability to each state, for each bin that 
fits the definition of having approximately equal comittor probabilities to each state.'''
    #The AbsorbTM is a matrix built from the original transition matrix, but now the rows corresponding to absorbtive states are zeroed.
    AbsorbTM = state_info.TM[...,...]
    for state_name in state_info.states.iterkeys():
        for index in state_info.states[state_name]['indices']:
            AbsorbTM[index] = np.zeros(AbsorbTM.shape[0])
    #This dictionary holds committor probabilities.  Keys are matrix indices; while values are probabilities
    CommProbs = {}
    #Calculate the committor probability to each STATE
    for state_name in state_info.states.iterkeys():
        #Get the "minor," deleting rows and columns corresponding to all indices of the bins in this state
        #First make a mask.  1's in the mask "let that data in that row/column through," while zeros cut off that row/column
        mask = np.ones(state_info.TM.shape[0], dtype=bool)  #Shortcut to make array of all TRUE values.
        for index in state_info.states[state_name]['indices']:
             mask[index] = False
        MinorTM = AbsorbTM[...,mask][mask,...]
        CommProbs[index] = np.zeros(MinorTM.shape[0])
        #Now subtract I.  This is simply a shortcut to solving the system of equations we wish to solve (it results from working out the equations given by the first step analysis and putting them into a matrix equation).
        SubMinTM = MinorTM - np.identity(MinorTM.shape[0])
        #Also from working out the matrix equations.  Here, the right hand side is the sum (over all the bins in the absorbtion state in question) of each "intermediate" bins' direct (not passing through other bins) transitions into that state.
        #First sum over each bin in absorbtion state
        RHS = np.zeros(AbsorbTM.shape[0])
        for index in state_info.states[state_name]['indices']:
            RHS += AbsorbTM[:,index]
        RHS = RHS[mask]
        #Now make the "right hand side" of the equation negative.
        RHS = -1 * RHS
        #Actually solve the matrix equation.  The result is the commutter probability for all the indices in the MinorTM.
        solution = solve(SubMinTM, RHS)
        names_of_bins_relative_to_new_indices_list = state_info.labels[mask]
        
        
        CommProbs[state_name] = dict(zip(names_of_bins_relative_to_new_indices_list ,solution[:]))
    #Check for each bin, if the committer probability is equal for each state
    equal_probability_committor_list = []

    #Count the number of absorbtive states:
    state_count = 0
    for state in state_info.states.iterkeys():
        if state != 'Intermediate' : state_count += 1
    #Calculate "ideal" probability flux to each state
    ideal_probability = 1/float(state_count)
    for index in state_info.state_index['Intermediate']:
        approximately_equal = True
        for state_name in state_info.states.iterkeys():
            if state_name != 'Intermediate':
                bin_name = state_info.index[index]
                if CommProbs[state_name][bin_name] < ideal_probability - error_margin or CommProbs[state_name][bin_name] > ideal_probability + error_margin:
                    approximately_equal = False
        if approximately_equal == True:
            equal_probability_committor_list.append( (state_info.index[index] , [CommProbs[state_name][state_info.index[index]] for state_name in state_info.states.iterkeys()] ))
    return equal_probability_committor_list
        
    
