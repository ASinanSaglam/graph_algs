#!/usr/bin/env python
from __future__ import print_function, division; __metaclass__ = type
import sys
import logging
import math
from numpy import index_exp

from westtools import (WESTTool, WESTDataReader,  
                       ProgressIndicatorComponent)
import numpy
from westpa import h5io
from westpa.h5io import WESTPAH5File
from westpa.extloader import get_object

class ComProb:
    '''
    Calculate the committor probabilities from each bin (corresponding to 
    rows of transition_matrix) into each state given in state_idx_list.
    This uses a first-step analysis.  ``state_map`` should be an length-N
    vector, given as a numpy array, where transition_matrix is of dimensions
    N*N.  Each element i of the state_map should denote to which state that 
    bin belongs.
    '''
    def __init__(self, transition_matrix, state_map, state_idx_list):
        self.transition_matrix = transition_matrix
        self.state_map = state_map
        self.state_idx_list = state_idx_list


    def remove_rows(self, arr, row_idxs):
        good_rows = range(arr.shape[0])
        for idx in row_idxs:
            good_rows.remove(idx)
        good_rows = numpy.array(good_rows)
        newarr = arr[good_rows]
        return newarr


    def remove_cols(self, arr, col_idxs):
        good_cols = range(arr.shape[1])
        for idx in col_idxs:
            good_cols.remove(idx)
        good_cols = numpy.array(good_cols)
        newarr = arr[:, good_cols]
        return newarr
        

    def get_minor(self, arr, row_idxs=None, col_idxs=None):
        '''Get the minor for ``arr``, an order-two tensor. Exclude rows and
        columns given in row_idxs and col_idxs, respectively.''' 
        print("column idxs to remove: " + repr(col_idxs))
        print("row idxs to remove: " + repr(row_idxs))
        newarr = self.remove_rows(self.remove_cols(arr, col_idxs), row_idxs)
        return newarr
         

    def solve(self):
        '''Calculate the committor probabilities.'''
        nbins = self.transition_matrix.shape[0]
        results = numpy.zeros((nbins, len(self.state_idx_list)))

        # Get the indices of bins that are in one of the specified states.
        bin_idxs_for_macrostates = []
        for state_idx in self.state_idx_list:
            bin_idxs_for_macrostates += \
                    list(numpy.where(self.state_map == state_idx)[0])
                                       
        # Get the minor for the transition matrix, removing elements
        # corresponding to absorptive states.
        minor = self.get_minor(self.transition_matrix,
                               row_idxs=bin_idxs_for_macrostates,
                               col_idxs=bin_idxs_for_macrostates) 
        # subtract identity from the minor
        minor_minus_identity = minor - numpy.identity(minor.shape[0])

        for istate, state_idx in enumerate(self.state_idx_list):
            # Get right hand side (rhs). For a given row, sum all the transition
            # probabilities corresponding to transitions into state_idx.
            rhs = numpy.empty(self.transition_matrix.shape[0])
            idxs_of_bins_in_cur_state = numpy.where(self.state_map == state_idx)[0]
            rhs = numpy.sum(self.transition_matrix[:, idxs_of_bins_in_cur_state],
                         axis=1)
            # Keep track of the rows that will be removed.
            idx_map = numpy.array(range(rhs.shape[0]))

            # Remove the rows corresponding to a bin in a macrostate (from
            # state_idx_list). This makes the indexing the same as for the
            # minor. 
            rhs = self.remove_rows(rhs, bin_idxs_for_macrostates)
            print('rhs: ' + repr(idx_map))
            idx_map = self.remove_rows(idx_map, bin_idxs_for_macrostates)
            print("idx_map: " + repr(idx_map))

            # Solve the matrix equation; ``sol`` gives the committor 
            # probabilities.
            #sol = numpy.linalg.solve(minor_minus_identity, -1*rhs)
            sol = numpy.linalg.lstsq(minor_minus_identity, -1*rhs)[0]
            print("solution: " + repr(sol))

            # Place the solution into the results array, mapping the indices
            # back to match the original.
            results[idx_map, istate] = sol

            # Set the committor probabilities to one for bins that are in the
            # current state.
            results[numpy.where(self.state_map == state_idx)[0], istate] = 1.0
        return results

    #def solve_without_minor(self):
    #    nbins = self.transition_matrix.shape[0]
    #    results = numpy.empty((nbins, len(self.state_idx_list)))

    #    transmat_minus_identity = self.transition_matrix - numpy.identity(nbins)
    #    for istate, state_idx in enumerate(self.state_idx_list):
    #        # Get right hand side (rhs). For a given row, sum all the transition
    #        # probabilities corresponding to transitions into state_idx.
    #        rhs = numpy.empty(self.transition_matrix.shape[0])
    #        idxs_of_bins_in_cur_state = numpy.where(self.state_map == state_idx)[0]
    #        rhs = numpy.sum(self.transition_matrix[:, idxs_of_bins_in_cur_state],
    #                     axis=1)
    #        # Solve the matrix equation; ``sol`` gives the committor 
    #        # probabilities.
    #        sol = numpy.linalg.solve(transmat_minus_identity, -1*rhs)
    #        results[:,i] = sol
    #    return results


class WComProb(WESTTool):
    prog ='w_comprob'
    description = '''\
Calculate the probability of a walker in a given bin next entering each of a 
set of states. Bin assignments (usually "assignments.h5") and a right
stochastic transition matrix (usually "multireweight.h5") must be supplied.
The transition matrix should be generated with a tool such as w_multi_reweight,
run with the "-t/--save-transition-matrices" flag. 

-----------------------------------------------------------------------------
Output format
-----------------------------------------------------------------------------
The output file (DEFAULT: "comprob.h5", otherwise specified by "-o/--output" 
flag) contains the following dataset:

  /committor_probabilities: [bin, state_0, state_1]

-----------------------------------------------------------------------------
Command-line options
-----------------------------------------------------------------------------
'''
    def __init__(self):
        super(WComProb, self).__init__()
        
        self.data_reader = WESTDataReader()
        self.progress = ProgressIndicatorComponent()
        
        #self.output_filename = None
        #self.transmat_filename = None
        #self.assignment_filename = None
        #
        #self.output_file = None
        #self.transmat_file = None
        #self.assignments_file = None
        
    def add_args(self, parser):
        ''' Add arguments for specifying input files, in addition to options
        on what transition matrix to use, as well as the tolerance for the 
        isocommittor surface.'''
        self.progress.add_args(parser)
        self.data_reader.add_args(parser)

        iogroup = parser.add_argument_group('input/output options')
        iogroup.add_argument('-a', '--assignments', default='assign.h5',
                             help='''Use bin and macrostate assignments stored
                             in ASSIGNMENTS. (default: %(default)s).''')

        iogroup.add_argument('-t', '--transition-matrices', 
                             dest='transition_matrices', 
                             default='multireweight.h5',
                             help='''Use transition matrices stored in
                             TRANSITION_MATRICES (default: %(default)s).''')

        iogroup.add_argument('-o', '--output', dest='output', 
                             default='comprob.h5',
                             help='''Store results in OUTPUT (default: 
                             %(default)s).''')

        cogroup = parser.add_argument_group('calculation options')
        cogroup.add_argument('-i', '--iteration', default=None, 
                             metavar='N_ITER', type=int,
                             dest='iteration',
                             help='''Use transition matrix stored for iteration 
                             N_ITER. By default, use the last iteration for
                             which a transition matrix is stored.''') 

        cogroup.add_argument('--state-indices', default='[0,1]',
                             metavar="STATE_IDX_LIST",
                             help='''Calculate committor probabilities to
                             states in STATE_IDX_LIST, a string that will be
                             parsed as a Python list.''')

        cogroup.add_argument('--alpha', default=.05, type=float, dest='alpha',
                              help='''Accept bins with committor probability 
                              between 0.5-ALPHA and 0.5+ALPHA as having equal
                              commmittor probabilities to either state.''')

        cogroup.add_argument('--use-color', action='store_true',  
                             dest='i_use_color',
                             help='''If specified, retain color labels on bins
                             during committor probability calculation.
                             Otherwise, reduce colored matrix to uncolored
                             matrix for analysis.''') 


    def _eval(self, string):
        '''Evaluate the given string to a Python object, and return it.'''
        return eval(string, {'numpy': numpy,
                             'np': numpy})
                            

    def process_args(self, args):
        '''Parse arguments from the command line, making them available as
        attributes.'''
        self.progress.process_args(args)
        self.data_reader.process_args(args)
        
        self.output_filename = args.output
        self.assignments_filename = args.assignments
        self.transmat_filename = args.transition_matrices
                
        self.alpha = args.alpha
        if self.alpha <= 0 or self.alpha > 0.5:
            raise ValueError('Parameter error -- ALPHA must be between 0 and'
                             '0.5 (supplied: {:f}).'.format(sel.alpha))

        self.state_idx_list = self._eval(args.state_indices)
        self.i_use_color = args.i_use_color
        self.iteration = args.iteration


    def open_files(self):
        '''Open output, assignments, and transition matrix files, and load
        some basic information from the assignments file.'''
        self.output_file = h5io.WESTPAH5File(self.output_filename, 'w', 
                                             creating_program=True)
        h5io.stamp_creator_data(self.output_file)

        self.assignments_file = h5io.WESTPAH5File(self.assignments_filename, 
                                                  'r')
        self.transmat_file = h5io.WESTPAH5File(self.transmat_filename, 'r')

        # Load some basic information needed early in the analysis.
        self.nstates = self.assignments_file.attrs['nstates']
        self.nbins = self.assignments_file.attrs['nbins']
        self.state_labels = self.assignments_file['state_labels'][...]
        self.state_map = numpy.array(self.assignments_file['state_map'][...])

        # remake the state_map if necessary
        if self.i_use_color:
            new_state_map = numpy.empty(self.nstate*self.nbins)
            for i in xrange(self.nbins):
                for j in xrange(self.nstates):
                    new_state_map[self.nstates*i+j] = self.state_map[i] 
            self.state_map = new_state_map


    def load_transition_matrix(self):
        '''Load the transition matrix from self.transmat_file, either looking 
        for the transition matrix at the iteration specified by self.iteration
        or otherwise defaulting to the final transition matrix stored. If color
        is not to be used, convert the colored transition matrix to an
        uncolored transition matrix. Store the transition matrix in 
        self.transition_matrix''' 

        if 'iterations' not in self.transmat_file.keys():
            raise KeyError('No transition matrices are stored in {:s}! '
                           'Please check that you have supplied the correct '
                           'transition matrix file, and that the file was '
                           'created using the "-t/--save-transition-matrices" '
                           'flag.'.format(self.transmat_filename))
        # Default to looking for last iteration
        if self.iteration is None: 
            key = sorted(self.transmat_file['iterations'].keys())[-1]
            transmat_group = self.transmat_file['iterations'][key]
        # Otherwise use the specified iteration
        else:
            try:
                transmat_group = self.transmat_file['iterations/iter_{:08d}'
                                                    .format(self.iteration)]
            except KeyError:
                 raise KeyError('No transition matrix is stored for iteration '
                                '{:d} in transition matrix file {:s}.'
                                .format(self.iteration, self.transmat_filename)
                                )

        # Convert sparse to dense matrix.
        nfbins = self.nbins*self.nstates
        self.transition_matrix = numpy.empty((nfbins, nfbins), 
                                             dtype=numpy.float64)
        rows = numpy.array(transmat_group['rows'], dtype=numpy.int16)
        cols = numpy.array(transmat_group['cols'], dtype=numpy.int16)
        self.transition_matrix[rows, cols] = transmat_group['k']
 
        # Convert colored to non-colored transition matrix, if needed
        if not self.i_use_color:
            self.transition_matrix = self.convert_colored_to_noncolored_matrix(
                                                         self.transition_matrix)
 

    def convert_colored_to_noncolored_matrix(self, mat):
        '''Convert the colored matrix ``mat`` to a noncolored matrix, using 
        information on the number of states and bins from self.nstates and 
        self.nbins. Return a new matrix.''' 
        nbins = self.nbins
        nstates = self.nstates
        newmat = numpy.empty((nbins, nbins))
        print("Dimensions of matrix: " + repr(mat.shape))
        for i in xrange(nbins):
            for j in xrange(nbins):
                newmat[i,j] = numpy.sum(mat[i*nstates:(i+1)*nstates,
                                            j*nstates:(j+1)*nstates]
                                        ) 
        return newmat

    def go(self):
        '''Run the main analysis.''' 
        pi = self.progress.indicator
        with pi:
            pi.new_operation('Initializing')
            self.open_files()
            self.load_transition_matrix()
            # Calculate the committor probabilities
            pi.new_operation('Calculating committor probabilities')
            comprob_arr = ComProb(self.transition_matrix,
                                  self.state_map,
                                  self.state_idx_list).solve()
            pi.new_operation('Finding isocommittor surface')
            # Find isocomittor surface. Each element of close_arr corresponds
            # to a given bin.  When set to `1`, this means that the committor
            # probability for this bin is close to the ideal value, which is 
            # usually 0.5 (however, we do not explicitly enforce that only two
            # states be specified; if three are specified, then the ideal value
            # would be 0.3333...). 
            close_arr = numpy.ones(comprob_arr.shape[0])
            # "ideal" is usually .5
            ideal = numpy.ones(comprob_arr.shape[0], dtype=float)
            ideal /= len(self.state_idx_list)
            # For each ending state, check that the value is close to ideal.
            # Recursive logical and statements ensure that once an ending state
            # is encountered where a given bin's committor probability is not 
            # close enough to the ideal value, the corresponding element of 
            # close_arr stays forever False.
            for i in xrange(comprob_arr.shape[1]):
                close_arr = numpy.logical_and(close_arr,
                                              numpy.isclose(comprob_arr[:,i],
                                                            ideal,
                                                            rtol=0.0,
                                                            atol=self.alpha)
                                              )
                
            isocom_bin_idxs = numpy.where(close_arr)[0]

            self.output_file.create_dataset('committor_probabilities',
                                            data=comprob_arr) 
            self.output_file.create_dataset('isocommittor_bins',
                                            data=isocom_bin_idxs)
                           
if __name__ == '__main__':
    WComProb().main()
