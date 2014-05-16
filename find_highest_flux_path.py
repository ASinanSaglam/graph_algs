#!/usr/bin/env python
'''This script is designed to return the "highest flux path" from one state in a graph to another state, where a "state" is defined as a collection of nodes.  More specifically, it finds the continuous path through which the flow rate from the first state to the second is the greatest.  In our case, we use it to find the "most common" path from the X-ray crystal structure of the villin domain to its NMR structure.  Again, "most common" is taken to mean the single continuous path most likely to be taken from any part of the "X-ray structure state" to the "NMR structure state."  In other words, it should be used to find an "important" transition state.  

To do so, this script must be passed a networkx wieghted digraph.  Each node should have an equilibrium population assigned as an attribute, with a key of "equilibrium_population". Each directed edge should have a weight assigned as an attribute, with a key of "k" (for the rate constant).  Take note that here we assume the Markov Property for node to node transitions.

This script then solves for the least cost path between every node in state 1 and every node in state 2.  The cost function is implemented through a compounded product of the rate (k) values between nodes in the pathway, all multiplied by the equilibrium population of the starting node (giving flux out of that node and to the destination node).  A higher product is lower cost.  This script will use an implementation of Dijkstra's algorithm.'''
def calculate_least_cost_path(input_graph):
	'''This function will calculate the least cost path, as defined above, given a weighted digraph.  It returns a tuple of 2 values.  The first value is a tuple of each node in the least cost path, in the order in which they are traversed.  The second value is the actual flux along this path.'''
	 
