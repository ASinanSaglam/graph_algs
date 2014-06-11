#!/usr/bin/env python
# Written 16 May 2014, Alex DeGrave
'''This script is designed to return the "highest flux path" from one state in a graph to another state, where a "state" is defined as a collection of nodes.  More specifically, it finds the continuous path through which the flow rate from the first state to the second is the greatest.  In our case, we use it to find the "most common" path from the X-ray crystal structure of the villin domain to its NMR structure.  Again, "most common" is taken to mean the single continuous path most likely to be taken from any part of the "X-ray structure state" to the "NMR structure state."  In other words, it should be used to find an "important" transition state.  

To do so, this script must be passed a networkx wieghted digraph.  Each node should have an equilibrium population assigned as an attribute, with a key of "equilibrium_population".  Additionally, each node should have an attribute with key of "state", referring to whether it corresponds to the NMR structure, the X-ray crystal structure, or neither.  Possible values are "A" "B" and "I" where A and B are starting and ending states, respectively, and I is the intermediate state (neither A nor B). Each directed edge should have a weight assigned as an attribute, with a key of "k" (for the rate constant).  Take note that here we assume the Markov Property for node to node transitions.

This script then solves for the least cost path between every node in state A and every node in state B.  The cost function is implemented through a compounded product of the rate (k) values between nodes in the pathway, all multiplied by the equilibrium population of the starting node (giving flux out of that node and to the destination node).  A higher product is lower cost.  This script will use an implementation of Dijkstra's algorithm.'''
def calculate_least_cost_path(input_graph):
    '''This function will calculate the least cost path, as defined above, given a weighted digraph.  It returns a tuple of 2 values.  The first value is a tuple of each node in the least cost path, in the order in which they are traversed.  The second value is the actual flux along this path.'''
    # This list will hold the least cost and associated path from every node in A to every node in B.
    cost_and_path_list = []
    for edge_name in input_graph.edges():
        if input_graph.edge[edge_name[0]][edge_name[1]]['k'] == 0:
            input_graph.remove_edge( edge_name[0] , edge_name[1])    
    # Iterate over each node in the graph, finding nodes with a state of A.  These are starting nodes.
    for starting_node_name in input_graph.nodes():
        if input_graph.node[starting_node_name]["state"] == 'A':
            print starting_node_name
            # We now must iterate over each ending node; that is, each node with a state of B
            for ending_node_name in input_graph.nodes():
                 if input_graph.node[ending_node_name]["state"] == 'B':
                
                     # We now implement Dijkstra's algorithm.
	             # Start by copying the input graph to a new graph
	             current_graph = input_graph
	             # Clear the unvisited_list from previous iterations.
	             unvisited_list = []

	             # Set all node cost values to 0 (greatest possible cost)
	             for node_name in current_graph.nodes():
                          current_graph.node[node_name]['cost'] = 0.0
		     # Add every node to the unvisited list.
		     unvisited_list.append(node_name)

	             # Set the starting node cost to its population.
	             current_graph.node[starting_node_name]['cost'] = current_graph.node[starting_node_name]['equilibrium_population'].real #* 1000000000000000000000
                     # Set the looping check parameter to False
	             done = False

	             # Iterate over all neighbors for every current node.
                     previous_cost = 0.0
	             while not done:

	                 # Find the least cost unvisited node and make it the current node:
		         current_node_name = unvisited_list[0]
		         for node_name in unvisited_list:
		             if current_graph.node[node_name]['cost'] > current_graph.node[current_node_name]['cost']:
                                 current_node_name = node_name
                         # Iterate over nodes adjacent to the current node and update their costs.
		         for adjacent_node_name in current_graph[current_node_name]:
	                     # Multiply the k value for the (directed) edge between the current and adjacent node by the cost of the current node.  If this value is greater than the current cost of the adjacent node, we update the adjacent node's cost.
		             if current_graph.edge[current_node_name][adjacent_node_name]['k'] * current_graph.node[current_node_name]['cost'] > current_graph.node[adjacent_node_name]['cost']:
                                 current_graph.node[adjacent_node_name]['cost'] = current_graph.edge[current_node_name][adjacent_node_name]['k'] * current_graph.node[current_node_name]['cost']

	                 # Remove the current node from the unvisited list
                         unvisited_list.remove(current_node_name)
                         # If the the unvisited list is empty, we are done.
                         if len(unvisited_list) == 0: done = True
               
                     # Take note of the final cost for the least cost path.  
                     cost = current_graph.node[ending_node_name]['cost']
                     # We now must find the path.
                     path = []
                     # We build the path from end to beginning, and later reverse it.  First add the last element.
                     path.append(ending_node_name)
                     current_node_name = ending_node_name
                     old_node_name = current_node_name
                     # Iterate over every node in the path  until we reach the starting node.
                     while current_node_name != starting_node_name:
                         # Used to check for endless loop.
                         old_node_name = current_node_name
                         for adjacent_node_name in current_graph.predecessors(current_node_name):
                             # We know that if the cost of a node adjacent to the current node is equal to the cost of the current node divided by the k value for the path FROM THE ADJACENT NODE TO  THE CURRENT NODE (since the graph is bidirectional, then this node must have been used to make the cost value for the current node.  Thus, it is part of the least cost path. Additionally, we must make sure not to backtrack.
                             if current_graph.node[adjacent_node_name]['cost'] * current_graph.edge[adjacent_node_name][current_node_name]['k'] == current_graph.node[current_node_name]['cost'] and not (adjacent_node_name in path):
                                 path.append(adjacent_node_name)
                                 current_node_name = adjacent_node_name
                         # Once we find the node through which the least cost path goes, we no longer need to check any more adjacent nodes.
                                 break

                         # If the current node did not change, then there is no path.  break the endless loop
                         if current_node_name == old_node_name:
                             print 'Broken Path on node ' + str(current_node_name)
                             break
                     # Reverse the path
                     path = path[::-1]
                     # Append the cost and path to a list.
                     cost_and_path_list.append( (cost,path) )
                     print cost
    # After calculating every path and its associated cost, we simply find the path with the least cost among these paths.
    least_cost_and_path = cost_and_path_list[0]
    for cost_and_path in cost_and_path_list:
        if cost_and_path[0] > least_cost_and_path[0]: least_cost_and_path = cost_and_path

    return least_cost_and_path

