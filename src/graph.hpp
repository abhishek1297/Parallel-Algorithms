/*
 * graph.hpp
 *
 *  Created on: 26-Nov-2020
 *      Author: abhishek
 */

#ifndef GRAPH_HPP_
#define GRAPH_HPP_
#include <vector>
#include <string>
#include <iostream>

using adjlist_edge_wt =  std::vector<std::vector<std::pair<int, int>>>;

class EdgeException {};

//Adjacency list representation of a Undirected Graph.
class Graph {

	public:


	int numVertices_m = 0, numEdges_m = 0;
	std::vector<int> adjacencyList_m; // all edges in contiguous order
	std::vector<int> edgeOffsets_m; // offsets of edges for vertex i = 0...|V| - 1 in the list.
	std::vector<int> vertexDegree_m; // total number of edges for vertex i = 0...|V| - 1, (offset + degree) indices
	std::vector<int> edgeWeights_m; // weights of all edges in contiguous order
	//general for printing
	std::string pathData_m;
	Graph(const std::string &path, const int &numInputs, const std::vector<int> &indexToRead, bool convertToZeroIdx, const std::string &mode);

 	friend std::ostream& operator <<(std::ostream &out, Graph &G);
	private:
 	void loadGraphFile(adjlist_edge_wt &adjList, const int &numInputs, const std::vector<int> &indexToRead, bool convertToZeroIdx);
};

#endif /* GRAPH_HPP_ */
