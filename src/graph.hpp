/*
 * graph.hpp
 *
 *  Created on: 26-Nov-2020
 *      Author: abhishek
 */

#ifndef GRAPH_HPP_
#define GRAPH_HPP_

#include <vector>
#include <map>


class EdgeException {};

//Adjacency list representation of a Undirected Graph.
class Graph {

	public:
	std::vector< int> adjacencyList_m; // all edges in contiguous order
	std::vector< int> edgeOffsets_m; // offsets of edges for vertex i = 0...|V| - 1 in the list.
	std::vector< int> vertexDegree_m; // total number of edges for vertex i = 0...|V| - 1, (offset + degree) indices

	int numVertices_m = 0, numEdges_m = 0;
 	Graph(const std::string &path); //Generates a Random graph provided the parameters.

	private:
 	void loadGraphFile(const std::string &path, std::map<int, std::vector<int>> &adjList);

};

#endif /* GRAPH_HPP_ */
