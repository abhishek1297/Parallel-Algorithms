#include "graph.hpp"
#include <ctime>
#include <fstream>
#include <sstream>

//Generates graph representation given a dataset
Graph::Graph(const std::string &path) {

	std::vector<std::vector<int>> adjList;
	loadGraphFile(path, adjList);
	int nV{static_cast<int>(adjList.size())};
	edgeOffsets_m.resize(nV);
	vertexDegree_m.resize(nV);

	for (int parent=0; parent<nV; ++parent) {

		 edgeOffsets_m[parent] = adjacencyList_m.size();
		 vertexDegree_m[parent] = adjList[parent].size();
		 for (int child: adjList[parent]) {
			 adjacencyList_m.push_back(child);
		 }
	}

	 numVertices_m = edgeOffsets_m.size();
	 numEdges_m = adjacencyList_m.size();
	 std::cout << "Vertices: " << numVertices_m << std::endl
			 	<< "Edges: " << numEdges_m << std::endl;

}

//loading adjacency list from the dataset file.
void Graph::loadGraphFile(const std::string &path,
		std::vector<std::vector<int>> &adjList) {
	std::ifstream ifs(path, std::ifstream::in);
	ifs.seekg(0, ifs.beg);
	if (ifs) {
		std::vector<int> vec;
		std::string nVStr, from, to;
		std::getline(ifs, nVStr);
		int u, v, nV;
		try {
			nV = std::stoi(nVStr);
			for (int i=0; i<nV; ++i,
			adjList.push_back(std::vector<int>()));
		}
		catch(std::exception &e) {
			std::cout << e.what();
		}
		while (ifs >> from >> to) {
			u = std::stoi(from);
			v = std::stoi(to);
			adjList[u].push_back(v);
		}
	}
	else
		std::cout << "Error Loading File" << std::endl;
}


std::ostream& operator <<(std::ostream &out, Graph &G) {

	out << "AdjList contiguous\n" ;
	for (auto x: G.adjacencyList_m) {
		out << x << " ";
	}
	out << std::endl;
	return out;
}
