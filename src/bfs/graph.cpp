#include "graph.hpp"
#include <ctime>
#include <fstream>
#include <sstream>
#include <functional>
#include <iostream>

//Generates graph representation given a dataset
Graph::Graph(const std::string &path, bool convertToZeroIdx):pathData_m(path) {

	std::vector<std::vector<int>> adjList;
	loadGraphFile(path, adjList, convertToZeroIdx);
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

}

//loading adjacency list from the dataset file.
void Graph::loadGraphFile(const std::string &path,
						  std::vector<std::vector<int>> &adjList,
						  bool convertToZeroIdx)
	{
	std::ifstream ifs(path, std::ifstream::in);
	ifs.seekg(0, ifs.beg);
	if (ifs) {
		std::vector<int> vec;
		std::string nVStr, from, to;
		std::getline(ifs, nVStr);
		int nV;
		try {
			nV = std::stoi(nVStr);
			for (int i=0; i<nV; ++i,
			adjList.push_back(std::vector<int>()));
//			adjList.resize(nV);
//			std::fill(adjList.begin(), adjList.end(), std::vector<int>());
		}
		catch(std::exception &e) {
			std::cout << e.what();
		}
		std::function<void (std::string, std::string)> store;
		if (convertToZeroIdx) {
			 store = [&adjList](std::string from, std::string to) {//zero based index
				int u = std::stoi(from) - 1;
				int v = std::stoi(to) - 1;
				adjList[u].push_back(v);
			 };
		}
		else {
			store = [&adjList](std::string from, std::string to) {
				int u = std::stoi(from);
				int v = std::stoi(to);
				adjList[u].push_back(v);
			};
		}
		while (ifs >> from >> to)
			store(from, to);
	}
	else
		std::cout << "Error Loading File" << std::endl;
}


std::ostream& operator <<(std::ostream &out, Graph &G) {

	out << "Name: " << G.pathData_m << std::endl
	<< "Vertices: " << G.numVertices_m << std::endl
	<< "Edges: " << G.numEdges_m << std::endl;
	out << std::endl;
	return out;
}
