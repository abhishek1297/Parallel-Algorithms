#include "graph.hpp"
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <sstream>
#include <fstream>

Graph::Graph(const std::string &path) {


	 int u, v;


	 std::map<int, std::vector<int>> adjList;
	 loadGraphFile(path, adjList);

	 for (auto it: adjList) {

		 int parent{it.first - 1};
		 vertexDegree_m.push_back(it.second.size());
		 edgeOffsets_m.push_back(adjacencyList_m.size());
		 for (int child: it.second) {
			 adjacencyList_m.push_back(child - 1);
		 }
	 }

	 numVertices_m = edgeOffsets_m.size();
	 numEdges_m = adjacencyList_m.size();
	 std::cout << "Vertices: " << numVertices_m << std::endl
			 	<< "Edges: " << numEdges_m << std::endl;

}

void Graph::loadGraphFile(const std::string &path, std::map<int, std::vector<int>> &adjList) {

	std::ifstream ifs(path, std::ifstream::in);
	if (ifs.is_open()) {
		std::string line{""}, token{""};
		int count{0}, key, val;
		while (std::getline(ifs, line)) {
			std::istringstream iss(line);
			count = 0;
			while (std::getline(iss, token, '\t')) {
				int *ptr = count == 0 ? &key : &val;
				*ptr = std::stoi(token);
				count ++;
				if (count == 2) break;
			}
			adjList[key].push_back(val);
		}
	}
	else
		std::cout << "Error Loading File" << std::endl;
}
