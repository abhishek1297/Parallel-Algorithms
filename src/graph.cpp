#include "graph.hpp"
#include <fstream>
#include <sstream>
#include <functional>
#include <string>
#include <iostream>
#include <algorithm>


/**
 * This constructor will generate a Adjacency list graph representation
 * for the given input file passed.
 *
 * Parameters:
 * path: absolute path of the input file
 * numInputs: Total number of inputs that needs to be read on each line of the file
 * indexToRead: a list indices(zero based) of each token needed {from, to, weight}
 * convertToZeroIdx: convert to zero based index if the vertex value starts with one
 * mode: bfs or sssp
 */

Graph::Graph(const std::string &path,
		const int &numInputs,
		const std::vector<int> &indexToRead,
		bool convertToZeroIdx,
		const std::string &mode) :pathData_m(path){

	adjlist_vec adjList;
	loadGraphFile(adjList, numInputs, indexToRead, convertToZeroIdx);
	int nV{static_cast<int>(adjList.size())};
	edgeOffsets_m.resize(nV);
	vertexDegree_m.resize(nV);

	std::function<void (std::vector<std::pair<int, int>> &)> func;
	if (mode == "bfs") {
		func = [this](std::vector<std::pair<int, int>> &parent) {
			for (auto &child : parent) {
				adjacencyList_m.push_back(child.first);
			}
		};

	}
	else
	if (mode == "sssp") {
		func = [this](std::vector<std::pair<int, int>> &parent) {

			for (auto &child: parent) {
				adjacencyList_m.push_back(child.first);
				edgeWeights_m.push_back(child.second);
			}
		};
	}

	for (int parent=0; parent<nV; ++parent) {

		 edgeOffsets_m[parent] = adjacencyList_m.size();
		 vertexDegree_m[parent] = adjList[parent].size();
		 func(adjList[parent]);
	}

	 numVertices_m = edgeOffsets_m.size();
	 numEdges_m = adjacencyList_m.size();

}

void Graph::loadGraphFile(adjlist_vec &adjList,
		const int &numInputs,
		const std::vector<int> &indexToRead,
		bool convertToZeroIdx) {

	std::ifstream ifs(pathData_m, std::ifstream::in);
	ifs.seekg(0, ifs.beg);
	if (ifs) {
		std::string nVStr;
		std::getline(ifs, nVStr);//1st |V| line
		int nV;
		try {
			nV = std::stoi(nVStr);
			for (int i=0; i<nV; ++i, adjList.push_back(std::vector<std::pair<int, int>>()));
		}
		catch(std::exception &e) {
			std::cout << "nV error" << std::endl;
			std::cout << e.what();
		}

		if (*(std::max(indexToRead.begin(), indexToRead.end())) >= indexToRead.size()) {
			std::cout << "invalid indices" << std::endl;
			exit(1);
		}

		int fromIdx = indexToRead[0];
		int toIdx = indexToRead[1];
		int weightIdx = indexToRead[2];

		std::function<void (std::string, std::string, std::string)> store;
		if (convertToZeroIdx) {
			store = [&adjList](std::string from, std::string to, std::string wt) {
					int u = std::stoi(from) - 1;
					int v = std::stoi(to) - 1;
					adjList[u].push_back(std::make_pair(v, stoi(wt)));
				};
		}
		else {
			store = [&adjList](std::string from, std::string to, std::string wt) {
					int u = std::stoi(from);
					int v = std::stoi(to);
					adjList[u].push_back(std::make_pair(v, stoi(wt)));
				};
		}


		std::string line;
		std::vector<std::string> token(4);

		while(std::getline(ifs, line)) {

			line.erase(std::remove(line.begin(), line.end(), ','), line.end());
			std::stringstream ss(line);
			std::string t;
			int i{0};
			while (ss >> t)
				token[i++] = t;
			store(token[fromIdx], token[toIdx], numInputs > 2 ? token[weightIdx]: "-1");
		}
		ifs.close();
	}
	else {
		std::cout << "Error Loading File" << std::endl;
		ifs.close();
		exit(1);
	}
}


std::ostream& operator <<(std::ostream &out, Graph &G) {

	out << "Name: " << G.pathData_m << std::endl
	<< "Vertices: " << G.numVertices_m << std::endl
	<< "Edges: " << G.numEdges_m << std::endl;
	out << std::endl;
	return out;
}
