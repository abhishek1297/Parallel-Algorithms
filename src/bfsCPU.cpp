#include "bfsCPU.hpp"
#include <queue>
#include <iostream>
#include <chrono>
#include <algorithm>

void bfsCPU( int source, Graph &G, std::vector<int> &distance, std::vector<bool> &visited) {

	std::queue< int> queue;
	 int vertex, child;

	queue.push(source);
	visited[source] = true;
	distance[source] = 0;

	while(!queue.empty()) {

		vertex = queue.front();
		queue.pop();

		for ( int i=G.edgeOffsets_m[vertex]; i<G.edgeOffsets_m[vertex]+
		G.vertexDegree_m[vertex]; ++i) {
			child = G.adjacencyList_m[i];
			if (!visited[child]) {

				visited[child] = true;
				distance[child] = distance[vertex] + 1;
				queue.push(child);
			}
		}
	}
}


double execBfsCPU(Graph &G, int nV) {

	std::vector<int> distance(nV, -1);
	std::vector<bool> visited(nV, false);

	auto start = std::chrono::high_resolution_clock::now();

	bfsCPU(0, G, distance, visited);

	auto end = std::chrono::high_resolution_clock::now();

	std::chrono::duration<double, std::micro> t = end - start;
	std::cout << "Time: " << t.count();

//	std::cout << "\n----------------\n";
//	for (int i=0; i<nV; ++i)
//		std::cout << i << ": " << distance[i] << std::endl;
	std::cout << std::endl;
	std::cout << *(std::max_element(distance.begin(), distance.end())) << std::endl;
	return t.count();
}
