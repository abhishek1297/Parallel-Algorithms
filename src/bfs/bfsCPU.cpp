#include "bfsCPU.hpp"
#include <queue>
#include <chrono>

namespace bfsCPU {

	/**
	 * Executing the Breadth-First search on the CPU
	 *
	 * Parameters
	 * G: A Graph object
	 * distanceCheck: list of distances calculated to verify later
	 * source: source index (zero based). deafult is zero.
	 */
	double execute(Graph &G, std::vector<int> &distanceCheck, int source) {

		std::vector<int> distance(G.numVertices_m, -1);//distance from source

		std::queue<int> queue;
		int vertex, child;

		queue.push(source);
		distance[source] = 0;

		auto start = std::chrono::high_resolution_clock::now();
		while(!queue.empty()) {

			vertex = queue.front();
			queue.pop();

			for ( int i=G.edgeOffsets_m[vertex]; i<G.edgeOffsets_m[vertex]+
												G.vertexDegree_m[vertex]; ++i) {

				child = G.adjacencyList_m[i];
				if (distance[child] == -1) {

					distance[child] = distance[vertex] + 1;
					queue.push(child);
				}
			}
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> t = end - start;
		distanceCheck = std::move(distance);
		return t.count();
	}
}
