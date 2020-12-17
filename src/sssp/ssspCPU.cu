#include "ssspCPU.hpp"

namespace ssspCPU {
	struct comparator {
		bool operator ()(const std::pair<int, int> &p1, const std::pair<int, int> &p2) const {
			return p1.second < p2.second;
		}
	};

	double execute(Graph &G, std::vector<int> &distance, int source) {

		std::vector<std::pair<int, int>> pQ;
		distance.resize(G.numVertices_m);
		std::fill(distance.begin(), distance.end(), INT_MAX);
		distance[source] = 0;
		pQ.push_back(std::make_pair(source, distance[source]));

		auto start = std::chrono::high_resolution_clock::now();
		while (!pQ.empty()) {
			std::sort(pQ.begin(), pQ.end(), comparator());
			int vertex = pQ[0].first;
			pQ.erase(pQ.begin());
			for (int i=G.edgeOffsets_m[vertex]; i<G.edgeOffsets_m[vertex]+G.vertexDegree_m[vertex]; ++i) {

				int child = G.adjacencyList_m[i];

				int alt = distance[vertex] + G.edgeWeights_m[i];

				if (alt < distance[child]) {

					distance[child] = alt;
					pQ.push_back(std::make_pair(child, alt));
				}
			}
		}
		auto end = std::chrono::high_resolution_clock::now();

		std::chrono::duration<double, std::milli> t = end - start;
		return t.count();
	}
}
