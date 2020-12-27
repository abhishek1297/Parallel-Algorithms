#include "main.hpp"
#include "../dataIO.hpp"
#include <algorithm>

//100 random values
bool verify(int n, std::vector<int> &v1, std::vector<int> &v2) {


	for (int i=1; i<=100; ++i) {//checking random distances

		int u = rand() % n;
		std::cout << (v1[u] == v2[u]);
		if (i % 10 == 0) std::cout << std::endl;
	}
	return true;
}


int bfsMain (Graph&& G) {

	std::vector<int> distanceCPU, distanceGPUH, distanceGPUN, distanceGPUB;
	double minCPU{-1};
	std::vector<double> minGPU(3, -1.0);
	enum GPU{naive=0, hierar, blocked};
	std::cout << "Executing..." << std::endl;
//	minCPU = bfsCPU::execute(G, distanceCPU, 0);
	minGPU[naive] = bfsGPU::naive::execute(G, distanceGPUN, 0);
	minGPU[hierar] = bfsGPU::hierarchical::execute(G, distanceGPUH, 0);
//	minGPU[blocked]= bfsGPU::blocked::execute(G, distanceGPUB, 0);

//	std::cout << "CPU Time: " << minCpu << " ms" << std::endl;
	std::cout << "GPU Naive Time: " << minGPU[naive] << " ms" << std::endl;
	std::cout << "GPU Hierarchical Time: " << minGPU[hierar] << " ms" << std::endl;
//	std::cout << "GPU Blocked Time: " << minGPU[blocked] << " ms" << std::endl;


//	verify(G.numVertices_m, distanceGPUN, distanceCPU);

	//compute properties of the graph
	auto it = std::max_element(G.vertexDegree_m.begin(), G.vertexDegree_m.end());
	int maxDegree = *it;
	auto it2 = std::min_element(G.vertexDegree_m.begin(), G.vertexDegree_m.end());
	int minDegree = *it2;
	double avgDegree = ((double)std::accumulate(G.vertexDegree_m.begin(), G.vertexDegree_m.end(), 0)) / G.numVertices_m;
	writeToFile(G, minCPU, minGPU, minDegree, maxDegree, avgDegree);
	return 0;
}
