#include "main.hpp"
#include "../dataIO.hpp"


int bfsMain (Graph&& G) {

	std::vector<int> distanceCPU, distanceGPUH, distanceGPUN, distanceGPUB;
	double cpu, gpu, minCpu{INT_MAX}, minGpu{INT_MAX};

	std::cout << "GPU Hierarchical Time: " << bfsGPU::hierarchical::execute(G, distanceGPUH, 0) << " ms" << std::endl;
	std::cout << "GPU Naive Time: " << bfsGPU::naive::execute(G, distanceGPUN, 0) << " ms" << std::endl;
	std::cout << "GPU Blocked Time: " << bfsGPU::blocked::execute(G, distanceGPUB, 0) << " ms" << std::endl;
//	for (int i=1; i<=100; ++i) {//checking random distances
//
//		int u = rand() % G.numVertices_m;
//		std::cout << (distanceGPUN[u] == distanceGPUH[u]);
//		if (i % 10 == 0) std::cout << std::endl;
//	}

//	writeToFile(G.pathData_m, G.numVertices_m, G.numEdges_m, minCpu, minGpu);

	return 0;
}
