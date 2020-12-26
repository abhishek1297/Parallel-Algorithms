#include "main.hpp"
#include "../dataIO.hpp"


int bfsMain (Graph&& G) {

	std::vector<int> distanceCPU, distanceGPUH, distanceGPUN, distanceGPUB;
	double minCpu, minGpuB{INT_MAX}, minGpuN{INT_MAX}, minGpuH{INT_MAX}, minGpuDPH{INT_MAX};

//	minCpu = bfsCPU::execute(G, distanceCPU, 0);

	for (int i=0; i<10; ++i) {
		minGpuH = std::min(minGpuH, bfsGPU::hierarchical::execute(G, distanceGPUH, 0));
		minGpuN = std::min(minGpuN, bfsGPU::naive::execute(G, distanceGPUN, 0));
//		minGpuB = std::min(minGpuB, bfsGPU::blocked::execute(G, distanceGPUB, 0));
	}

	printf("\n%s\n\n", cudaGetErrorString(cudaGetLastError()));

//	std::cout << "CPU Time: " << minCpu << " ms" << std::endl;
	std::cout << "GPU Naive Time: " << minGpuN << " ms" << std::endl;
	std::cout << "GPU Normal Hierarchical Time: " << minGpuH << " ms" << std::endl;
//	std::cout << "GPU Blocked Time: " << minGpuB << " ms" << std::endl;
	for (int i=1; i<=100; ++i) {//checking random distances

		int u = rand() % G.numVertices_m;
		std::cout << (distanceGPUN[u] == distanceGPUH[u]);
		if (i % 10 == 0) std::cout << std::endl;
	}
//	writeToFile(G.pathData_m, G.numVertices_m, G.numEdges_m, minCpu, minGpuN);

	return 0;
}
