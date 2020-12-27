#include "main.hpp"
#include <random>
int ssspMain (Graph&& G) {

	std::vector<int> distanceCPU, distanceGPU;
	std::cout << "Executing...\n";
	std::cout  << "CPU Time: " << ssspCPU::execute(G, distanceCPU) << " ms" << std::endl;
	std::cout  << "GPU Time: " << ssspGPU::execute(G, distanceGPU) << " ms" << std::endl;


	for (int i=1; i<=100; ++i) {//checking random distances

		int u = rand() % G.numVertices_m;
		std::cout << (distanceCPU[u] == distanceGPU[u]);
		if (i % 10 == 0) std::cout << std::endl;
	}
}
