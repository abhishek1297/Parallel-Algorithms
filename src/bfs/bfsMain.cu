/*
 ============================================================================
 Name        : bfsMain.cu
 Author      : Abhishek Purandare
 Version     :
 Copyright   : AbhishekPurandare2020
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include "bfsMain.hpp"
#include<random>


int bfsMain () {

	const std::string datasetDir{"/home/abhishek/Downloads/graph datasets/"};
	std::vector<std::string> files{"test.txt", "roadNet-CA.txt", "p2p-Gnutella31.txt",
								   "youtube.txt", "live-journal.txt"};
	std::vector<int> distanceCPU, distanceGPU;
	bool convertToZeroIdx {false};
	Graph G(datasetDir + files[4], convertToZeroIdx);
	double cpu, gpu;
	std::cout << G << std::endl;
	std::cout << "CPU Time: " << (cpu = bfsCPU::execute(G, distanceCPU, 0)) << " ms" << std::endl;
	std::cout << "GPU Time: " << (gpu = bfsGPU::execute(G, distanceGPU, 0)) << " ms" << std::endl;
	std::cout << ((gpu < cpu) ? "GPU Faster": "CPU Faster") << std::endl;

	for (int i=0; i<20; ++i) {

		int u = rand() % G.numVertices_m;
		std::cout << distanceCPU[u] << " " << distanceGPU[u] << std::endl;
	}

	return 0;
}
