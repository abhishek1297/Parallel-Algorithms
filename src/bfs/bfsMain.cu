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


int bfsMain () {

	const std::string datasetDir{"/home/abhishek/Downloads/graph datasets/"};
	std::vector<std::string> files{"test.txt", "roadNet-CA.txt", "p2p-Gnutella31.txt",
								   "youtube.txt", "live-journal.txt"};
	std::vector<int> distanceCPU, distanceGPU;
	bool convertToZeroIdx {true};
	Graph G(datasetDir + files[3], convertToZeroIdx);
	double cpu, gpu;
	std::cout << G << std::endl;
	std::cout << "CPU Time: " << (cpu = bfsCPU::execute(G, distanceCPU, 0)) << " ms" << std::endl;
	std::cout << "GPU Time: " << (gpu = bfsGPU::execute(G, distanceGPU, 0)) << " ms" << std::endl;
	std::cout << ((gpu < cpu) ? "GPU Faster": "CPU Faster") << std::endl;

	return 0;
}
