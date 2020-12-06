/*
 ============================================================================
 Name        : main.cu
 Author      : Abhishek Purandare
 Version     :
 Copyright   : AbhishekPurandare2020
 Description : CUDA compute reciprocals
 ============================================================================
 */

#include "graph.hpp"
#include "bfsCPU.hpp"
#include "bfsGPU.hpp"
#include <iostream>

bool evaluate (int nV, std::vector<int> &&v1, std::vector<int> &&v2) {

	for (int i=0;i<nV; ++i) {
		if (v1[i] != v2[i])
			return false;
	}
	return true;
}

int main () {
	std::cout.flush();
	const std::string datasetDir{"/home/abhishek/Downloads/graph datasets/"};
	std::vector<std::string> files{"test.txt", "roadNet-CA.txt", "p2p-Gnutella31.txt", "youtube.txt"};
	std::vector<int> distanceCPU, distanceGPU;

	Graph G(datasetDir + files[1]);
	double cpu, gpu;
	std::cout << "CPU Time: " << (cpu = bfsCPU::execute(G, distanceCPU, 0)) << " ms" << std::endl;
	std::cout << "GPU Time: " << (gpu = bfsGPU::execute(G, distanceGPU, 0)) << " ms" << std::endl;
	std::cout << ((gpu < cpu) ? "GPU Faster": "CPU Faster") << std::endl;
	std::cout << "Distances are " << std::boolalpha << evaluate(G.numVertices_m, std::move(distanceCPU), std::move(distanceGPU));

	return 0;
}
