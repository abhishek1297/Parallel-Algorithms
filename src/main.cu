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
//#include "bfsCPU.hpp"
#include "bfsGPU.hpp"
#include <iostream>

int main () {
	std::cout.flush();
	const std::string datasetDir{"/home/abhishek/Downloads/graph datasets/"};
	std::vector<std::string> files{"roadNet-CA.txt", "p2p-Gnutella31.txt", "youtube.txt"};
	Graph G(datasetDir + "test.txt");
	std::vector<int> distanceCPU, distanceGPU;

//	bfsCPU::execute(G, distanceCPU, 0);
	bfsGPU::execute(G, distanceGPU, 0);
	return 0;
}
