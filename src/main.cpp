#include "graph.hpp"
#include "bfsCPU.hpp"
#include <iostream>

//int evaluate() {
//
//	std::cout << "(V, E)\t\tTime(us)" << std::endl;
//
////	for ()
//
//}

int main(int argc, char *argv[]) {

	const std::string datasetDir{"/home/abhishek/cuda-workspace/Algorithms/src/graph datasets/"};
	std::vector<std::string> files;
	files.push_back("p2p-Gnutella04_adj.tsv");
	files.push_back("roadNet-CA.tsv");


	Graph G(datasetDir + files[1]);
	return 0;
}
