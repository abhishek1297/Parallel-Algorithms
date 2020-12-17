#include "ssspCPU.hpp"
#include "ssspGPU.hpp"

int main () {
	const std::string datasetDir{"/home/abhishek/Downloads/graph datasets/"};

	std::vector<int> distanceCPU, distanceGPU;
	std::string files[] {"USA-road-d.CAL.gr", "test.txt"};
	Graph G(datasetDir + files[0], 4, {1, 2, 3}, true, "sssp");
	std::cout << G << std::endl;
	std::cout << "Executing...\n";
//	std::cout  << "CPU Time: " << ssspCPU::execute(G, distanceCPU) << " ms" << std::endl;
	std::cout  << "GPU Time: " << ssspGPU::execute(G, distanceGPU) << " ms" << std::endl;


	for (int i=0; i<10; i++) {
		std::cout << /*distanceCPU[i] << " " <<*/ distanceGPU[i] << std::endl;
	}
}
