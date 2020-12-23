#ifndef BFSCPU_HPP_
#define BFSCPU_HPP_
#include "../graph.hpp"
namespace bfsCPU {
	double execute(Graph &G, std::vector<int> &distanceCheck,int source = 0);
}
#endif /* BFSCPU_HPP_ */
