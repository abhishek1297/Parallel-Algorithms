/*
 * bfsCPU.hpp
 *
 *  Created on: 27-Nov-2020
 *      Author: abhishek
 */

#ifndef BFSCPU_HPP_
#define BFSCPU_HPP_
#include "../graph.hpp"
namespace bfsCPU {
	void bfsUtil( int source, Graph &G, std::vector<int> &distance);

	double execute(Graph &G, std::vector<int> &distanceCheck,int source = 0);
}
#endif /* BFSCPU_HPP_ */
