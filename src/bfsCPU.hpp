/*
 * bfsCPU.hpp
 *
 *  Created on: 27-Nov-2020
 *      Author: abhishek
 */

#ifndef BFSCPU_HPP_
#define BFSCPU_HPP_
#include "graph.hpp"

void bfsCPU( int source, Graph &G, std::vector<int> &distance,
		std::vector<bool> &visited);

double execBfsCPU(Graph &G, int nV);

#endif /* BFSCPU_HPP_ */
