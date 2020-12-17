/*
 * SSSPCPU.hpp
 *
 *  Created on: 14-Dec-2020
 *      Author: abhishek
 */

#ifndef SSSPCPU_HPP_
#define SSSPCPU_HPP_

#include "../graph.hpp"
#include <climits>
#include <chrono>
#include <algorithm>


namespace ssspCPU {
	double execute(Graph &G, std::vector<int> &distance, int source = 0);
}

#endif /* SSSPCPU_HPP_ */
