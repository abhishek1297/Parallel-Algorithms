/*
 * dataIO.hpp
 *
 *  Created on: 20-Dec-2020
 *      Author: abhishek
 */

#ifndef DATAIO_HPP_
#define DATAIO_HPP_
#include <random>
#include <climits>
/**
 * boolean: to convert to zero based indexing since indices represent vertices
 * int: total inputs in a single of a file
 * vector<int>: which tokens are needed {from, to, weight} indices in that line
 * string: path to dataset
 */
struct DatasetInfo {

	bool toZeroIdx;
	int numInputs;
	std::vector<int> indexToRead;
	std::string fname;
};

extern struct DatasetInfo F[];
int writeToFile(std::string fname, int nV, int nE, double minCPU, double minGPU);



#endif /* DATAIO_HPP_ */
