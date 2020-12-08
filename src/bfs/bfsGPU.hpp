/*
 * bfsGPU.cuh
 *
 *  Created on: 27-Nov-2020
 *      Author: abhishek
 */

#ifndef BFSGPU_HPP_
#define BFSGPU_HPP_
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <device_launch_parameters.h>

namespace bfsGPU {
	void initMemory(Graph &G, int source, std::vector<int> &distanceCheck);

	extern "C" {

	__global__ void parentKernelBfs(int *d_adjList, int *d_edgeOffsets,
			int *d_vertexDegree, int *d_distance, int *d_currQ, int *d_nextQ, int *d_nextQSize);

	__global__ void childKernelBfs(int depth, int *d_adjList, int *d_edgeOffsets,
			int *d_vertexDegree, int *d_distance, int *d_currQ, int currQSize,
			int *d_nextQ, int *d_nextQSize);

	}
	double execute(Graph &G, std::vector<int> &distanceCheck, int source = 0);

	void freeMemory();
}
#endif /* BFSGPU_HPP_ */
