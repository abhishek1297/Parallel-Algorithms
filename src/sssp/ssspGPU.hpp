#ifndef SSSPGPU_HPP_
#define SSSPGPU_HPP_
#include "../graph.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace ssspGPU {
	void initMemory(Graph &G, int source, std::vector<int> &distanceCheck);

	extern "C" {

	__global__ void parentKernelSssp(int *d_adjList, int *d_edgeWeights, int *d_edgeOffsets, int *d_vertexDegree,
			int *d_distance, int *d_currQ, int *d_nextQ, int *d_nextQSize);

	__global__ void childKernelSssp(int *d_adjList, int *d_edgeWeights, int *d_edgeOffsets,
			int *d_vertexDegree, int *d_distance, int *d_currQ, int currQSize,
			int *d_nextQ, int *d_nextQSize);

	}
	double execute(Graph &G, std::vector<int> &distance, int source = 0);

	void freeMemory();
}



#endif /* SSSPGPU_HPP_ */
