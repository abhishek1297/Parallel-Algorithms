#include "graph.hpp"
#include "bfsGPU.hpp"
#include <chrono>
#include <iostream>

namespace bfsGPU {

	const int BLOCK_SIZE = 1024;
//	const int BLOCK_QUEUE_SIZE = 128;
	const int SUB_QUEUE_LEN = 32;
	const int NUM_SUB_QUEUES = 4;
//	const int CORES_PER_SM = 128; //CUDA cores per SM

	//device pointers
	int *d_adjList;
	int *d_edgeOffsets;
	int *d_vertexDegree;
	int *d_distance;
	int *d_parent;
	int *d_currQ;
	int *d_nextQ;
	int *d_currQSize;
	int *d_nextQSize;


	void initMemory(Graph &G, int source, std::vector<int> &distanceCheck) {

		//Allocation
		cudaMalloc(&d_adjList, G.numEdges_m * sizeof(int));
		cudaMalloc(&d_edgeOffsets, G.numVertices_m * sizeof(int));
		cudaMalloc(&d_vertexDegree, G.numVertices_m * sizeof(int));
		cudaMalloc(&d_distance, G.numVertices_m * sizeof(int));
		cudaMalloc(&d_currQ, G.numVertices_m * sizeof(int));
		cudaMalloc(&d_nextQ, G.numVertices_m * sizeof(int));
		cudaMalloc(&d_currQSize, sizeof(int));
		cudaMalloc(&d_nextQSize, sizeof(int));
		//Data transfer
		cudaMemcpy(d_adjList, G.adjacencyList_m.data(), G.numEdges_m * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_edgeOffsets, G.edgeOffsets_m.data(), G.numVertices_m * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_vertexDegree, G.vertexDegree_m.data(), G.numVertices_m * sizeof(int), cudaMemcpyHostToDevice);
		//Init kernel parameters
		cudaMemcpy(d_currQ, &source, sizeof(int), cudaMemcpyHostToDevice);
		const int currQSize = 1;
		cudaMemcpy(d_currQSize, &currQSize, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemset(d_nextQSize, 0, sizeof(int));
		//Init distance
		distanceCheck.resize(G.numVertices_m);
		std::fill(distanceCheck.begin(), distanceCheck.end(), -1);
		distanceCheck[source] = 0;
		cudaMemcpy(d_distance, distanceCheck.data(), G.numVertices_m * sizeof(int), cudaMemcpyHostToDevice);

	}

	extern "C" __global__ void kernelBfs(int depth,
										 int *d_adjList,
										 int *d_edgeOffsets,
										 int *d_vertexDegree,
										 int *d_distance,
										 int *d_currQ,
										 int *d_currQSize,
										 int *d_nextQ,
										 int *d_nextQSize) {

			/*
			 * d_variable is device allocated variables
			 * s_variable is shared memory variable
			 */
				__shared__ int s_subNextQ[NUM_SUB_QUEUES][SUB_QUEUE_LEN], s_subNextQSize[NUM_SUB_QUEUES];
				__shared__ int s_globalOffsets[NUM_SUB_QUEUES];
				//registers
				int child, parent,
				subSharedQIdx /*which row of queue < NUM_SUB_QUEUES */,
				subSharedQSize/*length of a sub queue to be incremented < SUB_QUEUE_LEN */,
				globalQIdx /*global level queue idx < |V| */;
				//obtain thread id
				int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
				if (threadIdx.x < NUM_SUB_QUEUES) //only one thread needed to set the size.
					s_subNextQSize[threadIdx.x] = 0;
				__syncthreads();

				if (tIdx < *d_currQSize) {

					parent = d_currQ[tIdx];//get current values in parallel
					subSharedQIdx = threadIdx.x & (NUM_SUB_QUEUES - 1);

					for (int i=d_edgeOffsets[parent]; i<d_edgeOffsets[parent]+d_vertexDegree[parent]; ++i) {

						child = d_adjList[i];
						if (atomicMin(&d_distance[child], INT_MAX) == -1) {

							d_distance[child] = depth + 1;
							// Increment sub queue size
							subSharedQSize = atomicAdd(&s_subNextQSize[subSharedQIdx], 1);
							if (subSharedQSize < SUB_QUEUE_LEN) {

								s_subNextQ[subSharedQIdx][subSharedQSize] = child;
							}
							else {

								s_subNextQSize[subSharedQIdx] = SUB_QUEUE_LEN;
								globalQIdx = atomicAdd(d_nextQSize, 1);
								d_nextQ[globalQIdx] = child;
							}
						}
					}
				}
				__syncthreads();

				if (threadIdx.x < NUM_SUB_QUEUES) // offsets for sub queues to global memory
					s_globalOffsets[threadIdx.x] = atomicAdd(d_nextQSize, s_subNextQSize[threadIdx.x]);
				__syncthreads();

				for (int t=threadIdx.x; t<SUB_QUEUE_LEN; t+=blockDim.x) {

					for (int i=0; i<NUM_SUB_QUEUES; ++i) {
						if (s_subNextQSize[i] != 0) {
							d_nextQ[s_globalOffsets[i] + t] = s_subNextQ[i][t];
						}
					}
				}
	}

	double execute(Graph &G, std::vector<int> &distanceCheck, int source) {

		//initialize data
		initMemory(G, source, distanceCheck);
		//execution

		int h_currQSize{1};
		int numBlocks{0}, depth{0};

		auto start = std::chrono::high_resolution_clock::now();
		while (h_currQSize) {

			numBlocks = ((h_currQSize - 1) / BLOCK_SIZE) + 1;
			kernelBfs<<<numBlocks, BLOCK_SIZE>>>(depth, d_adjList, d_edgeOffsets, d_vertexDegree, d_distance,
					d_currQ, d_currQSize, d_nextQ, d_nextQSize);
			cudaDeviceSynchronize(); // halt cpu
			std::swap(d_currQ, d_nextQ);
			cudaMemcpy(d_currQSize, d_nextQSize, sizeof(int), cudaMemcpyDeviceToDevice);
			cudaMemset(d_nextQSize, 0, sizeof(int));
			cudaMemcpy(&h_currQSize, d_currQSize, sizeof(int), cudaMemcpyDeviceToHost);
			++depth;

		}

		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> t = end - start;
		//fill the distances obtained for comparison
		cudaMemcpy(distanceCheck.data(), d_distance, G.numVertices_m * sizeof(int), cudaMemcpyDeviceToHost);
		//free device pointers
		freeMemory();

		return t.count();
	}

	void freeMemory(){

		cudaFree(d_adjList);
		cudaFree(d_edgeOffsets);
		cudaFree(d_vertexDegree);
		cudaFree(d_distance);
		cudaFree(d_parent);
		cudaFree(d_currQ);
		cudaFree(d_nextQ);
		cudaFree(d_currQSize);
		cudaFree(d_nextQSize);
	}
}
