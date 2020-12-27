#include "ssspGPU.hpp"
#include <chrono>
#include <climits>

namespace ssspGPU {

	const int BLOCK_SIZE = 1024;
	const int SUB_QUEUE_SIZE = 4;
	const int NUM_SUB_QUEUES = 32;

	//device pointers
	int *d_adjList;
	int *d_edgeWeights;
	int *d_edgeOffsets;
	int *d_vertexDegree;
	int *d_distance;
	int *d_parent;
	int *d_currQ;
	int *d_nextQ;
	int *d_nextQSize;

	struct sortByWeights {
		int *distance;
		__device__
		sortByWeights(int *distance_): distance(distance_) {}
		__device__
		bool operator ()(const int &x, const int &y) const {
			return distance[x] < distance[y];
		}
	};

	void initMemory(Graph &G, int source, std::vector<int> &distanceCheck) {

		//Allocation
		cudaMalloc(&d_adjList, G.numEdges_m * sizeof(int));
		cudaMalloc(&d_edgeWeights, G.numEdges_m * sizeof(int));
		cudaMalloc(&d_edgeOffsets, G.numVertices_m * sizeof(int));
		cudaMalloc(&d_vertexDegree, G.numVertices_m * sizeof(int));
		cudaMalloc(&d_distance, G.numVertices_m * sizeof(int));
		cudaMalloc(&d_currQ, G.numVertices_m * sizeof(int));
		cudaMalloc(&d_nextQ, G.numVertices_m * sizeof(int));
		cudaMalloc(&d_nextQSize, sizeof(int));
		//Data transfer
		cudaMemcpy(d_adjList, G.adjacencyList_m.data(), G.numEdges_m * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_edgeWeights, G.edgeWeights_m.data(), G.numEdges_m * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_edgeOffsets, G.edgeOffsets_m.data(), G.numVertices_m * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_vertexDegree, G.vertexDegree_m.data(), G.numVertices_m * sizeof(int), cudaMemcpyHostToDevice);
		//Init kernel parameters
		cudaMemcpy(d_currQ, &source, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemset(d_nextQSize, 0, sizeof(int));
		//Init distance
		distanceCheck.resize(G.numVertices_m);
		std::fill(distanceCheck.begin(), distanceCheck.end(), INT_MAX);
		distanceCheck[source] = 0;
		cudaMemcpy(d_distance, distanceCheck.data(), G.numVertices_m * sizeof(int), cudaMemcpyHostToDevice);

	}


	extern "C"
	__global__ void childKernelSssp(int *d_adjList,
			int *d_edgeWeights,
			int *d_edgeOffsets,
			int *d_vertexDegree,
			int *d_distance,
			int *d_currQ,
			int currQSize,
			int *d_nextQ,
			int *d_nextQSize) {

				/*
				 * d_variable is device allocated variables
				 * s_variable is shared memory variable
				 */
					__shared__ int s_subNextQ[NUM_SUB_QUEUES][SUB_QUEUE_SIZE], s_subNextQSize[NUM_SUB_QUEUES];
					__shared__ int s_globalOffsets[NUM_SUB_QUEUES];
					//registers
					int child, parent, wt,
					subSharedQIdx /*which row of queue < NUM_SUB_QUEUES */,
					subSharedQSize/*length of a sub queue to be incremented < SUB_QUEUE_SIZE */,
					globalQIdx /*global level queue idx < |V| */;
					//obtain thread id
					int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
					if (threadIdx.x < NUM_SUB_QUEUES) //only one thread needed to set the size.
						s_subNextQSize[threadIdx.x] = 0;
					__syncthreads();

					if (tIdx < currQSize) {

						parent = d_currQ[tIdx];//get current values in parallel
						subSharedQIdx = threadIdx.x & (NUM_SUB_QUEUES - 1);

						for (int i=d_edgeOffsets[parent]; i<d_edgeOffsets[parent]+d_vertexDegree[parent]; ++i) {

							child = d_adjList[i];
							wt = d_distance[child];
							atomicMin(&d_distance[child], d_edgeWeights[i] + d_distance[parent]);
							//if the node not visited or the key is decreased
							if (wt == INT_MAX || wt != d_distance[child]) {
								// Increment sub queue size
								subSharedQSize = atomicAdd(&s_subNextQSize[subSharedQIdx], 1);
								if (subSharedQSize < SUB_QUEUE_SIZE) {

									s_subNextQ[subSharedQIdx][subSharedQSize] = child;
								}
								else {

									s_subNextQSize[subSharedQIdx] = SUB_QUEUE_SIZE;
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

					for (int t=threadIdx.x; t<NUM_SUB_QUEUES*SUB_QUEUE_SIZE; t+=blockDim.x) {

						//row-major ordering lucky i guess
						const int row = t / SUB_QUEUE_SIZE;
						if (s_subNextQSize[row] == 0) continue;
						const int col = t % SUB_QUEUE_SIZE;
						int lim = (SUB_QUEUE_SIZE * row) + s_subNextQSize[row];
						if (t < lim)
							d_nextQ[s_globalOffsets[row] + col] = s_subNextQ[row][col];
					}
	}

	double execute(Graph &G, std::vector<int> &distance, int source) {

		//initialize data
		initMemory(G, source, distance);
		//execution
		int currQSize {1};
		auto start = std::chrono::high_resolution_clock::now();
		while (currQSize) {

				int numBlocks = ((currQSize - 1) / BLOCK_SIZE) + 1;
				childKernelSssp<<<numBlocks, BLOCK_SIZE>>>(d_adjList, d_edgeWeights, d_edgeOffsets,
						d_vertexDegree, d_distance,
						d_currQ, currQSize, d_nextQ, d_nextQSize);

				cudaDeviceSynchronize(); // halt gpu
				cudaMemcpyAsync(&currQSize, d_nextQSize, sizeof(int), cudaMemcpyDeviceToHost);
				cudaMemcpyAsync(d_currQ, d_nextQ, sizeof(int) * currQSize, cudaMemcpyDeviceToDevice);
				cudaMemsetAsync(d_nextQSize, 0, sizeof(int));
				//thrust::sort(thrust::device, d_currQ, d_currQ + currQSize, sortByWeights(d_distance));
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> t = end - start;
		//fill the distances obtained for comparison
		cudaMemcpy(distance.data(), d_distance, G.numVertices_m * sizeof(int), cudaMemcpyDeviceToHost);
		//free device pointers
		freeMemory();

		return t.count();
	}

	void freeMemory(){

		cudaFree(d_adjList);
		cudaFree(d_edgeOffsets);
		cudaFree(d_vertexDegree);
		cudaFree(d_distance);
		cudaFree(d_currQ);
		cudaFree(d_nextQ);
		cudaFree(d_nextQSize);
	}

}
