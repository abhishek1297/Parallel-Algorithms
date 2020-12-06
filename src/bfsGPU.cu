#include "graph.hpp"
#include "bfsGPU.hpp"
#include <chrono>
#include <iostream>

#define BLOCK_SIZE 1024
#define BLOCK_QUEUE_SIZE 64
#define WARP_SIZE 32
#define NUM_SUB_QUEUES 4
#define NUM_SP 128 //CUDA cores per SM
/*
CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction cuBFS;

#define CHECK_CUDA_RESULT(N) {											\
	CUresult result = N;												\
	if (result != 0) {													\
		printf("CUDA call on line %d returned error %d\n", __LINE__,	\
			result);													\
		exit(1);														\
	}																	\
}
*/


int * d_adjList;
int * d_edgeOffsets;
int * d_vertexDegree;
int * d_distance;
int * d_parent;
int * d_currQ;
int * d_nextQ;
int * d_currQSize;
int * d_nextQSize;



namespace bfsGPU {

	void initMemory(Graph &G, int source, std::vector<int> &distanceCheck) {

		//Initialize
		/*
		CHECK_CUDA_RESULT(cuInit(0));
		CHECK_CUDA_RESULT(cuDeviceGet(&cuDevice, 0));
		CHECK_CUDA_RESULT(cuCtxCreate_v2(&cuContext, 0, cuDevice));
		CHECK_CUDA_RESULT(cuModuleLoad(&cuModule, "bfsGPU.ptx"));
		CHECK_CUDA_RESULT(cuModuleGetFunction(&cuBFS, cuModule, "kernelBfs"));*/
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

		distanceCheck.resize(G.numVertices_m);
		std::fill(distanceCheck.begin(), distanceCheck.end(), -1);
		distanceCheck[source] = 0;
		cudaMemcpy(d_distance, distanceCheck.data(), G.numVertices_m * sizeof(int), cudaMemcpyHostToDevice);

	}

	extern "C" __global__ void kernelBfs(int depth, int *d_adjList, int *d_edgeOffsets,
				int *d_vertexDegree, int *d_distance, int *d_currQ, int *d_currQSize,
				int *d_nextQ, int *d_nextQSize) {

			/*
			 * d_variable is device allocated variables
			 * s_variable is shared memory variable
			 */
/*			int t = threadIdx.x + blockDim.x * blockIdx.x;

			if (t < *d_currQSize) {

				int parent = d_currQ[t];
				for (int i=d_edgeOffsets[parent]; i<d_edgeOffsets[parent]+d_vertexDegree[parent]; ++i) {

					int child = d_adjList[i];
					if (atomicMin(&d_distance[child], INT_MAX) == -1) {
						d_distance[child] = depth + 1;
						int idx = atomicAdd(d_nextQSize, 1);
						d_nextQ[idx] = child;
					}
				}
			}
			__syncthreads();*/
//			__shared__ int s_nextQ[BLOCK_QUEUE_SIZE];
			__shared__ int s_subNextQ[][], s_subQId;
			__shared__ int s_nextQSize, s_blockGlobalQIdx;

			//obtain thread id
			int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
			if (tIdx == 0) //only one thread needed to set the size.
				s_nextQSize = 0;
			__syncthreads();

			if (tIdx < *d_currQSize) {

				const int parent = d_currQ[tIdx];//get current values in parallel

				for (int i=d_edgeOffsets[parent]; i<d_edgeOffsets[parent]+d_vertexDegree[parent]; ++i) {

					const int child = d_adjList[i];
					if (atomicMin(&d_distance[child], INT_MAX) == -1) {

						d_distance[child] = depth + 1;
						const int sharedQIdx = atomicAdd(&s_nextQSize, 1);

						if (sharedQIdx < BLOCK_QUEUE_SIZE) { //if the shared memory is not full, fill the shared queue

							s_nextQ[sharedQIdx] = child;
						}
						else { //fill the global queue

							s_nextQSize = BLOCK_QUEUE_SIZE;
							const int globalQIdx = atomicAdd(d_nextQSize, 1);
							d_nextQ[globalQIdx] = child;
						}
					}
				}
			}
			__syncthreads();

			if (threadIdx.x == 0) //offset for global memory
				s_blockGlobalQIdx = atomicAdd(d_nextQSize, s_nextQSize);
			__syncthreads();

			for (int i=threadIdx.x; i<s_nextQSize; i+=blockDim.x) {// fill the global memory
				d_nextQ[s_blockGlobalQIdx + i] = s_nextQ[i];
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
			depth ++;

		}

		auto end = std::chrono::high_resolution_clock::now();

//		std::cout << cudaGetErrorString(cudaGetLastError())<< std::endl;
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
