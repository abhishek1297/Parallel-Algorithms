#include "graph.hpp"
#include "bfsGPU.hpp"


#define BLOCK_SIZE 1024
#define BLOCK_QUEUE_SIZE 1024
#define WARP_SIZE 32
#define NUM_SUB_QUEUES 4

CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction cuBFS;

CUdeviceptr d_adjList;
CUdeviceptr d_edgeOffsets;
CUdeviceptr d_vertexDegree;
CUdeviceptr d_distance;
CUdeviceptr d_parent;
CUdeviceptr d_currQ;
CUdeviceptr d_nextQ;
CUdeviceptr d_currQSize;
CUdeviceptr d_nextQSize;

#define CHECK_CUDA_RESULT(N) {											\
	CUresult result = N;												\
	if (result != 0) {													\
		printf("CUDA call on line %d returned error %d\n", __LINE__,	\
			result);													\
		exit(1);														\
	}																	\
}

namespace bfsGPU {

	void initMemory(Graph &G, int source, std::vector<int> &distanceCheck) {

		//Initialize
		CHECK_CUDA_RESULT(cuInit(0));
		CHECK_CUDA_RESULT(cuDeviceGet(&cuDevice, 1));
		CHECK_CUDA_RESULT(cuCtxCreate(&cuContext, 0, cuDevice));
		CHECK_CUDA_RESULT(cuModuleLoad(&cuModule, "bfsGPU"));
		CHECK_CUDA_RESULT(cuModuleGetFunction(&cuBFS, cuModule, "kernelBfs"));
		//Allocation
		CHECK_CUDA_RESULT(cuMemAlloc(&d_adjList, G.numEdges_m * sizeof(int)));
		CHECK_CUDA_RESULT(cuMemAlloc(&d_edgeOffsets, G.numVertices_m * sizeof(int)));
		CHECK_CUDA_RESULT(cuMemAlloc(&d_vertexDegree, G.numVertices_m * sizeof(int)));
		CHECK_CUDA_RESULT(cuMemAlloc(&d_distance, G.numVertices_m * sizeof(int)));
		CHECK_CUDA_RESULT(cuMemAlloc(&d_currQ, G.numVertices_m * sizeof(int)));
		CHECK_CUDA_RESULT(cuMemAlloc(&d_nextQ, G.numVertices_m * sizeof(int)));
		CHECK_CUDA_RESULT(cuMemAlloc(&d_currQSize, sizeof(int)));
		CHECK_CUDA_RESULT(cuMemAlloc(&d_nextQSize, sizeof(int)));
		//Data transfer
		CHECK_CUDA_RESULT(cuMemcpyHtoD(d_adjList, G.adjacencyList_m.data(), G.numEdges_m * sizeof(int)));
		CHECK_CUDA_RESULT(cuMemcpyHtoD(d_edgeOffsets, G.edgeOffsets_m.data(), G.numVertices_m * sizeof(int)));
		CHECK_CUDA_RESULT(cuMemcpyHtoD(d_vertexDegree, G.vertexDegree_m.data(), G.numVertices_m * sizeof(int)));

		distanceCheck.resize(G.numVertices_m);
		std::fill(distanceCheck.begin(), distanceCheck.end(), -1);
		distanceCheck[source] = 0;
		CHECK_CUDA_RESULT(cuMemcpyHtoD(d_distance, distanceCheck.data(), G.numVertices_m * sizeof(int)));
	}
	extern "C" {
		__global__ void kernelBfs(int depth, int *d_adjList, int *d_edgeOffsets,
				int *d_vertexDegree, int *d_distance, int *d_currQ, int *d_currQSize,
				int *d_nextQ, int *d_nextQSize) {

			/*
			 * d_variable is device allocated variables
			 * s_variable is shared memory variable
			 */

			__shared__ int s_nextQ[BLOCK_SIZE];
			__shared__ int s_nextQSize, s_blockGlobalQIdx;

			//obtain thread id
			int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
			if (tIdx == 0) //only one thread needed to set the size.
				s_nextQSize = 0;
			__syncthreads();

			if (tIdx < *d_currQSize) {

				const int parent = d_currQ[tIdx];//get frontier front value

				for (int i=d_edgeOffsets[parent]; i<d_edgeOffsets[parent]+d_vertexDegree[parent]; ++i) {

					const int child = d_adjList[i];
					const int oldDist = atomicMin(&d_distance[child], INT_MAX); //returns the old distance
					if (oldDist == -1) {

						d_distance[child] = d_distance[parent] + 1;
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
	}

	void execute(Graph &G, std::vector<int> &distanceCheck, int source) {

		//initialize data
		initMemory(G, source, distanceCheck);
		//execution

		int depth{0}, h_currQSize{1};
		int numBlocks{0};

		while (h_currQSize) {

			numBlocks = ((h_currQSize - 1) / BLOCK_SIZE) + 1;

			void *args[] = {&depth, &d_adjList, &d_edgeOffsets, &d_vertexDegree, &d_distance,
													&d_currQ, &d_currQSize, &d_nextQ, &d_nextQSize};
			CHECK_CUDA_RESULT(cuLaunchKernel(cuBFS, numBlocks, 1, 1, BLOCK_SIZE, 1, 1, 0, 0, args, 0));
			cuCtxSynchronize(); // halt cpu
			std::swap(d_currQ, d_nextQ);
			CHECK_CUDA_RESULT(cuMemcpyDtoD(d_currQSize, d_nextQSize, sizeof(int)));
			CHECK_CUDA_RESULT(cuMemsetD32_v2(d_nextQSize, 0, sizeof(int)));
			CHECK_CUDA_RESULT(cuMemcpyDtoH(&h_currQSize, d_currQSize, sizeof(int)));

			depth++;
		}

		//fill the distances obtained for comparison
		CHECK_CUDA_RESULT(cuMemcpyDtoH(distanceCheck.data(), d_distance, G.numVertices_m * sizeof(int)));
		//free device pointers
		freeMemory();

	}

	void freeMemory(){

		CHECK_CUDA_RESULT(cuMemFree(d_adjList));
		CHECK_CUDA_RESULT(cuMemFree(d_edgeOffsets));
		CHECK_CUDA_RESULT(cuMemFree(d_vertexDegree));
		CHECK_CUDA_RESULT(cuMemFree(d_distance));
		CHECK_CUDA_RESULT(cuMemFree(d_parent));
		CHECK_CUDA_RESULT(cuMemFree(d_currQ));
		CHECK_CUDA_RESULT(cuMemFree(d_nextQ));
		CHECK_CUDA_RESULT(cuMemFree(d_currQSize));
		CHECK_CUDA_RESULT(cuMemFree(d_nextQSize));
	}
}
