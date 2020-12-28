#include "bfsGPU.hpp"
#include <chrono>
#include <cstdio>
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
namespace bfsGPU {

	const int BLOCK_SIZE = 1024;
	const int BLOCK_QUEUE_SIZE = 32;
	const int SUB_QUEUE_SIZE = 4;
	const int NUM_SUB_QUEUES = 32;

	//device pointers
	int *d_adjList;
	int *d_edgeOffsets;
	int *d_vertexDegree;
	int *d_distance;
	int *d_parent;
	int *d_currQ;
	int *d_nextQ;
	int *d_nextQSize;
	texture<int, cudaTextureType1D, cudaReadModeElementType> tex_edgeOffsets;


	void initMemory(Graph &G, int source, std::vector<int> &distanceCheck) {

		//Allocation
		cudaMalloc(&d_adjList, G.numEdges_m * sizeof(int));
		cudaMalloc(&d_edgeOffsets, G.numVertices_m * sizeof(int));
		cudaMalloc(&d_vertexDegree, G.numVertices_m * sizeof(int));
		cudaMalloc(&d_distance, G.numVertices_m * sizeof(int));
		cudaMalloc(&d_currQ, G.numVertices_m * sizeof(int));
		cudaMalloc(&d_nextQ, G.numVertices_m * sizeof(int));
		cudaMalloc(&d_nextQSize, sizeof(int));
		//Data transfer
		cudaMemcpy(d_adjList, G.adjacencyList_m.data(), G.numEdges_m * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_edgeOffsets, G.edgeOffsets_m.data(), G.numVertices_m * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_vertexDegree, G.vertexDegree_m.data(), G.numVertices_m * sizeof(int), cudaMemcpyHostToDevice);
		//Init kernel parameters
		cudaMemcpy(d_currQ, &source, sizeof(int), cudaMemcpyHostToDevice);
		cudaMemset(d_nextQSize, 0, sizeof(int));
		//Init distance
		distanceCheck.resize(G.numVertices_m);
		std::fill(distanceCheck.begin(), distanceCheck.end(), -1);
		distanceCheck[source] = 0;
		cudaMemcpy(d_distance, distanceCheck.data(), G.numVertices_m * sizeof(int), cudaMemcpyHostToDevice);
		//texture reference assigned to the edge offsets
		size_t offset{0};
		cudaBindTexture(&offset, tex_edgeOffsets, d_currQ, G.numVertices_m * sizeof(int));
	}

	/**
	 * The parent kernel is similar to launching the kernel from the cpu.
	 * But in this case launching the workload from the gpu itself.
	 */
	/*
	extern "C"
	__global__ void hierarchical::parentKernel(int *d_adjList,
			int *d_edgeOffsets,
			int *d_vertexDegree,
			int *d_distance,
			int *d_currQ,
			int *d_nextQ,
			int *d_nextQSize) {

		int currQSize = 1;
		int dev_depth = 0;
		int numBlocks;
		while (currQSize) {

			numBlocks = ((currQSize - 1) / BLOCK_SIZE) + 1;
			childKernel<<<numBlocks, BLOCK_SIZE>>>(++dev_depth, d_adjList, d_edgeOffsets,
					d_vertexDegree, d_distance,
					d_currQ, currQSize, d_nextQ, d_nextQSize);

			cudaDeviceSynchronize(); // halt gpu
			currQSize = *d_nextQSize;
			cudaMemcpyAsync(d_currQ, d_nextQ, sizeof(int) * currQSize, cudaMemcpyDeviceToDevice);
			cudaMemsetAsync(d_nextQSize, 0, sizeof(int));

		}
	}

	double hierarchical::executeDP(Graph &G, std::vector<int> &distanceCheck, int source) {

		//initialize data
		initMemory(G, source, distanceCheck);
		//execution
		auto start = std::chrono::high_resolution_clock::now();
		parentKernel<<<1, 1>>>(d_adjList, d_edgeOffsets, d_vertexDegree, d_distance,
								  d_currQ, d_nextQ, d_nextQSize);
		cudaDeviceSynchronize();
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> t = end - start;
		//fill the distances obtained for comparison
		cudaMemcpy(distanceCheck.data(), d_distance, G.numVertices_m * sizeof(int), cudaMemcpyDeviceToHost);
		//free device pointers
		freeMemory();

		return t.count();
	}*/

	/**
	 *
	 * The kernel which performs the main operation of traversal over the graph.
	 */

	extern "C"
	__global__ void hierarchical::childKernel(int depth,
			int *d_adjList,
			int *d_edgeOffsets,
			int *d_vertexDegree,
			int *d_distance,
			int *d_currQ,
			int currQSize,
			int *d_nextQ,
			int *d_nextQSize) {

				/*
				 * d_variable is device allocated variable
				 * s_variable is shared memory variable
				 */
					__shared__ int s_subNextQ[NUM_SUB_QUEUES][SUB_QUEUE_SIZE], s_subNextQSize[NUM_SUB_QUEUES];
					__shared__ int s_globalOffsets[NUM_SUB_QUEUES];
					//registers
					int child, parent,
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
						subSharedQIdx = tIdx & (NUM_SUB_QUEUES - 1);
//						subSharedQIdx = (tIdx / SUB_QUEUE_SIZE) % NUM_SUB_QUEUES;

						//expand all children
						for (int i=d_edgeOffsets[parent]; i<d_edgeOffsets[parent]+d_vertexDegree[parent]; ++i) {

							child = d_adjList[i];
							if (atomicMin(&d_distance[child], INT_MAX) == -1) { // if not found

								d_distance[child] = depth;
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
						__syncthreads();
					}

					if (threadIdx.x < NUM_SUB_QUEUES) // offsets for sub queues to global memory
							s_globalOffsets[threadIdx.x] = atomicAdd(d_nextQSize, s_subNextQSize[threadIdx.x]);
						__syncthreads();

					/*for (int t=threadIdx.x; t<SUB_QUEUE_SIZE; t+=blockDim.x) {

							for (int i=0; i<NUM_SUB_QUEUES; ++i) {
								if (t < s_subNextQSize[i]) {
									d_nextQ[s_globalOffsets[i] + t] = s_subNextQ[i][t];
								}
							}
					}*/


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


	double hierarchical::execute(Graph &G, std::vector<int> &distanceCheck, int source) {

		//initialize data
		initMemory(G, source, distanceCheck);
		int currQSize{1};
		int depth{0}, numBlocks;
		//execution
		auto start = std::chrono::high_resolution_clock::now();
		while (currQSize) {
			numBlocks = ((currQSize - 1) / BLOCK_SIZE) + 1;
				childKernel<<<numBlocks, BLOCK_SIZE>>>(++depth, d_adjList, d_edgeOffsets,
						d_vertexDegree, d_distance,
						d_currQ, currQSize, d_nextQ, d_nextQSize);

				cudaDeviceSynchronize(); // halt gpu
				cudaMemcpyAsync(&currQSize, d_nextQSize, sizeof(int), cudaMemcpyDeviceToHost);
				cudaMemcpyAsync(d_currQ, d_nextQ, sizeof(int) * currQSize, cudaMemcpyDeviceToDevice);
				cudaMemsetAsync(d_nextQSize, 0, sizeof(int));
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> t = end - start;
		//fill the distances obtained for comparison
		cudaMemcpy(distanceCheck.data(), d_distance, G.numVertices_m * sizeof(int), cudaMemcpyDeviceToHost);
		//free device pointers
		freeMemory();

		return t.count();
	}

	extern "C"
	__global__ void blocked::kernelB(int depth, int *d_adjList, int *d_edgeOffsets,
				int *d_vertexDegree, int *d_distance, int *d_currQ, int currQSize,
				int *d_nextQ, int *d_nextQSize) {

			/*
			 * d_variable is device allocated variables
			 * s_variable is shared memory variable
			 */
			__shared__ int s_nextQ[BLOCK_QUEUE_SIZE];
			__shared__ int s_nextQSize, s_globalOffset;

			//obtain thread id
			int tIdx = threadIdx.x + blockIdx.x * blockDim.x;
			if (tIdx == 0) //only one thread needed to set the size.
				s_nextQSize = 0;
			__syncthreads();

			if (tIdx < currQSize) {

				const int parent = d_currQ[tIdx];//get current values in parallel

				for (int i=d_edgeOffsets[parent]; i<d_edgeOffsets[parent]+d_vertexDegree[parent]; ++i) {

					const int child = d_adjList[i];
					if (atomicMin(&d_distance[child], INT_MAX) == -1) {

						d_distance[child] = depth;
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
				s_globalOffset = atomicAdd(d_nextQSize, s_nextQSize);
			__syncthreads();

			if (threadIdx.x >= BLOCK_QUEUE_SIZE) return;
			for (int i=threadIdx.x; i<s_nextQSize; i+=blockDim.x) {// fill the global memory
				d_nextQ[s_globalOffset + i] = s_nextQ[i];
			}
		}

	double blocked::execute(Graph &G, std::vector<int> &distanceCheck, int source) {

		//initialize data
		initMemory(G, source, distanceCheck);
		int currQSize{1};
		int depth{0}, numBlocks;
		//execution
		auto start = std::chrono::high_resolution_clock::now();
		while (currQSize) {
			numBlocks = ((currQSize - 1) / BLOCK_SIZE) + 1;
				kernelB<<<numBlocks, BLOCK_SIZE>>>(++depth, d_adjList, d_edgeOffsets,
						d_vertexDegree, d_distance,
						d_currQ, currQSize, d_nextQ, d_nextQSize);

				cudaDeviceSynchronize(); // halt gpu
				cudaMemcpyAsync(&currQSize, d_nextQSize, sizeof(int), cudaMemcpyDeviceToHost);
				cudaMemcpyAsync(d_currQ, d_nextQ, sizeof(int) * currQSize, cudaMemcpyDeviceToDevice);
				cudaMemsetAsync(d_nextQSize, 0, sizeof(int));
		}
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> t = end - start;
		//fill the distances obtained for comparison
		cudaMemcpy(distanceCheck.data(), d_distance, G.numVertices_m * sizeof(int), cudaMemcpyDeviceToHost);
		//free device pointers
		freeMemory();

		return t.count();
	}

	extern "C"
	__global__ void naive::kernelN(int depth, int *d_adjList, int *d_edgeOffsets,
			int *d_vertexDegree, int *d_distance, int *d_currQ, int currQSize,
			int *d_nextQ, int *d_nextQSize) {

				int t = threadIdx.x + blockDim.x * blockIdx.x;
				if (t < currQSize) {
					int parent = d_currQ[t];
					for (int i=d_edgeOffsets[parent]; i<d_edgeOffsets[parent]+d_vertexDegree[parent]; ++i) {
						int child = d_adjList[i];
						if (atomicMin(&d_distance[child], INT_MAX) == -1) {
							d_distance[child] = depth;
							int idx = atomicAdd(d_nextQSize, 1);
							d_nextQ[idx] = child;
						}
					}
				}
				__syncthreads();
	}

	double naive::execute(Graph &G, std::vector<int> &distanceCheck, int source) {

		//initialize data
		initMemory(G, source, distanceCheck);
		int currQSize{1};
		int depth{0}, numBlocks;
		//execution
		auto start = std::chrono::high_resolution_clock::now();
		while (currQSize) {
			numBlocks = ((currQSize - 1) / BLOCK_SIZE) + 1;
				kernelN<<<numBlocks, BLOCK_SIZE>>>(++depth, d_adjList, d_edgeOffsets,
						d_vertexDegree, d_distance,
						d_currQ, currQSize, d_nextQ, d_nextQSize);

				cudaDeviceSynchronize(); // halt gpu
				cudaMemcpyAsync(&currQSize, d_nextQSize, sizeof(int), cudaMemcpyDeviceToHost);
				cudaMemcpyAsync(d_currQ, d_nextQ, sizeof(int) * currQSize, cudaMemcpyDeviceToDevice);
				cudaMemsetAsync(d_nextQSize, 0, sizeof(int));
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
		cudaFree(d_currQ);
		cudaFree(d_nextQ);
		cudaFree(d_nextQSize);
		cudaUnbindTexture(tex_edgeOffsets);
	}

}

/*
 *
 * Testing texture memory
extern "C" __global__ void kernelTex(int *d_edgeOffsets) {

		int x = threadIdx.x;

		printf("\ntid(%d) = %d %d", x, tex1Dfetch(tex_edgeOffsets, x * 5), d_edgeOffsets[x * 5]);
	}

	double executeTex(Graph &G, std::vector<int> &distanceCheck, int source) {
		//assigning edge offsets to texture memory
		size_t offset;
		int arr[10] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
		tex_edgeOffsets.addressMode[0] = cudaAddressModeWrap;
		tex_edgeOffsets.filterMode = cudaFilterModeLinear;
		tex_edgeOffsets.normalized = true;
		// Bind the array to the texture
		int *dev_ptr;
		cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();
		gpuErrchk(cudaMalloc(&dev_ptr, 10 * sizeof(int)));
		gpuErrchk(cudaMemcpy(dev_ptr, arr, sizeof(int) * 10, cudaMemcpyHostToDevice));

		gpuErrchk(cudaBindTexture(&offset, tex_edgeOffsets, dev_ptr, 10 * sizeof(int)));
		gpuErrchk(cudaBindTextureToArray(tex_edgeOffsets, dev_ptr, desc));

		initMemory(G, source, distanceCheck);
		kernelTex<<<1, 10>>>(d_edgeOffsets);
		cudaDeviceSynchronize();
		freeMemory();
		cudaUnbindTexture(tex_edgeOffsets);
		cudaFree(dev_ptr);
		printf("%s", cudaGetErrorString(cudaGetLastError()));
	}
*/
