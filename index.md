# Background
During the recent decade, the Graphics Processing Unit (GPU) has become an important aspect of computation and not just in the field of Gaming. GPU computing is the use of GPUs to accelerate general-purpose tasks which are executed on the CPU. GPU is used when there are applications which use computer-intensive time-consuming operations.  From user-level abstraction, it is simply an application which runs faster with help of parallel processing. This advantage of having a massive number of cores can help make applications more efficient and seamless.

When it comes to Graph, It is one of the most important data structures which can represent real-world scenarios such as Network Graphs in Routing, Flight Networks, Pathfinding for Navigation systems, etc. Evidently, we are talking about millions of vertices that need to pass information back and forth through their links. These large connections will increase the time in the order of vertices. We can see sequential implementations that run on supercomputers perform well, but have very expensive Hardware. Whereas GPUs are much cheaper when compared to these computers. GPUs are highly programming restrictive due to the nature of their hardware.

This project depicts the use of GPGPU to help improve the performance of the Pathfinding algorithms, Breadth-First Search and Single-Source Shortest Path specifically. I will present GPU implementations of these algorithms along with different approaches taken. Experimental results show that up to 6.52 times speedup is achieved over its CPU counterpart.

# CUDA Programming Model
A general-purpose programming model by Nvidia that leverages the use of GPUs to solve many complex problems in an efficient way than CPU. CUDA comes with a software environment that helps developers to build applications using C/C++. The main challenge is to use an increasing number of cores in GPUs for parallelizing parts of the application. The idea is to distribute the workload among all the cores parallelly by assigning a task to each thread. The model has three main abstractions Grids, Blocks, and Threads at the lowest granularity. Depending on your GPU model, cores, and compute capabilities will vary. These partitions divide the problems into subproblems to be solved independently. You can find out more here. The Nvidia GPUs have SIMT architecture similar to SIMD, where all the threads execute the same instruction set on different data in parallel.

# Graph Representation
A graph _G(V,E)_ can be represented as an Adjacency Matrix as well as an Adjacency List. Here, I have used an adjacency list due to their efficient use of space _O(V+E)_.  This list _A_ can be stored contiguously along with two more arrays,  _E_ which holds all the offsets, and, _O_ which holds the outdegree for a vertex _v_ which means _A[v]to A[v+O[v]]_ holds all the children. There can be an additional weighted array _W_  which is stored contiguously as well.  
<p align="center"> <img src="images/graph_rep.jpg" width="500" height="350" /> </p>

# Breadth-First Search
## Definition
Given an undirected, unweighted graph _G(V,E)_  and a given source _S_, find the minimum distance from _S_ to each vertex in Gsuch that all the vertices with the same depth must be visited before moving to the next level. Hence it is guaranteed to find the shortest path to a vertex.

## Serial Approach

The BFS Algorithm is,
- Adding S into the queue
- Until all vertices are not visited
  - Loop over all the vertices adjacent to the front vertex
  - To avoid cycles mark the vertices as visited
  - Add them at the back of the queue

The queue ensures level-wise progression. This serial implementation has the time complexity of _O(VE)_. Once the density of the graph increases it really becomes challenging for the CPU execution to finish in optimal time. Note that I have not used extra space to keep track of the visited vertices since the array of distances is sufficient to check whether the vertex is visited or not. Also, I do not store any previous vertices because in a parallel implementation the current vertex may be found via a different path.

## Parallel Approach (Naive)

This approach is somewhat similar to the serial implementation. While parallelizing BFS, the only way to traverse is level-wise. So as suggested in [B4] perform level-synchronous BFS where the current queue represents the current level. Instead of only visiting the front vertex of the queue, distribute each vertex to a thread to fill the next level queue in parallel. Even if the current vertex may have different paths, the level-synchronous approach makes sure that the overwrites by different threads will always be the same.

<p align="center"> <img src="images/bfs_rep.jpg" width="500" height="350" /> </p>

### Possible Problems
At a bare minimum, GPU requires around ~300 clock cycles to access global memory. It is always ideal to perform memory coalescing, where all the threads access the memory at the same time. It is important to maximize the bandwidth to global memory.
Since performing level-synchronization is necessary, All the threads need to update the same queue, which brings race conditions between them that could lead to inconsistencies.
Parallelizing the outermost loop is difficult because the algorithm goes level by level. Therefore, the CUDA kernel must be launched for each new level. This incurs the cost of time consumed for data transfer between CPU and GPU.
The graph density as well as regularness also play a huge role. It is possible that a graph may not utilize maximum GPU throughput i.e launching less number of threads at every level.
Highly irregular access to global memory will slow down the performance.
The load imbalance between threads is difficult to avoid i.e the outdegrees varying drastically at the same time.

### Blocked Queue Approach

CUDA provides an L2 cache for each block where only threads running in that block can access this shared memory. It is located on-chip so the access times are reduced due to high bandwidth and low latency. Depending on the number of memory allocated between multiple blocks execution times may vary due to resource use limitations of the GPU. Regardless, this shared memory could be used to maintain a block-level queue to ensure threads will only write to the global queue if and only if the block-level queue is full. This will help avoid collisions at the global queue. Once all threads explore their corresponding adjacent vertices, The block-level queue could coalesce the to the global memory if the sufficient number of threads are launched.

### Hierarchical Queue Approach
Hierarchical Queue Approach
As shown in [B1] it is possible to specialize the queue by adding another level for warps. Thus, avoiding collisions at the block-level queue as well. According to the indices of threads, each one is mapped to its respective sub-queue. The authors of the paper have implemented it such that they copy these sub-queues to the block-level queue. It seems unreasonable because both blocked-level and sub-queues reside inside shared memory. I have skipped copying to block-level and directly coalesced to global memory. Similar to the previous approach, when the sub-queue becomes full, the thread will write to the global queue.

<p align="center"> <img src="images/hierar.jpg" width="500" height="350" /> </p>


| **G** | **V** | **E** | **CPU** | **GPU-N** | **GPU-B** | **GPU-H** |
| --------- | --------- | --------- | --------- | --------- | --------- | --------- |
| New York  | 264,346 | 733,846 | 29.9261 ms | 1.21x | 0.98x | 0.92x |
| California | 1,890,815 | 4,657,742 | 237.658 ms | 2.62x | 2.38x | 2.10x |
| Western USA | 6,262,104 | 15,248,146 | 857.753 ms | 4.38x | 4.09x | 3.97x |
| Central USA | 14,081,816 | 34,292,496 | 2625.37 ms | 6.52x | 6.11x | 6.15x |
| Whole USA | 23,947,347 | 58,333,344 | 3400.58 ms | 5.80x | 5.35x | 5.30x |

Table shows the speed up against CPU execution over real-world sparse road graphs from 9th DIMACS shortest path Datasets of USA roads.

# Single-Source Shortest Path
## Definition
The weighted, undirected graph _G(V,E,W)_ and a source vertex _S_, find the shortest path to all the vertices from _S_ such that the distance between any two vertices _(u,v)_ should be minimum of all the paths from _u_ to _v_.

## Serial Approach
The Dijkstra’s Algorithm is,
- Set all the distances to infinity
- Add S to the queue.
- Until all vertices are not visited
  - Pick a vertex which is closer to the current vertex from the queue
  - Loop over all the adjacent vertices
  - If the adjacent vertex has been visited then if the current distance is smaller update distance to vertex to the current distance.
  - Otherwise, add them at the back of the queue.

Now, this queue can be a priority queue. Thus, we can extract the minimum weighted edge in constant time. I avoid the usage of the priority queue because it is maintained as a binary heap under the hood. Therefore, the insertion and deletion require _O(log n)_ which in large arrays could make the performance really slow at least in my case. It is better to perform insertions in constant time. So, instead, I sort the array at each iteration using STL’s sort method which uses Intro sort Algorithm which does hybrid sorting using Quicksort, Heapsort, and, Insertion sort. So this Dijkstra’s has a total running time of _O(VE+Vlog V)_. A handicap for the CPU would be to use a Fibonacci heap where insertion and minimum extraction takes _O(1)_. But I have not explored this because there won’t be much of a performance increase when compared with the parallel approach.

## Parallel Approach

BFS and Dijkstra’s hold very similar approaches except that we need to extract minimum value from the queue in case of Dijkstra’s. It was very easy to convert the Parallel BFS approach into finding the shortest path. There is no need for sorting or extracting the minimum value when you consider all of the vertices in the current queue are executed in parallel. Although, it is important to reflect the changes accordingly when you revisit a vertex via the shortest path. But the essence or the gist remains intact between both parallel approaches. I have implemented this algorithm using Hierarchical-queue approach.


# Attempted Optimizations
## Atomic Operations
To avoid inconsistencies at the global queue as well as the local queue(s). CUDA threads could make use of atomic operations to correctly fill the queues without overwriting at the same location. These atomic operations require an address at which the thread will make its manipulations. Thus, when calling these functions, threads will have exclusive access at that memory location. But the access becomes serialized and also affects the performance.

<p align="center"> <img src="images/atomic.jpg" width="500" height="350" /> </p>
## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/abhishek1297/Parallel-Algorithms/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Summary
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```



For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/abhishek1297/Parallel-Algorithms/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
