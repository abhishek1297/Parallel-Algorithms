
/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include "dataIO.hpp"
#include "bfs/main.hpp"
#include "sssp/main.hpp"

/**
 * Inputs to graph constructor defined in dataIO.cpp
 * Make sure to use weighted graphs for the SSSP.

    {true, 4, {1, 2, 3}, dir +"test.txt"},// 0
    {false, 2, {0, 1}, dir + "live-journal.txt"}, // 1
    {true, 2, {0, 1}, dir + "youtube.txt"}, // 2
    {false, 2, {0, 1}, dir + "roadNet-CA.txt"},// 3
    {false, 2, {0, 1}, dir + "p2p-Gnutella31.txt"}, // 4

    {true, 4, {1, 2, 3}, dir + "USA-road-d.CAL.gr"}, // 5
    {true, 4, {1, 2, 3}, dir + "USA-road-d.NY.gr"}, // 6
    {true, 4, {1, 2, 3}, dir + "USA-road-d.W.gr"}, // 7
    {true, 4, {1, 2, 3}, dir + "USA-road-d.CTR.gr"} // 8
    {true, 4, {1, 2, 3}, dir + "USA-road-d.USA.gr"} // 8
*/


int main() {
	const struct DatasetInfo &inp = F[0];
	Graph G(inp.fname, inp.numInputs, inp.indexToRead, inp.toZeroIdx, "bfs");
	std::cout << G << std::endl;
	return bfsMain(std::move(G));
//	return ssspMain(std::move(G));
}
