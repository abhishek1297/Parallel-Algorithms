#include "dataIO.hpp"
#include <fstream>
#include <iostream>

std::string dir{"/home/abhishek/Downloads/graph datasets/"};
struct DatasetInfo F[] {
    {true, 4, {1, 2, 3}, dir +"test.txt"},// 0
    {false, 2, {0, 1}, dir + "live-journal.txt"}, // 1
    {true, 2, {0, 1}, dir + "youtube.txt"}, // 2
    {false, 2, {0, 1}, dir + "roadNet-CA.txt"},// 3
    {false, 2, {0, 1}, dir + "p2p-Gnutella31.txt"}, // 4
    {true, 4, {1, 2, 3}, dir + "USA-road-d.CAL.gr"}, // 5
    {true, 4, {1, 2, 3}, dir + "USA-road-d.NY.gr"}, // 6
    {true, 4, {1, 2, 3}, dir + "USA-road-d.W.gr"}, // 7
    {true, 4, {1, 2, 3}, dir + "USA-road-d.CTR.gr"}, // 8
    {true, 4, {1, 2, 3}, dir + "USA-road-d.USA.gr"}, // 9
};

int writeToFile(const Graph& G, double minCPU, std::vector<double> minGPU, int minDegree, int maxDegree, double avgDegree) {

	std::ofstream out{"analysis.txt", std::ios::app};
	if (out) {

		out << G.pathData_m << std::endl;
		out << "V: " << G.numVertices_m << "\t" << "E: " << G.numEdges_m << std::endl;
		out << "CPU: " << minCPU << std::endl;
		out << "GPU (N, H, B): ";
		for (double d: minGPU)
			out << d << " ";
		out << std::endl;
		out << "Degree range: [" << minDegree << ", " << maxDegree << "]\t" << "Avg Degree: " << avgDegree << std::endl << std::endl;
		std::cout << "Data Appended" << std::endl;
		out.close();
		return 0;

	}
	out.close();
	return 1;

}
