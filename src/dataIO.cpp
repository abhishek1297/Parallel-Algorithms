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
    {true, 4, {1, 2, 3}, dir + "USA-road-d.CTR.gr"} // 8
};

int writeToFile(std::string fname, int nV, int nE, double minCPU, double minGPU) {

	std::ofstream out{"analysis.txt", std::ios::app};
	if (out) {

		out << fname << std::endl;
		out << "V: " << nV << "\t" << "E: " << nE << std::endl;
		out << "CPU: " << minCPU << "\t" << "GPU: " << minGPU << std::endl << std::endl;
		std::cout << "Data Appended" << std::endl;
		out.close();
		return 0;

	}
	out.close();
	return 1;

}
