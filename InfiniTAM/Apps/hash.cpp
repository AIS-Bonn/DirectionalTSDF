#include <iostream>
#include <ITMLib/Utils/ITMMath.h>
#include <ITMLib/Objects/Scene/ITMVoxelBlockHash.h>

int main(int argv, char **argc)
{
	Vector3s voxelPos(std::stoi(argc[1]), std::stoi(argc[2]), std::stoi(argc[3]));
	if (argv == 5)
	{
		std::cout << ITMLib::hashIndex(voxelPos, ITMLib::TSDFDirection(std::stoi(argc[4]))) << std::endl;
	}
	else
	{
		std::cout << ITMLib::hashIndex(voxelPos) << std::endl;
	}
	return 0;
}
