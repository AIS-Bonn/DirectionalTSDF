//
// Created by Malte Splietker on 05.05.20.
//

#include "TrajectorySourceEngine.h"

#include <sstream>
#include <fstream>
#include <ORUtils/SE3Pose.h>
#include <ITMLib/Utils/ITMMath.h>
#include <iostream>

namespace InputSource
{
void TrajectorySourceEngine::Read(const std::string& path)
{
	std::ifstream file(path);
	std::string line;
	if (not file.is_open())
		printf("error opening trajectory file %s\n", path.c_str());

	while (std::getline(file, line))
	{
		if (line[0] == '#')
			continue;
		std::istringstream lineStream(line);

		float timestamp, tx, ty, tz, qx, qy, qz, qw;
		lineStream >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

		ORUtils::SE3Pose& pose = poses.emplace_back();
		pose.SetFrom(tx, ty, tz, qx, qy, qz, qw);
	}
	it = poses.begin();
}

bool TrajectorySourceEngine::hasMorePoses() const
{
	return it != poses.end();
}

const ORUtils::SE3Pose* TrajectorySourceEngine::getPose()
{
	const ORUtils::SE3Pose* pose = &(*it);
	it++;
	return pose;
}

} // InputSource