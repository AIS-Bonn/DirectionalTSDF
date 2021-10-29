//
// Created by Malte Splietker on 05.05.20.
//

#pragma once

#include <string>
#include <vector>
#include <ORUtils/SE3Pose.h>

namespace InputSource
{

class TrajectorySourceEngine
{
public:
	TrajectorySourceEngine() = default;

	void Read(const std::string& path);

	[[nodiscard]]
	bool hasMorePoses() const;

	const ORUtils::SE3Pose* getPose();

private:
	std::vector<ORUtils::SE3Pose>::const_iterator it;
	std::vector<ORUtils::SE3Pose> poses;
};

} // InputSource