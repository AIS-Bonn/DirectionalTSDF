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
	while (std::getline(file, line))
	{
		std::istringstream lineStream(line);

		float timestamp, tx, ty, tz, qx, qy, qz, qw;
		lineStream >> timestamp >> tx >> ty >> tz >> qx >> qy >> qz >> qw;

		ORUtils::SE3Pose& pose = poses.emplace_back();

		// position is provided in WC-frame, rotation in CW-frame (see ITMLoggingEngine.cpp)

		Matrix3f R_CW, R_WC;
		R_CW.m[0] = 1 - 2 * qy * qy - 2 * qz * qz;
		R_CW.m[1] = 2 * qx * qy - 2 * qz * qw;
		R_CW.m[2] = 2 * qx * qz + 2 * qy * qw;
		R_CW.m[3] = 2 * qx * qy + 2 * qz * qw;
		R_CW.m[4] = 1 - 2 * qx * qx - 2 * qz * qz;
		R_CW.m[5] = 2 * qy * qz - 2 * qx * qw;
		R_CW.m[6] = 2 * qx * qz - 2 * qy * qw;
		R_CW.m[7] = 2 * qy * qz + 2 * qx * qw;
		R_CW.m[8] = 1 - 2 * qx * qx - 2 * qy * qy;
		R_CW.inv(R_WC);

		Matrix4f T_WC;
		T_WC.setIdentity();
		T_WC.m00 = R_WC.m00;
		T_WC.m01 = R_WC.m01;
		T_WC.m02 = R_WC.m02;
		T_WC.m10 = R_WC.m10;
		T_WC.m11 = R_WC.m11;
		T_WC.m12 = R_WC.m12;
		T_WC.m20 = R_WC.m20;
		T_WC.m21 = R_WC.m21;
		T_WC.m22 = R_WC.m22;
		T_WC.m30 =  tx;
		T_WC.m31 =  ty;
		T_WC.m32 =  tz;
		pose.SetInvM(T_WC);
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