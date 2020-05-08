//
// Created by Malte Splietker on 26.07.19.
//

#include <iomanip>
#include <experimental/filesystem>
#include <iostream>

#include "ITMLoggingEngine.h"

namespace fs = std::experimental::filesystem;

namespace ITMLib
{

void ITMLoggingEngine::Initialize(const std::string& outputDirectory)
{
	m_outputDirectory = outputDirectory;
	fs::create_directories(m_outputDirectory);

	m_timeStatsFile.open(m_outputDirectory + "/time_stats.txt");
	m_timeStatsFile.flags(std::ios::right);
	m_timeStatsFile.setf(std::ios::fixed);
	ITMTimeStats stats_dummy;
	stats_dummy.PrintHeader(m_timeStatsFile);

	m_trackingFile.open(m_outputDirectory + "/tracking.txt");
	m_trackingFile << std::fixed << std::left << std::setprecision(10);
	m_trackingFile << "# tx ty tz qx qy qz qw score result" << std::endl;
}

void ITMLoggingEngine::CloseAll()
{
	m_timeStatsFile.flush();
	m_timeStatsFile.close();

	m_trackingFile.flush();
	m_trackingFile.close();
}

void ITMLoggingEngine::LogTimeStats(const ITMTimeStats& timeStats)
{
	m_timeStatsFile << timeStats << std::endl;
}

Vector4f QuaternionFromTransformationMatrix(const Matrix4f& M)
{
	Matrix3f R;
	R.m[0 + 3 * 0] = M.m[0 + 4 * 0]; R.m[1 + 3 * 0] = M.m[1 + 4 * 0]; R.m[2 + 3 * 0] = M.m[2 + 4 * 0];
	R.m[0 + 3 * 1] = M.m[0 + 4 * 1]; R.m[1 + 3 * 1] = M.m[1 + 4 * 1]; R.m[2 + 3 * 1] = M.m[2 + 4 * 1];
	R.m[0 + 3 * 2] = M.m[0 + 4 * 2]; R.m[1 + 3 * 2] = M.m[1 + 4 * 2]; R.m[2 + 3 * 2] = M.m[2 + 4 * 2];
	R.m[0 + 3 * 0] = M.m[0 + 4 * 0]; R.m[1 + 3 * 0] = M.m[1 + 4 * 0]; R.m[2 + 3 * 0] = M.m[2 + 4 * 0];
	R.m[0 + 3 * 1] = M.m[0 + 4 * 1]; R.m[1 + 3 * 1] = M.m[1 + 4 * 1]; R.m[2 + 3 * 1] = M.m[2 + 4 * 1];
	R.m[0 + 3 * 2] = M.m[0 + 4 * 2]; R.m[1 + 3 * 2] = M.m[1 + 4 * 2]; R.m[2 + 3 * 2] = M.m[2 + 4 * 2];
	float values[4]; // x, y, z, w

	// "Quaternion Calculus and Fast Animation", Ken Shoemake, 1987 SIGGRAPH course notes
	float trace = R.m00 + R.m11 + R.m22;
	if (trace > 0)
	{
		trace = sqrt(1 + trace);
		float s = 1 / (2 * trace);
		values[3] = 0.5 * trace;
		values[0] = (R.m21 - R.m12) * s;
		values[1] = (R.m02 - R.m20) * s;
		values[2] = (R.m10 - R.m01) * s;
	}
	else
	{
		int i = 0;
		if (R.m11 > R.m00)
			i = 1;
		if (R.m22 > R.at(i, i))
			i = 2;
		int j = (i+1)%3;
		int k = (j+1)%3;

		trace = sqrt(R.at(i,i) - R.at(j,j) - R.at(k,k) + 1.0);
		float s = 1 / (2 * trace);
		values[i] = 0.5 * trace;
		values[3] = (R.at(k, j) - R.at(j, k)) * s;
		values[j] = (R.at(j, i) + R.at(i, j)) * s;
		values[k] = (R.at(k, i) + R.at(i, k)) * s;
	}
	return Vector4f(values[0], values[1], values[2], values[3]);
}

void ITMLoggingEngine::LogPose(const ITMTrackingState& trackingState)
{
	Matrix4f T_WC = trackingState.pose_d->GetInvM();
	Matrix4f T_CW = trackingState.pose_d->GetM();
	Vector3f T = (T_WC * Vector4f(0, 0, 0, 1)).toVector3();
	Vector4f Q = QuaternionFromTransformationMatrix(T_CW);

	m_trackingFile << T.x << " " << T.y << " " << T.z << " " << Q.x << " " << Q.y << " " << Q.z << " " << Q.w;
	m_trackingFile << " " << trackingState.trackerScore;
	m_trackingFile << " " << trackingState.trackerResult;
	m_trackingFile << std::endl;
}

} // namespace ITMLib